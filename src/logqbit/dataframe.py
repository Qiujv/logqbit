import threading
import time
import uuid
import warnings
from pathlib import Path

import pandas as pd


class DataFrameBuffer:
    """Buffer appended dataframe rows and persist them to a feather file.

    The background thread has a small state machine:
    wait until data becomes dirty, wait the current autosave interval to batch
    nearby appends, then write if the buffer is still dirty. ``flush()`` skips
    that delay and writes synchronously on the caller's thread.
    """

    def __init__(self, path: str | Path, autosave_interval: float = 0.2):
        self.path = Path(path)
        self._autosave_interval = autosave_interval
        self._segs: list[pd.DataFrame] = []
        if self.path.exists():
            self._segs.append(pd.read_feather(self.path))
        self._records: list[dict[str, float | int | str]] = []

        self._dirty = False
        self._closed = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get_df(self) -> pd.DataFrame:
        with self._condition:
            df = self._get_df_locked()
        return df

    def add_one_row(self, kwargs: dict[str, float | int | str]):
        with self._condition:
            self._records.append(kwargs)
            self._mark_dirty_locked()

    def add_multi_rows(self, df: pd.DataFrame):
        with self._condition:
            if self._records:
                self._segs.append(pd.DataFrame.from_records(self._records))
                self._records = []
            self._segs.append(df)
            self._mark_dirty_locked()

    def flush(self) -> pd.DataFrame:
        """Flush pending rows immediately, blocking until the save finishes."""
        with self._condition:
            if not self._dirty:
                return self._get_df_locked()
            return self._write_locked()

    def close(self) -> None:
        """Flush pending rows and stop the autosave thread."""
        with self._condition:
            if self._closed:
                return

        self.flush()

        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def _mark_dirty_locked(self) -> None:
        was_dirty = self._dirty
        self._dirty = True
        if not was_dirty:
            self._condition.notify_all()

    def _get_df_locked(self) -> pd.DataFrame:
        if self._records:
            self._segs.append(pd.DataFrame.from_records(self._records))
            self._records = []

        if len(self._segs) == 0:
            return pd.DataFrame({})
        if len(self._segs) == 1:
            return self._segs[0]

        df = pd.concat(self._segs)
        self._segs = [df]
        return df

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._dirty and not self._closed:
                    self._condition.wait()

                if self._closed:
                    return

                self._condition.wait(timeout=self._autosave_interval)
                if self._closed:
                    return
                if self._dirty:
                    self._write_locked()

    def _write_locked(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> pd.DataFrame:
        df = self._get_df_locked()
        tmp = self.path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        df.to_feather(tmp)
        self._replace_tmp(tmp, max_retries=max_retries, retry_delay=retry_delay)
        self._dirty = False
        self._autosave_interval = self._autosave_interval_for_rows(df.shape[0])
        return df

    def _autosave_interval_for_rows(self, row_count: int) -> float:
        if row_count < 1000:
            return 0.1
        if row_count < 10000:
            return 0.2
        if row_count < 100000:
            return 0.5
        return 1.0

    def _replace_tmp(
        self,
        tmp: Path,
        max_retries: int,
        retry_delay: float,
    ) -> None:
        for attempt in range(max_retries):
            try:
                tmp.replace(self.path)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    warnings.warn(
                        f"Failed to replace {self.path} after {max_retries} attempts"
                    )
                    raise
