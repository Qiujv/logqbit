import logging
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
_AUTOSAVE_RETRY_INTERVAL = 1.0
_OPEN_BUFFER_PATHS: set[Path] = set()
_OPEN_BUFFER_PATHS_LOCK = threading.Lock()


def _reserve_buffer_path(path: Path) -> Path:
    resolved = path.resolve()
    with _OPEN_BUFFER_PATHS_LOCK:
        if resolved in _OPEN_BUFFER_PATHS:
            raise RuntimeError(
                f"Data buffer at '{path}' already has an active writer; "
                "close it before opening another."
            )
        _OPEN_BUFFER_PATHS.add(resolved)
    return resolved


def _release_buffer_path(path: Path) -> None:
    with _OPEN_BUFFER_PATHS_LOCK:
        _OPEN_BUFFER_PATHS.discard(path)


def _autosave_interval_for_rows(row_count: int) -> float:
    if row_count < 1000:
        return 0.1
    if row_count < 10000:
        return 0.2
    if row_count < 100000:
        return 0.5
    return 1.0


class DataFrameBuffer:
    """Buffer appended dataframe rows and persist them to a feather file.

    The background thread has a small state machine:
    wait until data becomes dirty, wait the current autosave interval to batch
    nearby appends, then write if the buffer is still dirty. ``flush()`` skips
    that delay and writes synchronously on the caller's thread. Within one
    process, only one active buffer may own a path at a time.
    """

    def __init__(self, path: str | Path, autosave_interval: float = 0.2):
        self.path = Path(path)
        self._path_key = _reserve_buffer_path(self.path)
        try:
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
        except BaseException:
            _release_buffer_path(self._path_key)
            raise

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def get_df(self) -> pd.DataFrame:
        with self._condition:
            df = self._get_df_locked()
            return df.copy(deep=True)

    def add_one_row(self, kwargs: dict[str, float | int | str]):
        with self._condition:
            self._raise_if_closed()
            if not kwargs:
                raise ValueError("cannot add a row without any columns")
            self._records.append(dict(kwargs))
            self._mark_dirty_locked()

    def add_multi_rows(self, df: pd.DataFrame):
        with self._condition:
            self._raise_if_closed()
            if df.empty:
                return
            if self._records:
                self._segs.append(pd.DataFrame.from_records(self._records))
                self._records = []
            self._segs.append(df.copy(deep=True))
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
            if self._dirty:
                self._write_locked()
            self._closed = True
            self._condition.notify_all()
        try:
            if (
                self._thread.is_alive()
                and self._thread is not threading.current_thread()
            ):
                self._thread.join(timeout=2)
        finally:
            _release_buffer_path(self._path_key)

    def _mark_dirty_locked(self) -> None:
        was_dirty = self._dirty
        self._dirty = True
        if not was_dirty:
            self._condition.notify_all()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed DataFrameBuffer")

    def _get_df_locked(self) -> pd.DataFrame:
        if self._records:
            self._segs.append(pd.DataFrame.from_records(self._records))
            self._records = []

        if len(self._segs) == 0:
            return pd.DataFrame({})
        if len(self._segs) == 1:
            return self._segs[0]

        df = pd.concat(self._segs, ignore_index=True)
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
                    try:
                        self._write_locked()
                    except Exception:
                        self._autosave_interval = max(
                            self._autosave_interval,
                            _AUTOSAVE_RETRY_INTERVAL,
                        )
                        logger.exception(
                            "Failed to autosave %s; retrying in %.1f seconds",
                            self.path,
                            self._autosave_interval,
                        )

    def _write_locked(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> pd.DataFrame:
        df = self._get_df_locked()
        tmp = self.path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        try:
            df.to_feather(tmp)
            self._replace_tmp(tmp, max_retries=max_retries, retry_delay=retry_delay)
        finally:
            with suppress(OSError):
                tmp.unlink()
        self._dirty = False
        self._autosave_interval = _autosave_interval_for_rows(df.shape[0])
        return df

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
                    raise
