import logging
import threading
import time
import uuid
import weakref
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Finalizers may flush during interpreter shutdown, when importing pyarrow for
# the first time is already too late for it to register its own exit hooks.
import pyarrow.feather  # noqa: F401

logger = logging.getLogger(__name__)
_AUTOSAVE_RETRY_INTERVAL = 1.0


def _autosave_interval_for_rows(row_count: int) -> float:
    if row_count < 1000:
        return 0.1
    if row_count < 10000:
        return 0.2
    if row_count < 100000:
        return 0.5
    return 1.0


class _BufferState:
    """Mutable storage state shared by all handles for one resolved path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.segs: list[pd.DataFrame] = []
        if self.path.exists():
            self.segs.append(pd.read_feather(self.path))
        self.records: list[dict[str, float | int | str]] = []
        self.autosave_interval = _autosave_interval_for_rows(
            sum(len(segment) for segment in self.segs)
        )

        self.dirty = False
        self.closed = False
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def get_df(self) -> pd.DataFrame:
        with self.condition:
            return self._get_df_locked().copy(deep=True)

    def add_one_row(self, values: dict[str, float | int | str]) -> None:
        with self.condition:
            self._raise_if_closed()
            if not values:
                raise ValueError("cannot add a row without any columns")
            self.records.append(dict(values))
            self._mark_dirty_locked()

    def add_multi_rows(self, dataframe: pd.DataFrame) -> None:
        with self.condition:
            self._raise_if_closed()
            if dataframe.empty:
                return
            if self.records:
                self.segs.append(pd.DataFrame.from_records(self.records))
                self.records = []
            self.segs.append(dataframe.copy(deep=True))
            self._mark_dirty_locked()

    def flush(self) -> pd.DataFrame:
        with self.condition:
            self._raise_if_closed()
            if not self.dirty:
                return self._get_df_locked()
            return self._write_locked()

    def shutdown(self) -> None:
        """Flush pending rows and stop the autosave thread."""
        with self.condition:
            if self.closed:
                return
            if self.dirty:
                self._write_locked()
            self.closed = True
            self.condition.notify_all()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def _mark_dirty_locked(self) -> None:
        was_dirty = self.dirty
        self.dirty = True
        if not was_dirty:
            self.condition.notify_all()

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("cannot use a closed dataframe buffer")

    def _get_df_locked(self) -> pd.DataFrame:
        if self.records:
            self.segs.append(pd.DataFrame.from_records(self.records))
            self.records = []

        if len(self.segs) == 0:
            return pd.DataFrame({})
        if len(self.segs) == 1:
            return self.segs[0]

        dataframe = pd.concat(self.segs, ignore_index=True)
        self.segs = [dataframe]
        return dataframe

    def run(self) -> None:
        while True:
            with self.condition:
                while not self.dirty and not self.closed:
                    self.condition.wait()

                if self.closed:
                    return

                self.condition.wait(timeout=self.autosave_interval)
                if self.closed:
                    return
                if self.dirty:
                    try:
                        self._write_locked()
                    except Exception:
                        self.autosave_interval = max(
                            self.autosave_interval,
                            _AUTOSAVE_RETRY_INTERVAL,
                        )
                        logger.exception(
                            "Failed to autosave %s; retrying in %.1f seconds",
                            self.path,
                            self.autosave_interval,
                        )

    def _write_locked(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> pd.DataFrame:
        dataframe = self._get_df_locked()
        tmp = self.path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        try:
            dataframe.to_feather(tmp)
            self._replace_tmp(
                tmp,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        finally:
            with suppress(OSError):
                tmp.unlink()
        self.dirty = False
        self.autosave_interval = _autosave_interval_for_rows(dataframe.shape[0])
        return dataframe

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
                    retry_delay *= 2
                else:
                    raise


class DataFrameBuffer:
    """Shared dataframe buffer for one path.

    Use :meth:`open` to reuse the process-local buffer for a path. The last
    Python reference disappearing flushes pending data and stops its worker.
    """

    def __init__(
        self,
        state: _BufferState,
        path_key: Path,
        owner_token: object,
    ) -> None:
        self._state = state
        self._finalizer = weakref.finalize(
            self,
            _finalize_buffer,
            path_key,
            state,
            owner_token,
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
    ) -> "DataFrameBuffer":
        """Return the shared process-local buffer for ``path``."""
        return _get_shared_buffer(Path(path))

    @property
    def path(self) -> Path:
        return self._state.path

    def get_df(self) -> pd.DataFrame:
        return self._state.get_df()

    def add_one_row(self, values: dict[str, float | int | str]) -> None:
        self._state.add_one_row(values)

    def add_multi_rows(self, dataframe: pd.DataFrame) -> None:
        self._state.add_multi_rows(dataframe)

    def flush(self) -> pd.DataFrame:
        """Flush pending rows immediately, blocking until the save finishes."""
        return self._state.flush()


@dataclass
class _BufferEntry:
    state: _BufferState
    buffer_ref: weakref.ReferenceType[DataFrameBuffer]
    owner_token: object


_BUFFER_ENTRIES: dict[Path, _BufferEntry] = {}
_BUFFER_ENTRIES_LOCK = threading.Lock()


def _get_shared_buffer(
    path: Path,
) -> DataFrameBuffer:
    path_key = path.resolve()
    with _BUFFER_ENTRIES_LOCK:
        entry = _BUFFER_ENTRIES.get(path_key)
        if entry is not None:
            buffer = entry.buffer_ref()
            if buffer is not None:
                return buffer

            # A failed or still-pending finalizer may leave the state available
            # after its wrapper dies. A new token transfers ownership safely.
            owner_token = object()
            buffer = DataFrameBuffer(entry.state, path_key, owner_token)
            entry.buffer_ref = weakref.ref(buffer)
            entry.owner_token = owner_token
            return buffer

        state = _BufferState(path)
        owner_token = object()
        buffer = DataFrameBuffer(state, path_key, owner_token)
        _BUFFER_ENTRIES[path_key] = _BufferEntry(
            state=state,
            buffer_ref=weakref.ref(buffer),
            owner_token=owner_token,
        )
        return buffer


def _finalize_buffer(
    path_key: Path,
    state: _BufferState,
    owner_token: object,
) -> None:
    with _BUFFER_ENTRIES_LOCK:
        entry = _BUFFER_ENTRIES.get(path_key)
        if (
            entry is None
            or entry.state is not state
            or entry.owner_token is not owner_token
        ):
            return
        try:
            state.shutdown()
        except Exception:
            logger.exception("Failed to shut down dataframe buffer %s", state.path)
        else:
            del _BUFFER_ENTRIES[path_key]
