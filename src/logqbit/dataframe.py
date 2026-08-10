from __future__ import annotations

import logging
import threading
import time
import uuid
import weakref
from collections.abc import Iterable
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


class _DataFrameCache:
    """In-memory dataframe segments loaded once and later written as a whole."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.segments: list[pd.DataFrame] = []
        if path.exists():
            self.segments.append(pd.read_feather(path))
        self.records: list[dict[str, float | int | str]] = []
        self.dirty = False

    @property
    def row_count(self) -> int:
        return sum(len(segment) for segment in self.segments) + len(self.records)

    def add_one_row(self, values: dict[str, float | int | str]) -> None:
        if not values:
            raise ValueError("cannot add a row without any columns")
        self.records.append(dict(values))
        self.dirty = True

    def add_multi_rows(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            return
        if self.records:
            self.segments.append(pd.DataFrame.from_records(self.records))
            self.records = []
        self.segments.append(dataframe.copy(deep=True))
        self.dirty = True

    def get_df(self) -> pd.DataFrame:
        if self.records:
            self.segments.append(pd.DataFrame.from_records(self.records))
            self.records = []

        if len(self.segments) == 0:
            return pd.DataFrame({})
        if len(self.segments) == 1:
            return self.segments[0]

        dataframe = pd.concat(self.segments, ignore_index=True)
        self.segments = [dataframe]
        return dataframe

    def write(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> pd.DataFrame:
        dataframe = self.get_df()
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


@dataclass(frozen=True)
class BufferWorkerInfo:
    """Diagnostic snapshot of one dataframe buffer's autosave worker."""

    worker_id: str
    path: Path
    thread_name: str
    thread_ident: int | None
    thread_alive: bool
    dirty: bool
    owner_alive: bool
    last_error: str | None

    @property
    def orphaned(self) -> bool:
        """Whether the owning :class:`DataFrameBuffer` has been collected."""
        return not self.owner_alive


class _BufferWorker:
    """Synchronization and autosave lifecycle for one independent cache."""

    def __init__(self, path: Path) -> None:
        self.worker_id = uuid.uuid4().hex
        self.cache = _DataFrameCache(path)
        self.autosave_interval = _autosave_interval_for_rows(self.cache.row_count)
        self.closed = False
        self.last_error: str | None = None
        self.condition = threading.Condition()
        self.thread = threading.Thread(
            target=self.run,
            name=f"logqbit-buffer-{self.worker_id[:6]}-{path.name}",
            daemon=True,
        )
        self.thread.start()

    @property
    def path(self) -> Path:
        return self.cache.path

    def get_df(self) -> pd.DataFrame:
        with self.condition:
            self._raise_if_closed()
            return self.cache.get_df().copy(deep=True)

    def add_one_row(self, values: dict[str, float | int | str]) -> None:
        with self.condition:
            self._raise_if_closed()
            was_dirty = self.cache.dirty
            self.cache.add_one_row(values)
            self._notify_if_newly_dirty(was_dirty)

    def add_multi_rows(self, dataframe: pd.DataFrame) -> None:
        with self.condition:
            self._raise_if_closed()
            was_dirty = self.cache.dirty
            self.cache.add_multi_rows(dataframe)
            self._notify_if_newly_dirty(was_dirty)

    def flush(self) -> pd.DataFrame:
        with self.condition:
            self._raise_if_closed()
            if not self.cache.dirty:
                return self.cache.get_df()
            return self._write_locked()

    def close(self) -> None:
        """Flush pending rows and stop the autosave thread."""
        with self.condition:
            if not self.closed:
                if self.cache.dirty:
                    self._write_locked()
                self.closed = True
                self.condition.notify_all()

        if self.thread is threading.current_thread():
            raise RuntimeError("a dataframe buffer worker cannot join itself")
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.thread.is_alive():
            error = TimeoutError(
                f"autosave thread {self.thread.name!r} did not stop within 2 seconds"
            )
            self.last_error = _format_error(error)
            raise error

    def info(self, owner_alive: bool) -> BufferWorkerInfo:
        with self.condition:
            return BufferWorkerInfo(
                worker_id=self.worker_id,
                path=self.path,
                thread_name=self.thread.name,
                thread_ident=self.thread.ident,
                thread_alive=self.thread.is_alive(),
                dirty=self.cache.dirty,
                owner_alive=owner_alive,
                last_error=self.last_error,
            )

    def _notify_if_newly_dirty(self, was_dirty: bool) -> None:
        if self.cache.dirty and not was_dirty:
            self.condition.notify_all()

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("cannot use a closed dataframe buffer")

    def run(self) -> None:
        while True:
            with self.condition:
                while not self.cache.dirty and not self.closed:
                    self.condition.wait()

                if self.closed:
                    return

                self.condition.wait(timeout=self.autosave_interval)
                if self.closed:
                    return
                if self.cache.dirty:
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

    def _write_locked(self) -> pd.DataFrame:
        try:
            dataframe = self.cache.write()
        except Exception as error:
            self.last_error = _format_error(error)
            raise
        self.last_error = None
        self.autosave_interval = _autosave_interval_for_rows(dataframe.shape[0])
        return dataframe


def _format_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


class DataFrameBuffer:
    """Independent autosaving dataframe buffer for one path.

    Construction reads the backing file once and creates a new cache and
    worker. When the last reference to this object disappears, pending data is
    flushed and its worker is stopped automatically.
    """

    def __init__(self, path: str | Path) -> None:
        self._worker = _BufferWorker(Path(path))
        _BUFFER_REGISTRY.register(self, self._worker)
        self._finalizer = weakref.finalize(
            self,
            _finalize_buffer,
            self._worker.worker_id,
        )

    @classmethod
    def inspect_workers(cls) -> tuple[BufferWorkerInfo, ...]:
        """Return diagnostic snapshots of all registered autosave workers."""
        return _BUFFER_REGISTRY.inspect()

    @classmethod
    def close_workers(
        cls,
        worker_ids: Iterable[str] | str | None = None,
    ) -> tuple[BufferWorkerInfo, ...]:
        """Close selected workers, or all workers when no IDs are provided.

        Every selected worker is attempted. The returned snapshots describe
        workers that could not be closed and remain registered.
        """
        return _BUFFER_REGISTRY.close(worker_ids)

    @property
    def path(self) -> Path:
        return self._worker.path

    def get_df(self) -> pd.DataFrame:
        return self._worker.get_df()

    def add_one_row(self, values: dict[str, float | int | str]) -> None:
        self._worker.add_one_row(values)

    def add_multi_rows(self, dataframe: pd.DataFrame) -> None:
        self._worker.add_multi_rows(dataframe)

    def flush(self) -> pd.DataFrame:
        """Flush pending rows immediately, blocking until the save finishes."""
        return self._worker.flush()

    def close(self) -> None:
        """Flush pending rows and stop this buffer's autosave worker."""
        self._worker.close()
        _BUFFER_REGISTRY.discard(self._worker.worker_id, self._worker)
        self._finalizer.detach()


@dataclass
class _BufferEntry:
    worker: _BufferWorker
    owner_ref: weakref.ReferenceType[DataFrameBuffer]


class _BufferRegistry:
    """Track workers for diagnostics and recovery without sharing their data."""

    def __init__(self) -> None:
        self._entries: dict[str, _BufferEntry] = {}
        self._lock = threading.Lock()

    def register(self, owner: DataFrameBuffer, worker: _BufferWorker) -> None:
        with self._lock:
            self._entries[worker.worker_id] = _BufferEntry(
                worker=worker,
                owner_ref=weakref.ref(owner),
            )

    def inspect(
        self,
        worker_ids: Iterable[str] | None = None,
    ) -> tuple[BufferWorkerInfo, ...]:
        selected_ids = None if worker_ids is None else set(worker_ids)
        with self._lock:
            entries = [
                entry
                for worker_id, entry in self._entries.items()
                if selected_ids is None or worker_id in selected_ids
            ]
        return tuple(
            entry.worker.info(owner_alive=entry.owner_ref() is not None)
            for entry in entries
        )

    def close(
        self,
        worker_ids: Iterable[str] | str | None = None,
    ) -> tuple[BufferWorkerInfo, ...]:
        if isinstance(worker_ids, str):
            selected_ids = {worker_ids}
        elif worker_ids is None:
            selected_ids = None
        else:
            selected_ids = set(worker_ids)

        with self._lock:
            entries = [
                (worker_id, entry)
                for worker_id, entry in self._entries.items()
                if selected_ids is None or worker_id in selected_ids
            ]

        failed_ids: list[str] = []
        for worker_id, entry in entries:
            try:
                entry.worker.close()
            except Exception as error:
                entry.worker.last_error = _format_error(error)
                failed_ids.append(worker_id)
                logger.exception(
                    "Failed to close dataframe buffer worker %s for %s",
                    worker_id,
                    entry.worker.path,
                )
            else:
                self._discard_if_same(worker_id, entry.worker)

        return self.inspect(failed_ids)

    def finalize(self, worker_id: str) -> None:
        with self._lock:
            entry = self._entries.get(worker_id)
        if entry is None:
            return
        try:
            entry.worker.close()
        except Exception as error:
            entry.worker.last_error = _format_error(error)
            logger.exception(
                "Failed to finalize dataframe buffer worker %s for %s",
                worker_id,
                entry.worker.path,
            )
        else:
            self._discard_if_same(worker_id, entry.worker)

    def _discard_if_same(self, worker_id: str, worker: _BufferWorker) -> None:
        with self._lock:
            entry = self._entries.get(worker_id)
            if entry is not None and entry.worker is worker:
                del self._entries[worker_id]

    def discard(self, worker_id: str, worker: _BufferWorker) -> None:
        """Remove a worker after its owning buffer closes it directly."""
        self._discard_if_same(worker_id, worker)


_BUFFER_REGISTRY = _BufferRegistry()


def _finalize_buffer(worker_id: str) -> None:
    _BUFFER_REGISTRY.finalize(worker_id)
