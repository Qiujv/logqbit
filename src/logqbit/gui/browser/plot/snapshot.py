"""Bounded, asynchronous snapshots for a multi-record plot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pandas as pd
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from logqbit.file_version import FileVersion

# Limits are deliberately conservative; file size is only a preliminary check.
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class Snapshot:
    frame: pd.DataFrame
    version: FileVersion
    size: int


class _Signals(QObject):
    finished = Signal(object, object)


class _Load(QRunnable):
    def __init__(self, records, cache, generation):
        super().__init__()
        self.records, self.cache, self.generation = records, cache, generation
        self.signals = _Signals()
        self.cancelled = Event()

    def run(self):
        try:
            result = self._load()
        except Exception as exc:
            result = ({}, [f"Snapshot loading failed: {exc}"])
        self.signals.finished.emit(self.generation, result)

    def _load(self):
        snapshots, errors = {}, []
        total = 0
        for record in self.records:
            if self.cancelled.is_set():
                break
            version = FileVersion.from_path(record.data_path)
            if version is None:
                errors.append(f"#{record.log_id}: data file missing")
                continue
            cached = self.cache.get(record.path)
            if cached is not None and cached.version == version:
                snapshot = cached
            else:
                if (
                    version.size > MAX_SNAPSHOT_BYTES
                    or total + version.size > MAX_SNAPSHOT_BYTES
                ):
                    errors.append(f"#{record.log_id}: snapshot size limit exceeded")
                    continue
                frame = record.read_dataframe()
                if frame is None or FileVersion.from_path(record.data_path) != version:
                    errors.append(
                        f"#{record.log_id}: data could not be read or changed during reading"
                    )
                    continue
                snapshot = Snapshot(
                    frame, version, int(frame.memory_usage(index=True, deep=True).sum())
                )
            if total + snapshot.size > MAX_SNAPSHOT_BYTES:
                errors.append(f"#{record.log_id}: snapshot size limit exceeded")
                continue
            snapshots[record.path] = snapshot
            total += snapshot.size
        return snapshots, errors


class PlotSnapshots(QObject):
    ready = Signal(object, object)
    loading = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.records = []
        self.cache: dict[Path, Snapshot] = {}
        self.generation = 0
        self.active = None
        self.pending = False

    def set_records(self, records):
        self.records = list(records)
        paths = {record.path for record in records}
        self.cache = {
            path: value for path, value in self.cache.items() if path in paths
        }
        self.generation += 1
        self.pending = False
        if self.active is not None:
            self.active.cancelled.set()

    def clear(self):
        self.set_records([])
        self.cache.clear()

    def request(self):
        if not self.records:
            return
        self.generation += 1
        self.loading.emit()
        if self.active is not None:
            self.active.cancelled.set()
            self.pending = True
            return
        self._start()

    def _start(self):
        self.pending = False
        worker = _Load(self.records[:], self.cache.copy(), self.generation)
        self.active = worker
        worker.signals.finished.connect(self._finished)
        self.pool.start(worker)

    def _finished(self, generation, result):
        self.active = None
        if generation == self.generation:
            self.cache, errors = result
            self.ready.emit(self.cache, errors)
        if self.pending and self.records:
            self._start()
