"""Cached records and disk access for LogQbit catalogs."""

from __future__ import annotations

import logging
import shutil
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd
import pyarrow
import pyarrow.ipc

from .metadata import LogMetadata

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_KNOWN_LOG_FILENAMES = {"const.yaml", "data.feather", "metadata.json"}


@dataclass(frozen=True)
class _FileVersion:
    mtime_ns: int
    size: int
    inode: int

    @classmethod
    def from_path(cls, path: Path) -> _FileVersion | None:
        try:
            file_stat = path.stat()
        except OSError:
            return None
        if not stat.S_ISREG(file_stat.st_mode):
            return None
        return cls(
            file_stat.st_mtime_ns,
            file_stat.st_size,
            file_stat.st_ino,
        )


_RETRY_VERSION = _FileVersion(mtime_ns=-1, size=-1, inode=-1)


@dataclass(frozen=True)
class LogRecord:
    """One catalog record with cached summary fields and explicit disk access."""

    path: Path
    row_count: int = 0
    columns: tuple[str, ...] = ()
    data_version: _FileVersion | None = None

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_dir():
            raise FileNotFoundError(f"Log directory at '{path}' does not exist.")
        object.__setattr__(self, "path", path)

    @property
    def log_id(self) -> int | str:
        """Return numeric directory names as integers, otherwise as text."""
        return int(self.path.name) if self.path.name.isdecimal() else self.path.name

    @property
    def meta_path(self) -> Path:
        return self.path / "metadata.json"

    @property
    def data_path(self) -> Path:
        return self.path / "data.feather"

    @property
    def const_path(self) -> Path:
        return self.path / "const.yaml"

    @cached_property
    def meta(self) -> LogMetadata:
        """Proxy to modify metadata.json on setting values."""
        return LogMetadata(
            self.meta_path,
            create=False,
            default_on_error=True,
        )

    # Read-only metadata fields cached in memory. Sync by manual meta.reload().
    @property
    def title(self) -> str:
        return str(self.meta.root.get("title", "untitled"))

    @property
    def star(self) -> int:
        return int(self.meta.root.get("star", 0))

    @property
    def trash(self) -> bool:
        return bool(self.meta.root.get("trash", False))

    @property
    def plot_axes(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.meta.root.get("plot_axes", []))

    @property
    def plot_fields(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.meta.root.get("plot_fields", []))

    @property
    def create_time(self) -> str:
        return str(self.meta.root.get("create_time", ""))

    @property
    def create_machine(self) -> str:
        return str(self.meta.root.get("create_machine", ""))

    def read_dataframe(self) -> pd.DataFrame | None:
        """Read the current Feather file once, without retaining a cache."""
        if not self.data_path.exists():
            return None
        try:
            return pd.read_feather(_read_data_buffer(self.data_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to read feather file %s: %s", self.data_path, exc)
            return None

    def read_yaml_text(self) -> str:
        if not self.const_path.exists():
            return "const.yaml not found."
        try:
            text = self.const_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to read yaml file %s: %s", self.const_path, exc)
            return f"Failed to read const.yaml: {exc}"
        return text if text.strip() else "(const.yaml is empty)"

    def list_image_files(self) -> list[Path]:
        return self._list_files(lambda path: path.suffix.lower() in _IMAGE_EXTENSIONS)

    def list_other_files(self) -> list[Path]:
        return self._list_files(
            lambda path: (
                path.name not in _KNOWN_LOG_FILENAMES
                and path.suffix.lower() not in _IMAGE_EXTENSIONS
            )
        )

    def _list_files(self, predicate: Callable[[Path], bool]) -> list[Path]:
        try:
            files = [
                child
                for child in self.path.iterdir()
                if child.is_file() and predicate(child)
            ]
        except OSError:
            return []
        return sorted(files)


def _read_data_buffer(data_path: Path) -> pyarrow.BufferReader:
    # Close the disk handle before Arrow parsing to reduce Windows replace races.
    return pyarrow.BufferReader(data_path.read_bytes())


def _inspect_data(
    data_path: Path,
    version: _FileVersion | None,
) -> tuple[int, tuple[str, ...]]:
    if version is None:
        return 0, ()
    with pyarrow.ipc.open_file(_read_data_buffer(data_path)) as reader:
        row_count = sum(
            reader.get_batch(index).num_rows
            for index in range(reader.num_record_batches)
        )
        columns = tuple(str(name) for name in reader.schema.names)
    return row_count, columns


def _refresh_record(path: Path, previous: LogRecord | None = None) -> LogRecord:
    if previous is not None:
        previous.meta.reload()

    data_version = _FileVersion.from_path(path / "data.feather")
    if previous is not None and previous.data_version == data_version:
        return previous

    try:
        row_count, columns = _inspect_data(path / "data.feather", data_version)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to inspect feather file %s: %s",
            path / "data.feather",
            exc,
        )
        row_count, columns = 0, ()
        data_version = _RETRY_VERSION
    return LogRecord(
        path=path,
        row_count=row_count,
        columns=columns,
        data_version=data_version,
    )


class LogCatalog:
    """Scan one parent directory and reuse unchanged records."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self._directory = Path(directory) if directory is not None else None
        self._records: dict[Path, LogRecord] = {}

    def refresh(self, directory: str | Path | None = None) -> list[LogRecord]:
        if directory is None:
            if self._directory is None:
                raise TypeError("directory is required on the first refresh")
            directory = self._directory
        directory = Path(directory)
        if self._directory != directory:
            self._directory = directory
            self._records.clear()

        if not directory.is_dir():
            self._records.clear()
            return []

        paths = {
            path for path in directory.iterdir() if (path / "metadata.json").is_file()
        }

        for removed_path in self._records.keys() - paths:
            del self._records[removed_path]

        for path in paths:
            self._records[path] = _refresh_record(path, self._records.get(path))

        return sorted(
            self._records.values(),
            key=lambda record: _log_name_sort_key(record.path.name),
        )


def _log_name_sort_key(name: str) -> tuple[int, int | str]:
    if name.isdecimal():
        return 0, int(name)
    return 1, name.casefold()


def _next_export_logfolder_path(parent_path: Path) -> Path:
    max_index = max(
        (
            int(entry.name)
            for entry in parent_path.iterdir()
            if entry.is_dir() and entry.name.isdecimal()
        ),
        default=-1,
    )
    next_index = max_index + 1
    while (parent_path / str(next_index)).exists():
        next_index += 1
    return parent_path / str(next_index)


def export_records(
    records: Iterable[LogRecord], destination_parent: str | Path
) -> list[Path]:
    """Copy selected records into newly numbered destination directories."""
    destination_parent = Path(destination_parent)
    destination_parent.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []
    for record in sorted(
        records,
        key=lambda item: _log_name_sort_key(item.path.name),
    ):
        target_path = _next_export_logfolder_path(destination_parent)
        target_path.mkdir(parents=True, exist_ok=False)
        shutil.copytree(record.path, target_path, dirs_exist_ok=True)
        import_from_path = target_path / "import_from"
        if not import_from_path.exists():
            import_from_path.write_text(str(record.path), encoding="utf-8")
        exported_paths.append(target_path)
    return exported_paths
