"""Filesystem-backed records and caching for the log browser."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.ipc

try:  # pragma: no cover - fallback for direct browser execution
    from ..metadata import LogMetadata
except ImportError:  # pragma: no cover
    from logqbit.metadata import LogMetadata  # type: ignore

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
KNOWN_RECORD_FILENAMES = {"const.yaml", "data.feather", "metadata.json"}


def _next_export_logfolder_path(parent_path: Path) -> Path:
    parent_path = Path(parent_path)
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
    records: Iterable[LogRecord], destination_parent: Path
) -> list[Path]:
    destination_parent = Path(destination_parent)
    destination_parent.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []
    for record in sorted(records, key=lambda item: item.log_id):
        target_path = _next_export_logfolder_path(destination_parent)
        target_path.mkdir(parents=True, exist_ok=False)
        shutil.copytree(record.path, target_path, dirs_exist_ok=True)
        import_from_path = target_path / "import_from"
        if not import_from_path.exists():
            import_from_path.write_text(str(record.path), encoding="utf-8")
        exported_paths.append(target_path)
    return exported_paths


@dataclass(frozen=True)
class FileVersion:
    """Cheap identity for one on-disk file generation."""

    mtime_ns: int
    size: int
    inode: int

    @classmethod
    def from_path(cls, path: Path) -> FileVersion | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return cls(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            inode=stat.st_ino,
        )


@dataclass
class LogDetailData:
    """Lazy full-data cache for one log detail view."""

    dataframe: pd.DataFrame | None = field(default=None, repr=False)
    loaded_version: FileVersion | None = None

    def clear(self) -> None:
        self.dataframe = None
        self.loaded_version = None

    def load(
        self, path: Path | None, version: FileVersion | None
    ) -> pd.DataFrame | None:
        if (
            self.dataframe is not None
            and self.loaded_version == version
            and version is not None
        ):
            return self.dataframe
        if path is None or version is None:
            self.clear()
            return None

        dataframe = pd.read_feather(path)
        self.dataframe = dataframe
        self.loaded_version = version
        return dataframe


@dataclass
class LogRecord:
    """Lightweight catalog entry with a separate lazy detail-data cache."""

    log_id: int
    path: Path
    data_path: Path | None = None
    yaml_path: Path | None = None

    row_count: int = 0
    columns: list[str] = field(default_factory=list)

    title: str = "untitled"
    star: int = 0
    trash: bool = False
    plot_axes: list[str] = field(default_factory=list)
    create_time: str = ""
    create_machine: str = ""

    data_version: FileVersion | None = field(default=None, repr=False)
    metadata_version: FileVersion | None = field(default=None, repr=False)
    _detail_data: LogDetailData = field(
        default_factory=LogDetailData, init=False, repr=False
    )

    meta: LogMetadata = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.meta = LogMetadata(self.path / "metadata.json", create=True)
        self.refresh_metadata(force=True)
        if self.data_path is not None and self.data_version is None:
            self.data_version = FileVersion.from_path(self.data_path)

    @property
    def data_frame(self) -> pd.DataFrame | None:
        """Compatibility accessor for code that previously owned this cache."""
        return self._detail_data.dataframe

    @data_frame.setter
    def data_frame(self, value: pd.DataFrame | None) -> None:
        if value is None:
            self._detail_data.clear()
            return
        self._detail_data.dataframe = value
        self._detail_data.loaded_version = self.data_version

    def refresh_metadata(self, *, force: bool = False) -> bool:
        version = FileVersion.from_path(self.meta.path)
        if not force and version == self.metadata_version:
            return False

        self.meta.reload()
        root = self.meta.root
        self.title = str(root.get("title", "untitled"))
        self.star = int(root.get("star", 0))
        self.trash = bool(root.get("trash", False))
        self.plot_axes = [str(item) for item in root.get("plot_axes", [])]
        self.create_time = str(root.get("create_time", ""))
        self.create_machine = str(root.get("create_machine", ""))
        self.metadata_version = version
        return True

    def refresh_from_disk(self, *, inspect_data: bool = True) -> bool:
        """Refresh lightweight summary fields, re-reading Feather only if changed."""
        self.yaml_path = self.path / "const.yaml"
        if not self.yaml_path.exists():
            self.yaml_path = None

        data_path = self.path / "data.feather"
        new_version = FileVersion.from_path(data_path)
        self.data_path = data_path if new_version is not None else None
        data_changed = new_version != self.data_version

        if data_changed:
            self.data_version = new_version
            self._detail_data.clear()
            if self.data_path is None:
                self.row_count = 0
                self.columns = []
            elif inspect_data:
                try:
                    with pyarrow.ipc.open_file(self.data_path) as reader:
                        self.row_count = sum(
                            reader.get_batch(i).num_rows
                            for i in range(reader.num_record_batches)
                        )
                        self.columns = [str(name) for name in reader.schema.names]
                except FileNotFoundError:
                    self.data_path = None
                    self.data_version = None
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Failed to inspect feather file %s: %s",
                        self.data_path,
                        exc,
                    )

        return self.refresh_metadata() or data_changed

    def load_dataframe(self) -> pd.DataFrame | None:
        try:
            if self.data_path is not None:
                current_version = FileVersion.from_path(self.data_path)
                if current_version != self.data_version:
                    self.data_version = current_version
                    self._detail_data.clear()
            dataframe = self._detail_data.load(self.data_path, self.data_version)
            if dataframe is not None:
                self.row_count = len(dataframe)
                self.columns = [str(col) for col in dataframe.columns]
            return dataframe
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to read feather file %s: %s", self.data_path, exc)
            return None

    def read_yaml_text(self) -> str:
        if not self.yaml_path or not self.yaml_path.exists():
            return "const.yaml not found."
        try:
            text = self.yaml_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to read yaml file %s: %s", self.yaml_path, exc)
            return f"Failed to read const.yaml: {exc}"
        return text if text.strip() else "(const.yaml is empty)"

    def list_image_files(self) -> list[Path]:
        files: list[Path] = []
        try:
            children = list(self.path.iterdir())
        except OSError:
            return files
        for child in children:
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(child)
        files.sort()
        return files

    def list_other_files(self) -> list[Path]:
        files: list[Path] = []
        try:
            children = list(self.path.iterdir())
        except OSError:
            return files
        for child in children:
            if not child.is_file():
                continue
            if child.name in KNOWN_RECORD_FILENAMES:
                continue
            if child.suffix.lower() in IMAGE_EXTENSIONS:
                continue
            files.append(child)
        files.sort()
        return files

    @staticmethod
    def scan_directory(directory: Path) -> list[LogRecord]:
        return LogCatalog().refresh(directory)


class LogCatalog:
    """In-memory catalog that preserves unchanged log entries across refreshes."""

    def __init__(self) -> None:
        self._directory: Path | None = None
        self._records: dict[Path, LogRecord] = {}

    def refresh(
        self,
        directory: Path,
        *,
        skip_data_inspection_for: Path | None = None,
    ) -> list[LogRecord]:
        directory = Path(directory)
        if self._directory != directory:
            self._directory = directory
            self._records.clear()

        if not directory.exists() or not directory.is_dir():
            self._records.clear()
            return []

        paths = {
            path
            for path in directory.iterdir()
            if path.is_dir() and path.name.isdigit()
        }

        for removed_path in self._records.keys() - paths:
            del self._records[removed_path]

        for path in paths:
            record = self._records.get(path)
            if record is None:
                record = LogRecord(log_id=int(path.name), path=path)
                self._records[path] = record
            record.refresh_from_disk(
                inspect_data=path != skip_data_inspection_for
            )

        return sorted(self._records.values(), key=lambda record: record.log_id)
