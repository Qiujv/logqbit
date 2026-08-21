"""Cached records and disk access for LogQbit catalogs."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import pandas as pd
import pyarrow
import pyarrow.ipc

from .file_version import FileVersion
from .metadata import LogMetadata
from .registry import Registry

__all__ = [
    "LogCatalog",
    "LogRecord",
    "PlotColumns",
    "resolve_plot_columns",
    "LOGFOLDER_SID_COLUMN",
    "MergeRecordsError",
    "MergeRecordsResult",
    "PreparedMerge",
    "append_records_into_record",
    "merge_records_into_new",
    "export_records",
]

logger = logging.getLogger(__name__)

_FLOAT_LOG_ID_PATTERN = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?")


_RETRY_VERSION = FileVersion(mtime_ns=-1, size=-1, inode=-1)


@dataclass(frozen=True)
class PlotColumns:
    """Plot roles resolved against the columns available in one record."""

    axes: tuple[str, ...]
    fields: tuple[str, ...]
    groupby: tuple[str, ...]
    ignored: tuple[str, ...]


def resolve_plot_columns(
    columns: Sequence[str],
    preferred_axes: Sequence[str],
    preferred_fields: Sequence[str],
    preferred_groupby: Sequence[str] = (),
) -> PlotColumns:
    """Resolve stored plot preferences into ordered, disjoint column roles."""

    columns = _ordered_unique(columns)
    available = set(columns)
    groupby = [
        column for column in _ordered_unique(preferred_groupby) if column in available
    ]
    groupby_set = set(groupby)
    axes = [
        column
        for column in _ordered_unique(preferred_axes)
        if column in available and column not in groupby_set
    ]
    axes_set = set(axes)
    fields = [
        column
        for column in _ordered_unique(preferred_fields)
        if column in available and column not in groupby_set and column not in axes_set
    ]
    fields_set = set(fields)
    ignored = [
        column
        for column in columns
        if column not in groupby_set
        and column not in axes_set
        and column not in fields_set
    ]

    if not axes and ignored:
        axes.append(ignored.pop(0))
    if not fields and ignored:
        fields.append(ignored.pop(0))
    return PlotColumns(tuple(axes), tuple(fields), tuple(groupby), tuple(ignored))


def _ordered_unique(items: Sequence[str]) -> tuple[str, ...]:
    if isinstance(items, str):
        return (items,)
    return tuple(dict.fromkeys(str(item) for item in items))


@dataclass(frozen=True)
class _DataSummary:
    data_version: FileVersion | None
    row_count: int
    columns: tuple[str, ...]


class LogRecord:
    """Handle to one existing log record.

    The standard record files are exposed through lazy accessors without
    creating a dataframe writer or missing files. Catalog summary data is
    inspected on demand and refreshed in place.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Log directory at '{path}' does not exist.")
        self._path = path
        self._data_summary: _DataSummary | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path!r})"

    @property
    def path(self) -> Path:
        """Directory containing this record's files."""
        return self._path

    @property
    def log_id(self) -> int | float | str:
        """Return ASCII integer and decimal directory names numerically."""
        return _parse_log_id(self.path.name)

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

    @cached_property
    def reg(self) -> Registry:
        """Return the existing ``const.yaml`` registry without creating it."""
        return Registry(self.const_path, create=False)

    @property
    def const(self) -> Registry:
        """Alias for :attr:`reg`."""
        return self.reg

    # Common metadata shortcuts delegate to LogMetadata's synchronized fields.
    @property
    def title(self) -> str:
        return self.meta.title

    @property
    def star(self) -> int:
        return self.meta.star

    @property
    def trash(self) -> bool:
        return self.meta.trash

    @property
    def row_count(self) -> int:
        """Number of rows in the current on-disk dataframe."""
        return self._require_data_summary().row_count

    @property
    def columns(self) -> tuple[str, ...]:
        """Column names in the current on-disk dataframe."""
        return self._require_data_summary().columns

    @property
    def data_version(self) -> FileVersion | None:
        """Filesystem identity of the inspected ``data.feather`` generation."""
        return self._require_data_summary().data_version

    @property
    def resolved_plot_columns(self) -> PlotColumns:
        """Return the effective plot roles for the current data columns."""
        return resolve_plot_columns(
            self.columns,
            self.meta.plot_axes,
            self.meta.plot_fields,
            self.meta.plot_groupby,
        )

    @property
    def create_time(self) -> str:
        return self.meta.create_time

    @property
    def create_machine(self) -> str:
        return self.meta.create_machine

    def read_dataframe(self) -> pd.DataFrame | None:
        """Read the current Feather file, bypassing the cached :attr:`df`."""
        if not self.data_path.exists():
            return None
        try:
            return pd.read_feather(_read_data_buffer(self.data_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to read feather file %s: %s", self.data_path, exc)
            return None

    @cached_property
    def df(self) -> pd.DataFrame | None:
        """Return a lazily cached in-memory snapshot of ``data.feather``.

        Mutating the returned dataframe changes only this cached object. Call
        :meth:`read_dataframe` to read the current file without using it.
        """
        version_before = FileVersion.from_path(self.data_path)
        dataframe = self.read_dataframe()
        version_after = FileVersion.from_path(self.data_path)
        if dataframe is not None:
            if version_before == version_after:
                self._data_summary = _DataSummary(
                    data_version=version_after,
                    row_count=len(dataframe),
                    columns=tuple(str(column) for column in dataframe.columns),
                )
            else:
                self._data_summary = None
        elif version_after is None:
            self._data_summary = _DataSummary(None, 0, ())
        return dataframe

    def refresh(self) -> Self:
        """Refresh cached file-backed state in place and return this record."""
        metadata = self.__dict__.get("meta")
        if metadata is not None:
            metadata.reload()
        self._refresh_data_summary()
        return self

    def _require_data_summary(self) -> _DataSummary:
        if self._data_summary is None:
            self._refresh_data_summary()
        assert self._data_summary is not None
        return self._data_summary

    def _refresh_data_summary(self) -> bool:
        data_version = FileVersion.from_path(self.data_path)
        previous = self._data_summary
        if previous is not None and previous.data_version == data_version:
            return False

        try:
            row_count, columns = _inspect_data(self.data_path, data_version)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to inspect feather file %s: %s", self.data_path, exc)
            row_count, columns = 0, ()
            data_version = _RETRY_VERSION

        self._data_summary = _DataSummary(data_version, row_count, columns)
        if previous is not None:
            self.__dict__.pop("df", None)
        return True


def _read_data_buffer(data_path: Path) -> pyarrow.BufferReader:
    # Close the disk handle before Arrow parsing to reduce Windows replace races.
    return pyarrow.BufferReader(data_path.read_bytes())


def _inspect_data(
    data_path: Path,
    version: FileVersion | None,
) -> tuple[int, tuple[str, ...]]:
    if version is None:
        return 0, ()
    # TODO: Benchmark a lightweight schema/count_rows path with before/after
    # FileVersion checks, including Windows replace behavior, so catalog
    # inspection does not need to buffer the entire Feather file.
    with pyarrow.ipc.open_file(_read_data_buffer(data_path)) as reader:
        row_count = sum(
            reader.get_batch(index).num_rows
            for index in range(reader.num_record_batches)
        )
        columns = tuple(str(name) for name in reader.schema.names)
    return row_count, columns


class LogCatalog:
    """Scan one parent directory and reuse unchanged records."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self._directory = Path(directory) if directory is not None else None
        self._records: dict[Path, LogRecord] = {}

    def refresh(self, directory: str | Path | None = None) -> list[LogRecord]:
        """Refresh records, ordered by numeric then named directory."""
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
            record = self._records.get(path)
            if record is None:
                record = LogRecord(path)
                self._records[path] = record
            record.refresh()

        return sorted(
            self._records.values(),
            key=lambda record: _log_name_sort_key(record.path.name),
        )


LOGFOLDER_SID_COLUMN = "logfolder_sid"


class MergeRecordsError(RuntimeError):
    """Raised when selected records cannot be merged safely."""


@dataclass(frozen=True)
class MergeRecordsResult:
    """Summary of one completed record merge."""

    path: Path
    row_count: int
    appended_records: int
    skipped_records: int
    created: bool


@dataclass(frozen=True)
class _LoadedMergeRecord:
    record: LogRecord
    dataframe: pd.DataFrame
    version: FileVersion
    source_ids: frozenset[str]
    had_sid_column: bool


@dataclass(frozen=True)
class PreparedMerge:
    """Validated in-memory merge awaiting publication."""

    dataframe: pd.DataFrame
    loaded_records: tuple[_LoadedMergeRecord, ...]
    source: LogRecord
    target: LogRecord | None
    destination_parent: Path | None
    appended_records: int
    skipped_records: int
    staging_path: Path | None

    @property
    def is_noop(self) -> bool:
        return self.target is not None and self.appended_records == 0

    @property
    def row_count(self) -> int:
        return len(self.dataframe)

    @classmethod
    def for_new_folder(
        cls,
        records: Iterable[LogRecord],
        destination_parent: str | Path,
    ) -> PreparedMerge:
        """Prepare selected records for publication as a new LogFolder."""
        records = _validate_merge_records(records)
        loaded = _load_merge_records(records)
        _require_common_merge_columns(loaded)
        _reject_overlapping_sources(loaded)
        dataframe = pd.concat(
            [item.dataframe for item in loaded],
            ignore_index=True,
        )
        _require_unchanged_merge_sources(loaded)
        destination_parent = Path(destination_parent)
        staging_path = _stage_new_merged_logfolder(
            destination_parent,
            dataframe,
            records[0],
        )
        return cls(
            dataframe=dataframe,
            loaded_records=tuple(loaded),
            source=records[0],
            target=None,
            destination_parent=destination_parent,
            appended_records=len(records),
            skipped_records=0,
            staging_path=staging_path,
        )

    @classmethod
    def for_append(
        cls,
        records: Iterable[LogRecord],
    ) -> PreparedMerge:
        """Prepare records to append to their sole existing aggregate."""
        records = _validate_merge_records(records)
        target = cls.find_append_target(records)
        if target is None:
            raise MergeRecordsError(
                f"Append requires exactly one selected record with a "
                f"{LOGFOLDER_SID_COLUMN} column."
            )

        loaded = _load_merge_records(records)
        _require_common_merge_columns(loaded)
        loaded_target = next(item for item in loaded if item.record.path == target.path)
        sid_records = [item for item in loaded if item.had_sid_column]
        if len(sid_records) != 1 or sid_records[0] is not loaded_target:
            raise MergeRecordsError(
                f"Append requires exactly one selected record with a "
                f"{LOGFOLDER_SID_COLUMN} column."
            )

        source_ids = set(loaded_target.source_ids)
        frames = [loaded_target.dataframe]
        appended_records = 0
        skipped_records = 0
        for item in loaded:
            if item is loaded_target:
                continue
            if item.source_ids <= source_ids:
                skipped_records += 1
                continue
            overlap = item.source_ids & source_ids
            if overlap:
                overlap_text = ", ".join(sorted(overlap))
                raise MergeRecordsError(
                    f"Record #{item.record.log_id} partially overlaps the target "
                    f"through source ID(s): {overlap_text}."
                )
            frames.append(item.dataframe)
            source_ids.update(item.source_ids)
            appended_records += 1

        if appended_records == 0:
            _require_unchanged_merge_sources(loaded)
            return cls(
                dataframe=loaded_target.dataframe,
                loaded_records=tuple(loaded),
                source=target,
                target=target,
                destination_parent=None,
                appended_records=0,
                skipped_records=skipped_records,
                staging_path=None,
            )

        dataframe = pd.concat(frames, ignore_index=True)
        _require_unchanged_merge_sources(loaded)
        staging_path = _stage_dataframe(dataframe, target.data_path)
        return cls(
            dataframe=dataframe,
            loaded_records=tuple(loaded),
            source=target,
            target=target,
            destination_parent=None,
            appended_records=appended_records,
            skipped_records=skipped_records,
            staging_path=staging_path,
        )

    @staticmethod
    def find_append_target(records: Iterable[LogRecord]) -> LogRecord | None:
        """Find the sole selected aggregate using cached column summaries."""
        records = list(records)
        if len(records) < 2:
            return None
        targets = [
            record for record in records if LOGFOLDER_SID_COLUMN in record.columns
        ]
        return targets[0] if len(targets) == 1 else None

    def publish(self) -> MergeRecordsResult:
        """Publish this prepared merge after rechecking every source version."""
        if self.target is None:
            return self._publish_new_folder()
        return self._publish_append()

    def discard(self) -> None:
        """Remove temporary files retained by an unpublished merge."""
        if self.staging_path is None:
            return
        if self.staging_path.is_dir():
            shutil.rmtree(self.staging_path, ignore_errors=True)
            return
        try:
            self.staging_path.unlink()
        except FileNotFoundError:
            pass

    def _publish_new_folder(self) -> MergeRecordsResult:
        if self.staging_path is None:
            raise MergeRecordsError("The prepared merge has no staged LogFolder.")
        if self.destination_parent is None:
            raise MergeRecordsError("The prepared merge has no destination directory.")
        _require_unchanged_merge_sources(self.loaded_records)
        self.destination_parent.mkdir(parents=True, exist_ok=True)
        target_path = _publish_staged_logfolder(
            self.staging_path,
            self.destination_parent,
            before_publish=lambda: _require_unchanged_merge_sources(
                self.loaded_records
            ),
        )
        return MergeRecordsResult(
            path=target_path,
            row_count=self.row_count,
            appended_records=self.appended_records,
            skipped_records=self.skipped_records,
            created=True,
        )

    def _publish_append(self) -> MergeRecordsResult:
        assert self.target is not None
        if self.is_noop:
            _require_unchanged_merge_sources(self.loaded_records)
            return MergeRecordsResult(
                path=self.target.path,
                row_count=self.row_count,
                appended_records=0,
                skipped_records=self.skipped_records,
                created=False,
            )
        if self.staging_path is None:
            raise MergeRecordsError("The prepared append has no staged data file.")

        _require_unchanged_merge_sources(self.loaded_records)
        _replace_staged_file(
            self.staging_path,
            self.target.data_path,
            before_replace=lambda: _require_unchanged_merge_sources(
                self.loaded_records
            ),
        )
        target_metadata = LogMetadata(
            self.target.meta_path,
            create=False,
            default_on_error=True,
        )
        if not target_metadata.plot_groupby:
            target_metadata.plot_groupby = (LOGFOLDER_SID_COLUMN,)
        return MergeRecordsResult(
            path=self.target.path,
            row_count=self.row_count,
            appended_records=self.appended_records,
            skipped_records=self.skipped_records,
            created=False,
        )


def merge_records_into_new(
    records: Iterable[LogRecord],
    destination_parent: str | Path,
) -> MergeRecordsResult:
    """Merge records into a newly numbered LogFolder directory."""
    prepared = PreparedMerge.for_new_folder(records, destination_parent)
    try:
        return prepared.publish()
    finally:
        prepared.discard()


def append_records_into_record(
    records: Iterable[LogRecord],
) -> MergeRecordsResult:
    """Append records to the selected aggregate containing source IDs."""
    prepared = PreparedMerge.for_append(records)
    try:
        return prepared.publish()
    finally:
        prepared.discard()


def _validate_merge_records(records: Iterable[LogRecord]) -> list[LogRecord]:
    records = list(records)
    if len(records) < 2:
        raise MergeRecordsError("Select at least two records to merge.")
    paths = [record.path.resolve() for record in records]
    if len(set(paths)) != len(paths):
        raise MergeRecordsError("The selected records must be distinct.")
    return records


def _load_merge_records(
    records: Sequence[LogRecord],
) -> list[_LoadedMergeRecord]:
    loaded: list[_LoadedMergeRecord] = []
    for record in records:
        version = FileVersion.from_path(record.data_path)
        if version is None:
            raise MergeRecordsError(f"Record #{record.log_id} has no data file.")
        dataframe = record.read_dataframe()
        if dataframe is None:
            raise MergeRecordsError(
                f"Could not read data from record #{record.log_id}."
            )
        if FileVersion.from_path(record.data_path) != version:
            raise MergeRecordsError(
                f"Record #{record.log_id} changed while it was being read. Try again."
            )

        sid_column_count = sum(
            column == LOGFOLDER_SID_COLUMN for column in dataframe.columns
        )
        if sid_column_count > 1:
            raise MergeRecordsError(
                f"Record #{record.log_id} has multiple {LOGFOLDER_SID_COLUMN} columns."
            )
        had_sid_column = sid_column_count == 1
        dataframe = dataframe.copy()
        if had_sid_column:
            dataframe[LOGFOLDER_SID_COLUMN] = dataframe[LOGFOLDER_SID_COLUMN].astype(
                "string"
            )
            source_ids = frozenset(
                str(value)
                for value in pd.unique(
                    dataframe[LOGFOLDER_SID_COLUMN].dropna()
                ).tolist()
            )
        else:
            source_id = str(record.log_id)
            dataframe[LOGFOLDER_SID_COLUMN] = source_id
            source_ids = frozenset((source_id,))
        loaded.append(
            _LoadedMergeRecord(
                record=record,
                dataframe=dataframe,
                version=version,
                source_ids=source_ids,
                had_sid_column=had_sid_column,
            )
        )
    return loaded


def _reject_overlapping_sources(records: Sequence[_LoadedMergeRecord]) -> None:
    owners: dict[str, LogRecord] = {}
    for item in records:
        for source_id in item.source_ids:
            previous = owners.get(source_id)
            if previous is not None:
                raise MergeRecordsError(
                    f"Records #{previous.log_id} and #{item.record.log_id} both "
                    f"contain source ID {source_id!r}."
                )
            owners[source_id] = item.record


def _require_common_merge_columns(
    records: Sequence[_LoadedMergeRecord],
) -> None:
    common_columns = set(records[0].dataframe.columns)
    for item in records[1:]:
        common_columns.intersection_update(item.dataframe.columns)
    common_columns.discard(LOGFOLDER_SID_COLUMN)
    if len(common_columns) < 2:
        common_text = ", ".join(sorted(str(column) for column in common_columns))
        detail = common_text or "none"
        raise MergeRecordsError(
            "The selected records need at least two common data columns "
            f"(found: {detail})."
        )


def _require_unchanged_merge_sources(
    records: Sequence[_LoadedMergeRecord],
) -> None:
    for item in records:
        if FileVersion.from_path(item.record.data_path) != item.version:
            raise MergeRecordsError(
                f"Record #{item.record.log_id} changed during the merge. Try again."
            )


def _stage_new_merged_logfolder(
    parent: Path,
    dataframe: pd.DataFrame,
    source: LogRecord,
) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=".logqbit-merge-", dir=parent))
    try:
        dataframe.to_feather(staging_path / "data.feather")
        if source.const_path.is_file():
            shutil.copy2(source.const_path, staging_path / "const.yaml")
        source_metadata = LogMetadata(
            source.meta_path,
            create=False,
            default_on_error=True,
        )
        metadata = LogMetadata(
            staging_path / "metadata.json.pending",
            source_metadata.title,
        )
        metadata.update(
            plot_axes=source_metadata.plot_axes,
            plot_fields=source_metadata.plot_fields,
            plot_groupby=(LOGFOLDER_SID_COLUMN,),
        )
        return staging_path
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def _publish_staged_logfolder(
    staging_path: Path,
    parent: Path,
    *,
    before_publish: Callable[[], None],
) -> Path:
    target_path: Path | None = None
    try:
        before_publish()
        while True:
            candidate_path = _next_export_logfolder_path(parent)
            try:
                candidate_path.mkdir(exist_ok=False)
            except FileExistsError:  # pragma: no cover - concurrent allocation
                continue
            target_path = candidate_path
            break
        for source_name, target_name in (
            ("data.feather", "data.feather"),
            ("const.yaml", "const.yaml"),
            ("metadata.json.pending", "metadata.json"),
        ):
            source_path = staging_path / source_name
            if source_path.exists():
                source_path.replace(target_path / target_name)
        return target_path
    except Exception:
        if target_path is not None and target_path.exists():
            shutil.rmtree(target_path)
        raise
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)


def _stage_dataframe(dataframe: pd.DataFrame, target_path: Path) -> Path:
    staging_path = target_path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    try:
        dataframe.to_feather(staging_path)
    except Exception:
        try:
            staging_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return staging_path


def _replace_staged_file(
    staging_path: Path,
    target_path: Path,
    *,
    before_replace: Callable[[], None],
) -> None:
    retry_delay = 0.1
    for attempt in range(3):
        before_replace()
        try:
            staging_path.replace(target_path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(retry_delay)
            retry_delay *= 2


def _parse_log_id(name: str) -> int | float | str:
    if name.isdecimal() and name.isascii():
        return int(name)
    if _FLOAT_LOG_ID_PATTERN.fullmatch(name):
        return float(name)
    return name


def _log_name_sort_key(
    name: str,
) -> tuple[int, int | float, str] | tuple[int, str, str]:
    log_id = _parse_log_id(name)
    if isinstance(log_id, (int, float)):
        return 0, log_id, name
    return 1, name.casefold(), name


def _next_export_logfolder_path(parent_path: Path) -> Path:
    max_index = max(
        (
            int(entry.name)
            for entry in parent_path.iterdir()
            if entry.is_dir() and entry.name.isdecimal() and entry.name.isascii()
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
