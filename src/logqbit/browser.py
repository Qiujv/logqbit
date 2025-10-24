"""Interactive browser for experiment log folders."""

from __future__ import annotations

import logging
import numbers
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PySide6.QtCore import (
    QAbstractTableModel,
    QFileSystemWatcher,
    QModelIndex,
    Qt,
    QTimer,
)
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - fallback for direct execution
    from .index import LogIndex
except ImportError:  # pragma: no cover - fallback for direct execution
    from index import LogIndex  # type: ignore

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_TABLE_ROWS = 50000
REFRESH_DEBOUNCE_MS = 250

COL_ID = 0
COL_TITLE = 1
COL_ROWS = 2
COL_CREATE_TIME = 3
COL_CREATE_MACHINE = 4
COL_PLOT_AXES = 5


@dataclass
class IndexInfo:
    star: int = 0
    trash: bool = False
    title: str = ""
    create_time: str = ""
    create_machine: str = ""
    plot_axes: List[str] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    experiment_id: int
    path: Path
    index_info: IndexInfo
    row_count: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    plot_axes: List[str] = field(default_factory=list)
    data_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    index_path: Optional[Path] = None
    data_preview: Optional[pd.DataFrame] = field(default=None, repr=False)
    data_frame: Optional[pd.DataFrame] = field(default=None, repr=False)
    data_loaded_fully: bool = False


def _read_index_file(path: Path) -> IndexInfo:
    info = IndexInfo()
    if path is None or not path.exists():
        return info
    return _read_json_index(path)


def _write_index_file(path: Path, info: IndexInfo) -> None:
    _write_json_index(path, info)


def _read_json_index(path: Path) -> IndexInfo:
    info = IndexInfo()
    try:
        log_index = LogIndex(path, create=False)
    except FileNotFoundError:
        return info
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load index.json %s: %s", path, exc)
        return info
    try:
        log_index.reload()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Reload index %s failed: %s", path, exc)
    data = log_index.root
    info.title = str(data.get("title", ""))
    star_value = data.get("star")
    if star_value is None and "starred" in data:
        star_value = data.get("starred")
    try:
        info.star = int(star_value) if star_value is not None else 0
    except (TypeError, ValueError):
        info.star = 1 if star_value else 0
    trash_value = data.get("trash")
    if trash_value is None and "trashed" in data:
        trash_value = data.get("trashed")
    info.trash = bool(trash_value)
    info.create_time = str(data.get("create_time", ""))
    info.create_machine = str(data.get("create_machine", ""))
    axes_value = data.get("plot_axes", [])
    if isinstance(axes_value, list):
        info.plot_axes = [str(item) for item in axes_value]
    elif axes_value is None:
        info.plot_axes = []
    else:
        info.plot_axes = [str(axes_value)]
    return info


def _write_json_index(path: Path, info: IndexInfo) -> None:
    create = not path.exists()
    title = info.title or "untitled"
    log_index = LogIndex(path, title=title, create=True)
    try:
        log_index.reload()
    except Exception:  # pragma: no cover - defensive
        pass
    if info.title:
        log_index.root["title"] = info.title
    elif create:
        log_index.root.setdefault("title", title)
    log_index.root["star"] = int(info.star)
    log_index.root["trash"] = bool(info.trash)
    log_index.root["plot_axes"] = [str(item) for item in info.plot_axes]
    log_index.save(timeout=1.0)


def _detect_data_file(folder: Path) -> Optional[Path]:
    for candidate in ("data.parquent", "data.parquet"):
        path = folder / candidate
        if path.exists():
            return path
    return None


def _detect_meta_file(folder: Path) -> Optional[Path]:
    path = folder / "meta.yaml"
    return path if path.exists() else None


def _detect_index_file(folder: Path) -> Optional[Path]:
    json_path = folder / "index.json"
    return json_path if json_path.exists() else None


def _list_image_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for child in folder.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(child)
    files.sort()
    return files


def _read_meta_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return "meta.yaml not found."
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to read meta file %s: %s", path, exc)
        return f"Failed to read meta.yaml: {exc}"
    return text if text.strip() else "(meta.yaml is empty)"


def _read_parquet_summary(data_path: Optional[Path]) -> tuple[Optional[int], List[str]]:
    if not data_path:
        return None, []
    try:
        parquet_file = pq.ParquetFile(data_path)
        metadata = parquet_file.metadata
        row_count = metadata.num_rows if metadata is not None else None
        columns = (
            list(parquet_file.schema.names) if parquet_file.schema is not None else []
        )
        return row_count, columns
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to inspect parquet file %s: %s", data_path, exc)
        return None, []


def _load_dataframe(data_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not data_path:
        return None
    try:
        df = pd.read_parquet(data_path)
        if len(df) > MAX_TABLE_ROWS:
            return df.head(MAX_TABLE_ROWS)
        return df
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to read parquet file %s: %s", data_path, exc)
        return None


def _load_dataframe_preview(
    data_path: Optional[Path], limit: int
) -> Optional[pd.DataFrame]:
    if not data_path or limit <= 0:
        return None
    remaining = limit
    tables: List[pa.Table] = []
    try:
        parquet_file = pq.ParquetFile(data_path)
        for group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(group_index)
            if table is None or table.num_rows == 0:
                continue
            if table.num_rows > remaining:
                table = table.slice(0, remaining)
            tables.append(table)
            remaining -= table.num_rows
            if remaining <= 0:
                break
        if not tables:
            return pd.DataFrame()
        preview_table = (
            tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote=True)
        )
        return preview_table.to_pandas()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to read parquet preview %s: %s", data_path, exc)
        return None


class PandasTableModel(QAbstractTableModel):
    def __init__(
        self,
        frame: pd.DataFrame,
        parent: Optional[QWidget] = None,
        highlight_columns: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(parent)
        self._frame = frame
        self._highlight = (
            {str(name) for name in highlight_columns} if highlight_columns else set()
        )
        self._bold_font = QFont(parent.font() if parent else QFont())
        self._bold_font.setBold(True)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._frame.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: D401
        if not index.isValid():
            return None
        column_name = str(self._frame.columns[index.column()])
        if role == Qt.FontRole and column_name in self._highlight:
            return self._bold_font
        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self._frame.iat[index.row(), index.column()]
        if pd.isna(value):
            return ""
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            try:
                return format(value, ".6g")
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ):
        if role == Qt.FontRole and orientation == Qt.Horizontal:
            column_name = str(self._frame.columns[section])
            if column_name in self._highlight:
                return self._bold_font
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._frame.columns[section])
        return str(self._frame.index[section])


class ScaledImageLabel(QLabel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)

    def load_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = None
            self.setText(f"Failed to load {path.name}")
            return False
        self._pixmap = pixmap
        self.setText("")
        self._update_scaled_pixmap()
        return True

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override naming
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if not self._pixmap or self._pixmap.isNull():
            return
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        scaled = self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled)


class LogBrowserWindow(QMainWindow):
    _open_windows: List["LogBrowserWindow"] = []

    def __init__(
        self, directory: Optional[Path] = None, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Logqbit Log Browser")
        self.setWindowIcon(
            QIcon(QPixmap(str(files("logqbit") / "assets" / "browser.svg")))
        )
        self.resize(1200, 700)

        self._base_dir = Path(directory) if directory else Path.cwd()
        self._records: List[ExperimentRecord] = []
        self._current_record: Optional[ExperimentRecord] = None
        self._list_refresh_pending = False
        self._detail_refresh_pending = False
        self._show_trash = True
        self._suppress_item_changed = False
        self._image_tab_indices: List[int] = []
        self._sort_column = COL_ID
        self._sort_order = Qt.AscendingOrder
        self._suppress_sort_sync = False
        self._data_preview_limit = 100
        self._data_preview_active = False
        self._data_preview_cap = 100
        self._shortcuts: List[QAction] = []

        self._dir_watcher = QFileSystemWatcher(self)
        self._detail_watcher = QFileSystemWatcher(self)
        self._dir_watcher.directoryChanged.connect(self._schedule_list_refresh)
        self._dir_watcher.fileChanged.connect(self._schedule_list_refresh)
        self._detail_watcher.directoryChanged.connect(self._schedule_detail_refresh)
        self._detail_watcher.fileChanged.connect(self._schedule_detail_refresh)

        self._build_ui()
        self._sync_directory_watcher()
        self.refresh_experiments()

        LogBrowserWindow._open_windows.append(self)
        self.destroyed.connect(self._on_destroyed)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_bar = QHBoxLayout()
        self.directory_label = QLabel(str(self._base_dir))
        self.directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.directory_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        change_button = QPushButton("Change...")
        refresh_button = QPushButton("Refresh")
        change_button.clicked.connect(self._on_change_directory_clicked)
        refresh_button.clicked.connect(self._on_refresh_clicked)
        top_bar.addWidget(QLabel("Directory:"))
        top_bar.addWidget(self.directory_label)
        top_bar.addWidget(change_button)
        top_bar.addWidget(refresh_button)
        layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal, central)

        self.experiment_table = QTableWidget(0, 6, splitter)
        self.experiment_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Title",
                "Rows",
                "Create Time",
                "Create Machine",
                "Plot Axes",
            ]
        )
        self.experiment_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.experiment_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.experiment_table.verticalHeader().setVisible(False)
        header = self.experiment_table.horizontalHeader()
        header.setSectionResizeMode(COL_ID, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_TITLE, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_ROWS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_CREATE_TIME, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_CREATE_MACHINE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_PLOT_AXES, QHeaderView.ResizeToContents)
        header.setSectionsClickable(True)
        header.setSortIndicator(self._sort_column, self._sort_order)
        header.setSortIndicatorShown(True)
        header.sortIndicatorChanged.connect(self._on_sort_indicator_changed)
        self.experiment_table.itemSelectionChanged.connect(
            self._on_experiment_selection_changed
        )
        self.experiment_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.experiment_table.customContextMenuRequested.connect(
            self._open_table_context_menu
        )
        self.experiment_table.setSortingEnabled(True)
        self.experiment_table.setColumnHidden(COL_CREATE_TIME, True)
        self.experiment_table.setColumnHidden(COL_CREATE_MACHINE, True)
        self.experiment_table.setColumnHidden(COL_PLOT_AXES, True)

        detail_widget = QWidget(splitter)
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(6)

        detail_top = QHBoxLayout()
        self.detail_label = QLabel("No experiment selected.")
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        detail_top.addWidget(self.detail_label)
        detail_top.addStretch(1)
        detail_layout.addLayout(detail_top)

        self.tab_widget = QTabWidget(detail_widget)
        self.meta_view = QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.tab_widget.addTab(self.meta_view, "Meta")

        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        data_layout.setContentsMargins(4, 4, 4, 4)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self.data_status_label = QLabel("")
        self.data_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_status_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self.data_load_button = QPushButton("Show More Rows")
        self.data_load_button.setEnabled(False)
        self.data_load_button.clicked.connect(self._on_load_all_data_clicked)
        controls.addWidget(self.data_status_label)
        controls.addStretch(1)
        controls.addWidget(self.data_load_button)
        data_layout.addLayout(controls)

        self.data_table = QTableView()
        self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data_table.setSortingEnabled(False)
        self.data_table.setWordWrap(False)
        self.data_table.horizontalHeader().setStretchLastSection(False)
        self.data_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        row_height = self.data_table.fontMetrics().height() + 6
        self.data_table.verticalHeader().setDefaultSectionSize(row_height)
        self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.customContextMenuRequested.connect(self._open_data_context_menu)
        data_layout.addWidget(self.data_table)
        self.tab_widget.addTab(data_tab, "Data")

        detail_layout.addWidget(self.tab_widget)

        splitter.addWidget(self.experiment_table)
        splitter.addWidget(detail_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 600])

        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self._setup_shortcuts()

    def _on_destroyed(self) -> None:
        try:
            LogBrowserWindow._open_windows.remove(self)
        except ValueError:
            pass

    def _open_new_window(self) -> None:
        window = LogBrowserWindow(self._base_dir)
        window.show()

    def _sync_directory_watcher(self) -> None:
        try:
            if self._dir_watcher.directories():
                self._dir_watcher.removePaths(self._dir_watcher.directories())
        except Exception:  # pragma: no cover - defensive
            pass
        if self._base_dir.exists():
            self._dir_watcher.addPath(str(self._base_dir))

    def _schedule_list_refresh(self) -> None:
        if self._list_refresh_pending:
            return
        self._list_refresh_pending = True
        QTimer.singleShot(REFRESH_DEBOUNCE_MS, self._run_list_refresh)

    def _run_list_refresh(self) -> None:
        self._list_refresh_pending = False
        self.refresh_experiments()

    def _schedule_detail_refresh(self) -> None:
        if self._detail_refresh_pending:
            return
        self._detail_refresh_pending = True
        QTimer.singleShot(REFRESH_DEBOUNCE_MS, self._run_detail_refresh)

    def _run_detail_refresh(self) -> None:
        self._detail_refresh_pending = False
        self.refresh_current_experiment()

    def _on_change_directory_clicked(self) -> None:
        current = str(self._base_dir)
        chosen = QFileDialog.getExistingDirectory(self, "Select log directory", current)
        if chosen:
            self.set_directory(Path(chosen))

    def set_directory(self, directory: Path) -> None:
        if directory == self._base_dir:
            return
        self._base_dir = directory
        self.directory_label.setText(str(self._base_dir))
        self._sync_directory_watcher()
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        previous_id = (
            self._current_record.experiment_id if self._current_record else None
        )
        records = self._scan_directory(self._base_dir)
        self._records = records
        self._populate_experiment_table(records)
        row_count = self.experiment_table.rowCount()
        if row_count:
            self.detail_label.setText("Select an experiment to preview.")
            if previous_id is not None and self._select_experiment_by_id(previous_id):
                return
            if row_count:
                self.experiment_table.setCurrentCell(0, COL_ID)
        else:
            if records:
                self.detail_label.setText("No experiments to display.")
            else:
                self.detail_label.setText("No experiments found.")
            self._current_record = None
            self.experiment_table.clearSelection()
            self._clear_preview_panels()
            self._clear_detail_watcher()

    def _select_experiment_by_id(self, experiment_id: int) -> bool:
        table = self.experiment_table
        for row in range(table.rowCount()):
            record = self._record_for_row(row)
            if record and record.experiment_id == experiment_id:
                table.blockSignals(True)
                table.selectRow(row)
                table.setCurrentCell(row, COL_ID)
                table.blockSignals(False)
                self._current_record = record
                self._load_experiment(record)
                return True
        return False

    def refresh_current_experiment(self) -> None:
        if not self._current_record:
            return
        self._load_experiment(self._current_record)

    def _scan_directory(self, directory: Path) -> List[ExperimentRecord]:
        records: List[ExperimentRecord] = []
        if not directory.exists() or not directory.is_dir():
            return records
        for entry in directory.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.isdigit():
                continue
            exp_id = int(entry.name)
            meta = _detect_meta_file(entry)
            data_file = _detect_data_file(entry)
            index_path = _detect_index_file(entry)
            index_info = _read_index_file(index_path) if index_path else IndexInfo()
            row_count, columns = _read_parquet_summary(data_file)
            record = ExperimentRecord(
                experiment_id=exp_id,
                path=entry,
                index_info=index_info,
                row_count=row_count,
                columns=columns,
                plot_axes=index_info.plot_axes,
                data_path=data_file,
                meta_path=meta,
                index_path=index_path,
            )
            records.append(record)
        records.sort(key=lambda rec: rec.experiment_id)
        return records

    def _populate_experiment_table(self, records: Iterable[ExperimentRecord]) -> None:
        table = self.experiment_table
        was_sorting = table.isSortingEnabled()
        if was_sorting:
            table.setSortingEnabled(False)
        table.blockSignals(True)
        self._suppress_item_changed = True
        table.setRowCount(0)
        for record in records:
            if not self._show_trash and record.index_info.trash:
                continue
            row = table.rowCount()
            table.insertRow(row)
            self._set_row_items(row, record)
        table.blockSignals(False)
        self._suppress_item_changed = False
        if was_sorting:
            table.setSortingEnabled(True)
            self._apply_current_sort()
        else:
            table.resizeRowsToContents()

    def _set_row_items(self, row: int, record: ExperimentRecord) -> None:
        table = self.experiment_table
        star_count = max(int(record.index_info.star), 0)
        star_prefix = "⭐" * star_count

        parts: List[str] = []
        if record.index_info.trash:
            # parts.append("[trash]")
            parts.append("🗑️")
        if star_prefix:
            parts.append(star_prefix)
        title_text = record.index_info.title or "(untitled)"
        parts.append(title_text)
        display_title = " ".join(parts)

        id_item = QTableWidgetItem()
        id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        id_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        id_item.setData(Qt.DisplayRole, record.experiment_id)
        id_item.setData(Qt.UserRole, record)
        table.setItem(row, COL_ID, id_item)

        title_item = QTableWidgetItem(display_title)
        title_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        if star_prefix:
            title_item.setToolTip(f"Star count: {star_count}")
        title_font = title_item.font()
        title_font.setBold(star_count > 0)
        title_font.setStrikeOut(record.index_info.trash)
        title_item.setFont(title_font)
        table.setItem(row, COL_TITLE, title_item)

        rows_item = QTableWidgetItem()
        rows_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        rows_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        rows_value = record.row_count if record.row_count is not None else None
        rows_item.setData(Qt.DisplayRole, rows_value)
        table.setItem(row, COL_ROWS, rows_item)

        time_item = QTableWidgetItem(record.index_info.create_time or "")
        time_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        table.setItem(row, COL_CREATE_TIME, time_item)

        machine_item = QTableWidgetItem(record.index_info.create_machine or "")
        machine_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        table.setItem(row, COL_CREATE_MACHINE, machine_item)

        plot_axes_text = (
            ", ".join(record.index_info.plot_axes)
            if record.index_info.plot_axes
            else ""
        )
        plot_axes_item = QTableWidgetItem(plot_axes_text)
        plot_axes_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        table.setItem(row, COL_PLOT_AXES, plot_axes_item)

    def _apply_current_sort(self) -> None:
        if not self.experiment_table.isSortingEnabled():
            self.experiment_table.resizeRowsToContents()
            return
        self._suppress_sort_sync = True
        try:
            header = self.experiment_table.horizontalHeader()
            header.setSortIndicator(self._sort_column, self._sort_order)
            self.experiment_table.sortItems(self._sort_column, self._sort_order)
        finally:
            self._suppress_sort_sync = False
        self.experiment_table.resizeRowsToContents()

    def _on_sort_indicator_changed(self, section: int, order: Qt.SortOrder) -> None:
        if self._suppress_sort_sync:
            return
        self._sort_column = section
        self._sort_order = order
        self._apply_current_sort()

    def _record_for_row(self, row: int) -> Optional[ExperimentRecord]:
        item = self.experiment_table.item(row, COL_ID)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _find_row_for_record(self, record: ExperimentRecord) -> Optional[int]:
        table = self.experiment_table
        for row in range(table.rowCount()):
            row_record = self._record_for_row(row)
            if row_record is record:
                return row
            if (
                row_record
                and row_record.experiment_id == record.experiment_id
                and row_record.path == record.path
            ):
                return row
        return None

    def _update_row_items(self, record: ExperimentRecord) -> None:
        row = self._find_row_for_record(record)
        if row is None:
            return
        table = self.experiment_table
        table.blockSignals(True)
        self._suppress_item_changed = True
        try:
            self._set_row_items(row, record)
        finally:
            table.blockSignals(False)
            self._suppress_item_changed = False
        table.resizeRowToContents(row)

    def _toggle_column(self, column: int, visible: bool) -> None:
        self.experiment_table.setColumnHidden(column, not visible)

    def _on_experiment_selection_changed(self) -> None:
        table = self.experiment_table
        selected = table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        record = self._record_for_row(row)
        if record is None:
            return
        self._current_record = record
        self._load_experiment(record)

    def _load_experiment(self, record: ExperimentRecord) -> None:
        self.detail_label.setText(f"Experiment {record.experiment_id} - {record.path}")
        self.meta_view.setPlainText(_read_meta_text(record.meta_path))
        self._data_preview_active = False
        self.data_status_label.setText("Loading data preview…")
        self.data_load_button.setEnabled(False)
        self._data_preview_cap = self._data_preview_limit
        self._show_data_table(record, preview_only=True)

        image_files = _list_image_files(record.path)
        self._update_image_tabs(image_files)

        self._update_detail_watcher(record)

    def _set_data_table_empty(self, message: str = "No data to display.") -> None:
        self.data_table.setModel(None)
        self.data_status_label.setText(message)
        self.data_load_button.setEnabled(False)
        self._data_preview_active = False

    def _show_data_table(self, record: ExperimentRecord, preview_only: bool) -> None:
        if preview_only:
            limit = max(self._data_preview_cap, self._data_preview_limit)
        else:
            limit = 0
        if preview_only and record.row_count is not None and record.row_count <= limit:
            preview_only = False

        frame: Optional[pd.DataFrame]
        total_rows = record.row_count
        has_more = False

        if preview_only:
            frame = record.data_preview
            if frame is None or len(frame) < limit:
                if record.data_frame is not None and len(record.data_frame) >= limit:
                    frame = record.data_frame.head(limit).copy()
                else:
                    frame = _load_dataframe_preview(record.data_path, limit)
                record.data_preview = frame
            if frame is None:
                message = (
                    "Data file not found."
                    if not record.data_path or not record.data_path.exists()
                    else "Failed to load data."
                )
                record.columns = []
                self._set_data_table_empty(message)
                self._data_preview_cap = self._data_preview_limit
                return
            frame_rows = len(frame)
            if total_rows is not None:
                has_more = total_rows > frame_rows
            else:
                has_more = True if limit else False
        else:
            frame = record.data_frame
            if frame is None or not record.data_loaded_fully:
                frame = _load_dataframe(record.data_path)
                if frame is None:
                    message = (
                        "Data file not found."
                        if not record.data_path or not record.data_path.exists()
                        else "Failed to load data."
                    )
                    record.columns = []
                    self._set_data_table_empty(message)
                    self._data_preview_cap = self._data_preview_limit
                    return
                record.data_frame = frame
                record.data_loaded_fully = True
            record.data_preview = (
                frame.head(self._data_preview_limit).copy()
                if frame is not None
                else None
            )
            frame_rows = len(frame)
            total_rows = frame_rows
            has_more = False

        if frame is None:
            record.columns = []
            self._set_data_table_empty("No data to display.")
            return

        record.columns = [str(col) for col in frame.columns]
        highlight = [str(col) for col in record.index_info.plot_axes]
        model = PandasTableModel(frame, self.data_table, highlight_columns=highlight)
        self.data_table.setModel(model)
        self.data_table.resizeColumnsToContents()
        row_height = self.data_table.fontMetrics().height() + 6
        self.data_table.verticalHeader().setDefaultSectionSize(row_height)

        frame_rows = len(frame)
        if preview_only:
            if has_more:
                total_text = f" Total: {total_rows}." if total_rows is not None else ""
                self.data_status_label.setText(
                    f"Showing first {frame_rows} rows. " + total_text
                )
            else:
                self.data_status_label.setText(f"Showing all {frame_rows} rows.")
            self.data_load_button.setEnabled(has_more)
            self._data_preview_active = has_more
        else:
            self.data_status_label.setText(f"Showing all {frame_rows} rows.")
            self.data_load_button.setEnabled(False)
            self._data_preview_active = False

    def _on_load_all_data_clicked(self) -> None:
        record = self._current_record
        if record is None:
            return
        if not self._data_preview_active:
            self._show_data_table(record, preview_only=False)
            return
        previous_cap = self._data_preview_cap
        self._data_preview_cap += 3000
        self._show_data_table(record, preview_only=True)

    def _on_refresh_clicked(self) -> None:
        self.refresh_experiments()
        self.refresh_current_experiment()

    def _open_table_context_menu(self, point) -> None:
        row = self.experiment_table.indexAt(point).row()
        if 0 <= row < self.experiment_table.rowCount():
            self.experiment_table.selectRow(row)
        records = self._get_selected_records()
        record = records[0] if records else None
        menu = QMenu(self)
        rename_action = menu.addAction("Rename Title... (F2)")
        toggle_star_action = menu.addAction("Toggle ⭐Star (S)")
        toggle_star_action.setCheckable(True)
        toggle_trash_action = menu.addAction("Toggle 🗑️Trash (T)")
        toggle_trash_action.setCheckable(True)
        show_trash_action = menu.addAction("Show Trashed Items")
        show_trash_action.setCheckable(True)
        show_trash_action.setChecked(self._show_trash)
        menu.addSeparator()
        create_time_action = menu.addAction("Show Create Time Column")
        create_time_action.setCheckable(True)
        create_time_action.setChecked(
            not self.experiment_table.isColumnHidden(COL_CREATE_TIME)
        )
        create_machine_action = menu.addAction("Show Create Machine Column")
        create_machine_action.setCheckable(True)
        create_machine_action.setChecked(
            not self.experiment_table.isColumnHidden(COL_CREATE_MACHINE)
        )
        plot_axes_action = menu.addAction("Show Plot Axes Column")
        plot_axes_action.setCheckable(True)
        plot_axes_action.setChecked(
            not self.experiment_table.isColumnHidden(COL_PLOT_AXES)
        )
        menu.addSeparator()
        open_explorer = menu.addAction("Open in Explorer")
        if not records:
            rename_action.setEnabled(False)
            toggle_star_action.setEnabled(False)
            toggle_trash_action.setEnabled(False)
            open_explorer.setEnabled(False)
        else:
            rename_action.setEnabled(len(records) == 1)
            all_starred = all(rec.index_info.star > 0 for rec in records)
            all_trashed = all(rec.index_info.trash for rec in records)
            toggle_star_action.setEnabled(True)
            toggle_star_action.setChecked(all_starred)
            toggle_trash_action.setEnabled(True)
            toggle_trash_action.setChecked(all_trashed)
        chosen = menu.exec(self.experiment_table.viewport().mapToGlobal(point))
        if chosen is None:
            return
        if chosen == rename_action and records and len(records) == 1:
            self._rename_record_title(records[0])
        elif chosen == toggle_star_action and records:
            self._set_records_star_count(
                records, 1 if toggle_star_action.isChecked() else 0
            )
        elif chosen == toggle_trash_action and records:
            self._set_records_trash(records, toggle_trash_action.isChecked())
        elif chosen == show_trash_action:
            self._toggle_show_trash()
        elif chosen == create_time_action:
            self._toggle_column(COL_CREATE_TIME, create_time_action.isChecked())
        elif chosen == create_machine_action:
            self._toggle_column(COL_CREATE_MACHINE, create_machine_action.isChecked())
        elif chosen == plot_axes_action:
            self._toggle_column(COL_PLOT_AXES, plot_axes_action.isChecked())
        elif chosen == open_explorer and record is not None:
            self._open_record_in_explorer(record)

    def _get_selected_record(self) -> Optional[ExperimentRecord]:
        records = self._get_selected_records()
        return records[0] if records else None

    def _get_selected_records(self) -> List[ExperimentRecord]:
        selection_model = self.experiment_table.selectionModel()
        if selection_model is None:
            return []
        records: List[ExperimentRecord] = []
        for index in selection_model.selectedRows():
            record = self._record_for_row(index.row())
            if record is not None:
                records.append(record)
        return records

    def _set_record_star_count(
        self, record: ExperimentRecord, count: int, refresh: bool = True
    ) -> bool:
        new_value = max(int(count), 0)
        if record.index_info.star == new_value:
            return False
        previous = record.index_info.star
        record.index_info.star = new_value
        if not self._persist_index_info(record):
            record.index_info.star = previous
            return False
        if refresh:
            self.refresh_experiments()
        return True

    def _set_record_trash(
        self, record: ExperimentRecord, value: bool, refresh: bool = True
    ) -> bool:
        if record.index_info.trash == value:
            return False
        previous = record.index_info.trash
        record.index_info.trash = value
        if not self._persist_index_info(record):
            record.index_info.trash = previous
            return False
        if refresh:
            self.refresh_experiments()
        return True

    def _toggle_show_trash(self) -> None:
        self._show_trash = not self._show_trash
        self.refresh_experiments()

    def _set_records_star_count(
        self, records: Iterable[ExperimentRecord], count: int
    ) -> None:
        changed = False
        for record in records:
            changed |= self._set_record_star_count(record, count, refresh=False)
        if changed:
            self.refresh_experiments()

    def _set_records_trash(
        self, records: Iterable[ExperimentRecord], value: bool
    ) -> None:
        changed = False
        for record in records:
            changed |= self._set_record_trash(record, value, refresh=False)
        if changed:
            self.refresh_experiments()

    def _rename_record_title(self, record: ExperimentRecord) -> None:
        current_title = record.index_info.title or ""
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Rename Experiment")
        dialog.setLabelText("Enter new title:")
        dialog.setTextValue(current_title)
        dialog.setInputMode(QInputDialog.TextInput)
        dialog.resize(max(dialog.sizeHint().width(), 600), dialog.sizeHint().height())
        if dialog.exec() != QDialog.Accepted:
            return
        new_title = dialog.textValue().strip()
        if new_title == current_title:
            return
        previous = record.index_info.title
        record.index_info.title = new_title
        if not self._persist_index_info(record):
            record.index_info.title = previous
            return
        self.refresh_experiments()

    def _shortcut_set_star(self, count: int) -> None:
        records = self._get_selected_records()
        if not records:
            return
        self._set_records_star_count(records, count)

    def _shortcut_toggle_star(self) -> None:
        records = self._get_selected_records()
        if not records:
            return
        changed = False
        for record in records:
            target = 0 if record.index_info.star > 0 else 1
            changed |= self._set_record_star_count(record, target, refresh=False)
        if changed:
            self.refresh_experiments()

    def _shortcut_mark_trash(self) -> None:
        records = self._get_selected_records()
        if not records:
            return
        self._set_records_trash(records, True)

    def _shortcut_toggle_trash(self) -> None:
        records = self._get_selected_records()
        if not records:
            return
        changed = False
        for record in records:
            changed |= self._set_record_trash(
                record, not record.index_info.trash, refresh=False
            )
        if changed:
            self.refresh_experiments()

    def _shortcut_rename_title(self) -> None:
        records = self._get_selected_records()
        if len(records) != 1:
            return
        self._rename_record_title(records[0])

    def _setup_shortcuts(self) -> None:
        for action in self._shortcuts:
            self.removeAction(action)
        self._shortcuts.clear()

        def add_shortcut(key: int, callback) -> None:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda _checked=False, cb=callback: cb())
            self.addAction(action)
            self._shortcuts.append(action)

        add_shortcut(Qt.Key_Delete, self._shortcut_mark_trash)
        add_shortcut(Qt.Key_T, self._shortcut_toggle_trash)
        add_shortcut(Qt.Key_S, self._shortcut_toggle_star)
        add_shortcut(Qt.Key_F2, self._shortcut_rename_title)
        add_shortcut(Qt.Key_0, lambda: self._shortcut_set_star(0))
        add_shortcut(Qt.Key_1, lambda: self._shortcut_set_star(1))
        add_shortcut(Qt.Key_2, lambda: self._shortcut_set_star(2))
        add_shortcut(Qt.Key_3, lambda: self._shortcut_set_star(3))

    def _open_data_context_menu(self, point) -> None:
        record = self._current_record
        model = self.data_table.model()
        if record is None or model is None:
            return
        index = self.data_table.indexAt(point)
        column = (
            index.column()
            if index.isValid()
            else self.data_table.horizontalHeader().logicalIndexAt(point.x())
        )
        if column < 0:
            return
        column_name = str(model.headerData(column, Qt.Horizontal))
        if not column_name:
            return
        menu = QMenu(self.data_table)
        is_tracked = column_name in record.index_info.plot_axes
        toggle_action = menu.addAction("Toggle Plot Axes")
        toggle_action.setCheckable(True)
        toggle_action.setChecked(is_tracked)
        chosen = menu.exec(self.data_table.viewport().mapToGlobal(point))
        if chosen == toggle_action:
            self._toggle_column_plot_axes(record, column_name, not is_tracked)

    def _toggle_column_plot_axes(
        self, record: ExperimentRecord, column: str, enable: bool
    ) -> None:
        column = str(column)
        if not column:
            return
        updated = list(record.index_info.plot_axes)
        if enable:
            if column in updated:
                return
            updated.append(column)
        else:
            if column not in updated:
                return
            updated = [item for item in updated if item != column]
        record.index_info.plot_axes = updated
        record.plot_axes = list(updated)
        if self._persist_index_info(record):
            self._load_experiment(record)

    def _open_record_in_explorer(self, record: ExperimentRecord) -> None:
        path = record.path
        try:
            if sys.platform.startswith("win"):
                subprocess.run(["explorer", "/select,", str(path)], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to open explorer for %s: %s", path, exc)
            QMessageBox.warning(
                self, "Open in Explorer", f"Failed to open file browser: {exc}"
            )

    def _persist_index_info(self, record: ExperimentRecord) -> bool:
        index_path = record.index_path
        if index_path is None or not index_path.exists():
            index_path = record.path / "index.json"
        record.index_path = index_path
        try:
            _write_index_file(index_path, record.index_info)
        except TimeoutError as exc:  # pragma: no cover - defensive
            logger.warning("Timed out updating index file %s: %s", index_path, exc)
            QMessageBox.warning(
                self,
                "Update Failed",
                "Another process is updating this log. Please try again.",
            )
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to update index file %s: %s", index_path, exc)
            QMessageBox.warning(
                self,
                "Update Failed",
                f"Failed to update index file for experiment {record.experiment_id}: {exc}",
            )
            return False
        record.index_info = _read_index_file(index_path)
        record.plot_axes = list(record.index_info.plot_axes)
        self._update_row_items(record)
        self._apply_current_sort()
        if (
            self._current_record
            and self._current_record.experiment_id == record.experiment_id
        ):
            self._update_detail_watcher(record)
        return True

    def _clear_preview_panels(self) -> None:
        self.meta_view.setPlainText("")
        self._set_data_table_empty("")
        self._clear_image_tabs()

    def _clear_detail_watcher(self) -> None:
        try:
            paths = self._detail_watcher.files() + self._detail_watcher.directories()
            if paths:
                self._detail_watcher.removePaths(paths)
        except Exception:  # pragma: no cover - defensive
            pass

    def _clear_image_tabs(self) -> None:
        if not self._image_tab_indices:
            return
        for index in sorted(self._image_tab_indices, reverse=True):
            self.tab_widget.removeTab(index)
        self._image_tab_indices.clear()

    def _update_image_tabs(self, image_files: List[Path]) -> None:
        self._clear_image_tabs()
        for image_path in image_files:
            widget = ScaledImageLabel()
            widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
            widget.setWordWrap(True)
            widget.setToolTip(str(image_path))
            success = widget.load_image(image_path)
            if not success:
                widget.setWordWrap(True)
            index = self.tab_widget.addTab(widget, image_path.name)
            self._image_tab_indices.append(index)

    def _update_detail_watcher(self, record: ExperimentRecord) -> None:
        self._clear_detail_watcher()
        watch_paths: List[str] = [str(record.path)]
        for extra in (record.meta_path, record.data_path, record.index_path):
            if extra and extra.exists():
                watch_paths.append(str(extra))
        if watch_paths:
            self._detail_watcher.addPaths(watch_paths)


def ensure_application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("Logqbit Log Browser")
    return app


def open_browser(directory: Optional[str] = None) -> LogBrowserWindow:
    app = ensure_application()
    window = LogBrowserWindow(Path(directory) if directory else None)
    window.show()
    return window


def main(argv: Optional[List[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    directory = Path(args[0]).expanduser().resolve() if args else None
    app = ensure_application()
    window = LogBrowserWindow(directory)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual run
    sys.exit(main())
