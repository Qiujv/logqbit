"""Record detail views for the log browser."""

from __future__ import annotations

import logging
import numbers
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QFileSystemWatcher,
    QModelIndex,
    Signal,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import (
    QAction,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QMessageBox,
    QWidget,
)

from logqbit.file_version import FileVersion

from logqbit.gui.plot.widget import PlotManager

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord

logger = logging.getLogger(__name__)

REFRESH_DEBOUNCE_MS = 250
TAB_CONST = 0
TAB_DATA = 1
TAB_PLOT = 2


class PandasTableModel(QAbstractTableModel):
    """Table model for displaying DataFrames with an optional row limit."""

    def __init__(
        self,
        frame: pd.DataFrame,
        parent: QWidget | None = None,
        highlight_columns: Iterable[str] | None = None,
        preview_limit: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._df = frame
        self._preview_limit = preview_limit
        self._highlight = (
            {str(name) for name in highlight_columns} if highlight_columns else set()
        )
        self._bold_font = QFont(parent.font()) if parent else QFont()
        self._bold_font.setBold(True)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        total_rows = self._df.shape[0]
        if self._preview_limit is not None and self._preview_limit > 0:
            return min(total_rows, self._preview_limit)
        return total_rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._df.columns)

    def get_total_rows(self) -> int:
        return self._df.shape[0]

    def set_preview_limit(self, limit: int | None) -> None:
        old_count = self.rowCount()
        self._preview_limit = limit
        if self.rowCount() != old_count:
            self.beginResetModel()
            self.endResetModel()

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: D401
        if not index.isValid():
            return None
        column_name = str(self._df.columns[index.column()])
        if role == Qt.FontRole and column_name in self._highlight:
            return self._bold_font
        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self._df.iat[index.row(), index.column()]
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
            column_name = str(self._df.columns[section])
            if column_name in self._highlight:
                return self._bold_font
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


@dataclass
class DetailMetadataCache:
    """Track whether plot controls need synchronization from metadata."""

    path: Path | None = None
    version: FileVersion | None = None

    def clear(self) -> None:
        self.path = None
        self.version = None

    def update(self, record: LogRecord) -> bool:
        version = FileVersion.from_path(record.meta_path)
        changed = self.path != record.path or self.version != version
        self.path = record.path
        self.version = version
        return changed


class DetailDataCache:
    """Full-data cache owned by one detail view."""

    def __init__(self) -> None:
        self.dataframe: pd.DataFrame | None = None
        self.path: Path | None = None
        self.version: FileVersion | None = None

    def clear(self) -> None:
        self.dataframe = None
        self.path = None
        self.version = None

    def load(self, record: LogRecord) -> pd.DataFrame | None:
        version = FileVersion.from_path(record.data_path)
        if (
            self.dataframe is not None
            and self.path == record.path
            and self.version == version
            and version is not None
        ):
            return self.dataframe
        self.clear()
        dataframe = record.read_dataframe()
        if dataframe is not None:
            self.dataframe = dataframe
            self.path = record.path
            self.version = version
        return dataframe


class DataViewManager:
    INITIAL_PREVIEW_LIMIT = 100
    PREVIEW_INCREMENT = 1000

    def __init__(
        self,
        parent: QWidget | None = None,
        load_more_callback: Callable[[], None] | None = None,
    ):
        self._load_more_callback = load_more_callback
        self.widget = self._create_widget(parent)

    def _create_widget(self, parent: QWidget | None = None) -> QWidget:
        data_tab = QWidget(parent)
        data_layout = QVBoxLayout(data_tab)
        data_layout.setContentsMargins(4, 4, 4, 4)

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
        data_layout.addWidget(self.data_table)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self.data_status_label = QLabel("")
        self.data_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_status_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.data_load_button = QPushButton("Show More Rows")
        self.data_load_button.setEnabled(False)
        if self._load_more_callback:
            self.data_load_button.clicked.connect(self._load_more_callback)
        controls.addWidget(self.data_status_label)
        controls.addStretch(1)
        controls.addWidget(self.data_load_button)
        data_layout.addLayout(controls)
        return data_tab

    def set_empty(self, message: str = "No data to display.") -> None:
        self.data_table.setModel(None)
        self.data_status_label.setText(message)
        self.data_load_button.setEnabled(False)

    def show_data_table(
        self,
        record: LogRecord,
        dataframe: pd.DataFrame | None,
        preview_only: bool,
    ) -> None:
        if dataframe is None:
            message = (
                "Data file not found."
                if record.data_version is None
                else "Failed to load data."
            )
            self.set_empty(message)
            return

        total_rows = len(dataframe)
        preview_limit = None
        if preview_only and total_rows > self.INITIAL_PREVIEW_LIMIT:
            preview_limit = self.INITIAL_PREVIEW_LIMIT

        model = PandasTableModel(
            dataframe,
            self.data_table,
            highlight_columns=record.plot_axes,
            preview_limit=preview_limit,
        )
        self.data_table.setModel(model)
        self.data_table.resizeColumnsToContents()
        row_height = self.data_table.fontMetrics().height() + 6
        self.data_table.verticalHeader().setDefaultSectionSize(row_height)

        displayed_rows = model.rowCount()
        has_more = displayed_rows < total_rows
        if has_more:
            self.data_status_label.setText(
                f"Showing first {displayed_rows} rows. Total: {total_rows}."
            )
            self.data_load_button.setEnabled(True)
        else:
            self.data_status_label.setText(f"Showing all {displayed_rows} rows.")
            self.data_load_button.setEnabled(False)

    def load_more_data(
        self,
        record: LogRecord,
        dataframe: pd.DataFrame | None,
    ) -> None:
        model = self.data_table.model()
        if not isinstance(model, PandasTableModel):
            self.show_data_table(record, dataframe, preview_only=False)
            return

        total_rows = model.get_total_rows()
        current_limit = model.rowCount()
        if current_limit >= total_rows:
            return

        new_limit = min(current_limit + self.PREVIEW_INCREMENT, total_rows)
        model.set_preview_limit(new_limit)

        displayed_rows = model.rowCount()
        has_more = displayed_rows < total_rows
        if has_more:
            self.data_status_label.setText(
                f"Showing first {displayed_rows} rows. Total: {total_rows}."
            )
            self.data_load_button.setEnabled(True)
        else:
            self.data_status_label.setText(f"Showing all {displayed_rows} rows.")
            self.data_load_button.setEnabled(False)


def record_watch_paths(record: LogRecord) -> list[str]:
    paths: list[str] = [str(record.path)]
    for extra in (
        record.const_path,
        record.data_path,
        record.meta_path,
        *record.list_image_files(),
        *record.list_other_files(),
    ):
        if extra and extra.exists():
            paths.append(str(extra))
    return paths


def _open_path_in_explorer(path: Path, parent: QWidget | None = None) -> None:
    try:
        if sys.platform.startswith("win"):
            command = ["explorer", str(path)]
            if path.is_file():
                command = ["explorer", "/select,", str(path)]
            subprocess.run(command, check=False)
        elif sys.platform == "darwin":
            command = ["open", str(path)]
            if path.is_file():
                command = ["open", "-R", str(path)]
            subprocess.run(command, check=False)
        else:
            target = path if path.is_dir() else path.parent
            subprocess.run(["xdg-open", str(target)], check=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to open explorer for %s: %s", path, exc)
        if parent is not None:
            QMessageBox.warning(
                parent,
                "Open in Explorer",
                f"Failed to open file browser: {exc}",
            )


class ScaledImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
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

    def copy_image_to_clipboard(self) -> None:
        if self._pixmap is not None and not self._pixmap.isNull():
            QApplication.clipboard().setPixmap(self._pixmap)

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


class RecordDetailView(QWidget):
    """Reusable record detail widget with tabs and preview controls."""

    record_refreshed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        file_open_callback: Callable[[Path], None] | None = None,
        enable_tab_shortcuts: bool = True,
    ) -> None:
        super().__init__(parent)
        self._record: LogRecord | None = None
        self._data_cache = DetailDataCache()
        self._metadata_cache = DetailMetadataCache()
        self._file_open_callback = file_open_callback
        self._enable_tab_shortcuts = enable_tab_shortcuts
        self._shortcuts: list[QAction] = []
        self._detail_watcher = self._create_detail_watcher()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(REFRESH_DEBOUNCE_MS)
        self._refresh_timer.timeout.connect(self.refresh_current_record)

        self._build_ui()
        if self._enable_tab_shortcuts:
            self._setup_shortcuts()

    @property
    def current_record(self) -> LogRecord | None:
        return self._record

    @property
    def watch_enabled(self) -> bool:
        return self.watch_checkbox.isChecked()

    def set_watch_enabled(self, enabled: bool) -> None:
        self.watch_checkbox.setChecked(enabled)

    def current_tab_index(self) -> int:
        return self.tab_widget.currentIndex()

    def set_current_tab(self, index: int) -> None:
        if 0 <= index < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(index)

    def switch_tab(self, step: int) -> None:
        count = self.tab_widget.count()
        if count <= 1:
            return
        current = self.tab_widget.currentIndex()
        if current < 0:
            current = 0
        self.tab_widget.setCurrentIndex((current + step) % count)

    def load_record(self, record: LogRecord) -> None:
        update_plot_controls = self._metadata_cache.update(record)
        dataframe = self._data_cache.load(record)
        self._record = record
        self.detail_id_label.setText(f"#{record.log_id}")
        self.detail_label.setText(str(record.path))
        self.yaml_view.setPlainText(record.read_yaml_text())
        self.data_view_manager.show_data_table(
            record,
            dataframe,
            preview_only=True,
        )
        self._update_image_tabs(record.list_image_files())
        self.files_button.setEnabled(True)
        defer_plot = self.tab_widget.currentIndex() != TAB_PLOT
        if update_plot_controls:
            self.plot_manager.update_controls(record, dataframe)
        self.plot_manager.update_plot(record, dataframe, defer=defer_plot)
        self._sync_detail_watcher()

    def refresh_current_record(self, *, force: bool = False) -> None:
        if self._record is None:
            return
        if force:
            self._data_cache.clear()
        self._record = self._record.refresh()
        self.load_record(self._record)
        self.record_refreshed.emit(self._record)

    def clear(self, message: str = "No log selected.") -> None:
        self._refresh_timer.stop()
        self._clear_detail_watcher()
        self._record = None
        self._data_cache.clear()
        self._metadata_cache.clear()
        self.detail_id_label.setText("")
        self.detail_label.setText(message)
        self.yaml_view.setPlainText("")
        self.data_view_manager.set_empty("")
        self._clear_dynamic_tabs()
        self.files_button.setEnabled(False)
        self.files_menu.clear()
        self.plot_manager.reset_plot_state("")

    def _build_ui(self) -> None:
        detail_layout = QVBoxLayout(self)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(6)

        detail_top = QHBoxLayout()
        self.detail_id_label = QLabel()
        detail_top.addWidget(self.detail_id_label)
        self.detail_label = QLabel("No log selected.")
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_label.setWordWrap(True)
        self.detail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        detail_top.addWidget(self.detail_label)
        self.watch_checkbox = QCheckBox("auto update")
        self.watch_checkbox.setChecked(True)
        self.watch_checkbox.setToolTip(
            "Automatically refresh this detail view when files change"
        )
        self.watch_checkbox.toggled.connect(self._on_watch_toggled)
        detail_top.addWidget(self.watch_checkbox)
        detail_layout.addLayout(detail_top)

        self.tab_widget = QTabWidget(self)

        self.files_button = QToolButton(self.tab_widget)
        self.files_button.setText("Files...")
        self.files_button.setPopupMode(QToolButton.InstantPopup)
        self.files_button.setEnabled(False)
        self.files_menu = QMenu(self.files_button)
        self.files_menu.aboutToShow.connect(self._rebuild_files_menu)
        self.files_button.setMenu(self.files_menu)
        self.tab_widget.setCornerWidget(self.files_button, Qt.TopRightCorner)

        self.yaml_view = QPlainTextEdit()
        self.yaml_view.setReadOnly(True)
        self.yaml_view.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        self.tab_widget.addTab(self.yaml_view, "Const.")

        self.data_view_manager = DataViewManager(
            parent=self,
            load_more_callback=self._on_load_more,
        )
        self.tab_widget.addTab(self.data_view_manager.widget, "Data")

        self.plot_manager = PlotManager(parent=self)
        self.tab_widget.addTab(self.plot_manager.widget, "Plot")

        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        detail_layout.addWidget(self.tab_widget)

    def _setup_shortcuts(self) -> None:
        for action in self._shortcuts:
            self.removeAction(action)
        self._shortcuts.clear()

        def add_shortcut(key: int, callback: Callable[[], None]) -> None:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda _checked=False, cb=callback: cb())
            self.addAction(action)
            self._shortcuts.append(action)

        add_shortcut(Qt.Key_Left, lambda: self.switch_tab(-1))
        add_shortcut(Qt.Key_Right, lambda: self.switch_tab(1))

    def _on_tab_changed(self, index: int) -> None:
        if index == TAB_PLOT:
            self.plot_manager.refresh_if_needed()

    def _on_load_more(self) -> None:
        if self._record:
            self.data_view_manager.load_more_data(
                self._record,
                self._data_cache.load(self._record),
            )

    def _on_watch_toggled(self, enabled: bool) -> None:
        if enabled:
            self._sync_detail_watcher()
        else:
            self._refresh_timer.stop()
            self._clear_detail_watcher()

    def _schedule_detail_refresh(self) -> None:
        self._refresh_timer.start()

    def _create_detail_watcher(self) -> QFileSystemWatcher:
        watcher = QFileSystemWatcher(self)
        watcher.directoryChanged.connect(self._schedule_detail_refresh)
        watcher.fileChanged.connect(self._schedule_detail_refresh)
        return watcher

    def _clear_detail_watcher(self) -> None:
        try:
            paths = self._detail_watcher.files() + self._detail_watcher.directories()
            failed_paths = self._detail_watcher.removePaths(paths) if paths else []
            if failed_paths:
                old_watcher = self._detail_watcher
                self._detail_watcher = self._create_detail_watcher()
                old_watcher.deleteLater()
        except Exception:  # pragma: no cover - defensive
            pass

    def _sync_detail_watcher(self) -> None:
        self._clear_detail_watcher()
        if not self.watch_enabled or self._record is None:
            return
        watch_paths = record_watch_paths(self._record)
        if watch_paths:
            self._detail_watcher.addPaths(watch_paths)

    def _clear_dynamic_tabs(self) -> None:
        for index in range(self.tab_widget.count() - 1, TAB_PLOT, -1):
            self.tab_widget.removeTab(index)

    def _update_image_tabs(self, image_files: list[Path]) -> None:
        self._clear_dynamic_tabs()
        for image_path in image_files:
            image_tab = QWidget()
            layout = QVBoxLayout(image_tab)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

            image_label = ScaledImageLabel()
            image_label.setToolTip(str(image_path))
            image_loaded = image_label.load_image(image_path)
            layout.addWidget(image_label, stretch=1)

            button_row = QHBoxLayout()
            button_row.addStretch(1)
            copy_button = QPushButton("Copy image")
            copy_button.setEnabled(image_loaded)
            copy_button.setToolTip("Copy the original image to the clipboard")
            copy_button.clicked.connect(image_label.copy_image_to_clipboard)
            button_row.addWidget(copy_button)
            layout.addLayout(button_row)

            self.tab_widget.addTab(image_tab, image_path.name)

    def _rebuild_files_menu(self) -> None:
        self.files_menu.clear()
        record = self._record
        if record is None:
            return

        try:
            file_paths = sorted(
                (path for path in record.path.iterdir() if path.is_file()),
                key=lambda path: path.name.casefold(),
            )
        except OSError:
            file_paths = []

        if not file_paths:
            empty_action = self.files_menu.addAction("(No files)")
            empty_action.setEnabled(False)

        for path in file_paths:
            action = self.files_menu.addAction(path.name)
            action.setToolTip(str(path))
            action.triggered.connect(
                lambda _checked=False, file_path=path: self._open_file(file_path)
            )

        self.files_menu.addSeparator()
        show_action = self.files_menu.addAction("Show in Explorer")
        show_action.triggered.connect(self._show_record_in_explorer)

    def _open_file(self, path: Path) -> None:
        if self._file_open_callback:
            self._file_open_callback(path)
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.warning(
                self,
                "Open File",
                f"No application is available to open {path.name}.",
            )

    def _show_record_in_explorer(self) -> None:
        if self._record is not None:
            _open_path_in_explorer(self._record.path, parent=self)


class RecordDetailWindow(QMainWindow):
    """Standalone window showing the full detail panel for a single log record."""

    def __init__(
        self,
        record: LogRecord,
        initial_tab: int = TAB_CONST,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.resize(900, 600)

        self.detail_view = RecordDetailView(parent=self)
        self.detail_view.record_refreshed.connect(self._update_window_title)
        self.setCentralWidget(self.detail_view)

        self.load_record(record)
        self.detail_view.set_current_tab(initial_tab)

    def load_record(self, record: LogRecord) -> None:
        self._update_window_title(record)
        self.detail_view.load_record(record)

    def refresh_current_record(self) -> None:
        self.detail_view.refresh_current_record()

    def _update_window_title(self, record: LogRecord) -> None:
        title = record.title or "(untitled)"
        self.setWindowTitle(f"#{record.log_id} {title}")
