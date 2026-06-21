"""Reusable detail views for log records."""

from __future__ import annotations

import logging
import numbers
import subprocess
import sys
from collections.abc import Callable, Iterable
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QFileSystemWatcher, QModelIndex, Qt, QTimer
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QHeaderView,
    QWidget,
    QMessageBox,
)

from .plotter import PlotManager

if TYPE_CHECKING:
    from .browser import LogRecord

logger = logging.getLogger(__name__)

REFRESH_DEBOUNCE_MS = 250
TAB_CONST = 0
TAB_DATA = 1
TAB_PLOT = 2


def _load_window_icon() -> QIcon:
    try:
        icon_path = files("logqbit") / "assets" / "browser.svg"
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            return icon
    except Exception as exc:
        logger.debug("Failed to load window icon: %s", exc)
    return QIcon()


WINDOW_ICON = _load_window_icon()


def record_watch_paths(record: LogRecord) -> list[str]:
    paths: list[str] = [str(record.path)]
    for extra in (
        record.yaml_path,
        record.data_path,
        record.meta.path,
        *record.list_image_files(),
        *record.list_other_files(),
    ):
        if extra and extra.exists():
            paths.append(str(extra))
    return paths


def _open_path_in_explorer(path: Path, parent: QWidget | None = None) -> None:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["explorer", "/select,", str(path)], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path.parent)], check=False)
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


class PandasTableModel(QAbstractTableModel):
    """Table model for displaying pandas DataFrames with optional preview limit."""

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
        new_count = self.rowCount()
        if new_count != old_count:
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


class DataViewManager:
    INITIAL_PREVIEW_LIMIT = 100
    PREVIEW_INCREMENT = 1000

    def __init__(
        self,
        parent: QWidget | None = None,
        load_more_callback: Callable[[], None] | None = None,
        plot_axes_changed_callback: Callable[[LogRecord], None] | None = None,
    ):
        self._load_more_callback = load_more_callback
        self._plot_axes_changed_callback = plot_axes_changed_callback
        self._current_record: LogRecord | None = None
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
        self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.customContextMenuRequested.connect(self._open_context_menu)
        data_layout.addWidget(self.data_table)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self.data_status_label = QLabel("")
        self.data_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_status_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
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

    def show_data_table(self, record: LogRecord, preview_only: bool) -> None:
        self._current_record = record
        dataframe = record.load_dataframe()
        if dataframe is None:
            message = (
                "Data file not found."
                if not record.data_path or not record.data_path.exists()
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
            highlight_columns=record.meta.plot_axes,
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

    def _open_context_menu(self, point) -> None:
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
        is_tracked = column_name in record.meta.plot_axes
        toggle_action = menu.addAction("Toggle Plot Axes")
        toggle_action.setCheckable(True)
        toggle_action.setChecked(is_tracked)

        chosen = menu.exec(self.data_table.viewport().mapToGlobal(point))
        if chosen == toggle_action:
            self._toggle_plot_axes(record, column_name, not is_tracked)

    def _toggle_plot_axes(
        self, record: LogRecord, column_name: str, enable: bool
    ) -> None:
        column_name = str(column_name)
        if not column_name:
            return

        updated = list(record.meta.plot_axes)
        if enable:
            if column_name in updated:
                return
            updated.append(column_name)
        else:
            if column_name not in updated:
                return
            updated = [item for item in updated if item != column_name]

        record.meta.plot_axes = updated
        if self._plot_axes_changed_callback:
            self._plot_axes_changed_callback(record)

    def load_more_data(self, record: LogRecord) -> None:
        model = self.data_table.model()
        if not isinstance(model, PandasTableModel):
            self.show_data_table(record, preview_only=False)
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


class RecordDetailView(QWidget):
    """Reusable record detail widget with tabs and preview controls."""

    def __init__(
        self,
        parent: QWidget | None = None,
        record_changed_callback: Callable[[LogRecord], None] | None = None,
        file_open_callback: Callable[[Path], None] | None = None,
        watch_toggled_callback: Callable[[bool], None] | None = None,
        enable_tab_shortcuts: bool = True,
    ) -> None:
        super().__init__(parent)
        self._record: LogRecord | None = None
        self._record_changed_callback = record_changed_callback
        self._file_open_callback = file_open_callback
        self._watch_toggled_callback = watch_toggled_callback
        self._enable_tab_shortcuts = enable_tab_shortcuts
        self._image_tab_indices: list[int] = []
        self._extra_files_tab_index: int | None = None
        self._shortcuts: list[QAction] = []

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
        self._record = record
        self.detail_label.setText(f"#{record.log_id} - {record.path}")
        self.yaml_view.setPlainText(record.read_yaml_text())
        self.data_view_manager.show_data_table(record, preview_only=True)
        self._update_image_tabs(record.list_image_files())
        self._update_extra_files_tab(record.list_other_files())
        defer_plot = self.tab_widget.currentIndex() != TAB_PLOT
        self.plot_manager.update_plot_and_controls(record, defer_plot=defer_plot)

    def refresh_current_record(self) -> None:
        if self._record is None:
            return
        self._record.data_frame = None
        self.load_record(self._record)

    def clear(self, message: str = "No log selected.") -> None:
        self._record = None
        self.detail_label.setText(message)
        self.yaml_view.setPlainText("")
        self.data_view_manager.set_empty("")
        self._clear_image_tabs()
        self._update_extra_files_tab([])
        self.plot_manager.reset_plot_state("")

    def _build_ui(self) -> None:
        detail_layout = QVBoxLayout(self)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(6)

        detail_top = QHBoxLayout()
        self.detail_label = QLabel("No log selected.")
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        detail_top.addWidget(self.detail_label)
        detail_top.addStretch(1)
        self.watch_checkbox = QCheckBox("auto update")
        self.watch_checkbox.setChecked(True)
        self.watch_checkbox.setToolTip("Automatically refresh this detail view when files change")
        self.watch_checkbox.toggled.connect(self._on_watch_toggled)
        detail_top.addWidget(self.watch_checkbox)
        detail_layout.addLayout(detail_top)

        self.tab_widget = QTabWidget(self)

        self.yaml_view = QPlainTextEdit()
        self.yaml_view.setReadOnly(True)
        self.tab_widget.addTab(self.yaml_view, "Const.")

        self.data_view_manager = DataViewManager(
            parent=self,
            load_more_callback=self._on_load_more,
            plot_axes_changed_callback=self._on_plot_axes_changed,
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
            self.data_view_manager.load_more_data(self._record)

    def _on_plot_axes_changed(self, record: LogRecord) -> None:
        if self._record_changed_callback:
            self._record_changed_callback(record)
        self.load_record(record)

    def _on_watch_toggled(self, enabled: bool) -> None:
        if self._watch_toggled_callback:
            self._watch_toggled_callback(enabled)

    def _clear_image_tabs(self) -> None:
        if not self._image_tab_indices:
            return
        for index in sorted(self._image_tab_indices, reverse=True):
            self.tab_widget.removeTab(index)
        self._image_tab_indices.clear()

    def _update_image_tabs(self, image_files: list[Path]) -> None:
        self._clear_image_tabs()
        for image_path in image_files:
            widget = ScaledImageLabel()
            widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
            widget.setWordWrap(True)
            widget.setToolTip(str(image_path))
            widget.load_image(image_path)
            index = self.tab_widget.addTab(widget, image_path.name)
            self._image_tab_indices.append(index)

    def _update_extra_files_tab(self, extra_files: list[Path]) -> None:
        if self._extra_files_tab_index is not None:
            self.tab_widget.removeTab(self._extra_files_tab_index)
            self._extra_files_tab_index = None

        if not extra_files:
            return

        file_list = QListWidget(self.tab_widget)
        for file_path in extra_files:
            item = QListWidgetItem(file_path.name)
            item.setData(Qt.UserRole, file_path)
            item.setToolTip(str(file_path))
            file_list.addItem(item)

        file_list.itemClicked.connect(self._open_extra_file_item)
        file_list.itemActivated.connect(self._open_extra_file_item)
        self._extra_files_tab_index = self.tab_widget.addTab(file_list, "Files")

    def _open_extra_file_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if not isinstance(path, Path):
            return
        if self._file_open_callback:
            self._file_open_callback(path)
            return
        _open_path_in_explorer(path, parent=self)


class RecordDetailWindow(QMainWindow):
    """Standalone window showing the full detail panel for a single log record."""

    def __init__(
        self,
        record: LogRecord,
        initial_tab: int = TAB_CONST,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not WINDOW_ICON.isNull():
            self.setWindowIcon(WINDOW_ICON)
        self.resize(900, 600)
        self._detail_refresh_pending = False
        self._detail_watcher = QFileSystemWatcher(self)
        self._detail_watcher.directoryChanged.connect(self._schedule_detail_refresh)
        self._detail_watcher.fileChanged.connect(self._schedule_detail_refresh)

        self.detail_view = RecordDetailView(
            parent=self,
            watch_toggled_callback=self._on_watch_toggled,
        )
        self.setCentralWidget(self.detail_view)

        self.load_record(record)
        self.detail_view.set_current_tab(initial_tab)

    def load_record(self, record: LogRecord) -> None:
        title = record.meta.title or "(untitled)"
        self.setWindowTitle(f"#{record.log_id} {title}")
        self.detail_view.load_record(record)
        self._update_detail_watcher(record)

    def refresh_current_record(self) -> None:
        self.detail_view.refresh_current_record()
        record = self.detail_view.current_record
        if record is not None:
            self._update_detail_watcher(record)

    def _schedule_detail_refresh(self) -> None:
        if self._detail_refresh_pending:
            return
        self._detail_refresh_pending = True
        QTimer.singleShot(REFRESH_DEBOUNCE_MS, self._run_detail_refresh)

    def _run_detail_refresh(self) -> None:
        self._detail_refresh_pending = False
        self.refresh_current_record()

    def _clear_detail_watcher(self) -> None:
        try:
            paths = self._detail_watcher.files() + self._detail_watcher.directories()
            if paths:
                self._detail_watcher.removePaths(paths)
        except Exception:  # pragma: no cover - defensive
            pass

    def _update_detail_watcher(self, record: LogRecord) -> None:
        self._clear_detail_watcher()
        if not self.detail_view.watch_enabled:
            return
        watch_paths = record_watch_paths(record)
        if watch_paths:
            self._detail_watcher.addPaths(watch_paths)

    def _on_watch_toggled(self, enabled: bool) -> None:
        record = self.detail_view.current_record
        if record is None:
            self._clear_detail_watcher()
            return
        if enabled:
            self._update_detail_watcher(record)
        else:
            self._clear_detail_watcher()
