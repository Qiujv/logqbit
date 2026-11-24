"""Record detail container and standalone window."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import (
    QFileSystemWatcher,
    Signal,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QAction,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from logqbit.gui.browser.detail.data import DataViewManager
from logqbit.gui.browser.detail.files import (
    FilesMenu,
    ImageTab,
    list_image_files,
    open_file,
)
from logqbit.gui.browser.detail.const import ConstTab
from logqbit.gui.browser.plot.view import PlotView
from logqbit.gui.browser.plot.snapshot import PlotSnapshots
from logqbit.file_version import FileVersion

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord

logger = logging.getLogger(__name__)

REFRESH_DEBOUNCE_MS = 250
TAB_CONST = 0
TAB_DATA = 1
TAB_PLOT = 2


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


@dataclass
class DetailDataCache:
    """Full-data cache owned by one detail view."""

    dataframe: pd.DataFrame | None = None
    path: Path | None = None
    version: FileVersion | None = None

    def clear(self) -> None:
        self.dataframe = None
        self.path = None
        self.version = None

    def update(self, record: LogRecord) -> bool:
        version = FileVersion.from_path(record.data_path)
        if (
            self.dataframe is not None
            and self.path == record.path
            and self.version == version
            and version is not None
        ):
            return False
        self.clear()
        dataframe = record.read_dataframe()
        if dataframe is not None:
            self.dataframe = dataframe
            self.path = record.path
            self.version = version
        return True


def record_watch_paths(record: LogRecord) -> list[str]:
    paths: list[str] = [str(record.path)]
    for extra in (
        record.const_path,
        record.data_path,
        record.meta_path,
        *list_image_files(record.path),
    ):
        if extra and extra.exists():
            paths.append(str(extra))
    return paths


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
        self._selected_records = []
        self._snapshots = PlotSnapshots(self)
        self._snapshots.ready.connect(self._snapshot_ready)
        self._snapshots.loading.connect(self._snapshot_loading)
        self._data_cache = DetailDataCache()
        self._metadata_cache = DetailMetadataCache()
        self._file_open_callback = file_open_callback
        self._enable_tab_shortcuts = enable_tab_shortcuts
        self._shortcuts: list[QAction] = []
        self._preferred_tab_index = TAB_CONST
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
            self._preferred_tab_index = index
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
        metadata_changed = self._metadata_cache.update(record)
        data_changed = self._data_cache.update(record)
        dataframe = self._data_cache.dataframe
        self._record = record
        self._update_detail_id()
        self.detail_label.setText(record.path.as_posix())
        self.files_menu.set_record_path(record.path)
        self.const_tab.load_record(record)
        if metadata_changed or data_changed:
            self.data_view_manager.show_data_table(
                record,
                dataframe,
                preview_only=True,
            )
        self._update_image_tabs(list_image_files(record.path))
        self.files_button.setEnabled(True)
        defer_plot = self.tab_widget.currentIndex() != TAB_PLOT
        if not self.plot_view._comparison:
            if metadata_changed:
                self.plot_view.update_controls(record, dataframe)
            if metadata_changed or data_changed:
                self.plot_view.update_plot(record, dataframe, defer=defer_plot)
        self._sync_detail_watcher()

    def _update_detail_id(self) -> None:
        if self._record is None:
            self.detail_id_label.setText("")
            return
        suffix = (
            f" · {len(self._selected_records)} selected"
            if len(self._selected_records) > 1
            else ""
        )
        self.detail_id_label.setText(f"#{self._record.log_id}{suffix}")

    def set_selected_records(self, records) -> None:
        records = list(records)
        old_paths = [record.path for record in self._selected_records]
        paths = [record.path for record in records]
        was_comparison = self.plot_view._comparison
        self._selected_records = records
        self._update_detail_id()
        if len(records) < 2:
            self._snapshots.clear()
            self.plot_view.set_comparison(False)
            if was_comparison and self._record is not None:
                self.plot_view.update_controls(self._record, self._data_cache.dataframe)
                self.plot_view.update_plot(
                    self._record,
                    self._data_cache.dataframe,
                    defer=self.tab_widget.currentIndex() != TAB_PLOT,
                )
            return
        self.plot_view.set_comparison(True)
        if paths != old_paths:
            self._snapshots.set_records(records)
            self.plot_view._snapshot_frames = []
            self.plot_view._clear_plot("Preparing comparison snapshot…")
            if self.tab_widget.currentIndex() == TAB_PLOT:
                self._snapshots.request()

    def _snapshot_loading(self) -> None:
        self.plot_view._clear_plot("Loading comparison snapshot…")

    def _snapshot_ready(self, cache, errors) -> None:
        if not self.plot_view._comparison:
            return
        frames = [
            (record, cache[record.path].frame)
            for record in self._selected_records
            if record.path in cache
        ]
        self.plot_view.set_snapshot(
            frames, errors, defer=self.tab_widget.currentIndex() != TAB_PLOT
        )

    def refresh_comparison(self) -> None:
        """Refresh the multi-record plot snapshot after an explicit refresh."""
        if self.plot_view._comparison:
            self._snapshots.request()

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
        self._selected_records = []
        self._snapshots.clear()
        self.plot_view.set_comparison(False)
        self._data_cache.clear()
        self._metadata_cache.clear()
        self.detail_id_label.setText("")
        self.detail_label.setText(message)
        self.const_tab.clear("")
        self.data_view_manager.set_empty("")
        was_blocked = self.tab_widget.blockSignals(True)
        try:
            self._clear_dynamic_tabs()
        finally:
            self.tab_widget.blockSignals(was_blocked)
        self.files_button.setEnabled(False)
        self.files_menu.set_record_path(None)
        self.plot_view.reset_plot_state("")

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
        self.files_menu = FilesMenu(
            file_open_callback=self._open_file,
            parent=self.files_button,
        )
        self.files_button.setMenu(self.files_menu)
        self.tab_widget.setCornerWidget(self.files_button, Qt.TopRightCorner)

        self.const_tab = ConstTab(
            file_open_callback=self._open_file,
            parent=self,
        )
        self.const_tab.file_changed.connect(self._sync_detail_watcher)
        self.tab_widget.addTab(self.const_tab, "Const.")

        self.data_view_manager = DataViewManager(
            parent=self,
            load_more_callback=self._on_load_more,
        )
        self.tab_widget.addTab(self.data_view_manager.widget, "Data")

        self.plot_view = PlotView(parent=self)
        self.tab_widget.addTab(self.plot_view, "Plot")

        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.tab_widget.tabBarClicked.connect(self._remember_tab_choice)
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
        self._remember_tab_choice(index)
        if index == TAB_PLOT:
            if self.plot_view._comparison and not self.plot_view._snapshot_frames:
                self._snapshots.request()
            else:
                self.plot_view.refresh_if_needed()

    def _remember_tab_choice(self, index: int) -> None:
        if index >= 0:
            self._preferred_tab_index = index

    def _on_load_more(self) -> None:
        if self._record:
            self.data_view_manager.load_more_data(
                self._record,
                self._data_cache.dataframe,
            )

    def _on_watch_toggled(self, enabled: bool) -> None:
        if enabled:
            self._sync_detail_watcher()
        else:
            self._refresh_timer.stop()
            self._clear_detail_watcher()

    def _schedule_detail_refresh(self) -> None:
        if not self._refresh_timer.isActive():
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
        if not self.watch_enabled or self._record is None:
            self._clear_detail_watcher()
            return
        watch_paths = record_watch_paths(self._record)
        current_paths = set(
            self._detail_watcher.files() + self._detail_watcher.directories()
        )
        if current_paths == set(watch_paths):
            return
        self._clear_detail_watcher()
        if watch_paths:
            self._detail_watcher.addPaths(watch_paths)

    def _clear_dynamic_tabs(self) -> None:
        for index in range(self.tab_widget.count() - 1, TAB_PLOT, -1):
            widget = self.tab_widget.widget(index)
            self.tab_widget.removeTab(index)
            if widget is not None:
                widget.deleteLater()

    def _create_image_tab(self, image_path: Path) -> QWidget:
        return ImageTab(
            image_path,
            file_open_callback=self._open_file,
            refresh_callback=self.refresh_current_record,
        )

    def _update_image_tabs(self, image_files: list[Path]) -> None:
        was_blocked = self.tab_widget.blockSignals(True)
        try:
            self._clear_dynamic_tabs()
            for image_path in image_files:
                image_tab = self._create_image_tab(image_path)
                self.tab_widget.addTab(image_tab, image_path.name)

            self.tab_widget.setCurrentIndex(
                min(self._preferred_tab_index, self.tab_widget.count() - 1)
            )
        finally:
            self.tab_widget.blockSignals(was_blocked)

    def _open_file(self, path: Path) -> None:
        open_file(path, callback=self._file_open_callback, parent=self)


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
