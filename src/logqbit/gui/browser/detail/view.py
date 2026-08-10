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
    QUrl,
)
from PySide6.QtGui import (
    QAction,
    QDesktopServices,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QMessageBox,
    QWidget,
)
from send2trash import send2trash

from logqbit.file_version import FileVersion

from logqbit.gui.browser.detail.data import DataViewManager
from logqbit.gui.browser.detail.files import (
    ZoomableImageView,
    _copy_file_to_clipboard,
    open_in_file_manager,
)
from logqbit.gui.browser.detail.yaml import YamlView
from logqbit.gui.browser.plot.manager import PlotManager

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord

logger = logging.getLogger(__name__)

REFRESH_DEBOUNCE_MS = 250
TAB_CONST = 0
TAB_DATA = 1
TAB_PLOT = 2


def _format_file_size(size: int) -> str:
    """Return a compact, binary file-size label."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


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
        *record.list_image_files(),
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
        metadata_changed = self._metadata_cache.update(record)
        data_changed = self._data_cache.update(record)
        dataframe = self._data_cache.dataframe
        self._record = record
        self.detail_id_label.setText(f"#{record.log_id}")
        self.detail_label.setText(str(record.path))
        self.yaml_view.set_yaml_text(record.read_yaml_text())
        if metadata_changed or data_changed:
            self.data_view_manager.show_data_table(
                record,
                dataframe,
                preview_only=True,
            )
        self._update_image_tabs(record.list_image_files())
        self.files_button.setEnabled(True)
        defer_plot = self.tab_widget.currentIndex() != TAB_PLOT
        if metadata_changed:
            self.plot_manager.update_controls(record, dataframe)
        if metadata_changed or data_changed:
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
        self.yaml_view.set_yaml_text("")
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
        self.detail_label.setContextMenuPolicy(Qt.NoContextMenu)
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

        self.yaml_view = YamlView()
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
                self._data_cache.dataframe,
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
        image_tab = QWidget()
        image_tab.setContextMenuPolicy(Qt.CustomContextMenu)
        image_tab.customContextMenuRequested.connect(
            lambda position, path=image_path, tab=image_tab: self._open_image_context_menu(
                path, tab, position
            )
        )
        layout = QVBoxLayout(image_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        image_view = ZoomableImageView()
        image_view.setToolTip(str(image_path))
        image_view.set_image_context_menu_callback(
            lambda position, path=image_path, view=image_view: self._open_image_context_menu(
                path, view, position
            )
        )
        image_view.load_image(image_path)
        layout.addWidget(image_view, stretch=1)

        def copy_file(_checked: bool = False) -> None:
            _copy_file_to_clipboard(image_path)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_label = QLabel(
            f"File size: {_format_file_size(image_path.stat().st_size)}",
            image_tab,
        )
        status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_row.addWidget(status_label, stretch=1)
        copy_button = QPushButton("copy")
        copy_button.setToolTip("Copy the image file to the clipboard (Ctrl+C)")
        copy_button.clicked.connect(copy_file)
        status_row.addWidget(copy_button)
        layout.addLayout(status_row)

        copy_shortcut = QShortcut(QKeySequence.Copy, image_tab)
        copy_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        copy_shortcut.activated.connect(copy_file)
        return image_tab

    def _open_image_context_menu(
        self, image_path: Path, widget: QWidget, position
    ) -> None:
        menu = QMenu(widget)
        rename_action = menu.addAction("Rename File...")
        move_to_trash_action = menu.addAction("Move to Recycle Bin...")
        open_action = menu.addAction("Open File")
        chosen = menu.exec(widget.mapToGlobal(position))
        if chosen == rename_action:
            self._rename_image_file(image_path)
        elif chosen == move_to_trash_action:
            self._move_image_file_to_trash(image_path)
        elif chosen == open_action:
            self._open_file(image_path)

    def _rename_image_file(self, image_path: Path) -> None:
        new_name, accepted = QInputDialog.getText(
            self,
            "Rename Image File",
            "New file name:",
            text=image_path.stem,
        )
        new_name = new_name.strip()
        if not accepted or new_name == image_path.stem:
            return
        if (
            not new_name
            or new_name in {".", ".."}
            or Path(new_name).name != new_name
            or "/" in new_name
            or "\\" in new_name
        ):
            QMessageBox.warning(
                self,
                "Rename Image File",
                "Enter a valid file name without a path.",
            )
            return

        new_path = image_path.with_name(f"{new_name}{image_path.suffix}")
        if new_path.exists():
            QMessageBox.warning(
                self,
                "Rename Image File",
                f"A file named {new_name!r} already exists.",
            )
            return
        try:
            image_path.rename(new_path)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Rename Image File",
                f"Failed to rename file:\n{exc}",
            )
            return
        self.refresh_current_record()

    def _move_image_file_to_trash(self, image_path: Path) -> None:
        reply = QMessageBox.question(
            self,
            "Move Image File to Recycle Bin",
            f"Move this file to the Recycle Bin?\n\n{image_path}\n\n"
            "This operation can be undone from the Recycle Bin.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            send2trash(str(image_path))
        except Exception as exc:  # pragma: no cover - platform integration
            QMessageBox.warning(
                self,
                "Move Image File to Recycle Bin",
                f"Failed to move file to Recycle Bin:\n{exc}",
            )
            return
        self.refresh_current_record()

    def _update_image_tabs(self, image_files: list[Path]) -> None:
        first_image_index = TAB_PLOT + 1
        selected_image_offset = self.tab_widget.currentIndex() - first_image_index
        was_blocked = self.tab_widget.blockSignals(True)
        try:
            self._clear_dynamic_tabs()
            for image_path in image_files:
                image_tab = self._create_image_tab(image_path)
                self.tab_widget.addTab(image_tab, image_path.name)

            if selected_image_offset >= 0 and image_files:
                restored_offset = min(selected_image_offset, len(image_files) - 1)
                self.tab_widget.setCurrentIndex(first_image_index + restored_offset)
        finally:
            self.tab_widget.blockSignals(was_blocked)

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
            open_in_file_manager(self._record.path, parent=self)


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
