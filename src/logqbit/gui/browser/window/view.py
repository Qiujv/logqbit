"""Interactive browser window for log folders."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import (
    QFileSystemWatcher,
    QSignalBlocker,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QAction,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from send2trash import send2trash

from logqbit.catalog import LogCatalog, LogRecord, export_records

from logqbit.gui.browser.detail.files import open_in_file_manager
from logqbit.gui.browser.detail.view import RecordDetailView, RecordDetailWindow
from logqbit.gui.browser.plot.mesh import warmup_plotter_jit
from logqbit.gui.browser.window.model import (
    COL_CREATE_MACHINE,
    COL_CREATE_TIME,
    COL_ID,
    COL_PLOT_AXES,
    COL_ROWS,
    COL_TITLE,
    SORT_ROLE,
    LogListTableModel,
)
from logqbit.gui.browser.window.preferences import SettingsManager, ThemeManager

logger = logging.getLogger(__name__)

# Constants
REFRESH_DEBOUNCE_MS = 250
DISABLE_JIT_WARMUP_ENV = "LOGQBIT_BROWSER_DISABLE_JIT_WARMUP"

_plotter_jit_warmup_started = False


def _start_plotter_jit_warmup() -> None:
    global _plotter_jit_warmup_started
    if _plotter_jit_warmup_started:
        return
    if os.environ.get(DISABLE_JIT_WARMUP_ENV):
        return
    _plotter_jit_warmup_started = True

    def run_warmup() -> None:
        try:
            warmup_plotter_jit()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to warm up plotter JIT: %s", exc)

    thread = threading.Thread(
        target=run_warmup,
        name="logqbit-plotter-jit-warmup",
        daemon=True,
    )
    thread.start()


class LogBrowserWindow(QMainWindow):
    def __init__(
        self, directory: Path | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.resize(1200, 700)

        self.settings_manager = SettingsManager()

        # State
        self._base_dir = Path(directory) if directory else Path.cwd()
        self._selected_record: LogRecord | None = None
        self._show_trash = True
        self._show_starred_only = False
        self._shortcuts: list[QAction] = []
        self._list_refresh_pending = False
        self._detail_windows: list[RecordDetailWindow] = []
        self._catalog = LogCatalog()
        self._actions = _BrowserActions(self)

        # Theme management
        app = QApplication.instance()
        self.theme_manager = ThemeManager(app) if app else None
        self._theme_mode = self.settings_manager.load_theme_mode()

        # Recent directories
        recent = self.settings_manager.load_recent_directories()
        if directory:
            self._base_dir = Path(directory)
            self.settings_manager.update_recent_directories(self._base_dir)
        elif recent:
            self._base_dir = recent[0]

        # File watchers
        self._dir_watcher = QFileSystemWatcher(self)
        self._dir_watcher.directoryChanged.connect(self._schedule_list_refresh)
        self._dir_watcher.fileChanged.connect(self._schedule_list_refresh)

        # Build UI
        self._build_ui()
        if self.theme_manager:
            self.theme_manager.apply_theme(self._theme_mode)
        self._update_theme_button()
        self._update_window_title()
        self._sync_directory_watcher()
        self.refresh_logs()
        QTimer.singleShot(500, _start_plotter_jit_warmup)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_bar = self._create_top_bar()
        layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal, central)

        # Left: Log table
        self.log_table, self.table_model, self.table_proxy = self._create_log_table(
            splitter
        )

        # Right: Detail panel
        detail_widget = self._create_detail_panel(splitter)

        splitter.addWidget(self.log_table)
        splitter.addWidget(detail_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 600])

        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self._setup_shortcuts()
        self._rebuild_directory_menu()

    def _create_top_bar(self) -> QHBoxLayout:
        top_bar = QHBoxLayout()
        self.directory_label = QLabel(str(self._base_dir))
        self.directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.directory_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.directory_button = QToolButton()
        self.directory_button.setText("Change dir...")
        self.directory_button.setPopupMode(QToolButton.InstantPopup)
        self._directory_menu = QMenu(self.directory_button)
        self.directory_button.setMenu(self._directory_menu)
        refresh_button = QPushButton("🔄️Refresh")
        refresh_button.setToolTip("Refresh logs and current detail (F5)")
        refresh_button.clicked.connect(self._on_refresh_clicked)
        self.theme_button = QPushButton()
        self.theme_button.setFixedWidth(36)
        self.theme_button.setFocusPolicy(Qt.NoFocus)
        self.theme_button.clicked.connect(self._on_theme_button_clicked)
        top_bar.addWidget(QLabel("Directory:"))
        top_bar.addWidget(self.directory_label)
        top_bar.addWidget(self.directory_button)
        top_bar.addWidget(refresh_button)
        top_bar.addWidget(self.theme_button)
        return top_bar

    def _create_log_table(
        self, parent: QWidget
    ) -> tuple[QTableView, LogListTableModel, QSortFilterProxyModel]:
        model = LogListTableModel(parent)

        proxy = QSortFilterProxyModel(parent)
        proxy.setSourceModel(model)
        proxy.setSortRole(SORT_ROLE)

        table = QTableView(parent)
        table.setModel(proxy)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.setSelectionMode(QTableView.ExtendedSelection)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)

        font_height = table.fontMetrics().height()
        table.verticalHeader().setDefaultSectionSize(font_height + 4)

        header = table.horizontalHeader()
        header.setSectionResizeMode(COL_ID, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_TITLE, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_ROWS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_PLOT_AXES, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_CREATE_TIME, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_CREATE_MACHINE, QHeaderView.ResizeToContents)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(False)  # For compact view.
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(
            self._actions.open_header_context_menu
        )

        table.setColumnHidden(COL_CREATE_TIME, True)
        table.setColumnHidden(COL_CREATE_MACHINE, True)

        table.selectionModel().selectionChanged.connect(self._on_log_selection_changed)
        table.doubleClicked.connect(self._on_log_double_clicked)
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(self._actions.open_table_context_menu)

        table.sortByColumn(COL_ID, Qt.AscendingOrder)

        return table, model, proxy

    def _create_detail_panel(self, parent: QWidget) -> QWidget:
        self.detail_view = RecordDetailView(
            parent=parent,
            enable_tab_shortcuts=False,
        )
        self.detail_view.record_refreshed.connect(self._on_detail_record_refreshed)
        return self.detail_view

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

        add_shortcut(Qt.Key_Delete, self._actions.shortcut_send_to_recycle_bin)
        add_shortcut(Qt.Key_T, self._actions.shortcut_toggle_trash)
        add_shortcut(Qt.Key_S, self._actions.shortcut_toggle_star)
        add_shortcut(Qt.Key_F2, self._actions.shortcut_rename_title)
        add_shortcut(Qt.Key_F5, self._on_refresh_clicked)
        add_shortcut(Qt.Key_0, lambda: self._actions.shortcut_set_star(0))
        add_shortcut(Qt.Key_1, lambda: self._actions.shortcut_set_star(1))
        add_shortcut(Qt.Key_2, lambda: self._actions.shortcut_set_star(2))
        add_shortcut(Qt.Key_3, lambda: self._actions.shortcut_set_star(3))
        add_shortcut(Qt.Key_Left, lambda: self.detail_view.switch_tab(-1))
        add_shortcut(Qt.Key_Right, lambda: self.detail_view.switch_tab(1))

        self.open_explorer_shortcut = QShortcut(
            QKeySequence("Ctrl+Return"),
            self.log_table,
        )
        self.open_explorer_shortcut.setKeys(
            [QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")]
        )
        self.open_explorer_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.open_explorer_shortcut.activated.connect(
            self._actions.shortcut_open_in_explorer
        )

    def _rebuild_directory_menu(self) -> None:
        if self._directory_menu is None:
            return
        self._directory_menu.clear()
        recent = self.settings_manager._recent_directories
        # Filter out current directory
        menu_items = [path for path in recent if path != self._base_dir]
        for path in menu_items:
            action = self._directory_menu.addAction(str(path))
            action.triggered.connect(
                lambda _checked=False, target=path: self.set_directory(target)
            )
        if menu_items:
            self._directory_menu.addSeparator()
        open_action = self._directory_menu.addAction("Open Other Folder...")
        open_action.triggered.connect(self._open_directory_dialog)
        clear_action = self._directory_menu.addAction("Clear Recent Folders")
        clear_action.setEnabled(bool(menu_items))
        clear_action.triggered.connect(self._clear_recent_directories)
        new_window_action = self._directory_menu.addAction("New Window")
        new_window_action.triggered.connect(
            lambda: self._open_new_window(self._base_dir)
        )

    def _clear_recent_directories(self) -> None:
        self.settings_manager.clear_recent_directories(keep=self._base_dir)
        self._rebuild_directory_menu()

    def _update_theme_button(self) -> None:
        if not self.theme_manager:
            return
        emoji = self.theme_manager.get_theme_button_emoji(self._theme_mode)
        tooltip = self.theme_manager.get_theme_tooltip(self._theme_mode)
        if hasattr(self, "theme_button"):
            self.theme_button.setText(emoji)
            self.theme_button.setToolTip(tooltip)

    def _update_window_title(self) -> None:
        self.setWindowTitle(f"{self._base_dir.name} - LogQbit Browser")

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
        self.refresh_logs()

    def set_directory(self, directory: Path) -> None:
        path = Path(directory)
        if path != self._base_dir:
            self._base_dir = path
            self.directory_label.setText(str(self._base_dir))
            self._update_window_title()
            self._sync_directory_watcher()
            self.refresh_logs()
        else:
            self.directory_label.setText(str(self._base_dir))
        self.settings_manager.update_recent_directories(path)
        self._rebuild_directory_menu()

    def refresh_logs(self) -> None:
        previous_record = self._selected_record
        previous_path = previous_record.path if previous_record else None
        all_records = self._catalog.refresh(self._base_dir)

        # Filter out trash if needed
        if self._show_trash:
            records = all_records
        else:
            records = [record for record in all_records if not record.trash]
        if self._show_starred_only:
            records = [record for record in records if record.star > 0]

        self.table_model.set_records(records)

        row_count = self.table_proxy.rowCount()
        if row_count:
            selected_record = next(
                (record for record in records if record.path == previous_path),
                None,
            )
            if selected_record is None:
                source_row = 0
                selected_record = records[source_row]
            else:
                source_row = records.index(selected_record)

            source_index = self.table_model.index(source_row, 0)
            proxy_index = self.table_proxy.mapFromSource(source_index)
            selection_blocker = QSignalBlocker(self.log_table.selectionModel())
            try:
                self.log_table.selectRow(proxy_index.row())
            finally:
                selection_blocker.unblock()

            self._selected_record = selected_record
            if selected_record is not previous_record:
                self._load_log(selected_record)
        else:
            if all_records:
                self.detail_view.clear("No logs to display.")
            else:
                self.detail_view.clear("No logs found.")
            self._selected_record = None
            self.log_table.clearSelection()

    def refresh_current_log(self, *, force: bool = False) -> None:
        if not self._selected_record:
            return
        self.detail_view.refresh_current_record(force=force)

    def _on_log_double_clicked(self, proxy_index) -> None:
        """Open a standalone detail window for the double-clicked log record."""
        source_index = self.table_proxy.mapToSource(proxy_index)
        record = self.table_model.get_record(source_index.row())
        if record is None:
            return
        initial_tab = self.detail_view.current_tab_index()
        window = RecordDetailWindow(record, initial_tab=initial_tab, parent=None)
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        self._detail_windows.append(window)
        window.destroyed.connect(
            lambda: (
                self._detail_windows.remove(window)
                if window in self._detail_windows
                else None
            )
        )
        window.show()

    def _on_log_selection_changed(self) -> None:
        selected = self.log_table.selectionModel().selectedRows()
        if not selected:
            return
        proxy_index = selected[0]
        source_index = self.table_proxy.mapToSource(proxy_index)
        record = self.table_model.get_record(source_index.row())
        if record is None:
            return
        self._selected_record = record
        self._load_log(record)

    def _load_log(self, record: LogRecord) -> None:
        self.detail_view.load_record(record)

    def _on_detail_record_refreshed(self, record: LogRecord) -> None:
        if (
            self._selected_record is not None
            and self._selected_record.path == record.path
        ):
            self._selected_record = record
            self.table_model.notify_record_changed(record)

    def _on_refresh_clicked(self) -> None:
        previous_record = self._selected_record
        self.refresh_logs()
        self.refresh_current_log(force=self._selected_record is previous_record)

    def _on_theme_button_clicked(self) -> None:
        current_index = ThemeManager.THEME_MODES.index(self._theme_mode)
        next_index = (current_index + 1) % len(ThemeManager.THEME_MODES)
        self._theme_mode = ThemeManager.THEME_MODES[next_index]
        if self.theme_manager:
            self.theme_manager.apply_theme(self._theme_mode)
        self.settings_manager.save_theme_mode(self._theme_mode)
        self._update_theme_button()

    def _open_directory_dialog(self) -> None:
        current = str(self._base_dir)
        chosen = QFileDialog.getExistingDirectory(self, "Select log directory", current)
        if chosen:
            self.set_directory(Path(chosen))

    def _open_new_window(self, directory: Path) -> None:
        """Launch a new browser window in a separate process."""
        try:
            from logqbit.gui.browser.startup.launcher import start_browser

            start_browser(directory)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Launch Error",
                f"Failed to launch new window:\n{exc}",
            )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override naming
        self.settings_manager.save_recent_directories(
            self.settings_manager._recent_directories
        )
        self.settings_manager.save_theme_mode(self._theme_mode)
        super().closeEvent(event)


class _BrowserActions:
    """Group menus and record mutations initiated from the main window."""

    def __init__(self, window: LogBrowserWindow) -> None:
        self.window = window

    def get_selected_records(self) -> list[LogRecord]:
        selection_model = self.window.log_table.selectionModel()
        if selection_model is None:
            return []
        records: list[LogRecord] = []
        for proxy_index in selection_model.selectedRows():
            source_index = self.window.table_proxy.mapToSource(proxy_index)
            record = self.window.table_model.get_record(source_index.row())
            if record is not None:
                records.append(record)
        return records

    def open_table_context_menu(self, point) -> None:
        records = self.get_selected_records()
        menu = QMenu(self.window)
        show_trash_action = menu.addAction("Show Trashed Items")
        show_trash_action.setCheckable(True)
        show_trash_action.setChecked(self.window._show_trash)
        show_starred_action = menu.addAction("Show Starred Items Only")
        show_starred_action.setCheckable(True)
        show_starred_action.setChecked(self.window._show_starred_only)
        menu.addSeparator()
        rename_action = menu.addAction("Rename Title... (F2)")
        toggle_star_action = menu.addAction("Toggle ⭐Star (S)")
        toggle_star_action.setCheckable(True)
        toggle_trash_action = menu.addAction("Toggle 🗑️Trash (T)")
        toggle_trash_action.setCheckable(True)
        send_to_recycle_action = menu.addAction("Send to Recycle Bin (Del)")
        open_explorer = menu.addAction("Open in Explorer (Ctrl+Enter)")
        export_action = menu.addAction("Export Items...")
        if not records:
            rename_action.setEnabled(False)
            toggle_star_action.setEnabled(False)
            toggle_trash_action.setEnabled(False)
            send_to_recycle_action.setEnabled(False)
            open_explorer.setEnabled(False)
            export_action.setEnabled(False)
        else:
            rename_action.setEnabled(len(records) == 1)
            toggle_star_action.setChecked(all(record.star > 0 for record in records))
            toggle_trash_action.setChecked(all(record.trash for record in records))
        chosen = menu.exec(self.window.log_table.viewport().mapToGlobal(point))
        if chosen is None:
            return
        if chosen == rename_action and len(records) == 1:
            self.rename_record_title(records[0])
        elif chosen == toggle_star_action and records:
            self.set_records_star_count(
                records, 1 if toggle_star_action.isChecked() else 0
            )
        elif chosen == toggle_trash_action and records:
            self.set_records_trash(records, toggle_trash_action.isChecked())
        elif chosen == send_to_recycle_action and records:
            self.send_records_to_recycle_bin(records)
        elif chosen == show_trash_action:
            self.toggle_show_trash()
        elif chosen == show_starred_action:
            self.toggle_show_starred_only()
        elif chosen == open_explorer and records:
            self.open_path_in_explorer(records[0].path, len(records) != 1)
        elif chosen == export_action and records:
            self.export_records(records)

    def open_header_context_menu(self, point) -> None:
        menu = self.create_header_context_menu()
        header = self.window.log_table.horizontalHeader()
        menu.exec(header.mapToGlobal(point))

    def create_header_context_menu(self) -> QMenu:
        menu = QMenu(self.window)
        for label, column in (
            ("Show Plot Axes Column", COL_PLOT_AXES),
            ("Show Create Time Column", COL_CREATE_TIME),
            ("Show Create Machine Column", COL_CREATE_MACHINE),
        ):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(not self.window.log_table.isColumnHidden(column))
            action.triggered.connect(
                lambda checked=False, target=column: self.toggle_column(target, checked)
            )
        return menu

    def toggle_column(self, column: int, visible: bool) -> None:
        self.window.log_table.setColumnHidden(column, not visible)

    def toggle_show_trash(self) -> None:
        self.window._show_trash = not self.window._show_trash
        self.window.refresh_logs()

    def toggle_show_starred_only(self) -> None:
        self.window._show_starred_only = not self.window._show_starred_only
        self.window.refresh_logs()

    def rename_record_title(self, record: LogRecord) -> None:
        current_title = record.title
        dialog = QInputDialog(self.window)
        dialog.setWindowTitle("Rename Log")
        dialog.setLabelText("Enter new title:")
        dialog.setTextValue(current_title)
        dialog.setInputMode(QInputDialog.TextInput)
        dialog.resize(max(dialog.sizeHint().width(), 600), dialog.sizeHint().height())
        if dialog.exec() != QDialog.Accepted:
            return
        new_title = dialog.textValue().strip()
        if new_title == current_title:
            return
        record.meta.update(title=new_title)
        self.window.refresh_logs()

    def set_record_star_count(
        self, record: LogRecord, count: int, refresh: bool = True
    ) -> bool:
        count = max(int(count), 0)
        if record.star == count:
            return False
        record.meta.update(star=count)
        if refresh:
            self.window.refresh_logs()
        return True

    def set_record_trash(
        self, record: LogRecord, value: bool, refresh: bool = True
    ) -> bool:
        value = bool(value)
        if record.trash == value:
            return False
        record.meta.update(trash=value)
        if refresh:
            self.window.refresh_logs()
        return True

    def set_records_star_count(self, records: Iterable[LogRecord], count: int) -> None:
        changed = False
        for record in records:
            changed |= self.set_record_star_count(record, count, refresh=False)
        if changed:
            self.window.refresh_logs()

    def set_records_trash(self, records: Iterable[LogRecord], value: bool) -> None:
        changed = False
        for record in records:
            changed |= self.set_record_trash(record, value, refresh=False)
        if changed:
            self.window.refresh_logs()

    def open_path_in_explorer(self, path: Path, select: bool = False) -> None:
        open_in_file_manager(path, select=select, parent=self.window)

    def export_records(self, records: Iterable[LogRecord]) -> None:
        records_list = list(records)
        if not records_list:
            return

        chosen = QFileDialog.getExistingDirectory(
            self.window,
            "Select new parent folder for export",
            str(
                self.window._base_dir.parent
                if self.window._base_dir.parent.exists()
                else self.window._base_dir
            ),
        )
        if not chosen:
            return

        destination_parent = Path(chosen)
        try:
            exported_paths = export_records(records_list, destination_parent)
        except Exception as exc:
            QMessageBox.warning(
                self.window,
                "Export Failed",
                f"Failed to export selected log folders:\n{exc}",
            )
            return

        if len(exported_paths) == 1:
            message = f"Exported 1 log folder to:\n{exported_paths[0]}"
        else:
            message = (
                f"Exported {len(exported_paths)} log folders to parent folder:\n"
                f"{destination_parent}"
            )
        QMessageBox.information(self.window, "Export Complete", message)

    def send_records_to_recycle_bin(self, records: Iterable[LogRecord]) -> None:
        records_list = list(records)
        if not records_list:
            return

        if len(records_list) == 1:
            message = f"Send log folder #{records_list[0].log_id} to Recycle Bin?\n\n"
            message += f"Path: {records_list[0].path}\n\n"
            message += "This operation can be undone from the Recycle Bin."
        else:
            message = f"Send {len(records_list)} log folders to Recycle Bin?\n\n"
            message += "IDs: " + ", ".join(f"#{r.log_id}" for r in records_list[:10])
            if len(records_list) > 10:
                message += f", ... (+{len(records_list) - 10} more)"
            message += "\n\nThis operation can be undone from the Recycle Bin."

        reply = QMessageBox.question(
            self.window,
            "Confirm Send to Recycle Bin",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        failed_paths = []
        for record in records_list:
            try:
                send2trash(str(record.path))
            except Exception as exc:
                failed_paths.append(f"{record.path} ({exc})")

        if failed_paths:
            error_msg = "Failed to send some folders to Recycle Bin:\n\n"
            error_msg += "\n".join(failed_paths[:5])
            if len(failed_paths) > 5:
                error_msg += f"\n... and {len(failed_paths) - 5} more"
            QMessageBox.warning(self.window, "Error", error_msg)

        self.window.refresh_logs()

    def shortcut_set_star(self, count: int) -> None:
        records = self.get_selected_records()
        if records:
            self.set_records_star_count(records, count)

    def shortcut_toggle_star(self) -> None:
        records = self.get_selected_records()
        if not records:
            return
        all_starred = all(record.star > 0 for record in records)
        self.set_records_star_count(records, 0 if all_starred else 1)

    def shortcut_send_to_recycle_bin(self) -> None:
        records = self.get_selected_records()
        if records:
            self.send_records_to_recycle_bin(records)

    def shortcut_toggle_trash(self) -> None:
        records = self.get_selected_records()
        if not records:
            return
        all_trashed = all(record.trash for record in records)
        self.set_records_trash(records, not all_trashed)

    def shortcut_open_in_explorer(self) -> None:
        records = self.get_selected_records()
        if records:
            self.open_path_in_explorer(records[0].path, len(records) != 1)

    def shortcut_rename_title(self) -> None:
        records = self.get_selected_records()
        if len(records) == 1:
            self.rename_record_title(records[0])
