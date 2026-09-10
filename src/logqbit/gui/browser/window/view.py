"""Interactive Browser window and cross-component coordination."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from logqbit.catalog import LogRecord
from logqbit.gui.browser.detail.view import RecordDetailView, RecordDetailWindow
from logqbit.gui.browser.plot.mesh import warmup_plotter_jit
from logqbit.gui.browser.window.navigation import RecordNavigation
from logqbit.gui.browser.window.preferences import SettingsManager, ThemeManager
from logqbit.gui.browser.window.records import RecordActions

logger = logging.getLogger(__name__)
DISABLE_JIT_WARMUP_ENV = "LOGQBIT_BROWSER_DISABLE_JIT_WARMUP"
_plotter_jit_warmup_started = False


def _start_plotter_jit_warmup() -> None:
    global _plotter_jit_warmup_started
    if _plotter_jit_warmup_started or os.environ.get(DISABLE_JIT_WARMUP_ENV):
        return
    _plotter_jit_warmup_started = True

    def run_warmup() -> None:
        try:
            warmup_plotter_jit()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to warm up plotter JIT: %s", exc)

    threading.Thread(
        target=run_warmup, name="logqbit-plotter-jit-warmup", daemon=True
    ).start()


class LogBrowserWindow(QMainWindow):
    """Assemble Browser components and coordinate detail windows."""

    def __init__(
        self, directory: Path | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.resize(1200, 700)
        self.settings_manager = SettingsManager()
        recent = self.settings_manager.load_recent_directories()
        initial_directory = (
            Path(directory) if directory else (recent[0] if recent else Path.cwd())
        )
        if directory:
            self.settings_manager.update_recent_directories(initial_directory)
        self._theme_mode = self.settings_manager.load_theme_mode()
        app = QApplication.instance()
        self.theme_manager = ThemeManager(app) if app else None
        self._detail_windows: list[RecordDetailWindow] = []
        self._shortcuts: list[QAction] = []

        self._build_ui(initial_directory)
        self._actions = RecordActions(self.navigation, self)
        self._actions.records_changed.connect(self.navigation.refresh_logs)
        self._actions.files_written.connect(self.navigation.schedule_list_refresh)
        self._actions.path_renamed.connect(self._on_path_renamed)
        self.navigation.log_table.customContextMenuRequested.connect(
            self._actions.open_table_context_menu
        )
        self._setup_shortcuts()
        self._rebuild_directory_menu()
        if self.theme_manager:
            self.theme_manager.apply_theme(self._theme_mode)
        self._update_theme_button()
        self._update_window_title()
        self.navigation.refresh_logs()
        QTimer.singleShot(500, _start_plotter_jit_warmup)

    def _build_ui(self, directory: Path) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        splitter = QSplitter(Qt.Horizontal, central)
        self.navigation = RecordNavigation(directory, self.settings_manager, splitter)
        layout.addLayout(self._create_top_bar())
        self.detail_view = RecordDetailView(parent=splitter, enable_tab_shortcuts=False)
        splitter.addWidget(self.navigation)
        splitter.addWidget(self.detail_view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 600])
        layout.addWidget(splitter)
        self.setCentralWidget(central)
        self.navigation.display_record.connect(self.detail_view.load_record)
        self.navigation.open_record.connect(self._open_record_window)
        self.navigation.empty_selection.connect(self.detail_view.clear)
        self.navigation.directory_changed.connect(self._directory_changed)
        self.navigation.about_requested.connect(self.show_about_dialog)
        self.detail_view.record_refreshed.connect(self.navigation.notify_record_changed)

    def _create_top_bar(self) -> QHBoxLayout:
        top_bar = QHBoxLayout()
        self.directory_label = QLabel(self.navigation.base_dir.as_posix())
        self.directory_label.setWordWrap(True)
        self.directory_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.directory_label.setFocusPolicy(Qt.NoFocus)
        self.directory_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.directory_label.customContextMenuRequested.connect(
            self._show_top_bar_context_menu
        )
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

    def _setup_shortcuts(self) -> None:
        def add_shortcut(key: int | QKeySequence.StandardKey, callback) -> None:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda _checked=False, cb=callback: cb())
            self.addAction(action)
            self._shortcuts.append(action)

        actions = self._actions
        add_shortcut(Qt.Key_Delete, actions.shortcut_send_to_recycle_bin)
        add_shortcut(Qt.Key_T, actions.shortcut_toggle_trash)
        add_shortcut(Qt.Key_S, actions.shortcut_toggle_star)
        add_shortcut(Qt.Key_F2, actions.shortcut_rename_title)
        add_shortcut(Qt.Key_F3, actions.shortcut_change_id)
        add_shortcut(Qt.Key_F5, self._on_refresh_clicked)
        add_shortcut(QKeySequence("Ctrl+P"), actions.shortcut_pin_record)
        add_shortcut(QKeySequence.New, actions.shortcut_make_note)
        add_shortcut(Qt.Key_0, lambda: actions.shortcut_set_star(0))
        add_shortcut(Qt.Key_1, lambda: actions.shortcut_set_star(1))
        add_shortcut(Qt.Key_2, lambda: actions.shortcut_set_star(2))
        add_shortcut(Qt.Key_3, lambda: actions.shortcut_set_star(3))
        add_shortcut(Qt.Key_Left, lambda: self.detail_view.switch_tab(-1))
        add_shortcut(Qt.Key_Right, lambda: self.detail_view.switch_tab(1))
        self.open_explorer_shortcut = QShortcut(
            QKeySequence("Ctrl+Return"), self.navigation.log_table
        )
        self.open_explorer_shortcut.setKeys(
            [QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")]
        )
        self.open_explorer_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.open_explorer_shortcut.activated.connect(actions.shortcut_open_in_explorer)

    def _show_top_bar_context_menu(self, position) -> None:
        menu = self.navigation.create_header_context_menu()
        menu.exec(self.directory_label.mapToGlobal(position))

    def show_about_dialog(self) -> None:
        from logqbit.gui.browser.about import about_message

        dialog = QMessageBox(self)
        dialog.setWindowTitle("About LogQbit")
        browser_icon = self.windowIcon()
        if browser_icon.isNull():
            dialog.setIcon(QMessageBox.Information)
        else:
            dialog.setIconPixmap(browser_icon.pixmap(64, 64))
        dialog.setText(about_message())
        dialog.setTextInteractionFlags(
            Qt.TextSelectableByMouse
            | Qt.TextSelectableByKeyboard
            | Qt.LinksAccessibleByMouse
        )
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()

    def _rebuild_directory_menu(self) -> None:
        self._directory_menu.clear()
        recent = self.settings_manager._recent_directories
        menu_items = [path for path in recent if path != self.navigation.base_dir]
        for path in menu_items:
            action = self._directory_menu.addAction(str(path))
            action.triggered.connect(
                lambda _checked=False, target=path: self.navigation.set_directory(
                    target
                )
            )
        clear_action = self._directory_menu.addAction("Cleanup")
        if menu_items:
            self._directory_menu.addSeparator()
        open_action = self._directory_menu.addAction("Open Other...")
        open_action.triggered.connect(self._open_directory_dialog)
        clear_action.setEnabled(bool(menu_items))
        clear_action.triggered.connect(self._clear_recent_directories)
        new_window_action = self._directory_menu.addAction("New Window")
        new_window_action.triggered.connect(
            lambda: self._open_new_window(self.navigation.base_dir)
        )

    def _clear_recent_directories(self) -> None:
        existing = [
            path for path in self.settings_manager._recent_directories if path.exists()
        ]
        self.settings_manager.save_recent_directories(existing)
        self._rebuild_directory_menu()

    def _directory_changed(self, directory: Path) -> None:
        self.directory_label.setText(self.navigation.base_dir.as_posix())
        self._update_window_title()
        self._rebuild_directory_menu()

    def _update_theme_button(self) -> None:
        if self.theme_manager:
            self.theme_button.setText(
                self.theme_manager.get_theme_button_emoji(self._theme_mode)
            )
            self.theme_button.setToolTip(
                self.theme_manager.get_theme_tooltip(self._theme_mode)
            )

    def _update_window_title(self) -> None:
        self.setWindowTitle(
            f"{self.navigation.base_dir.parent.name} / {self.navigation.base_dir.name} - LogQbit Browser"
        )

    def _open_record_window(self, record: LogRecord) -> None:
        window = RecordDetailWindow(
            record, initial_tab=self.detail_view.current_tab_index(), parent=None
        )
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

    def _on_path_renamed(self, old_path: Path, new_path: Path) -> None:
        self.navigation.rename_pinned_record(old_path, new_path)
        self.navigation.refresh_logs(preferred_path=new_path)
        renamed_record = next(
            (
                record
                for record in self.navigation.all_records
                if record.path == new_path
            ),
            None,
        )
        if renamed_record is None:
            return
        for window in list(self._detail_windows):
            current = window.detail_view.current_record
            if current is not None and current.path == old_path:
                window.load_record(renamed_record)

    def _on_refresh_clicked(self) -> None:
        previous_record = self.navigation.selected_record
        self.navigation.refresh_logs()
        current = self.navigation.selected_record
        if current is not None:
            self.detail_view.refresh_current_record(force=current is previous_record)

    def _on_theme_button_clicked(self) -> None:
        current_index = ThemeManager.THEME_MODES.index(self._theme_mode)
        self._theme_mode = ThemeManager.THEME_MODES[
            (current_index + 1) % len(ThemeManager.THEME_MODES)
        ]
        if self.theme_manager:
            self.theme_manager.apply_theme(self._theme_mode)
        self.settings_manager.save_theme_mode(self._theme_mode)
        self._update_theme_button()

    def _open_directory_dialog(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Select log directory", str(self.navigation.base_dir.parent)
        )
        if chosen:
            self.navigation.set_directory(Path(chosen))

    def _open_new_window(self, directory: Path) -> None:
        try:
            from logqbit.gui.browser.startup import launch_browser

            launch_browser(directory)
        except Exception as exc:
            QMessageBox.warning(
                self, "Launch Error", f"Failed to launch new window:\n{exc}"
            )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.settings_manager.save_recent_directories(
            self.settings_manager._recent_directories
        )
        self.settings_manager.save_theme_mode(self._theme_mode)
        super().closeEvent(event)
