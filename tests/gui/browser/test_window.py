from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest

from logqbit import catalog as catalog_module
from logqbit.catalog import LogCatalog, LogRecord
from logqbit.gui.browser.window.model import (
    COL_CREATE_TIME,
    COL_ROWS,
)
from logqbit.gui.browser.window.view import LogBrowserWindow
from logqbit.gui.browser.startup.bootstrap import ensure_application


def scan_catalog(directory: Path) -> list[LogRecord]:
    return LogCatalog(directory).refresh()


class TestBrowserWindow:
    def test_browser_window_detail_shortcuts(self, sample_logfolder: Path) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_logfolder)
        window.show()
        app.processEvents()
        try:
            assert window.detail_view.current_record is not None

            window.detail_view.yaml_view.setFocus()
            QTest.keyClick(window.detail_view.yaml_view, Qt.Key_Right)
            app.processEvents()
            assert window.detail_view.current_tab_index() == 1

            window.detail_view.data_view_manager.data_table.setFocus()
            QTest.keyClick(window.detail_view.data_view_manager.data_table, Qt.Key_Left)
            app.processEvents()
            assert window.detail_view.current_tab_index() == 0
        finally:
            window.close()

    def test_browser_f5_runs_manual_refresh(
        self,
        sample_logfolder: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        calls: list[tuple[str, bool | None]] = []
        monkeypatch.setattr(
            window,
            "refresh_logs",
            lambda: calls.append(("logs", None)),
        )
        monkeypatch.setattr(
            window,
            "refresh_current_log",
            lambda *, force=False: calls.append(("detail", force)),
        )
        refresh_action = next(
            action
            for action in window._shortcuts
            if action.shortcut() == QKeySequence(Qt.Key_F5)
        )

        refresh_action.trigger()

        assert calls == [("logs", None), ("detail", True)]
        window.close()

    def test_directory_menu_can_clear_recent_folders(
        self,
        sample_logfolder: Path,
        tmp_path: Path,
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        window.settings_manager._settings = QSettings(
            str(tmp_path / "browser-settings.ini"),
            QSettings.IniFormat,
        )
        window.settings_manager.save_recent_directories(
            [sample_logfolder, tmp_path / "other"]
        )
        window._rebuild_directory_menu()

        clear_action = next(
            action
            for action in window._directory_menu.actions()
            if action.text() == "Clear Recent Folders"
        )
        assert clear_action.isEnabled()

        clear_action.trigger()

        assert window.settings_manager.load_recent_directories() == [sample_logfolder]
        rebuilt_clear_action = next(
            action
            for action in window._directory_menu.actions()
            if action.text() == "Clear Recent Folders"
        )
        assert not rebuilt_clear_action.isEnabled()
        window.close()

    def test_browser_can_show_starred_records_only(
        self,
        sample_records: list[LogRecord],
    ) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_records[0].path.parent)
        app.processEvents()
        try:
            assert window.table_model.rowCount() == 3

            window._actions.toggle_show_starred_only()

            assert window.table_model.rowCount() == 1
            record = window.table_model.get_record(0)
            assert record is not None
            assert record.star > 0

            window._actions.toggle_show_starred_only()
            assert window.table_model.rowCount() == 3
        finally:
            window.close()

    def test_column_visibility_actions_are_in_header_context_menu(
        self,
        sample_logfolder: Path,
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        try:
            assert window.log_table.isColumnHidden(COL_CREATE_TIME)

            menu = window._actions.create_header_context_menu()
            actions = menu.actions()

            assert [action.text() for action in actions] == [
                "Show Plot Axes Column",
                "Show Create Time Column",
                "Show Create Machine Column",
            ]
            actions[1].trigger()
            assert not window.log_table.isColumnHidden(COL_CREATE_TIME)
        finally:
            window.close()

    def test_open_explorer_shortcut_is_scoped_to_log_table(
        self,
        sample_logfolder: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_logfolder)
        window.show()
        app.processEvents()
        try:
            assert set(window.open_explorer_shortcut.keys()) == {
                QKeySequence("Ctrl+Return"),
                QKeySequence("Ctrl+Enter"),
            }
            assert (
                window.open_explorer_shortcut.context() == Qt.WidgetWithChildrenShortcut
            )
            assert window.open_explorer_shortcut.parent() is window.log_table

            opened: list[tuple[Path, bool]] = []
            monkeypatch.setattr(
                window._actions,
                "open_path_in_explorer",
                lambda path, select=False: opened.append((path, select)),
            )
            window.detail_view.yaml_view.setFocus()
            QTest.keyClick(
                window.detail_view.yaml_view,
                Qt.Key_Return,
                Qt.ControlModifier,
            )
            assert not opened

            window.log_table.setFocus()
            QTest.keyClick(window.log_table, Qt.Key_Return, Qt.ControlModifier)
            assert opened == [(window._selected_record.path, False)]
        finally:
            window.close()

    def test_list_refresh_preserves_current_detail_cache(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_logfolder)
        app.processEvents()
        try:
            record = window._selected_record
            assert record is not None
            dataframe = window.detail_view._data_cache.dataframe
            assert dataframe is not None

            read_count = 0
            original_read_feather = catalog_module.pd.read_feather

            def count_read_feather(*args, **kwargs):
                nonlocal read_count
                read_count += 1
                return original_read_feather(*args, **kwargs)

            monkeypatch.setattr(catalog_module.pd, "read_feather", count_read_feather)

            window.refresh_logs()

            assert window._selected_record is record
            assert window.detail_view._data_cache.dataframe is dataframe
            assert read_count == 0
        finally:
            window.close()

    def test_manual_refresh_reads_current_feather_once(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_logfolder)
        app.processEvents()
        try:
            read_count = 0
            original_read_feather = catalog_module.pd.read_feather

            def count_read_feather(*args, **kwargs):
                nonlocal read_count
                read_count += 1
                return original_read_feather(*args, **kwargs)

            monkeypatch.setattr(catalog_module.pd, "read_feather", count_read_feather)

            window._on_refresh_clicked()

            assert read_count == 1
        finally:
            window.close()

    def test_manual_refresh_updates_catalog_summary_and_detail(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_logfolder)
        app.processEvents()
        try:
            record = window._selected_record
            assert record is not None
            assert record.data_path.exists()
            pd.DataFrame({"x": range(10), "y": range(10)}).to_feather(record.data_path)

            inspect_count = 0
            read_count = 0
            original_open_file = catalog_module.pyarrow.ipc.open_file
            original_read_feather = catalog_module.pd.read_feather

            def count_open_file(*args, **kwargs):
                nonlocal inspect_count
                inspect_count += 1
                return original_open_file(*args, **kwargs)

            def count_read_feather(*args, **kwargs):
                nonlocal read_count
                read_count += 1
                return original_read_feather(*args, **kwargs)

            monkeypatch.setattr(
                catalog_module.pyarrow.ipc, "open_file", count_open_file
            )
            monkeypatch.setattr(catalog_module.pd, "read_feather", count_read_feather)

            window._on_refresh_clicked()

            assert inspect_count == 1
            assert read_count == 1
            assert record.row_count == 3
            assert window._selected_record is not None
            assert window._selected_record.row_count == 10
        finally:
            window.close()

    def test_detail_data_refresh_updates_log_list_rows(
        self, sample_logfolder: Path
    ) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_logfolder)
        app.processEvents()
        try:
            record = window._selected_record
            assert record is not None
            pd.DataFrame({"x": range(9), "y": range(9)}).to_feather(record.data_path)

            window.detail_view.refresh_current_record()

            refreshed = window._selected_record
            assert refreshed is not None
            assert refreshed is window.detail_view.current_record
            assert refreshed.row_count == 9
            source_row = next(
                row
                for row in range(window.table_model.rowCount())
                if window.table_model.get_record(row) is refreshed
            )
            assert (
                window.table_model.data(
                    window.table_model.index(source_row, COL_ROWS),
                    Qt.DisplayRole,
                )
                == "9"
            )

            dataframe = window.detail_view._data_cache.dataframe
            window.refresh_logs()

            assert window._selected_record is window.detail_view.current_record
            assert window._selected_record.row_count == 9
            assert window.detail_view._data_cache.dataframe is dataframe
        finally:
            window.close()

    def test_browser_window_watch_toggle_controls_watcher(
        self, sample_logfolder: Path
    ) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_logfolder)
        window.show()
        app.processEvents()
        try:
            record = window.detail_view.current_record
            assert record is not None
            assert str(record.path) in set(
                window.detail_view._detail_watcher.directories()
            )

            window.detail_view.set_watch_enabled(False)
            app.processEvents()
            assert not window.detail_view._detail_watcher.directories()
            assert not window.detail_view._detail_watcher.files()

            window.detail_view.set_watch_enabled(True)
            app.processEvents()
            assert str(record.path) in set(
                window.detail_view._detail_watcher.directories()
            )
        finally:
            window.close()

    def test_browser_window_log_table_focus_can_switch_detail_tabs(
        self, sample_logfolder: Path
    ) -> None:
        app = ensure_application()
        window = LogBrowserWindow(sample_logfolder)
        window.show()
        app.processEvents()
        try:
            window.log_table.setFocus()
            QTest.keyClick(window.log_table, Qt.Key_Right)
            app.processEvents()
            assert window.detail_view.current_tab_index() == 1

            QTest.keyClick(window.log_table, Qt.Key_Left)
            app.processEvents()
            assert window.detail_view.current_tab_index() == 0
        finally:
            window.close()
