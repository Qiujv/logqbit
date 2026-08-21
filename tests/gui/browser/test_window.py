from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6.QtCore import QEventLoop, QSettings, Qt, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

from logqbit import catalog as catalog_module
from logqbit.catalog import (
    LOGFOLDER_SID_COLUMN,
    LogCatalog,
    LogRecord,
    MergeRecordsError,
    MergeRecordsResult,
)
from logqbit.gui.browser.window.model import COL_ROWS
from logqbit.gui.browser.window import merge as merge_module
from logqbit.gui.browser.window.merge import MergeDialog
from logqbit.gui.browser.detail.view import RecordDetailWindow
from logqbit.gui.browser.window.view import LogBrowserWindow, _validated_log_id
from logqbit.metadata import LogMetadata


def _create_application() -> QApplication:
    app = QApplication.instance()
    assert app is not None
    return app


def scan_catalog(directory: Path) -> list[LogRecord]:
    return LogCatalog(directory).refresh()


class TestBrowserWindow:
    def test_make_note_creates_metadata_only_and_selects_it(
        self,
        sample_logfolder: Path,
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        try:
            assert window._actions.create_note("0.5", "Remember this")

            note_path = sample_logfolder / "0.5"
            assert sorted(path.name for path in note_path.iterdir()) == [
                "metadata.json"
            ]
            assert LogMetadata(note_path / "metadata.json", create=False).title == (
                "Remember this"
            )
            assert window._selected_record is not None
            assert window._selected_record.path == note_path
        finally:
            window.close()

    def test_make_note_does_not_replace_existing_directory(
        self,
        sample_logfolder: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        existing_path = sample_logfolder / "reserved"
        existing_path.mkdir()
        marker = existing_path / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        warnings: list[str] = []
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda _parent, _title, message: warnings.append(message),
        )
        window = LogBrowserWindow(sample_logfolder)
        try:
            assert not window._actions.create_note("reserved", "note")
            assert marker.read_text(encoding="utf-8") == "keep"
            assert warnings
        finally:
            window.close()

    def test_change_id_preserves_selection_and_updates_detail_window(
        self,
        sample_logfolder: Path,
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        record = window._selected_record
        assert record is not None
        detail_window = RecordDetailWindow(record)
        window._detail_windows.append(detail_window)
        try:
            old_path = record.path
            new_path = old_path.parent / "0.5"

            assert window._actions.rename_record_id(record, "0.5")

            assert not old_path.exists()
            assert new_path.is_dir()
            assert window._selected_record is not None
            assert window._selected_record.path == new_path
            assert detail_window.detail_view.current_record is not None
            assert detail_window.detail_view.current_record.path == new_path
        finally:
            detail_window.close()
            window.close()

    @pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a\\b", "CON"])
    def test_log_id_validation_rejects_unsafe_directory_names(self, value: str) -> None:
        with pytest.raises(ValueError):
            _validated_log_id(value)

    def test_browser_window_detail_shortcuts(self, sample_logfolder: Path) -> None:
        app = _create_application()
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

    def test_directory_menu_can_clear_missing_recent_folders(
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
            [sample_logfolder, tmp_path / "other", tmp_path / "missing"]
        )
        (tmp_path / "other").mkdir()
        window._rebuild_directory_menu()

        clear_action = next(
            action
            for action in window._directory_menu.actions()
            if action.text() == "Cleanup"
        )
        assert clear_action.isEnabled()

        clear_action.trigger()

        assert window.settings_manager.load_recent_directories() == [
            sample_logfolder,
            tmp_path / "other",
        ]
        window.close()

    def test_open_other_folder_starts_from_current_parent(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        try:
            requested_paths: list[str] = []
            monkeypatch.setattr(
                "logqbit.gui.browser.window.view.QFileDialog.getExistingDirectory",
                lambda _parent, _title, path: requested_paths.append(path) or "",
            )

            window._open_directory_dialog()

            assert requested_paths == [str(sample_logfolder.parent)]
        finally:
            window.close()

    def test_top_bar_only_offers_custom_about_menu(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        shown: list[tuple[object, str, str]] = []
        monkeypatch.setattr(
            QMessageBox,
            "about",
            lambda parent, title, text: shown.append((parent, title, text)),
        )
        try:
            assert window.directory_label.contextMenuPolicy() == Qt.CustomContextMenu

            window._actions.show_about_dialog()

            assert shown and shown[0][0] is window
            assert shown[0][1] == "About LogQbit"
        finally:
            window.close()

    def test_browser_can_show_starred_records_only(
        self,
        sample_records: list[LogRecord],
    ) -> None:
        app = _create_application()
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

    def test_append_opens_persistent_dialog_for_sole_sid_record(
        self,
        sample_records: list[LogRecord],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = LogBrowserWindow(sample_records[0].path.parent)
        shown: list[tuple[list[LogRecord], LogRecord | None]] = []
        monkeypatch.setattr(
            window._actions,
            "_show_merge_dialog",
            lambda records, target=None: shown.append((records, target)),
        )
        try:
            aggregate = sample_records[0]
            dataframe = aggregate.read_dataframe()
            assert dataframe is not None
            dataframe[LOGFOLDER_SID_COLUMN] = "aggregate"
            dataframe.to_feather(aggregate.data_path)
            aggregate.refresh()
            window._actions.append_into_existing_record([sample_records[1], aggregate])

            assert len(shown) == 1
            assert [record.path for record in shown[0][0]] == [
                sample_records[1].path,
                aggregate.path,
            ]
            assert shown[0][1] is aggregate
        finally:
            window.close()

    def test_merge_dialog_summary_lists_folders_and_write_button_up_front(
        self,
        sample_records: list[LogRecord],
    ) -> None:
        window = LogBrowserWindow(sample_records[0].path.parent)
        dialog = MergeDialog(
            window,
            sample_records[:2],
            sample_records[0].path.parent,
            target=sample_records[0],
        )
        try:
            assert dialog._summary_label.text() == (
                "Appending 1 folder into #0:\n#0: a, b, 1 rows\n#1: x, y, 1 rows"
            )
            assert dialog._write_button.text() == "Write File"
            assert not dialog._write_button.isEnabled()
        finally:
            dialog.close()
            window.close()

    def test_merge_dialog_prepares_and_writes_in_one_window(
        self,
        sample_logfolder: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        refreshes: list[bool] = []
        monkeypatch.setattr(window, "refresh_logs", lambda: refreshes.append(True))

        prepared = SimpleNamespace(
            is_noop=False,
            target=None,
            row_count=12,
            appended_records=2,
            skipped_records=0,
            staging_path=None,
            discard=lambda: None,
        )
        published: list[object] = []
        records = scan_catalog(sample_logfolder)

        def prepare(selected, destination):
            assert selected == records
            assert destination == sample_logfolder
            return prepared

        def publish():
            published.append(prepared)
            return MergeRecordsResult(
                path=sample_logfolder / "3",
                row_count=12,
                appended_records=2,
                skipped_records=0,
                created=True,
            )

        prepared.publish = publish
        monkeypatch.setattr(
            merge_module.PreparedMerge,
            "for_new_folder",
            classmethod(
                lambda _cls, selected, destination: prepare(selected, destination)
            ),
        )

        analysis_loop = QEventLoop()
        analyzed: list[bool] = []
        completed: list[bool] = []
        dialog = MergeDialog(
            window,
            records,
            sample_logfolder,
        )
        dialog.files_written.connect(window.refresh_logs)
        dialog.analysis_finished.connect(
            lambda succeeded: (analyzed.append(succeeded), analysis_loop.quit())
        )
        dialog.merge_finished.connect(completed.append)
        QTimer.singleShot(3000, analysis_loop.quit)
        try:
            assert "#0: x, y, z, 3 rows" in dialog._summary_label.text()
            assert dialog._write_button.text() == "Write File"
            assert not dialog._write_button.isEnabled()

            analysis_loop.exec()

            assert analyzed == [True]
            assert published == []
            assert dialog._write_button.isEnabled()
            assert dialog._status_label.text() == "Ready to merge."

            dialog._write_button.click()

            assert completed == [True]
            assert published == [prepared]
            assert refreshes == [True]
            assert dialog._status_label.text() == "Merge complete."
            assert dialog._detail_label.text() == (
                f"12 rows written to:\n{sample_logfolder / '3'}"
            )
            assert dialog._write_button.isHidden()
            assert dialog._cancel_button.text() == "Close"
            assert dialog._cancel_button.isEnabled()
        finally:
            dialog.close()
            window.close()

    def test_merge_dialog_keeps_validation_failure_in_same_window(
        self,
        sample_logfolder: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        records = scan_catalog(sample_logfolder)
        target = records[0]

        def prepare_append(selected):
            assert selected == records
            raise MergeRecordsError(
                "The selected records need at least two common data columns."
            )

        prepared = SimpleNamespace(
            is_noop=False,
            target=None,
            row_count=3,
            appended_records=1,
            skipped_records=0,
            staging_path=None,
            discard=lambda: None,
        )

        def prepare_new(selected, destination):
            assert selected == records
            assert destination == sample_logfolder
            return prepared

        monkeypatch.setattr(
            merge_module.PreparedMerge,
            "for_append",
            classmethod(lambda _cls, selected: prepare_append(selected)),
        )
        monkeypatch.setattr(
            merge_module.PreparedMerge,
            "for_new_folder",
            classmethod(
                lambda _cls, selected, destination: prepare_new(selected, destination)
            ),
        )

        loop = QEventLoop()
        analyzed: list[bool] = []
        dialog = MergeDialog(
            window,
            records,
            sample_logfolder,
            target=target,
        )
        dialog.analysis_finished.connect(
            lambda succeeded: (analyzed.append(succeeded), loop.quit())
        )
        QTimer.singleShot(3000, loop.quit)
        try:
            loop.exec()

            assert analyzed == [False]
            assert dialog._status_label.text() == "Fail to merge."
            assert dialog._detail_label.text() == (
                "The selected records need at least two common data columns."
            )
            assert not dialog._try_new_button.isHidden()
            assert dialog._cancel_button.text() == "Close"

            dialog._try_new_button.click()

            assert dialog._target is None
            assert "into a new folder" in dialog._summary_label.text()
            assert dialog._status_label.text() == "Ready to merge."
            assert dialog._write_button.isEnabled()
        finally:
            dialog.close()
            window.close()

    def test_merge_dialog_displays_noop_without_enabling_write_back(
        self,
        sample_logfolder: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = LogBrowserWindow(sample_logfolder)
        records = scan_catalog(sample_logfolder)
        target = records[0]
        prepared = SimpleNamespace(
            is_noop=True,
            target=target,
            row_count=20,
            appended_records=0,
            skipped_records=2,
            staging_path=None,
            discard=lambda: None,
        )
        monkeypatch.setattr(
            merge_module.PreparedMerge,
            "for_append",
            classmethod(lambda _cls, _records: prepared),
        )
        loop = QEventLoop()
        analyzed: list[bool] = []
        dialog = MergeDialog(
            window,
            records,
            sample_logfolder,
            target=target,
        )
        dialog.analysis_finished.connect(
            lambda succeeded: (analyzed.append(succeeded), loop.quit())
        )
        QTimer.singleShot(3000, loop.quit)
        try:
            assert dialog._write_button.text() == "Write File"
            assert not dialog._write_button.isEnabled()

            loop.exec()

            assert analyzed == [True]
            assert not dialog._write_button.isEnabled()
            assert dialog._status_label.text() == "No merge needed."
            assert (
                "already contains all selected sources" in dialog._detail_label.text()
            )
            assert dialog._cancel_button.text() == "Close"
        finally:
            dialog.close()
            window.close()

    def test_open_explorer_shortcut_is_scoped_to_log_table(
        self,
        sample_logfolder: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = _create_application()
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

    def test_manual_refresh_updates_catalog_summary_and_detail(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _create_application()
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
            assert record.row_count == 10
            assert window._selected_record is not None
            assert window._selected_record is record
            assert window._selected_record.row_count == 10
        finally:
            window.close()

    def test_detail_data_refresh_updates_log_list_rows(
        self, sample_logfolder: Path
    ) -> None:
        app = _create_application()
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
        app = _create_application()
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
        app = _create_application()
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
