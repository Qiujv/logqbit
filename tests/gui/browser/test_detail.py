from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QPushButton,
)

from logqbit import catalog as catalog_module
from logqbit.catalog import LogCatalog, LogRecord
from logqbit.gui.browser.detail.data import DataViewManager, PandasTableModel
from logqbit.gui.browser.detail.files import (
    ImageTab,
    ZoomableImageView,
    _format_file_size,
)
from logqbit.gui.browser.detail.view import (
    TAB_PLOT,
    RecordDetailView,
    RecordDetailWindow,
)
from logqbit.logfolder import LogFolder


def _create_application() -> QApplication:
    app = QApplication.instance()
    assert app is not None
    return app


def scan_catalog(directory: Path) -> list[LogRecord]:
    return LogCatalog(directory).refresh()


class TestRecordDetailWidgets:
    def test_image_tab_copies_image_file(self, sample_logfolder: Path) -> None:
        app = _create_application()
        record = scan_catalog(sample_logfolder)[0]
        image_path = record.path / "copy-test.png"
        image_path.write_bytes(b"not a renderable image")

        view = RecordDetailView()
        view.load_record(record)
        tab_names = [view.tab_widget.tabText(i) for i in range(view.tab_widget.count())]
        image_tab = view.tab_widget.widget(tab_names.index(image_path.name))
        assert image_tab is not None
        copy_button = image_tab.findChild(QPushButton)
        assert copy_button is not None
        assert copy_button.isEnabled()
        assert copy_button.text() == "copy"
        status_label = next(
            label
            for label in image_tab.findChildren(QLabel)
            if label.text().startswith("File size:")
        )
        assert status_label.text() == (
            f"File size: {image_path.stat().st_size} B. Double-click to zoom to fit."
        )
        copy_shortcut = image_tab.findChild(QShortcut)
        assert copy_shortcut is not None
        assert copy_shortcut.key() == QKeySequence.Copy
        assert copy_shortcut.context() == Qt.WidgetWithChildrenShortcut

        app.clipboard().clear()
        copy_shortcut.activated.emit()
        mime_data = app.clipboard().mimeData()
        assert mime_data.hasUrls()
        assert [Path(url.toLocalFile()) for url in mime_data.urls()] == [
            image_path.resolve()
        ]
        assert not mime_data.hasImage()

        app.clipboard().clear()
        copy_button.click()
        assert [
            Path(url.toLocalFile()) for url in app.clipboard().mimeData().urls()
        ] == [image_path.resolve()]

    def test_image_file_size_formatting(self) -> None:
        assert _format_file_size(0) == "0 B"
        assert _format_file_size(1023) == "1023 B"
        assert _format_file_size(1024) == "1.0 KiB"
        assert _format_file_size(1024**2) == "1.0 MiB"

    def test_image_tab_supports_zooming_and_panning(
        self, sample_logfolder: Path
    ) -> None:
        app = _create_application()
        record = scan_catalog(sample_logfolder)[0]
        image_path = record.path / "zoom-test.png"
        image = QPixmap(8, 8)
        image.fill(QColor("blue"))
        assert image.save(str(image_path))
        view = RecordDetailView()
        view.resize(600, 400)
        view.load_record(record)
        view.set_current_tab(TAB_PLOT + 1)
        view.show()
        app.processEvents()
        image_tab = view.tab_widget.widget(TAB_PLOT + 1)
        assert image_tab is not None
        image_view = image_tab.findChild(ZoomableImageView)
        assert image_view is not None
        assert not image_view.getPlotItem().getAxis("left").isVisible()
        assert not image_view.getPlotItem().getAxis("bottom").isVisible()
        assert image_view._view_box.state["yInverted"]
        image_view.zoom_fit()
        initial_range = image_view._view_box.viewRange()

        image_view._view_box.scaleBy((0.5, 0.5))
        image_view._view_box.zoom_fit_requested.emit()

        assert image_view._view_box.viewRange()[0] == pytest.approx(initial_range[0])
        assert image_view._view_box.viewRange()[1] == pytest.approx(initial_range[1])
        view.close()

    def test_rename_image_file_refreshes_tabs(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        image_path = record.path / "before.png"
        image_path.write_bytes(b"image")
        view = RecordDetailView()
        view.load_record(record)
        dialog_arguments = {}

        def get_new_name(*args, **kwargs):
            dialog_arguments.update(kwargs)
            return "after", True

        monkeypatch.setattr(
            "logqbit.gui.browser.detail.files.QInputDialog.getText",
            get_new_name,
        )

        image_tab = view.tab_widget.widget(
            [view.tab_widget.tabText(i) for i in range(view.tab_widget.count())].index(
                image_path.name
            )
        )
        assert isinstance(image_tab, ImageTab)
        image_tab._rename_file()

        assert not image_path.exists()
        assert (record.path / "after.png").exists()
        assert dialog_arguments["text"] == "before"
        tab_names = [view.tab_widget.tabText(i) for i in range(view.tab_widget.count())]
        assert "after.png" in tab_names
        assert "before.png" not in tab_names

    def test_move_image_file_to_trash_refreshes_tabs(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        image_path = record.path / "trash.png"
        image_path.write_bytes(b"image")
        view = RecordDetailView()
        view.load_record(record)
        moved_paths: list[str] = []
        monkeypatch.setattr(
            "logqbit.gui.browser.detail.files.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.Yes,
        )

        def move_to_trash(path: str) -> None:
            moved_paths.append(path)
            Path(path).unlink()

        monkeypatch.setattr(
            "logqbit.gui.browser.detail.files.send2trash", move_to_trash
        )

        image_tab = view.tab_widget.widget(
            [view.tab_widget.tabText(i) for i in range(view.tab_widget.count())].index(
                image_path.name
            )
        )
        assert isinstance(image_tab, ImageTab)
        image_tab._move_to_trash()

        assert moved_paths == [str(image_path)]
        assert not image_path.exists()
        tab_names = [view.tab_widget.tabText(i) for i in range(view.tab_widget.count())]
        assert "trash.png" not in tab_names

    def test_plot_tab_copies_current_view(self, sample_logfolder: Path) -> None:
        app = _create_application()
        record = scan_catalog(sample_logfolder)[0]
        view = RecordDetailView()
        view.resize(600, 400)
        view.load_record(record)
        view.set_current_tab(TAB_PLOT)
        assert view.plot_manager.copy_plot_button.text() == "copy"
        view.show()
        app.processEvents()
        try:
            app.clipboard().clear()
            view.plot_manager.copy_plot_button.click()
            copied = app.clipboard().image()
            assert not copied.isNull()
            first_pixel = copied.pixel(0, 0)
            assert any(
                copied.pixel(x, y) != first_pixel
                for y in range(copied.height())
                for x in range(copied.width())
            )
        finally:
            app.clipboard().clear()
            view.close()

    def test_detail_header_separates_id_and_selectable_wrapped_path(
        self, sample_logfolder: Path
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        view = RecordDetailView()
        view.load_record(record)

        assert view.detail_id_label.text() == f"#{record.log_id}"
        assert not (
            view.detail_id_label.textInteractionFlags() & Qt.TextSelectableByMouse
        )
        assert view.detail_label.text() == str(record.path)
        assert view.detail_label.textInteractionFlags() & Qt.TextSelectableByMouse
        assert view.detail_label.contextMenuPolicy() == Qt.NoContextMenu
        assert view.detail_label.wordWrap()

    def test_detail_refresh_keeps_dataframe_when_only_other_files_change(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        view = RecordDetailView()
        view.load_record(record)
        dataframe = view._data_cache.dataframe
        table_model = view.data_view_manager.data_table.model()
        assert dataframe is not None

        (record.path / "notes.txt").write_text("updated", encoding="utf-8")
        read_count = 0
        original_read_feather = catalog_module.pd.read_feather

        def count_read_feather(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read_feather(*args, **kwargs)

        monkeypatch.setattr(catalog_module.pd, "read_feather", count_read_feather)

        view.refresh_current_record()

        assert view._data_cache.dataframe is dataframe
        assert view.data_view_manager.data_table.model() is table_model
        assert read_count == 0

    def test_noop_refresh_preserves_plot_range_and_fit_selection(
        self,
        sample_logfolder: Path,
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        record.meta.update(plot_axes=["x"], plot_fields=["y"])
        view = RecordDetailView()
        view.load_record(record)
        view.set_current_tab(TAB_PLOT)
        plot_item = view.plot_manager.plot_widget.getPlotItem()
        plot_item.getViewBox().setXRange(1.0, 2.0, padding=0)
        view.plot_manager.exponential_fit_button.click()
        expected_range = plot_item.getViewBox().viewRange()[0]

        (record.path / "notes.txt").write_text("updated", encoding="utf-8")
        view.refresh_current_record()

        assert plot_item.getViewBox().viewRange()[0] == pytest.approx(expected_range)
        assert view.plot_manager.exponential_fit_button.isChecked()

    def test_sync_detail_watcher_keeps_unchanged_paths(
        self,
        sample_logfolder: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        view = RecordDetailView()
        view.load_record(record)
        clear_calls = 0

        def count_clear() -> None:
            nonlocal clear_calls
            clear_calls += 1

        monkeypatch.setattr(view, "_clear_detail_watcher", count_clear)

        view._sync_detail_watcher()

        assert clear_calls == 0

    def test_data_refresh_does_not_rebuild_tag_bar(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        view = RecordDetailView()
        view.load_record(record)
        view.set_current_tab(TAB_PLOT)
        set_columns_calls: list[tuple] = []
        monkeypatch.setattr(
            view.plot_manager.tag_bar,
            "set_columns",
            lambda *args: set_columns_calls.append(args),
        )

        pd.DataFrame({"x": range(5), "y": range(5), "z": range(5)}).to_feather(
            record.data_path
        )
        view.refresh_current_record()

        assert set_columns_calls == []

        record.meta.update(plot_axes=["y"], plot_fields=["z"])
        view.refresh_current_record()

        assert len(set_columns_calls) == 1

    def test_data_table_unique_values_use_full_column(self) -> None:
        manager = DataViewManager()
        frame = pd.DataFrame({"value": [1, 2, 1, None, 2]})
        manager.data_table.setModel(
            PandasTableModel(frame, manager.data_table, preview_limit=1)
        )

        values = manager._unique_values_for_column(0)

        assert values is not None
        assert values.iloc[:2].tolist() == [1.0, 2.0]
        assert pd.isna(values.iloc[2])
        assert len(values) == 3
        assert manager.data_table.contextMenuPolicy() == Qt.CustomContextMenu

    def test_files_corner_menu_lists_and_opens_all_record_files(
        self, sample_logfolder: Path
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        extra_file = record.path / "notes.txt"
        extra_file.write_text("hello", encoding="utf-8")
        opened_paths: list[Path] = []

        view = RecordDetailView(file_open_callback=opened_paths.append)
        view.load_record(record)
        view._rebuild_files_menu()

        assert view.tab_widget.cornerWidget(Qt.TopRightCorner) is view.files_button
        action_names = [action.text() for action in view.files_menu.actions()]
        assert "data.feather" in action_names
        assert "metadata.json" in action_names
        assert "notes.txt" in action_names
        assert action_names[-1] == "Show in Explorer"

        next(
            action
            for action in view.files_menu.actions()
            if action.text() == "notes.txt"
        ).trigger()
        assert opened_paths == [extra_file]

    def test_show_in_explorer_uses_current_record_folder(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        opened_paths: list[Path] = []
        monkeypatch.setattr(
            "logqbit.gui.browser.detail.view.open_in_file_manager",
            lambda path, parent=None: opened_paths.append(path),
        )
        view = RecordDetailView()
        view.load_record(record)
        view._rebuild_files_menu()

        view.files_menu.actions()[-1].trigger()
        assert opened_paths == [record.path]

    def test_switching_to_record_without_images_removes_image_tabs(
        self, sample_logfolder: Path
    ) -> None:
        record_with_image = scan_catalog(sample_logfolder)[0]
        image_path = record_with_image.path / "old-image.png"
        image = QPixmap(8, 8)
        image.fill(QColor("blue"))
        assert image.save(str(image_path))

        empty_path = sample_logfolder / "999"
        LogFolder(empty_path)
        record_without_image = next(
            record
            for record in scan_catalog(sample_logfolder)
            if record.path == empty_path
        )

        view = RecordDetailView()
        view.load_record(record_with_image)
        assert view.tab_widget.count() == TAB_PLOT + 2
        view.set_current_tab(TAB_PLOT + 1)

        view.load_record(record_without_image)
        assert view.tab_widget.count() == TAB_PLOT + 1
        assert view.current_tab_index() == TAB_PLOT

    def test_switching_records_preserves_image_tab_position(
        self,
        sample_logfolder: Path,
    ) -> None:
        first_record = scan_catalog(sample_logfolder)[0]
        second_path = sample_logfolder / "999"
        LogFolder(second_path)

        for record_path, color in (
            (first_record.path, "blue"),
            (second_path, "green"),
        ):
            for name in ("first.png", "second.png"):
                image = QPixmap(8, 8)
                image.fill(QColor(color))
                assert image.save(str(record_path / name))

        second_record = next(
            record
            for record in scan_catalog(sample_logfolder)
            if record.path == second_path
        )
        view = RecordDetailView()
        view.load_record(first_record)
        view.set_current_tab(TAB_PLOT + 2)

        view.load_record(second_record)

        assert view.current_tab_index() == TAB_PLOT + 2
        assert view.tab_widget.tabText(view.current_tab_index()) == "second.png"

    def test_detail_window_watch_toggle_controls_watcher(
        self, sample_logfolder: Path
    ) -> None:
        app = _create_application()
        record = scan_catalog(sample_logfolder)[0]
        window = RecordDetailWindow(record)
        window.show()
        app.processEvents()
        try:
            assert window.detail_view.watch_enabled
            assert window.detail_view._detail_watcher.directories()

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

    def test_detail_views_manage_watchers_independently(
        self, sample_logfolder: Path
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        first_view = RecordDetailView()
        second_view = RecordDetailView()
        try:
            first_view.load_record(record)
            second_view.load_record(record)

            assert str(record.path) in set(first_view._detail_watcher.directories())
            assert str(record.path) in set(second_view._detail_watcher.directories())

            first_view.set_watch_enabled(False)

            assert not first_view._detail_watcher.directories()
            assert str(record.path) in set(second_view._detail_watcher.directories())
        finally:
            first_view.close()
            second_view.close()

    def test_detail_views_own_independent_dataframe_caches(
        self, sample_logfolder: Path
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        first_view = RecordDetailView()
        second_view = RecordDetailView()
        try:
            first_view.load_record(record)
            second_view.load_record(record)

            first_frame = first_view._data_cache.dataframe
            second_frame = second_view._data_cache.dataframe
            assert first_frame is not None
            assert second_frame is not None
            assert first_frame is not second_frame

            first_view.refresh_current_record(force=True)

            assert first_view._data_cache.dataframe is not first_frame
            assert second_view._data_cache.dataframe is second_frame
        finally:
            first_view.close()
            second_view.close()

    def test_detail_window_has_file_watcher_and_tab_shortcuts(
        self, sample_logfolder: Path
    ) -> None:
        app = _create_application()
        record = scan_catalog(sample_logfolder)[0]
        extra_file = record.path / "notes.txt"
        extra_file.write_text("hello", encoding="utf-8")

        window = RecordDetailWindow(record)
        window.show()
        app.processEvents()
        try:
            assert str(record.path) in set(
                window.detail_view._detail_watcher.directories()
            )
            watched_files = set(window.detail_view._detail_watcher.files())
            assert str(record.meta_path) in watched_files
            assert str(extra_file) not in watched_files

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
