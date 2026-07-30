from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontDatabase, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton

from logqbit import catalog as catalog_module
from logqbit.catalog import (
    LogCatalog,
    LogRecord,
    export_records,
)
from logqbit.gui.browser import (
    COL_ID,
    COL_PLOT_AXES,
    COL_ROWS,
    COL_TITLE,
    LogBrowserWindow,
    LogListTableModel,
    SettingsManager,
    ensure_application,
)
from logqbit.gui.detail_view import (
    TAB_PLOT,
    PandasTableModel,
    RecordDetailView,
    RecordDetailWindow,
    record_watch_paths,
)
from logqbit.logfolder import LogFolder


def scan_catalog(directory: Path) -> list[LogRecord]:
    return LogCatalog(directory).refresh()


@pytest.fixture
def sample_logfolder(tmp_path: Path) -> Path:
    """Create a sample log folder with data and return its parent directory."""
    lf = LogFolder.new(tmp_path, title="test_log")
    lf.add_row(x=1.0, y=2.0, z=3.0)
    lf.add_row(x=1.5, y=2.5, z=3.5)
    lf.add_row(x=2.0, y=3.0, z=4.0)
    lf.flush()
    lf.meta.star = 1
    lf.meta.plot_axes = ["x", "y"]
    # Return the parent directory path, not the LogFolder object
    # This ensures the LogFolder is properly closed and data is flushed
    return tmp_path
    

@pytest.fixture
def sample_records(tmp_path: Path) -> list[LogRecord]:
    """Create multiple sample log records."""
    records = []
    
    # Record 0: basic log
    lf0 = LogFolder.new(tmp_path, title="log_zero")
    lf0.add_row(a=1, b=2)
    lf0.flush()
    
    # Record 1: starred log
    lf1 = LogFolder.new(tmp_path, title="log_one")
    lf1.add_row(x=10, y=20)
    lf1.flush()
    lf1.meta.star = 2
    
    # Record 2: trashed log
    lf2 = LogFolder.new(tmp_path, title="log_two")
    lf2.add_row(p=100, q=200)
    lf2.flush()
    lf2.meta.trash = True
    
    # Scan directory to get records
    records = scan_catalog(tmp_path)
    return records


class TestLogRecord:
    """Tests for LogRecord class."""
    
    def test_scan_catalog_finds_logs(self, tmp_path: Path) -> None:
        """Test scanning a directory for log records."""
        # Create multiple log folders
        LogFolder.new(tmp_path, title="log1").flush()
        LogFolder.new(tmp_path, title="log2").flush()
        
        records = scan_catalog(tmp_path)
        
        assert len(records) == 2
        assert all(isinstance(r, LogRecord) for r in records)
        assert {r.log_id for r in records} == {0, 1}
    
    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning an empty directory."""
        records = scan_catalog(tmp_path)
        assert records == []
    
    def test_scan_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test scanning a directory that doesn't exist."""
        records = scan_catalog(tmp_path / "nonexistent")
        assert records == []
    
    def test_entry_reads_dataframe(self, sample_logfolder: Path) -> None:
        """Test loading dataframe through the passive entry."""
        records = scan_catalog(sample_logfolder)
        assert len(records) == 1
        
        record = records[0]
        df = record.read_dataframe()
        
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["x", "y", "z"]
        assert record.row_count == 3
        assert record.columns == ("x", "y", "z")
    
    def test_read_yaml_text(self, sample_logfolder: Path) -> None:
        """Test reading YAML text from a log record."""
        records = scan_catalog(sample_logfolder)
        record = records[0]
        
        yaml_text = record.read_const_text()
        
        assert isinstance(yaml_text, str)
        assert len(yaml_text) > 0
    
    def test_read_yaml_missing_file(self, tmp_path: Path) -> None:
        """Test reading YAML when file doesn't exist."""
        lf = LogFolder.new(tmp_path)
        # Don't create yaml file
        lf.df_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [1]}).to_feather(lf.df_path)
        
        records = scan_catalog(tmp_path)
        record = records[0]
        
        yaml_text = record.read_const_text()
        assert "const.yaml not found" in yaml_text
    
    def test_list_image_files(self, tmp_path: Path) -> None:
        """Test listing image files in a log folder."""
        lf = LogFolder.new(tmp_path)
        lf.flush()
        
        # Create some image files
        (lf.path / "plot.png").touch()
        (lf.path / "result.jpg").touch()
        (lf.path / "data.txt").touch()  # Not an image
        
        records = scan_catalog(tmp_path)
        record = records[0]
        
        images = record.list_image_files()
        
        assert len(images) == 2
        assert all(img.suffix.lower() in {".png", ".jpg"} for img in images)

    def test_list_other_files(self, sample_logfolder: Path) -> None:
        """Test listing non-standard files in a log folder."""
        records = scan_catalog(sample_logfolder)
        record = records[0]

        extra_text = record.path / "notes.txt"
        extra_binary = record.path / "snapshot.bin"
        ignored_image = record.path / "plot.webp"
        extra_text.write_text("hello", encoding="utf-8")
        extra_binary.write_bytes(b"123")
        ignored_image.touch()

        other_files = record.list_other_files()

        assert [path.name for path in other_files] == ["notes.txt", "snapshot.bin"]

    def test_record_watch_paths_include_extra_files(self, sample_logfolder: Path) -> None:
        """Test watch path helper tracks record files beyond the standard trio."""
        records = scan_catalog(sample_logfolder)
        record = records[0]

        extra_file = record.path / "notes.txt"
        extra_file.write_text("watch me", encoding="utf-8")

        watch_paths = set(record_watch_paths(record))

        assert str(record.path) in watch_paths
        assert str(record.meta_path) in watch_paths
        assert str(extra_file) in watch_paths

    def test_export_records_copies_selected_logs_in_id_order(self, tmp_path: Path) -> None:
        source_parent = tmp_path / "source"
        source_parent.mkdir()

        low = LogFolder.new(source_parent, title="low")
        low.add_row(x=1, y=2)
        low.flush()
        (low.path / "note.txt").write_text("low-note", encoding="utf-8")

        high = LogFolder.new(source_parent, title="high")
        high.add_row(x=10, y=20)
        high.flush()
        (high.path / "snapshot.bin").write_bytes(b"abc")
        (high.path / "import_from").write_text("preserve-me", encoding="utf-8")

        records = scan_catalog(source_parent)
        record_by_title = {record.title: record for record in records}

        destination_parent = tmp_path / "exported"
        destination_parent.mkdir()
        (destination_parent / "0").mkdir()

        exported_paths = export_records(
            [
                record_by_title["high"],
                record_by_title["low"],
            ],
            destination_parent,
        )

        assert [path.name for path in exported_paths] == ["1", "2"]
        assert (exported_paths[0] / "note.txt").read_text(encoding="utf-8") == "low-note"
        assert (exported_paths[1] / "snapshot.bin").read_bytes() == b"abc"
        assert (exported_paths[0] / "import_from").read_text(encoding="utf-8") == str(record_by_title["low"].path)
        assert (exported_paths[1] / "import_from").read_text(encoding="utf-8") == "preserve-me"

        exported_records = scan_catalog(destination_parent)
        exported_titles = {record.log_id: record.title for record in exported_records}
        assert exported_titles[1] == "low"
        assert exported_titles[2] == "high"


class TestLogListTableModel:
    """Tests for LogListTableModel class."""
    
    def test_initial_state(self) -> None:
        """Test initial state of the table model."""
        model = LogListTableModel()
        
        assert model.rowCount() == 0
        assert model.columnCount() == 6
    
    def test_set_records(self, sample_records: list[LogRecord]) -> None:
        """Test setting records in the model."""
        model = LogListTableModel()
        model.set_records(sample_records)
        
        assert model.rowCount() == len(sample_records)
    
    def test_get_record(self, sample_records: list[LogRecord]) -> None:
        """Test getting a record by row index."""
        model = LogListTableModel()
        model.set_records(sample_records)
        
        record = model.get_record(0)
        assert record is not None
        assert record.log_id == sample_records[0].log_id
        
        # Test out of bounds
        assert model.get_record(-1) is None
        assert model.get_record(999) is None
    
    def test_data_display_id(self, sample_records: list[LogRecord]) -> None:
        """Test displaying ID column."""
        model = LogListTableModel()
        model.set_records(sample_records)
        
        index = model.index(0, COL_ID)
        data = model.data(index, Qt.DisplayRole)
        
        assert data == sample_records[0].log_id
    
    def test_data_display_title(self, sample_records: list[LogRecord]) -> None:
        """Test displaying title with stars and trash."""
        model = LogListTableModel()
        model.set_records(sample_records)
        
        # Regular title
        index0 = model.index(0, COL_TITLE)
        data0 = model.data(index0, Qt.DisplayRole)
        assert "log_zero" in data0
        
        # Starred title
        index1 = model.index(1, COL_TITLE)
        data1 = model.data(index1, Qt.DisplayRole)
        assert "⭐⭐" in data1
        assert "log_one" in data1
        
        # Trashed title
        index2 = model.index(2, COL_TITLE)
        data2 = model.data(index2, Qt.DisplayRole)
        assert "🗑️" in data2
        assert "log_two" in data2
    
    def test_data_display_rows(self, sample_records: list[LogRecord]) -> None:
        """Test displaying row count."""
        model = LogListTableModel()
        model.set_records(sample_records)
        
        index = model.index(0, COL_ROWS)
        data = model.data(index, Qt.DisplayRole)
        
        assert isinstance(data, int)
        assert data >= 0
    
    def test_data_display_plot_axes(self, sample_logfolder: Path) -> None:
        """Test displaying plot axes with abbreviations."""
        records = scan_catalog(sample_logfolder)
        model = LogListTableModel()
        model.set_records(records)
        
        index = model.index(0, COL_PLOT_AXES)
        data = model.data(index, Qt.DisplayRole)
        
        # Should show first 3 characters of each axis
        assert data == "2, x, y"  # "x" + "y"
    
    def test_data_tooltip_plot_axes(self, sample_logfolder: Path) -> None:
        """Test tooltip showing full plot axes names."""
        records = scan_catalog(sample_logfolder)
        model = LogListTableModel()
        model.set_records(records)
        
        index = model.index(0, COL_PLOT_AXES)
        tooltip = model.data(index, Qt.ToolTipRole)
        
        assert tooltip == "x, y"
    
    def test_data_font_role_starred(self, sample_records: list[LogRecord]) -> None:
        """Test font styling for starred items."""
        model = LogListTableModel()
        model.set_records(sample_records)
        
        # Starred item should be bold
        index1 = model.index(1, COL_TITLE)
        font = model.data(index1, Qt.FontRole)
        assert font is not None
        assert font.bold()
    
    def test_data_font_role_trashed(self, sample_records: list[LogRecord]) -> None:
        """Test font styling for trashed items."""
        model = LogListTableModel()
        model.set_records(sample_records)
        
        # Trashed item should be strikeout
        index2 = model.index(2, COL_TITLE)
        font = model.data(index2, Qt.FontRole)
        assert font is not None
        assert font.strikeOut()
    
    def test_notify_record_changed(self, sample_records: list[LogRecord]) -> None:
        """Test notifying views after a record has been refreshed."""
        model = LogListTableModel()
        model.set_records(sample_records)

        record = sample_records[0]
        record.meta.update(title="updated_title")

        model.notify_record_changed(record)
        
        index = model.index(0, COL_TITLE)
        data = model.data(index, Qt.DisplayRole)
        assert "updated_title" in data
    
    def test_header_data(self) -> None:
        """Test header data."""
        model = LogListTableModel()
        
        headers = []
        for col in range(6):
            header = model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
            headers.append(header)
        
        expected = ["ID", "Title", "Rows", "Axes", "Create Time", "Create Machine"]
        assert headers == expected


class TestSettingsManager:
    def test_update_recent_directories_can_be_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LOGQBIT_BROWSER_DISABLE_RECENT_DIRS", "1")
        manager = SettingsManager()
        original = list(manager.load_recent_directories())

        manager.update_recent_directories(tmp_path)

        assert manager.load_recent_directories() == original


class TestPandasTableModel:
    """Tests for PandasTableModel class."""
    
    def test_basic_dataframe_display(self) -> None:
        """Test displaying a basic dataframe."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        model = PandasTableModel(df)
        
        assert model.rowCount() == 3
        assert model.columnCount() == 2
        
        # Test data access
        index = model.index(0, 0)
        data = model.data(index, Qt.DisplayRole)
        assert data == "1"
    
    def test_preview_limit(self) -> None:
        """Test preview limit functionality."""
        df = pd.DataFrame({"x": range(100)})
        model = PandasTableModel(df, preview_limit=10)
        
        assert model.rowCount() == 10
        assert model.get_total_rows() == 100
    
    def test_set_preview_limit(self) -> None:
        """Test changing preview limit."""
        df = pd.DataFrame({"x": range(100)})
        model = PandasTableModel(df, preview_limit=10)
        
        assert model.rowCount() == 10
        
        model.set_preview_limit(50)
        assert model.rowCount() == 50
        
        model.set_preview_limit(None)
        assert model.rowCount() == 100
    
    def test_highlight_columns(self) -> None:
        """Test highlighting specific columns."""
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        model = PandasTableModel(df, highlight_columns=["x", "z"])
        
        # Highlighted column should have bold font
        index_x = model.index(0, 0)
        font = model.data(index_x, Qt.FontRole)
        assert font is not None
        assert font.bold()
        
        # Non-highlighted column should have no special font
        index_y = model.index(0, 1)
        font_y = model.data(index_y, Qt.FontRole)
        assert font_y is None
    
    def test_numeric_formatting(self) -> None:
        """Test numeric value formatting."""
        df = pd.DataFrame({"value": [1.234567890, 0.000123, 1234567.89]})
        model = PandasTableModel(df)
        
        # Should format to 6 significant figures
        data0 = model.data(model.index(0, 0), Qt.DisplayRole)
        assert "1.23457" in data0
    
    def test_nan_display(self) -> None:
        """Test displaying NaN values."""
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
        model = PandasTableModel(df)
        
        # NaN should display as empty string
        index = model.index(1, 0)
        data = model.data(index, Qt.DisplayRole)
        assert data == ""
    
    def test_header_data(self) -> None:
        """Test column headers."""
        df = pd.DataFrame({"alpha": [1], "beta": [2]})
        model = PandasTableModel(df)
        
        header0 = model.headerData(0, Qt.Horizontal, Qt.DisplayRole)
        header1 = model.headerData(1, Qt.Horizontal, Qt.DisplayRole)
        
        assert header0 == "alpha"
        assert header1 == "beta"
    
    def test_header_font_for_highlighted_columns(self) -> None:
        """Test that highlighted columns have bold headers."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        model = PandasTableModel(df, highlight_columns=["x"])
        
        # Highlighted column header should be bold
        font = model.headerData(0, Qt.Horizontal, Qt.FontRole)
        assert font is not None
        assert font.bold()
        
        # Non-highlighted column header should have no special font
        font_y = model.headerData(1, Qt.Horizontal, Qt.FontRole)
        assert font_y is None


class TestRecordDetailWidgets:
    def test_image_tab_copies_original_image(self, sample_logfolder: Path) -> None:
        app = ensure_application()
        record = scan_catalog(sample_logfolder)[0]
        image_path = record.path / "copy-test.png"
        source = QPixmap(13, 7)
        source.fill(QColor("red"))
        assert source.save(str(image_path))

        view = RecordDetailView()
        view.load_record(record)
        tab_names = [view.tab_widget.tabText(i) for i in range(view.tab_widget.count())]
        image_tab = view.tab_widget.widget(tab_names.index(image_path.name))
        assert image_tab is not None
        copy_button = image_tab.findChild(QPushButton)
        assert copy_button is not None

        app.clipboard().clear()
        copy_button.click()
        copied = app.clipboard().pixmap()
        assert copied.size() == source.size()

    def test_plot_tab_copies_current_view(self, sample_logfolder: Path) -> None:
        app = ensure_application()
        record = scan_catalog(sample_logfolder)[0]
        view = RecordDetailView()
        view.resize(600, 400)
        view.load_record(record)
        view.set_current_tab(TAB_PLOT)
        view.show()
        app.processEvents()
        try:
            app.clipboard().clear()
            view.plot_manager.copy_plot_button.click()
            copied = app.clipboard().pixmap()
            assert not copied.isNull()
            image = copied.toImage()
            first_pixel = image.pixel(0, 0)
            assert any(
                image.pixel(x, y) != first_pixel
                for y in range(image.height())
                for x in range(image.width())
            )
        finally:
            view.close()

    def test_detail_header_separates_id_and_selectable_wrapped_path(
        self, sample_logfolder: Path
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        view = RecordDetailView()
        view.load_record(record)

        assert view.detail_id_label.text() == f"#{record.log_id}"
        assert not (view.detail_id_label.textInteractionFlags() & Qt.TextSelectableByMouse)
        assert view.detail_label.text() == str(record.path)
        assert view.detail_label.textInteractionFlags() & Qt.TextSelectableByMouse
        assert view.detail_label.wordWrap()

    def test_detail_refresh_keeps_dataframe_when_only_other_files_change(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        view = RecordDetailView()
        view.load_record(record)
        dataframe = view._data_cache.dataframe
        assert dataframe is not None

        (record.path / "notes.txt").write_text("updated", encoding="utf-8")
        read_count = 0
        original_read_feather = catalog_module.pd.read_feather

        def count_read_feather(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read_feather(*args, **kwargs)

        monkeypatch.setattr(
            catalog_module.pd, "read_feather", count_read_feather
        )

        view.refresh_current_record()

        assert view._data_cache.dataframe is dataframe
        assert read_count == 0

    def test_const_view_uses_system_fixed_font(self) -> None:
        view = RecordDetailView()
        expected = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)

        assert view.yaml_view.font().family() == expected.family()

    def test_data_table_has_no_custom_context_menu(self) -> None:
        view = RecordDetailView()

        assert view.data_view_manager.data_table.contextMenuPolicy() == Qt.DefaultContextMenu

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
            action for action in view.files_menu.actions() if action.text() == "notes.txt"
        ).trigger()
        assert opened_paths == [extra_file]

    def test_show_in_explorer_uses_current_record_folder(
        self, sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        opened_paths: list[Path] = []
        monkeypatch.setattr(
            "logqbit.gui.detail_view._open_path_in_explorer",
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
        LogFolder(empty_path).close()
        record_without_image = next(
            record
            for record in scan_catalog(sample_logfolder)
            if record.path == empty_path
        )

        view = RecordDetailView()
        view.load_record(record_with_image)
        assert view.tab_widget.count() == TAB_PLOT + 2

        view.load_record(record_without_image)
        assert view.tab_widget.count() == TAB_PLOT + 1

    def test_detail_window_watch_toggle_controls_watcher(
        self, sample_logfolder: Path
    ) -> None:
        app = ensure_application()
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

            assert str(record.path) in set(
                first_view._detail_watcher.directories()
            )
            assert str(record.path) in set(
                second_view._detail_watcher.directories()
            )

            first_view.set_watch_enabled(False)

            assert not first_view._detail_watcher.directories()
            assert str(record.path) in set(
                second_view._detail_watcher.directories()
            )
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
        app = ensure_application()
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
            assert str(extra_file) in watched_files

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

            monkeypatch.setattr(
                catalog_module.pd, "read_feather", count_read_feather
            )

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

            monkeypatch.setattr(
                catalog_module.pd, "read_feather", count_read_feather
            )

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
            pd.DataFrame({"x": range(10), "y": range(10)}).to_feather(
                record.data_path
            )

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
            monkeypatch.setattr(
                catalog_module.pd, "read_feather", count_read_feather
            )

            window._on_refresh_clicked()

            assert inspect_count == 1
            assert read_count == 1
            assert record.row_count == 3
            assert window._selected_record is not None
            assert window._selected_record.row_count == 10
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
