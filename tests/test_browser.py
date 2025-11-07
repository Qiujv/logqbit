from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from logqbit.browser import (
    COL_CREATE_MACHINE,
    COL_CREATE_TIME,
    COL_ID,
    COL_PLOT_AXES,
    COL_ROWS,
    COL_TITLE,
    LogListTableModel,
    LogRecord,
    PandasTableModel,
)
from logqbit.logfolder import LogFolder


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
    records = LogRecord.scan_directory(tmp_path)
    return records


class TestLogRecord:
    """Tests for LogRecord class."""
    
    def test_scan_directory_finds_logs(self, tmp_path: Path) -> None:
        """Test scanning a directory for log records."""
        # Create multiple log folders
        LogFolder.new(tmp_path, title="log1").flush()
        LogFolder.new(tmp_path, title="log2").flush()
        
        records = LogRecord.scan_directory(tmp_path)
        
        assert len(records) == 2
        assert all(isinstance(r, LogRecord) for r in records)
        assert {r.log_id for r in records} == {0, 1}
    
    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning an empty directory."""
        records = LogRecord.scan_directory(tmp_path)
        assert records == []
    
    def test_scan_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test scanning a directory that doesn't exist."""
        records = LogRecord.scan_directory(tmp_path / "nonexistent")
        assert records == []
    
    def test_load_dataframe(self, sample_logfolder: Path) -> None:
        """Test loading dataframe from a log record."""
        records = LogRecord.scan_directory(sample_logfolder)
        assert len(records) == 1
        
        record = records[0]
        df = record.load_dataframe()
        
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["x", "y", "z"]
        assert record.row_count == 3
        assert record.columns == ["x", "y", "z"]
    
    def test_load_dataframe_caches_result(self, sample_logfolder: Path) -> None:
        """Test that loading dataframe caches the result."""
        records = LogRecord.scan_directory(sample_logfolder)
        record = records[0]
        
        df1 = record.load_dataframe()
        df2 = record.load_dataframe()
        
        assert df1 is df2  # Should be the same object
    
    def test_read_yaml_text(self, sample_logfolder: Path) -> None:
        """Test reading YAML text from a log record."""
        records = LogRecord.scan_directory(sample_logfolder)
        record = records[0]
        
        yaml_text = record.read_yaml_text()
        
        assert isinstance(yaml_text, str)
        assert len(yaml_text) > 0
    
    def test_read_yaml_missing_file(self, tmp_path: Path) -> None:
        """Test reading YAML when file doesn't exist."""
        lf = LogFolder.new(tmp_path)
        # Don't create yaml file
        lf.df_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [1]}).to_feather(lf.df_path)
        
        records = LogRecord.scan_directory(tmp_path)
        record = records[0]
        
        yaml_text = record.read_yaml_text()
        assert "const.yaml not found" in yaml_text
    
    def test_list_image_files(self, tmp_path: Path) -> None:
        """Test listing image files in a log folder."""
        lf = LogFolder.new(tmp_path)
        lf.flush()
        
        # Create some image files
        (lf.path / "plot.png").touch()
        (lf.path / "result.jpg").touch()
        (lf.path / "data.txt").touch()  # Not an image
        
        records = LogRecord.scan_directory(tmp_path)
        record = records[0]
        
        images = record.list_image_files()
        
        assert len(images) == 2
        assert all(img.suffix.lower() in {".png", ".jpg"} for img in images)


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
        records = LogRecord.scan_directory(sample_logfolder)
        model = LogListTableModel()
        model.set_records(records)
        
        index = model.index(0, COL_PLOT_AXES)
        data = model.data(index, Qt.DisplayRole)
        
        # Should show first 3 characters of each axis
        assert data == "x, y"  # "x" + "y"
    
    def test_data_tooltip_plot_axes(self, sample_logfolder: Path) -> None:
        """Test tooltip showing full plot axes names."""
        records = LogRecord.scan_directory(sample_logfolder)
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
    
    def test_update_record(self, sample_records: list[LogRecord]) -> None:
        """Test updating a record in the model."""
        model = LogListTableModel()
        model.set_records(sample_records)
        
        record = sample_records[0]
        record.meta.title = "updated_title"
        
        model.update_record(record)
        
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
