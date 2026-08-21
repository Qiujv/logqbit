from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt

from logqbit.catalog import LogCatalog, LogRecord
from logqbit.logfolder import LogFolder
from logqbit.gui.browser.window.model import (
    COL_ID,
    COL_PLOT_AXES,
    COL_ROWS,
    COL_TITLE,
    SORT_ROLE,
    LogListTableModel,
    LogListSortFilterProxyModel,
)


def scan_catalog(directory: Path) -> list[LogRecord]:
    return LogCatalog(directory).refresh()


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
        record = sample_records[0]
        pd.DataFrame({"x": range(1_234)}).to_feather(record.data_path)
        record.refresh()
        model = LogListTableModel()
        model.set_records([record])

        index = model.index(0, COL_ROWS)
        data = model.data(index, Qt.DisplayRole)

        assert data == "1,234"
        assert model.data(index, SORT_ROLE) == 1_234

    def test_proxy_sorts_numeric_ids_before_named_ids(self, tmp_path: Path) -> None:
        for name in ("beta", "10", "1.5", "2", "Alpha"):
            LogFolder(tmp_path / name)
        records = scan_catalog(tmp_path)
        model = LogListTableModel()
        model.set_records(list(reversed(records)))
        proxy = LogListSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(COL_ID, Qt.AscendingOrder)

        ordered_names = []
        for row in range(proxy.rowCount()):
            source = proxy.mapToSource(proxy.index(row, COL_ID))
            record = model.get_record(source.row())
            assert record is not None
            ordered_names.append(record.path.name)

        assert ordered_names == ["1.5", "2", "10", "Alpha", "beta"]

    def test_data_display_plot_axes(self, sample_logfolder: Path) -> None:
        """Test displaying plot axes with abbreviations."""
        records = scan_catalog(sample_logfolder)
        model = LogListTableModel()
        model.set_records(records)

        index = model.index(0, COL_PLOT_AXES)
        data = model.data(index, Qt.DisplayRole)

        # Should show first 3 characters of each axis
        assert data == "2,x,y"  # "x" + "y"

    def test_data_tooltip_plot_axes(self, sample_logfolder: Path) -> None:
        """Test tooltip showing full plot axes names."""
        records = scan_catalog(sample_logfolder)
        model = LogListTableModel()
        model.set_records(records)

        index = model.index(0, COL_PLOT_AXES)
        tooltip = model.data(index, Qt.ToolTipRole)

        assert tooltip == "x, y"

    def test_data_display_uses_resolved_plot_axes(
        self,
        sample_logfolder: Path,
    ) -> None:
        record = scan_catalog(sample_logfolder)[0]
        record.meta.update(
            plot_axes=["missing", "y", "y"],
            plot_fields=["x"],
        )
        model = LogListTableModel()
        model.set_records([record])
        index = model.index(0, COL_PLOT_AXES)

        assert record.meta.plot_axes == ("missing", "y", "y")
        assert model.data(index, Qt.DisplayRole) == "1,y"
        assert model.data(index, Qt.ToolTipRole) == "y"

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
