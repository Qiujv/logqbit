from __future__ import annotations


import pandas as pd
from PySide6.QtCore import Qt

from logqbit.gui.browser.detail.data import PandasTableModel


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

    def test_column_values_ignores_preview_and_uses_column_position(self) -> None:
        frame = pd.DataFrame(
            [[1, "a"], [2, "a"], [3, "b"]],
            columns=["value", "value"],
        )
        model = PandasTableModel(frame, preview_limit=1)

        assert model.column_values(1).tolist() == ["a", "a", "b"]

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
