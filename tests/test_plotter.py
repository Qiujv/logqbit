"""Tests for plotter module helper functions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QSizePolicy

from logqbit.gui.plot.fit import fit_exponential, fit_quadratic
from logqbit.gui.plot.mesh import (
    _build_grids_rect,
    _is_lexsorted,
    warmup_plotter_jit,
)
from logqbit.gui.plot.widget import (
    PlotManager,
    TagBar,
    _partition_columns,
)


class TestTagBar:
    def test_set_columns_deduplicates_without_reordering(self) -> None:
        tag_bar = TagBar()

        tag_bar.set_columns(
            ["x", "y", "signal", "reference"],
            ["y", "x", "y"],
            ["reference", "y", "signal", "reference"],
        )

        assert tag_bar.axes == ["y", "x"]
        assert tag_bar.fields == ["reference", "signal"]
        assert tag_bar._split()[2] == []

    def test_set_columns_uses_first_ignored_column_when_fields_are_empty(
        self,
    ) -> None:
        tag_bar = TagBar()

        tag_bar.set_columns(
            ["x", "signal", "reference"],
            ["x", "x"],
            [],
        )

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["signal"]
        assert tag_bar._split()[2] == ["reference"]


def test_partition_columns_fills_axes_before_fields() -> None:
    axes, fields, ignored = _partition_columns(
        ["x", "signal", "reference"],
        [],
        [],
    )

    assert axes == ["x"]
    assert fields == ["signal"]
    assert ignored == ["reference"]


class TestFits:
    def test_exponential_reports_decay_time_with_offset(self) -> None:
        x = np.linspace(2.0, 12.0, 80)
        y = 1.5 + 4.0 * np.exp(-(x - 2.0) / 2.75)

        result = fit_exponential(x, y)

        assert result.value == pytest.approx(2.75, rel=1e-5)
        assert result.label.startswith("τ = ")
        assert result.x[[0, -1]] == pytest.approx([2.0, 12.0])

    @pytest.mark.parametrize("coefficient", [-2.0, 2.0])
    def test_quadratic_reports_extremum(self, coefficient: float) -> None:
        x = np.linspace(1_000_000.0, 1_000_010.0, 51)
        y = coefficient * (x - 1_000_004.25) ** 2 + 3.0

        result = fit_quadratic(x, y)

        assert result.value == pytest.approx(1_000_004.25)
        assert result.label.startswith("x = ")

    def test_fit_rejects_too_few_selected_points(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            fit_exponential(np.array([0.0, 1.0, 2.0]), np.ones(3))


class TestPlotManagerFitAndColorBar:
    def test_fit_selection_uses_points_inside_both_rectangle_axes(self) -> None:
        manager = PlotManager()
        x = np.arange(-2.0, 4.0)
        y = x**2
        manager.fit_controller.set_series(x, y, "signal", "#1E90FF")
        manager.quadratic_fit_button.click()
        manager.fit_view_box.setXRange(-3.0, 3.0, padding=0)
        selection = QRectF(-1.1, -0.1, 2.2, 1.2)

        manager.fit_controller._fit_selection(
            "quadratic",
            selection,
        )

        assert "using 3 points" in manager.plot_status_label.text()
        assert "x = " in manager.plot_status_label.text()
        assert "minimum" not in manager.plot_status_label.text()
        assert "maximum" not in manager.plot_status_label.text()
        assert manager.quadratic_fit_button.isChecked()
        assert manager.fit_view_box._fit_kind == "quadratic"

        selected_points = manager.fit_controller._overlays[1]
        assert selected_points.opts["size"] == 6
        assert selected_points.opts["brush"].color().name() == "#1e90ff"
        result_text = manager.fit_controller._overlays[3]
        assert (
            result_text.pos().y() <= selection.top()
            or result_text.pos().y() >= selection.bottom()
        )

        manager.quadratic_fit_button.click()
        assert not manager.quadratic_fit_button.isChecked()
        assert manager.fit_view_box._fit_kind is None
        manager.widget.deleteLater()

    def test_fit_buttons_require_one_1d_field(self) -> None:
        manager = PlotManager()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {"x": [0.0, 1.0], "a": [1.0, 2.0], "b": [2.0, 3.0]}
        )

        manager._refresh_plot_1d("x", ["a"])
        assert manager.exponential_fit_button.text() == "fit exp"
        assert manager.quadratic_fit_button.text() == "fit x²"
        for button in (
            manager.exponential_fit_button,
            manager.quadratic_fit_button,
            manager.copy_plot_button,
        ):
            expected_width = button.fontMetrics().horizontalAdvance(button.text()) + 24
            assert button.width() == expected_width
        assert not manager.exponential_fit_button.isHidden()
        assert manager.exponential_fit_button.isEnabled()
        assert manager.quadratic_fit_button.isEnabled()
        assert (
            manager.plot_status_label.sizePolicy().horizontalPolicy()
            == QSizePolicy.Ignored
        )

        manager._refresh_plot_1d("x", ["a", "b"])
        assert not manager.exponential_fit_button.isEnabled()
        assert not manager.quadratic_fit_button.isEnabled()
        manager.widget.deleteLater()

    def test_copy_plot_temporarily_adds_record_path_to_title(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = PlotManager()
        manager._plot_record = SimpleNamespace(path=Path("/logs/example-record"))
        observed: dict[str, object] = {}

        class FakeExporter:
            def __init__(self, plot_item) -> None:
                self.plot_item = plot_item

            def export(self, *, copy: bool) -> None:
                observed["copy"] = copy
                observed["title"] = self.plot_item.titleLabel.text
                observed["visible"] = self.plot_item.titleLabel.isVisible()

        monkeypatch.setattr("logqbit.gui.plot.widget.ImageExporter", FakeExporter)

        manager.copy_plot_to_clipboard()

        assert observed == {
            "copy": True,
            "title": "/logs/example-record",
            "visible": True,
        }
        assert not manager.plot_widget.getPlotItem().titleLabel.isVisible()
        manager.widget.deleteLater()

    def test_color_bar_is_reused_and_removed_when_switching_to_1d(self) -> None:
        manager = PlotManager()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0, 1.0],
                "y": [0.0, 1.0, 0.0, 1.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        )

        manager._refresh_plot_2d("x", "y", "z")
        color_bar = manager._color_bar
        assert color_bar is not None
        assert color_bar.getAxis("left").labelText == ""
        assert manager.exponential_fit_button.isHidden()
        assert manager.quadratic_fit_button.isHidden()

        manager._refresh_plot_2d("x", "y", "z")
        assert manager._color_bar is color_bar

        manager._refresh_plot_1d("x", ["z"])
        assert manager._color_bar is None
        manager.widget.deleteLater()


class TestIsLexsorted:
    def test_already_sorted(self) -> None:
        x = np.array([1.0, 1.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 1.0, 2.0])
        assert _is_lexsorted(x, y) is True

    def test_not_sorted_by_x(self) -> None:
        x = np.array([2.0, 1.0, 2.0])
        y = np.array([1.0, 1.0, 2.0])
        assert _is_lexsorted(x, y) is False

    def test_not_sorted_by_y_within_same_x(self) -> None:
        # y reverses direction mid-column (1→3→2): not monotonic → False
        x = np.array([1.0, 1.0, 1.0, 2.0])
        y = np.array([1.0, 3.0, 2.0, 1.0])
        assert _is_lexsorted(x, y) is False

    def test_y_descending_within_column_is_ok(self) -> None:
        # y monotonically descending within x=1 is now allowed
        x = np.array([1.0, 1.0, 2.0])
        y = np.array([2.0, 1.0, 1.0])
        assert _is_lexsorted(x, y) is True

    def test_single_element(self) -> None:
        x = np.array([1.0])
        y = np.array([1.0])
        assert _is_lexsorted(x, y) is True

    def test_strictly_increasing_x(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 3.0, 1.0])  # y can be anything when x changes
        assert _is_lexsorted(x, y) is True


class TestBuildGridsRect:
    def _make_inputs(self, x_data, y_data, z_data):
        """Compute all inputs needed by _build_grids_rect from flat x/y/z arrays."""
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        z_data = np.asarray(z_data, dtype=float)
        N = len(x_data)

        change = np.empty(N, dtype=np.bool_)
        change[0] = True
        change[1:] = x_data[1:] != x_data[:-1]
        xu = x_data[change]
        col_starts = np.flatnonzero(change)
        nx_col = len(xu)
        col_ends = np.append(col_starts[1:], N)
        col_sizes = col_ends - col_starts
        max_ny = int(col_sizes.max())

        ref_col = int(np.argmax(col_sizes))
        ref_y = y_data[col_starts[ref_col] : col_ends[ref_col]]
        typical_dy = float(np.median(np.diff(ref_y))) if len(ref_y) > 1 else 1.0

        last_y = y_data[col_ends - 1]
        prev_idx = np.maximum(col_ends - 2, col_starts)
        step_c = np.where(col_sizes > 1, last_y - y_data[prev_idx], typical_dy)
        top_y = last_y + step_c

        return (
            x_data, y_data, z_data,
            col_starts, col_sizes, max_ny, nx_col,
            top_y, step_c,
        )

    def test_uniform_grid_shape(self) -> None:
        """A 3x2 grid: z_final shape (2, 5), y_final shape (3, 6)."""
        x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        z = np.arange(6.0)

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c = args

        z_final, y_final = _build_grids_rect(
            y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c
        )

        assert z_final.shape == (max_ny, 2 * nx_col - 1)   # (2, 5)
        assert y_final.shape == (max_ny + 1, 2 * nx_col)   # (3, 6)

    def test_z_values_placed_correctly(self) -> None:
        """Data columns sit at even indices; odd separator columns are NaN."""
        x = np.array([0.0, 0.0, 1.0, 1.0])
        y = np.array([0.0, 1.0, 0.0, 2.0])
        z = np.array([10.0, 20.0, 30.0, 40.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c = args

        z_final, _ = _build_grids_rect(
            y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c
        )

        # Column 0 (c2=0): z values 10, 20
        assert z_final[0, 0] == pytest.approx(10.0)
        assert z_final[1, 0] == pytest.approx(20.0)
        # Column 1 (c2=2): z values 30, 40
        assert z_final[0, 2] == pytest.approx(30.0)
        assert z_final[1, 2] == pytest.approx(40.0)
        # Odd separator column is NaN
        assert np.isnan(z_final[0, 1])

    def test_y_corners_horizontal_edges(self) -> None:
        """Left and right y corners per cell must be equal (horizontal edges)."""
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, 2.0, 3.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c = args

        _, y_final = _build_grids_rect(
            y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c
        )

        # Column 0 occupies c2=0,1; left and right y must match
        for r in range(max_ny + 1):
            assert y_final[r, 0] == pytest.approx(y_final[r, 1])
        # First 3 rows match input y
        assert y_final[0, 0] == pytest.approx(1.0)
        assert y_final[1, 0] == pytest.approx(2.0)
        assert y_final[2, 0] == pytest.approx(3.0)
        # Top extrapolated: 3 + 1 = 4
        assert y_final[3, 0] == pytest.approx(4.0)

    def test_unequal_column_sizes_no_crash(self) -> None:
        """Columns with different point counts should not crash."""
        x = np.array([0.0, 0.0, 0.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 5.0])
        z = np.array([1.0, 2.0, 3.0, 4.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c = args

        z_final, y_final = _build_grids_rect(
            y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c
        )

        assert z_final.shape == (max_ny, 2 * nx_col - 1)
        # Short column (x=1, c2=2): first row filled, rest NaN
        assert z_final[0, 2] == pytest.approx(4.0)
        assert np.isnan(z_final[1, 2])
        assert np.isnan(z_final[2, 2])
        # y corners for short column extrapolated (not NaN)
        assert not np.isnan(y_final[1, 2])
        assert not np.isnan(y_final[2, 2])
        assert not np.isnan(y_final[3, 2])


def test_warmup_plotter_jit() -> None:
    warmup_plotter_jit()
