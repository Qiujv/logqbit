"""Tests for plotter module helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from logqbit.plotter import _build_grids, _is_lexsorted


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
        x = np.array([1.0, 1.0, 2.0])
        y = np.array([2.0, 1.0, 1.0])  # y descending within x=1
        assert _is_lexsorted(x, y) is False

    def test_single_element(self) -> None:
        x = np.array([1.0])
        y = np.array([1.0])
        assert _is_lexsorted(x, y) is True

    def test_strictly_increasing_x(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 3.0, 1.0])  # y can be anything when x changes
        assert _is_lexsorted(x, y) is True


class TestBuildGrids:
    def _make_inputs(self, x_data, y_data, z_data):
        """Compute all inputs needed by _build_grids from flat x/y/z arrays."""
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
            top_y, step_c, typical_dy,
        )

    def test_uniform_grid_shape(self) -> None:
        """A 3x2 grid should produce z_grid (2, 3) and y_corners (3, 4)."""
        # 3 x-values, 2 y-values each → sorted lex order
        x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        z = np.arange(6.0)

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c, typical_dy = args

        z_grid, y_corners = _build_grids(
            x, y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c, typical_dy
        )

        assert z_grid.shape == (max_ny, nx_col)       # (2, 3)
        assert y_corners.shape == (max_ny + 1, nx_col + 1)  # (3, 4)

    def test_z_values_placed_correctly(self) -> None:
        """z_grid values should match input z values per column."""
        x = np.array([0.0, 0.0, 1.0, 1.0])
        y = np.array([0.0, 1.0, 0.0, 2.0])
        z = np.array([10.0, 20.0, 30.0, 40.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c, typical_dy = args

        z_grid, _ = _build_grids(
            x, y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c, typical_dy
        )

        # Column 0 (x=0): z[0]=10, z[1]=20
        assert z_grid[0, 0] == pytest.approx(10.0)
        assert z_grid[1, 0] == pytest.approx(20.0)
        # Column 1 (x=1): z[0]=30, z[1]=40
        assert z_grid[0, 1] == pytest.approx(30.0)
        assert z_grid[1, 1] == pytest.approx(40.0)

    def test_y_corners_first_column(self) -> None:
        """y_corners first column should contain the y data points."""
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, 2.0, 3.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c, typical_dy = args

        _, y_corners = _build_grids(
            x, y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c, typical_dy
        )

        assert y_corners[0, 0] == pytest.approx(1.0)
        assert y_corners[1, 0] == pytest.approx(2.0)
        assert y_corners[2, 0] == pytest.approx(3.0)
        # top extrapolated: 3.0 + 1.0 = 4.0
        assert y_corners[3, 0] == pytest.approx(4.0)

    def test_unequal_column_sizes_no_crash(self) -> None:
        """Columns with different point counts should not crash."""
        # col x=0 has 3 points, col x=1 has 1 point
        x = np.array([0.0, 0.0, 0.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 5.0])
        z = np.array([1.0, 2.0, 3.0, 4.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c, typical_dy = args

        z_grid, y_corners = _build_grids(
            x, y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c, typical_dy
        )

        assert z_grid.shape == (max_ny, nx_col)
        # Short column (x=1) should have z filled for row 0 and NaN beyond
        assert z_grid[0, 1] == pytest.approx(4.0)
        assert np.isnan(z_grid[1, 1])
        assert np.isnan(z_grid[2, 1])
        # y_corners for x=1 should be extrapolated (not NaN)
        assert not np.isnan(y_corners[1, 1])
        assert not np.isnan(y_corners[2, 1])
        assert not np.isnan(y_corners[3, 1])
