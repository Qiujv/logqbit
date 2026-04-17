"""Tests for plotter module helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from logqbit.plotter import _build_grids_rect, _is_lexsorted


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
