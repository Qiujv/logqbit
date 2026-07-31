"""Numerical preparation for two-dimensional plot meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange


@dataclass(frozen=True)
class PlotMeshData:
    """Prepared corners, values, and dimensions for a 2D color mesh."""

    x_corners: np.ndarray
    y_corners: np.ndarray
    z_grid: np.ndarray
    levels: tuple[float, float]
    point_count: int
    x_column_count: int
    max_y_count: int


@njit(cache=True)
def _is_lexsorted(x: np.ndarray, y: np.ndarray) -> bool:
    """Return whether x is ascending and each x-column has monotonic y."""
    point_count = len(x)
    previous_x = x[0]
    previous_y = y[0]
    y_direction = 0
    for i in range(1, point_count):
        current_x = x[i]
        if current_x < previous_x:
            return False
        if current_x == previous_x:
            current_y = y[i]
            if current_y > previous_y:
                if y_direction == -1:
                    return False
                y_direction = 1
            elif current_y < previous_y:
                if y_direction == 1:
                    return False
                y_direction = -1
            previous_y = current_y
        else:
            y_direction = 0
            previous_x = current_x
            previous_y = y[i]
    return True


@njit(parallel=True, cache=True)
def _build_grids_rect(ys, zs, col_starts, col_sizes, max_ny, nx_col, top_y, step_c):
    """Build rect-separated z/y arrays for PColorMeshItem in parallel."""
    z_final = np.full((max_ny, 2 * nx_col - 1), np.nan)
    y_final = np.empty((max_ny + 1, 2 * nx_col))

    for column in prange(nx_col):
        start = col_starts[column]
        size = col_sizes[column]
        output_column = column + column

        for row in range(size):
            z_final[row, output_column] = zs[start + row]
            y_value = ys[start + row]
            y_final[row, output_column] = y_value
            y_final[row, output_column + 1] = y_value

        column_top = top_y[column]
        y_final[size, output_column] = column_top
        y_final[size, output_column + 1] = column_top

        column_step = step_c[column]
        for row in range(size + 1, max_ny + 1):
            value = column_top + (row - size) * column_step
            y_final[row, output_column] = value
            y_final[row, output_column + 1] = value

    return z_final, y_final


def build_plot_mesh(
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
) -> PlotMeshData:
    """Prepare flat numeric x/y/z data for ``PColorMeshItem``."""
    point_count = len(x_data)
    if point_count > 1 and not _is_lexsorted(x_data, y_data):
        sort_indices = np.lexsort((y_data, x_data))
        x_data = x_data[sort_indices]
        y_data = y_data[sort_indices]
        z_data = z_data[sort_indices]

    column_change = np.empty(point_count, dtype=np.bool_)
    column_change[0] = True
    column_change[1:] = x_data[1:] != x_data[:-1]
    unique_x = x_data[column_change]
    column_starts = np.flatnonzero(column_change)
    x_column_count = len(unique_x)
    column_ends = np.append(column_starts[1:], point_count)
    column_sizes = column_ends - column_starts
    max_y_count = int(column_sizes.max())

    reference_column = int(np.argmax(column_sizes))
    reference_y = y_data[
        column_starts[reference_column] : column_ends[reference_column]
    ]
    typical_y_step = (
        float(np.median(np.diff(reference_y))) if len(reference_y) > 1 else 1.0
    )

    last_y = y_data[column_ends - 1]
    previous_indices = np.maximum(column_ends - 2, column_starts)
    column_steps = np.where(
        column_sizes > 1,
        last_y - y_data[previous_indices],
        typical_y_step,
    )
    top_y = last_y + column_steps

    x_edges = np.empty(x_column_count + 1)
    x_edges[:x_column_count] = unique_x
    last_x_step = unique_x[-1] - unique_x[-2] if x_column_count > 1 else 1.0
    x_edges[-1] = unique_x[-1] + last_x_step
    rectangular_x_edges = np.repeat(x_edges, 2)[1:-1]
    x_corners = np.broadcast_to(
        rectangular_x_edges,
        (max_y_count + 1, 2 * x_column_count),
    )

    z_grid, y_corners = _build_grids_rect(
        y_data,
        z_data,
        column_starts,
        column_sizes,
        max_y_count,
        x_column_count,
        top_y,
        column_steps,
    )

    z_min = float(np.min(z_data))
    z_max = float(np.max(z_data))
    if z_min == z_max:
        padding = abs(z_min) * 0.01 or 1.0
        z_min -= padding
        z_max += padding

    return PlotMeshData(
        x_corners=x_corners,
        y_corners=y_corners,
        z_grid=z_grid,
        levels=(z_min, z_max),
        point_count=point_count,
        x_column_count=x_column_count,
        max_y_count=max_y_count,
    )


def warmup_plotter_jit() -> None:
    """Compile numba-backed 2D plotting helpers before the first real plot."""
    x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    z = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    column_starts = np.array([0, 2], dtype=np.int64)
    column_sizes = np.array([2, 2], dtype=np.int64)
    top_y = np.array([2.0, 2.0], dtype=np.float64)
    column_steps = np.array([1.0, 1.0], dtype=np.float64)

    _is_lexsorted(x, y)
    _build_grids_rect(
        y,
        z,
        column_starts,
        column_sizes,
        2,
        2,
        top_y,
        column_steps,
    )
