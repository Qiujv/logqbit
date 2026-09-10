"""Stored-record plot data preparation and graphics-layer ownership."""

from __future__ import annotations

import functools
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt

from logqbit.gui.browser.plot.mesh import PlotMeshData, build_plot_mesh

MAX_PLOT_GROUPS = 50
COLOR_BAR_HEIGHT_FACTOR = 0.9
PLOT_COLORS = (
    "#1E90FF",
    "#FF6347",
    "#32CD32",
    "#FF8C00",
    "#9370DB",
    "#00CED1",
    "#FF1493",
    "#8B4513",
)


@dataclass(frozen=True)
class PlotGroup:
    """One dataframe slice and its user-facing group label."""

    label: str
    frame: pd.DataFrame


def iter_plot_groups(
    frame: pd.DataFrame,
    groupby: Sequence[str],
) -> Iterator[PlotGroup]:
    """Yield plot groups in first-observed order, retaining missing values."""
    columns = tuple(groupby)
    if not columns:
        yield PlotGroup("", frame)
        return

    grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
    grouped = frame.groupby(grouper, sort=False, dropna=False, observed=True)
    for key, group_frame in grouped:
        values = (key,) if len(columns) == 1 else tuple(key)
        label = ", ".join(
            f"{column}={_format_group_value(value)}"
            for column, value in zip(columns, values, strict=True)
        )
        yield PlotGroup(label, group_frame)


def _format_group_value(value: object) -> str:
    try:
        if bool(pd.isna(value)):
            return "<NA>"
    except (TypeError, ValueError):
        pass
    return str(value)


@dataclass(frozen=True)
class PlotSeries:
    """One numeric series drawn in a 1D plot."""

    x: np.ndarray
    y: np.ndarray
    name: str


def _plot_values(values: pd.Series) -> tuple[pd.Series, bool]:
    """Convert a plot column to numeric coordinates and report datetime input."""
    if not pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_numeric(values, errors="coerce"), False

    # DateAxisItem displays Unix timestamps in the local time zone, whereas
    # pandas stores datetime64 values as nanoseconds. Treat naive datetimes as
    # local too, preserving their displayed clock time. Keep missing values as
    # NaN so the normal per-series dropna() handling removes them.
    if values.dt.tz is None:
        values = values.dt.tz_localize(datetime.now().astimezone().tzinfo)
    timestamps = values.astype("int64").astype(float).div(1_000_000_000)
    return timestamps.where(values.notna()), True


def _groups_for_plot(
    frame: pd.DataFrame,
    groupby: Sequence[str],
) -> tuple[list[PlotGroup], int]:
    """Return at most the latest observed plot groups and their total count."""
    groups: deque[PlotGroup] = deque(maxlen=MAX_PLOT_GROUPS)
    total_count = 0
    for total_count, group in enumerate(iter_plot_groups(frame, groupby), start=1):
        groups.append(group)
    return list(groups), total_count


@dataclass(frozen=True)
class RenderResult:
    """Data-layer result consumed by the panel's interaction controllers."""

    status: str
    mode: Literal["empty", "1d", "2d"] = "empty"
    x_is_datetime: bool = False
    y_is_datetime: bool = False
    series: tuple[PlotSeries, ...] = ()
    fit_series: tuple[np.ndarray, np.ndarray, str, str] | None = None
    mesh: PlotMeshData | None = None
    group_label: str = ""
    axis_names: tuple[str, str, str] = ("", "", "")


class PlotRenderer:
    """Own data graphics, axes, legends and color-bar connections, never overlays."""

    def __init__(self, plot_widget: pg.PlotWidget) -> None:
        self.plot_widget = plot_widget
        self._data_items: list[object] = []
        self._color_bar: pg.ColorBarItem | None = None
        self._color_bar_mesh: pg.PColorMeshItem | None = None
        self._legend: pg.LegendItem | None = None
        self._mesh_item: pg.PColorMeshItem | None = None
        self._mesh_items: list[pg.PColorMeshItem] = []
        self._mesh_levels: tuple[float, float] | None = None
        self._mesh_z_column: str | None = None
        self.marker_size = 6

    @property
    def has_mesh(self) -> bool:
        return self._mesh_item is not None

    def clear(self, *, preserve_color_bar: bool = False) -> None:
        self._mesh_item = None
        self._mesh_items = []
        self._mesh_levels = None
        self._mesh_z_column = None
        self._clear_data_layer(preserve_color_bar=preserve_color_bar)

    def set_cursor_visible(self, active: bool) -> None:
        if self._mesh_item is None or self._mesh_levels is None:
            return
        if active:
            self._hide_color_bar()
        else:
            self._show_color_bar(
                self._mesh_item, self._mesh_levels, self._mesh_z_column
            )

    def _clear_data_layer(self, *, preserve_color_bar: bool = False) -> None:
        """Remove plot data while retaining user-owned interaction overlays."""
        if preserve_color_bar:
            self._disconnect_color_bar_mesh()
        else:
            self._hide_color_bar()
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            for item in self._data_items:
                plot_item.removeItem(item)
        self._data_items.clear()
        self._clear_legend()

    def _set_datetime_axes(self, x_is_datetime: bool, y_is_datetime: bool) -> None:
        """Install date axes for datetime coordinates and normal axes otherwise."""
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is None:
            return

        axis_items = {
            "bottom": (
                pg.DateAxisItem(orientation="bottom")
                if x_is_datetime
                else pg.AxisItem(orientation="bottom")
            ),
            "left": (
                pg.DateAxisItem(orientation="left")
                if y_is_datetime
                else pg.AxisItem(orientation="left")
            ),
        }
        for axis in axis_items.values():
            axis.setTextPen("k")
            axis.enableAutoSIPrefix(False)
        plot_item.setAxisItems(axis_items)

        if x_is_datetime:
            plot_item.ctrl.logXCheck.setChecked(False)
        if y_is_datetime:
            plot_item.ctrl.logYCheck.setChecked(False)

    def _clear_legend(self) -> None:
        if self._legend is None:
            return
        plot_item = self.plot_widget.getPlotItem()
        self._legend.clear()
        scene = self._legend.scene()
        if scene is not None:
            scene.removeItem(self._legend)
        if plot_item is not None:
            plot_item.legend = None
        self._legend = None

    def _show_legend(self) -> None:
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            self._legend = plot_item.addLegend(
                offset=(10, 10),
                horSpacing=1,
                verSpacing=0,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(255, 255, 255, 200),
            )
            self._legend.layout.setContentsMargins(2, 2, 2, 2)

    def _hide_color_bar(self) -> None:
        if self._color_bar is None:
            return
        plot_item = self.plot_widget.getPlotItem()
        self._disconnect_color_bar_mesh()
        if plot_item is not None:
            plot_item.layout.removeItem(self._color_bar)
        self._color_bar.setParentItem(None)
        self._color_bar.deleteLater()
        self._color_bar = None

    def _disconnect_color_bar_mesh(self) -> None:
        """Detach the color bar before its mesh leaves the graphics scene."""
        if self._color_bar is None:
            return
        if self._color_bar_mesh is not None:
            try:
                self._color_bar_mesh.sigLevelsChanged.disconnect(
                    self._color_bar._levelsChangedHandler
                )
            except RuntimeError:
                pass
        self._color_bar.setImageItem([])
        self._color_bar_mesh = None

    def _show_color_bar(
        self,
        mesh: pg.PColorMeshItem,
        levels: tuple[float, float],
        z_column: str,
    ) -> None:
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is None:
            return
        if self._color_bar is None:
            self._color_bar = plot_item.addColorBar(
                mesh,
                values=levels,
                width=12,
                colorMap=self.cmap,
                interactive=False,
                colorMapMenu=False,
                pen="k",
                label=z_column,
            )
            self._color_bar.axis.setPen("k")
            self._color_bar.axis.setTextPen("k")
            self._color_bar.axis.setWidth(38)
            self._color_bar.getAxis("left").setWidth(1)
            plot_item.layout.setColumnFixedWidth(4, 2)
            plot_item.layout.setColumnSpacing(4, 0)
        else:
            self._disconnect_color_bar_mesh()
            self._color_bar.setLevels(levels)
            self._color_bar.setImageItem(mesh)
            self._color_bar.getAxis("left").setLabel(z_column)
        self._color_bar_mesh = mesh
        self._resize_color_bar(plot_item)

    def _resize_color_bar(self, plot_item: pg.PlotItem) -> None:
        if self._color_bar is None or plot_item.vb.height() <= 0:
            return
        self._color_bar.setMaximumHeight(
            round(plot_item.vb.height() * COLOR_BAR_HEIGHT_FACTOR)
        )
        plot_item.layout.setAlignment(self._color_bar, Qt.AlignVCenter)

    def render_1d(
        self,
        frame: pd.DataFrame,
        x_col: str,
        y_cols: Sequence[str],
        groupby: Sequence[str] = (),
    ) -> RenderResult:
        for col in (x_col, *groupby):
            if col not in frame.columns:
                return RenderResult(f"Column '{col}' not in data.")

        self.clear()
        x_values, x_is_datetime = _plot_values(frame[x_col])
        y_datetime_columns = [
            pd.api.types.is_datetime64_any_dtype(frame[y_col])
            for y_col in y_cols
            if y_col in frame.columns
        ]
        if any(y_datetime_columns) and not all(y_datetime_columns):
            return RenderResult("Datetime and numeric fields cannot share a y axis.")
        y_is_datetime = bool(y_datetime_columns) and all(y_datetime_columns)
        self._set_datetime_axes(x_is_datetime, y_is_datetime)
        if groupby:
            self._show_legend()

        plotted = 0
        plotted_groups: set[str] = set()
        fit_series: tuple[np.ndarray, np.ndarray, str, str] | None = None
        cursor_series: list[PlotSeries] = []
        plot_groups, total_group_count = _groups_for_plot(frame, groupby)
        for plot_group in plot_groups:
            x_values, _ = _plot_values(plot_group.frame[x_col])
            for y_col in y_cols:
                if y_col not in plot_group.frame.columns:
                    continue
                y_values, _ = _plot_values(plot_group.frame[y_col])
                df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
                if df.empty:
                    continue
                color = PLOT_COLORS[plotted % len(PLOT_COLORS)]
                series_name = (
                    f"{y_col} | {plot_group.label}" if plot_group.label else y_col
                )
                legend_name = (
                    plot_group.label
                    if plot_group.label and len(y_cols) == 1
                    else series_name
                )
                show_markers = len(df) <= 2001
                pen = pg.mkPen(color=color, width=2)
                if show_markers:
                    item = self.plot_widget.plot(
                        df["x"].values,
                        df["y"].values,
                        pen=pen,
                        name=legend_name,
                        symbol="o",
                        symbolSize=self.marker_size,
                        symbolPen=pg.mkPen(color=color),
                        symbolBrush=pg.mkBrush("#FFFFFF"),
                    )
                else:
                    item = self.plot_widget.plot(
                        df["x"].values,
                        df["y"].values,
                        pen=pen,
                        name=legend_name,
                    )
                self._data_items.append(item)
                if fit_series is None:
                    fit_series = (
                        df["x"].to_numpy(dtype=float),
                        df["y"].to_numpy(dtype=float),
                        series_name,
                        color,
                    )
                cursor_series.append(
                    PlotSeries(
                        df["x"].to_numpy(dtype=float),
                        df["y"].to_numpy(dtype=float),
                        series_name,
                    )
                )
                plotted += 1
                plotted_groups.add(plot_group.label)

        if plotted == 0:
            return RenderResult("No numeric data to plot.")

        self.plot_widget.setLabel("bottom", x_col)
        self.plot_widget.setLabel("left", ", ".join(y_cols))
        status = f"1D plot: {x_col} vs {', '.join(y_cols[:3])}"
        if groupby:
            status += f" ({len(plotted_groups)} groups, {plotted} curves)"
            if total_group_count > MAX_PLOT_GROUPS:
                status += f"; showing latest {MAX_PLOT_GROUPS} of {total_group_count}"
        return RenderResult(
            status, "1d", x_is_datetime, y_is_datetime, tuple(cursor_series), fit_series
        )

    def render_2d(
        self,
        frame: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        groupby: Sequence[str] = (),
    ) -> RenderResult:
        for col in (x_col, y_col, z_col, *groupby):
            if col not in frame.columns:
                return RenderResult(f"Column '{col}' not in data.")

        x_is_datetime = pd.api.types.is_datetime64_any_dtype(frame[x_col])
        y_is_datetime = pd.api.types.is_datetime64_any_dtype(frame[y_col])

        mesh_groups = []
        plot_groups, total_group_count = _groups_for_plot(frame, groupby)
        for plot_group in plot_groups:
            sub = plot_group.frame[[x_col, y_col, z_col]]
            arr = np.column_stack(
                [_plot_values(sub[column])[0] for column in sub.columns]
            ).astype(float, copy=False)

            mask = ~np.isnan(arr).any(axis=1)
            if not mask.any():
                continue
            filtered = arr[mask]
            mesh = build_plot_mesh(filtered[:, 0], filtered[:, 1], filtered[:, 2])
            mesh_groups.append((plot_group.label, mesh))

        if not mesh_groups:
            return RenderResult("No numeric data to plot.")

        levels = (
            min(mesh.levels[0] for _, mesh in mesh_groups),
            max(mesh.levels[1] for _, mesh in mesh_groups),
        )

        self.clear(preserve_color_bar=True)
        self._set_datetime_axes(x_is_datetime, y_is_datetime)
        if groupby:
            self._show_legend()

        rendered_groups = []
        for index, (label, mesh) in enumerate(mesh_groups):
            pcm = pg.PColorMeshItem(
                mesh.x_corners,
                mesh.y_corners,
                mesh.z_grid,
                colorMap=self.cmap,
                levels=levels,
            )
            self.plot_widget.addItem(pcm)
            self._data_items.append(pcm)
            self._mesh_items.append(pcm)
            rendered_groups.append((label, mesh, pcm))

            if self._legend is not None:
                color = PLOT_COLORS[index % len(PLOT_COLORS)]
                legend_sample = pg.PlotDataItem(
                    pen=pg.mkPen(color=color, width=2),
                )
                self._legend.addItem(legend_sample, label)

        cursor_label, cursor_mesh, cursor_pcm = max(
            rendered_groups,
            key=lambda item: item[1].point_count,
        )
        self._mesh_item = cursor_pcm
        self._mesh_levels = levels
        self._mesh_z_column = z_col
        self._show_color_bar(cursor_pcm, levels, z_col)
        self.plot_widget.setLabel("bottom", x_col)
        self.plot_widget.setLabel("left", y_col)

        total_points = sum(mesh.point_count for _, mesh, _ in rendered_groups)
        if groupby:
            status = (
                f"2D plot: {total_points} points in {len(rendered_groups)} group(s); "
                f"cursor uses {cursor_label} ({cursor_mesh.point_count} points)"
            )
            if total_group_count > MAX_PLOT_GROUPS:
                status += f"; showing latest {MAX_PLOT_GROUPS} of {total_group_count}"
        else:
            status = (
                f"2D plot: {cursor_mesh.point_count} points → "
                f"{cursor_mesh.x_column_count}×{cursor_mesh.max_y_count} mesh"
            )
        return RenderResult(
            status,
            "2d",
            x_is_datetime,
            y_is_datetime,
            mesh=cursor_mesh,
            group_label=cursor_label,
            axis_names=(x_col, y_col, z_col),
        )

    @functools.cached_property
    def cmap(self):
        cmap = pg.colormap.get("RdBu_r", source="matplotlib")
        if cmap is None:
            cmap = pg.colormap.get("CET-D1")
        return cmap
