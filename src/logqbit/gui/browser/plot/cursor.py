"""Explicit draggable cursors for one- and two-dimensional plots."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, Qt
from PySide6.QtWidgets import QAbstractButton, QLabel

from logqbit.gui.browser.plot.mesh import PlotMeshData
from logqbit.gui.browser.plot.rendering import PlotSeries


class CursorController:
    """Own cursor graphics, readout, preview, and 2D section curves."""

    _PREVIEW_INTERVAL_MS = 33
    _PREVIEW_COLOR = (30, 144, 255, 120)

    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        horizontal_section_widget: pg.PlotWidget,
        vertical_section_widget: pg.PlotWidget,
        section_readout: QLabel,
        button: QAbstractButton,
        visibility_changed: Callable[[bool], None],
        activated: Callable[[], None],
        cancel_pending_click: Callable[[], None],
    ) -> None:
        self._plot_widget = plot_widget
        self._horizontal_widget = horizontal_section_widget
        self._vertical_widget = vertical_section_widget
        self._section_readout = section_readout
        self._button = button
        self._visibility_changed = visibility_changed
        self._activated = activated
        self._cancel_pending_click = cancel_pending_click
        self._mode: Literal["1d", "2d"] | None = None
        self._series: tuple[PlotSeries, ...] = ()
        self._mesh: PlotMeshData | None = None
        self._axis_names = ("x", "y", "z")
        self._group_label = ""
        self._items: list[object] = []
        self._vertical_line: pg.InfiniteLine | None = None
        self._horizontal_line: pg.InfiniteLine | None = None
        self._target: pg.TargetItem | None = None
        self._readout: pg.TextItem | None = None
        self._horizontal_curve: pg.PlotDataItem | None = None
        self._vertical_curve: pg.PlotDataItem | None = None
        self._preview_vertical_line: pg.InfiniteLine | None = None
        self._preview_horizontal_line: pg.InfiniteLine | None = None
        self._preview_readout: pg.TextItem | None = None
        self._preview_timer = QElapsedTimer()
        self._preview_visible = False

        button.clicked.connect(self._toggle)
        plot_widget.scene().sigMouseMoved.connect(self._update_preview)
        self._set_sections_visible(False)
        button.setEnabled(False)

    @property
    def active(self) -> bool:
        return self._button.isChecked()

    def configure_1d(
        self,
        series: Sequence[PlotSeries],
        *,
        preserve_overlay: bool = False,
    ) -> None:
        if preserve_overlay and self.active and self._mode == "1d":
            self._series = tuple(series)
            self._mesh = None
            self._group_label = ""
            return
        self.disable()
        self._mode = "1d"
        self._series = tuple(series)
        self._mesh = None
        self._group_label = ""
        self._button.setEnabled(bool(self._series))
        self._button.setToolTip("Enable a data cursor. Click the plot to move it.")

    def configure_2d(
        self,
        mesh: PlotMeshData,
        x_name: str,
        y_name: str,
        z_name: str,
        *,
        group_label: str = "",
        preserve_overlay: bool = False,
    ) -> None:
        if preserve_overlay and self.active and self._mode == "2d":
            self._series = ()
            self._mesh = mesh
            self._axis_names = (x_name, y_name, z_name)
            self._group_label = group_label
            return
        self.disable()
        self._mode = "2d"
        self._series = ()
        self._mesh = mesh
        self._axis_names = (x_name, y_name, z_name)
        self._group_label = group_label
        self._button.setEnabled(True)
        self._button.setToolTip(
            "Enable a crosshair and section plots. Click the plot to move it."
        )

    def disable(self) -> None:
        was_active = self.active
        self._cancel_pending_click()
        self._button.setChecked(False)
        self._remove_cursor_items()
        self._set_sections_visible(False)
        if was_active:
            self._visibility_changed(False)

    def clear(self) -> None:
        self.disable()
        self._mode = None
        self._series = ()
        self._mesh = None
        self._group_label = ""
        self._button.setEnabled(False)

    def _toggle(self, checked: bool) -> None:
        if checked:
            self._activated()
            if self._mode == "1d":
                self._enable_1d()
            elif self._mode == "2d":
                self._enable_2d()
            else:
                self._button.setChecked(False)
                return
        else:
            self._cancel_pending_click()
            self._remove_cursor_items()
            self._set_sections_visible(False)
        self._visibility_changed(self.active)

    def _enable_1d(self) -> None:
        reference = self._series[0]
        index = len(reference.x) // 2
        x = float(reference.x[index])
        line = pg.InfiniteLine(
            pos=x,
            angle=90,
            movable=False,
            pen=pg.mkPen("#222222", width=1.5),
        )
        line.setZValue(20)
        self._vertical_line = line
        self._add_main_item(line)
        self._readout = self._make_readout()
        self._make_preview_items()
        self._update_1d(x)

    def _enable_2d(self) -> None:
        mesh = self._mesh
        if mesh is None:
            return
        middle_column = mesh.x_column_count // 2
        x = float(mesh.x_values[middle_column])
        size = int(mesh.column_sizes[middle_column])
        render_column = 2 * middle_column
        y_values = mesh.y_corners[:size, render_column]
        y = float(y_values[len(y_values) // 2])

        vertical = pg.InfiniteLine(
            pos=x,
            angle=90,
            movable=False,
            pen=pg.mkPen("#222222", width=1.5),
        )
        horizontal = pg.InfiniteLine(
            pos=y,
            angle=0,
            movable=False,
            pen=pg.mkPen("#222222", width=1.5),
        )
        target = pg.TargetItem(
            pos=(x, y),
            size=12,
            symbol="s",
            pen=pg.mkPen("#222222", width=1.5),
            brush=pg.mkBrush(255, 255, 255, 190),
            movable=False,
        )
        vertical.setZValue(20)
        horizontal.setZValue(20)
        target.setZValue(21)
        self._vertical_line = vertical
        self._horizontal_line = horizontal
        self._target = target
        for item in (vertical, horizontal, target):
            self._add_main_item(item)
        self._readout = self._make_readout()
        self._make_preview_items()

        self._horizontal_curve = self._horizontal_widget.plot(
            pen=pg.mkPen("#1E90FF", width=1.5)
        )
        self._vertical_curve = self._vertical_widget.plot(
            pen=pg.mkPen("#1E90FF", width=1.5)
        )
        x_name, y_name, z_name = self._axis_names
        self._horizontal_widget.setLabel("bottom", x_name)
        self._horizontal_widget.setLabel("left", z_name)
        self._vertical_widget.setLabel("bottom", z_name)
        self._vertical_widget.setLabel("left", y_name)
        self._set_sections_visible(True)
        self._update_2d(x, y)

    def _make_readout(self) -> pg.TextItem:
        readout = pg.TextItem(
            color="#111111",
            fill=pg.mkBrush(255, 255, 255, 220),
            border=pg.mkPen("#777777"),
            anchor=(0, 1),
        )
        readout.setZValue(30)
        self._add_main_item(readout)
        return readout

    def _make_preview_items(self) -> None:
        vertical = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(self._PREVIEW_COLOR, width=1, style=Qt.DashLine),
        )
        vertical.setZValue(18)
        self._preview_vertical_line = vertical
        self._add_main_item(vertical)

        if self._mode == "2d":
            horizontal = pg.InfiniteLine(
                angle=0,
                movable=False,
                pen=pg.mkPen(self._PREVIEW_COLOR, width=1, style=Qt.DashLine),
            )
            horizontal.setZValue(18)
            self._preview_horizontal_line = horizontal
            self._add_main_item(horizontal)

        readout = pg.TextItem(
            color=self._PREVIEW_COLOR,
            fill=pg.mkBrush(255, 255, 255, 150),
            anchor=(0, 1),
        )
        readout.setZValue(19)
        self._preview_readout = readout
        self._add_main_item(readout)
        self._hide_preview()

    def _add_main_item(self, item: object) -> None:
        self._plot_widget.addItem(item)
        self._items.append(item)

    def _remove_cursor_items(self) -> None:
        for item in self._items:
            self._plot_widget.removeItem(item)
        self._items.clear()
        self._horizontal_widget.clear()
        self._vertical_widget.clear()
        self._vertical_line = None
        self._horizontal_line = None
        self._target = None
        self._readout = None
        self._horizontal_curve = None
        self._vertical_curve = None
        self._preview_vertical_line = None
        self._preview_horizontal_line = None
        self._preview_readout = None
        self._preview_visible = False

    def move_to_click(self, position) -> None:
        """Move the cursor from a confirmed single-click position."""
        if not self.active:
            return
        view_box = self._plot_widget.getPlotItem().vb
        point = view_box.mapToView(position)
        if self._mode == "1d":
            self._update_1d(float(point.x()))
        elif self._mode == "2d":
            self._update_2d(float(point.x()), float(point.y()))

    def _update_preview(self, scene_position) -> None:
        if not self.active or self._preview_readout is None:
            return
        plot_item = self._plot_widget.getPlotItem()
        if not plot_item.sceneBoundingRect().contains(scene_position):
            self._hide_preview()
            return
        if (
            self._preview_visible
            and self._preview_timer.elapsed() < self._PREVIEW_INTERVAL_MS
        ):
            return

        self._preview_timer.restart()
        self._preview_visible = True
        point = plot_item.vb.mapSceneToView(scene_position)
        x = self._snap_preview_coordinate(float(point.x()), "bottom")
        y = self._snap_preview_coordinate(float(point.y()), "left")
        if self._preview_vertical_line is not None:
            self._preview_vertical_line.setValue(x)
            self._preview_vertical_line.show()
        if self._preview_horizontal_line is not None:
            self._preview_horizontal_line.setValue(y)
            self._preview_horizontal_line.show()
        self._preview_readout.setText(
            f"x = {self._format_axis_value(x, 'bottom')}\n"
            f"y = {self._format_axis_value(y, 'left')}"
        )
        self._preview_readout.setPos(x, y)
        self._preview_readout.show()

    def _snap_preview_coordinate(self, value: float, axis_name: str) -> float:
        plot_item = self._plot_widget.getPlotItem()
        axis = plot_item.getAxis(axis_name)
        view_range = plot_item.vb.viewRange()[0 if axis_name == "bottom" else 1]
        pixel_length = (
            plot_item.vb.width() if axis_name == "bottom" else plot_item.vb.height()
        )
        if pixel_length <= 0:
            return value
        tick_levels = axis.tickSpacing(*view_range, pixel_length)
        if not tick_levels:
            return value
        spacing, offset = tick_levels[1] if len(tick_levels) > 1 else tick_levels[0]
        step = spacing * 0.1
        if not math.isfinite(step) or step <= 0:
            return value
        steps = (value - offset) / step
        nearest_step = math.floor(steps + 0.5) if steps >= 0 else math.ceil(steps - 0.5)
        return offset + nearest_step * step

    def _hide_preview(self) -> None:
        for item in (
            self._preview_vertical_line,
            self._preview_horizontal_line,
            self._preview_readout,
        ):
            if item is not None:
                item.hide()
        self._preview_visible = False

    def _update_1d(self, x: float) -> None:
        if not self._series or self._vertical_line is None or self._readout is None:
            return
        reference = self._series[0]
        reference_index = int(np.argmin(np.abs(reference.x - x)))
        snapped_x = float(reference.x[reference_index])
        lines = [f"x = {self._format_axis_value(snapped_x, 'bottom')}"]
        first_y = float(reference.y[reference_index])
        for series in self._series:
            index = int(np.argmin(np.abs(series.x - snapped_x)))
            lines.append(
                f"{series.name} = {self._format_axis_value(series.y[index], 'left')}"
            )
        self._vertical_line.setValue(snapped_x)
        self._readout.setText("\n".join(lines))
        self._readout.setPos(snapped_x, first_y)
        self._readout.show()

    def _format_axis_value(self, value: float, axis_name: str) -> str:
        """Format DateAxis coordinates like their visible tick labels."""
        axis = self._plot_widget.getPlotItem().getAxis(axis_name)
        if isinstance(axis, pg.DateAxisItem):
            try:
                return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            except (OverflowError, OSError, ValueError):
                pass
        return f"{value:.6g}"

    def _update_2d(self, x: float, y: float) -> None:
        mesh = self._mesh
        if mesh is None:
            return
        x, y, z = mesh.nearest_point(x, y)
        if self._vertical_line is not None:
            self._vertical_line.setValue(x)
        if self._horizontal_line is not None:
            self._horizontal_line.setValue(y)
        if self._target is not None:
            self._target.setPos(x, y)

        section_x, section_z = mesh.horizontal_section(y)
        _, section_y, vertical_z = mesh.vertical_section(x)
        if self._horizontal_curve is not None:
            self._horizontal_curve.setData(section_x, section_z)
            self._horizontal_curve.show()
        if self._vertical_curve is not None:
            self._vertical_curve.setData(vertical_z, section_y)
            self._vertical_curve.show()
        x_name, y_name, z_name = self._axis_names
        lines = [
            f"{x_name} = {self._format_axis_value(x, 'bottom')}",
            f"{y_name} = {self._format_axis_value(y, 'left')}",
            f"{z_name} = {z:.6g}",
        ]
        if self._group_label:
            lines.insert(0, self._group_label)
        text = "\n".join(lines)
        if self._readout is not None:
            self._readout.setText(text)
            self._readout.setPos(x, y)
            self._readout.show()
        self._section_readout.setText(text)

    def _set_sections_visible(self, visible: bool) -> None:
        self._horizontal_widget.setVisible(visible)
        self._vertical_widget.setVisible(visible)
        self._section_readout.setVisible(visible)
