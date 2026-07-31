"""Explicit draggable cursors for one- and two-dimensional plots."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QPushButton

from logqbit.gui.plot.mesh import PlotMeshData


@dataclass(frozen=True)
class CursorSeries:
    """One numeric series available to a 1D cursor."""

    x: np.ndarray
    y: np.ndarray
    name: str


class CursorController:
    """Own cursor graphics, drag state, readout, and 2D section curves."""

    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        horizontal_section_widget: pg.PlotWidget,
        vertical_section_widget: pg.PlotWidget,
        section_readout: QLabel,
        button: QPushButton,
        visibility_changed: Callable[[bool], None],
        activated: Callable[[], None],
    ) -> None:
        self._plot_widget = plot_widget
        self._horizontal_widget = horizontal_section_widget
        self._vertical_widget = vertical_section_widget
        self._section_readout = section_readout
        self._button = button
        self._visibility_changed = visibility_changed
        self._activated = activated
        self._mode: Literal["1d", "2d"] | None = None
        self._series: tuple[CursorSeries, ...] = ()
        self._mesh: PlotMeshData | None = None
        self._axis_names = ("x", "y", "z")
        self._items: list[object] = []
        self._vertical_line: pg.InfiniteLine | None = None
        self._horizontal_line: pg.InfiniteLine | None = None
        self._target: pg.TargetItem | None = None
        self._readout: pg.TextItem | None = None
        self._horizontal_curve: pg.PlotDataItem | None = None
        self._vertical_curve: pg.PlotDataItem | None = None
        self._syncing = False

        button.clicked.connect(self._toggle)
        self._set_sections_visible(False)
        button.setEnabled(False)

    @property
    def active(self) -> bool:
        return self._button.isChecked()

    def configure_1d(self, series: Sequence[CursorSeries]) -> None:
        self.disable()
        self._mode = "1d"
        self._series = tuple(series)
        self._mesh = None
        self._button.setEnabled(bool(self._series))
        self._button.setToolTip("Enable a draggable data cursor")

    def configure_2d(
        self,
        mesh: PlotMeshData,
        x_name: str,
        y_name: str,
        z_name: str,
    ) -> None:
        self.disable()
        self._mode = "2d"
        self._series = ()
        self._mesh = mesh
        self._axis_names = (x_name, y_name, z_name)
        self._button.setEnabled(True)
        self._button.setToolTip("Enable a draggable crosshair and section plots")

    def disable(self) -> None:
        was_active = self.active
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
            movable=True,
            pen=pg.mkPen("#222222", width=1.5),
            hoverPen=pg.mkPen("#1E90FF", width=2),
        )
        line.setZValue(20)
        line.sigDragged.connect(self._hide_results)
        line.sigPositionChangeFinished.connect(self._finish_1d_drag)
        self._vertical_line = line
        self._add_main_item(line)
        self._readout = self._make_readout()
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
            movable=True,
            pen=pg.mkPen("#222222", width=1.5),
            hoverPen=pg.mkPen("#1E90FF", width=2),
        )
        horizontal = pg.InfiniteLine(
            pos=y,
            angle=0,
            movable=True,
            pen=pg.mkPen("#222222", width=1.5),
            hoverPen=pg.mkPen("#1E90FF", width=2),
        )
        target = pg.TargetItem(
            pos=(x, y),
            size=12,
            symbol="s",
            pen=pg.mkPen("#222222", width=1.5),
            hoverPen=pg.mkPen("#1E90FF", width=2),
            brush=pg.mkBrush(255, 255, 255, 190),
            hoverBrush=pg.mkBrush(255, 255, 255, 230),
            movable=True,
        )
        vertical.setZValue(20)
        horizontal.setZValue(20)
        target.setZValue(21)
        vertical.sigDragged.connect(self._vertical_dragged)
        horizontal.sigDragged.connect(self._horizontal_dragged)
        target.sigPositionChanged.connect(self._target_dragged)
        vertical.sigPositionChangeFinished.connect(self._finish_2d_drag)
        horizontal.sigPositionChangeFinished.connect(self._finish_2d_drag)
        target.sigPositionChangeFinished.connect(self._finish_2d_drag)
        self._vertical_line = vertical
        self._horizontal_line = horizontal
        self._target = target
        for item in (vertical, horizontal, target):
            self._add_main_item(item)
        self._readout = self._make_readout()

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

    def _hide_results(self, *_args) -> None:
        if self._readout is not None:
            self._readout.hide()
        if self._horizontal_curve is not None:
            self._horizontal_curve.hide()
        if self._vertical_curve is not None:
            self._vertical_curve.hide()
        self._section_readout.clear()

    def _vertical_dragged(self, *_args) -> None:
        if self._syncing or self._vertical_line is None:
            return
        self._hide_results()
        self._syncing = True
        try:
            if self._target is not None:
                self._target.setPos(self._vertical_line.value(), self._target.pos().y())
        finally:
            self._syncing = False

    def _horizontal_dragged(self, *_args) -> None:
        if self._syncing or self._horizontal_line is None:
            return
        self._hide_results()
        self._syncing = True
        try:
            if self._target is not None:
                self._target.setPos(self._target.pos().x(), self._horizontal_line.value())
        finally:
            self._syncing = False

    def _target_dragged(self, *_args) -> None:
        if self._syncing or self._target is None:
            return
        self._hide_results()
        self._syncing = True
        try:
            if self._vertical_line is not None:
                self._vertical_line.setValue(self._target.pos().x())
            if self._horizontal_line is not None:
                self._horizontal_line.setValue(self._target.pos().y())
        finally:
            self._syncing = False

    def _finish_1d_drag(self, *_args) -> None:
        if self._vertical_line is not None:
            self._update_1d(float(self._vertical_line.value()))

    def _finish_2d_drag(self, *_args) -> None:
        if self._vertical_line is None or self._horizontal_line is None:
            return
        self._update_2d(
            float(self._vertical_line.value()),
            float(self._horizontal_line.value()),
        )

    def _update_1d(self, x: float) -> None:
        if not self._series or self._vertical_line is None or self._readout is None:
            return
        reference = self._series[0]
        reference_index = int(np.argmin(np.abs(reference.x - x)))
        snapped_x = float(reference.x[reference_index])
        lines = [f"x = {snapped_x:.6g}"]
        first_y = float(reference.y[reference_index])
        for series in self._series:
            index = int(np.argmin(np.abs(series.x - snapped_x)))
            lines.append(f"{series.name} = {series.y[index]:.6g}")
        self._vertical_line.setValue(snapped_x)
        self._readout.setText("\n".join(lines))
        self._readout.setPos(snapped_x, first_y)
        self._readout.show()

    def _update_2d(self, x: float, y: float) -> None:
        mesh = self._mesh
        if mesh is None:
            return
        x, y, z = mesh.nearest_point(x, y)
        self._syncing = True
        try:
            if self._vertical_line is not None:
                self._vertical_line.setValue(x)
            if self._horizontal_line is not None:
                self._horizontal_line.setValue(y)
            if self._target is not None:
                self._target.setPos(x, y)
        finally:
            self._syncing = False

        section_x, section_z = mesh.horizontal_section(y)
        _, section_y, vertical_z = mesh.vertical_section(x)
        if self._horizontal_curve is not None:
            self._horizontal_curve.setData(section_x, section_z)
            self._horizontal_curve.show()
        if self._vertical_curve is not None:
            self._vertical_curve.setData(vertical_z, section_y)
            self._vertical_curve.show()
        x_name, y_name, z_name = self._axis_names
        text = f"{x_name} = {x:.6g}\n{y_name} = {y:.6g}\n{z_name} = {z:.6g}"
        if self._readout is not None:
            self._readout.setText(text)
            self._readout.setPos(x, y)
            self._readout.show()
        self._section_readout.setText(text)

    def _set_sections_visible(self, visible: bool) -> None:
        self._horizontal_widget.setVisible(visible)
        self._vertical_widget.setVisible(visible)
        self._section_readout.setVisible(visible)
