"""Standalone plotter for log records."""

from __future__ import annotations

import functools
import html
import logging
from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..catalog import LogRecord

from .plot_fit import FitController, FitViewBox
from .plot_mesh import _build_grids_rect as _build_grids_rect
from .plot_mesh import _is_lexsorted as _is_lexsorted
from .plot_mesh import build_plot_mesh
from .plot_mesh import warmup_plotter_jit as warmup_plotter_jit
from .plot_tag_bar import TagBar as TagBar
from .plot_tag_bar import _partition_columns as _partition_columns

logger = logging.getLogger(__name__)


def _load_window_icon() -> QIcon:
    try:
        icon_path = files("logqbit") / "assets" / "browser.svg"
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            return icon
    except Exception as exc:
        logger.debug(f"Failed to load window icon: {exc}")
    return QIcon()


WINDOW_ICON = _load_window_icon()


class PlotManager:
    def __init__(self, parent: QWidget | None = None):
        self._plot_record: LogRecord | None = None
        self._plot_frame: pd.DataFrame | None = None
        self._suppress_updates = False
        self._needs_refresh = False
        self._color_bar: pg.ColorBarItem | None = None
        self.widget = self._create_widget(parent)

    def _create_widget(self, parent: QWidget | None = None) -> QWidget:
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Tag bar
        self.tag_bar = TagBar()
        self.tag_bar.changed.connect(self.refresh_plot)
        self.tag_bar.save_clicked.connect(self._save_tag_bar)
        layout.addWidget(self.tag_bar)

        # Plot widget
        self.fit_view_box = FitViewBox()
        plot_item = pg.PlotItem(viewBox=self.fit_view_box)
        self.plot_widget = pg.PlotWidget(plotItem=plot_item)
        self.plot_widget.setBackground("w")
        self.plot_widget.useOpenGL(True)
        self.plot_widget.setMinimumHeight(220)

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            # plot_item.setDownsampling(auto=True, mode="subsample")
            for axis in ["left", "bottom", "top", "right"]:
                plot_item.getAxis(axis).setTextPen("k")
                plot_item.getAxis(axis).enableAutoSIPrefix(False)

        layout.addWidget(self.plot_widget, stretch=1)

        # Status row
        status_row = QHBoxLayout()
        self.plot_status_label = QLabel("No data to plot.")
        self.plot_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.plot_status_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        status_row.addWidget(self.plot_status_label, stretch=1)
        self.exponential_fit_button = QPushButton("fit exp")
        self.exponential_fit_button.setCheckable(True)
        self._make_button_compact(self.exponential_fit_button)
        status_row.addWidget(self.exponential_fit_button)
        self.quadratic_fit_button = QPushButton("fit x²")
        self.quadratic_fit_button.setCheckable(True)
        self._make_button_compact(self.quadratic_fit_button)
        status_row.addWidget(self.quadratic_fit_button)
        self.copy_plot_button = QPushButton("Copy plot")
        self.copy_plot_button.setToolTip("Copy the current plot view to the clipboard")
        self.copy_plot_button.clicked.connect(self.copy_plot_to_clipboard)
        self._make_button_compact(self.copy_plot_button)
        status_row.addWidget(self.copy_plot_button)
        layout.addLayout(status_row)

        self.fit_controller = FitController(
            self.plot_widget,
            self.fit_view_box,
            self.plot_status_label,
            self.exponential_fit_button,
            self.quadratic_fit_button,
        )
        return container

    @staticmethod
    def _make_button_compact(button: QPushButton) -> None:
        text_width = button.fontMetrics().horizontalAdvance(button.text())
        button.setFixedWidth(text_width + 24)

    def copy_plot_to_clipboard(self) -> None:
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is None:
            return

        title_label = plot_item.titleLabel
        previous_title = title_label.text
        previous_options = dict(title_label.opts)
        title_was_visible = title_label.isVisible()
        record = self._plot_record
        if record is not None:
            plot_item.setTitle(
                html.escape(str(record.path)),
                color="k",
                size="9pt",
            )
            plot_item.layout.activate()
        try:
            exporter = ImageExporter(plot_item)
            exporter.export(copy=True)
        finally:
            if title_was_visible:
                plot_item.setTitle(previous_title, **previous_options)
            else:
                plot_item.setTitle(None)
            plot_item.layout.activate()

    # ── record loading ────────────────────────────────────────────────────────

    def _save_tag_bar(self) -> None:
        record = self._plot_record
        if record is None:
            return
        record.meta.update(
            plot_axes=self.tag_bar.axes,
            plot_fields=self.tag_bar.fields,
        )

    def reset_plot_state(self, message: str = "No data to plot.") -> None:
        self._plot_record = None
        self._plot_frame = None
        self.tag_bar.set_columns([], [], [])
        self._clear_plot(message)
        self._needs_refresh = False

    def mark_needs_refresh(self) -> None:
        self._needs_refresh = True

    def refresh_if_needed(self) -> None:
        if self._needs_refresh:
            self._needs_refresh = False
            self.refresh_plot()

    def update_controls(
        self,
        record: LogRecord,
        frame: pd.DataFrame | None,
    ) -> None:
        if frame is None or frame.empty or not len(frame.columns):
            self.tag_bar.set_columns([], [], [])
            return

        columns = list(frame.columns)
        plot_axes = record.plot_axes
        plot_fields = record.plot_fields

        self._suppress_updates = True
        self.tag_bar.set_columns(columns, plot_axes, plot_fields)
        self._suppress_updates = False

    def update_plot(
        self,
        record: LogRecord,
        frame: pd.DataFrame | None,
        *,
        defer: bool = False,
    ) -> None:
        self._plot_record = record
        self._plot_frame = frame

        if frame is None or frame.empty or not len(frame.columns):
            self._clear_plot("No columns available to plot.")
            self._needs_refresh = False
        elif defer:
            self._needs_refresh = True
        else:
            self.refresh_plot()

    # ── plotting ──────────────────────────────────────────────────────────────

    def refresh_plot(self) -> None:
        if self._suppress_updates:
            return

        axes = self.tag_bar.axes
        fields = self.tag_bar.fields

        if len(axes) == 1 and len(fields) >= 1:
            self._refresh_plot_1d(axes[0], fields)
        elif len(axes) >= 2 and len(fields) >= 1:
            self._refresh_plot_2d(axes[0], axes[1], fields[0])
        else:
            self._clear_plot("No data to plot.")

    def _clear_plot(
        self,
        message: str | None = None,
        *,
        hide_fit_buttons: bool = True,
    ) -> None:
        self.fit_controller.disable("Fit is available for a single 1D field.")
        if hide_fit_buttons:
            self.fit_controller.set_visible(False)
        self.plot_widget.clear()
        self._hide_color_bar()
        if message is not None:
            self.plot_status_label.setText(message)

    def _hide_color_bar(self) -> None:
        if self._color_bar is None:
            return
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.layout.removeItem(self._color_bar)
        self._color_bar.setParentItem(None)
        self._color_bar.deleteLater()
        self._color_bar = None

    def _show_color_bar(
        self,
        mesh: pg.PColorMeshItem,
        levels: tuple[float, float],
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
            )
            self._color_bar.axis.setPen("k")
            self._color_bar.axis.setTextPen("k")
            self._color_bar.axis.setWidth(38)
            self._color_bar.getAxis("left").setWidth(1)
            plot_item.layout.setColumnFixedWidth(4, 2)
            plot_item.layout.setColumnSpacing(4, 0)
        else:
            self._color_bar.setLevels(levels)
            self._color_bar.setImageItem(mesh)

    def _refresh_plot_1d(self, x_col: str, y_cols: list[str]) -> None:
        record = self._plot_record
        if record is None:
            self._clear_plot("No log selected.")
            return

        frame = self._plot_frame
        if frame is None or frame.empty:
            self._clear_plot("No data to plot.")
            return

        if x_col not in frame.columns:
            self._clear_plot(f"Column '{x_col}' not in data.")
            return

        x_values = pd.to_numeric(frame[x_col], errors="coerce")
        self._clear_plot(hide_fit_buttons=False)
        self.fit_controller.set_visible(True)

        COLORS = [
            "#1E90FF",
            "#FF6347",
            "#32CD32",
            "#FF8C00",
            "#9370DB",
            "#00CED1",
            "#FF1493",
            "#8B4513",
        ]
        plotted = 0
        single_series: tuple[np.ndarray, np.ndarray, str] | None = None
        for i, y_col in enumerate(y_cols):
            if y_col not in frame.columns:
                continue
            y_values = pd.to_numeric(frame[y_col], errors="coerce")
            df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
            if df.empty:
                continue
            color = COLORS[i % len(COLORS)]
            show_markers = len(df) <= 500
            pen = pg.mkPen(color=color, width=2)
            if show_markers:
                self.plot_widget.plot(
                    df["x"].values,
                    df["y"].values,
                    pen=pen,
                    name=y_col,
                    symbol="o",
                    symbolSize=6,
                    symbolPen=pg.mkPen(color=color),
                    symbolBrush=pg.mkBrush("#FFFFFF"),
                )
            else:
                self.plot_widget.plot(
                    df["x"].values, df["y"].values, pen=pen, name=y_col
                )
            if len(y_cols) == 1:
                single_series = (
                    df["x"].to_numpy(dtype=float),
                    df["y"].to_numpy(dtype=float),
                    y_col,
                    color,
                )
            plotted += 1

        if plotted == 0:
            self._clear_plot("No numeric data to plot.")
            return

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.enableAutoRange(enable=True)
            plot_item.autoRange()
        self.plot_widget.setLabel("bottom", x_col)
        self.plot_widget.setLabel("left", ", ".join(y_cols))
        self.plot_status_label.setText(f"1D plot: {x_col} vs {', '.join(y_cols[:3])}")
        if single_series is not None:
            self.fit_controller.set_series(*single_series)
        else:
            self.fit_controller.disable(
                "Keep exactly one field in the TagBar to enable fitting."
            )

    def _refresh_plot_2d(self, x_col: str, y_col: str, z_col: str) -> None:
        self.fit_controller.set_visible(False)
        record = self._plot_record
        if record is None:
            self._clear_plot("No log selected.")
            return

        frame = self._plot_frame
        if frame is None or frame.empty:
            self._clear_plot("No data to plot.")
            return

        for col in (x_col, y_col, z_col):
            if col not in frame.columns:
                self._clear_plot(f"Column '{col}' not in data.")
                return

        sub = frame[[x_col, y_col, z_col]]
        if all(np.issubdtype(t, np.number) for t in sub.dtypes):
            arr = sub.to_numpy(dtype=float, copy=False)
        else:
            arr = sub.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        mask = ~np.isnan(arr).any(axis=1)
        if not mask.any():
            self._clear_plot("No numeric data to plot.")
            return

        filtered = arr[mask]
        x_data, y_data, z_data = filtered[:, 0], filtered[:, 1], filtered[:, 2]
        mesh_data = build_plot_mesh(x_data, y_data, z_data)

        self.fit_controller.disable("Fit is only available for 1D plots.")
        self.plot_widget.clear()
        pcm = pg.PColorMeshItem(
            mesh_data.x_corners,
            mesh_data.y_corners,
            mesh_data.z_grid,
            colorMap=self.cmap,
            levels=mesh_data.levels,
        )
        self.plot_widget.addItem(pcm)
        self._show_color_bar(pcm, mesh_data.levels)
        self.plot_widget.setLabel("bottom", x_col)
        self.plot_widget.setLabel("left", y_col)

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.enableAutoRange(enable=True)
            plot_item.autoRange()

        self.plot_status_label.setText(
            f"2D plot: {mesh_data.point_count} points → "
            f"{mesh_data.x_column_count}×{mesh_data.max_y_count} mesh"
        )

    @functools.cached_property
    def cmap(self):
        cmap = pg.colormap.get("RdBu_r", source="matplotlib")
        if cmap is None:
            cmap = pg.colormap.get("CET-D1")
        return cmap
