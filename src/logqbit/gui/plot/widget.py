"""Plot widget for stored log records."""

from __future__ import annotations

import functools
import html
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord

from logqbit.catalog import resolve_plot_columns
from logqbit.gui.plot.cursor import CursorController, CursorSeries
from logqbit.gui.plot.fit import FitController, FitViewBox
from logqbit.gui.plot.mesh import build_plot_mesh


class TagBar(QWidget):
    """Assign columns to axes, fields, and ignored sections by dragging."""

    changed = Signal()
    save_clicked = Signal()

    _SEP = "|"
    _GRAY = QColor("#888888")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("axes | fields:"))

        self._list = QListWidget()
        self._list.setFlow(QListView.LeftToRight)
        self._list.setWrapping(False)
        self._list.setDragDropMode(QListWidget.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        row_height = self._list.fontMetrics().height() + 4
        self._list.setFixedHeight(row_height + self._list.frameWidth() * 2)
        self._list.installEventFilter(self)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.itemClicked.connect(lambda _: self._list.clearSelection())

        model = self._list.model()
        model.rowsInserted.connect(lambda *_: self._on_model_changed())
        model.rowsRemoved.connect(lambda *_: self._on_model_changed())
        model.rowsMoved.connect(lambda *_: self._on_model_changed())
        model.layoutChanged.connect(lambda: self._on_model_changed())

        self._loading = False
        layout.addWidget(self._list)

    def _on_model_changed(self) -> None:
        if self._loading:
            return
        self._update_item_colors()
        self.changed.emit()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction("Save", self.save_clicked.emit)
        menu.exec(self._list.mapToGlobal(pos))

    def eventFilter(self, obj, event):
        if obj is self._list and event.type() == QEvent.Wheel:
            bar = self._list.horizontalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y() // 2)
            return True
        return super().eventFilter(obj, event)

    def _make_sep(self) -> QListWidgetItem:
        item = QListWidgetItem(self._SEP)
        item.setForeground(self._GRAY)
        return item

    def _update_item_colors(self) -> None:
        separator_count = 0
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item.text() == self._SEP:
                separator_count += 1
            elif separator_count >= 2:
                item.setForeground(self._GRAY)
            else:
                item.setData(Qt.ForegroundRole, None)

    def set_columns(
        self,
        columns: Sequence[str],
        plot_axes: Sequence[str],
        plot_fields: Sequence[str],
    ) -> None:
        resolved = resolve_plot_columns(columns, plot_axes, plot_fields)

        self._loading = True
        try:
            self._list.clear()
            for name in resolved.axes:
                self._list.addItem(name)
            self._list.addItem(self._make_sep())
            for name in resolved.fields:
                self._list.addItem(name)
            self._list.addItem(self._make_sep())
            for name in resolved.ignored:
                item = QListWidgetItem(name)
                item.setForeground(self._GRAY)
                self._list.addItem(item)
        finally:
            self._loading = False

    def _split(self) -> tuple[list[str], list[str], list[str]]:
        sections: list[list[str]] = []
        current: list[str] = []
        for index in range(self._list.count()):
            text = self._list.item(index).text()
            if text == self._SEP:
                sections.append(current)
                current = []
            else:
                current.append(text)
        sections.append(current)
        while len(sections) < 3:
            sections.append([])
        return sections[0], sections[1], sections[2]

    @property
    def axes(self) -> list[str]:
        return self._split()[0]

    @property
    def fields(self) -> list[str]:
        return self._split()[1]


class PlotManager:
    def __init__(self, parent: QWidget | None = None):
        self._plot_record: LogRecord | None = None
        self._plot_frame: pd.DataFrame | None = None
        self._suppress_updates = False
        self._needs_refresh = False
        self._color_bar: pg.ColorBarItem | None = None
        self._mesh_item: pg.PColorMeshItem | None = None
        self._mesh_levels: tuple[float, float] | None = None
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

        plot_area = QWidget()
        self.plot_layout = QGridLayout(plot_area)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(4)
        self.plot_layout.addWidget(self.plot_widget, 0, 0)

        self.vertical_section_widget = pg.PlotWidget()
        self.vertical_section_widget.setBackground("w")
        self.vertical_section_widget.setMinimumWidth(120)
        self.horizontal_section_widget = pg.PlotWidget()
        self.horizontal_section_widget.setBackground("w")
        self.horizontal_section_widget.setMinimumHeight(100)
        self.section_readout = QLabel()
        self.section_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.section_readout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.section_readout.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.plot_layout.addWidget(self.vertical_section_widget, 0, 1)
        self.plot_layout.addWidget(self.horizontal_section_widget, 1, 0)
        self.plot_layout.addWidget(self.section_readout, 1, 1)
        self._set_section_layout_active(False)
        self.horizontal_section_widget.setXLink(self.plot_widget)
        self.vertical_section_widget.setYLink(self.plot_widget)
        for section_widget in (
            self.horizontal_section_widget,
            self.vertical_section_widget,
        ):
            plot_item = section_widget.getPlotItem()
            if plot_item is not None:
                for axis in ("left", "bottom", "top", "right"):
                    plot_item.getAxis(axis).setTextPen("k")
                    plot_item.getAxis(axis).enableAutoSIPrefix(False)

        layout.addWidget(plot_area, stretch=1)

        # Status row
        status_row = QHBoxLayout()
        self.plot_status_label = QLabel("No data to plot.")
        self.plot_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.plot_status_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        status_row.addWidget(self.plot_status_label, stretch=1)
        self.cursor_button = QPushButton("cursor")
        self.cursor_button.setCheckable(True)
        self._make_button_compact(self.cursor_button)
        status_row.addWidget(self.cursor_button)
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
        self.cursor_controller = CursorController(
            self.plot_widget,
            self.horizontal_section_widget,
            self.vertical_section_widget,
            self.section_readout,
            self.cursor_button,
            self._cursor_visibility_changed,
            self._cursor_activated,
        )
        self.exponential_fit_button.clicked.connect(self._fit_activated)
        self.quadratic_fit_button.clicked.connect(self._fit_activated)
        return container

    def _cursor_activated(self) -> None:
        self.fit_controller.cancel_selection()

    def _fit_activated(self, checked: bool) -> None:
        if checked:
            self.cursor_controller.disable()

    def _cursor_visibility_changed(self, active: bool) -> None:
        self._set_section_layout_active(active and self._mesh_item is not None)
        if self._mesh_item is None or self._mesh_levels is None:
            return
        if active:
            self._hide_color_bar()
        else:
            self._show_color_bar(self._mesh_item, self._mesh_levels)

    def _set_section_layout_active(self, active: bool) -> None:
        self.plot_layout.setColumnStretch(0, 5 if active else 1)
        self.plot_layout.setColumnStretch(1, 1 if active else 0)
        self.plot_layout.setRowStretch(0, 5 if active else 1)
        self.plot_layout.setRowStretch(1, 1 if active else 0)

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
        resolved = record.resolved_plot_columns

        self._suppress_updates = True
        self.tag_bar.set_columns(columns, resolved.axes, resolved.fields)
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
        self._mesh_item = None
        self._mesh_levels = None
        self.cursor_controller.clear()
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
        single_series: tuple[np.ndarray, np.ndarray, str, str] | None = None
        cursor_series: list[CursorSeries] = []
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
            cursor_series.append(
                CursorSeries(
                    df["x"].to_numpy(dtype=float),
                    df["y"].to_numpy(dtype=float),
                    y_col,
                )
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
        self.cursor_controller.configure_1d(cursor_series)
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

        self._mesh_item = None
        self._mesh_levels = None
        self.cursor_controller.clear()
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
        self._mesh_item = pcm
        self._mesh_levels = mesh_data.levels
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
        self.cursor_controller.configure_2d(mesh_data, x_col, y_col, z_col)

    @functools.cached_property
    def cmap(self):
        cmap = pg.colormap.get("RdBu_r", source="matplotlib")
        if cmap is None:
            cmap = pg.colormap.get("CET-D1")
        return cmap
