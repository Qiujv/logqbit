"""Plot widget for stored log records."""

from __future__ import annotations

import functools
import html
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QGridLayout,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord

from logqbit.catalog import resolve_plot_columns
from logqbit.gui.browser.plot.cursor import CursorController, CursorSeries
from logqbit.gui.browser.plot.fitting import FitController, FitViewBox
from logqbit.gui.browser.plot.grouping import iter_plot_groups
from logqbit.gui.browser.plot.mesh import build_plot_mesh

PLOT_EXPORT_SCALE = 2
PLOT_AUTO_RANGE_PADDING = 0.01
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


class TagBar(QWidget):
    """Assign columns to plot roles by dragging between sections."""

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
        self._columns: tuple[str, ...] = ()
        self._groupby_checks: dict[str, QCheckBox] = {}
        layout.addWidget(self._list)

        self.groupby_button = QToolButton()
        self.groupby_button.setText("group by")
        self.groupby_button.setPopupMode(QToolButton.InstantPopup)
        self.groupby_button.setEnabled(False)
        self.groupby_menu = QMenu(self.groupby_button)
        self._groupby_panel = QWidget(self.groupby_menu)
        self._groupby_layout = QVBoxLayout(self._groupby_panel)
        self._groupby_layout.setContentsMargins(6, 4, 6, 4)
        self._groupby_layout.setSpacing(2)
        self._groupby_action = QWidgetAction(self.groupby_menu)
        self._groupby_action.setDefaultWidget(self._groupby_panel)
        self.groupby_menu.addAction(self._groupby_action)
        self.groupby_button.setMenu(self.groupby_menu)
        layout.addWidget(self.groupby_button)

    def _on_model_changed(self) -> None:
        if self._loading:
            return
        axes, fields, _ = self._split()
        conflicts = set(axes + fields).intersection(self.groupby)
        if conflicts:
            self._loading = True
            try:
                for column in conflicts:
                    self._groupby_checks[column].setChecked(False)
                self._update_groupby_button()
            finally:
                self._loading = False
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
        plot_groupby: Sequence[str],
    ) -> None:
        resolved = resolve_plot_columns(
            columns,
            plot_axes,
            plot_fields,
            plot_groupby,
        )

        self._loading = True
        try:
            self._columns = tuple(dict.fromkeys(str(column) for column in columns))
            self._set_groupby_options(resolved.groupby)
            self._set_list_columns(resolved)
        finally:
            self._loading = False

    def _set_list_columns(self, resolved) -> None:
        self._list.clear()
        for name in resolved.axes:
            self._list.addItem(name)
        self._list.addItem(self._make_sep())
        for name in resolved.fields:
            self._list.addItem(name)
        self._list.addItem(self._make_sep())
        for name in (*resolved.groupby, *resolved.ignored):
            item = QListWidgetItem(name)
            item.setForeground(self._GRAY)
            self._list.addItem(item)

    def _set_groupby_options(self, selected: Sequence[str]) -> None:
        while self._groupby_layout.count():
            layout_item = self._groupby_layout.takeAt(0)
            if widget := layout_item.widget():
                widget.deleteLater()
        selected_set = set(selected)
        self._groupby_checks = {}
        for column in self._columns:
            checkbox = QCheckBox(column, self._groupby_panel)
            checkbox.setChecked(column in selected_set)
            checkbox.toggled.connect(self._on_groupby_toggled)
            self._groupby_layout.addWidget(checkbox)
            self._groupby_checks[column] = checkbox
        self._update_groupby_button()

    def _on_groupby_toggled(self) -> None:
        if self._loading:
            return
        axes, fields, _ = self._split()
        resolved = resolve_plot_columns(
            self._columns,
            axes,
            fields,
            self.groupby,
        )
        self._loading = True
        try:
            self._set_list_columns(resolved)
            self._update_groupby_button()
        finally:
            self._loading = False
        self.changed.emit()

    def _update_groupby_button(self) -> None:
        count = len(self.groupby)
        self.groupby_button.setText(f"group by ({count})" if count else "group by")
        self.groupby_button.setEnabled(bool(self._columns))
        self.groupby_button.setToolTip(
            ", ".join(self.groupby) if count else "Select columns used to group plots"
        )

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

    @property
    def groupby(self) -> list[str]:
        return [
            column
            for column in self._columns
            if self._groupby_checks[column].isChecked()
        ]


class PlotManager:
    def __init__(self, parent: QWidget | None = None):
        self._plot_record: LogRecord | None = None
        self._plot_frame: pd.DataFrame | None = None
        self._suppress_updates = False
        self._needs_refresh = False
        self._color_bar: pg.ColorBarItem | None = None
        self._legend: pg.LegendItem | None = None
        self._mesh_item: pg.PColorMeshItem | None = None
        self._mesh_items: list[pg.PColorMeshItem] = []
        self._mesh_levels: tuple[float, float] | None = None
        self._mesh_z_column: str | None = None
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
        self._setup_plot_context_menu(plot_item)
        self.fit_view_box.zoom_fit_requested.connect(self.zoom_fit_all)

        # plot_item.setDownsampling(auto=True, mode="subsample")
        plot_item.setContextMenuActionVisible("Points", False)
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
                plot_item.setContextMenuActionVisible("Points", False)
                for axis in ("left", "bottom", "top", "right"):
                    plot_item.getAxis(axis).setTextPen("k")
                    plot_item.getAxis(axis).enableAutoSIPrefix(False)

        layout.addWidget(plot_area, stretch=1)

        # Status row
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        self.plot_status_label = QLabel("No data to plot.")
        self.plot_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.plot_status_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        status_row.addWidget(self.plot_status_label, stretch=1)
        self.cursor_button = QToolButton()
        self.cursor_button.setText("cursor")
        self.cursor_button.setCheckable(True)
        status_row.addWidget(self.cursor_button)
        self.exponential_fit_button = QToolButton()
        self.exponential_fit_button.setText("fit exp")
        self.exponential_fit_button.setCheckable(True)
        status_row.addWidget(self.exponential_fit_button)
        self.quadratic_fit_button = QToolButton()
        self.quadratic_fit_button.setText("fit x²")
        self.quadratic_fit_button.setCheckable(True)
        status_row.addWidget(self.quadratic_fit_button)
        self.copy_plot_button = QToolButton()
        self.copy_plot_button.setText("copy")
        self.copy_plot_button.setToolTip(
            "copy the current plot view to the clipboard (Ctrl+C)"
        )
        self.copy_plot_button.clicked.connect(self.copy_plot_to_clipboard)
        status_row.addWidget(self.copy_plot_button)
        self.copy_plot_shortcut = QShortcut(QKeySequence.Copy, self.plot_widget)
        self.copy_plot_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.copy_plot_shortcut.activated.connect(self.copy_plot_to_clipboard)
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

    def _setup_plot_context_menu(self, plot_item: pg.PlotItem) -> None:
        menu = self.fit_view_box.getMenu(None)
        menu.addSeparator()
        self.save_plot_action = menu.addAction("Save plot")
        self.save_plot_action.triggered.connect(self.save_plot)

        self.log_x_action = menu.addAction("Log X")
        self.log_x_action.setCheckable(True)
        self.log_x_action.toggled.connect(plot_item.ctrl.logXCheck.setChecked)
        plot_item.ctrl.logXCheck.toggled.connect(self.log_x_action.setChecked)

        self.log_y_action = menu.addAction("Log Y")
        self.log_y_action.setCheckable(True)
        self.log_y_action.toggled.connect(plot_item.ctrl.logYCheck.setChecked)
        plot_item.ctrl.logYCheck.toggled.connect(self.log_y_action.setChecked)

    def zoom_fit_all(self) -> None:
        """Resize the plot view to include all plotted data."""
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.autoRange(padding=PLOT_AUTO_RANGE_PADDING)

    def _cursor_activated(self) -> None:
        self.fit_controller.cancel_selection()

    def _fit_activated(self, checked: bool) -> None:
        if checked:
            self.cursor_controller.disable()

    def _cursor_visibility_changed(self, active: bool) -> None:
        self._set_section_layout_active(active and self._mesh_item is not None)
        if (
            self._mesh_item is None
            or self._mesh_levels is None
            or self._mesh_z_column is None
        ):
            return
        if active:
            self._hide_color_bar()
        else:
            self._show_color_bar(
                self._mesh_item, self._mesh_levels, self._mesh_z_column
            )

    def _set_section_layout_active(self, active: bool) -> None:
        self.plot_layout.setColumnStretch(0, 5 if active else 1)
        self.plot_layout.setColumnStretch(1, 1 if active else 0)
        self.plot_layout.setRowStretch(0, 5 if active else 1)
        self.plot_layout.setRowStretch(1, 1 if active else 0)

    @contextmanager
    def _plot_with_record_title(self) -> Iterator[pg.PlotItem | None]:
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is None:
            yield None
            return

        title_label = plot_item.titleLabel
        previous_title = title_label.text
        previous_options = dict(title_label.opts)
        title_was_visible = title_label.isVisible()
        previous_text_width = title_label.item.textWidth()
        record = self._plot_record
        if record is not None:
            available_width = max(1.0, plot_item.vb.width())
            title_label.item.setTextWidth(available_width)
            title = html.escape(record.title)
            path = html.escape(str(record.path)).replace("/", "/<wbr>")
            plot_item.setTitle(
                f"{title} — {path}",
                color="k",
            )
            title_height = title_label.item.boundingRect().height()
            title_label.setMaximumHeight(title_height)
            plot_item.layout.setRowFixedHeight(0, title_height)
            plot_item.layout.activate()
        try:
            yield plot_item
        finally:
            if title_was_visible:
                plot_item.setTitle(previous_title, **previous_options)
                title_label.item.setTextWidth(previous_text_width)
                title_label.updateMin()
            else:
                title_label.item.setTextWidth(previous_text_width)
                title_label.setText("")
                plot_item.setTitle(None)
            plot_item.layout.activate()

    def _render_plot_image(self) -> QImage | None:
        with self._plot_with_record_title() as plot_item:
            if plot_item is None:
                return None
            return self._create_image_exporter(plot_item).export(toBytes=True)

    def copy_plot_to_clipboard(self) -> None:
        with self._plot_with_record_title() as plot_item:
            if plot_item is not None:
                self._create_image_exporter(plot_item).export(copy=True)

    def _create_image_exporter(self, plot_item) -> ImageExporter:
        exporter = ImageExporter(plot_item)
        parameters = exporter.parameters()
        parameters["width"] *= PLOT_EXPORT_SCALE
        if self._mesh_item is not None:
            parameters["antialias"] = False
        return exporter

    def save_plot(self) -> None:
        record = self._plot_record
        if record is None:
            return
        image = self._render_plot_image()
        if image is None:
            return

        output_path = self._next_plot_path(record.path)
        if image.save(str(output_path), "PNG"):
            self.plot_status_label.setText(f"Saved plot to {output_path}")
        else:
            self.plot_status_label.setText(f"Failed to save plot to {output_path}")

    @staticmethod
    def _next_plot_path(folder: Path) -> Path:
        output_path = folder / "plot.png"
        suffix = 1
        while output_path.exists():
            output_path = folder / f"plot-{suffix}.png"
            suffix += 1
        return output_path

    # ── record loading ────────────────────────────────────────────────────────

    def _save_tag_bar(self) -> None:
        record = self._plot_record
        if record is None:
            return
        record.meta.update(
            plot_axes=self.tag_bar.axes,
            plot_fields=self.tag_bar.fields,
            plot_groupby=self.tag_bar.groupby,
        )

    def reset_plot_state(self, message: str = "No data to plot.") -> None:
        self._plot_record = None
        self._plot_frame = None
        self.tag_bar.set_columns([], [], [], [])
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
            self.tag_bar.set_columns([], [], [], [])
            return

        columns = list(frame.columns)
        resolved = record.resolved_plot_columns

        self._suppress_updates = True
        self.tag_bar.set_columns(
            columns,
            resolved.axes,
            resolved.fields,
            resolved.groupby,
        )
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
        groupby = self.tag_bar.groupby

        if len(axes) == 1 and len(fields) >= 1:
            self._refresh_plot_1d(axes[0], fields, groupby)
        elif len(axes) >= 2 and len(fields) >= 1:
            self._refresh_plot_2d(axes[0], axes[1], fields[0], groupby)
        else:
            self._clear_plot("No data to plot.")

    def _clear_plot(
        self,
        message: str | None = None,
        *,
        hide_fit_buttons: bool = True,
    ) -> None:
        self._mesh_item = None
        self._mesh_items = []
        self._mesh_levels = None
        self._mesh_z_column = None
        self.cursor_controller.clear()
        self.fit_controller.disable("Fit is available for a single 1D field.")
        if hide_fit_buttons:
            self.fit_controller.set_visible(False)
        self.plot_widget.clear()
        self._clear_legend()
        self._hide_color_bar()
        if message is not None:
            self.plot_status_label.setText(message)

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
        if plot_item is not None and self._color_bar is not None:
            plot_item.layout.removeItem(self._color_bar)
        if self._color_bar is not None:
            self._color_bar.setParentItem(None)
            self._color_bar.deleteLater()
            self._color_bar = None

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
            self._color_bar.setLevels(levels)
            self._color_bar.setImageItem(mesh)
            self._color_bar.getAxis("left").setLabel(z_column)
        self._resize_color_bar(plot_item)

    def _resize_color_bar(self, plot_item: pg.PlotItem) -> None:
        if self._color_bar is None or plot_item.vb.height() <= 0:
            return
        self._color_bar.setMaximumHeight(
            round(plot_item.vb.height() * COLOR_BAR_HEIGHT_FACTOR)
        )
        plot_item.layout.setAlignment(self._color_bar, Qt.AlignVCenter)

    def _refresh_plot_1d(
        self,
        x_col: str,
        y_cols: list[str],
        groupby: Sequence[str] = (),
    ) -> None:
        record = self._plot_record
        if record is None:
            self._clear_plot("No log selected.")
            return

        frame = self._plot_frame
        if frame is None or frame.empty:
            self._clear_plot("No data to plot.")
            return

        for col in (x_col, *groupby):
            if col not in frame.columns:
                self._clear_plot(f"Column '{col}' not in data.")
                return

        self._clear_plot(hide_fit_buttons=False)
        self.fit_controller.set_visible(True)
        if groupby:
            self._show_legend()

        plotted = 0
        plotted_groups: set[str] = set()
        fit_series: tuple[np.ndarray, np.ndarray, str, str] | None = None
        cursor_series: list[CursorSeries] = []
        for plot_group in iter_plot_groups(frame, groupby):
            x_values = pd.to_numeric(plot_group.frame[x_col], errors="coerce")
            for y_col in y_cols:
                if y_col not in plot_group.frame.columns:
                    continue
                y_values = pd.to_numeric(plot_group.frame[y_col], errors="coerce")
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
                    self.plot_widget.plot(
                        df["x"].values,
                        df["y"].values,
                        pen=pen,
                        name=legend_name,
                        symbol="o",
                        symbolSize=6,
                        symbolPen=pg.mkPen(color=color),
                        symbolBrush=pg.mkBrush("#FFFFFF"),
                    )
                else:
                    self.plot_widget.plot(
                        df["x"].values,
                        df["y"].values,
                        pen=pen,
                        name=legend_name,
                    )
                if fit_series is None:
                    fit_series = (
                        df["x"].to_numpy(dtype=float),
                        df["y"].to_numpy(dtype=float),
                        series_name,
                        color,
                    )
                cursor_series.append(
                    CursorSeries(
                        df["x"].to_numpy(dtype=float),
                        df["y"].to_numpy(dtype=float),
                        series_name,
                    )
                )
                plotted += 1
                plotted_groups.add(plot_group.label)

        if plotted == 0:
            self._clear_plot("No numeric data to plot.")
            return

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.enableAutoRange(enable=True)
            plot_item.autoRange(padding=PLOT_AUTO_RANGE_PADDING)
        self.plot_widget.setLabel("bottom", x_col)
        self.plot_widget.setLabel("left", ", ".join(y_cols))
        status = f"1D plot: {x_col} vs {', '.join(y_cols[:3])}"
        if groupby:
            status += f" ({len(plotted_groups)} groups, {plotted} curves)"
        self.plot_status_label.setText(status)
        self.cursor_controller.configure_1d(cursor_series)
        if fit_series is not None:
            self.fit_controller.set_series(*fit_series)
        else:
            self.fit_controller.disable("No numeric field is available for fitting.")

    def _refresh_plot_2d(
        self,
        x_col: str,
        y_col: str,
        z_col: str,
        groupby: Sequence[str] = (),
    ) -> None:
        self.fit_controller.set_visible(False)
        record = self._plot_record
        if record is None:
            self._clear_plot("No log selected.")
            return

        frame = self._plot_frame
        if frame is None or frame.empty:
            self._clear_plot("No data to plot.")
            return

        for col in (x_col, y_col, z_col, *groupby):
            if col not in frame.columns:
                self._clear_plot(f"Column '{col}' not in data.")
                return

        mesh_groups = []
        for plot_group in iter_plot_groups(frame, groupby):
            sub = plot_group.frame[[x_col, y_col, z_col]]
            if all(np.issubdtype(t, np.number) for t in sub.dtypes):
                arr = sub.to_numpy(dtype=float, copy=False)
            else:
                arr = sub.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

            mask = ~np.isnan(arr).any(axis=1)
            if not mask.any():
                continue
            filtered = arr[mask]
            mesh = build_plot_mesh(filtered[:, 0], filtered[:, 1], filtered[:, 2])
            mesh_groups.append((plot_group.label, mesh))

        if not mesh_groups:
            self._clear_plot("No numeric data to plot.")
            return

        levels = (
            min(mesh.levels[0] for _, mesh in mesh_groups),
            max(mesh.levels[1] for _, mesh in mesh_groups),
        )

        self._mesh_item = None
        self._mesh_items = []
        self._mesh_levels = None
        self._mesh_z_column = None
        self.cursor_controller.clear()
        self.fit_controller.disable("Fit is only available for 1D plots.")
        self.plot_widget.clear()
        self._clear_legend()
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

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.enableAutoRange(enable=True)
            plot_item.autoRange(padding=PLOT_AUTO_RANGE_PADDING)

        total_points = sum(mesh.point_count for _, mesh, _ in rendered_groups)
        if groupby:
            status = (
                f"2D plot: {total_points} points in {len(rendered_groups)} group(s); "
                f"cursor uses {cursor_label} ({cursor_mesh.point_count} points)"
            )
        else:
            status = (
                f"2D plot: {cursor_mesh.point_count} points → "
                f"{cursor_mesh.x_column_count}×{cursor_mesh.max_y_count} mesh"
            )
        self.plot_status_label.setText(status)
        self.cursor_controller.configure_2d(
            cursor_mesh,
            x_col,
            y_col,
            z_col,
            group_label=cursor_label,
        )

    @functools.cached_property
    def cmap(self):
        cmap = pg.colormap.get("RdBu_r", source="matplotlib")
        if cmap is None:
            cmap = pg.colormap.get("CET-D1")
        return cmap
