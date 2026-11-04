"""Plot widget for stored log records."""

from __future__ import annotations

import html
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6.QtCore import Qt, Signal, QRectF, QTimer
from PySide6.QtGui import (
    QActionGroup,
    QImage,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QGridLayout,
    QFileDialog,
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord

from logqbit.gui.browser.plot.cursor import CursorController
from logqbit.gui.browser.plot.fitting import FitController, FitKind
from logqbit.gui.browser.plot.controls import TagBar
from logqbit.gui.browser.plot.rendering import (
    PlotRenderer,
    RenderResult,
    PlotGroup,
    iter_plot_groups,
    _plot_values,
    MAX_PLOT_GROUPS,
)

PLOT_EXPORT_SCALE = 2
# Larger values make text and other fixed-size elements larger in copied plots.
PLOT_COPY_CONTENT_SCALE = 1.4
PLOT_AUTO_RANGE_PADDING = 0.01
MARKER_SIZES = {"Small": 4, "Medium": 6, "Large": 8}


class PlotStatusLabel(QLabel):
    """Keep the full status accessible when the row clips its text."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setToolTip(text)

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        self.setToolTip(text)


class PlotViewBox(pg.ViewBox):
    """ViewBox that draws a fit-selection rectangle only while armed."""

    selection_finished = Signal(str, object)
    selection_canceled = Signal()
    zoom_fit_requested = Signal()
    cursor_click_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._fit_kind: FitKind | None = None
        self._pending_cursor_click = None
        self._cursor_click_timer = QTimer(self)
        self._cursor_click_timer.setSingleShot(True)
        self._cursor_click_timer.timeout.connect(self._emit_cursor_click)

    def _emit_cursor_click(self) -> None:
        if self._pending_cursor_click is not None:
            self.cursor_click_requested.emit(self._pending_cursor_click)
            self._pending_cursor_click = None

    def cancel_pending_cursor_click(self) -> None:
        self._cursor_click_timer.stop()
        self._pending_cursor_click = None

    def arm(self, kind: FitKind) -> None:
        self._fit_kind = kind
        self.setCursor(Qt.CrossCursor)

    def cancel_fit_selection(self, *, notify: bool = False) -> None:
        was_armed = self._fit_kind is not None
        self._fit_kind = None
        self.rbScaleBox.hide()
        self.unsetCursor()
        if notify and was_armed:
            self.selection_canceled.emit()

    def mouseDragEvent(self, event, axis=None) -> None:
        if self._fit_kind is None or event.button() != Qt.LeftButton:
            super().mouseDragEvent(event, axis=axis)
            return

        event.accept()
        if event.isFinish():
            self.rbScaleBox.hide()
            local_rect = QRectF(event.buttonDownPos(), event.pos()).normalized()
            data_rect = self.childGroup.mapRectFromParent(local_rect).normalized()
            kind = self._fit_kind
            self.selection_finished.emit(kind, data_rect)
        else:
            self.updateScaleBox(event.buttonDownPos(), event.pos())

    def mouseClickEvent(self, event) -> None:
        if event.double() and event.button() == Qt.LeftButton:
            self.cancel_pending_cursor_click()
            event.accept()
            self.zoom_fit_requested.emit()
            return
        if event.button() == Qt.LeftButton:
            self._pending_cursor_click = event.pos()
            self._cursor_click_timer.start(QApplication.doubleClickInterval())
            event.accept()
            return
        if self._fit_kind is not None and event.button() == Qt.RightButton:
            event.accept()
            self.cancel_fit_selection(notify=True)
            return
        super().mouseClickEvent(event)


class PlotView(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._plot_record: LogRecord | None = None
        self._plot_frame: pd.DataFrame | None = None
        self._comparison = False
        self._snapshot_frames = []
        self._snapshot_errors = []
        self._suppress_updates = False
        self._needs_refresh = False
        self._user_controls_view = False
        self._create_widget()

    def _create_widget(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Tag bar
        self.tag_bar = TagBar()
        self.tag_bar.changed.connect(self._on_plot_roles_changed)
        self.tag_bar.save_clicked.connect(self._save_tag_bar)
        layout.addWidget(self.tag_bar)

        # Plot widget
        self.view_box = PlotViewBox()
        plot_item = pg.PlotItem(viewBox=self.view_box)
        self.plot_widget = pg.PlotWidget(plotItem=plot_item)
        self.renderer = PlotRenderer(self.plot_widget)
        self.plot_widget.setBackground("w")
        self.plot_widget.useOpenGL(True)
        self.plot_widget.setMinimumHeight(220)
        self._setup_plot_context_menu(plot_item)
        self.view_box.zoom_fit_requested.connect(self.zoom_fit_all)
        self.view_box.sigRangeChangedManually.connect(
            lambda _mask: self._mark_user_controls_view()
        )

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
        self.plot_status_label = PlotStatusLabel("No data to plot.")
        self.plot_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.plot_status_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        status_row.addWidget(self.plot_status_label, stretch=1)
        self.snapshot_refresh_button = QToolButton()
        self.snapshot_refresh_button.setText("Refresh")
        self.snapshot_refresh_button.setToolTip(
            "Refresh the multi-record plot; automatic updates are paused."
        )
        self.snapshot_refresh_button.hide()
        status_row.addWidget(self.snapshot_refresh_button)
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
            self.view_box,
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
            self.view_box.cancel_pending_cursor_click,
        )
        self.view_box.cursor_click_requested.connect(
            self.cursor_controller.move_to_click
        )
        self.exponential_fit_button.clicked.connect(self._fit_activated)
        self.quadratic_fit_button.clicked.connect(self._fit_activated)

    def _setup_plot_context_menu(self, plot_item: pg.PlotItem) -> None:
        menu = self.view_box.getMenu(None)
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

        self.marker_size_menu = menu.addMenu("Marker size")
        self.marker_size_actions = QActionGroup(self.marker_size_menu)
        self.marker_size_actions.setExclusive(True)
        for label, size in MARKER_SIZES.items():
            action = self.marker_size_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(size == self.renderer.marker_size)
            action.triggered.connect(
                lambda checked, size=size: checked and self._set_marker_size(size)
            )
            self.marker_size_actions.addAction(action)
        self.marker_size_menu.menuAction().setVisible(False)

    def _set_marker_size(self, size: int) -> None:
        self.renderer.marker_size = size
        for item in self.plot_widget.getPlotItem().listDataItems():
            if item.opts["symbol"] is not None:
                item.setSymbolSize(size)

    def zoom_fit_all(self) -> None:
        """Resize the plot view to include all plotted data."""
        self._mark_user_controls_view()
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.autoRange(padding=PLOT_AUTO_RANGE_PADDING)

    def _cursor_activated(self) -> None:
        self._mark_user_controls_view()
        self.fit_controller.cancel_selection()

    def _fit_activated(self, checked: bool) -> None:
        if checked:
            self._mark_user_controls_view()
            self.cursor_controller.disable()

    def _mark_user_controls_view(self) -> None:
        self._user_controls_view = True

    def _on_plot_roles_changed(self) -> None:
        self._user_controls_view = False
        self.refresh_plot()

    def _cursor_visibility_changed(self, active: bool) -> None:
        self._set_section_layout_active(active and self.renderer.has_mesh)
        self.renderer.set_cursor_visible(active)

    def _set_section_layout_active(self, active: bool) -> None:
        self.plot_layout.setColumnStretch(0, 5 if active else 1)
        self.plot_layout.setColumnStretch(1, 1 if active else 0)
        self.plot_layout.setRowStretch(0, 5 if active else 1)
        self.plot_layout.setRowStretch(1, 1 if active else 0)

    @contextmanager
    def _plot_with_record_title(
        self, *, font_scale: float = 1.0
    ) -> Iterator[pg.PlotItem | None]:
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
            path = html.escape(record.path.as_posix()).replace("/", "/<wbr>")
            if self._comparison:
                title = "Comparison"
                path = html.escape(
                    ", ".join(f"#{item.log_id}" for item, _ in self._snapshot_frames)
                )
            axis_font = plot_item.getAxis("bottom").label.document().defaultFont()
            plot_item.setTitle(
                f"{title} — {path}",
                color="k",
                size=f"{axis_font.pointSizeF() * font_scale:g}pt",
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
        plot_item = self.plot_widget.getPlotItem()
        geometry = plot_item.geometry()
        try:
            plot_item.resize(
                geometry.width() / PLOT_COPY_CONTENT_SCALE,
                geometry.height() / PLOT_COPY_CONTENT_SCALE,
            )
            plot_item.layout.activate()
            with self._plot_with_record_title(
                font_scale=1 / PLOT_COPY_CONTENT_SCALE
            ) as plot_item:
                if plot_item is not None:
                    self._create_image_exporter(
                        plot_item, scale=PLOT_EXPORT_SCALE * PLOT_COPY_CONTENT_SCALE
                    ).export(copy=True)
        finally:
            plot_item.setGeometry(geometry)
            plot_item.layout.activate()

    def _create_image_exporter(
        self, plot_item, *, scale: float = PLOT_EXPORT_SCALE
    ) -> ImageExporter:
        exporter = ImageExporter(plot_item)
        parameters = exporter.parameters()
        parameters["width"] = int(parameters["width"] * scale)
        if self.renderer.has_mesh:
            parameters["antialias"] = False
        return exporter

    def save_plot(self) -> None:
        record = self._plot_record
        if record is None:
            return
        image = self._render_plot_image()
        if image is None:
            return

        if self._comparison:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save comparison plot", "plot.png", "PNG images (*.png)"
            )
            if not filename:
                return
            output_path = Path(filename)
        else:
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

    def _save_tag_bar(self) -> None:
        record = self._plot_record
        if record is None or self._comparison:
            return
        record.meta.update(
            plot_axes=self.tag_bar.axes,
            plot_fields=self.tag_bar.fields,
            plot_groupby=self.tag_bar.groupby,
        )

    def set_comparison(self, enabled: bool) -> None:
        self._comparison = enabled
        self.tag_bar.save_enabled = not enabled
        self.snapshot_refresh_button.setVisible(enabled)
        if not enabled:
            self._snapshot_frames = []
            self._snapshot_errors = []

    def set_snapshot(self, frames, errors) -> None:
        self._snapshot_frames = frames
        self._snapshot_errors = errors
        self._user_controls_view = False
        self.refresh_plot()

    def reset_plot_state(self, message: str = "No data to plot.") -> None:
        self._plot_record = None
        self._plot_frame = None
        self._user_controls_view = False
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
        self._user_controls_view = False
        if frame is None or frame.empty or not len(frame.columns):
            self.tag_bar.set_columns([], [], [], [])
            return

        columns = list(frame.columns)
        resolved = record.resolved_plot_columns

        self._suppress_updates = True
        try:
            self.tag_bar.set_columns(
                columns, resolved.axes, resolved.fields, resolved.groupby
            )
        finally:
            self._suppress_updates = False

    def update_plot(
        self,
        record: LogRecord,
        frame: pd.DataFrame | None,
        *,
        defer: bool = False,
    ) -> None:
        if self._plot_record is None or self._plot_record.path != record.path:
            self._user_controls_view = False
        self._plot_record = record
        self._plot_frame = frame

        if frame is None or frame.empty or not len(frame.columns):
            self._clear_plot("No columns available to plot.")
            self._needs_refresh = False
        elif defer:
            self._needs_refresh = True
        else:
            self.refresh_plot()

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

    def _clear_plot(self, message: str | None = None) -> None:
        self.cursor_controller.clear()
        self.fit_controller.disable("Fit is available for a single 1D field.")
        self.fit_controller.set_visible(False)
        self.marker_size_menu.menuAction().setVisible(False)
        self.renderer.clear()
        if message is not None:
            self.plot_status_label.setText(message)

    def _view_range_if_preserved(self) -> tuple[list[float], list[float]] | None:
        if not self._user_controls_view:
            return None
        x_range, y_range = self.view_box.viewRange()
        return list(x_range), list(y_range)

    def _restore_or_autorange(
        self,
        view_range: tuple[list[float], list[float]] | None,
    ) -> None:
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is None:
            return
        if view_range is None:
            plot_item.enableAutoRange(enable=True)
            plot_item.autoRange(padding=PLOT_AUTO_RANGE_PADDING)
            return
        plot_item.enableAutoRange(enable=False)
        self.view_box.setRange(
            xRange=view_range[0],
            yRange=view_range[1],
            padding=0,
        )

    def _refresh_plot_1d(
        self, x_col: str, y_cols: list[str], groupby: Sequence[str] = ()
    ) -> None:
        self._render([x_col], y_cols, groupby)

    def _refresh_plot_2d(
        self, x_col: str, y_col: str, z_col: str, groupby: Sequence[str] = ()
    ) -> None:
        self._render([x_col, y_col], [z_col], groupby)

    def _render(
        self, axes: Sequence[str], fields: Sequence[str], groupby: Sequence[str]
    ) -> None:
        """Replace data, preserving interaction overlays only after user activity."""
        if self._plot_record is None:
            self._clear_plot("No log selected.")
            return
        if self._comparison:
            self._render_comparison(axes, fields, groupby)
            return
        if self._plot_frame is None or self._plot_frame.empty:
            self._clear_plot("No data to plot.")
            return
        view_range = self._view_range_if_preserved()
        preserve = view_range is not None
        if not preserve:
            self.cursor_controller.clear()
            self.fit_controller.disable("Fit is available for a single 1D field.")
        with self.renderer.scene_update_transaction():
            if len(axes) == 1:
                result = self.renderer.render_1d(
                    self._plot_frame, axes[0], fields, groupby
                )
            else:
                result = self.renderer.render_2d(
                    self._plot_frame, axes[0], axes[1], fields[0], groupby
                )
            if result.mode == "empty":
                self._clear_plot(result.status)
                return
            self._restore_or_autorange(view_range)
        self.log_x_action.setEnabled(not result.x_is_datetime)
        self.log_y_action.setEnabled(not result.y_is_datetime)
        self.plot_status_label.setText(result.status)
        self.marker_size_menu.menuAction().setVisible(result.mode == "1d")
        self.fit_controller.set_visible(result.mode == "1d")
        self._configure_interactions(result, preserve=preserve)

    def _render_comparison(self, axes, fields, groupby) -> None:
        groups = []
        errors = list(self._snapshot_errors)
        reference_modes = None
        field_mode = None
        reference_frame = None
        for record, frame in self._snapshot_frames:
            required = list(axes[:2]) + (list(fields[:1]) if len(axes) >= 2 else [])
            missing = [name for name in required if name not in frame.columns]
            missing_fields = [name for name in fields if name not in frame.columns]
            if missing or (len(axes) == 1 and len(missing_fields) == len(fields)):
                errors.append(
                    f"#{record.log_id}: missing {', '.join(dict.fromkeys(missing + missing_fields))}"
                )
                continue
            if missing_fields:
                errors.append(f"#{record.log_id}: missing {', '.join(missing_fields)}")
            axis_modes = tuple(
                pd.api.types.is_datetime64_any_dtype(frame[name]) for name in axes[:2]
            )
            if reference_modes is None:
                reference_modes = axis_modes
                reference_frame = frame
            elif axis_modes != reference_modes:
                errors.append(f"#{record.log_id}: incompatible axis types")
                continue
            if len(axes) == 1:
                incompatible = []
                for name in fields:
                    if name not in frame.columns:
                        continue
                    mode = pd.api.types.is_datetime64_any_dtype(frame[name])
                    if field_mode is None:
                        field_mode = mode
                    elif mode != field_mode:
                        incompatible.append(name)
                if incompatible:
                    errors.append(
                        f"#{record.log_id}: incompatible field types {', '.join(incompatible)}"
                    )
                    frame = frame.drop(columns=incompatible)
                if not any(name in frame.columns for name in fields):
                    continue
            coordinate = _plot_values(frame[axes[0]])[0].notna()
            if len(axes) >= 2:
                valid = (
                    coordinate
                    & _plot_values(frame[axes[1]])[0].notna()
                    & _plot_values(frame[fields[0]])[0].notna()
                )
            else:
                valid = pd.Series(False, index=frame.index)
                for name in fields:
                    if name in frame.columns:
                        valid |= coordinate & _plot_values(frame[name])[0].notna()
            if not valid.any():
                errors.append(f"#{record.log_id}: no valid points")
                continue
            available_groupby = [name for name in groupby if name in frame.columns]
            absent = [name for name in groupby if name not in frame.columns]
            if absent:
                errors.append(
                    f"#{record.log_id}: missing group columns {', '.join(absent)}"
                )
            for group in iter_plot_groups(frame, available_groupby):
                labels = [f"#{record.log_id}"]
                if group.label:
                    labels.append(group.label)
                labels.extend(f"{name}=<missing column>" for name in absent)
                groups.append(PlotGroup(", ".join(labels), group.frame))
        if not groups or reference_frame is None:
            self._clear_plot("; ".join(errors) or "No data to plot.")
            return
        self.cursor_controller.clear()
        self.fit_controller.disable("Fit is available for a single 1D field.")
        with self.renderer.scene_update_transaction():
            if len(axes) == 1:
                result = self.renderer.render_1d(
                    reference_frame, axes[0], fields, groups=groups
                )
            else:
                result = self.renderer.render_2d(
                    reference_frame, axes[0], axes[1], fields[0], groups=groups
                )
            if result.mode == "empty":
                self._clear_plot("; ".join([result.status, *errors]))
                return
            self._restore_or_autorange(None)
        self.log_x_action.setEnabled(not result.x_is_datetime)
        self.log_y_action.setEnabled(not result.y_is_datetime)
        if len(groups) > MAX_PLOT_GROUPS:
            omitted = sorted(
                {group.label.split(",", 1)[0] for group in groups[:-MAX_PLOT_GROUPS]}
            )
            errors.append("omitted groups from " + ", ".join(omitted))
        if result.fit_series is not None:
            errors.insert(0, f"fit uses {result.fit_series[2]}")
        status = "; ".join([result.status, *errors])
        self.plot_status_label.setText(status)
        self.marker_size_menu.menuAction().setVisible(result.mode == "1d")
        self.fit_controller.set_visible(result.mode == "1d")
        self._configure_interactions(result, preserve=False)

    def _configure_interactions(self, result: RenderResult, *, preserve: bool) -> None:
        if result.mode == "1d":
            self.cursor_controller.configure_1d(
                result.series, preserve_overlay=preserve
            )
            if result.fit_series is not None and not result.y_is_datetime:
                self.fit_controller.set_series(
                    *result.fit_series,
                    x_is_datetime=result.x_is_datetime,
                    preserve_overlays=preserve,
                )
            else:
                self.fit_controller.disable("Fit is unavailable for datetime fields.")
        else:
            self.cursor_controller.configure_2d(
                result.mesh,
                *result.axis_names,
                group_label=result.group_label,
                preserve_overlay=preserve,
            )
