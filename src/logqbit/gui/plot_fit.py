"""Interactive fitting support for one-dimensional plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QGraphicsRectItem, QLabel, QPushButton
from scipy.optimize import curve_fit

FitKind = Literal["exponential", "quadratic"]


@dataclass(frozen=True)
class FitResult:
    """A fitted curve and the concise result shown to the user."""

    x: np.ndarray
    y: np.ndarray
    value: float
    label: str


def _prepare_fit_data(
    x: np.ndarray,
    y: np.ndarray,
    *,
    minimum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, x-sorted fitting data with enough distinct x values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) < minimum_points:
        raise ValueError(f"select at least {minimum_points} data points")

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    if np.unique(x).size < 3:
        raise ValueError("select points at three or more distinct x positions")
    return x, y


def fit_exponential(x: np.ndarray, y: np.ndarray) -> FitResult:
    """Fit ``offset + amplitude * exp(-(x - x0) / tau)``."""
    x, y = _prepare_fit_data(x, y, minimum_points=4)
    x0 = float(x[0])
    dx = x - x0
    span = float(dx[-1])
    if span <= 0:
        raise ValueError("selected x values have no range")

    tail_count = max(1, len(y) // 5)
    offset0 = float(np.mean(y[-tail_count:]))
    amplitude0 = float(y[0] - offset0)
    if np.isclose(amplitude0, 0):
        amplitude0 = float(np.ptp(y)) or 1.0
    tau0 = span / 3

    def model(
        x_offset: np.ndarray,
        amplitude: float,
        tau: float,
        offset: float,
    ) -> np.ndarray:
        return offset + amplitude * np.exp(-x_offset / tau)

    lower_tau = max(span * 1e-9, np.finfo(float).tiny)
    upper_tau = span * 1e9
    try:
        parameters, _ = curve_fit(
            model,
            dx,
            y,
            p0=(amplitude0, tau0, offset0),
            bounds=((-np.inf, lower_tau, -np.inf), (np.inf, upper_tau, np.inf)),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise ValueError("exponential fit did not converge") from exc

    amplitude, tau, offset = parameters
    x_fit = np.linspace(x[0], x[-1], 300)
    y_fit = model(x_fit - x0, amplitude, tau, offset)
    return FitResult(x=x_fit, y=y_fit, value=float(tau), label=f"τ = {tau:.6g}")


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> FitResult:
    """Fit a centered quadratic and report its extremum position."""
    x, y = _prepare_fit_data(x, y, minimum_points=3)
    x_ref = float(np.mean(x))
    dx = x - x_ref
    span = float(x[-1] - x[0])
    design = np.column_stack((dx**2, dx, np.ones_like(dx)))
    a, b, c = np.linalg.lstsq(design, y, rcond=None)[0]

    y_scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), 1.0)
    if abs(a) * span**2 <= np.finfo(float).eps ** 0.5 * y_scale:
        raise ValueError("selected points have too little curvature")

    extremum_x = x_ref - b / (2 * a)
    label = f"x = {extremum_x:.6g}"

    x_fit = np.linspace(x[0], x[-1], 300)
    fit_dx = x_fit - x_ref
    y_fit = a * fit_dx**2 + b * fit_dx + c
    return FitResult(x=x_fit, y=y_fit, value=float(extremum_x), label=label)


class FitViewBox(pg.ViewBox):
    """ViewBox that draws a fit-selection rectangle only while armed."""

    selection_finished = Signal(str, object)
    selection_canceled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._fit_kind: FitKind | None = None

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
        if self._fit_kind is not None and event.button() == Qt.RightButton:
            event.accept()
            self.cancel_fit_selection(notify=True)
            return
        super().mouseClickEvent(event)


class FitController:
    """Own the 1D fit controls, selection state, and plot overlays."""

    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        view_box: FitViewBox,
        status_label: QLabel,
        exponential_button: QPushButton,
        quadratic_button: QPushButton,
    ) -> None:
        self._plot_widget = plot_widget
        self._view_box = view_box
        self._status_label = status_label
        self._buttons = {
            "exponential": exponential_button,
            "quadratic": quadratic_button,
        }
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._field = ""
        self._color = "#1E90FF"
        self._overlays: list[object] = []

        exponential_button.clicked.connect(
            lambda checked: self._toggle("exponential", checked)
        )
        quadratic_button.clicked.connect(
            lambda checked: self._toggle("quadratic", checked)
        )
        view_box.selection_finished.connect(self._fit_selection)
        view_box.selection_canceled.connect(self._selection_canceled)

        self._escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), plot_widget)
        self._escape_shortcut.activated.connect(
            lambda: self.cancel_selection(show_status=True)
        )
        self.disable("Fit is available for a single 1D field.")

    def set_series(
        self,
        x: np.ndarray,
        y: np.ndarray,
        field: str,
        color: str,
    ) -> None:
        self.clear_overlays()
        self.cancel_selection()
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._field = field
        self._color = color
        for button in self._buttons.values():
            button.setEnabled(True)
            button.setToolTip(
                "Click, then drag a rectangle around the data points to fit"
            )

    def disable(self, reason: str) -> None:
        self.clear_overlays()
        self.cancel_selection()
        self._x = None
        self._y = None
        self._field = ""
        for button in self._buttons.values():
            button.setEnabled(False)
            button.setToolTip(reason)

    def set_visible(self, visible: bool) -> None:
        if not visible:
            self.cancel_selection()
        for button in self._buttons.values():
            button.setVisible(visible)

    def cancel_selection(self, *, show_status: bool = False) -> None:
        was_active = any(button.isChecked() for button in self._buttons.values())
        self._view_box.cancel_fit_selection()
        for button in self._buttons.values():
            button.setChecked(False)
        if show_status and was_active:
            self._status_label.setText("Fit selection canceled.")

    def clear_overlays(self) -> None:
        plot_item = self._plot_widget.getPlotItem()
        if plot_item is not None:
            for item in self._overlays:
                plot_item.removeItem(item)
        self._overlays.clear()

    def _toggle(self, kind: FitKind, checked: bool) -> None:
        if not checked:
            self.cancel_selection(show_status=True)
            return

        self.clear_overlays()
        for other_kind, button in self._buttons.items():
            if other_kind != kind:
                button.setChecked(False)
        self._view_box.arm(kind)
        self._status_label.setText(
            f"{kind.capitalize()} fit: drag a rectangle around points in "
            f"'{self._field}'. Right-click or press Esc to cancel."
        )

    def _selection_canceled(self) -> None:
        self.cancel_selection(show_status=True)

    def _fit_selection(self, kind: FitKind, rect: QRectF) -> None:
        self.clear_overlays()
        if self._x is None or self._y is None:
            return

        selected = (
            (self._x >= rect.left())
            & (self._x <= rect.right())
            & (self._y >= rect.top())
            & (self._y <= rect.bottom())
        )
        x = self._x[selected]
        y = self._y[selected]
        self._add_selection_overlays(rect, x, y)

        try:
            result = (
                fit_exponential(x, y) if kind == "exponential" else fit_quadratic(x, y)
            )
        except ValueError as exc:
            self._status_label.setText(f"{kind.capitalize()} fit failed: {exc}.")
            return

        fit_curve = pg.PlotDataItem(
            result.x,
            result.y,
            pen=pg.mkPen("#111111", width=2.5, style=Qt.DashLine),
        )
        result_text = pg.TextItem(
            result.label,
            color="#111111",
            fill=pg.mkBrush(255, 255, 255, 220),
            border=pg.mkPen("#555555"),
            anchor=(0, 0),
        )
        view_bottom, view_top = self._view_box.viewRange()[1]
        below_space = rect.top() - view_bottom
        above_space = view_top - rect.bottom()
        if above_space >= below_space:
            pixel_height = abs(self._view_box.viewPixelSize()[1])
            text_y = rect.bottom() + result_text.boundingRect().height() * pixel_height
        else:
            text_y = rect.top()
        result_text.setPos(rect.left(), text_y)
        plot_item = self._plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.addItem(fit_curve)
            plot_item.addItem(result_text, ignoreBounds=True)
            self._overlays.extend((fit_curve, result_text))
        self._status_label.setText(
            f"{kind.capitalize()} fit using {len(x)} points: {result.label}"
        )

    def _add_selection_overlays(
        self,
        rect: QRectF,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        plot_item = self._plot_widget.getPlotItem()
        if plot_item is None:
            return

        rectangle = QGraphicsRectItem(rect)
        rectangle.setPen(pg.mkPen("#E67E22", width=1.5))
        rectangle.setBrush(pg.mkBrush(230, 126, 34, 35))
        rectangle.setAcceptedMouseButtons(Qt.NoButton)
        selected_points = pg.ScatterPlotItem(
            x,
            y,
            symbol="o",
            size=6,
            pen=pg.mkPen(self._color),
            brush=pg.mkBrush(self._color),
        )
        plot_item.addItem(rectangle, ignoreBounds=True)
        plot_item.addItem(selected_points)
        self._overlays.extend((rectangle, selected_points))
