"""Pure numerical models used by interactive plot fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
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
