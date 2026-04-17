"""Standalone plotter for log records."""

from __future__ import annotations

import functools
import logging
from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from numba import njit, prange
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .browser import LogRecord

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


@njit(cache=True)
def _is_lexsorted(x: np.ndarray, y: np.ndarray) -> bool:
    """Return True if x is globally ascending and y is monotonic within each x-column.

    Unlike a strict lex-sort, y may be either ascending or descending within a
    column; the only requirement is that it doesn't reverse direction mid-column.
    """
    N = len(x)
    prev_x = x[0]
    prev_y = y[0]
    y_dir = 0  # 0=undetermined, 1=ascending, -1=descending; reset per column
    for i in range(1, N):
        xi = x[i]
        if xi < prev_x:
            return False
        if xi == prev_x:
            yi = y[i]
            if yi > prev_y:
                if y_dir == -1:
                    return False
                y_dir = 1
            elif yi < prev_y:
                if y_dir == 1:
                    return False
                y_dir = -1
            prev_y = yi
        else:
            y_dir = 0  # new column — reset direction
            prev_x = xi
            prev_y = y[i]
    return True


@njit(parallel=True, cache=True)
def _build_grids_rect(ys, zs, col_starts, col_sizes, max_ny, nx_col, top_y, step_c):
    """Build rect-separated z/y arrays for PColorMeshItem using parallel columns.

    Each data column occupies an even index (c*2) in the output; odd indices are
    NaN separator columns so adjacent columns share no edges.  Left and right y
    corners of each cell are identical → perfectly horizontal top/bottom edges.

    Returns:
        z_final : shape (max_ny, 2*nx_col-1)   — NaN at odd (separator) columns
        y_final : shape (max_ny+1, 2*nx_col)   — paired left/right y per column
    """
    z_final = np.full((max_ny, 2 * nx_col - 1), np.nan)
    y_final = np.empty((max_ny + 1, 2 * nx_col))

    for c in prange(nx_col):
        s = col_starts[c]
        n = col_sizes[c]
        c2 = c + c  # even column index

        for r in range(n):
            z_final[r, c2] = zs[s + r]
            yv = ys[s + r]
            y_final[r, c2] = yv
            y_final[r, c2 + 1] = yv

        top = top_y[c]
        y_final[n, c2] = top
        y_final[n, c2 + 1] = top

        sc = step_c[c]
        for r in range(n + 1, max_ny + 1):
            val = top + (r - n) * sc
            y_final[r, c2] = val
            y_final[r, c2 + 1] = val

    return z_final, y_final


class PlotManager:
    MARKER_AUTO_THRESHOLD = 500  # Auto-enable markers when point count <= this value

    def __init__(self, parent: QWidget | None = None):
        self._plot_record: LogRecord | None = None
        self._suppress_updates = False
        self._marker_auto = True
        self._needs_refresh = False

        # Current selections
        self._x_column: str = ""
        self._y_column: str = ""
        self._z_column: str = ""

        self.widget = self._create_widget(parent)

    def _create_widget(self, parent: QWidget | None = None) -> QWidget:
        """Create and return the plot tab widget."""
        plot_tab = QWidget(parent)
        plot_layout = QVBoxLayout(plot_tab)
        plot_layout.setContentsMargins(4, 4, 4, 4)

        # Plot controls
        plot_controls = QHBoxLayout()
        plot_controls.setContentsMargins(0, 0, 0, 0)

        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItem("1D", "1d")
        self.plot_mode_combo.addItem("2D", "2d")
        self.plot_mode_combo.setCurrentIndex(0)

        # Use QToolButton with popup menu for column selection
        self.plot_x_button = QToolButton()
        self.plot_y_button = QToolButton()
        self.plot_z_button = QToolButton()
        self.plot_x_button.setText("(none)")
        self.plot_y_button.setText("(none)")
        self.plot_z_button.setText("(none)")
        self.plot_x_button.setEnabled(False)
        self.plot_y_button.setEnabled(False)
        self.plot_z_button.setEnabled(False)

        # Menus for buttons
        self._x_menu = QMenu()
        self._y_menu = QMenu()
        self._z_menu = QMenu()
        self.plot_x_button.setMenu(self._x_menu)
        self.plot_y_button.setMenu(self._y_menu)
        self.plot_z_button.setMenu(self._z_menu)
        self.plot_x_button.setPopupMode(QToolButton.InstantPopup)
        self.plot_y_button.setPopupMode(QToolButton.InstantPopup)
        self.plot_z_button.setPopupMode(QToolButton.InstantPopup)

        plot_controls.addWidget(QLabel("Mode:"))
        plot_controls.addWidget(self.plot_mode_combo)
        plot_controls.addSpacing(6)
        plot_controls.addWidget(QLabel("X:"))
        plot_controls.addWidget(self.plot_x_button)
        plot_controls.addSpacing(6)
        plot_controls.addWidget(QLabel("Y:"))
        plot_controls.addWidget(self.plot_y_button)
        plot_controls.addSpacing(6)
        self.plot_z_label = QLabel("Z:")
        plot_controls.addWidget(self.plot_z_label)
        plot_controls.addWidget(self.plot_z_button)
        plot_controls.addSpacing(6)

        self.plot_marker_checkbox = QCheckBox("Show markers")
        self.plot_marker_checkbox.setEnabled(False)
        self.plot_marker_checkbox.setChecked(False)
        plot_controls.addWidget(self.plot_marker_checkbox)
        plot_controls.addStretch(1)
        plot_layout.addLayout(plot_controls)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        # self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.useOpenGL(True)  # Required for pcolormesh for large data.
        self.plot_widget.setMinimumHeight(220)

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.setDownsampling(auto=True, mode="subsample")
            for axis in ["left", "bottom", "top", "right"]:
                plot_item.getAxis(axis).setTextPen("k")
                plot_item.getAxis(axis).enableAutoSIPrefix(False)

        plot_layout.addWidget(self.plot_widget, stretch=1)

        # Status label
        self.plot_status_label = QLabel("No data to plot.")
        self.plot_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        plot_layout.addWidget(self.plot_status_label)

        # Connect signals
        self.plot_mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.plot_marker_checkbox.toggled.connect(self.on_marker_toggled)

        return plot_tab

    def reset_plot_state(self, message: str = "No data to plot.") -> None:
        """Reset plot to empty state."""
        self._plot_record = None
        self._suppress_updates = True
        self._x_column = ""
        self._y_column = ""
        self._z_column = ""
        self._x_menu.clear()
        self._y_menu.clear()
        self._z_menu.clear()
        self.plot_x_button.setText("(none)")
        self.plot_y_button.setText("(none)")
        self.plot_z_button.setText("(none)")
        self._suppress_updates = False
        self.plot_x_button.setEnabled(False)
        self.plot_y_button.setEnabled(False)
        self.plot_z_button.setEnabled(False)
        self._reset_marker_checkbox(enabled=False)
        self.plot_widget.clear()
        self.plot_status_label.setText(message)
        self._needs_refresh = False

    def _reset_marker_checkbox(self, enabled: bool) -> None:
        """Reset marker checkbox to unchecked state."""
        self._marker_auto = True
        self.plot_marker_checkbox.blockSignals(True)
        self.plot_marker_checkbox.setChecked(False)
        self.plot_marker_checkbox.blockSignals(False)
        self.plot_marker_checkbox.setEnabled(enabled)

    def mark_needs_refresh(self) -> None:
        self._needs_refresh = True

    def refresh_if_needed(self) -> None:
        if self._needs_refresh:
            self._needs_refresh = False
            self.refresh_plot()

    def update_plot_and_controls(
        self, record: LogRecord, defer_plot: bool = False
    ) -> None:
        same_record = record is self._plot_record
        previous_x = self._x_column if same_record else ""
        previous_y = self._y_column if same_record else ""
        previous_z = self._z_column if same_record else ""
        previous_mode = self.plot_mode_combo.currentData() if same_record else None
        self._plot_record = record

        frame = record.load_dataframe()

        if frame is None or frame.empty or not len(frame.columns):
            self._suppress_updates = True
            self._x_column = ""
            self._y_column = ""
            self._z_column = ""
            self._x_menu.clear()
            self._y_menu.clear()
            self._z_menu.clear()
            self.plot_x_button.setText("(none)")
            self.plot_y_button.setText("(none)")
            self.plot_z_button.setText("(none)")
            self._suppress_updates = False
            self.plot_x_button.setEnabled(False)
            self.plot_y_button.setEnabled(False)
            self.plot_z_button.setEnabled(False)
            self._reset_marker_checkbox(enabled=False)
            self.plot_widget.clear()
            self.plot_status_label.setText("No columns available to plot.")
            self._needs_refresh = False
            return

        columns = frame.columns
        plot_axes = [col for col in record.meta.plot_axes if col in columns]
        plot_zs = [col for col in columns if col not in plot_axes]

        # Populate menus
        self._suppress_updates = True
        self._x_menu.clear()
        self._y_menu.clear()
        self._z_menu.clear()

        for name in columns:
            x_action = self._x_menu.addAction(name)
            x_action.triggered.connect(
                lambda checked=False, col=name: self._on_x_selected(col)
            )
            y_action = self._y_menu.addAction(name)
            y_action.triggered.connect(
                lambda checked=False, col=name: self._on_y_selected(col)
            )
            z_action = self._z_menu.addAction(name)
            z_action.triggered.connect(
                lambda checked=False, col=name: self._on_z_selected(col)
            )

        auto_mode = "2d" if len(plot_axes) >= 2 else "1d"
        if previous_mode:
            auto_mode = previous_mode  # Keep user's choice

        if auto_mode == "2d":
            if len(plot_axes) >= 2:
                x_default = plot_axes[0]
                y_default = plot_axes[1]
                z_default = plot_zs[0] if plot_zs else columns[0]
            elif len(plot_axes) == 1:
                x_default = plot_axes[0]
                y_default = plot_zs[0] if plot_zs else columns[0]
                z_default = plot_zs[1] if len(plot_zs) > 1 else columns[0]
            else:
                x_default = columns[0]
                y_default = columns[1] if len(columns) > 1 else columns[0]
                z_default = columns[2] if len(columns) > 2 else columns[0]
        else:  # 1D mode
            if plot_axes:
                x_default = plot_axes[0]
                y_default = plot_zs[0] if plot_zs else columns[0]
            else:
                x_default = columns[0]
                y_default = columns[1] if len(columns) > 1 else columns[0]
            z_default = columns[0]  # Not used in 1D mode

        # Restore previous selections if available
        if previous_x and previous_x in columns:
            x_default = previous_x
        if previous_y and previous_y in columns:
            y_default = previous_y
        if previous_z and previous_z in columns:
            z_default = previous_z

        self._x_column = x_default
        self._y_column = y_default
        self._z_column = z_default
        self.plot_x_button.setText(x_default)
        self.plot_y_button.setText(y_default)
        self.plot_z_button.setText(z_default)
        self.plot_mode_combo.setCurrentIndex(0 if auto_mode == "1d" else 1)

        self._suppress_updates = False
        self.plot_x_button.setEnabled(True)
        self.plot_y_button.setEnabled(True)
        self.plot_z_button.setEnabled(auto_mode == "2d")
        self.plot_z_button.setVisible(auto_mode == "2d")
        self.plot_z_label.setVisible(auto_mode == "2d")
        self._reset_marker_checkbox(enabled=auto_mode == "1d")
        self.plot_marker_checkbox.setVisible(auto_mode == "1d")

        if defer_plot:
            self._needs_refresh = True
        else:
            self.refresh_plot()

    def _on_x_selected(self, column: str) -> None:
        if self._suppress_updates:
            return
        self._x_column = column
        self.plot_x_button.setText(column)
        self.refresh_plot()

    def _on_y_selected(self, column: str) -> None:
        if self._suppress_updates:
            return
        self._y_column = column
        self.plot_y_button.setText(column)
        self.refresh_plot()

    def _on_z_selected(self, column: str) -> None:
        if self._suppress_updates:
            return
        self._z_column = column
        self.plot_z_button.setText(column)
        self.refresh_plot()

    def on_mode_changed(self, _index: int = -1) -> None:
        mode = self.plot_mode_combo.currentData()
        if mode == "1d":
            self.plot_x_button.setEnabled(len(self._x_menu.actions()) > 0)
            self.plot_y_button.setEnabled(len(self._y_menu.actions()) > 0)
            self.plot_z_button.setEnabled(False)
            self.plot_z_button.setVisible(False)
            self.plot_z_label.setVisible(False)
            self.plot_marker_checkbox.setEnabled(len(self._x_menu.actions()) > 0)
            self.plot_marker_checkbox.setVisible(True)
            self.refresh_plot()
        else:  # 2d mode
            self.plot_x_button.setEnabled(len(self._x_menu.actions()) > 0)
            self.plot_y_button.setEnabled(len(self._y_menu.actions()) > 0)
            self.plot_z_button.setEnabled(len(self._z_menu.actions()) > 0)
            self.plot_z_button.setVisible(True)
            self.plot_z_label.setVisible(True)
            self._reset_marker_checkbox(enabled=False)
            self.plot_marker_checkbox.setVisible(False)
            self.refresh_plot()

    def on_marker_toggled(self, _checked: bool) -> None:
        if self._suppress_updates or self.plot_mode_combo.currentData() != "1d":
            return
        self._marker_auto = False
        self.refresh_plot()

    def refresh_plot(self) -> None:
        if self._suppress_updates:
            return

        mode = self.plot_mode_combo.currentData()

        if mode == "2d":
            self._refresh_plot_2d()
        else:
            self._refresh_plot_1d()

    def _refresh_plot_1d(self) -> None:
        def _disable_markers() -> None:
            self._reset_marker_checkbox(enabled=False)

        record = self._plot_record
        if record is None:
            self.plot_widget.clear()
            self.plot_status_label.setText("No log selected.")
            _disable_markers()
            return
        frame = record.load_dataframe()
        if frame is None or frame.empty:
            self.plot_widget.clear()
            self.plot_status_label.setText("No data to plot.")
            _disable_markers()
            return

        x_column = self._x_column
        y_column = self._y_column
        if not x_column or not y_column:
            self.plot_widget.clear()
            self.plot_status_label.setText("Select X and Y columns to plot.")
            _disable_markers()
            return
        if x_column not in frame.columns or y_column not in frame.columns:
            self.plot_widget.clear()
            self.plot_status_label.setText("Selected columns not in data.")
            _disable_markers()
            return
        x_values = pd.to_numeric(frame[x_column], errors="coerce")
        y_values = pd.to_numeric(frame[y_column], errors="coerce")
        if x_values.isna().all():
            self.plot_widget.clear()
            self.plot_status_label.setText(f"Column '{x_column}' is not numeric.")
            _disable_markers()
            return
        if y_values.isna().all():
            self.plot_widget.clear()
            self.plot_status_label.setText(f"Column '{y_column}' is not numeric.")
            _disable_markers()
            return
        df = pd.DataFrame({"x": x_values, "y": y_values})
        df.dropna(axis="index", how="any", inplace=True)
        if df.empty:
            self.plot_widget.clear()
            self.plot_status_label.setText(
                "No valid numeric rows after filtering NaN values."
            )
            _disable_markers()
            return
        show_markers = False
        if self._marker_auto:
            default_checked = len(df) <= self.MARKER_AUTO_THRESHOLD
            if self.plot_marker_checkbox.isChecked() != default_checked:
                self.plot_marker_checkbox.blockSignals(True)
                self.plot_marker_checkbox.setChecked(default_checked)
                self.plot_marker_checkbox.blockSignals(False)
            show_markers = default_checked
        else:
            show_markers = self.plot_marker_checkbox.isChecked()
        self.plot_marker_checkbox.setEnabled(True)
        self.plot_widget.clear()
        plot_pen = pg.mkPen(color="#1E90FF", width=2)
        if show_markers:
            self.plot_widget.plot(
                df["x"].values,
                df["y"].values,
                pen=plot_pen,
                symbol="o",
                symbolSize=6,
                symbolPen=pg.mkPen(color="#1E90FF"),
                symbolBrush=pg.mkBrush("#FFFFFF"),
            )
        else:
            self.plot_widget.plot(df["x"].values, df["y"].values, pen=plot_pen)
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.enableAutoRange(enable=True)
            plot_item.autoRange()
        self.plot_widget.setLabel("bottom", x_column)
        self.plot_widget.setLabel("left", y_column)
        self.plot_status_label.setText(f"Plotted {len(df)} rows.")

    def _refresh_plot_2d(self) -> None:
        """Refresh 2D plot using PColorMeshItem (handles uniform and non-uniform grids)."""
        record = self._plot_record
        if record is None:
            self.plot_widget.clear()
            self.plot_status_label.setText("No log selected.")
            return

        frame = record.load_dataframe()
        if frame is None or frame.empty:
            self.plot_widget.clear()
            self.plot_status_label.setText("No data to plot.")
            return

        x_column = self._x_column
        y_column = self._y_column
        z_column = self._z_column

        if not x_column or not y_column or not z_column:
            self.plot_widget.clear()
            self.plot_status_label.setText("Select X, Y, and Z columns to plot.")
            return

        if (
            x_column not in frame.columns
            or y_column not in frame.columns
            or z_column not in frame.columns
        ):
            self.plot_widget.clear()
            self.plot_status_label.setText("Selected columns not in data.")
            return

        sub = frame[[x_column, y_column, z_column]]
        if all(np.issubdtype(t, np.number) for t in sub.dtypes):
            arr = sub.to_numpy(dtype=float, copy=False)
        else:
            arr = sub.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        mask = ~np.isnan(arr).any(axis=1)
        if not mask.any():
            self.plot_widget.clear()
            self.plot_status_label.setText("No numeric data to plot.")
            return

        filtered = arr[mask]
        x_data = filtered[:, 0]
        y_data = filtered[:, 1]
        z_data = filtered[:, 2]
        N = len(x_data)

        # Sort only when necessary (early-exit check via numba)
        if N > 1 and not _is_lexsorted(x_data, y_data):
            sort_idx = np.lexsort((y_data, x_data))
            x_data = x_data[sort_idx]
            y_data = y_data[sort_idx]
            z_data = z_data[sort_idx]

        # Column grouping
        change = np.empty(N, dtype=np.bool_)
        change[0] = True
        change[1:] = x_data[1:] != x_data[:-1]
        xu = x_data[change]
        col_starts = np.flatnonzero(change)
        nx_col = len(xu)
        col_ends = np.append(col_starts[1:], N)
        col_sizes = col_ends - col_starts
        max_ny = int(col_sizes.max())

        ref_col = int(np.argmax(col_sizes))
        ref_y = y_data[col_starts[ref_col] : col_ends[ref_col]]
        typical_dy = float(np.median(np.diff(ref_y))) if len(ref_y) > 1 else 1.0

        # Per-column extrapolation parameters
        last_y = y_data[col_ends - 1]
        prev_idx = np.maximum(col_ends - 2, col_starts)
        step_c = np.where(col_sizes > 1, last_y - y_data[prev_idx], typical_dy)
        top_y = last_y + step_c

        # x corners — rect format: each column gets a left+right edge pair
        x_edges = np.empty(nx_col + 1)
        x_edges[:nx_col] = xu
        x_edges[-1] = xu[-1] + (xu[-1] - xu[-2] if nx_col > 1 else 1.0)
        x_edges_rect = np.repeat(x_edges, 2)[1:-1]  # shape (2*nx_col,)
        x_corners = np.broadcast_to(x_edges_rect, (max_ny + 1, 2 * nx_col))

        z_grid, y_corners = _build_grids_rect(
            y_data, z_data, col_starts, col_sizes, max_ny, nx_col,
            top_y, step_c,
        )

        self.plot_widget.clear()
        pcm = pg.PColorMeshItem(x_corners, y_corners, z_grid, colorMap=self.cmap)
        self.plot_widget.addItem(pcm)
        self.plot_widget.setLabel("bottom", x_column)
        self.plot_widget.setLabel("left", y_column)

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.enableAutoRange(enable=True)
            plot_item.autoRange()

        self.plot_status_label.setText(f"2D plot: {N} points → {nx_col}×{max_ny} mesh")

    @functools.cached_property
    def cmap(self):
        cmap = pg.colormap.get("RdBu_r", source="matplotlib")
        if cmap is None:
            cmap = pg.colormap.get("CET-D1")
            # cmap = pg.colormap.get("CET-L12")  # Blues
        return cmap


class PlotWindow(QMainWindow):
    """Standalone window showing a PlotManager for a single log record."""

    def __init__(self, record: LogRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if not WINDOW_ICON.isNull():
            self.setWindowIcon(WINDOW_ICON)
        self.resize(900, 600)
        self._record = record
        self._update_title()

        self.plot_manager = PlotManager(parent=self)
        self.setCentralWidget(self.plot_manager.widget)
        self.plot_manager.update_plot_and_controls(record)

    def _update_title(self) -> None:
        record = self._record
        title = record.meta.title or "(untitled)"
        self.setWindowTitle(f"#{record.log_id} {title} - Plot")

    def load_record(self, record: LogRecord) -> None:
        """Switch to a different log record."""
        self._record = record
        self._update_title()
        self.plot_manager.update_plot_and_controls(record)
