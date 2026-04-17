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
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
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


class TagBar(QWidget):
    """
    Single QListWidget holding axes / fields / ignored items in one row,
    separated by non-draggable " | " divider items.  Drag items across
    dividers to reassign them between sections.
    """

    changed = Signal()
    save_clicked = Signal()

    _SEP = "|"
    _GRAY = QColor("#888888")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4,0,4,0)
        layout.setSpacing(4)

        hint = QLabel("axes | fields:")
        layout.addWidget(hint)

        self._list = QListWidget()
        self._list.setFlow(QListView.LeftToRight)
        self._list.setWrapping(False)
        self._list.setDragDropMode(QListWidget.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        row_h = self._list.fontMetrics().height() + 4
        self._list.setFixedHeight(row_h + self._list.frameWidth() * 2)
        self._list.installEventFilter(self)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.itemClicked.connect(lambda _: self._list.clearSelection())

        m = self._list.model()
        m.rowsInserted.connect(lambda *_: self._on_model_changed())
        m.rowsRemoved.connect(lambda *_: self._on_model_changed())
        m.rowsMoved.connect(lambda *_: self._on_model_changed())
        m.layoutChanged.connect(lambda: self._on_model_changed())

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
        sep_count = 0
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.text() == self._SEP:
                sep_count += 1
            elif sep_count >= 2:
                item.setForeground(self._GRAY)
            else:
                item.setData(Qt.ForegroundRole, None)

    def set_columns(
        self,
        columns: list[str],
        plot_axes: list[str],
        plot_fields: list[str],
    ) -> None:
        col_set = set(columns)
        axes = [c for c in plot_axes if c in col_set]
        fields = [c for c in plot_fields if c in col_set]
        assigned = set(axes) | set(fields)
        ignored = [c for c in columns if c not in assigned]

        self._loading = True
        try:
            self._list.clear()
            for name in axes:
                self._list.addItem(name)
            self._list.addItem(self._make_sep())
            for name in fields:
                self._list.addItem(name)
            self._list.addItem(self._make_sep())
            for name in ignored:
                item = QListWidgetItem(name)
                item.setForeground(self._GRAY)
                self._list.addItem(item)
        finally:
            self._loading = False

    def _split(self) -> tuple[list[str], list[str], list[str]]:
        sections: list[list[str]] = []
        current: list[str] = []
        for i in range(self._list.count()):
            text = self._list.item(i).text()
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
        self._suppress_updates = False
        self._needs_refresh = False
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
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.useOpenGL(True)
        self.plot_widget.setMinimumHeight(220)

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.setDownsampling(auto=True, mode="subsample")
            for axis in ["left", "bottom", "top", "right"]:
                plot_item.getAxis(axis).setTextPen("k")
                plot_item.getAxis(axis).enableAutoSIPrefix(False)

        layout.addWidget(self.plot_widget, stretch=1)

        # Status label
        self.plot_status_label = QLabel("No data to plot.")
        self.plot_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.plot_status_label)

        return container

    # ── record loading ────────────────────────────────────────────────────────

    def _save_tag_bar(self) -> None:
        record = self._plot_record
        if record is None:
            return
        record.meta.plot_axes = self.tag_bar.axes
        record.meta.plot_fields = self.tag_bar.fields

    def reset_plot_state(self, message: str = "No data to plot.") -> None:
        self._plot_record = None
        self.tag_bar.set_columns([], [], [])
        self.plot_widget.clear()
        self.plot_status_label.setText(message)
        self._needs_refresh = False

    def mark_needs_refresh(self) -> None:
        self._needs_refresh = True

    def refresh_if_needed(self) -> None:
        if self._needs_refresh:
            self._needs_refresh = False
            self.refresh_plot()

    def update_plot_and_controls(
        self, record: LogRecord, defer_plot: bool = False
    ) -> None:
        self._plot_record = record
        frame = record.load_dataframe()

        if frame is None or frame.empty or not len(frame.columns):
            self.tag_bar.set_columns([], [], [])
            self.plot_widget.clear()
            self.plot_status_label.setText("No columns available to plot.")
            self._needs_refresh = False
            return

        columns = list(frame.columns)
        plot_axes = record.meta.plot_axes
        plot_fields = record.meta.plot_fields

        # If meta has no fields, auto-assign the first non-axes column
        if not plot_fields:
            axes_set = set(plot_axes)
            first_field = next((c for c in columns if c not in axes_set), None)
            if first_field:
                plot_fields = [first_field]

        self._suppress_updates = True
        self.tag_bar.set_columns(columns, plot_axes, plot_fields)
        self._suppress_updates = False

        if defer_plot:
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
            self.plot_widget.clear()
            self.plot_status_label.setText("No data to plot.")

    def _refresh_plot_1d(self, x_col: str, y_cols: list[str]) -> None:
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

        if x_col not in frame.columns:
            self.plot_widget.clear()
            self.plot_status_label.setText(f"Column '{x_col}' not in data.")
            return

        x_values = pd.to_numeric(frame[x_col], errors="coerce")
        self.plot_widget.clear()

        COLORS = ["#1E90FF", "#FF6347", "#32CD32", "#FF8C00", "#9370DB",
                  "#00CED1", "#FF1493", "#8B4513"]
        plotted = 0
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
                    df["x"].values, df["y"].values,
                    pen=pen, name=y_col,
                    symbol="o", symbolSize=6,
                    symbolPen=pg.mkPen(color=color),
                    symbolBrush=pg.mkBrush("#FFFFFF"),
                )
            else:
                self.plot_widget.plot(df["x"].values, df["y"].values, pen=pen, name=y_col)
            plotted += 1

        if plotted == 0:
            self.plot_widget.clear()
            self.plot_status_label.setText("No numeric data to plot.")
            return

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.enableAutoRange(enable=True)
            plot_item.autoRange()
        self.plot_widget.setLabel("bottom", x_col)
        self.plot_widget.setLabel("left", ", ".join(y_cols))
        self.plot_status_label.setText(f"1D plot: {x_col} vs {', '.join(y_cols[:3])}")

    def _refresh_plot_2d(self, x_col: str, y_col: str, z_col: str) -> None:
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

        for col in (x_col, y_col, z_col):
            if col not in frame.columns:
                self.plot_widget.clear()
                self.plot_status_label.setText(f"Column '{col}' not in data.")
                return

        sub = frame[[x_col, y_col, z_col]]
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
        x_data, y_data, z_data = filtered[:, 0], filtered[:, 1], filtered[:, 2]
        N = len(x_data)

        if N > 1 and not _is_lexsorted(x_data, y_data):
            sort_idx = np.lexsort((y_data, x_data))
            x_data = x_data[sort_idx]
            y_data = y_data[sort_idx]
            z_data = z_data[sort_idx]

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
        ref_y = y_data[col_starts[ref_col]: col_ends[ref_col]]
        typical_dy = float(np.median(np.diff(ref_y))) if len(ref_y) > 1 else 1.0

        last_y = y_data[col_ends - 1]
        prev_idx = np.maximum(col_ends - 2, col_starts)
        step_c = np.where(col_sizes > 1, last_y - y_data[prev_idx], typical_dy)
        top_y = last_y + step_c

        x_edges = np.empty(nx_col + 1)
        x_edges[:nx_col] = xu
        x_edges[-1] = xu[-1] + (xu[-1] - xu[-2] if nx_col > 1 else 1.0)
        x_edges_rect = np.repeat(x_edges, 2)[1:-1]
        x_corners = np.broadcast_to(x_edges_rect, (max_ny + 1, 2 * nx_col))

        z_grid, y_corners = _build_grids_rect(
            y_data, z_data, col_starts, col_sizes, max_ny, nx_col, top_y, step_c,
        )

        self.plot_widget.clear()
        pcm = pg.PColorMeshItem(x_corners, y_corners, z_grid, colorMap=self.cmap)
        self.plot_widget.addItem(pcm)
        self.plot_widget.setLabel("bottom", x_col)
        self.plot_widget.setLabel("left", y_col)

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
        return cmap



