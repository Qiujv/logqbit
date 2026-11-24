"""Place plot labels within the visible ViewBox."""

from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6.QtCore import QPointF


class ViewBoxLabel:
    """Keep a text item near its data anchor or the closest view edge."""

    def __init__(self, view_box: pg.ViewBox, item: pg.TextItem) -> None:
        self.view_box = view_box
        self.item = item
        self.anchor: QPointF | None = None
        self.enabled = False
        view_box.sigRangeChanged.connect(self.update)
        view_box.sigResized.connect(self.update)

    def set_anchor(self, x: float, y: float) -> None:
        self.anchor = QPointF(x, y)
        self.item.setPos(self.anchor)
        self.enabled = True
        self.update()

    def hide(self) -> None:
        self.enabled = False
        self.item.hide()

    def dispose(self) -> None:
        self.view_box.sigRangeChanged.disconnect(self.update)
        self.view_box.sigResized.disconnect(self.update)

    def update(self, *_args: object) -> None:
        if not self.enabled or self.anchor is None:
            return
        point = self.view_box.mapViewToScene(self.anchor)
        bounds = self.view_box.mapRectToScene(self.view_box.boundingRect())
        if not all(math.isfinite(v) for v in (point.x(), point.y())):
            self.item.hide()
            return

        # TextItem ignores view scaling. Measure its actual scene rectangle after
        # positioning it, then translate in scene coordinates before mapping back.
        self.item.setPos(self.anchor)
        self.item.show()
        label = self.item.mapRectToScene(self.item.boundingRect())
        margin = 4.0
        gap = 6.0
        x = point.x() + gap
        y = point.y() - gap - label.height()
        if x + label.width() > bounds.right() - margin:
            x = point.x() - gap - label.width()
        if y < bounds.top() + margin:
            y = point.y() + gap
        x = max(bounds.left() + margin, min(x, bounds.right() - margin - label.width()))
        y = max(
            bounds.top() + margin, min(y, bounds.bottom() - margin - label.height())
        )
        position = self.view_box.mapSceneToView(
            point + QPointF(x - label.left(), y - label.top())
        )
        self.item.setPos(position)
