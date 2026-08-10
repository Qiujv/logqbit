"""File-manager and image helpers for record details."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from collections.abc import Callable

from PySide6.QtCore import QEvent, QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

logger = logging.getLogger(__name__)


def open_in_file_manager(
    path: Path,
    *,
    select: bool = False,
    parent: QWidget | None = None,
) -> None:
    """Open a directory or reveal a path in the platform file manager."""
    try:
        if sys.platform.startswith("win"):
            command = (
                ["explorer", "/select,", str(path)]
                if select
                else ["explorer", str(path)]
            )
        elif sys.platform == "darwin":
            command = ["open", "-R", str(path)] if select else ["open", str(path)]
        else:
            target = path.parent if path.is_file() else path
            command = ["xdg-open", str(target)]
        subprocess.run(command, check=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to open explorer for %s: %s", path, exc)
        if parent is not None:
            QMessageBox.warning(
                parent,
                "Open in Explorer",
                f"Failed to open file browser: {exc}",
            )


def _copy_file_to_clipboard(path: Path) -> None:
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(path.resolve()))])
    QApplication.clipboard().setMimeData(mime_data)


class ZoomableImageView(QScrollArea):
    """Qt Image Viewer-style preview with wheel zooming and panning."""

    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0
    WHEEL_ZOOM_FACTOR = 1.1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._base_size = None
        self._zoom = 1.0
        self._drag_position: QPoint | None = None
        self._context_menu_callback: Callable[[QPoint], None] | None = None
        self._image_label = QLabel(self)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setBackgroundRole(QPalette.Base)
        self._image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_label.setScaledContents(True)
        self.setBackgroundRole(QPalette.Dark)
        self.setWidget(self._image_label)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self.viewport().installEventFilter(self)
        self._image_label.installEventFilter(self)

    def set_image_context_menu_callback(
        self, callback: Callable[[QPoint], None]
    ) -> None:
        self._context_menu_callback = callback
        for widget in (self.viewport(), self._image_label):
            widget.setContextMenuPolicy(Qt.CustomContextMenu)
            widget.customContextMenuRequested.connect(
                lambda position, source=widget: callback(source.mapTo(self, position))
            )

    def load_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = None
            self._image_label.setPixmap(QPixmap())
            self._image_label.setText(f"Failed to load {path.name}")
            self._image_label.resize(self.viewport().size())
            return False
        self._pixmap = pixmap
        self._image_label.setText("")
        self._image_label.setPixmap(pixmap)
        self._fit_image()
        return True

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override naming
        super().resizeEvent(event)
        if self._zoom == 1.0:
            self._fit_image()

    def eventFilter(self, source, event) -> bool:  # noqa: N802 - Qt override naming
        if event.type() == QEvent.Wheel:
            position = source.mapTo(self.viewport(), event.position().toPoint())
            self._zoom_from_wheel(event, position)
            return True
        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            self._drag_position = None
            self.viewport().unsetCursor()
            self._fit_image()
            return True
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._drag_position = source.mapTo(
                self.viewport(), event.position().toPoint()
            )
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return True
        if event.type() == QEvent.MouseMove and self._drag_position is not None:
            position = source.mapTo(self.viewport(), event.position().toPoint())
            delta = position - self._drag_position
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self._drag_position = position
            return True
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._drag_position is not None:
                self._drag_position = None
                self.viewport().unsetCursor()
                return True
        return super().eventFilter(source, event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override naming
        self._zoom_from_wheel(event, event.position().toPoint())

    def _zoom_from_wheel(self, event, position: QPoint) -> None:
        factor = (
            self.WHEEL_ZOOM_FACTOR
            if event.angleDelta().y() > 0
            else 1 / self.WHEEL_ZOOM_FACTOR
        )
        if self._apply_zoom(factor, position):
            event.accept()
        else:
            event.ignore()

    def _apply_zoom(self, factor: float, position: QPoint | None = None) -> bool:
        new_zoom = self._zoom * factor
        if (
            self._pixmap is None
            or self._base_size is None
            or not self.MIN_ZOOM <= new_zoom <= self.MAX_ZOOM
        ):
            return False
        if position is not None:
            image_position = self._image_label.mapFrom(self.viewport(), position)
            x_ratio = image_position.x() / max(self._image_label.width(), 1)
            y_ratio = image_position.y() / max(self._image_label.height(), 1)
        self._image_label.resize(self._base_size * new_zoom)
        self._zoom = new_zoom
        if position is not None:
            self.horizontalScrollBar().setValue(
                round(x_ratio * self._image_label.width() - position.x())
            )
            self.verticalScrollBar().setValue(
                round(y_ratio * self._image_label.height() - position.y())
            )
        return True

    def _fit_image(self) -> None:
        if self._pixmap is None or self.viewport().size().isEmpty():
            return
        self._base_size = self._pixmap.size().scaled(
            self.viewport().size(), Qt.KeepAspectRatio
        )
        self._image_label.resize(self._base_size)
        self._zoom = 1.0
