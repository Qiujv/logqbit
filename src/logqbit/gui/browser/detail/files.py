"""File-manager and image helpers for record details."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QMimeData, Qt, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QSizePolicy, QWidget

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


class ScaledImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)

    def load_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = None
            self.setText(f"Failed to load {path.name}")
            return False
        self._pixmap = pixmap
        self.setText("")
        self._update_scaled_pixmap()
        return True

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override naming
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if not self._pixmap or self._pixmap.isNull():
            return
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        scaled = self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled)
