"""File-manager and image helpers for record details."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from collections.abc import Callable

import pyqtgraph as pg
from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QMenu,
    QGraphicsPixmapItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from send2trash import send2trash

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


def _format_file_size(size: int) -> str:
    """Return a compact, binary file-size label."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


class ImageViewBox(pg.ViewBox):
    """ViewBox that fits the full image on a left-button double click."""

    zoom_fit_requested = Signal()

    def mouseClickEvent(self, event) -> None:
        if event.double() and event.button() == Qt.LeftButton:
            event.accept()
            self.zoom_fit_requested.emit()
            return
        super().mouseClickEvent(event)


class ZoomableImageView(pg.PlotWidget):
    """Image preview using pyqtgraph's native pan and zoom interactions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._view_box = ImageViewBox(enableMenu=False)
        super().__init__(parent=parent, viewBox=self._view_box, enableMenu=False)
        self._pixmap: QPixmap | None = None
        self._image_item = QGraphicsPixmapItem()
        self._view_box.addItem(self._image_item)
        self._view_box.setAspectLocked(True)
        self._view_box.setDefaultPadding(0)
        self._view_box.invertY(True)
        self.setMinimumSize(200, 200)
        self.getPlotItem().hideAxis("left")
        self.getPlotItem().hideAxis("bottom")
        self._view_box.zoom_fit_requested.connect(self.zoom_fit)

    def set_image_context_menu_callback(
        self, callback: Callable[[QPoint], None]
    ) -> None:
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(callback)

    def load_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = None
            self._image_item.setPixmap(QPixmap())
            return False
        self._pixmap = pixmap
        self._image_item.setPixmap(pixmap)
        self.zoom_fit()
        return True

    def zoom_fit(self) -> None:
        if self._pixmap is None:
            return
        self._view_box.autoRange(padding=0)


class ImageTab(QWidget):
    """Image preview tab with image-specific file actions."""

    def __init__(
        self,
        image_path: Path,
        *,
        file_open_callback: Callable[[Path], None],
        refresh_callback: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._file_open_callback = file_open_callback
        self._refresh_callback = refresh_callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._image_view = ZoomableImageView()
        self._image_view.setToolTip(str(image_path))
        self._image_view.set_image_context_menu_callback(
            lambda position: self._open_context_menu(self._image_view, position)
        )
        self._image_view.load_image(image_path)
        layout.addWidget(self._image_view, stretch=1)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda position: self._open_context_menu(self, position)
        )

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_label = QLabel(
            "File size: "
            f"{_format_file_size(image_path.stat().st_size)}. "
            "Double-click to zoom to fit.",
            self,
        )
        status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_row.addWidget(status_label, stretch=1)
        copy_button = QPushButton("copy")
        copy_button.setToolTip("Copy the image file to the clipboard (Ctrl+C)")
        copy_button.clicked.connect(self._copy_file)
        status_row.addWidget(copy_button)
        layout.addLayout(status_row)

        copy_shortcut = QShortcut(QKeySequence.Copy, self)
        copy_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        copy_shortcut.activated.connect(self._copy_file)

    def _copy_file(self, _checked: bool = False) -> None:
        _copy_file_to_clipboard(self._image_path)

    def _open_context_menu(self, widget: QWidget, position: QPoint) -> None:
        menu = QMenu(widget)
        view_all_action = menu.addAction("View all")
        menu.addSeparator()
        rename_action = menu.addAction("Rename File...")
        move_to_trash_action = menu.addAction("Move to Recycle Bin...")
        open_action = menu.addAction("Open File")
        chosen = menu.exec(widget.mapToGlobal(position))
        if chosen == view_all_action:
            self._image_view.zoom_fit()
        elif chosen == rename_action:
            self._rename_file()
        elif chosen == move_to_trash_action:
            self._move_to_trash()
        elif chosen == open_action:
            self._file_open_callback(self._image_path)

    def _rename_file(self) -> None:
        new_name, accepted = QInputDialog.getText(
            self,
            "Rename Image File",
            "New file name:",
            text=self._image_path.stem,
        )
        new_name = new_name.strip()
        if not accepted or new_name == self._image_path.stem:
            return
        if (
            not new_name
            or new_name in {".", ".."}
            or Path(new_name).name != new_name
            or "/" in new_name
            or "\\" in new_name
        ):
            QMessageBox.warning(
                self,
                "Rename Image File",
                "Enter a valid file name without a path.",
            )
            return

        new_path = self._image_path.with_name(f"{new_name}{self._image_path.suffix}")
        if new_path.exists():
            QMessageBox.warning(
                self,
                "Rename Image File",
                f"A file named {new_name!r} already exists.",
            )
            return
        try:
            self._image_path.rename(new_path)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Rename Image File",
                f"Failed to rename file:\n{exc}",
            )
            return
        self._refresh_callback()

    def _move_to_trash(self) -> None:
        reply = QMessageBox.question(
            self,
            "Move Image File to Recycle Bin",
            f"Move this file to the Recycle Bin?\n\n{self._image_path}\n\n"
            "This operation can be undone from the Recycle Bin.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            send2trash(str(self._image_path))
        except Exception as exc:  # pragma: no cover - platform integration
            QMessageBox.warning(
                self,
                "Move Image File to Recycle Bin",
                f"Failed to move file to Recycle Bin:\n{exc}",
            )
            return
        self._refresh_callback()
