"""Constants detail tab and read-only YAML view."""

from __future__ import annotations

import html
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import YamlLexer
from pygments.token import Text
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import (
    QMessageBox,
    QMenu,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from send2trash import send2trash

from logqbit.gui.browser.detail.files import open_file, read_yaml_text

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord


def yaml_view_font() -> QFont:
    """Return a readable monospace font with broadly available fallbacks."""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    installed_families = set(QFontDatabase.families())
    for family in ("Cascadia Mono", "Consolas"):
        if family in installed_families:
            font.setFamily(family)
            break
    font.setStyleHint(QFont.Monospace)
    if font.pointSize() < 10:
        font.setPointSize(10)
    return font


class YamlView(QTextEdit):
    """Display raw YAML as selectable, syntax-highlighted text."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source = ""
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)
        self.setFont(yaml_view_font())

    def set_yaml_text(self, text: str) -> None:
        self._source = text
        self._render(preserve_view=False)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override naming
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange and getattr(self, "_source", ""):
            self._render(preserve_view=True)

    def _render(self, *, preserve_view: bool) -> None:
        cursor_position = self.textCursor().position() if preserve_view else 0
        scroll_position = self.verticalScrollBar().value() if preserve_view else 0
        style = (
            "github-dark"
            if self.palette().color(QPalette.Base).lightness() < 128
            else "friendly"
        )
        formatter = HtmlFormatter(style=style, noclasses=True, nowrap=True)
        fragment = highlight(self._source, YamlLexer(), formatter)
        default_color = formatter.style.style_for_token(Text).get("color")
        if not default_color:
            default_color = self.palette().color(QPalette.Text).name()[1:]
        font = self.font()
        font_family = html.escape(font.family(), quote=True)
        document = (
            f'<pre style="color: #{default_color}; font-family: &quot;{font_family}&quot;; '
            f'font-size: {font.pointSize()}pt; margin: 4px;">{fragment}</pre>'
        )
        self.setHtml(document)

        cursor = self.textCursor()
        cursor.setPosition(min(cursor_position, len(self.toPlainText())))
        self.setTextCursor(cursor)
        self.verticalScrollBar().setValue(scroll_position)


class ConstTab(QWidget):
    """Display and manage one record's ``const.yaml`` file."""

    file_changed = Signal()

    def __init__(
        self,
        *,
        file_open_callback: Callable[[Path], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._record: LogRecord | None = None
        self._file_open_callback = file_open_callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.yaml_view = YamlView(self)
        self.yaml_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.yaml_view.customContextMenuRequested.connect(self._open_const_context_menu)
        layout.addWidget(self.yaml_view)

    def load_record(self, record: LogRecord) -> None:
        self._record = record
        self.yaml_view.set_yaml_text(read_yaml_text(record.const_path))

    def clear(self, message: str = "") -> None:
        self._record = None
        self.yaml_view.set_yaml_text(message)

    def _open_const_context_menu(self, position) -> None:
        menu = self._create_const_context_menu()
        menu.exec(self.yaml_view.mapToGlobal(position))

    def _create_const_context_menu(self) -> QMenu:
        menu = self.yaml_view.createStandardContextMenu()
        if menu.actions():
            menu.addSeparator()
        edit_action = menu.addAction("Edit...")
        delete_action = menu.addAction("Delete const.yaml")
        record = self._record
        edit_action.setEnabled(record is not None)
        delete_action.setEnabled(record is not None and record.const_path.exists())
        edit_action.triggered.connect(self._edit_const_file)
        delete_action.triggered.connect(self._delete_const_file)
        return menu

    def _edit_const_file(self) -> None:
        record = self._record
        if record is None:
            return
        path = record.const_path
        created = False
        try:
            if not path.exists():
                path.touch(exist_ok=False)
                created = True
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Edit const.yaml",
                f"Could not create const.yaml:\n{exc}",
            )
            return
        if created:
            self.yaml_view.set_yaml_text(read_yaml_text(path))
            self.file_changed.emit()
        self._open_file(path)

    def _delete_const_file(self) -> None:
        record = self._record
        if record is None or not record.const_path.exists():
            return
        try:
            send2trash(str(record.const_path))
        except Exception as exc:  # pragma: no cover - platform integration
            QMessageBox.warning(
                self,
                "Delete const.yaml",
                f"Could not move const.yaml to the Recycle Bin:\n{exc}",
            )
            return
        self.yaml_view.set_yaml_text(read_yaml_text(record.const_path))
        self.file_changed.emit()

    def _open_file(self, path: Path) -> None:
        open_file(path, callback=self._file_open_callback, parent=self)
