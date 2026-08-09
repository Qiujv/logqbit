"""Read-only YAML detail tab."""

from __future__ import annotations

import html

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import YamlLexer
from pygments.token import Text
from PySide6.QtCore import QEvent
from PySide6.QtGui import QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QTextEdit


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
