"""Tests for the read-only YAML detail view."""

from PySide6.QtGui import QColor, QPalette, QTextCursor
from PySide6.QtWidgets import QTextEdit

from logqbit.gui.browser.detail.yaml import YamlView


def test_yaml_view_uses_readable_font_and_syntax_highlighting() -> None:
    view = YamlView()
    source = "title: example\nenabled: true\n# note"

    view.set_yaml_text(source)

    assert view.font().pointSize() >= 10
    assert view.lineWrapMode() == QTextEdit.NoWrap
    assert view.toPlainText() == source
    document = view.document()
    key_cursor = QTextCursor(document)
    key_cursor.setPosition(0)
    comment_cursor = QTextCursor(document)
    comment_cursor.setPosition(source.index("#"))
    assert key_cursor.charFormat().foreground().color().isValid()
    assert (
        key_cursor.charFormat().foreground().color()
        != comment_cursor.charFormat().foreground().color()
    )


def test_yaml_view_rehighlights_for_dark_palette() -> None:
    view = YamlView()
    source = "title: example\n# note"
    view.set_yaml_text(source)
    light_html = view.document().toHtml()

    dark_palette = view.palette()
    dark_palette.setColor(QPalette.Base, QColor("black"))
    dark_palette.setColor(QPalette.Text, QColor("white"))
    view.setPalette(dark_palette)

    assert view.document().toHtml() != light_html
    assert view.toPlainText() == source
