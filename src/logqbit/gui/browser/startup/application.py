"""Lightweight QApplication setup shared by Browser startup paths."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.resources import files

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication


def ensure_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the Browser application, creating and configuring it if needed."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv) if argv is not None else sys.argv)

    app.setApplicationName("LogQbit Log Browser")
    icon = QIcon(str(files("logqbit") / "assets" / "browser.svg"))
    if not icon.isNull():
        app.setWindowIcon(icon)
    return app
