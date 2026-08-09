"""Fast visual bootstrap for the LogQbit browser."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen


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


def _show_startup_notice(app: QApplication) -> QSplashScreen:
    pixmap = QPixmap(380, 104)
    pixmap.fill(QColor("#f5f7fa"))
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.showMessage(
        "LogQbit Browser 正在启动…",
        Qt.AlignCenter,
        QColor("#202124"),
    )
    splash.show()
    app.processEvents()
    return splash


def main(argv: list[str] | None = None) -> int:
    """Show a lightweight splash before importing the full browser window."""
    args = argv if argv is not None else sys.argv[1:]
    app = ensure_application()
    splash = _show_startup_notice(app)

    try:
        from logqbit.gui.browser.window.view import LogBrowserWindow

        directory = Path(args[0]).expanduser().resolve() if args else None
        window = LogBrowserWindow(directory)
        window.show()
        splash.finish(window)
        return app.exec()
    finally:
        splash.close()
