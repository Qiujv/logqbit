"""Fast visual bootstrap for the LogQbit browser."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from logqbit.gui.browser.application import ensure_application


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
        from logqbit.gui.browser.window import LogBrowserWindow

        directory = Path(args[0]).expanduser().resolve() if args else None
        window = LogBrowserWindow(directory)
        window.show()
        splash.finish(window)
        return app.exec()
    finally:
        splash.close()
