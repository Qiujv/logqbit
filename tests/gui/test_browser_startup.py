from __future__ import annotations

from logqbit.gui.browser import bootstrap
from logqbit.gui.browser.application import ensure_application


def test_browser_startup_notice_is_shown_immediately() -> None:
    app = ensure_application()

    splash = bootstrap._show_startup_notice(app)
    try:
        assert splash.isVisible()
        assert "正在启动" in splash.message()
    finally:
        splash.close()
