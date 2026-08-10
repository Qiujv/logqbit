from __future__ import annotations

import builtins

from logqbit.gui.browser.startup import bootstrap
from logqbit.gui.browser.startup.bootstrap import ensure_application


def test_browser_startup_notice_is_shown_immediately() -> None:
    app = ensure_application()

    splash = bootstrap._show_startup_notice(app)
    try:
        assert splash.isVisible()
        assert "正在启动" in splash.message()
    finally:
        splash.close()


def test_browser_startup_failure_shows_error_dialog(monkeypatch) -> None:
    ensure_application()
    expected = RuntimeError("window import failed")
    shown: list[Exception] = []
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "logqbit.gui.browser.window.view":
            raise expected
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.setattr(bootstrap, "_show_startup_error", shown.append)

    assert bootstrap.main([]) == 1
    assert shown == [expected]


def test_startup_error_dialog_contains_exception_details(monkeypatch) -> None:
    ensure_application()
    captured: dict[str, object] = {}

    class FakeMessageBox:
        Critical = object()
        Ok = object()

        def setIcon(self, value) -> None:
            captured["icon"] = value

        def setWindowTitle(self, value) -> None:
            captured["title"] = value

        def setText(self, value) -> None:
            captured["text"] = value

        def setInformativeText(self, value) -> None:
            captured["info"] = value

        def setDetailedText(self, value) -> None:
            captured["details"] = value

        def setStandardButtons(self, value) -> None:
            captured["buttons"] = value

        def exec(self) -> None:
            captured["shown"] = True

    monkeypatch.setattr(bootstrap, "QMessageBox", FakeMessageBox)

    bootstrap._show_startup_error(ValueError("bad config"))

    assert captured["title"] == "LogQbit Browser 启动失败"
    assert captured["info"] == "ValueError: bad config"
    assert "ValueError: bad config" in str(captured["details"])
    assert captured["shown"] is True
