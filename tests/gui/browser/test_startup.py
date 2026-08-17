from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from logqbit.gui.browser import startup


def _create_application() -> QApplication:
    app = QApplication.instance()
    assert app is not None
    return app


def test_browser_startup_notice_is_shown_immediately() -> None:
    app = _create_application()

    splash = startup._show_startup_splash(app)
    try:
        assert splash.isVisible()
        assert "正在启动" in splash.message()
        assert splash.font().pointSize() == 14
        assert splash.font().bold()
    finally:
        splash.close()


def test_windows_taskbar_identity_is_set_before_qt_startup(monkeypatch) -> None:
    app_ids: list[str] = []
    fake_ctypes = SimpleNamespace(
        windll=SimpleNamespace(
            shell32=SimpleNamespace(
                SetCurrentProcessExplicitAppUserModelID=app_ids.append
            )
        )
    )
    monkeypatch.setattr(startup.platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    startup._set_windows_app_user_model_id()

    assert app_ids == [startup._WINDOWS_APP_USER_MODEL_ID]


def test_browser_startup_failure_shows_error_dialog(monkeypatch) -> None:
    app = _create_application()
    app.setWindowIcon(QIcon())
    expected = RuntimeError("window import failed")
    shown: list[Exception] = []
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "logqbit.gui.browser.window.view":
            raise expected
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.setattr(startup, "_show_startup_error", shown.append)

    assert startup.run_browser_application([]) == 1
    assert shown == [expected]
    assert not app.windowIcon().isNull()


def test_startup_error_dialog_contains_exception_details(monkeypatch) -> None:
    _create_application()
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

    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox", FakeMessageBox)

    startup._show_startup_error(ValueError("bad config"))

    assert captured["title"] == "LogQbit Browser 启动失败"
    assert captured["info"] == "ValueError: bad config"
    assert "ValueError: bad config" in str(captured["details"])
    assert captured["shown"] is True
