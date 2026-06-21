from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qt_application() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def disable_browser_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGQBIT_BROWSER_DISABLE_RECENT_DIRS", "1")
    monkeypatch.setenv("LOGQBIT_BROWSER_DISABLE_JIT_WARMUP", "1")
