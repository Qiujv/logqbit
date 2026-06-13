from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_browser_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGQBIT_BROWSER_DISABLE_RECENT_DIRS", "1")
    monkeypatch.setenv("LOGQBIT_BROWSER_DISABLE_JIT_WARMUP", "1")
