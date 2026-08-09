from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette

from logqbit.gui.browser.startup.bootstrap import ensure_application
from logqbit.gui.browser.window.preferences import SettingsManager
from logqbit.gui.browser.window.preferences import ThemeManager


class TestSettingsManager:
    def test_update_recent_directories_can_be_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LOGQBIT_BROWSER_DISABLE_RECENT_DIRS", "1")
        manager = SettingsManager()
        original = list(manager.load_recent_directories())

        manager.update_recent_directories(tmp_path)

        assert manager.load_recent_directories() == original

    def test_clear_recent_directories_preserves_requested_path(
        self, tmp_path: Path
    ) -> None:
        manager = SettingsManager()
        manager._settings = QSettings(
            str(tmp_path / "browser-settings.ini"),
            QSettings.IniFormat,
        )
        first = tmp_path / "first"
        manager.save_recent_directories([first, tmp_path / "second"])

        manager.clear_recent_directories(keep=first)

        assert manager.load_recent_directories() == [first]


def test_dark_palette_uses_white_highlighted_text() -> None:
    manager = ThemeManager(ensure_application())

    palette = manager._create_dark_palette()

    assert palette.color(QPalette.HighlightedText) == QColor("white")
