from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QSettings, Qt

from logqbit.gui.browser.window.preferences import SettingsManager, ThemeManager


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


@pytest.mark.parametrize(
    ("mode", "scheme"),
    [
        ("light", Qt.ColorScheme.Light),
        ("dark", Qt.ColorScheme.Dark),
    ],
)
def test_theme_manager_sets_explicit_color_scheme(mode: str, scheme) -> None:
    app = MagicMock()
    hints = app.styleHints.return_value

    ThemeManager(app).apply_theme(mode)

    hints.setColorScheme.assert_called_once_with(scheme)
    hints.unsetColorScheme.assert_not_called()
    app.setPalette.assert_not_called()
    app.setStyleSheet.assert_not_called()


@pytest.mark.parametrize("mode", ["system", "invalid"])
def test_theme_manager_uses_system_color_scheme_by_default(mode: str) -> None:
    app = MagicMock()
    hints = app.styleHints.return_value

    ThemeManager(app).apply_theme(mode)

    hints.unsetColorScheme.assert_called_once_with()
    hints.setColorScheme.assert_not_called()
