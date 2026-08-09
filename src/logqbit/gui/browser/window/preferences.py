"""Persistent preferences and color themes for the Browser window."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


DISABLE_RECENT_DIRS_ENV = "LOGQBIT_BROWSER_DISABLE_RECENT_DIRS"
SETTINGS_ORG = "LogQbit"
SETTINGS_APP = "LogBrowser"
SETTINGS_RECENT_DIRS_KEY = "recent/directories"
SETTINGS_THEME_KEY = "ui/theme"


class SettingsManager:
    """Persist browser preferences and recent directories."""

    def __init__(self) -> None:
        self._settings = QSettings(
            QSettings.IniFormat,
            QSettings.UserScope,
            SETTINGS_ORG,
            SETTINGS_APP,
        )
        self._recent_directories: list[Path] = []

    def load_recent_directories(self) -> list[Path]:
        stored = self._settings.value(SETTINGS_RECENT_DIRS_KEY, [])
        if isinstance(stored, str):
            candidates = [stored]
        elif isinstance(stored, (list, tuple)):
            candidates = list(stored)
        else:
            candidates = []
        recent_paths: list[Path] = []
        for item in candidates:
            text = str(item)
            if not text:
                continue
            try:
                path = Path(text)
            except Exception:
                continue
            if path not in recent_paths:
                recent_paths.append(path)
        self._recent_directories = recent_paths[:10]
        return self._recent_directories

    def save_recent_directories(self, directories: list[Path]) -> None:
        self._recent_directories = directories[:10]
        self._settings.setValue(
            SETTINGS_RECENT_DIRS_KEY,
            [str(path) for path in self._recent_directories],
        )
        self._settings.sync()

    def update_recent_directories(self, path: Path) -> list[Path]:
        if os.environ.get(DISABLE_RECENT_DIRS_ENV):
            return self.load_recent_directories()
        self.load_recent_directories()
        resolved = Path(path)
        entries = [resolved]
        for existing in self._recent_directories:
            if existing != resolved:
                entries.append(existing)
            if len(entries) >= 10:
                break
        self.save_recent_directories(entries)
        return self._recent_directories

    def clear_recent_directories(self, keep: Path | None = None) -> None:
        self.save_recent_directories([Path(keep)] if keep is not None else [])

    def load_theme_mode(self) -> str:
        saved_mode = self._settings.value(SETTINGS_THEME_KEY, "system")
        return saved_mode if saved_mode in ThemeManager.THEME_MODES else "system"

    def save_theme_mode(self, mode: str) -> None:
        self._settings.setValue(SETTINGS_THEME_KEY, mode)
        self._settings.sync()


class ThemeManager:
    """Apply the browser's light, dark, or system color theme."""

    THEME_MODES = ["light", "dark", "system"]

    def __init__(self, app: QApplication):
        self.app = app
        self._system_palette = app.palette()
        self._current_mode = "system"

    def apply_theme(self, mode: str) -> None:
        self._current_mode = mode
        style_hints = getattr(self.app, "styleHints", None)
        can_use_color_scheme = False
        hints = None
        if style_hints and hasattr(Qt, "ColorScheme"):
            hints = style_hints()
            can_use_color_scheme = hasattr(hints, "setColorScheme")

        if can_use_color_scheme:
            unknown_scheme = getattr(Qt.ColorScheme, "Unknown", Qt.ColorScheme.Light)
            scheme_map = {
                "dark": Qt.ColorScheme.Dark,
                "light": Qt.ColorScheme.Light,
                "system": unknown_scheme,
            }
            hints.setColorScheme(scheme_map.get(mode, unknown_scheme))
            palette = self._system_palette
        else:
            palette = {
                "dark": self._create_dark_palette(),
                "light": self._create_light_palette(),
                "system": self._system_palette,
            }.get(mode, self._system_palette)

        self.app.setPalette(palette)
        self.app.setStyleSheet("")

    def _create_light_palette(self) -> QPalette:
        palette = QPalette()
        colors = {
            QPalette.Window: (250, 250, 250),
            QPalette.WindowText: (30, 30, 30),
            QPalette.Base: (255, 255, 255),
            QPalette.AlternateBase: (245, 245, 245),
            QPalette.ToolTipBase: (255, 255, 255),
            QPalette.ToolTipText: (30, 30, 30),
            QPalette.Text: (30, 30, 30),
            QPalette.Button: (245, 245, 245),
            QPalette.ButtonText: (30, 30, 30),
            QPalette.BrightText: (255, 0, 0),
            QPalette.Link: (0, 122, 204),
            QPalette.Highlight: (51, 153, 255),
            QPalette.HighlightedText: (255, 255, 255),
        }
        for role, color in colors.items():
            palette.setColor(role, QColor(*color))
        return palette

    def _create_dark_palette(self) -> QPalette:
        palette = QPalette()
        colors = {
            QPalette.Window: (37, 37, 38),
            QPalette.WindowText: (220, 220, 220),
            QPalette.Base: (30, 30, 30),
            QPalette.AlternateBase: (45, 45, 45),
            QPalette.ToolTipBase: (255, 255, 255),
            QPalette.ToolTipText: (255, 255, 255),
            QPalette.Text: (220, 220, 220),
            QPalette.Button: (45, 45, 45),
            QPalette.ButtonText: (220, 220, 220),
            QPalette.BrightText: (255, 0, 0),
            QPalette.Link: (100, 160, 220),
            QPalette.Highlight: (42, 130, 218),
            QPalette.HighlightedText: (255, 255, 255),
        }
        for role, color in colors.items():
            palette.setColor(role, QColor(*color))
        return palette

    def get_theme_button_emoji(self, mode: str) -> str:
        return {"light": "🌝", "dark": "🌚", "system": "🌗"}.get(mode, "🌗")

    def get_theme_tooltip(self, mode: str) -> str:
        return {
            "light": "Light mode",
            "dark": "Dark mode",
            "system": "Follow system theme",
        }.get(mode, "Follow system theme")
