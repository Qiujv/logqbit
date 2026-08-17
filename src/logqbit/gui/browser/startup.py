"""Launch and bootstrap the LogQbit browser."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import traceback
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QSplashScreen

_FOREGROUND_OPTION = "--foreground"
_STARTUP_MODULE = "logqbit.gui.browser.startup"
_WINDOWS_APP_USER_MODEL_ID = "Qiujv.LogQbit.Browser"


def launch_browser(directory: str | Path | None = None) -> None:
    """Launch the browser in a new process using the platform's GUI Python."""
    executable = sys.executable
    environment = os.environ.copy()

    if platform.system() == "Windows":
        executable, overrides = _windows_gui_launch(executable)
        environment.update(overrides)

    command = [executable, "-m", _STARTUP_MODULE, _FOREGROUND_OPTION]
    if directory is not None:
        command.append(str(directory))

    _spawn_gui_process(command, environment)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the browser, optionally opening a directory."""
    args = list(sys.argv[1:] if argv is None else argv)
    foreground = _FOREGROUND_OPTION in args
    if foreground:
        args.remove(_FOREGROUND_OPTION)
    if len(args) > 1:
        print("usage: logqbit-browser [--foreground] [directory]", file=sys.stderr)
        return 2
    if foreground:
        return run_browser_application(args)
    launch_browser(args[0] if args else None)
    return 0


def _spawn_gui_process(command: list[str], environment: dict[str, str]) -> None:
    """Start the GUI without routing through shell or script launchers."""
    subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=platform.system() != "Windows",
    )


def windows_gui_executable(executable: str) -> str:
    """Prefer pythonw.exe so Windows GUI launches do not get a console window."""
    path = PureWindowsPath(executable) if "\\" in executable else Path(executable)
    if path.name.lower() != "python.exe":
        return executable

    candidate = path.with_name("pythonw.exe")
    if os.path.exists(candidate):
        return str(candidate)
    return executable


def _windows_gui_launch(executable: str) -> tuple[str, dict[str, str]]:
    """Resolve a real GUI interpreter while retaining the current venv."""
    gui_executable = windows_gui_executable(executable)
    environment: dict[str, str] = {}

    if sys.prefix == sys.base_prefix:
        return gui_executable, environment

    # Work around broken venv pythonw trampolines in uv minor-version installs.
    base_executable = getattr(sys, "_base_executable", "")
    base_gui_executable = windows_gui_executable(base_executable)
    base_gui_path = PureWindowsPath(base_gui_executable)
    if base_gui_path.name.lower() == "pythonw.exe" and os.path.exists(
        base_gui_executable
    ):
        gui_executable = base_gui_executable
        environment["__PYVENV_LAUNCHER__"] = executable

    return gui_executable, environment


def _set_windows_app_user_model_id() -> None:
    """Give the browser a stable Windows taskbar identity."""
    if platform.system() != "Windows":
        return

    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
        _WINDOWS_APP_USER_MODEL_ID
    )


def _browser_window_icon() -> QIcon:
    """Load the native Windows icon, falling back to the source SVG."""
    from PySide6.QtGui import QIcon

    assets = files("logqbit") / "assets"
    for filename in ("browser.ico", "browser.svg"):
        icon = QIcon(str(assets / filename))
        if not icon.isNull():
            return icon
    return QIcon()


def _show_startup_splash(app: QApplication) -> QSplashScreen:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    pixmap = QPixmap(380, 104)
    pixmap.fill(QColor("#dceaf7"))
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    font = splash.font()
    font.setPointSize(14)
    font.setBold(True)
    splash.setFont(font)
    splash.showMessage(
        "LogQbit Browser 正在启动...",
        Qt.AlignCenter,
        QColor("#202124"),
    )
    splash.show()
    app.processEvents()
    return splash


def _show_startup_error(error: Exception) -> None:
    """Show startup failures that would otherwise be hidden by ``pythonw.exe``."""
    from PySide6.QtWidgets import QMessageBox

    message = QMessageBox()
    message.setIcon(QMessageBox.Critical)
    message.setWindowTitle("LogQbit Browser 启动失败")
    message.setText("LogQbit Browser 无法启动。")
    message.setInformativeText(f"{type(error).__name__}: {error}")
    message.setDetailedText("".join(traceback.format_exception(error)))
    message.setStandardButtons(QMessageBox.Ok)
    message.exec()


def run_browser_application(argv: Sequence[str]) -> int:
    """Show a lightweight splash before importing the full browser window."""
    from PySide6.QtWidgets import QApplication

    _set_windows_app_user_model_id()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("LogQbit Log Browser")
    icon = _browser_window_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    splash = _show_startup_splash(app)

    try:
        from logqbit.gui.browser.window.view import LogBrowserWindow

        directory = Path(argv[0]).expanduser().resolve() if argv else None
        window = LogBrowserWindow(directory)
        if not icon.isNull():
            window.setWindowIcon(icon)
        window.show()
        splash.finish(window)
        return app.exec()
    except Exception as error:
        splash.close()
        _show_startup_error(error)
        return 1
    finally:
        splash.close()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
