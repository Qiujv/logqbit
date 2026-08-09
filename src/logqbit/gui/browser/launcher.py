"""Helpers for launching the LogQbit browser as a detached process."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path, PureWindowsPath


def start_browser(directory: str | Path | None = None) -> None:
    """Start the browser without tying its lifetime to the calling terminal."""
    executable = sys.executable
    command = [executable, "-m", "logqbit.gui.browser"]
    if directory is not None:
        command.append(str(directory))

    if platform.system() == "Windows":
        command[0] = windows_gui_executable(executable)
        _start_windows_detached(command)
    else:
        _start_posix_detached(command)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the browser detached, optionally opening a directory."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print("usage: logqbit-browser-detached [directory]", file=sys.stderr)
        return 2
    start_browser(args[0] if args else None)
    return 0


def _start_posix_detached(command: list[str]) -> None:
    launcher = (
        "import subprocess, sys\n"
        "subprocess.Popen(\n"
        "    sys.argv[1:],\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        "    close_fds=True,\n"
        "    start_new_session=True,\n"
        ")\n"
    )
    subprocess.Popen(
        [sys.executable, "-c", launcher, *command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
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


def _start_windows_detached(command: list[str]) -> None:
    """Ask Explorer to launch the GUI outside terminal or notebook job objects."""
    use_vbs = _windows_script_host_enabled()
    script_path = (
        _write_windows_vbs_launcher(command)
        if use_vbs
        else _write_windows_cmd_launcher(command)
    )

    try:
        _open_with_explorer(script_path)
    except OSError:
        if not use_vbs:
            raise
        _open_with_explorer(_write_windows_cmd_launcher(command))


def _open_with_explorer(path: Path) -> None:
    creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    creationflags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    subprocess.Popen(
        ["explorer.exe", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )


def _write_windows_vbs_launcher(command: list[str]) -> Path:
    """Write a hidden WSH launcher for the final browser process."""
    command_line = subprocess.list2cmdline(command)
    escaped_command = command_line.replace('"', '""')
    script = (
        'Set shell = CreateObject("WScript.Shell")\n'
        f'shell.Run "{escaped_command}", 0, False\n'
    )
    path = Path(tempfile.gettempdir()) / "logqbit_browser_launch.vbs"
    path.write_text(script, encoding="utf-8")
    return path


def _write_windows_cmd_launcher(command: list[str]) -> Path:
    """Write a fallback launcher for systems with Windows Script Host disabled."""
    command_line = subprocess.list2cmdline(command)
    script = f'@echo off\nstart "" /B {command_line}\n'
    path = Path(tempfile.gettempdir()) / "logqbit_browser_launch.cmd"
    path.write_text(script, encoding="utf-8")
    return path


def _windows_script_host_enabled() -> bool:
    try:
        import winreg
    except ImportError:
        return True

    subkey = r"Software\Microsoft\Windows Script Host\Settings"
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Enabled")
        except OSError:
            continue
        if int(value) == 0:
            return False
    return True


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(main())
