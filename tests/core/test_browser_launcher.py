from __future__ import annotations

import subprocess
from pathlib import Path

from logqbit.gui.browser import startup


def test_windows_gui_executable_prefers_pythonw(monkeypatch) -> None:
    monkeypatch.setattr(startup.os.path, "exists", lambda _path: True)

    assert startup.windows_gui_executable(r"C:\Python\python.exe") == (
        r"C:\Python\pythonw.exe"
    )


def test_windows_gui_executable_keeps_non_console_executable() -> None:
    executable = r"C:\Python\pythonw.exe"

    assert startup.windows_gui_executable(executable) == executable


def test_launch_browser_uses_windows_gui_launcher(monkeypatch, tmp_path: Path) -> None:
    launched: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(startup.platform, "system", lambda: "Windows")
    monkeypatch.setattr(startup.sys, "executable", r"C:\Python\python.exe")
    monkeypatch.setenv("LOGQBIT_TEST_ENV", "preserved")
    monkeypatch.setattr(
        startup,
        "_windows_gui_launch",
        lambda _executable: (
            r"C:\Python\pythonw.exe",
            {"__PYVENV_LAUNCHER__": r"C:\project\.venv\Scripts\python.exe"},
        ),
    )
    monkeypatch.setattr(
        startup,
        "_spawn_gui_process",
        lambda command, environment: launched.append((command, environment)),
    )

    startup.launch_browser(tmp_path)

    command, environment = launched[0]
    assert command == [
        r"C:\Python\pythonw.exe",
        "-m",
        "logqbit.gui.browser.startup",
        "--foreground",
        str(tmp_path),
    ]
    assert environment["LOGQBIT_TEST_ENV"] == "preserved"
    assert environment["__PYVENV_LAUNCHER__"] == (
        r"C:\project\.venv\Scripts\python.exe"
    )


def test_windows_gui_launch_uses_base_pythonw_for_uv_venv(monkeypatch) -> None:
    monkeypatch.setattr(startup.sys, "prefix", r"C:\project\.venv")
    monkeypatch.setattr(startup.sys, "base_prefix", r"C:\uv\python")
    monkeypatch.setattr(startup.sys, "_base_executable", r"C:\uv\python\python.exe")
    monkeypatch.setattr(startup.os.path, "exists", lambda _path: True)

    executable, environment = startup._windows_gui_launch(
        r"C:\project\.venv\Scripts\pythonw.exe"
    )

    assert executable == r"C:\uv\python\pythonw.exe"
    assert environment == {
        "__PYVENV_LAUNCHER__": r"C:\project\.venv\Scripts\pythonw.exe"
    }


def test_launch_browser_uses_current_interpreter_on_posix(
    monkeypatch, tmp_path: Path
) -> None:
    launched: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup.sys, "executable", "/usr/bin/python")
    monkeypatch.setattr(
        startup,
        "_spawn_gui_process",
        lambda command, environment: launched.append((command, environment)),
    )

    startup.launch_browser(tmp_path)

    assert launched[0][0] == [
        "/usr/bin/python",
        "-m",
        "logqbit.gui.browser.startup",
        "--foreground",
        str(tmp_path),
    ]


def test_spawn_gui_process_uses_one_direct_child(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(startup.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        startup.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    startup._spawn_gui_process(["pythonw.exe", "-m", "browser"], {"A": "B"})

    command, kwargs = calls[0]
    assert command == ["pythonw.exe", "-m", "browser"]
    assert kwargs == {
        "env": {"A": "B"},
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "start_new_session": False,
    }


def test_launcher_main_rejects_multiple_directories(capsys) -> None:
    assert startup.main(["first", "second"]) == 2
    assert "usage: logqbit-browser [--foreground] [directory]" in (
        capsys.readouterr().err
    )


def test_launcher_main_runs_in_foreground(monkeypatch) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(
        startup,
        "run_browser_application",
        lambda args: launched.append(list(args)) or 17,
    )

    assert startup.main(["records", "--foreground"]) == 17
    assert launched == [["records"]]
