from __future__ import annotations

from pathlib import Path

from logqbit.gui.browser.startup import launcher


def test_windows_gui_executable_prefers_pythonw(monkeypatch) -> None:
    monkeypatch.setattr(launcher.os.path, "exists", lambda _path: True)

    assert launcher.windows_gui_executable(r"C:\Python\python.exe") == (
        r"C:\Python\pythonw.exe"
    )


def test_windows_gui_executable_keeps_non_console_executable() -> None:
    executable = r"C:\Python\pythonw.exe"

    assert launcher.windows_gui_executable(executable) == executable


def test_start_browser_uses_windows_gui_launcher(monkeypatch, tmp_path: Path) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(launcher.sys, "executable", r"C:\Python\python.exe")
    monkeypatch.setattr(
        launcher,
        "windows_gui_executable",
        lambda _executable: r"C:\Python\pythonw.exe",
    )
    monkeypatch.setattr(launcher, "_start_windows_detached", launched.append)

    launcher.start_browser(tmp_path)

    assert launched == [
        [r"C:\Python\pythonw.exe", "-m", "logqbit.gui.browser", str(tmp_path)]
    ]


def test_start_browser_uses_posix_launcher(monkeypatch, tmp_path: Path) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python")
    monkeypatch.setattr(launcher, "_start_posix_detached", launched.append)

    launcher.start_browser(tmp_path)

    assert launched == [["/usr/bin/python", "-m", "logqbit.gui.browser", str(tmp_path)]]


def test_windows_launcher_prefers_hidden_vbs(monkeypatch, tmp_path: Path) -> None:
    vbs_path = tmp_path / "launch.vbs"
    opened: list[Path] = []
    monkeypatch.setattr(launcher, "_windows_script_host_enabled", lambda: True)
    monkeypatch.setattr(launcher, "_write_windows_vbs_launcher", lambda _cmd: vbs_path)
    monkeypatch.setattr(launcher, "_open_with_explorer", opened.append)

    launcher._start_windows_detached(["pythonw.exe", "-m", "logqbit.gui.browser"])

    assert opened == [vbs_path]


def test_windows_launcher_falls_back_to_cmd(monkeypatch, tmp_path: Path) -> None:
    vbs_path = tmp_path / "launch.vbs"
    cmd_path = tmp_path / "launch.cmd"
    opened: list[Path] = []
    monkeypatch.setattr(launcher, "_windows_script_host_enabled", lambda: True)
    monkeypatch.setattr(launcher, "_write_windows_vbs_launcher", lambda _cmd: vbs_path)
    monkeypatch.setattr(launcher, "_write_windows_cmd_launcher", lambda _cmd: cmd_path)

    def open_with_explorer(path: Path) -> None:
        opened.append(path)
        if path == vbs_path:
            raise OSError("VBS launch failed")

    monkeypatch.setattr(launcher, "_open_with_explorer", open_with_explorer)

    launcher._start_windows_detached(["pythonw.exe", "-m", "logqbit.gui.browser"])

    assert opened == [vbs_path, cmd_path]
