from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from logqbit import cli
from logqbit.cli import shortcuts
from logqbit.gui.browser import launcher


def test_copy_template(tmp_path: Path) -> None:
    output_path = tmp_path / "migration.py"

    assert cli.copy_template("move_from_labrad", output_path) == 0
    assert output_path.is_file()


def test_browser_demo_dispatch(monkeypatch) -> None:
    monkeypatch.setattr("logqbit.cli.demo.create_example_data", lambda: 17)

    assert cli.main(["browser-demo"]) == 17


def test_browser_detached_dispatch(monkeypatch, tmp_path: Path) -> None:
    launched: list[Path | None] = []
    monkeypatch.setattr(launcher, "start_browser", launched.append)

    assert cli.main(["browser", str(tmp_path), "--detach"]) == 0
    assert launched == [tmp_path]


def test_shortcuts_create_only_browser_shortcut(
    monkeypatch,
    tmp_path: Path,
) -> None:
    icon_path = tmp_path / "browser.ico"
    gui_python = tmp_path / "pythonw.exe"
    icon_path.touch()
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shortcuts, "_browser_icon", lambda: icon_path)
    monkeypatch.setattr(
        shortcuts,
        "windows_gui_executable",
        lambda _executable: str(gui_python),
    )
    monkeypatch.setattr(shortcuts.subprocess, "run", run)

    assert shortcuts.create_shortcuts(tmp_path) == 0
    assert len(commands) == 1
    powershell_script = commands[0][2]
    assert str(gui_python) in powershell_script
    assert "logqbit.gui.browser.launcher" in powershell_script
    assert "live_plotter" not in powershell_script
