from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from logqbit import cli
from logqbit.cli import demo
from logqbit.cli import shortcuts
from logqbit.gui.browser import startup
from logqbit.metadata import LogMetadata


def test_copy_template(tmp_path: Path) -> None:
    output_path = tmp_path / "migration.py"

    assert cli.copy_template("move_from_labrad", output_path) == 0
    assert output_path.is_file()


def test_browser_demo_dispatch(monkeypatch) -> None:
    monkeypatch.setattr("logqbit.cli.demo.create_example_data", lambda: 17)

    assert cli.main(["browser-demo"]) == 17


def test_browser_demo_preserves_existing_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    example_dir = tmp_path / "logqbit_example"
    example_dir.mkdir()
    marker = example_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    created_in: list[Path] = []
    launched: list[Path] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        demo,
        "_create_examples",
        lambda path: created_in.append(path) or 7,
    )
    monkeypatch.setattr(startup, "launch_browser", launched.append)

    assert demo.create_example_data() == 0
    assert marker.read_text(encoding="utf-8") == "keep"
    assert created_in == [example_dir]
    assert launched == [example_dir]


def test_grouped_examples_append_records_with_plot_metadata(tmp_path: Path) -> None:
    existing = tmp_path / "0"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    demo._create_grouped_1d_example(tmp_path)
    demo._create_grouped_2d_example(tmp_path)

    assert marker.read_text(encoding="utf-8") == "keep"
    one_d_meta = LogMetadata(tmp_path / "1" / "metadata.json", create=False)
    assert one_d_meta.plot_axes == ("time",)
    assert one_d_meta.plot_fields == ("signal",)
    assert one_d_meta.plot_groupby == ("device", "mode")
    one_d_frame = pd.read_feather(tmp_path / "1" / "data.feather")
    assert one_d_frame.groupby(["device", "mode"]).ngroups == 4

    two_d_meta = LogMetadata(tmp_path / "2" / "metadata.json", create=False)
    assert two_d_meta.plot_axes == ("x", "y")
    assert two_d_meta.plot_fields == ("response",)
    assert two_d_meta.plot_groupby == ("region",)
    two_d_frame = pd.read_feather(tmp_path / "2" / "data.feather")
    assert two_d_frame.groupby("region").size().to_dict() == {
        "left": 16**2,
        "right": 22**2,
    }


def test_browser_dispatch(monkeypatch, tmp_path: Path) -> None:
    launched: list[Path | None] = []
    monkeypatch.setattr(startup, "launch_browser", launched.append)

    assert cli.main(["browser", str(tmp_path)]) == 0
    assert launched == [tmp_path]


def test_browser_foreground_dispatch(monkeypatch, tmp_path: Path) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(
        startup,
        "run_browser_application",
        lambda args: launched.append(list(args)) or 17,
    )

    assert cli.main(["browser", str(tmp_path), "--foreground"]) == 17
    assert launched == [[str(tmp_path)]]


def test_shortcuts_create_only_browser_shortcut(
    monkeypatch,
    tmp_path: Path,
) -> None:
    icon_path = tmp_path / "browser.ico"
    gui_entrypoint = Path(shortcuts.sys.executable).with_name("logqbit-browser.exe")
    icon_path.touch()
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shortcuts, "_browser_icon", lambda: icon_path)
    monkeypatch.setattr(shortcuts.subprocess, "run", run)

    assert shortcuts.create_shortcuts(tmp_path) == 0
    assert len(commands) == 1
    powershell_script = commands[0][2]
    assert str(gui_entrypoint) in powershell_script
    assert "$Shortcut.Arguments" not in powershell_script
    assert "live_plotter" not in powershell_script
