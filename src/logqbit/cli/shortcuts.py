"""Windows desktop shortcut support for the LogQbit browser."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
from importlib.resources import files
from pathlib import Path


def _powershell_single_quoted(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _browser_icon() -> Path | None:
    assets_dir = files("logqbit") / "assets"
    browser_svg = assets_dir / "browser.svg"
    browser_ico = assets_dir / "browser.ico"
    if browser_ico.is_file():
        return Path(str(browser_ico))
    if not browser_svg.is_file():
        print(f"Error: SVG file not found: {browser_svg}", file=sys.stderr)
        return None

    try:
        from logqbit.cli.svg2ico import svg_to_ico

        print("Generating ICO file from SVG...")
        try:
            svg_to_ico(str(browser_svg), str(browser_ico))
            print(f"  ✓ Created: {browser_ico}")
            return Path(str(browser_ico))
        except (PermissionError, OSError):
            temporary_ico = Path(tempfile.gettempdir()) / "logqbit_browser.ico"
            svg_to_ico(str(browser_svg), str(temporary_ico))
            print(f"  ✓ Created: {temporary_ico} (package dir not writable)")
            return temporary_ico
    except ImportError as exc:
        print(f"Error: Could not import svg2ico: {exc}", file=sys.stderr)
        print("Please ensure PySide6 is installed.", file=sys.stderr)
    except Exception as exc:
        print(f"Error generating ICO file: {exc}", file=sys.stderr)
        traceback.print_exc()
    return None


def create_shortcuts(output_dir: Path | None = None) -> int:
    """Create a Windows desktop shortcut for the LogQbit browser."""
    try:
        browser_ico = _browser_icon()
        if browser_ico is None:
            return 1

        if output_dir is None:
            result = subprocess.run(
                ["powershell", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                print("Error: Could not determine Desktop path", file=sys.stderr)
                return 1
            output_dir = Path(result.stdout.strip())
            if not output_dir.exists():
                print(
                    f"Error: Desktop directory does not exist: {output_dir}",
                    file=sys.stderr,
                )
                return 1

        output_dir.mkdir(parents=True, exist_ok=True)
        shortcut_path = output_dir / "LogQbit Browser.lnk"
        browser_entrypoint = Path(sys.executable).with_name("logqbit-browser.exe")
        powershell_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut({_powershell_single_quoted(shortcut_path)})
$Shortcut.TargetPath = {_powershell_single_quoted(browser_entrypoint)}
$Shortcut.IconLocation = {_powershell_single_quoted(browser_ico)}
$Shortcut.Save()
"""
        result = subprocess.run(
            ["powershell", "-Command", powershell_script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("Error creating shortcut: LogQbit Browser", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1

        print(f"✓ Created: {shortcut_path}")
        print(f"\n✓ Shortcut created in: {output_dir}")
        return 0
    except Exception as exc:
        print(f"Error creating shortcut: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
