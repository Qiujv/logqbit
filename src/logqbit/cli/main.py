"""Command-line interface for LogQbit utilities."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path


def copy_template(template_name: str, output_path: Path | None = None) -> int:
    """Copy a packaged template to the requested output path."""
    template_file = f"{template_name}.py"
    template_path = files("logqbit") / "templates" / template_file
    if not template_path.is_file():
        print(f"Error: Template '{template_name}' not found.", file=sys.stderr)
        print("\nAvailable templates:", file=sys.stderr)
        for item in files("logqbit").joinpath("templates").iterdir():
            if item.name.endswith(".py"):
                print(f"  - {item.name[:-3]}", file=sys.stderr)
        return 1

    if output_path is None:
        output_path = Path.cwd() / template_file
    elif output_path.is_dir():
        output_path = output_path / template_file

    if output_path.exists():
        response = input(f"File '{output_path}' already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Cancelled.")
            return 0

    try:
        output_path.write_bytes(template_path.read_bytes())
    except OSError as exc:
        print(f"Error copying template: {exc}", file=sys.stderr)
        return 1

    print(f"✓ Template copied to: {output_path}")
    print("\nNext steps:")
    print("  1. Edit the file to configure your paths")
    print(f"  2. Run: python {output_path.name}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logqbit",
        description="LogQbit command-line utilities",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    copy_parser = subparsers.add_parser(
        "copy-template",
        help="Copy a template script to your working directory",
    )
    copy_parser.add_argument(
        "template",
        help="Template name (e.g., 'move_from_labrad')",
    )
    copy_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: current directory)",
    )

    browser_parser = subparsers.add_parser(
        "browser",
        help="Launch the log browser GUI",
    )
    browser_parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory to open (default: current directory)",
    )
    browser_parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in the current process and keep the terminal attached",
    )
    subparsers.add_parser(
        "browser-demo",
        help="Create example data and launch browser",
    )

    shortcuts_parser = subparsers.add_parser(
        "shortcuts",
        help="Create a Windows desktop shortcut for the browser",
    )
    shortcuts_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: Desktop)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected LogQbit command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "copy-template":
        return copy_template(args.template, args.output)
    if args.command == "browser":
        from logqbit.gui.browser.startup import (
            launch_browser,
            run_browser_application,
        )

        if args.foreground:
            browser_args = [str(args.directory)] if args.directory else []
            return run_browser_application(browser_args)
        launch_browser(args.directory)
        return 0
    if args.command == "browser-demo":
        from logqbit.cli.demo import create_example_data

        return create_example_data()
    if args.command == "shortcuts":
        from logqbit.cli.shortcuts import create_shortcuts

        return create_shortcuts(args.output)

    parser.print_help()
    return 0
