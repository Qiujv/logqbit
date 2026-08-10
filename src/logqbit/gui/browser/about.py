"""Information displayed by the Browser's About dialog."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import logqbit


def about_message() -> str:
    """Build the Browser About dialog message."""
    try:
        project_version = version("logqbit")
    except PackageNotFoundError:
        project_version = "development"
    package_path = Path(logqbit.__file__).resolve().parent
    return (
        "<b>LogQbit by Qiujv</b><br>"
        f"Version: {project_version}<br>"
        f"Path: {package_path}<br>"
    )
