"""Information displayed by the Browser's About dialog."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import logqbit
from logqbit._offline_docs import offline_docs_url

HOMEPAGE_URL = "https://github.com/Qiujv/logqbit"
DOCUMENTATION_URL = "https://qiujv.github.io/logqbit/"


def about_message() -> str:
    """Build the Browser About dialog message."""
    try:
        project_version = version("logqbit")
    except PackageNotFoundError:
        project_version = "development"
    package_path = Path(logqbit.__file__).resolve().parent
    offline_docs = offline_docs_url()
    offline_docs_line = (
        f'<a href="{offline_docs}">Documentation (offline)</a><br>'
        if offline_docs
        else ""
    )
    return (
        f"<b>LogQbit</b> · v{project_version}<br>"
        "A lab-data toolkit by Qiujv.<br>"
        f"Path: {package_path}<br>"
        f'<a href="{HOMEPAGE_URL}">Homepage</a><br>'
        f'<a href="{DOCUMENTATION_URL}">Documentation</a><br>'
        f"{offline_docs_line}"
        "Built with PySide6, pyqtgraph, pandas, and Apache Arrow.<br>"
    )
