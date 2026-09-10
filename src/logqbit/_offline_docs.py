"""Locations for the static documentation bundled in distribution artifacts."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def offline_docs_path() -> Path | None:
    """Return the installed offline documentation entry point, if available."""
    path = Path(files("logqbit").joinpath("_docs", "index.html"))
    return path if path.is_file() else None


def offline_docs_url() -> str | None:
    """Return a file URL for the installed offline documentation, if available."""
    path = offline_docs_path()
    return path.resolve().as_uri() if path else None
