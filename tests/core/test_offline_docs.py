from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from logqbit.gui.browser import about

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("mkdocs") is None, reason="MkDocs is a dev dependency")
def test_offline_docs_builder_compacts_mkdocs_site(tmp_path: Path) -> None:
    output = tmp_path / "offline-docs"

    subprocess.run(
        [sys.executable, "scripts/build_offline_docs.py", "--output", str(output)],
        cwd=ROOT,
        check=True,
    )

    page = (output / "index.html").read_text(encoding="utf-8")
    for title in ("首页与安装", "核心 API", "LogBrowser 使用指南", "命令行工具", "从 LabRAD 迁移"):
        assert title in page
    assert (output / "core" / "index.html").is_file()
    assert "LogFolder" in page
    assert "highlight.js" not in page
    assert not (output / "css" / "fonts").exists()
    assert not (output / "search").exists()
    assert not (output / "js").exists()
    assert "jQuery" not in page
    assert "wy-nav-top" in page
    assert 'href="index.html" class="icon icon-home"' in page
    core_page = (output / "core" / "index.html").read_text(encoding="utf-8")
    assert 'href="../index.html" class="icon icon-home"' in core_page


def test_about_message_links_to_bundled_docs(monkeypatch) -> None:
    docs_url = "file:///tmp/logqbit-docs/index.html"
    monkeypatch.setattr(about, "offline_docs_url", lambda: docs_url)

    assert f'href="{docs_url}"' in about.about_message()
    assert "Documentation (offline)" in about.about_message()
    assert "A lab-data toolkit by Qiujv." in about.about_message()
    assert "Built with PySide6, pyqtgraph, pandas, and Apache Arrow." in about.about_message()
