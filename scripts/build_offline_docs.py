"""Build a compact offline copy of the MkDocs site for wheel distribution."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src" / "logqbit" / "_docs"
UNNEEDED_PATHS = (
    Path("404.html"),
    Path("css/fonts"),
    Path("js"),
    Path("objects.inv"),
    Path("search"),
    Path("search.html"),
    Path("sitemap.xml"),
    Path("sitemap.xml.gz"),
)
HIGHLIGHT_ASSET = re.compile(
    r'\s*<(?:link[^>]+highlight\.js[^>]*|script[^>]+highlight\.min\.js[^>]*></script>|'
    r'script>hljs\.highlightAll\(\);</script>)',
    re.DOTALL,
)
SEARCH_FORM = re.compile(r'<div role="search">.*?</div>\s*</div>', re.DOTALL)
LEGACY_SCRIPT = re.compile(
    r"\s*<script src=\"(?:\.\./)*js/"
    r"(?:html5shiv\.min\.js|jquery-3\.6\.0\.min\.js|theme_extra\.js|theme\.js)\"></script>",
)
SEARCH_SCRIPT = re.compile(r'\s*<script src="(?:\.\./)*search/main\.js"></script>')
THEME_INITIALIZER = re.compile(
    r'\s*<script>\s*jQuery\(function \(\) \{\s*'
    r'SphinxRtdTheme\.Navigation\.enable\(true\);\s*\}\);\s*</script>',
    re.DOTALL,
)
DOC_LINK = re.compile(
    r'href="((?:\.\./)*(?:core|browser|cli|migration_guide)/)"'
)
HOME_LINK = re.compile(r'href="\."')
PARENT_HOME_LINK = re.compile(r'href="\.\."')
MOBILE_NAVIGATION = """
<script>
document.querySelectorAll('[data-toggle="wy-nav-top"]').forEach((toggle) => {
  toggle.addEventListener("click", () => {
    document.querySelectorAll('[data-toggle="wy-nav-shift"]').forEach((element) => {
      element.classList.toggle("shift");
    });
  });
});
</script>"""


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _compact_html(output: Path) -> None:
    for html_path in output.rglob("*.html"):
        page = html_path.read_text(encoding="utf-8")
        page = HIGHLIGHT_ASSET.sub("", page)
        page = SEARCH_FORM.sub("", page)
        page = LEGACY_SCRIPT.sub("", page)
        page = SEARCH_SCRIPT.sub("", page)
        page = THEME_INITIALIZER.sub("", page)
        page = DOC_LINK.sub(r'href="\1index.html"', page)
        page = PARENT_HOME_LINK.sub('href="../index.html"', page)
        page = HOME_LINK.sub('href="index.html"', page)
        page = page.replace("</body>", f"{MOBILE_NAVIGATION}\n</body>")
        html_path.write_text(page, encoding="utf-8")


def build(output: Path) -> None:
    """Build once with MkDocs, then remove assets unnecessary offline."""
    with tempfile.TemporaryDirectory(prefix="logqbit-mkdocs-") as temporary_dir:
        site_dir = Path(temporary_dir) / "site"
        subprocess.run(
            ["mkdocs", "build", "--strict", "--site-dir", str(site_dir)],
            cwd=ROOT,
            check=True,
        )
        _remove(output)
        shutil.copytree(site_dir, output)

    for relative_path in UNNEEDED_PATHS:
        _remove(output / relative_path)
    _compact_html(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
