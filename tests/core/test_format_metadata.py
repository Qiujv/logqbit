import json
import os
from pathlib import Path

from logqbit.cli import main
from logqbit.cli import format_metadata
from logqbit.metadata import LogMetadata


def test_format_metadata_preview_and_idempotence(tmp_path: Path, capsys) -> None:
    path = tmp_path / "nested" / "metadata.json"
    path.parent.mkdir()
    original = json.dumps({"title": "中文", "custom": {"备注": ["测试", 3]}})
    path.write_text(original, encoding="utf-8")
    os.chmod(path, 0o640)
    assert main(["format-metadata-json", str(tmp_path), "--dry-run"]) == 0
    assert path.read_text(encoding="utf-8") == original
    assert main(["format-metadata-json", str(tmp_path)]) == 0
    assert '    "title": "中文"' in path.read_text(encoding="utf-8")
    assert LogMetadata(path, create=False).root == json.loads(original)
    assert path.stat().st_mode & 0o777 == 0o640
    before = path.stat().st_mtime_ns
    assert main(["format-metadata-json", str(tmp_path)]) == 0
    assert path.stat().st_mtime_ns == before
    assert "unchanged: 1" in capsys.readouterr().out


def test_format_metadata_skips_unsafe_files_and_continues(
    tmp_path: Path, capsys
) -> None:
    for name, content in {
        "duplicate": '{"x": 1, "x": 2}',
        "invalid": "{invalid",
        "array": "[]",
        "precision": '{"x": 1.234567890123456789}',
        "valid": '{"title": "\\u4e2d"}',
    }.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "metadata.json").write_text(content, encoding="utf-8")
    assert main(["format-metadata-json", str(tmp_path)]) == 1
    assert (tmp_path / "duplicate" / "metadata.json").read_text() == '{"x": 1, "x": 2}'
    assert "中" in (tmp_path / "valid" / "metadata.json").read_text()
    assert "errors: 4" in capsys.readouterr().out


def test_changed_file_is_not_overwritten(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "metadata.json"
    path.write_text('{"title": "old"}', encoding="utf-8")
    replace = format_metadata._replace

    def change_first(path, original, formatted, initial):
        path.write_text('{"title": "external"}', encoding="utf-8")
        replace(path, original, formatted, initial)

    monkeypatch.setattr(format_metadata, "_replace", change_first)
    assert main(["format-metadata-json", str(tmp_path)]) == 1
    assert json.loads(path.read_text()) == {"title": "external"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_format_metadata_skips_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    path = target / "metadata.json"
    path.write_text('{"title":"\\u4e2d"}', encoding="utf-8")
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "linked").symlink_to(target, target_is_directory=True)
    (scan / "metadata.json").symlink_to(path)
    assert main(["format-metadata-json", str(scan)]) == 0
    assert path.read_text() == '{"title":"\\u4e2d"}'


def test_format_metadata_leaves_other_files_unchanged(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text('{"title":"\\u4e2d"}')
    data = tmp_path / "data.feather"
    contents = b"ARROW1\x00\xffsample data"
    data.write_bytes(contents)
    initial = data.stat()

    assert main(["format-metadata-json", str(tmp_path)]) == 0
    assert data.read_bytes() == contents
    assert data.stat() == initial
