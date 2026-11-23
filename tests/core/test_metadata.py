import json
import socket
from datetime import datetime
from pathlib import Path

import pytest

import logqbit.metadata as metadata_module
from logqbit.metadata import LogMetadata


def test_unicode_formats_remain_readable_and_updates_are_formatted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        '{"title": "\\u4e2d\\u6587", "custom": "\\u6d4b\\u8bd5"}', encoding="utf-8"
    )
    meta = LogMetadata(path, create=False)
    assert meta.title == "中文"
    meta.star = 2
    assert '    "title": "中文"' in path.read_text(encoding="utf-8")
    assert LogMetadata(path, create=False).root["custom"] == "测试"

    created = tmp_path / "new.json"
    LogMetadata(created, title="中文")
    assert '    "title": "中文"' in created.read_text(encoding="utf-8")


def test_logmetadata_creates_defaults(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta = LogMetadata(meta_path)

    assert meta_path.exists()
    assert meta.title == "untitled"
    assert meta.star == 0
    assert meta.trash is False
    assert meta.plot_axes == ()
    assert meta.plot_fields == ()
    assert meta.plot_groupby == ()
    datetime.strptime(meta.root["create_time"], "%Y-%m-%d %H:%M:%S")
    assert meta.root["create_machine"] == socket.gethostname()


def test_logmetadata_persists_updates(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta = LogMetadata(meta_path, title="demo")

    meta.title = "demo-updated"
    meta.star = 3
    meta.trash = True
    meta.plot_axes = ("x", "y")
    meta.plot_fields = "signal"
    meta.plot_groupby = "device"

    reloaded = LogMetadata(meta_path)
    assert reloaded.title == "demo-updated"
    assert reloaded.star == 3
    assert reloaded.trash is True
    assert reloaded.plot_axes == ("x", "y")
    assert reloaded.root["plot_axes"] == ["x", "y"]
    assert reloaded.plot_fields == ("signal",)
    assert reloaded.root["plot_fields"] == ["signal"]
    assert reloaded.plot_groupby == ("device",)
    assert reloaded.root["plot_groupby"] == ["device"]


def test_logmetadata_detects_external_change(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta = LogMetadata(meta_path)

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["title"] = "external"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")

    assert meta.title == "external"


def test_logmetadata_update_changes_loaded_values(tmp_path: Path) -> None:
    meta = LogMetadata(tmp_path / "metadata.json", title="before")

    meta.update(title="after", star=2, plot_axes=["x", "y"])

    assert meta.title == "after"
    assert meta.star == 2
    assert meta.plot_axes == ("x", "y")


def test_metadata_save_retries_windows_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"
    meta = LogMetadata(path)
    original_replace = Path.replace
    attempts = 0
    delays = []

    def replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise PermissionError("temporarily unavailable")
        return original_replace(source, target)

    monkeypatch.setattr(metadata_module.sys, "platform", "win32")
    monkeypatch.setattr(metadata_module.time, "sleep", delays.append)
    monkeypatch.setattr(Path, "replace", replace)
    meta.title = "saved"

    assert attempts == 3
    assert delays == [0.02, 0.04]
    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "saved"
    assert list(tmp_path.glob("*.tmp")) == []


def test_metadata_save_gives_up_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"
    meta = LogMetadata(path, title="original")
    error = PermissionError("access denied")
    attempts = []
    delays = []

    def fail_replace(source, target):
        attempts.append(source)
        raise error

    monkeypatch.setattr(metadata_module.sys, "platform", "win32")
    monkeypatch.setattr(metadata_module.time, "sleep", delays.append)
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError) as exc_info:
        meta.title = "unsaved"

    assert exc_info.value is error
    assert len(attempts) == 7
    assert delays == [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "original"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "platform,error", [("linux", PermissionError()), ("win32", OSError())]
)
def test_metadata_save_does_not_retry_other_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, error: OSError
) -> None:
    meta = LogMetadata(tmp_path / "metadata.json")
    attempts = []

    def fail_replace(source, target):
        attempts.append(source)
        raise error

    monkeypatch.setattr(metadata_module.sys, "platform", platform)
    monkeypatch.setattr(metadata_module.time, "sleep", lambda _: pytest.fail("slept"))
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(type(error)):
        meta.title = "unsaved"
    assert len(attempts) == 1


def test_logmetadata_update_preserves_external_and_unknown_fields(
    tmp_path: Path,
) -> None:
    meta_path = tmp_path / "metadata.json"
    first = LogMetadata(meta_path, title="initial")
    second = LogMetadata(meta_path)

    first.update({"custom": "preserved"}, title="external")
    second.update(star=3)

    assert second.title == "external"
    assert second.star == 3
    assert second.root["custom"] == "preserved"


def test_logmetadata_create_false(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        LogMetadata(missing, create=False)


def test_logmetadata_is_strict_for_invalid_json_by_default(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        LogMetadata(meta_path, create=False)


def test_logmetadata_rejects_non_object_root(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        LogMetadata(meta_path, create=False)

    tolerant = LogMetadata(
        meta_path,
        create=False,
        default_on_error=True,
    )
    assert tolerant.title == "<invalid metadata>"


def test_logmetadata_can_fallback_on_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text("{}", encoding="utf-8")

    def fail_load(stream):
        raise OSError("temporary read failure")

    monkeypatch.setattr(json, "load", fail_load)

    meta = LogMetadata(
        meta_path,
        create=False,
        default_on_error=True,
    )

    assert meta.title == "<invalid metadata>"
    assert meta.star == 0
