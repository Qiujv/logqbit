import json
import socket
from datetime import datetime
from pathlib import Path

import pytest

from logqbit.metadata import LogMetadata


def test_logmetadata_creates_defaults(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta = LogMetadata(meta_path)

    assert meta_path.exists()
    assert meta.title == "untitled"
    assert meta.star == 0
    assert meta.trash is False
    assert meta.plot_axes == ()
    assert meta.plot_fields == ()
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

    reloaded = LogMetadata(meta_path)
    assert reloaded.title == "demo-updated"
    assert reloaded.star == 3
    assert reloaded.trash is True
    assert reloaded.plot_axes == ("x", "y")
    assert reloaded.root["plot_axes"] == ["x", "y"]
    assert reloaded.plot_fields == ("signal",)
    assert reloaded.root["plot_fields"] == ["signal"]


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
