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
    assert meta.plot_axes == []
    datetime.strptime(meta.root["create_time"], "%Y-%m-%d %H:%M:%S")
    assert meta.root["create_machine"] == socket.gethostname()


def test_logmetadata_persists_updates(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta = LogMetadata(meta_path, title="demo")

    meta.title = "demo-updated"
    meta.star = 3
    meta.trash = True
    meta.plot_axes = ["x", "y"]

    reloaded = LogMetadata(meta_path)
    assert reloaded.title == "demo-updated"
    assert reloaded.star == 3
    assert reloaded.trash is True
    assert reloaded.plot_axes == ["x", "y"]


def test_logmetadata_detects_external_change(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta = LogMetadata(meta_path)

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["title"] = "external"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")

    assert meta.title == "external"


def test_logmetadata_create_false(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        LogMetadata(missing, create=False)

