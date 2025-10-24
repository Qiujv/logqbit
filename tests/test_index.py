import json
import socket
from datetime import datetime
from pathlib import Path

import pytest

from logqbit.index import FileLock, LogIndex


def test_logindex_creates_defaults(tmp_path: Path) -> None:
    idx_path = tmp_path / "index.json"
    idx = LogIndex(idx_path)

    assert idx_path.exists()
    assert idx.title == "untitled"
    assert idx.star == 0
    assert idx.trash is False
    assert idx.plot_axes == []
    datetime.strptime(idx.root["create_time"], "%Y-%m-%d %H:%M:%S")
    assert idx.root["create_machine"] == socket.gethostname()


def test_logindex_persists_updates(tmp_path: Path) -> None:
    idx_path = tmp_path / "index.json"
    idx = LogIndex(idx_path, title="demo")

    idx.title = "demo-updated"
    idx.star = 3
    idx.trash = True
    idx.plot_axes = ["x", "y"]

    reloaded = LogIndex(idx_path)
    assert reloaded.title == "demo-updated"
    assert reloaded.star == 3
    assert reloaded.trash is True
    assert reloaded.plot_axes == ["x", "y"]
    assert not idx_path.with_suffix(".lock").exists()


def test_logindex_detects_external_change(tmp_path: Path) -> None:
    idx_path = tmp_path / "index.json"
    idx = LogIndex(idx_path)

    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    payload["title"] = "external"
    idx_path.write_text(json.dumps(payload), encoding="utf-8")

    assert idx.title == "external"


def test_logindex_create_false(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        LogIndex(missing, create=False)


def test_filelock_context_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    lock_path = target.with_suffix(".lock")

    with FileLock(target, timeout=0.2, delete_on_release=True):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_filelock_timeout_when_held(tmp_path: Path) -> None:
    target = tmp_path / "resource.json"
    lock1 = FileLock(target, timeout=0.5, delete_on_release=False)
    lock1.acquire()
    try:
        lock2 = FileLock(target, timeout=0.05, delete_on_release=False)
        with pytest.raises(TimeoutError):
            lock2.acquire()
        if getattr(lock2, "_file", None):
            try:
                lock2._file.close()  # type: ignore[attr-defined]
            finally:
                lock2._file = None  # type: ignore[attr-defined]
    finally:
        lock1.release()
        lingering = target.with_suffix(".lock")
        if lingering.exists():
            lingering.unlink()
