from __future__ import annotations

import gc
import time
import weakref
from pathlib import Path

import pandas as pd
import pytest

from logqbit import dataframe as dataframe_module
from logqbit.dataframe import DataFrameBuffer, _autosave_interval_for_rows


def test_flush_writes_pending_rows(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer.open(path)
    frame.add_one_row({"x": 1, "y": 2.0})

    saved = frame.flush()

    expected = pd.DataFrame([{"x": 1, "y": 2.0}])
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(
        pd.read_feather(path).reset_index(drop=True),
        expected,
    )


def test_autosave_writes_pending_rows(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer.open(path)
    frame._state.autosave_interval = 0.01
    frame.add_multi_rows(pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}))

    deadline = time.monotonic() + 1
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    expected = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    pd.testing.assert_frame_equal(
        pd.read_feather(path).reset_index(drop=True),
        expected,
    )


@pytest.mark.parametrize(
    ("row_count", "expected"),
    [(999, 0.1), (1000, 0.2), (10_000, 0.5), (100_000, 1.0)],
)
def test_autosave_interval_for_rows(row_count: int, expected: float) -> None:
    assert _autosave_interval_for_rows(row_count) == expected


def test_autosave_recovers_and_cleans_up_after_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer.open(path)
    frame._state.autosave_interval = 0.01
    state = frame._state
    original_replace = state._replace_tmp
    replace_count = 0

    def replace_once_failed(*args, **kwargs) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 1:
            raise OSError("temporary write failure")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(dataframe_module, "_AUTOSAVE_RETRY_INTERVAL", 0.01)
    monkeypatch.setattr(state, "_replace_tmp", replace_once_failed)
    frame.add_one_row({"x": 1})
    deadline = time.monotonic() + 1
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert path.exists()
    assert state.thread.is_alive()
    assert replace_count >= 2
    assert not list(tmp_path.glob("data.*.tmp"))
    assert "temporary write failure" in caplog.text


def test_same_path_reuses_buffer(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"

    first = DataFrameBuffer.open(path)
    second = DataFrameBuffer.open(path)

    assert second is first
    assert second._state is first._state


def test_last_reference_flushes_and_stops_worker(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer.open(path)
    state = frame._state
    frame.add_one_row({"x": 1})
    frame_ref = weakref.ref(frame)

    del frame
    gc.collect()

    assert frame_ref() is None
    assert not state.thread.is_alive()
    pd.testing.assert_frame_equal(pd.read_feather(path), pd.DataFrame({"x": [1]}))

    reopened = DataFrameBuffer.open(path)
    assert reopened._state is not state


def test_failed_initialization_does_not_cache_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data.feather"
    path.touch()

    def fail_read(_path: Path) -> pd.DataFrame:
        raise OSError("invalid feather")

    with monkeypatch.context() as scoped:
        scoped.setattr(dataframe_module.pd, "read_feather", fail_read)
        with pytest.raises(OSError, match="invalid feather"):
            DataFrameBuffer.open(path)

    path.unlink()
    assert DataFrameBuffer.open(path).path == path


def test_failed_finalizer_state_can_be_reacquired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer.open(path)
    state = frame._state
    original_shutdown = state.shutdown

    def fail_shutdown() -> None:
        raise OSError("temporary shutdown failure")

    monkeypatch.setattr(state, "shutdown", fail_shutdown)
    del frame
    gc.collect()

    assert "temporary shutdown failure" in caplog.text
    reopened = DataFrameBuffer.open(path)
    assert reopened._state is state

    monkeypatch.setattr(state, "shutdown", original_shutdown)
