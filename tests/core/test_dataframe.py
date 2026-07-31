from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from logqbit import dataframe as dataframe_module
from logqbit.dataframe import DataFrameBuffer, _autosave_interval_for_rows


def test_flush_writes_pending_rows(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer(path)
    try:
        frame.add_one_row({"x": 1, "y": 2.0})

        saved = frame.flush()

        expected = pd.DataFrame([{"x": 1, "y": 2.0}])
        pd.testing.assert_frame_equal(saved.reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(
            pd.read_feather(path).reset_index(drop=True),
            expected,
        )
    finally:
        frame.close()


def test_autosave_writes_pending_rows(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer(path, autosave_interval=0.01)
    try:
        frame.add_multi_rows(pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}))

        deadline = time.monotonic() + 1
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)

        expected = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        pd.testing.assert_frame_equal(
            pd.read_feather(path).reset_index(drop=True),
            expected,
        )
    finally:
        frame.close()


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
    frame = DataFrameBuffer(path, autosave_interval=0.01)
    original_replace = frame._replace_tmp
    replace_count = 0

    def replace_once_failed(*args, **kwargs) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 1:
            raise OSError("temporary write failure")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(dataframe_module, "_AUTOSAVE_RETRY_INTERVAL", 0.01)
    monkeypatch.setattr(frame, "_replace_tmp", replace_once_failed)
    try:
        frame.add_one_row({"x": 1})
        deadline = time.monotonic() + 1
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert path.exists()
        assert frame._thread.is_alive()
        assert replace_count >= 2
        assert not list(tmp_path.glob("data.*.tmp"))
        assert "temporary write failure" in caplog.text
    finally:
        frame.close()


def test_closed_buffer_remains_readable_but_rejects_appends(tmp_path: Path) -> None:
    frame = DataFrameBuffer(tmp_path / "data.feather")
    frame.add_one_row({"x": 1})
    frame.close()

    assert frame.closed
    assert not frame._thread.is_alive()
    pd.testing.assert_frame_equal(frame.get_df(), pd.DataFrame({"x": [1]}))
    with pytest.raises(RuntimeError, match="closed DataFrameBuffer"):
        frame.add_one_row({"x": 2})
    with pytest.raises(RuntimeError, match="closed DataFrameBuffer"):
        frame.add_multi_rows(pd.DataFrame({"x": [2]}))


def test_failed_initialization_releases_writer_path(
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
            DataFrameBuffer(path)

    path.unlink()
    frame = DataFrameBuffer(path)
    frame.close()
