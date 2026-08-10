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
    frame = DataFrameBuffer(path)
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
    frame = DataFrameBuffer(path)
    frame._worker.autosave_interval = 0.01
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
    frame = DataFrameBuffer(path)
    frame._worker.autosave_interval = 0.01
    worker = frame._worker
    original_replace = worker.cache._replace_tmp
    replace_count = 0

    def replace_once_failed(*args, **kwargs) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 1:
            raise OSError("temporary write failure")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(dataframe_module, "_AUTOSAVE_RETRY_INTERVAL", 0.01)
    monkeypatch.setattr(worker.cache, "_replace_tmp", replace_once_failed)
    frame.add_one_row({"x": 1})
    deadline = time.monotonic() + 1
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert path.exists()
    assert worker.thread.is_alive()
    assert replace_count >= 2
    assert not list(tmp_path.glob("data.*.tmp"))
    assert "temporary write failure" in caplog.text


def test_same_path_creates_independent_buffers(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"

    first = DataFrameBuffer(path)
    second = DataFrameBuffer(path)

    assert second is not first
    assert second._worker is not first._worker
    assert second._worker.thread is not first._worker.thread


def test_independent_buffer_overwrites_changes_after_its_initial_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.feather"
    first = DataFrameBuffer(path)
    second = DataFrameBuffer(path)
    first._worker.autosave_interval = 10
    second._worker.autosave_interval = 10

    first.add_one_row({"x": 1})
    first.flush()
    assert second.get_df().empty

    second.add_one_row({"x": 2})
    second.flush()

    pd.testing.assert_frame_equal(pd.read_feather(path), pd.DataFrame({"x": [2]}))


def test_last_reference_flushes_and_stops_worker(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer(path)
    worker = frame._worker
    frame.add_one_row({"x": 1})
    frame_ref = weakref.ref(frame)

    del frame
    gc.collect()

    assert frame_ref() is None
    assert not worker.thread.is_alive()
    pd.testing.assert_frame_equal(pd.read_feather(path), pd.DataFrame({"x": [1]}))

    reopened = DataFrameBuffer(path)
    assert reopened._worker is not worker


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
            DataFrameBuffer(path)

    path.unlink()
    assert DataFrameBuffer(path).path == path


def test_inspect_and_close_selected_workers(tmp_path: Path) -> None:
    first = DataFrameBuffer(tmp_path / "first.feather")
    second = DataFrameBuffer(tmp_path / "second.feather")
    first._worker.autosave_interval = 10
    first.add_one_row({"x": 1})

    infos = {info.worker_id: info for info in DataFrameBuffer.inspect_workers()}
    first_info = infos[first._worker.worker_id]
    assert first_info.path == first.path
    assert first_info.thread_name == first._worker.thread.name
    assert first_info.thread_ident == first._worker.thread.ident
    assert first_info.thread_alive
    assert first_info.dirty
    assert first_info.owner_alive
    assert not first_info.orphaned
    assert first_info.last_error is None

    failures = DataFrameBuffer.close_workers(first_info.worker_id)

    assert failures == ()
    assert not first._worker.thread.is_alive()
    assert second._worker.thread.is_alive()
    assert first_info.worker_id not in {
        info.worker_id for info in DataFrameBuffer.inspect_workers()
    }
    with pytest.raises(RuntimeError, match="closed dataframe buffer"):
        first.add_one_row({"x": 2})


def test_failed_finalizer_remains_inspectable_and_can_be_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer(path)
    worker = frame._worker
    worker_id = worker.worker_id
    original_close = worker.close

    def fail_close() -> None:
        raise OSError("temporary shutdown failure")

    monkeypatch.setattr(worker, "close", fail_close)
    del frame
    gc.collect()

    assert "temporary shutdown failure" in caplog.text
    info = next(
        info
        for info in DataFrameBuffer.inspect_workers()
        if info.worker_id == worker_id
    )
    assert info.orphaned
    assert info.thread_alive
    assert info.last_error == "OSError: temporary shutdown failure"

    monkeypatch.setattr(worker, "close", original_close)
    assert DataFrameBuffer.close_workers(worker_id) == ()
    assert not worker.thread.is_alive()


def test_close_workers_continues_after_one_worker_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = DataFrameBuffer(tmp_path / "first.feather")
    second = DataFrameBuffer(tmp_path / "second.feather")
    first_worker = first._worker
    second_worker = second._worker
    original_close = first_worker.close

    def fail_close() -> None:
        raise OSError("cannot close first worker")

    monkeypatch.setattr(first_worker, "close", fail_close)
    failures = DataFrameBuffer.close_workers(
        [first_worker.worker_id, second_worker.worker_id]
    )

    assert [info.worker_id for info in failures] == [first_worker.worker_id]
    assert failures[0].last_error == "OSError: cannot close first worker"
    assert first_worker.thread.is_alive()
    assert not second_worker.thread.is_alive()

    monkeypatch.setattr(first_worker, "close", original_close)
    assert DataFrameBuffer.close_workers(first_worker.worker_id) == ()


def test_close_flushes_and_invalidates_buffer(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer(path)
    worker_id = frame._worker.worker_id
    frame.add_one_row({"x": 1})

    frame.close()

    assert not frame._worker.thread.is_alive()
    assert not frame._finalizer.alive
    assert worker_id not in {
        info.worker_id for info in DataFrameBuffer.inspect_workers()
    }
    pd.testing.assert_frame_equal(pd.read_feather(path), pd.DataFrame({"x": [1]}))
    with pytest.raises(RuntimeError, match="closed dataframe buffer"):
        frame.get_df()

    frame.close()
