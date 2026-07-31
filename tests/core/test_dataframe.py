from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from logqbit.dataframe import DataFrameBuffer


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


def test_autosave_interval_updates_from_row_count(tmp_path: Path) -> None:
    path = tmp_path / "data.feather"
    frame = DataFrameBuffer(path, autosave_interval=0.01)
    try:
        frame.add_multi_rows(pd.DataFrame({"x": range(1000)}))

        frame.flush()

        assert frame._autosave_interval == 0.2
    finally:
        frame.close()
