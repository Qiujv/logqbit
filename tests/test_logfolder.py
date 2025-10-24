from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from logqbit.logfolder import LogFolder


def test_new_creates_incremental_directory(tmp_path: Path) -> None:
    parent = tmp_path / "logs"
    parent.mkdir()
    (parent / "0").mkdir()
    (parent / "1").mkdir()

    lf = LogFolder.new(parent)

    assert lf.path.parent == parent
    assert lf.path.name == "2"


def test_add_row_scalar_and_save(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)
    lf.add_row(x=1.5, y=2.0)

    expected_df = pd.DataFrame([{"x": 1.5, "y": 2.0}])
    pd.testing.assert_frame_equal(lf.df.reset_index(drop=True), expected_df)

    lf.flush()

    assert lf.df_path.exists()
    saved_df = pd.read_parquet(lf.df_path)
    pd.testing.assert_frame_equal(saved_df.reset_index(drop=True), expected_df)

    loaded = LogFolder(lf.path)
    pd.testing.assert_frame_equal(loaded.df.reset_index(drop=True), expected_df)


def test_add_row_vector_creates_dataframe(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)
    lf.add_row(step=[0, 1, 2], current=[0.1, 0.2, 0.3])

    expected_df = pd.DataFrame({"step": [0, 1, 2], "current": [0.1, 0.2, 0.3]})

    pd.testing.assert_frame_equal(lf.df.reset_index(drop=True), expected_df)


def test_add_meta_covers_existing_meta(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)

    lf.add_meta({"experiment": {"name": "cooling"}})
    lf.add_meta_to_head(run=1, experiment={"operator": "alice"})

    with pytest.raises(KeyError):
        lf.reg["experiment"]["name"]
    assert lf.reg["experiment"]["operator"] == "alice"
    assert lf.reg["run"] == 1


def test_logfolder_index_persists_updates(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path, title="demo")
    lf.idx.star = 1
    lf.idx.trash = True

    reloaded = LogFolder(lf.path)
    assert reloaded.idx.star == 1
    assert reloaded.idx.trash is True
    assert reloaded.idx.title == "demo"


def test_load_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LogFolder(tmp_path / "nonexistent", create=False)
