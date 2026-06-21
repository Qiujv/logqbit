from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import weakref

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


def test_new_reserves_unique_directory_names(tmp_path: Path) -> None:
    parent = tmp_path / "logs"
    parent.mkdir()
    count = 20

    def create_logfolder() -> str:
        lf = LogFolder.new(parent)
        return lf.path.name

    with ThreadPoolExecutor(max_workers=8) as executor:
        names = list(executor.map(lambda _: create_logfolder(), range(count)))

    assert len(set(names)) == count
    assert sorted(map(int, names)) == list(range(count))


def test_add_row_scalar_and_save(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)
    lf.add_row(x=1.5, y=2.0)

    expected_df = pd.DataFrame([{"x": 1.5, "y": 2.0}])
    pd.testing.assert_frame_equal(lf.df.reset_index(drop=True), expected_df)

    lf.flush()

    assert lf.df_path.exists()
    saved_df = pd.read_feather(lf.df_path)
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

    lf.add_const({"experiment": {"name": "cooling"}})
    lf.add_const_to_head(run=1, experiment={"operator": "alice"})

    with pytest.raises(KeyError):
        lf.reg["experiment"]["name"]
    assert lf.reg["experiment"]["operator"] == "alice"
    assert lf.reg["run"] == 1


def test_logfolder_index_persists_updates(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path, title="demo")
    lf.meta.star = 1
    lf.meta.trash = True

    reloaded = LogFolder(lf.path)
    assert reloaded.meta.star == 1
    assert reloaded.meta.trash is True
    assert reloaded.meta.title == "demo"


def test_load_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LogFolder(tmp_path / "nonexistent", create=False)


def test_context_manager_closes_and_flushes(tmp_path: Path) -> None:
    with LogFolder.new(tmp_path) as lf:
        path = lf.df_path
        lf.add_row(x=1)

    saved_df = pd.read_feather(path)
    pd.testing.assert_frame_equal(saved_df.reset_index(drop=True), pd.DataFrame([{"x": 1}]))


def test_close_is_idempotent(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)
    lf.add_row(x=1)

    lf.close()
    lf.close()

    saved_df = pd.read_feather(lf.df_path)
    pd.testing.assert_frame_equal(saved_df.reset_index(drop=True), pd.DataFrame([{"x": 1}]))


def test_finalize_flushes_when_logfolder_is_collected(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)
    path = lf.df_path
    lf.add_row(x=1)
    ref = weakref.ref(lf)

    del lf
    gc.collect()

    assert ref() is None
    saved_df = pd.read_feather(path)
    pd.testing.assert_frame_equal(saved_df.reset_index(drop=True), pd.DataFrame([{"x": 1}]))
