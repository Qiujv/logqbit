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

    with LogFolder.new(parent) as lf:
        assert lf.path.parent == parent
        assert lf.path.name == "2"


def test_new_reserves_unique_directory_names(tmp_path: Path) -> None:
    parent = tmp_path / "logs"
    parent.mkdir()
    count = 20

    def create_logfolder() -> str:
        with LogFolder.new(parent) as lf:
            return lf.path.name

    with ThreadPoolExecutor(max_workers=8) as executor:
        names = list(executor.map(lambda _: create_logfolder(), range(count)))

    assert len(set(names)) == count
    assert sorted(map(int, names)) == list(range(count))


def test_add_row_scalar_and_save(tmp_path: Path) -> None:
    expected_df = pd.DataFrame([{"x": 1.5, "y": 2.0}])
    with LogFolder.new(tmp_path) as lf:
        path = lf.path
        lf.add_row(x=1.5, y=2.0)
        pd.testing.assert_frame_equal(lf.df.reset_index(drop=True), expected_df)

        lf.flush()
        assert lf.df_path.exists()
        saved_df = pd.read_feather(lf.df_path)
        pd.testing.assert_frame_equal(saved_df.reset_index(drop=True), expected_df)

    with LogFolder(path, create=False) as loaded:
        pd.testing.assert_frame_equal(loaded.df.reset_index(drop=True), expected_df)


def test_add_row_vector_creates_dataframe(tmp_path: Path) -> None:
    expected_df = pd.DataFrame({"step": [0, 1, 2], "current": [0.1, 0.2, 0.3]})
    with LogFolder.new(tmp_path) as lf:
        lf.add_row(step=[0, 1, 2], current=[0.1, 0.2, 0.3])
        pd.testing.assert_frame_equal(lf.df.reset_index(drop=True), expected_df)


def test_add_df_and_df_property_do_not_share_mutable_data(tmp_path: Path) -> None:
    source = pd.DataFrame({"step": [0, 1], "current": [0.1, 0.2]})
    expected = source.copy()
    with LogFolder.new(tmp_path) as lf:
        path = lf.df_path
        lf.add_df(source)
        source.loc[0, "current"] = 99

        snapshot = lf.df
        snapshot.loc[1, "current"] = 88

        pd.testing.assert_frame_equal(lf.df, expected)

    pd.testing.assert_frame_equal(pd.read_feather(path), expected)


def test_add_meta_covers_existing_meta(tmp_path: Path) -> None:
    with LogFolder.new(tmp_path) as lf:
        lf.add_const({"experiment": {"name": "cooling"}})
        lf.add_const_to_head(run=1, experiment={"operator": "alice"})

        with pytest.raises(KeyError):
            lf.reg["experiment"]["name"]
        assert lf.reg["experiment"]["operator"] == "alice"
        assert lf.reg["run"] == 1


def test_logfolder_index_persists_updates(tmp_path: Path) -> None:
    with LogFolder.new(tmp_path, title="demo") as lf:
        path = lf.path
        lf.meta.star = 1
        lf.meta.trash = True

    with LogFolder(path, create=False) as reloaded:
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


def test_context_manager_closes_after_body_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="experiment failed"):
        with LogFolder.new(tmp_path) as lf:
            path = lf.df_path
            lf.add_row(x=1)
            raise ValueError("experiment failed")

    assert lf.closed
    pd.testing.assert_frame_equal(pd.read_feather(path), pd.DataFrame({"x": [1]}))


def test_close_is_idempotent(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)
    lf.add_row(x=1)

    lf.close()
    lf.close()

    saved_df = pd.read_feather(lf.df_path)
    pd.testing.assert_frame_equal(saved_df.reset_index(drop=True), pd.DataFrame([{"x": 1}]))


def test_close_empty_logfolder_does_not_create_feather(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)

    lf.close()

    assert not lf.df_path.exists()
    assert lf.df.empty


def test_closed_logfolder_is_readable_but_rejects_new_data(tmp_path: Path) -> None:
    lf = LogFolder.new(tmp_path)
    lf.add_row(x=1)
    lf.close()

    assert lf.closed
    pd.testing.assert_frame_equal(lf.df, pd.DataFrame({"x": [1]}))
    message = "create a new LogFolder"
    with pytest.raises(RuntimeError, match=message):
        lf.add_row(x=2)
    with pytest.raises(RuntimeError, match=message):
        lf.add_df(pd.DataFrame({"x": [2]}))
    with pytest.raises(RuntimeError, match=message):
        lf.add_row(x=[1, 2], y=[1])
    with pytest.raises(RuntimeError, match=message):
        lf.capture(lambda x: {"y": x}, {"x": [1, 2]})

    pd.testing.assert_frame_equal(lf.df, pd.DataFrame({"x": [1]}))
    assert not (lf.path / "const.yaml").exists()
    assert lf.meta.plot_axes == ()

    lf.meta.title = "closed snapshot"
    lf.add_const(note="still editable")
    assert lf.meta.title == "closed snapshot"
    assert lf.reg["note"] == "still editable"


def test_logfolder_rejects_empty_row_and_ignores_empty_dataframe(
    tmp_path: Path,
) -> None:
    with LogFolder.new(tmp_path) as lf:
        with pytest.raises(ValueError, match="without any columns"):
            lf.add_row()
        lf.add_df(pd.DataFrame({"x": []}))
        path = lf.df_path

    assert not path.exists()


def test_only_one_active_writer_can_own_a_logfolder(tmp_path: Path) -> None:
    first = LogFolder.new(tmp_path)
    path = first.path
    try:
        first.add_row(x=1)
        with pytest.raises(RuntimeError, match="active writer"):
            LogFolder(path, create=False)
    finally:
        first.close()
    first_snapshot = first.df

    with LogFolder(path, create=False) as reopened:
        reopened.add_row(x=2)

    pd.testing.assert_frame_equal(
        pd.read_feather(path / "data.feather"),
        pd.DataFrame({"x": [1, 2]}),
    )
    pd.testing.assert_frame_equal(first_snapshot, pd.DataFrame({"x": [1]}))
    pd.testing.assert_frame_equal(first.df, first_snapshot)


def test_capture_records_data_axes_and_constants(tmp_path: Path) -> None:
    with LogFolder.new(tmp_path) as lf:
        lf.capture(
            lambda x, bias: {"y": 2 * x + bias},
            {"x": [1, 2], "bias": 0.5},
        )

        pd.testing.assert_frame_equal(
            lf.df,
            pd.DataFrame({"x": [1, 2], "y": [2.5, 4.5]}),
        )
        assert lf.meta.plot_axes == ("x",)
        assert lf.reg["const"] == {"bias": 0.5}
        assert lf.reg["dims"] == {"x": 2}


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

    with LogFolder(path.parent, create=False) as reopened:
        assert not reopened.closed
