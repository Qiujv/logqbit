from pathlib import Path

import pandas as pd
import pytest

from logqbit import catalog as catalog_module
from logqbit.catalog import (
    LOGFOLDER_SID_COLUMN,
    LogCatalog,
    LogRecord,
    MergeRecordsError,
    PlotColumns,
    PreparedMerge,
    append_records_into_record,
    merge_records_into_new,
    resolve_plot_columns,
)
from logqbit.logfolder import LogFolder
from logqbit.metadata import LogMetadata


@pytest.fixture
def sample_logfolder(tmp_path: Path) -> Path:
    with LogFolder.new(tmp_path, title="test_log") as logfolder:
        logfolder.add_row(x=1.0, y=2.0, z=3.0)
        logfolder.add_row(x=1.5, y=2.5, z=3.5)
        logfolder.add_row(x=2.0, y=3.0, z=4.0)
    return tmp_path


def test_refresh_reuses_unchanged_records(
    sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = LogCatalog()
    first_record = catalog.refresh(sample_logfolder)[0]

    def fail_open_file(*args, **kwargs):
        raise AssertionError("unchanged Feather should not be inspected")

    monkeypatch.setattr(catalog_module.pyarrow.ipc, "open_file", fail_open_file)

    second_record = catalog.refresh(sample_logfolder)[0]

    assert second_record is first_record


def test_resolve_plot_columns_applies_preferences_and_defaults() -> None:
    assert resolve_plot_columns(
        ["x", "y", "signal", "reference"],
        ["missing", "y", "y"],
        ["y", "signal", "signal"],
    ) == PlotColumns(
        axes=("y",),
        fields=("signal",),
        groupby=(),
        ignored=("x", "reference"),
    )

    assert resolve_plot_columns(["x", "signal", "reference"], [], []) == PlotColumns(
        axes=("x",),
        fields=("signal",),
        groupby=(),
        ignored=("reference",),
    )
    assert resolve_plot_columns(["x", "signal"], "x", "signal") == PlotColumns(
        axes=("x",),
        fields=("signal",),
        groupby=(),
        ignored=(),
    )

    assert resolve_plot_columns(
        ["device", "x", "signal", "note"],
        ["device", "x"],
        ["signal", "device"],
        ["missing", "device", "device"],
    ) == PlotColumns(
        axes=("x",),
        fields=("signal",),
        groupby=("device",),
        ignored=("note",),
    )


def test_catalog_uses_metadata_file_as_log_directory_marker(
    tmp_path: Path,
) -> None:
    named_path = tmp_path / "named-run"
    LogFolder(named_path, title="named")
    numeric_without_metadata = tmp_path / "123"
    numeric_without_metadata.mkdir()

    records = LogCatalog(tmp_path).refresh()

    assert [record.path for record in records] == [named_path]
    assert records[0].title == "named"


def test_refresh_orders_numeric_then_named_directories(tmp_path: Path) -> None:
    for name in ("10", "beta", "2", "Alpha", "1"):
        LogFolder(tmp_path / name)

    records = LogCatalog(tmp_path).refresh()

    assert [record.path.name for record in records] == [
        "1",
        "2",
        "10",
        "Alpha",
        "beta",
    ]


def test_refresh_replaces_only_changed_record(sample_logfolder: Path) -> None:
    catalog = LogCatalog()
    record = catalog.refresh(sample_logfolder)[0]

    pd.DataFrame({"x": range(10), "y": range(10)}).to_feather(record.data_path)
    refreshed = catalog.refresh(sample_logfolder)[0]

    assert refreshed is not record
    assert refreshed.row_count == 10
    assert refreshed.columns == ("x", "y")


def test_refresh_retries_data_inspection_after_failure(
    sample_logfolder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = LogCatalog()
    original_open_file = catalog_module.pyarrow.ipc.open_file

    def fail_open_file(*args, **kwargs):
        raise OSError("temporary read failure")

    monkeypatch.setattr(catalog_module.pyarrow.ipc, "open_file", fail_open_file)
    failed = catalog.refresh(sample_logfolder)[0]

    assert failed.row_count == 0
    assert failed.columns == ()
    assert failed.data_version is not None

    monkeypatch.setattr(
        catalog_module.pyarrow.ipc,
        "open_file",
        original_open_file,
    )
    recovered = catalog.refresh(sample_logfolder)[0]

    assert recovered is not failed
    assert recovered.row_count == 3
    assert recovered.columns == ("x", "y", "z")


def test_record_uses_editable_defaults_for_invalid_metadata(
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "broken"
    record_path.mkdir()
    metadata_path = record_path / "metadata.json"
    metadata_path.write_text("{invalid", encoding="utf-8")

    record = LogCatalog(tmp_path).refresh()[0]

    assert record.title == "<invalid metadata>"
    assert record.star == 0
    assert record.trash is False
    assert record.meta.plot_axes == ()
    assert record.meta.plot_fields == ()
    assert record.meta.plot_groupby == ()
    assert record.resolved_plot_columns == PlotColumns((), (), (), ())

    record.meta.update(title="repaired")

    assert record.title == "repaired"
    assert LogMetadata(metadata_path, create=False).title == "repaired"


def test_record_reads_dataframe_without_catalog_cache(
    sample_logfolder: Path,
) -> None:
    record = LogCatalog(sample_logfolder).refresh()[0]

    first = record.read_dataframe()
    second = record.read_dataframe()

    assert first is not None
    assert second is not None
    assert first is not second
    pd.testing.assert_frame_equal(first, second)


def test_record_reads_dataframe_from_memory_buffer(
    sample_logfolder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = LogCatalog(sample_logfolder).refresh()[0]
    original_read_feather = catalog_module.pd.read_feather
    sources = []

    def read_feather(source, *args, **kwargs):
        sources.append(source)
        return original_read_feather(source, *args, **kwargs)

    monkeypatch.setattr(catalog_module.pd, "read_feather", read_feather)

    assert record.read_dataframe() is not None
    assert len(sources) == 1
    assert isinstance(sources[0], catalog_module.pyarrow.BufferReader)


def test_catalog_inspects_data_from_memory_buffer(
    sample_logfolder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_file = catalog_module.pyarrow.ipc.open_file
    sources = []

    def open_file(source, *args, **kwargs):
        sources.append(source)
        return original_open_file(source, *args, **kwargs)

    monkeypatch.setattr(catalog_module.pyarrow.ipc, "open_file", open_file)

    records = LogCatalog(sample_logfolder).refresh()

    assert len(records) == 1
    assert len(sources) == 1
    assert isinstance(sources[0], catalog_module.pyarrow.BufferReader)


def test_record_reads_current_in_memory_metadata(
    sample_logfolder: Path,
) -> None:
    record = LogCatalog(sample_logfolder).refresh()[0]

    record.meta.update(
        title="updated",
        star=2,
        trash=True,
        plot_axes=["x"],
        plot_fields=["z"],
        plot_groupby=["y"],
    )

    assert record.title == "updated"
    assert record.star == 2
    assert record.trash is True
    assert record.meta.plot_axes == ("x",)
    assert record.meta.plot_fields == ("z",)
    assert record.meta.plot_groupby == ("y",)
    assert record.resolved_plot_columns == PlotColumns(
        axes=("x",),
        fields=("z",),
        groupby=("y",),
        ignored=(),
    )


def test_metadata_refresh_resolves_plot_columns_without_inspecting_data(
    sample_logfolder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = LogCatalog(sample_logfolder)
    record = catalog.refresh()[0]
    LogMetadata(record.meta_path, create=False).update(
        plot_axes=["missing", "z"],
        plot_fields=["x"],
    )

    def fail_open_file(*args, **kwargs):
        raise AssertionError("metadata-only refresh should not inspect Feather")

    monkeypatch.setattr(catalog_module.pyarrow.ipc, "open_file", fail_open_file)

    refreshed = catalog.refresh()[0]

    assert refreshed is record
    assert refreshed.resolved_plot_columns == PlotColumns(
        axes=("z",),
        fields=("x",),
        groupby=(),
        ignored=("y",),
    )


def test_logfolder_and_record_share_metadata_interface(tmp_path: Path) -> None:
    with LogFolder.new(tmp_path, title="writer") as logfolder:
        logfolder.meta.update(title="from-client", star=1)

        record = LogRecord(logfolder.path)
        assert record.meta.title == "from-client"

        record.meta.update(title="from-record", star=3)

        logfolder.meta.reload()
        assert logfolder.meta.title == "from-record"
        assert logfolder.meta.star == 3


def test_log_record_construction_has_no_file_creation(tmp_path: Path) -> None:
    path = tmp_path / "0"
    path.mkdir()

    record = LogRecord(path)

    assert record.path == path
    assert not record.meta_path.exists()
    assert not record.data_path.exists()
    assert not record.const_path.exists()


def _create_merge_record(
    parent: Path,
    dataframe: pd.DataFrame,
    *,
    title: str,
    plot_groupby: tuple[str, ...] = (),
) -> LogRecord:
    with LogFolder.new(parent, title=title) as logfolder:
        logfolder.add_df(dataframe)
        logfolder.meta.update(
            plot_axes=("x",),
            plot_fields=("value",),
            plot_groupby=plot_groupby,
        )
    return LogCatalog(parent).refresh()[-1]


def test_merge_records_into_new_copies_metadata_const_and_adds_sid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [1, 2], "value": [10, 20]}),
        title="first",
    )
    first.const_path.write_text("temperature: 20\n", encoding="utf-8")
    second = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [3], "value": [30], "extra": [300]}),
        title="second",
    )

    prepared = PreparedMerge.for_new_folder([first, second], tmp_path)

    assert prepared.staging_path is not None
    assert prepared.staging_path.is_dir()
    assert sorted(
        path.name for path in tmp_path.iterdir() if path.name.isdecimal()
    ) == [
        "0",
        "1",
    ]
    assert [record.path for record in LogCatalog(tmp_path).refresh()] == [
        first.path,
        second.path,
    ]
    monkeypatch.setattr(
        pd.DataFrame,
        "to_feather",
        lambda *_args, **_kwargs: pytest.fail(
            "publishing must not serialize the dataframe again"
        ),
    )

    result = prepared.publish()

    assert not prepared.staging_path.exists()

    merged = LogRecord(result.path)
    dataframe = merged.read_dataframe()
    assert dataframe is not None
    assert dataframe.index.tolist() == [0, 1, 2]
    assert dataframe[LOGFOLDER_SID_COLUMN].astype(str).tolist() == ["0", "0", "1"]
    assert set(dataframe.columns) == {"x", "value", "extra", LOGFOLDER_SID_COLUMN}
    assert merged.title == "first"
    assert merged.meta.plot_axes == ("x",)
    assert merged.meta.plot_fields == ("value",)
    assert merged.meta.plot_groupby == (LOGFOLDER_SID_COLUMN,)
    assert merged.const_path.read_text(encoding="utf-8") == "temperature: 20\n"
    assert result.created is True
    assert result.appended_records == 2


def test_merge_records_into_new_rejects_overlapping_source_ids(
    tmp_path: Path,
) -> None:
    aggregate = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [1], "value": [10], LOGFOLDER_SID_COLUMN: ["1"]}),
        title="aggregate",
    )
    raw = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [2], "value": [20]}),
        title="raw",
    )

    with pytest.raises(MergeRecordsError, match="both contain source ID '1'"):
        merge_records_into_new([aggregate, raw], tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["0", "1"]


def test_merge_records_requires_two_common_columns_excluding_sid(
    tmp_path: Path,
) -> None:
    first = _create_merge_record(
        tmp_path,
        pd.DataFrame({"value": [10], LOGFOLDER_SID_COLUMN: ["first"]}),
        title="first",
    )
    second = _create_merge_record(
        tmp_path,
        pd.DataFrame({"value": [20]}),
        title="second",
    )

    with pytest.raises(
        MergeRecordsError,
        match=r"at least two common data columns \(found: value\)",
    ):
        merge_records_into_new([first, second], tmp_path)
    with pytest.raises(
        MergeRecordsError,
        match=r"at least two common data columns \(found: value\)",
    ):
        append_records_into_record([first, second])


def test_merge_records_into_new_fails_before_creating_target_for_missing_data(
    tmp_path: Path,
) -> None:
    first = _create_merge_record(
        tmp_path,
        pd.DataFrame({"value": [10]}),
        title="first",
    )
    second = _create_merge_record(
        tmp_path,
        pd.DataFrame({"value": [20]}),
        title="second",
    )
    second.data_path.unlink()

    with pytest.raises(MergeRecordsError, match="has no data file"):
        merge_records_into_new([first, second], tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["0", "1"]


def test_append_records_into_existing_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [1], "value": [10], LOGFOLDER_SID_COLUMN: ["previous"]}),
        title="target",
    )
    source = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [2, 3], "value": [20, 30]}),
        title="source",
    )

    target_before = target.data_path.read_bytes()

    prepared = PreparedMerge.for_append([source, target])

    assert target.data_path.read_bytes() == target_before
    assert prepared.appended_records == 1
    assert prepared.staging_path is not None
    assert prepared.staging_path.is_file()
    monkeypatch.setattr(
        pd.DataFrame,
        "to_feather",
        lambda *_args, **_kwargs: pytest.fail(
            "publishing must not serialize the dataframe again"
        ),
    )

    result = prepared.publish()

    assert not prepared.staging_path.exists()

    dataframe = pd.read_feather(target.data_path)
    assert dataframe["value"].tolist() == [10, 20, 30]
    assert dataframe[LOGFOLDER_SID_COLUMN].astype(str).tolist() == [
        "previous",
        "1",
        "1",
    ]
    assert target.meta.plot_groupby == (LOGFOLDER_SID_COLUMN,)
    assert result.created is False
    assert result.appended_records == 1
    assert result.skipped_records == 0


def test_append_is_idempotent_and_preserves_existing_groupby(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _create_merge_record(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1],
                "value": [10],
                LOGFOLDER_SID_COLUMN: pd.Series(["1"], dtype="string"),
            }
        ),
        title="target",
        plot_groupby=("device",),
    )
    source = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [2], "value": [20]}),
        title="source",
    )
    original_version = target.data_path.stat().st_mtime_ns

    monkeypatch.setattr(
        catalog_module.pd,
        "concat",
        lambda *_args, **_kwargs: pytest.fail("no-op must not concatenate data"),
    )

    prepared = PreparedMerge.for_append([target, source])

    assert prepared.is_noop
    result = prepared.publish()

    assert result.appended_records == 0
    assert result.skipped_records == 1
    assert target.data_path.stat().st_mtime_ns == original_version
    assert target.meta.plot_groupby == ("device",)


def test_append_preserves_nonempty_groupby(
    tmp_path: Path,
) -> None:
    target = _create_merge_record(
        tmp_path,
        pd.DataFrame(
            {
                "value": [10],
                "device": ["a"],
                LOGFOLDER_SID_COLUMN: ["previous"],
            }
        ),
        title="target",
        plot_groupby=("device",),
    )
    source = _create_merge_record(
        tmp_path,
        pd.DataFrame({"value": [20], "device": ["b"]}),
        title="source",
    )

    result = append_records_into_record([target, source])

    assert result.appended_records == 1
    assert target.meta.plot_groupby == ("device",)


def test_append_aborts_when_a_source_changes_after_prepare(
    tmp_path: Path,
) -> None:
    target = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [1], "value": [10], LOGFOLDER_SID_COLUMN: ["previous"]}),
        title="target",
    )
    source = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [2], "value": [20]}),
        title="source",
    )
    target_before = target.data_path.read_bytes()
    prepared = PreparedMerge.for_append([target, source])
    pd.DataFrame({"x": [9], "value": [99]}).to_feather(source.data_path)

    with pytest.raises(MergeRecordsError, match="changed during the merge"):
        prepared.publish()

    assert target.data_path.read_bytes() == target_before
    prepared.discard()


def test_append_requires_exactly_one_sid_record(tmp_path: Path) -> None:
    first = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [1], "value": [10]}),
        title="first",
    )
    second = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [2], "value": [20]}),
        title="second",
    )

    with pytest.raises(MergeRecordsError, match="exactly one selected record"):
        append_records_into_record([first, second])

    first_aggregate = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [3], "value": [30], LOGFOLDER_SID_COLUMN: ["first"]}),
        title="first aggregate",
    )
    second_aggregate = _create_merge_record(
        tmp_path,
        pd.DataFrame({"x": [4], "value": [40], LOGFOLDER_SID_COLUMN: ["second"]}),
        title="second aggregate",
    )

    with pytest.raises(MergeRecordsError, match="exactly one selected record"):
        append_records_into_record([first_aggregate, second_aggregate])
