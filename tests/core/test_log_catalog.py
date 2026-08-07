from pathlib import Path

import pandas as pd
import pytest

from logqbit import catalog as catalog_module
from logqbit.catalog import LogCatalog, LogRecord
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
    assert record.plot_axes == ()
    assert record.plot_fields == ()

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
    )

    assert record.title == "updated"
    assert record.star == 2
    assert record.trash is True
    assert record.plot_axes == ("x",)
    assert record.plot_fields == ("z",)


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
