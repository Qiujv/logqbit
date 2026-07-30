from pathlib import Path

import pandas as pd
import pytest

from logqbit.gui import log_catalog as log_catalog_module
from logqbit.gui.log_catalog import LogCatalog
from logqbit.logfolder import LogFolder


@pytest.fixture
def sample_logfolder(tmp_path: Path) -> Path:
    logfolder = LogFolder.new(tmp_path, title="test_log")
    logfolder.add_row(x=1.0, y=2.0, z=3.0)
    logfolder.add_row(x=1.5, y=2.5, z=3.5)
    logfolder.add_row(x=2.0, y=3.0, z=4.0)
    logfolder.flush()
    return tmp_path


def test_refresh_reuses_unchanged_records(
    sample_logfolder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = LogCatalog()
    first_record = catalog.refresh(sample_logfolder)[0]
    dataframe = first_record.load_dataframe()

    def fail_open_file(*args, **kwargs):
        raise AssertionError("unchanged Feather should not be inspected")

    monkeypatch.setattr(
        log_catalog_module.pyarrow.ipc, "open_file", fail_open_file
    )

    second_record = catalog.refresh(sample_logfolder)[0]

    assert second_record is first_record
    assert second_record.load_dataframe() is dataframe


def test_refresh_invalidates_only_changed_feather(
    sample_logfolder: Path,
) -> None:
    catalog = LogCatalog()
    record = catalog.refresh(sample_logfolder)[0]
    assert record.load_dataframe() is not None

    pd.DataFrame({"x": range(10), "y": range(10)}).to_feather(record.data_path)

    refreshed_record = catalog.refresh(sample_logfolder)[0]

    assert refreshed_record is record
    assert refreshed_record.data_frame is None
    assert refreshed_record.row_count == 10
    assert refreshed_record.columns == ["x", "y"]
