from __future__ import annotations

from pathlib import Path

import pytest

from logqbit.catalog import LogCatalog, LogRecord
from logqbit.logfolder import LogFolder


@pytest.fixture
def sample_logfolder(tmp_path: Path) -> Path:
    """Create a sample log folder with data and return its parent directory."""
    with LogFolder.new(tmp_path, title="test_log") as log:
        log.add_row(x=1.0, y=2.0, z=3.0)
        log.add_row(x=1.5, y=2.5, z=3.5)
        log.add_row(x=2.0, y=3.0, z=4.0)
        log.meta.star = 1
        log.meta.plot_axes = ["x", "y"]
    return tmp_path


@pytest.fixture
def sample_records(tmp_path: Path) -> list[LogRecord]:
    """Create multiple sample log records."""
    with LogFolder.new(tmp_path, title="log_zero") as log:
        log.add_row(a=1, b=2)

    with LogFolder.new(tmp_path, title="log_one") as log:
        log.add_row(x=10, y=20)
        log.meta.star = 2

    with LogFolder.new(tmp_path, title="log_two") as log:
        log.add_row(p=100, q=200)
        log.meta.trash = True

    return LogCatalog(tmp_path).refresh()
