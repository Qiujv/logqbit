from __future__ import annotations

from pathlib import Path

import pytest

from logqbit.browser import LogRecord
from logqbit.logfolder import LogFolder


@pytest.fixture
def sample_logfolder_lf(tmp_path: Path) -> LogFolder:
    lf = LogFolder.new(tmp_path, title="test_log")
    lf.add_row(x=1.0, y=2.0, z=3.0)
    lf.flush()
    return lf  # 返回 LogFolder 对象

def test_load_dataframe_lf(sample_logfolder_lf: LogFolder):
    records = LogRecord.scan_directory(sample_logfolder_lf.path.parent)
    df = records[0].load_dataframe()
    assert df is not None  # ❌ 失败! df 是 None
