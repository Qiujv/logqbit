# %% [markdown]
# # Benchmark: LogFolder Flush Latency vs Row Count
#
# Tests how long `LogFolder.flush()` takes for 100 / 1k / 10k / 100k / 1M /
# 10M rows, each with 10 float64 columns.
#
# The comparison window mirrors `DataFrameBuffer`'s adaptive autosave interval:
# a synchronous flush should normally finish faster than the next autosave tick.

# %%
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from logqbit.logfolder import LogFolder  # noqa: E402


COLS = [f"c{i}" for i in range(10)]  # 10 float64 columns
DEFAULT_ROW_COUNTS = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
CHUNK_ROWS = 100_000
BYTES_PER_ROW = len(COLS) * 8  # float64 = 8 bytes


def row_counts_from_env() -> list[int]:
    raw = os.environ.get("LOGQBIT_BENCH_FLUSH_ROWS")
    if raw is None:
        return DEFAULT_ROW_COUNTS
    return [int(part.strip().replace("_", "")) for part in raw.split(",") if part.strip()]


def expected_autosave_interval(n: int) -> float:
    if n < 1_000:
        return 0.1
    if n < 10_000:
        return 0.2
    if n < 100_000:
        return 0.5
    return 1.0


def fmt_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def add_random_rows(logfolder: LogFolder, row_count: int) -> None:
    rng = np.random.default_rng(42)
    rows_left = row_count

    while rows_left > 0:
        batch = min(CHUNK_ROWS, rows_left)
        logfolder.add_row(**{column: rng.random(batch) for column in COLS})
        rows_left -= batch


# %%
results = []

print(
    f"{'Rows':>10}  {'Flush (s)':>10}  {'Autosave':>10}  {'Safe?':>6}  "
    f"{'File size':>10}  {'Raw data':>10}"
)
print("-" * 68)

with tempfile.TemporaryDirectory() as tmp_root:
    for row_count in row_counts_from_env():
        folder = Path(tmp_root) / f"bench_{row_count}"

        with LogFolder(folder, title="bench", create=True) as logfolder:
            add_random_rows(logfolder, row_count)

            t0 = time.perf_counter()
            logfolder.flush()
            elapsed = time.perf_counter() - t0

            file_bytes = logfolder.df_path.stat().st_size if logfolder.df_path.exists() else 0
            interval = expected_autosave_interval(row_count)
            safe = "yes" if elapsed < interval else "no"

            print(
                f"{row_count:>10,}  {elapsed:>10.3f}  {interval:>10.1f}  "
                f"{safe:>6}  {fmt_bytes(file_bytes):>10}  "
                f"{fmt_bytes(row_count * BYTES_PER_ROW):>10}"
            )

            results.append(
                {
                    "rows": row_count,
                    "elapsed": elapsed,
                    "autosave_interval": interval,
                    "file_bytes": file_bytes,
                }
            )

# %%
#       Rows   Flush (s)    Autosave   Safe?   File size    Raw data
# --------------------------------------------------------------------
#        100       0.165         0.1      no     12.5 KB      7.8 KB
#      1,000       0.001         0.2     yes     82.8 KB     78.1 KB
#     10,000       0.001         0.5     yes    786.0 KB    781.2 KB
#    100,000       0.003         1.0     yes      7.6 MB      7.6 MB
#  1,000,000       0.066         1.0     yes     80.1 MB     76.3 MB
# 10,000,000       0.484         1.0     yes    801.3 MB    762.9 MB
