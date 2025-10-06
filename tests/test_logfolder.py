from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from logqbit.client import LogFolder


def test_new_creates_incremental_directory(tmp_path: Path) -> None:
	parent = tmp_path / "logs"
	parent.mkdir()
	(parent / "0").mkdir()
	(parent / "1").mkdir()

	log_folder = LogFolder.new(parent)

	assert log_folder.path.parent == parent
	assert log_folder.path.name == "2"
	assert log_folder.meta["create_machine"] == LogFolder.CREATE_MACHINE
	assert "create_time" in log_folder.meta


def test_add_row_scalar_and_save(tmp_path: Path) -> None:
	log_folder = LogFolder.new(tmp_path)
	log_folder.add_row(temperature=1.5, voltage=2.0)

	expected_df = pd.DataFrame([{"temperature": 1.5, "voltage": 2.0}])
	pd.testing.assert_frame_equal(log_folder.df.reset_index(drop=True), expected_df)

	log_folder.save()

	assert log_folder.meta_path.exists()
	with log_folder.meta_path.open("r", encoding="utf-8") as meta_file:
		meta = json.load(meta_file)

	assert meta["create_machine"] == LogFolder.CREATE_MACHINE
	assert "create_time" in meta

	assert log_folder.data_path.exists()
	saved_df = pd.read_parquet(log_folder.data_path)
	pd.testing.assert_frame_equal(saved_df.reset_index(drop=True), expected_df)

	loaded = LogFolder.load(log_folder.path)
	assert loaded.meta["create_machine"] == LogFolder.CREATE_MACHINE
	assert "create_time" in loaded.meta


def test_add_row_vector_creates_dataframe(tmp_path: Path) -> None:
	log_folder = LogFolder.new(tmp_path)

	log_folder.add_row(step=[0, 1, 2], current=[0.1, 0.2, 0.3])

	expected_df = pd.DataFrame(
		{
			"step": [0, 1, 2],
			"current": [0.1, 0.2, 0.3],
		}
	)

	df = log_folder.df.reset_index(drop=True)
	pd.testing.assert_frame_equal(df, expected_df)


def test_meta_merge_preserves_existing_metadata(tmp_path: Path) -> None:
	log_folder = LogFolder.new(tmp_path)

	log_folder.add_meta({"experiment": {"name": "cooling"}})
	log_folder.add_meta(run=1, experiment={"operator": "alice"})

	assert log_folder.meta["experiment"]["name"] == "cooling"
	assert log_folder.meta["experiment"]["operator"] == "alice"
	assert log_folder.meta["run"] == 1
	assert "create_machine" in log_folder.meta
	assert "create_time" in log_folder.meta


def test_load_raises_for_missing_directory(tmp_path: Path) -> None:
	with pytest.raises(FileNotFoundError):
		LogFolder.load(tmp_path / "nonexistent")
