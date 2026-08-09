from __future__ import annotations

from pathlib import Path

import pandas as pd

from logqbit.catalog import LogCatalog, LogRecord, export_records
from logqbit.gui.browser.detail.view import record_watch_paths
from logqbit.logfolder import LogFolder


def scan_catalog(directory: Path) -> list[LogRecord]:
    return LogCatalog(directory).refresh()


class TestLogRecord:
    """Tests for LogRecord class."""

    def test_scan_catalog_finds_logs(self, tmp_path: Path) -> None:
        """Test scanning a directory for log records."""
        with LogFolder.new(tmp_path, title="log1"):
            pass
        with LogFolder.new(tmp_path, title="log2"):
            pass

        records = scan_catalog(tmp_path)

        assert len(records) == 2
        assert all(isinstance(r, LogRecord) for r in records)
        assert {r.log_id for r in records} == {0, 1}

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning an empty directory."""
        records = scan_catalog(tmp_path)
        assert records == []

    def test_scan_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test scanning a directory that doesn't exist."""
        records = scan_catalog(tmp_path / "nonexistent")
        assert records == []

    def test_entry_reads_dataframe(self, sample_logfolder: Path) -> None:
        """Test loading dataframe through the passive entry."""
        records = scan_catalog(sample_logfolder)
        assert len(records) == 1

        record = records[0]
        df = record.read_dataframe()

        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["x", "y", "z"]
        assert record.row_count == 3
        assert record.columns == ("x", "y", "z")

    def test_read_yaml_text(self, sample_logfolder: Path) -> None:
        """Test reading YAML text from a log record."""
        records = scan_catalog(sample_logfolder)
        record = records[0]

        yaml_text = record.read_yaml_text()

        assert isinstance(yaml_text, str)
        assert len(yaml_text) > 0

    def test_read_yaml_missing_file(self, tmp_path: Path) -> None:
        """Test reading YAML when file doesn't exist."""
        with LogFolder.new(tmp_path) as lf:
            pd.DataFrame({"x": [1]}).to_feather(lf.df_path)

        records = scan_catalog(tmp_path)
        record = records[0]

        yaml_text = record.read_yaml_text()
        assert "const.yaml not found" in yaml_text

    def test_list_image_files(self, tmp_path: Path) -> None:
        """Test listing image files in a log folder."""
        with LogFolder.new(tmp_path) as lf:
            path = lf.path

        (path / "plot.png").touch()
        (path / "result.jpg").touch()
        (path / "data.txt").touch()

        records = scan_catalog(tmp_path)
        record = records[0]

        images = record.list_image_files()

        assert len(images) == 2
        assert all(img.suffix.lower() in {".png", ".jpg"} for img in images)

    def test_list_other_files(self, sample_logfolder: Path) -> None:
        """Test listing non-standard files in a log folder."""
        records = scan_catalog(sample_logfolder)
        record = records[0]

        extra_text = record.path / "notes.txt"
        extra_binary = record.path / "snapshot.bin"
        ignored_image = record.path / "plot.webp"
        extra_text.write_text("hello", encoding="utf-8")
        extra_binary.write_bytes(b"123")
        ignored_image.touch()

        other_files = record.list_other_files()

        assert [path.name for path in other_files] == ["notes.txt", "snapshot.bin"]

    def test_record_watch_paths_ignore_unrendered_extra_files(
        self,
        sample_logfolder: Path,
    ) -> None:
        records = scan_catalog(sample_logfolder)
        record = records[0]

        extra_file = record.path / "notes.txt"
        extra_file.write_text("watch me", encoding="utf-8")
        hidden_file = record.path / ".DS_Store"
        hidden_file.write_bytes(b"finder")
        image_file = record.path / "plot.png"
        image_file.touch()

        watch_paths = set(record_watch_paths(record))

        assert str(record.path) in watch_paths
        assert str(record.meta_path) in watch_paths
        assert str(image_file) in watch_paths
        assert str(extra_file) not in watch_paths
        assert str(hidden_file) not in watch_paths

    def test_export_records_copies_selected_logs_in_id_order(
        self, tmp_path: Path
    ) -> None:
        source_parent = tmp_path / "source"
        source_parent.mkdir()

        with LogFolder.new(source_parent, title="low") as low:
            low.add_row(x=1, y=2)
            (low.path / "note.txt").write_text("low-note", encoding="utf-8")

        with LogFolder.new(source_parent, title="high") as high:
            high.add_row(x=10, y=20)
            (high.path / "snapshot.bin").write_bytes(b"abc")
            (high.path / "import_from").write_text("preserve-me", encoding="utf-8")

        records = scan_catalog(source_parent)
        record_by_title = {record.title: record for record in records}

        destination_parent = tmp_path / "exported"
        destination_parent.mkdir()
        (destination_parent / "0").mkdir()

        exported_paths = export_records(
            [
                record_by_title["high"],
                record_by_title["low"],
            ],
            destination_parent,
        )

        assert [path.name for path in exported_paths] == ["1", "2"]
        assert (exported_paths[0] / "note.txt").read_text(
            encoding="utf-8"
        ) == "low-note"
        assert (exported_paths[1] / "snapshot.bin").read_bytes() == b"abc"
        assert (exported_paths[0] / "import_from").read_text(encoding="utf-8") == str(
            record_by_title["low"].path
        )
        assert (exported_paths[1] / "import_from").read_text(
            encoding="utf-8"
        ) == "preserve-me"

        exported_records = scan_catalog(destination_parent)
        exported_titles = {record.log_id: record.title for record in exported_records}
        assert exported_titles[1] == "low"
        assert exported_titles[2] == "high"
