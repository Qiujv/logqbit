"""Interactive browser for experiment log folders."""

from __future__ import annotations

import logging
import numbers
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import pyarrow.parquet as pq
from PySide6.QtCore import (
	QAbstractTableModel,
	QFileSystemWatcher,
	QModelIndex,
	Qt,
	QTimer,
)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
	QApplication,
	QFileDialog,
	QHBoxLayout,
	QHeaderView,
	QLabel,
	QMainWindow,
	QMenu,
	QMessageBox,
	QPlainTextEdit,
	QPushButton,
	QSizePolicy,
	QSplitter,
	QTableView,
	QTableWidget,
	QTableWidgetItem,
	QTabWidget,
	QVBoxLayout,
	QWidget,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_TABLE_ROWS = 50000
REFRESH_DEBOUNCE_MS = 250


@dataclass
class IndexInfo:
	starred: bool = False
	trashed: bool = False
	title: str = ""
	extra_lines: List[str] = field(default_factory=list)


@dataclass
class ExperimentRecord:
	experiment_id: int
	path: Path
	index_info: IndexInfo
	row_count: Optional[int]
	data_path: Optional[Path]
	meta_path: Optional[Path]
	index_path: Optional[Path]


def _read_index_file(path: Path) -> IndexInfo:
	info = IndexInfo()
	if not path.exists():
		return info
	extra_lines: List[str] = []
	try:
		for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
			line = raw_line.strip()
			if not line or line.startswith("#"):
				extra_lines.append(raw_line)
				continue
			if ":" not in line:
				extra_lines.append(raw_line)
				continue
			key, value = line.split(":", 1)
			key = key.strip().lower()
			value = value.strip()
			if key == "title":
				info.title = value
			elif key == "starred":
				info.starred = value.lower() in {"1", "true", "yes", "y"}
			elif key == "trashed":
				info.trashed = value.lower() in {"1", "true", "yes", "y"}
			else:
				extra_lines.append(raw_line)
	except Exception as exc:  # pragma: no cover - defensive
		logger.warning("Failed to parse index file %s: %s", path, exc)
		return info
	info.extra_lines = extra_lines
	return info


def _write_index_file(path: Path, info: IndexInfo) -> None:
	lines: List[str] = []
	if info.title:
		lines.append(f"title: {info.title}")
	lines.append("starred: yes" if info.starred else "starred: no")
	lines.append("trashed: yes" if info.trashed else "trashed: no")
	if info.extra_lines:
		if lines and info.extra_lines[0].strip():
			lines.append("")
		lines.extend(info.extra_lines)
	text = "\n".join(lines)
	if text and not text.endswith("\n"):
		text += "\n"
	path.write_text(text, encoding="utf-8")


def _detect_data_file(folder: Path) -> Optional[Path]:
	for candidate in ("data.parquent", "data.parquet"):
		path = folder / candidate
		if path.exists():
			return path
	return None


def _detect_meta_file(folder: Path) -> Optional[Path]:
	path = folder / "meta.yaml"
	return path if path.exists() else None


def _detect_index_file(folder: Path) -> Optional[Path]:
	path = folder / "index"
	return path if path.exists() else None


def _list_image_files(folder: Path) -> List[Path]:
	files: List[Path] = []
	for child in folder.iterdir():
		if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
			files.append(child)
	files.sort()
	return files


def _read_meta_text(path: Optional[Path]) -> str:
	if not path:
		return "meta.yaml not found."
	try:
		return path.read_text(encoding="utf-8", errors="ignore")
	except Exception as exc:  # pragma: no cover - defensive
		logger.error("Failed to read meta file %s: %s", path, exc)
		return f"Failed to read meta.yaml: {exc}"


def _read_row_count(data_path: Optional[Path]) -> Optional[int]:
	if not data_path:
		return None
	try:
		return pq.ParquetFile(data_path).metadata.num_rows
	except Exception as exc:  # pragma: no cover - defensive
		logger.warning("Failed to read row count from %s: %s", data_path, exc)
		return None


def _load_dataframe(data_path: Optional[Path]) -> Optional[pd.DataFrame]:
	if not data_path:
		return None
	try:
		df = pd.read_parquet(data_path)
		if len(df) > MAX_TABLE_ROWS:
			return df.head(MAX_TABLE_ROWS)
		return df
	except Exception as exc:  # pragma: no cover - defensive
		logger.error("Failed to read parquet file %s: %s", data_path, exc)
		return None


class PandasTableModel(QAbstractTableModel):
	def __init__(self, frame: pd.DataFrame, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self._frame = frame

	def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
		return 0 if parent.isValid() else len(self._frame.index)

	def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
		return 0 if parent.isValid() else len(self._frame.columns)

	def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: D401
		if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
			return None
		value = self._frame.iat[index.row(), index.column()]
		if pd.isna(value):
			return ""
		if isinstance(value, numbers.Real) and not isinstance(value, bool):
			try:
				return format(value, ".6g")
			except (TypeError, ValueError):
				return str(value)
		return str(value)

	def headerData(  # noqa: N802
		self,
		section: int,
		orientation: Qt.Orientation,
		role: int = Qt.DisplayRole,
	):
		if role != Qt.DisplayRole:
			return None
		if orientation == Qt.Horizontal:
			return str(self._frame.columns[section])
		return str(self._frame.index[section])


class ScaledImageLabel(QLabel):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self._pixmap: Optional[QPixmap] = None
		self.setAlignment(Qt.AlignCenter)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.setMinimumSize(200, 200)

	def load_image(self, path: Path) -> bool:
		pixmap = QPixmap(str(path))
		if pixmap.isNull():
			self._pixmap = None
			self.setText(f"Failed to load {path.name}")
			return False
		self._pixmap = pixmap
		self.setText("")
		self._update_scaled_pixmap()
		return True

	def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override naming
		super().resizeEvent(event)
		self._update_scaled_pixmap()

	def _update_scaled_pixmap(self) -> None:
		if not self._pixmap or self._pixmap.isNull():
			return
		size = self.size()
		if size.width() <= 0 or size.height() <= 0:
			return
		scaled = self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		super().setPixmap(scaled)


class LogBrowserWindow(QMainWindow):
	_open_windows: List["LogBrowserWindow"] = []

	def __init__(self, directory: Optional[Path] = None, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self.setWindowTitle("Logqbit Log Browser")
		self.setWindowIcon(QIcon(QPixmap(str(files("logqbit") / "assets" / "browser.svg"))))
		self.resize(1200, 700)

		self._base_dir = Path(directory) if directory else Path.cwd()
		self._records: List[ExperimentRecord] = []
		self._display_records: List[ExperimentRecord] = []
		self._current_record: Optional[ExperimentRecord] = None
		self._list_refresh_pending = False
		self._detail_refresh_pending = False
		self._show_trashed = True
		self._suppress_item_changed = False
		self._image_tab_indices: List[int] = []

		self._dir_watcher = QFileSystemWatcher(self)
		self._detail_watcher = QFileSystemWatcher(self)
		self._dir_watcher.directoryChanged.connect(self._schedule_list_refresh)
		self._dir_watcher.fileChanged.connect(self._schedule_list_refresh)
		self._detail_watcher.directoryChanged.connect(self._schedule_detail_refresh)
		self._detail_watcher.fileChanged.connect(self._schedule_detail_refresh)

		self._build_ui()
		self._sync_directory_watcher()
		self.refresh_experiments()

		LogBrowserWindow._open_windows.append(self)
		self.destroyed.connect(self._on_destroyed)

	def _build_ui(self) -> None:
		central = QWidget(self)
		layout = QVBoxLayout(central)
		layout.setContentsMargins(8, 8, 8, 8)
		layout.setSpacing(6)

		top_bar = QHBoxLayout()
		self.directory_label = QLabel(str(self._base_dir))
		self.directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
		self.directory_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		change_button = QPushButton("Change...")
		refresh_button = QPushButton("Refresh")
		change_button.clicked.connect(self._on_change_directory_clicked)
		refresh_button.clicked.connect(self._on_refresh_clicked)
		top_bar.addWidget(QLabel("Directory:"))
		top_bar.addWidget(self.directory_label)
		top_bar.addWidget(change_button)
		top_bar.addWidget(refresh_button)
		layout.addLayout(top_bar)

		splitter = QSplitter(Qt.Horizontal, central)

		self.experiment_table = QTableWidget(0, 4, splitter)
		self.experiment_table.setHorizontalHeaderLabels(["Star", "ID", "Title", "Rows"])
		self.experiment_table.setSelectionBehavior(QTableWidget.SelectRows)
		self.experiment_table.setSelectionMode(QTableWidget.SingleSelection)
		self.experiment_table.verticalHeader().setVisible(False)
		header = self.experiment_table.horizontalHeader()
		header.setSectionResizeMode(0, QHeaderView.Fixed)
		header.setSectionResizeMode(1, QHeaderView.Interactive)
		header.setSectionResizeMode(3, QHeaderView.Interactive)
		header.setSectionResizeMode(2, QHeaderView.Stretch)
		header.resizeSection(0, 40)
		header.resizeSection(1, 60)
		header.resizeSection(3, 60)
		self.experiment_table.itemSelectionChanged.connect(self._on_experiment_selection_changed)
		self.experiment_table.setContextMenuPolicy(Qt.CustomContextMenu)
		self.experiment_table.customContextMenuRequested.connect(self._open_table_context_menu)
		self.experiment_table.itemChanged.connect(self._on_table_item_changed)

		detail_widget = QWidget(splitter)
		detail_layout = QVBoxLayout(detail_widget)
		detail_layout.setContentsMargins(0, 0, 0, 0)
		detail_layout.setSpacing(6)

		detail_top = QHBoxLayout()
		self.detail_label = QLabel("No experiment selected.")
		self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
		self.detail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		detail_top.addWidget(self.detail_label)
		detail_top.addStretch(1)
		detail_layout.addLayout(detail_top)

		self.tab_widget = QTabWidget(detail_widget)
		self.meta_view = QPlainTextEdit()
		self.meta_view.setReadOnly(True)
		self.tab_widget.addTab(self.meta_view, "Meta")

		data_tab = QWidget()
		data_layout = QVBoxLayout(data_tab)
		data_layout.setContentsMargins(4, 4, 4, 4)
		self.data_info_label = QLabel("No data loaded.")
		self.data_table = QTableView()
		self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.data_table.setSortingEnabled(False)
		self.data_table.horizontalHeader().setStretchLastSection(True)
		data_layout.addWidget(self.data_info_label)
		data_layout.addWidget(self.data_table)
		self.tab_widget.addTab(data_tab, "Data")

		detail_layout.addWidget(self.tab_widget)

		splitter.addWidget(self.experiment_table)
		splitter.addWidget(detail_widget)
		splitter.setStretchFactor(0, 1)
		splitter.setStretchFactor(1, 1)
		splitter.setSizes([600, 600])

		layout.addWidget(splitter)
		self.setCentralWidget(central)

	def _on_destroyed(self) -> None:
		try:
			LogBrowserWindow._open_windows.remove(self)
		except ValueError:
			pass

	def _open_new_window(self) -> None:
		window = LogBrowserWindow(self._base_dir)
		window.show()

	def _sync_directory_watcher(self) -> None:
		try:
			if self._dir_watcher.directories():
				self._dir_watcher.removePaths(self._dir_watcher.directories())
		except Exception:  # pragma: no cover - defensive
			pass
		if self._base_dir.exists():
			self._dir_watcher.addPath(str(self._base_dir))

	def _schedule_list_refresh(self) -> None:
		if self._list_refresh_pending:
			return
		self._list_refresh_pending = True
		QTimer.singleShot(REFRESH_DEBOUNCE_MS, self._run_list_refresh)

	def _run_list_refresh(self) -> None:
		self._list_refresh_pending = False
		self.refresh_experiments()

	def _schedule_detail_refresh(self) -> None:
		if self._detail_refresh_pending:
			return
		self._detail_refresh_pending = True
		QTimer.singleShot(REFRESH_DEBOUNCE_MS, self._run_detail_refresh)

	def _run_detail_refresh(self) -> None:
		self._detail_refresh_pending = False
		self.refresh_current_experiment()

	def _on_change_directory_clicked(self) -> None:
		current = str(self._base_dir)
		chosen = QFileDialog.getExistingDirectory(self, "Select log directory", current)
		if chosen:
			self.set_directory(Path(chosen))

	def set_directory(self, directory: Path) -> None:
		if directory == self._base_dir:
			return
		self._base_dir = directory
		self.directory_label.setText(str(self._base_dir))
		self._sync_directory_watcher()
		self.refresh_experiments()

	def refresh_experiments(self) -> None:
		previous_id = self._current_record.experiment_id if self._current_record else None
		records = self._scan_directory(self._base_dir)
		self._records = records
		self._populate_experiment_table(records)
		if self._display_records:
			self.detail_label.setText("Select an experiment to preview.")
			if previous_id is not None and self._select_experiment_by_id(previous_id):
				return
			if self._display_records:
				self.experiment_table.setCurrentCell(0, 0)
		else:
			if records:
				self.detail_label.setText("No experiments to display.")
			else:
				self.detail_label.setText("No experiments found.")
			self._current_record = None
			self.experiment_table.clearSelection()
			self._clear_preview_panels()
			self._clear_detail_watcher()

	def _select_experiment_by_id(self, experiment_id: int) -> bool:
		for row, record in enumerate(self._display_records):
			if record.experiment_id == experiment_id:
				table = self.experiment_table
				table.blockSignals(True)
				table.selectRow(row)
				table.setCurrentCell(row, 0)
				table.blockSignals(False)
				self._current_record = record
				self._load_experiment(record)
				return True
		return False

	def refresh_current_experiment(self) -> None:
		if not self._current_record:
			return
		self._load_experiment(self._current_record)

	def _scan_directory(self, directory: Path) -> List[ExperimentRecord]:
		records: List[ExperimentRecord] = []
		if not directory.exists() or not directory.is_dir():
			return records
		for entry in directory.iterdir():
			if not entry.is_dir():
				continue
			if not entry.name.isdigit():
				continue
			exp_id = int(entry.name)
			meta = _detect_meta_file(entry)
			data_file = _detect_data_file(entry)
			index_path = _detect_index_file(entry)
			index_info = _read_index_file(index_path) if index_path else IndexInfo()
			row_count = _read_row_count(data_file)
			record = ExperimentRecord(
				experiment_id=exp_id,
				path=entry,
				index_info=index_info,
				row_count=row_count,
				data_path=data_file,
				meta_path=meta,
				index_path=index_path,
			)
			records.append(record)
		records.sort(key=lambda rec: rec.experiment_id)
		return records

	def _populate_experiment_table(self, records: Iterable[ExperimentRecord]) -> None:
		table = self.experiment_table
		table.blockSignals(True)
		self._suppress_item_changed = True
		table.setRowCount(0)
		self._display_records = []
		for record in records:
			if not self._show_trashed and record.index_info.trashed:
				continue
			row = table.rowCount()
			table.insertRow(row)
			self._display_records.append(record)
			star_item = QTableWidgetItem()
			star_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
			star_item.setCheckState(Qt.Checked if record.index_info.starred else Qt.Unchecked)
			star_item.setData(Qt.TextAlignmentRole, Qt.AlignCenter)
			table.setItem(row, 0, star_item)

			id_item = QTableWidgetItem(str(record.experiment_id))
			id_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
			id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
			table.setItem(row, 1, id_item)

			title_text = record.index_info.title or "(untitled)"
			if record.index_info.trashed:
				title_text = "[trash] " + title_text
			title_item = QTableWidgetItem(title_text)
			title_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
			table.setItem(row, 2, title_item)

			rows_display = "" if record.row_count is None else f"{record.row_count:,}"
			rows_item = QTableWidgetItem(rows_display)
			rows_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
			rows_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
			table.setItem(row, 3, rows_item)
		table.blockSignals(False)
		self._suppress_item_changed = False
		table.resizeRowsToContents()

	def _on_experiment_selection_changed(self) -> None:
		table = self.experiment_table
		selected = table.selectionModel().selectedRows()
		if not selected:
			return
		row = selected[0].row()
		if row >= len(self._display_records):
			return
		record = self._display_records[row]
		self._current_record = record
		self._load_experiment(record)

	def _load_experiment(self, record: ExperimentRecord) -> None:
		self.detail_label.setText(f"Experiment {record.experiment_id} - {record.path}")
		self.meta_view.setPlainText(_read_meta_text(record.meta_path))

		df = _load_dataframe(record.data_path)
		if df is not None:
			model = PandasTableModel(df, self.data_table)
			self.data_table.setModel(model)
			total_rows = record.row_count if record.row_count is not None else len(df)
			truncated = "" if len(df) == total_rows else " (displaying first {0:,} rows)".format(len(df))
			rows_info = "unknown" if total_rows is None else f"{total_rows:,}"
			self.data_info_label.setText(f"Rows: {rows_info}{truncated}")
		else:
			self.data_table.setModel(None)
			self.data_info_label.setText("No data available.")

		image_files = _list_image_files(record.path)
		self._update_image_tabs(image_files)

		self._update_detail_watcher(record)

	def _on_refresh_clicked(self) -> None:
		self.refresh_experiments()
		self.refresh_current_experiment()

	def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
		if self._suppress_item_changed or item.column() != 0:
			return
		row = item.row()
		if row >= len(self._display_records):
			return
		record = self._display_records[row]
		starred = item.checkState() == Qt.Checked
		if record.index_info.starred == starred:
			return
		record.index_info.starred = starred
		self._persist_index_info(record)

	def _open_table_context_menu(self, point) -> None:
		row = self.experiment_table.indexAt(point).row()
		if 0 <= row < len(self._display_records):
			self.experiment_table.selectRow(row)
		record = self._get_selected_record()
		menu = QMenu(self)
		mark_star = menu.addAction("Mark as Starred")
		unstar = menu.addAction("Unstar")
		menu.addSeparator()
		mark_trash = menu.addAction("Mark as Trashed")
		restore_trash = menu.addAction("Restore from Trash")
		menu.addSeparator()
		show_trash_action = menu.addAction("Show Trashed Items")
		show_trash_action.setCheckable(True)
		show_trash_action.setChecked(self._show_trashed)
		menu.addSeparator()
		open_explorer = menu.addAction("Open in Explorer")
		if record is None:
			mark_star.setEnabled(False)
			unstar.setEnabled(False)
			mark_trash.setEnabled(False)
			restore_trash.setEnabled(False)
			open_explorer.setEnabled(False)
		else:
			mark_star.setEnabled(not record.index_info.starred)
			unstar.setEnabled(record.index_info.starred)
			mark_trash.setEnabled(not record.index_info.trashed)
			restore_trash.setEnabled(record.index_info.trashed)
		chosen = menu.exec(self.experiment_table.viewport().mapToGlobal(point))
		if chosen is None:
			return
		if chosen == mark_star and record is not None:
			self._set_record_starred(record, True)
		elif chosen == unstar and record is not None:
			self._set_record_starred(record, False)
		elif chosen == mark_trash and record is not None:
			self._set_record_trashed(record, True)
		elif chosen == restore_trash and record is not None:
			self._set_record_trashed(record, False)
		elif chosen == show_trash_action:
			self._toggle_show_trashed()
		elif chosen == open_explorer and record is not None:
			self._open_record_in_explorer(record)

	def _get_selected_record(self) -> Optional[ExperimentRecord]:
		selected = self.experiment_table.selectionModel().selectedRows()
		if not selected:
			return None
		row = selected[0].row()
		if 0 <= row < len(self._display_records):
			return self._display_records[row]
		return None

	def _set_record_starred(self, record: ExperimentRecord, value: bool) -> None:
		if record.index_info.starred == value:
			return
		record.index_info.starred = value
		self._persist_index_info(record)
		self.refresh_experiments()

	def _set_record_trashed(self, record: ExperimentRecord, value: bool) -> None:
		if record.index_info.trashed == value:
			return
		record.index_info.trashed = value
		self._persist_index_info(record)
		self.refresh_experiments()

	def _toggle_show_trashed(self) -> None:
		self._show_trashed = not self._show_trashed
		self.refresh_experiments()

	def _open_record_in_explorer(self, record: ExperimentRecord) -> None:
		path = record.path
		try:
			if sys.platform.startswith("win"):
				subprocess.run(["explorer", "/select,", str(path)], check=False)
			elif sys.platform == "darwin":
				subprocess.run(["open", "-R", str(path)], check=False)
			else:
				subprocess.run(["xdg-open", str(path)], check=False)
		except Exception as exc:  # pragma: no cover - defensive
			logger.error("Failed to open explorer for %s: %s", path, exc)
			QMessageBox.warning(self, "Open in Explorer", f"Failed to open file browser: {exc}")

	def _persist_index_info(self, record: ExperimentRecord) -> None:
		index_path = record.index_path or (record.path / "index")
		record.index_path = index_path
		try:
			_write_index_file(index_path, record.index_info)
		except Exception as exc:  # pragma: no cover - defensive
			logger.error("Failed to update index file %s: %s", index_path, exc)
			QMessageBox.warning(
				self,
				"Update Failed",
				f"Failed to update index file for experiment {record.experiment_id}: {exc}",
			)
			return
		if self._current_record and self._current_record.experiment_id == record.experiment_id:
			self._update_detail_watcher(record)

	def _clear_preview_panels(self) -> None:
		self.meta_view.setPlainText("")
		self.data_table.setModel(None)
		self.data_info_label.setText("No data loaded.")
		self._clear_image_tabs()

	def _clear_detail_watcher(self) -> None:
		try:
			paths = self._detail_watcher.files() + self._detail_watcher.directories()
			if paths:
				self._detail_watcher.removePaths(paths)
		except Exception:  # pragma: no cover - defensive
			pass

	def _clear_image_tabs(self) -> None:
		if not self._image_tab_indices:
			return
		for index in sorted(self._image_tab_indices, reverse=True):
			self.tab_widget.removeTab(index)
		self._image_tab_indices.clear()

	def _update_image_tabs(self, image_files: List[Path]) -> None:
		self._clear_image_tabs()
		for image_path in image_files:
			widget = ScaledImageLabel()
			widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
			widget.setWordWrap(True)
			widget.setToolTip(str(image_path))
			success = widget.load_image(image_path)
			if not success:
				widget.setWordWrap(True)
			index = self.tab_widget.addTab(widget, image_path.name)
			self._image_tab_indices.append(index)

	def _update_detail_watcher(self, record: ExperimentRecord) -> None:
		self._clear_detail_watcher()
		watch_paths: List[str] = [str(record.path)]
		for extra in (record.meta_path, record.data_path, record.index_path):
			if extra and extra.exists():
				watch_paths.append(str(extra))
		if watch_paths:
			self._detail_watcher.addPaths(watch_paths)


def ensure_application() -> QApplication:
	app = QApplication.instance()
	if app is None:
		app = QApplication(sys.argv)
		app.setApplicationName("Logqbit Log Browser")
	return app


def open_browser(directory: Optional[str] = None) -> LogBrowserWindow:
	app = ensure_application()
	window = LogBrowserWindow(Path(directory) if directory else None)
	window.show()
	return window


def main(argv: Optional[List[str]] = None) -> int:
	args = argv if argv is not None else sys.argv[1:]
	directory = Path(args[0]).expanduser().resolve() if args else None
	app = ensure_application()
	window = LogBrowserWindow(directory)
	window.show()
	return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual run
	sys.exit(main())
