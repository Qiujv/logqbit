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
from PySide6.QtCore import (QAbstractTableModel, QFileSystemWatcher,
                            QModelIndex, Qt, QTimer)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QSizePolicy, QSplitter, QTableView,
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

try:  # pragma: no cover - fallback for direct execution
	from .index import LogIndex
except ImportError:  # pragma: no cover - fallback for direct execution
	from index import LogIndex  # type: ignore

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_TABLE_ROWS = 50000
REFRESH_DEBOUNCE_MS = 250

COL_STAR = 0
COL_ID = 1
COL_TITLE = 2
COL_ROWS = 3
COL_CREATE_TIME = 4
COL_CREATE_MACHINE = 5


@dataclass
class IndexInfo:
	starred: bool = False
	trashed: bool = False
	title: str = ""
	create_time: str = ""
	create_machine: str = ""


@dataclass
class ExperimentRecord:
	experiment_id: int
	path: Path
	index_info: IndexInfo
	row_count: Optional[int] = None
	columns: List[str] = field(default_factory=list)
	data_path: Optional[Path] = None
	meta_path: Optional[Path] = None
	index_path: Optional[Path] = None


def _read_index_file(path: Path) -> IndexInfo:
	info = IndexInfo()
	if path is None or not path.exists():
		return info
	return _read_json_index(path)


def _write_index_file(path: Path, info: IndexInfo) -> None:
	_write_json_index(path, info)


def _read_json_index(path: Path) -> IndexInfo:
	info = IndexInfo()
	try:
		log_index = LogIndex(path, create=False)
	except FileNotFoundError:
		return info
	except Exception as exc:  # pragma: no cover - defensive
		logger.warning("Failed to load index.json %s: %s", path, exc)
		return info
	try:
		log_index.reload()
	except Exception as exc:  # pragma: no cover - defensive
		logger.debug("Reload index %s failed: %s", path, exc)
	data = log_index.root
	info.title = str(data.get("title", ""))
	info.starred = bool(data.get("starred", False))
	info.trashed = bool(data.get("trashed", False))
	info.create_time = str(data.get("create_time", ""))
	info.create_machine = str(data.get("create_machine", ""))
	return info


def _write_json_index(path: Path, info: IndexInfo) -> None:
	create = not path.exists()
	title = info.title or "untitled"
	log_index = LogIndex(path, title=title, create=True)
	try:
		log_index.reload()
	except Exception:  # pragma: no cover - defensive
		pass
	if info.title:
		log_index.root["title"] = info.title
	elif create:
		log_index.root.setdefault("title", title)
	log_index.root["starred"] = bool(info.starred)
	log_index.root["trashed"] = bool(info.trashed)
	log_index.save(timeout=1.0)


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
	json_path = folder / "index.json"
	return json_path if json_path.exists() else None


def _list_image_files(folder: Path) -> List[Path]:
	files: List[Path] = []
	for child in folder.iterdir():
		if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
			files.append(child)
	files.sort()
	return files


def _read_meta_text(path: Optional[Path]) -> str:
	if not path or not path.exists():
		return "meta.yaml not found."
	try:
		text = path.read_text(encoding="utf-8", errors="ignore")
	except Exception as exc:  # pragma: no cover - defensive
		logger.error("Failed to read meta file %s: %s", path, exc)
		return f"Failed to read meta.yaml: {exc}"
	return text if text.strip() else "(meta.yaml is empty)"


def _read_parquet_summary(data_path: Optional[Path]) -> tuple[Optional[int], List[str]]:
	if not data_path:
		return None, []
	try:
		parquet_file = pq.ParquetFile(data_path)
		metadata = parquet_file.metadata
		row_count = metadata.num_rows if metadata is not None else None
		columns = list(parquet_file.schema.names) if parquet_file.schema is not None else []
		return row_count, columns
	except Exception as exc:  # pragma: no cover - defensive
		logger.warning("Failed to inspect parquet file %s: %s", data_path, exc)
		return None, []


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
		self._current_record: Optional[ExperimentRecord] = None
		self._list_refresh_pending = False
		self._detail_refresh_pending = False
		self._show_trashed = True
		self._suppress_item_changed = False
		self._image_tab_indices: List[int] = []
		self._sort_column = COL_ID
		self._sort_order = Qt.AscendingOrder
		self._suppress_sort_sync = False

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

		self.experiment_table = QTableWidget(0, 6, splitter)
		self.experiment_table.setHorizontalHeaderLabels([
			"Star",
			"ID",
			"Title",
			"Rows",
			"Create Time",
			"Create Machine",
		])
		self.experiment_table.setSelectionBehavior(QTableWidget.SelectRows)
		self.experiment_table.setSelectionMode(QTableWidget.SingleSelection)
		self.experiment_table.verticalHeader().setVisible(False)
		header = self.experiment_table.horizontalHeader()
		header.setSectionResizeMode(0, QHeaderView.Fixed)
		header.setSectionResizeMode(1, QHeaderView.Interactive)
		header.setSectionResizeMode(3, QHeaderView.Interactive)
		header.setSectionResizeMode(2, QHeaderView.Stretch)
		header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
		header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
		header.resizeSection(0, 40)
		header.resizeSection(1, 60)
		header.resizeSection(3, 60)
		header.setSectionsClickable(True)
		header.setSortIndicator(self._sort_column, self._sort_order)
		header.setSortIndicatorShown(True)
		header.sortIndicatorChanged.connect(self._on_sort_indicator_changed)
		self.experiment_table.itemSelectionChanged.connect(self._on_experiment_selection_changed)
		self.experiment_table.setContextMenuPolicy(Qt.CustomContextMenu)
		self.experiment_table.customContextMenuRequested.connect(self._open_table_context_menu)
		self.experiment_table.itemChanged.connect(self._on_table_item_changed)
		self.experiment_table.setSortingEnabled(True)
		self.experiment_table.setColumnHidden(COL_CREATE_TIME, True)
		self.experiment_table.setColumnHidden(COL_CREATE_MACHINE, True)

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
		self.data_table = QTableView()
		self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.data_table.setSortingEnabled(False)
		self.data_table.horizontalHeader().setStretchLastSection(True)
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
		row_count = self.experiment_table.rowCount()
		if row_count:
			self.detail_label.setText("Select an experiment to preview.")
			if previous_id is not None and self._select_experiment_by_id(previous_id):
				return
			if row_count:
				self.experiment_table.setCurrentCell(0, COL_ID)
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
		table = self.experiment_table
		for row in range(table.rowCount()):
			record = self._record_for_row(row)
			if record and record.experiment_id == experiment_id:
				table.blockSignals(True)
				table.selectRow(row)
				table.setCurrentCell(row, COL_ID)
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
			row_count, columns = _read_parquet_summary(data_file)
			record = ExperimentRecord(
				experiment_id=exp_id,
				path=entry,
				index_info=index_info,
				row_count=row_count,
				columns=columns,
				data_path=data_file,
				meta_path=meta,
				index_path=index_path,
			)
			records.append(record)
		records.sort(key=lambda rec: rec.experiment_id)
		return records

	def _populate_experiment_table(self, records: Iterable[ExperimentRecord]) -> None:
		table = self.experiment_table
		was_sorting = table.isSortingEnabled()
		if was_sorting:
			table.setSortingEnabled(False)
		table.blockSignals(True)
		self._suppress_item_changed = True
		table.setRowCount(0)
		for record in records:
			if not self._show_trashed and record.index_info.trashed:
				continue
			row = table.rowCount()
			table.insertRow(row)
			self._set_row_items(row, record)
		table.blockSignals(False)
		self._suppress_item_changed = False
		if was_sorting:
			table.setSortingEnabled(True)
			self._apply_current_sort()
		else:
			table.resizeRowsToContents()

	def _set_row_items(self, row: int, record: ExperimentRecord) -> None:
		table = self.experiment_table
		star_item = QTableWidgetItem()
		star_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
		star_item.setCheckState(Qt.Checked if record.index_info.starred else Qt.Unchecked)
		star_item.setData(Qt.TextAlignmentRole, Qt.AlignCenter)
		star_item.setData(Qt.UserRole, record)
		table.setItem(row, COL_STAR, star_item)

		id_item = QTableWidgetItem()
		id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
		id_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
		id_item.setData(Qt.DisplayRole, record.experiment_id)
		table.setItem(row, COL_ID, id_item)

		title_text = record.index_info.title or "(untitled)"
		if record.index_info.trashed:
			title_text = "[trash] " + title_text
		title_item = QTableWidgetItem(title_text)
		title_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
		table.setItem(row, COL_TITLE, title_item)

		rows_item = QTableWidgetItem()
		rows_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
		rows_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		rows_value = record.row_count if record.row_count is not None else None
		rows_item.setData(Qt.DisplayRole, rows_value)
		table.setItem(row, COL_ROWS, rows_item)

		time_item = QTableWidgetItem(record.index_info.create_time or "")
		time_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
		table.setItem(row, COL_CREATE_TIME, time_item)

		machine_item = QTableWidgetItem(record.index_info.create_machine or "")
		machine_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
		table.setItem(row, COL_CREATE_MACHINE, machine_item)

	def _apply_current_sort(self) -> None:
		if not self.experiment_table.isSortingEnabled():
			self.experiment_table.resizeRowsToContents()
			return
		self._suppress_sort_sync = True
		try:
			header = self.experiment_table.horizontalHeader()
			header.setSortIndicator(self._sort_column, self._sort_order)
			self.experiment_table.sortItems(self._sort_column, self._sort_order)
		finally:
			self._suppress_sort_sync = False
		self.experiment_table.resizeRowsToContents()

	def _on_sort_indicator_changed(self, section: int, order: Qt.SortOrder) -> None:
		if self._suppress_sort_sync:
			return
		self._sort_column = section
		self._sort_order = order
		self._apply_current_sort()

	def _record_for_row(self, row: int) -> Optional[ExperimentRecord]:
		item = self.experiment_table.item(row, COL_STAR)
		if item is None:
			return None
		return item.data(Qt.UserRole)

	def _find_row_for_record(self, record: ExperimentRecord) -> Optional[int]:
		table = self.experiment_table
		for row in range(table.rowCount()):
			row_record = self._record_for_row(row)
			if row_record is record:
				return row
			if row_record and row_record.experiment_id == record.experiment_id and row_record.path == record.path:
				return row
		return None

	def _update_row_items(self, record: ExperimentRecord) -> None:
		row = self._find_row_for_record(record)
		if row is None:
			return
		table = self.experiment_table
		table.blockSignals(True)
		self._suppress_item_changed = True
		try:
			self._set_row_items(row, record)
		finally:
			table.blockSignals(False)
			self._suppress_item_changed = False
		table.resizeRowToContents(row)

	def _toggle_column(self, column: int, visible: bool) -> None:
		self.experiment_table.setColumnHidden(column, not visible)

	def _on_experiment_selection_changed(self) -> None:
		table = self.experiment_table
		selected = table.selectionModel().selectedRows()
		if not selected:
			return
		row = selected[0].row()
		record = self._record_for_row(row)
		if record is None:
			return
		self._current_record = record
		self._load_experiment(record)

	def _load_experiment(self, record: ExperimentRecord) -> None:
		self.detail_label.setText(f"Experiment {record.experiment_id} - {record.path}")
		self.meta_view.setPlainText(_read_meta_text(record.meta_path))

		df = _load_dataframe(record.data_path)
		if df is not None:
			model = PandasTableModel(df, self.data_table)
			self.data_table.setModel(model)
			record.columns = list(df.columns)
		else:
			self.data_table.setModel(None)

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
		record = self._record_for_row(row)
		if record is None:
			return
		starred = item.checkState() == Qt.Checked
		if record.index_info.starred == starred:
			return
		previous = record.index_info.starred
		record.index_info.starred = starred
		if not self._persist_index_info(record):
			record.index_info.starred = previous
			self._suppress_item_changed = True
			try:
				item.setCheckState(Qt.Checked if previous else Qt.Unchecked)
			except Exception:
				pass
			finally:
				self._suppress_item_changed = False

	def _open_table_context_menu(self, point) -> None:
		row = self.experiment_table.indexAt(point).row()
		if 0 <= row < self.experiment_table.rowCount():
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
		create_time_action = menu.addAction("Show Create Time Column")
		create_time_action.setCheckable(True)
		create_time_action.setChecked(not self.experiment_table.isColumnHidden(COL_CREATE_TIME))
		create_machine_action = menu.addAction("Show Create Machine Column")
		create_machine_action.setCheckable(True)
		create_machine_action.setChecked(not self.experiment_table.isColumnHidden(COL_CREATE_MACHINE))
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
		elif chosen == create_time_action:
			self._toggle_column(COL_CREATE_TIME, create_time_action.isChecked())
		elif chosen == create_machine_action:
			self._toggle_column(COL_CREATE_MACHINE, create_machine_action.isChecked())
		elif chosen == open_explorer and record is not None:
			self._open_record_in_explorer(record)

	def _get_selected_record(self) -> Optional[ExperimentRecord]:
		selected = self.experiment_table.selectionModel().selectedRows()
		if not selected:
			return None
		row = selected[0].row()
		return self._record_for_row(row)

	def _set_record_starred(self, record: ExperimentRecord, value: bool) -> None:
		if record.index_info.starred == value:
			return
		previous = record.index_info.starred
		record.index_info.starred = value
		if not self._persist_index_info(record):
			record.index_info.starred = previous
			return
		self.refresh_experiments()

	def _set_record_trashed(self, record: ExperimentRecord, value: bool) -> None:
		if record.index_info.trashed == value:
			return
		previous = record.index_info.trashed
		record.index_info.trashed = value
		if not self._persist_index_info(record):
			record.index_info.trashed = previous
			return
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

	def _persist_index_info(self, record: ExperimentRecord) -> bool:
		index_path = record.index_path
		if index_path is None or not index_path.exists():
			index_path = record.path / "index.json"
		record.index_path = index_path
		try:
			_write_index_file(index_path, record.index_info)
		except TimeoutError as exc:  # pragma: no cover - defensive
			logger.warning("Timed out updating index file %s: %s", index_path, exc)
			QMessageBox.warning(
				self,
				"Update Failed",
				"Another process is updating this log. Please try again.",
			)
			return False
		except Exception as exc:  # pragma: no cover - defensive
			logger.error("Failed to update index file %s: %s", index_path, exc)
			QMessageBox.warning(
				self,
				"Update Failed",
				f"Failed to update index file for experiment {record.experiment_id}: {exc}",
			)
			return False
		record.index_info = _read_index_file(index_path)
		self._update_row_items(record)
		self._apply_current_sort()
		if self._current_record and self._current_record.experiment_id == record.experiment_id:
			self._update_detail_watcher(record)
		return True

	def _clear_preview_panels(self) -> None:
		self.meta_view.setPlainText("")
		self.data_table.setModel(None)
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
