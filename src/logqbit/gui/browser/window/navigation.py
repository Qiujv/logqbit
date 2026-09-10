"""Record-list navigation for the Browser window."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QFileSystemWatcher,
    QAbstractTableModel,
    QModelIndex,
    QSignalBlocker,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPalette, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QItemDelegate,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QTableView,
    QToolButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QStyle,
    QStyleOptionViewItem,
    QWidget,
)

from logqbit.catalog import LogCatalog, _log_name_sort_key
from logqbit.gui.browser.window.preferences import SettingsManager

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord


COL_ID = 0
COL_TITLE = 1
COL_ROWS = 2
COL_PLOT_AXES = 3
COL_CREATE_TIME = 4
COL_CREATE_MACHINE = 5
SORT_ROLE = Qt.UserRole + 1

REFRESH_DEBOUNCE_MS = 250
DETAIL_LOAD_DEBOUNCE_MS = 100
CATALOG_CACHE_SIZE = 3


class LogListSortFilterProxyModel(QSortFilterProxyModel):
    """Sort log IDs by numeric value first and directory name second."""

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        if left.column() == COL_ID and right.column() == COL_ID:
            model = self.sourceModel()
            if isinstance(model, LogListTableModel):
                left_record = model.get_record(left.row())
                right_record = model.get_record(right.row())
                if left_record is not None and right_record is not None:
                    return _log_name_sort_key(
                        left_record.path.name
                    ) < _log_name_sort_key(right_record.path.name)
        return super().lessThan(left, right)


class LogListTableModel(QAbstractTableModel):
    """Table model for the browser's record list."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[LogRecord] = []
        self._bold_font = QFont()
        self._bold_font.setBold(True)
        self._strikeout_font = QFont()
        self._strikeout_font.setStrikeOut(True)
        self._bold_strikeout_font = QFont()
        self._bold_strikeout_font.setBold(True)
        self._bold_strikeout_font.setStrikeOut(True)

    def set_records(self, records: list[LogRecord]) -> None:
        self.beginResetModel()
        self._records = list(records)
        self.endResetModel()

    def get_record(self, row: int) -> LogRecord | None:
        if 0 <= row < len(self._records):
            return self._records[row]
        return None

    def notify_record_changed(self, record: LogRecord) -> None:
        row = next(
            (
                index
                for index, current in enumerate(self._records)
                if current.path == record.path
            ),
            None,
        )
        if row is None:
            return
        self._records[row] = record
        self.dataChanged.emit(
            self.index(row, 0), self.index(row, self.columnCount() - 1)
        )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else 6

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        record = self._records[index.row()]
        column = index.column()
        if role == Qt.DisplayRole:
            if column == COL_ID:
                return record.log_id
            if column == COL_TITLE:
                parts: list[str] = []
                if record.trash:
                    parts.append("🗑️")
                star_prefix = "⭐" * max(record.star, 0)
                if star_prefix:
                    parts.append(star_prefix)
                parts.append(record.title or "(untitled)")
                return " ".join(parts)
            if column == COL_ROWS:
                return f"{record.row_count:,}"
            if column == COL_CREATE_TIME:
                return record.create_time
            if column == COL_CREATE_MACHINE:
                return record.create_machine
            if column == COL_PLOT_AXES:
                plot_axes = record.resolved_plot_columns.axes
                if plot_axes:
                    return ",".join(
                        [str(len(plot_axes))] + [axis[:3] for axis in plot_axes]
                    )
                return ""
        if role == Qt.FontRole and column == COL_TITLE:
            is_bold = max(record.star, 0) > 0
            if is_bold and record.trash:
                return self._bold_strikeout_font
            if is_bold:
                return self._bold_font
            if record.trash:
                return self._strikeout_font
        if role == Qt.ToolTipRole:
            if column == COL_TITLE:
                return record.title or "(untitled)"
            if column == COL_PLOT_AXES:
                plot_axes = record.resolved_plot_columns.axes
                return ", ".join(plot_axes) if plot_axes else "(no plot axes)"
        if role == Qt.UserRole and column == COL_ID:
            return record
        if role == SORT_ROLE:
            if column == COL_ROWS:
                return record.row_count
            return self.data(index, Qt.DisplayRole)
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ):  # noqa: N802
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        headers = ["ID", "Title", "Rows", "Axes", "Create Time", "Create Machine"]
        return headers[section] if 0 <= section < len(headers) else None


class LogListItemDelegate(QItemDelegate):
    """Draw selected log-list text consistently across native platform styles."""

    @staticmethod
    def _display_option(option: QStyleOptionViewItem) -> QStyleOptionViewItem:
        display_option = QStyleOptionViewItem(option)
        if display_option.state & QStyle.State_Selected:
            palette = display_option.palette
            palette.setColor(QPalette.HighlightedText, QColor("white"))
            display_option.palette = palette
        return display_option

    def drawDisplay(self, painter, option, rect, text) -> None:  # noqa: N802
        super().drawDisplay(painter, self._display_option(option), rect, text)


class PinScrollArea(QScrollArea):
    """A compact pin strip that scrolls only along its horizontal axis."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.pixelDelta().x() or event.pixelDelta().y()
        if not delta:
            delta = event.angleDelta().x() or event.angleDelta().y()
        if delta:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta
            )
        event.accept()


class RecordNavigation(QWidget):
    """Own the record table, catalog, filters, watcher, and pin navigation."""

    display_record = Signal(object)
    open_record = Signal(object)
    empty_selection = Signal(str)
    directory_changed = Signal(object)
    about_requested = Signal()

    def __init__(
        self,
        directory: Path,
        settings_manager: SettingsManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings_manager = settings_manager
        self._base_dir = Path(directory)
        self._selected_record: LogRecord | None = None
        self._all_records: list[LogRecord] = []
        self._show_trash = True
        self._show_starred_only = False
        self._list_refresh_pending = False
        self._catalog = LogCatalog()
        self._catalog_cache: OrderedDict[Path, LogCatalog] = OrderedDict()
        self._pinned_record_names = settings_manager.load_pinned_records(self._base_dir)

        self._detail_load_timer = QTimer(self)
        self._detail_load_timer.setSingleShot(True)
        self._detail_load_timer.setInterval(DETAIL_LOAD_DEBOUNCE_MS)
        self._detail_load_timer.timeout.connect(self._emit_selected_record)
        self._dir_watcher = QFileSystemWatcher(self)
        self._dir_watcher.directoryChanged.connect(self.schedule_list_refresh)
        self._dir_watcher.fileChanged.connect(self.schedule_list_refresh)

        self._build_ui()
        self._sync_directory_watcher()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def selected_record(self) -> LogRecord | None:
        return self._selected_record

    @property
    def all_records(self) -> list[LogRecord]:
        return self._all_records

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.log_table = self._create_log_table(self)
        layout.addWidget(self.log_table, stretch=1)
        self.pin_bar = self._create_pin_bar(self)
        layout.addWidget(self.pin_bar)

    def _create_log_table(self, parent: QWidget) -> QTableView:
        self.table_model = LogListTableModel(parent)
        self.table_proxy = LogListSortFilterProxyModel(parent)
        self.table_proxy.setSourceModel(self.table_model)
        self.table_proxy.setSortRole(SORT_ROLE)
        table = QTableView(parent)
        table.setModel(self.table_proxy)
        table.setItemDelegate(LogListItemDelegate(table))
        table.setSelectionBehavior(QTableView.SelectRows)
        table.setSelectionMode(QTableView.ExtendedSelection)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(False)
        table.setSortingEnabled(True)
        table.verticalHeader().setDefaultSectionSize(table.fontMetrics().height())
        header = table.horizontalHeader()
        for column in (
            COL_ID,
            COL_ROWS,
            COL_PLOT_AXES,
            COL_CREATE_TIME,
            COL_CREATE_MACHINE,
        ):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        header.setSectionResizeMode(COL_TITLE, QHeaderView.Stretch)
        header.setResizeContentsPrecision(-1)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(False)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.open_header_context_menu)
        table.setColumnHidden(COL_CREATE_TIME, True)
        table.setColumnHidden(COL_CREATE_MACHINE, True)
        table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        table.doubleClicked.connect(self._on_double_clicked)
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.sortByColumn(COL_ID, Qt.AscendingOrder)
        return table

    def _create_pin_bar(self, parent: QWidget) -> QWidget:
        pin_bar = QWidget(parent)
        pin_height = pin_bar.fontMetrics().height() + 4
        pin_bar.setFixedHeight(pin_height)
        pin_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pin_layout = QHBoxLayout(pin_bar)
        pin_layout.setContentsMargins(4, 0, 4, 0)
        pin_layout.setSpacing(4)
        pin_layout.addWidget(QLabel("📌Pins:"))
        self.pin_scroll_area = PinScrollArea(pin_bar)
        self.pin_scroll_area.setWidgetResizable(True)
        self.pin_scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.pin_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pin_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pin_scroll_area.setFixedHeight(pin_height)
        self._pin_button_container = QWidget(self.pin_scroll_area)
        self._pin_button_layout = QHBoxLayout(self._pin_button_container)
        self._pin_button_layout.setContentsMargins(0, 0, 0, 0)
        self._pin_button_layout.setSpacing(2)
        self._pin_button_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pin_scroll_area.setWidget(self._pin_button_container)
        pin_layout.addWidget(self.pin_scroll_area, stretch=1)
        return pin_bar

    def open_header_context_menu(self, point) -> None:
        menu = self.create_header_context_menu()
        menu.exec(self.log_table.horizontalHeader().mapToGlobal(point))

    def create_header_context_menu(self) -> QMenu:
        menu = QMenu(self)
        show_trash = menu.addAction("Show Trashed Items")
        show_trash.setCheckable(True)
        show_trash.setChecked(self._show_trash)
        show_trash.triggered.connect(self.toggle_show_trash)
        show_starred = menu.addAction("Show Starred Items Only")
        show_starred.setCheckable(True)
        show_starred.setChecked(self._show_starred_only)
        show_starred.triggered.connect(self.toggle_show_starred_only)
        menu.addSeparator()
        for label, column in (
            ("Show Plot Axes Column", COL_PLOT_AXES),
            ("Show Create Time Column", COL_CREATE_TIME),
            ("Show Create Machine Column", COL_CREATE_MACHINE),
        ):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(not self.log_table.isColumnHidden(column))
            action.triggered.connect(
                lambda checked=False, target=column: self.toggle_column(target, checked)
            )
        menu.addSeparator()
        menu.addAction("About", self.about_requested)
        return menu

    def toggle_column(self, column: int, visible: bool) -> None:
        self.log_table.setColumnHidden(column, not visible)

    def toggle_show_trash(self) -> None:
        self._show_trash = not self._show_trash
        self.refresh_logs()

    def toggle_show_starred_only(self) -> None:
        self._show_starred_only = not self._show_starred_only
        self.refresh_logs()

    def selected_records(self) -> list[LogRecord]:
        records: list[LogRecord] = []
        for proxy_index in self.log_table.selectionModel().selectedRows():
            source_index = self.table_proxy.mapToSource(proxy_index)
            record = self.table_model.get_record(source_index.row())
            if record is not None:
                records.append(record)
        return records

    def set_directory(self, directory: Path) -> None:
        path = Path(directory)
        if path != self._base_dir:
            self._base_dir = path
            catalog = self._catalog_cache.get(path)
            self._catalog = catalog if catalog is not None else LogCatalog()
            self._pinned_record_names = self.settings_manager.load_pinned_records(path)
            self._sync_directory_watcher()
            self.refresh_logs()
        self.settings_manager.update_recent_directories(path)
        self.directory_changed.emit(path)

    def refresh_logs(self, *, preferred_path: Path | None = None) -> None:
        previous_record = self._selected_record
        previous_path = (
            Path(preferred_path)
            if preferred_path is not None
            else (previous_record.path if previous_record is not None else None)
        )
        all_records = self._catalog.refresh(self._base_dir)
        self._all_records = all_records
        self._cache_catalog(self._base_dir, self._catalog)
        records = all_records
        if not self._show_trash:
            records = [record for record in records if not record.trash]
        if self._show_starred_only:
            records = [record for record in records if record.star > 0]
        self.table_model.set_records(records)
        self._resize_content_columns()
        self._rebuild_pin_bar(all_records)
        if records:
            selected = next(
                (record for record in records if record.path == previous_path),
                records[0],
            )
            source_row = records.index(selected)
            proxy_index = self.table_proxy.mapFromSource(
                self.table_model.index(source_row, 0)
            )
            blocker = QSignalBlocker(self.log_table.selectionModel())
            try:
                self.log_table.selectRow(proxy_index.row())
            finally:
                del blocker
            self._selected_record = selected
            if selected is not previous_record:
                self._detail_load_timer.stop()
                self.display_record.emit(selected)
        else:
            self._detail_load_timer.stop()
            self._selected_record = None
            self.log_table.clearSelection()
            self.empty_selection.emit(
                "No logs to display." if all_records else "No logs found."
            )

    def _cache_catalog(self, directory: Path, catalog: LogCatalog) -> None:
        directory = Path(directory)
        self._catalog_cache.pop(directory, None)
        self._catalog_cache[directory] = catalog
        while len(self._catalog_cache) > CATALOG_CACHE_SIZE:
            self._catalog_cache.popitem(last=False)

    def _resize_content_columns(self) -> None:
        for column in (
            COL_ID,
            COL_ROWS,
            COL_PLOT_AXES,
            COL_CREATE_TIME,
            COL_CREATE_MACHINE,
        ):
            self.log_table.resizeColumnToContents(column)

    def _rebuild_pin_bar(self, all_records: list[LogRecord]) -> None:
        for index in range(self._pin_button_layout.count() - 1, -1, -1):
            item = self._pin_button_layout.takeAt(index)
            if widget := item.widget():
                widget.deleteLater()
        records_by_name = {record.path.name: record for record in all_records}
        visible_paths = {
            record.path
            for row in range(self.table_model.rowCount())
            if (record := self.table_model.get_record(row)) is not None
        }
        names = [name for name in self._pinned_record_names if name in records_by_name]
        if names != self._pinned_record_names:
            self._pinned_record_names = names
            self.settings_manager.save_pinned_records(self._base_dir, names)
        if not names:
            self._pin_button_layout.addWidget(QLabel("(none)"))
        for name in names:
            record = records_by_name[name]
            button = QToolButton(self._pin_button_container)
            button.setText(f"#{record.log_id}")
            button.setAutoRaise(True)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setToolTip(f"{record.title or '(untitled)'}\n{record.path}")
            button.setEnabled(record.path in visible_paths)
            if not button.isEnabled():
                button.setToolTip(f"Hidden by current filters.\n{button.toolTip()}")
            button.clicked.connect(
                lambda _checked=False, target=record: self._jump_to_pinned_record(
                    target
                )
            )
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            button.customContextMenuRequested.connect(
                lambda point, target=record, source=button: self._open_pin_context_menu(
                    source, point, target
                )
            )
            self._pin_button_layout.addWidget(button)
        self._pin_button_layout.activate()
        self._pin_button_container.setMinimumSize(self._pin_button_layout.sizeHint())

    def pin_record(self, record: LogRecord) -> None:
        if record.path.name not in self._pinned_record_names:
            self._pinned_record_names.append(record.path.name)
            self.settings_manager.save_pinned_records(
                self._base_dir, self._pinned_record_names
            )
            self._rebuild_pin_bar(self._all_records)

    def unpin_record(self, record: LogRecord) -> None:
        if record.path.name in self._pinned_record_names:
            self._pinned_record_names.remove(record.path.name)
            self.settings_manager.save_pinned_records(
                self._base_dir, self._pinned_record_names
            )
            self._rebuild_pin_bar(self._all_records)

    def toggle_pin_record(self, record: LogRecord) -> None:
        if record.path.name in self._pinned_record_names:
            self.unpin_record(record)
        else:
            self.pin_record(record)

    def rename_pinned_record(self, old_path: Path, new_path: Path) -> None:
        if old_path.parent != self._base_dir:
            return
        try:
            index = self._pinned_record_names.index(old_path.name)
        except ValueError:
            return
        self._pinned_record_names[index] = new_path.name
        self.settings_manager.save_pinned_records(
            self._base_dir, self._pinned_record_names
        )

    def _jump_to_pinned_record(self, record: LogRecord) -> None:
        source_row = next(
            (
                row
                for row in range(self.table_model.rowCount())
                if (current := self.table_model.get_record(row)) is not None
                and current.path == record.path
            ),
            None,
        )
        if source_row is None:
            return
        proxy_index = self.table_proxy.mapFromSource(
            self.table_model.index(source_row, 0)
        )
        if not proxy_index.isValid():
            return
        blocker = QSignalBlocker(self.log_table.selectionModel())
        try:
            self.log_table.selectRow(proxy_index.row())
        finally:
            del blocker
        self.log_table.scrollTo(proxy_index, QAbstractItemView.PositionAtCenter)
        self._selected_record = record
        self._detail_load_timer.stop()
        self.display_record.emit(record)

    def _open_pin_context_menu(
        self, button: QToolButton, point, record: LogRecord
    ) -> None:
        menu = QMenu(button)
        unpin = menu.addAction(f"Unpin #{record.log_id}")
        if menu.exec(button.mapToGlobal(point)) == unpin:
            self.unpin_record(record)

    def _on_selection_changed(self) -> None:
        selected = self.log_table.selectionModel().selectedRows()
        if not selected:
            return
        source_index = self.table_proxy.mapToSource(selected[0])
        record = self.table_model.get_record(source_index.row())
        if record is None:
            return
        self._selected_record = record
        self._detail_load_timer.start()

    def _emit_selected_record(self) -> None:
        if self._selected_record is not None:
            self.display_record.emit(self._selected_record)

    def _on_double_clicked(self, proxy_index) -> None:
        source_index = self.table_proxy.mapToSource(proxy_index)
        record = self.table_model.get_record(source_index.row())
        if record is not None:
            self.open_record.emit(record)

    def notify_record_changed(self, record: LogRecord) -> None:
        if (
            self._selected_record is not None
            and self._selected_record.path == record.path
        ):
            self._selected_record = record
        self.table_model.notify_record_changed(record)
        self._resize_content_columns()

    def schedule_list_refresh(self) -> None:
        if self._list_refresh_pending:
            return
        self._list_refresh_pending = True
        QTimer.singleShot(REFRESH_DEBOUNCE_MS, self._run_list_refresh)

    def _run_list_refresh(self) -> None:
        self._list_refresh_pending = False
        self.refresh_logs()

    def _sync_directory_watcher(self) -> None:
        try:
            if self._dir_watcher.directories():
                self._dir_watcher.removePaths(self._dir_watcher.directories())
        except Exception:
            pass
        if self._base_dir.exists():
            self._dir_watcher.addPath(str(self._base_dir))
