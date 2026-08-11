"""Table model for the Browser record list."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget

from logqbit.catalog import _log_name_sort_key

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord


COL_ID = 0
COL_TITLE = 1
COL_ROWS = 2
COL_PLOT_AXES = 3
COL_CREATE_TIME = 4
COL_CREATE_MACHINE = 5
SORT_ROLE = Qt.UserRole + 1


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

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: D401
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
            return None

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

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        headers = ["ID", "Title", "Rows", "Axes", "Create Time", "Create Machine"]
        if 0 <= section < len(headers):
            return headers[section]
        return None
