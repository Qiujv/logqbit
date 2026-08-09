"""Data table models and controls for record details."""

from __future__ import annotations

import logging
import numbers
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from logqbit.catalog import LogRecord

logger = logging.getLogger(__name__)


class PandasTableModel(QAbstractTableModel):
    """Table model for displaying DataFrames with an optional row limit."""

    def __init__(
        self,
        frame: pd.DataFrame,
        parent: QWidget | None = None,
        highlight_columns: Iterable[str] | None = None,
        preview_limit: int | None = None,
        missing_display: str = "",
    ) -> None:
        super().__init__(parent)
        self._df = frame
        self._preview_limit = preview_limit
        self._missing_display = missing_display
        self._highlight = (
            {str(name) for name in highlight_columns} if highlight_columns else set()
        )
        self._bold_font = QFont(parent.font()) if parent else QFont()
        self._bold_font.setBold(True)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        total_rows = self._df.shape[0]
        if self._preview_limit is not None and self._preview_limit > 0:
            return min(total_rows, self._preview_limit)
        return total_rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._df.columns)

    def get_total_rows(self) -> int:
        return self._df.shape[0]

    def column_values(self, column: int) -> pd.Series:
        """Return one full column by position, independent of the preview limit."""
        return self._df.iloc[:, column]

    def set_preview_limit(self, limit: int | None) -> None:
        old_count = self.rowCount()
        self._preview_limit = limit
        if self.rowCount() != old_count:
            self.beginResetModel()
            self.endResetModel()

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: D401
        if not index.isValid():
            return None
        column_name = str(self._df.columns[index.column()])
        if role == Qt.FontRole and column_name in self._highlight:
            return self._bold_font
        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self._df.iat[index.row(), index.column()]
        if pd.isna(value):
            return self._missing_display
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
        if role == Qt.FontRole and orientation == Qt.Horizontal:
            column_name = str(self._df.columns[section])
            if column_name in self._highlight:
                return self._bold_font
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class UniqueValuesDialog(QDialog):
    """Display the distinct values from one DataFrame column."""

    def __init__(
        self,
        column_name: str,
        values: pd.Series,
        source_row_count: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Unique Values — {column_name}")
        self.resize(480, 560)

        layout = QVBoxLayout(self)
        table = QTableView(self)
        table.setWordWrap(False)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.verticalHeader().setDefaultSectionSize(table.fontMetrics().height() + 6)
        table.setModel(
            PandasTableModel(
                values.to_frame(name=column_name),
                table,
                missing_display="<NA>",
            )
        )
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        status = QLabel(
            f"{len(values):,} unique values from {source_row_count:,} rows.",
            self,
        )
        layout.addWidget(status)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DataViewManager:
    INITIAL_PREVIEW_LIMIT = 100
    PREVIEW_INCREMENT = 1000

    def __init__(
        self,
        parent: QWidget | None = None,
        load_more_callback: Callable[[], None] | None = None,
    ):
        self._load_more_callback = load_more_callback
        self.widget = self._create_widget(parent)

    def _create_widget(self, parent: QWidget | None = None) -> QWidget:
        data_tab = QWidget(parent)
        data_layout = QVBoxLayout(data_tab)
        data_layout.setContentsMargins(4, 4, 4, 4)

        self.data_table = QTableView()
        self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data_table.setSortingEnabled(False)
        self.data_table.setWordWrap(False)
        self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.customContextMenuRequested.connect(self._open_context_menu)
        self.data_table.horizontalHeader().setStretchLastSection(False)
        self.data_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        row_height = self.data_table.fontMetrics().height() + 6
        self.data_table.verticalHeader().setDefaultSectionSize(row_height)
        data_layout.addWidget(self.data_table)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self.data_status_label = QLabel("")
        self.data_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_status_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.data_load_button = QPushButton("Show More Rows")
        self.data_load_button.setEnabled(False)
        if self._load_more_callback:
            self.data_load_button.clicked.connect(self._load_more_callback)
        controls.addWidget(self.data_status_label)
        controls.addStretch(1)
        controls.addWidget(self.data_load_button)
        data_layout.addLayout(controls)
        return data_tab

    def _open_context_menu(self, point) -> None:
        index = self.data_table.indexAt(point)
        if not index.isValid():
            return
        self.data_table.setCurrentIndex(index)
        menu = QMenu(self.data_table)
        show_unique_action = menu.addAction("Show Unique Values")
        show_unique_action.triggered.connect(
            lambda _checked=False, column=index.column(): self._show_unique_values(
                column
            )
        )
        menu.exec(self.data_table.viewport().mapToGlobal(point))

    def _unique_values_for_column(self, column: int) -> pd.Series | None:
        model = self.data_table.model()
        if not isinstance(model, PandasTableModel):
            return None
        return model.column_values(column).drop_duplicates(ignore_index=True)

    def _show_unique_values(self, column: int) -> None:
        model = self.data_table.model()
        if not isinstance(model, PandasTableModel):
            return
        error_message = ""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            values = self._unique_values_for_column(column)
        except Exception as exc:  # pragma: no cover - defensive
            values = None
            error_message = str(exc)
            logger.error("Failed to collect unique values: %s", error_message)
        finally:
            QApplication.restoreOverrideCursor()
        if error_message:
            QMessageBox.warning(
                self.widget,
                "Show Unique Values",
                f"Failed to collect unique values: {error_message}",
            )
            return
        if values is None:
            return
        column_name = str(model.headerData(column, Qt.Horizontal, Qt.DisplayRole))
        dialog = UniqueValuesDialog(
            column_name,
            values,
            model.get_total_rows(),
            self.widget,
        )
        dialog.exec()

    def set_empty(self, message: str = "No data to display.") -> None:
        self.data_table.setModel(None)
        self.data_status_label.setText(message)
        self.data_load_button.setEnabled(False)

    def show_data_table(
        self,
        record: LogRecord,
        dataframe: pd.DataFrame | None,
        preview_only: bool,
    ) -> None:
        if dataframe is None:
            message = (
                "Data file not found."
                if record.data_version is None
                else "Failed to load data."
            )
            self.set_empty(message)
            return

        total_rows = len(dataframe)
        preview_limit = None
        if preview_only and total_rows > self.INITIAL_PREVIEW_LIMIT:
            preview_limit = self.INITIAL_PREVIEW_LIMIT

        model = PandasTableModel(
            dataframe,
            self.data_table,
            highlight_columns=record.resolved_plot_columns.axes,
            preview_limit=preview_limit,
        )
        self.data_table.setModel(model)
        self.data_table.resizeColumnsToContents()
        row_height = self.data_table.fontMetrics().height() + 6
        self.data_table.verticalHeader().setDefaultSectionSize(row_height)

        displayed_rows = model.rowCount()
        has_more = displayed_rows < total_rows
        if has_more:
            self.data_status_label.setText(
                f"Showing first {displayed_rows} rows. Total: {total_rows}."
            )
            self.data_load_button.setEnabled(True)
        else:
            self.data_status_label.setText(f"Showing all {displayed_rows} rows.")
            self.data_load_button.setEnabled(False)

    def load_more_data(
        self,
        record: LogRecord,
        dataframe: pd.DataFrame | None,
    ) -> None:
        model = self.data_table.model()
        if not isinstance(model, PandasTableModel):
            self.show_data_table(record, dataframe, preview_only=False)
            return

        total_rows = model.get_total_rows()
        current_limit = model.rowCount()
        if current_limit >= total_rows:
            return

        new_limit = min(current_limit + self.PREVIEW_INCREMENT, total_rows)
        model.set_preview_limit(new_limit)

        displayed_rows = model.rowCount()
        has_more = displayed_rows < total_rows
        if has_more:
            self.data_status_label.setText(
                f"Showing first {displayed_rows} rows. Total: {total_rows}."
            )
            self.data_load_button.setEnabled(True)
        else:
            self.data_status_label.setText(f"Showing all {displayed_rows} rows.")
            self.data_load_button.setEnabled(False)
