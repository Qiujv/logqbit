"""Two-phase merge dialog for Browser records."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from logqbit.catalog import (
    LogRecord,
    MergeRecordsError,
    MergeRecordsResult,
    PreparedMerge,
)

logger = logging.getLogger(__name__)


class MergeDialog(QDialog):
    """Prepare a merge automatically, then publish only after confirmation."""

    analysis_finished = Signal(bool)
    merge_finished = Signal(bool)
    files_written = Signal()

    def __init__(
        self,
        parent: QWidget,
        records: Sequence[LogRecord],
        destination_parent: str | Path,
        *,
        target: LogRecord | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Data")
        self.setModal(True)
        self.setMinimumWidth(620)
        self._records = list(records)
        self._destination_parent = Path(destination_parent)
        self._target = target
        self._prepared: PreparedMerge | None = None
        self._closed = False

        layout = QVBoxLayout(self)
        self._summary_label = QLabel(self._summary_text(), self)
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._summary_label)

        self._status_label = QLabel("Preparing merge...", self)
        status_font = self._status_label.font()
        status_font.setBold(True)
        self._status_label.setFont(status_font)
        layout.addWidget(self._status_label)

        self._detail_label = QLabel("", self)
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._detail_label.hide()
        layout.addWidget(self._detail_label)

        button_layout = QHBoxLayout()
        self._try_new_button = QPushButton("Try Merge into New Folder", self)
        self._try_new_button.hide()
        button_layout.addWidget(self._try_new_button)
        button_layout.addStretch()
        self._write_button = QPushButton("Write File", self)
        self._write_button.setEnabled(False)
        self._cancel_button = QPushButton("Cancel", self)
        self._cancel_button.setDefault(True)
        button_layout.addWidget(self._write_button)
        button_layout.addWidget(self._cancel_button)
        layout.addLayout(button_layout)

        self._try_new_button.clicked.connect(self._try_new_folder)
        self._write_button.clicked.connect(self._start_publish)
        self._cancel_button.clicked.connect(self.reject)
        QTimer.singleShot(0, self._start_analysis)

    def _summary_text(self) -> str:
        if self._target is None:
            heading = f"Merging {len(self._records)} folders into a new folder:"
        else:
            source_count = len(self._records) - 1
            noun = "folder" if source_count == 1 else "folders"
            heading = f"Appending {source_count} {noun} into #{self._target.log_id}:"
        record_lines = []
        for record in self._records:
            columns = ", ".join(record.columns) or "(no columns)"
            record_lines.append(f"#{record.log_id}: {columns}, {record.row_count} rows")
        return "\n".join((heading, *record_lines))

    @Slot()
    def _start_analysis(self) -> None:
        if self._closed:
            return
        self._discard_prepared()
        self._try_new_button.hide()
        self._write_button.show()
        self._write_button.setEnabled(False)
        self._cancel_button.setText("Cancel")
        self._status_label.setText("Preparing merge...")
        self._set_status_color(QColor("#777777"))
        self._detail_label.hide()

        try:
            if self._target is None:
                self._prepared = PreparedMerge.for_new_folder(
                    self._records,
                    self._destination_parent,
                )
            else:
                self._prepared = PreparedMerge.for_append(self._records)
        except MergeRecordsError as exc:
            self._show_failure(str(exc))
            succeeded = False
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to prepare record merge")
            self._show_failure(f"{type(exc).__name__}: {exc}")
            succeeded = False
        else:
            if self._prepared.is_noop:
                self._show_ready("No merge needed.", self._prepared_message())
                self._cancel_button.setText("Close")
            else:
                self._show_ready("Ready to merge.", self._prepared_message())
                self._write_button.setEnabled(True)
            succeeded = True
        self._cancel_button.setDefault(True)
        self.analysis_finished.emit(succeeded)

    @Slot()
    def _start_publish(self) -> None:
        if self._prepared is None or self._prepared.is_noop:
            return
        self._write_button.setEnabled(False)
        try:
            result = self._prepared.publish()
        except MergeRecordsError as exc:
            self._show_failure(str(exc))
            succeeded = False
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to publish record merge")
            self._show_failure(f"{type(exc).__name__}: {exc}")
            succeeded = False
        else:
            self._status_label.setText("Merge complete.")
            self._set_status_color(QColor("#2e9d45"))
            self._detail_label.setText(self._success_message(result))
            self._detail_label.show()
            self.files_written.emit()
            succeeded = True
        self._write_button.hide()
        self._cancel_button.setText("Close")
        self._cancel_button.setDefault(True)
        self.merge_finished.emit(succeeded)

    def _show_failure(self, message: str) -> None:
        self._status_label.setText("Fail to merge.")
        self._set_status_color(QColor("#d64545"))
        self._detail_label.setText(message)
        self._detail_label.show()
        self._write_button.setEnabled(False)
        self._cancel_button.setText("Close")
        if self._target is not None:
            self._try_new_button.show()

    def _show_ready(self, status: str, detail: str) -> None:
        self._status_label.setText(status)
        self._set_status_color(QColor("#2e9d45"))
        self._detail_label.setText(detail)
        self._detail_label.show()

    def _set_status_color(self, color: QColor) -> None:
        palette = self._status_label.palette()
        palette.setColor(QPalette.WindowText, color)
        self._status_label.setPalette(palette)

    def _prepared_message(self) -> str:
        assert self._prepared is not None
        if self._prepared.is_noop:
            return (
                "The target already contains all selected sources.\n"
                f"Rows: {self._prepared.row_count}; "
                f"skipped folders: {self._prepared.skipped_records}."
            )
        return (
            f"Rows after merge: {self._prepared.row_count}; "
            f"folders appended: {self._prepared.appended_records}; "
            f"already merged: {self._prepared.skipped_records}."
        )

    @staticmethod
    def _success_message(result: MergeRecordsResult) -> str:
        return f"{result.row_count} rows written to:\n{result.path}"

    @Slot()
    def _try_new_folder(self) -> None:
        if self._target is None:
            return
        self._discard_prepared()
        self._target = None
        self._summary_label.setText(self._summary_text())
        self._start_analysis()

    def reject(self) -> None:
        self._closed = True
        self._discard_prepared()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override naming
        self._closed = True
        self._discard_prepared()
        super().closeEvent(event)

    def _discard_prepared(self) -> None:
        if self._prepared is not None:
            self._prepared.discard()
            self._prepared = None
