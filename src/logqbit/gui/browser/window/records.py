"""Record actions and context menus for the Browser."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QWidget,
)
from send2trash import send2trash

from logqbit.catalog import LogRecord, PreparedMerge, export_records
from logqbit.gui.browser.detail.files import open_in_file_manager
from logqbit.gui.browser.window.merge import MergeDialog
from logqbit.gui.browser.window.navigation import RecordNavigation
from logqbit.metadata import LogMetadata


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _validated_log_id(value: str) -> str:
    log_id = value.strip()
    if not log_id:
        raise ValueError("ID cannot be empty.")
    if log_id in {".", ".."}:
        raise ValueError("ID must be a directory name, not '.' or '..'.")
    if any(character in log_id for character in '/\\<>:"|?*'):
        raise ValueError("ID contains a character that is invalid in a directory name.")
    if any(ord(character) < 32 for character in log_id):
        raise ValueError("ID cannot contain control characters.")
    if log_id.endswith((" ", ".")):
        raise ValueError("ID cannot end with a space or period.")
    if log_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("ID is a reserved directory name on Windows.")
    return log_id


class _MakeNoteDialog(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Make Note")
        layout = QFormLayout(self)
        self.id_edit = QLineEdit(self)
        self.title_edit = QLineEdit(self)
        layout.addRow('ID (e.g. 5.1, "foobar"):', self.id_edit)
        layout.addRow("Title:", self.title_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.resize(max(self.sizeHint().width(), 500), self.sizeHint().height())


class RecordActions(QObject):
    """Own record menus and mutations, using navigation as the list boundary."""

    records_changed = Signal()
    files_written = Signal()
    path_renamed = Signal(object, object)

    def __init__(
        self,
        navigation: RecordNavigation,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self.navigation = navigation
        self._parent_widget = parent

    @property
    def parent_widget(self) -> QWidget:
        return self._parent_widget

    def get_selected_records(self) -> list[LogRecord]:
        return self.navigation.selected_records()

    def open_table_context_menu(self, point) -> None:
        records = self.get_selected_records()
        menu = QMenu(self.parent_widget)
        pin_action = menu.addAction("Toggle 📌Pin (Ctrl+P)")
        make_note_action = menu.addAction("Make 🏷️Note... (Ctrl+N)")
        rename_action = menu.addAction("Rename Title... (F2)")
        change_id_action = menu.addAction("Change ID... (F3)")
        toggle_star_action = menu.addAction("Toggle ⭐Star (S)")
        toggle_star_action.setCheckable(True)
        toggle_trash_action = menu.addAction("Toggle 🗑️Trash (T)")
        toggle_trash_action.setCheckable(True)
        send_to_recycle_action = menu.addAction("Send to Recycle Bin (Del)")
        open_explorer = menu.addAction("Open in Explorer (Ctrl+Enter)")
        menu.addSeparator()
        merge_new_action = menu.addAction("Merge into New LogFolder...")
        append_action = menu.addAction("Append into Existing LogFolder...")
        export_action = menu.addAction("Export Items...")
        can_merge = len(records) >= 2
        append_target = PreparedMerge.find_append_target(records)
        merge_new_action.setEnabled(can_merge)
        append_action.setEnabled(append_target is not None)
        if not records:
            for action in (
                rename_action,
                change_id_action,
                toggle_star_action,
                toggle_trash_action,
                send_to_recycle_action,
                open_explorer,
                pin_action,
                export_action,
            ):
                action.setEnabled(False)
        else:
            rename_action.setEnabled(len(records) == 1)
            change_id_action.setEnabled(len(records) == 1)
            toggle_star_action.setChecked(all(record.star > 0 for record in records))
            toggle_trash_action.setChecked(all(record.trash for record in records))
        chosen = menu.exec(self.navigation.log_table.viewport().mapToGlobal(point))
        if chosen is None:
            return
        if chosen == make_note_action:
            self.make_note()
        elif chosen == rename_action and len(records) == 1:
            self.rename_record_title(records[0])
        elif chosen == change_id_action and len(records) == 1:
            self.change_record_id(records[0])
        elif chosen == toggle_star_action and records:
            self.set_records_star_count(
                records, 1 if toggle_star_action.isChecked() else 0
            )
        elif chosen == toggle_trash_action and records:
            self.set_records_trash(records, toggle_trash_action.isChecked())
        elif chosen == send_to_recycle_action and records:
            self.send_records_to_recycle_bin(records)
        elif chosen == open_explorer and records:
            self.open_path_in_explorer(records[0].path, len(records) != 1)
        elif chosen == pin_action and records:
            self.navigation.toggle_pin_record(records[0])
        elif chosen == merge_new_action and can_merge:
            self.merge_into_new_logfolder(records)
        elif chosen == append_action and append_target is not None:
            self.append_into_existing_record(records)
        elif chosen == export_action and records:
            self.export_records(records)

    def merge_into_new_logfolder(self, records: Iterable[LogRecord]) -> None:
        self._show_merge_dialog(list(records))

    def append_into_existing_record(self, records: Iterable[LogRecord]) -> None:
        records = list(records)
        target = PreparedMerge.find_append_target(records)
        if target is not None:
            self._show_merge_dialog(records, target)

    def _show_merge_dialog(
        self, records: list[LogRecord], target: LogRecord | None = None
    ) -> None:
        dialog = MergeDialog(
            self.parent_widget, records, self.navigation.base_dir, target=target
        )
        dialog.files_written.connect(self.files_written.emit)
        dialog.exec()

    def rename_record_title(self, record: LogRecord) -> None:
        dialog = QInputDialog(self.parent_widget)
        dialog.setWindowTitle("Rename Log")
        dialog.setLabelText("Enter new title:")
        dialog.setTextValue(record.title)
        dialog.setInputMode(QInputDialog.TextInput)
        dialog.resize(max(dialog.sizeHint().width(), 600), dialog.sizeHint().height())
        if dialog.exec() != QDialog.Accepted:
            return
        new_title = dialog.textValue().strip()
        if new_title == record.title:
            return
        record.meta.update(title=new_title)
        self.records_changed.emit()

    def make_note(self) -> None:
        dialog = _MakeNoteDialog(self.parent_widget)
        if dialog.exec() == QDialog.Accepted:
            self.create_note(dialog.id_edit.text(), dialog.title_edit.text().strip())

    def create_note(self, log_id: str, title: str) -> bool:
        try:
            log_id = _validated_log_id(log_id)
        except ValueError as exc:
            QMessageBox.warning(self.parent_widget, "Invalid ID", str(exc))
            return False
        target_path = self.navigation.base_dir / log_id
        created = False
        try:
            target_path.mkdir(exist_ok=False)
            created = True
            LogMetadata(target_path / "metadata.json", title=title, create=True)
        except FileExistsError:
            QMessageBox.warning(
                self.parent_widget,
                "ID Already Exists",
                f"A directory with ID {log_id!r} already exists.",
            )
            return False
        except OSError as exc:
            if created:
                try:
                    (target_path / "metadata.json").unlink(missing_ok=True)
                    target_path.rmdir()
                except OSError:
                    pass
            QMessageBox.warning(
                self.parent_widget,
                "Could Not Make Note",
                f"Failed to create log folder:\n{exc}",
            )
            return False
        self.navigation.refresh_logs(preferred_path=target_path)
        return True

    def change_record_id(self, record: LogRecord) -> None:
        dialog = QInputDialog(self.parent_widget)
        dialog.setWindowTitle("Change Log ID")
        dialog.setLabelText('Enter new ID (e.g. 5.1, "foobar"):')
        dialog.setTextValue(record.path.name)
        dialog.setInputMode(QInputDialog.TextInput)
        dialog.resize(max(dialog.sizeHint().width(), 600), dialog.sizeHint().height())
        if dialog.exec() == QDialog.Accepted:
            self.rename_record_id(record, dialog.textValue())

    def rename_record_id(self, record: LogRecord, new_id: str) -> bool:
        try:
            new_id = _validated_log_id(new_id)
        except ValueError as exc:
            QMessageBox.warning(self.parent_widget, "Invalid ID", str(exc))
            return False
        old_path = record.path
        if new_id == old_path.name:
            return False
        target_path = old_path.parent / new_id
        if target_path.exists():
            QMessageBox.warning(
                self.parent_widget,
                "ID Already Exists",
                f"A directory with ID {new_id!r} already exists.",
            )
            return False
        try:
            old_path.rename(target_path)
        except OSError as exc:
            QMessageBox.warning(
                self.parent_widget,
                "Could Not Change ID",
                f"Failed to rename log folder:\n{exc}",
            )
            self.navigation.refresh_logs()
            return False
        self.path_renamed.emit(old_path, target_path)
        return True

    def set_record_star_count(
        self, record: LogRecord, count: int, refresh: bool = True
    ) -> bool:
        count = max(int(count), 0)
        if record.star == count:
            return False
        record.meta.update(star=count)
        if refresh:
            self.records_changed.emit()
        return True

    def set_record_trash(
        self, record: LogRecord, value: bool, refresh: bool = True
    ) -> bool:
        value = bool(value)
        if record.trash == value:
            return False
        record.meta.update(trash=value)
        if refresh:
            self.records_changed.emit()
        return True

    def set_records_star_count(self, records: Iterable[LogRecord], count: int) -> None:
        changed = False
        for record in records:
            changed |= self.set_record_star_count(record, count, refresh=False)
        if changed:
            self.records_changed.emit()

    def set_records_trash(self, records: Iterable[LogRecord], value: bool) -> None:
        changed = False
        for record in records:
            changed |= self.set_record_trash(record, value, refresh=False)
        if changed:
            self.records_changed.emit()

    def open_path_in_explorer(self, path: Path, select: bool = False) -> None:
        open_in_file_manager(path, select=select, parent=self.parent_widget)

    def export_records(self, records: Iterable[LogRecord]) -> None:
        records_list = list(records)
        if not records_list:
            return
        chosen = QFileDialog.getExistingDirectory(
            self.parent_widget,
            "Select new parent folder for export",
            str(
                self.navigation.base_dir.parent
                if self.navigation.base_dir.parent.exists()
                else self.navigation.base_dir
            ),
        )
        if not chosen:
            return
        destination_parent = Path(chosen)
        try:
            exported_paths = export_records(records_list, destination_parent)
        except Exception as exc:
            QMessageBox.warning(
                self.parent_widget,
                "Export Failed",
                f"Failed to export selected log folders:\n{exc}",
            )
            return
        if len(exported_paths) == 1:
            message = f"Exported 1 log folder to:\n{exported_paths[0]}"
        else:
            message = f"Exported {len(exported_paths)} log folders to parent folder:\n{destination_parent}"
        QMessageBox.information(self.parent_widget, "Export Complete", message)

    def send_records_to_recycle_bin(self, records: Iterable[LogRecord]) -> None:
        records_list = list(records)
        if not records_list:
            return
        if len(records_list) == 1:
            message = f"Send log folder #{records_list[0].log_id} to Recycle Bin?\n\nPath: {records_list[0].path}\n\nThis operation can be undone from the Recycle Bin."
        else:
            message = (
                f"Send {len(records_list)} log folders to Recycle Bin?\n\nIDs: "
                + ", ".join(f"#{r.log_id}" for r in records_list[:10])
            )
            if len(records_list) > 10:
                message += f", ... (+{len(records_list) - 10} more)"
            message += "\n\nThis operation can be undone from the Recycle Bin."
        if (
            QMessageBox.question(
                self.parent_widget,
                "Confirm Send to Recycle Bin",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        failed_paths = []
        for record in records_list:
            try:
                send2trash(str(record.path))
            except Exception as exc:
                failed_paths.append(f"{record.path} ({exc})")
        if failed_paths:
            error_msg = "Failed to send some folders to Recycle Bin:\n\n" + "\n".join(
                failed_paths[:5]
            )
            if len(failed_paths) > 5:
                error_msg += f"\n... and {len(failed_paths) - 5} more"
            QMessageBox.warning(self.parent_widget, "Error", error_msg)
        self.records_changed.emit()

    def shortcut_set_star(self, count: int) -> None:
        records = self.get_selected_records()
        if records:
            self.set_records_star_count(records, count)

    def shortcut_toggle_star(self) -> None:
        records = self.get_selected_records()
        if records:
            self.set_records_star_count(
                records, 0 if all(record.star > 0 for record in records) else 1
            )

    def shortcut_send_to_recycle_bin(self) -> None:
        records = self.get_selected_records()
        if records:
            self.send_records_to_recycle_bin(records)

    def shortcut_toggle_trash(self) -> None:
        records = self.get_selected_records()
        if records:
            self.set_records_trash(records, not all(record.trash for record in records))

    def shortcut_open_in_explorer(self) -> None:
        records = self.get_selected_records()
        if records:
            self.open_path_in_explorer(records[0].path, len(records) != 1)

    def shortcut_rename_title(self) -> None:
        records = self.get_selected_records()
        if len(records) == 1:
            self.rename_record_title(records[0])

    def shortcut_change_id(self) -> None:
        records = self.get_selected_records()
        if len(records) == 1:
            self.change_record_id(records[0])

    def shortcut_make_note(self) -> None:
        self.make_note()

    def shortcut_pin_record(self) -> None:
        records = self.get_selected_records()
        if records:
            self.navigation.toggle_pin_record(records[0])
