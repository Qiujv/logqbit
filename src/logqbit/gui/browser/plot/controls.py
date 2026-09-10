"""Plot widget for stored log records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QPalette,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QApplication,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

if TYPE_CHECKING:
    pass

from logqbit.catalog import resolve_plot_columns


class _CompactTagDelegate(QStyledItemDelegate):
    """Tighten tag padding while retaining the active native Qt style."""

    def __init__(self, horizontal_padding: int, parent: QWidget) -> None:
        super().__init__(parent)
        self._horizontal_padding = horizontal_padding

    def paint(self, painter, option, index) -> None:
        display_option = QStyleOptionViewItem(option)
        self.initStyleOption(display_option, index)
        text = display_option.text
        display_option.text = ""
        style = (
            display_option.widget.style()
            if display_option.widget
            else QApplication.style()
        )
        style.drawControl(
            QStyle.CE_ItemViewItem,
            display_option,
            painter,
            display_option.widget,
        )

        color_role = (
            QPalette.HighlightedText
            if display_option.state & QStyle.State_Selected
            else QPalette.Text
        )
        painter.save()
        painter.setPen(display_option.palette.color(color_role))
        painter.drawText(
            display_option.rect.adjusted(
                self._horizontal_padding,
                0,
                -self._horizontal_padding,
                0,
            ),
            Qt.AlignCenter,
            text,
        )
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802
        display_option = QStyleOptionViewItem(option)
        self.initStyleOption(display_option, index)
        text_width = display_option.fontMetrics.horizontalAdvance(display_option.text)
        if display_option.text != TagBar._SEP:
            text_width = max(
                text_width,
                display_option.fontMetrics.horizontalAdvance("00"),
            )
        size = super().sizeHint(option, index)
        size.setWidth(text_width + self._horizontal_padding * 2)
        return size


# Keep TagBar sizing in a delegate, rather than a stylesheet, so native palettes
# remain responsive to light and dark theme changes.


class _TagListWidget(QListWidget):
    """Clear the transient drag selection after every completed drag."""

    def startDrag(self, supported_actions) -> None:  # noqa: N802
        try:
            super().startDrag(supported_actions)
        finally:
            self.clearSelection()


class TagBar(QWidget):
    """Assign columns to plot roles by dragging between sections."""

    changed = Signal()
    save_clicked = Signal()

    _SEP = "|"
    _GRAY = QColor("#888888")
    _ITEM_SPACING = 2
    _ITEM_HORIZONTAL_PADDING = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(self._ITEM_SPACING)
        layout.addWidget(QLabel("axes | fields:"))

        self._list = _TagListWidget()
        self._list.setFlow(QListView.LeftToRight)
        self._list.setWrapping(False)
        self._list.setSpacing(self._ITEM_SPACING)
        self._list.setItemDelegate(
            _CompactTagDelegate(self._ITEM_HORIZONTAL_PADDING, self._list)
        )
        self._list.setDragDropMode(QListWidget.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setToolTip("Click to toggle field. Drag to reorder.")
        row_height = self._list.fontMetrics().height() + 4
        self._list.setFixedHeight(row_height + self._list.frameWidth() * 2)
        self._list.installEventFilter(self)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.itemClicked.connect(self._toggle_item_role)

        model = self._list.model()
        model.rowsInserted.connect(lambda *_: self._on_model_changed())
        model.rowsRemoved.connect(lambda *_: self._on_model_changed())
        model.rowsMoved.connect(lambda *_: self._on_model_changed())
        model.layoutChanged.connect(lambda: self._on_model_changed())

        self._loading = False
        self._columns: tuple[str, ...] = ()
        self._groupby_checks: dict[str, QCheckBox] = {}
        layout.addWidget(self._list)

        self.groupby_button = QToolButton()
        self.groupby_button.setText("group by")
        self.groupby_button.setPopupMode(QToolButton.InstantPopup)
        self.groupby_button.setEnabled(False)
        self.groupby_menu = QMenu(self.groupby_button)
        self._groupby_panel = QWidget(self.groupby_menu)
        self._groupby_layout = QVBoxLayout(self._groupby_panel)
        self._groupby_layout.setContentsMargins(6, 4, 6, 4)
        self._groupby_layout.setSpacing(2)
        self._groupby_action = QWidgetAction(self.groupby_menu)
        self._groupby_action.setDefaultWidget(self._groupby_panel)
        self.groupby_menu.addAction(self._groupby_action)
        self.groupby_button.setMenu(self.groupby_menu)
        layout.addWidget(self.groupby_button)

    def _on_model_changed(self) -> None:
        if self._loading:
            return
        self._list.clearSelection()
        axes, fields, _ = self._split()
        conflicts = set(axes + fields).intersection(self.groupby)
        if conflicts:
            self._loading = True
            try:
                for column in conflicts:
                    self._groupby_checks[column].setChecked(False)
                self._update_groupby_button()
            finally:
                self._loading = False
        self._update_item_colors()
        self.changed.emit()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction("Save", self.save_clicked.emit)
        menu.exec(self._list.mapToGlobal(pos))

    def _toggle_item_role(self, item: QListWidgetItem) -> None:
        item_index = self._list.row(item)
        separators = [
            index
            for index in range(self._list.count())
            if self._list.item(index).text() == self._SEP
        ]
        if item.text() == self._SEP or len(separators) < 2:
            self._list.clearSelection()
            return
        if item_index < separators[0]:
            self._list.clearSelection()
            return

        was_field = separators[0] < item_index < separators[1]
        self._loading = True
        try:
            item = self._list.takeItem(item_index)
            separator_indices = [
                index
                for index in range(self._list.count())
                if self._list.item(index).text() == self._SEP
            ]
            second_separator = separator_indices[1]
            self._list.insertItem(
                second_separator + 1 if was_field else second_separator,
                item,
            )
        finally:
            self._loading = False
        self._list.clearSelection()
        self._on_model_changed()

    def eventFilter(self, obj, event):
        if obj is self._list and event.type() == QEvent.Wheel:
            bar = self._list.horizontalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y() // 2)
            return True
        return super().eventFilter(obj, event)

    def _make_sep(self) -> QListWidgetItem:
        item = QListWidgetItem(self._SEP)
        item.setForeground(self._GRAY)
        return item

    def _make_tag_item(self, name: str) -> QListWidgetItem:
        return QListWidgetItem(name)

    def _update_item_colors(self) -> None:
        separator_count = 0
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item.text() == self._SEP:
                separator_count += 1
            elif separator_count >= 2:
                item.setForeground(self._GRAY)
            else:
                item.setData(Qt.ForegroundRole, None)

    def set_columns(
        self,
        columns: Sequence[str],
        plot_axes: Sequence[str],
        plot_fields: Sequence[str],
        plot_groupby: Sequence[str],
    ) -> None:
        resolved = resolve_plot_columns(
            columns,
            plot_axes,
            plot_fields,
            plot_groupby,
        )

        self._loading = True
        try:
            self._columns = tuple(dict.fromkeys(str(column) for column in columns))
            self._set_groupby_options(resolved.groupby)
            self._set_list_columns(resolved)
        finally:
            self._loading = False

    def _set_list_columns(self, resolved) -> None:
        self._list.clear()
        for name in resolved.axes:
            self._list.addItem(self._make_tag_item(name))
        self._list.addItem(self._make_sep())
        for name in resolved.fields:
            self._list.addItem(self._make_tag_item(name))
        self._list.addItem(self._make_sep())
        for name in (*resolved.groupby, *resolved.ignored):
            item = self._make_tag_item(name)
            item.setForeground(self._GRAY)
            self._list.addItem(item)

    def _set_groupby_options(self, selected: Sequence[str]) -> None:
        while self._groupby_layout.count():
            layout_item = self._groupby_layout.takeAt(0)
            if widget := layout_item.widget():
                widget.deleteLater()
        selected_set = set(selected)
        self._groupby_checks = {}
        for column in self._columns:
            checkbox = QCheckBox(column, self._groupby_panel)
            checkbox.setChecked(column in selected_set)
            checkbox.toggled.connect(self._on_groupby_toggled)
            self._groupby_layout.addWidget(checkbox)
            self._groupby_checks[column] = checkbox
        self._update_groupby_button()

    def _on_groupby_toggled(self) -> None:
        if self._loading:
            return
        axes, fields, _ = self._split()
        resolved = resolve_plot_columns(
            self._columns,
            axes,
            fields,
            self.groupby,
        )
        self._loading = True
        try:
            self._set_list_columns(resolved)
            self._update_groupby_button()
        finally:
            self._loading = False
        self.changed.emit()

    def _update_groupby_button(self) -> None:
        count = len(self.groupby)
        self.groupby_button.setText(f"group by ({count})" if count else "group by")
        self.groupby_button.setEnabled(bool(self._columns))
        self.groupby_button.setToolTip(
            ", ".join(self.groupby) if count else "Select columns used to group plots"
        )

    def _split(self) -> tuple[list[str], list[str], list[str]]:
        sections: list[list[str]] = []
        current: list[str] = []
        for index in range(self._list.count()):
            text = self._list.item(index).text()
            if text == self._SEP:
                sections.append(current)
                current = []
            else:
                current.append(text)
        sections.append(current)
        while len(sections) < 3:
            sections.append([])
        return sections[0], sections[1], sections[2]

    @property
    def axes(self) -> list[str]:
        return self._split()[0]

    @property
    def fields(self) -> list[str]:
        return self._split()[1]

    @property
    def groupby(self) -> list[str]:
        return [
            column
            for column in self._columns
            if self._groupby_checks[column].isChecked()
        ]
