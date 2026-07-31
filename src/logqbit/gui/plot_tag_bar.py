"""Tag bar for assigning data columns to plot roles."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QWidget,
)


def _ordered_unique(items: Sequence[str]) -> list[str]:
    """Remove duplicates while preserving the first occurrence of each item."""
    return list(dict.fromkeys(items))


def _partition_columns(
    columns: Sequence[str],
    plot_axes: Sequence[str],
    plot_fields: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return ordered, disjoint axes, fields, and ignored columns."""
    columns = _ordered_unique(columns)
    axes = [c for c in _ordered_unique(plot_axes) if c in columns]
    fields = [c for c in _ordered_unique(plot_fields) if c in columns and c not in axes]
    ignored = [c for c in columns if c not in axes and c not in fields]

    if not axes and ignored:
        axes.append(ignored.pop(0))
    if not fields and ignored:
        fields.append(ignored.pop(0))

    return axes, fields, ignored


class TagBar(QWidget):
    """Assign columns to axes, fields, and ignored sections by dragging."""

    changed = Signal()
    save_clicked = Signal()

    _SEP = "|"
    _GRAY = QColor("#888888")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)

        hint = QLabel("axes | fields:")
        layout.addWidget(hint)

        self._list = QListWidget()
        self._list.setFlow(QListView.LeftToRight)
        self._list.setWrapping(False)
        self._list.setDragDropMode(QListWidget.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        row_h = self._list.fontMetrics().height() + 4
        self._list.setFixedHeight(row_h + self._list.frameWidth() * 2)
        self._list.installEventFilter(self)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.itemClicked.connect(lambda _: self._list.clearSelection())

        model = self._list.model()
        model.rowsInserted.connect(lambda *_: self._on_model_changed())
        model.rowsRemoved.connect(lambda *_: self._on_model_changed())
        model.rowsMoved.connect(lambda *_: self._on_model_changed())
        model.layoutChanged.connect(lambda: self._on_model_changed())

        self._loading = False
        layout.addWidget(self._list)

    def _on_model_changed(self) -> None:
        if self._loading:
            return
        self._update_item_colors()
        self.changed.emit()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction("Save", self.save_clicked.emit)
        menu.exec(self._list.mapToGlobal(pos))

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

    def _update_item_colors(self) -> None:
        sep_count = 0
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.text() == self._SEP:
                sep_count += 1
            elif sep_count >= 2:
                item.setForeground(self._GRAY)
            else:
                item.setData(Qt.ForegroundRole, None)

    def set_columns(
        self,
        columns: Sequence[str],
        plot_axes: Sequence[str],
        plot_fields: Sequence[str],
    ) -> None:
        axes, fields, ignored = _partition_columns(
            columns,
            plot_axes,
            plot_fields,
        )

        self._loading = True
        try:
            self._list.clear()
            for name in axes:
                self._list.addItem(name)
            self._list.addItem(self._make_sep())
            for name in fields:
                self._list.addItem(name)
            self._list.addItem(self._make_sep())
            for name in ignored:
                item = QListWidgetItem(name)
                item.setForeground(self._GRAY)
                self._list.addItem(item)
        finally:
            self._loading = False

    def _split(self) -> tuple[list[str], list[str], list[str]]:
        sections: list[list[str]] = []
        current: list[str] = []
        for i in range(self._list.count()):
            text = self._list.item(i).text()
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
