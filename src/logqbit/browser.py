"""Interactive browser for log folders."""

from __future__ import annotations

import logging
import numbers
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pyarrow.ipc
from PySide6.QtCore import (
    QAbstractTableModel,
    QFileSystemWatcher,
    QModelIndex,
    QSettings,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from send2trash import send2trash

try:  # pragma: no cover - fallback for direct execution
    from .metadata import LogMetadata
    from .plotter import PlotManager, PlotWindow
except ImportError:  # pragma: no cover - fallback for direct execution
    from metadata import LogMetadata  # type: ignore
    from plotter import PlotManager, PlotWindow  # type: ignore

logger = logging.getLogger(__name__)

# Constants
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
REFRESH_DEBOUNCE_MS = 250

COL_ID = 0
COL_TITLE = 1
COL_ROWS = 2
COL_PLOT_AXES = 3
COL_CREATE_TIME = 4
COL_CREATE_MACHINE = 5

SETTINGS_ORG = "LogQbit"
SETTINGS_APP = "LogBrowser"
SETTINGS_RECENT_DIRS_KEY = "recent/directories"
SETTINGS_THEME_KEY = "ui/theme"

# Tab indices for detail panel
TAB_CONST = 0
TAB_DATA = 1
TAB_PLOT = 2


def _load_window_icon() -> QIcon:
    try:
        icon_path = files("logqbit") / "assets" / "browser.svg"
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            return icon
    except Exception as exc:
        logger.debug(f"Failed to load window icon: {exc}")
    return QIcon()  # Return null icon as fallback


WINDOW_ICON = _load_window_icon()


@dataclass
class LogRecord:
    log_id: int
    path: Path
    data_path: Path | None = None
    yaml_path: Path | None = None

    # Data metadata
    row_count: int = 0
    columns: list[str] = field(default_factory=list)

    # Cached data
    data_frame: pd.DataFrame | None = field(default=None, repr=False)

    # Metadata accessor (always available)
    meta: LogMetadata = field(init=False, repr=False)

    def __post_init__(self):
        self.meta = LogMetadata(self.path / "metadata.json", create=True)

    def load_dataframe(self) -> pd.DataFrame | None:
        if self.data_frame is not None:
            return self.data_frame
        if not self.data_path:
            return None
        try:
            self.data_frame = pd.read_feather(self.data_path)
            self.row_count = len(self.data_frame)
            self.columns = [str(col) for col in self.data_frame.columns]
            return self.data_frame
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to read feather file %s: %s", self.data_path, exc)
            return None

    def read_yaml_text(self) -> str:
        if not self.yaml_path or not self.yaml_path.exists():
            return "const.yaml not found."
        try:
            text = self.yaml_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to read yaml file %s: %s", self.yaml_path, exc)
            return f"Failed to read const.yaml: {exc}"
        return text if text.strip() else "(const.yaml is empty)"

    def list_image_files(self) -> list[Path]:
        files: list[Path] = []
        for child in self.path.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(child)
        files.sort()
        return files

    @staticmethod
    def scan_directory(directory: Path) -> list["LogRecord"]:
        records: list[LogRecord] = []
        if not directory.exists() or not directory.is_dir():
            return records

        for path in directory.iterdir():
            if not path.is_dir() or not path.name.isdigit():
                continue

            exp_id = int(path.name)
            yaml_path = path / "const.yaml"
            data_path = path / "data.feather"
            if not yaml_path.exists():
                yaml_path = None
            if not data_path.exists():
                data_path = None

            # Read feather summary if available
            row_count: int = 0
            columns: list[str] = []
            if data_path:
                try:
                    with pyarrow.ipc.open_file(data_path) as reader:
                        schema = reader.schema
                        row_count = sum(
                            reader.get_batch(i).num_rows
                            for i in range(reader.num_record_batches)
                        )
                        columns = [str(name) for name in schema.names]
                except FileNotFoundError:
                    pass
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Failed to inspect feather file %s: %s", data_path, exc
                    )

            record = LogRecord(
                log_id=exp_id,
                path=path,
                row_count=row_count,
                columns=columns,
                data_path=data_path,
                yaml_path=yaml_path,
            )

            records.append(record)

        return records


class LogListTableModel(QAbstractTableModel):
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

    def update_record(self, record: LogRecord) -> None:
        try:
            row = self._records.index(record)
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, self.columnCount() - 1)
            )
        except ValueError:
            pass

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else 6

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: D401
        if not index.isValid():
            return None

        record = self._records[index.row()]
        col = index.column()
        meta = record.meta

        # Display text
        if role == Qt.DisplayRole:
            if col == COL_ID:
                return record.log_id
            elif col == COL_TITLE:
                star_count = max(int(meta.star), 0)
                star_prefix = "⭐" * star_count
                parts: list[str] = []
                if meta.trash:
                    parts.append("🗑️")
                if star_prefix:
                    parts.append(star_prefix)
                title_text = meta.title or "(untitled)"
                parts.append(title_text)
                return " ".join(parts)
            elif col == COL_ROWS:
                return record.row_count
            elif col == COL_CREATE_TIME:
                return meta.create_time
            elif col == COL_CREATE_MACHINE:
                return meta.create_machine
            elif col == COL_PLOT_AXES:
                if meta.plot_axes:
                    n_axes = len(meta.plot_axes)
                    return f"{n_axes}, " + ", ".join([i[:3] for i in meta.plot_axes])
                else:
                    return ""

        # Font styling
        elif role == Qt.FontRole and col == COL_TITLE:
            star_count = max(int(meta.star), 0)
            is_bold = star_count > 0
            is_trash = meta.trash
            if is_bold and is_trash:
                return self._bold_strikeout_font
            elif is_bold:
                return self._bold_font
            elif is_trash:
                return self._strikeout_font
            return None

        # Tooltip
        elif role == Qt.ToolTipRole:
            if col == COL_TITLE:
                return meta.title or "(untitled)"
            elif col == COL_PLOT_AXES:
                return ", ".join(meta.plot_axes) if meta.plot_axes else "(no plot axes)"

        # User data - store record reference
        elif role == Qt.UserRole and col == COL_ID:
            return record

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


class PandasTableModel(QAbstractTableModel):
    """Table model for displaying pandas DataFrames with optional preview limit."""

    def __init__(
        self,
        frame: pd.DataFrame,
        parent: QWidget | None = None,
        highlight_columns: Iterable[str] | None = None,
        preview_limit: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._df = frame
        self._preview_limit = preview_limit
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

    def set_preview_limit(self, limit: int | None) -> None:
        old_count = self.rowCount()
        self._preview_limit = limit
        new_count = self.rowCount()
        if new_count != old_count:
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
        if role == Qt.FontRole and orientation == Qt.Horizontal:
            column_name = str(self._df.columns[section])
            if column_name in self._highlight:
                return self._bold_font
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class ScaledImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
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


class SettingsManager:
    def __init__(self):
        self._settings = QSettings(
            QSettings.IniFormat, QSettings.UserScope, SETTINGS_ORG, SETTINGS_APP
        )
        self._recent_directories: list[Path] = []

    def load_recent_directories(self) -> list[Path]:
        stored = self._settings.value(SETTINGS_RECENT_DIRS_KEY, [])
        if isinstance(stored, str):
            candidates = [stored]
        elif isinstance(stored, (list, tuple)):
            candidates = list(stored)
        else:
            candidates = []
        recent_paths: list[Path] = []
        for item in candidates:
            text = str(item)
            if not text:
                continue
            try:
                path = Path(text)
            except Exception:
                continue
            if path not in recent_paths:
                recent_paths.append(path)
        self._recent_directories = recent_paths[:10]
        return self._recent_directories

    def save_recent_directories(self, directories: list[Path]) -> None:
        self._recent_directories = directories[:10]
        self._settings.setValue(
            SETTINGS_RECENT_DIRS_KEY, [str(path) for path in self._recent_directories]
        )
        self._settings.sync()

    def update_recent_directories(self, path: Path) -> list[Path]:
        self.load_recent_directories()

        resolved = Path(path)
        entries = [resolved]
        for existing in self._recent_directories:
            if existing != resolved:
                entries.append(existing)
            if len(entries) >= 10:
                break
        self._recent_directories = entries
        self.save_recent_directories(self._recent_directories)
        return self._recent_directories

    def load_theme_mode(self) -> str:
        saved_mode = self._settings.value(SETTINGS_THEME_KEY, "system")
        if saved_mode in ThemeManager.THEME_MODES:
            return saved_mode
        return "system"

    def save_theme_mode(self, mode: str) -> None:
        self._settings.setValue(SETTINGS_THEME_KEY, mode)
        self._settings.sync()


class ThemeManager:
    THEME_MODES = ["light", "dark", "system"]

    def __init__(self, app: QApplication):
        self.app = app
        self._system_palette = app.palette() if app else QPalette()
        self._current_mode = "system"

    def apply_theme(self, mode: str) -> None:
        self._current_mode = mode
        style_hints = getattr(self.app, "styleHints", None)
        can_use_color_scheme = False
        hints = None
        if style_hints and hasattr(Qt, "ColorScheme"):
            hints = style_hints()
            can_use_color_scheme = hasattr(hints, "setColorScheme")

        if can_use_color_scheme:
            # Qt 6.5+ with ColorScheme support
            unknown_scheme = getattr(Qt.ColorScheme, "Unknown", Qt.ColorScheme.Light)
            scheme_map = {
                "dark": Qt.ColorScheme.Dark,
                "light": Qt.ColorScheme.Light,
                "system": unknown_scheme,
            }
            hints.setColorScheme(scheme_map.get(mode, unknown_scheme))
            palette = self._system_palette
        else:
            # Fallback to manual palette creation
            palette_map = {
                "dark": self._create_dark_palette(),
                "light": self._create_light_palette(),
                "system": self._system_palette,
            }
            palette = palette_map.get(mode, self._system_palette)

        self.app.setPalette(palette)
        self.app.setStyleSheet("")

    def _create_light_palette(self) -> QPalette:
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(250, 250, 250))
        palette.setColor(QPalette.WindowText, QColor(30, 30, 30))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(30, 30, 30))
        palette.setColor(QPalette.Text, QColor(30, 30, 30))
        palette.setColor(QPalette.Button, QColor(245, 245, 245))
        palette.setColor(QPalette.ButtonText, QColor(30, 30, 30))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(0, 122, 204))
        palette.setColor(QPalette.Highlight, QColor(51, 153, 255))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        return palette

    def _create_dark_palette(self) -> QPalette:
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(37, 37, 38))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(100, 160, 220))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        return palette

    def get_theme_button_emoji(self, mode: str) -> str:
        emoji_map = {
            "light": "🌝",
            "dark": "🌚",
            "system": "🌗",
        }
        return emoji_map.get(mode, "🌗")

    def get_theme_tooltip(self, mode: str) -> str:
        tooltip_map = {
            "light": "Light mode",
            "dark": "Dark mode",
            "system": "Follow system theme",
        }
        return tooltip_map.get(mode, "Follow system theme")


class DataViewManager:
    INITIAL_PREVIEW_LIMIT = 100
    PREVIEW_INCREMENT = 1000

    def __init__(
        self,
        parent: QWidget | None = None,
        load_more_callback=None,
        plot_axes_changed_callback=None,
    ):
        """Create data view manager with its own UI components.

        Args:
            parent: Parent widget
            load_more_callback: Callback when "Show More Rows" is clicked
            plot_axes_changed_callback: Callback(record, column_name, enabled) when plot axes toggled
        """
        self._load_more_callback = load_more_callback
        self._plot_axes_changed_callback = plot_axes_changed_callback
        self._current_record: LogRecord | None = None

        # Create UI components
        self.widget = self._create_widget(parent)

    def _create_widget(self, parent: QWidget | None = None) -> QWidget:
        """Create and return the data tab widget."""
        data_tab = QWidget(parent)
        data_layout = QVBoxLayout(data_tab)
        data_layout.setContentsMargins(4, 4, 4, 4)

        self.data_table = QTableView()
        self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data_table.setSortingEnabled(False)
        self.data_table.setWordWrap(False)
        self.data_table.horizontalHeader().setStretchLastSection(False)
        self.data_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        row_height = self.data_table.fontMetrics().height() + 6
        self.data_table.verticalHeader().setDefaultSectionSize(row_height)

        # Set up context menu
        self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.customContextMenuRequested.connect(self._open_context_menu)

        data_layout.addWidget(self.data_table)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self.data_status_label = QLabel("")
        self.data_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_status_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
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

    def set_empty(self, message: str = "No data to display.") -> None:
        self.data_table.setModel(None)
        self.data_status_label.setText(message)
        self.data_load_button.setEnabled(False)

    def show_data_table(self, record: LogRecord, preview_only: bool) -> None:
        self._current_record = record
        dataframe = record.load_dataframe()
        if dataframe is None:
            message = (
                "Data file not found."
                if not record.data_path or not record.data_path.exists()
                else "Failed to load data."
            )
            self.set_empty(message)
            return None

        total_rows = len(dataframe)
        preview_limit = None
        if preview_only and total_rows > self.INITIAL_PREVIEW_LIMIT:
            preview_limit = self.INITIAL_PREVIEW_LIMIT

        # Create model with preview limit
        model = PandasTableModel(
            dataframe,
            self.data_table,
            highlight_columns=record.meta.plot_axes,
            preview_limit=preview_limit,
        )
        self.data_table.setModel(model)
        self.data_table.resizeColumnsToContents()
        row_height = self.data_table.fontMetrics().height() + 6
        self.data_table.verticalHeader().setDefaultSectionSize(row_height)

        # Update status and button
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

    def _open_context_menu(self, point) -> None:
        """Handle context menu on data table."""
        record = self._current_record
        model = self.data_table.model()
        if record is None or model is None:
            return

        # Get column from click position
        index = self.data_table.indexAt(point)
        column = (
            index.column()
            if index.isValid()
            else self.data_table.horizontalHeader().logicalIndexAt(point.x())
        )
        if column < 0:
            return

        column_name = str(model.headerData(column, Qt.Horizontal))
        if not column_name:
            return

        # Create context menu
        menu = QMenu(self.data_table)
        is_tracked = column_name in record.meta.plot_axes
        toggle_action = menu.addAction("Toggle Plot Axes")
        toggle_action.setCheckable(True)
        toggle_action.setChecked(is_tracked)

        # Show menu and handle result
        chosen = menu.exec(self.data_table.viewport().mapToGlobal(point))
        if chosen == toggle_action:
            self._toggle_plot_axes(record, column_name, not is_tracked)

    def _toggle_plot_axes(
        self, record: LogRecord, column_name: str, enable: bool
    ) -> None:
        """Toggle plot axes tracking for a column."""
        column_name = str(column_name)
        if not column_name:
            return

        updated = list(record.meta.plot_axes)
        if enable:
            if column_name in updated:
                return
            updated.append(column_name)
        else:
            if column_name not in updated:
                return
            updated = [item for item in updated if item != column_name]

        record.meta.plot_axes = updated

        # Notify parent via callback
        if self._plot_axes_changed_callback:
            self._plot_axes_changed_callback(record, column_name, enable)

    def load_more_data(self, record: LogRecord) -> None:
        model = self.data_table.model()
        if not isinstance(model, PandasTableModel):
            # No preview active, show all data
            self.show_data_table(record, preview_only=False)
            return

        total_rows = model.get_total_rows()
        current_limit = model.rowCount()

        if current_limit >= total_rows:
            # Already showing all data
            return

        # Increase limit
        new_limit = min(current_limit + self.PREVIEW_INCREMENT, total_rows)
        model.set_preview_limit(new_limit)

        # Update status
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


class LogBrowserWindow(QMainWindow):
    def __init__(
        self, directory: Path | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        if not WINDOW_ICON.isNull():
            self.setWindowIcon(WINDOW_ICON)
        self.resize(1200, 700)

        self.settings_manager = SettingsManager()

        # State
        self._base_dir = Path(directory) if directory else Path.cwd()
        self._current_record: LogRecord | None = None
        self._show_trash = True
        self._image_tab_indices: list[int] = []
        self._shortcuts: list[QAction] = []
        self._list_refresh_pending = False
        self._detail_refresh_pending = False

        # Theme management
        app = QApplication.instance()
        self.theme_manager = ThemeManager(app) if app else None
        self._theme_mode = self.settings_manager.load_theme_mode()

        # Recent directories
        recent = self.settings_manager.load_recent_directories()
        if directory:
            self._base_dir = Path(directory)
            self.settings_manager.update_recent_directories(self._base_dir)
        elif recent:
            self._base_dir = recent[0]

        # File watchers
        self._dir_watcher = QFileSystemWatcher(self)
        self._detail_watcher = QFileSystemWatcher(self)
        self._dir_watcher.directoryChanged.connect(self._schedule_list_refresh)
        self._dir_watcher.fileChanged.connect(self._schedule_list_refresh)
        self._detail_watcher.directoryChanged.connect(self._schedule_detail_refresh)
        self._detail_watcher.fileChanged.connect(self._schedule_detail_refresh)

        # Build UI
        self._build_ui()
        if self.theme_manager:
            self.theme_manager.apply_theme(self._theme_mode)
        self._update_theme_button()
        self._update_window_title()
        self._sync_directory_watcher()
        self.refresh_logs()

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_bar = self._create_top_bar()
        layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal, central)

        # Left: Log table
        self.log_table, self.table_model, self.table_proxy = self._create_log_table(
            splitter
        )

        # Right: Detail panel
        detail_widget = self._create_detail_panel(splitter)

        splitter.addWidget(self.log_table)
        splitter.addWidget(detail_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 600])

        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self._setup_shortcuts()
        self._rebuild_directory_menu()

    def _create_top_bar(self) -> QHBoxLayout:
        top_bar = QHBoxLayout()
        self.directory_label = QLabel(str(self._base_dir))
        self.directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.directory_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.directory_button = QToolButton()
        self.directory_button.setText("Change dir...")
        self.directory_button.setPopupMode(QToolButton.InstantPopup)
        self._directory_menu = QMenu(self.directory_button)
        self.directory_button.setMenu(self._directory_menu)
        refresh_button = QPushButton("🔄️Refresh")
        refresh_button.clicked.connect(self._on_refresh_clicked)
        self.theme_button = QPushButton()
        self.theme_button.setFixedWidth(36)
        self.theme_button.setFocusPolicy(Qt.NoFocus)
        self.theme_button.clicked.connect(self._on_theme_button_clicked)
        top_bar.addWidget(QLabel("Directory:"))
        top_bar.addWidget(self.directory_label)
        top_bar.addWidget(self.directory_button)
        top_bar.addWidget(refresh_button)
        top_bar.addWidget(self.theme_button)
        return top_bar

    def _create_log_table(
        self, parent: QWidget
    ) -> tuple[QTableView, LogListTableModel, QSortFilterProxyModel]:
        model = LogListTableModel(parent)

        proxy = QSortFilterProxyModel(parent)
        proxy.setSourceModel(model)
        proxy.setSortRole(Qt.DisplayRole)

        table = QTableView(parent)
        table.setModel(proxy)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.setSelectionMode(QTableView.ExtendedSelection)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)

        font_height = table.fontMetrics().height()
        table.verticalHeader().setDefaultSectionSize(font_height + 4)

        header = table.horizontalHeader()
        header.setSectionResizeMode(COL_ID, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_TITLE, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_ROWS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_PLOT_AXES, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_CREATE_TIME, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_CREATE_MACHINE, QHeaderView.ResizeToContents)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(False)  # For compact view.

        table.setColumnHidden(COL_CREATE_TIME, True)
        table.setColumnHidden(COL_CREATE_MACHINE, True)

        table.selectionModel().selectionChanged.connect(self._on_log_selection_changed)
        table.doubleClicked.connect(self._on_log_double_clicked)
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(self._open_table_context_menu)

        table.sortByColumn(COL_ID, Qt.AscendingOrder)

        return table, model, proxy

    def _create_detail_panel(self, parent: QWidget) -> QWidget:
        detail_widget = QWidget(parent)
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(6)

        detail_top = QHBoxLayout()
        self.detail_label = QLabel("No log selected.")
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        detail_top.addWidget(self.detail_label)
        detail_top.addStretch(1)
        detail_layout.addLayout(detail_top)

        self.tab_widget = QTabWidget(detail_widget)

        self.yaml_view = QPlainTextEdit()
        self.yaml_view.setReadOnly(True)
        self.tab_widget.addTab(self.yaml_view, "Const.")

        # Create data view manager and add its widget
        self.data_view_manager = DataViewManager(
            parent=detail_widget,
            load_more_callback=self._on_load_data_clicked,
            plot_axes_changed_callback=self._on_plot_axes_changed,
        )
        self.tab_widget.addTab(self.data_view_manager.widget, "Data")

        # Create plot manager and add its widget
        self.plot_manager = PlotManager(parent=detail_widget)
        self.tab_widget.addTab(self.plot_manager.widget, "Plot")

        # Connect tab changed signal to refresh plot if needed
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        detail_layout.addWidget(self.tab_widget)
        return detail_widget

    def _setup_shortcuts(self) -> None:
        for action in self._shortcuts:
            self.removeAction(action)
        self._shortcuts.clear()

        def add_shortcut(key: int, callback) -> None:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda _checked=False, cb=callback: cb())
            self.addAction(action)
            self._shortcuts.append(action)

        add_shortcut(Qt.Key_Delete, self._shortcut_send_to_recycle_bin)
        add_shortcut(Qt.Key_T, self._shortcut_toggle_trash)
        add_shortcut(Qt.Key_S, self._shortcut_toggle_star)
        add_shortcut(Qt.Key_F2, self._shortcut_rename_title)
        add_shortcut(Qt.Key_0, lambda: self._shortcut_set_star(0))
        add_shortcut(Qt.Key_1, lambda: self._shortcut_set_star(1))
        add_shortcut(Qt.Key_2, lambda: self._shortcut_set_star(2))
        add_shortcut(Qt.Key_3, lambda: self._shortcut_set_star(3))

    def _rebuild_directory_menu(self) -> None:
        if self._directory_menu is None:
            return
        self._directory_menu.clear()
        recent = self.settings_manager._recent_directories
        # Filter out current directory
        menu_items = [path for path in recent if path != self._base_dir]
        for path in menu_items:
            action = self._directory_menu.addAction(str(path))
            action.triggered.connect(
                lambda _checked=False, target=path: self.set_directory(target)
            )
        if menu_items:
            self._directory_menu.addSeparator()
        open_action = self._directory_menu.addAction("Open Other Folder...")
        open_action.triggered.connect(self._open_directory_dialog)
        new_window_action = self._directory_menu.addAction("New Window")
        new_window_action.triggered.connect(
            lambda: self._open_new_window(self._base_dir)
        )

    def _update_theme_button(self) -> None:
        if not self.theme_manager:
            return
        emoji = self.theme_manager.get_theme_button_emoji(self._theme_mode)
        tooltip = self.theme_manager.get_theme_tooltip(self._theme_mode)
        if hasattr(self, "theme_button"):
            self.theme_button.setText(emoji)
            self.theme_button.setToolTip(tooltip)

    def _update_window_title(self) -> None:
        self.setWindowTitle(f"{self._base_dir.name} - LogQbit Browser")

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
        self.refresh_logs()

    def _schedule_detail_refresh(self) -> None:
        if self._detail_refresh_pending:
            return
        self._detail_refresh_pending = True
        QTimer.singleShot(REFRESH_DEBOUNCE_MS, self._run_detail_refresh)

    def _run_detail_refresh(self) -> None:
        self._detail_refresh_pending = False
        self.refresh_current_log()

    def set_directory(self, directory: Path) -> None:
        path = Path(directory)
        if path != self._base_dir:
            self._base_dir = path
            self.directory_label.setText(str(self._base_dir))
            self._update_window_title()
            self._sync_directory_watcher()
            self.refresh_logs()
        else:
            self.directory_label.setText(str(self._base_dir))
        self.settings_manager.update_recent_directories(path)
        self._rebuild_directory_menu()

    def refresh_logs(self) -> None:
        previous_id = self._current_record.log_id if self._current_record else None
        all_records = LogRecord.scan_directory(self._base_dir)

        # Filter out trash if needed
        if self._show_trash:
            records = all_records
        else:
            records = [r for r in all_records if not r.meta.trash]

        self.table_model.set_records(records)

        row_count = self.table_proxy.rowCount()
        if row_count:
            self.detail_label.setText("Select a log to preview.")
            if previous_id is not None:
                # Try to select previous log
                found = False
                for source_row in range(len(records)):
                    if records[source_row].log_id == previous_id:
                        # Map source row to proxy row
                        source_index = self.table_model.index(source_row, 0)
                        proxy_index = self.table_proxy.mapFromSource(source_index)
                        if proxy_index.isValid():
                            self.log_table.selectRow(proxy_index.row())
                            self._current_record = records[source_row]
                            self._load_log(self._current_record)
                            found = True
                            break
                if found:
                    return
            # Select first row if available
            if row_count:
                self.log_table.selectRow(0)
        else:
            if all_records:
                self.detail_label.setText("No logs to display.")
            else:
                self.detail_label.setText("No logs found.")
            self._current_record = None
            self.log_table.clearSelection()
            self._clear_preview_panels()
            self._clear_detail_watcher()

    def refresh_current_log(self) -> None:
        if not self._current_record:
            return
        # Clear cached dataframe to force reload from disk
        self._current_record.data_frame = None
        self._load_log(self._current_record)

    def _on_log_double_clicked(self, proxy_index) -> None:
        """Open a standalone PlotWindow for the double-clicked log record."""
        source_index = self.table_proxy.mapToSource(proxy_index)
        record = self.table_model.get_record(source_index.row())
        if record is None:
            return
        window = PlotWindow(record, parent=None)
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        if not WINDOW_ICON.isNull():
            window.setWindowIcon(WINDOW_ICON)
        # Keep a reference so the window isn't garbage-collected
        if not hasattr(self, "_plot_windows"):
            self._plot_windows: list[PlotWindow] = []
        self._plot_windows.append(window)
        window.destroyed.connect(
            lambda: self._plot_windows.remove(window)
            if window in self._plot_windows
            else None
        )
        window.show()

    def _on_log_selection_changed(self) -> None:
        selected = self.log_table.selectionModel().selectedRows()
        if not selected:
            return
        proxy_index = selected[0]
        source_index = self.table_proxy.mapToSource(proxy_index)
        record = self.table_model.get_record(source_index.row())
        if record is None:
            return
        self._current_record = record
        self._load_log(record)

    def _load_log(self, record: LogRecord) -> None:
        self.detail_label.setText(f"#{record.log_id} - {record.path}")
        self.yaml_view.setPlainText(record.read_yaml_text())
        self.data_view_manager.data_status_label.setText("Loading data preview…")
        self.data_view_manager.data_load_button.setEnabled(False)
        self.data_view_manager.show_data_table(record, preview_only=True)
        image_files = record.list_image_files()
        self._update_image_tabs(image_files)

        # Defer plot refresh if not on plot tab
        current_tab = self.tab_widget.currentIndex()
        defer_plot = current_tab != TAB_PLOT
        self.plot_manager.update_plot_and_controls(record, defer_plot=defer_plot)

        self._update_detail_watcher(record)

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab widget changes to trigger plot refresh if needed."""
        if index == TAB_PLOT:
            self.plot_manager.refresh_if_needed()

    def _clear_preview_panels(self) -> None:
        self.yaml_view.setPlainText("")
        self.data_view_manager.set_empty("")
        self._clear_image_tabs()
        self.plot_manager.reset_plot_state("")

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

    def _update_image_tabs(self, image_files: list[Path]) -> None:
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

    def _update_detail_watcher(self, record: LogRecord) -> None:
        self._clear_detail_watcher()
        watch_paths: list[str] = [str(record.path)]
        for extra in (record.yaml_path, record.data_path, record.meta.path):
            if extra and extra.exists():
                watch_paths.append(str(extra))
        if watch_paths:
            self._detail_watcher.addPaths(watch_paths)

    def _on_load_data_clicked(self) -> None:
        if self._current_record:
            self.data_view_manager.load_more_data(self._current_record)

    def _on_refresh_clicked(self) -> None:
        self.refresh_logs()
        self.refresh_current_log()

    def _on_theme_button_clicked(self) -> None:
        current_index = ThemeManager.THEME_MODES.index(self._theme_mode)
        next_index = (current_index + 1) % len(ThemeManager.THEME_MODES)
        self._theme_mode = ThemeManager.THEME_MODES[next_index]
        if self.theme_manager:
            self.theme_manager.apply_theme(self._theme_mode)
        self.settings_manager.save_theme_mode(self._theme_mode)
        self._update_theme_button()

    def _get_selected_records(self) -> list[LogRecord]:
        selection_model = self.log_table.selectionModel()
        if selection_model is None:
            return []
        records: list[LogRecord] = []
        for proxy_index in selection_model.selectedRows():
            source_index = self.table_proxy.mapToSource(proxy_index)
            record = self.table_model.get_record(source_index.row())
            if record is not None:
                records.append(record)
        return records

    def _open_directory_dialog(self) -> None:
        current = str(self._base_dir)
        chosen = QFileDialog.getExistingDirectory(self, "Select log directory", current)
        if chosen:
            self.set_directory(Path(chosen))

    def _open_new_window(self, directory: Path) -> None:
        """Launch a new browser window in a separate process."""
        import subprocess

        try:
            subprocess.Popen(
                [sys.executable, "-m", "logqbit.browser", str(directory)],
                cwd=Path.cwd(),
                start_new_session=True,  # Detach from parent process
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Launch Error",
                f"Failed to launch new window:\n{exc}",
            )

    def _open_table_context_menu(self, point) -> None:
        records = self._get_selected_records()
        menu = QMenu(self)
        rename_action = menu.addAction("Rename Title... (F2)")
        toggle_star_action = menu.addAction("Toggle ⭐Star (S)")
        toggle_star_action.setCheckable(True)
        toggle_trash_action = menu.addAction("Toggle 🗑️Trash (T)")
        toggle_trash_action.setCheckable(True)
        show_trash_action = menu.addAction("Show Trashed Items")
        show_trash_action.setCheckable(True)
        show_trash_action.setChecked(self._show_trash)
        send_to_recycle_action = menu.addAction("Send to Recycle Bin (Del)")
        menu.addSeparator()
        plot_axes_action = menu.addAction("Show Plot Axes Column")
        plot_axes_action.setCheckable(True)
        plot_axes_action.setChecked(not self.log_table.isColumnHidden(COL_PLOT_AXES))
        create_time_action = menu.addAction("Show Create Time Column")
        create_time_action.setCheckable(True)
        create_time_action.setChecked(
            not self.log_table.isColumnHidden(COL_CREATE_TIME)
        )
        create_machine_action = menu.addAction("Show Create Machine Column")
        create_machine_action.setCheckable(True)
        create_machine_action.setChecked(
            not self.log_table.isColumnHidden(COL_CREATE_MACHINE)
        )
        menu.addSeparator()
        open_explorer = menu.addAction("Open in Explorer")
        if not records:
            rename_action.setEnabled(False)
            toggle_star_action.setEnabled(False)
            toggle_trash_action.setEnabled(False)
            send_to_recycle_action.setEnabled(False)
            open_explorer.setEnabled(False)
        else:
            rename_action.setEnabled(len(records) == 1)
            all_starred = all(rec.meta.star > 0 for rec in records)
            all_trashed = all(rec.meta.trash for rec in records)
            toggle_star_action.setEnabled(True)
            toggle_star_action.setChecked(all_starred)
            toggle_trash_action.setEnabled(True)
            toggle_trash_action.setChecked(all_trashed)
            send_to_recycle_action.setEnabled(True)
        chosen = menu.exec(self.log_table.viewport().mapToGlobal(point))
        if chosen is None:
            return
        if chosen == rename_action and records and len(records) == 1:
            self._rename_record_title(records[0])
        elif chosen == toggle_star_action and records:
            self._set_records_star_count(
                records, 1 if toggle_star_action.isChecked() else 0
            )
        elif chosen == toggle_trash_action and records:
            self._set_records_trash(records, toggle_trash_action.isChecked())
        elif chosen == send_to_recycle_action and records:
            self._send_records_to_recycle_bin(records)
        elif chosen == show_trash_action:
            self._toggle_show_trash()
        elif chosen == create_time_action:
            self._toggle_column(COL_CREATE_TIME, create_time_action.isChecked())
        elif chosen == create_machine_action:
            self._toggle_column(COL_CREATE_MACHINE, create_machine_action.isChecked())
        elif chosen == plot_axes_action:
            self._toggle_column(COL_PLOT_AXES, plot_axes_action.isChecked())
        elif chosen == open_explorer and records:
            self._open_path_in_explorer(records[0].path, len(records) != 1)

    def _on_plot_axes_changed(
        self, record: LogRecord, column_name: str, enabled: bool
    ) -> None:
        """Callback when plot axes is toggled in DataViewManager."""
        self._update_ui_for_record(record)
        self._load_log(record)

    def _toggle_column(self, column: int, visible: bool) -> None:
        self.log_table.setColumnHidden(column, not visible)

    def _toggle_show_trash(self) -> None:
        self._show_trash = not self._show_trash
        self.refresh_logs()

    def _rename_record_title(self, record: LogRecord) -> None:
        current_title = record.meta.title
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Rename Log")
        dialog.setLabelText("Enter new title:")
        dialog.setTextValue(current_title)
        dialog.setInputMode(QInputDialog.TextInput)
        dialog.resize(max(dialog.sizeHint().width(), 600), dialog.sizeHint().height())
        if dialog.exec() != QDialog.Accepted:
            return
        new_title = dialog.textValue().strip()
        if new_title == current_title:
            return
        record.meta.title = new_title
        self._update_ui_for_record(record)
        self.refresh_logs()

    def _set_record_star_count(
        self, record: LogRecord, count: int, refresh: bool = True
    ) -> bool:
        new_value = max(int(count), 0)
        if record.meta.star == new_value:
            return False
        record.meta.star = new_value
        self._update_ui_for_record(record)
        if refresh:
            self.refresh_logs()
        return True

    def _set_record_trash(
        self, record: LogRecord, value: bool, refresh: bool = True
    ) -> bool:
        if record.meta.trash == value:
            return False
        record.meta.trash = value
        self._update_ui_for_record(record)
        if refresh:
            self.refresh_logs()
        return True

    def _set_records_star_count(self, records: Iterable[LogRecord], count: int) -> None:
        changed = False
        for record in records:
            changed |= self._set_record_star_count(record, count, refresh=False)
        if changed:
            self.refresh_logs()

    def _set_records_trash(self, records: Iterable[LogRecord], value: bool) -> None:
        changed = False
        for record in records:
            changed |= self._set_record_trash(record, value, refresh=False)
        if changed:
            self.refresh_logs()

    def _update_ui_for_record(self, record: LogRecord) -> None:
        self.table_model.update_record(record)
        if self._current_record and self._current_record.log_id == record.log_id:
            self._update_detail_watcher(record)

    def _open_path_in_explorer(self, path: Path, select: bool = False) -> None:
        try:
            if sys.platform.startswith("win"):
                if select:
                    subprocess.run(["explorer", "/select,", str(path)], check=False)
                else:
                    subprocess.run(["explorer", str(path)], check=False)
            elif sys.platform == "darwin":
                if select:
                    subprocess.run(["open", "-R", str(path)], check=False)
                else:
                    subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to open explorer for %s: %s", path, exc)
            QMessageBox.warning(
                self, "Open in Explorer", f"Failed to open file browser: {exc}"
            )

    def _send_records_to_recycle_bin(self, records: Iterable[LogRecord]) -> None:
        records_list = list(records)
        if not records_list:
            return

        # Prepare confirmation message
        if len(records_list) == 1:
            message = f"Send log folder #{records_list[0].log_id} to Recycle Bin?\n\n"
            message += f"Path: {records_list[0].path}\n\n"
            message += "This operation can be undone from the Recycle Bin."
        else:
            message = f"Send {len(records_list)} log folders to Recycle Bin?\n\n"
            message += "IDs: " + ", ".join(f"#{r.log_id}" for r in records_list[:10])
            if len(records_list) > 10:
                message += f", ... (+{len(records_list) - 10} more)"
            message += "\n\nThis operation can be undone from the Recycle Bin."

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Send to Recycle Bin",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        # Send to recycle bin
        failed_paths = []
        for record in records_list:
            try:
                send2trash(str(record.path))
            except Exception as exc:
                failed_paths.append(f"{record.path} ({exc})")

        # Show results
        if failed_paths:
            error_msg = "Failed to send some folders to Recycle Bin:\n\n"
            error_msg += "\n".join(failed_paths[:5])
            if len(failed_paths) > 5:
                error_msg += f"\n... and {len(failed_paths) - 5} more"
            QMessageBox.warning(self, "Error", error_msg)

        # Refresh list
        self.refresh_logs()

    def _shortcut_set_star(self, count: int) -> None:
        records = self._get_selected_records()
        if not records:
            return
        self._set_records_star_count(records, count)

    def _shortcut_toggle_star(self) -> None:
        records = self._get_selected_records()
        if not records:
            return
        all_starred = all(rec.meta.star > 0 for rec in records)
        self._set_records_star_count(records, 0 if all_starred else 1)

    def _shortcut_mark_trash(self) -> None:
        records = self._get_selected_records()
        if not records:
            return
        self._set_records_trash(records, True)

    def _shortcut_send_to_recycle_bin(self) -> None:
        records = self._get_selected_records()
        if not records:
            return
        self._send_records_to_recycle_bin(records)

    def _shortcut_toggle_trash(self) -> None:
        records = self._get_selected_records()
        if not records:
            return
        all_trashed = all(rec.meta.trash for rec in records)
        self._set_records_trash(records, not all_trashed)

    def _shortcut_rename_title(self) -> None:
        records = self._get_selected_records()
        if len(records) != 1:
            return
        self._rename_record_title(records[0])

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override naming
        self.settings_manager.save_recent_directories(
            self.settings_manager._recent_directories
        )
        self.settings_manager.save_theme_mode(self._theme_mode)
        super().closeEvent(event)


# ============================================================================
# Entry Point
# ============================================================================


def ensure_application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("Logqbit Log Browser")
    return app


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    directory = Path(args[0]).expanduser().resolve() if args else None
    app = ensure_application()
    window = LogBrowserWindow(directory)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual run
    sys.exit(main())
