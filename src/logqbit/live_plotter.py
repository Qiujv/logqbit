from __future__ import annotations

import pickle
import sys
import warnings
from collections import deque
from importlib.resources import files
from typing import Any, Mapping, Sequence

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .constants import HEAD_LENGTH, LIVE_PLOTTER_PIPE_NAME

ACTIVE_COLOR = (255, 94, 0)
INACTIVE_COLOR = (30, 144, 255, 120)
DEFAULT_LINE_COUNT = 4


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LivePlotterWindow(QMainWindow):
    def __init__(
        self,
        line_count: int = DEFAULT_LINE_COUNT,
        socket_name: str | None = LIVE_PLOTTER_PIPE_NAME,
    ) -> None:
        super().__init__()
        pg.setConfigOptions(antialias=True)
        self.setWindowTitle("LogQbit Live Plotter")
        self.setWindowIcon(QIcon(QPixmap(str(files("logqbit") / "assets" / "icon.svg"))))

        self.line_count = max(1, line_count)
        self._active_index = 0
        self._inactive_indices: deque[int] = deque(range(1, self.line_count))
        self._last_stepper_values: tuple[Any, ...] | None = None
        self._indeps: list[str] = []
        self._stepper_keys: list[str] = []
        self._dependent_keys: set[str] = set()
        self._current_y_key: str | None = None
        self._line_storage: list[list[dict[str, Any]]] = [
            [] for _ in range(self.line_count)
        ]
        self._show_markers = False
        self._marker_size = 4

        self._active_symbol_brush = pg.mkBrush(ACTIVE_COLOR)
        self._inactive_symbol_brush = pg.mkBrush(INACTIVE_COLOR)
        self._active_symbol_pen = pg.mkPen(ACTIVE_COLOR)
        self._inactive_symbol_pen = pg.mkPen(INACTIVE_COLOR)
        self._active_pen = pg.mkPen(color=ACTIVE_COLOR, width=2)
        self._inactive_pen = pg.mkPen(color=INACTIVE_COLOR, width=1)

        self._build_ui()
        self._configure_plot()

        self._command_server: PlotterCommandServer | None = None
        if socket_name:
            self._command_server = PlotterCommandServer(self, socket_name)

    # ------------------------------------------------------------------
    # Public API exposed to IPC clients
    # ------------------------------------------------------------------
    def set_indeps(self, indeps: Sequence[str]) -> None:
        indep_list = [str(item) for item in indeps if str(item)]
        if not indep_list:
            raise ValueError("'indeps' must contain at least one non-empty key")

        self._indeps = indep_list
        self._stepper_keys = indep_list[1:]
        self._dependent_keys.clear()
        self._current_y_key = None
        self._last_stepper_values = None

        self.plot_widget.setLabel("bottom", indep_list[0])

        for storage in self._line_storage:
            storage.clear()
        self._active_index = 0
        self._inactive_indices = deque(range(1, self.line_count))
        for idx in range(self.line_count):
            if idx == self._active_index:
                self._set_active(idx)
            else:
                self._set_inactive(idx)

        self._sync_y_selector()
        self._set_status_message("")
        self._refresh_all_lines()

    def add(
        self,
        record: Mapping[str, Any] | pd.Series | None = None,
        seg: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        rows: list[Mapping[str, Any]] = []

        if record is not None:
            if isinstance(record, pd.Series):
                rows.append(record.to_dict())
            elif isinstance(record, Mapping):
                rows.append(dict(record))
            else:
                raise TypeError("'record' must be a mapping or pandas Series")

        if seg is not None:
            if isinstance(seg, pd.DataFrame):
                seg_df = seg
            else:
                try:
                    seg_df = pd.DataFrame(seg)
                except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                    raise ValueError("'seg' must be convertible to a pandas DataFrame") from exc
            rows.extend(seg_df.to_dict(orient="records"))

        if not rows:
            return

        for row in rows:
            self._ingest_row(dict(row))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.plot_widget = pg.PlotWidget(background="w", parent=self)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        root.addWidget(self.plot_widget)

        controls = QHBoxLayout()
        controls.setContentsMargins(8, 4, 8, 4)
        controls.setSpacing(8)

        self.y_selector = QComboBox(self)
        self.y_selector.currentTextChanged.connect(self._on_y_changed)
        self.y_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        controls.addWidget(self.y_selector)

        self._status_bar = QStatusBar(self)
        self._status_bar.setSizeGripEnabled(False)
        self._status_bar.setContentsMargins(0, 0, 0, 0)
        self._status_bar.setStyleSheet("QStatusBar::item { border: none; }")
        self._status_label = QLabel("", self._status_bar)
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_bar.addWidget(self._status_label, 1)
        controls.addWidget(self._status_bar, 1)

        self.marker_button = QPushButton("Markers Off", self)
        self.marker_button.setCheckable(True)
        self.marker_button.setChecked(self._show_markers)
        self.marker_button.toggled.connect(self._on_marker_toggled)
        controls.addWidget(self.marker_button)

        root.addLayout(controls)

        central.setLayout(root)
        self.setCentralWidget(central)

        self._line_items: list[pg.PlotDataItem] = []
        for idx in range(self.line_count):
            pen = self._active_pen if idx == self._active_index else self._inactive_pen
            item = self.plot_widget.plot([], [], pen=pen)
            item.setZValue(10 if idx == self._active_index else 1)
            self._line_items.append(item)

    def _configure_plot(self) -> None:
        self.plot_widget.setLabel("left", "")
        self.plot_widget.setLabel("bottom", "")
        self.plot_widget.addLegend()

    def _on_marker_toggled(self, checked: bool) -> None:
        self._show_markers = checked
        self.marker_button.setText("Markers On" if checked else "Markers Off")
        self._refresh_all_lines()

    def _sync_y_selector(self) -> tuple[bool, str | None]:
        previous = self._current_y_key
        keys = sorted(self._dependent_keys)

        self.y_selector.blockSignals(True)
        self.y_selector.clear()
        for key in keys:
            self.y_selector.addItem(key)

        changed = False
        if not keys:
            self._current_y_key = None
            changed = previous is not None
        else:
            if previous in keys:
                index = keys.index(previous)
                self.y_selector.setCurrentIndex(index)
                self._current_y_key = previous
            else:
                self.y_selector.setCurrentIndex(0)
                self._current_y_key = keys[0]
                changed = previous != self._current_y_key
        self.y_selector.blockSignals(False)

        if changed:
            self._on_y_changed(self._current_y_key or "")
        return changed, self._current_y_key

    def _on_y_changed(self, text: str) -> None:
        if not text:
            self._current_y_key = None
            self.plot_widget.setLabel("left", "")
            for item in self._line_items:
                item.setData([], [])
            return

        self._current_y_key = text
        self.plot_widget.setLabel("left", text)
        self._refresh_all_lines()

    def _refresh_all_lines(self) -> None:
        for idx in range(self.line_count):
            self._refresh_line(idx)

    def _refresh_line(self, index: int) -> None:
        item = self._line_items[index]
        if not self._current_y_key:
            item.setData([], [])
            item.setSymbol(None)
            return

        storage = self._line_storage[index]
        if not storage:
            item.setData([], [])
            item.setSymbol(None)
            return

        x_values: list[float] = []
        y_values: list[float] = []
        for point in storage:
            y_raw = point["values"].get(self._current_y_key)
            y_value = _safe_float(y_raw)
            if y_value is None:
                continue
            x_value = _safe_float(point["x"])
            if x_value is None:
                continue
            x_values.append(x_value)
            y_values.append(y_value)

        item.setData(x_values, y_values)
        if self._show_markers and x_values:
            active = index == self._active_index
            item.setSymbol("o")
            item.setSymbolSize(self._marker_size)
            item.setSymbolBrush(self._active_symbol_brush if active else self._inactive_symbol_brush)
            item.setSymbolPen(self._active_symbol_pen if active else self._inactive_symbol_pen)
        else:
            item.setSymbol(None)

    def _roll_lines(self) -> None:
        if self.line_count == 1:
            self._line_storage[0].clear()
            self._refresh_line(0)
            return

        current = self._active_index
        self._set_inactive(current)
        self._inactive_indices.appendleft(current)

        new_active = self._inactive_indices.pop()
        self._active_index = new_active
        self._line_storage[new_active].clear()
        self._set_active(new_active)

        self._refresh_line(current)
        self._refresh_line(new_active)

    def _set_active(self, index: int) -> None:
        item = self._line_items[index]
        item.setPen(self._active_pen)
        item.setZValue(10)

    def _set_inactive(self, index: int) -> None:
        item = self._line_items[index]
        item.setPen(self._inactive_pen)
        item.setZValue(1)

    def _update_stepper_display(self, stepper: tuple[Any, ...]) -> None:
        if not self._stepper_keys:
            self._set_status_message("")
            return
        parts = [f"{key}={value}" for key, value in zip(self._stepper_keys, stepper)]
        self._set_status_message(" | ".join(parts))

    def _set_status_message(self, message: str) -> None:
        if hasattr(self, "_status_label"):
            self._status_label.setText(message)

    def _ingest_row(self, row: Mapping[str, Any]) -> None:
        if not self._indeps:
            return

        x_key = self._indeps[0]
        if x_key not in row:
            return

        x_value = _safe_float(row.get(x_key))
        if x_value is None:
            return

        stepper = tuple(row.get(key) for key in self._stepper_keys)
        dependent_values = {
            key: value
            for key, value in row.items()
            if key not in self._indeps
        }
        if not dependent_values:
            return

        if self._last_stepper_values is None:
            self._last_stepper_values = stepper
        elif stepper != self._last_stepper_values:
            self._roll_lines()
            self._last_stepper_values = stepper

        line_idx = self._active_index
        self._line_storage[line_idx].append({
            "x": x_value,
            "values": dict(dependent_values),
        })

        self._dependent_keys.update(dependent_values.keys())
        selection_changed, _ = self._sync_y_selector()
        if not selection_changed:
            self._refresh_line(line_idx)

        self._update_stepper_display(stepper)


class PlotterCommandServer(QObject):
    def __init__(self, window: LivePlotterWindow, socket_name: str) -> None:
        super().__init__(window)
        self._window = window
        self._socket_name = socket_name
        self._server = QLocalServer(self)
        self._connections: set[PlotterConnection] = set()

        QLocalServer.removeServer(self._socket_name)

        if not self._server.listen(self._socket_name):
            warnings.warn(
                f"LivePlotter IPC listen failed on '{self._socket_name}': {self._server.errorString()}",
                stacklevel=2,
            )
            return

        self._server.newConnection.connect(self._on_new_connection)

    def _on_new_connection(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            connection = PlotterConnection(socket, self._window, self)
            connection.finished.connect(self._on_connection_finished)
            self._connections.add(connection)

    def _on_connection_finished(self, connection: "PlotterConnection") -> None:
        self._connections.discard(connection)
        connection.deleteLater()


class PlotterConnection(QObject):
    finished = Signal(object)

    def __init__(
        self,
        socket: QLocalSocket,
        window: LivePlotterWindow,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._socket = socket
        self._window = window
        self._buffer = bytearray()
        self._expected_size: int | None = None

        self._socket.readyRead.connect(self._on_ready_read)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.errorOccurred.connect(self._on_error)

    def _on_ready_read(self) -> None:
        while self._socket.bytesAvailable():
            data = self._socket.readAll()
            if not data:
                break
            self._buffer.extend(bytes(data))
            self._process_buffer()

    def _process_buffer(self) -> None:
        while True:
            if self._expected_size is None:
                if len(self._buffer) < HEAD_LENGTH:
                    return
                header = bytes(self._buffer[:HEAD_LENGTH])
                self._expected_size = int.from_bytes(header, byteorder="big", signed=False)
                del self._buffer[:HEAD_LENGTH]

            if len(self._buffer) < (self._expected_size or 0):
                return

            payload = bytes(self._buffer[: self._expected_size]) if self._expected_size else b""
            del self._buffer[: self._expected_size or 0]
            self._expected_size = None
            if payload:
                self._process_payload(payload)

    def _process_payload(self, payload: bytes) -> None:
        try:
            message = pickle.loads(payload)
        except Exception as exc:  # pragma: no cover - defensive
            self._send_error("invalid_payload", f"Failed to unpickle payload: {exc}")
            return

        if not isinstance(message, dict):
            self._send_error("invalid_message", "Payload must be a dict")
            return

        command = message.get("cmd")
        if command == "set_indeps":
            self._handle_set_indeps(message)
        elif command == "add":
            self._handle_add(message)
        else:
            self._send_error("unknown_command", f"Command '{command}' is not supported")

    def _handle_set_indeps(self, payload: dict[str, Any]) -> None:
        indeps = payload.get("indeps")
        if not isinstance(indeps, Sequence):
            self._send_error("invalid_arguments", "'indeps' must be a sequence of strings")
            return
        try:
            self._window.set_indeps(list(indeps))
        except Exception as exc:  # pragma: no cover - GUI validation
            self._send_error("execution_error", f"{type(exc).__name__}: {exc}")
            return
        self._send_packet({"status": "ok", "cmd": "set_indeps"})

    def _handle_add(self, payload: dict[str, Any]) -> None:
        record = payload.get("record")
        if record is not None and not isinstance(record, (Mapping, pd.Series)):
            self._send_error("invalid_arguments", "'record' must be a mapping or pandas Series")
            return

        seg_payload = payload.get("seg")
        seg_df = None
        if seg_payload is not None:
            if isinstance(seg_payload, pd.DataFrame):
                seg_df = seg_payload
            else:
                try:
                    seg_df = pd.DataFrame(seg_payload)
                except (TypeError, ValueError) as exc:
                    self._send_error("invalid_arguments", f"'seg' conversion failed: {exc}")
                    return

        try:
            self._window.add(record=record, seg=seg_df)
        except Exception as exc:  # pragma: no cover - GUI validation
            self._send_error("execution_error", f"{type(exc).__name__}: {exc}")
            return

        self._send_packet({"status": "ok", "cmd": "add"})

    def _send_packet(self, payload: dict[str, Any]) -> None:
        try:
            body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # pragma: no cover - defensive
            body = pickle.dumps(
                {"status": "error", "code": "encoding_error", "message": str(exc)},
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        header = len(body).to_bytes(HEAD_LENGTH, byteorder="big", signed=False)
        self._socket.write(header)
        if body:
            self._socket.write(body)
        self._socket.flush()

    def _send_error(self, code: str, message: str) -> None:
        self._send_packet({"status": "error", "code": code, "message": message})

    def _on_disconnected(self) -> None:
        self.finished.emit(self)
        self._socket.deleteLater()
        self.deleteLater()

    def _on_error(self, _error: QLocalSocket.LocalSocketError) -> None:  # pragma: no cover - best effort
        self.finished.emit(self)
        self._socket.deleteLater()
        self.deleteLater()


class LivePlotterClient:
    def __init__(
        self,
        socket_name: str = LIVE_PLOTTER_PIPE_NAME,
        *,
        timeout_ms: int = 5000,
    ) -> None:
        self._socket_name = socket_name
        self._timeout_ms = timeout_ms
        self._socket: QLocalSocket | None = None
        self._owns_app = False
        self._app: QCoreApplication | None = None

    def connect(self) -> None:
        self._ensure_app()
        if self._socket is not None and self._socket.state() == QLocalSocket.ConnectedState:
            return

        socket = QLocalSocket()
        socket.connectToServer(self._socket_name)
        if not socket.waitForConnected(self._timeout_ms):
            error = socket.errorString()
            socket.deleteLater()
            raise ConnectionError(f"Could not connect to LivePlotter server '{self._socket_name}': {error}")

        self._socket = socket

    def close(self) -> None:
        if self._socket is None:
            return
        self._socket.disconnectFromServer()
        self._socket.waitForDisconnected(100)
        self._socket.deleteLater()
        self._socket = None

    # Context manager helpers -------------------------------------------------
    def __enter__(self) -> "LivePlotterClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # IPC commands ------------------------------------------------------------
    def set_indeps(self, indeps: Sequence[str]) -> None:
        payload = {"cmd": "set_indeps", "indeps": list(indeps)}
        self._invoke(payload)

    def add(
        self,
        *,
        record: Mapping[str, Any] | pd.Series | None = None,
        seg: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        seg_payload: Any = None
        if seg is not None and not isinstance(seg, pd.DataFrame):
            try:
                seg_payload = pd.DataFrame(seg)
            except (TypeError, ValueError) as exc:
                raise ValueError("'seg' must be convertible to a pandas DataFrame") from exc
        else:
            seg_payload = seg

        if record is not None and not isinstance(record, (Mapping, pd.Series)):
            raise TypeError("'record' must be a mapping or pandas Series")

        payload = {"cmd": "add", "record": record, "seg": seg_payload}
        self._invoke(payload)

    # Internal client helpers -------------------------------------------------
    def _ensure_app(self) -> None:
        app = QCoreApplication.instance()
        if app is None:
            self._app = QCoreApplication([])
            self._owns_app = True
        else:
            self._app = app

    def _invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("LivePlotterClient is not connected")

        body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        header = len(body).to_bytes(HEAD_LENGTH, byteorder="big", signed=False)
        self._socket.write(header)
        if body:
            self._socket.write(body)
        if not self._socket.waitForBytesWritten(self._timeout_ms):
            raise TimeoutError("Timed out sending data to LivePlotter server")

        response = self._read_packet()
        status = response.get("status")
        if status != "ok":
            code = response.get("code", "unknown_error")
            message = response.get("message", "Request failed")
            raise RuntimeError(f"LivePlotter error ({code}): {message}")
        return response

    def _read_packet(self) -> dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("LivePlotterClient is not connected")

        header = self._read_exact(HEAD_LENGTH)
        size = int.from_bytes(header, byteorder="big", signed=False)
        body = self._read_exact(size) if size else b""
        if not body:
            return {}
        return pickle.loads(body)

    def _read_exact(self, size: int) -> bytes:
        if self._socket is None:
            raise RuntimeError("LivePlotterClient is not connected")

        data = bytearray()
        while len(data) < size:
            chunk = self._socket.read(size - len(data))
            if chunk:
                data.extend(bytes(chunk))
                continue
            if not self._socket.waitForReadyRead(self._timeout_ms):
                raise TimeoutError("Timed out waiting for LivePlotter response")
            if self._socket.state() != QLocalSocket.ConnectedState and not self._socket.bytesAvailable():
                raise ConnectionError("LivePlotter server disconnected")
        return bytes(data)


def main() -> None:
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True

    window = LivePlotterWindow()
    window.show()

    if owns_app:
        sys.exit(app.exec())


__all__ = ["LivePlotterWindow", "LivePlotterClient", "main"]
