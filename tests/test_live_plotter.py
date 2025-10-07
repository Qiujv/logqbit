from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Iterator

import pandas as pd
import pytest

from logqbit.live_plotter import LIVE_PLOTTER_PIPE_NAME, LivePlotterClient


def _wait_for_server(timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        client = LivePlotterClient(timeout_ms=250)
        try:
            client.connect()
        except ConnectionError as exc:  # pragma: no cover - polling loop
            last_error = exc
            time.sleep(0.1)
        else:
            client.close()
            return
        finally:
            client.close()

    raise TimeoutError("Timed out waiting for LivePlotter server to be ready") from last_error


@pytest.fixture
def live_plotter_process() -> Iterator[subprocess.Popen[str]]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    process = subprocess.Popen(['logqbit-live-plotter'])

    try:
        _wait_for_server()
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_client_connect_failure_fast() -> None:
    client = LivePlotterClient(socket_name=f"{LIVE_PLOTTER_PIPE_NAME}-missing", timeout_ms=100)
    with pytest.raises(ConnectionError):
        client.connect()


def test_live_plotter_accepts_commands(live_plotter_process: subprocess.Popen[str]) -> None:
    with LivePlotterClient(timeout_ms=1000) as client:
        client.set_indeps(["time", "cycle"])

        client.add(record={"time": 0.0, "cycle": 0, "signal_a": 0.0})

        segment = pd.DataFrame(
            {
                "time": [0.1, 0.2, 0.3, 0.4],
                "cycle": [0, 0, 0, 0],
                "signal_a": [0.1, 0.2, 0.3, 0.4],
                "signal_b": [1.0, 0.9, 0.8, 0.7],
            }
        )
        client.add(seg=segment)

        client.add(
            seg=[
                {"time": 1.0, "cycle": 1, "signal_a": 0.5, "signal_b": 0.1},
                {"time": 1.1, "cycle": 1, "signal_a": 0.6, "signal_b": 0.2},
            ]
        )