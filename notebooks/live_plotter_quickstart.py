from __future__ import annotations

import math
import time

import pandas as pd

from logqbit.live_plotter import LivePlotterClient


def main() -> None:
    """Send a short demo sequence to a running LivePlotter window."""
    print("Connecting to LivePlotter server…")

    with LivePlotterClient() as client:
        client.set_indeps(["time", "cycle"])

        print("Sending three demo segments.")
        for cycle in range(3):
            times = [cycle + i * 0.1 for i in range(20)]
            segment = pd.DataFrame(
                {
                    "time": times,
                    "cycle": cycle,
                    "signal_a": [math.sin(t) for t in times],
                    "signal_b": [math.cos(t) for t in times],
                }
            )
            client.add(seg=segment)
            time.sleep(0.1)

    print("Done. Adjust markers in the LivePlotter window to explore the traces.")


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as exc:  # pragma: no cover - user feedback
        print("Failed to connect to LivePlotter. Make sure the GUI is running.")
        print(exc)
