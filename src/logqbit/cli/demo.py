"""Implementation of the ``logqbit browser-demo`` command."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from logqbit.logfolder import LogFolder


def create_example_data() -> int:
    """Append example logs under ``./logqbit_example`` and open the browser."""
    try:
        example_dir = Path.cwd() / "logqbit_example"
        example_dir.mkdir(parents=True, exist_ok=True)
        print(f"Appending example data in: {example_dir}")
        example_count = _create_examples(example_dir)

        print(f"\n✓ Created {example_count} example log folders in: {example_dir}")
        print("\nLaunching browser...")
        from logqbit.gui.browser.startup import launch_browser

        launch_browser(example_dir)
        return 0
    except ImportError as exc:
        print(f"Error: Missing required dependency: {exc}", file=sys.stderr)
        print("Please ensure logqbit is properly installed.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error creating example data: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


def _create_examples(example_dir: Path) -> int:
    """Create one batch of example records below an existing parent directory."""
    print("  Creating example: Linear relationship...")
    with LogFolder.new(example_dir, title="Linear Relationship Example") as log:
        log.meta.update(
            star=1,
            plot_axes=["x"],
        )
        x_values = np.linspace(0, 10, 50)
        log.add_df(
            pd.DataFrame({"x": x_values, "y": 2 * x_values + 1, "z": x_values**2})
        )
        log.add_const(
            description="y = 2x + 1, z = x^2",
            experiment_type="simulation",
        )

    print("  Creating example: Sinusoidal with noise...")
    with LogFolder.new(example_dir, title="Noisy Sinusoidal Signal") as log:
        log.meta.update(
            star=2,
            plot_axes=["time"],
        )
        time_values = np.linspace(0, 4 * np.pi, 100)
        signal = np.sin(time_values)
        noisy = signal + np.random.default_rng().normal(0, 0.1, len(signal))
        log.add_df(
            pd.DataFrame({"time": time_values, "signal": signal, "noisy": noisy})
        )
        log.add_const(
            description="sin(t) with Gaussian noise",
            frequency="1 Hz",
            noise_level=0.1,
        )

    print("  Creating example: 2D parameter scan...")
    with LogFolder.new(example_dir, title="2D Parameter Scan") as log:
        log.meta.update(
            star=3,
            plot_axes=["voltage", "frequency"],
        )
        voltage, frequency = np.meshgrid(
            np.linspace(-1, 1, 20),
            np.linspace(1, 10, 20),
            indexing="ij",
        )
        response = np.exp(-((frequency - 5.5) ** 2) / 2) * np.exp(
            -((voltage - 0.2) ** 2) / 0.5
        )
        response += np.random.default_rng().normal(0, 0.05, response.shape)
        log.add_df(
            pd.DataFrame(
                {
                    "voltage": voltage.ravel(),
                    "frequency": frequency.ravel(),
                    "response": response.ravel(),
                }
            )
        )
        log.add_const(
            description="Simulated resonance scan",
            voltage_unit="V",
            frequency_unit="GHz",
        )

    print("  Creating example: 1M-point 1D signal (large scale)...")
    with LogFolder.new(example_dir, title="1M-Point 1D Signal") as log:
        log.meta.update(plot_axes=["t"])
        point_count = 1_000_000
        time_values = np.linspace(0, 100, point_count)
        signal = np.sin(2 * np.pi * 0.5 * time_values) * np.exp(-time_values / 50)
        signal += np.random.default_rng(42).normal(0, 0.05, point_count)
        log.add_df(pd.DataFrame({"t": time_values, "signal": signal}))
        log.add_const(
            description="Damped sine, 1 million points",
            points=point_count,
        )

    print("  Creating example: 1M-point 2D scan (large scale)...")
    with LogFolder.new(example_dir, title="1M-Point 2D Scan") as log:
        log.meta.update(plot_axes=["x", "y"])
        x_count = y_count = 1000
        x_values = np.repeat(np.linspace(0, 1, x_count), y_count)
        y_values = np.tile(np.linspace(0, 1, y_count), x_count)
        z_values = np.sin(4 * np.pi * x_values) * np.cos(4 * np.pi * y_values)
        z_values += np.random.default_rng(7).normal(0, 0.05, len(z_values))
        log.add_df(pd.DataFrame({"x": x_values, "y": y_values, "z": z_values}))
        log.add_const(
            description="sin(4πx)cos(4πy) + noise, 1000×1000 grid",
            points=x_count * y_count,
        )

    print("  Creating example: Grouped 1D signals...")
    _create_grouped_1d_example(example_dir)

    print("  Creating example: Grouped 2D scans...")
    _create_grouped_2d_example(example_dir)

    return 7


def _create_grouped_1d_example(example_dir: Path) -> None:
    with LogFolder.new(example_dir, title="Grouped 1D Signals") as log:
        log.meta.update(
            plot_axes=["time"],
            plot_fields=["signal"],
            plot_groupby=["device", "mode"],
        )
        grouped_frames = []
        time_values = np.linspace(0, 6, 121)
        for device_index, device in enumerate(("A", "B")):
            for mode_index, mode in enumerate(("idle", "driven")):
                frequency = 0.5 + 0.15 * device_index + 0.25 * mode_index
                signal = (1 + 0.3 * device_index) * np.sin(
                    2 * np.pi * frequency * time_values
                )
                signal += 0.4 * mode_index + 0.2 * device_index
                grouped_frames.append(
                    pd.DataFrame(
                        {
                            "device": device,
                            "mode": mode,
                            "time": time_values,
                            "signal": signal,
                        }
                    )
                )
        log.add_df(pd.concat(grouped_frames, ignore_index=True))
        log.add_const(description="Grouped traces with two grouping columns")


def _create_grouped_2d_example(example_dir: Path) -> None:
    with LogFolder.new(example_dir, title="Grouped 2D Scans") as log:
        log.meta.update(
            plot_axes=["x", "y"],
            plot_fields=["response"],
            plot_groupby=["region"],
        )
        grouped_frames = []
        for region, x_range, size, center in (
            ("left", (-2.0, -0.2), 16, (-1.1, 0.2)),
            ("right", (0.2, 2.0), 22, (1.1, -0.2)),
        ):
            x_grid, y_grid = np.meshgrid(
                np.linspace(*x_range, size),
                np.linspace(-1, 1, size),
                indexing="ij",
            )
            response = np.exp(
                -((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2) / 0.3
            )
            grouped_frames.append(
                pd.DataFrame(
                    {
                        "region": region,
                        "x": x_grid.ravel(),
                        "y": y_grid.ravel(),
                        "response": response.ravel(),
                    }
                )
            )
        log.add_df(pd.concat(grouped_frames, ignore_index=True))
        log.add_const(description="Two grouped meshes with different point counts")
