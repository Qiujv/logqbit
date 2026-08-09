"""Implementation of the ``logqbit browser-demo`` command."""

from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from logqbit.logfolder import LogFolder


def create_example_data() -> int:
    """Create five example logs in ``./logqbit_example`` and open the browser."""
    try:
        example_dir = Path.cwd() / "logqbit_example"
        if example_dir.exists():
            response = input(
                f"Directory '{example_dir}' already exists. Overwrite? (y/N): "
            )
            if response.lower() != "y":
                print("Cancelled.")
                return 0
            shutil.rmtree(example_dir)

        example_dir.mkdir(parents=True)
        print(f"Creating example data in: {example_dir}")

        print("  Creating example 0: Linear relationship...")
        with LogFolder(example_dir / "0") as log:
            log.meta.update(
                title="Linear Relationship Example",
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

        print("  Creating example 1: Sinusoidal with noise...")
        with LogFolder(example_dir / "1") as log:
            log.meta.update(
                title="Noisy Sinusoidal Signal",
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

        print("  Creating example 2: 2D parameter scan...")
        with LogFolder(example_dir / "2") as log:
            log.meta.update(
                title="2D Parameter Scan",
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

        print("  Creating example 3: 1M-point 1D signal (large scale)...")
        with LogFolder(example_dir / "3") as log:
            log.meta.update(title="1M-Point 1D Signal", plot_axes=["t"])
            point_count = 1_000_000
            time_values = np.linspace(0, 100, point_count)
            signal = np.sin(2 * np.pi * 0.5 * time_values) * np.exp(-time_values / 50)
            signal += np.random.default_rng(42).normal(0, 0.05, point_count)
            log.add_df(pd.DataFrame({"t": time_values, "signal": signal}))
            log.add_const(
                description="Damped sine, 1 million points",
                points=point_count,
            )

        print("  Creating example 4: 1M-point 2D scan (large scale)...")
        with LogFolder(example_dir / "4") as log:
            log.meta.update(title="1M-Point 2D Scan", plot_axes=["x", "y"])
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

        print(f"\n✓ Created 5 example log folders in: {example_dir}")
        print("\nLaunching browser...")
        from logqbit.gui.browser.startup.bootstrap import main as browser_main

        return browser_main([str(example_dir)])
    except ImportError as exc:
        print(f"Error: Missing required dependency: {exc}", file=sys.stderr)
        print("Please ensure logqbit is properly installed.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error creating example data: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
