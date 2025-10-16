import inspect
import itertools
import os
import socket
import threading
import time
import weakref
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .registry import Registry, get_parser

yaml = get_parser()


class LogFolder:
    create_machine = socket.gethostname()  # Can be overridden.

    def __init__(
        self,
        path: str | Path,
        create: bool = True,
        save_delay_secs: float = 1.0,
    ):
        path = Path(path)
        meta_path = path / "meta.yaml"
        data_path = path / "data.parquet"
        if path.exists() and path.is_dir():
            pass
        elif create:
            path.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    {
                        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "create_machine": self.create_machine,
                    },
                    f,
                )
        else:
            raise FileNotFoundError(f"LogFolder at '{path}' does not exist.")

        self.path = path

        if meta_path.exists():
            self.reg = Registry(meta_path, auto_reload=False)
        else:
            self.reg = None

        self._handler = _DataHandler(data_path, save_delay_secs, self)

    @property
    def df(self) -> pd.DataFrame:
        """Get the full dataframe, flushing all data rows."""
        return self._handler.get_df()

    @classmethod
    def new(cls, parent_path: Path) -> "LogFolder":
        parent_path = Path(parent_path)
        max_index = max(
            (
                int(entry.name)
                for entry in os.scandir(parent_path)
                if entry.is_dir() and entry.name.isdecimal()
            ),
            default=-1,
        )
        new_index = max_index + 1
        while (parent_path / str(new_index)).exists():
            new_index += 1
        new_folder = parent_path / str(new_index)
        return cls(new_folder)

    def add_row(self, **kwargs) -> None:
        """
        Add a new row or multiple rows to the dataframe.
        Supports both scalar and vector input.
        For vector input, pandas will check length consistency.
        """
        is_multi_row = [
            k
            for k, v in kwargs.items()
            if hasattr(v, "__len__") and not isinstance(v, str)
        ]
        if is_multi_row:
            self._handler.add_multi_rows(pd.DataFrame(kwargs))
        else:
            self._handler.add_one_row(kwargs)

    def capture(
        self,
        func: Callable[[float], dict[str, float | list[float]]],
        axes: list[float | list[float]] | dict[str, float | list[float]],
    ):
        if not isinstance(axes, dict):  # Assumes isinstance(axes, list)
            fsig = inspect.signature(func)
            axes = dict(zip(fsig.parameters.keys(), axes))

        run_axs: dict[str, list[float]] = {}
        const_axs: dict[str, float] = {}
        for k, v in axes.items():
            if np.iterable(v):
                run_axs[k] = v
            else:
                const_axs[k] = v
        self.add_meta_to_head(
            const=const_axs,
            dims={k: [min(a), max(a), len(a)] for k, a in run_axs.items()},
        )

        step_table = list(itertools.product(*run_axs.values()))

        with logging_redirect_tqdm():
            for step in tqdm(step_table, ncols=80, desc=self.path.name):
                step_kws = dict(zip(run_axs.keys(), step))
                ret_kws = func(**step_kws, **const_axs)
                self.add_row(**step_kws, **ret_kws)

    def add_meta(self, meta: dict = None, /, **kwargs):
        if meta is None:
            meta = {}
        meta.update(kwargs)
        self.reg.root.update(meta)
        self.reg.save()

    def add_meta_to_head(self, meta: dict = None, /, **kwargs):
        if meta is None:
            meta = {}
        meta.update(kwargs)
        for i, (k, v) in enumerate(meta.items()):
            self.reg.root.insert(i, k, v)
        self.reg.save()

    @property
    def indeps(self) -> list[str]:
        """Running axes for plotting."""
        return self.reg["indeps"]  # Let KeyError raise if not exists.

    @indeps.setter
    def indeps(self, value: list[str]) -> None:
        if not isinstance(value, list):
            raise ValueError("indeps must be a list of strings.")
        if not all(isinstance(v, str) for v in value):
            raise ValueError("indeps must be a list of strings.")

        self.reg["indeps"] = value


class _DataHandler:
    def __init__(
        self,
        path: str | Path,
        save_delay_secs: float,
        parent: LogFolder,
    ):
        self.path = Path(path)
        self._segs: list[pd.DataFrame] = []
        if self.path.exists():
            self._segs.append(pd.read_parquet(self.path))
        self._records: list[dict[str, float | int | str]] = []

        self.save_delay_secs = save_delay_secs
        self._should_stop = False
        self._skip_debounce = threading.Event()
        self._dirty = threading.Event()
        self._save_done = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        weakref.finalize(parent, self._cleanup)

    def get_df(self, _clear: bool = False) -> pd.DataFrame:
        with self._lock:
            if self._records:
                self._segs.append(pd.DataFrame.from_records(self._records))
                self._records = []

            if len(self._segs) == 0:
                df = pd.DataFrame({})
            elif len(self._segs) == 1:
                df = self._segs[0]
            else:
                df = pd.concat(self._segs)
                self._segs = [df]

            if _clear:
                self._dirty.clear()
        return df

    def add_one_row(self, kwargs: dict[str, float | int | str]):
        with self._lock:
            self._records.append(kwargs)
            if not self._dirty.is_set():
                self._dirty.set()

    def add_multi_rows(self, df: pd.DataFrame):
        with self._lock:
            if self._records:
                self._segs.append(pd.DataFrame.from_records(self._records))
                self._records = []
            self._segs.append(df)
            if not self._dirty.is_set():
                self._dirty.set()

    def _run(self):
        while not self._should_stop:
            self._dirty.wait()
            if self._should_stop:
                break
            self._save_done = False
            if self._skip_debounce.wait(self.save_delay_secs):
                self._skip_debounce.clear()
            df = self.get_df(_clear=True)
            try:
                df.to_parquet(self.path, index=False)
            finally:
                self._save_done = True

    def _cleanup(self):
        try:
            self._should_stop = True
            self._skip_debounce.set()  # Process all pending data.
            self._dirty.set()  # Just break the run loop.
            if self._thread.is_alive():
                self._thread.join(timeout=2)
        except Exception:
            pass

    def flush(self):
        """Flash the pending data immediately, block until done."""
        self._skip_debounce.set()
        while not self._save_done:
            time.sleep(0.01)
