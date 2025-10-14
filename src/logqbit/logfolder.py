import inspect
import itertools
import os
import socket
import threading
import time
import warnings
import weakref
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .registry import get_parser

yaml = get_parser()


class LogFolder:
    META_FILENAME = "meta.yaml"
    DATA_FILENAME = "data.parquet"
    CREATE_MACHINE: str = socket.gethostname()
    SAVE_INTERVAL: float = 1.0  # seconds

    def __init__(self, path: str | Path, create: bool = True):
        self.path = Path(path)
        self.meta = OrderedDict()
        self._segs: list[pd.DataFrame] = []
        self._records: list[dict] = []
        self._last_add_row_time = datetime.now().timestamp()
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._lock = threading.Lock()

        _thread = threading.Thread(target=self._writer, daemon=True)
        _thread.start()
        weakref.finalize(self, self._cleanup, self._stop, _thread)

        if self.path.exists() and self.path.is_dir():
            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = yaml.load(f)
            if self.data_path.exists():
                self._segs = [pd.read_parquet(self.data_path)]
        elif create:
            self.path.mkdir(parents=True, exist_ok=True)
            self.meta = self.get_init_meta()
        else:
            raise FileNotFoundError(f"LogFolder path '{self.path}' does not exist.")

    @staticmethod
    def _cleanup(stop_event: threading.Event, thread: threading.Thread):
        try:
            stop_event.set()
            if thread.is_alive():
                thread.join(timeout=2)
        except Exception:
            pass

    @property
    def meta_path(self) -> Path:
        return self.path / self.META_FILENAME

    @property
    def data_path(self) -> Path:
        return self.path / self.DATA_FILENAME

    @classmethod
    def get_init_meta(cls):
        return OrderedDict(
            {
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "create_machine": cls.CREATE_MACHINE,
            }
        )

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

    @property
    def df(self) -> pd.DataFrame:
        """Get the full dataframe, flushing all data rows."""
        self._flush_rec_to_segs()
        if len(self._segs) == 0:
            warnings.warn("No data in DataLogger.", stacklevel=2)
            return pd.DataFrame({})
        elif len(self._segs) == 1:
            df = self._segs[0]
        else:
            df = pd.concat(self._segs)
            self._segs = [df]
        return df

    def _flush_rec_to_segs(self) -> None:
        if self._records:
            self._segs.append(pd.DataFrame.from_records(self._records))
            self._records = []

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
            self._flush_rec_to_segs()
            self._segs.append(pd.DataFrame(kwargs))
        else:
            self._records.append(kwargs)

        self._dirty.set()
        self._notify_plotter()

    def _notify_plotter(self) -> None:
        return None  # TODO: implement live plotter

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
        self.meta.update(meta)

    def add_meta_to_head(self, meta: dict = None, /, **kwargs):
        if meta is None:
            meta = {}
        meta.update(kwargs)
        self.meta.update(meta)
        for k in reversed(meta.keys()):
            self.meta.move_to_end(k, last=False)

    def save(self, force: bool = False) -> None:
        if not self._dirty.is_set() and not force:
            return

        with self._lock:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                yaml.dump(self.meta, f)

            self.df.to_parquet(self.data_path, index=False)

    def _writer(self):
        while not self._stop.is_set():
            self._dirty.wait()
            time.sleep(self.SAVE_INTERVAL)  # debounce interval
            self._dirty.clear()
            self.save()

    @property
    def indeps(self) -> list[str]:
        """Running axes for plotting."""
        return self.meta["indeps"]  # Let KeyError raise if not exists.

    @indeps.setter
    def indeps(self, value: list[str]) -> None:
        if not isinstance(value, list):
            raise ValueError("indeps must be a list of strings.")
        if not all(isinstance(v, str) for v in value):
            raise ValueError("indeps must be a list of strings.")

        self.meta["indeps"] = value
