import inspect
import itertools
import os
import socket
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import dpath
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

try:
    import json_tricks as json
except ImportError:
    import json


class LogFolder:
    META_FILENAME = "meta.json"
    DATA_FILENAME = "data.parquet"
    CREATE_MACHINE: str = socket.gethostname()
    SAVE_INTERVAL: float = 1.0  # seconds

    def __init__(self, path: Path, meta: dict | None = None) -> None:
        if meta is None:
            meta = self.get_init_meta()
        self.path = Path(path)
        self.meta = meta
        self._records: list[dict] = []
        self._segs: list[pd.DataFrame] = []
        self._io_worker = _IOWorker(self.data_path)
        self._last_add_row_time = datetime.now().timestamp()

    @classmethod
    def get_init_meta(cls) -> dict:
        return {
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "create_machine": cls.CREATE_MACHINE,
        }

    @classmethod
    def load(cls, path: Path) -> "LogFolder":
        path = Path(path)
        if not (path.exists() and path.is_dir()):
            raise FileNotFoundError(f"{path} is not a valid directory.")
        meta_path = path / cls.META_FILENAME
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        return cls(path, meta)

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
        new_folder.mkdir()
        return cls(new_folder)

    @property
    def df(self) -> pd.DataFrame:
        """Get the full dataframe, flushing all data rows."""
        self._flush_rec_to_segs()
        if len(self._segs) == 0:
            warnings.warn("No data in DataLogger.")
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

        self._notify_plotter()

        now = datetime.now().timestamp()
        if now - self._last_add_row_time > self.SAVE_INTERVAL:
            self._io_worker.save(self.df)
            self._last_add_row_time = now

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
        self.meta = dpath.merge(self.meta, meta)

    def add_meta_to_head(self, meta: dict = None, /, **kwargs):
        if meta is None:
            meta = {}
        meta.update(kwargs)
        self.meta = dpath.merge(meta, self.meta)

    def save(self) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

        self.df.to_parquet(self.data_path, index=False)

    def __del__(self):
        try:
            self.save()
        except Exception as e:
            warnings.warn(f"LogFile {self.path} save failed:\n{e}")

    @property
    def indeps(self) -> list[str]:
        """Running axes for plotting."""
        return self.meta["indeps"]  # Let KeyError raise if not exists.

    @indeps.setter
    def indeps(self, value: list[str]) -> None:
        self.meta["indeps"] = value

    @property
    def meta_path(self) -> Path:
        return self.path / self.META_FILENAME

    @property
    def data_path(self) -> Path:
        return self.path / self.DATA_FILENAME


class _IOWorker:
    def __init__(
        self,
        data_path: Path,
    ) -> None:
        self.data_path = data_path
        self.df = pd.DataFrame()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def save(self, df: pd.DataFrame) -> None:
        with self._lock:
            self.df = df
            self._event.set()

    def _run(self):
        while True:
            self._event.wait()
            self._event.clear()
            try:
                self.df.to_parquet(self.data_path, index=False)
            except Exception as e:
                print(f"IOWorker save error: {e}")
            time.sleep(0.01)
