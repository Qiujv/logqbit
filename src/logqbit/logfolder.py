import atexit
import inspect
import itertools
import os
import weakref
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing_extensions import deprecated

from .dataframe import DataFrameBuffer
from .metadata import LogMetadata
from .registry import Registry, get_parser

yaml = get_parser()


_ACTIVE_LOGFOLDERS: weakref.WeakSet["LogFolder"] = weakref.WeakSet()


def _close_active_logfolders() -> None:
    for logfolder in list(_ACTIVE_LOGFOLDERS):
        LogFolder._close_quietly(logfolder)


atexit.register(_close_active_logfolders)


class LogFolder:
    """Directory-backed experiment log with data, metadata, and constants.

    A ``LogFolder`` manages three files under one directory:

    - ``data.feather`` for tabular records
    - ``metadata.json`` for lightweight metadata
    - ``const.yaml`` for constant parameters and configuration
    """

    def __init__(
        self,
        path: str | Path,
        title: str = "untitled",
        create: bool = True,
    ):
        path = Path(path)
        if path.exists() and path.is_dir():
            pass
        elif create:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"LogFolder at '{path}' does not exist.")

        self.path = path
        # File created anyway.
        self.meta = LogMetadata(path / "metadata.json", title, create=True)
        # File create on setting values.
        self._handler = DataFrameBuffer(path / "data.feather")
        self._finalizer = weakref.finalize(self, self._close_handler_quietly, self._handler)
        _ACTIVE_LOGFOLDERS.add(self)

    def __enter__(self) -> "LogFolder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _close_handler_quietly(handler: DataFrameBuffer) -> None:
        try:
            handler.close()
        except Exception:
            pass

    @staticmethod
    def _close_quietly(logfolder: "LogFolder") -> None:
        try:
            logfolder.close()
        except Exception:
            pass

    @cached_property
    def reg(self) -> Registry:
        # File create on setting values.
        return Registry(self.path / "const.yaml", create=True)

    @property
    def const(self) -> Registry:
        """Alias for reg. Access the const.yaml registry."""
        return self.reg

    @property
    def df(self) -> pd.DataFrame:
        """Get the full dataframe, flushing all data rows."""
        return self._handler.get_df()

    @property
    def df_path(self) -> Path:
        """Path to the backing ``data.feather`` file."""
        return self._handler.path

    @classmethod
    def new(cls, parent_path: Path, title: str = "untitled") -> "LogFolder":
        """Create the next numeric log directory under ``parent_path``."""
        # TODO: add locking or something.
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
        return cls(new_folder, title=title, create=True)

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
        """Run a parameter sweep and append returned rows to this log."""
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
        self.add_const_to_head(
            const=const_axs,
            dims={k: len(a) for k, a in run_axs.items()},
        )
        self.meta.plot_axes = list(run_axs.keys())

        step_table = list(itertools.product(*run_axs.values()))

        with logging_redirect_tqdm():
            for step in tqdm(step_table, ncols=80, desc=self.path.name):
                step_kws = dict(zip(run_axs.keys(), step))
                ret_kws = func(**step_kws, **const_axs)
                self.add_row(**step_kws, **ret_kws)

    def add_const(self, meta: dict = None, /, **kwargs):
        """Append constant values to ``const.yaml`` and save immediately."""
        if meta is None:
            meta = {}
        meta.update(kwargs)
        self.reg.root.update(meta)
        self.reg.save()

    def add_const_to_head(self, meta: dict = None, /, **kwargs):
        """Insert constant values at the top of ``const.yaml`` and save."""
        if meta is None:
            meta = {}
        meta.update(kwargs)
        for i, (k, v) in enumerate(meta.items()):
            self.reg.root.insert(i, k, v)
        self.reg.save()

    @deprecated("Use `add_const` instead.")
    def add_meta(self, meta: dict = None, /, **kwargs):
        return self.add_const(meta, **kwargs)

    @deprecated("Use `add_const_to_head` instead.")
    def add_meta_to_head(self, meta: dict = None, /, **kwargs):
        return self.add_const_to_head(meta, **kwargs)

    def flush(self) -> None:
        """Flash the pending data immediately, block until done."""
        self._handler.flush()

    def close(self) -> None:
        """Flush pending data and stop the background autosave thread."""
        self._handler.close()
        self._finalizer.detach()
        _ACTIVE_LOGFOLDERS.discard(self)
