from __future__ import annotations
import json
import os
import socket
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar, overload
from collections.abc import Callable

try:
    from .registry import FileSnap
except ImportError:
    from registry import FileSnap  # type: ignore


_T = TypeVar("_T")


class _MetaField(Generic[_T]):
    """Descriptor for a single key in LogMetadata.root."""

    def __init__(self, key: str, default: _T, cast: Callable[..., _T]):
        self.key = key
        self.default = default
        self.cast = cast

    @overload
    def __get__(self, obj: None, objtype: type) -> _MetaField[_T]: ...
    @overload
    def __get__(self, obj: LogMetadata, objtype: type) -> _T: ...

    def __get__(self, obj: LogMetadata | None, objtype: type) -> _T | _MetaField[_T]:
        if obj is None:
            return self
        obj.reload()
        return self.cast(obj.root.get(self.key, self.default))

    def __set__(self, obj: LogMetadata, value: _T) -> None:
        obj[self.key] = self.cast(value)


class LogMetadata:
    title = _MetaField("title", "untitled", str)
    star = _MetaField("star", 0, int)
    trash = _MetaField("trash", False, bool)
    plot_axes = _MetaField("plot_axes", [], lambda v: [str(i) for i in v])
    plot_fields = _MetaField("plot_fields", [], lambda v: [str(i) for i in v])
    create_time = _MetaField("create_time", "", str)
    create_machine = _MetaField("create_machine", "", str)

    def __init__(self, path: str | Path, title: str = "untitled", create: bool = True):
        path = Path(path)
        if path.exists():
            pass
        elif create:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "title": title,
                        "star": 0,
                        "trash": False,
                        "plot_axes": [],
                        "plot_fields": [],
                        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "create_machine": socket.gethostname(),
                    },
                    f,
                )
        else:
            raise FileNotFoundError(f"Metadata file at '{path}' does not exist.")

        self.path = path
        self.root: dict = self.load()
        self._snap = FileSnap(self.path)

    def reload(self):
        if self._snap.changed():
            self.root = self.load()

    def load(self, path: str | Path | None = None) -> dict:
        path = self.path if path is None else path
        with open(self.path, "r", encoding="utf-8") as f:
            root = json.load(f)
        return root

    def save(self, path: str | Path | None = None) -> None:
        path = self.path if path is None else Path(path)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.root, f)
        Path(tmp).replace(path)

    def __getitem__(self, key: str):
        self.reload()
        return self.root[key]

    def __setitem__(self, key: str, value):
        self.reload()
        self.root[key] = value
        self.save()
