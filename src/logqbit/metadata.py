from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar, overload

from .file_version import FileVersion

_ReadT = TypeVar("_ReadT")
_WriteT = TypeVar("_WriteT")
logger = logging.getLogger(__name__)


def _default_root(title: str) -> dict[str, object]:
    return {
        "title": title,
        "star": 0,
        "trash": False,
        "plot_axes": [],
        "plot_fields": [],
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "create_machine": socket.gethostname(),
    }


class _MetaField(Generic[_ReadT, _WriteT]):
    """Descriptor for a single key in LogMetadata.root."""

    def __init__(
        self,
        key: str,
        default: _ReadT,
        cast: Callable[..., _ReadT],
    ):
        self.key = key
        self.default = default
        self.cast = cast

    @overload
    def __get__(
        self,
        obj: None,
        objtype: type,
    ) -> _MetaField[_ReadT, _WriteT]: ...

    @overload
    def __get__(self, obj: LogMetadata, objtype: type) -> _ReadT: ...

    def __get__(
        self,
        obj: LogMetadata | None,
        objtype: type,
    ) -> _ReadT | _MetaField[_ReadT, _WriteT]:
        if obj is None:
            return self
        obj.reload()
        return self.cast(obj.root.get(self.key, self.default))

    def __set__(self, obj: LogMetadata, value: _WriteT) -> None:
        obj[self.key] = self.cast(value)


class LogMetadata:
    """JSON-backed metadata helper for a log folder.

    Common fields such as ``title``, ``star``, and ``plot_axes`` are exposed as
    descriptors and synchronized to ``metadata.json`` on assignment.
    """

    title = _MetaField[str, str]("title", "untitled", str)
    star = _MetaField[int, int]("star", 0, int)
    trash = _MetaField[bool, bool]("trash", False, bool)
    plot_axes = _MetaField[tuple[str, ...], str | Iterable[str]](
        "plot_axes",
        (),
        lambda v: (v,) if isinstance(v, str) else tuple(str(i) for i in v),
    )
    plot_fields = _MetaField[tuple[str, ...], str | Iterable[str]](
        "plot_fields",
        (),
        lambda v: (v,) if isinstance(v, str) else tuple(str(i) for i in v),
    )
    create_time = _MetaField[str, str]("create_time", "", str)
    create_machine = _MetaField[str, str]("create_machine", "", str)

    def __init__(
        self,
        path: str | Path,
        title: str = "untitled",
        create: bool = True,
        *,
        default_on_error: bool = False,
    ):
        path = Path(path)
        if path.exists():
            pass
        elif create:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_default_root(title), f)
        else:
            raise FileNotFoundError(f"Metadata file at '{path}' does not exist.")

        self.path = path
        self._default_on_error = default_on_error
        self.root = self._load()
        self._file_version = FileVersion.require(self.path)

    def reload(self):
        current_version = FileVersion.require(self.path)
        if current_version != self._file_version:
            self.root = self._load()
            self._file_version = current_version

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                root = json.load(f)
            if not isinstance(root, dict):
                raise ValueError(
                    f"Metadata root in '{self.path}' must be a JSON object."
                )
            return root
        except (OSError, UnicodeError, ValueError) as exc:
            if not self._default_on_error:
                raise
            logger.warning("Failed to load metadata file %s: %s", self.path, exc)
            return _default_root("<invalid metadata>")

    def save(self, path: str | Path | None = None) -> None:
        path = self.path if path is None else Path(path)

        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.root, f)
            tmp_path.replace(path)
        finally:
            with suppress(OSError):
                tmp_path.unlink()

        if path == self.path:
            self._file_version = FileVersion.require(self.path)

    def update(
        self,
        values: Mapping[str, object] | None = None,
        /,
        **changes: object,
    ) -> None:
        """Atomically persist one metadata update."""
        pending = dict(values or {})
        pending.update(changes)
        if not pending:
            self.reload()
            return

        for key, value in pending.items():
            descriptor = getattr(type(self), key, None)
            if isinstance(descriptor, _MetaField):
                pending[key] = descriptor.cast(value)

        self.reload()
        self.root.update(pending)
        self.save()

    def __getitem__(self, key: str):
        self.reload()
        return self.root[key]

    def __setitem__(self, key: str, value):
        self.update({key: value})
