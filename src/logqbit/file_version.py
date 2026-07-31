"""Cheap file-version snapshots used by caches and file watchers."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileVersion:
    """Identity and content-change hints from one file stat call."""

    mtime_ns: int
    size: int
    inode: int

    @classmethod
    def from_path(cls, path: str | Path) -> FileVersion | None:
        """Return the current regular-file version, or ``None`` if unavailable."""
        try:
            file_stat = Path(path).stat()
        except OSError:
            return None
        if not stat.S_ISREG(file_stat.st_mode):
            return None
        return cls(
            file_stat.st_mtime_ns,
            file_stat.st_size,
            file_stat.st_ino,
        )

    @classmethod
    def require(cls, path: str | Path) -> FileVersion:
        """Return an existing regular-file version or raise a useful error."""
        path = Path(path)
        version = cls.from_path(path)
        if version is not None:
            return version
        path.stat()
        raise ValueError(f"'{path}' is not a regular file")
