import errno
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

try:
    from .registry import FileSnap
except ImportError:
    from registry import FileSnap  # type: ignore


class LogMetadata:
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

    def save(self, path: str | Path | None = None, timeout=0.1):
        path = self.path if path is None else Path(path)
        tmp = path.with_suffix(".tmp")

        with FileLock(path, timeout=timeout):
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.root, f)
            tmp.replace(path)

    def __getitem__(self, key: str):
        self.reload()
        return self.root[key]

    def __setitem__(self, key: str, value):
        self.reload()
        self.root[key] = value
        self.save()

    @property
    def title(self) -> str:
        return self["title"]

    @title.setter
    def title(self, value: str):
        self["title"] = str(value)

    @property
    def star(self) -> int:
        return int(self["star"])

    @star.setter
    def star(self, value: int):
        self["star"] = int(value)

    @property
    def trash(self) -> bool:
        return bool(self["trash"])

    @trash.setter
    def trash(self, value: bool):
        self["trash"] = bool(value)

    @property
    def plot_axes(self) -> list[str]:
        axes = self["plot_axes"]
        if isinstance(axes, list):
            return [str(item) for item in axes]
        return []

    @plot_axes.setter
    def plot_axes(self, value: list[str]):
        if not isinstance(value, list):
            raise TypeError("plot_axes must be a list of strings")
        self["plot_axes"] = [str(item) for item in value]

    @property
    def create_time(self) -> str:
        """Get creation time."""
        value = self["create_time"]
        return str(value) if value is not None else ""

    @property
    def create_machine(self) -> str:
        """Get creation machine."""
        value = self["create_machine"]
        return str(value) if value is not None else ""


class FileLock:
    def __init__(
        self,
        path: str | Path,
        timeout: float = 0.5,
        delete_on_release: bool = True,
    ):
        self.path = Path(path).with_suffix(".lock")
        self.timeout = timeout
        self.delete_on_release = delete_on_release
        self._file = None

    def acquire(self):
        lock_path = self.path
        self._file = open(lock_path, "a+b")

        deadline = time.time() + self.timeout
        while True:
            try:
                if os.name == "nt":
                    # Windows: 非阻塞锁定前 1 字节
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Unix: 非阻塞独占锁
                    fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return  # 成功获取锁
            except (OSError, BlockingIOError) as e:
                # 检查是否为"资源被占用"错误
                if getattr(e, "errno", None) not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.time() > deadline:
                    raise TimeoutError(f"Timeout while waiting for lock: {lock_path}")
                time.sleep(0.02)

    def release(self):
        if not self._file:
            return

        try:
            if os.name == "nt":
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._file, fcntl.LOCK_UN)
        finally:
            try:
                self._file.close()
            finally:
                if self.delete_on_release:
                    try:
                        if self.path.exists():
                            self.path.unlink()
                    except OSError:
                        pass
                self._file = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
