import copy
import logging
import os
import re
import sys
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence, Set
from contextlib import suppress
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from typing_extensions import deprecated

if TYPE_CHECKING:
    from ruamel.yaml.constructor import BaseConstructor
    from ruamel.yaml.nodes import ScalarNode
    from ruamel.yaml.representer import BaseRepresenter

_logger = logging.getLogger(__name__)
_sentinel = object()


class FileSnap:
    """Track a file's modification time and size for cheap change detection."""

    __slots__ = ("path", "mtime", "size")

    def __init__(self, path: Path):
        self.path = Path(path)
        self.refresh()

    def refresh(self):
        st = self.path.stat()
        self.mtime = st.st_mtime
        self.size = st.st_size

    def changed(self) -> bool:
        st = self.path.stat()
        return (st.st_mtime, st.st_size) != (self.mtime, self.size)


class Registry:
    """A simple registry based on YAML file.

    `get`/`set` values synchronized with the file unless explicitly calling `get_local`/`set_local`.
    ```python
    reg = Registry('config.yaml')
    reg['new_key/sub_key'] = 123  # synced with file
    ```

    Operations on `root` and subitems are **local** and needs to be saved manually. e.g.
    ```python
    reg.reload()
    reg.root['another_key'] = 456  # local change, not synced until save.
    reg.save()
    ```
    Local changes will be discarded when `reload()`.

    NOTE: Local operations is useful for batch update without frequent file I/O.
    """

    def __init__(self, path: str | Path, create: bool = True, history_size: int = 2):
        path = Path(path)
        if path.exists():
            pass
        elif create:
            path.touch()
        else:
            raise FileNotFoundError(f"Registry file at '{path}' does not exist.")

        self.path = path
        self._yaml = _RecoverableYAML()
        self.root: CommentedMap = self.load()
        self._snap = FileSnap(self.path)
        self._root_dirty: bool = False

        self._undo_stack: deque[CommentedMap] = deque(maxlen=history_size)
        self._redo_stack: deque[CommentedMap] = deque(maxlen=history_size)

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, value):
        self.set(key, value, create_parents=True)

    def get(self, key: str, default=_sentinel):
        self.reload()
        return self.get_local(key, default)

    def set(self, key: str, value, create_parents: bool = True):
        self.reload()
        self.set_local(key, value, create_parents)
        self.save()

    def get_local(self, key: str, default=_sentinel):
        obj = self.root
        keys = key.split("/")
        for k in keys:
            try:
                obj = obj[k]
            except (KeyError, IndexError, TypeError):
                if default is _sentinel:
                    raise
                return default
        return obj

    def set_local(self, key: str, value, create_parents: bool = True):
        obj = self.root
        self._root_dirty = True
        keys = key.split("/")
        for k in keys[:-1]:
            if not (k in obj and isinstance(obj[k], Mapping)):
                if create_parents:
                    obj[k] = CommentedMap()
                else:
                    raise KeyError(f"Parent key '{k}' does not exist.")
            obj = obj[k]
        obj[keys[-1]] = value

    def print_local(self):
        """Print the local content to stdout."""
        self._yaml.dump(self.root, sys.stdout)

    def reload(self):
        if self._root_dirty or self._snap.changed():
            self.root = self.load()
            self._snap.refresh()
            self._root_dirty = False

    def load(self, path: str | Path | None = None) -> CommentedMap:
        # NOTE: `yaml.load` also returns `CommentedSeq`, `float`, `str`, `None`
        # or other build-in types depending on the top-level YAML content. But
        # only `CommentedMap` is legal for the use case of this class.
        path = self.path if path is None else Path(path)
        with open(path, "r", encoding="utf-8") as f:
            root = self._yaml.load(f)
        if root is None:
            root = CommentedMap()
            root.fa.set_block_style()
        return root

    def save(self, path: str | Path | None = None, *, record_history: bool = True):
        path = self.path if path is None else Path(path)
        is_primary_path = path == self.path
        previous = self.load(path) if record_history and is_primary_path else None

        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                self._yaml.dump(self.root, f)
            tmp_path.replace(path)
        finally:
            with suppress(OSError):
                tmp_path.unlink()

        if is_primary_path:
            self._snap.refresh()
            self._root_dirty = False
        if previous is not None:
            self._undo_stack.append(previous)
            self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(copy.deepcopy(self.root))
        self.root = self._undo_stack.pop()
        self.save(record_history=False)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(copy.deepcopy(self.root))
        self.root = self._redo_stack.pop()
        self.save(record_history=False)
        return True

    @deprecated("For backward compatibility only.")
    def copy(self) -> dict:
        self.reload()
        return _to_builtins(self.root)

    @deprecated("For backward compatibility only.")
    def cwd(self) -> str:
        return self["data_folder"]


class _RecoverableYAML:
    """Reuse a YAML parser until a load or dump failure invalidates it."""

    @cached_property
    def parser(self) -> YAML:
        return get_parser()

    def _invalidate(self) -> None:
        self.__dict__.pop("parser", None)

    def load(self, stream):
        try:
            return self.parser.load(stream)
        except BaseException:
            self._invalidate()
            raise

    def dump(self, data, stream):
        try:
            return self.parser.dump(data, stream)
        except BaseException:
            self._invalidate()
            raise


def get_parser() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 100
    # enable best_style in representer.represent_sequence.
    yaml.default_flow_style = None
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.representer.add_representer(np.ndarray, _represent_numpy_array)
    yaml.representer.add_multi_representer(np.generic, _represent_numpy_scalar)
    _set_yaml_for_labrad_units(yaml)
    return yaml


def _represent_numpy_array(dumper: "BaseRepresenter", data: np.ndarray):
    # return dumper.represent_sequence("!numpy", data.tolist(), flow_style=True)
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data.tolist(), flow_style=True
    )


def _represent_numpy_scalar(dumper: "BaseRepresenter", data: np.generic):
    return dumper.represent_data(data.item())


####### labrad.units support #########


def _set_yaml_for_labrad_units(yaml: YAML) -> YAML:
    return yaml  # placeholder if labrad is not available


try:
    import labrad.units as lab_units
    from labrad.units import Unit, Value, WithUnit

    def _set_yaml_for_labrad_units(yaml: YAML) -> YAML:
        try:
            yaml.constructor.add_constructor("!labrad_unit", _construct_labrad_value)
            yaml.representer.add_representer(WithUnit, _represent_labrad_value)
            yaml.representer.add_representer(Value, _represent_labrad_value)
            yaml.resolver.add_implicit_resolver(
                "!labrad_unit", _UNIT_PATTERN, list("+-0123456789")
            )
        except Exception:
            _logger.warning(
                "Failed to register labrad.units YAML support.", exc_info=True
            )
        return yaml
except ModuleNotFoundError as exc:
    if exc.name != "labrad":
        _logger.warning(
            "Failed to import labrad.units; unit support is disabled.", exc_info=True
        )
except Exception:
    _logger.warning(
        "Failed to import labrad.units; unit support is disabled.", exc_info=True
    )


_UNIT_PATTERN = re.compile(r"^\s*([-+]?\d[\d_]*(?:\.\d[\d_]*)?)\s*([A-Za-z]*)\s*$")


def _construct_labrad_value(loader: "BaseConstructor", node: "ScalarNode"):
    raw: str = loader.construct_scalar(node)
    match = _UNIT_PATTERN.match(raw)
    if not match:
        return raw

    magnitude_raw, unit_name = match.groups()
    unit_obj: Unit | None = getattr(lab_units, unit_name, None)
    if unit_obj is None:
        return raw

    magnitude = float(magnitude_raw.replace("_", ""))
    return magnitude * unit_obj


def _represent_labrad_value(dumper: "BaseRepresenter", data: "WithUnit"):
    unit_name = data.unit.name
    magnitude: float = data._value
    if magnitude.is_integer():
        spaced = f"{int(magnitude)} {unit_name}"
    else:
        spaced = f"{magnitude} {unit_name}"
    # spaced = str(data)
    # return dumper.represent_scalar('tag:yaml.org,2002:str', "="+spaced, style="")
    return dumper.represent_scalar("!labrad_unit", spaced)


def _to_builtins(obj):
    if isinstance(obj, Mapping):
        return {_to_builtins(k): _to_builtins(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        # BUG: labRAD dump tuple only, remove it!!
        return tuple(_to_builtins(item) for item in obj)
    if isinstance(obj, Set):
        return {_to_builtins(item) for item in obj}

    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)

    return obj
