import re
import sys
import warnings
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from typing_extensions import deprecated

if TYPE_CHECKING:
    from ruamel.yaml.constructor import BaseConstructor
    from ruamel.yaml.nodes import ScalarNode, SequenceNode
    from ruamel.yaml.representer import BaseRepresenter

_sentinel = object()


class FileSnap:
    __slots__ = ("path", "mtime", "size")

    def __init__(self, path: Path):
        self.path = Path(path)
        self._refresh()

    def _refresh(self):
        st = self.path.stat()
        self.mtime = st.st_mtime
        self.size = st.st_size

    def changed(self) -> bool:
        st = self.path.stat()
        if (st.st_mtime, st.st_size) != (self.mtime, self.size):
            self._refresh()
            return True
        return False


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

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.yaml = get_parser()
        self.root: CommentedMap = None
        self.load()
        self._snap = FileSnap(self.path)

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
        self.yaml.dump(self.root, sys.stdout)

    def reload(self):
        """Reloads the file if it has changed since the last load."""
        if self._snap.changed():
            self.load()
        return self.root

    def load(self):
        """Load YAML file to self.root."""
        # NOTE: `yaml.load` also returns `CommentedSeq`, `float`, `str`, `None`
        # or other build-in types depending on the top-level YAML content. But
        # only `CommentedMap` is legal for the use case of this class.
        with open(self.path, "r", encoding="utf-8") as f:
            self.root = self.yaml.load(f)
        return self.root

    def save(self, path: str | Path | None = None):
        """Save self.root to YAML file."""
        path = self.path if path is None else path
        with open(path, "w", encoding="utf-8") as f:
            self.yaml.dump(self.root, f)

    @deprecated("For backward compatibility only.")
    def copy(self) -> dict:
        return _to_builtins(self.root)

    @deprecated("For backward compatibility only.")
    def cwd(self) -> str:
        return self["data_folder"]


def get_parser() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 100
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.constructor.add_constructor("!numpy", _construct_numpy)
    yaml.representer.add_representer(np.ndarray, _represent_numpy)
    _set_yaml_for_labrad_units(yaml)
    return yaml


def _represent_numpy(dumper: "BaseRepresenter", data: np.ndarray):
    return dumper.represent_sequence("!numpy", data.tolist(), flow_style=True)


def _construct_numpy(loader: "BaseConstructor", node: "SequenceNode"):
    return np.array(loader.construct_sequence(node))


####### labrad.units support #########


def _set_yaml_for_labrad_units(yaml: YAML) -> YAML:
    return yaml  # placeholder if labrad is not available


try:
    import labrad.units as lab_units
    from labrad.units import Unit, Value, WithUnit

    def _set_yaml_for_labrad_units(yaml: YAML) -> YAML:
        yaml.resolver.add_implicit_resolver(
            "!labrad_unit", _UNIT_PATTERN, list("+-0123456789")
        )
        yaml.constructor.add_constructor("!labrad_unit", _construct_labrad_value)
        yaml.representer.add_representer(WithUnit, _represent_labrad_value)
        yaml.representer.add_representer(Value, _represent_labrad_value)
except ImportError:
    warnings.warn("labrad.units not found, unit support disabled.", ImportWarning)


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


def _represent_labrad_value(dumper: "BaseRepresenter", data: WithUnit):
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
