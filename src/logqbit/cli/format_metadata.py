"""Format existing metadata without changing its JSON values."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from decimal import Decimal
from pathlib import Path


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number: {value}")


def _format_content(original: bytes) -> bytes:
    text = original.decode("utf-8")
    options = {"object_pairs_hook": _unique_object, "parse_constant": _reject_constant}
    exact = json.loads(text, parse_float=Decimal, **options)
    if not isinstance(exact, dict):
        raise ValueError("metadata root must be a JSON object")
    root = json.loads(text, **options)
    formatted = json.dumps(root, ensure_ascii=False, indent=4, allow_nan=False)
    # A float conversion must not silently round a custom numeric value.
    if json.loads(formatted, parse_float=Decimal, **options) != exact:
        raise ValueError("numeric value cannot be preserved")
    return formatted.encode("utf-8")


def _replace(
    path: Path, original: bytes, formatted: bytes, initial: os.stat_result
) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(formatted)
        os.chmod(temporary, stat.S_IMODE(initial.st_mode))
        current = path.stat(follow_symlinks=False)
        if (current.st_ino, current.st_mtime_ns, current.st_size) != (
            initial.st_ino,
            initial.st_mtime_ns,
            initial.st_size,
        ) or path.read_bytes() != original:
            raise ValueError("file changed during formatting")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def format_metadata_json(directory: Path, *, dry_run: bool = False) -> int:
    """Recursively format metadata files and report individual failures."""
    if directory.is_symlink() or not directory.is_dir():
        print(
            f"Error: not a directory or a symbolic link: {directory}", file=sys.stderr
        )
        return 1

    changed = unchanged = failed = skipped = 0

    def scan_error(error: OSError) -> None:
        nonlocal failed
        failed += 1
        print(f"Error scanning {error.filename}: {error}", file=sys.stderr)

    for root, dirs, files in os.walk(directory, followlinks=False, onerror=scan_error):
        root_path = Path(root)
        for name in list(dirs):
            if (root_path / name).is_symlink():
                dirs.remove(name)
                skipped += 1
        if "metadata.json" not in files:
            continue
        path = root_path / "metadata.json"
        if path.is_symlink():
            skipped += 1
            continue
        try:
            initial = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(initial.st_mode):
                skipped += 1
                continue
            original = path.read_bytes()
            formatted = _format_content(original)
            if formatted == original:
                unchanged += 1
                continue
            if not dry_run:
                _replace(path, original, formatted, initial)
            changed += 1
            print(f"{'Would format' if dry_run else 'Formatted'}: {path}")
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            failed += 1
            print(f"Error: {path}: {exc}", file=sys.stderr)

    label = "Would format" if dry_run else "Formatted"
    print(
        f"{label}: {changed}; unchanged: {unchanged}; skipped: {skipped}; errors: {failed}"
    )
    return int(failed > 0)
