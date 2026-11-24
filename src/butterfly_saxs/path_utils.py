"""Shared path/discovery rules and atomic directory staging helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


IMAGE_SUFFIXES = frozenset(
    {
        ".cbf", ".edf", ".tif", ".tiff", ".npy", ".npz", ".h5", ".hdf5",
        ".hdf", ".csv", ".txt",
    }
)


def is_supported_image_path(value: Any) -> bool:
    try:
        return Path(value).suffix.casefold() in IMAGE_SUFFIXES
    except (TypeError, ValueError):
        return False


def filter_supported_image_paths(values: Iterable[Any]) -> list[Path]:
    """Return files only, excluding detector sidecars such as ``.dat``."""

    result: list[Path] = []
    for value in values:
        try:
            path = Path(value)
        except (TypeError, ValueError):
            continue
        if path.is_file() and is_supported_image_path(path):
            result.append(path)
    return result


def canonical_path(value: str | os.PathLike[str] | Path) -> str:
    try:
        path = Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        path = Path(os.path.abspath(os.fspath(value)))
    return os.path.normcase(path.as_posix()).replace("\\", "/")


@contextmanager
def atomic_stage_dir(
    destination: str | os.PathLike[str] | Path,
    *,
    prefix: str | None = None,
    overwrite: bool = False,
    existing_message: str | None = None,
) -> Iterator[Path]:
    """Stage an export in a sibling temporary directory and publish atomically.

    When ``overwrite=False`` (default), ``destination`` must not already exist.
    When ``overwrite=True``, an existing ``destination`` directory is swapped
    through a sibling ``.<name>-previous`` backup and restored if ``os.replace``
    fails.  Any ``BaseException`` (including ``KeyboardInterrupt`` or
    cancellation) cleans up the staging directory before re-raising.
    """

    raw = Path(destination).expanduser()
    target = raw.resolve(strict=False)
    if not target.name:
        raise ValueError("destination must name an output directory")
    if not overwrite and (os.path.lexists(raw) or os.path.lexists(target)):
        raise FileExistsError(
            existing_message or f"export target already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    stage_prefix = prefix if prefix is not None else f".{target.name}.staging-"
    stage = Path(tempfile.mkdtemp(prefix=stage_prefix, dir=target.parent))
    committed = False
    try:
        yield stage
        if not overwrite:
            if os.path.lexists(raw) or os.path.lexists(target):
                raise FileExistsError(
                    existing_message or f"export target already exists: {target}"
                )
            os.rename(stage, target)
            committed = True
        else:
            backup = target.with_name(f".{target.name}-previous")
            if backup.exists():
                raise FileExistsError(
                    f"Previous backup exists: {backup}; recover it before replacing {target}."
                )
            if target.exists():
                os.replace(target, backup)
            try:
                os.replace(stage, target)
                committed = True
            except BaseException:
                if backup.exists():
                    try:
                        os.replace(backup, target)
                    except OSError:
                        pass
                raise
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if not committed and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


__all__ = [
    "IMAGE_SUFFIXES",
    "atomic_stage_dir",
    "canonical_path",
    "filter_supported_image_paths",
    "is_supported_image_path",
]
