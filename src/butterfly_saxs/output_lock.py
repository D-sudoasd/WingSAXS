"""Cross-process locks for output directories."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
import os
from pathlib import Path
import stat
import threading


_LOCK_NAME = ".wingsaxs.lock"
_STATE_LOCK = threading.Lock()
_ACTIVE: dict[str, int] = {}
_THREAD_STATE = threading.local()


def _acquire_file_lock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_file_lock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _locked_error(root: Path, cause: OSError | None = None) -> OSError:
    error = OSError(
        f"Output directory is locked by another process or thread: {root}; "
        "wait for the active operation to finish, then retry."
    )
    if cause is not None:
        error.__cause__ = cause
    return error


@contextmanager
def output_lock(
    root: str | os.PathLike[str], *, create: bool = False
) -> Iterator[Path]:
    """Exclusively lock an existing output directory until the context exits.

    The stable lock file is intentionally retained after release. Nested use by
    the owning thread is reentrant; another thread or process fails promptly.
    """
    path = Path(root).expanduser()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"output directory does not exist: {path}") from None
    if not resolved.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {resolved}")
    lock_path = resolved / _LOCK_NAME
    try:
        info = lock_path.lstat()
    except FileNotFoundError:
        info = None
    if info is not None and (stat.S_ISLNK(info.st_mode) or info.st_nlink > 1
                             or getattr(info, "st_file_attributes", 0) & 0x400):
        raise OSError(f"Refusing a linked output lock file: {lock_path}; use a regular lock file inside the output directory.")

    key = os.path.normcase(os.fspath(resolved))
    held: dict[str, list[object]] = getattr(_THREAD_STATE, "held", {})
    _THREAD_STATE.held = held
    nested = held.get(key)
    if nested is not None:
        nested[1] = int(nested[1]) + 1
        try:
            yield resolved
        finally:
            nested[1] = int(nested[1]) - 1
        return

    owner = threading.get_ident()
    handle = None
    with _STATE_LOCK:
        if key in _ACTIVE:
            raise _locked_error(resolved)
        try:
            handle = lock_path.open("a+b")
            _acquire_file_lock(handle)
        except OSError as exc:
            if handle is not None:
                handle.close()
            raise _locked_error(resolved, exc) from exc
        _ACTIVE[key] = owner
        state: list[object] = [handle, 1]
        held[key] = state

    try:
        yield resolved
    finally:
        state[1] = int(state[1]) - 1
        if int(state[1]) == 0:
            with _STATE_LOCK:
                held.pop(key, None)
                _ACTIVE.pop(key, None)
                try:
                    _release_file_lock(handle)
                finally:
                    handle.close()
