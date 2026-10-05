"""Streamed content digests and task-local file version evidence."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


def stream_digest(
    handle: Any, *, check_cancel: Callable[[], None] | None = None,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        if check_cancel is not None:
            check_cancel()
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


@lru_cache(maxsize=1)
def _windows_usn_query() -> Callable[[Path], int]:
    """Bind read-only per-file USN access; retain no handles or input results."""

    import ctypes
    import struct
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    control = kernel.DeviceIoControl
    control.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
    ]
    control.restype = wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL

    def read(path: Path) -> int:
        # FILE_READ_ATTRIBUTES, share read/write/delete, OPEN_EXISTING.
        handle = create(str(path), 0x80, 7, None, 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            versions = (ctypes.c_ushort * 2)(2, 3)
            result = ctypes.create_string_buffer(1024)
            length = wintypes.DWORD()
            # FSCTL_READ_FILE_USN_DATA, with V2/V3 record layouts.
            if not control(
                handle, 0x000900EB, versions, ctypes.sizeof(versions),
                result, ctypes.sizeof(result), ctypes.byref(length), None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            record_length, major = struct.unpack_from("<IH", result.raw)
            offset = {2: 24, 3: 40}.get(major)
            if offset is None or not offset + 8 <= record_length <= length.value:
                raise OSError("unsupported file USN record")
            return struct.unpack_from("<q", result.raw, offset)[0]
        finally:
            close(handle)

    return read


def file_version(path: Path, info: os.stat_result) -> tuple[int | None, ...]:
    """Return metadata plus a write-sensitive token, or None for no proof."""

    revision = info.st_ctime_ns
    if os.name == "nt":
        # Windows ctime is creation time; even ChangeTime is a timestamp and
        # can repeat for rapid writes. Only a nonzero USN authorizes reuse.
        try:
            value = _windows_usn_query()(path)
            revision = value if value > 0 else None
        except OSError:
            revision = None
    return (
        info.st_dev, info.st_ino, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns, revision,
    )
