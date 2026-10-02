"""Read-only freshness checks for one explicitly selected delivery receipt."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from .delivery_dependencies import local_path_problem


class _ChangedDuringRead(OSError):
    pass


class _NotRegularFile(OSError):
    pass


def _snapshot(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _matches_open_file(path_snapshot: tuple[int, ...], handle_snapshot: tuple[int, ...]) -> bool:
    # Windows Python can expose creation time through stat/lstat and change
    # time through fstat. Path stat also adds synthetic execute bits for
    # .exe/.bat/.cmd/.com, while fstat cannot inspect the filename. Compare the
    # shared identity/content metadata across APIs, then retain each API's full
    # snapshot (including ctime and all mode bits) for before/after checks.
    mode_mask = ~0o111 if os.name == "nt" else -1
    return (path_snapshot[:2] == handle_snapshot[:2]
            and (path_snapshot[2] & mode_mask) == (handle_snapshot[2] & mode_mask)
            and path_snapshot[3:-1] == handle_snapshot[3:-1])


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Receipt has duplicate JSON key {key!r}; select an unambiguous receipt.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Receipt contains non-JSON value {value}; use a valid JSON receipt.")


def _hash_file(root: Path, path: Path) -> tuple[str, int, tuple[int, ...]]:
    if local_path_problem(root, path):
        raise _ChangedDuringRead("path became unsafe")
    before = _snapshot(path.lstat())
    if not stat.S_ISREG(before[2]):
        raise _NotRegularFile("expected a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = _snapshot(os.fstat(handle.fileno()))
        if (not _matches_open_file(before, opened)
                or local_path_problem(root, path) or _snapshot(path.lstat()) != before):
            raise _ChangedDuringRead("file changed before reading")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if _snapshot(os.fstat(handle.fileno())) != opened:
            raise _ChangedDuringRead("file changed during reading")
    if local_path_problem(root, path) or _snapshot(path.stat()) != before:
        raise _ChangedDuringRead("file changed during reading")
    return digest.hexdigest(), before[3], before


def verify_delivery_bindings(receipt: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Compare only the selected receipt's top-level ``bindings`` with disk.

    Each binding must have a portable relative ``path`` and a SHA-256 string.
    Invalid or unsafe input raises ``ValueError``; stale/missing named files
    return ``exit_code=1``. A current result returns zero and means file-binding
    integrity only, never scientific acceptance or validation of historical
    nested records. Nothing is written, discovered recursively, or repaired.
    """
    receipt = Path(receipt).expanduser().absolute()
    if receipt.is_symlink():
        raise ValueError("Receipt must be a regular file, not a symlink.")
    receipt = receipt.resolve()
    root = Path(root).expanduser().resolve() if root is not None else receipt.parent
    if not root.is_dir():
        raise ValueError(f"Delivery root is not a directory: {root}")
    receipt_snapshot = _snapshot(receipt.stat())
    if not stat.S_ISREG(receipt_snapshot[2]):
        raise ValueError("Receipt must be a regular JSON file.")
    with receipt.open("r", encoding="utf-8-sig") as handle:
        opened = _snapshot(os.fstat(handle.fileno()))
        if (not _matches_open_file(receipt_snapshot, opened)
                or receipt.is_symlink() or _snapshot(receipt.stat()) != receipt_snapshot):
            raise ValueError("Receipt changed before reading; retry after its writer finishes.")
        try:
            document = json.load(handle, object_pairs_hook=_unique_object,
                                 parse_constant=_invalid_constant)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read delivery receipt as JSON: {exc}") from exc
        if _snapshot(os.fstat(handle.fileno())) != opened:
            raise ValueError("Receipt changed during reading; retry after its writer finishes.")
    if not isinstance(document, Mapping):
        raise ValueError("Delivery receipt must be a JSON object with a top-level bindings list.")
    entries = document.get("bindings")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Delivery receipt needs a nonempty top-level bindings list; select the current closeout receipt.")

    selected: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Each delivery binding must be an object with path and sha256 strings.")
        relative, expected = entry.get("path"), entry.get("sha256")
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or ":" in relative or "\x00" in relative
                or any(not part or part.startswith(".") for part in relative.split("/"))):
            raise ValueError(f"Unsafe binding path {relative!r}; use a non-hidden relative path with '/' separators inside the delivery root.")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"Binding {relative!r} needs a 64-character hexadecimal SHA-256.")
        path = root / relative
        problem = local_path_problem(root, path)
        if problem:
            raise ValueError(f"Unsafe binding path {relative!r}: {problem}; bind a regular file inside the delivery root.")
        if path == receipt or (path.exists() and path.samefile(receipt)):
            raise ValueError(f"Receipt cannot bind itself ({relative!r}); select a parent closeout receipt.")
        expected = expected.lower()
        if relative in selected and selected[relative] != expected:
            raise ValueError(f"Conflicting duplicate binding for {relative!r}; select an unambiguous receipt.")
        selected[relative] = expected

    checks: list[dict[str, Any]] = []
    snapshots: dict[str, tuple[int, ...]] = {}
    for relative, expected in selected.items():
        row: dict[str, Any] = {"path": relative, "expected_sha256": expected}
        try:
            actual, size, snapshot = _hash_file(root, root / relative)
            snapshots[relative] = snapshot
            row.update(actual_sha256=actual, size_bytes=size,
                       status="current" if actual == expected else "sha256_mismatch")
        except _ChangedDuringRead:
            row["status"] = "changed_during_read"
        except FileNotFoundError:
            row["status"] = "missing_file"
        except _NotRegularFile:
            row["status"] = "not_regular_file"
        except OSError:
            row["status"] = "unreadable_file"
        checks.append(row)

    # A later file's read can overlap a writer changing an earlier file. Do not
    # accept a mixed-generation collection just because each read was stable.
    for row in checks:
        relative = row["path"]
        if relative not in snapshots:
            continue
        path = root / relative
        try:
            changed = local_path_problem(root, path) or _snapshot(path.stat()) != snapshots[relative]
        except OSError:
            changed = True
        if changed:
            row["status"] = "changed_during_read"
    if receipt.is_symlink() or _snapshot(receipt.stat()) != receipt_snapshot:
        raise ValueError("Receipt changed during verification; select the finished current receipt and retry.")

    actions = {
        "sha256_mismatch": "The recorded SHA is stale or the file changed; select a newly finalized receipt or restore its exact named file.",
        "missing_file": "Restore the named file inside the delivery root, or select the receipt for the files actually delivered.",
        "changed_during_read": "Wait for the file writer to finish, then verify the current receipt again.",
        "not_regular_file": "Restore the named regular file; a directory or special file cannot satisfy a binding.",
        "unreadable_file": "Make the named file readable, then verify the same receipt again.",
    }
    for row in checks:
        if row["status"] != "current":
            row["action"] = actions[row["status"]]
    current = sum(row["status"] == "current" for row in checks)
    stale = len(checks) - current
    return {"schema_version": "wingsaxs.delivery-bindings.v1",
            "status": "stale" if stale else "current", "exit_code": 1 if stale else 0,
            "receipt": str(receipt), "root": str(root), "checks": checks,
            "counts": {"checked": len(checks), "current": current, "stale": stale},
            "scientific_acceptance_assessed": False}
