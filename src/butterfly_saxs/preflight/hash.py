"""Content hashing, path authorization, and read-time provenance for preflight."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..file_identity import (
    sha256_bytes as _sha256_bytes_core,
    sha256_file as _sha256_file_core,
)
from ..path_contract import (
    PathContractError,
    display_path as _display_path_contract,
    resolve_authorized_path,
)
from ..producer import iter_calculation_source_files
from ..serialization import json_safe as _json_safe_core


class PreflightError(ValueError):
    """Raised when a package cannot satisfy the preflight contract."""


def json_safe(value: Any) -> Any:
    """Convert values to strict-JSON-compatible Python objects."""

    return _json_safe_core(value)


def strict_json(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def display_path(path: Path, package: Path) -> str:
    return _display_path_contract(path, package)


def resolve_path(
    package: Path,
    value: str | os.PathLike[str] | Path,
    *,
    base_dir: Path | None = None,
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    label: str = "input",
) -> Path:
    try:
        return resolve_authorized_path(
            value,
            package_root=package,
            base_dir=base_dir,
            external_roots=external_roots,
            label=label,
        )
    except PathContractError as exc:
        raise PreflightError(str(exc)) from exc


def resolve_output_path(value: str | os.PathLike[str] | Path) -> Path:
    """Resolve output relative to the caller's current working directory."""

    return Path(value).expanduser().resolve(strict=False)


def sha256_bytes(data: bytes) -> str:
    return _sha256_bytes_core(data)


def sha256_file(path: Path, *, error_type: type[Exception] = PreflightError) -> str:
    try:
        return _sha256_file_core(path)
    except OSError as exc:
        raise error_type(f"cannot read input for SHA-256: {path}: {exc}") from exc


def hashable_inline(
    value: Any,
    *,
    json_safe: Callable[[Any], Any] = json_safe,
) -> Any:
    """Describe inline arrays without expanding detector-sized data to JSON."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": [int(item) for item in array.shape],
            "sha256": sha256_bytes(array.tobytes()),
        }
    if isinstance(value, Mapping):
        return {
            str(key): hashable_inline(item, json_safe=json_safe)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [hashable_inline(item, json_safe=json_safe) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return hashable_inline(value.item(), json_safe=json_safe)
    return json_safe(value)


def inline_hash(
    value: Any,
    *,
    json_safe: Callable[[Any], Any] = json_safe,
) -> str:
    canonical = json.dumps(
        hashable_inline(value, json_safe=json_safe),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_bytes(canonical.encode("utf-8"))


def read_file_record(
    path: Path,
    package: Path,
    reader: Callable[[], Any],
    records: list[dict[str, Any]],
    *,
    display_path: Callable[[Path, Path], str] = display_path,
    error_type: type[Exception] = PreflightError,
) -> Any:
    """Run a reader bracketed by SHA-256 checks and record both digests."""

    before = sha256_file(path, error_type=error_type)
    try:
        value = reader()
    except Exception as exc:
        after: str | None = None
        post_hash_error: str | None = None
        try:
            after = sha256_file(path, error_type=error_type)
        except Exception as hash_exc:  # pragma: no cover - damaged/removed source
            post_hash_error = f"{type(hash_exc).__name__}: {hash_exc}"
        record = {
            "path": display_path(path, package),
            "algorithm": "sha256",
            "sha256_before": before,
            "sha256_after": after,
            "before": before,
            "after": after,
            "unchanged": bool(after is not None and before == after),
            "read_status": "error",
            "read_error": f"{type(exc).__name__}: {exc}",
        }
        if post_hash_error is not None:
            record["post_read_hash_error"] = post_hash_error
        records.append(record)
        if after is not None and before != after:
            raise error_type(f"input changed while being read: {path}") from exc
        raise
    after = sha256_file(path, error_type=error_type)
    record = {
        "path": display_path(path, package),
        "algorithm": "sha256",
        "sha256_before": before,
        "sha256_after": after,
        "before": before,
        "after": after,
        "unchanged": before == after,
    }
    records.append(record)
    if before != after:
        raise error_type(f"input changed while being read: {path}")
    return value


def source_tree_sha256(package_root: Path | None = None) -> str:
    """Recursively hash calculation source modules sorted by POSIX relative path."""

    digest = hashlib.sha256()
    for rel_posix, path in iter_calculation_source_files(package_root):
        digest.update(rel_posix.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "PreflightError",
    "display_path",
    "hashable_inline",
    "inline_hash",
    "json_safe",
    "read_file_record",
    "resolve_output_path",
    "resolve_path",
    "sha256_bytes",
    "sha256_file",
    "source_tree_sha256",
    "strict_json",
]
