"""Backward-compatible re-export shim for ``butterfly_saxs.lamellar.sources``."""

from __future__ import annotations

from .lamellar.sources import (
    _attach_single_npz as _attach_single_npz,
    _batch_prefix as _batch_prefix,
    _resolve_native_pair as _resolve_native_pair,
    load_lamellar_sources,
)

__all__ = ["load_lamellar_sources"]
