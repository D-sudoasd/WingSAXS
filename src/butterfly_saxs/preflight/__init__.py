"""Read-only package preflight subpackage for two-dimensional SAXS data."""

from __future__ import annotations

from ..io import load_image
from .hash import PreflightError, source_tree_sha256
from .runner import PREFLIGHT_SCHEMA_VERSION, run_preflight

__all__ = [
    "PREFLIGHT_SCHEMA_VERSION",
    "PreflightError",
    "load_image",
    "run_preflight",
    "source_tree_sha256",
]
