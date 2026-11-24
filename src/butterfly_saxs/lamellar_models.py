"""Backward-compatible re-export shim for ``butterfly_saxs.lamellar.models``."""

from __future__ import annotations

from .lamellar.models import LamellarScene, LamellarSettings, _empty_scene as _empty_scene

__all__ = ["LamellarScene", "LamellarSettings"]
