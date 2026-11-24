"""Backward-compatible re-export shim for ``butterfly_saxs.lamellar.scattering``."""

from __future__ import annotations

from .lamellar.scattering import (
    DEFAULT_GRID_SHAPE,
    MAX_GRID_PIXELS,
    MAX_GRID_SIDE,
    MAX_RASTER_CANDIDATE_PIXELS,
    LamellarScatteringResult,
    _scene_arrays as _scene_arrays,
    export_lamellar_scattering,
    render_lamellar_scattering,
    simulate_projected_density_fft,
)

__all__ = [
    "DEFAULT_GRID_SHAPE",
    "LamellarScatteringResult",
    "MAX_GRID_PIXELS",
    "MAX_GRID_SIDE",
    "MAX_RASTER_CANDIDATE_PIXELS",
    "export_lamellar_scattering",
    "render_lamellar_scattering",
    "simulate_projected_density_fft",
]
