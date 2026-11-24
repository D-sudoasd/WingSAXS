"""Backward-compatible re-export shim for ``butterfly_saxs.lamellar.render``."""

from __future__ import annotations

from .lamellar.render import (
    _FORMAT_VERSION as _FORMAT_VERSION,
    _camera_orientation as _camera_orientation,
    _colors as _colors,
    _convex_hull_2d as _convex_hull_2d,
    _draw_scale_bar as _draw_scale_bar,
    _font as _font,
    _image_extent as _image_extent,
    _image_to_array as _image_to_array,
    _title_suffix as _title_suffix,
    render_lamellar_2d,
    render_lamellar_3d,
    render_lamellar_combined,
)

__all__ = [
    "render_lamellar_2d",
    "render_lamellar_3d",
    "render_lamellar_combined",
]
