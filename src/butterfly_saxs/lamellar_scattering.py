"""Illustrative 2D projection FFTs of finite lamellar scene geometry.

This module transforms the current schematic geometry into a projected
contrast-density field and its 2D Fourier power.  It does not model detector
response or estimate measured SAXS intensity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from numbers import Integral
from typing import Any

import numpy as np

from .font_support import font_properties
from .lamellar_models import LamellarScene


DEFAULT_GRID_SHAPE = (512, 512)
MAX_GRID_SIDE = 2048
MAX_GRID_PIXELS = 2_097_152
MAX_RASTER_CANDIDATE_PIXELS = 75_000_000
_CORNER_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ),
    dtype=float,
)


@dataclass(frozen=True)
class LamellarScatteringResult:
    """Projected density and uncalibrated 2D Fourier power for one scene."""

    projected_density: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    intensity: np.ndarray
    bounds_xy: np.ndarray
    pixel_size: tuple[float, float]
    length_unit: str
    diagnostics: dict[str, Any]
    assumptions: tuple[str, ...]
    scene_status: str
    scene_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        density = np.array(self.projected_density, dtype=float, copy=True)
        qx = np.array(self.qx, dtype=float, copy=True)
        qy = np.array(self.qy, dtype=float, copy=True)
        intensity = np.array(self.intensity, dtype=float, copy=True)
        bounds = np.array(self.bounds_xy, dtype=float, copy=True)
        if density.ndim != 2 or intensity.shape != density.shape:
            raise ValueError("projected_density and intensity must share a 2D shape")
        if qx.shape != (density.shape[1],) or qy.shape != (density.shape[0],):
            raise ValueError("qx and qy must match the density columns and rows")
        if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
            raise ValueError("bounds_xy must have finite shape (2, 2)")
        if not all(np.all(np.isfinite(array)) for array in (density, qx, qy, intensity)):
            raise ValueError("scattering result arrays must be finite")
        if np.any(intensity < 0.0):
            raise ValueError("intensity must be nonnegative")
        if len(self.pixel_size) != 2 or not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in self.pixel_size
        ):
            raise ValueError("pixel_size must contain two finite positive values")
        if self.length_unit not in {"nm", "relative"}:
            raise ValueError("length_unit must be 'nm' or 'relative'")
        object.__setattr__(self, "projected_density", density)
        object.__setattr__(self, "qx", qx)
        object.__setattr__(self, "qy", qy)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "bounds_xy", bounds)
        object.__setattr__(
            self,
            "pixel_size",
            (float(self.pixel_size[0]), float(self.pixel_size[1])),
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "assumptions", tuple(str(v) for v in self.assumptions))
        object.__setattr__(self, "scene_flags", tuple(str(v) for v in self.scene_flags))

    @property
    def normalized_intensity(self) -> np.ndarray:
        """Return power divided by its maximum, or zeros for an empty pattern."""

        maximum = float(np.max(self.intensity)) if self.intensity.size else 0.0
        if maximum <= 0.0:
            return np.zeros_like(self.intensity)
        return self.intensity / maximum

    @property
    def density_unit(self) -> str:
        if self.length_unit == "nm":
            return "model e⁻ nm⁻²"
        return "model density × relative length"

    @property
    def q_unit(self) -> str:
        return "nm⁻¹" if self.length_unit == "nm" else "relative⁻¹"

    def metadata(self) -> dict[str, Any]:
        """Return a strict-JSON-safe record describing this conditional model."""

        return {
            "schema": "butterfly_saxs.lamellar_projection_fft.v1",
            "model_scope": "illustrative_projected_density_fft_only",
            "interpretation": (
                "Conditional forward visualization from the supplied LamellarScene; "
                "not measured scattering intensity, a fit, or a unique reconstruction."
            ),
            "scene_status": self.scene_status,
            "scene_flags": list(self.scene_flags),
            "length_unit": self.length_unit,
            "density_contrast": {
                "value": 1.0,
                "definition": (
                    "unit model electron-density contrast per scene-length-unit cubed; "
                    "this is a normalization convention, not a calibrated material value"
                ),
                "projected_density_unit": self.density_unit,
            },
            "q_convention": "q = 2*pi*fftshift(fftfreq); q = (qx, qy)",
            "q_unit": self.q_unit,
            "fft_intensity_definition": "abs(fft2(projected_density - mean(projected_density)))**2",
            "intensity_calibration": "uncalibrated_model_power",
            "grid_shape_rows_columns": [
                int(self.projected_density.shape[0]),
                int(self.projected_density.shape[1]),
            ],
            "canvas_bounds_xy": [
                [float(value) for value in self.bounds_xy[0]],
                [float(value) for value in self.bounds_xy[1]],
            ],
            "pixel_size_xy": [float(value) for value in self.pixel_size],
            "diagnostics": _json_finite(self.diagnostics),
            "assumptions": list(self.assumptions),
        }


def _validate_grid_shape(shape: Sequence[int]) -> tuple[int, int]:
    if isinstance(shape, (str, bytes)) or len(shape) != 2:
        raise ValueError("shape must contain (rows, columns)")
    dimensions: list[int] = []
    for value in shape:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("grid dimensions must be integers")
        dimension = int(value)
        if dimension < 16 or dimension > MAX_GRID_SIDE:
            raise ValueError(f"each grid dimension must be in [16, {MAX_GRID_SIDE}]")
        dimensions.append(dimension)
    rows, columns = dimensions
    if rows * columns > MAX_GRID_PIXELS:
        raise ValueError(
            f"grid is too large: at most {MAX_GRID_PIXELS:,} pixels are supported"
        )
    return rows, columns


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Return the counter-clockwise hull of a small set of 2D points."""

    unique = sorted({(float(row[0]), float(row[1])) for row in points})
    if len(unique) < 3:
        raise ValueError("a projected slab polygon must have nonzero area")

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        raise ValueError("a projected slab polygon must have nonzero area")
    return hull


def _inside_convex_polygon(
    x_grid: np.ndarray, y_grid: np.ndarray, hull: np.ndarray, tolerance: float
) -> np.ndarray:
    inside = np.ones(x_grid.shape, dtype=bool)
    for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        edge_x = end[0] - start[0]
        edge_y = end[1] - start[1]
        cross = edge_x * (y_grid - start[1]) - edge_y * (x_grid - start[0])
        inside &= cross >= -tolerance
    return inside


def _ray_thickness(
    x_relative: np.ndarray,
    y_relative: np.ndarray,
    size: np.ndarray,
    orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect z-directed rays with an oriented finite rectangular slab."""

    lower = np.full(x_relative.shape, -np.inf, dtype=float)
    upper = np.full(x_relative.shape, np.inf, dtype=float)
    valid = np.ones(x_relative.shape, dtype=bool)
    half_size = 0.5 * size
    tolerance = (
        np.finfo(float).eps
        * max(float(np.max(size)), np.finfo(float).tiny)
        * 32.0
    )
    for local_axis in range(3):
        basis = orientation[:, local_axis]
        beam_coefficient = float(basis[2])
        in_plane = basis[0] * x_relative + basis[1] * y_relative
        if abs(beam_coefficient) <= 1e-12:
            valid &= np.abs(in_plane) <= half_size[local_axis] + tolerance
            continue
        first = (-half_size[local_axis] - in_plane) / beam_coefficient
        second = (half_size[local_axis] - in_plane) / beam_coefficient
        lower = np.maximum(lower, np.minimum(first, second))
        upper = np.minimum(upper, np.maximum(first, second))
    thickness = np.maximum(upper - lower, 0.0)
    thickness[~valid] = 0.0
    return thickness, valid


def _scene_arrays(scene: LamellarScene) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(scene, LamellarScene):
        raise TypeError("scene must be a LamellarScene")
    centers = np.asarray(scene.centers, dtype=float)
    sizes = np.asarray(scene.sizes, dtype=float)
    orientations = np.asarray(scene.orientations, dtype=float)
    vertices = np.asarray(scene.vertices, dtype=float)
    count = len(centers)
    if count == 0:
        raise ValueError("the current lamellar scene contains no finite slabs")
    if (
        centers.shape != (count, 3)
        or sizes.shape != (count, 3)
        or orientations.shape != (count, 3, 3)
        or vertices.shape != (count, 8, 3)
    ):
        raise ValueError("scene centers, sizes, orientations, and vertices are inconsistent")
    if not all(
        np.all(np.isfinite(array))
        for array in (centers, sizes, orientations, vertices)
    ):
        raise ValueError("scene geometry must be finite")
    if np.any(sizes <= 0.0):
        raise ValueError("all scene slab dimensions must be positive")
    identity = np.eye(3)
    if not np.allclose(
        np.swapaxes(orientations, 1, 2) @ orientations,
        identity,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("scene orientations must be orthonormal")
    expected = (
        (_CORNER_SIGNS[None, :, :] * (0.5 * sizes[:, None, :]))
        @ np.swapaxes(orientations, 1, 2)
        + centers[:, None, :]
    )
    expected = expected.reshape(count, 8, 3)
    expected_order = np.lexsort(
        (expected[:, :, 2], expected[:, :, 1], expected[:, :, 0]), axis=1
    )
    vertices_order = np.lexsort(
        (vertices[:, :, 2], vertices[:, :, 1], vertices[:, :, 0]), axis=1
    )
    expected_sorted = np.take_along_axis(expected, expected_order[:, :, None], axis=1)
    vertices_sorted = np.take_along_axis(vertices, vertices_order[:, :, None], axis=1)
    scale = max(
        float(np.max(np.abs(vertices))),
        float(np.max(sizes)),
        np.finfo(float).tiny,
    )
    if not np.allclose(
        expected_sorted, vertices_sorted, rtol=1e-6, atol=scale * 1e-7
    ):
        raise ValueError("scene vertices do not match centers, sizes, and orientations")
    return centers, sizes, orientations, vertices


def simulate_projected_density_fft(
    scene: LamellarScene,
    *,
    shape: Sequence[int] = DEFAULT_GRID_SHAPE,
    margin_fraction: float = 0.1,
) -> LamellarScatteringResult:
    """Rasterize finite slabs as projected density, then compute 2D FFT power.

    The projection integrates a uniform unit model electron-density contrast
    along scene +z. Overlapping slabs add their path lengths. The field is
    centered by subtracting its mean before the transform. Grid dimensions are
    ``(rows, columns)`` and all coordinates retain the scene's existing length
    unit; relative scenes therefore produce relative inverse-length q axes.
    """

    rows, columns = _validate_grid_shape(shape)
    try:
        margin = float(margin_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("margin_fraction must be finite in [0, 1]") from exc
    if not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
        raise ValueError("margin_fraction must be finite in [0, 1]")
    centers, sizes, orientations, vertices = _scene_arrays(scene)

    projected_vertices = vertices[:, :, :2].reshape(-1, 2)
    object_min = np.min(projected_vertices, axis=0)
    object_max = np.max(projected_vertices, axis=0)
    object_span = object_max - object_min
    if not np.all(np.isfinite(object_span)) or np.any(object_span <= 0.0):
        raise ValueError("projected scene bounds must have positive finite width and height")
    padding = object_span * margin
    canvas_min = object_min - padding
    canvas_max = object_max + padding
    extent = canvas_max - canvas_min
    dx = float(extent[0] / columns)
    dy = float(extent[1] / rows)
    if not all(math.isfinite(value) and value > 0.0 for value in (dx, dy)):
        raise ValueError("grid pixel sizes must be finite and positive")
    x_axis = canvas_min[0] + (np.arange(columns, dtype=float) + 0.5) * dx
    y_axis = canvas_min[1] + (np.arange(rows, dtype=float) + 0.5) * dy
    density = np.zeros((rows, columns), dtype=float)
    candidate_pixels = 0

    for center, size, orientation, box_vertices in zip(
        centers, sizes, orientations, vertices, strict=True
    ):
        hull = _convex_hull_xy(box_vertices[:, :2])
        col_start = max(0, int(np.searchsorted(x_axis, float(np.min(hull[:, 0])), side="left")))
        col_stop = min(columns, int(np.searchsorted(x_axis, float(np.max(hull[:, 0])), side="right")))
        row_start = max(0, int(np.searchsorted(y_axis, float(np.min(hull[:, 1])), side="left")))
        row_stop = min(rows, int(np.searchsorted(y_axis, float(np.max(hull[:, 1])), side="right")))
        if col_start >= col_stop or row_start >= row_stop:
            continue
        candidate_pixels += (row_stop - row_start) * (col_stop - col_start)
        if candidate_pixels > MAX_RASTER_CANDIDATE_PIXELS:
            raise ValueError(
                "the scene is too large for the selected raster grid; reduce slab counts "
                "or choose a smaller grid, then review the resolution diagnostics"
            )
        x_grid, y_grid = np.meshgrid(
            x_axis[col_start:col_stop], y_axis[row_start:row_stop]
        )
        hull_scale = max(
            float(np.max(np.ptp(hull, axis=0))), np.finfo(float).tiny
        )
        inside = _inside_convex_polygon(
            x_grid,
            y_grid,
            hull,
            tolerance=np.finfo(float).eps * hull_scale**2 * 64.0,
        )
        if not np.any(inside):
            continue
        thickness, intersects = _ray_thickness(
            x_grid - center[0],
            y_grid - center[1],
            size,
            orientation,
        )
        thickness[~inside | ~intersects] = 0.0
        density[row_start:row_stop, col_start:col_stop] += thickness

    if not np.all(np.isfinite(density)):
        raise ValueError("projected density contains non-finite values")
    if not np.any(density > 0.0):
        raise ValueError(
            "the raster grid missed every slab; increase the grid dimensions or "
            "reduce the scene extent"
        )

    centered_density = density - float(np.mean(density))
    transform = np.fft.fftshift(np.fft.fft2(centered_density))
    intensity = np.asarray(np.abs(transform) ** 2, dtype=float)
    if not np.all(np.isfinite(intensity)) or float(np.max(intensity)) <= 0.0:
        raise ValueError("the projected scene produced an unusable FFT")
    qx = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(columns, d=dx))
    qy = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(rows, d=dy))
    period_samples = float(np.min(sizes[:, 2]) / min(dx, dy))
    warnings: list[str] = []
    if period_samples < 2.0:
        warnings.append(
            "the thinnest slab dimension spans fewer than two raster pixels; "
            "increase grid size or reduce the modeled scene extent"
        )
    if margin == 0.0:
        warnings.append(
            "no vacuum margin is present; the periodic FFT boundary may create edge effects"
        )
    if candidate_pixels > 0.75 * MAX_RASTER_CANDIDATE_PIXELS:
        warnings.append("raster work is close to the configured computational bound")
    diagnostics = {
        "grid_shape_rows_columns": [rows, columns],
        "grid_pixel_count": int(rows * columns),
        "raster_candidate_pixel_count": int(candidate_pixels),
        "pixel_size_xy": [dx, dy],
        "q_resolution_xy": [float(2.0 * np.pi / extent[0]), float(2.0 * np.pi / extent[1])],
        "nyquist_q_xy": [float(np.pi / dx), float(np.pi / dy)],
        "thinnest_slab_dimension_pixels": period_samples,
        "margin_fraction_per_side": margin,
        "mean_projected_density_before_centering": float(np.mean(density)),
        "maximum_projected_density": float(np.max(density)),
        "warnings": warnings,
    }
    assumptions = (
        "Each LamellarScene slab is a uniform finite rectangular prism; its projected density is the exact z-directed ray-intersection length at raster-pixel centers.",
        "The electron-density contrast is fixed to one model unit per scene-length-unit cubed; it is not a measured or calibrated material contrast.",
        "Overlapping slabs add their line-integrated projected density, and the mean is subtracted over the full rectangular canvas before the FFT.",
        "The canvas follows the projected scene bounds with the requested zero-density margin; the FFT treats the raster as periodically repeated.",
        "Intensity is the unnormalized squared magnitude of a 2D FFT. No form factors, instrument resolution, coherence, background, noise, or detector calibration are applied.",
        "This conditional 2D projection is illustrative. It does not uniquely reconstruct 3D morphology or validate measured SAXS intensity.",
    )
    return LamellarScatteringResult(
        projected_density=density,
        qx=qx,
        qy=qy,
        intensity=intensity,
        bounds_xy=np.vstack((canvas_min, canvas_max)),
        pixel_size=(dx, dy),
        length_unit=scene.length_unit,
        diagnostics=diagnostics,
        assumptions=assumptions,
        scene_status=scene.status,
        scene_flags=tuple(scene.metadata.get("flags", ())),
    )


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata contains a non-finite value")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"metadata value {type(value).__name__} is not JSON-compatible")


def render_lamellar_scattering(
    result: LamellarScatteringResult,
    *,
    figure: Any | None = None,
    language: str = "en",
) -> Any:
    """Render the paired real-space and reciprocal-space panels."""

    from matplotlib.figure import Figure
    from matplotlib.colors import LogNorm

    if figure is None:
        figure = Figure(figsize=(12.0, 5.1), constrained_layout=True, facecolor="#fbfcfd")
    else:
        figure.clear()
        figure.set_layout_engine("constrained")
        figure.set_facecolor("#fbfcfd")
    density_axis, intensity_axis = figure.subplots(1, 2)
    is_chinese = str(language).lower().startswith("zh")
    font = font_properties(size=None, language="zh") if is_chinese else None
    font_kwargs = {"fontproperties": font} if font is not None else {}
    for axis in (density_axis, intensity_axis):
        axis.set_facecolor("#f1f3f5")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors="#53616b", labelsize=9)
        axis.xaxis.label.set_color("#30424d")
        axis.yaxis.label.set_color("#30424d")
    if is_chinese:
        x_label = "x（nm）" if result.length_unit == "nm" else "x（相对长度）"
        y_label = "y（nm）" if result.length_unit == "nm" else "y（相对长度）"
        density_label = (
            "模型电子密度 × nm"
            if result.length_unit == "nm"
            else "模型密度 × 相对长度"
        )
        inverse_unit = "nm⁻¹" if result.length_unit == "nm" else "相对长度⁻¹"
    else:
        x_label = "x (nm)" if result.length_unit == "nm" else "x (relative length)"
        y_label = "y (nm)" if result.length_unit == "nm" else "y (relative length)"
        density_label = result.density_unit
        inverse_unit = "nm⁻¹" if result.length_unit == "nm" else "relative⁻¹"
    density_image = density_axis.imshow(
        result.projected_density,
        origin="lower",
        extent=(
            result.bounds_xy[0, 0],
            result.bounds_xy[1, 0],
            result.bounds_xy[0, 1],
            result.bounds_xy[1, 1],
        ),
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    density_axis.set_title(
        "投影密度" if is_chinese else "Projected density",
        loc="left",
        fontsize=13,
        color="#243746",
        pad=10,
        **font_kwargs,
    )
    density_axis.set_xlabel(x_label, **font_kwargs)
    density_axis.set_ylabel(y_label, **font_kwargs)
    density_colorbar = figure.colorbar(density_image, ax=density_axis, fraction=0.046, pad=0.035)
    density_colorbar.set_label(density_label, color="#30424d", **font_kwargs)
    normalized = result.normalized_intensity
    positive = normalized[normalized > 0.0]
    floor = max(float(np.max(positive)) * 1e-8, np.finfo(float).tiny)
    qx_edges = _centers_to_edges(result.qx)
    qy_edges = _centers_to_edges(result.qy)
    intensity_image = intensity_axis.imshow(
        normalized,
        origin="lower",
        extent=(qx_edges[0], qx_edges[-1], qy_edges[0], qy_edges[-1]),
        cmap="cividis",
        norm=LogNorm(vmin=floor, vmax=1.0),
        interpolation="nearest",
        aspect="equal",
    )
    intensity_axis.set_title(
        "二维 FFT 功率" if is_chinese else "2D FFT power",
        loc="left",
        fontsize=13,
        color="#243746",
        pad=10,
        **font_kwargs,
    )
    intensity_axis.set_xlabel(
        f"q_x（{inverse_unit.replace('⁻¹', '^-1')}）"
        if is_chinese
        else f"q_x ({inverse_unit.replace('⁻¹', '^-1')})",
        **font_kwargs,
    )
    intensity_axis.set_ylabel(
        f"q_y（{inverse_unit.replace('⁻¹', '^-1')}）"
        if is_chinese
        else f"q_y ({inverse_unit.replace('⁻¹', '^-1')})",
        **font_kwargs,
    )
    intensity_colorbar = figure.colorbar(intensity_image, ax=intensity_axis, fraction=0.046, pad=0.035)
    intensity_colorbar.set_label(
        "对数色标，I / max(I)" if is_chinese else "log scale, I / max(I)",
        color="#30424d",
        **font_kwargs,
    )
    warning_count = len(result.diagnostics.get("warnings", ()))
    qualifier = (
        " · 分辨率提示" if is_chinese and warning_count else " · resolution warning" if warning_count else ""
    )
    figure.suptitle(
        (
            f"片层投影 FFT 示意 · Δρ = 1 模型单位{qualifier}"
            if is_chinese
            else f"Illustrative lamellar projection FFT · Δρ = 1 model unit{qualifier}"
        ),
        fontsize=14,
        color="#213440",
        fontweight="semibold",
        **font_kwargs,
    )

    if font is not None:
        for axis in (density_axis, intensity_axis):
            for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
                tick.set_fontproperties(font)
        for colorbar in (density_colorbar, intensity_colorbar):
            for tick in (*colorbar.ax.get_xticklabels(), *colorbar.ax.get_yticklabels()):
                tick.set_fontproperties(font)

    return figure


def _plot_path(result: LamellarScatteringResult, path: Path, *, language: str) -> None:
    figure = render_lamellar_scattering(result, language=language)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("q axes need at least two regularly spaced values")
    half_step = 0.5 * float(values[1] - values[0])
    return np.concatenate(([values[0] - half_step], values + half_step))


def export_lamellar_scattering(
    result: LamellarScatteringResult,
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    language: str = "en",
) -> tuple[Path, Path, Path]:
    """Write compressed arrays, strict-JSON assumptions, and a paired plot.

    ``output_path`` is the NPZ filename or a stem. The companion JSON and PNG
    share that stem. Existing files are preserved unless ``overwrite=True``.
    """

    target = Path(output_path).expanduser()
    if target.suffix.lower() == ".npz":
        stem = target.with_suffix("")
        npz_path = target
    else:
        stem = target
        npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")
    png_path = stem.with_suffix(".png")
    targets = (npz_path, json_path, png_path)
    if not stem.parent.exists():
        raise FileNotFoundError(f"output directory does not exist: {stem.parent}")
    if not stem.parent.is_dir():
        raise NotADirectoryError(stem.parent)
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing scattering output: "
                + ", ".join(str(path) for path in existing)
            )
    metadata = result.metadata()
    json_text = json.dumps(
        metadata, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    )
    with tempfile.TemporaryDirectory(prefix=".lamellar-scattering-", dir=stem.parent) as temp_name:
        temp_dir = Path(temp_name)
        temp_npz = temp_dir / npz_path.name
        temp_json = temp_dir / json_path.name
        temp_png = temp_dir / png_path.name
        np.savez_compressed(
            temp_npz,
            projected_density=result.projected_density,
            qx=result.qx,
            qy=result.qy,
            intensity=result.intensity,
            intensity_normalized=result.normalized_intensity,
            metadata_json=np.asarray(json_text),
        )
        temp_json.write_text(json_text + "\n", encoding="utf-8")
        _plot_path(result, temp_png, language=language)
        published: list[Path] = []
        try:
            for source, destination in zip(
                (temp_npz, temp_json, temp_png), targets, strict=True
            ):
                if overwrite:
                    os.replace(source, destination)
                else:
                    # A same-filesystem hard link publishes the complete file
                    # atomically and fails if another writer created the target.
                    os.link(source, destination)
                    published.append(destination)
        except Exception:
            if not overwrite:
                for destination in published:
                    try:
                        destination.unlink()
                    except OSError:
                        pass
            raise
    return targets


__all__ = (
    "DEFAULT_GRID_SHAPE",
    "LamellarScatteringResult",
    "export_lamellar_scattering",
    "render_lamellar_scattering",
    "simulate_projected_density_fft",
)
