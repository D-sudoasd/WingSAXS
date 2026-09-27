"""Manual point and line measurements on observed two-dimensional SAXS data.

The functions here take detector arrays and the q-coordinate maps already
resolved by WingSAXS.  They do not load or synthesize a one-dimensional input.
Pixel samples excluded by the supplied detector-valid mask remain missing in
line profiles and are never interpolated across for an FWHM estimate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .settings import canonical_q_unit


_PHYSICAL_LENGTH_UNITS = {"nm⁻¹": "nm", "Å⁻¹": "Å"}


def _array2d(value: Any, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric two-dimensional array") from exc
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape!r} must match image shape {shape!r}")
    return array


def _canonical_unit(value: Any) -> str:
    unit = canonical_q_unit(value)
    return unit if unit in {"nm⁻¹", "Å⁻¹", "pixel-q"} else str(value or "unknown")


def spacing_from_q(qx: float, qy: float, q_unit: str) -> tuple[float | None, str | None]:
    """Return 2π/|q| only when the q map declares a physical reciprocal unit."""

    unit = _canonical_unit(q_unit)
    length_unit = _PHYSICAL_LENGTH_UNITS.get(unit)
    q = math.hypot(float(qx), float(qy))
    if length_unit is None or not math.isfinite(q) or q <= 0.0:
        return None, None
    return 2.0 * math.pi / q, length_unit


def _usable_arrays(
    data: Any,
    qx: Any,
    qy: Any,
    valid_mask: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = _array2d(data, "data")
    shape = observed.shape
    qx_values = _array2d(qx, "qx", shape)
    qy_values = _array2d(qy, "qy", shape)
    if valid_mask is None:
        usable = np.ones(shape, dtype=bool)
    else:
        raw_mask = np.asarray(valid_mask)
        if raw_mask.shape != shape:
            raise ValueError(f"valid_mask shape {raw_mask.shape!r} must match image shape {shape!r}")
        usable = raw_mask.astype(bool, copy=False)
    usable = usable & np.isfinite(observed) & np.isfinite(qx_values) & np.isfinite(qy_values)
    return observed, qx_values, qy_values, usable


def _pixel(row: Any, col: Any, shape: tuple[int, int]) -> tuple[int, int]:
    try:
        y, x = float(row), float(col)
    except (TypeError, ValueError) as exc:
        raise ValueError("pixel row and column must be finite numbers") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("pixel row and column must be finite numbers")
    iy, ix = int(round(y)), int(round(x))
    if not (0 <= iy < shape[0] and 0 <= ix < shape[1]):
        raise ValueError(f"selected pixel ({ix}, {iy}) is outside image bounds {shape[1]}×{shape[0]}")
    return iy, ix


@dataclass(frozen=True)
class PointMeasurement:
    """An observed detector pixel and optional valid-neighbourhood summary."""

    row: int
    col: int
    intensity: float
    qx: float
    qy: float
    q: float
    q_unit: str
    spacing: float | None
    spacing_unit: str | None
    patch_radius: int = 0
    patch_count: int = 1
    patch_mean: float | None = None
    patch_median: float | None = None
    patch_std: float | None = None
    patch_q_median: float | None = None
    source: str | None = None
    kind: str = field(default="point", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "row": self.row,
            "col": self.col,
            "intensity": self.intensity,
            "qx": self.qx,
            "qy": self.qy,
            "q": self.q,
            "q_unit": self.q_unit,
            "spacing": self.spacing,
            "spacing_unit": self.spacing_unit,
            "patch_radius": self.patch_radius,
            "patch_count": self.patch_count,
            "patch_mean": self.patch_mean,
            "patch_median": self.patch_median,
            "patch_std": self.patch_std,
            "patch_q_median": self.patch_q_median,
            "source": self.source,
        }


def measure_point(
    data: Any,
    qx: Any,
    qy: Any,
    row: Any,
    col: Any,
    *,
    valid_mask: Any = None,
    q_unit: str = "unknown",
    patch_radius: int = 1,
    source: str | None = None,
) -> PointMeasurement:
    """Measure one observed pixel and robust statistics in a valid local patch.

    ``valid_mask`` uses ``True`` for usable pixels.  The clicked pixel itself
    must be valid; masked, non-finite and out-of-range picks fail explicitly.
    """

    observed, qx_values, qy_values, usable = _usable_arrays(data, qx, qy, valid_mask)
    iy, ix = _pixel(row, col, observed.shape)
    if not usable[iy, ix]:
        raise ValueError("selected pixel is masked or has non-finite intensity/q coordinates")
    try:
        radius = int(patch_radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("patch_radius must be a non-negative integer") from exc
    if radius < 0 or radius != patch_radius:
        raise ValueError("patch_radius must be a non-negative integer")

    r0, r1 = max(0, iy - radius), min(observed.shape[0], iy + radius + 1)
    c0, c1 = max(0, ix - radius), min(observed.shape[1], ix + radius + 1)
    patch_valid = usable[r0:r1, c0:c1]
    patch_intensity = observed[r0:r1, c0:c1][patch_valid]
    patch_q = np.hypot(qx_values[r0:r1, c0:c1][patch_valid], qy_values[r0:r1, c0:c1][patch_valid])
    qx_value, qy_value = float(qx_values[iy, ix]), float(qy_values[iy, ix])
    q_value = math.hypot(qx_value, qy_value)
    spacing, spacing_unit = spacing_from_q(qx_value, qy_value, q_unit)
    return PointMeasurement(
        row=iy,
        col=ix,
        intensity=float(observed[iy, ix]),
        qx=qx_value,
        qy=qy_value,
        q=q_value,
        q_unit=_canonical_unit(q_unit),
        spacing=spacing,
        spacing_unit=spacing_unit,
        patch_radius=radius,
        patch_count=int(patch_intensity.size),
        patch_mean=float(np.mean(patch_intensity)),
        patch_median=float(np.median(patch_intensity)),
        patch_std=float(np.std(patch_intensity, ddof=1)) if patch_intensity.size > 1 else 0.0,
        patch_q_median=float(np.median(patch_q)),
        source=str(source) if source is not None else None,
    )


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return integer detector pixels on an inclusive rasterized line."""

    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    xs: list[int] = []
    ys: list[int] = []
    while True:
        xs.append(x0)
        ys.append(y0)
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x0 += sx
        if twice < dx:
            error += dx
            y0 += sy
    return np.asarray(ys, dtype=int), np.asarray(xs, dtype=int)


@dataclass(frozen=True)
class FWHMEstimate:
    """Half-height width of the strongest peak within observed contiguous data."""

    value: float | None
    x_left: float | None
    x_right: float | None
    x_peak: float | None
    y_peak: float | None
    baseline: float | None
    half_height: float | None
    x_unit: str
    status: str
    reason: str | None = None
    n_segment: int = 0
    flags: tuple[str, ...] = ()
    length_proxy: float | None = None
    length_unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "x_left": self.x_left,
            "x_right": self.x_right,
            "x_peak": self.x_peak,
            "y_peak": self.y_peak,
            "baseline": self.baseline,
            "half_height": self.half_height,
            "x_unit": self.x_unit,
            "status": self.status,
            "reason": self.reason,
            "n_segment": self.n_segment,
            "flags": list(self.flags),
            "length_proxy": self.length_proxy,
            "length_unit": self.length_unit,
            "method": "linear half-height crossings in one contiguous observed segment",
        }


def estimate_fwhm(
    x: Any,
    intensity: Any,
    *,
    valid: Any = None,
    x_unit: str = "unknown",
) -> FWHMEstimate:
    """Estimate FWHM without smoothing, filling gaps or crossing invalid data.

    The strongest sample selects the peak.  Its baseline is the fifth
    percentile of that same contiguous valid segment; both half-height
    crossings must be observed on the segment.  The result is a profile
    descriptor.  A 2π/FWHM length proxy is reported only for physical q units.
    """

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(intensity, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.shape != y_values.shape:
        raise ValueError("x and intensity must be one-dimensional arrays with matching shapes")
    if valid is None:
        valid_values = np.ones(x_values.shape, dtype=bool)
    else:
        valid_values = np.asarray(valid, dtype=bool)
        if valid_values.shape != x_values.shape:
            raise ValueError("valid must have the same shape as x and intensity")
    finite = valid_values & np.isfinite(x_values) & np.isfinite(y_values)
    starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
    ends = np.flatnonzero(finite & ~np.r_[finite[1:], False]) + 1
    segments = [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]
    unit = _canonical_unit(x_unit)
    if not segments:
        return FWHMEstimate(None, None, None, None, None, None, None, unit, "unavailable", "no_valid_samples")

    peak_segment = max(segments, key=lambda bounds: float(np.max(y_values[bounds[0]:bounds[1]])))
    begin, finish = peak_segment
    xs = x_values[begin:finish]
    ys = y_values[begin:finish]
    n = int(xs.size)
    i_peak = int(np.argmax(ys))
    peak_x, peak_y = float(xs[i_peak]), float(ys[i_peak])
    baseline = float(np.percentile(ys, 5.0))
    half = baseline + 0.5 * (peak_y - baseline)
    base_fields = (peak_x, peak_y, baseline, half, n)

    if n < 3:
        return FWHMEstimate(None, None, None, *base_fields[:4], unit, "unavailable", "insufficient_contiguous_support", n)
    differences = np.diff(xs)
    if np.any(~np.isfinite(differences)) or np.any(differences <= 0.0):
        return FWHMEstimate(None, None, None, *base_fields[:4], unit, "unavailable", "profile_axis_not_strictly_increasing", n)
    if not math.isfinite(baseline) or peak_y <= baseline:
        return FWHMEstimate(None, None, None, peak_x, peak_y, baseline, half, unit, "unavailable", "no_positive_peak", n)

    left_cross = None
    for index in range(i_peak - 1, -1, -1):
        if ys[index] < half <= ys[index + 1]:
            left_cross = _interpolate_crossing(xs[index], xs[index + 1], ys[index], ys[index + 1], half)
            break
    right_cross = None
    for index in range(i_peak, n - 1):
        if ys[index] >= half > ys[index + 1]:
            right_cross = _interpolate_crossing(xs[index], xs[index + 1], ys[index], ys[index + 1], half)
            break
    if left_cross is None or right_cross is None:
        return FWHMEstimate(None, left_cross, right_cross, peak_x, peak_y, baseline, half, unit,
                            "unavailable", "half_height_not_crossed_on_both_sides", n)
    width = float(right_cross - left_cross)
    if not math.isfinite(width) or width <= 0.0:
        return FWHMEstimate(None, left_cross, right_cross, peak_x, peak_y, baseline, half, unit,
                            "unavailable", "nonpositive_width", n)

    flags = []
    status = "estimated"
    if n < 7:
        status = "low_support"
        flags.append("fewer_than_7_contiguous_samples")
    length_unit = _PHYSICAL_LENGTH_UNITS.get(unit)
    length_proxy = 2.0 * math.pi / width if length_unit and width > 0.0 else None
    return FWHMEstimate(width, float(left_cross), float(right_cross), peak_x, peak_y, baseline,
                        half, unit, status, None, n, tuple(flags), length_proxy, length_unit)


def _interpolate_crossing(x0: float, x1: float, y0: float, y1: float, target: float) -> float:
    if y1 == y0:
        return float(x0)
    fraction = min(1.0, max(0.0, (target - y0) / (y1 - y0)))
    return float(x0 + fraction * (x1 - x0))


@dataclass(frozen=True)
class LineProfile:
    """Observed samples along a detector-pixel line, including invalid gaps."""

    rows: np.ndarray
    cols: np.ndarray
    pixel_distance: np.ndarray
    q_distance: np.ndarray
    intensity: np.ndarray
    valid: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    axis: np.ndarray
    axis_unit: str
    q_unit: str
    fwhm: FWHMEstimate
    source: str | None = None
    kind: str = field(default="line", init=False)

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": self.kind,
            "q_unit": self.q_unit,
            "axis_unit": self.axis_unit,
            "fwhm": self.fwhm.to_dict(),
            "source": self.source,
        }
        if include_samples:
            record["samples"] = [
                {
                    "row": int(row),
                    "col": int(col),
                    "pixel_distance": float(pixel_distance),
                    "q_distance": _finite_or_none(q_distance),
                    "axis": _finite_or_none(axis),
                    "intensity": _finite_or_none(value),
                    "valid": bool(valid),
                    "qx": _finite_or_none(qx),
                    "qy": _finite_or_none(qy),
                }
                for row, col, pixel_distance, q_distance, axis, value, valid, qx, qy in zip(
                    self.rows, self.cols, self.pixel_distance, self.q_distance,
                    self.axis, self.intensity, self.valid, self.qx, self.qy,
                )
            ]
        return record


def extract_line_profile(
    data: Any,
    qx: Any,
    qy: Any,
    start: tuple[Any, Any],
    end: tuple[Any, Any],
    *,
    valid_mask: Any = None,
    q_unit: str = "unknown",
    source: str | None = None,
) -> LineProfile:
    """Extract a Bresenham path of detector samples without interpolating gaps.

    Pixel coordinates are supplied as ``(column, row)``.  When calibrated q
    coordinates are finite at every sampled detector pixel, the profile axis
    is cumulative path length in that q unit.  Otherwise the axis is detector
    path length in pixels.  Invalid observed pixels keep a ``NaN`` intensity.
    """

    observed, qx_values, qy_values, usable = _usable_arrays(data, qx, qy, valid_mask)
    if len(start) != 2 or len(end) != 2:
        raise ValueError("start and end must each be (column, row) pairs")
    start_row, start_col = _pixel(start[1], start[0], observed.shape)
    end_row, end_col = _pixel(end[1], end[0], observed.shape)
    rows, cols = _bresenham((start_col, start_row), (end_col, end_row))
    pixel_steps = np.hypot(np.diff(cols), np.diff(rows))
    pixel_distance = np.r_[0.0, np.cumsum(pixel_steps)]
    intensities = observed[rows, cols].astype(float, copy=True)
    valid = usable[rows, cols].copy()
    intensities[~valid] = np.nan
    sample_qx = qx_values[rows, cols].astype(float, copy=True)
    sample_qy = qy_values[rows, cols].astype(float, copy=True)
    q_step = np.hypot(np.diff(sample_qx), np.diff(sample_qy))
    q_distance = np.r_[0.0, np.cumsum(q_step)]
    canonical_unit = _canonical_unit(q_unit)
    if (
        canonical_unit in {"nm⁻¹", "Å⁻¹", "pixel-q"}
        and np.all(np.isfinite(q_distance))
        and np.all(np.diff(q_distance) > 0.0)
    ):
        axis, axis_unit = q_distance.copy(), canonical_unit
    else:
        axis, axis_unit = pixel_distance.copy(), "pixel"
    fwhm = estimate_fwhm(axis, intensities, valid=valid, x_unit=axis_unit)
    return LineProfile(
        rows=rows,
        cols=cols,
        pixel_distance=pixel_distance,
        q_distance=q_distance,
        intensity=intensities,
        valid=valid,
        qx=sample_qx,
        qy=sample_qy,
        axis=axis,
        axis_unit=axis_unit,
        q_unit=canonical_unit,
        fwhm=fwhm,
        source=str(source) if source is not None else None,
    )


@dataclass(frozen=True)
class ROIOrientation:
    """Intensity-weighted principal axis of an observed q-space stripe ROI."""

    start: tuple[int, int]
    end: tuple[int, int]
    half_width_pixels: float
    n_pixels: int
    q_unit: str
    orientation_deg: float | None
    center_qx: float | None
    center_qy: float | None
    principal_vector: tuple[float, float] | None
    eigenvalues: tuple[float, float] | None
    eigenvalue_ratio: float | None
    status: str
    reason: str | None = None
    weighting: str = "non-negative observed intensity"
    interpretation: str = (
        "Observed in-plane intensity axis in q space; it does not identify a unique lamellar tilt or mechanism."
    )
    source: str | None = None
    kind: str = field(default="roi_orientation", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": list(self.start),
            "end": list(self.end),
            "half_width_pixels": self.half_width_pixels,
            "n_pixels": self.n_pixels,
            "q_unit": self.q_unit,
            "orientation_deg": self.orientation_deg,
            "center_qx": self.center_qx,
            "center_qy": self.center_qy,
            "principal_vector": list(self.principal_vector) if self.principal_vector else None,
            "eigenvalues": list(self.eigenvalues) if self.eigenvalues else None,
            "eigenvalue_ratio": self.eigenvalue_ratio,
            "status": self.status,
            "reason": self.reason,
            "weighting": self.weighting,
            "interpretation": self.interpretation,
            "source": self.source,
        }


def measure_roi_orientation(
    data: Any,
    qx: Any,
    qy: Any,
    start: tuple[Any, Any],
    end: tuple[Any, Any],
    *,
    valid_mask: Any = None,
    q_unit: str = "unknown",
    half_width_pixels: float = 3.0,
    source: str | None = None,
) -> ROIOrientation:
    """Estimate a descriptive q-space intensity axis inside a detector strip.

    ``start`` and ``end`` are detector ``(column, row)`` pairs.  The strip is
    formed from existing pixels whose projection lies between the endpoints;
    masked or non-finite pixels are omitted, never interpolated.  Covariance is
    computed from qx/qy with non-negative raw-intensity weights.  No background
    subtraction is assumed, so the result is explicitly an observed intensity
    axis and not a unique structural orientation.
    """

    observed, qx_values, qy_values, usable = _usable_arrays(data, qx, qy, valid_mask)
    if len(start) != 2 or len(end) != 2:
        raise ValueError("start and end must each be (column, row) pairs")
    sy, sx = _pixel(start[1], start[0], observed.shape)
    ey, ex = _pixel(end[1], end[0], observed.shape)
    try:
        half_width = float(half_width_pixels)
    except (TypeError, ValueError) as exc:
        raise ValueError("half_width_pixels must be finite and positive") from exc
    if not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("half_width_pixels must be finite and positive")
    vx, vy = float(ex - sx), float(ey - sy)
    length_squared = vx * vx + vy * vy
    canonical_unit = _canonical_unit(q_unit)
    if length_squared == 0.0:
        return ROIOrientation((sx, sy), (ex, ey), half_width, 0, canonical_unit,
                              None, None, None, None, None, None, "unavailable",
                              "zero_length_roi_line", source=str(source) if source is not None else None)

    margin = int(math.ceil(half_width))
    x0, x1 = max(0, min(sx, ex) - margin), min(observed.shape[1], max(sx, ex) + margin + 1)
    y0, y1 = max(0, min(sy, ey) - margin), min(observed.shape[0], max(sy, ey) + margin + 1)
    rows, cols = np.indices((y1 - y0, x1 - x0), dtype=float)
    rows += y0
    cols += x0
    projection = ((cols - sx) * vx + (rows - sy) * vy) / length_squared
    perpendicular = np.abs((cols - sx) * vy - (rows - sy) * vx) / math.sqrt(length_squared)
    selected_local = (
        usable[y0:y1, x0:x1]
        & (projection >= 0.0)
        & (projection <= 1.0)
        & (perpendicular <= half_width)
    )
    count = int(np.count_nonzero(selected_local))
    if count < 3:
        return ROIOrientation((sx, sy), (ex, ey), half_width, count, canonical_unit,
                              None, None, None, None, None, None, "unavailable",
                              "fewer_than_3_observed_pixels", source=str(source) if source is not None else None)

    coordinates = np.column_stack((qx_values[y0:y1, x0:x1][selected_local],
                                   qy_values[y0:y1, x0:x1][selected_local]))
    raw_intensity = observed[y0:y1, x0:x1][selected_local]
    weights = np.maximum(raw_intensity, 0.0)
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return ROIOrientation((sx, sy), (ex, ey), half_width, count, canonical_unit,
                              None, None, None, None, None, None, "unavailable",
                              "no_positive_intensity_weights", source=str(source) if source is not None else None)
    weights /= total_weight
    center = np.sum(coordinates * weights[:, None], axis=0)
    centered = coordinates - center
    covariance = (centered * weights[:, None]).T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return ROIOrientation((sx, sy), (ex, ey), half_width, count, canonical_unit,
                              None, float(center[0]), float(center[1]), None, None, None,
                              "unavailable", "covariance_decomposition_failed",
                              source=str(source) if source is not None else None)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vector = eigenvectors[:, int(order[0])]
    angle = (math.degrees(math.atan2(float(vector[1]), float(vector[0]))) + 90.0) % 180.0 - 90.0
    ratio = float(eigenvalues[0] / max(float(eigenvalues[1]), np.finfo(float).eps))
    status, reason = ("estimated", None) if ratio >= 2.0 else ("low_anisotropy", "principal_axis_is_weakly_defined")
    return ROIOrientation(
        (sx, sy), (ex, ey), half_width, count, canonical_unit, float(angle),
        float(center[0]), float(center[1]), (float(vector[0]), float(vector[1])),
        (float(eigenvalues[0]), float(eigenvalues[1])), ratio, status, reason,
        source=str(source) if source is not None else None,
    )


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plain_record(record: Any, *, include_samples: bool = True) -> dict[str, Any]:
    if isinstance(record, PointMeasurement):
        return record.to_dict()
    if isinstance(record, LineProfile):
        return record.to_dict(include_samples=include_samples)
    if isinstance(record, ROIOrientation):
        return record.to_dict()
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(f"unsupported local measurement record: {type(record).__name__}")


def measurement_document(records: Iterable[Any]) -> dict[str, Any]:
    """Return a strict-JSON-compatible, self-contained measurement document."""

    return {
        "schema": "butterfly-saxs/local-measurements-v1",
        "measurements": [_plain_record(record) for record in records],
    }


def export_local_measurements_json(
    path: str | Path,
    records: Iterable[Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write point records and complete line-profile samples as strict JSON."""

    target = Path(path)
    document = measurement_document(records)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing measurement export: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return target


_CSV_FIELDS = (
    "record_index", "row_type", "kind", "source", "row", "col", "q_unit", "q", "qx", "qy",
    "spacing", "spacing_unit", "intensity", "patch_radius", "patch_count",
    "patch_mean", "patch_median", "patch_std", "patch_q_median", "axis_unit",
    "sample_index", "pixel_distance", "q_distance", "axis", "valid", "fwhm",
    "fwhm_status", "fwhm_reason", "fwhm_length_proxy", "fwhm_length_unit",
    "start_col", "start_row", "end_col", "end_row", "half_width_pixels",
    "n_pixels", "orientation_deg", "center_qx", "center_qy", "principal_qx",
    "principal_qy", "eigenvalue_major", "eigenvalue_minor", "eigenvalue_ratio",
    "weighting", "interpretation", "status", "reason",
)


def export_local_measurements_csv(
    path: str | Path,
    records: Iterable[Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write point summaries and every line sample (masked gaps as blank cells)."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing measurement export: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for index, item in enumerate(records):
            record = _plain_record(item)
            common = {
                "record_index": index,
                "kind": record.get("kind", ""),
                "source": record.get("source", ""),
                "q_unit": record.get("q_unit", ""),
            }
            if record.get("kind") == "point":
                writer.writerow({
                    **common,
                    "row_type": "point",
                    "row": record.get("row"),
                    "col": record.get("col"),
                    "q": record.get("q"),
                    "qx": record.get("qx"),
                    "qy": record.get("qy"),
                    "spacing": record.get("spacing"),
                    "spacing_unit": record.get("spacing_unit"),
                    "intensity": record.get("intensity"),
                    "patch_radius": record.get("patch_radius"),
                    "patch_count": record.get("patch_count"),
                    "patch_mean": record.get("patch_mean"),
                    "patch_median": record.get("patch_median"),
                    "patch_std": record.get("patch_std"),
                    "patch_q_median": record.get("patch_q_median"),
                })
                continue

            if record.get("kind") == "roi_orientation":
                start = record.get("start", (None, None))
                end = record.get("end", (None, None))
                vector = record.get("principal_vector") or (None, None)
                eigenvalues = record.get("eigenvalues") or (None, None)
                writer.writerow({
                    **common,
                    "row_type": "roi_orientation",
                    "start_col": start[0] if len(start) > 0 else None,
                    "start_row": start[1] if len(start) > 1 else None,
                    "end_col": end[0] if len(end) > 0 else None,
                    "end_row": end[1] if len(end) > 1 else None,
                    "half_width_pixels": record.get("half_width_pixels"),
                    "n_pixels": record.get("n_pixels"),
                    "orientation_deg": record.get("orientation_deg"),
                    "center_qx": record.get("center_qx"),
                    "center_qy": record.get("center_qy"),
                    "principal_qx": vector[0] if len(vector) > 0 else None,
                    "principal_qy": vector[1] if len(vector) > 1 else None,
                    "eigenvalue_major": eigenvalues[0] if len(eigenvalues) > 0 else None,
                    "eigenvalue_minor": eigenvalues[1] if len(eigenvalues) > 1 else None,
                    "eigenvalue_ratio": record.get("eigenvalue_ratio"),
                    "status": record.get("status", ""),
                    "reason": record.get("reason", ""),
                    "weighting": record.get("weighting", ""),
                    "interpretation": record.get("interpretation", ""),
                })
                continue

            fwhm = record.get("fwhm", {})
            writer.writerow({
                **common,
                "row_type": "line_summary",
                "axis_unit": record.get("axis_unit", ""),
                "fwhm": fwhm.get("value"),
                "fwhm_status": fwhm.get("status", ""),
                "fwhm_reason": fwhm.get("reason", ""),
                "fwhm_length_proxy": fwhm.get("length_proxy"),
                "fwhm_length_unit": fwhm.get("length_unit", ""),
            })
            for sample_index, sample in enumerate(record.get("samples", [])):
                writer.writerow({
                    **common,
                    "row_type": "profile_sample",
                    "axis_unit": record.get("axis_unit", ""),
                    "sample_index": sample_index,
                    "row": sample.get("row"),
                    "col": sample.get("col"),
                    "pixel_distance": sample.get("pixel_distance"),
                    "q_distance": sample.get("q_distance"),
                    "axis": sample.get("axis"),
                    "intensity": sample.get("intensity"),
                    "qx": sample.get("qx"),
                    "qy": sample.get("qy"),
                    "valid": sample.get("valid"),
                    "fwhm": fwhm.get("value"),
                    "fwhm_status": fwhm.get("status", ""),
                    "fwhm_reason": fwhm.get("reason", ""),
                    "fwhm_length_proxy": fwhm.get("length_proxy"),
                    "fwhm_length_unit": fwhm.get("length_unit", ""),
                })
    return target


__all__ = [
    "FWHMEstimate",
    "LineProfile",
    "PointMeasurement",
    "ROIOrientation",
    "estimate_fwhm",
    "export_local_measurements_csv",
    "export_local_measurements_json",
    "extract_line_profile",
    "measure_point",
    "measure_roi_orientation",
    "measurement_document",
    "spacing_from_q",
]
