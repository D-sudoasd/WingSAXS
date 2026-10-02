"""Descriptive measurements from observed two-dimensional SAXS frames.

This module bins the supplied intensity image in reciprocal-space radius and
azimuth. It does not fit peaks, interpolate unsupported bins, or infer a
three-dimensional structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .local_measurement import spacing_from_q
from .settings import canonical_q_unit


_MAX_POLAR_CELLS = 1_000_000
_PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)


def _array2d(value: Any, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric two-dimensional array") from exc
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape!r} must match image shape {shape!r}")
    return array


def _bin_count(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric < 1 or numeric != int(numeric):
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _q_edges(q_values: np.ndarray, q_edges: Any, radial_bins: int) -> np.ndarray:
    if q_edges is not None:
        try:
            edges = np.asarray(q_edges, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("q_edges must be a one-dimensional numeric sequence") from exc
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("q_edges must contain at least two one-dimensional boundaries")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError("q_edges must be finite and strictly increasing")
        if edges[0] < 0.0:
            raise ValueError("q_edges must be non-negative because radial q is a magnitude")
        return edges.copy()

    q_min = float(np.min(q_values))
    q_max = float(np.max(q_values))
    if q_min == q_max:
        # A constant q map still has a finite, useful domain. Give it a small
        # interval so all requested bin edges remain distinct and q itself is
        # included (the final edge is closed).
        half_width = max(abs(q_min), 1.0) * 1.0e-9
        low = max(0.0, q_min - half_width)
        high = q_max + half_width
        if not np.isfinite(high):
            high = q_max
            low = max(0.0, q_min - 2.0 * half_width)
        if low == high:
            low = max(0.0, float(np.nextafter(q_min, -np.inf)))
            high = float(np.nextafter(q_max, np.inf))
        if not np.isfinite(high) or low == high:
            raise ValueError("could not construct finite q bins for the supplied q map")
        return np.linspace(low, high, radial_bins + 1, dtype=float)
    return np.linspace(q_min, q_max, radial_bins + 1, dtype=float)


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return indices and an in-range mask, with the final edge included."""

    n_bins = edges.size - 1
    inside = (values >= edges[0]) & (values <= edges[-1])
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = n_bins - 1
    return indices, inside


def _profile_stats(
    n_bins: int,
    observed_indices: np.ndarray,
    observed_values: np.ndarray,
    candidate_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Accumulate sample statistics without allocating per-bin pixel lists."""

    counts = np.bincount(observed_indices, minlength=n_bins).astype(np.int64, copy=False)
    candidate_counts = np.bincount(candidate_indices, minlength=n_bins).astype(np.int64, copy=False)
    sums = np.bincount(observed_indices, weights=observed_values, minlength=n_bins).astype(float, copy=False)
    means = np.full(n_bins, np.nan, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)

    stds = np.full(n_bins, np.nan, dtype=float)
    sems = np.full(n_bins, np.nan, dtype=float)
    repeated_means = means[observed_indices]
    squared_deviations = np.square(observed_values - repeated_means)
    sums_squared_deviation = np.bincount(
        observed_indices,
        weights=squared_deviations,
        minlength=n_bins,
    )
    has_sample_variance = counts > 1
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        stds[has_sample_variance] = np.sqrt(
            sums_squared_deviation[has_sample_variance] / (counts[has_sample_variance] - 1)
        )
        sems[has_sample_variance] = stds[has_sample_variance] / np.sqrt(counts[has_sample_variance])

    coverage = np.full(n_bins, np.nan, dtype=float)
    np.divide(counts, candidate_counts, out=coverage, where=candidate_counts > 0)
    # Empty bins have no measured sum. Keep them missing internally as well
    # as in the row representation so they cannot be mistaken for zero data.
    sums[counts == 0] = np.nan
    return {
        "mean": means,
        "sum": sums,
        "std": stds,
        "sem": sems,
        "count": counts,
        "candidate_count": candidate_counts,
        "coverage": coverage,
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _row_value(values: np.ndarray, index: int) -> float | int | None:
    if values.dtype.kind in "iu":
        return int(values[index])
    return _finite_number(values[index])


def _profile_rows(
    stats: dict[str, np.ndarray],
    lower_edges: np.ndarray,
    upper_edges: np.ndarray,
    *,
    radial: bool,
) -> list[dict[str, Any]]:
    centers = lower_edges + (upper_edges - lower_edges) / 2.0
    rows: list[dict[str, Any]] = []
    for index in range(lower_edges.size):
        if radial:
            row: dict[str, Any] = {
                "bin_index": index,
                "q_min": _finite_number(lower_edges[index]),
                "q_max": _finite_number(upper_edges[index]),
                "q_center": _finite_number(centers[index]),
            }
        else:
            row = {
                "bin_index": index,
                "angle_min_deg": _finite_number(lower_edges[index]),
                "angle_max_deg": _finite_number(upper_edges[index]),
                "angle_center_deg": _finite_number(centers[index]),
            }
        for key in ("mean", "sum", "std", "sem", "count", "candidate_count", "coverage"):
            row[key] = _row_value(stats[key], index)
        row["sem_kind"] = "pixel_sampling_descriptive_not_fit_uncertainty"
        rows.append(row)
    return rows


def _spacing_proxy(q_min: float | None, q_max: float | None, q_unit: str) -> dict[str, Any]:
    length_unit = {"nm⁻¹": "nm", "Å⁻¹": "Å"}.get(q_unit)
    if length_unit is None:
        return {
            "status": "unavailable",
            "reason": "q unit is not a declared physical reciprocal-length unit",
            "unit": None,
            "minimum": None,
            "maximum": None,
            "definition": "2π/q; reciprocal-space period proxy only",
        }
    if q_max is None or q_max <= 0.0:
        return {
            "status": "unavailable",
            "reason": "no positive observed q values",
            "unit": length_unit,
            "minimum": None,
            "maximum": None,
            "definition": "2π/q; reciprocal-space period proxy only",
        }
    minimum, _ = spacing_from_q(q_max, 0.0, q_unit)
    maximum = None
    if q_min is not None and q_min > 0.0:
        maximum, _ = spacing_from_q(q_min, 0.0, q_unit)
    return {
        "status": "estimated",
        "reason": None if maximum is not None else "observed q range includes zero",
        "unit": length_unit,
        "minimum": _finite_number(minimum),
        "maximum": _finite_number(maximum),
        "definition": "2π/q; reciprocal-space period proxy only, not a unique structural length",
    }


def _intensity_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        stats: dict[str, Any] = {
            "count": 0,
            "sum": None,
            "mean": None,
            "median": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "negative_count": 0,
        }
        stats.update({f"p{int(percentile):02d}": None for percentile in _PERCENTILES})
        return stats

    with np.errstate(invalid="ignore", over="ignore"):
        total = np.sum(values, dtype=float)
        mean = np.mean(values, dtype=float)
        median = np.median(values)
        std = np.std(values, ddof=1) if values.size > 1 else np.nan
        percentiles = np.percentile(values, _PERCENTILES)
    stats = {
        "count": int(values.size),
        "sum": _finite_number(total),
        "mean": _finite_number(mean),
        "median": _finite_number(median),
        "std": _finite_number(std),
        "minimum": _finite_number(np.min(values)),
        "maximum": _finite_number(np.max(values)),
        "negative_count": int(np.count_nonzero(values < 0.0)),
    }
    stats.update({f"p{int(percentile):02d}": _finite_number(value) for percentile, value in zip(_PERCENTILES, percentiles)})
    return stats


def _quadrants(
    qx: np.ndarray,
    qy: np.ndarray,
    image: np.ndarray,
    observed: np.ndarray,
    geometry: np.ndarray,
) -> dict[str, dict[str, Any]]:
    # Half-open assignments make quadrant counts disjoint and exhaustive:
    # qx == 0 goes to the right half, qy == 0 to the upper half.
    memberships = {
        "QI": (qx >= 0.0) & (qy >= 0.0),
        "QII": (qx < 0.0) & (qy >= 0.0),
        "QIII": (qx < 0.0) & (qy < 0.0),
        "QIV": (qx >= 0.0) & (qy < 0.0),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, membership in memberships.items():
        candidates = geometry & membership
        measured = observed & membership
        count = int(np.count_nonzero(measured))
        candidate_count = int(np.count_nonzero(candidates))
        selected = image[measured]
        total = np.sum(selected, dtype=float) if count else np.nan
        result[name] = {
            "sum": _finite_number(total),
            "count": count,
            "candidate_count": candidate_count,
            "coverage": count / candidate_count if candidate_count else None,
        }
    return result


def _weighted_tensor(qx: np.ndarray, qy: np.ndarray, image: np.ndarray) -> dict[str, Any]:
    weights = np.maximum(image, 0.0)
    weight_sum = float(np.sum(weights, dtype=float))
    base = {
        "weighting": "non-negative observed intensity: max(I, 0)",
        "tensor_definition": "intensity-weighted second moment E[q q^T] about the reciprocal-space origin",
        "anisotropy_definition": "(lambda_max - lambda_min) / (lambda_max + lambda_min)",
        "axis_convention": "principal eigenvector angle from +qx toward +qy, modulo 180 degrees",
        "interpretation": "descriptive in-plane reciprocal-space orientation; does not determine a unique 3D structure",
        "axis_reason": None,
    }
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return {
            **base,
            "status": "unavailable",
            "reason": "no positive non-negative-intensity weight",
            "axis_reason": "no_positive_intensity_weight",
            "tensor": None,
            "eigenvalues": None,
            "principal_axis_deg": None,
            "anisotropy": None,
        }

    q_scale = float(max(np.max(np.abs(qx)), np.max(np.abs(qy))))
    if not np.isfinite(q_scale) or q_scale <= 0.0:
        return {
            **base,
            "status": "unavailable",
            "reason": "observed reciprocal-space coordinates have zero scale",
            "axis_reason": "zero_reciprocal_space_scale",
            "tensor": None,
            "eigenvalues": None,
            "principal_axis_deg": None,
            "anisotropy": None,
        }

    # Scale q before products to avoid overflow. The tensor is then the
    # dimensionless moment E[(q/q_scale)(q/q_scale)^T]; eigenvectors and the
    # normalized anisotropy are unchanged by this common scale.
    sx = qx / q_scale
    sy = qy / q_scale
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        tensor = np.array(
            [
                [np.sum(weights * sx * sx) / weight_sum, np.sum(weights * sx * sy) / weight_sum],
                [np.sum(weights * sx * sy) / weight_sum, np.sum(weights * sy * sy) / weight_sum],
            ],
            dtype=float,
        )
    if not np.all(np.isfinite(tensor)):
        return {
            **base,
            "status": "unavailable",
            "reason": "weighted tensor is not finite",
            "axis_reason": "nonfinite_weighted_tensor",
            "tensor": None,
            "eigenvalues": None,
            "principal_axis_deg": None,
            "anisotropy": None,
        }
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    trace = float(np.sum(eigenvalues))
    if trace <= 0.0:
        return {
            **base,
            "status": "unavailable",
            "reason": "weighted reciprocal-space second moment is zero",
            "axis_reason": "zero_second_moment",
            "tensor": tensor.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "principal_axis_deg": None,
            "anisotropy": None,
        }
    principal = eigenvectors[:, -1]
    # Eigenvectors within floating-point eigensolver resolution are not a
    # defined orientation. This tolerance follows machine precision only; it
    # does not impose a scientific anisotropy threshold.
    eigenvalue_scale = float(np.max(np.abs(eigenvalues)))
    axis_undefined = (eigenvalues[-1] - eigenvalues[0]) <= np.finfo(float).eps * eigenvalue_scale
    anisotropy = 0.0 if axis_undefined else float((eigenvalues[-1] - eigenvalues[0]) / trace)
    angle = None
    if not axis_undefined:
        angle = float(np.degrees(np.arctan2(principal[1], principal[0])) % 180.0)
    return {
        **base,
        "status": "estimated",
        "reason": None,
        "q_scale": q_scale,
        "q_unit": "dimensionless after division by q_scale",
        "tensor": tensor.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "principal_axis_deg": angle,
        "axis_reason": "eigenvalue_degeneracy" if axis_undefined else None,
        "anisotropy": anisotropy,
    }


def measure_frame(
    image: Any,
    qx: Any,
    qy: Any,
    *,
    valid_mask: Any = None,
    q_unit: str = "pixel",
    q_edges: Any = None,
    radial_bins: int = 128,
    angular_bins: int = 72,
) -> dict[str, Any]:
    """Summarize observed frame intensity and its 2-D reciprocal-space layout.

    ``valid_mask`` is an inclusion mask: true pixels are eligible for
    observation statistics. Finite q-map pixels define the geometric
    candidate domain used for counts and coverage, including masked or
    non-finite-intensity pixels. Both radial and angular profiles retain empty
    bins with ``None`` statistics; polar numeric arrays use NaN for unsupported
    means, sums, and coverage, while count arrays contain zero.

    Supplied ``q_edges`` define the radial-profile and polar grid. Frame-wide
    intensity, q-range, quadrant, and tensor summaries still include all
    eligible pixels, including q values outside those analysis-grid edges.

    The returned ``sem`` is the pixel-sampling standard error of the mean,
    computed as sample standard deviation divided by square root of the pixel
    count. It is descriptive spatial variability, not fit uncertainty.
    """

    observed_image = _array2d(image, "image")
    shape = observed_image.shape
    qx_map = _array2d(qx, "qx", shape)
    qy_map = _array2d(qy, "qy", shape)
    n_radial = _bin_count(radial_bins, "radial_bins")
    n_angular = _bin_count(angular_bins, "angular_bins")
    if q_edges is not None:
        try:
            edge_count = np.asarray(q_edges).size - 1
        except (TypeError, ValueError) as exc:
            raise ValueError("q_edges must be a one-dimensional numeric sequence") from exc
        if edge_count < 1:
            raise ValueError("q_edges must contain at least two boundaries")
        n_radial = edge_count
    if n_radial * n_angular > _MAX_POLAR_CELLS:
        raise ValueError(
            f"radial_bins * angular_bins must be <= {_MAX_POLAR_CELLS} to bound polar output memory"
        )

    if valid_mask is None:
        inclusion = np.ones(shape, dtype=bool)
    else:
        try:
            supplied_mask = np.asarray(valid_mask)
        except (TypeError, ValueError) as exc:
            raise ValueError("valid_mask must match image shape") from exc
        if supplied_mask.shape != shape:
            raise ValueError(f"valid_mask shape {supplied_mask.shape!r} must match image shape {shape!r}")
        inclusion = supplied_mask.astype(bool, copy=False)

    image_flat = observed_image.ravel()
    qx_flat = qx_map.ravel()
    qy_flat = qy_map.ravel()
    inclusion_flat = inclusion.ravel()
    q_magnitude = np.hypot(qx_flat, qy_flat)
    geometry = np.isfinite(qx_flat) & np.isfinite(qy_flat) & np.isfinite(q_magnitude)
    if not np.any(geometry):
        raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
    usable = geometry & inclusion_flat & np.isfinite(image_flat)
    q_unit_canonical = canonical_q_unit(q_unit)
    q_unit_label = q_unit_canonical if q_unit_canonical in {"nm⁻¹", "Å⁻¹", "pixel-q"} else str(q_unit or "unknown")

    candidate_q = q_magnitude[geometry]
    edges = _q_edges(candidate_q, q_edges, n_radial)
    n_radial = edges.size - 1
    angle_edges = np.linspace(-180.0, 180.0, n_angular + 1, dtype=float)
    angles = np.degrees(np.arctan2(qy_flat, qx_flat))

    q_index, q_inside = _bin_indices(q_magnitude, edges)
    angle_index, angle_inside = _bin_indices(angles, angle_edges)
    candidate_radial_index = q_index[geometry & q_inside]
    observed_radial = usable & q_inside
    observed_radial_index = q_index[observed_radial]

    candidate_polar = geometry & q_inside & angle_inside
    observed_polar = usable & q_inside & angle_inside
    candidate_polar_index = q_index[candidate_polar] * n_angular + angle_index[candidate_polar]
    observed_polar_index = q_index[observed_polar] * n_angular + angle_index[observed_polar]
    observed_values = image_flat[observed_polar]

    radial_stats = _profile_stats(
        n_radial,
        observed_radial_index,
        image_flat[observed_radial],
        candidate_radial_index,
    )
    angular_candidate = candidate_polar_index % n_angular
    angular_observed = observed_polar_index % n_angular
    angular_stats = _profile_stats(
        n_angular,
        angular_observed,
        observed_values,
        angular_candidate,
    )
    polar_stats = _profile_stats(
        n_radial * n_angular,
        observed_polar_index,
        observed_values,
        candidate_polar_index,
    )
    polar = {
        key: values.reshape(n_radial, n_angular)
        for key, values in polar_stats.items()
    }
    polar["q_edges"] = edges.copy()
    polar["angle_edges"] = angle_edges

    observed_q = q_magnitude[usable]
    q_candidate_min = _finite_number(np.min(candidate_q))
    q_candidate_max = _finite_number(np.max(candidate_q))
    q_observed_min = _finite_number(np.min(observed_q)) if observed_q.size else None
    q_observed_max = _finite_number(np.max(observed_q)) if observed_q.size else None
    intensity_stats = _intensity_summary(image_flat[usable])

    if np.any(usable):
        max_flat = int(np.flatnonzero(usable)[np.argmax(image_flat[usable])])
        maxima: dict[str, Any] = {
            "intensity": _finite_number(image_flat[max_flat]),
            "row": max_flat // shape[1],
            "column": max_flat % shape[1],
            "qx": _finite_number(qx_flat[max_flat]),
            "qy": _finite_number(qy_flat[max_flat]),
            "q": _finite_number(q_magnitude[max_flat]),
            "angle_deg": _finite_number(angles[max_flat]),
        }
    else:
        maxima = {key: None for key in ("intensity", "row", "column", "qx", "qy", "q", "angle_deg")}

    tensor = _weighted_tensor(qx_flat[usable], qy_flat[usable], image_flat[usable])
    summary = {
        "q_unit": q_unit_label,
        "shape": [int(shape[0]), int(shape[1])],
        "pixel_counts": {
            "candidate_qmap": int(np.count_nonzero(geometry)),
            "observed": int(np.count_nonzero(usable)),
            "valid_mask_excluded": int(np.count_nonzero(geometry & ~inclusion_flat)),
            "nonfinite_intensity_after_mask": int(np.count_nonzero(geometry & inclusion_flat & ~np.isfinite(image_flat))),
            "invalid_qmap": int(image_flat.size - np.count_nonzero(geometry)),
        },
        "intensity": intensity_stats,
        "q_range": {
            "candidate_min": q_candidate_min,
            "candidate_max": q_candidate_max,
            "observed_min": q_observed_min,
            "observed_max": q_observed_max,
        },
        "reciprocal_spacing_proxy": _spacing_proxy(q_observed_min, q_observed_max, q_unit_canonical),
        "quadrants": _quadrants(qx_flat, qy_flat, image_flat, usable, geometry),
        "maxima": maxima,
        "in_plane_intensity_tensor": tensor,
        "sem_definition": (
            "sample standard deviation across observed pixels divided by sqrt(count); "
            "descriptive pixel-sampling statistic, not fit uncertainty"
        ),
        "quadrant_axis_assignment": "qx=0 is assigned to the right half and qy=0 to the upper half",
    }

    return {
        "summary": summary,
        "radial_rows": _profile_rows(radial_stats, edges[:-1], edges[1:], radial=True),
        "angular_rows": _profile_rows(angular_stats, angle_edges[:-1], angle_edges[1:], radial=False),
        "polar": polar,
    }


__all__ = ["measure_frame"]
