"""Descriptive measurements from observed two-dimensional SAXS frames.

This module bins the supplied intensity image in reciprocal-space radius and
azimuth. It does not fit peaks, interpolate unsupported bins, or infer a
three-dimensional structure.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .local_measurement import spacing_from_q
from .saxs_physics import (
    _radial_metrics_from_1d_profile,
    compute_azimuthal_harmonics_and_order,
    fit_c2_bridge_four_lobe,
)
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


def _bin_indices(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    uniform: bool = False,
    all_inside: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``intp`` indices (zero-copy for ``np.bincount``) and an optional in-range mask."""

    n_bins = edges.size - 1
    e_lo = float(edges[0])
    e_hi = float(edges[-1])
    if all_inside and uniform and e_hi > e_lo:
        inv_step = float(n_bins) / (e_hi - e_lo)
        scaled = values - e_lo
        scaled *= inv_step
        indices = scaled.astype(np.intp)
        indices[indices >= n_bins] = n_bins - 1
        return indices, None
    inside = (values >= e_lo) & (values <= e_hi)
    if uniform and e_hi > e_lo:
        inv_step = float(n_bins) / (e_hi - e_lo)
        scaled = values - e_lo
        scaled *= inv_step
        np.floor(scaled, out=scaled)
        indices = scaled.astype(np.intp)
        np.clip(indices, 0, n_bins - 1, out=indices)
    else:
        indices = np.searchsorted(edges, values, side="right").astype(np.intp) - 1
        indices[values == e_hi] = n_bins - 1
    return indices, inside


def _profile_stats_with_m2(
    n_bins: int,
    observed_indices: np.ndarray,
    observed_values: np.ndarray,
    candidate_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Accumulate sample statistics and within-bin M2 without allocating per-bin pixel lists."""

    counts = np.bincount(observed_indices, minlength=n_bins).astype(np.int64, copy=False)
    candidate_counts = np.bincount(candidate_indices, minlength=n_bins).astype(np.int64, copy=False)
    sums = np.bincount(observed_indices, weights=observed_values, minlength=n_bins).astype(float, copy=False)
    means = np.full(n_bins, np.nan, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)

    stds = np.full(n_bins, np.nan, dtype=float)
    sems = np.full(n_bins, np.nan, dtype=float)
    if observed_values.size:
        with np.errstate(invalid="ignore", over="ignore"):
            sq_vals = observed_values * observed_values
            sum_sq = np.bincount(observed_indices, weights=sq_vals, minlength=n_bins).astype(float, copy=False)
            m2 = np.zeros(n_bins, dtype=float)
            has_cnt = counts > 1
            mean_sq_term = np.zeros(n_bins, dtype=float)
            mean_sq_term[has_cnt] = counts[has_cnt] * (means[has_cnt] * means[has_cnt])
            m2[has_cnt] = sum_sq[has_cnt] - mean_sq_term[has_cnt]
            # Fall back to exact centered two-pass M2 whenever cancellation or non-finite sum_sq occurs
            needs_centered = has_cnt & (
                ~np.isfinite(m2)
                | (m2 < 0.0)
                | ((sum_sq > 0.0) & (m2 < 1.0e-10 * sum_sq))
            )
            if np.any(needs_centered):
                sel_c = needs_centered[observed_indices]
                idx_c = observed_indices[sel_c]
                dev_c = observed_values[sel_c] - means[idx_c]
                np.square(dev_c, out=dev_c)
                m2_c = np.bincount(idx_c, weights=dev_c, minlength=n_bins).astype(float, copy=False)
                m2[needs_centered] = m2_c[needs_centered]
    else:
        m2 = np.zeros(n_bins, dtype=float)
    has_sample_variance = counts > 1
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        stds[has_sample_variance] = np.sqrt(m2[has_sample_variance] / (counts[has_sample_variance] - 1))
        sems[has_sample_variance] = stds[has_sample_variance] / np.sqrt(counts[has_sample_variance])

    coverage = np.full(n_bins, np.nan, dtype=float)
    np.divide(counts, candidate_counts, out=coverage, where=candidate_counts > 0)
    # Empty bins have no measured sum. Keep them missing internally as well
    # as in the row representation so they cannot be mistaken for zero data.
    sums_out = sums.copy()
    sums_out[counts == 0] = np.nan
    return (
        {
            "mean": means,
            "sum": sums_out,
            "std": stds,
            "sem": sems,
            "count": counts,
            "candidate_count": candidate_counts,
            "coverage": coverage,
        },
        m2,
    )


def _profile_stats(
    n_bins: int,
    observed_indices: np.ndarray,
    observed_values: np.ndarray,
    candidate_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Accumulate sample statistics without allocating per-bin pixel lists."""
    stats, _ = _profile_stats_with_m2(n_bins, observed_indices, observed_values, candidate_indices)
    return stats


def _marginalize_polar_stats(
    counts_2d: np.ndarray,
    cand_2d: np.ndarray,
    sums_2d: np.ndarray,
    means_2d: np.ndarray,
    m2_2d: np.ndarray,
    *,
    axis: int,
) -> dict[str, np.ndarray]:
    """Exact O(n_radial * n_angular) Chan parallel-variance marginalization of polar stats."""
    counts = np.sum(counts_2d, axis=axis, dtype=np.int64)
    candidate_counts = np.sum(cand_2d, axis=axis, dtype=np.int64)
    sums_raw = np.sum(np.where(counts_2d > 0, sums_2d, 0.0), axis=axis, dtype=float)
    means = np.full(counts.shape, np.nan, dtype=float)
    np.divide(sums_raw, counts, out=means, where=counts > 0)

    mean_bcast = np.expand_dims(means, axis=axis)
    has_cell = counts_2d > 0
    delta = np.where(has_cell, means_2d - mean_bcast, 0.0)
    m2_total = np.sum(m2_2d + counts_2d * (delta * delta), axis=axis, dtype=float)

    stds = np.full(counts.shape, np.nan, dtype=float)
    sems = np.full(counts.shape, np.nan, dtype=float)
    has_var = counts > 1
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        stds[has_var] = np.sqrt(m2_total[has_var] / (counts[has_var] - 1))
        sems[has_var] = stds[has_var] / np.sqrt(counts[has_var])

    coverage = np.full(counts.shape, np.nan, dtype=float)
    np.divide(counts, candidate_counts, out=coverage, where=candidate_counts > 0)
    sums_out = sums_raw.copy()
    sums_out[counts == 0] = np.nan
    return {
        "mean": means,
        "sum": sums_out,
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

    n = int(values.size)
    with np.errstate(invalid="ignore", over="ignore"):
        total = float(np.sum(values, dtype=float))
        mean = total / float(n)
        if n > 1:
            sum_sq = float(np.dot(values, values))
            ss_fast = sum_sq - float(n) * (mean * mean)
            if math.isfinite(sum_sq) and sum_sq > 0.0 and ss_fast > 1.0e-6 * sum_sq:
                std = float(math.sqrt(ss_fast / float(n - 1)))
            else:
                diff = values - mean
                std = float(math.sqrt(float(np.dot(diff, diff)) / float(n - 1)))
        else:
            std = np.nan

        if n >= 32:
            # Single np.partition computes all 7 exact linear-interpolated percentiles,
            # plus exact min/max from the 1% tails and exact negative_count via partition bounds.
            pos_f = [(p / 100.0) * (n - 1) for p in _PERCENTILES]
            lo_idx = [int(math.floor(x)) for x in pos_f]
            hi_idx = [int(math.ceil(x)) for x in pos_f]
            kths = sorted(set(lo_idx + hi_idx))
            part = np.partition(values, kths)
            percentiles = [
                float(part[lo] + (pf - lo) * (part[hi] - part[lo]))
                for pf, lo, hi in zip(pos_f, lo_idx, hi_idx)
            ]
            median = percentiles[3]
            vmin = float(np.min(part[: lo_idx[0] + 1]))
            vmax = float(np.max(part[hi_idx[-1] :]))
            # Use partition order statistics to narrow negative_count scan when possible
            neg_count: int | None = None
            for k_idx in kths:
                if part[k_idx] >= 0.0:
                    neg_count = int(np.count_nonzero(part[:k_idx] < 0.0))
                    break
            if neg_count is None:
                k_last = kths[-1]
                neg_count = k_last + 1 + int(np.count_nonzero(part[k_last + 1 :] < 0.0))
        else:
            percentiles = [float(v) for v in np.percentile(values, _PERCENTILES)]
            median = percentiles[3]
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            neg_count = int(np.count_nonzero(values < 0.0))
    stats = {
        "count": n,
        "sum": _finite_number(total),
        "mean": _finite_number(mean),
        "median": _finite_number(median),
        "std": _finite_number(std),
        "minimum": _finite_number(vmin),
        "maximum": _finite_number(vmax),
        "negative_count": neg_count,
    }
    stats.update({f"p{int(percentile):02d}": _finite_number(value) for percentile, value in zip(_PERCENTILES, percentiles)})
    return stats


def _quadrants(
    qx: np.ndarray,
    qy: np.ndarray,
    image: np.ndarray,
    observed: np.ndarray,
    geometry: np.ndarray | None,
    *,
    obs_qx: np.ndarray | None = None,
    obs_qy: np.ndarray | None = None,
    obs_img: np.ndarray | None = None,
    all_geom_finite: bool = False,
) -> dict[str, dict[str, Any]]:
    # Half-open assignments make quadrant counts disjoint and exhaustive:
    # qx == 0 goes to the right half, qy == 0 to the upper half.
    if all_geom_finite:
        nx_g = qx < 0.0
        ny_g = qy < 0.0
        c2 = int(np.count_nonzero(nx_g & ~ny_g))
        c3 = int(np.count_nonzero(nx_g & ny_g))
        c4 = int(np.count_nonzero(~nx_g & ny_g))
        c1 = int(qx.size) - c2 - c3 - c4
        cand_map = {"QI": c1, "QII": c2, "QIII": c3, "QIV": c4}
        nx_o = nx_g[observed]
        ny_o = ny_g[observed]
        img_o = image[observed] if obs_img is None else obs_img
    else:
        assert geometry is not None
        qx_g = qx[geometry]
        qy_g = qy[geometry]
        nx_g = qx_g < 0.0
        ny_g = qy_g < 0.0
        c2 = int(np.count_nonzero(nx_g & ~ny_g))
        c3 = int(np.count_nonzero(nx_g & ny_g))
        c4 = int(np.count_nonzero(~nx_g & ny_g))
        c1 = int(qx_g.size) - c2 - c3 - c4
        cand_map = {"QI": c1, "QII": c2, "QIII": c3, "QIV": c4}
        qx_o = qx[observed] if obs_qx is None else obs_qx
        qy_o = qy[observed] if obs_qy is None else obs_qy
        nx_o = qx_o < 0.0
        ny_o = qy_o < 0.0
        img_o = image[observed] if obs_img is None else obs_img

    pos_x = ~nx_o
    pos_y = ~ny_o
    obs_masks = (
        ("QI", pos_x & pos_y),
        ("QII", nx_o & pos_y),
        ("QIII", nx_o & ny_o),
        ("QIV", pos_x & ny_o),
    )
    result: dict[str, dict[str, Any]] = {}
    for name, mask in obs_masks:
        count = int(np.count_nonzero(mask))
        candidate_count = cand_map[name]
        total = float(np.sum(img_o, where=mask, dtype=float)) if count else np.nan
        result[name] = {
            "sum": _finite_number(total),
            "count": count,
            "candidate_count": candidate_count,
            "coverage": count / candidate_count if candidate_count else None,
        }
    return result


def _weighted_tensor(
    qx: np.ndarray,
    qy: np.ndarray,
    image: np.ndarray,
    *,
    all_nonnegative: bool = False,
    precomputed_weight_sum: float | None = None,
) -> dict[str, Any]:
    if all_nonnegative:
        weights = image
        weight_sum = float(precomputed_weight_sum) if precomputed_weight_sum is not None else float(np.sum(image, dtype=float))
    else:
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

    q_scale = float(
        max(
            abs(float(np.min(qx))),
            abs(float(np.max(qx))),
            abs(float(np.min(qy))),
            abs(float(np.max(qy))),
        )
    )
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

    # Scale q before products only if extreme magnitudes risk overflow/underflow;
    # otherwise compute BLAS dot products directly and apply common scale factor once.
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        if 1.0e-100 <= q_scale <= 1.0e100:
            inv_denom = 1.0 / (weight_sum * q_scale * q_scale)
            buf = weights * qx
            t_xx = float(np.dot(buf, qx)) * inv_denom
            t_xy = float(np.dot(buf, qy)) * inv_denom
            np.multiply(weights, qy, out=buf)
            t_yy = float(np.dot(buf, qy)) * inv_denom
        else:
            inv_scale = 1.0 / q_scale
            sx = qx * inv_scale
            sy = qy * inv_scale
            buf = weights * sx
            t_xx = float(np.dot(buf, sx)) / weight_sum
            t_xy = float(np.dot(buf, sy)) / weight_sum
            np.multiply(weights, sy, out=buf)
            t_yy = float(np.dot(buf, sy)) / weight_sum
        tensor = np.array([[t_xx, t_xy], [t_xy, t_yy]], dtype=float)
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


def _saxs_physics_summary(
    *,
    edges: np.ndarray,
    radial_stats: dict[str, np.ndarray],
    angle_edges: np.ndarray,
    angular_stats: dict[str, np.ndarray],
    q_unit: str,
) -> dict[str, Any]:
    """Compute additive strict-JSON-safe saxs_physics metrics from observed profiles."""
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    ang_mean = angular_stats["mean"]
    ang_count = angular_stats["count"]

    harmonics = compute_azimuthal_harmonics_and_order(
        angle_deg=angle_centers,
        intensity=ang_mean,
        counts=ang_count,
        q_unit=q_unit,
    )
    radial = _radial_metrics_from_1d_profile(
        edges,
        radial_stats["mean"],
        radial_stats["count"],
        radial_stats["candidate_count"],
        q_unit=q_unit,
    )

    wing_half_opening = harmonics.get("wing_half_opening_estimate_deg")
    n_obs_ang = int(harmonics.get("n_observed_bins") or 0)
    a4_val = harmonics.get("a4")
    if n_obs_ang >= 12 and a4_val is not None and a4_val >= 0.02:
        c2_fit = fit_c2_bridge_four_lobe(
            angle_centers,
            ang_mean,
            counts=ang_count,
            initial_chi_sym_deg=harmonics.get("friedel_bisector_deg"),
            initial_alpha_deg=wing_half_opening,
            max_nfev=400,
            include_flat_comparison=False,
        )
        if c2_fit.get("success") and c2_fit.get("alpha_deg") is not None:
            wing_half_opening = c2_fit["alpha_deg"]

    caveats: list[str] = [
        "q_win_invariant is a window-bounded scattering invariant proxy over observed q bins only; "
        "it does not estimate delta_rho or volume fraction.",
    ]
    if q_unit not in {"nm⁻¹", "Å⁻¹"}:
        caveats.append(
            f"L_star_direct_nm and L_star_kratky_nm are None because q_unit={q_unit!r} is not a calibrated physical unit."
        )

    return {
        "status": harmonics.get("status", "unavailable"),
        "a2": harmonics.get("a2"),
        "a4": harmonics.get("a4"),
        "a4_over_a2": harmonics.get("a4_over_a2"),
        "f2_hermans": harmonics.get("f2"),
        "f4_order": harmonics.get("f4"),
        "f2_nominal": harmonics.get("f2_nominal"),
        "f4_nominal": harmonics.get("f4_nominal"),
        "phi2_deg": harmonics.get("phi2_deg"),
        "phi4_deg": harmonics.get("phi4_deg"),
        "friedel_bisector_deg": harmonics.get("friedel_bisector_deg"),
        "wing_half_opening_deg": _finite_number(wing_half_opening),
        "q_win_invariant": radial["q_win_invariant"],
        "q_star_direct": radial["q_star_direct"],
        "L_star_direct_nm": radial["L_star_direct_nm"],
        "edge_hit_direct": radial["edge_hit_direct"],
        "q_star_kratky": radial["q_star_kratky"],
        "L_star_kratky_nm": radial["L_star_kratky_nm"],
        "edge_hit_kratky": radial["edge_hit_kratky"],
        "friedel_q1_q3_ratio": harmonics.get("q1_over_q3"),
        "friedel_q2_q4_ratio": harmonics.get("q2_over_q4"),
        "flags": list(harmonics.get("flags") or ()),
        "caveats": caveats,
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
    q_mag: Any = None,
    chi_deg: Any = None,
    geometry_cache: Any = None,
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

    # Pull precomputed q_mag / chi_deg from geometry_cache when shape matches
    if geometry_cache is not None:
        if q_mag is None:
            if getattr(geometry_cache, "shape", None) == shape and hasattr(geometry_cache.geometry, "q_nm_inv"):
                q_mag = geometry_cache.geometry.q_nm_inv
            elif getattr(geometry_cache, "roi_shape", None) == shape:
                q_mag = geometry_cache.q_mag_roi
        if chi_deg is None:
            if getattr(geometry_cache, "shape", None) == shape and hasattr(geometry_cache.geometry, "chi_deg"):
                chi_deg = geometry_cache.geometry.chi_deg
            elif getattr(geometry_cache, "roi_shape", None) == shape:
                chi_deg = geometry_cache.chi_deg_roi

    image_flat = observed_image.ravel()
    qx_flat = qx_map.ravel()
    qy_flat = qy_map.ravel()
    inclusion_flat = inclusion.ravel()

    if q_mag is not None:
        q_mag_arr = _array2d(q_mag, "q_mag", shape)
        q_magnitude = q_mag_arr.ravel()
        if geometry_cache is not None and getattr(geometry_cache, "q_coordinates_validated", False):
            all_geom_finite = True
            geometry = None
        else:
            q_lo_f, q_hi_f = float(np.min(q_magnitude)), float(np.max(q_magnitude))
            qx_lo_f, qx_hi_f = float(np.min(qx_flat)), float(np.max(qx_flat))
            qy_lo_f, qy_hi_f = float(np.min(qy_flat)), float(np.max(qy_flat))
            all_geom_finite = (
                math.isfinite(q_lo_f)
                and math.isfinite(q_hi_f)
                and math.isfinite(qx_lo_f)
                and math.isfinite(qx_hi_f)
                and math.isfinite(qy_lo_f)
                and math.isfinite(qy_hi_f)
            )
            geometry = None if all_geom_finite else (np.isfinite(qx_flat) & np.isfinite(qy_flat) & np.isfinite(q_magnitude))
    else:
        qx_lo_f, qx_hi_f = float(np.min(qx_flat)), float(np.max(qx_flat))
        qy_lo_f, qy_hi_f = float(np.min(qy_flat)), float(np.max(qy_flat))
        all_coords_finite = (
            math.isfinite(qx_lo_f)
            and math.isfinite(qx_hi_f)
            and math.isfinite(qy_lo_f)
            and math.isfinite(qy_hi_f)
        )
        if all_coords_finite and max(abs(qx_lo_f), abs(qx_hi_f), abs(qy_lo_f), abs(qy_hi_f)) < 1.0e100:
            q_magnitude = qx_flat * qx_flat
            q_magnitude += qy_flat * qy_flat
            np.sqrt(q_magnitude, out=q_magnitude)
            all_geom_finite = True
            geometry = None
        else:
            q_magnitude = np.hypot(qx_flat, qy_flat)
            geometry = np.isfinite(q_magnitude)
            all_geom_finite = bool(np.all(geometry))

    if not all_geom_finite and not np.any(geometry):
        raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
    geom_and_incl = inclusion_flat if all_geom_finite else (geometry & inclusion_flat)
    usable = geom_and_incl & np.isfinite(image_flat)
    q_unit_canonical = canonical_q_unit(q_unit)
    q_unit_label = q_unit_canonical if q_unit_canonical in {"nm⁻¹", "Å⁻¹", "pixel-q"} else str(q_unit or "unknown")

    candidate_q = q_magnitude if all_geom_finite else q_magnitude[geometry]
    q_cand_min_val = float(np.min(candidate_q))
    q_cand_max_val = float(np.max(candidate_q))
    if q_edges is None:
        if q_cand_min_val == q_cand_max_val:
            edges = _q_edges(candidate_q, None, n_radial)
        else:
            edges = np.linspace(q_cand_min_val, q_cand_max_val, n_radial + 1, dtype=float)
    else:
        edges = _q_edges(candidate_q, q_edges, n_radial)
    n_radial = edges.size - 1
    angle_edges = np.linspace(-180.0, 180.0, n_angular + 1, dtype=float)

    uniform_q = q_edges is None
    all_full_domain = all_geom_finite and uniform_q
    q_index, q_inside = _bin_indices(q_magnitude, edges, uniform=uniform_q, all_inside=all_full_domain)

    def _chi_deg_to_angle_bins(chi_vals: np.ndarray) -> np.ndarray:
        """Map ``chi_deg`` in ``[0, 360)`` to ``intp`` bins over ``[-180, 180]`` without float copies."""
        inv_step = float(n_angular) / 360.0
        if n_angular % 2 == 0:
            half = n_angular // 2
            idx = (chi_vals * inv_step).astype(np.intp)
            idx += half
            idx[chi_vals > 180.0] -= n_angular
            idx[idx >= n_angular] = n_angular - 1
            idx[idx < 0] = 0
            return idx
        shifted = np.where(chi_vals > 180.0, chi_vals - 180.0, chi_vals + 180.0)
        idx = (shifted * inv_step).astype(np.intp)
        idx[idx >= n_angular] = n_angular - 1
        idx[idx < 0] = 0
        return idx

    # Single fused in-place intp polar-index path: zero redundant copies or int casts
    if all_full_domain:
        if chi_deg is not None:
            chi_arr = _array2d(chi_deg, "chi_deg", shape).ravel()
            cand_angle_idx = _chi_deg_to_angle_bins(chi_arr)
        else:
            inv_ang_rad = float(n_angular) / (2.0 * math.pi)
            ang_buf = np.arctan2(qy_flat, qx_flat)
            ang_buf += math.pi
            ang_buf *= inv_ang_rad
            cand_angle_idx = ang_buf.astype(np.intp)
            cand_angle_idx[cand_angle_idx >= n_angular] = n_angular - 1
        q_index *= n_angular
        q_index += cand_angle_idx
        candidate_polar_index = q_index
        observed_polar_index = candidate_polar_index[usable]
        obs_img = image_flat[usable]
        observed_values = obs_img
    else:
        assert q_inside is not None
        candidate_polar = q_inside if all_geom_finite else (geometry & q_inside)
        observed_polar = usable & q_inside
        obs_in_cand = observed_polar[candidate_polar]
        if chi_deg is not None:
            chi_arr = _array2d(chi_deg, "chi_deg", shape).ravel()
            cand_angle_idx = _chi_deg_to_angle_bins(chi_arr[candidate_polar])
        else:
            ang_buf = np.arctan2(qy_flat[candidate_polar], qx_flat[candidate_polar])
            ang_buf += math.pi
            ang_buf *= float(n_angular) / (2.0 * math.pi)
            cand_angle_idx = ang_buf.astype(np.intp)
            cand_angle_idx[cand_angle_idx >= n_angular] = n_angular - 1
        cand_q_idx = q_index[candidate_polar]
        cand_q_idx *= n_angular
        cand_q_idx += cand_angle_idx
        candidate_polar_index = cand_q_idx
        observed_polar_index = candidate_polar_index[obs_in_cand]
        observed_values = image_flat[observed_polar]
        obs_img = image_flat[usable]

    # Single full-frame polar pass + exact O(n_radial * n_angular) Chan marginalization
    polar_stats, polar_m2 = _profile_stats_with_m2(
        n_radial * n_angular,
        observed_polar_index,
        observed_values,
        candidate_polar_index,
    )
    polar = {
        key: values.reshape(n_radial, n_angular)
        for key, values in polar_stats.items()
    }
    m2_2d = polar_m2.reshape(n_radial, n_angular)
    radial_stats = _marginalize_polar_stats(
        polar["count"], polar["candidate_count"], polar["sum"], polar["mean"], m2_2d, axis=1
    )
    angular_stats = _marginalize_polar_stats(
        polar["count"], polar["candidate_count"], polar["sum"], polar["mean"], m2_2d, axis=0
    )
    polar["q_edges"] = edges.copy()
    polar["angle_edges"] = angle_edges

    # Extract usable 1D vectors ONCE and avoid copying observed_q array
    obs_qx = qx_flat[usable]
    obs_qy = qy_flat[usable]
    q_candidate_min = _finite_number(q_cand_min_val)
    q_candidate_max = _finite_number(q_cand_max_val)
    if obs_img.size:
        q_observed_min = _finite_number(np.min(q_magnitude, initial=np.inf, where=usable))
        q_observed_max = _finite_number(np.max(q_magnitude, initial=-np.inf, where=usable))
    else:
        q_observed_min = None
        q_observed_max = None
    intensity_stats = _intensity_summary(obs_img)

    if obs_img.size:
        max_local = int(np.argmax(obs_img))
        max_val = float(obs_img[max_local])
        # Deterministic first maximum in row-major order without allocating full flatnonzero(usable)
        max_flat = int(np.argmax(usable & (image_flat == max_val)))
        max_qx_val = float(obs_qx[max_local])
        max_qy_val = float(obs_qy[max_local])
        maxima: dict[str, Any] = {
            "intensity": _finite_number(max_val),
            "row": max_flat // shape[1],
            "column": max_flat % shape[1],
            "qx": _finite_number(max_qx_val),
            "qy": _finite_number(max_qy_val),
            "q": _finite_number(q_magnitude[max_flat]),
            "angle_deg": _finite_number(math.degrees(math.atan2(max_qy_val, max_qx_val))),
        }
    else:
        maxima = {key: None for key in ("intensity", "row", "column", "qx", "qy", "q", "angle_deg")}

    tensor = _weighted_tensor(
        obs_qx,
        obs_qy,
        obs_img,
        all_nonnegative=(intensity_stats["negative_count"] == 0),
        precomputed_weight_sum=intensity_stats["sum"],
    )
    saxs_physics_summary = _saxs_physics_summary(
        edges=edges,
        radial_stats=radial_stats,
        angle_edges=angle_edges,
        angular_stats=angular_stats,
        q_unit=q_unit_canonical,
    )
    n_geom = int(image_flat.size) if all_geom_finite else int(np.count_nonzero(geometry))
    n_geom_incl = int(np.count_nonzero(geom_and_incl))
    n_obs = int(obs_img.size)
    summary = {
        "q_unit": q_unit_label,
        "shape": [int(shape[0]), int(shape[1])],
        "pixel_counts": {
            "candidate_qmap": n_geom,
            "observed": n_obs,
            "valid_mask_excluded": n_geom - n_geom_incl,
            "nonfinite_intensity_after_mask": n_geom_incl - n_obs,
            "invalid_qmap": int(image_flat.size) - n_geom,
        },
        "intensity": intensity_stats,
        "q_range": {
            "candidate_min": q_candidate_min,
            "candidate_max": q_candidate_max,
            "observed_min": q_observed_min,
            "observed_max": q_observed_max,
        },
        "reciprocal_spacing_proxy": _spacing_proxy(q_observed_min, q_observed_max, q_unit_canonical),
        "quadrants": _quadrants(
            qx_flat,
            qy_flat,
            image_flat,
            usable,
            geometry,
            obs_qx=obs_qx,
            obs_qy=obs_qy,
            obs_img=obs_img,
            all_geom_finite=all_geom_finite,
        ),
        "maxima": maxima,
        "in_plane_intensity_tensor": tensor,
        "sem_definition": (
            "sample standard deviation across observed pixels divided by sqrt(count); "
            "descriptive pixel-sampling statistic, not fit uncertainty"
        ),
        "quadrant_axis_assignment": "qx=0 is assigned to the right half and qy=0 to the upper half",
        "a2": saxs_physics_summary["a2"],
        "a4": saxs_physics_summary["a4"],
        "a4_over_a2": saxs_physics_summary["a4_over_a2"],
        "f2_hermans": saxs_physics_summary["f2_hermans"],
        "f4_order": saxs_physics_summary["f4_order"],
        "friedel_bisector_deg": saxs_physics_summary["friedel_bisector_deg"],
        "wing_half_opening_deg": saxs_physics_summary["wing_half_opening_deg"],
        "q_win_invariant": saxs_physics_summary["q_win_invariant"],
        "q_star_direct": saxs_physics_summary["q_star_direct"],
        "L_star_direct_nm": saxs_physics_summary["L_star_direct_nm"],
        "q_star_kratky": saxs_physics_summary["q_star_kratky"],
        "L_star_kratky_nm": saxs_physics_summary["L_star_kratky_nm"],
        "friedel_q1_q3_ratio": saxs_physics_summary["friedel_q1_q3_ratio"],
        "friedel_q2_q4_ratio": saxs_physics_summary["friedel_q2_q4_ratio"],
        "saxs_physics": saxs_physics_summary,
    }

    return {
        "summary": summary,
        "radial_rows": _profile_rows(radial_stats, edges[:-1], edges[1:], radial=True),
        "angular_rows": _profile_rows(angular_stats, angle_edges[:-1], angle_edges[1:], radial=False),
        "polar": polar,
    }


__all__ = ["measure_frame"]
