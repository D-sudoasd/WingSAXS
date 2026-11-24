"""Scaled q-space graph surface curvature, candidate detection, and multiscale NMS."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .._primitives import safe_gradient as _safe_gradient
from ..ridge_inputs import (
    coordinate_derivatives as _coordinate_derivatives,
    robust_noise as _robust_noise,
    sample as _sample,
)

CANDIDATE_NOISE_METHOD = "mask_normalized_raw_minus_smoothed_mad"
CANDIDATE_NOISE_MIN_SAFE_SAMPLES = 32
CANDIDATE_NOISE_EDGE_TRUNCATE = 4.0


def _check_cancelled(cancel_event: Any, stage: str) -> None:
    if cancel_event is None:
        return
    from ..cancellation import raise_if_cancelled

    raise_if_cancelled(cancel_event, stage)


def _offset_nearest(array: np.ndarray, drow: np.ndarray, dcol: np.ndarray) -> np.ndarray:
    """Sample ``array`` at each pixel plus a local (row, col) offset.

    The curvature stencil only needs a one-pixel sign/height check.  Full-image
    ``map_coordinates`` on a 2.5 Mpx detector was the dominant evaluate cost.
    """

    rows, cols = array.shape
    row_index = np.arange(rows, dtype=np.float64)[:, None]
    col_index = np.arange(cols, dtype=np.float64)[None, :]
    yi = np.clip(
        np.rint(np.nan_to_num(row_index - drow, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.intp),
        0,
        rows - 1,
    )
    xi = np.clip(
        np.rint(np.nan_to_num(col_index - dcol, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.intp),
        0,
        cols - 1,
    )
    return np.asarray(array[yi, xi], dtype=array.dtype)


@dataclass
class _CurvatureField:
    sigma: float
    smooth: np.ndarray
    support: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    valid: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    lambda_min: np.ndarray
    lambda_max: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    normal_slope: np.ndarray
    slope_minus: np.ndarray
    slope_plus: np.ndarray
    intensity_minus: np.ndarray
    intensity_plus: np.ndarray
    offset_fraction: np.ndarray
    q_normal_step: np.ndarray
    q_scale_anisotropy: np.ndarray
    axis_separation: np.ndarray
    candidate: np.ndarray
    accepted: np.ndarray
    score: np.ndarray
    curvature_threshold: float
    baseline: float
    noise: float
    q_step: float
    height_scale: float
    structural_spread: float
    noise_support_count: int
    noise_support_fraction: float
    noise_status: str
    noise_edge_guard_pixels: int
    coordinate_derivatives: Mapping[str, np.ndarray] | None = None


def _masked_highpass_noise(
    image: np.ndarray,
    smooth: np.ndarray,
    support: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: float,
    support_min: float,
) -> tuple[float, str, int, float, int]:
    """Estimate detector noise from valid raw-minus-smooth samples only."""

    safe = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(image)
        & np.isfinite(smooth)
        & (support >= float(support_min))
    )
    edge_guard = int(np.ceil(CANDIDATE_NOISE_EDGE_TRUNCATE * float(sigma)))
    if edge_guard:
        rows, cols = safe.shape
        if rows <= 2 * edge_guard or cols <= 2 * edge_guard:
            safe[:] = False
        else:
            safe[:edge_guard, :] = False
            safe[-edge_guard:, :] = False
            safe[:, :edge_guard] = False
            safe[:, -edge_guard:] = False
    sample_count = int(np.count_nonzero(safe))
    valid_count = int(np.count_nonzero(valid))
    support_fraction = float(sample_count / max(1, valid_count))
    if sample_count < CANDIDATE_NOISE_MIN_SAFE_SAMPLES:
        return float("inf"), "insufficient_safe_support", sample_count, support_fraction, edge_guard
    residual = np.asarray(image[safe], dtype=float) - np.asarray(smooth[safe], dtype=float)
    centre = float(np.median(residual))
    noise = float(1.4826 * np.median(np.abs(residual - centre)))
    scale = max(
        1.0,
        float(np.max(np.abs(image[safe]))),
        float(np.max(np.abs(smooth[safe]))),
    )
    if not np.isfinite(noise) or noise <= np.finfo(float).eps * scale:
        return float("inf"), "degenerate_highpass_mad", sample_count, support_fraction, edge_guard
    return noise, "ok", sample_count, support_fraction, edge_guard


def _scaled_surface_field(
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: float,
    options: Mapping[str, Any],
    coordinate_derivatives: tuple[dict[str, np.ndarray], np.ndarray, float] | None = None,
) -> _CurvatureField:
    if gaussian_filter is None:  # pragma: no cover
        raise RuntimeError("ridge tracing requires scipy.ndimage")
    weights = valid.astype(float)
    support = gaussian_filter(weights, sigma=float(sigma), mode="nearest")
    numerator = gaussian_filter(np.where(valid, image, 0.0), sigma=float(sigma), mode="nearest")
    smooth = np.divide(numerator, support, out=np.full_like(image, np.nan, dtype=float), where=support > 1e-8)
    finite_smooth = np.isfinite(smooth) & (support > 1e-8)
    fill = float(np.nanmedian(smooth[finite_smooth])) if np.any(finite_smooth) else 0.0
    surface = np.where(finite_smooth, smooth, fill)
    finite_values = surface[valid & np.isfinite(surface)]
    baseline = float(np.nanpercentile(finite_values, 10.0)) if finite_values.size else 0.0
    height_scale = float(np.nanpercentile(finite_values, 90.0) - np.nanpercentile(finite_values, 10.0)) if finite_values.size else 1.0
    structural_spread = _robust_noise(finite_values - baseline)
    if not np.isfinite(structural_spread) or structural_spread <= np.finfo(float).eps:
        structural_spread = max(float(np.nanstd(finite_values)), np.finfo(float).eps) if finite_values.size else 1.0
    height_scale = max(height_scale, 3.0 * structural_spread, np.finfo(float).eps)
    noise, noise_status, noise_support_count, noise_support_fraction, noise_edge_guard = _masked_highpass_noise(
        image,
        smooth,
        support,
        valid,
        sigma=float(sigma),
        support_min=float(options["support_min"]),
    )
    z = (surface - baseline) / height_scale
    if coordinate_derivatives is None:
        derivatives, _inverse, q_step = _coordinate_derivatives(qx, qy)
    else:
        derivatives, _inverse, q_step = coordinate_derivatives
    inv00, inv01 = derivatives["inv00"], derivatives["inv01"]
    inv10, inv11 = derivatives["inv10"], derivatives["inv11"]
    z_y, z_x = _safe_gradient(z, 0), _safe_gradient(z, 1)
    z_yy, z_yx = _safe_gradient(z_y, 0), _safe_gradient(z_y, 1)
    _z_xy, z_xx = _safe_gradient(z_x, 0), _safe_gradient(z_x, 1)
    z_xy = 0.5 * (z_yx + _z_xy)
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient_x = inv00 * z_x + inv10 * z_y
        gradient_y = inv01 * z_x + inv11 * z_y
        # Remove the coordinate-map second derivatives before transforming
        # the detector Hessian.  This is the chain rule for a graph surface,
        # rather than a pixel-grid Hessian used as a curvature surrogate.
        residual_xx = z_xx - gradient_x * derivatives["qx_xx"] - gradient_y * derivatives["qy_xx"]
        residual_xy = z_xy - gradient_x * derivatives["qx_xy"] - gradient_y * derivatives["qy_xy"]
        residual_yy = z_yy - gradient_x * derivatives["qx_yy"] - gradient_y * derivatives["qy_yy"]
        hxx = inv00 * (residual_xx * inv00 + residual_xy * inv10) + inv10 * (residual_xy * inv00 + residual_yy * inv10)
        hxy = inv00 * (residual_xx * inv01 + residual_xy * inv11) + inv10 * (residual_xy * inv01 + residual_yy * inv11)
        hyy = inv01 * (residual_xx * inv01 + residual_xy * inv11) + inv11 * (residual_xy * inv01 + residual_yy * inv11)
    # q coordinates are scaled so one typical detector q step is one unit.
    gradient_x *= q_step
    gradient_y *= q_step
    hxx *= q_step * q_step
    hxy *= q_step * q_step
    hyy *= q_step * q_step
    with np.errstate(divide="ignore", invalid="ignore"):
        graph_metric_det = 1.0 + gradient_x * gradient_x + gradient_y * gradient_y
        graph_normal = np.sqrt(graph_metric_det)
        first_e = 1.0 + gradient_x * gradient_x
        first_f = gradient_x * gradient_y
        first_g = 1.0 + gradient_y * gradient_y
        second_e = hxx / graph_normal
        second_f = hxy / graph_normal
        second_g = hyy / graph_normal
        mean_curvature = (first_e * second_g - 2.0 * first_f * second_f + first_g * second_e) / (2.0 * graph_metric_det)
        gaussian_curvature = (second_e * second_g - second_f * second_f) / graph_metric_det
        discriminant = np.sqrt(np.maximum(mean_curvature * mean_curvature - gaussian_curvature, 0.0))
        lambda_min = mean_curvature - discriminant
        lambda_max = mean_curvature + discriminant
        # Shape operator = metric^{-1} * second fundamental form.  The
        # selected eigenvector is a direction in the scaled q tangent plane.
        shape_a11 = (first_g * second_e - first_f * second_f) / graph_metric_det
        shape_a12 = (first_g * second_f - first_f * second_g) / graph_metric_det
        shape_a21 = (-first_f * second_e + first_e * second_f) / graph_metric_det
        shape_a22 = (-first_f * second_f + first_e * second_g) / graph_metric_det
        normal_x = shape_a12
        normal_y = lambda_min - shape_a11
        alternative = np.hypot(normal_x, normal_y) <= 1e-10
        normal_x = np.where(alternative, lambda_min - shape_a22, normal_x)
        normal_y = np.where(alternative, shape_a21, normal_y)
        normal_norm = np.hypot(normal_x, normal_y)
        normal_x = np.divide(normal_x, normal_norm, out=np.full_like(normal_x, np.nan), where=normal_norm > 1e-10)
        normal_y = np.divide(normal_y, normal_norm, out=np.full_like(normal_y, np.nan), where=normal_norm > 1e-10)
    tangent_x, tangent_y = -normal_y, normal_x
    with np.errstate(divide="ignore", invalid="ignore"):
        axis_separation = (lambda_max - lambda_min) / np.maximum(
            np.maximum(np.abs(lambda_min), np.abs(lambda_max)), np.finfo(float).eps
        )
        # One q step along the local principal normal, expressed in detector
        # pixels using the physical q-map Jacobian.
        displacement_x = q_step * (inv00 * normal_x + inv01 * normal_y)
        displacement_y = q_step * (inv10 * normal_x + inv11 * normal_y)
        q_normal_step = np.hypot(
            derivatives["qx_x"] * displacement_x + derivatives["qx_y"] * displacement_y,
            derivatives["qy_x"] * displacement_x + derivatives["qy_y"] * displacement_y,
        )
        singular_trace = derivatives["step_x"] ** 2 + derivatives["step_y"] ** 2
        singular_disc = np.sqrt(np.maximum((derivatives["step_x"] ** 2 - derivatives["step_y"] ** 2) ** 2 + 4.0 * (derivatives["qx_x"] * derivatives["qx_y"] + derivatives["qy_x"] * derivatives["qy_y"]) ** 2, 0.0))
        singular_max = np.sqrt(np.maximum((singular_trace + singular_disc) / 2.0, 0.0))
        singular_min = np.sqrt(np.maximum((singular_trace - singular_disc) / 2.0, 0.0))
        q_scale_anisotropy = singular_max / np.maximum(singular_min, np.finfo(float).eps)
        normal_slope = gradient_x * normal_x + gradient_y * normal_y
    step_factor = float(options.get("normal_step_factor", 1.0))
    sample_dx = displacement_x * step_factor
    sample_dy = displacement_y * step_factor
    # Project neighbour gradients onto the centre pixel's normal vector so
    # arbitrary +/-1 eigenvector sign flips at adjacent pixels cannot invert
    # the directional slope derivative.
    grad_x_minus = _offset_nearest(gradient_x, sample_dy, sample_dx)
    grad_y_minus = _offset_nearest(gradient_y, sample_dy, sample_dx)
    grad_x_plus = _offset_nearest(gradient_x, -sample_dy, -sample_dx)
    grad_y_plus = _offset_nearest(gradient_y, -sample_dy, -sample_dx)
    slope_minus = grad_x_minus * normal_x + grad_y_minus * normal_y
    slope_plus = grad_x_plus * normal_x + grad_y_plus * normal_y
    intensity_minus = _offset_nearest(z, sample_dy, sample_dx)
    intensity_plus = _offset_nearest(z, -sample_dy, -sample_dx)
    slope_derivative = (slope_plus - slope_minus) / np.maximum(2.0 * q_step * step_factor, np.finfo(float).eps)
    offset_q = np.divide(-normal_slope, slope_derivative, out=np.full_like(normal_slope, np.nan), where=slope_derivative < -1e-10)
    offset_fraction = np.divide(offset_q, max(q_step * step_factor, np.finfo(float).eps))
    max_offset = float(options.get("max_offset_fraction", 0.75))
    offset_fraction = np.clip(offset_fraction, -max_offset, max_offset)
    # Mask and crop boundaries must have one complete local stencil.
    edge = np.ones(image.shape, dtype=bool)
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    support_ok = support >= float(options["support_min"])
    finite = (
        valid
        & support_ok
        & edge
        & np.isfinite(lambda_min)
        & np.isfinite(normal_slope)
        & np.isfinite(offset_fraction)
        & np.isfinite(q_scale_anisotropy)
    )
    concave = lambda_min < 0.0
    local_maximum = (intensity_minus <= z) & (intensity_plus <= z)
    sign_change = ((slope_minus <= 0.0) & (slope_plus >= 0.0)) | ((slope_minus >= 0.0) & (slope_plus <= 0.0))
    slope_small = np.abs(normal_slope) <= np.nanpercentile(np.abs(normal_slope[finite]), 35.0) if np.any(finite) else np.zeros_like(finite)
    signal_ok = surface >= baseline + float(options["candidate_snr_min"]) * noise
    potential = finite & concave & local_maximum & signal_ok & (sign_change | slope_small)
    strengths = -lambda_min[potential]
    threshold = float(np.nanpercentile(strengths, float(options["curvature_percentile"]))) if strengths.size else float("inf")
    threshold = max(threshold, np.finfo(float).eps)
    score = np.maximum(-lambda_min, 0.0) * np.maximum(surface - baseline, 0.0) / max(noise / height_scale, np.finfo(float).eps)
    accepted = potential & (-lambda_min >= threshold)
    accepted &= axis_separation >= float(options["axis_separation_min"])
    accepted &= q_scale_anisotropy <= float(options["q_scale_anisotropy_max"])
    return _CurvatureField(
        sigma=float(sigma),
        # ``surface`` is already the mask-normalized Gaussian signal in the
        # image's original intensity units.  ``z`` is the separately scaled
        # surface used only for derivatives/curvature; do not scale the
        # metadata surface a second time.
        smooth=surface,
        support=support,
        qx=qx,
        qy=qy,
        valid=valid,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        normal_x=normal_x,
        normal_y=normal_y,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        normal_slope=normal_slope,
        slope_minus=slope_minus,
        slope_plus=slope_plus,
        intensity_minus=intensity_minus,
        intensity_plus=intensity_plus,
        offset_fraction=offset_fraction,
        q_normal_step=q_normal_step,
        q_scale_anisotropy=q_scale_anisotropy,
        axis_separation=axis_separation,
        candidate=potential,
        accepted=accepted,
        score=score,
        curvature_threshold=threshold,
        baseline=baseline,
        noise=noise,
        q_step=q_step,
        height_scale=height_scale,
        structural_spread=float(structural_spread),
        noise_support_count=noise_support_count,
        noise_support_fraction=noise_support_fraction,
        noise_status=noise_status,
        noise_edge_guard_pixels=noise_edge_guard,
        coordinate_derivatives=derivatives,
    )


def _raw_candidates(
    field: _CurvatureField,
    *,
    row_offset: int,
    col_offset: int,
    max_candidates: int,
    options: Mapping[str, Any],
) -> list[dict[str, Any]]:
    flat = np.flatnonzero(field.candidate.ravel())
    if flat.size > max_candidates:
        ranking = np.argsort(field.score.ravel()[flat])[::-1][:max_candidates]
        flat = flat[ranking]
    if flat.size == 0:
        return []
    rows, cols = np.unravel_index(flat, field.candidate.shape)
    fraction = field.offset_fraction[rows, cols]
    # The exact detector displacement is reconstructed from q-normal direction
    # and the local q-map Jacobian.  A local finite difference is sufficient
    # here because the q map is re-sampled at the resulting coordinates.
    derivatives = field.coordinate_derivatives
    if derivatives is None:  # Compatibility for manually constructed fields.
        derivatives, _inverse, _q_step = _coordinate_derivatives(field.qx, field.qy)
    inv00 = derivatives["inv00"]
    inv01 = derivatives["inv01"]
    inv10 = derivatives["inv10"]
    inv11 = derivatives["inv11"]
    displacement_x = field.q_step * (inv00[rows, cols] * field.normal_x[rows, cols] + inv01[rows, cols] * field.normal_y[rows, cols])
    displacement_y = field.q_step * (inv10[rows, cols] * field.normal_x[rows, cols] + inv11[rows, cols] * field.normal_y[rows, cols])
    step_factor = float(options.get("normal_step_factor", 1.0))
    sub_x = cols.astype(float) + fraction * step_factor * displacement_x
    sub_y = rows.astype(float) + fraction * step_factor * displacement_y
    qx_sub = _sample(field.qx, sub_y, sub_x)
    qy_sub = _sample(field.qy, sub_y, sub_x)
    intensity = _sample(field.smooth, sub_y, sub_x)
    local_noise = max(field.noise, np.finfo(float).eps)
    local_snr = (intensity - field.baseline) / local_noise
    results: list[dict[str, Any]] = []
    for index in range(flat.size):
        row, col = int(rows[index]), int(cols[index])
        accepted = bool(field.accepted[row, col])
        reason = "accepted"
        if not accepted:
            if float(field.axis_separation[row, col]) < float(options["axis_separation_min"]):
                reason = "ambiguous_local_axes"
            elif float(field.q_scale_anisotropy[row, col]) > float(options["q_scale_anisotropy_max"]):
                reason = "unstable_q_scale"
            elif float(-field.lambda_min[row, col]) < field.curvature_threshold:
                reason = "weak_curvature"
            elif not np.isfinite(local_snr[index]) or local_snr[index] < float(options["snr_threshold"]):
                reason = "low_snr"
            else:
                reason = "ridge_candidate_rejected"
        results.append(
            {
                "qx": float(qx_sub[index]),
                "qy": float(qy_sub[index]),
                "pixel_x": float(sub_x[index] + col_offset),
                "pixel_y": float(sub_y[index] + row_offset),
                "_local_pixel_x": float(sub_x[index]),
                "_local_pixel_y": float(sub_y[index]),
                "intensity": float(intensity[index]),
                "snr": float(local_snr[index]),
                "normal_qx": float(field.normal_x[row, col]),
                "normal_qy": float(field.normal_y[row, col]),
                "tangent_qx": float(field.tangent_x[row, col]),
                "tangent_qy": float(field.tangent_y[row, col]),
                "curvature": float(field.lambda_min[row, col]),
                "curvature_max": float(field.lambda_max[row, col]),
                "principal_curvature_min": float(field.lambda_min[row, col]),
                "principal_curvature_max": float(field.lambda_max[row, col]),
                "curvature_method": "scaled_graph_shape_operator",
                "axis_separation": float(field.axis_separation[row, col]),
                "q_normal_step": float(field.q_normal_step[row, col]),
                "q_scale_anisotropy": float(field.q_scale_anisotropy[row, col]),
                "scale": float(field.sigma),
                "scale_stability": float("nan"),
                "scale_stable": False,
                "topology_flag": "candidate",
                "topology": "candidate",
                "topology_flags": ["candidate"],
                "accepted": accepted,
                "valid": True,
                "reason": reason,
                "arc_id": -1,
                "branch_id": -1,
                "quadrant": "unknown",
                "side": "unknown",
                "score": float(field.score[row, col]),
                "_scale_index": int(round(field.sigma * 1000.0)),
            }
        )
    return results


def _nms(candidates: list[dict[str, Any]], options: Mapping[str, Any], q_step: float) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda point: (
            not bool(point["accepted"]),
            -float(point.get("score", 0.0)),
            float(point["qx"]),
            float(point["qy"]),
            float(point["scale"]),
        ),
    )
    nms_factor = float(options["nms_radius_factor"])
    base = max(0.25 * q_step, nms_factor * q_step, np.finfo(float).eps)
    finite_scales = [
        float(point["scale"])
        for point in ordered
        if math.isfinite(float(point.get("scale", 1.0)))
    ]
    max_scale = max(finite_scales, default=1.0)
    max_radius = max(base, nms_factor * max_scale * q_step)
    cell_reach = max(2, int(math.ceil(max_radius / base)))
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept: list[dict[str, Any]] = []
    for point in ordered:
        px, py = float(point["qx"]), float(point["qy"])
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        point_scale = float(point.get("scale", 1.0))
        if not math.isfinite(point_scale):
            point_scale = 1.0
        radius = max(base, nms_factor * point_scale * q_step)
        bx, by = int(math.floor(px / base)), int(math.floor(py / base))
        blocked = False
        for ix in range(bx - cell_reach, bx + cell_reach + 1):
            for iy in range(by - cell_reach, by + cell_reach + 1):
                for kept_index in bins.get((ix, iy), []):
                    other = kept[kept_index]
                    distance = math.hypot(px - float(other["qx"]), py - float(other["qy"]))
                    if distance >= max(radius, nms_factor * float(other["scale"]) * q_step):
                        continue
                    tangent_dot = abs(float(point["tangent_qx"]) * float(other["tangent_qx"]) + float(point["tangent_qy"]) * float(other["tangent_qy"]))
                    if tangent_dot >= 0.35:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if not blocked:
            index = len(kept)
            kept.append(point)
            bins[(bx, by)].append(index)
    for point in kept:
        point["_nms_kept"] = True
    return kept


def _annotate_scale_stability(
    candidates: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
    q_step: float,
    scales: Sequence[float],
    *,
    cancel_event: Any = None,
) -> None:
    """Annotate NMS points from an exact physical-q neighborhood query.

    The previous implementation compared every surviving point with every raw
    candidate.  A ``cKDTree`` gives the same closed-ball support set without
    silently dropping or downsampling raw candidates.  The small binned
    fallback remains exact when scipy.spatial is unavailable.
    """

    if not candidates:
        return
    radius = 1.8 * max(float(q_step), np.finfo(float).eps)
    raw_xy = np.asarray([[float(point["qx"]), float(point["qy"])] for point in raw_candidates], dtype=float)
    finite_raw = np.flatnonzero(np.all(np.isfinite(raw_xy), axis=1)) if raw_xy.size else np.asarray([], dtype=int)
    tree = cKDTree(raw_xy[finite_raw]) if finite_raw.size and cKDTree is not None else None
    bins: dict[tuple[int, int], list[int]] = {}
    if tree is None and raw_xy.size:
        cell = radius
        for raw_index in finite_raw.tolist():
            point = raw_candidates[raw_index]
            if raw_index % 2048 == 0:
                _check_cancelled(cancel_event, "ridge-trace:scale-stability-index")
            key = (int(math.floor(float(point["qx"]) / cell)), int(math.floor(float(point["qy"]) / cell)))
            bins.setdefault(key, []).append(raw_index)
    n_scales = max(len(scales), 1)
    for index, point in enumerate(candidates):
        if index % 128 == 0:
            _check_cancelled(cancel_event, "ridge-trace:scale-stability-query")
        xy = [float(point["qx"]), float(point["qy"])]
        if tree is not None:
            support_indices = finite_raw[np.asarray(tree.query_ball_point(xy, radius), dtype=int)].tolist()
        elif raw_xy.size:
            cell = radius
            bx = int(math.floor(xy[0] / cell))
            by = int(math.floor(xy[1] / cell))
            support_indices = [
                raw_index
                for ix in range(bx - 1, bx + 2)
                for iy in range(by - 1, by + 2)
                for raw_index in bins.get((ix, iy), ())
                if math.hypot(raw_xy[raw_index, 0] - xy[0], raw_xy[raw_index, 1] - xy[1]) <= radius
            ]
        else:
            support_indices = []
        _check_cancelled(cancel_event, "ridge-trace:scale-stability-query")
        unique_scales = len({round(float(raw_candidates[raw_index]["scale"]), 5) for raw_index in support_indices})
        point["scale_stability"] = float(unique_scales / n_scales)
        point["scale_stable"] = bool(point["scale_stability"] >= 0.34)
        if point["scale_stability"] < 0.34:
            point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["single_scale_support"]))


def _finite_scale_stability_summary(
    points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize finite scale support without emitting all-NaN warnings."""

    values: list[float] = []
    for point in points:
        try:
            value = float(point.get("scale_stability", float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {
            "mean": None,
            "min": None,
            "n_finite": 0,
            "n_candidates": int(len(points)),
        }
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "n_finite": int(len(values)),
        "n_candidates": int(len(points)),
    }
