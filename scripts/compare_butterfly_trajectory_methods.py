"""Compare three WingSAXS trace methods and one fixed-qy baseline on paired images.

The benchmark runs ``annular_peak``, ``radial_sector``, ``curvature`` and a
simple ``cartesian_slice`` baseline on the same generated intensity image, q
map, invalid-pixel mask and q window. The Cartesian baseline is a fixed-qy
detector-row peak scan on a rectilinear q grid; it is not a reproduction of a
published slice or curvature algorithm.

Example:
    py -3.13 scripts/compare_butterfly_trajectory_methods.py \
      --output results/validation/trajectory_method_comparison/run-01 \
      --cases ellipse_ratio_100,partial_arcs,missing_branch,null
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from butterfly_saxs.benchmark_arcs import (  # noqa: E402
    DEFAULT_CASE_IDS,
    GENERATOR_HASH,
    GENERATOR_VERSION,
    generate_arc_case,
)
from butterfly_saxs.butterfly import analyze_butterfly  # noqa: E402


METHODS: dict[str, dict[str, Any]] = {
    "annular_peak": {
        "definition": "Angular intensity maxima sampled on successive prescribed q annuli.",
        "comparison_limit": "An annulus is a sampling radius, not a radial reflection peak or ridge normal.",
        "options": {"annular_radial_bins": 48, "annular_angle_bins": 144},
    },
    "radial_sector": {
        "definition": "Radial-profile peak from a finite azimuthal sector, linked across sector centers.",
        "comparison_limit": "This is not a Cartesian x-slice implementation from the literature.",
        "options": {"sector_width_deg": 8.0, "sector_step_deg": 4.0},
    },
    "curvature": {
        "definition": "Multiscale principal-curvature ridge candidates, refined against observed local profiles.",
        "comparison_limit": "No equivalence to a published curvature-tracing implementation is asserted.",
        "options": {},
    },
    "cartesian_slice": {
        "definition": "Observed local intensity maxima along fixed-qy Cartesian detector rows.",
        "comparison_limit": (
            "A simple fixed-coordinate slice baseline on a rectilinear q grid; not a full reproduction "
            "of Murthy and Grubb or another published slice algorithm."
        ),
        "options": {
            "grid_requirement": "rectilinear qx/qy grid; qy constant across each row and qx monotonic across columns",
            "sampling": "observed detector pixels only; no interpolation across mask or q-window gaps",
            "smoothing_sigma_bins": 1.0,
            "min_prominence_sigma": 4.0,
            "min_prominence_fraction": 0.02,
            "min_peak_separation_bins": 2,
            "min_width_bins": 0.8,
            "min_run_samples": 5,
        },
    },
}

METHOD_ORDER = tuple(METHODS)
Q_WINDOW = (0.15, 0.95)
DEFAULT_NOISE_SIGMAS = (0.0, 0.02)
DEFAULT_MASK_VARIANTS = ("none", "wedge")
SCHEMA_VERSION = "trajectory-method-comparison-v2"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_hash(array: Any) -> str:
    data = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(data.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(data.shape), separators=(",", ":")).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def _input_hash(image: np.ndarray, qx: np.ndarray, qy: np.ndarray, invalid: np.ndarray) -> str:
    parts = {
        "image": _array_hash(image),
        "qx": _array_hash(qx),
        "qy": _array_hash(qy),
        "invalid_mask": _array_hash(invalid),
    }
    return _sha256_bytes(json.dumps(parts, sort_keys=True).encode("ascii"))


def _median_pixel_q_step(qx: np.ndarray, qy: np.ndarray) -> float:
    steps: list[np.ndarray] = []
    for axis in (0, 1):
        dx = np.diff(qx, axis=axis)
        dy = np.diff(qy, axis=axis)
        distance = np.hypot(dx, dy)
        finite = distance[np.isfinite(distance) & (distance > 0)]
        if finite.size:
            steps.append(finite)
    if not steps:
        raise ValueError("q map has no finite adjacent-pixel spacing")
    value = float(np.median(np.concatenate(steps)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("q map has invalid adjacent-pixel spacing")
    return value


def _wedge_mask(qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """Mask one paired angular wedge through the generator's narrow lobes."""
    q = np.hypot(qx, qy)
    angle = np.arctan2(qy, qx)
    # 0.08 rad is close to the long axis of the default b/a=0.1 case.
    centers = (0.08, 0.08 + np.pi)
    selected = np.zeros(q.shape, dtype=bool)
    for center in centers:
        delta = np.angle(np.exp(1j * (angle - center)))
        selected |= np.abs(delta) <= np.deg2rad(7.0)
    return selected & (q >= 0.28) & (q <= 0.78)


def _noise_image(base: np.ndarray, sigma: float, seed: int, invalid: np.ndarray) -> np.ndarray:
    image = np.asarray(base, dtype=np.float64).copy()
    if sigma:
        noise = np.random.default_rng(seed).normal(0.0, sigma, size=image.shape)
        valid = np.isfinite(image) & ~invalid
        image[valid] += noise[valid]
    return image


def _truth_arc_samples(
    truth: dict[str, Any],
    qx: np.ndarray,
    qy: np.ndarray,
    invalid: np.ndarray,
    q_window: tuple[float, float],
    spacing: float,
) -> dict[str, np.ndarray]:
    """Densify visible generator polylines in calibrated q coordinates."""
    from scipy.spatial import cKDTree

    detector_q = np.column_stack((qx.ravel(), qy.ravel()))
    finite_detector = np.all(np.isfinite(detector_q), axis=1)
    detector_q = detector_q[finite_detector]
    detector_invalid = invalid.ravel()[finite_detector]
    tree = cKDTree(detector_q)
    output: dict[str, np.ndarray] = {}
    for arc in truth.get("actual_observable_arcs", []):
        polyline = np.asarray(arc.get("polyline_q", []), dtype=np.float64)
        if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] != 2:
            continue
        samples: list[np.ndarray] = []
        for start, stop in zip(polyline[:-1], polyline[1:], strict=True):
            length = float(np.linalg.norm(stop - start))
            count = max(1, int(math.ceil(length / max(spacing, 1.0e-12))))
            samples.append(start + np.linspace(0.0, 1.0, count + 1)[:, None] * (stop - start))
        points = np.unique(np.concatenate(samples, axis=0), axis=0)
        radius = np.hypot(points[:, 0], points[:, 1])
        in_window = (radius >= q_window[0]) & (radius <= q_window[1])
        _, nearest = tree.query(points, k=1)
        visible = in_window & ~detector_invalid[np.asarray(nearest, dtype=int)]
        points = points[visible]
        if points.size:
            output[str(arc["arc_id"])] = points
    return output


def _point_arrays(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    coords: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        try:
            qx, qy = float(point["qx"]), float(point["qy"])
        except (KeyError, TypeError, ValueError):
            qx = qy = float("nan")
        accepted = point.get("accepted") is True and math.isfinite(qx) and math.isfinite(qy)
        row = {
            "source_index": index,
            "point_id": point.get("point_id"),
            "accepted": bool(point.get("accepted") is True),
            "valid": point.get("valid"),
            "qx": qx if math.isfinite(qx) else None,
            "qy": qy if math.isfinite(qy) else None,
            "q_radius": math.hypot(qx, qy) if math.isfinite(qx) and math.isfinite(qy) else None,
            "branch_id": point.get("branch_id"),
            "side": point.get("side"),
            "intensity": point.get("intensity"),
            "snr": point.get("snr"),
            "prominence": point.get("prominence"),
            "reason": point.get("reason"),
            "topology_flags": list(point.get("topology_flags", [])),
            "source_method": point.get("source_method", point.get("method")),
            "accepted_for_localization": accepted,
        }
        rows.append(row)
        if accepted:
            coords.append((qx, qy))
    coord_array = np.asarray(coords, dtype=np.float64).reshape((-1, 2))
    return rows, coord_array


def _role_arc_id(point: dict[str, Any]) -> str | None:
    branch = point.get("branch_id")
    side = point.get("side")
    if branch not in (0, 1) or side not in ("upper", "lower"):
        return None
    return f"{'plus' if branch == 0 else 'minus'}_{side}"


def _distance_to_cloud(query: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    if not len(query) or not len(cloud):
        return np.full(len(query), np.nan, dtype=np.float64)
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(cloud).query(query, k=1)
    return np.asarray(distances, dtype=np.float64)


def _make_comparison_figure(
    output: Path,
    *,
    case_id: str,
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    truth: dict[str, Any],
    invalid: np.ndarray,
    method_rows: dict[str, list[dict[str, Any]]],
    method_records: dict[str, dict[str, Any]],
    input_sha256: str,
) -> dict[str, str]:
    """Render one paired-input overview with true support and each trace."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [
        float(np.nanmin(qx)), float(np.nanmax(qx)),
        float(np.nanmin(qy)), float(np.nanmax(qy)),
    ]
    image_view = np.where(invalid | ~np.isfinite(image), np.nan, image)
    finite_values = image_view[np.isfinite(image_view)]
    color_limit = float(np.percentile(finite_values, 99.5)) if finite_values.size else 1.0
    if not math.isfinite(color_limit) or color_limit <= 0:
        color_limit = 1.0
    colors = {
        "annular_peak": "#0072B2",
        "radial_sector": "#D55E00",
        "curvature": "#009E73",
        "cartesian_slice": "#CC79A7",
    }
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 7.3), constrained_layout=True)
    for axis, method in zip(axes.ravel(), METHOD_ORDER, strict=True):
        axis.pcolormesh(qx, qy, image_view, cmap="gray_r", vmin=0,
                        vmax=color_limit, shading="auto", rasterized=True)
        for arc in truth.get("actual_observable_arcs", []):
            polyline = np.asarray(arc.get("polyline_q", []), dtype=float)
            if len(polyline):
                axis.plot(polyline[:, 0], polyline[:, 1], color="#CC79A7", linewidth=1.0,
                          linestyle="--", alpha=0.85)
        extracted = [row for row in method_rows[method] if row.get("accepted_for_localization")]
        if extracted:
            axis.scatter([row["qx"] for row in extracted], [row["qy"] for row in extracted],
                         s=12, color=colors[method], edgecolors="white", linewidths=0.2,
                         label="accepted observed points", zorder=4)
        record = method_records[method]
        median = record.get("median_point_to_visible_truth_sample_cloud_distance_pixels")
        coverage = record.get("visible_truth_sample_cloud_coverage_within_tolerance")
        median_text = "n/a" if median is None else f"{median:.2f} px"
        coverage_text = "n/a" if coverage is None else f"{100 * coverage:.0f}%"
        title = (
            f"{method}  (n={len(extracted)})\n"
            f"median distance={median_text}; coverage@1.5 px={coverage_text}"
        )
        axis.set_title(title, fontsize=8, pad=4)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("qₓ (nm⁻¹)")
        axis.set_ylabel("qᵧ (nm⁻¹)")
    figure.suptitle(
        f"Four-method paired comparison · {case_id} · added noise sigma={method_records[METHOD_ORDER[0]].get('extra_gaussian_noise_sigma', 0):g} · "
        f"mask={method_records[METHOD_ORDER[0]].get('invalid_mask_variant', 'none')} · input {input_sha256[:12]}\n"
        "Dashed curve: generator-declared observable support; dots: accepted extracted points",
        fontsize=10,
    )
    png = output / "trajectory_method_comparison.png"
    svg = output / "trajectory_method_comparison.svg"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)
    return {"png": png.name, "svg": svg.name}


def score_extraction(
    points: list[dict[str, Any]],
    truth: dict[str, Any],
    qx: np.ndarray,
    qy: np.ndarray,
    invalid: np.ndarray,
    q_window: tuple[float, float],
    pixel_q_step: float,
    tolerance_pixels: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score point localization and visible arc support; keep failures explicit."""
    rows, accepted_coords = _point_arrays(points)
    visible_arcs = _truth_arc_samples(
        truth, qx, qy, invalid, q_window, spacing=0.5 * pixel_q_step
    )
    all_truth = (
        np.concatenate(list(visible_arcs.values()), axis=0)
        if visible_arcs
        else np.empty((0, 2), dtype=np.float64)
    )
    tolerance_q = float(tolerance_pixels) * pixel_q_step
    global_point_distances = _distance_to_cloud(accepted_coords, all_truth)
    for row_index, row in enumerate(rows):
        if not row["accepted_for_localization"] or not len(all_truth):
            row.update(distance_to_visible_truth_samples_q=None,
                       point_to_truth_sample_cloud_distance_pixels=None,
                       nearest_truth_arc_id=None, truth_role_match=None)
            continue
        point = accepted_coords[sum(r["accepted_for_localization"] for r in rows[:row_index + 1]) - 1]
        distances_by_arc = {
            arc_id: float(_distance_to_cloud(point.reshape(1, 2), cloud)[0])
            for arc_id, cloud in visible_arcs.items()
        }
        nearest_arc = min(distances_by_arc, key=distances_by_arc.get)
        error_q = distances_by_arc[nearest_arc]
        assigned_arc = _role_arc_id(row)
        row.update(
            distance_to_visible_truth_samples_q=error_q,
            point_to_truth_sample_cloud_distance_pixels=error_q / pixel_q_step,
            nearest_truth_arc_id=nearest_arc,
            truth_role_match=(assigned_arc == nearest_arc) if assigned_arc else None,
        )

    matched_errors = np.asarray(
        [row["point_to_truth_sample_cloud_distance_pixels"] for row in rows
         if row.get("point_to_truth_sample_cloud_distance_pixels") is not None], dtype=np.float64
    )
    per_arc: dict[str, Any] = {}
    for arc_id, cloud in visible_arcs.items():
        role_points = np.asarray(
            [
                (float(row["qx"]), float(row["qy"]))
                for row in rows
                if row["accepted_for_localization"] and _role_arc_id(row) == arc_id
            ],
            dtype=np.float64,
        ).reshape((-1, 2))
        any_distances = _distance_to_cloud(cloud, accepted_coords)
        role_distances = _distance_to_cloud(cloud, role_points)
        per_arc[arc_id] = {
            "visible_truth_samples": int(len(cloud)),
            "assigned_accepted_points": int(len(role_points)),
            "coverage_any_point_within_tolerance": (
                float(np.mean(any_distances <= tolerance_q)) if len(any_distances) else None
            ),
            "coverage_assigned_points_within_tolerance": (
                float(np.mean(role_distances <= tolerance_q)) if len(role_distances) else 0.0
            ),
        }
    summary = {
        "candidate_point_count": len(points),
        "accepted_point_count": int(len(accepted_coords)),
        "rejected_point_count": int(len(points) - len(accepted_coords)),
        "generator_arc_count": len(truth.get("actual_observable_arcs", [])),
        "visible_arc_count": len(visible_arcs),
        "visible_truth_sample_count": int(len(all_truth)),
        "localization_tolerance_pixels": float(tolerance_pixels),
        "localization_tolerance_q": tolerance_q,
        "median_point_to_visible_truth_sample_cloud_distance_pixels": (
            float(np.median(matched_errors)) if matched_errors.size else None
        ),
        "p95_point_to_visible_truth_sample_cloud_distance_pixels": (
            float(np.percentile(matched_errors, 95)) if matched_errors.size else None
        ),
        "fraction_points_within_sample_cloud_tolerance": (
            float(np.mean(global_point_distances <= tolerance_q))
            if global_point_distances.size and len(all_truth) else None
        ),
        "visible_truth_sample_cloud_coverage_within_tolerance": (
            float(np.mean(_distance_to_cloud(all_truth, accepted_coords) <= tolerance_q))
            if len(all_truth) and len(accepted_coords) else (0.0 if len(all_truth) else None)
        ),
        "unassigned_or_mismatched_role_points": int(sum(
            row["accepted_for_localization"] and row.get("truth_role_match") is not True
            for row in rows
        )),
        "per_truth_arc": per_arc,
    }
    return summary, rows


def _trace_options(method: str) -> dict[str, Any]:
    return {"stage": "trace", "resamples": 0, "sensitivity": False,
            "trace_method": method, **METHODS[method]["options"]}


def _cartesian_slice_points(
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    invalid: np.ndarray,
    q_window: tuple[float, float],
    pixel_q_step: float,
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    """Find observed local maxima on fixed-qy detector rows, without gap filling."""
    from scipy.signal import find_peaks

    from butterfly_saxs.sector_peaks import (
        _contiguous_runs,
        _profile_noise,
        _robust_raw_baseline,
        _smooth_supported,
    )

    image = np.asarray(image, dtype=np.float64)
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    invalid = np.asarray(invalid, dtype=bool)
    if image.ndim != 2 or image.shape != qx.shape or qx.shape != qy.shape or qy.shape != invalid.shape:
        raise ValueError("cartesian_slice requires matching two-dimensional image, qx, qy and mask arrays")
    if min(image.shape) < 2:
        raise ValueError("cartesian_slice requires at least two detector rows and columns")

    grid_tolerance = max(1.0e-12, float(pixel_q_step) * 1.0e-6)
    expected_qx = np.broadcast_to(qx[0:1, :], qx.shape)
    expected_qy = np.broadcast_to(qy[:, 0:1], qy.shape)
    if not np.all(np.isfinite(qx)) or not np.all(np.isfinite(qy)):
        raise ValueError("cartesian_slice requires a finite rectilinear q grid")
    if not np.allclose(qx, expected_qx, rtol=0.0, atol=grid_tolerance):
        raise ValueError("cartesian_slice requires qx to be constant down detector columns")
    if not np.allclose(qy, expected_qy, rtol=0.0, atol=grid_tolerance):
        raise ValueError("cartesian_slice requires qy to be constant across each detector row")
    qx_axis = qx[0]
    qx_steps = np.diff(qx_axis)
    if not (np.all(qx_steps > 0.0) or np.all(qx_steps < 0.0)):
        raise ValueError("cartesian_slice requires qx to be strictly monotonic across detector columns")
    if len(np.unique(qy[:, 0])) != qy.shape[0]:
        raise ValueError("cartesian_slice requires distinct fixed-qy detector rows")

    method_options = METHODS["cartesian_slice"]["options"]
    smoothing_sigma = float(method_options["smoothing_sigma_bins"])
    min_prominence_sigma = float(method_options["min_prominence_sigma"])
    min_prominence_fraction = float(method_options["min_prominence_fraction"])
    min_distance = int(method_options["min_peak_separation_bins"])
    min_width = float(method_options["min_width_bins"])
    min_run_samples = int(method_options["min_run_samples"])
    points: list[dict[str, Any]] = []
    supported_run_count = 0
    short_run_count = 0
    peak_candidate_count = 0
    rejected_peak_count = 0

    for row_index, qy_target in enumerate(qy[:, 0]):
        row_qx = qx[row_index].copy()
        row_qy = qy[row_index].copy()
        profile = image[row_index].copy()
        row_invalid = invalid[row_index].copy()
        detector_columns = np.arange(image.shape[1], dtype=int)
        if qx_steps[0] < 0:
            row_qx = row_qx[::-1]
            row_qy = row_qy[::-1]
            profile = profile[::-1]
            row_invalid = row_invalid[::-1]
            detector_columns = detector_columns[::-1]
        q_radius = np.hypot(row_qx, row_qy)
        supported = (
            ~row_invalid & np.isfinite(profile) & np.isfinite(row_qx) & np.isfinite(row_qy)
            & (q_radius >= q_window[0]) & (q_radius <= q_window[1])
        )
        for run_index, (left, right) in enumerate(_contiguous_runs(supported)):
            if right - left < min_run_samples:
                short_run_count += 1
                continue
            supported_run_count += 1
            raw_segment = profile[left:right]
            all_supported = np.ones(raw_segment.shape, dtype=bool)
            smooth_segment = _smooth_supported(raw_segment, all_supported, smoothing_sigma)
            noise = _profile_noise(raw_segment, smooth_segment, all_supported)
            span = max(float(np.ptp(raw_segment)), np.finfo(float).eps)
            prominence_threshold = max(
                min_prominence_sigma * noise,
                min_prominence_fraction * span,
            )
            baseline = _robust_raw_baseline(raw_segment)
            peak_indices, properties = find_peaks(
                smooth_segment,
                prominence=prominence_threshold,
                distance=min_distance,
                width=min_width,
            )
            prominences = np.asarray(properties.get("prominences", []), dtype=float)
            widths = np.asarray(properties.get("widths", []), dtype=float)
            for peak_order, (peak_index, prominence, width) in enumerate(
                zip(peak_indices, prominences, widths, strict=True)
            ):
                peak_candidate_count += 1
                local_index = int(peak_index)
                global_index = left + local_index
                peak_height = float(smooth_segment[local_index] - baseline)
                accepted = peak_height >= min_prominence_sigma * noise
                if not accepted:
                    rejected_peak_count += 1
                detector_column = int(detector_columns[global_index])
                point_id = f"slice-r{row_index}-s{run_index}-p{peak_order}"
                points.append({
                    "point_id": point_id,
                    "qx": float(row_qx[global_index]),
                    "qy": float(row_qy[global_index]),
                    "slice_qy_target": float(qy_target),
                    "slice_qy_deviation_q": float(row_qy[global_index] - qy_target),
                    "detector_row": row_index,
                    "detector_column": detector_column,
                    "intensity": float(profile[global_index]),
                    "smoothed_intensity": float(smooth_segment[local_index]),
                    "prominence": float(prominence),
                    "prominence_threshold": float(prominence_threshold),
                    "profile_noise_estimate": float(noise),
                    "snr": float(prominence / max(noise, np.finfo(float).eps)),
                    "width_bins": float(width),
                    "baseline": float(baseline),
                    "accepted": bool(accepted),
                    "valid": bool(accepted),
                    "reason": None if accepted else "peak_height_below_profile_noise_gate",
                    "source_method": "cartesian_slice",
                })

    diagnostics: dict[str, int | str] = {
        "fixed_qy_row_count": int(qy.shape[0]),
        "supported_run_count": supported_run_count,
        "short_run_count": short_run_count,
        "peak_candidate_count": peak_candidate_count,
        "rejected_peak_count": rejected_peak_count,
        "sampling_rule": "fixed detector rows; observed pixels only; gaps are not bridged",
    }
    return points, diagnostics


def analyze_variant(
    *,
    case_id: str,
    truth: dict[str, Any],
    image: np.ndarray,
    qmap: Any,
    invalid: np.ndarray,
    q_window: tuple[float, float] = Q_WINDOW,
    tolerance_pixels: float = 1.5,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Run all methods on one exact image/q-map/mask tuple."""
    qx, qy = np.asarray(qmap.qx), np.asarray(qmap.qy)
    pitch = _median_pixel_q_step(qx, qy)
    input_sha256 = _input_hash(image, qx, qy, invalid)
    records: list[dict[str, Any]] = []
    point_rows: dict[str, list[dict[str, Any]]] = {}
    for method in METHOD_ORDER:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "case_id": case_id,
            "method": method,
            "same_input_sha256": input_sha256,
            "q_window": list(q_window),
            "q_unit": getattr(qmap, "q_unit", "unknown"),
            "median_pixel_q_step": pitch,
            "method_options": METHODS[method]["options"],
        }
        try:
            if method == "cartesian_slice":
                points, slice_diagnostics = _cartesian_slice_points(
                    image, qx, qy, invalid, q_window, pitch
                )
                method_version = "fixed-qy-cartesian-slice-v1"
                extraction_method = "fixed-qy detector-row local maxima"
            else:
                result = analyze_butterfly(
                    image.copy(), qmap, q_window, mask=invalid.copy(),
                    options=_trace_options(method), reference_axis_deg=0.0,
                )
                points = list(result.get("points", []))
                method_version = result.get("method_version")
                extraction_method = result.get("diagnostics", {}).get("method")
                slice_diagnostics = None
            score, rows = score_extraction(
                points, truth, qx, qy, invalid, q_window, pitch, tolerance_pixels
            )
            record.update(
                execution_status="complete",
                method_version=method_version,
                extraction_method=extraction_method,
                elapsed_s=float(time.perf_counter() - started),
                method_diagnostics=slice_diagnostics,
                **score,
            )
            point_rows[method] = rows
        except Exception as exc:  # preserve a method failure alongside peer results
            record.update(
                execution_status="error",
                elapsed_s=float(time.perf_counter() - started),
                error=f"{type(exc).__name__}: {exc}",
                candidate_point_count=0,
                accepted_point_count=0,
                rejected_point_count=0,
                generator_arc_count=len(truth.get("actual_observable_arcs", [])),
                visible_arc_count=None,
                visible_truth_sample_count=None,
                median_point_to_visible_truth_sample_cloud_distance_pixels=None,
                p95_point_to_visible_truth_sample_cloud_distance_pixels=None,
                fraction_points_within_sample_cloud_tolerance=None,
                visible_truth_sample_cloud_coverage_within_tolerance=None,
                per_truth_arc={},
            )
            point_rows[method] = []
        records.append(record)
    return records, point_rows


def _parse_csv_values(value: str, *, cast: Any, name: str) -> tuple[Any, ...]:
    try:
        parsed = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return parsed


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not fieldnames:
        fieldnames = ["record_status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _code_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        REPO / "src/butterfly_saxs/benchmark_arcs.py",
        REPO / "src/butterfly_saxs/butterfly.py",
        REPO / "src/butterfly_saxs/butterfly_ridge.py",
        REPO / "src/butterfly_saxs/annular_trace.py",
        REPO / "src/butterfly_saxs/sector_trace.py",
        REPO / "src/butterfly_saxs/sector_peaks.py",
        REPO / "src/butterfly_saxs/butterfly_settings.py",
    )
    return {
        path.relative_to(REPO).as_posix(): _sha256_bytes(path.read_bytes())
        for path in paths
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    generated_names = {"summary.json", "summary.csv", "method_summary.csv", "points.csv", "truth.json",
                       "figure_caption.md", "benchmark_summary.md",
                       "trajectory_method_comparison.png", "trajectory_method_comparison.svg"}
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output path is not a directory: {output}")
    if output.exists():
        existing_files = [path for path in output.rglob("*") if path.is_file()]
        if existing_files and not args.force:
            raise FileExistsError(f"choose a new output directory or pass --force: {output}")
        unexpected = []
        for path in existing_files:
            relative = path.relative_to(output)
            if relative.parts == ("inputs",) or relative.parts == ("points",):
                unexpected.append(path)
            elif relative.parts[0] == "inputs" and path.suffix == ".npz":
                continue
            elif relative.parts[0] == "points" and path.suffix == ".csv":
                continue
            elif relative.parts == (path.name,) and path.name in generated_names:
                continue
            else:
                unexpected.append(path)
        if unexpected:
            raise FileExistsError(f"--force can replace only files created by this benchmark; unexpected file: {unexpected[0]}")
        for path in existing_files:
            path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    input_dir = output / "inputs"
    points_dir = output / "points"
    input_dir.mkdir(exist_ok=True)
    points_dir.mkdir(exist_ok=True)

    case_ids = tuple(args.cases)
    noise_sigmas = tuple(args.noise_sigmas)
    mask_variants = tuple(args.mask_variants)
    if any(not math.isfinite(value) or value < 0 for value in noise_sigmas):
        raise ValueError("noise sigma values must be finite and non-negative")
    if any(value not in {"none", "wedge"} for value in mask_variants):
        raise ValueError("mask variants must be 'none' or 'wedge'")
    q_window = tuple(args.q_window)
    if len(q_window) != 2 or not 0 <= q_window[0] < q_window[1]:
        raise ValueError("q window must contain two increasing non-negative values")

    start_time = datetime.now(timezone.utc).isoformat()
    before_hashes = _code_hashes()
    records: list[dict[str, Any]] = []
    all_point_rows: list[dict[str, Any]] = []
    pooled_errors: dict[str, list[float]] = {method: [] for method in METHOD_ORDER}
    figure_data: tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]],
                       dict[str, dict[str, Any]], str] | None = None
    for case_index, case_id in enumerate(case_ids):
        case = generate_arc_case(case_id, shape=(args.shape, args.shape))
        truth = case["truth"]
        base_image = np.asarray(case["image"], dtype=np.float64)
        qmap = case["qmap"]
        for noise_index, sigma in enumerate(noise_sigmas):
            noise_seed = int(truth["seed"] + 10_000 + noise_index)
            noisy_image = _noise_image(base_image, sigma, noise_seed, np.asarray(case["mask"]))
            for mask_variant in mask_variants:
                invalid = np.asarray(case["mask"], dtype=bool).copy()
                if mask_variant == "wedge":
                    invalid |= _wedge_mask(qmap.qx, qmap.qy)
                digest = _input_hash(noisy_image, qmap.qx, qmap.qy, invalid)
                key = f"{case_id}__noise-{sigma:g}__mask-{mask_variant}"
                input_path = input_dir / f"{key}.npz"
                np.savez_compressed(
                    input_path, image=noisy_image, qx=qmap.qx, qy=qmap.qy,
                    mask=invalid, q_unit=np.asarray(qmap.q_unit),
                )
                case_record_start = len(records)
                variant_records, variant_point_rows = analyze_variant(
                    case_id=case_id, truth=truth, image=noisy_image, qmap=qmap,
                    invalid=invalid, q_window=q_window,
                    tolerance_pixels=args.tolerance_pixels,
                )
                for record in variant_records:
                    record.update(
                        category=truth.get("category"),
                        truth_geometry=("none" if truth.get("category") == "null" else
                                        "smooth_nonellipse" if truth.get("category") == "non_elliptic" else "ellipse"),
                        seed=int(truth["seed"]),
                        extra_gaussian_noise_sigma=float(sigma),
                        extra_noise_seed=noise_seed,
                        invalid_mask_variant=mask_variant,
                        input_file=f"inputs/{input_path.name}",
                        image_qmask_sha256=digest,
                        input_file_sha256=_sha256_bytes(input_path.read_bytes()),
                        generator_version=truth.get("generator_version"),
                        generator_sha256=truth.get("generator_sha256"),
                    )
                    records.append(record)
                    for point_row in variant_point_rows[record["method"]]:
                        point_record = {
                            "case_id": case_id,
                            "method": record["method"],
                            "extra_gaussian_noise_sigma": float(sigma),
                            "invalid_mask_variant": mask_variant,
                            "input_sha256": digest,
                            **point_row,
                        }
                        all_point_rows.append(point_record)
                        if point_row.get("point_to_truth_sample_cloud_distance_pixels") is not None:
                            pooled_errors[record["method"]].append(
                                float(point_row["point_to_truth_sample_cloud_distance_pixels"])
                            )
                for method in METHOD_ORDER:
                    file_stem = f"{key}__{method}"
                    method_points = variant_point_rows[method]
                    fields = list(method_points[0]) if method_points else [
                        "source_index", "point_id", "accepted", "qx", "qy"
                    ]
                    _write_csv(points_dir / f"{file_stem}.csv", method_points, fields)
                if figure_data is None:
                    figure_data = (
                        {"case_id": case_id, "truth": truth},
                        {"image": noisy_image, "qx": qmap.qx, "qy": qmap.qy, "invalid": invalid},
                        variant_point_rows,
                        {row["method"]: {**row, "extra_gaussian_noise_sigma": sigma,
                                          "invalid_mask_variant": mask_variant}
                         for row in variant_records},
                        digest,
                    )
                if len(records) != case_record_start + len(METHOD_ORDER):
                    raise RuntimeError("paired method records were not complete")
                print(f"{key}: " + ", ".join(
                    f"{row['method']}={row.get('accepted_point_count', 0)}" for row in variant_records
                ), file=sys.stderr, flush=True)

    after_hashes = _code_hashes()
    errors_by_method = {
        method: sum(row["method"] == method and row["execution_status"] == "error" for row in records)
        for method in METHOD_ORDER
    }
    method_summaries = []
    for method in METHOD_ORDER:
        rows = [row for row in records if row["method"] == method]
        errors = np.asarray(pooled_errors[method], dtype=np.float64)
        method_summaries.append({
            "method": method,
            "paired_variants": len(rows),
            "execution_errors": errors_by_method[method],
            "total_accepted_points": sum(int(row.get("accepted_point_count", 0)) for row in rows),
            "pooled_localized_points": int(errors.size),
            "pooled_median_point_to_truth_sample_cloud_distance_pixels": float(np.median(errors)) if errors.size else None,
            "pooled_p95_point_to_truth_sample_cloud_distance_pixels": float(np.percentile(errors, 95)) if errors.size else None,
            "mean_variant_visible_truth_sample_cloud_coverage_at_tolerance": float(np.mean([
                row["visible_truth_sample_cloud_coverage_within_tolerance"]
                for row in rows if row.get("visible_truth_sample_cloud_coverage_within_tolerance") is not None
            ])) if any(row.get("visible_truth_sample_cloud_coverage_within_tolerance") is not None for row in rows) else None,
            "total_null_case_accepted_points": sum(
                int(row.get("accepted_point_count", 0))
                for row in rows if row.get("category") == "null"
            ),
        })

    figure_files = None
    if figure_data is not None:
        figure_case, figure_input, figure_points, figure_records, figure_hash = figure_data
        figure_files = _make_comparison_figure(
            output, case_id=figure_case["case_id"], image=figure_input["image"],
            qx=figure_input["qx"], qy=figure_input["qy"], truth=figure_case["truth"],
            invalid=figure_input["invalid"], method_rows=figure_points,
            method_records=figure_records, input_sha256=figure_hash,
        )
        (output / "figure_caption.md").write_text(
            "Paired comparison of three WingSAXS trace methods and one fixed-qy Cartesian slice baseline "
            "on one generated SAXS-like image. Each panel overlays the same generator-declared visible "
            "support as dashed curves and that method's accepted observed points. The baseline uses "
            "rectilinear detector rows, contiguous supported segments and recorded prominence/noise gates; "
            "it is not a full reproduction of a published slice algorithm. "
            "Panel annotations report accepted-point count, median nearest-reference distance "
            "in median pixel-q spacings, and the fraction of visible truth samples covered within the "
            "operational 1.5-pixel tolerance. This is a synthetic localization illustration, not a real-data "
            "validation or a comparison to published Cartesian-slice or curvature implementations.\n",
            encoding="utf-8",
        )

    context = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": start_time,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy")
        },
        "generator": {
            "name": "butterfly_saxs.benchmark_arcs.generate_arc_case",
            "version": GENERATOR_VERSION,
            "source_sha256": GENERATOR_HASH,
            "truth_scope": "known observable generator arcs; not a physical SAXS forward model",
        },
        "analysis": {
            "methods": METHODS,
            "method_order": list(METHOD_ORDER),
            "q_window": list(q_window),
            "q_unit": "nm^-1",
            "shape": [args.shape, args.shape],
            "stage": "trace",
            "ellipse_fit": False,
            "uncertainty_resampling": False,
            "localization_tolerance_pixels": args.tolerance_pixels,
            "localization_tolerance_definition": "tolerance_pixels multiplied by median adjacent-pixel q spacing; operational synthetic benchmark threshold, not a calibrated resolution or confidence interval",
            "extra_gaussian_noise_sigmas": list(noise_sigmas),
            "generator_baseline_noise": "benchmark_arcs adds Gaussian image noise with sigma 0.004, or 0.002 for the null case; sigma=0 below means no additional noise was added, not a noiseless image",
            "mask_variants": list(mask_variants),
            "mask_true_means": "excluded or invalid detector pixel",
        },
        "case_ids": list(case_ids),
        "source_sha256": before_hashes,
        "source_unchanged": before_hashes == after_hashes,
        "input_identity_rule": "all four methods receive the same image/q-map/invalid-mask hashes and q window within each variant",
        "aggregation_note": "Pooled point-distance quantiles weight accepted localized points equally, so denser method outputs contribute more points; mean variant coverage weights each eligible image variant equally.",
        "result_files": {
            "summary": "summary.json",
            "summary_csv": "summary.csv",
            "method_summary_csv": "method_summary.csv",
            "summary_markdown": "benchmark_summary.md",
            "point_records": "points/*.csv",
            "saved_inputs": "inputs/*.npz",
            "comparison_figure": figure_files,
        },
        "limitations": [
            "Synthetic image validation only; no real experimental sequence was provided.",
            "The generator rasterizes known arcs and is not an independent physical scattering model.",
            "The radial_sector method is an azimuthal-sector radial-profile method, distinct from the fixed-qy Cartesian baseline.",
            "The cartesian_slice method is a simple fixed-qy detector-row baseline requiring a rectilinear q grid; it is not a full reproduction of a published slice method.",
            "This benchmark does not establish equivalence to any published curvature method.",
            "Point-localization tolerance is an operational pixel-scale threshold, not an accepted scientific tolerance.",
            "No uncertainty-interval coverage or parameter identifiability is assessed here.",
        ],
    }
    report = {"context": context, "method_summary": method_summaries, "results": records}
    _json_dump(output / "summary.json", report)
    summary_fields = list(dict.fromkeys(key for row in records for key, value in row.items()
                                        if not isinstance(value, (dict, list))))
    _write_csv(output / "summary.csv", records, summary_fields)
    _write_csv(output / "method_summary.csv", method_summaries, list(method_summaries[0]))
    _write_csv(output / "points.csv", all_point_rows,
               list(dict.fromkeys(key for row in all_point_rows for key in row)))
    _json_dump(output / "truth.json", {
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": GENERATOR_HASH,
        "case_truth": {
            case_id: generate_arc_case(case_id, shape=(args.shape, args.shape))["truth"]
            for case_id in case_ids
        },
    })
    summary_markdown = [
        "# Paired trajectory-method benchmark results",
        "",
        f"The run compared {len(records) // len(METHOD_ORDER)} paired image variants across {len(case_ids)} "
        "synthetic cases. Each method received the same image, q map, invalid-pixel mask, and q window "
        "within a variant. The trace stage ran without ellipse fitting or uncertainty resampling.",
        "",
        "| Method | Variants | Execution errors | Accepted points | Localized points | Pooled median distance (px) | Pooled P95 (px) | Mean variant coverage | Null detections |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def format_summary_value(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    for row in method_summaries:
        summary_markdown.append(
            f"| {row['method']} | {row['paired_variants']} | {row['execution_errors']} | "
            f"{row['total_accepted_points']} | {row['pooled_localized_points']} | "
            f"{format_summary_value(row['pooled_median_point_to_truth_sample_cloud_distance_pixels'])} | "
            f"{format_summary_value(row['pooled_p95_point_to_truth_sample_cloud_distance_pixels'])} | "
            f"{format_summary_value(row['mean_variant_visible_truth_sample_cloud_coverage_at_tolerance'])} | "
            f"{row['total_null_case_accepted_points']} |"
        )
    summary_markdown.extend([
        "",
        "Distances are nearest-neighbor distances to generator-declared visible truth samples, densified "
        "at half the median adjacent-pixel q spacing. They are not exact point-to-segment normal distances. "
        "The default 1.5-pixel coverage tolerance is an operational synthetic threshold, not an instrument "
        "resolution, scientific acceptance criterion, or confidence interval.",
        "",
        "Pooled point-distance quantiles weight each accepted localized point equally, so methods returning "
        "more points contribute more heavily. Mean coverage gives equal weight to eligible image variants. "
        "Null detections count accepted output points on background-only inputs. These summaries are "
        "descriptive and do not establish universal method superiority.",
        "",
        "This run is synthetic only. The Cartesian literature slice comparator is not implemented, and no "
        "equivalence to a published curvature method is claimed.",
        "",
    ])
    (output / "benchmark_summary.md").write_text("\n".join(summary_markdown), encoding="utf-8")
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASE_IDS),
                        help="comma-separated canonical arc benchmark case IDs")
    parser.add_argument("--shape", type=int, default=96)
    parser.add_argument("--q-window", type=float, nargs=2, default=Q_WINDOW, metavar=("QMIN", "QMAX"))
    parser.add_argument("--noise-sigmas", default=",".join(str(x) for x in DEFAULT_NOISE_SIGMAS),
                        help="comma-separated additional Gaussian noise SDs in image-intensity units")
    parser.add_argument("--mask-variants", default=",".join(DEFAULT_MASK_VARIANTS),
                        help="comma-separated mask variants: none,wedge")
    parser.add_argument("--tolerance-pixels", type=float, default=1.5)
    parser.add_argument("--force", action="store_true",
                        help="replace files previously generated by this benchmark in the output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    args.cases = _parse_csv_values(args.cases, cast=str, name="case IDs")
    unknown = set(args.cases) - set(DEFAULT_CASE_IDS)
    if unknown:
        parser.error(f"unknown case IDs: {', '.join(sorted(unknown))}")
    args.noise_sigmas = _parse_csv_values(args.noise_sigmas, cast=float, name="noise sigmas")
    args.mask_variants = _parse_csv_values(args.mask_variants, cast=str, name="mask variants")
    if args.shape < 48:
        parser.error("shape must be at least 48 pixels per side")
    if not math.isfinite(args.tolerance_pixels) or args.tolerance_pixels <= 0:
        parser.error("tolerance-pixels must be positive and finite")
    try:
        report = run_benchmark(args)
    except Exception as exc:
        parser.exit(2, f"benchmark failed: {type(exc).__name__}: {exc}\n")
    execution_errors = sum(row["execution_status"] == "error" for row in report["results"])
    print(json.dumps({
        "summary": str((args.output.resolve() / "summary.json")),
        "paired_variant_count": len(report["results"]) // len(METHOD_ORDER),
        "execution_errors": execution_errors,
        "source_unchanged": report["context"]["source_unchanged"],
        "method_summary": report["method_summary"],
    }, ensure_ascii=True, allow_nan=False))
    return 0 if report["context"]["source_unchanged"] and execution_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
