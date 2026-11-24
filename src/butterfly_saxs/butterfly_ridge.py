"""Independent observed ridge and arc extraction for butterfly SAXS images.

This module orchestrates multiscale surface curvature ridge detection,
first-order radial I(q) hint reconciliation, normal-profile refinement, and
observed q-space support freezing via ``butterfly_saxs.tracing``.
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

import numpy as np

from ._primitives import safe_gradient
from .ridge_inputs import (
    as_image as _as_image,
    as_qmap as _as_qmap,
    coordinate_derivatives as _shared_coordinate_derivatives,
)
from .tracing.curvature import (
    CANDIDATE_NOISE_EDGE_TRUNCATE,
    CANDIDATE_NOISE_METHOD,
    CANDIDATE_NOISE_MIN_SAFE_SAMPLES,
    _CurvatureField as _CurvatureField,
    _annotate_scale_stability,
    _check_cancelled,
    _finite_scale_stability_summary,
    _masked_highpass_noise as _masked_highpass_noise,
    _nms,
    _offset_nearest as _offset_nearest,
    _raw_candidates,
    _scaled_surface_field,
)
from .tracing.radial_hint import (
    FIRST_ORDER_PEAK_AMBIGUITY_MIN_WIDTH_BINS as FIRST_ORDER_PEAK_AMBIGUITY_MIN_WIDTH_BINS,
    FIRST_ORDER_PEAK_AMBIGUITY_NOISE_MULTIPLE as FIRST_ORDER_PEAK_AMBIGUITY_NOISE_MULTIPLE,
    FIRST_ORDER_PEAK_FAMILY_SPAN as FIRST_ORDER_PEAK_FAMILY_SPAN,
    FIRST_ORDER_PEAK_MIN_BIN_COUNT as FIRST_ORDER_PEAK_MIN_BIN_COUNT,
    FIRST_ORDER_PEAK_MIN_CONTIGUOUS_RUN_BINS as FIRST_ORDER_PEAK_MIN_CONTIGUOUS_RUN_BINS,
    FIRST_ORDER_PEAK_MIN_EFFECTIVE_SAMPLES as FIRST_ORDER_PEAK_MIN_EFFECTIVE_SAMPLES,
    FIRST_ORDER_PEAK_MIN_TOTAL_SUPPORTED_BINS as FIRST_ORDER_PEAK_MIN_TOTAL_SUPPORTED_BINS,
    FIRST_ORDER_PEAK_NOISE_SIGMA_BINS as FIRST_ORDER_PEAK_NOISE_SIGMA_BINS,
    FIRST_ORDER_PEAK_PROMINENCE_NOISE_MULTIPLE as FIRST_ORDER_PEAK_PROMINENCE_NOISE_MULTIPLE,
    RADIAL_HINT_CACHE_MAX_BYTES as RADIAL_HINT_CACHE_MAX_BYTES,
    RADIAL_HINT_MAX_SMOOTHING_SIGMA_BINS as RADIAL_HINT_MAX_SMOOTHING_SIGMA_BINS,
    RADIAL_HINT_SMOOTHING_FWHM_Q_PIXEL_FRACTION as RADIAL_HINT_SMOOTHING_FWHM_Q_PIXEL_FRACTION,
    _RadialHintGeometryCache as _RadialHintGeometryCache,
    _angular_delta_deg as _angular_delta_deg,
    _apply_hint_independent_arc_support,
    _axis_deviation_deg as _axis_deviation_deg,
    _circular_angular_span_deg as _circular_angular_span_deg,
    _contiguous_profile_runs as _contiguous_profile_runs,
    _demote_secondary_radial_population,
    _effective_radial_bin_samples as _effective_radial_bin_samples,
    _fill_sparse_first_order_ring,
    _first_order_q_hint,
    _next_arc_id as _next_arc_id,
    _prefer_first_order_scores,
    _prepare_sparse_first_order_samples as _prepare_sparse_first_order_samples,
    _radial_profile_noise_mad as _radial_profile_noise_mad,
    _radial_q_pixel_step as _radial_q_pixel_step,
    _refine_radial_peak_q as _refine_radial_peak_q,
    _sector_first_order_peak as _sector_first_order_peak,
    _select_first_order_peak_index as _select_first_order_peak_index,
    _sparse_first_order_coverage as _sparse_first_order_coverage,
)
from .tracing.topology import (
    _UnionFind as _UnionFind,
    _apply_edits,
    _apply_seeds,
    _arc_topology,
    _assign_component_identities as _assign_component_identities,
    _assign_point_ids,
    _assign_reference_branches,
    _attach_unresolved_to_nearest_arc as _attach_unresolved_to_nearest_arc,
    _build_identity_arc as _build_identity_arc,
    _component_major_direction as _component_major_direction,
    _component_tangent_errors as _component_tangent_errors,
    _demote_unresolved_identity_point as _demote_unresolved_identity_point,
    _graph_arcs,
    _normalise_options,
    _ordered_component as _ordered_component,
    _parse_q_window,
    _point_identity_key as _point_identity_key,
    _point_signature,
    _polygon_mask as _polygon_mask,
    _public_point,
    _quadrant_branch,
    _refresh_arc_identity,
    _side_axis_threshold,
    _side_from_reference_axis as _side_from_reference_axis,
    _split_component_identities as _split_component_identities,
)

_safe_gradient = safe_gradient
METHOD_VERSION = "butterfly-observed-ridges-v1.3"


def _coordinate_derivatives(
    qx: np.ndarray, qy: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Return detector-to-q derivatives, local inverse Jacobian, q step."""

    return _shared_coordinate_derivatives(qx, qy)


def _apply_profile_refinement(
    point: dict[str, Any],
    profile: Mapping[str, Any],
    arcs_by_id: Mapping[int, Mapping[str, Any]],
    options: Mapping[str, Any],
    q_step: float,
) -> None:
    """Apply an observed single-peak normal shift with consistency gates.

    The curvature zero is retained as ``curvature_seed_*``.  A profile fit is
    allowed to move the public point only when it is a supported, unambiguous
    nearby observation and the shift preserves the pre-fit branch/quadrant
    and side assignment.  Failed or ambiguous profiles remain visible without
    promoting a seed to an accepted observation.
    """

    point.setdefault("curvature_seed_qx", float(point.get("qx", float("nan"))))
    point.setdefault("curvature_seed_qy", float(point.get("qy", float("nan"))))
    point.setdefault("curvature_seed_pixel_x", float(point.get("pixel_x", float("nan"))))
    point.setdefault("curvature_seed_pixel_y", float(point.get("pixel_y", float("nan"))))
    point.setdefault("profile_refinement_applied", False)
    point.setdefault("profile_shift_q", 0.0)
    point.setdefault("profile_shift_pixel", 0.0)
    point.setdefault("profile_refinement_reason", "not_attempted")
    point.setdefault("profile_center_offset_q", float("nan"))
    if not bool(options.get("profile_refinement", True)):
        point["profile_refinement_reason"] = "disabled_by_options"
        return
    if not bool(point.get("accepted", False)):
        point["profile_refinement_reason"] = "curvature_candidate_not_accepted"
        return
    if not bool(profile.get("valid", False)):
        point["profile_refinement_reason"] = "profile_unavailable"
        return
    if bool(profile.get("ambiguous", False)) or int(profile.get("peak_count", 0)) != 1:
        point["profile_refinement_reason"] = "ambiguous_or_multiple_normal_peaks"
        point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["profile_refinement_not_applied_ambiguous"]))
        return
    profile_snr = float(profile.get("snr", float("nan")))
    if not np.isfinite(profile_snr) or profile_snr < float(options["profile_refinement_min_snr"]):
        point["profile_refinement_reason"] = "profile_snr_below_refinement_floor"
        return
    support_fraction = float(profile.get("support_fraction", 0.0))
    if not np.isfinite(support_fraction) or support_fraction < float(options["support_min"]):
        point["profile_refinement_reason"] = "profile_support_below_refinement_floor"
        return
    offset_q = float(profile.get("peak_center_q", float("nan")))
    step = float(point.get("q_normal_step", q_step))
    max_shift = float(options["profile_refinement_max_q_step"]) * max(step, q_step, np.finfo(float).eps)
    if not np.isfinite(offset_q) or abs(offset_q) > max_shift:
        point["profile_refinement_reason"] = "profile_shift_exceeds_local_scale"
        point["profile_center_offset_q"] = offset_q
        return
    normal_x = float(point.get("normal_qx", float("nan")))
    normal_y = float(point.get("normal_qy", float("nan")))
    normal_norm = float(np.hypot(normal_x, normal_y))
    if not np.isfinite(normal_norm) or normal_norm <= np.finfo(float).eps:
        point["profile_refinement_reason"] = "profile_shift_missing_unit_normal"
        return
    normal_x, normal_y = normal_x / normal_norm, normal_y / normal_norm
    old_qx, old_qy = float(point["qx"]), float(point["qy"])
    new_qx = float(profile.get("peak_center_qx", float("nan")))
    new_qy = float(profile.get("peak_center_qy", float("nan")))
    if not np.isfinite(new_qx) or not np.isfinite(new_qy):
        new_qx, new_qy = old_qx + offset_q * normal_x, old_qy + offset_q * normal_y
    new_pixel_x = float(profile.get("peak_center_pixel_x", float("nan")))
    new_pixel_y = float(profile.get("peak_center_pixel_y", float("nan")))
    if not np.isfinite(new_pixel_x) or not np.isfinite(new_pixel_y):
        new_pixel_x, new_pixel_y = float(point["pixel_x"]), float(point["pixel_y"])
    center_x, center_y = float(options["center_qx"]), float(options["center_qy"])
    branch, quadrant = _quadrant_branch(
        new_qx,
        new_qy,
        center_x=center_x,
        center_y=center_y,
        reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
        tolerance_deg=float(options["quadrant_axis_tol_deg"]),
    )
    old_branch = int(point.get("branch_id", -1))
    old_quadrant = str(point.get("quadrant", "unknown"))
    if old_branch in {0, 1} and branch != old_branch:
        point["profile_refinement_reason"] = "profile_shift_crosses_branch_boundary"
        return
    if old_quadrant not in {"unknown", "axis_boundary"} and quadrant not in {old_quadrant, "axis_boundary"}:
        point["profile_refinement_reason"] = "profile_shift_crosses_quadrant_boundary"
        return
    arc = arcs_by_id.get(int(point.get("arc_id", -1)))
    new_side = str(point.get("side", "unknown"))
    if arc is not None and bool(arc.get("major_direction_resolved", False)):
        major_x = float(arc.get("major_direction_qx", float("nan")))
        major_y = float(arc.get("major_direction_qy", float("nan")))
        radial_x, radial_y = new_qx - center_x, new_qy - center_y
        side_value = radial_x * (-major_y) + radial_y * major_x
        side_scale = _side_axis_threshold(float(np.hypot(radial_x, radial_y)), q_step)
        new_side = "upper" if side_value > side_scale else "lower" if side_value < -side_scale else "unknown"
        if str(point.get("side", "unknown")) in {"upper", "lower"} and new_side != str(point["side"]):
            point["profile_refinement_reason"] = "profile_shift_crosses_side_boundary"
            return
    point["profile_center_offset_q"] = offset_q
    point["profile_shift_q"] = float(np.hypot(new_qx - old_qx, new_qy - old_qy))
    point["profile_shift_pixel"] = float(np.hypot(new_pixel_x - float(point["pixel_x"]), new_pixel_y - float(point["pixel_y"])))
    point["profile_refined_qx"] = new_qx
    point["profile_refined_qy"] = new_qy
    point["profile_refined_pixel_x"] = new_pixel_x
    point["profile_refined_pixel_y"] = new_pixel_y
    point["qx"], point["qy"] = new_qx, new_qy
    point["pixel_x"], point["pixel_y"] = new_pixel_x, new_pixel_y
    point["branch_id"], point["quadrant"], point["side"] = int(branch), quadrant, new_side
    point["profile_refinement_applied"] = True
    point["profile_refinement_reason"] = "observed_single_normal_peak"
    point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["profile_refined_observed_peak"]))


def _deferred_rejected_profile(point: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a review stub without sampling a detector-sized profile."""

    return {
        "point_id": str(point.get("point_id", "")),
        "valid": True,
        "model": "deferred_candidate_profile",
        "reason": "fit_deferred_rejected_candidate",
        "fit_deferred": True,
        "ambiguous": False,
        "offset_q": [],
        "raw_intensity": [],
        "fit_intensity": [],
        "peaks": [],
        "peak_count": 0,
        "snr": float("nan"),
        "normal_fwhm_q": float("nan"),
        "localization_sigma_q": float("nan"),
        "sampling_sigma_q": float("nan"),
        "uncertainty_source": "not_fit_rejected_candidate",
        "support_fraction": 0.0,
    }


def trace_butterfly_ridges(
    image: np.ndarray | Mapping[str, Any] | Any,
    qmap: Mapping[str, Any] | Any,
    q_window: Any,
    *,
    mask: Any = None,
    reference_axis_deg: float = 0.0,
    options: Any = None,
    edits: Any = None,
    cancel_event: Any = None,
    radial_hint_cache: _RadialHintGeometryCache | None = None,
) -> dict[str, Any]:
    """Trace observed butterfly ridges and return points, arcs and profiles.

    The output is intentionally a plain mapping so the GUI and JSON export
    can preserve rejected candidates without depending on the legacy
    ``observables`` dataclasses.  ``branch_id`` is 0 for QI/QIII, 1 for
    QII/QIV, and -1 when the reference-axis quadrant is ambiguous.  The
    result contains no ellipse fit and makes that topology-first boundary
    explicit in ``diagnostics``.
    """

    _check_cancelled(cancel_event, "ridge-trace:input")
    started = time.perf_counter()
    image_array, image_mask = _as_image(image)
    qx, qy, q, qmap_mask = _as_qmap(qmap, image_array.shape)
    q_min, q_max = _parse_q_window(q_window, q)
    opts = _normalise_options(options)
    if edits is None and opts.get("edits") is not None:
        edits = opts.get("edits")
    opts["reference_axis_deg"] = float(reference_axis_deg)
    if not np.isfinite(opts["reference_axis_deg"]):
        raise ValueError("reference_axis_deg must be finite")
    center_hint = opts.get("center_hint", opts.get("center_q", None))
    if center_hint is not None:
        opts["center_qx"], opts["center_qy"] = float(center_hint[0]), float(center_hint[1])
    valid = np.isfinite(image_array) & np.isfinite(qx) & np.isfinite(qy) & np.isfinite(q) & (q >= q_min) & (q <= q_max)
    if image_mask is not None:
        valid &= ~image_mask
    if qmap_mask is not None:
        valid &= ~qmap_mask
    if mask is not None:
        try:
            explicit_mask = np.asarray(np.broadcast_to(np.asarray(mask, dtype=bool), image_array.shape), dtype=bool)
        except ValueError as exc:
            raise ValueError("mask must broadcast to image shape") from exc
        valid &= ~explicit_mask
    valid, applied_edits, seed_actions = _apply_edits(valid, qx, qy, edits)
    finite_domain = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(q) & (q >= q_min) & (q <= q_max)
    first_order = _first_order_q_hint(
        q, image_array, valid, q_min, q_max, geometry_cache=radial_hint_cache
    )
    first_order["applied"] = False
    if not np.any(finite_domain):
        return {
            "points": [],
            "arcs": [],
            "profiles": {},
            "diagnostics": {"topology_before_ellipse_fit": True, "ellipse_fit": None, "reason": "empty_q_window"},
            "method_version": METHOD_VERSION,
        }
    ys, xs = np.where(finite_domain)
    pad = int(np.ceil(5.0 * max(opts["smoothing_scales"]))) + 3
    row0, row1 = max(0, int(ys.min()) - pad), min(image_array.shape[0], int(ys.max()) + pad + 1)
    col0, col1 = max(0, int(xs.min()) - pad), min(image_array.shape[1], int(xs.max()) + pad + 1)
    crop_image = image_array[row0:row1, col0:col1]
    crop_qx, crop_qy = qx[row0:row1, col0:col1], qy[row0:row1, col0:col1]
    crop_valid = valid[row0:row1, col0:col1]
    _check_cancelled(cancel_event, "ridge-trace:crop")
    signature = _point_signature(image_array, opts, (q_min, q_max))
    raw_candidates: list[dict[str, Any]] = []
    noise_estimates_by_scale: list[dict[str, Any]] = []
    coordinate_cache = _coordinate_derivatives(crop_qx, crop_qy)
    for scale in opts["smoothing_scales"]:
        _check_cancelled(cancel_event, f"ridge-trace:curvature:{scale:g}")
        field = _scaled_surface_field(
            crop_image,
            crop_qx,
            crop_qy,
            crop_valid,
            sigma=float(scale),
            options=opts,
            coordinate_derivatives=coordinate_cache,
        )
        noise_estimates_by_scale.append(
            {
                "smoothing_scale": float(scale),
                "method": CANDIDATE_NOISE_METHOD,
                "status": field.noise_status,
                "noise_sigma": float(field.noise) if np.isfinite(field.noise) else None,
                "candidate_snr_min": float(opts["candidate_snr_min"]),
                "noise_threshold_delta": (
                    float(opts["candidate_snr_min"] * field.noise)
                    if np.isfinite(field.noise)
                    else None
                ),
                "noise_sample_count": int(field.noise_support_count),
                "noise_support_fraction": float(field.noise_support_fraction),
                "minimum_support_fraction": float(opts["support_min"]),
                "edge_guard_pixels": int(field.noise_edge_guard_pixels),
                "structural_spread_for_height_scale": float(field.structural_spread),
                "height_scale": float(field.height_scale),
            }
        )
        raw_candidates.extend(
            _raw_candidates(
                field,
                row_offset=row0,
                col_offset=col0,
                max_candidates=int(opts["max_candidates"]),
                options=opts,
            )
        )
    raw_steps = [
        float(candidate["q_normal_step"])
        for candidate in raw_candidates
        if np.isfinite(candidate["q_normal_step"]) and candidate["q_normal_step"] > 0
    ]
    q_step = float(np.nanmedian(raw_steps)) if raw_steps else float(coordinate_cache[2])
    if not np.isfinite(q_step) or q_step <= 0:
        q_step = 1.0
    if first_order.get("candidate_peaks") and first_order.get("selection_status") in {
        "selected",
        "ambiguous",
    }:
        # Resolve competing radial peaks against the actual curvature
        # trajectories before the hint can alter candidate scores, NMS, or
        # topology.  This pass deliberately stops before profile fitting and
        # never mutates the raw candidates used by the final trace.
        _check_cancelled(cancel_event, "ridge-trace:first-order-arc-support")
        support_candidates = [dict(candidate) for candidate in raw_candidates]
        support_candidates = _nms(support_candidates, opts, q_step)
        _assign_point_ids(support_candidates, signature)
        _assign_reference_branches(support_candidates, opts)
        support_groups, support_edges = _graph_arcs(support_candidates, opts, q_step)
        support_arcs = _arc_topology(
            support_groups, support_edges, support_candidates, opts, q_step
        )
        first_order = _apply_hint_independent_arc_support(
            first_order,
            support_arcs,
            support_candidates,
            q_min=q_min,
            q_max=q_max,
            radii=np.asarray(q[valid], dtype=float),
            weights=np.clip(np.asarray(image_array[valid], dtype=float), 0.0, None),
        )
        first_order["applied"] = False
    if bool(opts.get("first_order_prefer", True)):
        n_weighted = _prefer_first_order_scores(raw_candidates, first_order.get("q_star"))
        first_order["applied"] = bool(n_weighted)
        first_order["n_weighted"] = int(n_weighted)
    candidates = _nms(raw_candidates, opts, q_step)
    if opts["max_points"] is not None and len(candidates) > int(opts["max_points"]):
        candidates = sorted(candidates, key=lambda point: (-float(point.get("score", 0.0)), float(point["qx"]), float(point["qy"])))[: int(opts["max_points"])]
    _assign_point_ids(candidates, signature)
    # Scale support is measured after NMS so the point records can expose it
    # without carrying duplicate candidate pixels into the graph.
    _annotate_scale_stability(
        candidates,
        raw_candidates,
        q_step,
        opts["smoothing_scales"],
        cancel_event=cancel_event,
    )
    _check_cancelled(cancel_event, "ridge-trace:nms")
    _assign_reference_branches(candidates, opts)
    groups, edges = _graph_arcs(candidates, opts, q_step)
    arcs = _arc_topology(groups, edges, candidates, opts, q_step)
    _check_cancelled(cancel_event, "ridge-trace:topology")
    # Local profiles are intentionally imported lazily: the legacy
    # observables module can call this module without creating an import loop.
    from .ridge_profiles import build_profile_context, extract_normal_profile, wang2007_vertical_slice_check

    profile_context = (
        build_profile_context(image_array, {"qx": qx, "qy": qy, "q": q}, mask=~valid)
        if candidates
        else None
    )
    arcs_by_id = {int(arc["arc_id"]): arc for arc in arcs}
    profiles: dict[str, dict[str, Any]] = {}
    n_profiled = 0
    for point in candidates:
        _check_cancelled(cancel_event, "ridge-trace:profiles")
        if not bool(point.get("accepted", False)):
            profile = _deferred_rejected_profile(point)
        else:
            n_profiled += 1
            profile = extract_normal_profile(
                image_array,
                {"qx": qx, "qy": qy, "q": q},
                point,
                mask=~valid,
                options={
                    **opts,
                    "_profile_context": profile_context,
                    "fit_gaussian": True,
                    "profile_half_width_q": opts["profile_half_width_q"] or 5.0 * max(float(point.get("q_normal_step", q_step)), q_step),
                },
                cancel_event=cancel_event,
            )
        point_id = str(point["point_id"])
        profiles[point_id] = profile
        point["normal_fwhm_q"] = float(profile.get("normal_fwhm_q", float("nan")))
        point["localization_sigma_q"] = float(profile.get("localization_sigma_q", float("nan")))
        point["sampling_sigma_q"] = float(profile.get("sampling_sigma_q", float("nan")))
        point["uncertainty_source"] = str(profile.get("uncertainty_source", "unknown"))
        profile_snr = float(profile.get("snr", float("nan")))
        if np.isfinite(profile_snr):
            point["snr"] = profile_snr
        if not bool(profile.get("valid", False)):
            point["accepted"] = False
            point["reason"] = "profile_unavailable"
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["profile_unavailable"]))
        elif bool(profile.get("fit_deferred", False)):
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["profile_fit_deferred_for_rejected_candidate"]))
        elif bool(profile.get("ambiguous", False)):
            point["accepted"] = False
            point["reason"] = "ambiguous_normal_profile"
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["ambiguous_normal_profile"]))
        elif not profile.get("peaks"):
            point["accepted"] = False
            point["reason"] = "no_normal_peak"
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["no_normal_peak"]))
        _apply_profile_refinement(point, profile, arcs_by_id, opts, q_step)
    excluded_ids: set[str] = set()
    for action in list(edits or [] if not isinstance(edits, Mapping) else edits.get("actions", edits.get("edits", []))):
        if isinstance(action, Mapping) and str(action.get("type", "")).lower().replace("-", "_") == "exclude_point":
            point_id = str(action.get("point_id", ""))
            if point_id:
                excluded_ids.add(point_id)

    def _apply_excluded_points() -> None:
        if not excluded_ids:
            return
        for point in candidates:
            if str(point.get("point_id", "")) in excluded_ids:
                point["accepted"] = False
                if "valid" in point:
                    point["valid"] = False
                point["reason"] = "excluded_point_edit"
                point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["excluded_point_edit"]))

    _apply_excluded_points()
    seed_records = _apply_seeds(candidates, arcs, seed_actions, opts, q_step)
    _apply_excluded_points()
    _refresh_arc_identity(arcs, candidates)
    radial_split = _demote_secondary_radial_population(
        candidates, prefer_radius=first_order.get("q_star")
    )
    ring_fill = _fill_sparse_first_order_ring(
        candidates,
        qx=qx,
        qy=qy,
        q=q,
        intensity=image_array,
        valid=valid,
        hint=first_order.get("q_star"),
        options=opts,
        q_step=q_step,
        signature=signature,
        arcs=arcs,
    )
    _apply_excluded_points()
    _refresh_arc_identity(arcs, candidates)
    for point in candidates:
        point_id = str(point.get("point_id", ""))
        if point_id and point_id not in profiles:
            profiles[point_id] = {
                "point_id": point_id,
                "valid": True,
                "model": "azimuthal_radial_peak",
                "reason": "first_order_ring_sample",
                "fit_deferred": False,
                "ambiguous": False,
                "offset_q": [],
                "raw_intensity": [],
                "fit_intensity": [],
                "peaks": [{"q": float(np.hypot(float(point["qx"]), float(point["qy"])))}],
                "peak_count": 1,
                "snr": float(point.get("snr", float("nan"))),
                "normal_fwhm_q": float("nan"),
                "localization_sigma_q": float(point.get("localization_sigma_q", q_step)),
                "sampling_sigma_q": float(q_step),
                "uncertainty_source": "azimuthal_radial_peak",
                "support_fraction": 1.0,
            }
    _check_cancelled(cancel_event, "ridge-trace:edits")
    # Freeze finite observed support only after profile refinement and all
    # seed/exclude edits.  The freezer mutates the working records so its
    # point-local/arc-local JSON fields survive public serialization; retain
    # only its compact summary in tracer diagnostics.
    from .arc_support import freeze_observed_support

    _check_cancelled(cancel_event, "ridge-trace:support")
    support_envelope = freeze_observed_support(
        candidates, arcs, summary_only=True, cancel_event=cancel_event
    )
    support_summary = dict(support_envelope.get("summary", {}))
    support_summary.update(
        {
            "support_version": support_envelope.get("support_version", support_envelope.get("version")),
            "support_status": support_envelope.get("support_status", support_envelope.get("status", "unavailable")),
            "support_frozen": bool(support_envelope.get("support_frozen", support_envelope.get("frozen", False))),
        }
    )
    if bool(opts.get("run_wang_check", True)):
        wang = wang2007_vertical_slice_check(
            image_array,
            {"qx": qx, "qy": qy, "q": q},
            (q_min, q_max),
            mask=~valid,
            reference_axis_deg=float(reference_axis_deg),
            options=opts,
            cancel_event=cancel_event,
        )
    else:
        wang = {"applicable": False, "diagnostic_only": True, "used_for_acceptance": False, "reason": "disabled_by_options"}
    public_points = [_public_point(point) for point in candidates]
    scale_stability_summary = _finite_scale_stability_summary(public_points)
    diagnostics: dict[str, Any] = {
        "topology_before_ellipse_fit": True,
        "ellipse_fit": None,
        "curvature_definition": "minimum principal curvature of the scaled q-space graph surface from first/second fundamental forms",
        "ridge_condition": "directional slope zero with negative minimum principal curvature",
        "method": "mask_normalized_multiscale_principal_curvature_zero_slope",
        "reference_axis_deg": float(reference_axis_deg),
        "center_q": [float(opts["center_qx"]), float(opts["center_qy"])],
        "q_window": [float(q_min), float(q_max)],
        "first_order_q_hint": first_order,
        "first_order_ring_fill": ring_fill,
        "smoothing_scales": [float(value) for value in opts["smoothing_scales"]],
        "crop": {"row_start": row0, "row_stop": row1, "col_start": col0, "col_stop": col1, "shape": [row1 - row0, col1 - col0], "original_shape": list(image_array.shape)},
        "n_raw_candidates": int(len(raw_candidates)),
        "n_points": int(len(public_points)),
        "n_profiled_points": int(n_profiled),
        "n_accepted_points": int(sum(bool(point["accepted"]) for point in public_points)),
        "elapsed_s": float(time.perf_counter() - started),
        "n_arcs": int(len(arcs)),
        "n_valid_arcs": int(sum(bool(arc["valid"]) for arc in arcs)),
        "scale_stability": scale_stability_summary,
        "candidate_noise": {
            "method": CANDIDATE_NOISE_METHOD,
            "threshold_basis": "mask-normalized raw-minus-smoothed high-pass MAD",
            "minimum_safe_samples": CANDIDATE_NOISE_MIN_SAFE_SAMPLES,
            "support_min": float(opts["support_min"]),
            "crop_edge_guard_truncate_sigma": CANDIDATE_NOISE_EDGE_TRUNCATE,
            "estimates_by_scale": noise_estimates_by_scale,
        },
        "mask_fraction_in_q_window": float(1.0 - np.count_nonzero(crop_valid) / max(1, np.count_nonzero(finite_domain[row0:row1, col0:col1]))),
        "edits": applied_edits,
        "seed_matches": seed_records,
        "excluded_point_ids": sorted(excluded_ids),
        "observed_support": support_summary,
        "radial_population": radial_split,
        "wang2007_vertical_slice": wang,
        "flags": ["observed_only", "no_ellipse_fit", "rejected_candidates_retained", "topology_before_ellipse_fit"]
        + (["mixed_radial_populations"] if radial_split.get("selection_status") == "ambiguous" else []),
    }
    return {"points": public_points, "arcs": arcs, "profiles": profiles, "diagnostics": diagnostics, "method_version": METHOD_VERSION}


__all__ = ["METHOD_VERSION", "trace_butterfly_ridges"]
