"""Parameter-level evidence checks for observed arcs, not optimizer acceptance.

Limits here are explicitly provisional engineering screens. Statistical
confidence and material interpretation require independent calibration evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numpy as np

from ._primitives import finite_float as _finite

# A nearly circular candidate has no identifiable ellipse direction. This is
# a provisional geometry screen, not a statistical test of isotropy.
NEAR_CIRCULAR_AXIS_RATIO_MIN = 0.95


PARAMETERS = ("a", "b", "axis_ratio", "theta_deg")

ELLIPSE_CANDIDATE_LIMITATION_FLAGS = frozenset(
    {
        "axis_ratio_at_bound",
        "axis_ratio_collapsed_to_line",
        "major_axis_exceeds_observed_extent",
        "minor_axis_not_resolved_against_localization",
        "annular_outer_window_truncated",
        "parameter_at_explicit_bound",
    }
)


def _flag_tokens(flags) -> list[str]:
    if isinstance(flags, Mapping):
        nested = flags.get("flags") or ()
        if isinstance(nested, str):
            tokens = [part.strip() for part in nested.split(",") if part.strip()]
        else:
            tokens = [str(item) for item in nested if item]
        if flags.get("axis_ratio"):
            tokens.append("axis_ratio_at_bound")
        bounds = flags.get("bound_flags", flags.get("bound_status"))
        if isinstance(bounds, Mapping) and any(value is True for value in bounds.values()):
            tokens.append("parameter_at_explicit_bound")
        return list(dict.fromkeys(tokens))
    if isinstance(flags, str):
        return [part.strip() for part in flags.split(",") if part.strip()]
    return [str(item) for item in (flags or ()) if item]


def ellipse_candidate_is_limited(flags=None) -> bool:
    """Return whether geometry flags require a constrained-candidate label.

    These diagnostics describe parameter bounds or incomplete q-space support.
    They affect confidence in the ellipse, not whether a finite small ``b/a``
    is an ellipse or a ring.
    """

    return bool(ELLIPSE_CANDIDATE_LIMITATION_FLAGS.intersection(_flag_tokens(flags)))


def classify_ellipse_publication(
    *,
    quality_status=None,
    axis_ratio=None,
    flags=None,
) -> str:
    """Classify shape without granting quantitative or scientific acceptance.

    This classifies the fitted shape. Bound/support flags remain diagnostics
    for candidate status; they do not convert a finite ellipse into a ring.
    """

    quality = str(quality_status or "").strip().upper()
    flag_tokens = _flag_tokens(flags)
    ratio = _finite(axis_ratio)
    if ratio is not None and not 0.0 < ratio <= 1.0:
        return "fail"
    # A circular pattern may fail a four-side butterfly fit. Its shape and
    # unavailable ellipse-axis direction remain distinguishable from that fit.
    if ((ratio is not None and ratio >= NEAR_CIRCULAR_AXIS_RATIO_MIN)
            or "near_circular_ellipse_axis_unidentifiable" in flag_tokens):
        return "ring"
    if quality in {"FAIL", "FAILED", "INVALID"}:
        return "fail"
    if ratio is not None:
        return "ellipse"
    return "undetermined"


def unpublished_ellipse_shape(*, quality_status=None, axis_ratio=None, flags=None) -> bool:
    """True only when near-circular degeneracy makes the ellipse axis unusable."""

    return classify_ellipse_publication(
        quality_status=quality_status,
        axis_ratio=axis_ratio,
        flags=flags,
    ) == "ring"




def _first_present(mapping, *keys, default=None):
    if not isinstance(mapping, Mapping):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _component_count(row):
    explicit = _finite(row.get("support_component_count"))
    if explicit is not None:
        return max(0, int(explicit))
    components = row.get("support_components")
    if isinstance(components, Sequence) and not isinstance(components, (str, bytes)):
        return len(components)
    return 0


def extract_arc_rows(
    trace: Mapping | Sequence | None,
    candidate: Mapping | None = None,
    *,
    include_unassigned: bool = False,
    deep_copy: bool = False,
) -> list[dict]:
    """Read lossless per-arc evidence from either fit payload or trace.

    The fit adds projection/normal-residual evidence to the traced support.
    Prefer those post-fit diagnostics over the pre-fit rows when both are present.
    """

    candidates = []
    if isinstance(candidate, Mapping):
        candidates.extend((candidate.get("arc_diagnostics"), candidate.get("arc_support")))
    if isinstance(trace, Mapping):
        candidates.extend((trace.get("arc_diagnostics"), trace.get("arc_support")))
        diagnostics = trace.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            candidates.extend((diagnostics.get("arc_diagnostics"), diagnostics.get("arc_support")))
    elif isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
        candidates.append(trace)
    for value in candidates:
        if isinstance(value, Mapping):
            value = value.get("rows", value.get("arcs", value.get("records")))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
            if deep_copy:
                from copy import deepcopy

                rows = [
                    deepcopy(dict(item))
                    for item in value
                    if isinstance(item, Mapping)
                    and (
                        include_unassigned
                        or _finite(item.get("arc_id")) is None
                        or _finite(item["arc_id"]) >= 0
                    )
                ]
            else:
                # Unassigned/rejected points remain in point diagnostics but are
                # not an additional measured arc or a manually constrained arc.
                rows = [
                    dict(item)
                    for item in value
                    if isinstance(item, Mapping)
                    and (
                        include_unassigned
                        or _finite(item.get("arc_id")) is None
                        or _finite(item["arc_id"]) >= 0
                    )
                ]
            if rows:
                return rows
    return []


def _arc_rows(trace, candidate):
    """Read lossless per-arc evidence for quality gating."""

    return extract_arc_rows(trace, candidate, include_unassigned=False, deep_copy=False)


def _dedupe_reasons(reasons):
    return list(dict.fromkeys(str(reason) for reason in reasons if reason))


def _arc_support_quality(arc_rows):
    """Aggregate support/evidence without promoting a fit to acceptance."""

    support_statuses = [str(row.get("support_status", "unavailable")) for row in arc_rows]
    support_frozen = sum(status in {"frozen", "manual_only"} for status in support_statuses)
    support_missing = sum(status not in {"frozen", "manual_only"} for status in support_statuses)
    manual_only = sum(status == "manual_only" for status in support_statuses)
    invalid = [_finite(_first_present(row, "infeasible_projection_count", "support_infeasible_count")) or 0.0 for row in arc_rows]
    endpoint = [value for row in arc_rows if (value := _finite(_first_present(row, "support_endpoint_fraction", "endpoint_fraction"))) is not None]
    manual = [value for row in arc_rows if (value := _finite(row.get("manual_endpoint_fraction"))) is not None]
    normal = [value for row in arc_rows if (value := _finite(row.get("normal_residual_q_rms"))) is not None]
    normal_ratio = [value for row in arc_rows if (value := _finite(row.get("normal_residual_localization_ratio_rms"))) is not None]
    coverage = [value for row in arc_rows if (value := _finite(row.get("projection_rmse_q"))) is not None]
    gaps = [_finite(row.get("support_gap_count")) or 0.0 for row in arc_rows]
    short = sum(bool(row.get("short_arc")) or "short_arc" in (row.get("topology_flags") or ()) for row in arc_rows)
    return {
        "arc_count": len(arc_rows),
        "support_frozen_count": int(support_frozen),
        "support_unavailable_count": int(support_missing),
        "manual_only_arc_count": int(manual_only),
        "support_component_count": int(sum(_component_count(row) for row in arc_rows)),
        "support_gap_count": int(sum(int(value) for value in gaps)),
        "short_arc_count": int(short),
        "infeasible_projection_count": int(sum(int(value) for value in invalid)),
        "projection_rmse_q_median": float(np.median(coverage)) if coverage else None,
        "normal_residual_q_rms_median": float(np.median(normal)) if normal else None,
        "normal_residual_localization_ratio_rms_median": float(np.median(normal_ratio)) if normal_ratio else None,
        "support_endpoint_fraction_median": float(np.median(endpoint)) if endpoint else None,
        "manual_endpoint_fraction_median": float(np.median(manual)) if manual else None,
    }


def evaluate_arc_evidence(trace, candidate, *, uncertainty=None, sensitivity=None):
    """Keep a candidate, its empirical support, and publishability separate."""
    trace = trace if isinstance(trace, Mapping) else {}
    candidate = candidate if isinstance(candidate, Mapping) else {}
    fit_points = candidate.get("point_diagnostics")
    used_ids = None
    if isinstance(fit_points, (list, tuple)):
        used_ids = {str(p.get("point_id", index)) for index, p in enumerate(fit_points)
                    if isinstance(p, Mapping) and p.get("used") is True}
    points = [p for index, p in enumerate(trace.get("points", []))
              if isinstance(p, Mapping) and p.get("accepted", p.get("valid", False))
              and p.get("valid", True)
              and (used_ids is None or str(p.get("point_id", index)) in used_ids)
              and (_finite(p.get("arc_id")) is None or _finite(p["arc_id"]) >= 0)
              and _finite(p.get("qx")) is not None and _finite(p.get("qy")) is not None
              and p.get("side") in ("upper", "lower") and p.get("branch_id") in (0, 1)]
    groups = {f"{branch}:{side}": sum(p.get("branch_id") == branch and p.get("side") == side
                                    for p in points)
              for branch in (0, 1) for side in ("upper", "lower")}
    common = []
    occupied_sides = sum(
        1 for count in groups.values() if isinstance(count, (int, float)) and count > 0
    )
    if not candidate or not candidate.get("success"):
        common.append("solver_or_arc_support_unavailable")
    if occupied_sides < 3:
        common.append("insufficient_occupied_sides")
    if any(count < 3 for count in groups.values()):
        common.append("insufficient_independent_side_support")
    if any(p.get("trajectory_confidence") == "low"
           or p.get("trajectory_ambiguous") for p in points):
        common.append("annular_trajectory_support_limited")
    ratio = _finite(candidate.get("axis_ratio"))
    if ratio is not None and ratio >= NEAR_CIRCULAR_AXIS_RATIO_MIN:
        common.append("near_circular_ellipse_axis_unidentifiable")
    bound_flags = candidate.get("bound_flags", {}) or {}
    flag_names = {str(item) for item in (candidate.get("flags") or ())}
    if bool(bound_flags.get("axis_ratio")) or "axis_ratio_at_bound" in flag_names:
        common.append("axis_ratio_at_bound")
    sigmas = np.asarray([v for p in points if (v := _finite(p.get("localization_sigma_q")))
                         is not None and v > 0], dtype=float)
    median_sigma = float(np.median(sigmas)) if sigmas.size else None
    origin_qs = []
    candidate_centered_qs = []
    center_qx = _finite(candidate.get("center_qx"))
    center_qy = _finite(candidate.get("center_qy"))
    center = candidate.get("center")
    if (center_qx is None or center_qy is None) and isinstance(center, (list, tuple)) and len(center) == 2:
        center_qx = _finite(center[0]) if center_qx is None else center_qx
        center_qy = _finite(center[1]) if center_qy is None else center_qy
    center_qx = 0.0 if center_qx is None else center_qx
    center_qy = 0.0 if center_qy is None else center_qy
    for point in points:
        qx = _finite(point.get("qx"))
        qy = _finite(point.get("qy"))
        if qx is not None and qy is not None:
            dx, dy = qx - center_qx, qy - center_qy
            origin_qs.append(math.hypot(qx, qy))
            candidate_centered_qs.append(math.hypot(dx, dy))
    first_order = None
    diagnostics = trace.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        first_order = diagnostics.get("first_order_q_hint")
        if isinstance(first_order, Mapping):
            if first_order.get("selection_status") == "ambiguous":
                common.append("first_order_q_hint_ambiguous")
        radial_population = diagnostics.get("radial_population")
        if (
            isinstance(radial_population, Mapping)
            and radial_population.get("split")
            and radial_population.get("selection_status") == "ambiguous"
        ):
            common.append("mixed_radial_populations")
        if diagnostics.get("outer_window_truncated"):
            common.append("annular_outer_window_truncated")
    # The actual fitted points define the observed radial extent. q* remains
    # useful for reporting peak-selection ambiguity, but it is not an outer
    # support boundary for elongated butterfly wings.
    q_extent = max(origin_qs) if origin_qs else None
    candidate_centered_q_extent = max(candidate_centered_qs) if candidate_centered_qs else None
    a_value = _finite(candidate.get("a"))
    extent_tolerance = 3.0 * median_sigma if median_sigma is not None else 0.0
    major_exceeds_extent = (
        a_value is not None
        and candidate_centered_q_extent is not None
        and candidate_centered_q_extent > 0
        and a_value > candidate_centered_q_extent + extent_tolerance
    )
    if major_exceeds_extent:
        common.append("major_axis_exceeds_observed_extent")
    arc_rows = _arc_rows(trace, candidate)
    arc_metrics = _arc_support_quality(arc_rows)
    if not arc_rows:
        common.append("per_arc_support_evidence_unavailable")
    else:
        if arc_metrics["support_unavailable_count"]:
            common.append("observed_support_unavailable")
        if arc_metrics["manual_only_arc_count"]:
            common.append("manual_arc_bounds_not_experimental_evidence")
        if arc_metrics["short_arc_count"]:
            common.append("short_observed_arc")
        if arc_metrics["infeasible_projection_count"]:
            common.append("observed_support_infeasible")
        if arc_metrics["support_gap_count"]:
            common.append("disconnected_observed_support")
        endpoint_dependence = any(
            (_finite(_first_present(row, "support_endpoint_fraction", "endpoint_fraction")) or 0.0) > 0.0
            or (_finite(row.get("manual_endpoint_fraction")) or 0.0) > 0.0
            or bool(row.get("support_endpoint_clipped"))
            or bool(row.get("manual_bound_clipped"))
            for row in arc_rows
        )
        if endpoint_dependence:
            common.append("arc_endpoint_or_manual_bound_dependent")
        mirror_available = [row for row in arc_rows if _first_present(row, "mirror_symmetry_status", "mirror_status") in {"available", "observed", "supported"}]
        mirror_missing = [row for row in arc_rows if _first_present(row, "mirror_symmetry_status", "mirror_status") in {None, "unavailable", "ambiguous", "unpaired"}]
        if mirror_missing or (arc_rows and not mirror_available):
            common.append("mirror_symmetry_evidence_unavailable_or_ambiguous")
        normal_missing = [row for row in arc_rows if _finite(row.get("normal_residual_q_rms")) is None and _finite(row.get("normal_residual_q_p95")) is None]
        if normal_missing:
            common.append("per_arc_normal_residual_evidence_unavailable")
        normal_screen = []
        for row in arc_rows:
            value = _finite(_first_present(row, "normal_residual_q_rms", "normal_residual_q_p95"))
            scale = _finite(_first_present(row, "median_localization_sigma_q", "localization_sigma_q"))
            if value is not None and scale is not None and scale > 0:
                normal_screen.append(value > 3.0 * scale)
        if any(normal_screen):
            common.append("per_arc_normal_residual_exceeds_localization_scale")
    trace_diagnostics = trace.get("diagnostics")
    seed_matches = (
        _first_present(trace_diagnostics, "seed_matches", "seed_actions", default=[])
        if isinstance(trace_diagnostics, Mapping)
        else []
    ) or []
    if any(not match.get("matched", False) for match in seed_matches if isinstance(match, Mapping)):
        common.append("seed_not_supported_by_observed_arc")
    if median_sigma is not None:
        if any(
            (value := _finite(_first_present(row, "normal_residual_q_rms", "normal_residual_q_p95"))) is not None
            and value > 3.0 * median_sigma
            for row in arc_rows
        ):
            common.append("per_arc_normal_residual_exceeds_localization_scale")
    rmse = _finite(candidate.get("rmse"))
    if median_sigma is None:
        common.append("localization_uncertainty_unavailable")
    elif rmse is not None and rmse > 3 * median_sigma:
        common.append("residual_exceeds_localization_scale")
    condition = _finite(candidate.get("condition"))
    if condition is None or condition > 1e8:
        common.append("ill_conditioned_geometry")
    uncertainty = dict(uncertainty or {})
    intervals = uncertainty.get("intervals", {})
    sensitivity = dict(sensitivity or {})
    for holdout in sensitivity.get("held_out_arcs", []):
        if not holdout.get("success"):
            common.append("held_out_arc_not_identifiable")
        elif median_sigma is not None:
            error = _finite(holdout.get("predictive_rmse_q"))
            if error is not None and error > 3 * median_sigma:
                common.append("held_out_arc_prediction_exceeds_localization")
    rows = {}
    b = _finite(candidate.get("b"))
    geometry_usable = bool(
        candidate.get("success") and candidate_centered_qs
        and a_value is not None and b is not None and 0 < b <= a_value
        and ratio is not None and 0 < ratio <= 1
        and _finite(candidate.get("theta_deg")) is not None
    )
    width_scales = [v for p in points if (v := _finite(p.get("normal_fwhm_q")))
                    is not None and v > 0]
    bound_flags = candidate.get("bound_flags") or {}
    if not isinstance(bound_flags, Mapping):
        bound_flags = {}
    for name in PARAMETERS:
        reasons = list(common)
        value = _finite(candidate.get(name))
        if value is None:
            reasons.append("parameter_unavailable")
        bound_name = "theta" if name == "theta_deg" else name
        if bound_flags.get(bound_name) or bound_flags.get(name) or (
            name == "b" and (bound_flags.get("a") or bound_flags.get("axis_ratio"))
        ):
            reasons.append("parameter_at_explicit_bound")
        if name in ("b", "axis_ratio") and b is not None and median_sigma is not None and b <= 2 * median_sigma:
            reasons.append("minor_axis_not_resolved_against_localization")
        interval = intervals.get(name)
        if isinstance(interval, Mapping):
            interval = [interval.get("low", interval.get("lower")), interval.get("high", interval.get("upper"))]
        finite_interval = None
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            lo, hi = (_finite(x) for x in interval)
            if lo is not None and hi is not None and hi >= lo:
                finite_interval = [lo, hi]
                if name != "theta_deg" and lo <= 0:
                    reasons.append("interval_reaches_degenerate_geometry")
        if finite_interval is None:
            reasons.append("image_resampling_interval_unavailable")
        success_fraction = _finite(uncertainty.get("success_fraction"))
        if success_fraction is not None and success_fraction < 0.9:
            reasons.append("resampling_topology_or_fit_unstable")
        parameter_ranges = sensitivity.get("parameter_ranges")
        shifts = parameter_ranges.get(name) if isinstance(parameter_ranges, Mapping) else None
        if isinstance(shifts, (list, tuple)) and len(shifts) == 2 and finite_interval is not None:
            shift_lo, shift_hi = _finite(shifts[0]), _finite(shifts[1])
            if shift_lo is not None and shift_hi is not None:
                if shift_lo < finite_interval[0] or shift_hi > finite_interval[1]:
                    reasons.append("analysis_choices_exceed_resampling_interval")
        if sensitivity.get("failed_variants"):
            reasons.append("analysis_choice_topology_unstable")
        if not sensitivity.get("completed", False):
            reasons.append("analysis_choice_sensitivity_unassessed")
        reasons = _dedupe_reasons(reasons)
        # Optional uncertainty work describes an estimate; it does not decide
        # whether an already fitted numerical value can be inspected/exported.
        assessment_only = {
            "image_resampling_interval_unavailable",
            "analysis_choice_sensitivity_unassessed",
        }
        empirical_reasons = [reason for reason in reasons if reason not in assessment_only]
        support = "supported" if geometry_usable and not empirical_reasons else "undetermined"
        calibrated = bool(uncertainty.get("coverage_calibrated", False))
        publication_reasons = list(reasons)
        if not calibrated:
            publication_reasons.append("interval_coverage_not_calibrated")
        available = geometry_usable and value is not None
        publication_available = available and not publication_reasons
        estimate_status = (
            "unavailable" if not available else "available" if publication_available
            else "estimate" if support == "supported" else "candidate"
        )
        rows[name] = {
            "value": value if available else None,
            "candidate_value": value,
            "status": estimate_status,
            "publication_status": "available" if publication_available else "not_assessed",
            "confidence": "unavailable" if not available else "empirical" if support == "supported" else "limited",
            "empirical_status": support,
            "reason": "; ".join(_dedupe_reasons(publication_reasons)),
            "reasons": _dedupe_reasons(publication_reasons),
            "interval": finite_interval,
            "interval_kind": "calibrated_confidence_interval" if calibrated else "empirical_central_95_percent",
            "unit": "degree" if name == "theta_deg" else "dimensionless" if name == "axis_ratio" else candidate.get("q_unit", "unknown"),
        }
    status = "supported" if all(r["empirical_status"] == "supported" for r in rows.values()) else "undetermined"
    # Reusing a finite, resolved fit as an optimizer initial value is not a
    # publication decision. Each subsequent frame is measured independently.
    seed_blockers = {
        "insufficient_occupied_sides", "insufficient_independent_side_support",
        "near_circular_ellipse_axis_unidentifiable", "axis_ratio_collapsed_to_line",
        "axis_ratio_at_bound",
        "observed_support_infeasible", "ill_conditioned_geometry",
        "residual_exceeds_localization_scale", "annular_trajectory_support_limited",
    }
    warm_start_eligible = geometry_usable and not seed_blockers.intersection(common)
    return {
        "quantitative_parameters": rows,
        "measurement_status": status,
        "warm_start_eligible": warm_start_eligible,
        "quality": {
            "status": (
                "FAIL"
                if not candidate.get("success") or occupied_sides < 3
                else "WARN"
            ),
            "engineering_status": "WARN" if status != "supported" else "PASS",
            "scientific_status": "NOT_ACCEPTED",
            "scientific_reason": "Numerical estimates and their limitations are retained; structural interpretation uses the image, calibration and model assumptions",
            "thresholds_version": "butterfly-arcs-engineering-provisional-v1",
            "thresholds_frozen": False,
            "metrics": {"side_counts": groups, "occupied_sides": occupied_sides,
                        "observed_q_extent": q_extent,
                        "candidate_centered_q_extent": candidate_centered_q_extent,
                        "a_over_observed_extent": (
                            a_value / q_extent
                            if a_value is not None and q_extent not in (None, 0)
                            else None
                        ),
                        "median_localization_sigma_q": median_sigma,
                        "median_normal_fwhm_q": float(np.median(width_scales)) if width_scales else None,
                        "residual_sigma_ratio": rmse / median_sigma if rmse is not None and median_sigma else None,
                        "arc_support": arc_metrics},
            "arc_support": arc_metrics,
            "provisional_limits": {"side_points_min": 3, "residual_sigma_ratio_max": 3,
                                   "minor_axis_sigma_ratio_min": 2, "condition_max": 1e8,
                                   "resampling_success_fraction_min": 0.9},
            "flags": sorted({reason for row in rows.values() for reason in row["reasons"]}),
        },
    }


def _remediation_json_value(value):
    """Convert numeric/container values into strict JSON-safe scalars."""
    if isinstance(value, np.generic):
        return _remediation_json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_remediation_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _remediation_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_remediation_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _metric_float(metrics: Mapping[str, object] | None, *keys: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    for key in keys:
        if key in metrics:
            val = _finite(metrics[key])
            if val is not None:
                return val
    return None


def _round_param(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def build_quality_remediation(
    flags: Sequence[str],
    metrics: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Map quality flags and observed metrics to concrete bilingual remediation steps.

    Returns a deterministic, strict-JSON-serializable list of auditable action
    objects. Each item preserves the originating ``flag`` and provides
    ``severity``, ``summary_en``, ``summary_zh``, ``recommended_action``, and
    ``suggested_parameters``.
    """

    tokens = _flag_tokens(flags)
    if not tokens:
        return []
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    items: list[dict[str, object]] = []

    q_min = _metric_float(metric_map, "q_min", "window_q_min")
    q_max = _metric_float(metric_map, "q_max", "window_q_max")
    q_window = metric_map.get("q_window")
    if (q_min is None or q_max is None) and isinstance(q_window, (list, tuple)) and len(q_window) == 2:
        q_min = _finite(q_window[0]) if q_min is None else q_min
        q_max = _finite(q_window[1]) if q_max is None else q_max
    observed_q_extent = _metric_float(metric_map, "observed_q_extent", "candidate_centered_q_extent")
    q_cap_max = _metric_float(metric_map, "q_active_max", "q_valid_max", "detector_q_max")
    rec_q_min = _metric_float(metric_map, "recommended_q_min", "q_beamstop_safe", "beamstop_safe_q_min")
    rec_mask_radius = _metric_float(metric_map, "recommended_mask_radius_px", "beamstop_mask_radius_px")
    bisector_deg = _metric_float(metric_map, "friedel_bisector_deg", "recommended_draw_axis_deg", "detected_axis_deg")
    axis_ratio_val = _metric_float(metric_map, "axis_ratio", "axis_ratio_b_over_a")
    collapse_rate = _metric_float(metric_map, "bound_collapse_rate", "axis_ratio_bound_collapse_rate")
    dup_count = _metric_float(metric_map, "duplicate_frame_count")
    unique_ratio = _metric_float(metric_map, "unique_ratio")

    def _clamped_widened_q_max(base_hi: float | None) -> float | None:
        if base_hi is None or base_hi <= 0.0:
            return None
        candidate = base_hi * 1.25
        if q_cap_max is not None and q_cap_max > 0.0:
            candidate = min(candidate, q_cap_max)
        return _round_param(candidate)

    for flag in tokens:
        severity = "info"
        summary_en = f"Review quality flag '{flag}' against observed intensity and analysis settings."
        summary_zh = f"请结合观测强度与分析设置复核质量标记 '{flag}'。"
        recommended_action = f"Inspect diagnostics for '{flag}' before reporting quantitative estimates."
        suggested_parameters: dict[str, object] = {}

        if flag == "annular_outer_window_truncated":
            severity = "high"
            base_hi = q_max if q_max is not None else observed_q_extent
            target_q_max = _clamped_widened_q_max(base_hi)
            summary_en = (
                "Butterfly wings reach the outer q-window boundary, leaving the major-axis tip unobserved."
            )
            summary_zh = "蝴蝶花瓣延伸至分析 q 窗口外边界，长轴端点未被完整观测。"
            recommended_action = (
                f"Widen q_max (e.g., to ~{target_q_max}) if detector coverage remains valid, "
                "or retain major axis 'a' as a constrained candidate."
                if target_q_max is not None
                else "Widen q_max if valid detector signal extends further, or keep major axis 'a' as a constrained candidate."
            )
            suggested_parameters = {
                "widen_q_max": True,
                "treat_major_axis_as_candidate": True,
            }
            if target_q_max is not None:
                suggested_parameters["q_max"] = target_q_max

        elif flag == "low_q_edge_hit":
            severity = "high"
            target_q_min = _round_param(rec_q_min)
            if target_q_min is None and q_min is not None and q_min > 0:
                target_q_min = _round_param(max(q_min * 0.5, 0.01))
            summary_en = (
                "Radial peak candidates pile up at the inner q_min boundary, indicating low-q peak clipping."
            )
            summary_zh = "径向峰候选值堆积在 q_min 内边界，表明低 q 相关峰被窗口截断。"
            recommended_action = (
                f"Apply an explicit circular beamstop exclusion mask and lower q_min toward {target_q_min} "
                "to recover the low-q correlation peak."
                if target_q_min is not None
                else "Apply an explicit circular beamstop exclusion mask, then lower q_min to recover the clipped low-q peak."
            )
            suggested_parameters = {"apply_beamstop_mask": True, "lower_q_min": True}
            if target_q_min is not None:
                suggested_parameters["q_min"] = target_q_min
            if rec_mask_radius is not None:
                suggested_parameters["beamstop_mask_radius_px"] = _round_param(rec_mask_radius, 2)

        elif flag == "high_q_edge_hit":
            severity = "medium"
            target_q_max = _clamped_widened_q_max(q_max)
            summary_en = "Peak candidates pile up at the outer q_max boundary."
            summary_zh = "峰候选值堆积在 q_max 外边界。"
            recommended_action = (
                f"Increase q_max toward {target_q_max} if valid detector pixels exist at higher q."
                if target_q_max is not None
                else "Increase q_max if valid detector coverage extends to higher q."
            )
            suggested_parameters = {"widen_q_max": True}
            if target_q_max is not None:
                suggested_parameters["q_max"] = target_q_max

        elif flag in {"axis_ratio_at_bound", "axis_ratio_collapsed_to_line"}:
            severity = "high"
            rate_note_en = (
                f" ({collapse_rate * 100.0:.1f}% of frames collapsed)"
                if collapse_rate is not None
                else ""
            )
            rate_note_zh = (
                f"（{collapse_rate * 100.0:.1f}% 帧坍缩至下界）"
                if collapse_rate is not None
                else ""
            )
            summary_en = (
                f"Ellipse axis ratio b/a hit its explicit bound{rate_note_en}; origin-centered ellipse "
                "minor axis and Ln are degenerate for straight radial butterfly streaks."
            )
            summary_zh = (
                f"椭圆轴比 b/a 触及拟合下界{rate_note_zh}；对于过原点的直射线状蝴蝶翼，原点中心椭圆短轴与 Ln 存在几何简并。"
            )
            recommended_action = (
                "Keep ellipse a/b/Ln as constrained candidates only; report model-free wing ray opening "
                "angle, azimuthal harmonics (A2, A4), and sector radial q* / L* separately."
            )
            suggested_parameters = {
                "treat_ellipse_axes_as_candidate_only": True,
                "report_model_free_ray_angles": True,
                "recommended_workflow": "annular",
                "ellipse_preset": "flat_ellipse",
            }
            if axis_ratio_val is not None:
                suggested_parameters["observed_axis_ratio"] = _round_param(axis_ratio_val, 6)

        elif flag == "mixed_radial_populations":
            severity = "medium"
            summary_en = (
                "Multiple separated radial trajectory populations were detected within the active q_window."
            )
            summary_zh = "当前分析 q 窗口内检测到分离的多组径向轨迹群。"
            recommended_action = (
                "Split q_window into separate radial sub-windows around each population and compare fits "
                "independently instead of merging orders."
            )
            suggested_parameters = {
                "split_q_window": True,
                "compare_subwindow_fits": True,
            }

        elif flag in {
            "near_circular_axis_angle_unconstrained",
            "near_circular_ellipse_axis_unidentifiable",
        }:
            severity = "medium"
            summary_en = (
                "Near-circular scattering pattern (b/a >= 0.95) has no identifiable ellipse major-axis orientation."
            )
            summary_zh = "近圆形散射图样（b/a >= 0.95）的椭圆主轴方向不可辨识。"
            recommended_action = (
                "Publish ring q* and L = 2*pi/q* (when physical q is calibrated) only; suppress ellipse "
                "theta_deg and conditional Ln/Lz."
            )
            suggested_parameters = {
                "publish_shape": "ring",
                "report_ellipse_theta": False,
                "recommended_workflow": "sector",
            }

        elif flag == "orthogonal_axis_flip":
            severity = "high"
            axis_note_en = f" (~{bisector_deg:.1f} deg)" if bisector_deg is not None else ""
            axis_note_zh = f"（约 {bisector_deg:.1f}°）" if bisector_deg is not None else ""
            summary_en = (
                f"Principal symmetry axis{axis_note_en} is rotated ~90 deg relative to nominal_draw_axis_deg."
            )
            summary_zh = f"主对称轴{axis_note_zh}相对标称拉伸轴 nominal_draw_axis_deg 旋转约 90°。"
            recommended_action = (
                "Verify sample mounting orientation vs physical habit-plane switching before pooling with "
                "other runs; align angular analysis to the detected Friedel bisector."
            )
            suggested_parameters = {
                "verify_sample_mounting_orientation": True,
                "use_detected_friedel_bisector": True,
            }
            if bisector_deg is not None:
                suggested_parameters["draw_axis_deg"] = _round_param(bisector_deg, 2)

        elif flag == "duplicate_frames_detected":
            severity = "high"
            dup_note_en = (
                f" ({int(dup_count)} duplicate frames, unique ratio {unique_ratio:.3f})"
                if dup_count is not None and unique_ratio is not None
                else ""
            )
            dup_note_zh = (
                f"（{int(dup_count)} 帧重复，唯一帧占比 {unique_ratio:.3f}）"
                if dup_count is not None and unique_ratio is not None
                else ""
            )
            summary_en = (
                f"Identical/frozen pixel frames detected in sequence{dup_note_en}."
            )
            summary_zh = f"序列中检测到像素完全相同的冻结/重复帧{dup_note_zh}。"
            recommended_action = (
                "Quarantine duplicate frames to QC/SI diagnostics and exclude them from kinetic fitting "
                "without interpolating or fabricating replacement frames."
            )
            suggested_parameters = {
                "quarantine_duplicates": True,
                "exclude_from_kinetics": True,
            }

        elif flag == "beamstop_unmasked_low_q":
            severity = "high"
            summary_en = (
                "Unmasked central beamstop shadow/rim detected inside the valid detector region."
            )
            summary_zh = "有效探测器区域内存在未掩膜的中心束挡阴影或边缘伪影。"
            recommended_action = (
                f"Mask a circular exclusion region of radius >= {rec_mask_radius:.1f} px around the beam "
                "center before lowering q_min."
                if rec_mask_radius is not None
                else "Apply a circular beamstop exclusion mask with a safety margin around the beam center before lowering q_min."
            )
            suggested_parameters = {"apply_beamstop_mask": True}
            if rec_mask_radius is not None:
                suggested_parameters["beamstop_mask_radius_px"] = _round_param(rec_mask_radius, 2)
            beam_center = metric_map.get("beam_center_xy")
            if isinstance(beam_center, (list, tuple)) and len(beam_center) == 2:
                cx, cy = _finite(beam_center[0]), _finite(beam_center[1])
                if cx is not None and cy is not None:
                    suggested_parameters["beam_center_xy"] = [_round_param(cx, 2), _round_param(cy, 2)]
            if rec_q_min is not None:
                suggested_parameters["q_min"] = _round_param(rec_q_min)

        elif flag == "major_axis_exceeds_observed_extent":
            severity = "high"
            summary_en = (
                "Fitted ellipse major semi-axis 'a' extrapolates beyond the observed q-space extent."
            )
            summary_zh = "拟合椭圆半长轴 a 超出观测 q 空间范围，属于外推值。"
            recommended_action = (
                "Widen q_max if butterfly intensity continues outward; otherwise keep 'a' and Ln as "
                "extrapolated candidates only."
            )
            suggested_parameters = {
                "treat_major_axis_as_candidate": True,
                "widen_q_max": True,
            }
            target_ext_max = _clamped_widened_q_max(observed_q_extent)
            if target_ext_max is not None:
                suggested_parameters["q_max"] = target_ext_max

        elif flag == "parameter_at_explicit_bound":
            severity = "medium"
            summary_en = "One or more fitted ellipse parameters reached an explicit optimizer bound."
            summary_zh = "一个或多个拟合椭圆参数触及显式优化边界。"
            recommended_action = (
                "Inspect bound_flags; widen bounds only if physically supported by observed arcs, "
                "and retain the fit as a constrained candidate."
            )
            suggested_parameters = {"review_bound_flags": True, "keep_candidate_label": True}

        elif flag == "minor_axis_not_resolved_against_localization":
            severity = "medium"
            summary_en = "Minor semi-axis 'b' is smaller than 2x the ridge localization uncertainty."
            summary_zh = "拟合短半轴 b 小于 2 倍脊线定位不确定度，尚未被空间分辨。"
            recommended_action = (
                "Retain b and Ln as unresolved candidates; report ray opening angle and sector q* instead."
            )
            suggested_parameters = {"treat_minor_axis_as_candidate": True}

        elif flag == "first_order_q_hint_ambiguous":
            severity = "medium"
            summary_en = "Automatic first-order peak selection is ambiguous across multiple radial peaks."
            summary_zh = "自动首阶峰提示存在多峰歧义。"
            recommended_action = (
                "Restrict q_window to the primary first-order scattering band and verify via sector I(q)."
            )
            suggested_parameters = {"narrow_q_window_to_first_order": True, "verify_sector_profile": True}

        elif flag in {"insufficient_occupied_sides", "insufficient_independent_side_support"}:
            severity = "high"
            summary_en = "Too few independent butterfly quadrants/sides have observed arc support."
            summary_zh = "具有有效观测弧支持的独立象限/侧边数量不足。"
            recommended_action = (
                "Inspect detector gap masks and q_window; if only two lobes exist, switch to sector/two-lobe analysis "
                "without mirroring missing quadrants."
            )
            suggested_parameters = {
                "inspect_mask_gaps": True,
                "do_not_mirror_missing_quadrants": True,
                "recommended_workflow": "sector",
            }

        elif flag in {"ill_conditioned_geometry", "poor_match", "solver_or_arc_support_unavailable"}:
            severity = "high"
            summary_en = "Ellipse geometry fit is ill-conditioned, poorly matched, or lacks arc support."
            summary_zh = "椭圆几何拟合病态、失配或缺少有效弧线支持。"
            recommended_action = (
                "Verify mask polarity, beamstop exclusion, q_window, and draw_axis_deg; do not promote "
                "the fit to quantitative estimates."
            )
            suggested_parameters = {
                "verify_mask_and_q_window": True,
                "keep_candidate_label": True,
            }

        elif flag in {
            "residual_exceeds_localization_scale",
            "per_arc_normal_residual_exceeds_localization_scale",
        }:
            severity = "medium"
            summary_en = "Geometric fit residuals exceed 3x the observed ridge localization uncertainty."
            summary_zh = "几何拟合残差超过观测脊线定位不确定度的 3 倍。"
            recommended_action = (
                "Check for straight-streak vs ellipse model mismatch, unmasked parasitic streaks, "
                "or mixed radial populations."
            )
            suggested_parameters = {"inspect_fit_residuals": True, "check_model_mismatch": True}

        elif flag in {
            "annular_trajectory_support_limited",
            "disconnected_observed_support",
            "short_observed_arc",
            "observed_support_infeasible",
            "arc_endpoint_or_manual_bound_dependent",
        }:
            severity = "medium"
            summary_en = "Observed arc support is short, disconnected, or endpoint-dependent."
            summary_zh = "观测弧支持较短、不连续或依赖端点/手动边界。"
            recommended_action = (
                "Adjust annular_radial_bins / annular_angle_bins to match detector Nyquist sampling "
                "and inspect detector gap intersections."
            )
            suggested_parameters = {"check_nyquist_binning": True, "inspect_mask_gaps": True}

        elif flag in {"uncalibrated_pixel_q", "spacing_unavailable_unknown_q_unit"}:
            severity = "info"
            summary_en = "Coordinates are in pixel-q or unknown q units; physical spacing L in nm is unavailable."
            summary_zh = "当前为未标定 pixel-q 或未知 q 单位，无法计算物理周期 L（nm）。"
            recommended_action = "Supply a calibrated pyFAI .poni file to convert q to nm^-1 and obtain nm spacings."
            suggested_parameters = {"require_poni_calibration": True}

        elif flag == "mixed_intensity_units":
            severity = "high"
            summary_en = "Sequence mixes calibrated absolute-intensity (e.g. cm^-1) and raw-count frames."
            summary_zh = "序列混杂了绝对强度标定帧（如 cm^-1）与原始计数帧。"
            recommended_action = (
                "Separate calibrated and raw-count runs; restrict raw-count frames to dimensionless "
                "normalized observables (q*, L*, A2, A4, angles) and never pool raw counts into Q_win."
            )
            suggested_parameters = {
                "separate_calibrated_and_raw_frames": True,
                "raw_frames_dimensionless_only": True,
            }

        elif flag == "timestamp_anomaly":
            severity = "medium"
            summary_en = "Sequence timestamps exhibit acquisition gaps, stalls, or non-monotonic chronology."
            summary_zh = "序列时间戳存在采集断档、停滞或非单调顺序异常。"
            recommended_action = (
                "Audit beamline acquisition logs and keep gaps explicit in kinetic plots without interpolating."
            )
            suggested_parameters = {"keep_time_gaps_explicit": True, "do_not_interpolate_gaps": True}

        elif flag == "coverage_drop":
            severity = "medium"
            summary_en = "Significant detector or azimuthal coverage drop detected in one or more frames."
            summary_zh = "部分帧出现显著的探测器或方位角有效覆盖率下降。"
            recommended_action = (
                "Inspect shutter drop, mask mismatch, or dead detector modules on flagged frames."
            )
            suggested_parameters = {"inspect_flagged_frame_masks": True}

        elif flag == "abrupt_orientation_transition":
            severity = "medium"
            summary_en = "Abrupt frame-to-frame orientation jump or regime shift detected in the sequence."
            summary_zh = "序列中检测到帧间主轴取向突变或结构机制转换。"
            recommended_action = (
                "Inspect the transition frames to distinguish true structural reorientation from mounting "
                "shifts or warm-start branch jumps; consider independent mode across the boundary."
            )
            suggested_parameters = {"batch_mode": "independent", "inspect_transition_frames": True}

        elif flag == "streak_contaminated":
            severity = "medium"
            summary_en = "Sharp parasitic radial streak detected; may bias azimuthal harmonics and ellipse fits."
            summary_zh = "检测到尖锐寄生径向条纹，可能干扰方位角谐波与椭圆拟合。"
            recommended_action = (
                "Mask the parasitic streak sector/ROI explicitly and re-run geometry analysis."
            )
            suggested_parameters = {"mask_parasitic_streak_roi": True, "recommended_workflow": "sector"}

        elif flag == "low_signal":
            severity = "high"
            summary_en = "Observed scattering intensity has very low signal-to-noise ratio in the detector domain."
            summary_zh = "观测散射图样信噪比过低，难以可靠提取特征峰或弧线。"
            recommended_action = (
                "Verify beam exposure/frame integrity or use coarser radial/angular binning for diagnostic checks."
            )
            suggested_parameters = {"recommended_workflow": "sector", "verify_beam_exposure": True}

        items.append(
            {
                "flag": str(flag),
                "severity": severity,
                "summary_en": summary_en,
                "summary_zh": summary_zh,
                "recommended_action": recommended_action,
                "suggested_parameters": _remediation_json_value(suggested_parameters),
            }
        )
    return items
