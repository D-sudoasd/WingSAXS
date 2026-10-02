"""Ellipse-based q-space analysis for apparent lamellar morphology.

This module exports fitted geometry and measurements that can support a
lamellar schematic.  It does not invert an ellipse into a unique 3-D structure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .butterfly_quality import NEAR_CIRCULAR_AXIS_RATIO_MIN
from .lamellar_adapter import (
    _execution_failure,
    _radial_peaks,
    _source_mapping,
    observed_direction_support,
)
from .lamellar import _radial_peak_direction_support
from .observables import (
    _SPACING_CENTER_REL_TOL,
    _normalized_q_unit,
    _q_to_nm_inverse_scale,
    ellipse_radius,
)
from .public_ellipse import _origin_centered_periods, public_ellipse_payload

_ROW_FIELDS = (
    "parameter",
    "value",
    "candidate_value",
    "unit",
    "status",
    "source",
    "reason",
    "branch_id",
)


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            candidate = value[name]
        else:
            try:
                candidate = getattr(value, name)
            except (AttributeError, TypeError, ValueError):
                continue
        if candidate is not None:
            return candidate
    return default


def _finite(value: Any) -> float | None:
    if isinstance(value, Mapping):
        mapped_value = value.get("value")
        value = mapped_value if mapped_value is not None else value.get("candidate_value")
    if isinstance(value, bool):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def _sequence(value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    if hasattr(value, "ndim") and getattr(value, "ndim", None) == 0:
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return [item()]
            except (TypeError, ValueError):
                return []
    return []


def _result_source(source: Any) -> tuple[Any, tuple[Any, ...]]:
    result = _get(source, "result")
    if result is None:
        return source, (source,)
    return result, (result, source)


def _source_value(containers: Sequence[Any], *names: str, default: Any = None) -> Any:
    for container in containers:
        value = _get(container, *names)
        if value is not None:
            return value
    return default


def _analysis_settings(containers: Sequence[Any]) -> Any:
    for container in containers:
        analysis = _get(container, "analysis_settings", "analysis", "settings")
        if isinstance(analysis, Mapping):
            nested = analysis.get("analysis")
            return nested if isinstance(nested, Mapping) else analysis
    return {}


def _geometry_source(containers: Sequence[Any]) -> Any:
    for container in containers:
        fit = _get(container, "ellipse_fit", "ellipse", "ellipse_result")
        if fit is not None:
            return fit
        observables = _get(container, "observables", "measurements")
        fit = _get(observables, "ellipse_fit", "ellipse", "ellipse_result")
        if fit is not None:
            return fit
    for container in containers:
        if any(
            _get(container, name) is not None
            for name in (
                "a",
                "semi_major",
                "parameters",
                "parameter_values",
                "ellipses",
                "ellipse_pair",
            )
        ):
            return container
    return None


def _q_unit(containers: Sequence[Any], ellipse: Mapping[str, Any]) -> str:
    candidates = [
        ellipse.get("q_unit"),
        *(_get(item, "q_unit", "source_q_unit") for item in containers),
    ]
    for container in containers:
        qmap = _get(container, "qmap", "geometry")
        candidates.append(_get(qmap, "q_unit", "source_q_unit", "unit"))
        qmap_metadata = _get(container, "qmap_metadata")
        candidates.append(_get(qmap_metadata, "q_unit", "source_q_unit", "unit"))
        observables = _get(container, "observables", "measurements")
        candidates.append(_get(observables, "q_unit", "source_q_unit"))
    for value in candidates:
        if value is not None and str(value).strip().casefold() not in {
            "",
            "none",
            "unknown",
        }:
            return str(value)
    return "unknown"


def _axes_and_center(ellipse: Mapping[str, Any]) -> tuple[dict[str, float | None], dict[str, float | None]]:
    parameters = ellipse.get("parameters", ellipse.get("parameter_values", {}))
    parameters = parameters if isinstance(parameters, Mapping) else {}
    values = {
        "a": _finite(_source_value((parameters, ellipse), "a", "semi_major")),
        "b": _finite(_source_value((parameters, ellipse), "b", "semi_minor")),
        "axis_ratio": _finite(
            _source_value((parameters, ellipse), "axis_ratio", "axes_ratio")
        ),
        "theta_deg": _finite(
            _source_value(
                (parameters, ellipse),
                "theta_deg",
                "ellipse_axis_tilt_deg",
                "angle_deg",
            )
        ),
    }
    if values["theta_deg"] is None:
        theta = _finite(_source_value((parameters, ellipse), "theta"))
        if theta is not None:
            values["theta_deg"] = math.degrees(theta)
    if values["b"] is None and values["a"] is not None and values["axis_ratio"] is not None:
        values["b"] = values["a"] * values["axis_ratio"]
    if values["axis_ratio"] is None and values["a"] not in (None, 0.0) and values["b"] is not None:
        values["axis_ratio"] = values["b"] / values["a"]

    center = _source_value((ellipse, parameters), "center", "centre")
    cx = cy = None
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and len(center) >= 2:
        cx, cy = _finite(center[0]), _finite(center[1])
    if cx is None:
        cx = _finite(_source_value((parameters, ellipse), "center_qx", "cx"))
    if cy is None:
        cy = _finite(_source_value((parameters, ellipse), "center_qy", "cy"))
    return values, {"cx": cx, "cy": cy}


def _parameter_evidence(ellipse: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = ellipse.get("quantitative_parameters")
    return evidence if isinstance(evidence, Mapping) else {}


def _evidence_for(evidence: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    aliases = {
        "a": ("a", "semi_major"),
        "b": ("b", "semi_minor"),
        "axis_ratio": ("axis_ratio", "axes_ratio"),
        "theta_deg": ("theta_deg", "ellipse_axis_tilt_deg", "angle_deg"),
    }
    for alias in aliases[name]:
        row = evidence.get(alias)
        if isinstance(row, Mapping):
            return row
    return None


def _candidate_and_value(
    name: str,
    geometry: Mapping[str, float | None],
    evidence: Mapping[str, Any],
    *,
    source_status: str,
) -> tuple[float | None, float | None, str, str]:
    row = _evidence_for(evidence, name)
    candidate = _finite(row.get("candidate_value")) if row else None
    if candidate is None:
        candidate = geometry.get(name)
    status = _text(row.get("status", row.get("identifiability_status")), "candidate") if row else "candidate"
    value = _finite(row.get("value", row.get("estimate"))) if row else None
    if status.casefold() not in {"available", "estimate"}:
        value = None
    reason = _text(row.get("reason", row.get("reasons"))) if row else ""
    if not reason:
        if value is not None and status.casefold() in {"available", "estimate"}:
            reason = "quantitative_parameter_available"
        elif candidate is not None:
            reason = "finite_fit_candidate_without_quantitative_evidence"
        else:
            failed = source_status.strip().casefold() in {
                "failed", "failure", "error", "invalid", "cancelled", "canceled", "aborted"
            }
            status = "failed" if failed else "unavailable"
            reason = "ellipse_parameter_unavailable"
    return candidate, value, status, reason


def _origin_centered(cx: float | None, cy: float | None, a: float | None, b: float | None) -> bool:
    if None in (cx, cy, a, b):
        return False
    assert cx is not None and cy is not None and a is not None and b is not None
    if a <= 0.0 or b <= 0.0:
        return False
    tolerance = _SPACING_CENTER_REL_TOL * max(1.0, abs(a), abs(b))
    return math.hypot(cx, cy) <= tolerance


def _q_conversion_factor(
    source_unit: Any,
    target_unit: Any,
    *,
    allow_same_unknown: bool = False,
) -> float | None:
    """Convert a q value from source units into target units when declared."""
    source_scale = _q_to_nm_inverse_scale(source_unit)
    target_scale = _q_to_nm_inverse_scale(target_unit)
    if source_scale is not None and target_scale is not None:
        return source_scale / target_scale
    source_key = _normalized_q_unit(source_unit)
    target_key = _normalized_q_unit(target_unit)
    if source_key == target_key and source_key not in {"", "none"}:
        if source_key != "unknown" or allow_same_unknown:
            return 1.0
    return None


def _model_radius_in_unit(
    member: Mapping[str, Any], angle_deg: float, target_unit: Any
) -> float | None:
    radius = _radius_at(member, angle_deg)
    if radius is None:
        return None
    factor = _q_conversion_factor(member.get("q_unit", "unknown"), target_unit)
    return radius * factor if factor is not None else None


def _member_id(member: Mapping[str, Any], index: int) -> Any:
    value = member.get("branch_id", member.get("fit_branch_id"))
    if isinstance(value, bool):
        return index
    if isinstance(value, (int, str)):
        return value
    return index


def _members(
    ellipse: Mapping[str, Any],
    values: Mapping[str, float | None],
    center: Mapping[str, float | None],
    reference_axis_deg: float,
    default_q_unit: str,
) -> list[dict[str, Any]]:
    raw_members = _sequence(ellipse.get("ellipses"))
    members: list[dict[str, Any]] = []
    if not raw_members and all(values.get(name) is not None for name in ("a", "b", "theta_deg")):
        theta = float(values["theta_deg"])
        raw_members = [
            {"theta_deg": reference_axis_deg + theta},
            {"theta_deg": reference_axis_deg - theta},
        ]
    for index, raw in enumerate(raw_members):
        if not isinstance(raw, Mapping) and not hasattr(raw, "__dict__"):
            continue
        normalized = public_ellipse_payload(raw)
        member_q_unit = _text(_get(normalized, "q_unit", "source_q_unit"), default_q_unit)
        global_to_member = _q_conversion_factor(
            default_q_unit, member_q_unit, allow_same_unknown=True
        )
        angle = _finite(_get(normalized, "angle_deg", "theta_deg"))
        if angle is None:
            theta = _finite(_get(normalized, "theta"))
            angle = math.degrees(theta) if theta is not None else None
        if angle is None and values.get("theta_deg") is not None:
            theta = float(values["theta_deg"])
            angle = reference_axis_deg + (theta if index == 0 else -theta)
        a = _finite(_get(normalized, "a", "semi_major"))
        b = _finite(_get(normalized, "b", "semi_minor"))
        ratio = _finite(_get(normalized, "axis_ratio", "axes_ratio"))
        if a is None:
            value = values.get("a")
            a = value * global_to_member if value is not None and global_to_member is not None else None
        if b is None:
            value = values.get("b")
            b = value * global_to_member if value is not None and global_to_member is not None else None
        if ratio is None:
            ratio = values.get("axis_ratio")
        if b is None and a is not None and ratio is not None:
            b = a * ratio
        if ratio is None and a not in (None, 0.0) and b is not None:
            ratio = b / a
        member_center = _get(normalized, "center", "centre")
        cx = cy = None
        if isinstance(member_center, Sequence) and not isinstance(member_center, (str, bytes)) and len(member_center) >= 2:
            cx, cy = _finite(member_center[0]), _finite(member_center[1])
        if cx is None:
            cx = _finite(_get(normalized, "center_qx", "cx"))
        if cy is None:
            cy = _finite(_get(normalized, "center_qy", "cy"))
        if cx is None:
            value = center.get("cx")
            cx = value * global_to_member if value is not None and global_to_member is not None else None
        if cy is None:
            value = center.get("cy")
            cy = value * global_to_member if value is not None and global_to_member is not None else None
        if angle is None or a is None or b is None:
            continue
        members.append(
            {
                "branch_id": _member_id(normalized, index),
                "angle_deg": float(angle),
                "a": a,
                "b": b,
                "axis_ratio": ratio,
                "center_qx": cx,
                "center_qy": cy,
                "q_unit": member_q_unit,
                "status": "candidate",
            }
        )
    return members


def _ray_ellipse_radius(
    angle_deg: float,
    a: float,
    b: float,
    theta_deg: float,
    cx: float,
    cy: float,
) -> float | None:
    """Return the nearest positive intersection of a q ray and translated ellipse."""
    phi = math.radians(angle_deg)
    theta = math.radians(theta_deg)
    ctheta, stheta = math.cos(theta), math.sin(theta)
    dx, dy = math.cos(phi), math.sin(phi)
    u_direction = dx * ctheta + dy * stheta
    v_direction = -dx * stheta + dy * ctheta
    u_center = cx * ctheta + cy * stheta
    v_center = -cx * stheta + cy * ctheta
    qa = (u_direction / a) ** 2 + (v_direction / b) ** 2
    qb = -2.0 * (u_direction * u_center / (a * a) + v_direction * v_center / (b * b))
    qc = (u_center / a) ** 2 + (v_center / b) ** 2 - 1.0
    discriminant = qb * qb - 4.0 * qa * qc
    if qa <= 0.0 or discriminant < 0.0:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates = [value for value in ((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)) if value > 0.0]
    return min(candidates) if candidates else None


def _radius_at(member: Mapping[str, Any], angle_deg: float) -> float | None:
    a, b = _finite(member.get("a")), _finite(member.get("b"))
    theta = _finite(member.get("angle_deg"))
    cx, cy = _finite(member.get("center_qx")), _finite(member.get("center_qy"))
    if None in (a, b, theta, cx, cy) or a <= 0.0 or b <= 0.0:
        return None
    if _origin_centered(cx, cy, a, b):
        radius = float(ellipse_radius(math.radians(angle_deg), a, b, math.radians(theta)))
        return radius if math.isfinite(radius) and radius > 0.0 else None
    return _ray_ellipse_radius(angle_deg, a, b, theta, cx, cy)


def _period_at(member: Mapping[str, Any], angle_deg: float, q_unit: str) -> float | None:
    a, b = _finite(member.get("a")), _finite(member.get("b"))
    theta = _finite(member.get("angle_deg"))
    cx, cy = _finite(member.get("center_qx")), _finite(member.get("center_qy"))
    if None in (a, b, theta, cx, cy) or not _origin_centered(cx, cy, a, b):
        return None
    _, period, _, _ = _origin_centered_periods(
        a=float(a),
        b=float(b),
        theta_deg=float(theta),
        cx=float(cx),
        cy=float(cy),
        q_unit=q_unit,
        draw_axis_deg=float(angle_deg),
    )
    return float(period) if math.isfinite(period) else None


def _curve(
    member: Mapping[str, Any],
    samples: int,
    supported_count: int,
    q_unit: str,
    support_source: str | None,
    support_statuses: Sequence[str],
    unassigned_support_count: int,
) -> dict[str, Any]:
    if member.get("center_qx") is None or member.get("center_qy") is None:
        supported = supported_count > 0
        return {
            "branch_id": member.get("branch_id"),
            "qx": [],
            "qy": [],
            "q_radius": [],
            "angle_deg": [],
            "q_unit": q_unit,
            "status": "unavailable_missing_ellipse_center",
            "geometry_status": member.get("status", "candidate"),
            "observed_support": supported,
            "support_count": int(supported_count),
            "support_source": support_source if supported else None,
            "support_statuses": sorted(support_statuses),
            "unassigned_support_count": int(unassigned_support_count),
            "source": "ellipse_fit",
            "reason": "ellipse_center_missing; measured observations are retained separately",
            "curve_role": "apparent_ellipse_geometry",
        }
    a, b = float(member["a"]), float(member["b"])
    theta = math.radians(float(member["angle_deg"]))
    cx, cy = float(member["center_qx"]), float(member["center_qy"])
    qx: list[float] = []
    qy: list[float] = []
    radii: list[float] = []
    angles: list[float] = []
    for index in range(samples):
        phi = 2.0 * math.pi * index / (samples - 1)
        local_a, local_b = a * math.cos(phi), b * math.sin(phi)
        x = cx + local_a * math.cos(theta) - local_b * math.sin(theta)
        y = cy + local_a * math.sin(theta) + local_b * math.cos(theta)
        qx.append(x)
        qy.append(y)
        radii.append(math.hypot(x, y))
        angles.append(float(math.degrees(math.atan2(y, x)) % 360.0))
    supported = supported_count > 0
    model_state = "available_model" if member.get("status") == "available" else "candidate_model"
    return {
        "branch_id": member.get("branch_id"),
        "qx": qx,
        "qy": qy,
        "q_radius": radii,
        "angle_deg": angles,
        "q_unit": q_unit,
        "status": f"{model_state}_with_observed_support" if supported else f"{model_state}_no_observed_support",
        "geometry_status": member.get("status", "candidate"),
        "observed_support": supported,
        "support_count": int(supported_count),
        "support_source": support_source if supported else None,
        "support_statuses": sorted(support_statuses),
        "unassigned_support_count": int(unassigned_support_count),
        "source": "ellipse_fit",
        "reason": "fitted_q_space_geometry_not_observed_pixels",
        "curve_role": "apparent_ellipse_geometry",
    }


def _geometry_parameter_row(
    name: str,
    unit: str,
    candidate: float | None,
    value: float | None,
    status: str,
    reason: str,
    branch_id: Any = None,
) -> dict[str, Any]:
    row = {
        "parameter": name,
        "value": value,
        "candidate_value": candidate if value is None else None,
        "unit": unit,
        "status": status,
        "source": "ellipse_fit",
        "reason": reason,
        "branch_id": branch_id,
    }
    return {key: row[key] for key in _ROW_FIELDS}


def _derived_row(
    name: str,
    candidate: float | None,
    status: str,
    reason: str,
    branch_id: Any = None,
) -> dict[str, Any]:
    return _geometry_parameter_row(
        name,
        "nm",
        candidate,
        candidate if status == "available" else None,
        status,
        reason,
        branch_id,
    )


def _sample_angles(count: int, draw_axis_deg: float) -> list[float]:
    angles = [360.0 * index / (count - 1) for index in range(count)]
    draw = float(draw_axis_deg) % 360.0
    if not any(math.isclose(angle % 360.0, draw, abs_tol=1e-10) for angle in angles):
        angles.append(draw)
        angles.sort()
    return angles


def _near_circle(ratio: float | None) -> bool:
    return ratio is not None and ratio >= NEAR_CIRCULAR_AXIS_RATIO_MIN


def analyze_lamellar_morphology(source: Any, *, angular_samples: int = 181) -> dict[str, Any]:
    """Summarize ellipse geometry, directional q radii, and observed support.

    ``period_nm`` values are conditional apparent periods from the fitted
    ellipse.  They require calibrated physical q and an origin-centred fit;
    ellipse geometry does not identify a unique three-dimensional structure.
    """
    if isinstance(angular_samples, bool) or not isinstance(angular_samples, int) or angular_samples < 3:
        raise ValueError("angular_samples must be an integer greater than or equal to 3")

    result, containers = _result_source(source)
    raw_ellipse = _geometry_source(containers)
    raw_mapping = _source_mapping(raw_ellipse) if raw_ellipse is not None else {}
    ellipse = public_ellipse_payload(raw_ellipse) if raw_ellipse is not None else {}
    values, center = _axes_and_center(ellipse)
    evidence = _parameter_evidence(raw_mapping)

    settings = _analysis_settings(containers)
    reference_axis = _finite(
        _source_value(containers, "reference_axis_deg")
    )
    if reference_axis is None:
        reference_axis = _finite(ellipse.get("reference_axis_deg"))
    if reference_axis is None:
        configured_draw = _finite(_get(settings, "draw_axis_deg"))
        reference_axis = configured_draw - 90.0 if configured_draw is not None else 0.0
    draw_axis = _finite(_source_value(containers, "draw_axis_deg"))
    if draw_axis is None:
        for container in containers:
            observables = _get(container, "observables", "measurements")
            draw_axis = _finite(_get(observables, "draw_axis_deg"))
            if draw_axis is not None:
                break
    if draw_axis is None:
        draw_axis = _finite(_get(settings, "draw_axis_deg"))
    if draw_axis is None:
        draw_axis = reference_axis + 90.0

    q_unit = _q_unit(containers, ellipse)
    q_scale = _q_to_nm_inverse_scale(q_unit)
    source_status = _text(
        _get(ellipse, "status", "solver_status", "measurement_status"),
        _text(_source_value(containers, "source_status", "status", "measurement_status"), "unavailable"),
    )
    source_reason = _text(
        _get(ellipse, "message", "reason", "error"),
        _text(_source_value(containers, "error", "diagnostic", "reason")),
    )
    failure_states = {"failed", "failure", "error", "invalid", "cancelled", "canceled", "aborted"}
    source_failed = source_status.strip().casefold() in failure_states
    source_mapping = _source_mapping(source)
    execution_failure = _execution_failure(source_mapping)
    execution_status = _source_value((source,), "execution_status")
    if execution_status is None and execution_failure is not None:
        execution_status = "failed"
    elif execution_status is None:
        execution_status = _get(source, "status", "state")
    execution_reason = _text(
        _source_value(containers, "execution_reason", "error", "diagnostic", "reason"),
        execution_failure or "",
    )
    failed_source = source_failed or execution_failure is not None

    for name in ("a", "b", "axis_ratio", "theta_deg"):
        candidate, value, status, _ = _candidate_and_value(
            name, values, evidence, source_status=source_status
        )
        if not failed_source and status.casefold() == "available" and value is not None:
            values[name] = value
        elif values.get(name) is None:
            values[name] = candidate if candidate is not None else value
    if values["b"] is None and values["a"] is not None and values["axis_ratio"] is not None:
        values["b"] = values["a"] * values["axis_ratio"]
    if values["axis_ratio"] is None and values["a"] not in (None, 0.0) and values["b"] is not None:
        values["axis_ratio"] = values["b"] / values["a"]

    result_mapping = _source_mapping(result)
    radial_peak_records = _radial_peaks(result_mapping)
    observed = observed_direction_support(result)
    model_support = (
        observed if observed else _radial_peak_direction_support(radial_peak_records)
    )
    support_source = (
        "accepted_observed_ridge_points"
        if observed
        else "retained_lobe_radial_peaks"
        if model_support
        else None
    )
    support_by_branch: dict[Any, int] = {}
    support_statuses_by_branch: dict[Any, set[str]] = {}
    unassigned_support_count = 0
    for row in model_support:
        branch = row.get("branch_id")
        if branch is None or (isinstance(branch, int) and branch < 0):
            unassigned_support_count += 1
            continue
        support_by_branch[branch] = support_by_branch.get(branch, 0) + 1
        support_statuses_by_branch.setdefault(branch, set()).add(
            _text(row.get("status"), "observed").casefold()
        )

    axis_ratio = values.get("axis_ratio")
    axis_unidentified = _near_circle(axis_ratio) or (
        "near_circular_ellipse_axis_unidentifiable" in _sequence(ellipse.get("flags"))
    )
    members = _members(ellipse, values, center, reference_axis, q_unit)
    for member in members:
        member["status"] = "candidate"
    all_geometry_available = bool(evidence) and all(
        _text((_evidence_for(evidence, name) or {}).get("status"), "").casefold() == "available"
        and _finite((_evidence_for(evidence, name) or {}).get("value")) is not None
        for name in ("a", "b", "axis_ratio", "theta_deg")
    )
    if all_geometry_available and not failed_source and not axis_unidentified:
        for member in members:
            member["status"] = "available"
    for member in members:
        member["observed_support"] = support_by_branch.get(member["branch_id"], 0) > 0
        member["support_count"] = int(support_by_branch.get(member["branch_id"], 0))
        member["support_source"] = support_source if member["observed_support"] else None
        member["support_statuses"] = sorted(
            support_statuses_by_branch.get(member["branch_id"], set())
        )
        member["unassigned_support_count"] = int(unassigned_support_count)

    candidate_geometry_exists = all(values.get(name) is not None for name in ("a", "b", "theta_deg"))
    if candidate_geometry_exists and not members:
        candidate_geometry_exists = False
    analysis_status = (
        "failed"
        if raw_ellipse is None and failed_source
        else "unavailable"
        if raw_ellipse is None
        else "failed"
        if not candidate_geometry_exists and failed_source
        else "unavailable"
        if not candidate_geometry_exists
        else "available"
        if all_geometry_available and not failed_source and not axis_unidentified
        else "candidate"
    )
    if raw_ellipse is None:
        source_reason = source_reason or "ellipse_fit_missing"
    elif not candidate_geometry_exists:
        source_reason = source_reason or "finite_ellipse_geometry_unavailable"
    elif source_failed:
        source_reason = source_reason or "source_fit_failed_finite_geometry_retained_as_candidate"
    elif execution_failure is not None:
        source_reason = source_reason or "frame_execution_failed_finite_fit_retained_as_candidate"

    flags = ["apparent_geometry_only", "nonunique_inverse_problem"]
    if source_failed:
        flags.append("ellipse_fit_failed")
    if execution_failure is not None:
        flags.append("source_execution_failed")
    if axis_unidentified:
        flags.append("near_circular_ellipse_axis_unidentifiable")
    if q_scale is None:
        flags.append("spacing_unavailable_unknown_q_unit")
    if center["cx"] is None or center["cy"] is None:
        flags.append("spacing_unavailable_missing_ellipse_center")
    elif all(values.get(name) is not None for name in ("a", "b")) and not _origin_centered(center["cx"], center["cy"], values.get("a"), values.get("b")):
        flags.append("spacing_unavailable_nonzero_center")

    parameter_rows: list[dict[str, Any]] = []
    parameter_states: dict[str, tuple[float | None, float | None, str, str]] = {}
    if raw_ellipse is not None:
        for name, label, unit in (
            ("a", "a", q_unit),
            ("b", "b", q_unit),
            ("axis_ratio", "axis_ratio", "dimensionless"),
            ("theta_deg", "ellipse_axis_tilt_deg", "deg"),
        ):
            candidate, value, status, reason = _candidate_and_value(
                name, values, evidence, source_status=source_status
            )
            if failed_source:
                value = None
                status = "candidate" if candidate is not None else ("failed" if source_failed or execution_failure else "unavailable")
                reason = "source_execution_failed_candidate_retained" if execution_failure is not None else "ellipse_fit_failed_candidate_retained"
            parameter_states[name] = (candidate, value, status, reason)
            parameter_rows.append(_geometry_parameter_row(label, unit, candidate, value, status, reason))

    dim_status = all(
        name in parameter_states
        and parameter_states[name][2].casefold() == "available"
        and parameter_states[name][1] is not None
        for name in ("a", "b")
    )
    theta_status = (
        "theta_deg" in parameter_states
        and parameter_states["theta_deg"][2].casefold() == "available"
        and parameter_states["theta_deg"][1] is not None
    )
    origin = _origin_centered(center["cx"], center["cy"], values.get("a"), values.get("b"))
    base_periods: tuple[float, float, float, tuple[str, ...]] | None = None
    if candidate_geometry_exists and center["cx"] is not None and center["cy"] is not None:
        base_periods = _origin_centered_periods(
            a=float(values["a"]),
            b=float(values["b"]),
            theta_deg=reference_axis + float(values["theta_deg"]),
            cx=float(center["cx"]),
            cy=float(center["cy"]),
            q_unit=q_unit,
            draw_axis_deg=float(draw_axis),
        )
    if base_periods is None:
        ln = l_major = None
        spacing_flags: tuple[str, ...] = ("spacing_unavailable_missing_ellipse_center",)
    else:
        ln_raw, _lz_default_raw, l_major_raw, spacing_flags = base_periods
        ln = ln_raw if math.isfinite(ln_raw) else None
        l_major = l_major_raw if math.isfinite(l_major_raw) else None
    derived_status = "available" if dim_status and origin and q_scale is not None and not failed_source else "candidate"
    derived_reason = (
        "origin_centered_ellipse_spacing_assumption"
        if derived_status == "available"
        else "derived_from_finite_ellipse_candidate"
    )
    if ln is None:
        derived_status = "unavailable"
        derived_reason = spacing_flags[0] if spacing_flags else "ellipse_spacing_unavailable"
    if raw_ellipse is not None:
        parameter_rows.append(_derived_row("Ln_from_minor_axis_nm", ln, derived_status, derived_reason))
        parameter_rows.append(_derived_row("L_from_major_axis_nm", l_major, derived_status, derived_reason))

    sample_angles = _sample_angles(angular_samples, float(draw_axis))
    directional_rows: list[dict[str, Any]] = []
    model_curves: list[dict[str, Any]] = []
    for member in members:
        branch_id = member["branch_id"]
        member_q_unit = _text(member.get("q_unit"), q_unit)
        member_q_scale = _q_to_nm_inverse_scale(member_q_unit)
        support_count = int(support_by_branch.get(branch_id, 0))
        model_curves.append(
            _curve(
                member,
                angular_samples,
                support_count,
                member_q_unit,
                support_source,
                sorted(support_statuses_by_branch.get(branch_id, set())),
                unassigned_support_count,
            )
        )
        member_centered = _origin_centered(
            member.get("center_qx"), member.get("center_qy"), member.get("a"), member.get("b")
        )
        _, branch_lz_raw, _, branch_spacing_flags = _origin_centered_periods(
            a=float(member["a"]),
            b=float(member["b"]),
            theta_deg=float(member["angle_deg"]),
            cx=float(member["center_qx"]),
            cy=float(member["center_qy"]),
            q_unit=member_q_unit,
            draw_axis_deg=float(draw_axis),
        ) if member.get("center_qx") is not None and member.get("center_qy") is not None else (
            math.nan, math.nan, math.nan, ("spacing_unavailable_missing_ellipse_center",)
        )
        branch_lz = branch_lz_raw if math.isfinite(branch_lz_raw) else None
        lz_status = (
            "available"
            if member.get("status") == "available"
            and dim_status
            and theta_status
            and member_centered
            and member_q_scale is not None
            and not failed_source
            else "candidate"
        )
        if support_count == 0 and branch_lz is not None:
            lz_status = (
                "model_only_no_observed_support"
                if lz_status == "available"
                else "candidate_model_no_observed_support"
            )
        if branch_lz is None:
            lz_status = "unavailable"
            lz_reason = branch_spacing_flags[0] if branch_spacing_flags else "ellipse_spacing_unavailable"
        elif support_count == 0:
            lz_reason = "no_retained_accepted_valid_points_for_branch"
        else:
            lz_reason = "origin_centered_ellipse_spacing_assumption" if lz_status == "available" else "derived_from_finite_ellipse_candidate"
        parameter_rows.append(_derived_row("Lz_from_draw_axis_nm", branch_lz, lz_status, lz_reason, branch_id))
        for angle in sample_angles:
            radius = _radius_at(member, angle)
            relative = (
                float(member["b"]) / radius
                if member_centered and radius is not None and radius > 0.0
                else None
            )
            period = (
                2.0 * math.pi / (radius * member_q_scale)
                if member_centered and radius is not None and member_q_scale is not None and radius > 0.0
                else None
            )
            if math.isclose(angle % 360.0, float(draw_axis) % 360.0, abs_tol=1e-10):
                period = branch_lz
            if not member_centered:
                period_reason = (
                    "spacing_unavailable_missing_ellipse_center"
                    if member.get("center_qx") is None or member.get("center_qy") is None
                    else "spacing_unavailable_nonzero_center"
                )
            elif member_q_scale is None:
                period_reason = "spacing_unavailable_unknown_q_unit"
            else:
                period_reason = "origin_centered_ellipse_spacing_assumption"
            supported = support_count > 0
            if radius is None:
                row_status = "unavailable_no_centered_ray_intersection"
                reason = "ellipse_ray_does_not_intersect_fitted_geometry_or_center_missing"
            elif not supported:
                row_status = (
                    "available_model_no_observed_support"
                    if member.get("status") == "available"
                    else "candidate_model_no_observed_support"
                )
                reason = "no_retained_accepted_valid_points_for_branch"
            else:
                row_status = (
                    "available_model_with_observed_support"
                    if member.get("status") == "available"
                    else "candidate_model_with_observed_support"
                )
                reason = (
                    "fitted_ellipse_direction_with_retained_ridge_support"
                    if support_source == "accepted_observed_ridge_points"
                    else "fitted_ellipse_direction_with_retained_radial_peak_support"
                )
            directional_rows.append(
                {
                    "branch_id": branch_id,
                    "angle_deg": float(angle),
                    "q_radius": radius,
                    "q_unit": member_q_unit,
                    "period_nm": period,
                    "period_reason": period_reason,
                    "relative_period": relative,
                    "status": row_status,
                    "geometry_status": member.get("status", "candidate"),
                    "observed_support": supported,
                    "support_count": support_count,
                    "support_source": support_source if supported else None,
                    "support_statuses": sorted(support_statuses_by_branch.get(branch_id, set())),
                    "unassigned_support_count": int(unassigned_support_count),
                    "source": "ellipse_fit",
                    "reason": reason,
                }
            )

    observed_directions: list[dict[str, Any]] = []
    for item in observed:
        branch_id = item.get("branch_id")
        angle = _finite(item.get("angle_deg"))
        observation_unit = _text(item.get("q_unit"), q_unit)
        matching = next((member for member in members if member.get("branch_id") == branch_id), None)
        predicted = (
            _model_radius_in_unit(matching, angle, observation_unit)
            if matching is not None and angle is not None
            else None
        )
        predicted_period = (
            _period_at(matching, angle, _text(matching.get("q_unit"), q_unit))
            if matching is not None and angle is not None
            else None
        )
        prediction_status = matching.get("status", "candidate") if matching is not None else None
        comparison_reason = "model_and_observation_converted_to_observation_q_unit"
        if matching is None and branch_id is None and members:
            predictions = [
                (_model_radius_in_unit(member, angle, observation_unit), member)
                for member in members
                if angle is not None
            ]
            finite_predictions = [
                (radius, member) for radius, member in predictions if radius is not None
            ]
            if len(finite_predictions) == 1:
                predicted, matching = finite_predictions[0]
            elif len(finite_predictions) == len(members) and finite_predictions:
                radii = [radius for radius, _ in finite_predictions]
                scale = max(1.0, *(abs(radius) for radius in radii))
                if max(radii) - min(radii) <= 8.0 * math.ulp(scale):
                    predicted = sum(radii) / len(radii)
                    statuses = {member.get("status", "candidate") for _, member in finite_predictions}
                    prediction_status = "available" if statuses == {"available"} else "candidate"
                    matching = finite_predictions[0][1]
            if matching is not None:
                predicted_period = (
                    _period_at(matching, angle, _text(matching.get("q_unit"), q_unit))
                    if angle is not None
                    else None
                )
                if prediction_status is None:
                    prediction_status = matching.get("status", "candidate")
            elif not finite_predictions and members:
                model_radii = [
                    _radius_at(member, angle)
                    for member in members
                    if angle is not None
                ]
                if any(radius is not None for radius in model_radii):
                    comparison_reason = "q_units_incommensurate_or_unknown"
                else:
                    comparison_reason = "ellipse_model_unavailable_for_observed_direction"
            else:
                comparison_reason = "multiple_branch_models_disagree_at_observed_angle"
        elif matching is None:
            comparison_reason = "no_model_member_assigned_to_observed_branch"
        elif predicted is None:
            raw_prediction = _radius_at(matching, angle) if angle is not None else None
            comparison_reason = (
                "q_units_incommensurate_or_unknown"
                if raw_prediction is not None
                else "ellipse_model_unavailable_for_observed_direction"
            )
        q_radius = _finite(item.get("q"))
        observation_status = _text(item.get("status"), "observed")
        observed_directions.append(
            {
                "branch_id": branch_id,
                "angle_deg": angle,
                "q_radius": q_radius,
                "q_unit": observation_unit,
                "observation_q_unit": observation_unit,
                "comparison_q_unit": observation_unit if predicted is not None else None,
                "model_q_unit": observation_unit if predicted is not None else None,
                "period_nm": None,
                "period_reason": "observed_q_direction_has_no_bragg_order_assigned",
                "status": observation_status,
                "observation_kind": "retained_observed_ridge",
                "source": item.get("source", "ridge_points"),
                "reason": item.get("reason") or "accepted_valid_observed_ridge_point; no Bragg order assigned",
                "support_count": 1,
                "branch_support_count": support_by_branch.get(branch_id, 0) if branch_id is not None else None,
                "point_index": item.get("point_index"),
                "qx": item.get("qx"),
                "qy": item.get("qy"),
                "model_q_radius": predicted,
                "model_period_nm": predicted_period,
                "model_geometry_status": prediction_status,
                "residual_q": q_radius - predicted if q_radius is not None and predicted is not None else None,
                "residual_q_unit": observation_unit if predicted is not None else None,
                "comparison_reason": comparison_reason,
                "branch_id_source": item.get("branch_id_source"),
                "source_branch_id": item.get("source_branch_id"),
                "source_branch_id_source": item.get("source_branch_id_source"),
                "coverage": item.get("coverage"),
                "support": item.get("support"),
                "localization_sigma_q": item.get("localization_sigma_q"),
                "normal_fwhm_q": item.get("normal_fwhm_q"),
            }
        )

    observed_radial_peaks: list[dict[str, Any]] = []
    for peak in radial_peak_records:
        angle = _finite(peak.get("angle_deg"))
        q = _finite(peak.get("q"))
        peak_unit = _text(peak.get("q_unit"), q_unit)
        q_nm_inv = _finite(peak.get("q_nm_inv"))
        branch = peak.get("branch_id")
        comparisons = []
        for member in members:
            if branch is not None and member.get("branch_id") != branch:
                continue
            comparison_unit = peak_unit
            observed_peak_q = q
            predicted = (
                _model_radius_in_unit(member, angle, comparison_unit)
                if angle is not None
                else None
            )
            if predicted is None and q_nm_inv is not None and angle is not None:
                comparison_unit = "nm^-1"
                observed_peak_q = q_nm_inv
                predicted = _model_radius_in_unit(member, angle, comparison_unit)
            raw_prediction = _radius_at(member, angle) if angle is not None else None
            comparisons.append(
                {
                    "branch_id": member.get("branch_id"),
                    "model_q_radius": predicted,
                    "observed_q": observed_peak_q if predicted is not None else None,
                    "observed_minus_model_q": (
                        observed_peak_q - predicted
                        if observed_peak_q is not None and predicted is not None
                        else None
                    ),
                    "q_unit": comparison_unit if predicted is not None else None,
                    "status": "compared" if observed_peak_q is not None and predicted is not None else "unavailable",
                    "reason": (
                        "model_and_observation_converted_to_comparison_q_unit"
                        if predicted is not None
                        else "q_units_incommensurate_or_unknown"
                        if raw_prediction is not None
                        else "ellipse_model_unavailable_for_observed_direction"
                    ),
                }
            )
        observed_radial_peaks.append(
            {
                "peak_index": peak.get("index"),
                "branch_id": branch,
                "angle_deg": angle,
                "q": q,
                "q_unit": peak_unit,
                "q_nm_inv": q_nm_inv,
                "status": peak.get("status", "available"),
                "source": "lobe_radial_peaks",
                "reason": peak.get("reason", ""),
                "comparisons": comparisons,
            }
        )

    ellipse_summary = None
    if raw_ellipse is not None:
        ellipse_summary = {
            "a": values.get("a"),
            "b": values.get("b"),
            "axis_ratio": values.get("axis_ratio"),
            "center_qx": center.get("cx"),
            "center_qy": center.get("cy"),
            "reference_axis_deg": float(reference_axis),
            "ellipse_axis_tilt_deg": values.get("theta_deg"),
            "q_unit": q_unit,
            "member_q_units": [member.get("q_unit", q_unit) for member in members],
            "source_status": source_status,
            "status": analysis_status,
            "reason": source_reason,
            "support_source": support_source,
            "observed_support_count": len(model_support),
            "unassigned_support_count": int(unassigned_support_count),
            "axis_direction_status": "not_identified_near_circle" if axis_unidentified else "apparent_ellipse_axis_only",
            "axis_direction_reason": (
                "near_circular_ellipse_axis_unidentifiable"
                if axis_unidentified
                else "ellipse_axis_does_not_identify_a_structural_normal"
            ),
            "members": members,
        }
    assumptions = [
        "Ellipse parameters describe apparent q-space geometry of measured intensity.",
        "Ellipse-derived periods are conditional on an origin-centred ellipse and calibrated physical q.",
        "relative_period is the dimensionless ratio b/R and is reported only for an origin-centred ellipse.",
        "Ellipse geometry does not identify a unique three-dimensional lamellar structure or structural normal.",
        "Sampled ellipse curves are fitted geometry; observed support is listed separately.",
    ]
    return {
        "schema_version": 1,
        "status": analysis_status,
        "source_status": source_status,
        "source_reason": source_reason,
        "execution_status": execution_status,
        "execution_reason": execution_reason,
        "q_unit": q_unit,
        "observed_support_source": support_source,
        "observed_support_count": len(model_support),
        "unassigned_support_count": int(unassigned_support_count),
        "branch_support_counts": [
            {
                "branch_id": branch_id,
                "support_count": int(count),
                "support_source": support_source,
                "support_statuses": sorted(support_statuses_by_branch.get(branch_id, set())),
            }
            for branch_id, count in sorted(
                support_by_branch.items(), key=lambda item: str(item[0])
            )
        ],
        "reference_axis_deg": float(reference_axis),
        "draw_axis_deg": float(draw_axis),
        "ellipse": ellipse_summary,
        "flags": list(dict.fromkeys(flags)),
        "assumptions": assumptions,
        "parameter_rows": parameter_rows,
        "directional_rows": directional_rows,
        "observed_directions": observed_directions,
        "observed_radial_peaks": observed_radial_peaks,
        "model_curves": model_curves,
    }


__all__ = ["analyze_lamellar_morphology"]
