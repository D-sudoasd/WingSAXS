"""Scientific source adaptation and deterministic lamellar schematic geometry."""

from __future__ import annotations
import os
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np

from .lamellar_models import LamellarScene, LamellarSettings, _empty_scene
from .lamellar_utils import (
    _finite,
    _finite_positive,
    _normalise_q_unit,
    _q_scale_to_nm,
    _read,
)
from .lamellar_adapter import (
    _exact_integer,
    _ellipse_components,
    _ellipse_parameter_records,
    _ellipse_payload,
    _parameter_value,
    _direction_for_group,
    _execution_failure,
    _group_populations,
    _metadata_base,
    _period_record,
    _q_unit_from_source,
    _radial_peaks,
    _source_identity,
    _source_mapping,
    _stable_seed,
    observed_direction_support,
)
from .public_ellipse import _origin_centered_periods

_MISSING = object()
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
_BLUE = np.asarray((0.20, 0.45, 0.72, 0.82), dtype=float)
_ORANGE = np.asarray((0.92, 0.47, 0.18, 0.82), dtype=float)


def _axial_distance_deg(first: float, second: float) -> float:
    difference = (float(second) - float(first) + 90.0) % 180.0 - 90.0
    return abs(difference)


def _direction_only_group(
    rows: list[Mapping[str, Any]],
) -> tuple[float | None, dict[str, float], str | None]:
    angles = np.asarray([float(row["angle_deg"]) for row in rows], dtype=float)
    doubled = np.radians(2.0 * angles)
    resultant = float(np.abs(np.mean(np.exp(1j * doubled)))) if angles.size else 0.0
    mean = float(np.degrees(0.5 * np.angle(np.mean(np.exp(1j * doubled))))) if angles.size else 0.0
    deviations = [_axial_distance_deg(mean, float(value)) for value in angles]
    maximum_deviation = max(deviations, default=0.0)
    diagnostics = {
        "axial_resultant": resultant,
        "maximum_axial_deviation_deg": float(maximum_deviation),
    }
    if resultant < float(np.cos(np.deg2rad(30.0))) or maximum_deviation > 15.0:
        return None, diagnostics, (
            "Observed ridge directions without branch labels are too dispersed "
            f"to define one lamellar normal (axial resultant={resultant:.3g}, "
            f"maximum axial deviation={maximum_deviation:.3g}°)."
        )
    return mean, diagnostics, None


def _direction_source_label(rows: Sequence[Mapping[str, Any]]) -> str:
    sources = {str(row.get("source", "")) for row in rows}
    if sources and all("lobe_radial_peaks" in value for value in sources):
        return "retained_lobe_radial_peaks"
    if sources and all("ridge" in value for value in sources):
        return "accepted_observed_ridge_points"
    return "retained_observed_q_directions"


def _direction_populations(
    support: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Group only explicit branches; retain unlabelled support as one direction."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    unassigned: list[dict[str, Any]] = []
    for row in support:
        branch = row.get("branch_id")
        try:
            branch_id = int(branch) if branch is not None else None
        except (TypeError, ValueError, OverflowError):
            branch_id = None
        if branch_id is None or branch_id < 0:
            unassigned.append(row)
        else:
            grouped.setdefault(branch_id, []).append(row)
    populations: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"unassigned_direction": None}
    for branch_id, rows in sorted(grouped.items()):
        angle, inconsistent = _direction_for_group(rows)
        orientation_source = _direction_source_label(rows)
        if inconsistent:
            diagnostics.setdefault("inconsistent_explicit_branches", []).append(branch_id)
        populations.append(
            {
                "branch_id": branch_id,
                "period": None,
                "length_unit": "relative",
                "angle_deg": angle,
                "status": "candidate" if inconsistent else "direction_only",
                "orientation_source": orientation_source,
                "branch_id_source": str(rows[0].get("branch_id_source") or "source_branch_id"),
                "support_indices": [int(row["point_index"]) for row in rows],
                "support_q_values": [float(row["q"]) for row in rows],
                "support_q_units": [str(row["q_unit"]) for row in rows],
                "inconsistent_direction_support": bool(inconsistent),
            }
        )
    if unassigned:
        angle, group_diagnostics, reason = _direction_only_group(unassigned)
        orientation_source = _direction_source_label(unassigned)
        diagnostics["unassigned_direction"] = group_diagnostics
        if reason is not None:
            diagnostics["unassigned_direction"]["reason"] = reason
        populations.append(
            {
                "branch_id": -1,
                "branch_label": "Observed direction",
                "branch_id_source": "unassigned_direction_only",
                "period": None,
                "length_unit": "relative",
                "angle_deg": None if reason is not None else float(angle),
                "status": "unavailable" if reason is not None else "direction_only",
                "period_reason": reason,
                "orientation_source": orientation_source,
                "support_indices": [int(row["point_index"]) for row in unassigned],
                "support_q_values": [float(row["q"]) for row in unassigned],
                "support_q_units": [str(row["q_unit"]) for row in unassigned],
            }
        )
    return populations, diagnostics, reason if unassigned else None


def _ellipse_field(
    ellipse: Any, names: Sequence[str]
) -> tuple[float | None, bool, str]:
    """Read one fitted ellipse parameter while retaining candidate status."""

    containers = (
        _read(ellipse, ("quantitative_parameters",), None),
        _read(ellipse, ("parameters", "parameter_values"), None),
        ellipse,
    )
    found: tuple[float | None, bool, str] | None = None
    for container in containers:
        raw = _read(container, names, _MISSING)
        if raw is _MISSING:
            continue
        found = _parameter_value(raw)
        break
    if found is None:
        return None, False, "unavailable"
    value, candidate, status = found
    fit_status = str(
        _read(ellipse, ("status", "solver_status", "measurement_status"), "") or ""
    ).strip().lower()
    fit_success = _read(ellipse, ("success",), _MISSING)
    if fit_status in {"candidate", "estimate", "provisional", "undetermined", "failed", "failure", "error"} or fit_success is False:
        candidate = True
        status = fit_status or "candidate"
    return value, candidate, status


def _ellipse_center(ellipse: Any) -> tuple[float, float] | None:
    center = _read(ellipse, ("center", "centre"), _MISSING)
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and len(center) >= 2:
        cx, cy = _finite(center[0]), _finite(center[1])
    else:
        cx = _finite(_read(ellipse, ("center_qx", "cx"), _MISSING))
        cy = _finite(_read(ellipse, ("center_qy", "cy"), _MISSING))
    if cx is None or cy is None:
        parameters = _read(ellipse, ("parameters", "parameter_values"), None)
        if cx is None:
            cx = _finite(_read(parameters, ("center_qx", "cx"), _MISSING))
        if cy is None:
            cy = _finite(_read(parameters, ("center_qy", "cy"), _MISSING))
    return (float(cx), float(cy)) if cx is not None and cy is not None else None


def _origin_centered(
    center: tuple[float, float] | None, a: float | None, b: float | None
) -> bool:
    if center is None or a is None or b is None or a <= 0.0 or b <= 0.0:
        return False
    tolerance = 1.0e-8 * max(1.0, abs(a), abs(b))
    return float(np.hypot(*center)) <= tolerance


def _ellipse_direction_members(
    ellipse: Any, draw_axis_deg: float, source_q_unit: str
) -> list[dict[str, Any]]:
    """Normalize fitted q-space ellipses using the canonical member convention."""

    if ellipse is None:
        return []
    a, candidate_a, _ = _ellipse_field(ellipse, ("a", "semi_major", "major_axis"))
    b, candidate_b, _ = _ellipse_field(ellipse, ("b", "semi_minor", "minor_axis"))
    _, candidate_b_alias, _, _, _, _ = _ellipse_components(ellipse)
    candidate_b = candidate_b or candidate_b_alias
    ratio, candidate_ratio, _ = _ellipse_field(ellipse, ("axis_ratio", "axes_ratio"))
    if a is None and b is not None and ratio is not None and ratio > 0.0:
        a = b / ratio
    if b is None and a is not None and ratio is not None and ratio > 0.0:
        b = a * ratio
    theta, candidate_theta, _ = _ellipse_field(
        ellipse, ("theta_deg", "ellipse_axis_tilt_deg", "angle_deg")
    )
    if theta is None:
        theta_rad, candidate_theta_rad, _ = _ellipse_field(ellipse, ("theta",))
        if theta_rad is not None:
            theta = float(np.degrees(theta_rad))
            candidate_theta = candidate_theta_rad
    reference_axis = _finite(_read(ellipse, ("reference_axis_deg",), _MISSING))
    if reference_axis is None:
        reference_axis = float(draw_axis_deg) - 90.0
    center = _ellipse_center(ellipse)
    global_q_unit = str(
        _read(ellipse, ("q_unit", "source_q_unit"), source_q_unit) or source_q_unit
    )
    raw_members = _read(ellipse, ("ellipses", "ellipse_pair"), _MISSING)
    if isinstance(raw_members, Sequence) and not isinstance(raw_members, (str, bytes)):
        members = list(raw_members)
    elif isinstance(raw_members, Mapping) or hasattr(raw_members, "__dict__"):
        members = [raw_members]
    else:
        members = []
    if not members and a is not None and b is not None and theta is not None:
        # This is the canonical two-member representation of one symmetric
        # fit; it defines q-space geometry only, never observed scene normals.
        members = [None, None]
    normalized: list[dict[str, Any]] = []
    fit_status = str(
        _read(ellipse, ("status", "solver_status", "measurement_status"), "") or ""
    ).strip().lower()
    fit_success = _read(ellipse, ("success",), _MISSING)
    fit_candidate = fit_status in {
        "candidate", "estimate", "provisional", "undetermined", "failed", "failure", "error"
    } or fit_success is False
    for index, raw in enumerate(members):
        local_a, local_candidate_a, _ = _ellipse_field(raw, ("a", "semi_major", "major_axis")) if raw is not None else (None, False, "unavailable")
        local_b, local_candidate_b, _ = _ellipse_field(raw, ("b", "semi_minor", "minor_axis")) if raw is not None else (None, False, "unavailable")
        local_theta, local_candidate_theta, _ = _ellipse_field(
            raw, ("angle_deg", "theta_deg", "ellipse_axis_tilt_deg")
        ) if raw is not None else (None, False, "unavailable")
        if local_theta is None and raw is not None:
            theta_rad, candidate_rad, _ = _ellipse_field(raw, ("theta",))
            if theta_rad is not None:
                local_theta = float(np.degrees(theta_rad))
                local_candidate_theta = candidate_rad
        if local_a is None:
            local_a = a
        if local_b is None:
            local_b = b
        if local_theta is None and theta is not None:
            local_theta = float(reference_axis + (theta if index == 0 else -theta))
            local_candidate_theta = candidate_theta
        if local_a is None or local_b is None or local_theta is None:
            continue
        local_center = _ellipse_center(raw) if raw is not None else None
        if local_center is None:
            local_center = center
        local_unit = str(
            _read(raw, ("q_unit", "source_q_unit"), global_q_unit) if raw is not None else global_q_unit
        )
        source_branch = _read(raw, ("branch_id", "fit_branch_id"), _MISSING) if raw is not None else _MISSING
        branch_id = _exact_integer(source_branch) if source_branch is not _MISSING else None
        branch_source = "explicit_fit_member_branch_id" if branch_id is not None else None
        if branch_id is None and len(members) == 2:
            branch_id = index
            branch_source = "canonical_fit_member_order"
        normalized.append(
            {
                "branch_id": branch_id,
                "branch_id_source": branch_source,
                "a": float(local_a),
                "b": float(local_b),
                "theta_deg": float(local_theta),
                "center": local_center,
                "q_unit": local_unit,
                "candidate": bool(
                    candidate_a
                    or candidate_b
                    or candidate_ratio
                    or candidate_theta
                    or local_candidate_a
                    or local_candidate_b
                    or local_candidate_theta
                    or fit_candidate
                ),
            }
        )
    return normalized


def _ellipse_q_radius(member: Mapping[str, Any], angle_deg: float) -> float | None:
    a, b = float(member["a"]), float(member["b"])
    if a <= 0.0 or b <= 0.0 or not _origin_centered(member.get("center"), a, b):
        return None
    delta = np.deg2rad(float(angle_deg) - float(member["theta_deg"]))
    denominator = float(np.hypot(b * np.cos(delta), a * np.sin(delta)))
    radius = a * b / denominator if denominator > 0.0 else float("nan")
    return float(radius) if np.isfinite(radius) and radius > 0.0 else None


def _radial_peak_direction_support(
    peaks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose retained radial-peak observations as direction evidence."""

    rows: list[dict[str, Any]] = []
    for item in peaks:
        rows.append(
            {
                "point_index": int(item["index"]),
                "q": float(item["q"]),
                "angle_deg": float(item["angle_deg"]),
                "q_unit": str(item["q_unit"]),
                "branch_id": item.get("branch_id"),
                "branch_id_source": "radial_peak_branch_id" if item.get("branch_id") is not None else None,
                "source": "lobe_radial_peaks",
                "status": str(item.get("status", "available")),
                "reason": str(item.get("reason", "")),
                "valid": True,
                "accepted": True,
            }
        )
    return rows


def _apply_ellipse_directional_periods(
    root: Mapping[str, Any],
    settings: LamellarSettings,
    populations: list[dict[str, Any]],
    support: list[dict[str, Any]],
    draw_axis_deg: float,
    source_q_unit: str,
    parameter_sources: list[dict[str, Any]],
    flags: list[str],
    assumptions: list[str],
) -> str | None:
    """Assign periods from fitted ellipse radii along retained directions."""

    ellipse = _ellipse_payload(root)
    b, candidate_b, _, _, ellipse_unit, existing_period = _ellipse_components(ellipse)
    unit_value = str(
        _read(ellipse, ("q_unit", "source_q_unit"), source_q_unit or ellipse_unit)
        or source_q_unit
        or ellipse_unit
    )
    unit_scale = _q_scale_to_nm(unit_value)
    support_units = {_normalise_q_unit(row.get("q_unit")) for row in support}
    reference_q = float(support[0]["q"]) if support else None
    support_unit_match = (
        len(support_units) == 1
        and next(iter(support_units), "unknown") == _normalise_q_unit(unit_value)
    )
    base = float(settings.reference_period) if settings.reference_period is not None else 1.0

    # Keep Ln as a global ellipse parameter. Directional periods below are
    # evaluated independently at each observed normal.
    global_period = None
    global_status = "unavailable"
    global_reason = "ellipse minor axis b is missing or nonpositive"
    ellipse_center = _ellipse_center(ellipse)
    if ellipse_center is not None and b is not None and b > 0.0:
        if _origin_centered(ellipse_center, b, b):
            if unit_scale is not None:
                global_period = (
                    float(existing_period)
                    if existing_period is not None and existing_period > 0.0
                    else float(2.0 * np.pi / (b * unit_scale))
                )
                global_reason = "global 2π/b reference from origin-centred ellipse and calibrated q"
            elif reference_q is not None and support_unit_match:
                global_period = float(base * reference_q / b)
                global_reason = "global b reference on an explicit relative scale; q units are not calibrated"
                flags.append("spacing_unavailable_unknown_q_unit")
            else:
                global_reason = "ellipse b reference requires one matching observed q unit for relative normalization"
            if global_period is not None:
                global_status = "candidate" if candidate_b else "available"
        else:
            global_reason = "nonzero ellipse centre makes origin-centred spacing unavailable"
            flags.append("spacing_unavailable_nonzero_center")
    elif ellipse_center is None:
        global_reason = "ellipse centre is unavailable, so origin centring cannot be checked"
    parameter_sources.append(
        _period_record(
            "period_from_ellipse_minor_axis",
            None if global_status != "available" else global_period,
            "nm" if unit_scale is not None else "relative",
            "ellipse_fit",
            global_status,
            global_reason,
            candidate_value=global_period if global_status == "candidate" else None,
        )
    )

    members = _ellipse_direction_members(ellipse, draw_axis_deg, source_q_unit)
    if not members:
        reason = "finite ellipse a, b, and theta geometry is required for direction-dependent periods"
        for population in populations:
            population.update(period=None, status="unavailable", period_reason=reason)
        return reason

    first_error: str | None = None
    for population in populations:
        angle = _finite(population.get("angle_deg"))
        branch_id = _exact_integer(population.get("branch_id"))
        if angle is None:
            reason = "observed direction group has no finite angle"
            population.update(period=None, status="unavailable", period_reason=reason)
            first_error = first_error or reason
            continue
        if len(members) == 1:
            sole_member = members[0]
            member_branch = _exact_integer(sole_member.get("branch_id"))
            if member_branch is None:
                # A genuinely unlabelled single ellipse is one global fitted
                # geometry and can be evaluated along any retained direction.
                candidates = members
                member_assignment_resolved = False
            elif branch_id is not None and branch_id == member_branch:
                candidates = members
                member_assignment_resolved = True
            else:
                # A source-labelled member supports only its corresponding
                # observed branch; a singleton does not erase that identity.
                candidates = []
                member_assignment_resolved = False
        elif branch_id is not None and branch_id >= 0:
            candidates = [item for item in members if item.get("branch_id") == branch_id]
            member_assignment_resolved = True
        else:
            candidates = members
            member_assignment_resolved = False
        radii = [_ellipse_q_radius(member, angle) for member in candidates]
        finite_candidates = [
            (member, radius)
            for member, radius in zip(candidates, radii, strict=True)
            if radius is not None
        ]
        if not finite_candidates:
            reason = (
                f"no origin-centred fitted ellipse member matches observed branch {branch_id}"
                if branch_id is not None and branch_id >= 0
                else "observed direction has no finite origin-centred ellipse member radius"
            )
            population.update(period=None, status="unavailable", period_reason=reason)
            first_error = first_error or reason
            continue
        if len(finite_candidates) > 1:
            radii_values = [float(radius) for _, radius in finite_candidates]
            radius_scale = max(np.finfo(float).tiny, *(abs(value) for value in radii_values))
            tolerance = 64.0 * np.finfo(float).eps * radius_scale
            if max(radii_values) - min(radii_values) > tolerance:
                reason = (
                    "unassigned observed direction intersects fitted ellipse members "
                    f"at different q radii ({', '.join(f'{value:.6g}' for value in radii_values)}); "
                    "an ellipse member cannot be assigned without a measured branch ID"
                )
                population.update(period=None, status="unavailable", period_reason=reason)
                first_error = first_error or reason
                continue
        member, radius = finite_candidates[0]
        member_unit = str(member.get("q_unit") or unit_value)
        scale = _q_scale_to_nm(member_unit)
        if scale is not None:
            # Reuse the canonical physical-spacing calculation and its origin
            # tolerance/flags, evaluated along this observed q-space direction.
            a, b_member = float(member["a"]), float(member["b"])
            _, period, _, spacing_flags = _origin_centered_periods(
                a=a,
                b=b_member,
                theta_deg=float(member["theta_deg"]),
                cx=float(member["center"][0]),
                cy=float(member["center"][1]),
                q_unit=member_unit,
                draw_axis_deg=float(angle),
            )
            flags.extend(flag for flag in spacing_flags if flag not in flags)
            unit = "nm"
        else:
            if not support_unit_match or reference_q is None:
                reason = "relative ellipse spacing requires one retained observed q unit matching the fitted ellipse"
                population.update(period=None, status="unavailable", period_reason=reason)
                first_error = first_error or reason
                continue
            period = float(base * reference_q / float(radius))
            unit = "relative"
            flags.append("spacing_unavailable_unknown_q_unit")
            assumptions.append(
                "Ellipse directional periods with pixel-q or unknown q units use an explicit relative scale; no nm value is assigned."
            )
        if not np.isfinite(period) or period <= 0.0:
            reason = "ellipse direction-dependent period is unavailable for the fitted geometry and q unit"
            population.update(period=None, status="unavailable", period_reason=reason)
            first_error = first_error or reason
            continue
        candidate = bool(
            member.get("candidate")
            or population.get("status") == "candidate"
            or population.get("inconsistent_direction_support")
        )
        support_statuses = {
            str(row.get("status", "")).strip().lower()
            for row in support
            if int(row.get("point_index", -1)) in set(population.get("support_indices", ()))
        }
        candidate |= bool(support_statuses & {"candidate", "provisional", "undetermined", "estimate"})
        population.update(
            period=float(period),
            length_unit=unit,
            status="candidate" if candidate else "directional_ellipse",
            period_source="ellipse_fit_directional_radius",
            q_radius=float(radius),
            q_unit=member_unit,
            fitted_member_branch_id=(
                member.get("branch_id") if member_assignment_resolved else None
            ),
            fitted_member_branch_id_source=(
                member.get("branch_id_source")
                if member_assignment_resolved
                else "equal_radius_direction_only" if len(finite_candidates) > 1 else "single_model_member"
            ),
            period_reason="2π/R(beta) from an origin-centred fitted ellipse along the accepted observed direction",
        )
        parameter_sources.append(
            _period_record(
                "period_from_ellipse_directional_radius",
                None if candidate else float(period),
                unit,
                "ellipse_fit_directional_radius",
                "candidate" if candidate else "available",
                str(population["period_reason"]),
                branch_id=branch_id,
                candidate_value=float(period) if candidate else None,
            )
        )
        parameter_sources.append(
            _period_record(
                "q_radius_from_ellipse_directional_radius",
                None if candidate else float(radius),
                member_unit,
                "ellipse_fit_directional_radius",
                "available" if not candidate else "candidate",
                "fitted q-space ellipse radius evaluated at the observed group angle",
                branch_id=branch_id,
                candidate_value=float(radius) if candidate else None,
            )
        )
        if candidate:
            flags.append("candidate_ellipse_directional_period")
            assumptions.append(
                "Finite candidate ellipse parameters drive a provisional direction-dependent schematic; formal derived values remain null."
            )
    if first_error is not None and not any(
        _finite_positive(item.get("period")) is not None for item in populations
    ):
        return first_error
    return first_error


def build_lamellar_scene(
    source: Any, settings: Mapping[str, Any] | LamellarSettings | None = None
) -> LamellarScene:
    resolved = LamellarSettings.from_mapping(settings)
    root = _source_mapping(source)
    identity = _source_identity(source, root)
    raw_peaks = _radial_peaks(root)
    q_unit = _q_unit_from_source(root, raw_peaks)
    draw_axis = _finite(_read(root, ("draw_axis_deg",), _MISSING))
    if draw_axis is None:
        observables = _read(root, ("observables", "measurements"), None)
        nested_draw_axis = _finite(
            _read(
                observables,
                ("draw_axis_deg",),
                _read(_read(root, ("analysis",), None), ("draw_axis_deg",), 90.0),
            )
        )
        draw_axis = 90.0 if nested_draw_axis is None else nested_draw_axis
    source_status = str(_read(root, ("status", "measurement_status"), "") or "").lower()
    metadata_root = _read(root, ("metadata",), {})
    if not source_status and isinstance(metadata_root, Mapping):
        source_status = str(
            metadata_root.get("status", metadata_root.get("measurement_status", ""))
            or ""
        ).lower()
    execution_failure = _execution_failure(root)
    stale = source_status == "stale" or bool(_read(root, ("stale",), False))
    assumptions = [
        "Box dimensions are explicit ratios and do not identify a unique three-dimensional structure.",
    ]
    if resolved.period_source != "manual":
        assumptions.extend(
            (
                "Measured q-angle is used as the schematic lamellar-normal direction after draw-axis display rotation.",
                "Opposite q directions represent one unoriented normal; original branch identifiers are retained.",
            )
        )
    if resolved.spacing_jitter_pct > 0.0:
        assumptions.append(
            "Each interlayer gap varies independently and uniformly within "
            f"±{resolved.spacing_jitter_pct:g}% of the period; this is a schematic "
            "assumption, not fitted uncertainty."
        )
    flags: list[str] = []
    if execution_failure is not None and resolved.period_source != "manual":
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message=f"Source execution status is {execution_failure}; measured peaks are not treated as current.",
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            q_unit=q_unit,
            flags=("source_execution_failed",),
        )
        return _empty_scene(resolved, metadata)
    if stale and resolved.period_source != "manual":
        metadata = _metadata_base(
            resolved,
            identity,
            status="stale",
            available=False,
            message="The source result is marked stale; no current lamellar scene is available.",
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            q_unit=q_unit,
            flags=("stale_source",),
        )
        return _empty_scene(resolved, metadata)
    populations: list[dict[str, Any]] = []
    parameter_sources: list[dict[str, Any]] = []
    direction_support: list[dict[str, Any]] | None = None
    direction_diagnostics: dict[str, Any] | None = None
    ellipse_period_error: str | None = None
    length_unit = "relative"
    ellipse_for_metadata = _ellipse_payload(root)
    parameter_sources.extend(_ellipse_parameter_records(ellipse_for_metadata))
    if resolved.period_source == "manual":
        period = float(resolved.manual_period)
        angle = float(resolved.manual_angle_deg)
        if resolved.manual_second_orientation:
            directions = ((0, angle), (1, float(resolved.manual_second_angle_deg)))
            if resolved.selected_branch >= 0:
                directions = (directions[int(resolved.selected_branch)],)
            assumptions.append(
                "A second manual lamellar-normal direction is included at the explicitly set angle; the two manual direction groups are allocated as evenly as possible and seed-shuffled for visualization, not fitted populations or measured fractions."
            )
        else:
            branch = 0 if resolved.selected_branch < 0 else int(resolved.selected_branch)
            directions = ((branch, angle),)
        populations = [
            {
                "branch_id": branch,
                "period": period,
                "length_unit": resolved.manual_unit,
                "angle_deg": direction_angle,
                "status": "manual",
                "orientation_assumption": "explicit manual direction",
            }
            for branch, direction_angle in directions
        ]
        parameter_sources.append(
            _period_record(
                "period",
                period,
                resolved.manual_unit,
                "manual",
                "manual",
                "explicit manual assumption",
            )
        )
        length_unit = resolved.manual_unit
        assumptions.append("Period and angle are explicit manual assumptions supplied by the caller.")
    elif not raw_peaks and resolved.period_source != "ellipse":
        parameter_sources.append(
            _period_record(
                "period",
                None,
                "relative",
                "lobe_radial_peaks",
                "unavailable",
                "no valid radial lobe peaks; annulus q is not accepted",
            )
        )
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message="No valid lobe_radial_peaks with a positive radial q_star were found; annulus/azimuthal q is not used.",
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            parameter_sources=parameter_sources,
            q_unit=q_unit,
            flags=("no_valid_lobe_radial_peaks", "spacing_unavailable_unknown_q_unit"),
        )
        return _empty_scene(resolved, metadata)
    elif resolved.period_source == "ellipse":
        support = observed_direction_support(root)
        if not support:
            support = _radial_peak_direction_support(raw_peaks)
        if not support:
            metadata = _metadata_base(
                resolved,
                identity,
                status="unavailable",
                available=False,
                message=(
                    "The ellipse supplies reciprocal-space shape and period candidates, "
                    "but no retained observed ridge points or radial lobe peaks are "
                    "available to set a lamellar-normal direction."
                ),
                draw_axis_deg=draw_axis,
                reference_period=resolved.reference_period or 1.0,
                assumptions=assumptions,
                parameter_sources=parameter_sources,
                q_unit=q_unit,
                flags=("no_observed_direction_support",),
            )
            metadata["direction_support"] = []
            return _empty_scene(resolved, metadata)
        direction_support = support
        populations, direction_diagnostics, direction_error = _direction_populations(support)
        explicit_populations = [
            item for item in populations if _exact_integer(item.get("branch_id")) is not None
            and int(item["branch_id"]) >= 0
        ]
        if direction_error is not None and not explicit_populations:
            metadata = _metadata_base(
                resolved,
                identity,
                status="unavailable",
                available=False,
                message=direction_error,
                draw_axis_deg=draw_axis,
                reference_period=resolved.reference_period or 1.0,
                assumptions=assumptions,
                parameter_sources=parameter_sources,
                q_unit=q_unit,
                flags=("ambiguous_observed_directions",),
            )
            metadata["direction_support"] = support
            metadata["direction_diagnostics"] = direction_diagnostics
            return _empty_scene(resolved, metadata)
        if direction_error is not None:
            flags.append("ambiguous_unassigned_direction_retained")
        parameter_sources.append(
            _period_record(
                "observed_direction_support",
                None,
                "degree",
                _direction_source_label(support),
                "available",
                "direction only; retained observed q-space support is not replaced by fitted ellipse angle",
            )
        )
        assumptions.append(
            "Ellipse-mode schematic normals use only retained observed ridge-point or radial lobe-peak directions. The fitted ellipse theta defines q-space geometry and is not treated as the lamellar-normal angle."
        )
        flags.append(
            "ellipse_direction_from_radial_lobe_peaks"
            if all("lobe_radial_peaks" in str(row.get("source", "")) for row in support)
            else "ellipse_direction_from_observed_ridge_points"
        )
    else:
        groups = _group_populations(raw_peaks)
        reference_q_native = float(
            next(
                (item["q"] for item in raw_peaks if item["q_nm_inv"] is None),
                raw_peaks[0]["q"],
            )
        )
        for branch, group in groups:
            angle, inconsistent_angle = _direction_for_group(group)
            first = group[0]
            q_nm_values = [
                item["q_nm_inv"] for item in group if item["q_nm_inv"] is not None
            ]
            q_nm = float(q_nm_values[0]) if q_nm_values else None
            inconsistent_q = False
            uncertain_q = False
            if len(q_nm_values) == len(group):
                inconsistent_q = any(
                    abs(float(item) - q_nm) > max(1e-12, 0.05 * q_nm)
                    for item in q_nm_values[1:]
                )
            else:
                native_units = {_normalise_q_unit(item["q_unit"]) for item in group}
                if len(native_units) == 1:
                    native_q = float(first["q"])
                    inconsistent_q = any(
                        abs(float(item["q"]) - native_q) > max(1e-12, 0.05 * native_q)
                        for item in group[1:]
                    )
                else:
                    uncertain_q = len(group) > 1
            if q_nm is not None:
                period = float(2.0 * np.pi / q_nm)
                unit = "nm"
                period_status = (
                    "candidate"
                    if any(
                        item["status"] in {"candidate", "provisional", "undetermined"}
                        for item in group
                    )
                    else "available"
                )
                period_source_name = "lobe_radial_peaks"
                period_reason = (
                    "2π/q from the first valid radial lobe peak in this branch"
                )
                length_unit = "nm"
            else:
                q_value = float(first["q"])
                base = (
                    float(resolved.reference_period)
                    if resolved.reference_period is not None
                    else 1.0
                )
                period = float(base * reference_q_native / q_value)
                unit = "relative"
                period_status = (
                    "candidate"
                    if any(
                        item["status"] in {"candidate", "provisional", "undetermined"}
                        for item in group
                    )
                    else "available"
                )
                period_source_name = "lobe_radial_peaks"
                period_reason = f"normalized to explicit reference q={reference_q_native:g} ({first['q_unit'] or 'unknown'})"
                length_unit = "relative"
                flags.append("spacing_unavailable_unknown_q_unit")
                assumptions.append(
                    "Unknown q units are displayed on an explicit relative scale normalized to the retained reference q."
                )
            if uncertain_q:
                flags.append("mixed_q_unit_support")
                period_reason += "; opposite q units are mixed and cannot be compared"
                period_status = "candidate"
            if inconsistent_angle or inconsistent_q:
                flags.append("inconsistent_opposite_support")
                period_reason += f"; opposite q values={[float(item['q']) for item in group]} are inconsistent and were not mirrored or fabricated"
                period_status = (
                    "candidate" if period_status == "available" else period_status
                )
            if any(
                item["status"] in {"candidate", "provisional", "undetermined"}
                for item in group
            ):
                flags.append("candidate_radial_support")
            populations.append(
                {
                    "branch_id": int(branch),
                    "period": period,
                    "length_unit": unit,
                    "angle_deg": angle,
                    "status": "candidate"
                    if period_status == "candidate"
                    else "schematic",
                    "support_indices": [int(item["index"]) for item in group],
                    "support_q_values": [float(item["q"]) for item in group],
                    "support_q_units": [str(item["q_unit"]) for item in group],
                }
            )
            parameter_sources.append(
                _period_record(
                    "period",
                    None if period_status == "candidate" else period,
                    unit,
                    period_source_name,
                    period_status,
                    period_reason,
                    candidate_value=period if period_status == "candidate" else None,
                )
            )
            q_record = q_nm if q_nm is not None else float(first["q"])
            parameter_sources.append(
                _period_record(
                    "q_star",
                    None if period_status == "candidate" else q_record,
                    "nm^-1" if q_nm is not None else str(first["q_unit"]),
                    "lobe_radial_peaks",
                    "available" if period_status == "available" else period_status,
                    "retained radial peak",
                    candidate_value=q_record if period_status == "candidate" else None,
                )
            )
    if resolved.period_source == "ellipse":
        assert direction_support is not None
        ellipse_period_error = _apply_ellipse_directional_periods(
            root,
            resolved,
            populations,
            direction_support,
            draw_axis,
            q_unit,
            parameter_sources,
            flags,
            assumptions,
        )
    selected = [
        population
        for population in populations
        if resolved.selected_branch < 0
        or int(population["branch_id"]) == int(resolved.selected_branch)
    ]
    if resolved.period_source == "ellipse":
        selected = [
            population
            for population in selected
            if _finite_positive(population.get("period")) is not None
        ]
    if not selected:
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message=(
                ellipse_period_error
                or f"Selected branch {resolved.selected_branch} has no supported lamellar population."
            ),
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            parameter_sources=parameter_sources,
            populations=populations,
            q_unit=q_unit,
            flags=(*flags, "selected_branch_unavailable"),
        )
        if direction_support is not None:
            metadata["direction_support"] = direction_support
            metadata["direction_diagnostics"] = direction_diagnostics
        return _empty_scene(resolved, metadata)
    selected_units = {
        str(population.get("length_unit", length_unit))
        for population in selected
        if _finite_positive(population.get("period")) is not None
    }
    if len(selected_units) > 1:
        mixed_units = sorted(selected_units)
        message = (
            "Selected populations use incompatible length units ("
            + " and ".join(mixed_units)
            + "); select one branch before generating a scene."
        )
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message=message,
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            parameter_sources=parameter_sources,
            populations=populations,
            q_unit=q_unit,
            flags=(*flags, "mixed_length_units"),
        )
        metadata["length_unit"] = "relative"
        metadata["mixed_length_units"] = mixed_units
        if direction_support is not None:
            metadata["direction_support"] = direction_support
            metadata["direction_diagnostics"] = direction_diagnostics
        return _empty_scene(resolved, metadata)
    if (
        len(selected) < 2
        and resolved.selected_branch < 0
        and resolved.period_source != "manual"
    ):
        flags.append("single_branch_supported")
    if selected_units:
        length_unit = next(iter(selected_units))
    periods = [
        float(item["period"])
        for item in selected
        if _finite_positive(item.get("period")) is not None
    ]
    if not periods:
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message=(
                ellipse_period_error
                or "No finite positive period remained after source adaptation."
            ),
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            parameter_sources=parameter_sources,
            populations=populations,
            q_unit=q_unit,
            flags=flags,
        )
        if direction_support is not None:
            metadata["direction_support"] = direction_support
            metadata["direction_diagnostics"] = direction_diagnostics
        return _empty_scene(resolved, metadata)
    if resolved.period_source == "ellipse" and any(
        population.get("status") == "unavailable" for population in populations
    ):
        flags.append("partial_ellipse_direction_support")
    layout_period = (
        float(resolved.reference_period)
        if resolved.reference_period is not None
        else float(max(periods))
    )
    # A retained project scale may predate this frame or be a manual
    # normalization smaller than its current period. Use it for lower-bound
    # layout only when it is large enough to contain the generated packets.
    layout_period = max(layout_period, max(periods))
    n_layers = int(resolved.layer_count)
    centres: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    orientations: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    stack_ids: list[int] = []
    branch_ids: list[int] = []
    colors: list[np.ndarray] = []
    # Separate whole packets, including their stack length and optional slip.
    packet_width = resolved.width_ratio + (n_layers - 1) * abs(
        resolved.lateral_shift_ratio
    )
    max_gap_ratio = 1.0 + float(resolved.spacing_jitter_pct) / 100.0
    packet_length = (n_layers - 1) * max_gap_ratio + resolved.thickness_ratio
    packet_diameter = float(
        np.linalg.norm((packet_width, resolved.depth_ratio, packet_length))
    )
    packet_gap_ratio = 1.0
    spacing = layout_period * (packet_diameter + packet_gap_ratio)
    slot_entries: list[tuple[int, int, int]] = []
    manual_assignment: np.ndarray | None = None
    if (
        resolved.period_source == "manual"
        and resolved.mode == "multi"
        and resolved.manual_second_orientation
        and resolved.selected_branch < 0
        and len(selected) >= 2
    ):
        manual_assignment = np.arange(int(resolved.stack_count), dtype=int) % len(selected)
        assignment_rng = np.random.default_rng(
            _stable_seed(resolved.seed, 97, int(resolved.stack_count))
        )
        assignment_rng.shuffle(manual_assignment)
    if resolved.mode == "single":
        if resolved.stack_count > 0:
            for population_index, population in enumerate(selected):
                branch = int(population["branch_id"])
                slot_entries.append(
                    (
                        branch if branch in {0, 1} else population_index,
                        population_index,
                        0,
                    )
                )
    elif resolved.stack_count > 0:
        has_two = len(selected) >= 2
        for slot in range(int(resolved.stack_count)):
            if manual_assignment is not None:
                population_index = int(manual_assignment[slot])
            elif has_two:
                population_index = slot % len(selected)
            else:
                # Preserve the established even-slot identity when the other
                # fitted branch is absent in a frame. Manual two-direction
                # previews still fill every slot through their two explicit
                # assumptions above.
                if resolved.period_source != "manual" and slot % 2:
                    continue
                population_index = 0
            slot_entries.append((slot, population_index, slot))
        assumptions.append(
            "Mesoscale field repeats only branch families supported by the current source on a jittered grid. Packet centers and equal allocation among supported branches are schematic spatial assumptions, not measured positions or population fractions."
        )
    total_slots = max(1, int(resolved.stack_count))
    grid_columns = max(1, int(np.ceil(np.sqrt(total_slots))))
    grid_rows = int(np.ceil(total_slots / grid_columns))
    for stack_id, population_index, slot in slot_entries:
        population = selected[population_index]
        branch = int(population["branch_id"])
        period = float(population["period"])
        angle = float(population["angle_deg"])
        branch_slot = branch if branch in {0, 1} else population_index
        rng = np.random.default_rng(_stable_seed(resolved.seed, 0, slot))
        deviation = (
            float(rng.uniform(-float(resolved.spread_deg), float(resolved.spread_deg)))
            if resolved.spread_deg
            else 0.0
        )
        if resolved.mode == "single":
            population_offset = (
                (float(population_index) - (len(selected) - 1) / 2.0) * spacing
                if len(selected) > 1
                else 0.0
            )
            base_center = np.asarray((population_offset, 0.0, 0.0), dtype=float)
        else:
            column = slot % grid_columns
            row = slot // grid_columns
            base_center = np.asarray(
                (
                    (column - (grid_columns - 1) / 2.0) * spacing,
                    (row - (grid_rows - 1) / 2.0) * spacing,
                    0.0,
                ),
                dtype=float,
            )
            jitter = (
                0.5
                * packet_gap_ratio
                * layout_period
                * float(resolved.position_jitter_pct)
                / 100.0
            )
            base_center[:2] += rng.uniform(-jitter, jitter, size=2)
        display_angle = np.radians(angle + deviation + (90.0 - float(draw_axis)))
        tilt = np.radians(float(resolved.out_of_plane_deg))
        normal = np.asarray(
            (
                np.cos(tilt) * np.cos(display_angle),
                np.cos(tilt) * np.sin(display_angle),
                np.sin(tilt),
            ),
            dtype=float,
        )
        width_basis = np.asarray(
            (-np.sin(display_angle), np.cos(display_angle), 0.0), dtype=float
        )
        depth_basis = np.cross(normal, width_basis)
        depth_norm = float(np.linalg.norm(depth_basis))
        if depth_norm <= 0.0 or not np.isfinite(depth_norm):
            depth_basis = np.asarray((0.0, 0.0, 1.0), dtype=float)
        else:
            depth_basis /= depth_norm
        orientation = np.column_stack((width_basis, depth_basis, normal))
        if resolved.spacing_jitter_pct > 0.0 and n_layers > 1:
            jitter_fraction = float(resolved.spacing_jitter_pct) / 100.0
            gap_ratios = 1.0 + rng.uniform(
                -jitter_fraction, jitter_fraction, size=n_layers - 1
            )
            layer_offsets = np.concatenate((np.zeros(1), np.cumsum(gap_ratios)))
            layer_offsets -= float(np.mean(layer_offsets))
        else:
            layer_offsets = np.arange(n_layers, dtype=float) - (n_layers - 1) / 2.0
        for layer_index in range(n_layers):
            centred_index = float(layer_index) - (n_layers - 1) / 2.0
            centre = (
                base_center
                + normal * (float(layer_offsets[layer_index]) * period)
                + width_basis
                * (centred_index * float(resolved.lateral_shift_ratio) * period)
            )
            size = np.asarray(
                (
                    float(resolved.width_ratio) * period,
                    float(resolved.depth_ratio) * period,
                    float(resolved.thickness_ratio) * period,
                ),
                dtype=float,
            )
            local_corners = _CORNER_SIGNS * (0.5 * size)
            corners = local_corners @ orientation.T + centre
            centres.append(centre)
            sizes.append(size)
            orientations.append(orientation)
            vertices.append(corners)
            stack_ids.append(stack_id)
            branch_ids.append(branch)
            colors.append(_BLUE.copy() if branch_slot % 2 == 0 else _ORANGE.copy())
    centre_array = (
        np.asarray(centres, dtype=float).reshape((-1, 3))
        if centres
        else np.empty((0, 3), dtype=float)
    )
    size_array = (
        np.asarray(sizes, dtype=float).reshape((-1, 3))
        if sizes
        else np.empty((0, 3), dtype=float)
    )
    orientation_array = (
        np.asarray(orientations, dtype=float).reshape((-1, 3, 3))
        if orientations
        else np.empty((0, 3, 3), dtype=float)
    )
    vertex_array = (
        np.asarray(vertices, dtype=float).reshape((-1, 8, 3))
        if vertices
        else np.empty((0, 8, 3), dtype=float)
    )
    color_array = (
        np.asarray(colors, dtype=float).reshape((-1, 4))
        if colors
        else np.empty((0, 4), dtype=float)
    )
    if len(vertex_array):
        bounds = np.vstack(
            (
                np.min(vertex_array.reshape(-1, 3), axis=0),
                np.max(vertex_array.reshape(-1, 3), axis=0),
            )
        )
    else:
        bounds = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=float)
    scene_status = (
        "candidate"
        if any(item.get("status") == "candidate" for item in selected)
        else ("manual" if resolved.period_source == "manual" else "schematic")
    )
    message = (
        "Manual-assumption lamellar schematic."
        if scene_status == "manual"
        else (
            "Candidate parameters drive this schematic; formal values remain provisional."
            if scene_status == "candidate"
            else (
                "Schematic generated from observed directions and fitted ellipse q-space radii."
                if resolved.period_source == "ellipse"
                else "Schematic generated from measured radial lobe peaks."
            )
        )
    )
    metadata = _metadata_base(
        resolved,
        identity,
        status=scene_status,
        available=True,
        message=message,
        draw_axis_deg=draw_axis,
        reference_period=layout_period,
        assumptions=assumptions,
        parameter_sources=parameter_sources,
        populations=populations,
        q_unit=q_unit,
        flags=flags,
    )
    metadata["length_unit"] = length_unit
    metadata["period_source"] = resolved.period_source
    metadata["selected_branch"] = int(resolved.selected_branch)
    if direction_support is not None:
        metadata["direction_support"] = direction_support
        metadata["direction_diagnostics"] = direction_diagnostics
    return LamellarScene(
        centers=centre_array,
        sizes=size_array,
        orientations=orientation_array,
        vertices=vertex_array,
        stack_ids=np.asarray(stack_ids, dtype=int),
        branch_ids=np.asarray(branch_ids, dtype=int),
        colors=color_array,
        bounds=bounds,
        length_unit=length_unit,
        metadata=metadata,
    )


def load_lamellar_sources(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    from .lamellar_sources import load_lamellar_sources as _load

    return _load(path)


__all__ = (
    "LamellarScene",
    "LamellarSettings",
    "build_lamellar_scene",
    "load_lamellar_sources",
)
