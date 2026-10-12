"""Candidate ellipse geometry shared by GUI overlays and figure exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .butterfly_quality import (
    classify_ellipse_publication,
    ellipse_candidate_is_limited,
)
from .ellipse import EllipseGeometry
from .serialization import json_safe

_CURVE_PHI = np.linspace(0.0, 2.0 * np.pi, 361, dtype=np.float64)

def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _flag(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        candidates = list(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        candidates = list(value)
    else:
        return []
    return [dict(row) for row in candidates if isinstance(row, Mapping)]


def _candidate(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = result.get("candidate_fit")
    return value if isinstance(value, Mapping) else None


def _ellipse_state(
    candidate: Mapping[str, Any] | None, result: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    if candidate is None:
        return "not_available", {"reason": "candidate_fit_missing"}
    status_text = str(
        candidate.get("status", candidate.get("solver_status", ""))
    ).lower()
    quality = candidate.get("quality", result.get("quality", {}))
    if not isinstance(quality, Mapping):
        quality = {}
    flags: list[str] = []
    for source in (candidate, quality):
        raw_flags = source.get("flags", ())
        if isinstance(raw_flags, str):
            flags.extend(part.strip() for part in raw_flags.split(",") if part.strip())
        else:
            flags.extend(str(item) for item in (raw_flags or ()) if item)
    bound_flags = candidate.get("bound_flags", candidate.get("bound_status", {}))
    if isinstance(bound_flags, Mapping):
        active_bounds = [str(name) for name, active in bound_flags.items() if active is True]
        if active_bounds:
            flags.append("parameter_at_explicit_bound")
        if bound_flags.get("axis_ratio"):
            flags.append("axis_ratio_at_bound")
    flags = list(dict.fromkeys(flags))
    ratio = _finite(candidate.get("axis_ratio", candidate.get("axes_ratio")))
    if ratio is None:
        a = _finite(candidate.get("a", candidate.get("semi_major")))
        b = _finite(candidate.get("b", candidate.get("semi_minor")))
        if a and b is not None and a > 0.0:
            ratio = b / a
    publication_kind = classify_ellipse_publication(
        quality_status=quality.get("status", quality.get("engineering_status")),
        axis_ratio=ratio,
        flags=flags,
    )
    if _flag(candidate.get("success")) is False or status_text in {
        "fail", "failed", "error", "invalid", "unconverged"
    }:
        return "candidate_unconverged", {
            "reason": "solver_did_not_converge",
            "candidate_label": "unsuccessful geometry candidate; diagnostic only",
            "solver_status": status_text or "unknown",
            "message": str(candidate.get("message", "")),
            "quality_status": quality.get("status"),
            "flags": json_safe(flags),
        }
    if status_text in {"ring", "ring_only", "ring-only"} or publication_kind == "ring":
        return "ring_only_candidate", {
            "reason": "candidate_geometry_is_near_circular_or_ring_only",
            "candidate_label": "near-circular ring candidate; ellipse axis is not identifiable",
            "quality_status": quality.get("status"),
            "flags": json_safe(flags),
        }
    if _flag(candidate.get("success")) is True:
        limited = ellipse_candidate_is_limited(flags)
        return "candidate", {
            "reason": (
                "ellipse_candidate_has_bound_or_support_limitations"
                if limited
                else "diagnostic_candidate_only"
            ),
            "candidate_label": (
                "constrained ellipse candidate; not scientifically accepted"
                if limited
                else "candidate ellipse; not scientifically accepted"
            ),
            "bound_or_support_limited": limited,
            "quality_status": quality.get("status"),
            "scientific_status": (
                "NOT_ACCEPTED" if limited else quality.get("scientific_status", "NOT_ACCEPTED")
            ),
            "flags": json_safe(flags),
        }
    return "candidate_unconverged", {
        "reason": "finite_geometry_retained_for_diagnostic_review",
        "candidate_label": "candidate solver result; not quantitatively accepted",
        "solver_status": status_text or "unknown",
        "message": str(candidate.get("message", "")),
        "quality_status": quality.get("status"),
        "flags": json_safe(flags),
    }


def _member_geometry(
    member: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    index: int,
    reference_axis_deg: float | None,
) -> tuple[EllipseGeometry | None, int | str | None, int | str | None]:
    center = member.get("center", member.get("centre"))
    if (
        isinstance(center, Sequence)
        and not isinstance(center, (str, bytes))
        and len(center) == 2
    ):
        cx, cy = _finite(center[0]), _finite(center[1])
    else:
        parameters = candidate.get("parameters", candidate.get("parameter_values", {}))
        if not isinstance(parameters, Mapping):
            parameters = {}
        candidate_center = candidate.get("center", candidate.get("centre"))
        if (
            isinstance(candidate_center, Sequence)
            and not isinstance(candidate_center, (str, bytes))
            and len(candidate_center) == 2
        ):
            default_cx, default_cy = candidate_center
        else:
            default_cx = candidate.get(
                "center_qx", parameters.get("center_qx", parameters.get("cx"))
            )
            default_cy = candidate.get(
                "center_qy", parameters.get("center_qy", parameters.get("cy"))
            )
        cx = _finite(member.get("center_qx", member.get("cx", default_cx)))
        cy = _finite(member.get("center_qy", member.get("cy", default_cy)))
    a = _finite(member.get("a", member.get("semi_major", candidate.get("a"))))
    b = _finite(member.get("b", member.get("semi_minor", candidate.get("b"))))
    ratio = _finite(
        member.get(
            "axis_ratio",
            member.get(
                "axes_ratio", candidate.get("axis_ratio", candidate.get("axes_ratio"))
            ),
        )
    )
    if ratio is None and a is not None and b is not None and a > 0.0:
        ratio = b / a
    angle = _finite(member.get("angle_deg"))
    if angle is None:
        angle = _finite(member.get("theta_deg"))
    if angle is None:
        angle_rad = _finite(member.get("theta"))
        if angle_rad is not None:
            angle = math.degrees(angle_rad)
    if angle is None:
        theta = _finite(candidate.get("theta_deg"))
        if theta is None:
            theta_rad = _finite(candidate.get("theta"))
            theta = math.degrees(theta_rad) if theta_rad is not None else None
        if theta is not None and reference_axis_deg is not None:
            sign = 1.0 if index == 0 else -1.0
            angle = reference_axis_deg + sign * theta
    if cx is None or cy is None or a is None or ratio is None or angle is None:
        return None, member.get("branch_id"), member.get("fit_branch_id")
    try:
        geometry = EllipseGeometry(
            cx=float(cx),
            cy=float(cy),
            a=float(a),
            axis_ratio=float(ratio),
            theta=math.radians(float(angle)),
        )
    except (TypeError, ValueError):
        return None, member.get("branch_id"), member.get("fit_branch_id")
    branch_id = member.get("branch_id")
    fit_branch_id = member.get("fit_branch_id")
    return geometry, branch_id, fit_branch_id


def _candidate_curves(
    result: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    candidate = _candidate(result)
    state, state_metadata = _ellipse_state(candidate, result)
    if (
        state not in {"candidate", "candidate_unconverged", "ring_only_candidate"}
        or candidate is None
    ):
        return state, [], state_metadata
    diagnostics = result.get("diagnostics")
    reference = None
    if isinstance(diagnostics, Mapping):
        reference = _finite(diagnostics.get("reference_axis_deg"))
    candidate_reference = _finite(candidate.get("reference_axis_deg"))
    if candidate_reference is not None:
        reference = candidate_reference
    members = _mapping_rows(candidate.get("ellipses"))
    if not members:
        members = [{}]
        if _finite(candidate.get("a", candidate.get("semi_major"))) is not None:
            members.append({})
    curves: list[dict[str, Any]] = []
    for index, member in enumerate(members):
        geometry, branch_id, fit_branch_id = _member_geometry(
            member,
            candidate,
            index=index,
            reference_axis_deg=reference,
        )
        if geometry is None:
            continue
        points = np.asarray(geometry.point(_CURVE_PHI), dtype=np.float64)
        if points.shape != (_CURVE_PHI.size, 2) or not np.all(np.isfinite(points)):
            continue
        branch_label = (
            f"branch {branch_id}"
            if isinstance(branch_id, (int, np.integer))
            and not isinstance(branch_id, (bool, np.bool_))
            else f"fit branch {fit_branch_id}"
            if isinstance(fit_branch_id, (int, np.integer))
            and not isinstance(fit_branch_id, (bool, np.bool_))
            else f"ellipse {index + 1}"
        )
        curves.append(
            {
                "curve_index": index,
                "branch_id": branch_id,
                "fit_branch_id": fit_branch_id,
                "label": branch_label,
                "points": points,
                "phi_rad": _CURVE_PHI,
                "curve_role": "candidate_full_curve_not_observed_support",
                "scientific_status": "NOT_ACCEPTED",
            }
        )
    if not curves:
        state = "fit_unavailable"
        state_metadata = {
            **state_metadata,
            "reason": "candidate_geometry_unavailable_or_nonfinite",
        }
    else:
        state_metadata = {
            **state_metadata,
            "curves_role": "candidate diagnostic curves; full extent is not observed support",
        }
    return state, curves, state_metadata
