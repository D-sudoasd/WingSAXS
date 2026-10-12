"""Direct q-resolved petal measurements, independent of ellipse convergence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


QUADRANTS = ("QI", "QII", "QIII", "QIV")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _delta(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def summarize_annular_measurements(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize actual selected angular peaks, retaining gaps and asymmetry.

    Each row uses the prescribed annulus coordinate. No radial reflection
    position or period is derived from that coordinate. Angle variation along
    a trajectory and the half-height span of I(chi) are different observables;
    neither is a calibrated orientation distribution or confidence interval.
    """
    annular = trace.get("annular_peaks") or {}
    selected: dict[tuple[int, str], Mapping[str, Any]] = {}
    for point in trace.get("points", ()):
        if not point.get("accepted") or point.get("quadrant") not in QUADRANTS:
            continue
        if _finite(point.get("chi_deg")) is None or _finite(point.get("q_annulus")) is None:
            continue
        selected[(int(point["annulus_index"]), point["quadrant"])] = point

    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for annulus in annular.get("annuli", ()):
        index = int(annulus["annulus_index"])
        for quadrant in QUADRANTS:
            point = selected.get((index, quadrant))
            row = {
                "annulus_index": index,
                "q_annulus": _finite(annulus.get("q_center")),
                "q_min": _finite(annulus.get("q_min")),
                "q_max": _finite(annulus.get("q_max")),
                "quadrant": quadrant,
                "status": "observed" if point is not None else "missing",
                "point_id": point.get("point_id") if point is not None else None,
                "trajectory_id": point.get("trajectory_id") if point is not None else None,
                "arc_id": point.get("arc_id") if point is not None else None,
                "chi_deg": _finite(point.get("chi_deg")) if point is not None else None,
                "peak_intensity_raw_mean": _finite(point.get("intensity")) if point is not None else None,
                "prominence": _finite(point.get("prominence")) if point is not None else None,
                "azimuthal_fwhm_deg": None,
                "width_status": "unavailable",
                "coverage": _finite(point.get("coverage")) if point is not None else None,
                "snr": _finite(point.get("snr")) if point is not None else None,
                "trajectory_confidence": point.get("trajectory_confidence") if point is not None else None,
                "flags": [],
            }
            if point is not None:
                width = _finite(point.get("azimuthal_fwhm_deg"))
                if width is not None and width > 0.0:
                    row["azimuthal_fwhm_deg"] = width
                    row["width_status"] = "smoothed_profile_half_height_span"
                    for other_quadrant in QUADRANTS:
                        other = selected.get((index, other_quadrant))
                        if other is None or other_quadrant == quadrant:
                            continue
                        other_width = _finite(other.get("azimuthal_fwhm_deg"))
                        separation = abs(_delta(float(other["chi_deg"]) - float(point["chi_deg"])))
                        if other_width is not None and other_width > 0.0 and (width + other_width) / 2.0 >= separation:
                            row["flags"].append("angular_half_height_spans_overlap")
                            break
                else:
                    row["flags"].append("angular_half_height_crossings_unavailable")
                if point.get("trajectory_confidence") == "low":
                    row["flags"].append("low_trajectory_confidence")
            rows.append(row)

        for left, right in (("QI", "QIII"), ("QII", "QIV")):
            first, second = selected.get((index, left)), selected.get((index, right))
            supported = first is not None and second is not None
            p1 = _finite(first.get("prominence")) if first is not None else None
            p2 = _finite(second.get("prominence")) if second is not None else None
            asymmetry = None
            if supported and p1 is not None and p2 is not None and p1 >= 0.0 and p2 >= 0.0 and p1 + p2 > 0.0:
                asymmetry = (p1 - p2) / (p1 + p2)
            pairs.append({
                "annulus_index": index,
                "q_annulus": _finite(annulus.get("q_center")),
                "quadrants": [left, right],
                "status": "observed_pair" if supported else "missing_counterpart",
                "opposition_error_deg": _delta(float(second["chi_deg"]) - float(first["chi_deg"]) - 180.0) if supported else None,
                "prominence_asymmetry": asymmetry,
                "q_difference_independent": False,
            })

    trajectories = []
    for quadrant in QUADRANTS:
        points = [row for row in rows if row["quadrant"] == quadrant and row["status"] == "observed"]
        angles = np.asarray([row["chi_deg"] for row in points], dtype=float)
        unwrapped = np.degrees(np.unwrap(np.radians(angles)))
        trajectories.append({
            "quadrant": quadrant,
            "status": "observed" if points else "missing",
            "n_observed_annuli": len(points),
            "n_missing_annuli": len(annular.get("annuli", ())) - len(points),
            "q_min_observed": min((row["q_annulus"] for row in points), default=None),
            "q_max_observed": max((row["q_annulus"] for row in points), default=None),
            "chi_median_deg": float(np.median(unwrapped) % 360.0) if points else None,
            "chi_span_deg": float(np.ptp(unwrapped)) if points else None,
            "n_unavailable_widths": sum(row["width_status"] == "unavailable" for row in points),
            "n_overlapping_widths": sum("angular_half_height_spans_overlap" in row["flags"] for row in points),
            "n_low_confidence_points": sum(row["trajectory_confidence"] == "low" for row in points),
            "n_trajectory_segments": len({row["trajectory_id"] for row in points if row["trajectory_id"] is not None}),
            "n_supported_arc_points": sum(row["arc_id"] is not None and row["arc_id"] >= 0 for row in points),
            "trajectory_evidence_status": (
                "supported_segments" if any(row["arc_id"] is not None and row["arc_id"] >= 0 and row["trajectory_confidence"] != "low" for row in points)
                else "local_candidates_only" if points else "missing"
            ),
        })
    return {
        "method_version": "annular-direct-measurements-v1",
        "q_unit": trace.get("q_unit", annular.get("q_unit", "unknown")),
        "reference_axis_deg": trace.get("branch_pairing", {}).get("reference_axis_deg"),
        "measurement_definition": "selected observed I(chi) maxima and smoothed-profile half-height spans at prescribed q annuli",
        "width_basis": "finite supported half-height brackets in the mask-aware smoothed angular profile",
        "intensity_semantics": "source angular-bin mean; input correction state and units apply",
        "ellipse_required": False,
        "q_annulus_is_radial_peak": False,
        "trajectory_angle_span_is_orientation_width": False,
        "uncertainty": "sampling and selection descriptors; no calibrated confidence interval",
        "rows": rows,
        "opposite_pairs": pairs,
        "trajectories": trajectories,
    }
