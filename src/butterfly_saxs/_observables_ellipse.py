"""Constrained symmetric double-ellipse fitting adapter and quadrant symmetry diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._observables_types import (
    APPARENT_FLAGS,
    _SPACING_CENTER_REL_TOL,
    DoubleEllipseFit,
    EllipseGeometry,
    LobeMetrics,
    RidgeTrack,
    _array_field,
    _flags_with_q_unit,
    _get_field,
    _q_to_nm_inverse_scale,
    ellipse_radius,
)
from .cancellation import AnalysisCancelled, raise_if_cancelled
from .ellipse import (
    EllipseGeometry as _CanonicalEllipseGeometry,
    ellipse_sampson_residuals as _canonical_sampson_residuals,
    fit_symmetric_ellipses as _fit_canonical_symmetric_ellipses,
)


def _ridge_xy(points: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(points, RidgeTrack):
        points = points.points
    # A numeric ``(n, 2)`` array follows the conventional optimizer contract:
    # its columns are Cartesian qx/qy coordinates.  The generic sequence
    # fallback below intentionally retains the legacy ``(angle, q)`` tuple
    # interpretation, so it must not be used for NumPy point clouds.
    if isinstance(points, np.ndarray):
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("ridge point array must have shape (n, 2) for qx/qy")
        xarr, yarr = array[:, 0], array[:, 1]
        return xarr, yarr, np.isfinite(xarr) & np.isfinite(yarr)
    if isinstance(points, Mapping):
        x = _array_field(points, ("x", "qx", "q_x"))
        y = _array_field(points, ("y", "qy", "q_y"))
        q = _array_field(points, ("q", "q_star", "radius"))
        angle = _array_field(points, ("angle", "azimuth", "phi"))
        if x is None or y is None:
            if q is None or angle is None:
                raise ValueError("ridge mapping needs x/y or q/angle")
            q, angle = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(angle, dtype=float))
            x, y = q * np.cos(angle), q * np.sin(angle)
        else:
            x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return x.ravel(), y.ravel(), np.isfinite(x.ravel()) & np.isfinite(y.ravel())
    seq = list(points)
    xs: list[float] = []
    ys: list[float] = []
    for point in seq:
        x = _get_field(point, ("x", "qx", "q_x"))
        y = _get_field(point, ("y", "qy", "q_y"))
        if _get_field(point, ("valid",), True) is False:
            # Shared RidgePoint intentionally rejects NaN coordinates.  Keep
            # invalid sectors out of a downstream ellipse fit explicitly.
            x, y = np.nan, np.nan
        if x is None or y is None:
            q = _get_field(point, ("q", "q_star", "radius"))
            angle = _get_field(point, ("angle", "azimuth", "phi"))
            if q is None or angle is None:
                try:
                    angle, q = point[0], point[1]
                except Exception as exc:
                    raise ValueError("each ridge point needs x/y or angle/q") from exc
            x, y = float(q) * np.cos(float(angle)), float(q) * np.sin(float(angle))
        xs.append(float(x))
        ys.append(float(y))
    xarr, yarr = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    return xarr, yarr, np.isfinite(xarr) & np.isfinite(yarr)


def _ridge_source(points: Any) -> Any:
    """Return the concrete point collection behind a ridge adapter."""

    if isinstance(points, RidgeTrack):
        return points.points
    if isinstance(points, Mapping):
        nested = _get_field(points, ("points", "ridges", "ridge_points"))
        if nested is not None:
            return nested
    return points


def _ridge_components(points: Any) -> np.ndarray | None:
    """Read aligned branch labels while retaining valid partial labels.

    ``NaN`` marks an invalid or unassigned point.  Returning the aligned
    array lets strict butterfly callers retain every valid label instead of
    silently discarding the whole topology because one masked sector is
    unlabeled.
    """

    source = _ridge_source(points)
    if isinstance(source, Mapping):
        value = _get_field(source, ("component", "components", "labels", "branch", "branch_id"))
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 0:
            return None
        try:
            numeric = array.astype(float, copy=False).ravel()
        except (TypeError, ValueError, OverflowError):
            return None
        valid = np.isfinite(numeric) & np.isin(numeric, (0, 1))
        if not np.any(valid):
            return None
        result = np.full(numeric.shape, np.nan, dtype=float)
        result[valid] = numeric[valid]
        return result
    if isinstance(source, np.ndarray) or isinstance(source, (str, bytes)):
        return None
    try:
        sequence = list(source)
    except TypeError:
        return None
    values: list[Any] = []
    found = False
    for point in sequence:
        component = _get_field(point, ("component", "branch", "label", "branch_id"))
        if component is None:
            metadata = _get_field(point, ("metadata",), {})
            component = _get_field(metadata, ("component", "branch", "label", "branch_id"))
        if component is None:
            values.append(np.nan)
        else:
            values.append(component)
            found = True
    if not found:
        return None
    array = np.asarray(values)
    try:
        numeric = array.astype(float, copy=False)
    except (TypeError, ValueError):
        return None
    valid = np.isfinite(numeric) & np.isin(numeric, (0, 1))
    if not np.any(valid):
        return None
    numeric = numeric.astype(float, copy=False)
    numeric[~valid] = np.nan
    return numeric


def _ridge_weights(points: Any) -> np.ndarray | None:
    """Read optional positive point weights for the canonical solver."""

    source = _ridge_source(points)
    if isinstance(source, Mapping):
        value = _get_field(source, ("weight", "weights"))
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        return array.ravel() if array.ndim else None
    if isinstance(source, np.ndarray) or isinstance(source, (str, bytes)):
        return None
    try:
        sequence = list(source)
    except TypeError:
        return None
    values: list[float] = []
    found = False
    for point in sequence:
        value = _get_field(point, ("weight",))
        if value is None:
            values.append(1.0)
        else:
            values.append(float(value))
            found = True
    return np.asarray(values, dtype=float) if found else None


def _infer_ridge_q_unit(points: Any) -> str:
    """Read q-unit provenance from a ridge adapter without guessing units."""

    value = _get_field(points, ("q_unit",), None)
    if value is not None:
        return str(value or "unknown")
    source = _ridge_source(points)
    if isinstance(source, Mapping) or isinstance(source, (str, bytes, np.ndarray)):
        return str(_get_field(source, ("q_unit",), "unknown") or "unknown")
    try:
        sequence = list(source)
    except TypeError:
        return "unknown"
    for point in sequence:
        value = _get_field(point, ("q_unit",), None)
        if value is not None:
            return str(value or "unknown")
        metadata = _get_field(point, ("metadata",), {})
        value = _get_field(metadata, ("q_unit", "unit"), None)
        if value is not None:
            return str(value or "unknown")
    return "unknown"


def _initial_to_canonical(initial: Any) -> Any:
    """Translate the legacy ``a,b,theta`` initialiser to canonical names."""

    if initial is None:
        return None
    # ParameterSet carries editable/fixed/tied specifications and must not be
    # flattened through ``dict()`` (its items are resolved numeric values).
    if hasattr(initial, "spec_items") and hasattr(initial, "free_names"):
        return initial
    if not isinstance(initial, Mapping):
        values = list(initial)
        if len(values) < 3:
            raise ValueError("ellipse initial must contain a, b, theta")
        a_value, b_value = float(values[0]), float(values[1])
        if a_value <= 0:
            raise ValueError("ellipse initial a must be positive")
        return {"a": a_value, "axis_ratio": b_value / a_value, "theta": float(values[2])}

    data = dict(initial)
    if "a" not in data:
        for alias in ("semi_major", "major", "a0"):
            if alias in data:
                data["a"] = data[alias]
                break
    if "b" not in data:
        for alias in ("semi_minor", "minor", "b0"):
            if alias in data:
                data["b"] = data[alias]
                break
    if "theta" not in data and "tilt" in data:
        data["theta"] = data["tilt"]
    if "center" in data:
        center = data.pop("center")
        if len(center) != 2:
            raise ValueError("ellipse center must contain (cx, cy)")
        data.setdefault("cx", center[0])
        data.setdefault("cy", center[1])
    if "centre" in data:
        center = data.pop("centre")
        if len(center) != 2:
            raise ValueError("ellipse centre must contain (cx, cy)")
        data.setdefault("cx", center[0])
        data.setdefault("cy", center[1])

    # ``ellipse.py`` accepts theta_deg directly.  Keep it in that form so a
    # ParameterSpec with min/max/vary remains editable in degree units.
    if "theta_deg" not in data and "theta" not in data and "tilt_deg" in data:
        data["theta_deg"] = data["tilt_deg"]
    if "axis_ratio" not in data and "ratio" in data:
        data["axis_ratio"] = data["ratio"]
    if "axis_ratio" not in data and "b" in data:
        if "a" not in data:
            raise ValueError("ellipse initial with b requires a")
        def _number(value: Any) -> float:
            if isinstance(value, Mapping):
                value = value.get("value", value.get("estimate"))
            elif hasattr(value, "value"):
                value = value.value
            return float(value)
        a_value, b_value = _number(data["a"]), _number(data["b"])
        if a_value <= 0:
            raise ValueError("ellipse initial a must be positive")
        data["axis_ratio"] = b_value / a_value
        # b is derived in the canonical parameterisation.  Removing a plain
        # legacy b also prevents it from overriding the tied expression.
        data.pop("b", None)
    return data


def _ellipse_center_hint(parameters: Any) -> tuple[tuple[float, float], bool]:
    """Return a fixed-center hint for pre-fit quadrant labeling."""

    if parameters is None:
        return (0.0, 0.0), False
    values: Mapping[str, Any] | None = None
    fixed = False
    if hasattr(parameters, "resolve"):
        try:
            resolved = parameters.resolve()
            if isinstance(resolved, Mapping):
                values = resolved
            specs = dict(parameters.spec_items()) if hasattr(parameters, "spec_items") else {}
            fixed = bool(
                specs
                and all(
                    name in specs and (not bool(getattr(specs[name], "vary", True)))
                    for name in ("cx", "cy")
                )
            )
        except (TypeError, ValueError):
            values = None
    elif isinstance(parameters, Mapping):
        values = parameters
        center = parameters.get("center", parameters.get("centre"))
        if isinstance(center, Sequence) and len(center) == 2:
            values = {**parameters, "cx": center[0], "cy": center[1]}
        fixed = bool(parameters.get("fixed_center", False))
        if not fixed:
            cx_spec = parameters.get("cx")
            cy_spec = parameters.get("cy")
            fixed = bool(
                isinstance(cx_spec, Mapping)
                and isinstance(cy_spec, Mapping)
                and cx_spec.get("vary") is False
                and cy_spec.get("vary") is False
            )
    if values is None:
        return (0.0, 0.0), fixed
    try:
        def numeric(value: Any) -> float:
            if isinstance(value, Mapping):
                value = value.get("value", value.get("estimate", 0.0))
            return float(value)

        return (numeric(values.get("cx", 0.0)), numeric(values.get("cy", 0.0))), fixed
    except (TypeError, ValueError):
        return (0.0, 0.0), False


def _empty_double_ellipse(
    n_points: int,
    message: str,
    *flags: str,
    q_unit: str = "unknown",
) -> DoubleEllipseFit:
    nan = float("nan")
    geometry = EllipseGeometry(nan, nan, nan, (nan, nan))
    unit = str(q_unit or "unknown")
    return DoubleEllipseFit(
        a=nan,
        b=nan,
        theta=nan,
        ellipses=(geometry, geometry),
        ellipticity=nan,
        axes_ratio=nan,
        rmse=nan,
        rss=nan,
        n_points=int(n_points),
        branch_counts=(0, 0),
        success=False,
        message=message,
        flags=_flags_with_q_unit(APPARENT_FLAGS + tuple(flags), unit),
        q_unit=unit,
    )




def _quadrant_labels(
    points: np.ndarray,
    *,
    center: tuple[float, float],
    reference_axis_deg: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Classify observed points in the local q-centred reference frame."""

    names = ("QI", "QII", "QIII", "QIV")
    relative = points - np.asarray(center, dtype=float)[None, :]
    angle = np.mod(
        np.arctan2(relative[:, 1], relative[:, 0]) - np.deg2rad(reference_axis_deg),
        2.0 * np.pi,
    )
    indices = np.floor(angle / (0.5 * np.pi)).astype(int)
    indices = np.clip(indices, 0, 3)
    return indices, names


def _symmetry_diagnostics(
    points: np.ndarray,
    labels: np.ndarray | None,
    *,
    center: tuple[float, float],
    reference_axis_deg: float,
    unassigned_count: int = 0,
    center_verified: bool = False,
    azimuthal: bool = False,
) -> dict[str, Any]:
    """Describe observed quadrant pairing without manufacturing counterparts."""

    if points.size == 0:
        return {
            "policy": "strict_butterfly_quadrant_pairing",
            "symmetry_status": "FAIL",
            "flags": ["no_observed_symmetry_points"],
            "quadrant_counts": {name: 0 for name in ("QI", "QII", "QIII", "QIV")},
            "branch_quadrant_counts": {"0": {}, "1": {}},
            "paired_support": {},
            "central_symmetry": {},
            "branch_leaks": {"direct": 0, "swapped": 0, "selected": 0},
            "unassigned_count": int(unassigned_count),
        }
    quadrant, names = _quadrant_labels(
        points,
        center=center,
        reference_axis_deg=reference_axis_deg,
    )
    quadrant_counts = {
        name: int(np.count_nonzero(quadrant == index))
        for index, name in enumerate(names)
    }
    diagnostics: dict[str, Any] = {
        "policy": "strict_butterfly_quadrant_pairing",
        "reference_axis_deg": float(reference_axis_deg),
        "center_qx": float(center[0]),
        "center_qy": float(center[1]),
        "center_verified": bool(center_verified),
        "quadrant_counts": quadrant_counts,
        "branch_quadrant_counts": {"0": {name: 0 for name in names}, "1": {name: 0 for name in names}},
        "paired_support": {},
        "central_symmetry": {},
        "branch_leaks": {"direct": 0, "swapped": 0, "selected": 0, "global_swap": False},
        "unassigned_count": int(unassigned_count),
        "q_difference_independent": not bool(azimuthal),
        "flags": [],
    }
    if labels is None:
        diagnostics["symmetry_status"] = "WARN"
        diagnostics["flags"].append("symmetry_unverified_unlabeled")
        return diagnostics
    labels = np.asarray(labels, dtype=float).ravel()
    if labels.size != points.shape[0]:
        diagnostics["symmetry_status"] = "FAIL"
        diagnostics["flags"].append("symmetry_label_alignment_error")
        return diagnostics
    relative = points - np.asarray(center, dtype=float)[None, :]
    local_angle = np.mod(
        np.arctan2(relative[:, 1], relative[:, 0]) - np.deg2rad(reference_axis_deg),
        2.0 * np.pi,
    )
    sector_phase = np.mod(local_angle, 0.5 * np.pi)
    boundary_ambiguous = np.minimum(sector_phase, 0.5 * np.pi - sector_phase) < np.deg2rad(2.0)
    valid_labels = np.isfinite(labels) & np.isin(labels, (0, 1)) & ~boundary_ambiguous
    diagnostics["unassigned_count"] = int(unassigned_count + np.count_nonzero(boundary_ambiguous))
    if np.any(boundary_ambiguous):
        diagnostics["flags"].append("quadrant_boundary_ambiguous")
    for branch_id in (0, 1):
        for index, name in enumerate(names):
            diagnostics["branch_quadrant_counts"][str(branch_id)][name] = int(
                np.count_nonzero(valid_labels & (labels == branch_id) & (quadrant == index))
            )
    expected = np.asarray((0, 1, 0, 1), dtype=int)
    direct_leaks = int(np.count_nonzero(valid_labels & (labels != expected[quadrant])))
    swapped_leaks = int(np.count_nonzero(valid_labels & (1 - labels != expected[quadrant])))
    selected_swap = bool(swapped_leaks < direct_leaks)
    selected_leaks = min(direct_leaks, swapped_leaks)
    diagnostics["branch_leaks"] = {
        "direct": direct_leaks,
        "swapped": swapped_leaks,
        "selected": selected_leaks,
        "global_swap": selected_swap,
    }
    if selected_swap:
        diagnostics["flags"].append("global_branch_swap_equivalent")
    working_labels = 1.0 - labels if selected_swap else labels
    for branch_id in (0, 1):
        for index, name in enumerate(names):
            diagnostics["branch_quadrant_counts"][str(branch_id)][name] = int(
                np.count_nonzero(valid_labels & (working_labels == branch_id) & (quadrant == index))
            )
    effective_unassigned = int(diagnostics["unassigned_count"])
    if effective_unassigned or np.count_nonzero(~valid_labels):
        diagnostics["flags"].append("unassigned_branch_points")

    radius = np.hypot(relative[:, 0], relative[:, 1])
    angle = np.mod(
        np.arctan2(relative[:, 1], relative[:, 0]) - np.deg2rad(reference_axis_deg),
        2.0 * np.pi,
    )
    valid_radii = radius[valid_labels]
    finite_valid_radii = valid_radii[np.isfinite(valid_radii)]
    radius_scale = (
        max(float(np.median(finite_valid_radii)), np.finfo(float).eps)
        if finite_valid_radii.size
        else np.finfo(float).eps
    )
    angle_tolerance = np.deg2rad(20.0)
    q_tolerance = max(0.05 * radius_scale, 1.0e-8)
    pair_specs = (("QI", "QIII", 0, 2), ("QII", "QIV", 1, 3))
    all_distance: list[float] = []
    all_angle_error: list[float] = []
    all_q_difference: list[float] = []
    missing_total = 0
    for branch_id, (first_name, second_name, first_quadrant, second_quadrant) in enumerate(pair_specs):
        # Greedy one-to-one matching keeps memory O(n); do not materialize a
        # full point-by-point distance matrix for large azimuthal tracks.
        first = np.flatnonzero(valid_labels & (working_labels == branch_id) & (quadrant == first_quadrant))
        second = np.flatnonzero(valid_labels & (working_labels == branch_id) & (quadrant == second_quadrant))
        unused = set(int(index) for index in second)
        pairs: list[tuple[int, int]] = []
        for first_index in first:
            if not unused:
                break
            best_match: tuple[float, float, float, int] | None = None
            r1 = float(radius[first_index])
            a1 = float(angle[first_index])
            for second_index in unused:
                angular_error = abs(
                    (float(angle[second_index]) - a1 - np.pi + np.pi)
                    % (2.0 * np.pi)
                    - np.pi
                )
                if angular_error > angle_tolerance:
                    continue
                r2 = float(radius[second_index])
                q_difference = abs(r1 - r2)
                if q_difference > max(q_tolerance, 0.25 * max(r1, r2)):
                    continue
                vector_deviation = float(np.linalg.norm(relative[first_index] + relative[second_index]))
                candidate_key = (vector_deviation, angular_error, q_difference, int(second_index))
                if best_match is None or candidate_key < best_match:
                    best_match = candidate_key
            if best_match is not None:
                vector_deviation, angular_error, q_difference, second_index = best_match
                pairs.append((int(first_index), second_index))
                unused.remove(second_index)
                all_distance.append(vector_deviation)
                all_angle_error.append(float(np.degrees(angular_error)))
                all_q_difference.append(q_difference)
        missing = int(len(first) + len(second) - 2 * len(pairs))
        missing_total += missing
        diagnostics["paired_support"][str(branch_id)] = {
            "quadrant_pair": f"{first_name}+{second_name}",
            "observed_side_counts": [int(len(first)), int(len(second))],
            "paired_count": int(len(pairs)),
            "missing_opposite_count": missing,
            "pair_fraction": float(2 * len(pairs) / max(1, len(first) + len(second))),
        }
    diagnostics["central_symmetry"] = {
        "matched_pair_count": len(all_distance),
        "median_deviation_q": float(np.median(all_distance)) if all_distance else None,
        "p95_deviation_q": float(np.percentile(all_distance, 95.0)) if all_distance else None,
        "median_angular_deviation_deg": float(np.median(all_angle_error)) if all_angle_error else None,
        "p95_angular_deviation_deg": float(np.percentile(all_angle_error, 95.0)) if all_angle_error else None,
        "median_q_difference": float(np.median(all_q_difference)) if all_q_difference else None,
        "p95_q_difference": float(np.percentile(all_q_difference, 95.0)) if all_q_difference else None,
        "q_difference_note": (
            "same-annulus sampling can impose q agreement; use angle/deviation for independent symmetry evidence"
            if azimuthal
            else "observed radial q difference between measured opposite points"
        ),
    }
    if missing_total:
        diagnostics["flags"].append("missing_opposite_support")
    if selected_leaks:
        diagnostics["flags"].append("branch_quadrant_leak")
    if not center_verified:
        diagnostics["flags"].append("symmetry_center_unverified")
    if selected_leaks:
        diagnostics["symmetry_status"] = "FAIL"
    elif missing_total or effective_unassigned or not center_verified:
        diagnostics["symmetry_status"] = "WARN"
    else:
        diagnostics["symmetry_status"] = "PASS"
    return diagnostics




def fit_symmetric_double_ellipse(
    points: Any,
    *,
    initial: Sequence[float] | Mapping[str, Any] | None = None,
    parameters: Any = None,
    params: Any = None,
    config: Any = None,
    robust_loss: str = "soft_l1",
    f_scale: float = 1.0,
    residual: str = "sampson",
    residual_kind: str | None = None,
    labels: Any = None,
    weights: Any = None,
    max_nfev: int | None = None,
    multistart: int = 7,
    reference_axis_deg: float = 0.0,
    q_unit: str | None = None,
    strict_symmetry: bool = False,
    cancel_event: Any = None,
) -> DoubleEllipseFit:
    """Fit a shared-centre pair of q-space ellipses at ``+/-theta``.

    This is the public observables adapter around
    :func:`butterfly_saxs.ellipse.fit_symmetric_ellipses`.  It retains the
    historical ``a/b/theta`` result fields while exposing canonical fit
    diagnostics (centre, covariance, standard errors, condition number,
    coverage and bound status).  Unlabelled ridge points are assigned to the
    nearer branch by the canonical residual function; no mirrored
    observations are fabricated.
    """

    if parameters is not None and params is not None:
        raise ValueError("supply only one of parameters or params")
    if parameters is not None and initial is not None:
        raise ValueError("supply only one of initial or parameters")
    if params is not None and initial is not None:
        raise ValueError("supply only one of initial or params")
    if config is not None and parameters is not None:
        raise ValueError("parameters are already supplied by config")
    if _fit_canonical_symmetric_ellipses is None:
        return _empty_double_ellipse(0, "canonical ellipse solver is unavailable", "solver_unavailable")

    resolved_q_unit = str(q_unit if q_unit is not None else _infer_ridge_q_unit(points) or "unknown")
    source_points = _ridge_source(points)
    point_flags: tuple[Any, ...] = ()
    if isinstance(points, Mapping) or isinstance(points, RidgeTrack):
        raw_flags = _get_field(points, ("flags",), ())
        try:
            point_flags = tuple(raw_flags or ())
        except TypeError:
            point_flags = ()
    azimuthal_trajectory = "azimuthal_peak_ridge" in point_flags
    if not azimuthal_trajectory and not isinstance(source_points, (Mapping, np.ndarray, str, bytes)):
        try:
            azimuthal_trajectory = any(
                str(_get_field(point, ("method",), "")).lower().replace("-", "_")
                == "azimuthal_peak"
                for point in source_points
            )
        except TypeError:
            azimuthal_trajectory = False

    source_parameters = parameters if parameters is not None else (params if params is not None else initial)
    center_hint, center_verified = _ellipse_center_hint(source_parameters)
    x, y, finite_xy = _ridge_xy(points)
    radius = np.hypot(x, y)
    inferred_labels = _ridge_components(points) if labels is None else np.asarray(labels)
    unassigned_count = 0
    if inferred_labels is not None:
        inferred_labels = np.asarray(inferred_labels, dtype=float).ravel()
        if inferred_labels.size != x.size:
            raise ValueError(f"labels must contain one value per ridge point ({x.size})")
        label_valid = np.isfinite(inferred_labels) & np.isin(inferred_labels, (0, 1))
        unassigned_count = int(np.count_nonzero(~label_valid))
        if labels is not None and unassigned_count and not strict_symmetry:
            raise ValueError("labels must contain only 0 or 1")
        if unassigned_count and not strict_symmetry:
            inferred_labels = None

    keep = finite_xy & np.isfinite(radius) & (radius > 0.0)
    if strict_symmetry and inferred_labels is None:
        return _empty_double_ellipse(
            int(np.count_nonzero(keep)),
            "strict symmetry requires aligned observed branch labels",
            "symmetry_unverified_unlabeled",
            q_unit=resolved_q_unit,
        )
    if strict_symmetry and inferred_labels is not None:
        keep &= np.isfinite(inferred_labels) & np.isin(inferred_labels, (0, 1))
        relative = np.column_stack((x - center_hint[0], y - center_hint[1]))
        local_angle = np.mod(
            np.arctan2(relative[:, 1], relative[:, 0]) - np.deg2rad(reference_axis_deg),
            2.0 * np.pi,
        )
        phase = np.mod(local_angle, 0.5 * np.pi)
        ambiguous = np.minimum(phase, 0.5 * np.pi - phase) < np.deg2rad(2.0)
        keep &= ~ambiguous
        unassigned_count += int(np.count_nonzero(ambiguous & finite_xy & (radius > 0.0)))
    xy = np.column_stack((x[keep], y[keep]))
    n_points = int(xy.shape[0])
    if n_points < 5:
        return _empty_double_ellipse(
            n_points,
            "at least five finite nonzero ridge points are required",
            "insufficient_points",
            q_unit=resolved_q_unit,
        )
    if inferred_labels is not None:
        inferred_labels = inferred_labels[keep].astype(int)

    point_weights = _ridge_weights(points) if weights is None else np.asarray(weights, dtype=float).ravel()
    if point_weights is not None:
        if point_weights.size != x.size:
            raise ValueError(f"weights must contain one value per ridge point ({x.size})")
        point_weights = point_weights[keep]

    canonical_parameters = _initial_to_canonical(source_parameters)
    if residual_kind is not None:
        residual = residual_kind
    label_options = [inferred_labels]
    if inferred_labels is not None and (labels is None or strict_symmetry):
        # Lobe-derived branch IDs are identities, not signed physical labels.
        # Try the one allowed global swap so a negative generator/model theta
        # is represented by the same canonical +|theta|/-|theta| pair.
        label_options.append(1 - inferred_labels)
    try:
        raise_if_cancelled(cancel_event, "observables:ellipse:start")
        labelled_results = [
            (
                option,
                _fit_canonical_symmetric_ellipses(
                    xy,
                    parameters=canonical_parameters,
                    residual=residual,
                    loss=robust_loss,
                    f_scale=f_scale,
                    labels=option,
                    weights=point_weights,
                    max_nfev=max_nfev,
                    multistart=multistart,
                    config=config,
                    reference_axis_deg=reference_axis_deg,
                    cancel_event=cancel_event,
                ),
            )
            for option in label_options
        ]
        selected_label_option, canonical_result = min(
            labelled_results,
            key=lambda item: (
                not bool(item[1].success),
                not bool(np.isfinite(item[1].cost)),
                float(item[1].cost) if np.isfinite(item[1].cost) else float("inf"),
            ),
        )
        swap_applied = bool(
            inferred_labels is not None
            and len(label_options) > 1
            and np.array_equal(selected_label_option, label_options[1])
        )
        inferred_labels = selected_label_option
    except AnalysisCancelled:
        raise
    except Exception as exc:
        # Invalid editable configurations should be visible to a caller, but
        # a numerical failure for measured data remains a structured result
        # suitable for batch processing.
        if isinstance(exc, ValueError) and (canonical_parameters is not None or config is not None):
            raise
        return _empty_double_ellipse(
            n_points,
            f"ellipse solver failed: {exc}",
            "solver_failed",
            q_unit=resolved_q_unit,
        )

    values = dict(canonical_result.values)
    cx = float(values.get("cx", float("nan")))
    cy = float(values.get("cy", float("nan")))
    a = float(values.get("a", float("nan")))
    ratio = float(values.get("axis_ratio", float("nan")))
    b = float(values.get("b", a * ratio if np.isfinite(a * ratio) else float("nan")))
    theta = float(values.get("theta", float("nan")))
    centre = (cx, cy)
    reference_axis = float(getattr(canonical_result, "reference_axis_deg", reference_axis_deg))
    reference_rad = np.deg2rad(reference_axis)
    plus = EllipseGeometry(a, b, reference_rad + theta, centre)
    minus = EllipseGeometry(a, b, reference_rad - theta, centre)

    # Use the same q-space residual family as the canonical solver for branch
    # accounting.  Explicit labels take precedence over nearest-branch
    # assignment, which is important for supervised component comparisons.
    branch: np.ndarray
    if inferred_labels is not None:
        branch = inferred_labels == 0
    elif _CanonicalEllipseGeometry is not None:
        geometry = _CanonicalEllipseGeometry.from_values(values)
        residual_name = str(residual).lower()
        if residual_name in {"geometric", "distance", "closest"}:
            from .ellipse import ellipse_geometric_residuals as residual_function
        else:
            residual_function = _canonical_sampson_residuals
        plus_geometry = _CanonicalEllipseGeometry(
            geometry.cx,
            geometry.cy,
            geometry.a,
            geometry.axis_ratio,
            reference_rad + geometry.theta,
        )
        minus_geometry = _CanonicalEllipseGeometry(
            geometry.cx,
            geometry.cy,
            geometry.a,
            geometry.axis_ratio,
            reference_rad - geometry.theta,
        )
        r_plus = np.asarray(residual_function(xy, plus_geometry), dtype=float)
        r_minus = np.asarray(
            residual_function(xy, minus_geometry),
            dtype=float,
        )
        branch = np.abs(r_plus) <= np.abs(r_minus)
    else:  # pragma: no cover - canonical solver import guarantees this path
        branch = np.ones(n_points, dtype=bool)

    branch_counts = (int(np.count_nonzero(branch)), int(np.count_nonzero(~branch)))
    symmetry = _symmetry_diagnostics(
        xy,
        inferred_labels if inferred_labels is not None else None,
        center=(cx, cy),
        reference_axis_deg=reference_axis,
        unassigned_count=unassigned_count,
        center_verified=center_verified,
        azimuthal=azimuthal_trajectory,
    )
    if swap_applied:
        symmetry["branch_leaks"]["global_swap"] = True
        symmetry["flags"].append("global_branch_swap_applied")
    if strict_symmetry and unassigned_count:
        symmetry["flags"].append("partial_branch_labels_excluded")
    ellipticity = float(np.sqrt(max(0.0, 1.0 - ratio * ratio))) if np.isfinite(ratio) else float("nan")
    stderr = {str(name): float(value) for name, value in dict(canonical_result.stderr).items()}
    theta_stderr = stderr.get("theta", float("nan"))
    stderr.setdefault("theta_deg", float(np.degrees(theta_stderr)) if np.isfinite(theta_stderr) else float("nan"))
    bound_flags = {str(name): bool(value) for name, value in dict(canonical_result.bound_flags).items()}
    bound_status = {str(name): value for name, value in dict(canonical_result.bound_status).items()}
    unit = resolved_q_unit
    flags = list(_flags_with_q_unit(APPARENT_FLAGS, unit))
    if branch_counts[0] == 0 or branch_counts[1] == 0:
        flags.append("single_branch_supported")
    if any(bound_flags.values()):
        flags.append("parameter_at_bound")
    if not canonical_result.success:
        flags.append("solver_failed")

    q_scale = _q_to_nm_inverse_scale(unit)
    Ln_minor = float("nan")
    Lz = float("nan")
    if azimuthal_trajectory:
        flags.append("spacing_unavailable_azimuthal_trajectory")
    elif q_scale is not None and b > 0:
        # The reported spacing formulas describe an origin-centred q ellipse.
        # A fitted non-zero centre can represent beam-centre error or a real
        # translation; without an explicit physical model, suppress both
        # derived lengths instead of silently applying the origin formula.
        center_scale = max(1.0, abs(a), abs(b)) if np.isfinite(a) and np.isfinite(b) else 1.0
        center_tol = _SPACING_CENTER_REL_TOL * center_scale
        centered = (
            np.isfinite(cx)
            and np.isfinite(cy)
            and np.hypot(cx, cy) <= center_tol
        )
        if not centered:
            flags.extend(("spacing_unavailable_nonzero_center", "spacing_requires_origin_centered_ellipse_assumption"))
        else:
            Ln_minor = float(2.0 * np.pi / (b * q_scale))
            qz = float(ellipse_radius(np.pi / 2.0, a, b, theta))
            if qz > 0:
                Lz = float(2.0 * np.pi / (qz * q_scale))
            flags.append("spacing_requires_origin_centered_ellipse_assumption")

    residuals = np.asarray(canonical_result.residuals, dtype=float)
    rss = float(np.sum(np.square(residuals))) if residuals.size else float("nan")
    return DoubleEllipseFit(
        a=a,
        b=b,
        theta=theta,
        ellipses=(plus, minus),
        ellipticity=ellipticity,
        axes_ratio=ratio,
        rmse=float(canonical_result.rmse),
        rss=rss,
        n_points=n_points,
        branch_counts=branch_counts,
        success=bool(canonical_result.success),
        message=str(canonical_result.message),
        flags=tuple(flags),
        center=centre,
        stderr=stderr,
        covariance=canonical_result.covariance,
        condition_number=float(canonical_result.condition_number),
        coverage=canonical_result.coverage,
        bound_flags=bound_flags,
        bound_status=bound_status,
        free_names=tuple(canonical_result.free_names),
        parameter_values={
            **values,
            "reference_axis_deg": reference_axis,
            "ellipse_axis_tilt_deg": float(np.degrees(theta)),
        },
        reference_axis_deg=reference_axis,
        q_unit=unit,
        Ln_from_minor_axis_nm=Ln_minor,
        Lz_from_draw_axis_nm=Lz,
        branch_assignment=getattr(canonical_result, "branch_assignment", None),
        candidate_solutions=tuple(getattr(canonical_result, "candidate_solutions", ()) or ()),
        selected_start_index=int(getattr(canonical_result, "selected_start_index", 0)),
        multistart_count=int(getattr(canonical_result, "multistart_count", 1)),
        symmetry=symmetry,
        branch_assignment_indices=np.flatnonzero(keep),
    )


def _write_fit_branch_ids(ridge: RidgeTrack, fit: Any) -> None:
    """Copy an optional fit assignment onto the actual measured points."""

    assignment = _get_field(fit, ("branch_assignment", "branch_assignments"), None)
    if assignment is None:
        return
    if isinstance(assignment, Mapping):
        assignment = _get_field(assignment, ("point_branch", "labels", "branch_id", "values"), None)
    if assignment is None:
        return
    values = np.asarray(assignment).ravel()
    valid_indices = np.flatnonzero(ridge.valid)
    explicit_indices = _get_field(fit, ("branch_assignment_indices",), None)
    if explicit_indices is not None:
        explicit_indices = np.asarray(explicit_indices, dtype=int).ravel()
    if explicit_indices is not None and values.size == explicit_indices.size:
        target_indices = explicit_indices
    elif values.size == len(ridge.points):
        target_indices = np.arange(len(ridge.points), dtype=int)
    elif values.size == valid_indices.size:
        target_indices = valid_indices
    else:
        return
    for index, value in zip(target_indices, values):
        if value is None:
            continue
        try:
            branch_id = int(value)
        except (TypeError, ValueError):
            continue
        if branch_id not in (0, 1):
            continue
        point = ridge.points[int(index)]
        point.branch_id = branch_id
        quadrant = point.metadata.get("quadrant")
        if quadrant in {"QI", "QIII"}:
            point.metadata["quadrant_pair"] = "QI+QIII"
        elif quadrant in {"QII", "QIV"}:
            point.metadata["quadrant_pair"] = "QII+QIV"
        point.metadata["branch_id"] = branch_id


def _assign_symmetric_lobe_branches(
    ridge: RidgeTrack,
    lobes: Sequence[LobeMetrics],
    *,
    reference_axis_deg: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Assign fixed opposite-quadrant branch identity to observed points.

    The branch is determined once from the q-centred reference quadrants:
    QI+QIII -> 0 and QII+QIV -> 1.  It is never selected point-by-point by
    whichever ellipse happens to have the smaller residual near a crossing.
    """

    for point in ridge.points:
        if not point.valid or not np.isfinite(point.q):
            continue
        point_x = _get_field(point, ("qx", "x"), None)
        point_y = _get_field(point, ("qy", "y"), None)
        if point_x is not None and point_y is not None:
            relative_angle = float(
                (
                    np.arctan2(
                        float(point_y) - float(center[1]),
                        float(point_x) - float(center[0]),
                    )
                    - np.deg2rad(float(reference_axis_deg))
                )
                % (2.0 * np.pi)
            )
        else:
            relative_angle = float(
                (point.angle - np.deg2rad(float(reference_axis_deg))) % (2.0 * np.pi)
            )
        quadrant = int(np.floor(relative_angle / (0.5 * np.pi))) % 4
        boundary_phase = np.mod(relative_angle, 0.5 * np.pi)
        if min(boundary_phase, 0.5 * np.pi - boundary_phase) < np.deg2rad(2.0):
            point.branch_id = None
            point.metadata["quadrant"] = ("QI", "QII", "QIII", "QIV")[quadrant]
            point.metadata["quadrant_pair"] = (
                "QI+QIII" if quadrant in (0, 2) else "QII+QIV"
            )
            point.metadata["branch_assignment_source"] = "quadrant_boundary_ambiguous"
            point.metadata["symmetry_flags"] = ("quadrant_boundary_ambiguous",)
            continue
        point.branch_id = 0 if quadrant in (0, 2) else 1
        point.metadata["quadrant"] = ("QI", "QII", "QIII", "QIV")[quadrant]
        point.metadata["quadrant_pair"] = "QI+QIII" if point.branch_id == 0 else "QII+QIV"
        point.metadata["branch_assignment_source"] = "reference_quadrant"
        point.metadata["symmetry_flags"] = ("quadrant_pairing_observed",)
        point.metadata["branch_id"] = point.branch_id


fit_symmetric_ellipses = fit_symmetric_double_ellipse
fit_double_ellipse = fit_symmetric_double_ellipse


__all__ = [
    "ellipse_radius",
    "fit_double_ellipse",
    "fit_symmetric_double_ellipse",
    "fit_symmetric_ellipses",
    "_assign_symmetric_lobe_branches",
    "_ellipse_center_hint",
    "_empty_double_ellipse",
    "_infer_ridge_q_unit",
    "_initial_to_canonical",
    "_quadrant_labels",
    "_ridge_components",
    "_ridge_source",
    "_ridge_weights",
    "_ridge_xy",
    "_symmetry_diagnostics",
    "_write_fit_branch_ids",
]
