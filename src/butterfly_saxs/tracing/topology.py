"""Observed ridge graph connectivity, reference-quadrant arc topology, and manual edits/seeds."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np

from .._primitives import parse_q_window as _shared_parse_q_window


def _normalise_options(options: Any) -> dict[str, Any]:
    if options is None:
        values: dict[str, Any] = {}
    elif isinstance(options, Mapping):
        values = dict(options)
    else:
        try:
            values = dict(vars(options))
        except TypeError:
            values = {}
    defaults: dict[str, Any] = {
        "smoothing_scales": (1.2, 2.2, 3.6),
        "support_min": 0.78,
        "curvature_percentile": 65.0,
        "candidate_snr_min": 3.0,
        "axis_separation_min": 0.12,
        "q_scale_anisotropy_max": 4.0,
        "normal_step_factor": 1.0,
        "max_offset_fraction": 0.75,
        "nms_radius_factor": 0.45,
        "graph_radius_factor": 3.5,
        "graph_normal_factor": 1.8,
        "graph_tangent_angle_deg": 65.0,
        "max_arc_tangent_error_deg": 45.0,
        "min_arc_points": 6,
        "max_candidates": 30000,
        "max_points": None,
        "snr_threshold": 1.5,
        "profile_half_width_q": None,
        "profile_samples": 33,
        "profile_delta_bic": 4.0,
        "profile_refinement": True,
        "profile_refinement_max_q_step": 1.5,
        "profile_refinement_min_snr": 3.0,
        "run_wang_check": True,
        "quadrant_axis_tol_deg": 4.0,
        "seed_snap_factor": 3.0,
        "center_qx": 0.0,
        "center_qy": 0.0,
        "first_order_prefer": True,
        "fill_sparse_first_order_ring": True,
    }
    merged = {**defaults, **values}
    if "scales" in values and "smoothing_scales" not in values:
        merged["smoothing_scales"] = values["scales"]
    if "smooth_scales" in values and "smoothing_scales" not in values:
        merged["smoothing_scales"] = values["smooth_scales"]
    if "scales_px" in values and "smoothing_scales" not in values:
        merged["smoothing_scales"] = values["scales_px"]
    if "min_candidate_snr" in values and "candidate_snr_min" not in values:
        merged["candidate_snr_min"] = values["min_candidate_snr"]
    if "refine_from_profile" in values and "profile_refinement" not in values:
        merged["profile_refinement"] = values["refine_from_profile"]
    if "max_points" in values and values["max_points"] is not None:
        merged["max_candidates"] = max(int(values["max_points"]), int(merged["max_candidates"]))
    center = values.get("center_hint", values.get("center_q", None))
    if center is not None:
        try:
            merged["center_qx"], merged["center_qy"] = float(center[0]), float(center[1])
        except Exception as exc:
            raise ValueError("center_hint/center_q must be a two-element sequence") from exc
    try:
        scales = tuple(sorted({float(scale) for scale in merged["smoothing_scales"]}))
    except (TypeError, ValueError) as exc:
        raise ValueError("smoothing_scales must contain finite positive values") from exc
    if not scales or any(not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("smoothing_scales must contain finite positive values")
    merged["smoothing_scales"] = scales
    for name, lower, upper in (
        ("support_min", 0.0, 1.0),
        ("axis_separation_min", 0.0, 1.0),
        ("curvature_percentile", 0.0, 100.0),
        ("nms_radius_factor", 0.05, 10.0),
        ("graph_radius_factor", 0.5, 20.0),
        ("graph_normal_factor", 0.1, 20.0),
        ("graph_tangent_angle_deg", 1.0, 90.0),
        ("max_arc_tangent_error_deg", 1.0, 90.0),
        ("q_scale_anisotropy_max", 1.0, 1e6),
        ("snr_threshold", 0.0, 1e9),
        ("candidate_snr_min", 0.0, 1e9),
        ("profile_refinement_max_q_step", 1e-6, 10.0),
        ("profile_refinement_min_snr", 0.0, 1e9),
        ("quadrant_axis_tol_deg", 0.0, 45.0),
        ("seed_snap_factor", 0.1, 100.0),
    ):
        number = float(merged[name])
        if not np.isfinite(number) or number < lower or number > upper:
            raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        merged[name] = number
    merged["min_arc_points"] = max(2, int(merged["min_arc_points"]))
    merged["max_candidates"] = max(1, int(merged["max_candidates"]))
    if merged["max_points"] is not None:
        merged["max_points"] = max(1, int(merged["max_points"]))
    merged["profile_samples"] = max(9, int(merged["profile_samples"]))
    if not isinstance(merged["profile_refinement"], (bool, np.bool_)):
        raise ValueError("profile_refinement must be boolean")
    merged["profile_refinement"] = bool(merged["profile_refinement"])
    if "first_order_band" in values and "first_order_prefer" not in values:
        merged["first_order_prefer"] = values["first_order_band"]
    if not isinstance(merged["first_order_prefer"], (bool, np.bool_)):
        raise ValueError("first_order_prefer must be boolean")
    merged["first_order_prefer"] = bool(merged["first_order_prefer"])
    if not isinstance(merged["fill_sparse_first_order_ring"], (bool, np.bool_)):
        raise ValueError("fill_sparse_first_order_ring must be boolean")
    merged["fill_sparse_first_order_ring"] = bool(merged["fill_sparse_first_order_ring"])
    return merged


def _parse_q_window(q_window: Any, q: np.ndarray) -> tuple[float, float]:
    parsed = _shared_parse_q_window(
        q_window,
        q,
        empty_fallback=(float("nan"), float("nan")),
    )
    assert parsed is not None
    return parsed


def _polygon_mask(qx: np.ndarray, qy: np.ndarray, vertices: Any) -> np.ndarray:
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3 or not np.all(np.isfinite(points)):
        raise ValueError("polygon points must be a finite (n>=3, 2) array")
    inside = np.zeros(qx.shape, dtype=bool)
    x, y = qx, qy
    x0, y0 = points[-1]
    for x1, y1 in points:
        crosses = (y1 > y) != (y0 > y)
        denominator = y0 - y1
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = (x0 - x1) * (y - y1) / denominator + x1
        inside ^= crosses & (x < crossing_x)
        x0, y0 = x1, y1
    return inside


def _apply_edits(
    valid: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    edits: Any,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    if edits is None:
        return valid, [], []
    if isinstance(edits, Mapping):
        actions = edits.get("actions", edits.get("edits", []))
    else:
        actions = edits
    if isinstance(actions, Mapping) or isinstance(actions, (str, bytes)):
        actions = [actions]
    base_valid = np.asarray(valid, dtype=bool)
    edited = base_valid.copy()
    applied: list[dict[str, Any]] = []
    seed_actions: list[dict[str, Any]] = []
    for action in list(actions or []):
        if not isinstance(action, Mapping):
            continue
        kind = str(action.get("type", action.get("kind", ""))).strip().lower().replace("-", "_")
        if kind == "seed":
            seed_actions.append(dict(action))
            continue
        if kind not in {"exclude_polygon", "include_polygon"}:
            continue
        polygon = _polygon_mask(qx, qy, action.get("points", []))
        if kind == "exclude_polygon":
            edited[polygon] = False
        else:
            # Restore pre-edit validity only. Include can undo earlier polygon
            # edits, but cannot override input masks (including ROI masks),
            # finite-value checks, or the q-window.
            edited[polygon] = base_valid[polygon]
        applied.append({"type": kind, "n_pixels": int(np.count_nonzero(polygon))})
    return edited, applied, seed_actions


def _point_signature(image: np.ndarray, options: Mapping[str, Any], q_window: Any = None) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(image).view(np.uint8)).hexdigest()[:16]
    stable_options = {
        key: value
        for key, value in options.items()
        if key not in {"edits", "seed", "resamples", "sensitivity", "stage"}
    }
    try:
        option_text = json.dumps(
            {"options": stable_options, "q_window": q_window},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    except TypeError:
        option_text = repr((sorted((str(key), repr(value)) for key, value in stable_options.items()), repr(q_window)))
    option_digest = hashlib.sha1(option_text.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{option_digest}"


def _assign_point_ids(candidates: list[dict[str, Any]], signature: str) -> None:
    for point in candidates:
        text = f"{signature}|{point['qx']:.8g}|{point['qy']:.8g}|{point['pixel_x']:.5f}|{point['pixel_y']:.5f}"
        point["point_id"] = "ridge-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _graph_arcs(candidates: list[dict[str, Any]], options: Mapping[str, Any], q_step: float) -> tuple[list[list[int]], list[tuple[int, int]]]:
    if not candidates:
        return [], []
    graph_factor = float(options["graph_radius_factor"])
    radius = graph_factor * max(q_step, np.finfo(float).eps)
    base = max(radius / 2.0, np.finfo(float).eps)
    finite_scales = [
        float(point["scale"])
        for point in candidates
        if math.isfinite(float(point.get("scale", 1.0)))
    ]
    max_scale = max(finite_scales, default=1.0)
    max_search_radius = max(radius, max_scale * q_step * graph_factor)
    cell_reach = max(2, int(math.ceil(max_search_radius / base)))
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(candidates):
        px, py = float(point["qx"]), float(point["qy"])
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        bins[(int(math.floor(px / base)), int(math.floor(py / base)))].append(index)
    union = _UnionFind(len(candidates))
    edges: list[tuple[int, int]] = []
    tangent_cos = math.cos(math.radians(float(options["graph_tangent_angle_deg"])))
    for index, point in enumerate(candidates):
        px, py = float(point["qx"]), float(point["qy"])
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        bx, by = int(math.floor(px / base)), int(math.floor(py / base))
        point_scale_raw = float(point.get("scale", 1.0))
        if not math.isfinite(point_scale_raw):
            point_scale_raw = 1.0
        scale = max(point_scale_raw * q_step, q_step)
        for ix in range(bx - cell_reach, bx + cell_reach + 1):
            for iy in range(by - cell_reach, by + cell_reach + 1):
                for other_index in bins.get((ix, iy), []):
                    if other_index <= index:
                        continue
                    other = candidates[other_index]
                    # Topology is built from candidates that pass the local
                    # curvature/axis gates.  Rejected candidates remain in
                    # ``points`` for UI review but must not act as bridges
                    # that stitch unrelated arcs through noise or a mask.
                    if not bool(point.get("accepted", False)) or not bool(other.get("accepted", False)):
                        continue
                    # Reference-axis branch identity is assigned before graph
                    # connectivity in the production trace.  Do not let an
                    # unresolved axis-boundary point bridge two branches.
                    branch = int(point.get("branch_id", -1))
                    other_branch = int(other.get("branch_id", -1))
                    if branch not in (0, 1) or other_branch not in (0, 1) or branch != other_branch:
                        continue
                    dx, dy = float(other["qx"]) - px, float(other["qy"]) - py
                    distance = math.hypot(dx, dy)
                    other_scale_raw = float(other.get("scale", 1.0))
                    if not math.isfinite(other_scale_raw):
                        other_scale_raw = 1.0
                    max_distance = max(
                        radius,
                        max(point_scale_raw, other_scale_raw) * q_step * graph_factor,
                    )
                    if distance <= np.finfo(float).eps or distance > max_distance:
                        continue
                    dot_tangent = abs(float(point["tangent_qx"]) * float(other["tangent_qx"]) + float(point["tangent_qy"]) * float(other["tangent_qy"]))
                    if not np.isfinite(dot_tangent) or dot_tangent < tangent_cos:
                        continue
                    normal_projection = min(
                        abs(dx * float(point["normal_qx"]) + dy * float(point["normal_qy"])),
                        abs(dx * float(other["normal_qx"]) + dy * float(other["normal_qy"])),
                    )
                    if normal_projection > float(options["graph_normal_factor"]) * max(scale, other_scale_raw * q_step):
                        continue
                    union.union(index, other_index)
                    edges.append((index, other_index))
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        components[union.find(index)].append(index)
    groups = sorted(components.values(), key=lambda values: min(values))
    return groups, edges


def _ordered_component(component: list[int], edges: list[tuple[int, int]], candidates: list[dict[str, Any]]) -> list[int]:
    if len(component) <= 2:
        return sorted(component)
    members = set(component)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        if left in members and right in members:
            adjacency[left].append(right)
            adjacency[right].append(left)
    endpoints = [index for index in component if len(adjacency[index]) == 1]
    if endpoints:
        start = min(endpoints, key=lambda index: (float(candidates[index]["qx"]), float(candidates[index]["qy"])))
    else:
        start = min(component, key=lambda index: (float(candidates[index]["qx"]), float(candidates[index]["qy"])))
    ordered = [start]
    previous = -1
    current = start
    while True:
        neighbours = [value for value in adjacency[current] if value != previous and value not in ordered]
        if not neighbours:
            break
        cx = float(candidates[current]["qx"])
        cy = float(candidates[current]["qy"])
        if previous >= 0:
            vx = cx - float(candidates[previous]["qx"])
            vy = cy - float(candidates[previous]["qy"])
            norm = math.hypot(vx, vy) or 1.0
            vx, vy = vx / norm, vy / norm
            forward: list[tuple[float, float, int]] = []
            fallback: list[tuple[float, float, int]] = []
            for index in neighbours:
                nx = float(candidates[index]["qx"]) - cx
                ny = float(candidates[index]["qy"]) - cy
                n_norm = math.hypot(nx, ny)
                if n_norm <= np.finfo(float).eps:
                    cosine = -2.0
                else:
                    cosine = (vx * nx + vy * ny) / n_norm
                if cosine >= 0.0:
                    forward.append((n_norm, -cosine, index))
                else:
                    fallback.append((-cosine, n_norm, index))
            if forward:
                # Step to the nearest forward-continuing neighbour so multi-scale
                # chord edges do not leap over intermediate arc points.
                current = min(forward)[2]
            else:
                current = min(fallback)[2]
        else:
            current = min(
                neighbours,
                key=lambda index: (
                    math.hypot(float(candidates[index]["qx"]) - cx, float(candidates[index]["qy"]) - cy),
                    index,
                ),
            )
        ordered.append(current)
        previous, current = ordered[-2], ordered[-1]
        if len(ordered) == len(component):
            break
    if len(ordered) != len(component):
        remaining = [index for index in component if index not in ordered]
        for index in sorted(remaining):
            px = float(candidates[index]["qx"])
            py = float(candidates[index]["qy"])
            nearest_pos = min(
                range(len(ordered)),
                key=lambda pos: math.hypot(
                    float(candidates[ordered[pos]]["qx"]) - px,
                    float(candidates[ordered[pos]]["qy"]) - py,
                ),
            )
            ordered.insert(nearest_pos + 1, index)
    return ordered


def _quadrant_branch(qx: float, qy: float, *, center_x: float, center_y: float, reference_axis_deg: float, tolerance_deg: float) -> tuple[int, str]:
    theta = np.deg2rad(float(reference_axis_deg))
    dx, dy = qx - center_x, qy - center_y
    along = dx * np.cos(theta) + dy * np.sin(theta)
    perpendicular = -dx * np.sin(theta) + dy * np.cos(theta)
    angle = float(np.degrees(np.arctan2(perpendicular, along)))
    if abs(abs(angle) - 90.0) <= tolerance_deg or abs(abs(angle) - 180.0) <= tolerance_deg or abs(angle) <= tolerance_deg:
        quadrant = "axis_boundary"
        return -1, quadrant
    if along >= 0 and perpendicular >= 0:
        return 0, "QI"
    if along < 0 and perpendicular >= 0:
        return 1, "QII"
    if along < 0 and perpendicular < 0:
        return 0, "QIII"
    return 1, "QIV"


def _assign_reference_branches(candidates: list[dict[str, Any]], options: Mapping[str, Any]) -> None:
    """Assign reference-axis identities before graph connectivity is formed."""

    center_x, center_y = float(options["center_qx"]), float(options["center_qy"])
    reference = float(options.get("reference_axis_deg", 0.0))
    tolerance = float(options["quadrant_axis_tol_deg"])
    for point in candidates:
        branch, quadrant = _quadrant_branch(
            float(point["qx"]),
            float(point["qy"]),
            center_x=center_x,
            center_y=center_y,
            reference_axis_deg=reference,
            tolerance_deg=tolerance,
        )
        point["branch_id"], point["quadrant"] = int(branch), quadrant
        if branch not in (0, 1):
            point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["unresolved_reference_branch"]))


def _side_from_reference_axis(
    qx: float,
    qy: float,
    *,
    center_x: float,
    center_y: float,
    reference_axis_deg: float,
    q_step: float,
) -> str:
    """Upper/lower from the declared draw axis when the local major is unusable."""

    theta = math.radians(float(reference_axis_deg))
    dx = float(qx) - float(center_x)
    dy = float(qy) - float(center_y)
    perpendicular = -dx * math.sin(theta) + dy * math.cos(theta)
    scale = _side_axis_threshold(math.hypot(dx, dy), q_step)
    if perpendicular > scale:
        return "upper"
    if perpendicular < -scale:
        return "lower"
    return "unknown"


def _side_axis_threshold(radius: float, q_step: float) -> float:
    """Perpendicular band that is still 'on' the local major axis.

    This is an angular criterion, not a detector-pixel floor.  A one-pixel
    band swallows an entire flat butterfly wing once ``b`` is smaller than
    the q-grid step, leaving every point ``side=unknown``.
    """

    del q_step
    finite_radius = abs(float(radius)) if np.isfinite(radius) else 0.0
    return max(finite_radius * math.sin(math.radians(0.20)), float(np.finfo(float).eps))


def _point_identity_key(point: Mapping[str, Any]) -> tuple[int, str] | None:
    """Return a quantitative (branch, side) key, or None when it is unresolved."""

    try:
        branch = int(point.get("branch_id", -1))
    except (TypeError, ValueError):
        return None
    side = str(point.get("side", "unknown") or "unknown")
    if branch in (0, 1) and side in {"upper", "lower"}:
        return branch, side
    return None


def _demote_unresolved_identity_point(point: dict[str, Any], reason: str, *flags: str) -> None:
    """Keep a tip/axis candidate visible without letting it poison siblings."""

    extra = list(flags) + [reason]
    point["topology_flags"] = list(dict.fromkeys(list(point.get("topology_flags", [])) + extra))
    if bool(point.get("accepted", False)):
        point["accepted"] = False
        point["reason"] = reason


def _component_major_direction(
    ordered: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    *,
    center_x: float,
    center_y: float,
    reference_vector: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    positions = np.asarray(
        [[candidates[index]["qx"], candidates[index]["qy"]] for index in ordered],
        dtype=float,
    )
    centered = positions - np.asarray([center_x, center_y])
    if len(positions) >= 3 and np.linalg.matrix_rank(np.cov(centered.T)) > 0:
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    else:
        tangents = np.asarray(
            [[candidates[index]["tangent_qx"], candidates[index]["tangent_qy"]] for index in ordered],
            dtype=float,
        )
        major = np.nanmedian(tangents, axis=0)
    major_norm = float(np.hypot(*major))
    direction_confidence = 0.0
    if major_norm > 1e-10 and np.all(np.isfinite(major)):
        major = major / major_norm
        alignment = abs(float(np.dot(major, reference_vector)))
        direction_confidence = float(min(1.0, alignment))
        if float(np.dot(major, reference_vector)) < 0:
            major = -major
    else:
        major = np.asarray([np.nan, np.nan], dtype=float)
    resolved = bool(np.all(np.isfinite(major)) and direction_confidence >= 0.20)
    return major, direction_confidence, resolved


def _assign_component_identities(
    ordered: Sequence[int],
    candidates: list[dict[str, Any]],
    options: Mapping[str, Any],
    q_step: float,
    *,
    center_x: float,
    center_y: float,
    major: np.ndarray,
    resolved: bool,
) -> None:
    positions = np.asarray(
        [[candidates[index]["qx"], candidates[index]["qy"]] for index in ordered],
        dtype=float,
    )
    centered = positions - np.asarray([center_x, center_y])
    side_normal = np.asarray([-major[1], major[0]]) if resolved else np.asarray([np.nan, np.nan])
    median_radius = float(np.nanmedian(np.linalg.norm(centered, axis=1)))
    side_scale = _side_axis_threshold(median_radius, q_step)
    for order_index, index in enumerate(ordered):
        point = candidates[index]
        point["arc_order"] = int(order_index)
        branch, quadrant = _quadrant_branch(
            float(point["qx"]),
            float(point["qy"]),
            center_x=center_x,
            center_y=center_y,
            reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
            tolerance_deg=float(options["quadrant_axis_tol_deg"]),
        )
        point["branch_id"], point["quadrant"] = int(branch), quadrant
        short_arc = len(ordered) < int(options["min_arc_points"])
        use_reference_side = (not resolved) or short_arc
        if use_reference_side:
            point["side"] = _side_from_reference_axis(
                float(point["qx"]),
                float(point["qy"]),
                center_x=center_x,
                center_y=center_y,
                reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
                q_step=q_step,
            )
        else:
            radial = np.asarray([float(point["qx"]) - center_x, float(point["qy"]) - center_y])
            side_value = float(np.dot(radial, side_normal))
            point["side"] = (
                "upper" if side_value > side_scale else "lower" if side_value < -side_scale else "unknown"
            )
        flags = list(point.get("topology_flags", []))
        flags.append("connected_arc")
        point["topology_flag"] = "connected_observed_arc"
        point["topology"] = "connected_observed_arc"
        if short_arc:
            flags.append("short_arc")
        if not resolved:
            flags.append("major_direction_ambiguous")
        if use_reference_side and point["side"] in {"upper", "lower"}:
            flags.append("side_from_reference_axis")
        elif not resolved and bool(point.get("accepted", False)):
            point["reason"] = "ambiguous_major_direction"
        if _point_identity_key(point) is None:
            reason = "side_axis_boundary" if int(point["branch_id"]) in (0, 1) else "unresolved_reference_branch"
            flags.append(reason)
            if bool(point.get("accepted", False)):
                point["accepted"] = False
                point["reason"] = reason
        point["topology_flags"] = list(dict.fromkeys(flags))


def _component_tangent_errors(
    ordered: Sequence[int],
    candidates: list[dict[str, Any]],
) -> list[float]:
    tangent_errors: list[float] = []
    for order_index, index in enumerate(ordered):
        if len(ordered) == 1:
            tangent_errors.append(float("nan"))
            continue
        neighbour = ordered[min(order_index + 1, len(ordered) - 1)] if order_index == 0 else ordered[order_index - 1]
        dx = float(candidates[neighbour]["qx"]) - float(candidates[index]["qx"])
        dy = float(candidates[neighbour]["qy"]) - float(candidates[index]["qy"])
        norm = math.hypot(dx, dy)
        if norm <= np.finfo(float).eps:
            tangent_errors.append(float("nan"))
            continue
        observed = np.asarray([dx / norm, dy / norm])
        tangent = np.asarray(
            [float(candidates[index]["tangent_qx"]), float(candidates[index]["tangent_qy"])]
        )
        dot = float(np.clip(abs(np.dot(observed, tangent)), -1.0, 1.0))
        error = float(np.degrees(np.arccos(dot)))
        tangent_errors.append(error)
        candidates[index]["tangent_error_deg"] = error
    return tangent_errors


def _split_component_identities(ordered: Sequence[int], candidates: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], list[int]]:
    """Split one connected wing into identity-pure (branch, side) groups.

    A continuous ellipse wing often crosses the local major axis at the tip.
    Those two sides belong to the same observed component but must not share
    one arc identity, or the later refresh would discard the whole wing.
    """

    groups: dict[tuple[int, str], list[int]] = {}
    for index in ordered:
        key = _point_identity_key(candidates[index])
        if key is None:
            continue
        groups.setdefault(key, []).append(index)
    return groups


def _build_identity_arc(
    *,
    arc_id: int,
    ordered: Sequence[int],
    candidates: list[dict[str, Any]],
    options: Mapping[str, Any],
    major: np.ndarray,
    direction_confidence: float,
    resolved: bool,
    parent_split: bool,
) -> dict[str, Any]:
    for order_index, index in enumerate(ordered):
        candidates[index]["arc_id"] = int(arc_id)
        candidates[index]["arc_order"] = int(order_index)
        if parent_split:
            candidates[index]["topology_flags"] = list(
                dict.fromkeys(candidates[index].get("topology_flags", []) + ["identity_split_from_connected_wing"])
            )
    tangent_errors = _component_tangent_errors(ordered, candidates)
    accepted_count = int(sum(bool(candidates[index]["accepted"]) for index in ordered))
    scale_values = np.asarray([float(candidates[index]["scale"]) for index in ordered], dtype=float)
    scale_stability = (
        float(1.0 / (1.0 + np.nanstd(scale_values) / max(np.nanmean(scale_values), np.finfo(float).eps)))
        if scale_values.size
        else float("nan")
    )
    for index in ordered:
        candidates[index]["scale_stability"] = scale_stability
        candidates[index]["scale_stable"] = bool(scale_stability >= 0.5)
        if scale_stability < 0.5:
            candidates[index]["topology_flags"] = list(
                dict.fromkeys(candidates[index]["topology_flags"] + ["scale_unstable"])
            )
    tangent_error_median = (
        float(np.nanmedian(tangent_errors)) if np.any(np.isfinite(tangent_errors)) else float("nan")
    )
    branch_ids = sorted({int(candidates[index]["branch_id"]) for index in ordered})
    sides = sorted({str(candidates[index]["side"]) for index in ordered if _point_identity_key(candidates[index])})
    identity_resolved = bool(len(branch_ids) == 1 and branch_ids[0] in (0, 1) and len(sides) == 1 and sides[0] in {"upper", "lower"})
    identity_flags: list[str] = []
    if not (len(branch_ids) == 1 and branch_ids[0] in (0, 1)):
        identity_flags.append("unresolved_or_mixed_branch")
    if not (len(sides) == 1 and sides[0] in {"upper", "lower"}):
        identity_flags.append("unresolved_or_mixed_side")
    if parent_split:
        identity_flags.append("identity_split_from_connected_wing")
    arc_valid = bool(
        accepted_count >= int(options["min_arc_points"])
        and np.isfinite(tangent_error_median)
        and tangent_error_median <= float(options.get("max_arc_tangent_error_deg", 45.0))
        and scale_stability >= 0.4
        and identity_resolved
    )
    return {
        "arc_id": int(arc_id),
        "point_ids": [str(candidates[index]["point_id"]) for index in ordered],
        "ordered_point_ids": [str(candidates[index]["point_id"]) for index in ordered],
        "n_points": int(len(ordered)),
        "accepted_points": accepted_count,
        "valid": arc_valid,
        "reason": "accepted" if arc_valid else "short_or_rejected_arc",
        "major_direction_qx": float(major[0]),
        "major_direction_qy": float(major[1]),
        "major_direction_resolved": resolved,
        "major_direction_confidence": direction_confidence,
        "side_reference": "observed_arc_major_direction_and_center",
        "reference_axis_deg": float(options.get("reference_axis_deg", 0.0)),
        "branch_ids": branch_ids,
        "quadrants": sorted({str(candidates[index]["quadrant"]) for index in ordered}),
        "sides": sides,
        "accepted_branch_ids": branch_ids,
        "accepted_sides": sides,
        "identity_resolved": identity_resolved,
        "identity_flags": identity_flags,
        "tangent_error_deg": tangent_errors,
        "tangent_error_median_deg": tangent_error_median,
        "scale_stability": scale_stability,
        "topology_flag": "connected_observed_arc",
        "topology_flags": ["topology_before_ellipse_fit", *identity_flags],
        "parent_component_split": bool(parent_split),
    }


def _attach_unresolved_to_nearest_arc(
    unresolved: Sequence[int],
    arcs: Sequence[Mapping[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    """Keep tip/axis points on a same-branch sibling arc for review only."""

    if not unresolved or not arcs:
        return
    by_id = {str(item.get("point_id")): item for item in candidates}
    for index in unresolved:
        point = candidates[index]
        if int(point.get("arc_id", -1)) >= 0:
            continue
        branch = int(point.get("branch_id", -1))
        qx, qy = float(point["qx"]), float(point["qy"])
        best_id = None
        best_distance = float("inf")
        for arc in arcs:
            if branch in (0, 1) and list(arc.get("branch_ids", ())) != [branch]:
                continue
            for point_id in arc.get("ordered_point_ids", ()):
                sibling = by_id.get(str(point_id))
                if sibling is None:
                    continue
                distance = math.hypot(qx - float(sibling["qx"]), qy - float(sibling["qy"]))
                if distance < best_distance:
                    best_distance = distance
                    best_id = int(arc["arc_id"])
        if best_id is None:
            continue
        point["arc_id"] = best_id
        point["topology_flags"] = list(
            dict.fromkeys(point.get("topology_flags", []) + ["attached_unresolved_identity"])
        )


def _arc_topology(
    groups: list[list[int]],
    edges: list[tuple[int, int]],
    candidates: list[dict[str, Any]],
    options: Mapping[str, Any],
    q_step: float,
) -> list[dict[str, Any]]:
    center_x, center_y = float(options["center_qx"]), float(options["center_qy"])
    reference = np.deg2rad(float(options.get("reference_axis_deg", 0.0)))
    reference_vector = np.asarray([np.cos(reference), np.sin(reference)], dtype=float)
    arcs: list[dict[str, Any]] = []
    arc_id = 0
    for component in groups:
        if len(component) < 2:
            continue
        ordered = _ordered_component(component, edges, candidates)
        major, direction_confidence, resolved = _component_major_direction(
            ordered,
            candidates,
            center_x=center_x,
            center_y=center_y,
            reference_vector=reference_vector,
        )
        _assign_component_identities(
            ordered,
            candidates,
            options,
            q_step,
            center_x=center_x,
            center_y=center_y,
            major=major,
            resolved=resolved,
        )
        identity_groups = _split_component_identities(ordered, candidates)
        parent_split = len(identity_groups) > 1
        unresolved = [index for index in ordered if _point_identity_key(candidates[index]) is None]
        created: list[dict[str, Any]] = []
        for key in sorted(identity_groups):
            group = identity_groups[key]
            arc = _build_identity_arc(
                arc_id=arc_id,
                ordered=group,
                candidates=candidates,
                options=options,
                major=major,
                direction_confidence=direction_confidence,
                resolved=resolved,
                parent_split=parent_split,
            )
            arcs.append(arc)
            created.append(arc)
            arc_id += 1
        _attach_unresolved_to_nearest_arc(unresolved, created, candidates)
    for point in candidates:
        if int(point.get("arc_id", -1)) < 0:
            branch, quadrant = _quadrant_branch(
                float(point["qx"]),
                float(point["qy"]),
                center_x=center_x,
                center_y=center_y,
                reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
                tolerance_deg=float(options["quadrant_axis_tol_deg"]),
            )
            point["branch_id"], point["quadrant"] = int(branch), quadrant
            if str(point.get("side", "unknown")) not in {"upper", "lower"}:
                point["side"] = _side_from_reference_axis(
                    float(point["qx"]),
                    float(point["qy"]),
                    center_x=center_x,
                    center_y=center_y,
                    reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
                    q_step=q_step,
                )
            extra_flags = ["unconnected_candidate"]
            if point["side"] in {"upper", "lower"}:
                extra_flags.append("side_from_reference_axis")
            point["topology_flag"] = "unconnected_candidate"
            point["topology"] = "unconnected_candidate"
            point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + extra_flags))
    return arcs


def _refresh_arc_identity(arcs: list[dict[str, Any]], points: list[dict[str, Any]]) -> None:
    """Recheck post-profile/edit identities without dropping observed records."""

    points_by_arc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        arc_id = int(point.get("arc_id", -1))
        if arc_id >= 0:
            points_by_arc[arc_id].append(point)
    for arc in arcs:
        members = points_by_arc.get(int(arc.get("arc_id", -1)), [])
        for point in members:
            if _point_identity_key(point) is not None:
                continue
            reason = (
                "side_axis_boundary"
                if int(point.get("branch_id", -1)) in (0, 1)
                else "unresolved_point_identity"
            )
            _demote_unresolved_identity_point(point, reason, "unresolved_point_identity")
        branch_ids = sorted({int(point.get("branch_id", -1)) for point in members})
        sides = sorted({str(point.get("side", "unknown")) for point in members})
        accepted_members = [
            point
            for point in members
            if bool(point.get("accepted", False)) and bool(point.get("valid", False))
        ]
        accepted_keys = [_point_identity_key(point) for point in accepted_members]
        known_accepted = [key for key in accepted_keys if key is not None]
        accepted_branch_ids = sorted({key[0] for key in known_accepted})
        accepted_sides = sorted({key[1] for key in known_accepted})
        branch_resolved = len(accepted_branch_ids) == 1 and accepted_branch_ids[0] in (0, 1)
        side_resolved = len(accepted_sides) == 1 and accepted_sides[0] in {"upper", "lower"}
        identity_resolved = bool(branch_resolved and side_resolved and known_accepted)
        identity_flags: list[str] = []
        if not branch_resolved:
            identity_flags.append("unresolved_or_mixed_branch")
        if not side_resolved:
            identity_flags.append("unresolved_or_mixed_side")
        arc["branch_ids"] = branch_ids
        arc["sides"] = sides
        arc["accepted_branch_ids"] = accepted_branch_ids
        arc["accepted_sides"] = accepted_sides
        arc["identity_resolved"] = identity_resolved
        arc["identity_flags"] = identity_flags
        arc["topology_flags"] = list(dict.fromkeys(list(arc.get("topology_flags", [])) + identity_flags))
        if identity_resolved:
            continue
        # Only mixed *known* identities reject the remaining accepted points.
        # Unresolved tip/axis points were already demoted above.
        if known_accepted and not identity_resolved:
            arc["valid"] = False
            arc["reason"] = "unresolved_or_mixed_branch_or_side"
            for point in accepted_members:
                if _point_identity_key(point) is None:
                    continue
                point["accepted"] = False
                point["reason"] = "unresolved_arc_identity"
                point["topology_flags"] = list(
                    dict.fromkeys(list(point.get("topology_flags", [])) + identity_flags + ["unresolved_arc_identity"])
                )
        elif not known_accepted:
            arc["valid"] = False
            if not arc.get("reason"):
                arc["reason"] = "unresolved_or_mixed_branch_or_side"


def _apply_seeds(
    points: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    seed_actions: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
    q_step: float,
) -> list[dict[str, Any]]:
    """Use seeds to select coherent supported graph components.

    Seeds are UI priors over *observed* graph components.  They cannot create
    points or promote a candidate whose local profile/validity gate failed.
    Auto seeds choose the nearest strongest supported component; competing
    components remain in the output with an explicit rejection reason.
    """

    arc_by_id = {int(arc["arc_id"]): arc for arc in arcs}
    points_by_arc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        arc_id = int(point.get("arc_id", -1))
        if arc_id >= 0:
            points_by_arc[arc_id].append(point)

    def prior_values(action: Mapping[str, Any]) -> tuple[int, str, str]:
        raw_branch = action.get("branch_id", -1)
        try:
            branch = int(raw_branch) if raw_branch is not None else -1
        except (TypeError, ValueError):
            branch = -1
        if branch not in {0, 1}:
            branch = -1
        raw_side = str(action.get("side", "unknown") or "unknown").strip().lower()
        side = raw_side if raw_side in {"upper", "lower"} else "unknown"
        mode = "explicit_prior" if branch in {0, 1} or side != "unknown" else "auto"
        return branch, side, mode

    def arc_conflict(
        arc_points: Sequence[Mapping[str, Any]],
        branch_prior: int,
        side_prior: str,
    ) -> str | None:
        accepted_points = [point for point in arc_points if bool(point.get("accepted", False)) and bool(point.get("valid", False))]
        known_branches = {int(point.get("branch_id", -1)) for point in accepted_points if int(point.get("branch_id", -1)) in {0, 1}}
        known_sides = {str(point.get("side", "unknown")) for point in accepted_points if str(point.get("side", "unknown")) in {"upper", "lower"}}
        if branch_prior in {0, 1} and known_branches and known_branches != {branch_prior}:
            return "seed_branch_conflict"
        if side_prior in {"upper", "lower"} and known_sides and known_sides != {side_prior}:
            return "seed_side_conflict"
        return None

    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for seed_index, action in enumerate(seed_actions):
        try:
            qx, qy = float(action["qx"]), float(action["qy"])
        except (KeyError, TypeError, ValueError):
            records.append({"matched": False, "seed_index": seed_index, "reason": "invalid_seed"})
            continue
        branch_prior, side_prior, mode = prior_values(action)
        nearby_by_arc: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        rejected_nearby: list[dict[str, Any]] = []
        for point in points:
            if not np.isfinite(point.get("qx", np.nan)) or not np.isfinite(point.get("qy", np.nan)):
                continue
            distance = math.hypot(float(point["qx"]) - qx, float(point["qy"]) - qy)
            tolerance = float(options["seed_snap_factor"]) * max(q_step, float(point.get("q_normal_step", q_step)))
            if not np.isfinite(distance) or distance > tolerance:
                continue
            if not bool(point.get("valid", False)) or not bool(point.get("accepted", False)) or int(point.get("arc_id", -1)) < 0:
                rejected_nearby.append({"point_id": str(point.get("point_id", "")), "arc_id": int(point.get("arc_id", -1)), "reason": "seed_candidate_failed_profile_or_validity"})
                continue
            if side_prior == "unknown":
                arc_state = arc_by_id.get(int(point.get("arc_id", -1)), {})
                known_sides = {
                    str(value)
                    for value in arc_state.get("accepted_sides", arc_state.get("sides", []))
                    if str(value) in {"upper", "lower"}
                }
                if len(known_sides) != 1:
                    rejected_nearby.append({"point_id": str(point.get("point_id", "")), "arc_id": int(point.get("arc_id", -1)), "reason": "seed_candidate_unresolved_arc_identity"})
                    continue
            nearby_by_arc[int(point["arc_id"])].append((distance, point))
        if not nearby_by_arc:
            reason = "seed_no_supported_observation" if rejected_nearby else "seed_outside_snap_scale"
            records.append({"matched": False, "seed_index": seed_index, "qx": qx, "qy": qy, "reason": reason, "rejected_candidates": rejected_nearby})
            continue
        compatible: list[tuple[tuple[float, float, float, float], int, str | None]] = []
        conflicts: list[dict[str, Any]] = []
        for arc_id, candidates in nearby_by_arc.items():
            arc_points = points_by_arc.get(arc_id, [])
            conflict = arc_conflict(arc_points, branch_prior, side_prior)
            if conflict is not None:
                conflicts.append({"arc_id": arc_id, "reason": conflict})
                continue
            nearest_distance, nearest_point = min(candidates, key=lambda item: (item[0], -float(item[1].get("snr", 0.0))))
            arc = arc_by_id.get(arc_id, {})
            support = float(arc.get("accepted_points", 0))
            stability = float(arc.get("scale_stability", 0.0))
            compatible.append(((nearest_distance, -support, -stability, -float(nearest_point.get("snr", 0.0))), arc_id, None))
        if not compatible:
            records.append({"matched": False, "seed_index": seed_index, "qx": qx, "qy": qy, "reason": conflicts[0]["reason"] if conflicts else "seed_prior_conflict", "conflicts": conflicts, "rejected_candidates": rejected_nearby})
            continue
        _rank, selected_arc_id, _ = min(compatible, key=lambda item: item[0])
        selected_candidates = nearby_by_arc[selected_arc_id]
        distance, snapped = min(selected_candidates, key=lambda item: (item[0], -float(item[1].get("snr", 0.0))))
        selected_points = points_by_arc.get(selected_arc_id, [])
        competing_arc_ids = sorted(arc_id for _score, arc_id, _reason in compatible if arc_id != selected_arc_id)
        selection = {
            "seed_index": seed_index,
            "selected_arc_id": int(selected_arc_id),
            "selected_points": selected_points,
            "competing_arc_ids": competing_arc_ids,
            "branch_prior": branch_prior,
            "side_prior": side_prior,
            "mode": mode,
        }
        selections.append(selection)
        records.append({
            "matched": True,
            "seed_index": seed_index,
            "qx": qx,
            "qy": qy,
            "point_id": str(snapped["point_id"]),
            "distance_q": float(distance),
            "arc_id": int(selected_arc_id),
            "selected_arc_id": int(selected_arc_id),
            "affected_point_ids": [str(point["point_id"]) for point in selected_points],
            "competing_arc_ids": competing_arc_ids,
            "competing_point_ids": [
                str(point["point_id"])
                for competing_arc_id in competing_arc_ids
                for point in points_by_arc.get(int(competing_arc_id), [])
            ],
            "branch_prior": branch_prior,
            "side_prior": side_prior,
            "selection_mode": mode,
            "manual_prior_provenance": "manual_seed_branch_side_prior" if mode == "explicit_prior" else "manual_seed_auto_component",
            "rejected_candidates": rejected_nearby,
            "reason": "selected_supported_observed_arc",
        })

    selected_arc_ids = {int(selection["selected_arc_id"]) for selection in selections}
    for selection in selections:
        arc_id = int(selection["selected_arc_id"])
        arc = arc_by_id.get(arc_id)
        if arc is not None:
            arc["seed_selected"] = True
            arc["seed_selection_mode"] = selection["mode"]
            arc["seed_provenance"] = "manual_seed_branch_side_prior" if selection["mode"] == "explicit_prior" else "manual_seed_auto_component"
            arc["seed_prior_branch_id"] = selection["branch_prior"]
            arc["seed_prior_side"] = selection["side_prior"]
        for point in selection["selected_points"]:
            point["seeded"] = True
            point["seed_selected"] = True
            point["seed_selection_status"] = "selected_observed_arc"
            point["seed_provenance"] = "manual_seed_branch_side_prior" if selection["mode"] == "explicit_prior" else "manual_seed_auto_component"
            point["seed_ids"] = list(dict.fromkeys(list(point.get("seed_ids", [])) + [selection["seed_index"]]))
            point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["seed_selected_observed_arc"]))
            if not bool(point.get("accepted", False)) or not bool(point.get("valid", False)):
                continue
            branch_prior = int(selection["branch_prior"])
            if branch_prior in {0, 1}:
                point["branch_id"] = branch_prior
                point["seed_branch_prior_applied"] = True
            side_prior = str(selection["side_prior"])
            if side_prior in {"upper", "lower"}:
                point["side"] = side_prior
                point["seed_side_prior_applied"] = True
        if arc is not None:
            accepted_arc_points = [point for point in selection["selected_points"] if bool(point.get("accepted", False)) and bool(point.get("valid", False))]
            arc["branch_ids"] = sorted({int(point.get("branch_id", -1)) for point in accepted_arc_points})
            arc["sides"] = sorted({str(point.get("side", "unknown")) for point in accepted_arc_points})
            arc["seed_coherent"] = bool(
                (selection["branch_prior"] in {-1, 0, 1})
                and (selection["side_prior"] in {"unknown", "upper", "lower"})
                and (selection["side_prior"] == "unknown" or set(arc["sides"]) <= {selection["side_prior"]})
            )
            arc["seed_affected_point_ids"] = [str(point["point_id"]) for point in selection["selected_points"]]

    # A selected seed makes a competing supported fragment explicitly
    # ineligible for this seeded trajectory while retaining every point and
    # its original acceptance state for review/correction.
    for selection in selections:
        for competing_arc_id in selection["competing_arc_ids"]:
            if int(competing_arc_id) in selected_arc_ids:
                continue
            arc = arc_by_id.get(int(competing_arc_id))
            if arc is not None:
                arc["seed_competing"] = True
                arc["seed_selection_reason"] = "competing_seed_component"
            for point in points_by_arc.get(int(competing_arc_id), []):
                point["seed_competing"] = True
                point["seed_selection_status"] = "competing_seed_component"
                point["seed_selection_reason"] = "competing_seed_component"
                point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["seed_competing_component"]))
                if bool(point.get("accepted", False)):
                    point["seed_original_reason"] = point.get("reason", "accepted")
                    point["accepted"] = False
                    point["reason"] = "competing_seed_component"
    for arc in arcs:
        arc_points = points_by_arc.get(int(arc["arc_id"]), [])
        arc["accepted_points_after_seed"] = int(sum(bool(point.get("accepted", False)) for point in arc_points))
        arc["valid_after_seed"] = bool(arc["accepted_points_after_seed"] >= int(options.get("min_arc_points", 2)))
    return records


def _public_point(point: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in point.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    result.setdefault("branch_id", -1)
    result.setdefault("quadrant", "unknown")
    result.setdefault("arc_id", -1)
    result.setdefault("side", "unknown")
    result.setdefault("accepted", False)
    result.setdefault("valid", False)
    result.setdefault("reason", "unknown")
    result.setdefault("topology_flags", [])
    result.setdefault("topology_flag", "unknown")
    result.setdefault("topology", result["topology_flag"])
    result.setdefault("scale_stability", float("nan"))
    result.setdefault("scale_stable", False)
    result.setdefault("normal_fwhm_q", float("nan"))
    result.setdefault("localization_sigma_q", float("nan"))
    result.setdefault("sampling_sigma_q", float("nan"))
    result.setdefault("profile_refinement_applied", False)
    result.setdefault("profile_shift_q", 0.0)
    result.setdefault("profile_shift_pixel", 0.0)
    result.setdefault("profile_refinement_reason", "not_attempted")
    return result
