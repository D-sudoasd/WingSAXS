"""Evidence-preserving plots for ellipse-derived lamellar morphology reports.

These figures show the measured q-space directions alongside the fitted
ellipse and its conditional directional-period calculations.  The ellipse is
an apparent reciprocal-space geometry; the rendering does not infer a unique
three-dimensional lamellar structure.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .lamellar_render import render_lamellar_combined
from .lamellar_utils import _q_scale_to_nm
from .report_figures import (
    _GRAY,
    _ORANGE,
    _BLUE,
    _attach_agg,
    _canonical_q_unit,
    _float,
    _format_list,
    _q_label,
    _reserve_figure_footer,
    _rows,
    _save,
    _styles,
    _time_axis,
)


_BRANCH_COLORS = (_BLUE, _ORANGE, "#009E73", "#CC79A7", "#666666")
_CANDIDATE = "#D55E00"
_GEOMETRY_PARAMETERS = (
    ("a", "Ellipse semi-major axis a"),
    ("b", "Ellipse semi-minor axis b"),
    ("axis_ratio", "Ellipse axis ratio b/a"),
    ("ellipse_axis_tilt_deg", "Geometric ellipse-axis angle theta"),
    ("theta_deg", "Geometric ellipse-axis angle theta"),
    ("Ln_from_minor_axis_nm", "Conditional normal period Lₙ"),
    ("L_from_major_axis_nm", "Conditional major-axis period"),
    ("Lz_from_draw_axis_nm", "Conditional draw-axis period Lz"),
)


def _branch_color(branch_id: Any) -> str:
    try:
        return _BRANCH_COLORS[int(branch_id) % len(_BRANCH_COLORS)]
    except (TypeError, ValueError, OverflowError):
        return _BLUE


def _branch_key(value: Any) -> str:
    return "unlabeled" if value is None or str(value).strip().casefold() in {"", "none", "nan"} else str(value)


def _branch_label(branch: str) -> str:
    return "branch unspecified" if branch == "unlabeled" else f"branch {branch}"


def _display_unit(unit: Any) -> str:
    text = str(unit or "").strip()
    canonical = _canonical_q_unit(text)
    if canonical == "nm^-1":
        return "nm^-1"
    if canonical == "Å^-1":
        return "angstrom^-1"
    return text


def _status_kind(status: Any) -> str:
    """Classify plot styling without replacing the source status label."""

    value = str(status or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if "mixed" in value:
        return "mixed"
    if any(token in value for token in ("candidate", "provisional", "undetermined")):
        return "candidate"
    if any(token in value for token in ("failed", "rejected", "invalid", "unsupported")):
        return "failed"
    return "observed"


def _q_radius_in_unit(value: Any, source_unit: Any, target_unit: Any) -> tuple[float, bool]:
    """Convert a q radius into the analysis display unit when comparable."""

    radius = _float(value)
    if not np.isfinite(radius):
        return float("nan"), False
    source = str(source_unit or "").strip()
    target = str(target_unit or "").strip()
    source_scale = _q_scale_to_nm(source)
    target_scale = _q_scale_to_nm(target)
    if source_scale is not None and target_scale is not None:
        converted = radius * source_scale / target_scale
        return (converted, True) if np.isfinite(converted) else (float("nan"), False)
    unknown = {"", "unknown", "none", "null", "unspecified"}
    if source.casefold() not in unknown and target.casefold() not in unknown:
        if _canonical_q_unit(source) == _canonical_q_unit(target):
            return radius, True
    return float("nan"), False


def _curve_has_observed_support(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        value = row.get("observed_support")
        if isinstance(value, Mapping):
            value = value.get("support_count", value.get("count", value.get("supported")))
        if isinstance(value, str):
            if value.strip().casefold() in {"true", "yes", "supported", "observed"}:
                return True
            try:
                value = float(value)
            except ValueError:
                continue
        if isinstance(value, (bool, np.bool_)) and bool(value):
            return True
        if _float(row.get("support_count")) > 0 or _float(value) > 0:
            return True
    return False


def _support_source(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        value = row.get("support_source")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _support_phrase(source: Any, present: bool) -> str:
    normalized = str(source or "").casefold().replace("-", "_")
    if "radial" in normalized or "lobe" in normalized:
        return "radial-peak support present" if present else "no radial-peak support"
    if "ridge" in normalized:
        return "retained ridge directions present" if present else "no retained ridge directions"
    return "measured support present" if present else "no measured support"


def _is_candidate(row: Mapping[str, Any]) -> bool:
    status_kind = _status_kind(row.get("status"))
    if status_kind == "candidate":
        return True
    has_formal_value = any(np.isfinite(_float(row.get(name))) for name in ("value", "estimate", "parameter_value"))
    if has_formal_value:
        return False
    return np.isfinite(_float(row.get("candidate_value")))


def _value(row: Mapping[str, Any], *, candidate_ok: bool = True) -> float:
    for name in ("value", "estimate", "parameter_value"):
        number = _float(row.get(name))
        if np.isfinite(number):
            return number
    if candidate_ok:
        return _float(row.get("candidate_value"))
    return float("nan")


def _parameter_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _rows(analysis.get("parameter_rows"))


def _curve_records(value: Any) -> list[dict[str, Any]]:
    """Expand per-branch arrays without treating model values as observations."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        # Column-oriented arrays are accepted alongside a single curve record.
        if "angle_deg" in value or "q_radius" in value:
            candidates = [dict(value)]
        else:
            candidates = []
            for branch, curve in value.items():
                if isinstance(curve, Mapping):
                    item = dict(curve)
                    item.setdefault("branch_id", branch)
                    candidates.append(item)
                else:
                    candidates.extend({"branch_id": branch, **row} for row in _rows(curve))
    else:
        try:
            candidates = [dict(row) for row in value if isinstance(row, Mapping)]
        except TypeError:
            candidates = []

    expanded: list[dict[str, Any]] = []
    for curve in candidates:
        angles = curve.get("angle_deg")
        radii = curve.get("q_radius")
        try:
            angle_array = np.asarray(angles, dtype=float)
            radius_array = np.asarray(radii, dtype=float)
        except (TypeError, ValueError):
            continue
        if angle_array.ndim == 0 and radius_array.ndim == 0:
            expanded.append(curve)
            continue
        try:
            angle_array, radius_array = np.broadcast_arrays(angle_array, radius_array)
        except ValueError:
            continue
        status = curve.get("status")
        supported = curve.get("supported")
        try:
            support_array = np.broadcast_to(np.asarray(supported, dtype=bool), angle_array.shape)
        except (TypeError, ValueError):
            support_array = np.zeros(angle_array.shape, dtype=bool)
        support_count = curve.get("observed_support_count")
        try:
            support_count_array = np.broadcast_to(np.asarray(support_count, dtype=float), angle_array.shape)
        except (TypeError, ValueError):
            support_count_array = np.zeros(angle_array.shape, dtype=float)
        for angle, radius, is_supported, count in zip(
            angle_array.reshape(-1),
            radius_array.reshape(-1),
            support_array.reshape(-1),
            support_count_array.reshape(-1),
        ):
            expanded.append(
                {
                    **curve,
                    "angle_deg": float(angle),
                    "q_radius": float(radius),
                    "status": status,
                    "supported": bool(is_supported),
                    "observed_support_count": float(count),
                }
            )
    return expanded


def _valid_vertices(scene: Any) -> bool:
    status = str(getattr(scene, "status", "") or "").casefold()
    if status in {"unavailable", "stale", "failed", "error"}:
        return False
    try:
        vertices = np.asarray(getattr(scene, "vertices"), dtype=float)
    except (TypeError, ValueError, AttributeError):
        return False
    return vertices.ndim == 3 and vertices.shape[1:] == (8, 3) and len(vertices) > 0 and bool(np.all(np.isfinite(vertices)))


def _masked_observed(observed: Any, valid_mask: Any | None) -> Any:
    if observed is None or valid_mask is None:
        return observed
    image = np.asarray(observed)
    mask = np.asarray(valid_mask, dtype=bool)
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("valid_mask must have the same two-dimensional shape as observed")
    result = np.asarray(image, dtype=float).copy()
    result[~mask] = np.nan
    return result


def _analysis_text(analysis: Mapping[str, Any]) -> list[str]:
    rows = _parameter_rows(analysis)
    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_name[str(row.get("parameter", ""))].append(row)

    geometry = analysis.get("ellipse")
    ellipse = geometry if isinstance(geometry, Mapping) else {}
    for name in ("a", "b", "axis_ratio", "ellipse_axis_tilt_deg", "theta_deg"):
        if by_name.get(name):
            continue
        fallback_names = ("theta_deg", "ellipse_axis_tilt_deg") if name in {"theta_deg", "ellipse_axis_tilt_deg"} else (name,)
        for fallback in fallback_names:
            if fallback in ellipse:
                by_name[name].append({"parameter": name, "value": ellipse[fallback], "unit": ellipse.get("q_unit" if name in {"a", "b"} else "unit", "")})
                break

    lines: list[str] = []
    used: set[str] = set()
    for parameter, label in _GEOMETRY_PARAMETERS:
        if parameter in used:
            continue
        if parameter in {"ellipse_axis_tilt_deg", "theta_deg"} and used.intersection({"ellipse_axis_tilt_deg", "theta_deg"}):
            continue
        matching = by_name.get(parameter, [])
        if not matching:
            continue
        # Preserve branch-specific conditional periods when the analysis reports them.
        items = matching if parameter in {"Ln_from_minor_axis_nm", "L_from_major_axis_nm", "Lz_from_draw_axis_nm"} else matching[:1]
        for row in items:
            number = _value(row)
            if not np.isfinite(number):
                continue
            unit = _display_unit(row.get("unit", ""))
            branch = row.get("branch_id")
            suffix = f" · branch {branch}" if branch is not None and parameter in {"Ln_from_minor_axis_nm", "L_from_major_axis_nm", "Lz_from_draw_axis_nm"} else ""
            qualifier = "candidate " if _is_candidate(row) else ""
            lines.append(f"{qualifier}{label}{suffix}: {number:.5g}{f' {unit}' if unit else ''}")
        used.add(parameter)
        if parameter in {"ellipse_axis_tilt_deg", "theta_deg"}:
            used.update({"ellipse_axis_tilt_deg", "theta_deg"})

    source_status = str(analysis.get("source_status", analysis.get("status", "unspecified")) or "unspecified")
    source_reason = str(analysis.get("source_reason", "") or "").strip()
    lines.insert(0, f"Analysis status: {source_status}")
    if source_reason:
        lines.append(f"Reason: {source_reason}")
    if len(lines) == 1:
        lines.append("No finite ellipse parameter estimate is available.")
    return lines


def _period_figure(analysis: Mapping[str, Any], *, title: str) -> Figure:
    q_unit = str(analysis.get("q_unit", "unknown") or "unknown")
    q_scale = _q_scale_to_nm(q_unit)
    physical_q = q_scale is not None
    fig = _attach_agg(Figure(figsize=(12.0, 8.0), dpi=160, layout="constrained", facecolor="white"))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.1, 1.0), height_ratios=(0.95, 1.05), hspace=0.12, wspace=0.15)
    polar = fig.add_subplot(grid[:, 0], projection="polar")
    text_ax = fig.add_subplot(grid[0, 1])
    period_ax = fig.add_subplot(grid[1, 1])

    polar.set_theta_zero_location("E")
    polar.set_theta_direction(1)
    polar.set_thetagrids(np.arange(0, 360, 45))
    polar.set_rlabel_position(135)
    polar.set_title(f"Fitted q-ellipse and retained ridge directions\n{_q_label(q_unit)}", pad=18)

    analysis_status = str(analysis.get("source_status", analysis.get("status", "")) or "").casefold()
    analysis_candidate = _status_kind(analysis_status) == "candidate"
    model_curves = _curve_records(analysis.get("model_curves"))
    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unit_notes: dict[tuple[str, str], int] = defaultdict(int)
    for row in model_curves:
        angle = _float(row.get("angle_deg"))
        source_unit = row.get("q_unit", q_unit)
        radius, comparable = _q_radius_in_unit(row.get("q_radius"), source_unit, q_unit)
        if np.isfinite(_float(row.get("q_radius"))) and not comparable:
            unit_notes[("model q-ellipse samples", str(source_unit or "unknown"))] += 1
            continue
        if np.isfinite(angle) and np.isfinite(radius) and radius > 0:
            by_branch[_branch_key(row.get("branch_id"))].append({**row, "q_radius": radius, "q_unit": q_unit})

    for branch, rows in by_branch.items():
        rows.sort(key=lambda item: _float(item.get("angle_deg")))
        angles = np.asarray([_float(item.get("angle_deg")) for item in rows], dtype=float)
        radii = np.asarray([_float(item.get("q_radius")) for item in rows], dtype=float)
        radians = np.deg2rad(angles)
        color = _branch_color(branch)
        branch_label = _branch_label(branch)
        has_observed_support = _curve_has_observed_support(rows)
        support_label = _support_phrase(_support_source(rows), has_observed_support)
        candidate_model = analysis_candidate or any(_status_kind(row.get("status")) == "candidate" for row in rows)
        model_label = "candidate ellipse model" if candidate_model else "ellipse model"
        label = f"{model_label} · {branch_label} · {support_label}"
        polar.plot(radians, radii, color=color, lw=1.35, ls="--", alpha=0.85, label=label)

    observed_rows = _rows(analysis.get("observed_directions"))
    observation_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observed_rows:
        angle = _float(row.get("angle_deg"))
        source_unit = row.get("q_unit", q_unit)
        radius, comparable = _q_radius_in_unit(row.get("q_radius"), source_unit, q_unit)
        if np.isfinite(_float(row.get("q_radius"))) and not comparable:
            unit_notes[("observed ridge points", str(source_unit or "unknown"))] += 1
            continue
        if np.isfinite(angle) and np.isfinite(radius) and radius > 0:
            status = str(row.get("status", "measured") or "measured").casefold()
            observation_groups[(_branch_key(row.get("branch_id")), status)].append({**row, "q_radius": radius, "q_unit": q_unit})

    for (branch, status), rows in observation_groups.items():
        angles = np.deg2rad(np.asarray([_float(row.get("angle_deg")) for row in rows], dtype=float))
        radii = np.asarray([_float(row.get("q_radius")) for row in rows], dtype=float)
        counts = np.asarray([max(_float(row.get("support_count")), 0.0) for row in rows], dtype=float)
        sizes = 28 + 5 * np.sqrt(counts)
        kind = _status_kind(status)
        candidate = kind == "candidate"
        failed = kind == "failed"
        mixed = kind == "mixed"
        marker = "D" if candidate else "s" if mixed else "x" if failed else "o"
        color = "#000000" if failed else "#7A5195" if mixed else _branch_color(branch)
        polar.scatter(
            angles,
            radii,
            s=sizes,
            marker=marker,
            color=color if not (candidate or mixed) else "none",
            edgecolors=_CANDIDATE if candidate else color,
            linewidths=1.25,
            label=f"Retained observed ridge · {_branch_label(branch)} · status: {status}",
            zorder=6,
        )

    radial_peak_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not observed_rows:
        for row in _rows(analysis.get("observed_radial_peaks")):
            angle = _float(row.get("angle_deg"))
            source_unit = row.get("q_unit", q_unit)
            source_radius = row.get("q")
            radius, comparable = _q_radius_in_unit(source_radius, source_unit, q_unit)
            if not comparable:
                # Some envelopes preserve a separately converted physical q even
                # when the original peak coordinate is in detector units.
                radius, comparable = _q_radius_in_unit(row.get("q_nm_inv"), "nm^-1", q_unit)
            if not comparable:
                unit_notes[("observed radial peaks", str(source_unit or "unknown"))] += 1
                continue
            if np.isfinite(angle) and np.isfinite(radius) and radius > 0:
                status = str(row.get("status", "available") or "available").casefold()
                radial_peak_groups[(_branch_key(row.get("branch_id")), status)].append(
                    {**row, "q": radius, "q_unit": q_unit}
                )

    for (branch, status), rows in radial_peak_groups.items():
        kind = _status_kind(status)
        candidate = kind == "candidate"
        color = _CANDIDATE if candidate else "#7A5195" if kind == "mixed" else "#000000" if kind == "failed" else _branch_color(branch)
        polar.scatter(
            np.deg2rad(np.asarray([_float(row.get("angle_deg")) for row in rows], dtype=float)),
            np.asarray([_float(row.get("q")) for row in rows], dtype=float),
            marker="^",
            s=48,
            color="none" if kind in {"candidate", "mixed"} else color,
            edgecolors=color,
            linewidths=1.25,
            label=f"Measured radial peak · {_branch_label(branch)} · status: {status} · lobe_radial_peaks",
            zorder=5,
        )

    if not by_branch and not observation_groups and not radial_peak_groups:
        polar.text(0.5, 0.5, "No finite q-ellipse model or retained ridge point", transform=polar.transAxes, ha="center", va="center", color=_GRAY, wrap=True)
    polar.set_rlabel_position(135)
    polar.grid(alpha=0.22, linewidth=0.65)
    handles, labels = polar.get_legend_handles_labels()
    if handles:
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        polar.legend(unique.values(), unique.keys(), loc="lower left", bbox_to_anchor=(-0.05, -0.04), fontsize=7, frameon=False)

    text_ax.axis("off")
    text_ax.set_title("Ellipse parameters and source status", loc="left", pad=8)
    analysis_lines = _analysis_text(analysis)
    for (kind, source_unit), count in unit_notes.items():
        analysis_lines.append(
            f"Not overlaid: {count} {kind} in q unit {_display_unit(source_unit)!r} are not comparable with {_q_label(q_unit)}; source values remain in the per-frame analysis data."
        )
    text_ax.text(0.01, 0.96, "\n".join(analysis_lines), transform=text_ax.transAxes, ha="left", va="top", fontsize=9, color="#25323d", linespacing=1.45, wrap=True)

    directional_rows = _rows(analysis.get("directional_rows"))
    finite_nm_periods = any(np.isfinite(_float(row.get("period_nm"))) and _float(row.get("period_nm")) > 0 for row in directional_rows)
    finite_relative_periods = any(np.isfinite(_float(row.get("relative_period"))) and _float(row.get("relative_period")) > 0 for row in directional_rows)
    period_unit = "nm" if physical_q and finite_nm_periods else "relative" if finite_relative_periods else "nm" if physical_q else "relative"
    period_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in directional_rows:
        angle = _float(row.get("angle_deg"))
        # Report period in nm only when q itself carries physical inverse-length units.
        period = _float(row.get("period_nm")) if period_unit == "nm" else _float(row.get("relative_period"))
        if np.isfinite(angle) and np.isfinite(period) and period > 0:
            period_groups[_branch_key(row.get("branch_id"))].append({
                "angle_deg": angle,
                "period": period,
                "status": row.get("status"),
                "support_source": row.get("support_source"),
                "observed_support": row.get("observed_support"),
                "support_count": row.get("support_count"),
            })

    has_period = False
    for branch, rows in period_groups.items():
        rows.sort(key=lambda item: item["angle_deg"])
        x = np.asarray([item["angle_deg"] for item in rows], dtype=float)
        y = np.asarray([item["period"] for item in rows], dtype=float)
        color = _branch_color(branch)
        status = str(rows[0].get("status", "") or "").casefold()
        period_support = _curve_has_observed_support(rows)
        support_text = _support_phrase(_support_source(rows), period_support)
        if "no_observed_support" in status:
            model_kind = "candidate model" if analysis_candidate or _status_kind(status) == "candidate" else "model"
            status_label = f"{model_kind} only; {support_text}"
        elif "observed_support" in status:
            model_kind = "candidate model" if _status_kind(status) == "candidate" else "model"
            status_label = f"{model_kind}; {support_text}"
        elif _status_kind(status) == "candidate":
            status_label = "candidate model"
        else:
            status_label = "model-derived"
        period_ax.plot(x, y, color=color, lw=1.45, ls="--", label=f"Ellipse model · {_branch_label(branch)} · {status_label}")
        has_period = True
    period_ax.set_title("Conditional directional period from fitted q ellipse", loc="left", pad=8)
    period_ax.set_xlabel("q direction from +qx (degree)")
    period_ax.set_ylabel("Directional period (nm)" if period_unit == "nm" else "Relative period (dimensionless)")
    angle_values = np.asarray([_float(row.get("angle_deg")) for row in directional_rows], dtype=float)
    angle_values = angle_values[np.isfinite(angle_values)]
    if angle_values.size:
        period_ax.set_xlim(float(np.min(angle_values)), float(np.max(angle_values)))
    else:
        period_ax.set_xlim(0.0, 360.0)
    period_ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    if has_period:
        period_ax.legend(frameon=False, fontsize=7, loc="best")
    else:
        reason = str(analysis.get("source_reason", "") or "No finite directional period is available.")
        period_ax.text(0.5, 0.5, reason, transform=period_ax.transAxes, ha="center", va="center", color=_GRAY, wrap=True)
    _styles(period_ax)

    fig.suptitle(title or "Ellipse-derived lamellar morphology", fontsize=13, fontweight="semibold")
    assumptions = (
        "Theta is the fitted q-space ellipse-axis angle; it is not a lamellar orientation angle.\n"
        "Thickness, in-plane width and stack depth are schematic assumptions.\n"
        "One q-space ellipse does not determine a unique three-dimensional morphology."
    )
    fig.text(0.5, 0.012, assumptions, ha="center", va="bottom", fontsize=7.5, color=_GRAY, wrap=True)
    _reserve_figure_footer(fig, bottom=0.075, top=0.92)
    return fig


def render_lamellar_report_frame(
    output_dir: str | Path,
    *,
    analysis: Mapping[str, Any],
    scene: Any,
    observed: Any,
    qx: Any,
    qy: Any,
    valid_mask: Any | None = None,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 180,
    title: str = "",
) -> dict[str, Path]:
    """Render one frame's q-ellipse/period analysis and available scene figure.

    The combined observed/q-geometry/3-D scene is saved only when the supplied
    scene contains finite geometry.  The periods figure can still report
    measured directions and estimates when a scene is unavailable.
    """

    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be a mapping")
    if isinstance(dpi, bool) or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    output = Path(output_dir)
    fmt = _format_list(formats)
    paths: dict[str, Path] = {}
    if _valid_vertices(scene):
        # A view along the stack normal hides the layer spacing. Use a fixed
        # oblique offset from the first scene normal, without changing geometry.
        camera = {"elev": 25.0, "azim": -60.0}
        normals = np.asarray(getattr(scene, "orientations", []), dtype=float)
        if normals.ndim == 3 and normals.shape[1:] == (3, 3):
            for normal in normals[:, :, 2]:
                if np.all(np.isfinite(normal)) and np.hypot(*normal[:2]) > 1e-12:
                    camera["azim"] = float(np.rad2deg(np.arctan2(normal[1], normal[0])) + 45.0)
                    break
        combined = render_lamellar_combined(
            scene,
            observed=_masked_observed(observed, valid_mask),
            qx=qx,
            qy=qy,
            language="en",
            camera=camera,
        )
        # The source/status banner belongs to the scene renderer; place it below
        # the report title so its evidence labels remain visible.
        if title:
            for text in combined.texts:
                if text.get_position()[1] >= 0.98:
                    text.set_position((0.02, 0.945))
            combined.suptitle(title, y=0.99, fontsize=13, fontweight="semibold")
            combined.subplots_adjust(top=0.84)
        paths.update(_save(combined, output, "lamellar_structure", fmt, int(dpi), description="Observed frame with the supplied LamellarScene projection and three-dimensional view. Scene dimensions are schematic unless independently calibrated."))
        combined.clear()

    period_figure = _period_figure(analysis, title=title)
    paths.update(_save(period_figure, output, "lamellar_periods", fmt, int(dpi), description="Fitted reciprocal-space ellipse trajectories, retained measured ridge directions, and conditional directional periods. Model curves are distinguished from observed support."))
    period_figure.clear()
    return paths


def _frame_summary(frame: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = {
        key: frame.get(key)
        for key in ("frame_index", "frame_id", "time_s", "time", "time_unit", "q_unit", "status")
        if key in frame
    }
    result.setdefault("frame_index", index)
    if result.get("status") is None:
        analysis = frame.get("lamellar_analysis")
        if isinstance(analysis, Mapping):
            result["status"] = analysis.get("source_status", analysis.get("status", "unspecified"))
    return result


def _sequence_parameter_value(analysis: Mapping[str, Any], names: set[str]) -> tuple[float, bool, str]:
    for row in _parameter_rows(analysis):
        if str(row.get("parameter", "")) not in names:
            continue
        value = _value(row)
        if np.isfinite(value):
            return value, _is_candidate(row), str(row.get("status", "estimate") or "estimate")
    ellipse = analysis.get("ellipse")
    if isinstance(ellipse, Mapping):
        for name in names:
            aliases = ("theta_deg", "ellipse_axis_tilt_deg") if name in {"theta_deg", "ellipse_axis_tilt_deg"} else (name,)
            for alias in aliases:
                value = _float(ellipse.get(alias))
                if np.isfinite(value):
                    status = str(ellipse.get("status", "estimate") or "estimate")
                    return value, _status_kind(status) == "candidate", status
    return float("nan"), False, "missing"


def _observed_direction_summaries(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = analysis.get("observed_direction_summary")
    if value is None:
        return []
    if isinstance(value, Mapping) and not any(
        name in value
        for name in ("branch_id", "mean_angle_deg", "axial_mean_deg", "angle_mean_deg", "angle_deg")
    ):
        rows = []
        for branch, item in value.items():
            if isinstance(item, Mapping):
                row = dict(item)
                row.setdefault("branch_id", branch)
                rows.append(row)
        return rows
    return _rows(value)


def _sequence_figure(frames: Sequence[Mapping[str, Any]], *, title: str) -> Figure | None:
    summaries = [_frame_summary(frame, index) for index, frame in enumerate(frames)]
    x, x_label, _ = _time_axis(summaries)
    panel_data: list[dict[str, Any]] = []
    shared_series = (
        ("Conditional normal period Lₙ", {"Ln_from_minor_axis_nm"}, "nm"),
        ("Ellipse axis ratio b/a", {"axis_ratio"}, ""),
        ("Geometric q-ellipse axis angle theta", {"ellipse_axis_tilt_deg", "theta_deg"}, "degree"),
    )
    for label, names, unit in shared_series:
        y = np.full(len(frames), np.nan, dtype=float)
        candidate = np.zeros(len(frames), dtype=bool)
        for index, frame in enumerate(frames):
            analysis = frame.get("lamellar_analysis")
            if not isinstance(analysis, Mapping):
                continue
            value, is_candidate, _status = _sequence_parameter_value(analysis, names)
            # Physical directional periods are meaningful only when this frame's
            # source q coordinate was calibrated to inverse length.
            if unit == "nm" and _q_scale_to_nm(frame.get("q_unit", analysis.get("q_unit"))) is None:
                continue
            y[index] = value
            candidate[index] = is_candidate
        if np.any(np.isfinite(y)):
            panel_data.append({"label": label, "unit": unit, "kind": "parameter", "series": [("", y, candidate)]})

    lz_by_branch: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, frame in enumerate(frames):
        analysis = frame.get("lamellar_analysis")
        if not isinstance(analysis, Mapping):
            continue
        if _q_scale_to_nm(frame.get("q_unit", analysis.get("q_unit"))) is None:
            continue
        for row in _parameter_rows(analysis):
            if str(row.get("parameter", "")) != "Lz_from_draw_axis_nm":
                continue
            branch = _branch_key(row.get("branch_id"))
            if branch not in lz_by_branch:
                lz_by_branch[branch] = (np.full(len(frames), np.nan, dtype=float), np.zeros(len(frames), dtype=bool))
            y, candidate = lz_by_branch[branch]
            value = _value(row)
            if np.isfinite(value):
                y[index] = value
                candidate[index] = _is_candidate(row)
    lz_series = [(branch, y, candidate) for branch, (y, candidate) in lz_by_branch.items() if np.any(np.isfinite(y))]
    if lz_series:
        panel_data.append({"label": "Conditional draw-axis period Lz", "unit": "nm", "kind": "parameter", "series": [(branch, y, candidate) for branch, y, candidate in lz_series]})

    observed_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    has_summaries = False
    for index, frame in enumerate(frames):
        analysis = frame.get("lamellar_analysis")
        if not isinstance(analysis, Mapping):
            continue
        summary_rows = _observed_direction_summaries(analysis)
        if summary_rows:
            has_summaries = True
            for row in summary_rows:
                mean = _float(row.get("axial_mean_deg", row.get("mean_angle_deg", row.get("angle_mean_deg", row.get("angle_deg")))))
                if np.isfinite(mean):
                    branch = _branch_key(row.get("branch_id"))
                    summary_by_branch[branch].append({
                        "time": float(x[index]),
                        "mean": mean,
                        "minimum": _float(row.get("angle_min_deg", row.get("minimum_angle_deg", row.get("min_angle_deg")))),
                        "maximum": _float(row.get("angle_max_deg", row.get("maximum_angle_deg", row.get("max_angle_deg")))),
                        "count": _float(row.get("count", row.get("support_count"))),
                        "status": str(row.get("status", "observed_summary") or "observed_summary"),
                    })
            continue
        for row in _rows(analysis.get("observed_directions")):
            angle = _float(row.get("angle_deg"))
            if np.isfinite(angle):
                observed_by_branch[_branch_key(row.get("branch_id"))].append({"time": float(x[index]), "angle": angle, "status": str(row.get("status", "measured") or "measured")})
    if observed_by_branch or summary_by_branch:
        # This panel stores actual per-frame observations only.  It deliberately
        # has no connecting lines and leaves an unmeasured branch/frame empty.
        label = "Mean observed q-direction angle and range" if has_summaries else "Retained observed q-direction angles"
        panel_data.append({"label": label, "unit": "degree", "kind": "observed", "raw": observed_by_branch, "summaries": summary_by_branch})

    if not panel_data:
        return None
    rows = len(panel_data)
    fig = _attach_agg(Figure(figsize=(9.5, max(3.1, 2.65 * rows)), dpi=160, layout="constrained", facecolor="white"))
    axes = [fig.add_subplot(rows, 1, index + 1) for index in range(rows)]
    for index, (ax, item) in enumerate(zip(axes, panel_data)):
        label, unit, kind = item["label"], item["unit"], item["kind"]
        if kind == "observed":
            for branch, values in item["raw"].items():
                color = _branch_color(branch)
                by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for value in values:
                    by_status[value["status"].casefold()].append(value)
                for status, status_values in by_status.items():
                    values_array = np.asarray([(value["time"], value["angle"]) for value in status_values], dtype=float)
                    status_kind = _status_kind(status)
                    marker = "D" if status_kind == "candidate" else "s" if status_kind == "mixed" else "x" if status_kind == "failed" else "o"
                    status_color = _CANDIDATE if status_kind == "candidate" else "#7A5195" if status_kind == "mixed" else "#000000" if status_kind == "failed" else color
                    label_kind = "Candidate-like" if status_kind == "candidate" else "Mixed-status" if status_kind == "mixed" else "Failed" if status_kind == "failed" else "Observed"
                    ax.scatter(
                        values_array[:, 0],
                        values_array[:, 1],
                        color=status_color if status_kind not in {"candidate", "mixed"} else "none",
                        edgecolors=status_color if status_kind in {"candidate", "mixed", "failed"} else color,
                        marker=marker,
                        s=38 if status_kind in {"candidate", "mixed"} else 28,
                        label=f"{label_kind} ridge · {_branch_label(branch)} · status: {status}",
                        zorder=4 if status_kind != "observed" else 3,
                    )
            for branch, values in item["summaries"].items():
                for value in values:
                    source_status = str(value.get("status", "observed_summary") or "observed_summary")
                    status_kind = _status_kind(source_status)
                    marker = "D" if status_kind == "candidate" else "s" if status_kind == "mixed" else "x" if status_kind == "failed" else "o"
                    color = _CANDIDATE if status_kind == "candidate" else "#7A5195" if status_kind == "mixed" else "#000000" if status_kind == "failed" else _branch_color(branch)
                    marker_face = "none" if status_kind in {"candidate", "mixed"} else color
                    label_kind = "Candidate-like" if status_kind == "candidate" else "Mixed-status" if status_kind == "mixed" else "Failed" if status_kind == "failed" else "Axial"
                    low, high = value["minimum"], value["maximum"]
                    legend_tail = f"{_branch_label(branch)} · status: {source_status} · full directions retained per frame"
                    if np.isfinite(low) and np.isfinite(high):
                        # q directions are axial: 0 and 180 degrees represent
                        # the same unoriented line, so ranges may cross 0/180.
                        lower_error = (value["mean"] - low) % 180.0
                        upper_error = (high - value["mean"]) % 180.0
                        errors = np.asarray([[lower_error], [upper_error]])
                        ax.errorbar(
                            [value["time"]],
                            [value["mean"]],
                            yerr=errors,
                            fmt=marker,
                            color=color,
                            ecolor=color,
                            markerfacecolor=marker_face,
                            markeredgecolor=color,
                            capsize=2,
                            markersize=5,
                            label=f"{label_kind} mean and observed range · {legend_tail}",
                            zorder=4,
                        )
                    else:
                        ax.scatter(
                            [value["time"]],
                            [value["mean"]],
                            color=marker_face,
                            edgecolors=color,
                            marker=marker,
                            s=38 if status_kind in {"candidate", "mixed"} else 30,
                            label=f"{label_kind} mean · {legend_tail}",
                            zorder=4,
                        )
            ax.set_ylabel("degree")
            ax.set_ylim(-180.0, 360.0)
            ax.set_yticks(np.arange(-180.0, 361.0, 90.0))
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            if unique:
                ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7, loc="best")
        else:
            for series_name, yy, candidate in item["series"]:
                xx = x
                finite = np.isfinite(yy)
                actual = finite & ~candidate
                color = _branch_color(series_name) if series_name else _BLUE
                branch_suffix = f" · {_branch_label(series_name)}" if series_name else ""
                if np.any(finite):
                    ax.plot(xx, np.where(finite, yy, np.nan), color=color, lw=1.0, zorder=1)
                if np.any(actual):
                    ax.scatter(xx[actual], yy[actual], color=color, s=27, label=f"Estimate{branch_suffix}", zorder=3)
                if np.any(candidate & finite):
                    ax.scatter(xx[candidate & finite], yy[candidate & finite], facecolors="none", edgecolors=_CANDIDATE, marker="D", s=38, label=f"Candidate value{branch_suffix}", zorder=4)
            ax.set_ylabel(unit or label.rsplit(" ", 1)[-1])
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            if unique:
                ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7, loc="best")
        ax.set_title(label, loc="left", fontsize=10, pad=5)
        ax.set_xlabel(x_label)
        if len(x) > 1:
            finite_x = np.asarray(x, dtype=float)
            finite_x = finite_x[np.isfinite(finite_x)]
            if finite_x.size:
                low, high = float(np.min(finite_x)), float(np.max(finite_x))
                span = high - low
                padding = 0.04 * span if span > 0 else 0.5
                ax.set_xlim(low - padding, high + padding)
        if x_label.startswith("Frame index"):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _styles(ax)
    fig.suptitle(title or "Lamellar morphology across frames", fontsize=13, fontweight="semibold")
    _reserve_figure_footer(fig, bottom=0.035, top=0.94)
    return fig


def render_lamellar_sequence_report(
    output_dir: str | Path,
    *,
    frames: Sequence[Mapping[str, Any]],
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 180,
) -> dict[str, Path]:
    """Render compact time-series plots from per-frame lamellar summaries."""

    if isinstance(dpi, bool) or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    if isinstance(frames, (str, bytes)):
        raise TypeError("each frame must be a mapping")
    frame_rows = list(frames)
    if any(not isinstance(frame, Mapping) for frame in frame_rows):
        raise TypeError("each frame must be a mapping")
    if not frame_rows:
        return {}
    fmt = _format_list(formats)
    figure = _sequence_figure(frame_rows, title="Lamellar morphology through the sequence")
    if figure is None:
        return {}
    paths = _save(figure, Path(output_dir), "lamellar_sequence_morphology", fmt, int(dpi), description="Sequence of ellipse-derived conditional periods, fitted q-ellipse geometry and retained observed ridge directions. Missing frame values remain gaps; candidates are shown separately.")
    figure.clear()
    return paths


__all__ = ["render_lamellar_report_frame", "render_lamellar_sequence_report"]
