"""Compact, evidence-preserving figures for batch analysis reports.

The functions in this module own their Matplotlib ``Figure`` objects and attach
Agg canvases directly.  They do not select a process-wide backend or register
figures with pyplot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_PURPLE = "#CC79A7"
_GRAY = "#666666"
_MISSING = "#E5E7EB"


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _time_parts(row: Mapping[str, Any]) -> tuple[float, str | None]:
    seconds = _float(row.get("time_s"))
    if np.isfinite(seconds):
        return seconds, "s"
    value = _float(row.get("time"))
    if not np.isfinite(value):
        return float("nan"), None
    unit = str(row.get("time_unit") or "").strip()
    return value, unit or None


def _time_label(parts: Sequence[tuple[float, str | None]]) -> str | None:
    if not parts or not all(np.isfinite(value) for value, _unit in parts):
        return None
    units = [unit for _value, unit in parts]
    if all(unit is None for unit in units):
        return "Time (unit unspecified)"
    if units[0] is not None and all(unit == units[0] for unit in units):
        return f"Time ({units[0]})"
    return None


def _is_time_axis(label: str) -> bool:
    return label.startswith("Time (")


def _q_label(unit: Any) -> str:
    text = str(unit or "unknown").strip()
    normalized = text.lower().replace(" ", "")
    if normalized in {"1/nm", "nm^-1", "nm−1", "nm⁻¹", "nm-1"}:
        return r"$q$ (nm$^{-1}$)"
    if normalized in {"1/a", "1/å", "a^-1", "a−1", "å^-1", "å⁻¹", "angstrom^-1"}:
        return r"$q$ (Å$^{-1}$)"
    if normalized in {"pixel", "pixel-q", "pixel_q", "pixelq"}:
        return "q (pixel-q)"
    return f"q ({text})"


def _canonical_q_unit(unit: Any) -> str:
    text = str(unit or "unknown").strip()
    normalized = text.lower().replace(" ", "")
    if normalized in {"1/nm", "nm^-1", "nm−1", "nm⁻¹", "nm-1"}:
        return "nm^-1"
    if normalized in {"1/a", "1/å", "a^-1", "a−1", "å^-1", "å⁻¹", "angstrom^-1"}:
        return "Å^-1"
    if normalized in {"pixel", "pixel-q", "pixel_q", "pixelq"}:
        return "pixel-q"
    return text


def _q_xy_labels(unit: Any) -> tuple[str, str]:
    label = _q_label(unit)
    suffix = label[label.find("(") + 1 : -1] if "(" in label else "unknown"
    return rf"$q_x$ ({suffix})", rf"$q_y$ ({suffix})"


def _rows(value: Any) -> list[dict[str, Any]]:
    """Normalize row records and column-oriented arrays without inventing data."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        if not value:
            return []
        lengths = []
        for column in value.values():
            if isinstance(column, (str, bytes)) or np.isscalar(column):
                continue
            try:
                lengths.append(len(column))
            except TypeError:
                continue
        if not lengths:
            return [dict(value)]
        size = max(lengths)
        output: list[dict[str, Any]] = []
        for index in range(size):
            row: dict[str, Any] = {}
            for key, column in value.items():
                if isinstance(column, (str, bytes)) or np.isscalar(column):
                    row[str(key)] = column
                    continue
                try:
                    row[str(key)] = column[index] if index < len(column) else None
                except (TypeError, IndexError):
                    row[str(key)] = None
            output.append(row)
        return output
    if isinstance(value, (str, bytes)):
        return []
    try:
        return [dict(row) for row in value if isinstance(row, Mapping)]
    except TypeError:
        return []


def _styles(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


def _finite_limits(values: Any, *, percentile: float = 99.5) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not data.size:
        return 0.0, 1.0
    lo = float(np.min(data))
    hi = float(np.percentile(data, percentile))
    if not np.isfinite(hi) or hi <= lo:
        pad = max(abs(lo) * 0.02, 1.0)
        return lo - pad, hi + pad
    return lo, hi


def _intensity_norm(values: Any) -> tuple[Normalize, str]:
    """Use a stated log scale only for strictly positive, broad-range data."""

    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    if not finite.size:
        return Normalize(0.0, 1.0), "linear scale; no finite intensity bins"
    lo, hi = _finite_limits(finite)
    if np.all(finite > 0) and hi / max(float(np.min(finite)), np.finfo(float).tiny) >= 20:
        return (
            LogNorm(vmin=max(float(np.min(finite)), np.finfo(float).tiny), vmax=hi, clip=True),
            "log color scale; upper values clipped at the 99.5th percentile",
        )
    return Normalize(vmin=lo, vmax=hi, clip=True), "linear color scale"


def _format_list(formats: Sequence[str]) -> tuple[str, ...]:
    if isinstance(formats, (str, bytes)):
        formats = (str(formats),)
    cleaned: list[str] = []
    for item in formats:
        fmt = str(item).strip().lower().lstrip(".")
        if fmt not in {"png", "svg", "pdf", "tiff"}:
            raise ValueError("formats must contain only png, svg, pdf, and tiff")
        if fmt not in cleaned:
            cleaned.append(fmt)
    if not cleaned:
        raise ValueError("formats must contain at least one of png, svg, pdf, or tiff")
    return tuple(cleaned)


def _save(
    figure: Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
    *,
    description: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    title = figure._suptitle.get_text() if figure._suptitle is not None else stem
    paths: dict[str, Path] = {}
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        if fmt == "png":
            metadata = {"Title": title, "Description": description, "Software": "butterfly_saxs"}
            figure.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white", metadata=metadata)
        elif fmt == "tiff":
            figure.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
        elif fmt == "svg":
            metadata = {"Title": title, "Description": description, "Creator": "butterfly_saxs"}
            figure.savefig(path, format=fmt, bbox_inches="tight", facecolor="white", metadata=metadata)
        else:
            metadata = {"Title": title, "Subject": description, "Creator": "butterfly_saxs"}
            figure.savefig(path, format=fmt, bbox_inches="tight", facecolor="white", metadata=metadata)
        paths[f"{stem}.{fmt}"] = path
    return paths


def _reserve_figure_footer(figure: Figure, *, bottom: float = 0.045, top: float = 0.96) -> None:
    engine = figure.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(rect=(0.015, bottom, 0.97, top - bottom))


def _normalization_note(label: str) -> str:
    if label.startswith("log color scale"):
        return "Log color scale; values above the 99.5th percentile are clipped."
    if label.startswith("linear color scale"):
        return "Linear color scale."
    return label


def _attach_agg(figure: Figure) -> Figure:
    FigureCanvasAgg(figure)
    return figure


def _masked_mesh(ax: Any, x: np.ndarray, y: np.ndarray, values: np.ndarray, *, cmap: str, norm: Normalize) -> Any:
    color_map = matplotlib.colormaps[cmap].with_extremes(bad=_MISSING)
    return ax.pcolormesh(
        x,
        y,
        np.ma.masked_invalid(values),
        shading="auto",
        cmap=color_map,
        norm=norm,
        rasterized=True,
    )


def _coordinate_mesh(qx: np.ndarray, qy: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_coordinates = np.isfinite(qx) & np.isfinite(qy)
    if not np.any(valid_coordinates):
        raise ValueError("qx/qy contain no finite coordinates")
    if np.all(valid_coordinates):
        return qx, qy, values

    # Invalid coordinate cells are blank.  Nearest finite coordinates keep the
    # curvilinear mesh valid around those cells; their intensities stay masked.
    from scipy.ndimage import distance_transform_edt

    nearest = distance_transform_edt(~valid_coordinates, return_distances=False, return_indices=True)
    safe_x = qx[tuple(nearest)]
    safe_y = qy[tuple(nearest)]
    safe_values = np.where(valid_coordinates, values, np.nan)
    return safe_x, safe_y, safe_values


def _valid_image_inputs(
    image: Any, qx: Any, qy: Any, valid_mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(image, dtype=float)
    x = np.asarray(qx, dtype=float)
    y = np.asarray(qy, dtype=float)
    if observed.ndim != 2 or min(observed.shape, default=0) < 2:
        raise ValueError("image must be a two-dimensional array at least 2 by 2")
    if x.shape != observed.shape or y.shape != observed.shape:
        raise ValueError("qx and qy must have the same shape as image")
    if valid_mask is None:
        valid = np.ones(observed.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).copy()
        if valid.shape != observed.shape:
            raise ValueError("valid_mask must have the same shape as image")
    valid &= np.isfinite(observed)
    x, y, masked_values = _coordinate_mesh(x, y, np.where(valid, observed, np.nan))
    valid &= np.isfinite(x) & np.isfinite(y)
    observed_masked = np.where(valid, masked_values, np.nan)
    return observed, x, y, observed_masked


def _ridge_xy(measurement: Mapping[str, Any], fit_record: Mapping[str, Any] | None) -> np.ndarray | None:
    """Read only ridge coordinates explicitly named as q coordinates."""

    for source in (measurement, fit_record or {}):
        if not isinstance(source, Mapping):
            continue
        for x_key, y_key in (("measured_ridge_qx", "measured_ridge_qy"), ("ridge_qx", "ridge_qy")):
            if x_key in source and y_key in source:
                x = np.asarray(source[x_key], dtype=float).reshape(-1)
                y = np.asarray(source[y_key], dtype=float).reshape(-1)
                if x.shape == y.shape:
                    points = np.column_stack((x, y))
                    return points[np.all(np.isfinite(points), axis=1)]
        for key in ("measured_ridge_q", "ridge_points_q"):
            if key in source:
                points = np.asarray(source[key], dtype=float)
                if points.ndim == 2 and points.shape[1] == 2:
                    return points[np.all(np.isfinite(points), axis=1)]
    return None


def _has_descriptive_sem(rows: list[dict[str, Any]]) -> bool:
    x, mean, sem, _count = _profile_xy(
        rows,
        coordinate="q_center",
        min_key="q_min",
        max_key="q_max",
        mean_key="mean",
    )
    return bool(np.any(np.isfinite(x) & np.isfinite(mean) & np.isfinite(sem)))


def _frame_overview(
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    q_unit: str,
    measurement: Mapping[str, Any],
    fit_record: Mapping[str, Any] | None,
    title: str,
) -> Figure:
    summary = measurement.get("summary", {})
    summary = summary if isinstance(summary, Mapping) else {}
    radial = _rows(measurement.get("radial_rows", ()))
    angular = _rows(measurement.get("angular_rows", ()))
    polar = measurement.get("polar", {})
    polar = polar if isinstance(polar, Mapping) else {}

    fig = _attach_agg(Figure(figsize=(12.2, 7.4), dpi=160, layout="constrained", facecolor="white"))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.12, 1.0, 1.0), height_ratios=(1, 1))
    ax_q = fig.add_subplot(grid[:, 0])
    ax_radial = fig.add_subplot(grid[0, 1])
    ax_angular = fig.add_subplot(grid[1, 1])
    ax_polar = fig.add_subplot(grid[0, 2], projection="polar")
    ax_coverage = fig.add_subplot(grid[1, 2], projection="polar")

    show = np.where(np.isfinite(image), image, np.nan)
    norm, norm_label = _intensity_norm(show)
    color = _masked_mesh(ax_q, qx, qy, show, cmap="cividis", norm=norm)
    ax_q.set_aspect("equal", adjustable="box")
    xlabel, ylabel = _q_xy_labels(q_unit)
    ax_q.set_xlabel(xlabel)
    ax_q.set_ylabel(ylabel)
    ax_q.set_title("Observed intensity")
    fig.colorbar(color, ax=ax_q, shrink=0.78, pad=0.02, label="Intensity (input units)")
    points = _ridge_xy(measurement, fit_record)
    if points is not None and len(points):
        ax_q.scatter(points[:, 0], points[:, 1], s=11, facecolors="none", edgecolors="#F0E442", linewidths=0.75, label="Measured ridge")
        ax_q.legend(frameon=False, fontsize=7, loc="best")

    _plot_radial(ax_radial, radial, q_unit)
    _plot_angular(ax_angular, angular)
    polar_note = _plot_polar(ax_polar, ax_coverage, polar)

    frame_index = summary.get("frame_index")
    status = str(summary.get("status", "")).strip()
    subtitle = []
    if frame_index is not None:
        subtitle.append(f"Frame {frame_index}")
    time_value, time_unit = _time_parts(summary)
    if np.isfinite(time_value):
        subtitle.append(f"t = {time_value:g} {time_unit or 'unit unspecified'}")
    if status:
        subtitle.append(status.replace("_", " "))
    heading = title.strip() or "Frame measurement report"
    if subtitle:
        heading += "  |  " + " · ".join(subtitle)
    fig.suptitle(heading, fontsize=12, fontweight="semibold")
    notes = [f"Image map: {_normalization_note(norm_label)}"]
    if polar_note:
        notes.append(polar_note)
    if _has_descriptive_sem(radial):
        notes.append("Radial band: descriptive within-bin SEM, not fit uncertainty.")
    fig.text(0.5, 0.012, " · ".join(notes), ha="center", va="bottom", fontsize=7, color=_GRAY)
    _reserve_figure_footer(fig, bottom=0.055, top=0.94)
    return fig


def _profile_xy(rows: list[dict[str, Any]], *, coordinate: str, min_key: str, max_key: str, mean_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x: list[float] = []
    y: list[float] = []
    sem: list[float] = []
    count: list[float] = []
    for row in rows:
        center = _float(row.get(coordinate))
        if not np.isfinite(center):
            lower, upper = _float(row.get(min_key)), _float(row.get(max_key))
            center = 0.5 * (lower + upper) if np.isfinite(lower) and np.isfinite(upper) else float("nan")
        mean = _float(row.get(mean_key))
        spread = _float(row.get("sem"))
        n = _float(row.get("count"))
        if not np.isfinite(spread):
            std = _float(row.get("std"))
            if np.isfinite(std) and np.isfinite(n) and n > 0:
                spread = std / np.sqrt(n)
        x.append(center)
        y.append(mean)
        sem.append(spread if np.isfinite(spread) and spread >= 0 else float("nan"))
        count.append(n)
    return np.asarray(x), np.asarray(y), np.asarray(sem), np.asarray(count)


def _plot_radial(ax: Any, rows: list[dict[str, Any]], q_unit: str) -> None:
    x, y, sem, count = _profile_xy(rows, coordinate="q_center", min_key="q_min", max_key="q_max", mean_key="mean")
    valid = np.isfinite(x) & np.isfinite(y)
    if np.any(valid):
        ax.plot(x, y, color=_BLUE, lw=1.5, marker="o", ms=2.7, label="Bin mean")
        band = valid & np.isfinite(sem)
        if np.any(band):
            ax.fill_between(x, y - sem, y + sem, where=band, color=_BLUE, alpha=0.20, linewidth=0, label="Descriptive SEM")
    else:
        ax.text(0.5, 0.5, "No finite radial bins", transform=ax.transAxes, ha="center", va="center", color=_GRAY)
    ax.set_title("Radial mean intensity")
    ax.set_xlabel(_q_label(q_unit))
    ax.set_ylabel("Intensity (input units)")
    _styles(ax)
    if np.any(np.isfinite(sem)):
        ax.legend(frameon=False, fontsize=7, loc="best")


def _plot_angular(ax: Any, rows: list[dict[str, Any]]) -> None:
    x, y, _, _ = _profile_xy(rows, coordinate="angle_center_deg", min_key="angle_min_deg", max_key="angle_max_deg", mean_key="mean")
    valid = np.isfinite(x) & np.isfinite(y)
    if np.any(valid):
        ax.plot(x, y, color=_ORANGE, lw=1.45, marker="o", ms=2.5)
        finite_x = x[np.isfinite(x)]
        if finite_x.size and np.min(finite_x) >= -180 and np.max(finite_x) <= 180:
            ax.set_xlim(-180, 180)
    else:
        ax.text(0.5, 0.5, "No finite angular bins", transform=ax.transAxes, ha="center", va="center", color=_GRAY)
    ax.set_title("Angular mean intensity")
    ax.set_xlabel("Azimuth (degree)")
    ax.set_ylabel("Intensity (input units)")
    _styles(ax)


def _polar_arrays(polar: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_edges = np.asarray(polar.get("q_edges", ()), dtype=float)
    angle_edges = np.asarray(polar.get("angle_edges", ()), dtype=float)
    mean = np.asarray(polar.get("mean", ()), dtype=float)
    coverage = np.asarray(polar.get("coverage", ()), dtype=float)
    count = np.asarray(polar.get("count", ()), dtype=float)
    if q_edges.ndim != 1 or angle_edges.ndim != 1 or len(q_edges) < 2 or len(angle_edges) < 2:
        raise ValueError("polar q_edges and angle_edges must be one-dimensional bin edges")
    if not np.all(np.isfinite(q_edges)) or not np.all(np.diff(q_edges) > 0):
        raise ValueError("polar q_edges must be finite and strictly increasing")
    if not np.all(np.isfinite(angle_edges)) or not np.all(np.diff(angle_edges) > 0):
        raise ValueError("polar angle_edges must be finite and strictly increasing")
    expected = (len(q_edges) - 1, len(angle_edges) - 1)
    if mean.shape != expected:
        raise ValueError(f"polar mean must have shape {expected} (q bins, angle bins)")
    if coverage.size and coverage.shape != expected:
        raise ValueError(f"polar coverage must have shape {expected} (q bins, angle bins)")
    if count.size and count.shape != expected:
        raise ValueError(f"polar count must have shape {expected} (q bins, angle bins)")
    if not coverage.size:
        coverage = np.full(expected, np.nan)
    if not count.size:
        count = np.full(expected, np.nan)
    mean = np.where((count > 0) | ~np.isfinite(count), mean, np.nan)
    return q_edges, angle_edges, mean, coverage, count


def _plot_polar(mean_ax: Any, coverage_ax: Any, polar: Mapping[str, Any]) -> str | None:
    if not polar:
        for ax, message in ((mean_ax, "No polar measurement"), (coverage_ax, "No coverage data")):
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=_GRAY)
            ax.set_axis_off()
        return None
    q_edges, angle_edges, mean, coverage, count = _polar_arrays(polar)
    theta_edges = np.deg2rad(angle_edges)
    mean_norm, mean_norm_label = _intensity_norm(mean)
    mean_cmap = matplotlib.colormaps["cividis"].with_extremes(bad=_MISSING)
    mesh = mean_ax.pcolormesh(theta_edges, q_edges, np.ma.masked_invalid(mean), shading="auto", cmap=mean_cmap, norm=mean_norm, rasterized=True)
    mean_ax.set_theta_zero_location("E")
    mean_ax.set_theta_direction(1)
    mean_ax.set_title("Polar mean intensity", va="bottom")
    mean_ax.set_rlabel_position(24)
    mean_ax.grid(alpha=0.24, linewidth=0.6)
    fig = mean_ax.figure
    fig.colorbar(mesh, ax=mean_ax, shrink=0.72, pad=0.12, label="Mean intensity (input units)")

    finite_coverage = coverage[np.isfinite(coverage)]
    coverage_max = float(np.max(finite_coverage)) if finite_coverage.size else 1.0
    percent = coverage_max > 1.0 + 1e-8
    top = 100.0 if percent else 1.0
    cover_cmap = matplotlib.colormaps["viridis"].with_extremes(bad=_MISSING)
    cover = coverage_ax.pcolormesh(theta_edges, q_edges, np.ma.masked_invalid(coverage), shading="auto", cmap=cover_cmap, norm=Normalize(0, top, clip=True), rasterized=True)
    coverage_ax.set_theta_zero_location("E")
    coverage_ax.set_theta_direction(1)
    coverage_ax.set_title("Observed angular coverage", va="bottom")
    coverage_ax.set_rlabel_position(24)
    coverage_ax.grid(alpha=0.24, linewidth=0.6)
    coverage_label = "Coverage (%)" if percent else "Coverage (fraction)"
    fig.colorbar(cover, ax=coverage_ax, shrink=0.72, pad=0.12, label=coverage_label)
    candidate_count = _float(np.nansum(np.asarray(polar.get("candidate_count", 0), dtype=float)))
    if np.isfinite(candidate_count) and candidate_count > 0:
        coverage_ax.text(0.5, -0.17, f"Candidates: {candidate_count:g}", transform=coverage_ax.transAxes, ha="center", va="top", fontsize=7, color=_GRAY)
    return f"Polar mean: {_normalization_note(mean_norm_label)}"


def _frame_fit_figure(
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    valid_mask: Any | None,
    q_unit: str,
    model: Any,
    residual: Any | None,
    fit_record: Mapping[str, Any] | None,
    title: str,
) -> Figure:
    mod = np.asarray(model, dtype=float)
    if mod.shape != image.shape:
        raise ValueError("model must have the same shape as image")
    # The public full2d worker stores model-minus-observed residuals.  Reports
    # use the conventional observed-minus-model sign consistently; the source
    # residual array is retained separately by the NPZ exporter.
    if residual is not None and np.shape(residual) != image.shape:
        raise ValueError("residual must have the same shape as image")
    residual_array = image - mod
    valid = np.isfinite(image) & np.isfinite(mod) & np.isfinite(residual_array)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError("valid_mask must have the same shape as image")
        valid &= mask
    obs = np.where(valid, image, np.nan)
    pred = np.where(valid, mod, np.nan)
    resid = np.where(valid, residual_array, np.nan)
    safe_x, safe_y, _ = _coordinate_mesh(qx, qy, obs)
    fig = _attach_agg(Figure(figsize=(12.4, 4.4), dpi=160, layout="constrained", facecolor="white"))
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    qx_label, qy_label = _q_xy_labels(q_unit)
    norm, norm_label = _intensity_norm(np.concatenate((obs[np.isfinite(obs)], pred[np.isfinite(pred)])))
    for ax, data, label in zip(axes[:2], (obs, pred), ("Observed intensity", "Supplied full2d model")):
        mesh = _masked_mesh(ax, safe_x, safe_y, data, cmap="cividis", norm=norm)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(qx_label)
        ax.set_ylabel(qy_label)
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax, shrink=0.80, pad=0.02, label=f"Intensity (input units); {norm_label}")
    finite_resid = resid[np.isfinite(resid)]
    limit = float(np.percentile(np.abs(finite_resid), 99.5)) if finite_resid.size else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    residual_cmap = matplotlib.colormaps["PuOr"].with_extremes(bad=_MISSING)
    rmesh = axes[2].pcolormesh(safe_x, safe_y, np.ma.masked_invalid(resid), shading="auto", cmap=residual_cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), rasterized=True)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel(qx_label)
    axes[2].set_ylabel(qy_label)
    axes[2].set_title("Residual (observed − model)")
    fig.colorbar(rmesh, ax=axes[2], shrink=0.80, pad=0.02, label="Intensity difference (input units)")
    heading = title.strip() or "Full2D fit diagnostic"
    status = str((fit_record or {}).get("status", "")).strip()
    if status:
        heading += f"  |  {status.replace('_', ' ')}"
    fig.suptitle(heading, fontsize=12, fontweight="semibold")
    return fig


def _record_rows(value: Any, *, id_key: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        for key in ("rows", "records", "points", "arcs"):
            nested = value.get(key)
            if isinstance(nested, (Mapping, Sequence)) and not isinstance(nested, (str, bytes)):
                return _record_rows(nested, id_key=id_key)
        if any(key in value for key in (id_key, "used", "normal_residual_q", "arc_id", "support_status")):
            return [dict(value)]
        rows = []
        for record_id, row in value.items():
            if isinstance(row, Mapping):
                record = dict(row)
                record.setdefault(id_key, record_id)
                rows.append(record)
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _fit_payloads(fit_record: Mapping[str, Any] | None, measurement: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result: Mapping[str, Any] = fit_record or {}
    nested_result = result.get("result")
    if isinstance(nested_result, Mapping):
        result = nested_result
    observables = result.get("observables", {})
    observables = observables if isinstance(observables, Mapping) else {}
    butterfly_candidates = [
        result.get("butterfly"),
        observables.get("butterfly"),
        measurement.get("butterfly"),
    ]
    butterfly = next((item for item in butterfly_candidates if isinstance(item, Mapping)), {})
    ellipse_candidates = [
        result.get("ellipse_fit"),
        observables.get("ellipse_fit"),
        butterfly.get("candidate_fit"),
    ]
    ellipse = next((item for item in ellipse_candidates if isinstance(item, Mapping)), {})
    point_rows = _record_rows(ellipse.get("point_diagnostics", ()), id_key="point_id")
    arc_rows = _record_rows(ellipse.get("arc_diagnostics", ()), id_key="arc_id")
    if not point_rows:
        point_rows = _record_rows(butterfly.get("point_diagnostics", butterfly.get("points", ())), id_key="point_id")
    if not arc_rows:
        arc_rows = _record_rows(butterfly.get("arc_diagnostics", butterfly.get("arcs", ())), id_key="arc_id")

    source_profiles = butterfly.get("profiles", {})
    profiles: list[dict[str, Any]] = []
    if isinstance(source_profiles, Mapping):
        if any(name in source_profiles for name in ("offset_q", "raw_intensity", "fit_intensity")):
            profiles.append(dict(source_profiles))
        else:
            for point_id, profile in source_profiles.items():
                if isinstance(profile, Mapping):
                    row = dict(profile)
                    row.setdefault("point_id", point_id)
                    profiles.append(row)
    elif isinstance(source_profiles, Sequence) and not isinstance(source_profiles, (str, bytes)):
        profiles = [dict(profile) for profile in source_profiles if isinstance(profile, Mapping)]
    # Point identifiers are stable across the native profile dictionary, the
    # fitted diagnostics, and normal_profiles.csv.  Sorting makes pages stable
    # even when an upstream mapping was assembled in a different order.
    profiles.sort(key=lambda row: str(row.get("point_id", "")))
    return profiles, point_rows, arc_rows


def _evenly_spaced_indices(size: int, maximum: int) -> list[int]:
    if size <= maximum:
        return list(range(size))
    return list(dict.fromkeys(int(value) for value in np.linspace(0, size - 1, maximum)))


def _profile_pages(profiles: list[dict[str, Any]], page_size: int = 8) -> list[list[dict[str, Any]]]:
    if len(profiles) <= page_size:
        return [profiles]
    chosen = _evenly_spaced_indices(len(profiles), page_size)
    representative = [profiles[index] for index in chosen]
    rest = [profile for index, profile in enumerate(profiles) if index not in set(chosen)]
    pages = [representative]
    pages.extend(rest[start : start + page_size] for start in range(0, len(rest), page_size))
    return pages


def _profile_vectors(profile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinate = np.asarray(profile.get("offset_q", ()), dtype=float).reshape(-1)
    raw = np.asarray(profile.get("raw_intensity", ()), dtype=float).reshape(-1)
    fitted = np.asarray(profile.get("fit_intensity", ()), dtype=float).reshape(-1)
    residual = np.asarray(profile.get("residual", ()), dtype=float).reshape(-1)
    return coordinate, raw, fitted, residual


def _has_profile_samples(profile: Mapping[str, Any], *, sample_key: str) -> bool:
    offset, raw, fitted, residual = _profile_vectors(profile)
    samples = {"raw_intensity": raw, "fit_intensity": fitted, "residual": residual}.get(sample_key)
    if samples is None or not offset.size or offset.size != samples.size:
        return False
    return bool(np.any(np.isfinite(offset) & np.isfinite(samples)))


def _profile_page_axes(figure: Figure, profile_count: int) -> list[Any]:
    columns = 2
    rows = (profile_count + columns - 1) // columns
    grid = figure.add_gridspec(rows, columns)
    axes: list[Any] = []
    for index in range(profile_count):
        row, column = divmod(index, columns)
        if profile_count % columns and index == profile_count - 1:
            axes.append(figure.add_subplot(grid[row, :]))
        else:
            axes.append(figure.add_subplot(grid[row, column]))
    return axes


def _local_profile_figure(
    profiles: list[dict[str, Any]],
    page_index: int,
    page_count: int,
    *,
    q_unit: str,
    title: str,
) -> Figure:
    rows = (len(profiles) + 1) // 2
    fig = _attach_agg(Figure(figsize=(11.4, max(3.0, 2.0 * rows + 0.9)), dpi=160, layout="constrained", facecolor="white"))
    axes = _profile_page_axes(fig, len(profiles))
    for panel_index, ax in enumerate(axes):
        profile = profiles[panel_index]
        offset, raw, fitted, _residual = _profile_vectors(profile)
        point_id = str(profile.get("point_id", f"profile-{panel_index}"))
        if offset.size == raw.size and raw.size:
            finite = np.isfinite(offset) & np.isfinite(raw)
            if np.any(finite):
                ax.scatter(offset[finite], raw[finite], s=11, color=_BLUE, label="Observed samples", zorder=3)
        if offset.size == fitted.size and fitted.size:
            finite_fit = np.isfinite(offset) & np.isfinite(fitted)
            if np.any(finite_fit):
                ax.plot(offset, np.where(np.isfinite(fitted), fitted, np.nan), color=_ORANGE, lw=1.2, label="Supplied local fit")
        background = _float(profile.get("background"))
        if np.isfinite(background):
            ax.axhline(background, color=_GRAY, lw=0.8, ls="--", label="Recorded background")
        model_name = str(profile.get("model", "")).strip()
        panel_title = point_id if not model_name else f"{point_id} · {model_name}"
        title_artist = ax.set_title(panel_title, fontsize=8)
        title_artist.set_url("../../normal_profiles.csv")
        ax.set_xlabel(f"Normal offset q ({q_unit})", fontsize=7)
        ax.set_ylabel("Intensity (input units)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.16, linewidth=0.5)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=6, loc="best")
        reason = str(profile.get("reason", "")).strip()
        if reason and reason not in {"accepted", "ok"}:
            ax.text(0.02, 0.02, reason.replace("_", " "), transform=ax.transAxes, fontsize=6, color=_GRAY, va="bottom")
    heading = title.strip() or "Measured local normal profiles"
    fig.suptitle(f"{heading} · page {page_index + 1}/{page_count}", fontsize=12, fontweight="semibold")
    link = fig.text(0.5, 0.012, "Full measured profile samples: normal_profiles.csv; fit record: fit_details.json.", ha="center", va="bottom", fontsize=7, color=_GRAY)
    link.set_url("../../normal_profiles.csv")
    _reserve_figure_footer(fig, bottom=0.08, top=0.94)
    return fig


def _profile_residual_figure(
    profiles: list[dict[str, Any]],
    page_index: int,
    page_count: int,
    *,
    q_unit: str,
    title: str,
) -> Figure:
    rows = (len(profiles) + 1) // 2
    fig = _attach_agg(Figure(figsize=(11.4, max(3.0, 2.0 * rows + 0.9)), dpi=160, layout="constrained", facecolor="white"))
    axes = _profile_page_axes(fig, len(profiles))
    for panel_index, ax in enumerate(axes):
        profile = profiles[panel_index]
        offset, _raw, _fitted, residual = _profile_vectors(profile)
        point_id = str(profile.get("point_id", f"profile-{panel_index}"))
        if offset.size == residual.size and residual.size:
            finite = np.isfinite(offset) & np.isfinite(residual)
            if np.any(finite):
                ax.axhline(0.0, color=_GRAY, lw=0.7)
                ax.plot(offset, np.where(np.isfinite(residual), residual, np.nan), color=_PURPLE, lw=1.0, marker="o", ms=2.1)
            else:
                ax.text(0.5, 0.5, "No finite residual samples", transform=ax.transAxes, ha="center", va="center", color=_GRAY)
        else:
            ax.text(0.5, 0.5, "No stored profile residual", transform=ax.transAxes, ha="center", va="center", color=_GRAY)
        title_artist = ax.set_title(point_id, fontsize=8)
        title_artist.set_url("fit_details.json")
        ax.set_xlabel(f"Normal offset q ({q_unit})", fontsize=7)
        ax.set_ylabel("Observed − local fit (input units)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.16, linewidth=0.5)
    fig.suptitle(f"{title or 'Measured local profile residuals'} · page {page_index + 1}/{page_count}", fontsize=12, fontweight="semibold")
    link = fig.text(0.5, 0.012, "Stored profile residual samples; source fit record: fit_details.json.", ha="center", va="bottom", fontsize=7, color=_GRAY)
    link.set_url("fit_details.json")
    _reserve_figure_footer(fig, bottom=0.08, top=0.94)
    return fig


def _point_x(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    return np.arange(len(rows), dtype=float), [str(row.get("point_id", index)) for index, row in enumerate(rows)]


def _category_ticks(ax: Any, x: np.ndarray, labels: list[str]) -> None:
    if not len(x):
        return
    indices = np.arange(len(x))
    selected = indices if len(indices) <= 10 else np.unique(np.linspace(0, len(indices) - 1, 8).astype(int))
    ax.set_xticks(x[selected])
    ax.set_xticklabels([labels[index] for index in selected], rotation=45, ha="right", fontsize=6)


def _diagnostic_scatter(ax: Any, rows: list[dict[str, Any]], key: str, label: str, color: str, *, unit: str = "", category: str) -> bool:
    x, labels = _point_x(rows)
    values = np.asarray([_float(row.get(key)) for row in rows], dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return False
    used_values = [row.get("used") for row in rows]
    if any(isinstance(item, (bool, np.bool_)) for item in used_values):
        used = np.asarray([item is True or isinstance(item, np.bool_) and bool(item) for item in used_values]) & valid
        unused = (~np.asarray([item is True or isinstance(item, np.bool_) and bool(item) for item in used_values])) & valid
        if np.any(used):
            ax.scatter(x[used], values[used], s=22, color=color, marker="o", label=f"{label} (used)")
        if np.any(unused):
            ax.scatter(x[unused], values[unused], s=22, facecolors="none", edgecolors=color, marker="o", label=f"{label} (not used)")
    else:
        ax.scatter(x[valid], values[valid], s=22, color=color, marker="o", label=label)
    ax.set_xticks([])
    ax.set_xlabel("Point record order; see point IDs in fit_details.json")
    ax.set_ylabel(f"{label}{f' ({unit})' if unit else ''}")
    ax.legend(frameon=False, fontsize=6, loc="best")
    return True


def _arc_scatter(ax: Any, rows: list[dict[str, Any]], key: str, label: str, color: str, *, unit: str = "", labels: list[str] | None = None) -> bool:
    values = np.asarray([_float(row.get(key)) for row in rows], dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return False
    arc_ids = [_float(row.get("arc_id")) for row in rows]
    if arc_ids and all(np.isfinite(value) for value in arc_ids) and len(set(arc_ids)) == len(arc_ids):
        x = np.asarray(arc_ids, dtype=float)
        tick_labels = [str(int(value)) if float(value).is_integer() else f"{value:g}" for value in arc_ids]
    else:
        x = np.arange(len(rows), dtype=float)
        tick_labels = labels if labels is not None else [str(index) for index in range(len(rows))]
    ax.scatter(x[valid], values[valid], s=25, color=color, marker="o", label=label)
    if len(x):
        _category_ticks(ax, x, tick_labels)
    else:
        ax.set_xticks([])
    ax.set_xlabel("Arc ID; full records in fit_details.json")
    ax.set_ylabel(f"{label}{f' ({unit})' if unit else ''}")
    ax.legend(frameon=False, fontsize=6, loc="best")
    return True


def _fit_support_figure(
    point_rows: list[dict[str, Any]],
    arc_rows: list[dict[str, Any]],
    q_unit: str,
    title: str,
) -> Figure:
    fig = _attach_agg(Figure(figsize=(13.0, 7.4), dpi=160, layout="constrained", facecolor="white"))
    axes = [fig.add_subplot(2, 3, index + 1) for index in range(6)]
    q_label = str(q_unit or "unknown")
    point_axes, localization_ax, arc_residual_ax, ratio_ax, endpoint_ax, counts_ax = axes
    point_plotted = False
    for key, label, color in (
        ("normal_residual_q", "Normal residual", _BLUE),
        ("projection_residual_q", "Signed projection residual", _ORANGE),
        ("distance_q", "Projection distance", _GREEN),
    ):
        point_plotted |= _diagnostic_scatter(point_axes, point_rows, key, label, color, unit=q_label, category="point")
    if not point_plotted:
        point_axes.text(0.5, 0.5, "No point residual records", transform=point_axes.transAxes, ha="center", va="center", color=_GRAY)
    point_axes.set_title("Measured ridge residuals and distances")
    point_axes.set_ylabel(f"Residual or distance ({q_label})")

    if not _diagnostic_scatter(localization_ax, point_rows, "localization_sigma_q", "Local profile localization scale", _PURPLE, unit=q_label, category="point"):
        localization_ax.text(0.5, 0.5, "No localization scale reported", transform=localization_ax.transAxes, ha="center", va="center", color=_GRAY)
    localization_ax.set_title("Localization scale")
    localization_ax.set_ylabel(f"Local normal-profile scale ({q_label})")

    arc_plotted = False
    for key, label, color in (
        ("normal_residual_q_rms", "Normal residual RMS", _BLUE),
        ("projection_rmse_q", "Projection RMSE", _ORANGE),
    ):
        arc_plotted |= _arc_scatter(arc_residual_ax, arc_rows, key, label, color, unit=q_label)
    if not arc_plotted:
        arc_residual_ax.text(0.5, 0.5, "No arc residual summary reported", transform=arc_residual_ax.transAxes, ha="center", va="center", color=_GRAY)
    arc_residual_ax.set_title("Arc residuals (q scale)")
    arc_residual_ax.set_ylabel(f"Arc residual ({q_label})")

    if not _arc_scatter(ratio_ax, arc_rows, "normal_residual_localization_ratio_rms", "Residual / localization RMS", _ORANGE, unit="dimensionless"):
        ratio_ax.text(0.5, 0.5, "No residual/localization ratio reported", transform=ratio_ax.transAxes, ha="center", va="center", color=_GRAY)
    ratio_ax.set_title("Residual relative to localization scale")

    support_plotted = False
    for key, label, color in (
        ("support_endpoint_fraction", "Support endpoint fraction", _GREEN),
        ("manual_endpoint_fraction", "Manual endpoint fraction", _PURPLE),
    ):
        support_plotted |= _arc_scatter(endpoint_ax, arc_rows, key, label, color, unit="fraction")
    if not support_plotted:
        endpoint_ax.text(0.5, 0.5, "No endpoint support fractions reported", transform=endpoint_ax.transAxes, ha="center", va="center", color=_GRAY)
    endpoint_ax.set_title("Arc support endpoints")
    endpoint_ax.set_ylabel("Arc support fraction")

    count_plotted = False
    for key, label, color in (
        ("support_gap_count", "Support gaps", _BLUE),
        ("infeasible_projection_count", "Infeasible projections", _ORANGE),
    ):
        count_plotted |= _arc_scatter(counts_ax, arc_rows, key, label, color, unit="count")
    if not count_plotted:
        counts_ax.text(0.5, 0.5, "No support counts reported", transform=counts_ax.transAxes, ha="center", va="center", color=_GRAY)
    counts_ax.set_title("Arc support counts")
    counts_ax.set_ylabel("Count")

    for ax in axes:
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
    link = fig.text(0.5, 0.005, "Recorded point and arc diagnostics are shown by their native scales; full source records: fit_details.json.", ha="center", fontsize=7, color=_GRAY)
    link.set_url("fit_details.json")
    fig.suptitle(title or "Measured ridge residuals and arc support", fontsize=12, fontweight="semibold")
    return fig


def render_frame_report(
    output_dir: str | Path,
    *,
    image: Any,
    qx: Any,
    qy: Any,
    valid_mask: Any | None,
    q_unit: str,
    measurement: Mapping[str, Any],
    fit_record: Mapping[str, Any] | None = None,
    model: Any | None = None,
    residual: Any | None = None,
    title: str = "",
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 160,
) -> dict[str, Path]:
    """Render a compact frame overview and, when supplied, a full2d diagnostic.

    The radial band is shown only where the input measurement provides a finite
    SEM (or a sample standard deviation and count from which descriptive SEM can
    be calculated).  It describes within-bin spread and is not fit uncertainty.
    """

    if not isinstance(measurement, Mapping):
        raise TypeError("measurement must be a mapping")
    if isinstance(dpi, bool) or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    fmt = _format_list(formats)
    observed, x, y, shown = _valid_image_inputs(image, qx, qy, valid_mask)
    output = Path(output_dir)
    summary = measurement.get("summary", {})
    frame_index = summary.get("frame_index") if isinstance(summary, Mapping) else None
    try:
        suffix = f"_{int(frame_index):04d}" if frame_index is not None else ""
    except (TypeError, ValueError, OverflowError):
        suffix = ""
    overview = _frame_overview(shown, x, y, q_unit, measurement, fit_record, title)
    paths = _save(overview, output, f"frame{suffix}_overview", fmt, int(dpi), description="Observed q-space map, radial mean, angular mean, polar mean, and angular support coverage.")
    overview.clear()
    if model is not None:
        fit_figure = _frame_fit_figure(shown, x, y, valid_mask, q_unit, model, residual, fit_record, title)
        paths.update(_save(fit_figure, output, f"frame{suffix}_full2d_fit", fmt, int(dpi), description="Measured intensity, supplied full2d model, and measured-minus-model residual in q space."))
        fit_figure.clear()
    profiles, point_rows, arc_rows = _fit_payloads(fit_record, measurement)
    measured_profiles = [profile for profile in profiles if _has_profile_samples(profile, sample_key="raw_intensity")]
    if measured_profiles:
        pages = _profile_pages(measured_profiles, 8)
        for page_index, page in enumerate(pages):
            stem = f"frame{suffix}_local_profiles_{page_index + 1:03d}"
            profile_figure = _local_profile_figure(page, page_index, len(pages), q_unit=q_unit, title=title)
            ids = ", ".join(str(row.get("point_id", "")) for row in page)
            paths.update(_save(profile_figure, output, stem, fmt, int(dpi), description=f"Supplied observed local normal profiles and supplied profile fits; point IDs {ids}; complete source vectors in normal_profiles.csv and fit_details.json; no refit performed."))
            profile_figure.clear()
    residual_profiles = [profile for profile in profiles if _has_profile_samples(profile, sample_key="residual")]
    if residual_profiles:
        residual_pages = _profile_pages(residual_profiles, 8)
        for page_index, page in enumerate(residual_pages):
            residual_figure = _profile_residual_figure(page, page_index, len(residual_pages), q_unit=q_unit, title=title)
            ids = ", ".join(str(row.get("point_id", "")) for row in page)
            paths.update(_save(residual_figure, output, f"frame{suffix}_local_profile_residuals_{page_index + 1:03d}", fmt, int(dpi), description=f"Stored observed-minus-local-fit profile residuals for point IDs {ids}; no missing residual samples are calculated or filled."))
            residual_figure.clear()
    if point_rows or arc_rows:
        support_figure = _fit_support_figure(point_rows, arc_rows, q_unit, title)
        paths.update(_save(support_figure, output, f"frame{suffix}_fit_support", fmt, int(dpi), description="Recorded point residuals and localization scales with per-arc fit residual and observed-support diagnostics."))
        support_figure.clear()
    return paths


def _time_axis(summaries: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, str, bool]:
    parts = [_time_parts(summary) for summary in summaries]
    label = _time_label(parts)
    if label is not None:
        return np.asarray([value for value, _unit in parts], dtype=float), label, True
    return np.arange(len(summaries), dtype=float), "Frame index (0-based report order; time units mixed or missing)", False


def _edges_from_centers(centers: np.ndarray, *, fallback_width: float = 1.0) -> np.ndarray:
    values = np.asarray(centers, dtype=float)
    if not len(values):
        return np.array([0.0, fallback_width])
    if len(values) == 1:
        width = float(fallback_width)
        if not np.isfinite(width) or width <= 0:
            width = 1.0
        return np.array([values[0] - 0.5 * width, values[0] + 0.5 * width])
    mids = (values[:-1] + values[1:]) / 2
    return np.concatenate(([values[0] - (mids[0] - values[0])], mids, [values[-1] + (values[-1] - mids[-1])]))


def _edges_from_rows(rows: list[dict[str, Any]], centers: np.ndarray, *, min_key: str, max_key: str) -> np.ndarray:
    if len(centers) == 1 and rows:
        for row in rows:
            lo, hi = _float(row.get(min_key)), _float(row.get(max_key))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return np.array([lo, hi])
    return _edges_from_centers(centers)


def _time_edges(values: np.ndarray, *, fallback_time: bool) -> np.ndarray:
    if not len(values):
        return np.array([-0.5, 0.5])
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    diffs = np.diff(values)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        # Preserve report order when timestamps repeat or move backward; label
        # each row with its actual time while keeping the raster in frame order.
        return _edges_from_centers(np.arange(len(values), dtype=float))
    return _edges_from_centers(values)


def _intensity_matrix(
    profile_sets: Sequence[Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    unit: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_center = "q_center" if kind == "radial" else "angle_center_deg"
    key_min = "q_min" if kind == "radial" else "angle_min_deg"
    key_max = "q_max" if kind == "radial" else "angle_max_deg"
    if unit is not None:
        selected_frames = [
            i for i, summary in enumerate(summaries)
            if _canonical_q_unit(summary.get("q_unit")) == unit
        ]
    else:
        selected_frames = list(range(len(summaries)))
    coord_values: set[float] = set()
    for frame_index in selected_frames:
        for row in _rows(profile_sets[frame_index]):
            coordinate = _float(row.get(key_center))
            mean = _float(row.get("mean"))
            if np.isfinite(coordinate) and np.isfinite(mean):
                coord_values.add(coordinate)
    coords = sorted(coord_values)
    centers = np.asarray(coords, dtype=float)
    matrix = np.full((len(summaries), len(centers)), np.nan, dtype=float)
    coord_index = {value: i for i, value in enumerate(coords)}
    for frame_index in selected_frames:
        for row in _rows(profile_sets[frame_index]):
            x = _float(row.get(key_center))
            y = _float(row.get("mean"))
            if np.isfinite(x) and x in coord_index:
                matrix[frame_index, coord_index[x]] = y
    if centers.size:
        edges = _shared_profile_edges(profile_sets, selected_frames, centers, key_center, key_min, key_max)
    else:
        edges = np.array([0.0, 1.0])
    return matrix, centers, edges


def _shared_profile_edges(
    profile_sets: Sequence[Any],
    selected_frames: list[int],
    centers: np.ndarray,
    center_key: str,
    min_key: str,
    max_key: str,
) -> np.ndarray:
    """Use recorded common bin edges when all centers have contiguous bounds."""

    for frame_index in selected_frames:
        rows = _rows(profile_sets[frame_index])
        lookup = {
            _float(row.get(center_key)): row
            for row in rows
            if np.isfinite(_float(row.get(center_key)))
        }
        if not all(center in lookup for center in centers):
            continue
        lower = np.asarray([_float(lookup[center].get(min_key)) for center in centers])
        upper = np.asarray([_float(lookup[center].get(max_key)) for center in centers])
        if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(upper > lower):
            tolerance = max(float(np.ptp(np.concatenate((lower, upper)))) * 1e-9, 1e-12)
            if np.allclose(upper[:-1], lower[1:], rtol=1e-9, atol=tolerance):
                return np.concatenate((lower[:1], upper))
    return _edges_from_centers(centers)


def _profile_heatmap(
    summaries: Sequence[Mapping[str, Any]],
    profile_sets: Sequence[Any],
    *,
    kind: str,
    unit: str | None,
    x_values: np.ndarray,
    x_label: str,
) -> Figure:
    matrix, centers, q_edges = _intensity_matrix(profile_sets, summaries, kind=kind, unit=unit)
    y_edges = _time_edges(x_values, fallback_time=_is_time_axis(x_label))
    fig = _attach_agg(Figure(figsize=(9.2, 4.8), dpi=160, layout="constrained", facecolor="white"))
    ax = fig.add_subplot(1, 1, 1)
    if matrix.size and centers.size:
        norm, norm_label = _intensity_norm(matrix)
        mesh = _masked_mesh(ax, q_edges, y_edges, matrix, cmap="cividis", norm=norm)
        fig.colorbar(mesh, ax=ax, pad=0.02, label="Bin mean intensity (input units)")
        if kind == "radial":
            ax.set_xlabel(_q_label(unit))
            ax.set_title("Radial mean intensity by frame")
        else:
            ax.set_xlabel("Azimuth (degree)")
            ax.set_title("Angular mean intensity by frame")
            ax.set_xlim(-180, 180)
        ax.set_ylabel(x_label)
        if x_label.startswith("Frame index"):
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if _is_time_axis(x_label) and len(x_values) > 1 and not (np.all(np.diff(x_values) > 0) or np.all(np.diff(x_values) < 0)):
            ax.set_ylabel("Frame row (tick labels are time in s)")
            ticks = np.arange(len(x_values), dtype=float)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{value:g}" for value in x_values])
        fig.text(0.5, 0.012, _normalization_note(norm_label), ha="center", va="bottom", fontsize=7, color=_GRAY)
        _reserve_figure_footer(fig, bottom=0.065, top=0.985)
    else:
        ax.text(0.5, 0.5, "No finite profile bins", transform=ax.transAxes, ha="center", va="center", color=_GRAY)
        ax.set_axis_off()
    _styles(ax)
    return fig


def _tensor_values(summary: Mapping[str, Any]) -> tuple[float, float]:
    """Return normalized eigenvalue contrast and principal direction in degrees."""

    tensor = summary.get("in_plane_intensity_tensor")
    if isinstance(tensor, Mapping):
        reported_anisotropy = _float(tensor.get("anisotropy"))
        reported_angle = _float(tensor.get("principal_axis_deg"))
        if np.isfinite(reported_anisotropy):
            return reported_anisotropy, reported_angle
        tensor_matrix = np.asarray(tensor.get("tensor", ()), dtype=float)
        if tensor_matrix.shape == (2, 2):
            xx = float(tensor_matrix[0, 0])
            xy = float(0.5 * (tensor_matrix[0, 1] + tensor_matrix[1, 0]))
            yy = float(tensor_matrix[1, 1])
        else:
            normalized = {str(key).lower().replace("_", ""): value for key, value in tensor.items()}
            xx = next((_float(normalized[key]) for key in ("xx", "ixx", "mxx", "tensorxx") if key in normalized), float("nan"))
            xy = next((_float(normalized[key]) for key in ("xy", "ixy", "mxy", "tensoryx", "tensorxy") if key in normalized), float("nan"))
            yy = next((_float(normalized[key]) for key in ("yy", "iyy", "myy", "tensoryy") if key in normalized), float("nan"))
    else:
        array = np.asarray(tensor, dtype=float) if tensor is not None else np.asarray([])
        if array.shape == (2, 2):
            xx, xy, yy = float(array[0, 0]), float(0.5 * (array[0, 1] + array[1, 0])), float(array[1, 1])
        else:
            return float("nan"), float("nan")
    if not all(np.isfinite(value) for value in (xx, xy, yy)):
        return float("nan"), float("nan")
    matrix = np.array([[xx, xy], [xy, yy]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    denominator = float(np.sum(eigenvalues))
    if denominator <= 0:
        return float("nan"), float("nan")
    anisotropy = float((eigenvalues[1] - eigenvalues[0]) / denominator)
    angle = float(np.rad2deg(0.5 * np.arctan2(2 * xy, xx - yy)))
    return anisotropy, angle


def _summary_scalar(summary: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in summary:
            value = _float(summary.get(key))
            if np.isfinite(value):
                return value
    intensity = summary.get("intensity")
    if isinstance(intensity, Mapping):
        for key in keys:
            short = key.removeprefix("intensity_")
            if short == "total":
                short = "sum"
            if short in intensity:
                value = _float(intensity.get(short))
                if np.isfinite(value):
                    return value
    elif "intensity" in keys:
        return _float(intensity)
    return float("nan")


def _has_finite_profile_data(
    profile_sets: Sequence[Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    unit: str,
) -> bool:
    key_center = "q_center" if kind == "radial" else "angle_center_deg"
    for index, summary in enumerate(summaries):
        if _canonical_q_unit(summary.get("q_unit")) != unit:
            continue
        for row in _rows(profile_sets[index]):
            if np.isfinite(_float(row.get(key_center))) and np.isfinite(_float(row.get("mean"))):
                return True
    return False


def _has_finite_intensity_summary(
    summaries: Sequence[Mapping[str, Any]], unit: str,
) -> bool:
    for summary in summaries:
        if _canonical_q_unit(summary.get("q_unit")) != unit:
            continue
        if np.isfinite(_summary_scalar(summary, "intensity_sum", "intensity_total")):
            return True
        if np.isfinite(_summary_scalar(summary, "intensity_mean", "mean_intensity", "intensity")):
            return True
        anisotropy, axis = _tensor_values(summary)
        if np.isfinite(anisotropy) or np.isfinite(axis):
            return True
    return False


def _parameter_row_value(row: Mapping[str, Any]) -> tuple[float, bool]:
    for key in ("value", "estimate", "parameter_value"):
        value = _float(row.get(key))
        if np.isfinite(value):
            return value, False
    candidate = _float(row.get("candidate_value"))
    return candidate, np.isfinite(candidate)


def _status_markers(ax: Any, x: np.ndarray, y: np.ndarray, summaries: Sequence[Mapping[str, Any]], *, color: str) -> None:
    statuses = [str(summary.get("status", "unspecified") or "unspecified").lower() for summary in summaries]
    finite = np.isfinite(y)
    if np.any(finite):
        ax.plot(x, y, color="#8A8A8A", lw=1.0, zorder=1)
    for status in dict.fromkeys(statuses):
        selected = np.asarray([item == status for item in statuses]) & finite
        if not np.any(selected):
            continue
        failed = status in {"failed", "failure", "error", "cancelled", "canceled"}
        ax.scatter(x[selected], y[selected], color=("#000000" if failed else color), marker=("x" if failed else "o"), s=24, zorder=3, label=f"{status.replace('_', ' ')}")


def _intensity_anisotropy_figure(summaries: Sequence[Mapping[str, Any]], x: np.ndarray, x_label: str, unit: str | None) -> Figure:
    in_unit = np.asarray([unit is None or _canonical_q_unit(item.get("q_unit")) == unit for item in summaries])
    sum_values = np.asarray([_summary_scalar(item, "intensity_sum", "intensity_total") for item in summaries])
    mean_values = np.asarray([_summary_scalar(item, "intensity_mean", "mean_intensity", "intensity") for item in summaries])
    anisotropy_and_angle = [_tensor_values(item) for item in summaries]
    anisotropy = np.asarray([item[0] for item in anisotropy_and_angle])
    angle = np.asarray([item[1] for item in anisotropy_and_angle])
    sum_values = np.where(in_unit, sum_values, np.nan)
    mean_values = np.where(in_unit, mean_values, np.nan)
    anisotropy = np.where(in_unit, anisotropy, np.nan)
    angle = np.where(in_unit, angle, np.nan)
    series = [
        (sum_values, "Summed intensity", _BLUE),
        (mean_values, "Mean intensity", _GREEN),
        (anisotropy, "Normalized tensor anisotropy", _ORANGE),
        (angle, "Principal tensor direction (degree)", _PURPLE),
    ]
    fig = _attach_agg(Figure(figsize=(9.4, 6.5), dpi=160, layout="constrained", facecolor="white"))
    axes = [fig.add_subplot(2, 2, index + 1) for index in range(4)]
    for ax, (values, label, color) in zip(axes, series):
        if np.any(np.isfinite(values)):
            _status_markers(ax, x, values, summaries, color=color)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "Not available in summary data", transform=ax.transAxes, ha="center", va="center", color=_GRAY)
        ax.set_title(label)
        ax.set_xlabel(x_label)
        if label == "Normalized tensor anisotropy":
            ax.set_ylabel(r"$(\lambda_{max}-\lambda_{min})/(\lambda_{max}+\lambda_{min})$")
            ax.set_ylim(-0.03, 1.03)
        elif label == "Principal tensor direction (degree)":
            ax.set_ylabel("Direction from qx (degree)")
        elif label == "Summed intensity":
            ax.set_ylabel("Intensity sum (input units)")
        else:
            ax.set_ylabel("Mean intensity (input units)")
        _styles(ax)
    if x_label.startswith("Frame index"):
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    q_desc = f"; q unit: {unit}" if unit else ""
    fig.suptitle(f"Intensity and in-plane tensor evolution{q_desc}", fontsize=12, fontweight="semibold")
    fig.text(0.5, 0.012, "Tensor anisotropy is the normalized eigenvalue contrast; uncertainty is not estimated.", ha="center", va="bottom", fontsize=7, color=_GRAY)
    _reserve_figure_footer(fig, bottom=0.06, top=0.94)
    return fig


def _flatten_parameter_rows(value: Any, frame_count: int) -> list[tuple[int | None, dict[str, Any]]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    try:
        items = list(value)
    except TypeError:
        return []
    output: list[tuple[int | None, dict[str, Any]]] = []
    for outer_index, item in enumerate(items):
        if isinstance(item, Mapping):
            output.append((None, dict(item)))
        else:
            try:
                nested = list(item)
            except TypeError:
                continue
            if outer_index < frame_count:
                for row in nested:
                    if isinstance(row, Mapping):
                        output.append((outer_index, dict(row)))
    return output


def _parameter_groups(
    parameter_rows: Any,
    summaries: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    raw = _flatten_parameter_rows(parameter_rows, len(summaries))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    summary_slots: dict[str, int] = {}
    for index, summary in enumerate(summaries):
        if summary.get("frame_index") is not None:
            summary_slots[str(summary.get("frame_index"))] = index

    def slot_for(row: Mapping[str, Any], outer: int | None, fallback: int) -> int | None:
        for key in ("frame_slot", "frame_index", "frame"):
            if row.get(key) is not None:
                value = str(row.get(key))
                if value in summary_slots:
                    return summary_slots[value]
                try:
                    candidate = int(float(value))
                    if 0 <= candidate < len(summaries):
                        return candidate
                except (TypeError, ValueError, OverflowError):
                    pass
        return outer if outer is not None else (fallback if fallback < len(summaries) else None)

    next_fallback: dict[tuple[str, str], int] = defaultdict(int)
    for outer, row in raw:
        nested = row.get("parameters")
        if isinstance(nested, Mapping):
            for name, value in nested.items():
                if isinstance(value, Mapping):
                    item = dict(value)
                    item.setdefault("parameter", name)
                    item.setdefault("unit", row.get(f"{name}_unit", "unit unspecified"))
                    item.setdefault("frame_index", row.get("frame_index", row.get("frame")))
                    item.setdefault("time_s", row.get("time_s"))
                    item.setdefault("time", row.get("time"))
                    item.setdefault("time_unit", row.get("time_unit"))
                    value_record = item
                else:
                    value_record = {"parameter": name, "value": value, "unit": row.get(f"{name}_unit", "unit unspecified"), "frame_index": row.get("frame_index", row.get("frame")), "time_s": row.get("time_s"), "time": row.get("time"), "time_unit": row.get("time_unit"), "status": row.get("status")}
                name_key = str(value_record.get("parameter", name))
                unit_key = str(value_record.get("unit") or "unit unspecified")
                group_key = (name_key, unit_key)
                slot = slot_for(value_record, outer, next_fallback[group_key])
                next_fallback[group_key] += 1
                groups[group_key].append({**value_record, "_slot": slot})
            continue

        name = row.get("parameter", row.get("name"))
        if name is not None and any(key in row for key in ("value", "estimate", "parameter_value")):
            name_key = str(name)
            unit_key = str(row.get("unit") or "unit unspecified")
            group_key = (name_key, unit_key)
            slot = slot_for(row, outer, next_fallback[group_key])
            next_fallback[group_key] += 1
            groups[group_key].append({**row, "_slot": slot})
            continue

        # Wide records are accepted for callers that already have one row/frame.
        metadata = {"frame_index", "frame_slot", "frame", "time_s", "time", "time_unit", "status", "file", "filename", "path", "q_unit", "intensity_unit"}
        outer_frame = row.get("frame_index", row.get("frame"))
        for name_key, value in row.items():
            if name_key in metadata or name_key.endswith(("_stderr", "_unit", "_status")) or name_key in {"unit", "stderr", "parameter", "value"}:
                continue
            if not (value is None or np.isscalar(value)):
                continue
            unit_key = str(row.get(f"{name_key}_unit") or "unit unspecified")
            group_key = (str(name_key), unit_key)
            slot = slot_for({"frame_index": outer_frame}, None, next_fallback[group_key])
            next_fallback[group_key] += 1
            groups[group_key].append({
                "parameter": str(name_key),
                "value": value,
                "stderr": row.get(f"{name_key}_stderr"),
                "status": row.get(f"{name_key}_status", row.get("status")),
                "time_s": row.get("time_s"),
                "time": row.get("time"),
                "time_unit": row.get("time_unit"),
                "_slot": slot,
            })
    return groups


def _parameter_label(name: str, unit: str) -> str:
    labels = {
        "a": "Ellipse semi-major axis a",
        "b": "Ellipse semi-minor axis b",
        "axis_ratio": "Axis ratio b/a",
        "theta_deg": "Apparent ellipse axis tilt",
        "q_star": "Radial peak q*",
        "q_star_from_arcs": "Observed-arc radial peak q*",
        "Ln_nm": "Radial peak period 2π/q*",
        "L_from_observed_radius_nm": "Observed ring period 2π/q*",
        "Ln_from_minor_axis_nm": "Conditional ellipse normal spacing Ln",
        "L_N": "Conditional ellipse normal spacing L_N",
        "Lz_from_draw_axis_nm": "Conditional ellipse draw-axis spacing Lz",
        "L_z": "Conditional ellipse draw-axis spacing L_z",
        "Ln_candidate_from_minor_axis_nm": "Candidate ellipse normal spacing Ln",
        "Lz_candidate_from_draw_axis_nm": "Candidate ellipse draw-axis spacing Lz",
        "L_candidate_from_major_axis_nm": "Candidate major-axis period",
    }
    label = labels.get(name, name.replace("_", " "))
    return f"{label} ({unit})"


def _slug(value: str) -> str:
    output = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in output:
        output = output.replace("__", "_")
    return output.strip("_") or "unknown"


def _parameter_figure(
    name: str,
    unit: str,
    rows: list[dict[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    x: np.ndarray,
    x_label: str,
) -> Figure:
    y = np.full(len(summaries), np.nan)
    err = np.full(len(summaries), np.nan)
    candidate_only = np.zeros(len(summaries), dtype=bool)
    status = np.asarray([str(summary.get("status", "unspecified") or "unspecified") for summary in summaries], dtype=object)
    for row in rows:
        slot = row.get("_slot")
        if slot is None or not 0 <= int(slot) < len(summaries):
            continue
        index = int(slot)
        y[index], candidate_only[index] = _parameter_row_value(row)
        error = _float(row.get("stderr"))
        err[index] = error if np.isfinite(error) and error >= 0 else np.nan
        if row.get("status") is not None:
            status[index] = str(row.get("status"))

    fig = _attach_agg(Figure(figsize=(7.7, 3.7), dpi=160, layout="constrained", facecolor="white"))
    ax = fig.add_subplot(1, 1, 1)
    finite = np.isfinite(y)
    if np.any(finite):
        ax.plot(x, y, color="#888888", lw=1.0, zorder=1)
        for state in dict.fromkeys(status):
            selected = (status == state) & finite & ~candidate_only
            if not np.any(selected):
                continue
            failed = str(state).lower() in {"failed", "failure", "error", "bad_fit", "cancelled", "canceled"}
            ax.scatter(x[selected], y[selected], color=("#000000" if failed else _BLUE), marker=("x" if failed else "o"), s=28, label=str(state).replace("_", " "), zorder=3)
        if np.any(candidate_only & finite):
            ax.scatter(x[candidate_only & finite], y[candidate_only & finite], facecolors="none", edgecolors=_ORANGE, marker="D", s=34, label="Candidate value only", zorder=4)
        good_err = finite & np.isfinite(err)
        if np.any(good_err):
            ax.errorbar(x[good_err], y[good_err], yerr=err[good_err], fmt="none", ecolor=_GRAY, elinewidth=0.9, capsize=2, label="Reported standard error", zorder=2)
        missing = ~finite
        if np.any(missing):
            ax.scatter(x[missing], np.full(np.count_nonzero(missing), 0.025), transform=ax.get_xaxis_transform(), marker="|", color=_GRAY, s=65, label="No finite estimate", zorder=3)
        ax.legend(frameon=False, fontsize=7, loc="best")
    else:
        ax.text(0.5, 0.5, "No finite estimates", transform=ax.transAxes, ha="center", va="center", color=_GRAY)
    ax.set_title(_parameter_label(name, unit))
    ax.set_xlabel(x_label)
    ax.set_ylabel(_parameter_label(name, unit))
    if x_label.startswith("Frame index"):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _styles(ax)
    return fig


def render_sequence_report(
    output_dir: str | Path,
    *,
    summaries: Sequence[Mapping[str, Any]],
    radial_profiles: Sequence[Any],
    angular_profiles: Sequence[Any],
    parameter_rows: Any,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 160,
) -> dict[str, Path]:
    """Render sequence profiles, intensity/tensor trends, and long parameters.

    Frames stay in input order.  Profile and parameter gaps remain empty, and
    q-dependent radial and angular plots are grouped by recorded q unit.
    """

    if isinstance(dpi, bool) or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    fmt = _format_list(formats)
    if any(not isinstance(row, Mapping) for row in summaries):
        raise TypeError("each summary must be a mapping")
    summary_rows = list(summaries)
    if len(radial_profiles) != len(summary_rows) or len(angular_profiles) != len(summary_rows):
        raise ValueError("radial_profiles and angular_profiles must match summaries length")
    output = Path(output_dir)
    x, x_label, _ = _time_axis(summary_rows)
    paths: dict[str, Path] = {}
    units = list(dict.fromkeys(_canonical_q_unit(row.get("q_unit")) for row in summary_rows))
    for unit in units:
        if _has_finite_profile_data(radial_profiles, summary_rows, kind="radial", unit=unit):
            radial = _profile_heatmap(summary_rows, radial_profiles, kind="radial", unit=unit, x_values=x, x_label=x_label)
            paths.update(_save(radial, output, f"sequence_radial_evolution_{_slug(unit)}", fmt, int(dpi), description=f"Observed radial mean intensity by frame; grouped by q unit {unit}; missing bins remain blank."))
            radial.clear()

        if _has_finite_profile_data(angular_profiles, summary_rows, kind="angular", unit=unit):
            angular = _profile_heatmap(summary_rows, angular_profiles, kind="angular", unit=unit, x_values=x, x_label=x_label)
            paths.update(_save(angular, output, f"sequence_angular_evolution_{_slug(unit)}", fmt, int(dpi), description=f"Observed angular mean intensity by frame; grouped by q unit {unit}; missing bins remain blank."))
            angular.clear()

        if _has_finite_intensity_summary(summary_rows, unit):
            evolution = _intensity_anisotropy_figure(summary_rows, x, x_label, unit)
            paths.update(_save(evolution, output, f"sequence_intensity_anisotropy_{_slug(unit)}", fmt, int(dpi), description=f"Intensity summaries and normalized in-plane tensor eigenvalue contrast by frame; grouped by q unit {unit}."))
            evolution.clear()

    groups = _parameter_groups(parameter_rows, summary_rows)
    for (name, unit), rows in groups.items():
        if not any(
            np.isfinite(_parameter_row_value(row)[0])
            and row.get("_slot") is not None
            and 0 <= int(row["_slot"]) < len(summary_rows)
            for row in rows
        ):
            continue
        figure = _parameter_figure(name, unit, rows, summary_rows, x, x_label)
        stem = f"sequence_parameter_{_slug(name)}_{_slug(unit)}"
        paths.update(_save(figure, output, stem, fmt, int(dpi), description=f"Recorded parameter {name} ({unit}) by frame with missing values and reported standard errors."))
        figure.clear()
    return paths
