"""Scientific diagnostic figures shared by the UI and exporters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


OKABE_ITO = {
    "data": "#0072B2",
    "model": "#D55E00",
    "ridge": "#F0E442",
    "ellipse_a": "#56B4E9",
    "ellipse_b": "#E69F00",
    "failed": "#000000",
}


def _diagnostic_q_unit(q_unit: str | None) -> str:
    """Return the small set of q-unit labels that can be shown honestly."""

    normalized = str(q_unit or "unknown").strip().lower().replace(" ", "")
    if normalized in {"1/nm", "nm^-1", "nm^−1", "nm−1", "nm-1", "nm⁻¹"}:
        return "nm^-1"
    if normalized in {
        "1/a",
        "a^-1",
        "a−1",
        "a-1",
        "angstrom^-1",
        "å^-1",
        "å^−1",
        "å−1",
        "å⁻¹",
    }:
        return "Å^-1"
    if normalized in {"pixel-q", "pixel_q", "pixelq", "pixel"}:
        return "pixel-q"
    return "unknown"


def _diagnostic_q_axis_labels(q_unit: str | None) -> tuple[str, str]:
    unit = _diagnostic_q_unit(q_unit)
    if unit == "nm^-1":
        suffix = r"nm$^{-1}$"
    elif unit == "Å^-1":
        suffix = r"Å$^{-1}$"
    else:
        suffix = unit
    return rf"$q_x$ ({suffix})", rf"$q_y$ ({suffix})"


def _finite_limits(values: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(finite[0])
        delta = max(abs(center) * 0.01, 1.0)
        return center - delta, center + delta
    return float(lo), float(hi)


def _display_transform(values: np.ndarray, scale: str) -> np.ndarray:
    """Apply a display-only contrast transform while preserving sign."""

    mode = str(scale or "linear").strip().lower().replace("-", "_")
    array = np.asarray(values, dtype=float)
    if mode in {"linear", "raw", "none"}:
        return array
    if mode in {"log", "log1p", "signed_log"}:
        return np.sign(array) * np.log1p(np.abs(array))
    if mode in {"asinh", "arcsinh"}:
        return np.arcsinh(array)
    raise ValueError("display_scale must be 'linear', 'log1p', or 'asinh'")


def _extent(qx: np.ndarray, qy: np.ndarray) -> tuple[float, float, float, float]:
    finite_x = np.asarray(qx, dtype=float)[np.isfinite(qx)]
    finite_y = np.asarray(qy, dtype=float)[np.isfinite(qy)]
    if finite_x.size == 0 or finite_y.size == 0:
        raise ValueError("qx/qy 中没有有限坐标")
    return (
        float(np.min(finite_x)),
        float(np.max(finite_x)),
        float(np.min(finite_y)),
        float(np.max(finite_y)),
    )


def plot_fit_diagnostics(
    observed: np.ndarray,
    model: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    q_unit: str = "unknown",
    ridge_xy: np.ndarray | Sequence[Sequence[float]] | None = None,
    ellipse_curves: Iterable[np.ndarray | Sequence[Sequence[float]]] = (),
    output: str | Path | None = None,
    title: str | None = None,
    dpi: int = 300,
    display_scale: str = "linear",
    display_percentile: float = 99.5,
) -> Any:
    """Create observed/model/residual/overlay diagnostics with honest shared scales.

    Observed and model panels use the same intensity limits. Residuals always use a
    zero-centered diverging scale. The function returns the Matplotlib ``Figure``
    and optionally writes a lossless PNG (or a vector format selected by suffix).
    """

    import matplotlib.pyplot as plt

    obs = np.asarray(observed, dtype=float)
    mod = np.asarray(model, dtype=float)
    if obs.shape != mod.shape:
        raise ValueError("observed 与 model 的 shape 必须一致")
    if np.shape(qx) != obs.shape or np.shape(qy) != obs.shape:
        raise ValueError("qx/qy 必须与图像 shape 一致")
    valid = np.isfinite(obs) & np.isfinite(mod)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != obs.shape:
            raise ValueError("valid_mask 与图像 shape 不一致")
        valid &= mask
    try:
        percentile = float(display_percentile)
    except (TypeError, ValueError) as exc:
        raise ValueError("display_percentile must be between 50 and 100") from exc
    if not np.isfinite(percentile) or not 50.0 <= percentile <= 100.0:
        raise ValueError("display_percentile must be between 50 and 100")
    residual = np.where(valid, obs - mod, np.nan)
    obs_show = _display_transform(np.where(valid, obs, np.nan), display_scale)
    mod_show = _display_transform(np.where(valid, mod, np.nan), display_scale)
    residual_show = _display_transform(residual, display_scale)
    data_lo, data_hi = _finite_limits(
        np.concatenate([obs_show[valid], mod_show[valid]]),
        upper=percentile,
    )
    resid_finite = residual_show[np.isfinite(residual_show)]
    resid_lim = float(np.percentile(np.abs(resid_finite), percentile)) if resid_finite.size else 1.0
    if not np.isfinite(resid_lim) or resid_lim <= 0:
        resid_lim = 1.0
    ext = _extent(np.asarray(qx), np.asarray(qy))
    qx_label, qy_label = _diagnostic_q_axis_labels(q_unit)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    panels = (
        (axes[0, 0], obs_show, "Observed", "cividis", data_lo, data_hi),
        (axes[0, 1], mod_show, "Model", "cividis", data_lo, data_hi),
        (axes[1, 0], residual_show, "Residual", "PuOr", -resid_lim, resid_lim),
        (axes[1, 1], obs_show, "Overlay", "cividis", data_lo, data_hi),
    )
    for label, (ax, array, panel_title, cmap, vmin, vmax) in zip("ABCD", panels):
        image = ax.imshow(
            array,
            origin="lower",
            extent=ext,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(qx_label)
        ax.set_ylabel(qy_label)
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
        color_label = (
            "Intensity (input units)"
            if panel_title != "Residual"
            else "Data - model"
        )
        if str(display_scale).strip().lower() not in {"linear", "raw", "none"}:
            color_label += f" · display {display_scale}"
        fig.colorbar(image, ax=ax, shrink=0.82, label=color_label)

    overlay = axes[1, 1]
    if ridge_xy is not None:
        points = np.asarray(ridge_xy, dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and points.size:
            overlay.scatter(
                points[:, 0],
                points[:, 1],
                s=10,
                marker="o",
                facecolors="none",
                edgecolors=OKABE_ITO["ridge"],
                linewidths=0.7,
                label="Observed ridge",
            )
    for index, curve in enumerate(ellipse_curves):
        xy = np.asarray(curve, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not xy.size:
            continue
        color = OKABE_ITO["ellipse_a"] if index % 2 == 0 else OKABE_ITO["ellipse_b"]
        overlay.plot(xy[:, 0], xy[:, 1], color=color, lw=1.3, ls="-" if index % 2 == 0 else "--", label=f"Ellipse {index + 1}")
    handles, labels = overlay.get_legend_handles_labels()
    if handles:
        overlay.legend(frameon=False, fontsize=7, loc="best")
    if title:
        fig.suptitle(title, fontsize=10)
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if target.suffix.lower() in {".png", ".tif", ".tiff"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(target, **save_kwargs)
    return fig

# These labels describe the recorded quantities, not an inferred 3D structure.
# q-coordinate units must be supplied explicitly; they cannot be inferred from
# the magnitude of an ellipse radius or a radial peak.
_EVOLUTION_QUANTITIES = {
    "a": ("Ellipse semi-major axis a", None),
    "b": ("Ellipse semi-minor axis b", None),
    "axis_ratio": ("Axis ratio b/a", "dimensionless"),
    "ellipticity": ("Ellipticity", "dimensionless"),
    "eccentricity": ("Eccentricity", "dimensionless"),
    "theta": ("Apparent ellipse axis tilt", "rad"),
    "theta_deg": ("Apparent ellipse axis tilt", "deg"),
    "reference_axis_deg": ("Reference axis angle", "deg"),
    "lobe_angle": ("Lobe angle", "rad"),
    "lobe_angle_deg": ("Lobe angle", "deg"),
    "angular_width": ("Angular width", "rad"),
    "angular_width_deg": ("Angular width", "deg"),
    "radial_sigma": ("Radial Gaussian width", None),
    "radial_gamma": ("Radial Lorentzian width", None),
    "q_star": ("Radial peak q*", None),
    "q_star_nm_inv": ("Radial peak q*", "nm^-1"),
    "q_star_Ainv": ("Radial peak q*", "Å^-1"),
    "q_star_from_arcs": ("Observed-arc radius q*", None),
    "Ln_nm": ("Radial peak spacing 2π/q*", "nm"),
    "Ln_from_minor_axis_nm": ("Conditional minor-axis spacing Ln", "nm"),
    "Lz_from_draw_axis_nm": ("Conditional draw-axis spacing Lz", "nm"),
    "Ln_candidate_from_minor_axis_nm": ("Candidate minor-axis spacing Ln", "nm"),
    "Lz_candidate_from_draw_axis_nm": ("Candidate draw-axis spacing Lz", "nm"),
    "L_candidate_from_major_axis_nm": ("Candidate major-axis spacing", "nm"),
    "L_from_observed_radius_nm": ("Observed-radius spacing 2π/q*", "nm"),
}


def _evolution_number(value: Any) -> float:
    """Read one finite scalar without inventing a value for unavailable data."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _parameter_evolution_x(
    rows: Sequence[Mapping[str, Any]], x_key: str,
) -> tuple[np.ndarray, str]:
    """Use one coordinate system for the entire series, in original row order."""

    values = np.asarray([_evolution_number(row.get(x_key)) for row in rows])
    if len(rows) and np.all(np.isfinite(values)):
        label = {"time_s": "Time (s)", "frame_index": "Frame index"}.get(
            x_key, x_key.replace("_", " ")
        )
        return values, label
    source = "time" if x_key == "time_s" else x_key.replace("_", " ")
    return np.arange(len(rows), dtype=float), (
        f"Frame index (0-based row order; {source} incomplete)"
    )


def _parameter_evolution_label(
    name: str,
    parameter_labels: Mapping[str, str] | None = None,
    parameter_units: Mapping[str, str | None] | None = None,
) -> str:
    """Label documented quantities; never assume a physical q/intensity unit."""

    label, unit = _EVOLUTION_QUANTITIES.get(name, (name.replace("_", " "), None))
    if parameter_labels is not None and name in parameter_labels:
        label = parameter_labels[name]
    if parameter_units is not None and name in parameter_units:
        unit = parameter_units[name]
    suffix = str(unit).strip() if unit is not None else ""
    return f"{label} ({suffix or 'unit unspecified'})"


def plot_parameter_evolution(
    rows: Sequence[Mapping[str, Any]],
    *,
    parameters: Sequence[str],
    x_key: str = "time_s",
    output: str | Path | None = None,
    dpi: int = 300,
    parameter_labels: Mapping[str, str] | None = None,
    parameter_units: Mapping[str, str | None] | None = None,
) -> Any:
    """Plot recorded estimates, statuses, gaps and available standard errors.

    Finite values are retained even for warning or failed fits; their status is
    shown by separate markers. A ``<parameter>_status`` field takes precedence
    over the row's ``status`` so parameter-specific candidates remain visible.
    Missing values break the connecting line and
    appear as ticks in an axes-relative bottom strip, never as measured zeros.
    Only finite, nonnegative reported standard errors produce error bars. If
    any requested x coordinate is unavailable, *all* points use zero-based row
    order rather than mixing time and frame numbers. Rows are never sorted.

    ``parameter_labels`` and ``parameter_units`` optionally describe quantities
    known by the caller. They change labels only, with no unit conversion.
    Undeclared units remain unspecified, except for documented quantities with
    explicit units (e.g. ``theta_deg`` and ``Ln_nm``) and dimensionless ratios.
    The returned Matplotlib Figure is owned by the caller.
    """

    import matplotlib.pyplot as plt

    if not parameters:
        raise ValueError("parameters 不能为空")
    x, x_label = _parameter_evolution_x(rows, x_key)
    # These are display categories only; "available" is not scientific acceptance.
    ordinary_estimate_statuses = {"ok", "success", "available"}
    failed_statuses = {"failed", "failure", "error", "bad_fit", "cancelled", "canceled"}

    fig, axes = plt.subplots(
        len(parameters), 1,
        figsize=(7.8, max(2.8, 2.7 * len(parameters))),
        sharex=True, constrained_layout=True,
    )
    axes_arr = np.atleast_1d(axes)
    for ax, name in zip(axes_arr, parameters):
        statuses = np.asarray([
            str(row.get(f"{name}_status", row.get("status")) or "unspecified")
            .strip().lower() or "unspecified"
            for row in rows
        ], dtype=object)
        status_order = list(dict.fromkeys(statuses))
        y = np.asarray([_evolution_number(row.get(name)) for row in rows])
        err = np.full(len(rows), np.nan, dtype=float)
        for i, row in enumerate(rows):
            for key in (f"{name}_stderr", f"stderr_{name}"):
                candidate = _evolution_number(row.get(key))
                if np.isfinite(candidate) and candidate >= 0:
                    err[i] = candidate
                    break
        finite = np.isfinite(y)
        if np.any(finite):
            # Keep NaNs in place: compressing to finite points joins gaps.
            ax.plot(x, y, color="#8A8A8A", lw=1.0, zorder=1, label="_sequence")
        for status in status_order:
            selected = statuses == status
            if status in ordinary_estimate_statuses:
                color, marker = OKABE_ITO["data"], "o"
            elif status in failed_statuses:
                color, marker = OKABE_ITO["failed"], "x"
            else:
                color, marker = OKABE_ITO["ellipse_b"], "^"
            status_label = status.replace("_", " ")
            estimates = selected & finite
            if np.any(estimates):
                ax.scatter(
                    x[estimates], y[estimates], color=color, marker=marker,
                    s=25, linewidths=1.1, zorder=3,
                    label=f"Estimate ({status_label})",
                )
            missing = selected & ~finite
            if np.any(missing):
                ax.scatter(
                    x[missing], np.full(np.count_nonzero(missing), 0.025),
                    transform=ax.get_xaxis_transform(), marker="|", color=color,
                    s=70, linewidths=1.4, zorder=3,
                    label=f"No finite estimate ({status_label})",
                )
        known_error = finite & np.isfinite(err)
        if np.any(known_error):
            # Missing/invalid errors are omitted, not replaced with zero.
            ax.errorbar(
                x[known_error], y[known_error], yerr=err[known_error], fmt="none",
                ecolor="#666666", elinewidth=1.0, capsize=2, zorder=2,
                label="Reported standard error",
            )
        if not np.any(finite):
            ax.text(
                0.5, 0.5, "No finite estimates" if len(rows) else "No frames",
                transform=ax.transAxes, ha="center", va="center", color="#666666",
            )
        ax.set_ylabel(_parameter_evolution_label(name, parameter_labels, parameter_units))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.margins(x=0.05, y=0.15)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=7, loc="best")
    axes_arr[-1].set_xlabel(x_label)
    if x_label.startswith("Frame index"):
        from matplotlib.ticker import MaxNLocator

        axes_arr[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if target.suffix.lower() in {".png", ".tif", ".tiff"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(target, **save_kwargs)
    return fig
