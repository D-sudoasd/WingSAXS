"""Multi-window sensitivity audit for 2D SAXS butterfly observables and harmonic models."""

from __future__ import annotations

from collections.abc import Sequence
import math
import sys
from typing import Any

import numpy as np

from ..serialization import json_safe
from ._common import (
    _bin_azimuthal_from_2d,
    _finite,
    _normalise_intensity_mode,
    _q_unit_info,
    _resolve_2d_context,
    _validate_q_window,
)
from .harmonics import compute_azimuthal_harmonics_and_order, fit_c2_bridge_four_lobe
from .radial import measure_radial_coarsening_observables


def _resolve_fit_c2():
    sp_mod = sys.modules.get("butterfly_saxs.saxs_physics")
    if sp_mod is not None:
        return getattr(sp_mod, "fit_c2_bridge_four_lobe", fit_c2_bridge_four_lobe)
    return fit_c2_bridge_four_lobe


def run_window_sensitivity_audit(
    intensity: Any,
    qx: Any = None,
    qy: Any = None,
    valid_mask: Any = None,
    *,
    q_unit: str = "nm^-1",
    q_windows: Sequence[tuple[float, float]] = (
        (0.25, 1.35),
        (0.11, 1.35),
        (0.12, 1.20),
        (0.15, 1.00),
    ),
    nominal_draw_axis_deg: float = 90.0,
    bright_axis_convention: str = "reference",
    intensity_mode: str = "calibrated",
    sigma: Any = None,
    geometry_cache: Any = None,
    n_radial_bins: int = 64,
    n_azimuthal_bins: int = 180,
    sector_half_width_deg: float = 20.0,
    azimuthal_q_max: float = 0.45,
    systematic_k_uncertainty: float | str | None = None,
) -> dict[str, Any]:
    """Compare direct/Kratky peaks, ``Q_win``, and flat vs C2-bridge models across q-windows.

    Performs a single ROI pre-filter / geometry pass and evaluates every requested
    ``q_window`` strictly on observed valid pixels, returning a strict-JSON
    (``allow_nan=False``) audit summary.
    """
    mode = _normalise_intensity_mode(intensity_mode)
    convention = str(bright_axis_convention or "reference").strip().lower()
    if convention not in {"draw", "reference"}:
        raise ValueError("bright_axis_convention must be 'draw' or 'reference'")
    validated_windows = [_validate_q_window(w) for w in q_windows]
    if not validated_windows:
        raise ValueError("q_windows must contain at least one (q_min, q_max) pair")
    if isinstance(n_radial_bins, (bool, np.bool_)) or int(n_radial_bins) < 4:
        raise ValueError("n_radial_bins must be an integer >= 4")
    if isinstance(n_azimuthal_bins, (bool, np.bool_)) or int(n_azimuthal_bins) < 12:
        raise ValueError("n_azimuthal_bins must be an integer >= 12")

    q_unit_label, _, unit_caveat = _q_unit_info(q_unit)
    global_q_max = max(w[1] for w in validated_windows)
    ref_span = max(w[1] - w[0] for w in validated_windows)

    img, qx_arr, qy_arr, q_mag, chi_deg, geom_mask, inclusion, sig_arr, pixel_dq_hint = _resolve_2d_context(
        intensity,
        qx,
        qy,
        valid_mask,
        geometry_cache=geometry_cache,
        sigma=sigma,
        required_q_max=global_q_max,
    )

    # Build a single compact in-memory ROI cache for the whole multi-window pass
    if q_mag is None:
        in_box = (np.abs(qx_arr) <= global_q_max) & (np.abs(qy_arr) <= global_q_max)
        row_any = np.any(in_box, axis=1)
        if not np.any(row_any):
            if not np.any(np.isfinite(qx_arr) & np.isfinite(qy_arr)):
                raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
            sl = (slice(0, img.shape[0]), slice(0, img.shape[1]))
        else:
            col_any = np.any(in_box, axis=0)
            y0 = int(np.argmax(row_any))
            y1 = int(img.shape[0] - np.argmax(row_any[::-1]))
            x0 = int(np.argmax(col_any))
            x1 = int(img.shape[1] - np.argmax(col_any[::-1]))
            sl = (slice(y0, y1), slice(x0, x1))
        img_roi = np.ascontiguousarray(img[sl])
        qx_roi = np.ascontiguousarray(qx_arr[sl])
        qy_roi = np.ascontiguousarray(qy_arr[sl])
        q_roi = np.hypot(qx_roi, qy_roi)
        chi_roi = np.mod(np.degrees(np.arctan2(qy_roi, qx_roi)), 360.0)
        geom_roi = np.isfinite(q_roi)
        valid_roi = np.ascontiguousarray(inclusion[sl]) & geom_roi
        sig_roi = np.ascontiguousarray(sig_arr[sl]) if sig_arr is not None else None
    else:
        img_roi = img
        qx_roi = qx_arr
        qy_roi = qy_arr
        q_roi = q_mag
        chi_roi = (
            np.mod(chi_deg, 360.0)
            if chi_deg is not None
            else np.mod(np.degrees(np.arctan2(qy_roi, qx_roi)), 360.0)
        )
        geom_roi = geom_mask if geom_mask is not None else np.isfinite(q_roi)
        valid_roi = inclusion & geom_roi
        sig_roi = sig_arr

    from ..geometry import SAXSGeometryCache

    pass_cache = SAXSGeometryCache(
        geometry=geometry_cache.geometry if geometry_cache is not None else {"qx": qx_roi, "qy": qy_roi, "q": q_roi},
        shape=img_roi.shape,
        beam_center_xy=getattr(geometry_cache, "beam_center_xy", (0.0, 0.0)),
        pixel_dq=float(pixel_dq_hint) if pixel_dq_hint is not None else 1.0,
        base_valid_mask=valid_roi,
        beamstop_exclusion_mask=np.zeros(img_roi.shape, dtype=bool),
        valid_mask=valid_roi,
        roi_slices=(slice(0, img_roi.shape[0]), slice(0, img_roi.shape[1])),
        q_mag_roi=q_roi,
        qx_roi=qx_roi,
        qy_roi=qy_roi,
        chi_deg_roi=chi_roi,
        valid_roi=valid_roi,
        q_coordinates_validated=True,
        q_roi_max=global_q_max,
        q_unit=q_unit_label,
        geometry_roi_mask=geom_roi,
    )

    # Seed reference lobe centers once from the widest window's azimuthal band
    widest_lo, widest_hi = max(validated_windows, key=lambda w: w[1] - w[0])
    seed_az_hi = min(float(azimuthal_q_max), widest_hi)
    seed_az_win = (widest_lo, seed_az_hi) if seed_az_hi > widest_lo else (widest_lo, widest_hi)
    seed_ang, seed_prof, seed_cnt, _, _, _ = _bin_azimuthal_from_2d(
        img_roi,
        qx_roi,
        qy_roi,
        geom_roi,
        None,
        q_window=seed_az_win,
        n_bins=int(n_azimuthal_bins),
        q_mag=q_roi,
        chi_deg=chi_roi,
        inclusion=valid_roi,
    )
    seed_harm = compute_azimuthal_harmonics_and_order(
        angle_deg=seed_ang,
        intensity=seed_prof,
        counts=seed_cnt,
        q_unit=q_unit,
        nominal_draw_axis_deg=nominal_draw_axis_deg,
        bright_axis_convention=convention,
    )
    shared_lobe_centers: list[float] | None = None
    if (seed_harm.get("a4") or 0.0) >= 0.02 and int(seed_harm.get("n_observed_bins") or 0) >= 12:
        seed_fit = _resolve_fit_c2()(
            seed_ang,
            seed_prof,
            counts=seed_cnt,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            bright_axis_convention=convention,
            initial_chi_sym_deg=seed_harm.get("friedel_bisector_deg"),
            initial_alpha_deg=seed_harm.get("wing_half_opening_estimate_deg"),
            max_nfev=1500,
        )
        if seed_fit.get("success") and seed_fit.get("lobe_centers_deg"):
            centers = [float(c) for c in seed_fit["lobe_centers_deg"] if c is not None]
            if len(centers) == 4:
                shared_lobe_centers = centers

    window_rows: list[dict[str, Any]] = []
    all_flags: list[str] = []
    any_sigma = False

    for idx_w, (w_lo, w_hi) in enumerate(validated_windows):
        n_bins_w = max(16, int(round(float(n_radial_bins) * (w_hi - w_lo) / max(ref_span, 1.0e-12))))
        rad = measure_radial_coarsening_observables(
            img_roi,
            q_window=(w_lo, w_hi),
            q_unit=q_unit,
            n_radial_bins=n_bins_w,
            lobe_angles_deg=shared_lobe_centers,
            sector_half_width_deg=sector_half_width_deg,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            geometry_cache=pass_cache,
            intensity_mode=mode,
            sigma=sig_roi,
            systematic_k_uncertainty=systematic_k_uncertainty,
        )
        if rad.get("sigma_source") == "supplied_statistical_sigma":
            any_sigma = True

        az_hi = min(float(azimuthal_q_max), w_hi)
        az_win = (w_lo, az_hi) if az_hi > w_lo else (w_lo, w_hi)
        w_ang, w_prof, w_cnt, _, _, _ = _bin_azimuthal_from_2d(
            img_roi,
            qx_roi,
            qy_roi,
            geom_roi,
            None,
            q_window=az_win,
            n_bins=int(n_azimuthal_bins),
            q_mag=q_roi,
            chi_deg=chi_roi,
            inclusion=valid_roi,
        )
        harm = compute_azimuthal_harmonics_and_order(
            angle_deg=w_ang,
            intensity=w_prof,
            counts=w_cnt,
            q_unit=q_unit,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            bright_axis_convention=convention,
        )
        c2_fit = _resolve_fit_c2()(
            w_ang,
            w_prof,
            counts=w_cnt,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            bright_axis_convention=convention,
            initial_chi_sym_deg=harm.get("friedel_bisector_deg"),
            initial_alpha_deg=harm.get("wing_half_opening_estimate_deg"),
            max_nfev=1500,
        )
        fcmp = c2_fit.get("flat_baseline_comparison") or {}
        all_flags.extend(rad.get("flags") or ())
        all_flags.extend(harm.get("flags") or ())

        window_rows.append({
            "window_index": idx_w,
            "window_label": f"{w_lo:g}_{w_hi:g}",
            "q_window": [w_lo, w_hi],
            "azimuthal_q_window": [az_win[0], az_win[1]],
            "n_radial_bins": n_bins_w,
            "peak_channel_source": rad.get("peak_channel_source"),
            "q_star_direct": rad.get("q_star_direct"),
            "L_star_direct_nm": rad.get("L_star_direct_nm"),
            "edge_hit_direct": rad.get("edge_hit_direct"),
            "direct_low_q_edge_hit": rad.get("direct_low_q_edge_hit"),
            "q_star_kratky": rad.get("q_star_kratky"),
            "L_star_kratky_nm": rad.get("L_star_kratky_nm"),
            "edge_hit_kratky": rad.get("edge_hit_kratky"),
            "kratky_low_q_edge_hit": rad.get("kratky_low_q_edge_hit"),
            "I_bg_estimate": rad.get("I_bg_estimate"),
            "q_win_invariant": rad.get("q_win_invariant"),
            "q_win_bg_subtracted": rad.get("q_win_bg_subtracted"),
            "q_win_stat_stderr": rad.get("q_win_stat_stderr"),
            "a2": harm.get("a2"),
            "a4": harm.get("a4"),
            "a4_over_a2": harm.get("a4_over_a2"),
            "friedel_bisector_deg": harm.get("friedel_bisector_deg"),
            "orthogonal_rotation_detected": bool(harm.get("orthogonal_rotation_detected", False)),
            "c2_bridge_model": {
                "success": bool(c2_fit.get("success", False)),
                "c0": c2_fit.get("c0"),
                "c2": c2_fit.get("c2"),
                "chi_sym_deg": c2_fit.get("chi_sym_deg"),
                "alpha_deg": c2_fit.get("alpha_deg"),
                "lobe_fwhm_deg": c2_fit.get("lobe_fwhm_deg"),
                "rmse": c2_fit.get("rmse"),
                "r_squared": c2_fit.get("r_squared"),
                "aic": c2_fit.get("aic"),
                "bic": c2_fit.get("bic"),
            },
            "flat_baseline_model": {
                "success": bool(fcmp.get("success", False)),
                "c0": fcmp.get("c0"),
                "chi_sym_deg": fcmp.get("chi_sym_deg"),
                "alpha_deg": fcmp.get("alpha_deg"),
                "lobe_fwhm_deg": fcmp.get("lobe_fwhm_deg"),
                "rmse": fcmp.get("rmse"),
                "r_squared": fcmp.get("r_squared"),
                "aic": fcmp.get("aic"),
                "bic": fcmp.get("bic"),
            },
            "model_comparison": {
                "delta_rmse": fcmp.get("delta_rmse"),
                "delta_r_squared": fcmp.get("delta_r_squared"),
                "delta_aic": fcmp.get("delta_aic"),
                "delta_bic": fcmp.get("delta_bic"),
                "bridge_f_stat": fcmp.get("bridge_f_stat"),
                "bridge_improved": bool(fcmp.get("bridge_improved", False)),
                "lobe_fwhm_bias_deg": fcmp.get("lobe_fwhm_bias_deg"),
            },
        })

    def _span(values: list[float | None]) -> list[float] | None:
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        return [min(finite), max(finite)] if finite else None

    delta_bics = [
        float(r["model_comparison"]["delta_bic"])
        for r in window_rows
        if r["model_comparison"]["delta_bic"] is not None
    ]
    caveats = [
        "Window sensitivity audit evaluates strictly observed valid pixels across each declared q_window "
        "without fabricating or extrapolating outside the measured reciprocal-space domain.",
    ]
    if unit_caveat is not None:
        caveats.append(unit_caveat)
    if mode == "raw":
        caveats.append(
            "intensity_mode='raw': q_win_invariant and q_win_bg_subtracted are withheld (None) across all windows."
        )

    sys_k = "unavailable" if systematic_k_uncertainty is None else systematic_k_uncertainty
    return json_safe({
        "schema_version": "wingsaxs.window_sensitivity_audit.v1",
        "status": "estimated" if window_rows else "unavailable",
        "q_unit": q_unit_label,
        "intensity_mode": mode,
        "nominal_draw_axis_deg": float(nominal_draw_axis_deg) % 180.0,
        "bright_axis_convention": convention,
        "sigma_source": "supplied_statistical_sigma" if any_sigma else "unavailable",
        "systematic_k_uncertainty": sys_k,
        "windows": window_rows,
        "summary": {
            "n_windows": len(window_rows),
            "n_windows_direct_peak": sum(1 for r in window_rows if r["q_star_direct"] is not None),
            "n_windows_kratky_peak": sum(1 for r in window_rows if r["q_star_kratky"] is not None),
            "n_windows_bridge_improved": sum(1 for r in window_rows if r["model_comparison"]["bridge_improved"]),
            "q_star_direct_range": _span([r["q_star_direct"] for r in window_rows]),
            "L_star_direct_nm_range": _span([r["L_star_direct_nm"] for r in window_rows]),
            "q_star_kratky_range": _span([r["q_star_kratky"] for r in window_rows]),
            "L_star_kratky_nm_range": _span([r["L_star_kratky_nm"] for r in window_rows]),
            "q_win_bg_subtracted_range": _span([r["q_win_bg_subtracted"] for r in window_rows]),
            "median_delta_bic_c2_vs_flat": _finite(float(np.median(delta_bics))) if delta_bics else None,
        },
        "flags": list(dict.fromkeys(all_flags)),
        "caveats": caveats,
    })
