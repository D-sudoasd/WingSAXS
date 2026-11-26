"""Radial coarsening observables, Kratky/Porod window invariants, and 1D correlation."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from ..serialization import json_safe
from ._common import (
    _bin_azimuthal_from_2d,
    _contiguous_segments,
    _finite,
    _finite_list,
    _normalise_intensity_mode,
    _q_unit_info,
    _resolve_2d_context,
    _spacing_nm_from_q,
    _validate_q_window,
)
from .harmonics import _detect_wing_angles_from_profile


def _interpolate_crossing(x0: float, x1: float, y0: float, y1: float, target: float) -> float:
    if y1 == y0:
        return float(x0)
    frac = min(1.0, max(0.0, (target - y0) / (y1 - y0)))
    return float(x0 + frac * (x1 - x0))


def _integrate_observed_segments_weights(
    x: np.ndarray,
    valid: np.ndarray,
    dx_bins: np.ndarray,
) -> np.ndarray:
    """Return exact per-bin linear weights matching :func:`_integrate_observed_segments`."""
    weights = np.zeros(x.shape, dtype=float)
    for a, b in _contiguous_segments(valid):
        n_seg = b - a
        if n_seg == 1:
            weights[a] = float(dx_bins[a])
        else:
            dx_int = np.diff(x[a:b])
            weights[a] = 0.5 * float(dx_bins[a]) + 0.5 * float(dx_int[0])
            weights[b - 1] = 0.5 * float(dx_bins[b - 1]) + 0.5 * float(dx_int[-1])
            if n_seg > 2:
                weights[a + 1 : b - 1] = 0.5 * (dx_int[:-1] + dx_int[1:])
    return weights


def _integrate_observed_segments(x: np.ndarray, y: np.ndarray, dx_bins: np.ndarray) -> float | None:
    """Integrate ``y(x)`` strictly over contiguous observed segments without interpolating gaps."""
    valid = np.isfinite(x) & np.isfinite(y)
    segments = _contiguous_segments(valid)
    if not segments:
        return None
    total = 0.0
    for a, b in segments:
        if b - a >= 2:
            # Trapezoid over interior plus half-bin end caps so uniform full-coverage
            # grids match midpoint quadrature over [q_min, q_max]
            seg_int = float(np.trapezoid(y[a:b], x[a:b]))
            seg_int += 0.5 * float(y[a]) * float(dx_bins[a]) + 0.5 * float(y[b - 1]) * float(dx_bins[b - 1])
            total += seg_int
        else:
            total += float(y[a]) * float(dx_bins[a])
    return _finite(total)


def _estimate_local_noise_sigma(values: np.ndarray, valid: np.ndarray) -> float:
    """Estimate local bin-to-bin noise via 3rd differences (annihilates quadratic curvature)."""
    if int(np.count_nonzero(valid)) < 16:
        return 0.0
    diffs: list[np.ndarray] = []
    for a, b in _contiguous_segments(valid):
        if b - a >= 4:
            ys = values[a:b]
            d3 = ys[3:] - 3.0 * ys[2:-1] + 3.0 * ys[1:-2] - ys[:-3]
            finite_d3 = d3[np.isfinite(d3)]
            if finite_d3.size:
                diffs.append(np.abs(finite_d3))
    if not diffs:
        return 0.0
    all_d3 = np.concatenate(diffs)
    if all_d3.size < 4:
        return 0.0
    return float(1.4826 * float(np.median(all_d3)) / math.sqrt(20.0))


def _estimate_radial_background_and_active_mask(
    q_centers: np.ndarray,
    I_q: np.ndarray,
    counts: np.ndarray,
    *,
    zero_pixel_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, int]:
    """Exclude outer zero-padded dead shells and estimate the active high-q diffuse floor.

    Returns ``(active_mask, I_bg_estimate, tail_noise_mad, n_dead_outer_bins)``.
    """
    obs = np.isfinite(q_centers) & np.isfinite(I_q) & (counts > 0)
    if not np.any(obs):
        return obs, 0.0, 0.0, 0

    obs_idx = np.flatnonzero(obs)
    max_val = float(np.max(I_q[obs_idx]))
    dead_outer = np.zeros(q_centers.shape, dtype=bool)

    if max_val > 1.0e-9 and obs_idx.size >= 4:
        if zero_pixel_fraction is not None:
            pos = obs_idx.size - 1
            dead_positions: list[int] = []
            # Allow up to 1 outermost sparse corner shell when adjacent inner shell has >85% zeros.
            if (
                pos >= 1
                and float(zero_pixel_fraction[obs_idx[pos]]) <= 0.75
                and float(zero_pixel_fraction[obs_idx[pos - 1]]) > 0.85
            ):
                dead_positions.append(int(obs_idx[pos]))
                pos -= 1
            while pos >= 0 and float(zero_pixel_fraction[obs_idx[pos]]) > 0.75:
                dead_positions.append(int(obs_idx[pos]))
                pos -= 1
            if (
                dead_positions
                and pos >= 2
                and bool(np.any(zero_pixel_fraction[obs_idx[: pos + 1]] < 0.50))
            ):
                for bin_idx in dead_positions:
                    dead_outer[bin_idx] = True
                inner_slice = zero_pixel_fraction[obs_idx[max(0, pos - 8) : pos]]
                inner_plateau_zf = float(np.median(inner_slice)) if inner_slice.size else 0.0
                trans_thresh = min(0.74, max(0.50, inner_plateau_zf + 0.12))
                if float(zero_pixel_fraction[obs_idx[pos]]) > trans_thresh:
                    dead_outer[obs_idx[pos]] = True
        else:
            # 1D-profile fallback: detect trailing exact-zero shells preceded by a sharp cliff
            pos = obs_idx.size - 1
            while pos >= 0 and abs(float(I_q[obs_idx[pos]])) <= 1.0e-12:
                pos -= 1
            n_trailing_zero = (obs_idx.size - 1) - pos
            if n_trailing_zero >= 1 and pos >= 0:
                prev_val = float(I_q[obs_idx[pos]])
                if prev_val > max(1.0e-4 * max_val, 1.0e-6):
                    for k in range(pos + 1, obs_idx.size):
                        dead_outer[obs_idx[k]] = True

    active = obs & ~dead_outer
    if not np.any(active):
        active = obs
        dead_outer[:] = False
    n_dead = int(np.count_nonzero(dead_outer))

    y_act = I_q[active]
    if y_act.size >= 5:
        n_tail = max(5, int(round(0.15 * y_act.size)))
        tail_vals = y_act[-n_tail:]
        tail_med = float(np.median(tail_vals))
        tail_mad = float(1.4826 * float(np.median(np.abs(tail_vals - tail_med))))
        # Cap at 25th percentile of active profile so a peak near q_max is never subtracted as floor
        p25_global = float(np.percentile(y_act, 25.0))
        I_bg = max(0.0, min(tail_med, p25_global))
    else:
        I_bg = max(0.0, float(np.min(y_act)))
        tail_mad = 0.0
    return active, I_bg, tail_mad, n_dead


def _extract_radial_peak_from_profile(
    q_centers: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
    *,
    q_scale_nm_inv: float | None,
    q_unit_label: str,
    channel_name: str,
    active_mask: np.ndarray | None = None,
    floor_value: float | None = None,
    noise_sigma_per_bin: np.ndarray | None = None,
    min_rel_prominence: float = 0.02,
    min_rel_height: float = 0.05,
    min_snr: float = 5.0,
) -> dict[str, Any]:
    """Extract peak q*, L*_nm, FWHM, prominence, and edge-hit status from a 1D observed profile."""
    valid = np.isfinite(q_centers) & np.isfinite(values) & (counts > 0)
    if active_mask is not None:
        valid = valid & np.asarray(active_mask, dtype=bool)
    n_obs = int(np.count_nonzero(valid))
    caveats: list[str] = []
    if q_scale_nm_inv is None:
        caveats.append(f"L_star_nm unavailable for {channel_name}: q_unit={q_unit_label!r} is not a physical reciprocal length.")

    if n_obs < 3:
        return {
            "status": "unavailable",
            "reason": "insufficient_observed_radial_bins",
            "q_star": None,
            "L_star_nm": None,
            "peak_value": None,
            "fwhm_q": None,
            "prominence": None,
            "edge_hit": False,
            "edge_hit_side": None,
            "n_observed_bins": n_obs,
            "caveats": caveats,
        }

    obs_indices = np.flatnonzero(valid)
    first_obs = int(obs_indices[0])
    last_obs = int(obs_indices[-1])
    y_obs = values[obs_indices]
    global_max_pos = int(obs_indices[int(np.argmax(y_obs))])
    global_max_val = float(values[global_max_pos])
    floor = float(floor_value) if (floor_value is not None and math.isfinite(float(floor_value))) else float(np.min(y_obs))
    y_range = max(global_max_val - floor, 0.0)
    local_d3_sigma = _estimate_local_noise_sigma(values, valid)

    # Search for genuine interior peaks inside each contiguous observed segment
    candidates: list[tuple[float, float, int, int, int]] = []  # (prominence, height, global_idx, seg_a, seg_b)
    segments = _contiguous_segments(valid)
    for a, b in segments:
        if b - a < 3:
            continue
        ys = values[a:b]
        pks, _ = find_peaks(ys)
        if pks.size == 0:
            continue
        proms, _, _ = peak_prominences(ys, pks)
        for pk_local, prom in zip(pks, proms):
            prom_f = float(prom)
            if not math.isfinite(prom_f) or prom_f <= 0.0:
                continue
            pk_global_i = a + int(pk_local)
            height_f = float(ys[pk_local])
            bin_noise = (
                float(noise_sigma_per_bin[pk_global_i])
                if (noise_sigma_per_bin is not None and math.isfinite(float(noise_sigma_per_bin[pk_global_i])))
                else 0.0
            )
            eff_noise = max(local_d3_sigma, bin_noise)
            prom_thresh = max(min_snr * eff_noise, min_rel_prominence * y_range)
            height_thresh = max(min_snr * eff_noise, min_rel_height * y_range)
            if prom_f >= prom_thresh and (height_f - floor) >= height_thresh:
                candidates.append((prom_f, height_f, pk_global_i, a, b))

    if not candidates or y_range <= 1.0e-15:
        edge_side = None
        edge_hit = False
        if global_max_pos <= first_obs + 1:
            edge_hit = True
            edge_side = "low_q"
        elif global_max_pos >= last_obs - 1:
            edge_hit = True
            edge_side = "high_q"
        return {
            "status": "boundary_monotonic" if edge_hit else "no_interior_peak",
            "reason": f"monotonic_maximum_at_{edge_side}_boundary" if edge_hit else "flat_or_no_interior_maximum",
            "q_star": None,
            "L_star_nm": None,
            "peak_value": _finite(global_max_val),
            "boundary_q": _finite(q_centers[global_max_pos]) if edge_hit else None,
            "fwhm_q": None,
            "prominence": None,
            "edge_hit": edge_hit,
            "edge_hit_side": edge_side,
            "n_observed_bins": n_obs,
            "caveats": caveats,
        }

    # Prefer the peak with the largest prominence (tie-break by peak height)
    prom_best, peak_val, pk_global, seg_a, seg_b = max(candidates, key=lambda item: (item[0], item[1]))

    # Sub-bin parabolic refinement using immediate observed neighbors
    q_star = float(q_centers[pk_global])
    if seg_a < pk_global < seg_b - 1:
        y_m = float(values[pk_global - 1])
        y_0 = float(values[pk_global])
        y_p = float(values[pk_global + 1])
        denom = y_m - 2.0 * y_0 + y_p
        if denom < -1.0e-15:
            shift = 0.5 * (y_m - y_p) / denom
            if abs(shift) <= 0.5:
                dq_step = 0.5 * float(q_centers[pk_global + 1] - q_centers[pk_global - 1])
                q_star = float(q_centers[pk_global] + shift * dq_step)

    # Half-height width on the contiguous observed segment (never crosses gaps)
    xs = q_centers[seg_a:seg_b]
    ys = values[seg_a:seg_b]
    pk_loc = pk_global - seg_a
    # Local baseline from the prominence base level (peak_val - prom_best)
    baseline = peak_val - prom_best
    half_height = baseline + 0.5 * prom_best
    left_cross = None
    for i in range(pk_loc - 1, -1, -1):
        if ys[i] <= half_height <= ys[i + 1]:
            left_cross = _interpolate_crossing(float(xs[i]), float(xs[i + 1]), float(ys[i]), float(ys[i + 1]), half_height)
            break
    right_cross = None
    for i in range(pk_loc, ys.size - 1):
        if ys[i] >= half_height >= ys[i + 1]:
            right_cross = _interpolate_crossing(float(xs[i]), float(xs[i + 1]), float(ys[i]), float(ys[i + 1]), half_height)
            break
    fwhm_q = _finite(right_cross - left_cross) if (left_cross is not None and right_cross is not None and right_cross > left_cross) else None

    # Edge-hit classification
    edge_hit = False
    edge_side = None
    if pk_global <= first_obs + 1 or (global_max_pos == first_obs and global_max_val > peak_val and left_cross is None):
        edge_hit = True
        edge_side = "low_q"
    elif pk_global >= last_obs - 1 or (global_max_pos == last_obs and global_max_val > peak_val and right_cross is None):
        edge_hit = True
        edge_side = "high_q"

    L_star_nm = _spacing_nm_from_q(q_star, q_scale_nm_inv)
    status = "edge_truncated_peak" if edge_hit else "estimated"
    return {
        "status": status,
        "reason": f"peak_near_{edge_side}_window_boundary" if edge_hit else None,
        "q_star": _finite(q_star),
        "L_star_nm": L_star_nm,
        "peak_value": _finite(peak_val),
        "fwhm_q": fwhm_q,
        "prominence": _finite(prom_best),
        "edge_hit": edge_hit,
        "edge_hit_side": edge_side,
        "n_observed_bins": n_obs,
        "caveats": caveats,
    }


def _summarize_radial_channel(
    q_edges: np.ndarray,
    I_q: np.ndarray,
    counts: np.ndarray,
    candidate_counts: np.ndarray,
    *,
    q_scale_nm_inv: float | None,
    q_unit_label: str,
    channel_name: str,
    zero_pixel_fraction: np.ndarray | None = None,
    background_level: float | None = None,
    intensity_mode: str = "calibrated",
    sigma_radial: np.ndarray | None = None,
) -> dict[str, Any]:
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    dq_bins = q_edges[1:] - q_edges[:-1]
    obs_mask = np.isfinite(I_q) & (counts > 0)
    active_mask, I_bg_auto, tail_mad, n_dead_outer = _estimate_radial_background_and_active_mask(
        q_centers, I_q, counts, zero_pixel_fraction=zero_pixel_fraction
    )
    if background_level is not None:
        bg_val = float(background_level)
        if not math.isfinite(bg_val) or bg_val < 0.0:
            raise ValueError("background_level must be finite and non-negative when provided")
        I_bg = bg_val
    else:
        I_bg = I_bg_auto
    kratky_raw = np.where(obs_mask, (q_centers ** 2) * I_q, np.nan)
    I_excess = np.where(active_mask, np.maximum(I_q - I_bg, 0.0), np.nan)
    kratky_sub = np.where(active_mask, (q_centers ** 2) * I_excess, np.nan)
    coverage = np.divide(
        counts,
        candidate_counts,
        out=np.full(q_centers.shape, np.nan, dtype=float),
        where=candidate_counts > 0,
    )
    direct_noise = np.full(q_centers.shape, tail_mad, dtype=float)
    kratky_noise = (q_centers ** 2) * tail_mad
    direct_peak = _extract_radial_peak_from_profile(
        q_centers,
        I_q,
        counts,
        q_scale_nm_inv=q_scale_nm_inv,
        q_unit_label=q_unit_label,
        channel_name=f"{channel_name}_direct",
        active_mask=active_mask,
        floor_value=I_bg,
        noise_sigma_per_bin=direct_noise,
    )
    kratky_peak = _extract_radial_peak_from_profile(
        q_centers,
        kratky_sub,
        counts,
        q_scale_nm_inv=q_scale_nm_inv,
        q_unit_label=q_unit_label,
        channel_name=f"{channel_name}_kratky",
        active_mask=active_mask,
        floor_value=0.0,
        noise_sigma_per_bin=kratky_noise,
    )
    kratky_peak["kratky_background_subtracted"] = True
    kratky_peak["I_bg_estimate"] = _finite(I_bg)

    if intensity_mode == "raw":
        q_win = None
        q_win_bg_sub = None
        q_win_stat_stderr = None
    else:
        q_win = _integrate_observed_segments(q_centers, kratky_raw, dq_bins)
        q_win_bg_sub = _integrate_observed_segments(q_centers, kratky_sub, dq_bins)
        q_win_stat_stderr = None
        if sigma_radial is not None:
            sig_ok = obs_mask & np.isfinite(sigma_radial) & (sigma_radial >= 0.0)
            if np.any(sig_ok):
                seg_w = _integrate_observed_segments_weights(q_centers, obs_mask, dq_bins)
                qwin_var = float(np.sum(((q_centers[sig_ok] ** 2) * seg_w[sig_ok] * sigma_radial[sig_ok]) ** 2))
                if math.isfinite(qwin_var) and qwin_var >= 0.0:
                    q_win_stat_stderr = _finite(math.sqrt(qwin_var))

    obs_bins = int(np.count_nonzero(obs_mask))
    cand_bins = int(np.count_nonzero(candidate_counts > 0))
    radial_cov = float(obs_bins / cand_bins) if cand_bins > 0 else 0.0

    return {
        "status": "estimated" if obs_bins > 0 else "unavailable",
        "q_centers": _finite_list(q_centers),
        "q_edges": _finite_list(q_edges),
        "I_q": _finite_list(I_q),
        "sigma_radial": _finite_list(sigma_radial) if sigma_radial is not None else None,
        "q2_I_q": _finite_list(kratky_raw),
        "q2_I_sub_q": _finite_list(kratky_sub),
        "I_bg_estimate": _finite(I_bg),
        "I_bg_noise_mad": _finite(tail_mad),
        "kratky_background_subtracted": True,
        "n_outer_zero_padded_bins_excluded": n_dead_outer,
        "counts": [int(c) for c in counts],
        "candidate_counts": [int(c) for c in candidate_counts],
        "coverage": _finite_list(coverage),
        "radial_bin_coverage": _finite(radial_cov),
        "direct_peak": direct_peak,
        "kratky_peak": kratky_peak,
        "q_win_invariant": q_win,
        "q_win_bg_subtracted": q_win_bg_sub,
        "q_win_stat_stderr": q_win_stat_stderr,
    }


def _compute_cosine_correlation_1d(
    q_centers: np.ndarray,
    kratky: np.ndarray,
    dq_bins: np.ndarray,
    *,
    q_scale_nm_inv: float | None,
    q_unit_label: str,
    r_max_nm: float | None,
    n_r: int,
) -> dict[str, Any]:
    """Compute window-bounded 1D cosine correlation function K(r) for calibrated physical q only."""
    if q_scale_nm_inv is None:
        return {
            "status": "unavailable",
            "reason": "uncalibrated_q_unit",
            "caveat": (
                f"1D cosine correlation function K(r) in nm requires calibrated physical q "
                f"(nm^-1 or A^-1); unavailable for q_unit={q_unit_label!r}."
            ),
            "r_nm": None,
            "K_r": None,
            "first_maximum_r_nm": None,
            "truncation_warnings": ["uncalibrated_q_unit_no_physical_correlation"],
        }

    valid = np.isfinite(q_centers) & np.isfinite(kratky)
    segments = _contiguous_segments(valid)
    if not segments:
        return {
            "status": "unavailable",
            "reason": "no_observed_radial_bins",
            "caveat": "No observed radial bins available for K(r).",
            "r_nm": None,
            "K_r": None,
            "first_maximum_r_nm": None,
            "truncation_warnings": ["no_observed_radial_bins"],
        }

    q_nm = q_centers * q_scale_nm_inv
    dq_nm = dq_bins * q_scale_nm_inv
    # Scale q^2 I(q) consistently to nm^-1 axis: q_nm^2 * I = (q_scale^2) * kratky
    k_nm = kratky * (q_scale_nm_inv ** 2)
    quad_w = _integrate_observed_segments_weights(q_nm, valid, dq_nm)
    q_win_nm = _finite(float(np.dot(k_nm[valid], quad_w[valid])))
    if q_win_nm is None or q_win_nm <= 0.0:
        return {
            "status": "unavailable",
            "reason": "nonpositive_window_invariant",
            "caveat": "Window-bounded invariant integral is non-positive; K(r) cannot be normalized.",
            "r_nm": None,
            "K_r": None,
            "first_maximum_r_nm": None,
            "truncation_warnings": ["nonpositive_window_invariant"],
        }

    obs_q_nm = q_nm[valid]
    q_lo_nm = float(np.min(obs_q_nm))
    q_hi_nm = float(np.max(obs_q_nm))
    if r_max_nm is None:
        r_max = min(150.0, 2.0 * math.pi / max(q_lo_nm, 0.04))
    else:
        r_max = float(r_max_nm)
        if not math.isfinite(r_max) or r_max <= 0.0:
            raise ValueError("r_max_nm must be finite and positive")

    n_pts = max(int(n_r), 16)
    r_grid = np.linspace(0.0, r_max, n_pts, dtype=float)
    weighted_k = k_nm[valid] * quad_w[valid]
    k_vals = (np.cos(np.outer(r_grid, obs_q_nm)) @ weighted_k) / q_win_nm

    # First positive interior maximum (classic 1D correlation peak; ignores negative Gibbs ripples in the trough)
    first_max_r: float | None = None
    pks, _ = find_peaks(k_vals)
    pos_pks = [int(p) for p in pks if math.isfinite(float(k_vals[p])) and float(k_vals[p]) > 0.0]
    if pos_pks:
        first_max_r = _finite(r_grid[pos_pks[0]])

    trunc_warnings = [
        "window_bounded_cosine_transform_only_no_guinier_or_porod_extrapolation",
    ]
    if len(segments) > 1:
        trunc_warnings.append("missing_radial_bins_excluded_not_interpolated")

    return {
        "status": "estimated",
        "reason": None,
        "r_nm": _finite_list(r_grid),
        "K_r": _finite_list(k_vals),
        "first_maximum_r_nm": first_max_r,
        "r_min_resolution_nm": _finite(math.pi / q_hi_nm) if q_hi_nm > 0.0 else None,
        "r_max_window_nm": _finite(math.pi / q_lo_nm) if q_lo_nm > 0.0 else None,
        "q_range_nm_inv": [_finite(q_lo_nm), _finite(q_hi_nm)],
        "truncation_warnings": trunc_warnings,
        "caveat": (
            "Window-bounded 1D cosine correlation function K(r) computed strictly over observed "
            "[q_min, q_max] without Guinier or Porod extrapolation; finite-window truncation "
            "ripples are expected."
        ),
    }




def _detect_wing_angles_from_1d_pixels(
    angles_obs_deg: np.ndarray,
    img_obs: np.ndarray,
    *,
    nominal_draw_axis_deg: float,
    n_bins: int = 180,
) -> tuple[list[float], str]:
    """Detect butterfly wing ray angles directly from already-filtered 1D observed pixels."""
    if angles_obs_deg.size < 16:
        return [], "insufficient_azimuthal_support"
    step = 360.0 / float(n_bins)
    ang = (np.arange(n_bins, dtype=float) + 0.5) * step
    idx = np.floor(angles_obs_deg * (float(n_bins) / 360.0)).astype(np.int64)
    np.clip(idx, 0, n_bins - 1, out=idx)
    cnts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    sums = np.bincount(idx, weights=img_obs, minlength=n_bins).astype(float)
    prof = np.full(n_bins, np.nan, dtype=float)
    np.divide(sums, cnts, out=prof, where=cnts > 0)
    return _detect_wing_angles_from_profile(
        ang, prof, cnts, nominal_draw_axis_deg=nominal_draw_axis_deg
    )


def _detect_wing_angles_for_sectors(
    img: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    geometry: np.ndarray | None,
    usable: np.ndarray | None,
    *,
    q_window: tuple[float, float],
    nominal_draw_axis_deg: float,
    q_mag: np.ndarray | None = None,
    chi_deg: np.ndarray | None = None,
    inclusion: np.ndarray | None = None,
) -> tuple[list[float], str]:
    """Detect butterfly wing ray angles strictly from observed pixels in ``q_window``."""
    ang, prof, cnts, _, _, _ = _bin_azimuthal_from_2d(
        img,
        qx,
        qy,
        geometry,
        usable,
        q_window=q_window,
        n_bins=180,
        q_mag=q_mag,
        chi_deg=chi_deg,
        inclusion=inclusion,
    )
    return _detect_wing_angles_from_profile(
        ang, prof, cnts, nominal_draw_axis_deg=nominal_draw_axis_deg
    )


def measure_radial_coarsening_observables(
    image: Any,
    qx: Any = None,
    qy: Any = None,
    *,
    valid_mask: Any = None,
    q_window: tuple[float, float] | None = None,
    q_unit: str = "unknown",
    n_radial_bins: int = 128,
    lobe_angles_deg: Sequence[float] | None = None,
    sector_half_width_deg: float = 15.0,
    nominal_draw_axis_deg: float = 90.0,
    background_level: float | None = None,
    compute_correlation: bool = False,
    r_max_nm: float | None = None,
    n_r: int = 128,
    geometry_cache: Any = None,
    intensity_mode: str = "calibrated",
    sigma: Any = None,
    systematic_k_uncertainty: float | str | None = None,
) -> dict[str, Any]:
    """Compute dual-channel direct I(q) & Kratky q^2 I(q) radial observables and Q_win.

    Extracts BOTH full-azimuth and butterfly lobe-sectored radial profiles strictly
    from observed pixels, returning direct ``I(q)`` peaks, Kratky ``q^2 I(q)``
    peaks, window-bounded scattering invariant proxy ``Q_win`` (strictly ``None``
    when ``intensity_mode='raw'``), optional propagated statistical uncertainty
    (``sigma_radial`` and ``q_win_stat_stderr`` when ``sigma`` is provided), and
    optional calibrated 1D cosine correlation function ``K(r)``.
    """
    mode = _normalise_intensity_mode(intensity_mode)
    if isinstance(n_radial_bins, (bool, np.bool_)) or int(n_radial_bins) < 4:
        raise ValueError("n_radial_bins must be an integer >= 4")
    n_bins = int(n_radial_bins)
    half_w = float(sector_half_width_deg)
    if not math.isfinite(half_w) or not (0.0 < half_w <= 90.0):
        raise ValueError("sector_half_width_deg must be finite and in (0, 90]")

    req_q_max: float | None = None
    if q_window is not None:
        q_min, q_max = _validate_q_window(q_window)
        req_q_max = q_max

    img, qx_arr, qy_arr, q_mag, chi_deg, geom_mask, inclusion, sig_arr, _ = _resolve_2d_context(
        image,
        qx,
        qy,
        valid_mask,
        geometry_cache=geometry_cache,
        sigma=sigma,
        required_q_max=req_q_max,
    )
    q_unit_label, q_scale_nm_inv, unit_caveat = _q_unit_info(q_unit)

    # Pre-filter strictly to pixels inside [q_min, q_max] BEFORE arctan2, binning, and sector checks
    if q_window is not None and q_mag is None:
        in_box = (np.abs(qx_arr) <= q_max) & (np.abs(qy_arr) <= q_max)
        if not np.any(in_box):
            if not np.any(np.isfinite(qx_arr) & np.isfinite(qy_arr)):
                raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
            q_c = np.empty(0, dtype=float)
            qx_c = np.empty(0, dtype=float)
            qy_c = np.empty(0, dtype=float)
            img_c = np.empty(0, dtype=float)
            inc_c = np.empty(0, dtype=bool)
            sig_c = np.empty(0, dtype=float) if sig_arr is not None else None
            chi_c = None
        else:
            bx = qx_arr[in_box]
            by = qy_arr[in_box]
            bq = np.hypot(bx, by)
            in_q = np.isfinite(bq) & (bq >= q_min) & (bq <= q_max)
            q_c = bq[in_q]
            qx_c = bx[in_q]
            qy_c = by[in_q]
            img_c = img[in_box][in_q]
            inc_c = inclusion[in_box][in_q]
            sig_c = sig_arr[in_box][in_q] if sig_arr is not None else None
            chi_c = None
    else:
        assert q_mag is not None and geom_mask is not None
        if q_window is None:
            usable_full = geom_mask & inclusion & np.isfinite(img)
            obs_q = q_mag[usable_full]
            if obs_q.size == 0:
                obs_q = q_mag[geom_mask]
            q_min = float(np.min(obs_q))
            q_max = float(np.max(obs_q))
            if q_min == q_max:
                half = max(abs(q_min), 1.0) * 1.0e-6
                q_min = max(0.0, q_min - half)
                q_max = q_max + half
        in_q = geom_mask & (q_mag >= q_min) & (q_mag <= q_max)
        q_c = q_mag[in_q]
        qx_c = qx_arr[in_q]
        qy_c = qy_arr[in_q]
        img_c = img[in_q]
        inc_c = inclusion[in_q]
        sig_c = sig_arr[in_q] if sig_arr is not None else None
        chi_c = np.mod(chi_deg[in_q], 360.0) if chi_deg is not None else None

    obs_c = inc_c & np.isfinite(img_c)
    if sig_c is not None and np.any(obs_c & np.isfinite(sig_c) & (sig_c < 0.0)):
        raise ValueError("sigma must be non-negative on valid observed pixels")

    q_edges = np.linspace(q_min, q_max, n_bins + 1, dtype=float)
    inv_dq = float(n_bins) / max(q_max - q_min, 1.0e-30)
    q_idx_c = np.floor((q_c - q_min) * inv_dq).astype(np.int64)
    np.clip(q_idx_c, 0, n_bins - 1, out=q_idx_c)

    def _bin_radial_1d(
        cand_sel: np.ndarray | None,
        obs_sel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        c_idx = q_idx_c if cand_sel is None else q_idx_c[cand_sel]
        o_idx = q_idx_c[obs_sel]
        o_img = img_c[obs_sel]
        c_cnt = np.bincount(c_idx, minlength=n_bins).astype(np.int64)
        o_cnt = np.bincount(o_idx, minlength=n_bins).astype(np.int64)
        o_sum = np.bincount(o_idx, weights=o_img, minlength=n_bins).astype(float)
        o_mean = np.full(n_bins, np.nan, dtype=float)
        np.divide(o_sum, o_cnt, out=o_mean, where=o_cnt > 0)
        z_mask = np.abs(o_img) <= 1.0e-12
        z_cnt = np.bincount(o_idx[z_mask], minlength=n_bins).astype(np.int64) if np.any(z_mask) else np.zeros(n_bins, dtype=np.int64)
        z_frac = np.divide(z_cnt, o_cnt, out=np.zeros(n_bins, dtype=float), where=o_cnt > 0)

        se_rad: np.ndarray | None = None
        if sig_c is not None:
            o_sig = sig_c[obs_sel]
            sig_ok = np.isfinite(o_sig) & (o_sig >= 0.0)
            if np.any(sig_ok):
                s_idx = o_idx[sig_ok]
                s_val = o_sig[sig_ok]
                s_cnt = np.bincount(s_idx, minlength=n_bins).astype(float)
                s_var = np.bincount(s_idx, weights=s_val * s_val, minlength=n_bins).astype(float)
                se_rad = np.full(n_bins, np.nan, dtype=float)
                good_s = s_cnt > 0
                se_rad[good_s] = np.sqrt(s_var[good_s]) / s_cnt[good_s]
        return o_mean, o_cnt, c_cnt, z_frac, se_rad

    # Full-azimuth channel
    full_I, full_cnt, full_cand, full_zfrac, full_se = _bin_radial_1d(None, obs_c)
    full_summary = _summarize_radial_channel(
        q_edges,
        full_I,
        full_cnt,
        full_cand,
        q_scale_nm_inv=q_scale_nm_inv,
        q_unit_label=q_unit_label,
        channel_name="full_azimuth",
        zero_pixel_fraction=full_zfrac,
        background_level=background_level,
        intensity_mode=mode,
        sigma_radial=full_se,
    )

    # Azimuthal angles computed strictly on the pre-filtered 1D in-window pixels
    angles_c = chi_c if chi_c is not None else np.mod(np.degrees(np.arctan2(qy_c, qx_c)), 360.0)
    pos_q = q_c > 0.0

    # Lobe-sectored channel
    if lobe_angles_deg is not None:
        sector_angles = [float(a) % 360.0 for a in lobe_angles_deg if math.isfinite(float(a))]
        sector_source = "user_supplied"
    else:
        pos_obs = obs_c & pos_q
        sector_angles, sector_source = _detect_wing_angles_from_1d_pixels(
            angles_c[pos_obs],
            img_c[pos_obs],
            nominal_draw_axis_deg=nominal_draw_axis_deg,
        )

    if sector_angles and angles_c.size > 0:
        in_sector = np.zeros(angles_c.shape, dtype=bool)
        for ray_deg in sector_angles:
            in_sector |= np.abs(((angles_c - ray_deg + 180.0) % 360.0) - 180.0) <= half_w
        in_sector &= pos_q
        lobe_I, lobe_cnt, lobe_cand, lobe_zfrac, lobe_se = _bin_radial_1d(in_sector, obs_c & in_sector)
        lobe_summary = _summarize_radial_channel(
            q_edges,
            lobe_I,
            lobe_cnt,
            lobe_cand,
            q_scale_nm_inv=q_scale_nm_inv,
            q_unit_label=q_unit_label,
            channel_name="lobe_sectored",
            zero_pixel_fraction=lobe_zfrac,
            background_level=background_level,
            intensity_mode=mode,
            sigma_radial=lobe_se,
        )
        lobe_summary["lobe_angles_deg"] = sector_angles
        lobe_summary["sector_half_width_deg"] = half_w
        lobe_summary["lobe_angle_source"] = sector_source
    else:
        lobe_summary = {
            "status": "unavailable",
            "reason": sector_source,
            "lobe_angles_deg": [],
            "sector_half_width_deg": half_w,
            "lobe_angle_source": sector_source,
            "direct_peak": None,
            "kratky_peak": None,
            "q_win_invariant": None,
            "q_win_bg_subtracted": None,
            "q_win_stat_stderr": None,
            "sigma_radial": None,
        }

    # Primary top-level peaks: prefer lobe-sectored when detected and populated, else full-azimuth
    use_lobe = (
        lobe_summary.get("status") == "estimated"
        and isinstance(lobe_summary.get("direct_peak"), dict)
        and isinstance(lobe_summary.get("kratky_peak"), dict)
    )
    primary = lobe_summary if use_lobe else full_summary
    primary_source = "lobe_sectored" if use_lobe else "full_azimuth"

    direct_pk = primary["direct_peak"]
    kratky_pk = primary["kratky_peak"]

    # Optional 1D cosine correlation function K(r) from full-azimuth observed Kratky
    correlation = None
    if compute_correlation:
        q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
        dq_bins = q_edges[1:] - q_edges[:-1]
        kratky_full = np.where(np.isfinite(full_I) & (full_cnt > 0), (q_centers ** 2) * full_I, np.nan)
        correlation = _compute_cosine_correlation_1d(
            q_centers,
            kratky_full,
            dq_bins,
            q_scale_nm_inv=q_scale_nm_inv,
            q_unit_label=q_unit_label,
            r_max_nm=r_max_nm,
            n_r=n_r,
        )

    caveats = [
        "Q_win is a window-bounded scattering invariant proxy over observed [q_min, q_max] only; "
        "it does not estimate scattering-length-density contrast delta_rho or volume fraction.",
        "Ring/sector q* and L* = 2*pi/q* are distinct from ellipse geometry (a, b/a, theta_deg, conditional Ln/Lz).",
        "Kratky peak q_star_kratky is extracted from high-q diffuse-floor-subtracted q^2 * max(I(q) - I_bg, 0); "
        "raw q^2 * I(q) is retained in full_azimuth/lobe_sectored['q2_I_q'] and Q_win.",
    ]
    if unit_caveat is not None:
        caveats.append(unit_caveat)
    if mode == "raw":
        caveats.append(
            "intensity_mode='raw': q_win_invariant and q_win_bg_subtracted are withheld (None) "
            "because uncalibrated raw detector counts cannot produce an absolute scattering invariant."
        )

    flags: list[str] = []
    if mode == "raw":
        flags.append("raw_intensity_mode_q_win_withheld")
    if q_scale_nm_inv is None:
        flags.append("uncalibrated_pixel_q" if q_unit_label == "pixel-q" else "spacing_unavailable_unknown_q_unit")
    if direct_pk["edge_hit"]:
        flags.append(f"direct_peak_edge_hit_{direct_pk['edge_hit_side']}")
    if kratky_pk["edge_hit"]:
        flags.append(f"kratky_peak_edge_hit_{kratky_pk['edge_hit_side']}")
    if (full_summary.get("radial_bin_coverage") or 0.0) < 1.0:
        flags.append("incomplete_radial_coverage")
    if (full_summary.get("n_outer_zero_padded_bins_excluded") or 0) > 0:
        flags.append("outer_zero_padded_shells_excluded")

    direct_low_q_edge_hit = bool(direct_pk["edge_hit"] and direct_pk["edge_hit_side"] == "low_q")
    kratky_low_q_edge_hit = bool(kratky_pk["edge_hit"] and kratky_pk["edge_hit_side"] == "low_q")
    has_sigma = full_se is not None
    sys_k = "unavailable" if systematic_k_uncertainty is None else systematic_k_uncertainty

    return json_safe({
        "status": "estimated" if full_summary["status"] == "estimated" else "unavailable",
        "q_unit": q_unit_label,
        "intensity_mode": mode,
        "q_window": [q_min, q_max],
        "peak_channel_source": primary_source,
        "I_bg_estimate": primary.get("I_bg_estimate"),
        "I_bg_noise_mad": primary.get("I_bg_noise_mad"),
        "kratky_background_subtracted": True,
        "q_star_direct": direct_pk["q_star"],
        "L_star_direct_nm": direct_pk["L_star_nm"],
        "fwhm_direct_q": direct_pk["fwhm_q"],
        "prominence_direct": direct_pk["prominence"],
        "edge_hit_direct": direct_pk["edge_hit"],
        "edge_hit_direct_side": direct_pk["edge_hit_side"],
        "direct_low_q_edge_hit": direct_low_q_edge_hit,
        "q_star_kratky": kratky_pk["q_star"],
        "L_star_kratky_nm": kratky_pk["L_star_nm"],
        "fwhm_kratky_q": kratky_pk["fwhm_q"],
        "prominence_kratky": kratky_pk["prominence"],
        "edge_hit_kratky": kratky_pk["edge_hit"],
        "edge_hit_kratky_side": kratky_pk["edge_hit_side"],
        "kratky_low_q_edge_hit": kratky_low_q_edge_hit,
        "q_win_invariant": full_summary["q_win_invariant"],
        "q_win_bg_subtracted": full_summary["q_win_bg_subtracted"],
        "q_win_stat_stderr": full_summary["q_win_stat_stderr"],
        "sigma_radial": full_summary["sigma_radial"],
        "sigma_source": "supplied_statistical_sigma" if has_sigma else "unavailable",
        "systematic_k_uncertainty": sys_k,
        "q_win_units": f"intensity * ({q_unit_label})^3" if mode == "calibrated" else None,
        "full_azimuth": full_summary,
        "lobe_sectored": lobe_summary,
        "correlation_function": correlation,
        "flags": list(dict.fromkeys(flags)),
        "caveats": caveats,
    })


def _radial_metrics_from_1d_profile(
    q_edges: np.ndarray,
    I_q: np.ndarray,
    counts: np.ndarray,
    candidate_counts: np.ndarray,
    *,
    q_unit: str,
    intensity_mode: str = "calibrated",
) -> dict[str, Any]:
    """Lightweight helper used by ``report_measurements.measure_frame`` on pre-binned profiles."""
    q_unit_label, q_scale_nm_inv, _ = _q_unit_info(q_unit)
    channel = _summarize_radial_channel(
        q_edges,
        I_q,
        counts,
        candidate_counts,
        q_scale_nm_inv=q_scale_nm_inv,
        q_unit_label=q_unit_label,
        channel_name="frame_radial",
        intensity_mode=intensity_mode,
    )
    return {
        "q_win_invariant": channel["q_win_invariant"],
        "q_star_direct": channel["direct_peak"]["q_star"],
        "L_star_direct_nm": channel["direct_peak"]["L_star_nm"],
        "edge_hit_direct": channel["direct_peak"]["edge_hit"],
        "q_star_kratky": channel["kratky_peak"]["q_star"],
        "L_star_kratky_nm": channel["kratky_peak"]["L_star_nm"],
        "edge_hit_kratky": channel["kratky_peak"]["edge_hit"],
    }
