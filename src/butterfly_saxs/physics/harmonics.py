"""Azimuthal harmonics, Hermans orientation order, and C2-bridge four-lobe fitting."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from ..serialization import json_safe
from ._common import (
    _FWHM_PER_SIGMA,
    _bin_azimuthal_from_2d,
    _finite,
    _finite_list,
    _normalise_intensity_mode,
    _periodic_dist_deg,
    _q_unit_info,
    _resolve_2d_context,
    _unwrap_near_deg,
    _validate_q_window,
)


def _hermans_order(
    angle_deg: np.ndarray,
    intensity: np.ndarray,
    axis_deg: float,
) -> tuple[float | None, float | None]:
    """Compute Hermans f2 (P2) and 4-fold Legendre f4 (P4) relative to ``axis_deg``."""
    w_int = np.maximum(intensity, 0.0)
    beta = np.deg2rad(angle_deg - float(axis_deg))
    weight = w_int * np.abs(np.sin(beta))
    denom = float(np.sum(weight))
    if not math.isfinite(denom) or denom <= 0.0:
        return None, None
    cos2 = np.cos(beta) ** 2
    mean_cos2 = float(np.sum(weight * cos2) / denom)
    mean_cos4 = float(np.sum(weight * (cos2 * cos2)) / denom)
    f2 = 0.5 * (3.0 * mean_cos2 - 1.0)
    f4 = 0.125 * (35.0 * mean_cos4 - 30.0 * mean_cos2 + 3.0)
    return _finite(f2), _finite(f4)


def _friedel_quadrant_ratios(
    angle_deg: np.ndarray,
    intensity: np.ndarray,
    observed: np.ndarray,
) -> tuple[float | None, float | None, int, int]:
    """Compute opposite-quadrant Friedel symmetry ratios from matched observed pairs."""
    n = angle_deg.size
    if n < 4 or not np.any(observed):
        return None, None, 0, 0

    angles = np.mod(angle_deg, 360.0)
    order = np.argsort(angles)
    angles = angles[order]
    intensity = intensity[order]
    observed = observed[order]
    # Detect uniform even-bin grid for exact O(n) opposite-bin pairing
    diffs = np.diff(angles)
    is_uniform_even = (
        n % 2 == 0
        and n >= 4
        and np.all(np.isfinite(diffs))
        and np.allclose(diffs, 360.0 / n, rtol=1e-3, atol=1e-3)
    )
    if is_uniform_even:
        half = n // 2
        opp = (np.arange(n) + half) % n
        both = observed & observed[opp]
        q1_sel = both & (angles >= 0.0) & (angles < 90.0)
        q2_sel = both & (angles >= 90.0) & (angles < 180.0)
        n13 = int(np.count_nonzero(q1_sel))
        n24 = int(np.count_nonzero(q2_sel))
        r13 = None
        if n13 > 0:
            s1 = float(np.sum(intensity[q1_sel]))
            s3 = float(np.sum(intensity[opp[q1_sel]]))
            if s3 > 0.0 and s1 >= 0.0:
                r13 = _finite(s1 / s3)
        r24 = None
        if n24 > 0:
            s2 = float(np.sum(intensity[q2_sel]))
            s4 = float(np.sum(intensity[opp[q2_sel]]))
            if s4 > 0.0 and s2 >= 0.0:
                r24 = _finite(s2 / s4)
        return r13, r24, n13, n24

    # General fallback for irregular or odd-sized angular grids: match within half a bin step
    obs_idx = np.flatnonzero(observed)
    obs_ang = angles[obs_idx]
    obs_val = intensity[obs_idx]
    sorted_ang = np.sort(obs_ang)
    tol = 0.55 * float(np.median(np.diff(sorted_ang))) if sorted_ang.size > 1 else 2.5
    tol = min(max(tol, 0.5), 10.0)

    def _pair_ratio(lo_deg: float, hi_deg: float) -> tuple[float | None, int]:
        sel_a = np.flatnonzero((obs_ang >= lo_deg) & (obs_ang < hi_deg))
        sel_b = np.flatnonzero((obs_ang >= lo_deg + 180.0) & (obs_ang < hi_deg + 180.0))
        if sel_a.size == 0 or sel_b.size == 0:
            return None, 0
        used_b: set[int] = set()
        sum_a = 0.0
        sum_b = 0.0
        matched = 0
        for ia in sel_a:
            target = (obs_ang[ia] + 180.0) % 360.0
            dists = np.abs(_periodic_dist_deg(obs_ang[sel_b], target, 360.0))
            best_local = int(np.argmin(dists))
            ib = int(sel_b[best_local])
            if dists[best_local] <= tol and ib not in used_b:
                used_b.add(ib)
                sum_a += float(obs_val[ia])
                sum_b += float(obs_val[ib])
                matched += 1
        if matched == 0 or sum_b <= 0.0 or sum_a < 0.0:
            return None, matched
        return _finite(sum_a / sum_b), matched

    r13, n13 = _pair_ratio(0.0, 90.0)
    r24, n24 = _pair_ratio(90.0, 180.0)
    return r13, r24, n13, n24


def _detect_folded_lobe_opening(
    angles_obs: np.ndarray,
    intensity_obs: np.ndarray,
    axis_deg: float,
) -> tuple[float | None, float]:
    """Fold observed intensity around ``axis_deg`` after detrending ``c0 + c2*cos(2*d)``.

    Returns ``(alpha_est_deg_or_None, interior_contrast)`` where ``alpha_est_deg``
    lies in ``(6, 84)`` if a genuine interior 4-lobe peak exists.
    """
    if angles_obs.size < 8:
        return None, 0.0
    delta_rad = np.deg2rad(angles_obs - float(axis_deg))
    X = np.column_stack((np.ones_like(delta_rad), np.cos(2.0 * delta_rad)))
    if np.linalg.cond(X) > 1e6:
        return None, 0.0
    beta, *_ = np.linalg.lstsq(X, intensity_obs, rcond=None)
    resid = intensity_obs - X @ beta

    folded = np.abs(_periodic_dist_deg(angles_obs, axis_deg, 180.0))  # [0, 90]
    n_fold = 30  # 3-degree folded bins
    edges = np.linspace(0.0, 90.0, n_fold + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.searchsorted(edges, folded, side="right") - 1, 0, n_fold - 1)
    counts = np.bincount(idx, minlength=n_fold)
    sums = np.bincount(idx, weights=resid, minlength=n_fold)
    prof = np.full(n_fold, np.nan, dtype=float)
    np.divide(sums, counts, out=prof, where=counts > 0)

    valid = np.isfinite(prof)
    if np.count_nonzero(valid) < 5:
        return None, 0.0

    # Find strongest observed bin in interior (6..84 deg)
    interior = valid & (centers >= 6.0) & (centers <= 84.0)
    if not np.any(interior):
        return None, 0.0

    int_indices = np.flatnonzero(interior)
    peak_i = int(int_indices[np.argmax(prof[int_indices])])
    peak_val = float(prof[peak_i])
    val_range = float(np.nanmax(prof[valid]) - np.nanmin(prof[valid]))
    if val_range <= 1e-12:
        return None, 0.0

    # Check that it is an interior peak rather than a boundary slope
    edge_vals = [float(prof[j]) for j in (0, 1, n_fold - 2, n_fold - 1) if valid[j]]
    max_edge = max(edge_vals) if edge_vals else -np.inf
    if peak_val <= max_edge + 0.05 * val_range:
        return None, 0.0

    # Parabolic refinement when immediate neighbors are observed
    alpha_est = float(centers[peak_i])
    if 0 < peak_i < n_fold - 1 and valid[peak_i - 1] and valid[peak_i + 1]:
        y_m, y_0, y_p = float(prof[peak_i - 1]), float(prof[peak_i]), float(prof[peak_i + 1])
        denom = y_m - 2.0 * y_0 + y_p
        if denom < -1e-15:
            shift = 0.5 * (y_m - y_p) / denom
            if abs(shift) <= 1.0:
                alpha_est += shift * (90.0 / n_fold)
    return float(np.clip(alpha_est, 3.0, 87.0)), float(peak_val / max(val_range, 1e-15))


def _fit_even_harmonics_ols(
    angle_deg_obs: np.ndarray,
    intensity_obs: np.ndarray,
) -> tuple[dict[str, float | None], list[int], list[str]]:
    """Estimate even Fourier harmonics (m = 0, 2, 4, 6) strictly on observed bins via OLS."""
    flags: list[str] = []
    n_obs = angle_deg_obs.size
    if n_obs == 0:
        return {}, [], ["no_observed_angular_bins"]

    rad = np.deg2rad(angle_deg_obs)
    chosen_orders: list[int] = []
    beta: np.ndarray | None = None

    for candidate in ((2, 4, 6), (2, 4), (2,), ()):
        n_params = 1 + 2 * len(candidate)
        if n_obs < n_params:
            continue
        cols = [np.ones(n_obs, dtype=float)]
        for m in candidate:
            cols.append(np.cos(m * rad))
            cols.append(np.sin(m * rad))
        X = np.column_stack(cols)
        cond = float(np.linalg.cond(X))
        if math.isfinite(cond) and cond < 1.0e4:
            beta, *_ = np.linalg.lstsq(X, intensity_obs, rcond=None)
            chosen_orders = list(candidate)
            if len(candidate) < 3:
                flags.append("reduced_harmonic_order_due_to_mask_coverage")
            break
    if beta is None:
        flags.append("harmonic_design_ill_conditioned")
        return {"A0": _finite(np.mean(intensity_obs))}, [], flags

    out: dict[str, float | None] = {"A0": _finite(beta[0])}
    for idx, m in enumerate(chosen_orders):
        cm = float(beta[1 + 2 * idx])
        sm = float(beta[2 + 2 * idx])
        out[f"c{m}"] = _finite(cm)
        out[f"s{m}"] = _finite(sm)
        out[f"A{m}"] = _finite(math.hypot(cm, sm))
    return out, chosen_orders, flags


def compute_azimuthal_harmonics_and_order(
    image: Any = None,
    qx: Any = None,
    qy: Any = None,
    *,
    valid_mask: Any = None,
    q_window: tuple[float, float] | None = None,
    q_unit: str = "unknown",
    n_bins: int = 360,
    nominal_draw_axis_deg: float = 90.0,
    bright_axis_convention: str = "draw",
    expected_bright_axis_deg: float | None = None,
    angle_deg: Any = None,
    intensity: Any = None,
    counts: Any = None,
    reference_phase_deg: float | None = None,
    geometry_cache: Any = None,
    intensity_mode: str = "calibrated",
    sigma: Any = None,
) -> dict[str, Any]:
    """Compute even Fourier harmonics, Friedel symmetry bisector, and Hermans order parameters.

    Accepts either 2D arrays ``(image, qx, qy)`` (with optional ``valid_mask``,
    ``q_window``, and ``geometry_cache``) OR a pre-binned 1D angular profile
    ``(angle_deg, intensity)``. All quantities are derived strictly from
    observed pixels/bins; masked gaps are never filled or mirrored.

    ``bright_axis_convention`` controls the expected axis for ``orthogonal_rotation_detected``
    and ``mounting_tilt_deg`` when ``expected_bright_axis_deg`` is omitted:
    - ``"draw"`` (default): expects the bright C2 bridge / acute lobe bisector along
      ``nominal_draw_axis_deg``.
    - ``"reference"``: expects the bright C2 bridge / acute lobe bisector along
      ``(nominal_draw_axis_deg - 90) % 180`` (WingSAXS ``reference_axis_deg``, matching
      equatorial-bridged butterflies such as NO50S with vertical wire ``draw_axis_deg=90``).
    """
    _normalise_intensity_mode(intensity_mode)
    q_unit_label, _, _ = _q_unit_info(q_unit)
    nominal_axis = float(nominal_draw_axis_deg) % 180.0
    nominal_reference_axis = float((nominal_axis - 90.0) % 180.0)
    convention = str(bright_axis_convention or "draw").strip().lower()
    if convention not in {"draw", "reference"}:
        raise ValueError("bright_axis_convention must be 'draw' or 'reference'")
    if expected_bright_axis_deg is not None:
        exp_val = float(expected_bright_axis_deg)
        if not math.isfinite(exp_val):
            raise ValueError("expected_bright_axis_deg must be finite when provided")
        expected_axis = float(exp_val % 180.0)
    else:
        expected_axis = nominal_axis if convention == "draw" else nominal_reference_axis
    flags: list[str] = []
    quadrant_means: dict[str, float | None] = {"QI": None, "QII": None, "QIII": None, "QIV": None}
    q_bounds: tuple[float | None, float | None] = (None, None)

    if image is not None:
        if geometry_cache is None and (qx is None or qy is None):
            raise ValueError("qx and qy must be supplied when image is provided")
        if isinstance(n_bins, (bool, np.bool_)) or int(n_bins) < 8:
            raise ValueError("n_bins must be an integer >= 8")
        req_q_max = _validate_q_window(q_window)[1] if q_window is not None else None
        img, qx_arr, qy_arr, q_mag, chi_deg, geom_mask, inclusion, _, _ = _resolve_2d_context(
            image,
            qx,
            qy,
            valid_mask,
            geometry_cache=geometry_cache,
            sigma=sigma,
            required_q_max=req_q_max,
        )
        ang, prof, cnts, geom_cnts, quadrant_means, q_bounds = _bin_azimuthal_from_2d(
            img,
            qx_arr,
            qy_arr,
            geom_mask,
            None,
            q_window=q_window,
            n_bins=int(n_bins),
            q_mag=q_mag,
            chi_deg=chi_deg,
            inclusion=inclusion,
        )
        observed = np.isfinite(ang) & np.isfinite(prof) & (cnts > 0)
        geom_total = int(np.count_nonzero(geom_cnts > 0))
        obs_total = int(np.count_nonzero(observed))
        angular_coverage = float(obs_total / geom_total) if geom_total > 0 else 0.0
    elif angle_deg is not None and intensity is not None:
        ang = np.asarray(angle_deg, dtype=float)
        prof = np.asarray(intensity, dtype=float)
        if ang.ndim != 1 or prof.ndim != 1 or ang.shape != prof.shape or ang.size == 0:
            raise ValueError("angle_deg and intensity must be non-empty matching 1D arrays")
        observed = np.isfinite(ang) & np.isfinite(prof)
        if counts is not None:
            cnts = np.asarray(counts)
            if cnts.shape != ang.shape:
                raise ValueError("counts must match angle_deg shape")
            observed &= cnts > 0
        else:
            cnts = observed.astype(np.int64)
        if valid_mask is not None:
            vm = np.asarray(valid_mask, dtype=bool)
            if vm.shape != ang.shape:
                raise ValueError("valid_mask must match angle_deg shape")
            observed &= vm
        angular_coverage = float(np.count_nonzero(observed) / ang.size)
        ang_mod = np.mod(ang, 360.0)
        for label, lo, hi in (("QI", 0.0, 90.0), ("QII", 90.0, 180.0), ("QIII", 180.0, 270.0), ("QIV", 270.0, 360.0)):
            sel = observed & (ang_mod >= lo) & (ang_mod < hi)
            quadrant_means[label] = _finite(np.mean(prof[sel])) if np.any(sel) else None
    else:
        raise ValueError("supply either (image, qx, qy) or (angle_deg, intensity)")

    if angular_coverage < 1.0:
        flags.append("incomplete_azimuthal_coverage")

    n_obs = int(np.count_nonzero(observed))
    if n_obs == 0:
        flags.append("no_observed_angular_bins")
        return json_safe({
            "status": "unavailable",
            "q_unit": q_unit_label,
            "q_window": list(q_bounds) if q_bounds[0] is not None else None,
            "n_observed_bins": 0,
            "n_total_bins": int(ang.size),
            "angular_coverage": 0.0,
            "nominal_draw_axis_deg": nominal_axis,
            "nominal_reference_axis_deg": nominal_reference_axis,
            "bright_axis_convention": convention,
            "expected_bright_axis_deg": expected_axis,
            "A0": None,
            "A2": None,
            "A4": None,
            "A6": None,
            "a2": None,
            "a4": None,
            "a6": None,
            "a4_over_a2": None,
            "phi2_deg": None,
            "phi4_deg": None,
            "phi6_deg": None,
            "phi2_unwrapped_draw_deg": None,
            "phi4_unwrapped_draw_deg": None,
            "friedel_bisector_deg": None,
            "orthogonal_axis_deg": None,
            "bright_bridge_axis_deg": None,
            "dark_notch_axis_deg": None,
            "angle_to_draw_axis_deg": None,
            "angle_to_reference_axis_deg": None,
            "mounting_tilt_deg": None,
            "orthogonal_rotation_detected": False,
            "wing_half_opening_estimate_deg": None,
            "f2": None,
            "f4": None,
            "f2_nominal": None,
            "f4_nominal": None,
            "q1_over_q3": None,
            "q2_over_q4": None,
            "friedel_matched_bins_q1_q3": 0,
            "friedel_matched_bins_q2_q4": 0,
            "quadrant_means": quadrant_means,
            "flags": list(dict.fromkeys(flags)),
        })

    ang_obs = np.mod(ang[observed], 360.0)
    y_obs = prof[observed]

    harm, orders, harm_flags = _fit_even_harmonics_ols(ang_obs, y_obs)
    flags.extend(harm_flags)

    A0 = harm.get("A0")
    A2 = harm.get("A2")
    A4 = harm.get("A4")
    A6 = harm.get("A6")
    c2, s2 = harm.get("c2"), harm.get("s2")
    c4, s4 = harm.get("c4"), harm.get("s4")
    c6, s6 = harm.get("c6"), harm.get("s6")

    scale = max(abs(A0 or 0.0), float(np.std(y_obs)), 1.0e-15)
    amp_tol = 1.0e-10 * scale

    a2 = _finite(A2 / A0) if (A0 is not None and A0 > 0.0 and A2 is not None) else None
    a4 = _finite(A4 / A0) if (A0 is not None and A0 > 0.0 and A4 is not None) else None
    a6 = _finite(A6 / A0) if (A0 is not None and A0 > 0.0 and A6 is not None) else None
    a4_over_a2 = _finite(A4 / A2) if (A2 is not None and A4 is not None and A2 > amp_tol) else None

    phi2_raw = (
        float((0.5 * math.degrees(math.atan2(s2, c2))) % 180.0)
        if (c2 is not None and s2 is not None and A2 is not None and A2 > amp_tol)
        else None
    )
    phi4_raw = (
        float((0.25 * math.degrees(math.atan2(s4, c4))) % 90.0)
        if (c4 is not None and s4 is not None and A4 is not None and A4 > amp_tol)
        else None
    )
    phi6_raw = (
        float(((1.0 / 6.0) * math.degrees(math.atan2(s6, c6))) % 60.0)
        if (c6 is not None and s6 is not None and A6 is not None and A6 > amp_tol)
        else None
    )

    if reference_phase_deg is not None and math.isfinite(float(reference_phase_deg)):
        ref_deg = float(reference_phase_deg)
        phi2_deg = _unwrap_near_deg(phi2_raw, ref_deg, 180.0)
        phi4_deg = _unwrap_near_deg(phi4_raw, ref_deg, 90.0)
        phi6_deg = _unwrap_near_deg(phi6_raw, ref_deg, 60.0)
    else:
        phi2_deg = phi2_raw
        phi4_deg = phi4_raw
        phi6_deg = phi6_raw

    phi2_unwrapped_draw = _unwrap_near_deg(phi2_raw, nominal_axis, 180.0)
    phi4_unwrapped_draw = _unwrap_near_deg(phi4_raw, nominal_axis, 90.0)

    # Mirror-symmetry bisector from even Fourier harmonics
    aniso_amp = math.sqrt(sum((harm.get(f"A{m}") or 0.0) ** 2 for m in orders))
    friedel_bisector_deg: float | None = None
    orthogonal_axis_deg: float | None = None
    wing_half_opening_est: float | None = None

    if orders and aniso_amp > 1.0e-8 * scale:
        harm_terms = [
            (m, float(harm[f"c{m}"]), float(harm[f"s{m}"]))
            for m in orders
            if harm.get(f"c{m}") is not None and harm.get(f"s{m}") is not None
        ]

        def _mirror_loss(psi_rad: float) -> float:
            total = 0.0
            for m, cm, sm in harm_terms:
                val = -cm * math.sin(m * psi_rad) + sm * math.cos(m * psi_rad)
                total += val * val
            return total

        grid_rad = np.linspace(0.0, 0.5 * math.pi, 360, endpoint=False)
        losses = np.zeros_like(grid_rad)
        for m, cm, sm in harm_terms:
            val = -cm * np.sin(m * grid_rad) + sm * np.cos(m * grid_rad)
            losses += val * val
        best_idx = int(np.argmin(losses))
        step_rad = 0.5 * math.pi / 360.0
        lo_rad = float(grid_rad[best_idx]) - 1.5 * step_rad
        hi_rad = float(grid_rad[best_idx]) + 1.5 * step_rad
        opt = minimize_scalar(_mirror_loss, bounds=(lo_rad, hi_rad), method="bounded")
        psi_a_deg = float(math.degrees(float(opt.x)) % 180.0)
        psi_b_deg = float((psi_a_deg + 90.0) % 180.0)

        # Disambiguate between the two orthogonal mirror axes (psi_a vs psi_b)
        alpha_a, contrast_a = _detect_folded_lobe_opening(ang_obs, y_obs, psi_a_deg)
        if alpha_a is not None and contrast_a > 0.12 and (A4 or 0.0) > 0.02 * aniso_amp:
            if alpha_a < 44.0:
                friedel_bisector_deg = psi_a_deg
                wing_half_opening_est = alpha_a
            elif alpha_a > 46.0:
                friedel_bisector_deg = psi_b_deg
                wing_half_opening_est = 90.0 - alpha_a
            else:
                # Near-square 45 deg butterfly: tie-break by proximity to expected bright axis
                da = abs(float(_periodic_dist_deg(psi_a_deg, expected_axis, 180.0)))
                db = abs(float(_periodic_dist_deg(psi_b_deg, expected_axis, 180.0)))
                friedel_bisector_deg = psi_a_deg if da <= db else psi_b_deg
                wing_half_opening_est = alpha_a if da <= db else (90.0 - alpha_a)
        elif c2 is not None and s2 is not None and A2 is not None and A2 > amp_tol:
            # 2-lobe / nematic pattern: pick the mirror axis where C2 is maximal
            proj_a = c2 * math.cos(2.0 * math.radians(psi_a_deg)) + s2 * math.sin(2.0 * math.radians(psi_a_deg))
            friedel_bisector_deg = psi_a_deg if proj_a >= 0.0 else psi_b_deg
        else:
            da = abs(float(_periodic_dist_deg(psi_a_deg, expected_axis, 180.0)))
            db = abs(float(_periodic_dist_deg(psi_b_deg, expected_axis, 180.0)))
            friedel_bisector_deg = psi_a_deg if da <= db else psi_b_deg

        orthogonal_axis_deg = float((friedel_bisector_deg + 90.0) % 180.0)
    else:
        flags.append("isotropic_symmetry_axis_undefined")

    angle_to_draw_axis_deg: float | None = None
    angle_to_reference_axis_deg: float | None = None
    mounting_tilt_deg: float | None = None
    bright_bridge_axis_deg: float | None = None
    dark_notch_axis_deg: float | None = None
    orthogonal_rotation_detected = False
    if friedel_bisector_deg is not None:
        angle_to_draw_axis_deg = float(_periodic_dist_deg(friedel_bisector_deg, nominal_axis, 180.0))
        angle_to_reference_axis_deg = float(_periodic_dist_deg(friedel_bisector_deg, nominal_reference_axis, 180.0))
        mounting_tilt_deg = float(_periodic_dist_deg(friedel_bisector_deg, expected_axis, 180.0))
        if abs(mounting_tilt_deg) >= 65.0:
            orthogonal_rotation_detected = True
            flags.append("orthogonal_symmetry_axis_rotation")
        bright_bridge_axis_deg = phi2_raw if phi2_raw is not None else friedel_bisector_deg
        dark_notch_axis_deg = float((bright_bridge_axis_deg + 90.0) % 180.0)

    f2_nominal, f4_nominal = _hermans_order(ang_obs, y_obs, nominal_axis)
    if friedel_bisector_deg is not None:
        f2_sym, f4_sym = _hermans_order(ang_obs, y_obs, friedel_bisector_deg)
    else:
        # For an isotropic distribution P2 and P4 are axis-independent (~0)
        f2_sym, f4_sym = f2_nominal, f4_nominal

    q1_over_q3, q2_over_q4, n13, n24 = _friedel_quadrant_ratios(ang, prof, observed)
    if n13 == 0 or n24 == 0:
        flags.append("incomplete_friedel_opposite_quadrant_pairs")

    return json_safe({
        "status": "estimated",
        "q_unit": q_unit_label,
        "q_window": list(q_bounds) if q_bounds[0] is not None else None,
        "n_observed_bins": n_obs,
        "n_total_bins": int(ang.size),
        "angular_coverage": _finite(angular_coverage),
        "nominal_draw_axis_deg": nominal_axis,
        "nominal_reference_axis_deg": nominal_reference_axis,
        "bright_axis_convention": convention,
        "expected_bright_axis_deg": expected_axis,
        "A0": A0,
        "A2": A2,
        "A4": A4,
        "A6": A6,
        "a2": a2,
        "a4": a4,
        "a6": a6,
        "a4_over_a2": a4_over_a2,
        "phi2_deg": _finite(phi2_deg),
        "phi4_deg": _finite(phi4_deg),
        "phi6_deg": _finite(phi6_deg),
        "phi2_unwrapped_draw_deg": _finite(phi2_unwrapped_draw),
        "phi4_unwrapped_draw_deg": _finite(phi4_unwrapped_draw),
        "friedel_bisector_deg": _finite(friedel_bisector_deg),
        "orthogonal_axis_deg": _finite(orthogonal_axis_deg),
        "bright_bridge_axis_deg": _finite(bright_bridge_axis_deg),
        "dark_notch_axis_deg": _finite(dark_notch_axis_deg),
        "angle_to_draw_axis_deg": _finite(angle_to_draw_axis_deg),
        "angle_to_reference_axis_deg": _finite(angle_to_reference_axis_deg),
        "mounting_tilt_deg": _finite(mounting_tilt_deg),
        "orthogonal_rotation_detected": orthogonal_rotation_detected,
        "wing_half_opening_estimate_deg": _finite(wing_half_opening_est),
        "f2": f2_sym,
        "f4": f4_sym,
        "f2_nominal": f2_nominal,
        "f4_nominal": f4_nominal,
        "q1_over_q3": q1_over_q3,
        "q2_over_q4": q2_over_q4,
        "friedel_matched_bins_q1_q3": n13,
        "friedel_matched_bins_q2_q4": n24,
        "quadrant_means": quadrant_means,
        "flags": list(dict.fromkeys(flags)),
        "definitions": {
            "harmonics": "Truncated even Fourier OLS on observed angular bins only: A0 + sum_{m=2,4,6} A_m cos(m*(chi - phi_m))",
            "hermans_f2_f4": "Uniaxial fiber-symmetry Legendre moments <P2(cos(beta))> and <P4(cos(beta))> weighted by max(I, 0)*|sin(beta)| over observed bins",
            "friedel_ratios": "Sum(I_obs(Q1))/Sum(I_obs(Q3)) and Sum(I_obs(Q2))/Sum(I_obs(Q4)) over matched opposite observed angular bins only",
        },
    })


# ---------------------------------------------------------------------------
# 2. C2-Bridge + 4-Lobe Azimuthal Model
# ---------------------------------------------------------------------------


def _butterfly_lobe_centers_deg(chi_sym_deg: float, alpha_deg: float) -> tuple[float, float, float, float]:
    return (
        float((chi_sym_deg - alpha_deg) % 360.0),
        float((chi_sym_deg + alpha_deg) % 360.0),
        float((chi_sym_deg + 180.0 - alpha_deg) % 360.0),
        float((chi_sym_deg + 180.0 + alpha_deg) % 360.0),
    )


def evaluate_c2_bridge_four_lobe(
    angle_deg: Any,
    *,
    c0: float,
    c2: float,
    chi_sym_deg: float,
    alpha_deg: float,
    lobe_amplitudes: Sequence[float],
    lobe_sigmas_deg: Sequence[float] | float,
) -> np.ndarray:
    """Evaluate the C2-bridge + 4-lobe periodic butterfly azimuthal model."""
    angles = np.asarray(angle_deg, dtype=float)
    amps = [float(a) for a in lobe_amplitudes]
    if len(amps) != 4:
        raise ValueError("lobe_amplitudes must contain exactly 4 values")
    if isinstance(lobe_sigmas_deg, (int, float, np.floating)):
        sigmas = [float(lobe_sigmas_deg)] * 4
    else:
        sigmas = [float(s) for s in lobe_sigmas_deg]
        if len(sigmas) != 4:
            raise ValueError("lobe_sigmas_deg must be a scalar or 4-element sequence")

    centers = _butterfly_lobe_centers_deg(chi_sym_deg, alpha_deg)
    result = float(c0) + float(c2) * np.cos(2.0 * np.deg2rad(angles - float(chi_sym_deg)))
    for amp, mu, sigma in zip(amps, centers, sigmas):
        sig = max(sigma, 1.0e-6)
        dist = _periodic_dist_deg(angles, mu, 360.0)
        result = result + amp * np.exp(-0.5 * (dist / sig) ** 2)
    return result


def _canonicalize_butterfly_params(
    chi_sym_deg: float,
    alpha_deg: float,
    c2: float,
    amps: list[float],
    sigmas: list[float],
    *,
    prefer_acute_opening: bool,
    nominal_draw_axis_deg: float,
) -> tuple[float, float, float, list[float], list[float]]:
    """Map butterfly parameters to a unique canonical representation without altering I(chi)."""
    old_centers = _butterfly_lobe_centers_deg(chi_sym_deg, alpha_deg)
    new_chi = float(chi_sym_deg % 180.0)
    new_alpha = float(alpha_deg)
    new_c2 = float(c2)

    if prefer_acute_opening and new_alpha > 45.0:
        new_chi = float((new_chi + 90.0) % 180.0)
        new_alpha = float(90.0 - new_alpha)
        new_c2 = -new_c2
    elif prefer_acute_opening and abs(new_alpha - 45.0) <= 0.5:
        alt_chi = float((new_chi + 90.0) % 180.0)
        d_cur = abs(float(_periodic_dist_deg(new_chi, nominal_draw_axis_deg, 180.0)))
        d_alt = abs(float(_periodic_dist_deg(alt_chi, nominal_draw_axis_deg, 180.0)))
        if d_alt < d_cur:
            new_chi = alt_chi
            new_alpha = float(90.0 - new_alpha)
            new_c2 = -new_c2

    new_centers = _butterfly_lobe_centers_deg(new_chi, new_alpha)
    # Match each new geometric lobe center to its exact original center
    new_amps: list[float] = []
    new_sigmas: list[float] = []
    for nc in new_centers:
        dists = [abs(float(_periodic_dist_deg(nc, oc, 360.0))) for oc in old_centers]
        best = int(np.argmin(dists))
        new_amps.append(float(amps[best]))
        new_sigmas.append(float(sigmas[best]))
    return new_chi, new_alpha, new_c2, new_amps, new_sigmas


def _aic_bic(sse: float, n_obs: int, n_params: int) -> tuple[float | None, float | None]:
    if n_obs <= n_params or not math.isfinite(sse):
        return None, None
    mse = max(sse / float(n_obs), 1.0e-30)
    aic = n_obs * math.log(mse) + 2.0 * n_params
    bic = n_obs * math.log(mse) + n_params * math.log(float(n_obs))
    return _finite(aic), _finite(bic)


def fit_c2_bridge_four_lobe(
    angle_deg: Any,
    intensity: Any,
    *,
    valid_mask: Any = None,
    counts: Any = None,
    nominal_draw_axis_deg: float = 90.0,
    bright_axis_convention: str = "draw",
    initial_chi_sym_deg: float | None = None,
    initial_alpha_deg: float | None = None,
    initial_fwhm_deg: float = 30.0,
    fwhm_bounds_deg: tuple[float, float] = (2.0, 120.0),
    alpha_bounds_deg: tuple[float, float] = (3.0, 87.0),
    shared_width: bool = True,
    prefer_acute_opening: bool = True,
    loss: str = "linear",
    max_nfev: int = 5000,
    include_flat_comparison: bool = True,
) -> dict[str, Any]:
    """Fit a Friedel-symmetric C2-bridge + 4-lobe periodic model to observed angular bins.

    Model:
        ``I(chi) = c0 + c2 * cos(2*(chi - chi_sym)) + sum_{k=1..4} A_k * exp(-d(chi, mu_k)^2 / (2*sigma_k^2))``
    with ``mu = (chi_sym - alpha, chi_sym + alpha, chi_sym + 180 - alpha, chi_sym + 180 + alpha)``.

    Also fits the nested flat-baseline (``c2 = 0``) 4-lobe model on the identical
    observed bins for direct RMSE, R^2, AIC/BIC, and FWHM-bias comparison.
    """
    x = np.asarray(angle_deg, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 8:
        raise ValueError("angle_deg and intensity must be matching 1D arrays with at least 8 bins")

    convention = str(bright_axis_convention or "draw").strip().lower()
    if convention not in {"draw", "reference"}:
        raise ValueError("bright_axis_convention must be 'draw' or 'reference'")
    canonical_tie_axis_deg = (
        float(nominal_draw_axis_deg) % 180.0
        if convention == "draw"
        else float((float(nominal_draw_axis_deg) - 90.0) % 180.0)
    )

    fwhm_lo, fwhm_hi = float(fwhm_bounds_deg[0]), float(fwhm_bounds_deg[1])
    if not (math.isfinite(fwhm_lo) and math.isfinite(fwhm_hi) and 0.0 < fwhm_lo < fwhm_hi <= 360.0):
        raise ValueError("fwhm_bounds_deg must satisfy 0 < low < high <= 360")
    alpha_lo, alpha_hi = float(alpha_bounds_deg[0]), float(alpha_bounds_deg[1])
    if not (math.isfinite(alpha_lo) and math.isfinite(alpha_hi) and 0.0 < alpha_lo < alpha_hi < 90.0):
        raise ValueError("alpha_bounds_deg must satisfy 0 < low < high < 90")
    if loss not in {"linear", "soft_l1", "huber", "cauchy"}:
        raise ValueError("loss must be one of 'linear', 'soft_l1', 'huber', 'cauchy'")

    observed = np.isfinite(x) & np.isfinite(y)
    if counts is not None:
        cnt = np.asarray(counts)
        if cnt.shape != x.shape:
            raise ValueError("counts must match angle_deg shape")
        observed &= cnt > 0
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != x.shape:
            raise ValueError("valid_mask must match angle_deg shape")
        observed &= vm

    n_obs = int(np.count_nonzero(observed))
    n_params_c2 = 9 if shared_width else 12
    flags: list[str] = []
    if n_obs < x.size:
        flags.append("partial_angular_masking")

    if n_obs < n_params_c2 + 2:
        flags.append("insufficient_observed_bins_for_c2_four_lobe_fit")
        return json_safe({
            "success": False,
            "status": "unavailable",
            "message": f"Need at least {n_params_c2 + 2} observed angular bins; found {n_obs}.",
            "n_observed_bins": n_obs,
            "n_total_bins": int(x.size),
            "angular_coverage": _finite(n_obs / float(x.size)),
            "c0": None,
            "c2": None,
            "chi_sym_deg": None,
            "alpha_deg": None,
            "lobe_centers_deg": None,
            "lobe_amplitudes": None,
            "lobe_sigmas_deg": None,
            "lobe_fwhms_deg": None,
            "lobe_fwhm_deg": None,
            "lobe_supported": [False, False, False, False],
            "rmse": None,
            "r_squared": None,
            "aic": None,
            "bic": None,
            "angle_deg": _finite_list(x),
            "observed": _finite_list(np.where(observed, y, np.nan)),
            "model": [None] * int(x.size),
            "bridge_component": [None] * int(x.size),
            "lobes_component": [None] * int(x.size),
            "residual": [None] * int(x.size),
            "flat_baseline_comparison": None,
            "flags": list(dict.fromkeys(flags)),
        })

    xo = np.mod(x[observed], 360.0)
    yo = y[observed]
    y_range = max(float(np.max(yo) - np.min(yo)), 1.0e-12)
    y_abs_max = max(float(np.max(np.abs(yo))), 1.0)
    sigma_lo = fwhm_lo / _FWHM_PER_SIGMA
    sigma_hi = fwhm_hi / _FWHM_PER_SIGMA
    sigma_init = float(np.clip(float(initial_fwhm_deg) / _FWHM_PER_SIGMA, sigma_lo * 1.01, sigma_hi * 0.99))
    amp_hi = max(10.0 * y_range, 10.0 * y_abs_max, 1.0)
    c2_bound = max(5.0 * y_range, 5.0 * y_abs_max, 1.0)

    # Seed candidate symmetry axes: skip full harmonics recompute when caller already seeded chi_sym
    seed_axes: list[float] = []
    if initial_chi_sym_deg is not None and math.isfinite(float(initial_chi_sym_deg)):
        ax0 = float(initial_chi_sym_deg) % 180.0
        seed_axes.append(ax0)
        seed_axes.append((ax0 + 90.0) % 180.0)
    else:
        harm_ols, harm_orders, _ = _fit_even_harmonics_ols(xo, yo)
        bisector_seed: float | None = None
        orth_seed: float | None = None
        if harm_orders:
            harm_terms = [
                (m, float(harm_ols[f"c{m}"]), float(harm_ols[f"s{m}"]))
                for m in harm_orders
                if harm_ols.get(f"c{m}") is not None and harm_ols.get(f"s{m}") is not None
            ]
            if harm_terms:
                grid_rad = np.linspace(0.0, 0.5 * math.pi, 180, endpoint=False)
                losses = np.zeros_like(grid_rad)
                for m, cm, sm in harm_terms:
                    val = -cm * np.sin(m * grid_rad) + sm * np.cos(m * grid_rad)
                    losses += val * val
                best_deg = float(math.degrees(float(grid_rad[int(np.argmin(losses))])) % 180.0)
                bisector_seed = best_deg
                orth_seed = (best_deg + 90.0) % 180.0
        for candidate_axis in (
            bisector_seed,
            orth_seed,
            nominal_draw_axis_deg % 180.0,
            (nominal_draw_axis_deg + 90.0) % 180.0,
        ):
            if candidate_axis is not None and math.isfinite(float(candidate_axis)):
                ax_mod = float(candidate_axis) % 180.0
                if not any(abs(float(_periodic_dist_deg(ax_mod, s, 180.0))) < 5.0 for s in seed_axes):
                    seed_axes.append(ax_mod)

    _lobe_signs = np.array([-1.0, 1.0, -1.0, 1.0], dtype=float)
    _lobe_offsets = np.array([0.0, 0.0, 180.0, 180.0], dtype=float)
    _two_rad = 2.0 * (math.pi / 180.0)

    def _unpack(params: np.ndarray, include_c2: bool) -> tuple[float, float, float, float, list[float], list[float]]:
        if include_c2:
            c0_v, c2_v, chi_v, alpha_v = float(params[0]), float(params[1]), float(params[2]), float(params[3])
            amps_v = [float(v) for v in params[4:8]]
            sigs_v = [float(params[8])] * 4 if shared_width else [float(v) for v in params[8:12]]
        else:
            c0_v, c2_v, chi_v, alpha_v = float(params[0]), 0.0, float(params[1]), float(params[2])
            amps_v = [float(v) for v in params[3:7]]
            sigs_v = [float(params[7])] * 4 if shared_width else [float(v) for v in params[7:11]]
        return c0_v, c2_v, chi_v, alpha_v, amps_v, sigs_v

    def _fit_one_seed(axis_seed: float, alpha_seed: float, include_c2: bool):
        # Project out c0 + c2*cos(2*(chi - axis_seed)) on observed bins for clean initial baseline
        rad = np.deg2rad(xo - axis_seed)
        if include_c2:
            X_bg = np.column_stack((np.ones_like(rad), np.cos(2.0 * rad)))
            if np.linalg.cond(X_bg) < 1.0e5:
                bg_beta, *_ = np.linalg.lstsq(X_bg, yo, rcond=None)
                c0_0 = float(bg_beta[0])
                c2_0 = float(np.clip(bg_beta[1], -0.95 * c2_bound, 0.95 * c2_bound))
            else:
                c0_0 = float(np.percentile(yo, 20.0))
                c2_0 = 0.0
            resid0 = yo - (c0_0 + c2_0 * np.cos(2.0 * rad))
        else:
            c0_0 = float(np.percentile(yo, 20.0))
            c2_0 = 0.0
            resid0 = yo - c0_0

        centers0 = _butterfly_lobe_centers_deg(axis_seed, alpha_seed)
        amps0: list[float] = []
        win_deg = max(12.0, 0.8 * initial_fwhm_deg)
        for mu in centers0:
            near = np.abs(_periodic_dist_deg(xo, mu, 360.0)) <= win_deg
            est = float(np.max(resid0[near])) if np.any(near) else 0.25 * y_range
            amps0.append(float(np.clip(max(est, 0.05 * y_range), 1.0e-6, 0.95 * amp_hi)))

        sig_vec = [sigma_init] if shared_width else [sigma_init] * 4
        chi_lo, chi_hi = axis_seed - 45.0, axis_seed + 45.0

        if include_c2:
            p0 = np.array([c0_0, c2_0, axis_seed, alpha_seed, *amps0, *sig_vec], dtype=float)
            lo = np.array([-np.inf, -c2_bound, chi_lo, alpha_lo, 0.0, 0.0, 0.0, 0.0, *([sigma_lo] * len(sig_vec))], dtype=float)
            hi = np.array([np.inf, c2_bound, chi_hi, alpha_hi, amp_hi, amp_hi, amp_hi, amp_hi, *([sigma_hi] * len(sig_vec))], dtype=float)
        else:
            p0 = np.array([c0_0, axis_seed, alpha_seed, *amps0, *sig_vec], dtype=float)
            lo = np.array([-np.inf, chi_lo, alpha_lo, 0.0, 0.0, 0.0, 0.0, *([sigma_lo] * len(sig_vec))], dtype=float)
            hi = np.array([np.inf, chi_hi, alpha_hi, amp_hi, amp_hi, amp_hi, amp_hi, *([sigma_hi] * len(sig_vec))], dtype=float)

        eps_lo = np.where(np.isfinite(lo), 1.0e-9, 0.0)
        eps_hi = np.where(np.isfinite(hi), 1.0e-9, 0.0)
        p0 = np.minimum(np.maximum(p0, lo + eps_lo), hi - eps_hi)

        if include_c2:
            def _res(params: np.ndarray) -> np.ndarray:
                c0_v = params[0]
                c2_v = params[1]
                chi_v = params[2]
                alpha_v = params[3]
                amps_arr = params[4:8]
                mu = chi_v + _lobe_offsets + _lobe_signs * alpha_v
                d = ((xo[None, :] - mu[:, None] + 180.0) % 360.0) - 180.0
                if shared_width:
                    inv_s = 1.0 / max(float(params[8]), 1.0e-6)
                    u = d * inv_s
                else:
                    inv_s = 1.0 / np.maximum(params[8:12], 1.0e-6)
                    u = d * inv_s[:, None]
                g = np.exp(-0.5 * (u * u))
                return c0_v + c2_v * np.cos(_two_rad * (xo - chi_v)) + (amps_arr @ g) - yo

            def _jac(params: np.ndarray) -> np.ndarray:
                c2_v = params[1]
                chi_v = params[2]
                alpha_v = params[3]
                amps_arr = params[4:8]
                mu = chi_v + _lobe_offsets + _lobe_signs * alpha_v
                d = ((xo[None, :] - mu[:, None] + 180.0) % 360.0) - 180.0
                if shared_width:
                    inv_s = 1.0 / max(float(params[8]), 1.0e-6)
                    u = d * inv_s
                    g = np.exp(-0.5 * (u * u))
                    L = amps_arr[:, None] * g
                    dL_dmu = L * (u * inv_s)
                    dL_dsig = L * ((u * u) * inv_s)
                else:
                    inv_s = 1.0 / np.maximum(params[8:12], 1.0e-6)
                    u = d * inv_s[:, None]
                    g = np.exp(-0.5 * (u * u))
                    L = amps_arr[:, None] * g
                    dL_dmu = L * (u * inv_s[:, None])
                    dL_dsig = L * ((u * u) * inv_s[:, None])
                phase = _two_rad * (xo - chi_v)
                J = np.empty((n_obs, params.size), dtype=float)
                J[:, 0] = 1.0
                J[:, 1] = np.cos(phase)
                J[:, 2] = _two_rad * c2_v * np.sin(phase) + np.sum(dL_dmu, axis=0)
                J[:, 3] = _lobe_signs @ dL_dmu
                J[:, 4:8] = g.T
                if shared_width:
                    J[:, 8] = np.sum(dL_dsig, axis=0)
                else:
                    J[:, 8:12] = dL_dsig.T
                return J
        else:
            def _res(params: np.ndarray) -> np.ndarray:
                c0_v = params[0]
                chi_v = params[1]
                alpha_v = params[2]
                amps_arr = params[3:7]
                mu = chi_v + _lobe_offsets + _lobe_signs * alpha_v
                d = ((xo[None, :] - mu[:, None] + 180.0) % 360.0) - 180.0
                if shared_width:
                    inv_s = 1.0 / max(float(params[7]), 1.0e-6)
                    u = d * inv_s
                else:
                    inv_s = 1.0 / np.maximum(params[7:11], 1.0e-6)
                    u = d * inv_s[:, None]
                g = np.exp(-0.5 * (u * u))
                return c0_v + (amps_arr @ g) - yo

            def _jac(params: np.ndarray) -> np.ndarray:
                chi_v = params[1]
                alpha_v = params[2]
                amps_arr = params[3:7]
                mu = chi_v + _lobe_offsets + _lobe_signs * alpha_v
                d = ((xo[None, :] - mu[:, None] + 180.0) % 360.0) - 180.0
                if shared_width:
                    inv_s = 1.0 / max(float(params[7]), 1.0e-6)
                    u = d * inv_s
                    g = np.exp(-0.5 * (u * u))
                    L = amps_arr[:, None] * g
                    dL_dmu = L * (u * inv_s)
                    dL_dsig = L * ((u * u) * inv_s)
                else:
                    inv_s = 1.0 / np.maximum(params[7:11], 1.0e-6)
                    u = d * inv_s[:, None]
                    g = np.exp(-0.5 * (u * u))
                    L = amps_arr[:, None] * g
                    dL_dmu = L * (u * inv_s[:, None])
                    dL_dsig = L * ((u * u) * inv_s[:, None])
                J = np.empty((n_obs, params.size), dtype=float)
                J[:, 0] = 1.0
                J[:, 1] = np.sum(dL_dmu, axis=0)
                J[:, 2] = _lobe_signs @ dL_dmu
                J[:, 3:7] = g.T
                if shared_width:
                    J[:, 7] = np.sum(dL_dsig, axis=0)
                else:
                    J[:, 7:11] = dL_dsig.T
                return J

        return least_squares(
            _res,
            p0,
            jac=_jac,
            bounds=(lo, hi),
            loss=loss,
            f_scale=max(0.05 * y_range, 1.0e-12),
            max_nfev=max_nfev,
        )

    best_c2_fit = None
    best_c2_cost = float("inf")
    best_c2_sse = float("inf")
    initial_centers_used: tuple[float, ...] = ()

    for ax_idx, ax_seed in enumerate(seed_axes[:4]):
        alpha_candidates: list[float] = []
        if initial_alpha_deg is not None and math.isfinite(float(initial_alpha_deg)):
            a_init = float(initial_alpha_deg) if ax_idx == 0 else (90.0 - float(initial_alpha_deg))
            alpha_candidates.append(float(np.clip(a_init, alpha_lo + 1.0, alpha_hi - 1.0)))
        alpha_est, _ = _detect_folded_lobe_opening(xo, yo, ax_seed)
        if alpha_est is not None:
            a_clipped = float(np.clip(alpha_est, alpha_lo + 1.0, alpha_hi - 1.0))
            if not any(abs(a_clipped - existing) < 6.0 for existing in alpha_candidates):
                alpha_candidates.append(a_clipped)
        if not alpha_candidates:
            alpha_candidates = [float(np.clip(35.0, alpha_lo + 1.0, alpha_hi - 1.0))]
        for a_seed in alpha_candidates[:2]:
            try:
                res = _fit_one_seed(ax_seed, a_seed, include_c2=True)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            cost = float(res.cost)
            sse = float(np.sum(res.fun ** 2))
            if math.isfinite(cost) and cost < best_c2_cost:
                best_c2_cost = cost
                best_c2_sse = sse
                best_c2_fit = res
                initial_centers_used = _butterfly_lobe_centers_deg(ax_seed, a_seed)
        # Early exit when an effectively exact noiseless fit is already found
        if best_c2_sse <= 1.0e-10 * (y_range ** 2) * n_obs:
            break

    if best_c2_fit is None:
        flags.append("c2_four_lobe_fit_failed")
        return json_safe({
            "success": False,
            "status": "failed",
            "message": "C2-bridge + 4-lobe optimizer failed on all initial seeds.",
            "n_observed_bins": n_obs,
            "n_total_bins": int(x.size),
            "angular_coverage": _finite(n_obs / float(x.size)),
            "c0": None,
            "c2": None,
            "chi_sym_deg": None,
            "alpha_deg": None,
            "lobe_centers_deg": None,
            "lobe_amplitudes": None,
            "lobe_sigmas_deg": None,
            "lobe_fwhms_deg": None,
            "lobe_fwhm_deg": None,
            "lobe_supported": [False, False, False, False],
            "rmse": None,
            "r_squared": None,
            "aic": None,
            "bic": None,
            "angle_deg": _finite_list(x),
            "observed": _finite_list(np.where(observed, y, np.nan)),
            "model": [None] * int(x.size),
            "bridge_component": [None] * int(x.size),
            "lobes_component": [None] * int(x.size),
            "residual": [None] * int(x.size),
            "flat_baseline_comparison": None,
            "flags": list(dict.fromkeys(flags)),
        })

    c0_raw, c2_raw, chi_raw, alpha_raw, amps_raw, sigs_raw = _unpack(best_c2_fit.x, include_c2=True)
    chi_sym, alpha_deg, c2_val, amps, sigmas = _canonicalize_butterfly_params(
        chi_raw,
        alpha_raw,
        c2_raw,
        amps_raw,
        sigs_raw,
        prefer_acute_opening=prefer_acute_opening and (alpha_lo <= 45.0 <= alpha_hi),
        nominal_draw_axis_deg=canonical_tie_axis_deg,
    )
    centers = _butterfly_lobe_centers_deg(chi_sym, alpha_deg)
    fwhms = [float(s * _FWHM_PER_SIGMA) for s in sigmas]

    # Check observational support around each lobe
    lobe_supported: list[bool] = []
    for idx_k, (mu, sig) in enumerate(zip(centers, sigmas), start=1):
        near_bins = int(np.count_nonzero(np.abs(_periodic_dist_deg(xo, mu, 360.0)) <= max(12.0, 1.5 * sig)))
        supported_k = near_bins >= 3
        lobe_supported.append(supported_k)
        if not supported_k:
            flags.append(f"lobe_{idx_k}_weak_or_masked_support")

    supported_fwhms = [fw for fw, sup in zip(fwhms, lobe_supported) if sup]
    lobe_fwhm_summary = float(np.median(supported_fwhms)) if supported_fwhms else float(np.median(fwhms))

    # Bound diagnostics
    bound_tol = 1.0e-4
    if abs(alpha_deg - alpha_lo) <= bound_tol or abs(alpha_deg - alpha_hi) <= bound_tol:
        flags.append("alpha_at_bound")
    if any(abs(fw - fwhm_lo) <= bound_tol * max(fwhm_lo, 1.0) or abs(fw - fwhm_hi) <= bound_tol * max(fwhm_hi, 1.0) for fw in fwhms):
        flags.append("lobe_width_at_bound")

    # Evaluate components and residuals strictly on observed bins
    model_all = evaluate_c2_bridge_four_lobe(
        x,
        c0=c0_raw,
        c2=c2_val,
        chi_sym_deg=chi_sym,
        alpha_deg=alpha_deg,
        lobe_amplitudes=amps,
        lobe_sigmas_deg=sigmas,
    )
    bridge_all = c0_raw + c2_val * np.cos(2.0 * np.deg2rad(x - chi_sym))
    lobes_all = model_all - bridge_all
    residual_all = np.full(x.shape, np.nan, dtype=float)
    residual_all[observed] = y[observed] - model_all[observed]

    err_obs = residual_all[observed]
    sse_c2 = float(np.sum(err_obs ** 2))
    rmse_c2 = float(math.sqrt(sse_c2 / float(n_obs)))
    sst = float(np.sum((yo - float(np.mean(yo))) ** 2))
    r2_c2 = float(1.0 - sse_c2 / sst) if sst > 1.0e-30 else None
    aic_c2, bic_c2 = _aic_bic(sse_c2, n_obs, n_params_c2)

    success = bool(best_c2_fit.success and np.all(np.isfinite(best_c2_fit.x)) and np.all(np.isfinite(model_all)))
    if not success:
        flags.append("optimizer_did_not_converge")

    # Nested flat-baseline (c2 = 0) 4-lobe fit for direct scientific comparison
    n_params_flat = n_params_c2 - 1
    best_flat_fit = None
    best_flat_cost = float("inf")
    if include_flat_comparison:
        flat_seeds = [(chi_sym, alpha_deg), ((chi_sym + 90.0) % 180.0, max(alpha_lo + 1.0, min(alpha_hi - 1.0, 90.0 - alpha_deg)))]
        for ax_s, a_s in flat_seeds:
            try:
                fres = _fit_one_seed(ax_s, float(np.clip(a_s, alpha_lo + 0.5, alpha_hi - 0.5)), include_c2=False)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            fcost = float(fres.cost)
            if math.isfinite(fcost) and fcost < best_flat_cost:
                best_flat_cost = fcost
                best_flat_fit = fres

    flat_comparison: dict[str, Any] | None = None
    if best_flat_fit is not None:
        fc0, _, fchi, falpha, famps, fsigs = _unpack(best_flat_fit.x, include_c2=False)
        fchi, falpha, _, famps, fsigs = _canonicalize_butterfly_params(
            fchi,
            falpha,
            0.0,
            famps,
            fsigs,
            prefer_acute_opening=prefer_acute_opening and (alpha_lo <= 45.0 <= alpha_hi),
            nominal_draw_axis_deg=canonical_tie_axis_deg,
        )
        ffwhms = [float(s * _FWHM_PER_SIGMA) for s in fsigs]
        flat_fwhm_med = float(np.median(ffwhms))
        fmodel = evaluate_c2_bridge_four_lobe(
            xo,
            c0=fc0,
            c2=0.0,
            chi_sym_deg=fchi,
            alpha_deg=falpha,
            lobe_amplitudes=famps,
            lobe_sigmas_deg=fsigs,
        )
        ferr = yo - fmodel
        sse_flat = float(np.sum(ferr ** 2))
        rmse_flat = float(math.sqrt(sse_flat / float(n_obs)))
        r2_flat = float(1.0 - sse_flat / sst) if sst > 1.0e-30 else None
        aic_flat, bic_flat = _aic_bic(sse_flat, n_obs, n_params_flat)

        dof_c2 = n_obs - n_params_c2
        f_stat: float | None = None
        if dof_c2 > 0 and sse_c2 > 1.0e-30 and sse_flat >= sse_c2:
            f_stat = _finite(((sse_flat - sse_c2) / 1.0) / (sse_c2 / float(dof_c2)))
        elif dof_c2 > 0 and sse_flat > 1.0e-20 and sse_c2 <= 1.0e-30:
            f_stat = None  # exact noiseless fit with zero residual

        delta_bic = _finite(bic_flat - bic_c2) if (bic_flat is not None and bic_c2 is not None) else None
        delta_aic = _finite(aic_flat - aic_c2) if (aic_flat is not None and aic_c2 is not None) else None
        bridge_improved = bool(
            (delta_bic is not None and delta_bic > 2.0)
            and (rmse_flat - rmse_c2 > 1.0e-6 * y_range)
        )
        flat_comparison = {
            "success": bool(best_flat_fit.success),
            "c0": _finite(fc0),
            "chi_sym_deg": _finite(fchi),
            "alpha_deg": _finite(falpha),
            "lobe_amplitudes": [_finite(v) for v in famps],
            "lobe_fwhms_deg": [_finite(v) for v in ffwhms],
            "lobe_fwhm_deg": _finite(flat_fwhm_med),
            "rmse": _finite(rmse_flat),
            "r_squared": _finite(r2_flat),
            "aic": aic_flat,
            "bic": bic_flat,
            "delta_rmse": _finite(rmse_flat - rmse_c2),
            "delta_r_squared": _finite(r2_c2 - r2_flat) if (r2_c2 is not None and r2_flat is not None) else None,
            "delta_aic": delta_aic,
            "delta_bic": delta_bic,
            "bridge_f_stat": f_stat,
            "bridge_improved": bridge_improved,
            "lobe_fwhm_bias_deg": _finite(flat_fwhm_med - lobe_fwhm_summary),
        }

    return json_safe({
        "success": success,
        "status": "estimated" if success else "warning",
        "message": str(best_c2_fit.message),
        "n_observed_bins": n_obs,
        "n_total_bins": int(x.size),
        "angular_coverage": _finite(n_obs / float(x.size)),
        "c0": _finite(c0_raw),
        "c2": _finite(c2_val),
        "chi_sym_deg": _finite(chi_sym),
        "alpha_deg": _finite(alpha_deg),
        "lobe_centers_deg": [_finite(v) for v in centers],
        "lobe_amplitudes": [_finite(v) for v in amps],
        "lobe_sigmas_deg": [_finite(v) for v in sigmas],
        "lobe_fwhms_deg": [_finite(v) for v in fwhms],
        "lobe_fwhm_deg": _finite(lobe_fwhm_summary),
        "lobe_supported": lobe_supported,
        "initial_centres_deg": [_finite(v) for v in initial_centers_used],
        "rmse": _finite(rmse_c2),
        "r_squared": _finite(r2_c2),
        "aic": aic_c2,
        "bic": bic_c2,
        "angle_deg": _finite_list(x),
        "observed": _finite_list(np.where(observed, y, np.nan)),
        "model": _finite_list(model_all),
        "bridge_component": _finite_list(bridge_all),
        "lobes_component": _finite_list(lobes_all),
        "residual": _finite_list(residual_all),
        "flat_baseline_comparison": flat_comparison,
        "flags": list(dict.fromkeys(flags)),
    })


def _detect_wing_angles_from_profile(
    ang: np.ndarray,
    prof: np.ndarray,
    cnts: np.ndarray,
    *,
    nominal_draw_axis_deg: float,
) -> tuple[list[float], str]:
    """Detect butterfly wing ray angles from a binned observed azimuthal profile."""
    obs = np.isfinite(prof) & (cnts > 0)
    if int(np.count_nonzero(obs)) < 16:
        return [], "insufficient_azimuthal_support"
    harm = compute_azimuthal_harmonics_and_order(
        angle_deg=ang,
        intensity=prof,
        counts=cnts,
        nominal_draw_axis_deg=nominal_draw_axis_deg,
    )
    a2 = harm.get("a2") or 0.0
    a4 = harm.get("a4") or 0.0
    bisector = harm.get("friedel_bisector_deg")
    if bisector is None or (a2 < 0.03 and a4 < 0.03):
        return [], "isotropic_no_distinct_wings"

    if a4 >= 0.03:
        fit = fit_c2_bridge_four_lobe(
            ang,
            prof,
            counts=cnts,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            initial_chi_sym_deg=bisector,
            initial_alpha_deg=harm.get("wing_half_opening_estimate_deg"),
            max_nfev=1500,
            include_flat_comparison=False,
        )
        if fit.get("success") and fit.get("lobe_centers_deg"):
            centers = [float(c) for c in fit["lobe_centers_deg"] if c is not None]
            if len(centers) == 4:
                return centers, "c2_bridge_four_lobe"
        alpha_est = harm.get("wing_half_opening_estimate_deg")
        if alpha_est is not None:
            return list(_butterfly_lobe_centers_deg(bisector, float(alpha_est))), "harmonic_folded_wings"

    return [float(bisector % 360.0), float((bisector + 180.0) % 360.0)], "two_lobe_axis"
