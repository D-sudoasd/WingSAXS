"""Perpendicular wing streak width and Williamson-Hall / Ruland lateral coherence models."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ..serialization import json_safe
from ._common import (
    _FWHM_PER_SIGMA,
    _finite,
    _normalise_intensity_mode,
    _periodic_dist_deg,
    _q_unit_info,
    _resolve_2d_context,
    _validate_q_window,
)
from .radial import _detect_wing_angles_for_sectors


def _fit_1d_transverse_gaussian(
    q_perp_centers: np.ndarray,
    profile: np.ndarray,
    counts: np.ndarray,
    *,
    q_perp_max: float,
    min_r2: float,
) -> dict[str, Any]:
    """Fit ``b + A * exp(-(q_perp - mu)^2 / (2 * sigma^2))`` strictly on observed bins."""
    obs = np.isfinite(q_perp_centers) & np.isfinite(profile) & (counts > 0)
    n_obs = int(np.count_nonzero(obs))
    support_frac = float(n_obs / float(q_perp_centers.size)) if q_perp_centers.size > 0 else 0.0
    if n_obs < 7:
        return {"valid": False, "reason": "insufficient_observed_transverse_bins", "support_fraction": support_frac, "n_observed_bins": n_obs}

    xo = q_perp_centers[obs]
    yo = profile[obs]
    y_min, y_max = float(np.min(yo)), float(np.max(yo))
    y_range = y_max - y_min
    if not math.isfinite(y_range) or y_range <= 1.0e-12:
        return {"valid": False, "reason": "flat_transverse_profile", "support_fraction": support_frac, "n_observed_bins": n_obs}

    # Check peak is near the ray center and observed on both sides
    i_max = int(np.argmax(yo))
    mu_0 = float(xo[i_max])
    if abs(mu_0) > 0.45 * q_perp_max:
        return {"valid": False, "reason": "peak_off_ray_center", "support_fraction": support_frac, "n_observed_bins": n_obs}
    left_obs = int(np.count_nonzero(xo < mu_0))
    right_obs = int(np.count_nonzero(xo > mu_0))
    if left_obs < 2 or right_obs < 2:
        return {"valid": False, "reason": "one_sided_transverse_support", "support_fraction": support_frac, "n_observed_bins": n_obs}

    dq_step = float(np.median(np.diff(q_perp_centers))) if q_perp_centers.size > 1 else q_perp_max / 10.0
    sig_lo = max(0.35 * dq_step, 1.0e-6)
    sig_hi = max(0.85 * q_perp_max, sig_lo * 2.0)

    b_0 = float(np.percentile(yo, 10.0))
    a_0 = max(y_max - b_0, 0.1 * y_range)
    # Initial sigma from half-height bin count
    above_half = xo[yo >= b_0 + 0.5 * a_0]
    if above_half.size >= 2:
        sig_0 = float((np.max(above_half) - np.min(above_half)) / _FWHM_PER_SIGMA)
    else:
        sig_0 = 0.2 * q_perp_max
    sig_0 = float(np.clip(sig_0, sig_lo * 1.1, sig_hi * 0.9))

    p0 = np.array([b_0, a_0, float(np.clip(mu_0, -0.4 * q_perp_max, 0.4 * q_perp_max)), sig_0], dtype=float)
    lo = np.array([-np.inf, 0.0, -0.45 * q_perp_max, sig_lo], dtype=float)
    hi = np.array([np.inf, max(10.0 * y_range, 10.0 * abs(y_max), 1.0), 0.45 * q_perp_max, sig_hi], dtype=float)

    def _model(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        u = (x - p[2]) / p[3]
        return p[0] + p[1] * np.exp(-0.5 * (u * u))

    def _res_1d(p: np.ndarray) -> np.ndarray:
        u = (xo - p[2]) / p[3]
        return p[0] + p[1] * np.exp(-0.5 * (u * u)) - yo

    def _jac_1d(p: np.ndarray) -> np.ndarray:
        inv_s = 1.0 / p[3]
        u = (xo - p[2]) * inv_s
        g = np.exp(-0.5 * (u * u))
        ag = p[1] * g
        J = np.empty((n_obs, 4), dtype=float)
        J[:, 0] = 1.0
        J[:, 1] = g
        J[:, 2] = ag * (u * inv_s)
        J[:, 3] = ag * ((u * u) * inv_s)
        return J

    try:
        fit = least_squares(_res_1d, p0, jac=_jac_1d, bounds=(lo, hi), loss="linear", max_nfev=1500)
    except (ValueError, RuntimeError, FloatingPointError):
        return {"valid": False, "reason": "transverse_optimizer_failed", "support_fraction": support_frac, "n_observed_bins": n_obs}

    b_f, a_f, mu_f, sig_f = (float(v) for v in fit.x)
    pred = _model(fit.x, xo)
    resid = yo - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((yo - float(np.mean(yo))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1.0e-30 else 0.0
    rmse = float(math.sqrt(sse / float(n_obs)))
    fwhm_f = sig_f * _FWHM_PER_SIGMA

    # Validate fit quality and non-degeneracy
    at_sig_bound = abs(sig_f - sig_lo) <= 1.0e-4 * max(sig_lo, 1.0) or abs(sig_f - sig_hi) <= 1.0e-4 * max(sig_hi, 1.0)
    at_mu_bound = abs(abs(mu_f) - 0.45 * q_perp_max) <= 1.0e-4 * max(q_perp_max, 1.0)
    reason = None
    valid = True
    if not fit.success:
        valid, reason = False, "optimizer_did_not_converge"
    elif a_f <= 0.0:
        valid, reason = False, "nonpositive_amplitude"
    elif r2 < min_r2:
        valid, reason = False, "low_r_squared"
    elif at_sig_bound:
        valid, reason = False, "width_at_bound"
    elif at_mu_bound:
        valid, reason = False, "center_at_bound"

    # Estimate sigma stderr from Jacobian (only when Jacobian is finite and full rank)
    fwhm_stderr = None
    if valid and n_obs > 4 and fit.jac is not None and np.all(np.isfinite(fit.jac)):
        try:
            svals = np.linalg.svd(fit.jac, compute_uv=False)
            tol = float(svals[0]) * max(fit.jac.shape) * np.finfo(float).eps if svals.size else 0.0
            if svals.size == 4 and float(svals[-1]) > tol:
                jtj = fit.jac.T @ fit.jac
                s2 = sse / float(n_obs - 4)
                cov = np.linalg.pinv(jtj) * s2
                var_sig = float(cov[3, 3])
                if math.isfinite(var_sig) and var_sig >= 0.0:
                    fwhm_stderr = _finite(math.sqrt(var_sig) * _FWHM_PER_SIGMA)
        except np.linalg.LinAlgError:
            pass

    return {
        "valid": valid,
        "reason": reason,
        "support_fraction": _finite(support_frac),
        "n_observed_bins": n_obs,
        "baseline": _finite(b_f),
        "amplitude": _finite(a_f),
        "center_q_perp": _finite(mu_f),
        "sigma_q_perp": _finite(sig_f),
        "fwhm_q_perp": _finite(fwhm_f),
        "fwhm_q_perp_stderr": fwhm_stderr,
        "r_squared": _finite(r2),
        "rmse": _finite(rmse),
    }


def _ols_line_with_stderr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Exact ordinary least squares fit ``y = intercept + slope * x`` with stderr and R^2."""
    n = int(x.size)
    X = np.column_stack((np.ones(n, dtype=float), x))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    pred = X @ beta
    resid = y - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1.0e-30 else (1.0 if sse <= 1.0e-30 else 0.0)
    rmse = float(math.sqrt(sse / float(n)))

    intercept_se = None
    slope_se = None
    if n > 2:
        s2 = sse / float(n - 2)
        cov = np.linalg.pinv(X.T @ X) * s2
        if cov[0, 0] >= 0.0:
            intercept_se = _finite(math.sqrt(float(cov[0, 0])))
        if cov[1, 1] >= 0.0:
            slope_se = _finite(math.sqrt(float(cov[1, 1])))
    return {
        "intercept": intercept,
        "slope": slope,
        "intercept_stderr": intercept_se,
        "slope_stderr": slope_se,
        "r_squared": _finite(r2),
        "rmse": _finite(rmse),
    }


def _fit_streak_width_models(
    q_vals: np.ndarray,
    fwhm_vals: np.ndarray,
    *,
    q_scale_nm_inv: float | None,
    q_unit_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit linear and quadrature Grubb/Williamson-Hall models to ``fwhm(q)``."""
    n = int(q_vals.size)
    if n < 3 or float(np.ptp(q_vals)) <= 1.0e-12:
        unavail = {
            "status": "unavailable",
            "valid": False,
            "dq_perp_0": None,
            "dq_perp_0_stderr": None,
            "mosaic_spread_rad": None,
            "mosaic_spread_deg": None,
            "mosaic_spread_stderr_deg": None,
            "W_nm": None,
            "W_stderr_nm": None,
            "r_squared": None,
            "rmse": None,
            "flags": ["insufficient_valid_slices"],
        }
        return dict(unavail), dict(unavail)

    # 1. Linear model: dq_perp(q) = dq_perp_0 + delta_alpha_rad * q
    lin = _ols_line_with_stderr(q_vals, fwhm_vals)
    lin_flags: list[str] = []
    dq0_lin = lin["intercept"]
    slope_lin = lin["slope"]
    dq0_lin_se = lin["intercept_stderr"]
    slope_lin_se = lin["slope_stderr"]

    mosaic_deg_lin: float | None = None
    mosaic_se_deg_lin: float | None = None
    if slope_lin >= 0.0:
        mosaic_deg_lin = _finite(math.degrees(slope_lin))
        mosaic_se_deg_lin = _finite(math.degrees(slope_lin_se)) if slope_lin_se is not None else None
    else:
        lin_flags.append("negative_mosaic_slope")

    w_lin_nm: float | None = None
    w_lin_se_nm: float | None = None
    if dq0_lin <= 0.0:
        lin_flags.append("nonpositive_intercept_W_unbounded")
    elif q_scale_nm_inv is None:
        lin_flags.append("uncalibrated_pixel_q" if q_unit_label == "pixel-q" else "spacing_unavailable_unknown_q_unit")
    else:
        dq0_nm_inv = dq0_lin * q_scale_nm_inv
        w_lin_nm = _finite(2.0 * math.pi / dq0_nm_inv)
        if w_lin_nm is not None and dq0_lin_se is not None:
            w_lin_se_nm = _finite(w_lin_nm * (dq0_lin_se / dq0_lin))

    lin_valid = bool(dq0_lin > 0.0 and slope_lin >= 0.0)
    linear_result = {
        "status": "estimated" if lin_valid else "warning",
        "valid": lin_valid,
        "dq_perp_0": _finite(dq0_lin),
        "dq_perp_0_stderr": dq0_lin_se,
        "mosaic_spread_rad": _finite(slope_lin),
        "mosaic_spread_deg": mosaic_deg_lin,
        "mosaic_spread_stderr_deg": mosaic_se_deg_lin,
        "W_nm": w_lin_nm,
        "W_stderr_nm": w_lin_se_nm,
        "r_squared": lin["r_squared"],
        "rmse": lin["rmse"],
        "flags": lin_flags,
    }

    # 2. Quadrature model: dq_perp(q)^2 = dq_perp_0^2 + (delta_alpha_rad * q)^2
    quad_ols = _ols_line_with_stderr(q_vals ** 2, fwhm_vals ** 2)
    quad_flags: list[str] = []
    c0_q = quad_ols["intercept"]
    c1_q = quad_ols["slope"]
    c0_se = quad_ols["intercept_stderr"]
    c1_se = quad_ols["slope_stderr"]

    dq0_quad: float | None = None
    dq0_quad_se: float | None = None
    if c0_q > 0.0:
        dq0_quad = math.sqrt(c0_q)
        if c0_se is not None:
            dq0_quad_se = _finite(c0_se / (2.0 * dq0_quad))
    else:
        quad_flags.append("nonpositive_quadrature_intercept_W_unbounded")

    mosaic_rad_quad: float | None = None
    mosaic_deg_quad: float | None = None
    mosaic_se_deg_quad: float | None = None
    if c1_q >= 0.0:
        mosaic_rad_quad = math.sqrt(c1_q)
        mosaic_deg_quad = _finite(math.degrees(mosaic_rad_quad))
        if c1_se is not None and mosaic_rad_quad > 1.0e-12:
            mosaic_se_deg_quad = _finite(math.degrees(c1_se / (2.0 * mosaic_rad_quad)))
    else:
        quad_flags.append("negative_quadrature_mosaic_slope")

    w_quad_nm: float | None = None
    w_quad_se_nm: float | None = None
    if dq0_quad is not None and dq0_quad > 0.0:
        if q_scale_nm_inv is None:
            quad_flags.append("uncalibrated_pixel_q" if q_unit_label == "pixel-q" else "spacing_unavailable_unknown_q_unit")
        else:
            dq0_nm_inv = dq0_quad * q_scale_nm_inv
            w_quad_nm = _finite(2.0 * math.pi / dq0_nm_inv)
            if w_quad_nm is not None and dq0_quad_se is not None:
                w_quad_se_nm = _finite(w_quad_nm * (dq0_quad_se / dq0_quad))

    # Compute R^2 and RMSE in FWHM space as well as squared space
    quad_r2 = quad_ols["r_squared"]
    quad_rmse = quad_ols["rmse"]
    if dq0_quad is not None and mosaic_rad_quad is not None:
        pred_fwhm = np.sqrt(dq0_quad ** 2 + (mosaic_rad_quad * q_vals) ** 2)
        resid_fwhm = fwhm_vals - pred_fwhm
        sse_f = float(np.sum(resid_fwhm ** 2))
        sst_f = float(np.sum((fwhm_vals - float(np.mean(fwhm_vals))) ** 2))
        quad_r2 = _finite(1.0 - sse_f / sst_f) if sst_f > 1.0e-30 else (1.0 if sse_f <= 1.0e-30 else 0.0)
        quad_rmse = _finite(math.sqrt(sse_f / float(n)))

    quad_valid = bool(dq0_quad is not None and dq0_quad > 0.0 and mosaic_rad_quad is not None)
    quadrature_result = {
        "status": "estimated" if quad_valid else "warning",
        "valid": quad_valid,
        "dq_perp_0": _finite(dq0_quad),
        "dq_perp_0_stderr": dq0_quad_se,
        "dq_perp_0_squared": _finite(c0_q),
        "mosaic_spread_rad": _finite(mosaic_rad_quad),
        "mosaic_spread_deg": mosaic_deg_quad,
        "mosaic_spread_stderr_deg": mosaic_se_deg_quad,
        "W_nm": w_quad_nm,
        "W_stderr_nm": w_quad_se_nm,
        "r_squared": quad_r2,
        "r_squared_in_q2": quad_ols["r_squared"],
        "rmse": quad_rmse,
        "flags": quad_flags,
    }
    return linear_result, quadrature_result


def measure_perpendicular_streak_width(
    image: Any,
    qx: Any = None,
    qy: Any = None,
    *,
    valid_mask: Any = None,
    q_window: tuple[float, float],
    q_unit: str = "unknown",
    wing_angles_deg: Sequence[float] | None = None,
    n_q_slices: int = 12,
    transverse_half_angle_deg: float = 20.0,
    min_q_perp_half_width: float | None = None,
    n_perp_bins: int | None = None,
    min_slice_pixels: int = 12,
    min_slice_r2: float = 0.5,
    nominal_draw_axis_deg: float = 90.0,
    geometry_cache: Any = None,
    intensity_mode: str = "calibrated",
    sigma: Any = None,
) -> dict[str, Any]:
    """Measure transverse butterfly streak width ``dq_perp(q)`` along detected wing rays.

    Separates intrinsic angular mosaic spread ``delta_alpha`` (deg) from apparent
    lateral coherence length ``W = 2*pi / dq_perp(0)`` (nm, calibrated physical q
    only) using both linear (Williamson-Hall) and quadrature (Gaussian-convolution)
    fits on strictly observed transverse slices.
    """
    _normalise_intensity_mode(intensity_mode)
    q_min, q_max = _validate_q_window(q_window)
    if isinstance(n_q_slices, (bool, np.bool_)) or int(n_q_slices) < 3:
        raise ValueError("n_q_slices must be an integer >= 3")
    n_slices = int(n_q_slices)
    half_ang = float(transverse_half_angle_deg)
    if not math.isfinite(half_ang) or not (1.0 <= half_ang <= 60.0):
        raise ValueError("transverse_half_angle_deg must be finite and in [1, 60]")

    # Conservative radial upper bound covering all transverse slices before resolving ROI
    tan_half_max = math.tan(math.radians(half_ang))
    q_perp_prelim = q_max * tan_half_max
    if min_q_perp_half_width is not None and math.isfinite(float(min_q_perp_half_width)):
        q_perp_prelim = max(q_perp_prelim, float(min_q_perp_half_width))
    q_roi_prelim = math.hypot(q_max, max(q_perp_prelim, 0.25 * q_max))

    img, qx_arr, qy_arr, q_mag, chi_deg, geom_mask, inclusion, _, pixel_dq_hint = _resolve_2d_context(
        image,
        qx,
        qy,
        valid_mask,
        geometry_cache=geometry_cache,
        sigma=sigma,
        required_q_max=q_roi_prelim,
    )
    q_unit_label, q_scale_nm_inv, unit_caveat = _q_unit_info(q_unit)

    if wing_angles_deg is not None:
        rays = [float(a) % 360.0 for a in wing_angles_deg if math.isfinite(float(a))]
        ray_source = "user_supplied"
    else:
        rays, ray_source = _detect_wing_angles_for_sectors(
            img,
            qx_arr,
            qy_arr,
            geom_mask,
            None,
            q_window=(q_min, q_max),
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            q_mag=q_mag,
            chi_deg=chi_deg,
            inclusion=inclusion,
        )

    caveats = [
        "Apparent lateral coherence length W = 2*pi/dq_perp(0) is a model-dependent "
        "Grubb/Ruland/Williamson-Hall streak-width estimate affected by instrumental "
        "resolution, q-window range, and profile shape; it is not a unique 3D reconstruction."
    ]
    if unit_caveat is not None:
        caveats.append(unit_caveat)

    if not rays:
        return json_safe({
            "status": "unavailable",
            "reason": ray_source,
            "q_unit": q_unit_label,
            "q_window": [q_min, q_max],
            "wing_angle_source": ray_source,
            "wings": [],
            "summary": {
                "n_wings_analyzed": 0,
                "n_valid_wings_linear": 0,
                "n_valid_wings_quadrature": 0,
                "total_valid_slices": 0,
                "linear": {"dq_perp_0": None, "mosaic_spread_deg": None, "W_nm": None},
                "quadrature": {"dq_perp_0": None, "mosaic_spread_deg": None, "W_nm": None},
            },
            "flags": ["no_butterfly_wings_detected"],
            "caveats": caveats,
        })

    # Fast pixel reciprocal spacing estimate (reuse cache or strided subsample; avoids 2.48M-px nanmedian)
    if pixel_dq_hint is not None and math.isfinite(float(pixel_dq_hint)) and float(pixel_dq_hint) > 0.0:
        pixel_dq = float(pixel_dq_hint)
    else:
        stride_y = max(1, qx_arr.shape[0] // 64)
        stride_x = max(1, qx_arr.shape[1] // 64)
        dx_med = (
            float(
                np.nanmedian(
                    np.hypot(
                        np.diff(qx_arr[::stride_y, :], axis=1),
                        np.diff(qy_arr[::stride_y, :], axis=1),
                    )
                )
            )
            if qx_arr.shape[1] > 1
            else 0.0
        )
        dy_med = (
            float(
                np.nanmedian(
                    np.hypot(
                        np.diff(qx_arr[:, ::stride_x], axis=0),
                        np.diff(qy_arr[:, ::stride_x], axis=0),
                    )
                )
            )
            if qy_arr.shape[0] > 1
            else 0.0
        )
        pos_steps = [v for v in (dx_med, dy_med) if math.isfinite(v) and v > 0.0]
        pixel_dq = max(
            min(pos_steps) if pos_steps else (q_max - q_min) / 100.0,
            1.0e-9,
        )

    slice_edges = np.linspace(q_min, q_max, n_slices + 1, dtype=float)
    slice_centers = 0.5 * (slice_edges[:-1] + slice_edges[1:])
    slice_half_w = 0.5 * (slice_edges[1:] - slice_edges[:-1])

    # Exact radial ROI upper bound covering every slice of every wing
    q_perp_upper = max(
        q_max * tan_half_max,
        float(min_q_perp_half_width)
        if (min_q_perp_half_width is not None and math.isfinite(float(min_q_perp_half_width)))
        else 6.0 * pixel_dq,
    )
    q_roi_upper = math.hypot(q_max, q_perp_upper)

    if q_mag is None:
        in_box = (np.abs(qx_arr) <= q_roi_upper) & (np.abs(qy_arr) <= q_roi_upper)
        if not np.any(in_box):
            if not np.any(np.isfinite(qx_arr) & np.isfinite(qy_arr)):
                raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
            obs_qx = np.empty(0, dtype=float)
            obs_qy = np.empty(0, dtype=float)
            obs_img = np.empty(0, dtype=float)
        else:
            bx = qx_arr[in_box]
            by = qy_arr[in_box]
            bimg = img[in_box]
            binc = inclusion[in_box]
            bq = np.hypot(bx, by)
            in_ring = np.isfinite(bq) & (bq >= q_min) & (bq <= q_roi_upper) & binc & np.isfinite(bimg)
            obs_qx = bx[in_ring]
            obs_qy = by[in_ring]
            obs_img = bimg[in_ring]
    else:
        assert geom_mask is not None
        in_ring = geom_mask & inclusion & np.isfinite(img) & (q_mag >= q_min) & (q_mag <= q_roi_upper)
        obs_qx = qx_arr[in_ring]
        obs_qy = qy_arr[in_ring]
        obs_img = img[in_ring]

    wings_out: list[dict[str, Any]] = []
    all_flags: list[str] = []
    if q_scale_nm_inv is None:
        all_flags.append("uncalibrated_pixel_q" if q_unit_label == "pixel-q" else "spacing_unavailable_unknown_q_unit")

    for wing_idx, theta_deg in enumerate(rays):
        theta_rad = math.radians(theta_deg)
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
        q_par = obs_qx * cos_t + obs_qy * sin_t
        q_perp = -obs_qx * sin_t + obs_qy * cos_t

        # Determine nearest neighboring wing to avoid cross-wing contamination
        other_seps = [
            abs(float(_periodic_dist_deg(other, theta_deg, 360.0)))
            for j, other in enumerate(rays)
            if j != wing_idx
        ]
        min_neighbor_sep = min(other_seps) if other_seps else 180.0
        safe_half_ang = min(half_ang, max(5.0, 0.42 * min_neighbor_sep))
        tan_half = math.tan(math.radians(safe_half_ang))
        tan_neighbor_cap = math.tan(math.radians(max(6.0, 0.46 * min_neighbor_sep)))

        # Precompute per-slice q_perp_max and filter & sort once per wing (replaces 12x full scans)
        slice_q_perp_max = np.empty(n_slices, dtype=float)
        for s_idx in range(n_slices):
            qc = float(slice_centers[s_idx])
            qpm = qc * tan_half
            if min_q_perp_half_width is not None and math.isfinite(float(min_q_perp_half_width)):
                qpm = max(qpm, float(min_q_perp_half_width))
            else:
                qpm = max(qpm, 6.0 * pixel_dq)
            if min_neighbor_sep < 160.0:
                qpm = min(qpm, max(qc * tan_neighbor_cap, 4.0 * pixel_dq))
            slice_q_perp_max[s_idx] = qpm

        wing_q_perp_cap = float(np.max(slice_q_perp_max)) if n_slices > 0 else q_perp_upper
        in_wing = (q_par >= q_min) & (q_par <= q_max) & (np.abs(q_perp) <= wing_q_perp_cap)
        if np.any(in_wing):
            qp_w = q_par[in_wing]
            qt_w = q_perp[in_wing]
            im_w = obs_img[in_wing]
            order_w = np.argsort(qp_w)
            qp_s = qp_w[order_w]
            qt_s = qt_w[order_w]
            im_s = im_w[order_w]
        else:
            qp_s = np.empty(0, dtype=float)
            qt_s = np.empty(0, dtype=float)
            im_s = np.empty(0, dtype=float)

        slices_out: list[dict[str, Any]] = []
        valid_q_list: list[float] = []
        valid_fwhm_list: list[float] = []

        for s_idx in range(n_slices):
            qc = float(slice_centers[s_idx])
            hw_par = float(slice_half_w[s_idx])
            q_perp_max = float(slice_q_perp_max[s_idx])

            lo_i = int(np.searchsorted(qp_s, qc - hw_par, side="left"))
            hi_i = int(np.searchsorted(qp_s, qc + hw_par, side="right"))
            if hi_i > lo_i:
                qt_seg = qt_s[lo_i:hi_i]
                im_seg = im_s[lo_i:hi_i]
                sel_perp = np.abs(qt_seg) <= q_perp_max
                p_vals = qt_seg[sel_perp]
                i_vals = im_seg[sel_perp]
                n_px = int(p_vals.size)
            else:
                p_vals = np.empty(0, dtype=float)
                i_vals = np.empty(0, dtype=float)
                n_px = 0

            if n_px < int(min_slice_pixels):
                slices_out.append({
                    "q_center": _finite(qc),
                    "q_half_width": _finite(hw_par),
                    "q_perp_max": _finite(q_perp_max),
                    "n_pixels": n_px,
                    "valid": False,
                    "reason": "insufficient_pixels",
                    "fwhm_q_perp": None,
                    "sigma_q_perp": None,
                    "r_squared": None,
                })
                continue

            if n_perp_bins is not None:
                nb = max(int(n_perp_bins), 11)
            else:
                nb = int(np.clip(round((2.0 * q_perp_max) / max(1.1 * pixel_dq, 1.0e-9)), 15, 61))
                if nb % 2 == 0:
                    nb += 1

            p_edges = np.linspace(-q_perp_max, q_perp_max, nb + 1, dtype=float)
            p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
            p_idx = np.floor((p_vals + q_perp_max) * (float(nb) / max(2.0 * q_perp_max, 1.0e-30))).astype(np.int64)
            np.clip(p_idx, 0, nb - 1, out=p_idx)
            p_cnt = np.bincount(p_idx, minlength=nb).astype(np.int64)
            p_sum = np.bincount(p_idx, weights=i_vals, minlength=nb).astype(float)
            p_mean = np.full(nb, np.nan, dtype=float)
            np.divide(p_sum, p_cnt, out=p_mean, where=p_cnt > 0)

            sfit = _fit_1d_transverse_gaussian(
                p_centers, p_mean, p_cnt, q_perp_max=q_perp_max, min_r2=float(min_slice_r2)
            )
            slice_record = {
                "q_center": _finite(qc),
                "q_half_width": _finite(hw_par),
                "q_perp_max": _finite(q_perp_max),
                "n_pixels": n_px,
                **sfit,
            }
            slices_out.append(slice_record)
            if sfit["valid"] and sfit.get("fwhm_q_perp") is not None:
                valid_q_list.append(qc)
                valid_fwhm_list.append(float(sfit["fwhm_q_perp"]))

        q_arr = np.asarray(valid_q_list, dtype=float)
        fwhm_arr = np.asarray(valid_fwhm_list, dtype=float)
        linear_fit, quad_fit = _fit_streak_width_models(
            q_arr, fwhm_arr, q_scale_nm_inv=q_scale_nm_inv, q_unit_label=q_unit_label
        )
        wing_flags = list(dict.fromkeys([*linear_fit["flags"], *quad_fit["flags"]]))
        if len(valid_q_list) < n_slices:
            wing_flags.append("partial_slice_validity")
        all_flags.extend(wing_flags)

        wings_out.append({
            "wing_index": wing_idx,
            "wing_angle_deg": _finite(theta_deg),
            "n_total_slices": n_slices,
            "n_valid_slices": len(valid_q_list),
            "valid_q": [_finite(v) for v in valid_q_list],
            "valid_fwhm_q_perp": [_finite(v) for v in valid_fwhm_list],
            "linear_fit": linear_fit,
            "quadrature_fit": quad_fit,
            "slices": slices_out,
            "flags": wing_flags,
        })

    # Aggregate median estimates across valid wings
    def _agg(model_key: str) -> dict[str, Any]:
        dq0s = [w[model_key]["dq_perp_0"] for w in wings_out if w[model_key]["valid"] and w[model_key]["dq_perp_0"] is not None]
        mosaics = [w[model_key]["mosaic_spread_deg"] for w in wings_out if w[model_key]["valid"] and w[model_key]["mosaic_spread_deg"] is not None]
        ws = [w[model_key]["W_nm"] for w in wings_out if w[model_key]["valid"] and w[model_key]["W_nm"] is not None]
        r2s = [w[model_key]["r_squared"] for w in wings_out if w[model_key]["valid"] and w[model_key]["r_squared"] is not None]
        return {
            "n_valid_wings": len(dq0s),
            "dq_perp_0": _finite(float(np.median(dq0s))) if dq0s else None,
            "mosaic_spread_deg": _finite(float(np.median(mosaics))) if mosaics else None,
            "W_nm": _finite(float(np.median(ws))) if ws else None,
            "median_r_squared": _finite(float(np.median(r2s))) if r2s else None,
        }

    lin_agg = _agg("linear_fit")
    quad_agg = _agg("quadrature_fit")
    total_valid = sum(w["n_valid_slices"] for w in wings_out)
    overall_status = (
        "estimated"
        if (lin_agg["n_valid_wings"] > 0 or quad_agg["n_valid_wings"] > 0)
        else ("warning" if total_valid > 0 else "unavailable")
    )

    return json_safe({
        "status": overall_status,
        "q_unit": q_unit_label,
        "q_window": [q_min, q_max],
        "wing_angle_source": ray_source,
        "wings": wings_out,
        "summary": {
            "n_wings_analyzed": len(wings_out),
            "n_valid_wings_linear": lin_agg["n_valid_wings"],
            "n_valid_wings_quadrature": quad_agg["n_valid_wings"],
            "total_valid_slices": total_valid,
            "linear": lin_agg,
            "quadrature": quad_agg,
        },
        "flags": list(dict.fromkeys(all_flags)),
        "caveats": caveats,
    })
