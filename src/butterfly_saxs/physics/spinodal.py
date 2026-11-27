"""Metallic spinodal decomposition kinetics, dynamical scaling, and Khachaturyan coherent scaling."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from ..serialization import json_safe
from ._common import _finite, _finite_list, _validate_q_window


def _ols_simple(x: np.ndarray, y: np.ndarray, *, confidence_level: float = 0.95) -> dict[str, Any]:
    """Return OLS ``y = intercept + slope * x`` statistics with Student-t CIs."""
    n = int(x.size)
    if n < 2 or float(np.ptp(x)) <= 1.0e-15:
        return {
            "n_points": n,
            "dof": 0,
            "slope": None,
            "intercept": None,
            "slope_stderr": None,
            "intercept_stderr": None,
            "slope_ci": None,
            "r_squared": None,
            "rmse": None,
        }
    X = np.column_stack((np.ones(n, dtype=float), x))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(beta[0])
    slope = float(beta[1])
    pred = X @ beta
    resid = y - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1.0e-30 else (1.0 if sse <= 1.0e-30 else 0.0)
    rmse = float(math.sqrt(sse / float(n)))

    dof = n - 2
    slope_se: float | None = None
    intercept_se: float | None = None
    slope_ci: list[float | None] | None = None
    if dof > 0:
        s2 = sse / float(dof)
        cov = np.linalg.pinv(X.T @ X) * s2
        t_crit = float(student_t.ppf(0.5 * (1.0 + float(confidence_level)), df=dof))
        if cov[0, 0] >= 0.0:
            intercept_se = _finite(math.sqrt(float(cov[0, 0])))
        if cov[1, 1] >= 0.0:
            slope_se = _finite(math.sqrt(float(cov[1, 1])))
            if slope_se is not None:
                slope_ci = [_finite(slope - t_crit * slope_se), _finite(slope + t_crit * slope_se)]
    return {
        "n_points": n,
        "dof": dof,
        "slope": _finite(slope),
        "intercept": _finite(intercept),
        "slope_stderr": slope_se,
        "intercept_stderr": intercept_se,
        "slope_ci": slope_ci,
        "r_squared": _finite(r2),
        "rmse": _finite(rmse),
    }


def _interior_peak_1d(q: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, bool]:
    """Locate interior maximum with 3-point quadratic interpolation; reject endpoints."""
    if q.size < 3:
        return None, None, True
    idx = int(np.argmax(y))
    if idx == 0 or idx == q.size - 1:
        return None, None, True
    y_m, y_0, y_p = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    x_m, x_0, x_p = float(q[idx - 1]), float(q[idx]), float(q[idx + 1])
    denom = (x_m - x_0) * (x_m - x_p) * (x_0 - x_p)
    if abs(denom) > 1.0e-30:
        a_coef = (x_p * (y_0 - y_m) + x_0 * (y_m - y_p) + x_m * (y_p - y_0)) / denom
        b_coef = (
            (x_p ** 2) * (y_m - y_0)
            + (x_0 ** 2) * (y_p - y_m)
            + (x_m ** 2) * (y_0 - y_p)
        ) / denom
        if a_coef < -1.0e-30:
            x_v = -b_coef / (2.0 * a_coef)
            if x_m <= x_v <= x_p:
                y_v = float(np.interp(x_v, [x_m, x_0, x_p], [y_m, y_0, y_p]))
                c_coef = y_0 - a_coef * (x_0 ** 2) - b_coef * x_0
                y_quad = a_coef * (x_v ** 2) + b_coef * x_v + c_coef
                return _finite(x_v), _finite(max(y_v, y_quad)), False
    return _finite(x_0), _finite(y_0), False


def compute_dynamical_scaling_collapse(
    q_series: Any,
    I_series: Any,
    *,
    times: Any = None,
    q_mode: str = "q1",
    q_window: tuple[float, float] | None = None,
    x_grid: Any = None,
    furukawa_d: int = 3,
    furukawa_gamma: float | None = None,
    late_fraction: float = 0.5,
) -> dict[str, Any]:
    """Evaluate LMK/Furukawa dynamical scaling ``F(x,t) = q_char^3 * I(q,t) / Q(t)``.

    Accepts either a shared 1D ``q`` array paired with a 2D ``(n_frames, n_q)``
    intensity matrix or matching sequences of 1D ``q`` and ``I`` arrays.
    """
    mode = str(q_mode or "q1").strip().lower()
    if mode not in {"q1", "q_peak", "q_kratky", "q_peak_or_q1"}:
        raise ValueError(f"Unsupported q_mode={q_mode!r}; use 'q1', 'q_peak', 'q_kratky', or 'q_peak_or_q1'")
    if q_window is not None:
        q_lo, q_hi = _validate_q_window(q_window)
    else:
        q_lo, q_hi = None, None
    gamma = float(furukawa_gamma) if furukawa_gamma is not None else float(int(furukawa_d) + 1)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("furukawa_gamma must be finite and positive")

    I_raw = np.asarray(I_series, dtype=float) if not isinstance(I_series, (list, tuple)) else I_series
    if isinstance(I_raw, np.ndarray) and I_raw.ndim == 2:
        n_frames = I_raw.shape[0]
        I_list = [I_raw[i] for i in range(n_frames)]
    elif isinstance(I_series, Sequence):
        I_list = [np.asarray(arr, dtype=float) for arr in I_series]
        n_frames = len(I_list)
    else:
        raise ValueError("I_series must be a 2D array or sequence of 1D arrays")
    if n_frames == 0:
        raise ValueError("I_series must contain at least one curve")

    q_arr_raw = np.asarray(q_series, dtype=float) if not isinstance(q_series, (list, tuple)) else q_series
    if isinstance(q_arr_raw, np.ndarray) and q_arr_raw.ndim == 1:
        q_list = [q_arr_raw] * n_frames
    elif isinstance(q_arr_raw, np.ndarray) and q_arr_raw.ndim == 2 and q_arr_raw.shape[0] == n_frames:
        q_list = [q_arr_raw[i] for i in range(n_frames)]
    elif isinstance(q_series, Sequence) and len(q_series) == n_frames:
        q_list = [np.asarray(arr, dtype=float) for arr in q_series]
    elif isinstance(q_series, Sequence) and len(q_series) > 0 and isinstance(q_series[0], (int, float, np.number)):
        q_shared = np.asarray(q_series, dtype=float)
        q_list = [q_shared] * n_frames
    else:
        raise ValueError("q_series must be a shared 1D array or match I_series length")

    if times is not None:
        t_arr = np.asarray(times, dtype=float)
        if t_arr.shape != (n_frames,):
            raise ValueError("times must match I_series frame count")
    else:
        t_arr = np.full(n_frames, np.nan, dtype=float)

    if x_grid is not None:
        xg = np.asarray(x_grid, dtype=float)
        if xg.ndim != 1 or xg.size < 5 or np.any(~np.isfinite(xg)) or np.any(xg <= 0.0):
            raise ValueError("x_grid must be a finite positive 1D array with >= 5 points")
        xg = np.sort(xg)
    else:
        xg = np.linspace(0.4, 2.4, 65, dtype=float)

    frame_records: list[dict[str, Any]] = []
    f_lmk_rows: list[np.ndarray] = []
    f_peak_rows: list[np.ndarray] = []
    valid_indices: list[int] = []
    flags: list[str] = []

    for idx in range(n_frames):
        qv = q_list[idx]
        Iv = I_list[idx]
        if qv.ndim != 1 or Iv.ndim != 1 or qv.shape != Iv.shape:
            raise ValueError(f"Frame {idx} q and I must be matching 1D arrays")
        mask = np.isfinite(qv) & np.isfinite(Iv) & (qv > 0.0)
        if q_lo is not None and q_hi is not None:
            mask &= (qv >= q_lo) & (qv <= q_hi)
        qu = qv[mask]
        Iu = Iv[mask]
        if qu.size > 1:
            order = np.argsort(qu, kind="stable")
            qu = qu[order]
            Iu = Iu[order]
        if qu.size < 6:
            frame_records.append({
                "frame_index": idx,
                "time": _finite(t_arr[idx]),
                "valid": False,
                "reason": "insufficient_points",
                "Q_win": None,
                "q1": None,
                "q_peak": None,
                "I_peak": None,
                "q_kratky": None,
                "q_char": None,
                "q_char_source": None,
                "negative_fraction": None,
            })
            f_lmk_rows.append(np.full_like(xg, np.nan))
            f_peak_rows.append(np.full_like(xg, np.nan))
            continue

        neg_frac = float(np.mean(Iu < 0.0))
        I_pos = np.maximum(Iu, 0.0)
        q2I = (qu ** 2) * I_pos
        q3I = (qu ** 3) * I_pos
        q_inv = float(np.trapezoid(q2I, qu))
        if not math.isfinite(q_inv) or q_inv <= 0.0:
            frame_records.append({
                "frame_index": idx,
                "time": _finite(t_arr[idx]),
                "valid": False,
                "reason": "nonpositive_invariant",
                "Q_win": _finite(q_inv),
                "q1": None,
                "q_peak": None,
                "I_peak": None,
                "q_kratky": None,
                "q_char": None,
                "q_char_source": None,
                "negative_fraction": _finite(neg_frac),
            })
            f_lmk_rows.append(np.full_like(xg, np.nan))
            f_peak_rows.append(np.full_like(xg, np.nan))
            continue

        q1_val = float(np.trapezoid(q3I, qu) / q_inv)
        q_pk, I_pk, pk_edge = _interior_peak_1d(qu, Iu)
        q_kr, _, kr_edge = _interior_peak_1d(qu, (qu ** 2) * Iu)

        if mode == "q1":
            q_char, q_src = q1_val, "q1"
        elif mode == "q_peak":
            q_char, q_src = (q_pk, "q_peak") if q_pk is not None else (None, "q_peak_unavailable")
        elif mode == "q_kratky":
            q_char, q_src = (q_kr, "q_kratky") if q_kr is not None else (None, "q_kratky_unavailable")
        else:
            q_char, q_src = (q_pk, "q_peak") if q_pk is not None else (q1_val, "q1_fallback")

        if q_char is None or not math.isfinite(q_char) or q_char <= 0.0:
            frame_records.append({
                "frame_index": idx,
                "time": _finite(t_arr[idx]),
                "valid": False,
                "reason": f"{mode}_unavailable",
                "Q_win": _finite(q_inv),
                "q1": _finite(q1_val),
                "q_peak": q_pk,
                "I_peak": I_pk,
                "q_kratky": q_kr,
                "q_char": None,
                "q_char_source": q_src,
                "negative_fraction": _finite(neg_frac),
            })
            f_lmk_rows.append(np.full_like(xg, np.nan))
            f_peak_rows.append(np.full_like(xg, np.nan))
            continue

        x_obs = qu / q_char
        F_obs = (q_char ** 3) * I_pos / q_inv
        F_interp = np.interp(xg, x_obs, F_obs, left=np.nan, right=np.nan)
        I_at_qc = float(np.interp(q_char, qu, I_pos))
        if I_at_qc > 0.0 and math.isfinite(I_at_qc):
            F_pk_interp = np.interp(xg, x_obs, I_pos / I_at_qc, left=np.nan, right=np.nan)
        else:
            F_pk_interp = np.full_like(xg, np.nan)

        valid_indices.append(idx)
        f_lmk_rows.append(F_interp)
        f_peak_rows.append(F_pk_interp)
        frame_records.append({
            "frame_index": idx,
            "time": _finite(t_arr[idx]),
            "valid": True,
            "reason": None,
            "Q_win": _finite(q_inv),
            "q1": _finite(q1_val),
            "q_peak": q_pk,
            "peak_edge_hit": bool(pk_edge),
            "I_peak": I_pk,
            "q_kratky": q_kr,
            "kratky_edge_hit": bool(kr_edge),
            "q_char": _finite(q_char),
            "q_char_source": q_src,
            "negative_fraction": _finite(neg_frac),
        })

    if len(valid_indices) < 2:
        flags.append("insufficient_valid_frames_for_collapse")
        return json_safe({
            "status": "unavailable",
            "n_frames_total": n_frames,
            "n_frames_valid": len(valid_indices),
            "q_mode": mode,
            "furukawa_gamma": gamma,
            "x_grid": _finite_list(xg),
            "frames": frame_records,
            "F_master": None,
            "F_master_std": None,
            "F_furukawa": _finite_list((1.0 + 0.5 * gamma) * (xg ** 2) / (0.5 * gamma + (xg ** (2.0 + gamma)))),
            "collapse_cv_all": None,
            "collapse_cv_late": None,
            "furukawa_peak_rms": None,
            "low_x_slope": None,
            "high_x_slope": None,
            "flags": flags,
        })

    F_mat = np.asarray([f_lmk_rows[i] for i in valid_indices], dtype=float)
    F_pk_mat = np.asarray([f_peak_rows[i] for i in valid_indices], dtype=float)

    def _collapse_stats(sub_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
        counts = np.sum(np.isfinite(sub_mat), axis=0)
        mean_v = np.full(xg.shape, np.nan, dtype=float)
        std_v = np.full(xg.shape, np.nan, dtype=float)
        ok = counts >= 2
        if np.any(ok):
            mean_v[ok] = np.nanmean(sub_mat[:, ok], axis=0)
            std_v[ok] = np.nanstd(sub_mat[:, ok], axis=0)
        core = ok & (xg >= 0.5) & (xg <= 2.0) & (mean_v > 1.0e-12)
        if np.sum(core) < 3:
            core = ok & (mean_v > 1.0e-12)
        cv = float(np.mean(std_v[core] / mean_v[core])) if np.any(core) else None
        return mean_v, std_v, _finite(cv)

    F_master, F_std, cv_all = _collapse_stats(F_mat)
    n_late = max(2, min(len(valid_indices), int(math.ceil(float(late_fraction) * len(valid_indices)))))
    F_late_mean, _, cv_late = _collapse_stats(F_mat[-n_late:])
    _, _, cv_peak_all = _collapse_stats(F_pk_mat)
    _, _, cv_peak_late = _collapse_stats(F_pk_mat[-n_late:])
    F_pk_master = np.full(xg.shape, np.nan, dtype=float)
    ok_pk = np.sum(np.isfinite(F_pk_mat), axis=0) >= 2
    if np.any(ok_pk):
        F_pk_master[ok_pk] = np.nanmean(F_pk_mat[:, ok_pk], axis=0)
        pk_max = float(np.nanmax(F_pk_master[ok_pk]))
        if math.isfinite(pk_max) and pk_max > 0.0:
            F_pk_master[ok_pk] = F_pk_master[ok_pk] / pk_max

    F_furukawa = (1.0 + 0.5 * gamma) * (xg ** 2) / (0.5 * gamma + (xg ** (2.0 + gamma)))
    furukawa_rms: float | None = None
    f_cmp = ok_pk & np.isfinite(F_pk_master) & (xg >= 0.45) & (xg <= 2.2)
    if np.sum(f_cmp) >= 5:
        furukawa_rms = _finite(math.sqrt(float(np.mean((F_pk_master[f_cmp] - F_furukawa[f_cmp]) ** 2))))

    # Asymptotic log-log scaling slopes on collapsed late master curve (fall back to all)
    F_slope_ref = np.where(np.isfinite(F_late_mean), F_late_mean, F_master)
    low_m = np.isfinite(F_slope_ref) & (F_slope_ref > 0.0) & (xg >= 0.45) & (xg <= 0.85)
    high_m = np.isfinite(F_slope_ref) & (F_slope_ref > 0.0) & (xg >= 1.35) & (xg <= 2.30)
    low_ols = _ols_simple(np.log(xg[low_m]), np.log(F_slope_ref[low_m])) if np.sum(low_m) >= 3 else {"slope": None, "r_squared": None}
    high_ols = _ols_simple(np.log(xg[high_m]), np.log(F_slope_ref[high_m])) if np.sum(high_m) >= 3 else {"slope": None, "r_squared": None}

    return json_safe({
        "status": "estimated",
        "n_frames_total": n_frames,
        "n_frames_valid": len(valid_indices),
        "n_late_frames": n_late,
        "q_mode": mode,
        "furukawa_gamma": gamma,
        "x_grid": _finite_list(xg),
        "frames": frame_records,
        "F_master": _finite_list(F_master),
        "F_master_std": _finite_list(F_std),
        "F_late_master": _finite_list(F_late_mean),
        "F_peak_norm_master": _finite_list(F_pk_master),
        "F_furukawa": _finite_list(F_furukawa),
        "collapse_cv_all": cv_all,
        "collapse_cv_late": cv_late,
        "collapse_cv_peak_all": cv_peak_all,
        "collapse_cv_peak_late": cv_peak_late,
        "furukawa_peak_rms": furukawa_rms,
        "low_x_slope": low_ols["slope"],
        "low_x_slope_r2": low_ols["r_squared"],
        "high_x_slope": high_ols["slope"],
        "high_x_slope_r2": high_ols["r_squared"],
        "flags": flags,
        "caveat": (
            "LMK/Furukawa normalization int x^2 F(x) dx = 1 tests dynamical self-similarity; "
            "low-x slope >~3 favors bicontinuous conserved spinodal (Yeung x^4) over droplet x^2."
        ),
    })


def analyze_joint_spinodal_exponents(
    times: Any,
    q_char: Any,
    I_peak: Any,
    Q_inv: Any,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Jointly fit ``q_char ~ t^-a``, ``I_peak ~ t^b``, ``Q_inv ~ t^c`` and classify ``b/a``.

    Distinguishes active spinodal unmixing (``b/a > 3``, ``c > 0``), self-similar
    coarsening (``b/a ~ 3``, ``Q ~ const``), elastic-retarded arrest (``a -> 0``),
    and dissolution/reversion (``c < 0``).
    """
    t_arr = np.asarray(times, dtype=float)
    q_arr = np.asarray(q_char, dtype=float)
    I_arr = np.asarray(I_peak, dtype=float)
    Q_arr = np.asarray(Q_inv, dtype=float)
    if (
        t_arr.ndim != 1
        or q_arr.shape != t_arr.shape
        or I_arr.shape != t_arr.shape
        or Q_arr.shape != t_arr.shape
        or t_arr.size == 0
    ):
        raise ValueError("times, q_char, I_peak, and Q_inv must be non-empty matching 1D arrays")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must lie in (0, 1)")

    mask = (
        np.isfinite(t_arr) & (t_arr > 0.0)
        & np.isfinite(q_arr) & (q_arr > 0.0)
        & np.isfinite(I_arr) & (I_arr > 0.0)
        & np.isfinite(Q_arr) & (Q_arr > 0.0)
    )
    if t_min is not None:
        mask &= t_arr >= float(t_min)
    if t_max is not None:
        mask &= t_arr <= float(t_max)

    tu = t_arr[mask]
    qu = q_arr[mask]
    Iu = I_arr[mask]
    Qu = Q_arr[mask]
    order = np.argsort(tu, kind="stable")
    tu, qu, Iu, Qu = tu[order], qu[order], Iu[order], Qu[order]

    n_pts = int(tu.size)
    if n_pts < 3 or int(np.unique(tu).size) < 3 or float(np.ptp(tu)) <= 1.0e-12:
        return json_safe({
            "success": False,
            "status": "unavailable",
            "n_points": n_pts,
            "a": None,
            "a_stderr": None,
            "b": None,
            "b_stderr": None,
            "c": None,
            "c_stderr": None,
            "ba_ratio": None,
            "ba_ratio_stderr": None,
            "scaling_closure_residual": None,
            "regime": "insufficient_points",
            "flags": ["insufficient_valid_spinodal_series_points"],
        })

    lx = np.log(tu)
    fit_q = _ols_simple(lx, np.log(qu), confidence_level=confidence_level)
    fit_I = _ols_simple(lx, np.log(Iu), confidence_level=confidence_level)
    fit_Q = _ols_simple(lx, np.log(Qu), confidence_level=confidence_level)

    a_val = -fit_q["slope"] if fit_q["slope"] is not None else None
    a_se = fit_q["slope_stderr"]
    b_val = fit_I["slope"]
    b_se = fit_I["slope_stderr"]
    c_val = fit_Q["slope"]
    c_se = fit_Q["slope_stderr"]

    ba_ratio: float | None = None
    ba_se: float | None = None
    if a_val is not None and b_val is not None and abs(a_val) > 0.015:
        ba_ratio = _finite(b_val / a_val)
        if ba_ratio is not None and a_se is not None and b_se is not None and abs(b_val) > 1.0e-12:
            rel2 = (a_se / a_val) ** 2 + (b_se / b_val) ** 2
            ba_se = _finite(abs(ba_ratio) * math.sqrt(rel2))

    closure_res = (
        _finite(b_val - 3.0 * a_val - c_val)
        if (a_val is not None and b_val is not None and c_val is not None)
        else None
    )

    flags: list[str] = []
    if c_val is not None and c_val < -0.05 and (b_val is None or b_val < 0.0):
        regime = "dissolution_or_reversion"
    elif a_val is not None and abs(a_val) <= 0.03:
        if b_val is not None and b_val > 0.05 and (c_val is not None and c_val > 0.02):
            regime = "fixed_wavelength_amplification_or_arrest"
        else:
            regime = "stationary_or_elastic_arrest"
    elif a_val is not None and a_val > 0.03:
        if ba_ratio is not None and (ba_ratio > 3.5 or (c_val is not None and c_val > 0.05 and ba_ratio > 3.05)):
            regime = "active_spinodal_amplification_and_coarsening"
        elif ba_ratio is not None and 2.5 <= ba_ratio <= 3.5:
            regime = "self_similar_coarsening_lsw" if a_val >= 0.27 else "self_similar_elastic_retarded"
        else:
            regime = "broadening_or_transitional_coarsening"
        if a_val < 0.25:
            flags.append("sub_lsw_elastic_retarded_coarsening")
    else:
        regime = "contracting_or_peak_emergence"

    return json_safe({
        "success": True,
        "status": "estimated",
        "n_points": n_pts,
        "t_min": _finite(float(tu[0])),
        "t_max": _finite(float(tu[-1])),
        "a": _finite(a_val),
        "a_stderr": a_se,
        "a_r_squared": fit_q["r_squared"],
        "b": _finite(b_val),
        "b_stderr": b_se,
        "b_r_squared": fit_I["r_squared"],
        "c": _finite(c_val),
        "c_stderr": c_se,
        "c_r_squared": fit_Q["r_squared"],
        "ba_ratio": ba_ratio,
        "ba_ratio_stderr": ba_se,
        "scaling_closure_residual": closure_res,
        "regime": regime,
        "flags": flags,
        "caveat": (
            "b/a > 3 with c > 0 indicates active amplitude amplification alongside domain coarsening; "
            "b/a ~ 3 with c ~ 0 indicates self-similar late-stage coarsening."
        ),
    })


def analyze_cahn_hilliard_cook_early(
    q: Any,
    I_matrix: Any,
    times_s: Any,
    *,
    q_window: tuple[float, float] | None = None,
    fit_q_range: tuple[float, float] | None = None,
    min_frames: int = 3,
) -> dict[str, Any]:
    """Test linearized Cahn-Hilliard-Cook growth ``R(q)/q^2 = -D_eff - 2*M*kappa*q^2``.

    Quantifies amplification rates ``R(q) = 0.5 * d ln I(q,t) / dt``, tests the
    linear dispersion relation around the peak, and tracks pairwise crossover
    wavevector ``q_cross(t_i, t_{i+1})`` stability to distinguish genuine early
    linearized amplification from nonlinear or coarsening regimes (Khachaturyan ch05).
    """
    q_arr = np.asarray(q, dtype=float)
    I_mat = np.asarray(I_matrix, dtype=float)
    t_arr = np.asarray(times_s, dtype=float)
    if q_arr.ndim != 1 or I_mat.ndim != 2 or t_arr.ndim != 1:
        raise ValueError("q must be 1D, I_matrix must be 2D (n_frames, n_q), times_s must be 1D")
    if I_mat.shape != (t_arr.size, q_arr.size):
        raise ValueError("I_matrix shape must equal (len(times_s), len(q))")
    if t_arr.size < int(min_frames):
        return json_safe({
            "success": False,
            "status": "unavailable",
            "reason": "insufficient_frames",
            "n_frames": int(t_arr.size),
            "linear_chc_valid": False,
            "flags": ["insufficient_frames_for_chc"],
        })

    # Sort frames chronologically and filter finite time stamps
    t_ok = np.isfinite(t_arr)
    t_arr = t_arr[t_ok]
    I_mat = I_mat[t_ok, :]
    order_t = np.argsort(t_arr, kind="stable")
    t_arr = t_arr[order_t]
    I_mat = I_mat[order_t, :]

    q_mask = np.isfinite(q_arr) & (q_arr > 0.0)
    if q_window is not None:
        q_lo, q_hi = _validate_q_window(q_window)
        q_mask &= (q_arr >= q_lo) & (q_arr <= q_hi)
    q_sel = q_arr[q_mask]
    I_sel = I_mat[:, q_mask]
    order_q = np.argsort(q_sel, kind="stable")
    q_sel = q_sel[order_q]
    I_sel = I_sel[:, order_q]

    pos_q = np.all(np.isfinite(I_sel) & (I_sel > 0.0), axis=0)
    q_pos = q_sel[pos_q]
    I_pos = I_sel[:, pos_q]

    if t_arr.size < int(min_frames) or q_pos.size < 6 or float(np.ptp(t_arr)) <= 1.0e-12:
        return json_safe({
            "success": False,
            "status": "unavailable",
            "reason": "insufficient_positive_bins_or_time_span",
            "n_frames": int(t_arr.size),
            "n_positive_q": int(q_pos.size),
            "linear_chc_valid": False,
            "flags": ["insufficient_valid_bins_for_chc"],
        })

    ln_I = np.log(I_pos)
    t_centered = t_arr - float(np.mean(t_arr))
    denom_t = float(np.sum(t_centered ** 2))
    slopes = (t_centered[:, None] * (ln_I - np.mean(ln_I, axis=0))).sum(axis=0) / denom_t
    R_q = 0.5 * slopes

    # Pairwise crossover wavevector q_cross above peak where I_{i+1} drops below I_i
    ref_peak_idx = int(np.argmax(I_pos[-1]))
    q_cross_vals: list[float] = []
    for i in range(I_pos.shape[0] - 1):
        diff = I_pos[i + 1] - I_pos[i]
        found: float | None = None
        for j in range(max(ref_peak_idx, 0), q_pos.size - 1):
            if diff[j] > 0.0 and diff[j + 1] <= 0.0:
                denom = diff[j] - diff[j + 1]
                frac = float(diff[j] / denom) if abs(denom) > 1.0e-30 else 0.0
                found = float(q_pos[j] + frac * (q_pos[j + 1] - q_pos[j]))
                break
        if found is not None and math.isfinite(found):
            q_cross_vals.append(found)

    q_cross_med = _finite(float(np.median(q_cross_vals))) if q_cross_vals else None
    q_cross_shift_frac: float | None = None
    if len(q_cross_vals) >= 2 and q_cross_med is not None and q_cross_med > 0.0:
        q_cross_shift_frac = _finite((max(q_cross_vals) - min(q_cross_vals)) / q_cross_med)

    # Select CHC regression band
    if fit_q_range is not None:
        flo, fhi = _validate_q_window(fit_q_range)
        fit_mask = (q_pos >= flo) & (q_pos <= fhi)
    else:
        pos_r = np.flatnonzero(R_q > 0.0)
        if pos_r.size >= 5:
            i_start = int(pos_r[0])
            i_end = min(int(pos_r[-1]) + max(3, pos_r.size // 4), q_pos.size - 1)
            fit_mask = np.zeros(q_pos.size, dtype=bool)
            fit_mask[i_start : i_end + 1] = True
        else:
            fit_mask = np.ones(q_pos.size, dtype=bool)

    if int(np.count_nonzero(fit_mask)) < 4:
        fit_mask = np.ones(q_pos.size, dtype=bool)

    qf = q_pos[fit_mask]
    Rf = R_q[fit_mask]
    x_chc = qf ** 2
    y_chc = Rf / np.maximum(x_chc, 1.0e-30)
    ols = _ols_simple(x_chc, y_chc)

    intercept = ols["intercept"]
    slope = ols["slope"]
    r2 = ols["r_squared"]

    uphill_sign = bool(
        intercept is not None and slope is not None and intercept > 0.0 and slope < 0.0
    )
    D_eff = _finite(-intercept) if intercept is not None else None
    two_M_kappa = _finite(-slope) if slope is not None else None
    q_c: float | None = None
    q_m_chc: float | None = None
    R_max_chc: float | None = None
    if uphill_sign:
        assert intercept is not None and two_M_kappa is not None
        qc2 = intercept / two_M_kappa
        if qc2 > 0.0:
            q_c = _finite(math.sqrt(qc2))
            q_m_chc = _finite(math.sqrt(0.5 * qc2))
            R_max_chc = _finite((intercept ** 2) / (4.0 * two_M_kappa))

    flags: list[str] = []
    if not uphill_sign:
        flags.append("non_chc_dispersion_sign")
    if r2 is None or r2 < 0.85:
        flags.append("nonlinear_or_cook_regime_detected")
    if q_cross_shift_frac is not None and q_cross_shift_frac > 0.15:
        flags.append("moving_q_cross_coarsening_active")

    linear_valid = bool(
        uphill_sign
        and r2 is not None and r2 >= 0.85
        and (q_cross_shift_frac is None or q_cross_shift_frac <= 0.15)
    )

    return json_safe({
        "success": True,
        "status": "estimated" if linear_valid else "warning",
        "n_frames": int(t_arr.size),
        "n_fit_q_bins": int(qf.size),
        "fit_q_range": [_finite(float(qf[0])), _finite(float(qf[-1]))],
        "uphill_diffusion_sign": uphill_sign,
        "linear_chc_valid": linear_valid,
        "D_eff": D_eff,
        "two_M_kappa": two_M_kappa,
        "q_c": q_c,
        "q_m_chc": q_m_chc,
        "R_max_chc": R_max_chc,
        "r_squared": r2,
        "q_cross_values": _finite_list(np.asarray(q_cross_vals, dtype=float)),
        "q_cross_median": q_cross_med,
        "q_cross_relative_shift": q_cross_shift_frac,
        "q_bins": _finite_list(q_pos),
        "R_q": _finite_list(R_q),
        "R_over_q2": _finite_list(R_q / (q_pos ** 2)),
        "flags": flags,
        "caveat": (
            "Per Khachaturyan ch05, linearized Cahn-Hilliard holds only when R(q)/q^2 vs q^2 "
            "is linear with negative D_eff, positive 2*M*kappa, and stationary q_cross."
        ),
    })


def analyze_khachaturyan_complex_scaling(
    L_star_nm: Any,
    W_lateral_nm: Any,
    *,
    times: Any = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Evaluate Khachaturyan ch10 coherent modulated complex scaling ``lambda_0 = sqrt(r_1 * L_t)``.

    Computes ``r_1 = lambda_0^2 / L_t`` (with ``lambda_0 = L_star_nm`` and transverse
    macro-domain / Ruland lateral coherence ``L_t = W_lateral_nm``) and fits
    ``d ln(lambda_0) / d ln(L_t)`` against the theoretical 0.50 expectation.
    """
    L_arr = np.asarray(L_star_nm, dtype=float)
    W_arr = np.asarray(W_lateral_nm, dtype=float)
    if L_arr.ndim != 1 or W_arr.shape != L_arr.shape or L_arr.size == 0:
        raise ValueError("L_star_nm and W_lateral_nm must be non-empty matching 1D arrays")

    mask = np.isfinite(L_arr) & (L_arr > 0.0) & np.isfinite(W_arr) & (W_arr > 0.0)
    Lu = L_arr[mask]
    Wu = W_arr[mask]
    n_valid = int(Lu.size)
    if n_valid == 0:
        return json_safe({
            "success": False,
            "status": "unavailable",
            "n_valid": 0,
            "r1_median_nm": None,
            "r1_iqr_nm": None,
            "dlnL_dlnW": None,
            "dlnL_dlnW_stderr": None,
            "r_squared": None,
            "regime": "no_valid_pairs",
            "r1_series_nm": [None] * L_arr.size,
            "flags": ["no_valid_L_W_pairs"],
        })

    r1_all = np.where(mask, (L_arr ** 2) / np.where(mask, W_arr, 1.0), np.nan)
    r1_u = (Lu ** 2) / Wu
    med_r1 = float(np.median(r1_u))
    q25, q75 = float(np.percentile(r1_u, 25.0)), float(np.percentile(r1_u, 75.0))
    w_span_ratio = float(np.max(Wu) / np.min(Wu)) if float(np.min(Wu)) > 0.0 else 1.0

    ols = _ols_simple(np.log(Wu), np.log(Lu), confidence_level=confidence_level)
    alpha = ols["slope"]
    alpha_se = ols["slope_stderr"]
    r2 = ols["r_squared"]

    if n_valid < 3 or w_span_ratio <= 1.05:
        regime = "insufficient_dynamic_range"
    elif alpha is not None and 0.30 <= alpha <= 0.70 and (r2 is not None and r2 >= 0.30):
        regime = "coherent_complex_scaling"
    elif alpha is not None and abs(alpha) < 0.15 and w_span_ratio >= 1.25:
        regime = "elastic_wavelength_arrest"
    else:
        regime = "transitional_or_weak_coupling"

    time_L_exp: float | None = None
    time_W_exp: float | None = None
    if times is not None:
        t_arr = np.asarray(times, dtype=float)
        if t_arr.shape != L_arr.shape:
            raise ValueError("times must match L_star_nm shape")
        tm = mask & np.isfinite(t_arr) & (t_arr > 0.0)
        if int(np.count_nonzero(tm)) >= 3 and int(np.unique(t_arr[tm]).size) >= 3:
            time_L_exp = _ols_simple(np.log(t_arr[tm]), np.log(L_arr[tm]))["slope"]
            time_W_exp = _ols_simple(np.log(t_arr[tm]), np.log(W_arr[tm]))["slope"]

    return json_safe({
        "success": True,
        "status": "estimated",
        "n_valid": n_valid,
        "L_star_median_nm": _finite(float(np.median(Lu))),
        "W_lateral_median_nm": _finite(float(np.median(Wu))),
        "W_span_ratio": _finite(w_span_ratio),
        "r1_median_nm": _finite(med_r1),
        "r1_mean_nm": _finite(float(np.mean(r1_u))),
        "r1_iqr_nm": [_finite(q25), _finite(q75)],
        "dlnL_dlnW": alpha,
        "dlnL_dlnW_stderr": alpha_se,
        "dlnL_dlnW_ci": ols["slope_ci"],
        "dlnL_dlnW_r_squared": r2,
        "theory_prediction_dlnL_dlnW": 0.5,
        "time_exponent_L": time_L_exp,
        "time_exponent_W": time_W_exp,
        "regime": regime,
        "r1_series_nm": _finite_list(r1_all),
        "flags": [],
        "caveat": (
            "Khachaturyan ch10 Eqs. (10.3.33)-(10.3.35) predict lambda_0 = sqrt(r_1 * L_t) "
            "(d ln lambda_0 / d ln L_t = 0.50) before elastic wavelength arrest sets a plateau lambda_0*."
        ),
    })


__all__ = [
    "analyze_cahn_hilliard_cook_early",
    "analyze_joint_spinodal_exponents",
    "analyze_khachaturyan_complex_scaling",
    "compute_dynamical_scaling_collapse",
]
