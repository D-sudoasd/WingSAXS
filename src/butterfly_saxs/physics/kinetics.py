"""Power-law lamellar coarsening and Arrhenius rate kinetics fitting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from ..serialization import json_safe
from ._common import _GAS_CONSTANT_KJ_MOL_K, _KJ_MOL_PER_EV, _finite, _finite_list


def _resolve_1d_weights(
    n: int,
    sigma: Any,
    weights: Any,
) -> tuple[np.ndarray, bool]:
    """Return ``(sqrt_weights, has_absolute_sigma)`` after validating inputs."""
    if sigma is not None and weights is not None:
        raise ValueError("provide either sigma or weights, not both")
    if sigma is not None:
        s = np.asarray(sigma, dtype=float)
        if s.shape != (n,):
            raise ValueError("sigma must match data shape")
        if np.any(~np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError("sigma values must be finite and strictly positive")
        return 1.0 / s, True
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must match data shape")
        if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError("weights must be finite and strictly positive")
        return np.sqrt(w), False
    return np.ones(n, dtype=float), False


def fit_power_law_coarsening(
    t: Any,
    L: Any,
    *,
    sigma_L: Any = None,
    weights: Any = None,
    model: str = "power_law",
    t_min: float | None = None,
    t_max: float | None = None,
    confidence_level: float = 0.95,
    fix_L0: float | None = None,
) -> dict[str, Any]:
    """Fit power-law coarsening kinetics ``L(t) = A * t^n`` or ``L(t) = L0 + k * t^n``.

    Supports weighted (``sigma_L`` or ``weights``) and unweighted regression,
    strictly omitting non-finite samples and returning parameter standard errors
    and Student-t confidence intervals.
    """
    t_arr = np.asarray(t, dtype=float)
    L_arr = np.asarray(L, dtype=float)
    if t_arr.ndim != 1 or L_arr.ndim != 1 or t_arr.shape != L_arr.shape or t_arr.size == 0:
        raise ValueError("t and L must be non-empty matching 1D arrays")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must lie in (0, 1)")

    norm_model = str(model or "power_law").strip().lower().replace("-", "_")
    if norm_model in {"power_law", "pure", "at_n", "a_t_n"}:
        model_kind = "power_law"
    elif norm_model in {"offset_power_law", "offset", "l0_kt_n", "l0_k_t_n", "lsw"}:
        model_kind = "offset_power_law"
    else:
        raise ValueError("model must be 'power_law' (A*t^n) or 'offset_power_law' (L0 + k*t^n)")

    mask = np.isfinite(t_arr) & np.isfinite(L_arr)
    if t_min is not None:
        mask &= t_arr >= float(t_min)
    if t_max is not None:
        mask &= t_arr <= float(t_max)
    if model_kind == "power_law":
        mask &= (t_arr > 0.0) & (L_arr > 0.0)
    else:
        mask &= t_arr >= 0.0

    # Validate weights/sigma on selected finite entries
    idx_sel = np.flatnonzero(mask)
    if sigma_L is not None and weights is not None:
        raise ValueError("provide either sigma_L or weights, not both")
    sig_sel = None
    w_sel = None
    if sigma_L is not None:
        s_full = np.asarray(sigma_L, dtype=float)
        if s_full.shape != t_arr.shape:
            raise ValueError("sigma_L must match t and L shape")
        sig_sel = s_full[idx_sel]
    if weights is not None:
        w_full = np.asarray(weights, dtype=float)
        if w_full.shape != t_arr.shape:
            raise ValueError("weights must match t and L shape")
        w_sel = w_full[idx_sel]

    t_use = t_arr[idx_sel]
    L_use = L_arr[idx_sel]
    n_pts = int(t_use.size)
    n_params = 2 if (model_kind == "power_law" or fix_L0 is not None) else 3
    n_distinct_t = int(np.unique(t_use).size) if n_pts else 0
    t_span = float(np.ptp(t_use)) if n_pts else 0.0
    if n_pts < n_params or n_distinct_t < n_params or t_span <= 1.0e-12:
        unavail_flag = (
            "insufficient_points_for_coarsening_fit"
            if n_pts < n_params
            else "degenerate_time_span_for_coarsening_fit"
        )
        unavail_msg = (
            f"Need at least {n_params} valid points for {model_kind}; found {n_pts}."
            if n_pts < n_params
            else f"Need at least {n_params} distinct time points spanning > 0 for {model_kind}; found {n_distinct_t}."
        )
        return json_safe({
            "success": False,
            "status": "unavailable",
            "model": model_kind,
            "message": unavail_msg,
            "n_points": n_pts,
            "dof": 0,
            "parameters": None,
            "stderr": None,
            "ci": None,
            "r_squared": None,
            "rmse": None,
            "weighted_rmse": None,
            "reduced_chi_square": None,
            "flags": [unavail_flag],
        })

    sw, abs_sigma = _resolve_1d_weights(n_pts, sig_sel, w_sel)
    flags: list[str] = []

    if model_kind == "power_law":
        param_names = ("A", "n")
        # Weighted log-log linear seed
        lx = np.log(t_use)
        ly = np.log(L_use)
        X_ll = np.column_stack((np.ones(n_pts, dtype=float), lx))
        w_ll = sw * L_use  # delta-method weight for log(L)
        b_ll, *_ = np.linalg.lstsq(X_ll * w_ll[:, None], ly * w_ll, rcond=None)
        A0 = max(math.exp(float(np.clip(b_ll[0], -50.0, 50.0))), 1.0e-12)
        n0 = float(np.clip(b_ll[1], -5.0, 5.0))
        p0 = np.array([A0, n0], dtype=float)
        lo = np.array([1.0e-15, -10.0], dtype=float)
        hi = np.array([np.inf, 10.0], dtype=float)

        def _eval(p: np.ndarray, tv: np.ndarray) -> np.ndarray:
            return p[0] * np.power(tv, p[1])

        fit = least_squares(lambda p: sw * (_eval(p, t_use) - L_use), p0, bounds=(lo, hi), max_nfev=4000)
    else:
        # Offset power-law: L(t) = L0 + k * t^n
        fixed_l0 = float(fix_L0) if fix_L0 is not None else None
        if fixed_l0 is not None and not math.isfinite(fixed_l0):
            raise ValueError("fix_L0 must be finite when provided")

        best_fit = None
        best_cost = float("inf")
        n_grid = (0.15, 0.25, 1.0 / 3.0, 0.5, 0.75, 1.0)

        if fixed_l0 is not None:
            param_names = ("k", "n")

            def _eval(p: np.ndarray, tv: np.ndarray) -> np.ndarray:
                return fixed_l0 + p[0] * np.power(tv, p[1])

            for n_seed in n_grid:
                tn = np.power(t_use, n_seed)
                denom = float(np.sum((sw * tn) ** 2))
                k_seed = float(np.sum((sw ** 2) * tn * (L_use - fixed_l0)) / denom) if denom > 0.0 else 1.0
                k_seed = max(k_seed, 1.0e-9)
                p0 = np.array([k_seed, n_seed], dtype=float)
                res = least_squares(
                    lambda p: sw * (_eval(p, t_use) - L_use),
                    p0,
                    bounds=([0.0, 1.0e-3], [np.inf, 5.0]),
                    max_nfev=4000,
                )
                if math.isfinite(res.cost) and res.cost < best_cost:
                    best_cost = float(res.cost)
                    best_fit = res
        else:
            param_names = ("L0", "k", "n")

            def _eval(p: np.ndarray, tv: np.ndarray) -> np.ndarray:
                return p[0] + p[1] * np.power(tv, p[2])

            for n_seed in n_grid:
                tn = np.power(t_use, n_seed)
                X_seed = np.column_stack((np.ones(n_pts, dtype=float), tn))
                beta_s, *_ = np.linalg.lstsq(X_seed * sw[:, None], L_use * sw, rcond=None)
                l0_s = max(float(beta_s[0]), 0.0)
                k_s = max(float(beta_s[1]), 1.0e-9)
                p0 = np.array([l0_s, k_s, n_seed], dtype=float)
                res = least_squares(
                    lambda p: sw * (_eval(p, t_use) - L_use),
                    p0,
                    bounds=([0.0, 0.0, 1.0e-3], [np.inf, np.inf, 5.0]),
                    max_nfev=4000,
                )
                if math.isfinite(res.cost) and res.cost < best_cost:
                    best_cost = float(res.cost)
                    best_fit = res
        if best_fit is None:
            return json_safe({
                "success": False,
                "status": "failed",
                "model": model_kind,
                "message": "Offset power-law optimizer failed to produce a finite fit on all seeds.",
                "n_points": n_pts,
                "dof": max(n_pts - n_params, 0),
                "parameters": None,
                "stderr": None,
                "ci": None,
                "r_squared": None,
                "rmse": None,
                "weighted_rmse": None,
                "reduced_chi_square": None,
                "flags": ["coarsening_optimizer_failed"],
            })
        fit = best_fit

    pred = _eval(fit.x, t_use)
    raw_resid = L_use - pred
    w_resid = sw * raw_resid
    sse_raw = float(np.sum(raw_resid ** 2))
    sse_w = float(np.sum(w_resid ** 2))
    rmse = float(math.sqrt(sse_raw / float(n_pts)))
    w_rmse = float(math.sqrt(sse_w / float(n_pts)))
    sst_raw = float(np.sum((L_use - float(np.mean(L_use))) ** 2))
    r2 = float(1.0 - sse_raw / sst_raw) if sst_raw > 1.0e-30 else (1.0 if sse_raw <= 1.0e-30 else None)

    dof = n_pts - n_params
    red_chi2 = _finite(sse_w / float(dof)) if dof > 0 else None

    params_dict: dict[str, float | None] = {name: _finite(val) for name, val in zip(param_names, fit.x)}
    if model_kind == "offset_power_law" and fix_L0 is not None:
        params_dict["L0"] = _finite(fix_L0)

    stderr_dict: dict[str, float | None] = {name: None for name in param_names}
    ci_dict: dict[str, list[float | None] | None] = {name: None for name in param_names}

    if dof > 0 and fit.jac is not None and np.all(np.isfinite(fit.jac)):
        try:
            svals = np.linalg.svd(fit.jac, compute_uv=False)
            tol = float(svals[0]) * max(fit.jac.shape) * np.finfo(float).eps if svals.size else 0.0
            if svals.size < n_params or float(svals[-1]) <= tol:
                flags.append("rank_deficient_jacobian")
            else:
                jtj = fit.jac.T @ fit.jac
                scale_var = 1.0 if (abs_sigma and (red_chi2 is None or red_chi2 <= 1.0)) else (sse_w / float(dof))
                cov = np.linalg.pinv(jtj) * scale_var
                t_crit = float(student_t.ppf(0.5 * (1.0 + float(confidence_level)), df=dof))
                for i, name in enumerate(param_names):
                    var_i = float(cov[i, i])
                    if math.isfinite(var_i) and var_i >= 0.0:
                        se_i = math.sqrt(var_i)
                        stderr_dict[name] = _finite(se_i)
                        val_i = float(fit.x[i])
                        ci_dict[name] = [_finite(val_i - t_crit * se_i), _finite(val_i + t_crit * se_i)]
        except np.linalg.LinAlgError:
            flags.append("covariance_estimation_failed")
    elif dof > 0:
        flags.append("covariance_estimation_failed")
    else:
        flags.append("zero_dof_no_uncertainty_interval")

    if not fit.success:
        flags.append("optimizer_did_not_converge")

    return json_safe({
        "success": bool(fit.success),
        "status": "estimated" if fit.success else "warning",
        "model": model_kind,
        "message": str(fit.message),
        "n_points": n_pts,
        "dof": dof,
        "confidence_level": float(confidence_level),
        "parameters": params_dict,
        "stderr": stderr_dict,
        "ci": ci_dict,
        "r_squared": _finite(r2),
        "rmse": _finite(rmse),
        "weighted_rmse": _finite(w_rmse),
        "reduced_chi_square": red_chi2,
        "t_used": _finite_list(t_use),
        "L_used": _finite_list(L_use),
        "fitted_values": _finite_list(pred),
        "residuals": _finite_list(raw_resid),
        "flags": list(dict.fromkeys(flags)),
    })


def fit_arrhenius_rate(
    temperatures: Any,
    rates: Any,
    *,
    temperature_unit: str = "K",
    sigma_rates: Any = None,
    weights: Any = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Fit Arrhenius relation ``ln(k) = ln(A) - Ea / (R * T)`` returning ``Ea`` in kJ/mol."""
    T_raw = np.asarray(temperatures, dtype=float)
    k_raw = np.asarray(rates, dtype=float)
    if T_raw.ndim != 1 or k_raw.ndim != 1 or T_raw.shape != k_raw.shape or T_raw.size == 0:
        raise ValueError("temperatures and rates must be non-empty matching 1D arrays")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must lie in (0, 1)")

    unit_norm = str(temperature_unit or "K").strip().lower().replace("°", "")
    if unit_norm in {"k", "kelvin"}:
        T_kelvin = T_raw.copy()
    elif unit_norm in {"c", "degc", "celsius"}:
        T_kelvin = T_raw + 273.15
    else:
        raise ValueError("temperature_unit must be 'K' or 'C'")

    mask = np.isfinite(T_kelvin) & (T_kelvin > 0.0) & np.isfinite(k_raw) & (k_raw > 0.0)
    idx_sel = np.flatnonzero(mask)
    if sigma_rates is not None and weights is not None:
        raise ValueError("provide either sigma_rates or weights, not both")

    T_use = T_kelvin[idx_sel]
    k_use = k_raw[idx_sel]
    n_pts = int(T_use.size)
    if n_pts < 2 or float(np.ptp(T_use)) <= 1.0e-12:
        return json_safe({
            "success": False,
            "status": "unavailable",
            "message": f"Need at least 2 distinct positive temperatures with positive rates; found {n_pts}.",
            "n_points": n_pts,
            "dof": 0,
            "Ea_kJ_mol": None,
            "Ea_stderr_kJ_mol": None,
            "Ea_ci_kJ_mol": None,
            "Ea_eV": None,
            "ln_A": None,
            "ln_A_stderr": None,
            "A_prefactor": None,
            "r_squared": None,
            "rmse_ln": None,
            "flags": ["insufficient_points_for_arrhenius_fit"],
        })

    sig_ln = None
    w_ln = None
    if sigma_rates is not None:
        s_full = np.asarray(sigma_rates, dtype=float)
        if s_full.shape != T_raw.shape:
            raise ValueError("sigma_rates must match temperatures shape")
        s_sel = s_full[idx_sel]
        if np.any(~np.isfinite(s_sel)) or np.any(s_sel <= 0.0):
            raise ValueError("sigma_rates must be finite and strictly positive")
        sig_ln = s_sel / k_use  # first-order error propagation for ln(k)
    if weights is not None:
        w_full = np.asarray(weights, dtype=float)
        if w_full.shape != T_raw.shape:
            raise ValueError("weights must match temperatures shape")
        w_ln = w_full[idx_sel]

    sw, abs_sigma = _resolve_1d_weights(n_pts, sig_ln, w_ln)

    # x = 1 / (R * T) in mol/kJ, y = ln(k) => y = ln_A - Ea * x
    x = 1.0 / (_GAS_CONSTANT_KJ_MOL_K * T_use)
    y = np.log(k_use)
    X = np.column_stack((np.ones(n_pts, dtype=float), -x))
    Xw = X * sw[:, None]
    yw = y * sw

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    ln_A = float(beta[0])
    Ea_kJ = float(beta[1])
    pred_ln = X @ beta
    resid_ln = y - pred_ln
    sse_ln = float(np.sum(resid_ln ** 2))
    sse_w = float(np.sum((sw * resid_ln) ** 2))
    sst_ln = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse_ln / sst_ln) if sst_ln > 1.0e-30 else (1.0 if sse_ln <= 1.0e-30 else None)
    rmse_ln = float(math.sqrt(sse_ln / float(n_pts)))

    dof = n_pts - 2
    flags: list[str] = []
    Ea_se: float | None = None
    ln_A_se: float | None = None
    Ea_ci: list[float | None] | None = None
    ln_A_ci: list[float | None] | None = None

    if dof > 0:
        red_chi2 = sse_w / float(dof)
        scale_var = 1.0 if (abs_sigma and red_chi2 <= 1.0) else red_chi2
        cov = np.linalg.pinv(Xw.T @ Xw) * scale_var
        t_crit = float(student_t.ppf(0.5 * (1.0 + float(confidence_level)), df=dof))
        if cov[0, 0] >= 0.0:
            ln_A_se = _finite(math.sqrt(float(cov[0, 0])))
            if ln_A_se is not None:
                ln_A_ci = [_finite(ln_A - t_crit * ln_A_se), _finite(ln_A + t_crit * ln_A_se)]
        if cov[1, 1] >= 0.0:
            Ea_se = _finite(math.sqrt(float(cov[1, 1])))
            if Ea_se is not None:
                Ea_ci = [_finite(Ea_kJ - t_crit * Ea_se), _finite(Ea_kJ + t_crit * Ea_se)]
    else:
        flags.append("zero_dof_exact_two_point_fit")

    if Ea_kJ < 0.0:
        flags.append("negative_apparent_activation_energy")

    A_prefactor = _finite(math.exp(ln_A)) if abs(ln_A) < 700.0 else None

    return json_safe({
        "success": True,
        "status": "estimated",
        "n_points": n_pts,
        "dof": dof,
        "confidence_level": float(confidence_level),
        "Ea_kJ_mol": _finite(Ea_kJ),
        "Ea_stderr_kJ_mol": Ea_se,
        "Ea_ci_kJ_mol": Ea_ci,
        "Ea_eV": _finite(Ea_kJ / _KJ_MOL_PER_EV),
        "ln_A": _finite(ln_A),
        "ln_A_stderr": ln_A_se,
        "ln_A_ci": ln_A_ci,
        "A_prefactor": A_prefactor,
        "r_squared": _finite(r2),
        "rmse_ln": _finite(rmse_ln),
        "temperatures_K": _finite_list(T_use),
        "inv_T_K_inv": _finite_list(1.0 / T_use),
        "inv_RT_mol_per_kJ": _finite_list(x),
        "ln_rates": _finite_list(y),
        "fitted_ln_rates": _finite_list(pred_ln),
        "residuals_ln": _finite_list(resid_ln),
        "flags": flags,
        "caveat": (
            "Apparent activation energy Ea from Arrhenius regression of ln(rate) vs 1/(R*T) "
            "represents an effective kinetic barrier over the sampled temperature window."
        ),
    })
