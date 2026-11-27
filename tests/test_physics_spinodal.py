from __future__ import annotations

import json
import math

import numpy as np
import pytest

from butterfly_saxs.saxs_physics import (
    analyze_cahn_hilliard_cook_early,
    analyze_joint_spinodal_exponents,
    analyze_khachaturyan_complex_scaling,
    compute_dynamical_scaling_collapse,
)


def _assert_strict_json(payload: dict) -> None:
    dumped = json.dumps(payload, allow_nan=False, ensure_ascii=True)
    assert isinstance(dumped, str) and len(dumped) > 2


def test_dynamical_scaling_collapse_and_furukawa_self_similar_lsw() -> None:
    # Exact self-similar Furukawa d=3 (gamma=4) series: a = 1/3, b = 1.0, c = 0.0
    times = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
    q_grid = np.linspace(0.005, 5.0, 900)
    a_true = 1.0 / 3.0
    b_true = 1.0
    q_m = 0.45 * (times / 10.0) ** (-a_true)
    I_m = 5.0 * (times / 10.0) ** b_true
    gamma = 4.0

    I_series = []
    q_peaks = []
    I_peaks = []
    Q_invs = []
    for qm_t, Im_t in zip(q_m, I_m):
        x = q_grid / qm_t
        F = (1.0 + 0.5 * gamma) * (x ** 2) / (0.5 * gamma + (x ** (2.0 + gamma)))
        I_t = Im_t * F
        I_series.append(I_t)
        q_peaks.append(qm_t)
        I_peaks.append(Im_t)
        Q_invs.append(float(np.trapezoid(I_t * (q_grid ** 2), q_grid)))

    res = compute_dynamical_scaling_collapse(
        q_grid,
        I_series,
        times=times,
        q_mode="q_peak",
        q_window=(0.01, 4.8),
        furukawa_d=3,
    )
    _assert_strict_json(res)
    assert res["status"] == "estimated"
    assert res["n_frames_valid"] == 6
    assert res["collapse_cv_all"] < 0.04
    assert res["collapse_cv_late"] < 0.02
    assert res["collapse_cv_peak_all"] < 0.01
    assert res["furukawa_peak_rms"] < 0.02

    joint = analyze_joint_spinodal_exponents(times, q_peaks, I_peaks, Q_invs)
    _assert_strict_json(joint)
    assert joint["success"] is True
    assert joint["a"] == pytest.approx(a_true, abs=1e-4)
    assert joint["b"] == pytest.approx(b_true, abs=1e-4)
    assert abs(joint["c"]) < 0.03
    assert joint["ba_ratio"] == pytest.approx(3.0, abs=1e-2)
    assert joint["regime"] == "self_similar_coarsening_lsw"


def test_joint_exponents_distinguish_active_amplification_vs_arrest() -> None:
    times = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    # Active amplification inside spinodal: slow coarsening a=0.10, strong b=0.80 => c=0.50, b/a=8.0
    q_act = 0.25 * (times ** -0.10)
    I_act = 10.0 * (times ** 0.80)
    Q_act = 2.0 * (times ** 0.50)
    act = analyze_joint_spinodal_exponents(times, q_act, I_act, Q_act)
    _assert_strict_json(act)
    assert act["ba_ratio"] == pytest.approx(8.0, rel=1e-5)
    assert act["scaling_closure_residual"] == pytest.approx(0.0, abs=1e-9)
    assert act["regime"] == "active_spinodal_amplification_and_coarsening"
    assert "sub_lsw_elastic_retarded_coarsening" in act["flags"]

    # Elastic arrest: constant q_char (a ~ 0), static or mild growth
    q_arr = np.full_like(times, 0.20)
    I_arr = 8.0 * (times ** 0.15)
    Q_arr = 3.0 * (times ** 0.15)
    arr = analyze_joint_spinodal_exponents(times, q_arr, I_arr, Q_arr)
    _assert_strict_json(arr)
    assert arr["ba_ratio"] is None
    assert arr["regime"] == "fixed_wavelength_amplification_or_arrest"


def test_cahn_hilliard_cook_early_exact_recovery_and_nonlinear_flag() -> None:
    q = np.linspace(0.08, 0.70, 50)
    times_s = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
    D_eff_true = -0.04  # negative chemical diffusivity => uphill diffusion D_uphill = +0.04
    two_M_kappa_true = 0.16  # => q_c = sqrt(0.04 / 0.16) = 0.50, q_m = 0.50 / sqrt(2)
    R_true = (q ** 2) * (-D_eff_true - two_M_kappa_true * (q ** 2))
    I0 = np.full_like(q, 1.5)
    I_mat = np.vstack([I0 * np.exp(2.0 * R_true * t) for t in times_s])

    chc = analyze_cahn_hilliard_cook_early(q, I_mat, times_s, fit_q_range=(0.10, 0.56))
    _assert_strict_json(chc)
    assert chc["success"] is True
    assert chc["uphill_diffusion_sign"] is True
    assert chc["linear_chc_valid"] is True
    assert chc["D_eff"] == pytest.approx(D_eff_true, rel=1e-4)
    assert chc["two_M_kappa"] == pytest.approx(two_M_kappa_true, rel=1e-4)
    assert chc["q_c"] == pytest.approx(0.50, rel=1e-4)
    assert chc["q_m_chc"] == pytest.approx(0.50 / math.sqrt(2.0), rel=1e-4)
    assert chc["q_cross_median"] == pytest.approx(0.50, abs=0.015)
    assert chc["r_squared"] > 0.9999

    # Nonlinear coarsening test: moving peak q_m(t) shifts q_cross and breaks CHC linearity
    q_nl = np.linspace(0.10, 0.90, 80)
    times_nl = np.array([10.0, 30.0, 80.0, 200.0, 500.0])
    I_nl = np.vstack([
        (1.0 + 0.02 * t) * np.exp(-0.5 * ((q_nl - 0.50 * (t / 10.0) ** -0.25) / 0.08) ** 2) + 0.05
        for t in times_nl
    ])
    chc_nl = analyze_cahn_hilliard_cook_early(q_nl, I_nl, times_nl)
    _assert_strict_json(chc_nl)
    assert chc_nl["linear_chc_valid"] is False
    assert len(chc_nl["flags"]) >= 1


def test_khachaturyan_complex_scaling_laws_and_edge_cases() -> None:
    r1_true = 5.0
    W = np.array([40.0, 60.0, 90.0, 140.0, 220.0, 320.0])
    L_coherent = np.sqrt(r1_true * W)
    times = np.array([10.0, 25.0, 60.0, 150.0, 350.0, 800.0])

    coh = analyze_khachaturyan_complex_scaling(L_coherent, W, times=times)
    _assert_strict_json(coh)
    assert coh["success"] is True
    assert coh["r1_median_nm"] == pytest.approx(r1_true, rel=1e-6)
    assert coh["dlnL_dlnW"] == pytest.approx(0.5, abs=1e-6)
    assert coh["dlnL_dlnW_r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert coh["regime"] == "coherent_complex_scaling"
    assert coh["time_exponent_W"] > 0.4

    # Elastic arrest: constant lambda_0* while W expands
    L_arrest = np.full_like(W, 20.0)
    arr = analyze_khachaturyan_complex_scaling(L_arrest, W)
    _assert_strict_json(arr)
    assert arr["dlnL_dlnW"] == pytest.approx(0.0, abs=1e-6)
    assert arr["regime"] == "elastic_wavelength_arrest"

    # NaN preservation and empty pairs
    nan_res = analyze_khachaturyan_complex_scaling([np.nan, 20.0, np.nan], [100.0, np.nan, 120.0])
    _assert_strict_json(nan_res)
    assert nan_res["success"] is False
    assert nan_res["r1_series_nm"] == [None, None, None]
