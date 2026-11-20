from __future__ import annotations

import json
import math

import numpy as np
import pytest

from butterfly_saxs.report_measurements import measure_frame
from butterfly_saxs.saxs_physics import (
    compute_azimuthal_harmonics_and_order,
    evaluate_c2_bridge_four_lobe,
    fit_arrhenius_rate,
    fit_c2_bridge_four_lobe,
    fit_power_law_coarsening,
    measure_perpendicular_streak_width,
    measure_radial_coarsening_observables,
    unwrap_periodic_angles_deg,
)

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _assert_strict_json(payload: dict) -> None:
    dumped = json.dumps(payload, allow_nan=False, ensure_ascii=True)
    assert isinstance(dumped, str) and len(dumped) > 2


def _make_butterfly_2d(
    *,
    n_grid: int = 241,
    q_extent: float = 1.0,
    q_star: float = 0.38,
    sigma_q: float = 0.06,
    chi_sym_deg: float = 90.0,
    alpha_deg: float = 36.0,
    c0: float = 5.0,
    c2: float = -3.0,
    lobe_amplitudes: tuple[float, float, float, float] = (30.0, 30.0, 30.0, 30.0),
    lobe_fwhm_deg: float = 22.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-q_extent, q_extent, n_grid)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    chi = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    radial_env = np.exp(-0.5 * ((q - q_star) / sigma_q) ** 2)
    sigma_chi = lobe_fwhm_deg / _FWHM_PER_SIGMA
    angular_profile = evaluate_c2_bridge_four_lobe(
        chi,
        c0=c0,
        c2=c2,
        chi_sym_deg=chi_sym_deg,
        alpha_deg=alpha_deg,
        lobe_amplitudes=lobe_amplitudes,
        lobe_sigmas_deg=sigma_chi,
    )
    image = 0.5 + radial_env * angular_profile
    return image, qx, qy


def _make_streak_butterfly_2d(
    *,
    n_grid: int = 251,
    q_extent: float = 1.0,
    chi_sym_deg: float = 90.0,
    alpha_deg: float = 37.0,
    dq_perp_0: float = 0.045,
    mosaic_deg: float = 9.0,
    mode: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    axis = np.linspace(-q_extent, q_extent, n_grid)
    qx, qy = np.meshgrid(axis, axis)
    wing_angles = [
        (chi_sym_deg - alpha_deg) % 360.0,
        (chi_sym_deg + alpha_deg) % 360.0,
        (chi_sym_deg + 180.0 - alpha_deg) % 360.0,
        (chi_sym_deg + 180.0 + alpha_deg) % 360.0,
    ]
    mosaic_rad = math.radians(mosaic_deg)
    image = np.full(qx.shape, 2.0, dtype=float)
    for theta in wing_angles:
        tr = math.radians(theta)
        q_par = qx * math.cos(tr) + qy * math.sin(tr)
        q_perp = -qx * math.sin(tr) + qy * math.cos(tr)
        pos = q_par > 0.05
        if mode == "linear":
            fwhm = dq_perp_0 + mosaic_rad * np.maximum(q_par, 0.0)
        else:
            fwhm = np.sqrt(dq_perp_0 ** 2 + (mosaic_rad * np.maximum(q_par, 0.0)) ** 2)
        sigma = fwhm / _FWHM_PER_SIGMA
        radial_amp = 40.0 * np.exp(-0.5 * ((q_par - 0.45) / 0.35) ** 2)
        wing_contrib = np.where(pos, radial_amp * np.exp(-0.5 * (q_perp / sigma) ** 2), 0.0)
        image += wing_contrib
    return image, qx, qy, wing_angles


# ---------------------------------------------------------------------------
# 1. Azimuthal Harmonics & Order Parameters
# ---------------------------------------------------------------------------


def test_harmonics_exact_fourier_recovery_with_and_without_detector_mask_gaps():
    angles = np.linspace(0.5, 359.5, 360)
    # Known exact even harmonics
    profile = (
        12.0
        + 3.6 * np.cos(2.0 * np.deg2rad(angles - 25.0))
        + 4.8 * np.cos(4.0 * np.deg2rad(angles - 25.0))
        + 1.2 * np.cos(6.0 * np.deg2rad(angles - 25.0))
    )
    full = compute_azimuthal_harmonics_and_order(angle_deg=angles, intensity=profile, q_unit="nm^-1")
    _assert_strict_json(full)
    assert full["A0"] == pytest.approx(12.0, abs=1e-6)
    assert full["A2"] == pytest.approx(3.6, abs=1e-6)
    assert full["A4"] == pytest.approx(4.8, abs=1e-6)
    assert full["A6"] == pytest.approx(1.2, abs=1e-6)
    assert full["a2"] == pytest.approx(0.3, abs=1e-6)
    assert full["a4"] == pytest.approx(0.4, abs=1e-6)
    assert full["a4_over_a2"] == pytest.approx(4.8 / 3.6, abs=1e-6)
    assert full["phi2_deg"] == pytest.approx(25.0, abs=1e-4)
    assert full["phi4_deg"] == pytest.approx(25.0, abs=1e-4)
    assert full["q1_over_q3"] == pytest.approx(1.0, abs=1e-6)
    assert full["q2_over_q4"] == pytest.approx(1.0, abs=1e-6)

    # Mask out two asymmetric detector module gaps (15 deg in Q1, 18 deg in Q2)
    valid = ~(((angles >= 30.0) & (angles <= 45.0)) | ((angles >= 120.0) & (angles <= 138.0)))
    masked = compute_azimuthal_harmonics_and_order(
        angle_deg=angles, intensity=profile, valid_mask=valid, q_unit="nm^-1"
    )
    _assert_strict_json(masked)
    assert "incomplete_azimuthal_coverage" in masked["flags"]
    assert masked["A0"] == pytest.approx(12.0, abs=1e-5)
    assert masked["A2"] == pytest.approx(3.6, abs=1e-5)
    assert masked["A4"] == pytest.approx(4.8, abs=1e-5)
    assert masked["A6"] == pytest.approx(1.2, abs=1e-5)
    assert masked["q1_over_q3"] == pytest.approx(1.0, abs=1e-5)
    assert masked["q2_over_q4"] == pytest.approx(1.0, abs=1e-5)


def test_harmonics_90deg_rotation_invariance_and_friedel_asymmetry():
    img_v, qx, qy = _make_butterfly_2d(chi_sym_deg=90.0, alpha_deg=35.0, c0=8.0, c2=-4.0)
    img_h, _, _ = _make_butterfly_2d(chi_sym_deg=0.0, alpha_deg=35.0, c0=8.0, c2=-4.0)

    res_v = compute_azimuthal_harmonics_and_order(
        img_v, qx, qy, q_window=(0.28, 0.48), q_unit="nm^-1", nominal_draw_axis_deg=90.0
    )
    res_h = compute_azimuthal_harmonics_and_order(
        img_h, qx, qy, q_window=(0.28, 0.48), q_unit="nm^-1", nominal_draw_axis_deg=90.0
    )
    _assert_strict_json(res_v)
    _assert_strict_json(res_h)

    # Vertical butterfly aligned with nominal draw axis 90 deg
    assert res_v["friedel_bisector_deg"] == pytest.approx(90.0, abs=1.0)
    assert abs(res_v["mounting_tilt_deg"]) < 1.0
    assert res_v["orthogonal_rotation_detected"] is False
    assert res_v["f2"] == pytest.approx(res_v["f2_nominal"], abs=0.01)

    # 90-deg rotated butterfly (like SPring-8 run22!)
    dist_to_zero = abs(((res_h["friedel_bisector_deg"] + 90.0) % 180.0) - 90.0)
    assert dist_to_zero < 1.0
    assert res_h["orthogonal_rotation_detected"] is True
    assert abs(abs(res_h["mounting_tilt_deg"]) - 90.0) < 1.0
    # Intrinsic symmetry-frame f2 and f4 remain invariant under 90-deg sample rotation
    assert res_h["f2"] == pytest.approx(res_v["f2"], abs=0.02)
    assert res_h["f4"] == pytest.approx(res_v["f4"], abs=0.02)
    # Nominal-frame f2 shifts relative to fixed 90-deg lab axis (partially offset by C2 bridge rotation)
    assert abs(res_h["f2_nominal"] - res_v["f2_nominal"]) > 0.05

    # Explicit Friedel asymmetry in Q1 only
    asym = img_v.copy()
    asym[(qx >= 0.0) & (qy >= 0.0)] *= 1.5
    res_asym = compute_azimuthal_harmonics_and_order(asym, qx, qy, q_window=(0.28, 0.48), q_unit="nm^-1")
    assert res_asym["q1_over_q3"] == pytest.approx(1.5, rel=0.03)
    assert res_asym["q2_over_q4"] == pytest.approx(1.0, rel=0.03)


def test_harmonics_isotropic_ring_and_angle_unwrapping():
    angles = np.linspace(0.5, 359.5, 360)
    flat = np.full_like(angles, 7.5)
    iso = compute_azimuthal_harmonics_and_order(angle_deg=angles, intensity=flat)
    _assert_strict_json(iso)
    assert iso["a2"] == pytest.approx(0.0, abs=1e-9)
    assert iso["a4"] == pytest.approx(0.0, abs=1e-9)
    assert iso["friedel_bisector_deg"] is None
    assert "isotropic_symmetry_axis_undefined" in iso["flags"]
    assert iso["f2"] == pytest.approx(0.0, abs=1e-3)
    assert iso["f4"] == pytest.approx(0.0, abs=1e-3)

    unwrapped = unwrap_periodic_angles_deg([178.0, 179.5, None, 1.0, 3.0, 179.0], period_deg=180.0)
    assert unwrapped[2] is None
    assert unwrapped[3] == pytest.approx(181.0)
    assert unwrapped[4] == pytest.approx(183.0)
    assert unwrapped[5] == pytest.approx(179.0)


# ---------------------------------------------------------------------------
# 2. C2-Bridge + 4-Lobe Azimuthal Model
# ---------------------------------------------------------------------------


def test_c2_bridge_four_lobe_recovers_bridge_and_outperforms_flat_baseline():
    angles = np.linspace(0.5, 359.5, 360)
    true_c0 = 15.0
    true_c2 = -6.0  # strong equatorial bridge at 0/180 deg
    true_chi_sym = 93.2
    true_alpha = 37.8
    true_fwhm = 24.0
    true_amps = (28.0, 26.5, 27.5, 26.0)

    y = evaluate_c2_bridge_four_lobe(
        angles,
        c0=true_c0,
        c2=true_c2,
        chi_sym_deg=true_chi_sym,
        alpha_deg=true_alpha,
        lobe_amplitudes=true_amps,
        lobe_sigmas_deg=true_fwhm / _FWHM_PER_SIGMA,
    )
    fit = fit_c2_bridge_four_lobe(angles, y)
    _assert_strict_json(fit)

    assert fit["success"] is True
    assert fit["c0"] == pytest.approx(true_c0, abs=0.05)
    assert fit["c2"] == pytest.approx(true_c2, abs=0.05)
    assert fit["chi_sym_deg"] == pytest.approx(true_chi_sym, abs=0.2)
    assert fit["alpha_deg"] == pytest.approx(true_alpha, abs=0.2)
    assert fit["lobe_fwhm_deg"] == pytest.approx(true_fwhm, abs=0.2)
    assert fit["rmse"] < 1e-4
    assert fit["r_squared"] > 0.99999

    comp = fit["flat_baseline_comparison"]
    assert comp is not None
    assert comp["bridge_improved"] is True
    assert comp["rmse"] > 1.0  # flat baseline leaves large systematic C2 bridge residual
    assert comp["delta_r_squared"] > 0.01
    assert comp["delta_bic"] > 100.0


def test_c2_bridge_four_lobe_handles_90deg_rotation_and_masked_detector_gaps():
    angles = np.linspace(0.5, 359.5, 360)
    # Rotated near 0/180 wraparound (chi_sym = 2.5 deg) with masked gaps
    true_chi_sym = 2.5
    true_alpha = 34.0
    true_c2 = -4.5
    true_fwhm = 20.0
    y = evaluate_c2_bridge_four_lobe(
        angles,
        c0=10.0,
        c2=true_c2,
        chi_sym_deg=true_chi_sym,
        alpha_deg=true_alpha,
        lobe_amplitudes=(22.0, 21.0, 22.0, 21.0),
        lobe_sigmas_deg=true_fwhm / _FWHM_PER_SIGMA,
    )
    # Mask a 16-deg gap near one lobe wing and an 18-deg gap on the bridge
    valid = ~(((angles >= 82.0) & (angles <= 100.0)) | ((angles >= 195.0) & (angles <= 211.0)))
    y_masked = y.copy()
    y_masked[~valid] = np.nan

    fit = fit_c2_bridge_four_lobe(angles, y_masked, valid_mask=valid)
    _assert_strict_json(fit)

    assert fit["success"] is True
    assert "partial_angular_masking" in fit["flags"]
    assert fit["chi_sym_deg"] == pytest.approx(true_chi_sym, abs=0.3)
    assert fit["alpha_deg"] == pytest.approx(true_alpha, abs=0.3)
    assert fit["c2"] == pytest.approx(true_c2, abs=0.1)
    assert fit["lobe_fwhm_deg"] == pytest.approx(true_fwhm, abs=0.3)
    assert fit["rmse"] < 1e-3
    # Unobserved bins MUST remain None in residual (never filled or fabricated)
    for idx in np.flatnonzero(~valid):
        assert fit["residual"][idx] is None
    for idx in np.flatnonzero(valid):
        assert fit["residual"][idx] is not None


def test_c2_bridge_four_lobe_zero_bridge_does_not_claim_improvement():
    angles = np.linspace(0.5, 359.5, 360)
    y = evaluate_c2_bridge_four_lobe(
        angles,
        c0=8.0,
        c2=0.0,
        chi_sym_deg=90.0,
        alpha_deg=35.0,
        lobe_amplitudes=(20.0, 20.0, 20.0, 20.0),
        lobe_sigmas_deg=22.0 / _FWHM_PER_SIGMA,
    )
    fit = fit_c2_bridge_four_lobe(angles, y)
    assert fit["success"] is True
    assert abs(fit["c2"]) < 1e-3
    assert fit["flat_baseline_comparison"]["bridge_improved"] is False


# ---------------------------------------------------------------------------
# 3. Dual-Channel Radial Coarsening & Invariant Observables
# ---------------------------------------------------------------------------


def test_radial_coarsening_recovers_direct_and_kratky_peaks_qwin_and_correlation():
    q_star_true = 0.35
    sigma_q = 0.055
    img, qx, qy = _make_butterfly_2d(
        n_grid=261, q_extent=1.0, q_star=q_star_true, sigma_q=sigma_q, chi_sym_deg=90.0, alpha_deg=36.0
    )
    # Apply a realistic central beamstop mask r < 0.09 nm^-1
    valid = np.hypot(qx, qy) >= 0.09

    res_nm = measure_radial_coarsening_observables(
        img,
        qx,
        qy,
        valid_mask=valid,
        q_window=(0.11, 0.90),
        q_unit="nm^-1",
        n_radial_bins=120,
        compute_correlation=True,
    )
    _assert_strict_json(res_nm)

    assert res_nm["status"] == "estimated"
    assert res_nm["peak_channel_source"] == "lobe_sectored"
    assert res_nm["q_star_direct"] == pytest.approx(q_star_true, abs=0.01)
    assert res_nm["L_star_direct_nm"] == pytest.approx(2.0 * math.pi / q_star_true, rel=0.03)
    assert res_nm["edge_hit_direct"] is False
    assert res_nm["fwhm_direct_q"] == pytest.approx(sigma_q * _FWHM_PER_SIGMA, rel=0.15)
    assert res_nm["prominence_direct"] > 0.0

    # Kratky q^2 I(q) peak is physically shifted to slightly higher q than direct I(q)
    assert res_nm["q_star_kratky"] is not None
    assert res_nm["q_star_kratky"] > res_nm["q_star_direct"]
    assert res_nm["L_star_kratky_nm"] == pytest.approx(2.0 * math.pi / res_nm["q_star_kratky"], rel=1e-6)
    assert res_nm["q_win_invariant"] > 0.0

    # 1D cosine correlation function K(r) in nm
    corr = res_nm["correlation_function"]
    assert corr is not None
    assert corr["status"] == "estimated"
    assert corr["K_r"][0] == pytest.approx(1.0, abs=1e-6)
    assert corr["first_maximum_r_nm"] is not None
    assert corr["first_maximum_r_nm"] == pytest.approx(2.0 * math.pi / q_star_true, rel=0.2)
    assert len(corr["truncation_warnings"]) >= 1

    # Transparent A^-1 conversion must yield identical physical L* in nm
    res_ang = measure_radial_coarsening_observables(
        img,
        qx / 10.0,
        qy / 10.0,
        valid_mask=valid,
        q_window=(0.011, 0.090),
        q_unit="A^-1",
        n_radial_bins=120,
    )
    assert res_ang["L_star_direct_nm"] == pytest.approx(res_nm["L_star_direct_nm"], rel=1e-4)
    assert res_ang["L_star_kratky_nm"] == pytest.approx(res_nm["L_star_kratky_nm"], rel=1e-4)

    # pixel-q MUST gate nm lengths and K(r) to None/unavailable with explicit caveat
    res_px = measure_radial_coarsening_observables(
        img,
        qx,
        qy,
        valid_mask=valid,
        q_window=(0.11, 0.90),
        q_unit="pixel-q",
        n_radial_bins=120,
        compute_correlation=True,
    )
    _assert_strict_json(res_px)
    assert res_px["q_star_direct"] is not None
    assert res_px["L_star_direct_nm"] is None
    assert res_px["L_star_kratky_nm"] is None
    assert "uncalibrated_pixel_q" in res_px["flags"]
    assert res_px["correlation_function"]["status"] == "unavailable"
    assert res_px["correlation_function"]["r_nm"] is None
    assert any("pixel-q" in msg for msg in res_px["caveats"])


def test_radial_coarsening_monotonic_decay_flags_direct_edge_hit_while_kratky_recovers_peak():
    axis = np.linspace(-1.0, 1.0, 221)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    qc = 0.28  # Debye-Bueche I(q) = 1 / (1 + (q/qc)^2)^2 -> q^2 I(q) peaks strictly at q = qc
    img = 100.0 / (1.0 + (q / qc) ** 2) ** 2
    valid = q >= 0.08

    res = measure_radial_coarsening_observables(
        img, qx, qy, valid_mask=valid, q_window=(0.10, 0.90), q_unit="nm^-1", n_radial_bins=100
    )
    _assert_strict_json(res)
    assert res["q_star_direct"] is None
    assert res["L_star_direct_nm"] is None
    assert res["edge_hit_direct"] is True
    assert res["edge_hit_direct_side"] == "low_q"

    assert res["q_star_kratky"] == pytest.approx(qc, abs=0.015)
    assert res["L_star_kratky_nm"] == pytest.approx(2.0 * math.pi / qc, rel=0.06)
    assert res["edge_hit_kratky"] is False


# ---------------------------------------------------------------------------
# 4. Perpendicular Butterfly Streak Width / Grubb-Williamson-Hall Analysis
# ---------------------------------------------------------------------------


def test_perpendicular_streak_width_linear_and_quadrature_and_masking_and_pixel_q_gate():
    dq0_true = 0.045
    mosaic_true = 9.0
    w_true_nm = 2.0 * math.pi / dq0_true

    img_lin, qx, qy, rays = _make_streak_butterfly_2d(
        dq_perp_0=dq0_true, mosaic_deg=mosaic_true, mode="linear"
    )
    res_lin = measure_perpendicular_streak_width(
        img_lin,
        qx,
        qy,
        q_window=(0.20, 0.80),
        q_unit="nm^-1",
        wing_angles_deg=rays,
        n_q_slices=10,
    )
    _assert_strict_json(res_lin)
    assert res_lin["status"] == "estimated"
    assert res_lin["summary"]["n_valid_wings_linear"] == 4
    assert res_lin["summary"]["linear"]["dq_perp_0"] == pytest.approx(dq0_true, rel=0.08)
    assert res_lin["summary"]["linear"]["mosaic_spread_deg"] == pytest.approx(mosaic_true, rel=0.08)
    assert res_lin["summary"]["linear"]["W_nm"] == pytest.approx(w_true_nm, rel=0.08)
    assert res_lin["summary"]["linear"]["median_r_squared"] > 0.98

    # Quadrature convolution synthetic streak
    img_quad, qx, qy, rays = _make_streak_butterfly_2d(
        dq_perp_0=dq0_true, mosaic_deg=mosaic_true, mode="quadrature"
    )
    res_quad = measure_perpendicular_streak_width(
        img_quad,
        qx,
        qy,
        q_window=(0.20, 0.80),
        q_unit="nm^-1",
        wing_angles_deg=rays,
        n_q_slices=10,
    )
    _assert_strict_json(res_quad)
    assert res_quad["summary"]["n_valid_wings_quadrature"] == 4
    assert res_quad["summary"]["quadrature"]["dq_perp_0"] == pytest.approx(dq0_true, rel=0.08)
    assert res_quad["summary"]["quadrature"]["mosaic_spread_deg"] == pytest.approx(mosaic_true, rel=0.08)
    assert res_quad["summary"]["quadrature"]["W_nm"] == pytest.approx(w_true_nm, rel=0.08)
    assert res_quad["summary"]["quadrature"]["median_r_squared"] > 0.98

    # Detector gap masking a horizontal strip across the wings should not crash or fabricate
    valid = np.abs(qy - 0.35) > 0.03
    res_gap = measure_perpendicular_streak_width(
        img_lin,
        qx,
        qy,
        valid_mask=valid,
        q_window=(0.20, 0.80),
        q_unit="nm^-1",
        wing_angles_deg=rays,
        n_q_slices=10,
    )
    _assert_strict_json(res_gap)
    assert res_gap["summary"]["n_valid_wings_linear"] == 4
    assert res_gap["summary"]["linear"]["mosaic_spread_deg"] == pytest.approx(mosaic_true, rel=0.12)

    # pixel-q MUST return None for W_nm while keeping dq_perp_0 and mosaic_spread_deg
    res_px = measure_perpendicular_streak_width(
        img_lin,
        qx,
        qy,
        q_window=(0.20, 0.80),
        q_unit="pixel-q",
        wing_angles_deg=rays,
        n_q_slices=10,
    )
    _assert_strict_json(res_px)
    assert res_px["summary"]["linear"]["dq_perp_0"] is not None
    assert res_px["summary"]["linear"]["mosaic_spread_deg"] is not None
    assert res_px["summary"]["linear"]["W_nm"] is None
    assert res_px["summary"]["quadrature"]["W_nm"] is None
    assert "uncalibrated_pixel_q" in res_px["flags"]

    # Pure angular-spread streak (dq_perp_0 = 0): never fabricates negative W_nm
    img_mosaic, qx, qy, rays = _make_streak_butterfly_2d(
        dq_perp_0=1e-6, mosaic_deg=11.0, mode="linear"
    )
    res_zero = measure_perpendicular_streak_width(
        img_mosaic,
        qx,
        qy,
        q_window=(0.25, 0.80),
        q_unit="nm^-1",
        wing_angles_deg=rays,
        n_q_slices=9,
    )
    _assert_strict_json(res_zero)
    for wing in res_zero["wings"]:
        if wing["linear_fit"]["dq_perp_0"] is not None and wing["linear_fit"]["dq_perp_0"] <= 0.0:
            assert wing["linear_fit"]["W_nm"] is None
            assert "nonpositive_intercept_W_unbounded" in wing["linear_fit"]["flags"]


# ---------------------------------------------------------------------------
# 5. Kinetics & Thermal Scaling Helpers
# ---------------------------------------------------------------------------


def test_power_law_coarsening_pure_and_offset_with_weights_and_gaps():
    t = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    rng = np.random.default_rng(42)

    # Pure power-law L(t) = A * t^n with sigma weights
    L_pure = 5.5 * (t ** 0.34) + rng.normal(0.0, 0.05, size=t.size)
    sigma = np.full_like(t, 0.05)
    fit_pure = fit_power_law_coarsening(t, L_pure, sigma_L=sigma, model="power_law")
    _assert_strict_json(fit_pure)
    assert fit_pure["success"] is True
    assert fit_pure["parameters"]["A"] == pytest.approx(5.5, rel=0.03)
    assert fit_pure["parameters"]["n"] == pytest.approx(0.34, rel=0.04)
    ci_n = fit_pure["ci"]["n"]
    assert ci_n is not None and ci_n[0] <= 0.34 <= ci_n[1]

    # Offset LSW power-law L(t) = L0 + k * t^(1/3) with a NaN gap omitted cleanly
    t_off = np.array([0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 180.0])
    L_off = 12.0 + 1.8 * (t_off ** (1.0 / 3.0))
    L_off[3] = np.nan
    fit_off = fit_power_law_coarsening(t_off, L_off, model="offset_power_law")
    _assert_strict_json(fit_off)
    assert fit_off["success"] is True
    assert fit_off["n_points"] == 8
    assert fit_off["parameters"]["L0"] == pytest.approx(12.0, abs=0.05)
    assert fit_off["parameters"]["k"] == pytest.approx(1.8, abs=0.05)
    assert fit_off["parameters"]["n"] == pytest.approx(1.0 / 3.0, abs=0.01)

    with pytest.raises(ValueError, match="not both"):
        fit_power_law_coarsening(t, L_pure, sigma_L=sigma, weights=np.ones_like(t))


def test_arrhenius_rate_celsius_and_kelvin_and_weighted():
    temps_c = np.array([350.0, 375.0, 400.0, 425.0, 450.0, 475.0])
    temps_k = temps_c + 273.15
    true_Ea = 92.5  # kJ/mol
    true_A = 1.0e6
    R_kJ = 8.31446261815324e-3
    rates = true_A * np.exp(-true_Ea / (R_kJ * temps_k))

    res_c = fit_arrhenius_rate(temps_c, rates, temperature_unit="C")
    res_k = fit_arrhenius_rate(temps_k, rates, temperature_unit="K", sigma_rates=0.05 * rates)
    _assert_strict_json(res_c)
    _assert_strict_json(res_k)

    assert res_c["success"] is True
    assert res_c["Ea_kJ_mol"] == pytest.approx(true_Ea, abs=1e-6)
    assert res_c["ln_A"] == pytest.approx(math.log(true_A), abs=1e-6)
    assert res_c["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert res_k["Ea_kJ_mol"] == pytest.approx(true_Ea, abs=1e-6)
    assert res_k["Ea_stderr_kJ_mol"] is not None and res_k["Ea_stderr_kJ_mol"] > 0.0
    assert res_k["Ea_ci_kJ_mol"][0] < true_Ea < res_k["Ea_ci_kJ_mol"][1]

    with pytest.raises(ValueError, match="temperature_unit"):
        fit_arrhenius_rate(temps_c, rates, temperature_unit="F")


# ---------------------------------------------------------------------------
# 6. Additive wiring in report_measurements.measure_frame
# ---------------------------------------------------------------------------


def test_measure_frame_wires_all_saxs_physics_metrics_and_gates_pixel_q():
    img, qx, qy = _make_butterfly_2d(
        n_grid=181, q_extent=1.0, q_star=0.40, sigma_q=0.06, chi_sym_deg=90.0, alpha_deg=35.0, c2=-3.5
    )
    valid = np.hypot(qx, qy) >= 0.08
    required_keys = {
        "a2",
        "a4",
        "a4_over_a2",
        "f2_hermans",
        "f4_order",
        "friedel_bisector_deg",
        "wing_half_opening_deg",
        "q_win_invariant",
        "q_star_direct",
        "L_star_direct_nm",
        "q_star_kratky",
        "L_star_kratky_nm",
        "friedel_q1_q3_ratio",
        "friedel_q2_q4_ratio",
    }

    phys = measure_frame(
        img, qx, qy, valid_mask=valid, q_unit="nm^-1", q_edges=np.linspace(0.10, 0.90, 81), angular_bins=72
    )
    summary = phys["summary"]
    _assert_strict_json(summary)
    assert required_keys <= set(summary.keys())
    assert required_keys <= set(summary["saxs_physics"].keys())

    assert summary["a4"] is not None and summary["a4"] > 0.1
    assert summary["friedel_bisector_deg"] == pytest.approx(90.0, abs=2.5)
    assert summary["wing_half_opening_deg"] == pytest.approx(35.0, abs=2.5)
    assert summary["q_star_direct"] == pytest.approx(0.40, abs=0.02)
    assert summary["L_star_direct_nm"] == pytest.approx(2.0 * math.pi / 0.40, rel=0.05)
    assert summary["q_star_kratky"] > summary["q_star_direct"]
    assert summary["L_star_kratky_nm"] is not None
    assert summary["q_win_invariant"] > 0.0
    assert summary["friedel_q1_q3_ratio"] == pytest.approx(1.0, abs=0.05)
    assert summary["friedel_q2_q4_ratio"] == pytest.approx(1.0, abs=0.05)

    # Uncalibrated pixel-q MUST return None for nm lengths
    px = measure_frame(
        img, qx, qy, valid_mask=valid, q_unit="pixel-q", q_edges=np.linspace(0.10, 0.90, 81), angular_bins=72
    )
    _assert_strict_json(px["summary"])
    assert px["summary"]["q_star_direct"] is not None
    assert px["summary"]["L_star_direct_nm"] is None
    assert px["summary"]["L_star_kratky_nm"] is None


# ---------------------------------------------------------------------------
# 7. Real BL19B2 Edge-Case Regressions (BUG 3 & BUG 4)
# ---------------------------------------------------------------------------


def test_bright_axis_convention_reference_vs_draw_and_custom_expected():
    # NO50S-like equatorial-bridged butterfly: vertical wire draw_axis=90 deg,
    # bright bridge / acute bisector along reference_axis=0 deg (normal),
    # and orthogonally flipped butterfly with bright bridge along 90 deg (run22-like).
    img_normal, qx, qy = _make_butterfly_2d(chi_sym_deg=1.5, alpha_deg=38.0, c0=12.0, c2=6.0)
    img_flipped, _, _ = _make_butterfly_2d(chi_sym_deg=85.0, alpha_deg=33.0, c0=12.0, c2=6.0)

    # Under bright_axis_convention="reference" (nominal_draw_axis_deg=90 -> expected=0):
    norm_ref = compute_azimuthal_harmonics_and_order(
        img_normal,
        qx,
        qy,
        q_window=(0.25, 0.50),
        q_unit="nm^-1",
        nominal_draw_axis_deg=90.0,
        bright_axis_convention="reference",
    )
    flip_ref = compute_azimuthal_harmonics_and_order(
        img_flipped,
        qx,
        qy,
        q_window=(0.25, 0.50),
        q_unit="nm^-1",
        nominal_draw_axis_deg=90.0,
        bright_axis_convention="reference",
    )
    _assert_strict_json(norm_ref)
    _assert_strict_json(flip_ref)

    assert norm_ref["bright_axis_convention"] == "reference"
    assert norm_ref["expected_bright_axis_deg"] == pytest.approx(0.0)
    assert abs(norm_ref["angle_to_reference_axis_deg"]) < 3.0
    assert abs(abs(norm_ref["angle_to_draw_axis_deg"]) - 90.0) < 3.0
    assert norm_ref["orthogonal_rotation_detected"] is False

    assert flip_ref["orthogonal_rotation_detected"] is True
    assert abs(flip_ref["angle_to_reference_axis_deg"]) > 75.0
    assert abs(flip_ref["angle_to_draw_axis_deg"]) < 10.0

    # Default bright_axis_convention="draw" preserves legacy behavior (expected=90 deg)
    norm_draw = compute_azimuthal_harmonics_and_order(
        img_normal, qx, qy, q_window=(0.25, 0.50), q_unit="nm^-1", nominal_draw_axis_deg=90.0
    )
    assert norm_draw["bright_axis_convention"] == "draw"
    assert norm_draw["expected_bright_axis_deg"] == pytest.approx(90.0)
    assert norm_draw["orthogonal_rotation_detected"] is True

    # Explicit expected_bright_axis_deg overrides convention
    custom = compute_azimuthal_harmonics_and_order(
        img_flipped,
        qx,
        qy,
        q_window=(0.25, 0.50),
        q_unit="nm^-1",
        nominal_draw_axis_deg=90.0,
        expected_bright_axis_deg=85.0,
    )
    assert custom["expected_bright_axis_deg"] == pytest.approx(85.0)
    assert custom["orthogonal_rotation_detected"] is False

    with pytest.raises(ValueError, match="bright_axis_convention"):
        compute_azimuthal_harmonics_and_order(
            img_normal, qx, qy, q_window=(0.25, 0.50), bright_axis_convention="invalid"
        )


def test_radial_coarsening_flat_diffuse_floor_and_run02_kratky_guard():
    # Replicate BL19B2 run02 (375C): correlation peak at q=0.39 nm^-1 sitting on
    # a flat 6.0 cm^-1 diffuse background + outer zero-padded dead region at q > 1.45.
    rng = np.random.default_rng(123)
    axis = np.linspace(-1.6, 1.6, 261)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    chi = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    I_bg_true = 6.0
    angular = 0.6 + 0.4 * np.cos(2.0 * np.deg2rad(chi)) + np.exp(-0.5 * ((np.abs(((chi % 180.0) - 90.0)) - 52.0) / 10.0) ** 2)
    img = I_bg_true + 32.0 * np.exp(-0.5 * ((q - 0.39) / 0.065) ** 2) * angular + rng.normal(0.0, 0.25, size=q.shape)
    # Zero-padded outer corners (like abs2d_v4 H5 beyond circular active region)
    img[q > 1.45] = 0.0
    valid = q >= 0.09

    res = measure_radial_coarsening_observables(
        img, qx, qy, valid_mask=valid, q_window=(0.11, 1.55), q_unit="nm^-1", n_radial_bins=120
    )
    _assert_strict_json(res)
    assert res["kratky_background_subtracted"] is True
    assert res["I_bg_estimate"] == pytest.approx(I_bg_true, abs=0.5)
    assert "outer_zero_padded_shells_excluded" in res["flags"]
    assert res["q_star_direct"] == pytest.approx(0.39, abs=0.02)
    # Kratky MUST stay near the true ~0.40-0.43 nm^-1 peak, NOT climb to 1.15-1.45 on the diffuse floor!
    assert res["q_star_kratky"] is not None
    assert 0.38 <= res["q_star_kratky"] <= 0.46
    assert res["L_star_kratky_nm"] == pytest.approx(2.0 * math.pi / res["q_star_kratky"], rel=1e-6)
    # Raw q^2 I(q) profile remains preserved alongside subtracted Kratky
    assert "q2_I_q" in res["full_azimuth"]
    assert "q2_I_sub_q" in res["full_azimuth"]


def test_radial_coarsening_steep_low_q_decay_with_noisy_flat_tail_rejects_ripple_run22_like():
    # Replicate BL19B2 run22 (475C): steep low-q decay (5000+ cm^-1 at q=0.11)
    # down to a noisy 6.9 cm^-1 flat diffuse floor, with a real Kratky shoulder at q~0.16 nm^-1.
    rng = np.random.default_rng(77)
    axis = np.linspace(-1.45, 1.45, 261)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    I_bg_true = 6.9
    qc = 0.16
    # Debye-Bueche low-q scattering (~5200 at q=0.11) + flat diffuse floor + statistical noise
    signal = 11500.0 / (1.0 + (q / qc) ** 2) ** 2
    img = I_bg_true + signal + rng.normal(0.0, 0.35, size=q.shape)
    valid = q >= 0.09

    res = measure_radial_coarsening_observables(
        img, qx, qy, valid_mask=valid, q_window=(0.11, 1.35), q_unit="nm^-1", n_radial_bins=120
    )
    _assert_strict_json(res)
    expected_tail = I_bg_true + 11500.0 / (1.0 + (1.25 / qc) ** 2) ** 2
    assert res["I_bg_estimate"] == pytest.approx(expected_tail, abs=0.8)
    # Direct I(q) MUST NOT pick a tiny statistical ripple in the flat 6.9 cm^-1 tail
    assert res["q_star_direct"] is None
    assert res["L_star_direct_nm"] is None
    assert res["edge_hit_direct"] is True
    assert res["direct_low_q_edge_hit"] is True
    # Baseline-subtracted Kratky MUST recover the true ~0.16 nm^-1 correlation shoulder
    assert res["q_star_kratky"] is not None
    assert res["q_star_kratky"] == pytest.approx(qc, abs=0.025)
    assert res["L_star_kratky_nm"] == pytest.approx(2.0 * math.pi / res["q_star_kratky"], rel=1e-6)
    assert res["kratky_low_q_edge_hit"] is False

