from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.auto_advisor import (
    MORPHOLOGY_CLASSES,
    SCHEMA_VERSION,
    advise_frame,
    advise_from_arrays,
    build_beamstop_exclusion_mask,
)
from butterfly_saxs.pipeline import inspect_frame, synthetic_butterfly


def _make_grid(
    shape: tuple[int, int] = (160, 160),
    *,
    center: tuple[float, float] | None = None,
    dq: float = 0.008,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    h, w = shape
    cx, cy = (0.5 * (w - 1), 0.5 * (h - 1)) if center is None else center
    yy, xx = np.indices(shape, dtype=np.float64)
    qx = (xx - cx) * dq
    qy = (yy - cy) * dq
    q = np.hypot(qx, qy)
    chi_deg = np.degrees(np.arctan2(qy, qx)) % 360.0
    r_px = np.hypot(xx - cx, yy - cy)
    return q, chi_deg, r_px, (cx, cy)


def _angular_gaussian(chi_deg: np.ndarray, center_deg: float, sigma_deg: float) -> np.ndarray:
    diff = ((chi_deg - center_deg + 180.0) % 360.0) - 180.0
    return np.exp(-0.5 * (diff / sigma_deg) ** 2)


def _make_four_lobe_butterfly(
    q: np.ndarray,
    chi_deg: np.ndarray,
    *,
    bisector_deg: float = 90.0,
    half_opening_deg: float = 34.0,
    q_star: float = 0.28,
    radial_sigma: float = 0.08,
    angular_sigma_deg: float = 11.0,
    amplitude: float = 120.0,
    bg: float = 2.0,
    straight_ray: bool = False,
) -> np.ndarray:
    angles = [
        (bisector_deg - half_opening_deg) % 360.0,
        (bisector_deg + half_opening_deg) % 360.0,
        (bisector_deg + 180.0 - half_opening_deg) % 360.0,
        (bisector_deg + 180.0 + half_opening_deg) % 360.0,
    ]
    ang_term = np.zeros_like(q)
    for ang in angles:
        ang_term += _angular_gaussian(chi_deg, ang, angular_sigma_deg)
    if straight_ray:
        # Radial X-streak spanning from low q outward along fixed azimuths
        rad_term = np.exp(-q / 0.45) + 0.5 * np.exp(-0.5 * ((q - q_star) / 0.14) ** 2)
    else:
        rad_term = np.exp(-0.5 * ((q - q_star) / radial_sigma) ** 2)
    return bg + amplitude * rad_term * ang_term


def test_build_beamstop_exclusion_mask_geometry_and_validation() -> None:
    mask = build_beamstop_exclusion_mask((41, 51), (25.0, 20.0), 10.0)
    assert mask.shape == (41, 51)
    assert mask.dtype == np.bool_
    # True means excluded (inside disk), False means retained
    assert mask[20, 25] is np.True_
    assert mask[20, 35] is np.True_
    assert mask[20, 36] is np.False_
    assert mask[0, 0] is np.False_

    empty = build_beamstop_exclusion_mask((16, 16), (8.0, 8.0), 0.0)
    assert not np.any(empty)

    with pytest.raises(ValueError, match="shape"):
        build_beamstop_exclusion_mask((0, 10), (5.0, 5.0), 2.0)
    with pytest.raises(ValueError, match="radius_px"):
        build_beamstop_exclusion_mask((16, 16), (8.0, 8.0), -1.0)
    with pytest.raises(ValueError, match="beam_center_xy"):
        build_beamstop_exclusion_mask((16, 16), (float("nan"), 8.0), 3.0)


def test_beamstop_shadow_detection_and_low_q_peak_recovery() -> None:
    # Mimics SPring-8 BL19B2: dq=0.008 nm^-1/px, unmasked beamstop r<=10 px,
    # coarsened low-q correlation peak at q*=0.15 nm^-1 (r ~ 18.75 px).
    q, chi_deg, r_px, center = _make_grid((180, 180), dq=0.008)
    porod = 40.0 / (q ** 2 + 0.04)
    lobes = _make_four_lobe_butterfly(
        q,
        chi_deg,
        bisector_deg=92.0,
        half_opening_deg=35.0,
        q_star=0.15,
        radial_sigma=0.04,
        amplitude=220.0,
        bg=1.5,
    )
    image = porod + lobes
    image[r_px <= 10.0] = 0.0  # Unmasked beamstop shadow

    valid = np.ones_like(image, dtype=bool)
    advice = advise_from_arrays(
        image, q, chi_deg, valid, q_unit="nm^-1", nominal_draw_axis_deg=90.0
    )
    json.dumps(advice, allow_nan=False)

    assert advice["schema_version"] == SCHEMA_VERSION
    assert set(advice) >= {
        "schema_version",
        "beamstop",
        "q_window",
        "symmetry",
        "parameters",
        "warnings",
        "remediation",
    }
    bs = advice["beamstop"]
    assert bs["detected"] is True
    assert bs["unmasked_beamstop"] is True
    assert 9.0 <= bs["shadow_radius_px"] <= 11.5
    assert bs["recommended_mask_radius_px"] >= bs["shadow_radius_px"] + 2.0
    assert abs(bs["beam_center_xy"][0] - center[0]) < 0.5
    assert abs(bs["beam_center_xy"][1] - center[1]) < 0.5

    # Low-q window must stay below the q*=0.15 peak instead of clipping at 0.25
    q_min, q_max = advice["q_window"]["window"]
    assert 0.08 <= q_min < 0.14
    assert q_max > 0.25
    assert "beamstop_unmasked_low_q" in advice["warnings"]
    assert any(item["flag"] == "beamstop_unmasked_low_q" for item in advice["remediation"])


def test_premasked_and_low_count_beamstop_modes() -> None:
    q, chi_deg, r_px, _ = _make_grid((140, 140), dq=0.01)
    base = 30.0 / (q ** 2 + 0.05) + _make_four_lobe_butterfly(q, chi_deg, q_star=0.25, bg=2.0)

    # Pre-masked beamstop
    valid_masked = r_px > 11.0
    res_masked = advise_from_arrays(base, q, chi_deg, valid_masked, q_unit="nm^-1")
    assert res_masked["beamstop"]["detected"] is True
    assert res_masked["beamstop"]["unmasked_beamstop"] is False
    assert 10.0 <= res_masked["beamstop"]["shadow_radius_px"] <= 12.0
    assert "beamstop_unmasked_low_q" not in res_masked["warnings"]

    # Non-zero low-count unmasked shadow core (2% leakage)
    low_core = base.copy()
    low_core[r_px <= 9.0] = 0.4
    res_low = advise_from_arrays(low_core, q, chi_deg, np.ones_like(base, dtype=bool), q_unit="nm^-1")
    assert res_low["beamstop"]["detected"] is True
    assert res_low["beamstop"]["unmasked_beamstop"] is True
    assert 8.0 <= res_low["beamstop"]["shadow_radius_px"] <= 11.0


def test_isotropic_ring_with_detector_gaps_stays_isotropic() -> None:
    q, chi_deg, r_px, _ = _make_grid((160, 160), dq=0.01)
    image = 2.0 + 100.0 * np.exp(-0.5 * ((q - 0.35) / 0.05) ** 2)
    # Mask wide module gaps + asymmetric wedge to verify no gap-filling artifacts
    valid = np.ones_like(image, dtype=bool)
    valid[76:84, :] = False
    valid[:, 76:84] = False
    valid[(chi_deg >= 15.0) & (chi_deg <= 45.0)] = False
    valid[r_px <= 6.0] = False
    # Inject NaN/Inf in masked pixels to ensure strict JSON safety
    image[0, 0] = np.nan
    image[0, 1] = np.inf
    valid[0, 0] = True

    advice = advise_from_arrays(image, q, chi_deg, valid, q_unit="nm^-1")
    json.dumps(advice, allow_nan=False)

    sym = advice["symmetry"]
    assert sym["morphology_class"] == "isotropic_ring"
    assert sym["orthogonal_flip_detected"] is False
    assert sym["friedel_bisector_deg"] is None
    assert sym["harmonics"]["a2"] < 0.05
    assert sym["harmonics"]["a4"] < 0.05
    assert advice["parameters"]["recommended_workflow"] == "sector"
    assert "near_circular_axis_angle_unconstrained" in advice["warnings"]


def test_two_lobe_four_lobe_tilt_and_orthogonal_flip() -> None:
    q, chi_deg, _, _ = _make_grid((160, 160), dq=0.01)
    valid = np.ones_like(q, dtype=bool)

    # 1. Two-lobe pattern along 94 deg
    two_lobe = 2.0 + 100.0 * np.exp(-0.5 * ((q - 0.30) / 0.06) ** 2) * (
        _angular_gaussian(chi_deg, 94.0, 18.0) + _angular_gaussian(chi_deg, 274.0, 18.0)
    )
    res_two = advise_from_arrays(two_lobe, q, chi_deg, valid, q_unit="nm^-1", nominal_draw_axis_deg=90.0)
    assert res_two["symmetry"]["morphology_class"] == "two_lobe"
    assert res_two["symmetry"]["orthogonal_flip_detected"] is False
    assert abs(res_two["symmetry"]["friedel_bisector_deg"] - 94.0) < 3.0
    assert abs(res_two["symmetry"]["mounting_tilt_deg"] - 4.0) < 3.0
    assert res_two["parameters"]["recommended_workflow"] == "sector"

    # 2. Four-lobe butterfly with +6 deg mounting tilt around nominal 90 deg
    four_lobe = _make_four_lobe_butterfly(
        q, chi_deg, bisector_deg=96.0, half_opening_deg=33.0, q_star=0.30, bg=2.0
    )
    res_four = advise_from_arrays(four_lobe, q, chi_deg, valid, q_unit="nm^-1", nominal_draw_axis_deg=90.0)
    assert res_four["symmetry"]["morphology_class"] == "four_lobe_butterfly"
    assert res_four["symmetry"]["orthogonal_flip_detected"] is False
    assert abs(res_four["symmetry"]["friedel_bisector_deg"] - 96.0) < 3.0
    assert abs(res_four["symmetry"]["mounting_tilt_deg"] - 6.0) < 3.0
    assert res_four["symmetry"]["harmonics"]["a4"] > 0.15

    # 3. Orthogonal-flipped butterfly (~90 deg rotated, bisector at 4 deg vs nominal 90 deg)
    flipped = _make_four_lobe_butterfly(
        q, chi_deg, bisector_deg=4.0, half_opening_deg=32.0, q_star=0.30, bg=2.0
    )
    res_flip = advise_from_arrays(flipped, q, chi_deg, valid, q_unit="nm^-1", nominal_draw_axis_deg=90.0)
    assert res_flip["symmetry"]["morphology_class"] == "orthogonal_flipped_butterfly"
    assert res_flip["symmetry"]["orthogonal_flip_detected"] is True
    assert "orthogonal_axis_flip" in res_flip["warnings"]
    assert any(item["flag"] == "orthogonal_axis_flip" for item in res_flip["remediation"])


def test_streak_contamination_and_low_signal_classification() -> None:
    q, chi_deg, r_px, _ = _make_grid((160, 160), dq=0.01)
    valid = r_px > 6.0

    # Sharp continuous parasitic horizontal streak (FWHM ~ 5 deg across all q)
    streak = 2.0 + 180.0 * np.exp(-0.5 * (q / 0.6) ** 2) * (
        _angular_gaussian(chi_deg, 0.0, 2.2) + _angular_gaussian(chi_deg, 180.0, 2.2)
    )
    res_streak = advise_from_arrays(streak, q, chi_deg, valid, q_unit="nm^-1")
    assert res_streak["symmetry"]["morphology_class"] == "streak_contaminated"
    assert "streak_contaminated" in res_streak["warnings"]
    assert res_streak["parameters"]["recommended_workflow"] == "sector"

    # Pure low-SNR noise image
    rng = np.random.default_rng(42)
    noise = 10.0 + rng.normal(0.0, 1.0, size=q.shape)
    res_low = advise_from_arrays(noise, q, chi_deg, valid, q_unit="nm^-1")
    assert res_low["symmetry"]["morphology_class"] == "low_signal"
    assert "low_signal" in res_low["warnings"]
    assert res_low["symmetry"]["morphology_class"] in MORPHOLOGY_CLASSES

    # Flat constant image also classifies as low_signal without NaN/Inf
    flat = np.full_like(q, 5.0)
    res_flat = advise_from_arrays(flat, q, chi_deg, valid, q_unit="nm^-1")
    json.dumps(res_flat, allow_nan=False)
    assert res_flat["symmetry"]["morphology_class"] == "low_signal"


def test_straight_radial_x_streak_triggers_high_ellipse_identifiability_risk() -> None:
    q, chi_deg, r_px, _ = _make_grid((180, 180), dq=0.01)
    valid = r_px > 8.0

    # Straight radial X-streaks radiating at fixed angles from origin
    x_streak = _make_four_lobe_butterfly(
        q,
        chi_deg,
        bisector_deg=90.0,
        half_opening_deg=35.0,
        q_star=0.28,
        angular_sigma_deg=10.0,
        amplitude=150.0,
        bg=1.5,
        straight_ray=True,
    )
    res_straight = advise_from_arrays(x_streak, q, chi_deg, valid, q_unit="nm^-1")
    assert res_straight["symmetry"]["morphology_class"] == "four_lobe_butterfly"
    assert res_straight["parameters"]["ellipse_identifiability_risk"] == "high"
    assert "axis_ratio_at_bound" in res_straight["warnings"]

    # Nyquist-safe bin bounds
    params = res_straight["parameters"]
    assert 4 <= params["annular_radial_bins"] <= 192
    assert params["annular_radial_bins"] <= params["nyquist_max_radial_bins"]
    assert 16 <= params["annular_angle_bins"] <= 720
    assert params["annular_angle_bins"] <= params["nyquist_max_angle_bins"]
    assert params["recommended_workflow"] in {"annular", "curvature", "sector"}

    # Curved double-ellipse pattern (synthetic_butterfly) should not look like a straight radial ray
    curved_img, qmap = synthetic_butterfly((128, 128), q0=24.0, width=2.2, ellipticity=2.2, angle_deg=32.0, return_qmap=True)
    curved_q = qmap["q"]
    curved_chi = np.degrees(np.arctan2(qmap["qy"], qmap["qx"])) % 360.0
    res_curved = advise_from_arrays(
        curved_img, curved_q, curved_chi, np.ones_like(curved_img, dtype=bool), q_unit="pixel-q", nominal_draw_axis_deg=0.0
    )
    assert res_curved["parameters"]["ellipse_identifiability_risk"] in {"low", "moderate"}
    assert "uncalibrated_pixel_q" in res_curved["warnings"]


def test_advise_frame_and_pipeline_inspect_frame_recommend(tmp_path: Path) -> None:
    q, chi_deg, r_px, _ = _make_grid((128, 128), dq=0.01)
    image = _make_four_lobe_butterfly(q, chi_deg, bisector_deg=90.0, half_opening_deg=34.0, q_star=0.28)
    image[r_px <= 8.0] = 0.0
    qx = q * np.cos(np.deg2rad(chi_deg))
    qy = q * np.sin(np.deg2rad(chi_deg))

    npz_path = tmp_path / "butterfly_frame.npz"
    np.savez(npz_path, data=image, qx=qx, qy=qy, q=q, q_unit=np.asarray("nm^-1"))

    # Direct file API
    advice = advise_frame(npz_path, nominal_draw_axis_deg=90.0)
    json.dumps(advice, allow_nan=False)
    assert advice["schema_version"] == SCHEMA_VERSION
    assert advice["provenance"]["image_path"] == str(npz_path)
    assert advice["beamstop"]["unmasked_beamstop"] is True

    # Pipeline backward compatibility: default recommend=False has no 'advisor' key
    legacy_report = inspect_frame(npz_path)
    assert "advisor" not in legacy_report
    json.dumps(legacy_report, allow_nan=False)

    # Opt-in recommend=True attaches additive 'advisor' key
    rec_report = inspect_frame(npz_path, recommend=True, nominal_draw_axis_deg=90.0)
    assert "advisor" in rec_report
    assert rec_report["advisor"]["schema_version"] == SCHEMA_VERSION
    assert rec_report["advisor"]["beamstop"]["unmasked_beamstop"] is True
    assert set(legacy_report.keys()).issubset(set(rec_report.keys()))
    json.dumps(rec_report, allow_nan=False)

    # Direct file API with external mask file (.npy, True=excluded)
    bs_mask = build_beamstop_exclusion_mask(image.shape, (63.5, 63.5), 10.0)
    mask_path = tmp_path / "beamstop_mask.npy"
    np.save(mask_path, bs_mask)
    advice_masked = advise_frame(npz_path, mask_path=mask_path, nominal_draw_axis_deg=90.0)
    json.dumps(advice_masked, allow_nan=False)
    assert advice_masked["beamstop"]["detected"] is True
    assert advice_masked["beamstop"]["unmasked_beamstop"] is False
    assert "beamstop_unmasked_low_q" not in advice_masked["warnings"]


def test_advise_from_arrays_rejects_invalid_inputs() -> None:
    q, chi_deg, _, _ = _make_grid((32, 32), dq=0.01)
    img = np.ones_like(q)

    with pytest.raises(ValueError, match="empty"):
        advise_from_arrays(img, q, chi_deg, np.zeros_like(img, dtype=bool), q_unit="nm^-1")
    with pytest.raises(ValueError, match="shape"):
        advise_from_arrays(img, q[:16, :16], chi_deg, np.ones_like(img, dtype=bool), q_unit="nm^-1")
    with pytest.raises(ValueError, match="nominal_draw_axis_deg"):
        advise_from_arrays(img, q, chi_deg, np.ones_like(img, dtype=bool), q_unit="nm^-1", nominal_draw_axis_deg=math.nan)
    with pytest.raises(ValueError, match="bright_axis_convention"):
        advise_from_arrays(img, q, chi_deg, np.ones_like(img, dtype=bool), q_unit="nm^-1", bright_axis_convention="invalid")


def test_bl19b2_outer_zero_padding_2px_beamstop_rim_and_diffuse_floor() -> None:
    """Replicate real SPring-8 BL19B2 run06 edge cases:
    - circular outer zero-fill cliff at q >= 6.0 nm^-1 (detector corners out to ~10 nm^-1)
    - flat active diffuse floor ~5.0 cm^-1 with ~1.5 cm^-1 statistical noise across q = 1.5..6.0
    - unmasked beamstop core r=0..9 px (10..23 cm^-1) with 2-px rising rim r=10 (383), r=11 (696), r=12..13 (850+)
    """
    # Use a 260x260 grid scaled so r=10 px -> q~0.08 nm^-1, q_cutoff=6.0 nm^-1, corners ~8.5 nm^-1
    # Wait: to match both r=10 px beamstop and q=6.0 cutoff cleanly, use dq=0.055? No, test exact
    # BL19B2 dq=0.008 on a 200x200 box OR scaled dq=0.06 where r_bs=10 -> q=0.6? Better: test BOTH
    # exact r=10 px rim on dq=0.04 (corners q=5.65, zero cliff at q=4.2) so every physical feature
    # (10 px beamstop, 2-px rim, SAXS signal up to ~1.2, flat 5.0 floor to 4.2, zero fill 4.2..5.65)
    # is faithfully resolved without huge memory!
    rng = np.random.default_rng(2026)
    dq = 0.04
    q, chi_deg, r_px, _ = _make_grid((220, 220), dq=dq)
    # Flat diffuse floor ~5.0 cm^-1 + realistic statistical noise (~1.5 cm^-1)
    diffuse = 5.0 + rng.normal(0.0, 1.5, size=q.shape)
    # Strong low-q coarsened butterfly signal decaying into diffuse floor by q ~ 1.4 nm^-1
    lobes = _make_four_lobe_butterfly(
        q,
        chi_deg,
        bisector_deg=1.3,
        half_opening_deg=39.0,
        q_star=0.55,
        radial_sigma=0.22,
        angular_sigma_deg=14.0,
        amplitude=500.0,
        bg=0.0,
    )
    low_q_halo = 550.0 * np.exp(-0.5 * ((q - 0.50) / 0.25) ** 2)
    image = np.maximum(diffuse + lobes + low_q_halo, 0.1)

    # Unmasked beamstop shadow with realistic 2-px partial-shadow rim (run06 profile):
    # r<=9 px: 10..23 cm^-1; r=10 px: ~380 cm^-1; r=11 px: ~700 cm^-1; r>=12: ~880 cm^-1
    image[r_px <= 9.4] = 10.0 + 1.4 * r_px[r_px <= 9.4]
    rim10 = (r_px > 9.4) & (r_px <= 10.4)
    image[rim10] = 380.0
    rim11 = (r_px > 10.4) & (r_px <= 11.4)
    image[rim11] = 700.0

    # Unmasked circular outer zero-fill beyond q >= 3.8 nm^-1 (detector corners reach ~6.2 nm^-1)
    image[q >= 3.8] = 0.0

    valid = np.ones_like(image, dtype=bool)
    advice = advise_from_arrays(
        image,
        q,
        chi_deg,
        valid,
        q_unit="nm^-1",
        nominal_draw_axis_deg=90.0,
        bright_axis_convention="reference",
    )
    json.dumps(advice, allow_nan=False)

    # BUG 1: 2-px rim beamstop shadow MUST be detected even with outer zero-padding
    bs = advice["beamstop"]
    assert bs["detected"] is True
    assert bs["unmasked_beamstop"] is True
    assert 9.0 <= bs["shadow_radius_px"] <= 11.0
    assert 11.0 <= bs["recommended_mask_radius_px"] <= 13.5
    assert "beamstop_unmasked_low_q" in advice["warnings"]

    # BUG 2: outer zero-padded zone excluded; diffuse floor ~5.0 recovered; q_max stops near SAXS end
    qw = advice["q_window"]
    assert qw["outer_zero_padded"] is True
    assert qw["q_active_max"] < 3.9
    assert qw["q_valid_max"] > 5.0
    assert 3.0 <= qw["diffuse_floor_estimate"] <= 7.0
    assert qw["q_max"] < 2.2  # Must NOT extend to the q=3.8 cliff or q=6.2 corner
    if qw["q_star_kratky_estimate"] is not None:
        assert qw["q_star_kratky_estimate"] < 2.0  # Must NOT peak at the zero-cliff
    assert "annular_outer_window_truncated" not in advice["warnings"]
    for rem in advice["remediation"]:
        suggested_q_max = rem["suggested_parameters"].get("q_max")
        if suggested_q_max is not None:
            assert suggested_q_max <= qw["q_active_max"] + 1e-6


def test_bright_axis_convention_reference_vs_draw_for_equatorial_bridge_butterflies() -> None:
    """Verify BUG 3 fix: NO50S equatorial-bridged butterflies (draw_axis_deg=90,
    reference_axis=0) vs orthogonal-flipped run22 (bright bridge at ~84-90 deg)."""
    q, chi_deg, r_px, _ = _make_grid((160, 160), dq=0.01)
    valid = r_px > 8.0

    # Normal NO50S run06-like: 4 lobes around 0 deg (39 deg, 141 deg, 219 deg, 321 deg)
    # + bright equatorial C2 bridge at 0/180 deg, dark vertical notch at 90/270 deg.
    normal_no50s = (
        2.0
        + _make_four_lobe_butterfly(
            q, chi_deg, bisector_deg=1.5, half_opening_deg=39.0, q_star=0.30, amplitude=140.0, bg=0.0
        )
        + 45.0 * np.exp(-0.5 * ((q - 0.30) / 0.08) ** 2) * (0.5 + 0.5 * np.cos(np.deg2rad(2.0 * (chi_deg - 1.5))))
    )

    # Flipped NO50S run22-like: 4 lobes around 84 deg (52 deg, 116 deg, 232 deg, 296 deg)
    # + bright vertical C2 bridge at 84/264 deg, dark horizontal notch at 174/354 deg.
    flipped_no50s = (
        2.0
        + _make_four_lobe_butterfly(
            q, chi_deg, bisector_deg=84.0, half_opening_deg=32.0, q_star=0.30, amplitude=140.0, bg=0.0
        )
        + 45.0 * np.exp(-0.5 * ((q - 0.30) / 0.08) ** 2) * (0.5 + 0.5 * np.cos(np.deg2rad(2.0 * (chi_deg - 84.0))))
    )

    # Under bright_axis_convention="reference" with nominal_draw_axis_deg=90.0:
    res_norm_ref = advise_from_arrays(
        normal_no50s,
        q,
        chi_deg,
        valid,
        q_unit="nm^-1",
        nominal_draw_axis_deg=90.0,
        bright_axis_convention="reference",
    )
    sym_nr = res_norm_ref["symmetry"]
    assert sym_nr["bright_axis_convention"] == "reference"
    assert sym_nr["expected_bright_axis_deg"] == pytest.approx(0.0)
    assert sym_nr["orthogonal_flip_detected"] is False
    assert sym_nr["morphology_class"] == "four_lobe_butterfly"
    assert abs(sym_nr["angle_to_reference_axis_deg"]) < 5.0
    assert abs(abs(sym_nr["angle_to_draw_axis_deg"]) - 90.0) < 5.0
    assert abs(((sym_nr["bright_bridge_axis_deg"] - 1.5 + 90.0) % 180.0) - 90.0) < 5.0
    assert abs(((sym_nr["dark_notch_axis_deg"] - 91.5 + 90.0) % 180.0) - 90.0) < 5.0
    assert "orthogonal_axis_flip" not in res_norm_ref["warnings"]

    res_flip_ref = advise_from_arrays(
        flipped_no50s,
        q,
        chi_deg,
        valid,
        q_unit="nm^-1",
        nominal_draw_axis_deg=90.0,
        bright_axis_convention="reference",
    )
    sym_fr = res_flip_ref["symmetry"]
    assert sym_fr["orthogonal_flip_detected"] is True
    assert sym_fr["morphology_class"] == "orthogonal_flipped_butterfly"
    assert abs(sym_fr["angle_to_draw_axis_deg"]) < 10.0
    assert abs(abs(sym_fr["angle_to_reference_axis_deg"]) - 90.0) < 10.0
    assert "orthogonal_axis_flip" in res_flip_ref["warnings"]

    # Explicit expected_bright_axis_deg overrides convention cleanly
    res_explicit = advise_from_arrays(
        normal_no50s,
        q,
        chi_deg,
        valid,
        q_unit="nm^-1",
        nominal_draw_axis_deg=90.0,
        bright_axis_convention="draw",
        expected_bright_axis_deg=0.0,
    )
    assert res_explicit["symmetry"]["orthogonal_flip_detected"] is False
    assert res_explicit["symmetry"]["morphology_class"] == "four_lobe_butterfly"


def test_advise_frame_loads_poni_once_and_supports_beamstop_and_geometry_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import butterfly_saxs.geometry as geom_mod

    q, chi_deg, r_px, center = _make_grid((128, 128), dq=0.01)
    image = _make_four_lobe_butterfly(q, chi_deg, bisector_deg=90.0, half_opening_deg=34.0, q_star=0.28)
    image[r_px <= 8.0] = 0.0
    qx = q * np.cos(np.deg2rad(chi_deg))
    qy = q * np.sin(np.deg2rad(chi_deg))

    npz_path = tmp_path / "frame.npz"
    np.savez(npz_path, data=image, qx=qx, qy=qy, q=q, q_unit=np.asarray("nm^-1"))
    fake_poni = tmp_path / "fake.poni"
    fake_poni.write_text("# dummy\n", encoding="utf-8")

    class _FakeAI:
        def qArray(self, shape: tuple[int, int]) -> np.ndarray:
            return q

        def center_array(self, shape: tuple[int, int], unit: str = "chi_rad", scale: bool = True) -> np.ndarray:
            return np.deg2rad(chi_deg)

        def getFit2D(self) -> dict[str, float]:
            return {"centerX": center[0], "centerY": center[1]}

    load_calls = 0
    fake_ai = _FakeAI()

    def _counting_load_poni(poni_arg):
        nonlocal load_calls
        if poni_arg is fake_ai:
            return fake_ai
        load_calls += 1
        return fake_ai

    monkeypatch.setattr(geom_mod, "load_poni", _counting_load_poni)

    res = advise_frame(npz_path, poni_path=fake_poni, nominal_draw_axis_deg=90.0)
    json.dumps(res, allow_nan=False)
    assert load_calls == 1
    assert res["beamstop"]["unmasked_beamstop"] is True

    # Supplying beamstop_radius_px masks the unmasked beamstop in memory without mutating files
    res_bs = advise_frame(npz_path, poni_path=fake_poni, beamstop_radius_px=10.0, nominal_draw_axis_deg=90.0)
    json.dumps(res_bs, allow_nan=False)
    assert load_calls == 2  # +1 single load for second call
    assert res_bs["beamstop"]["unmasked_beamstop"] is False
    assert res_bs["provenance"]["beamstop_radius_px"] == pytest.approx(10.0)

    # Passing geometry_cache avoids loading PONI completely
    geom_maps = geom_mod.GeometryMaps(
        q_nm_inv=q,
        chi_rad=np.deg2rad(chi_deg),
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=np.ones_like(image, dtype=bool),
        metadata={"q_unit": "nm^-1"},
    )
    cache = geom_mod.build_geometry_cache(geom_maps, beamstop_radius_px=10.0, beam_center_xy=center, pixel_dq=0.01)
    res_cached = advise_frame(npz_path, geometry_cache=cache, nominal_draw_axis_deg=90.0)
    json.dumps(res_cached, allow_nan=False)
    assert load_calls == 2  # zero additional PONI loads
    assert res_cached["beamstop"]["unmasked_beamstop"] is False
    assert res_cached["symmetry"]["morphology_class"] == "four_lobe_butterfly"

