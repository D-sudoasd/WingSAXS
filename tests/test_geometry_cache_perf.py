from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from butterfly_saxs.geometry import (
    GeometryError,
    GeometryMaps,
    GeometryShapeError,
    SAXSGeometryCache,
    apply_beamstop_mask,
    build_geometry_cache,
)
from butterfly_saxs.report_measurements import measure_frame
from butterfly_saxs.saxs_physics import (
    _fit_1d_transverse_gaussian,
    compute_azimuthal_harmonics_and_order,
    evaluate_c2_bridge_four_lobe,
    fit_c2_bridge_four_lobe,
    measure_perpendicular_streak_width,
    measure_radial_coarsening_observables,
    run_window_sensitivity_audit,
)

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _make_synthetic_geometry(
    shape: tuple[int, int] = (201, 221),
    *,
    center_xy: tuple[float, float] = (108.4, 97.6),
    dq: float = 0.012,
) -> GeometryMaps:
    h, w = shape
    cx, cy = center_xy
    y = np.arange(h, dtype=np.float64) - cy
    x = np.arange(w, dtype=np.float64) - cx
    qx = np.broadcast_to((x * dq)[None, :], shape).copy()
    qy = np.broadcast_to((y * dq)[:, None], shape).copy()
    q = np.hypot(qx, qy)
    chi_rad = np.mod(np.arctan2(qy, qx), 2.0 * np.pi)
    valid = np.ones(shape, dtype=bool)
    valid[h // 2 - 2 : h // 2 + 2, :] = False  # horizontal module gap
    return GeometryMaps(
        q_nm_inv=q,
        chi_rad=chi_rad,
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=valid,
        metadata={"q_unit": "nm^-1"},
    )


def _make_butterfly_image(geom: GeometryMaps, *, q_star: float = 0.36) -> np.ndarray:
    q = geom.q_nm_inv
    chi = geom.chi_deg
    assert chi is not None
    rad_env = np.exp(-0.5 * ((q - q_star) / 0.065) ** 2)
    ang_env = evaluate_c2_bridge_four_lobe(
        chi,
        c0=12.0,
        c2=6.0,
        chi_sym_deg=2.0,
        alpha_deg=37.0,
        lobe_amplitudes=(35.0, 34.0, 35.0, 34.0),
        lobe_sigmas_deg=22.0 / _FWHM_PER_SIGMA,
    )
    return 4.5 + rad_env * ang_env


def test_apply_beamstop_mask_exact_broadcast_and_validation() -> None:
    shape = (91, 113)
    center = (53.25, 44.75)
    radius = 12.5
    base_valid = np.ones(shape, dtype=bool)
    base_valid[10:20, 15:25] = False

    combined, excl = apply_beamstop_mask(base_valid, shape, center, radius)
    yy, xx = np.indices(shape, dtype=np.float64)
    expected_excl = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= (radius ** 2)

    assert excl.dtype == np.bool_
    assert combined.dtype == np.bool_
    assert np.array_equal(excl, expected_excl)
    assert np.array_equal(combined, base_valid & ~expected_excl)
    # Input base_valid must never be mutated
    assert base_valid[45, 53] is np.True_

    # valid_mask=None and radius=0
    comb_none, excl_none = apply_beamstop_mask(None, shape, center, radius)
    assert np.array_equal(comb_none, ~expected_excl)
    assert np.array_equal(excl_none, expected_excl)

    comb_zero, excl_zero = apply_beamstop_mask(base_valid, shape, center, 0.0)
    assert not np.any(excl_zero)
    assert np.array_equal(comb_zero, base_valid)

    # Validation errors
    with pytest.raises(ValueError, match="shape"):
        apply_beamstop_mask(None, (0, 10), center, 5.0)
    with pytest.raises(ValueError, match="beam_center_xy"):
        apply_beamstop_mask(None, shape, (float("nan"), 10.0), 5.0)
    with pytest.raises(ValueError, match="radius_px"):
        apply_beamstop_mask(None, shape, center, -1.0)
    with pytest.raises(GeometryShapeError, match="valid_mask shape"):
        apply_beamstop_mask(np.ones((10, 10), dtype=bool), shape, center, 5.0)


def test_build_geometry_cache_attributes_roi_and_validation() -> None:
    geom = _make_synthetic_geometry((201, 221), center_xy=(108.4, 97.6), dq=0.012)
    cache = build_geometry_cache(geom, beamstop_radius_px=10.0, q_roi_max=0.95)

    assert isinstance(cache, SAXSGeometryCache)
    assert cache.geometry is geom
    assert cache.shape == (201, 221)
    assert cache.q_coordinates_validated is True
    assert cache.beam_center_xy[0] == pytest.approx(108.4, abs=1e-4)
    assert cache.beam_center_xy[1] == pytest.approx(97.6, abs=1e-4)
    assert cache.pixel_dq == pytest.approx(0.012, rel=1e-4)
    assert cache.base_valid_mask.shape == geom.shape
    assert cache.beamstop_exclusion_mask.shape == geom.shape
    assert cache.valid_mask.shape == geom.shape
    assert np.array_equal(cache.valid_mask, cache.base_valid_mask & ~cache.beamstop_exclusion_mask)

    # ROI must be strictly smaller than full frame and enclose every pixel with q <= q_roi_max
    assert cache.q_mag_roi.shape[0] < geom.shape[0]
    assert cache.q_mag_roi.shape[1] < geom.shape[1]
    assert cache.qx_roi.shape == cache.q_mag_roi.shape
    assert cache.qy_roi.shape == cache.q_mag_roi.shape
    assert cache.chi_deg_roi.shape == cache.q_mag_roi.shape
    assert cache.valid_roi.shape == cache.q_mag_roi.shape

    inside_roi = np.zeros(geom.shape, dtype=bool)
    inside_roi[cache.roi_slices] = True
    assert np.all(inside_roi[geom.q_nm_inv <= 0.95])

    # slice_frame returns exact ROI slice and rejects bad shapes
    frame = np.arange(geom.shape[0] * geom.shape[1], dtype=np.float64).reshape(geom.shape)
    sliced = cache.slice_frame(frame)
    assert np.array_equal(sliced, frame[cache.roi_slices])
    with pytest.raises(GeometryShapeError):
        cache.slice_frame(np.zeros((10, 10)))

    # Mapping input is also supported
    qmap_dict = {"qx": geom.qx, "qy": geom.qy, "q": geom.q, "valid_mask": geom.valid_mask, "q_unit": "nm^-1"}
    cache_map = build_geometry_cache(qmap_dict, beamstop_radius_px=10.0, q_roi_max=0.95)
    assert cache_map.roi_slices == cache.roi_slices
    assert np.allclose(cache_map.q_mag_roi, cache.q_mag_roi)

    with pytest.raises(GeometryError, match="q_roi_max"):
        build_geometry_cache(geom, q_roi_max=-1.0)


def test_geometry_cache_matches_uncached_physics_and_falls_back_when_window_exceeds_roi() -> None:
    geom = _make_synthetic_geometry((201, 221), center_xy=(108.4, 97.6), dq=0.012)
    img = _make_butterfly_image(geom, q_star=0.36)
    sigma = np.full_like(img, 0.25)

    cache = build_geometry_cache(geom, beamstop_radius_px=9.0, q_roi_max=0.90)
    valid = cache.valid_mask

    harm_uncached = compute_azimuthal_harmonics_and_order(
        img, geom.qx, geom.qy, valid_mask=valid, q_window=(0.20, 0.52), q_unit="nm^-1", n_bins=180, bright_axis_convention="reference"
    )
    harm_cached = compute_azimuthal_harmonics_and_order(
        img, q_window=(0.20, 0.52), q_unit="nm^-1", n_bins=180, bright_axis_convention="reference", geometry_cache=cache
    )
    assert harm_cached["a2"] == pytest.approx(harm_uncached["a2"], abs=1e-12)
    assert harm_cached["a4"] == pytest.approx(harm_uncached["a4"], abs=1e-12)
    assert harm_cached["friedel_bisector_deg"] == pytest.approx(harm_uncached["friedel_bisector_deg"], abs=1e-10)

    rad_uncached = measure_radial_coarsening_observables(
        img, geom.qx, geom.qy, valid_mask=valid, q_window=(0.12, 0.85), q_unit="nm^-1", n_radial_bins=64, sigma=sigma
    )
    rad_cached = measure_radial_coarsening_observables(
        img, q_window=(0.12, 0.85), q_unit="nm^-1", n_radial_bins=64, sigma=sigma, geometry_cache=cache
    )
    assert rad_cached["q_star_direct"] == pytest.approx(rad_uncached["q_star_direct"], abs=1e-10)
    assert rad_cached["q_star_kratky"] == pytest.approx(rad_uncached["q_star_kratky"], abs=1e-10)
    assert rad_cached["q_win_invariant"] == pytest.approx(rad_uncached["q_win_invariant"], rel=1e-12)
    assert rad_cached["q_win_stat_stderr"] == pytest.approx(rad_uncached["q_win_stat_stderr"], rel=1e-12)

    # Requesting q_window > cache.q_roi_max must transparently fall back to full geometry without truncating!
    rad_wide_uncached = measure_radial_coarsening_observables(
        img, geom.qx, geom.qy, valid_mask=valid, q_window=(0.12, 1.15), q_unit="nm^-1", n_radial_bins=64
    )
    rad_wide_cached = measure_radial_coarsening_observables(
        img, q_window=(0.12, 1.15), q_unit="nm^-1", n_radial_bins=64, geometry_cache=cache
    )
    assert rad_wide_cached["q_win_invariant"] == pytest.approx(rad_wide_uncached["q_win_invariant"], rel=1e-12)

    # Streak width cached vs uncached
    rays = [325.0, 39.0, 145.0, 219.0]
    sw_u = measure_perpendicular_streak_width(
        img, geom.qx, geom.qy, valid_mask=valid, q_window=(0.20, 0.60), q_unit="nm^-1", wing_angles_deg=rays, n_q_slices=8
    )
    sw_c = measure_perpendicular_streak_width(
        img, q_window=(0.20, 0.60), q_unit="nm^-1", wing_angles_deg=rays, n_q_slices=8, geometry_cache=cache
    )
    assert sw_c["status"] == sw_u["status"]
    assert sw_c["summary"]["total_valid_slices"] == sw_u["summary"]["total_valid_slices"]

    # Window sensitivity audit cached vs uncached
    audit_u = run_window_sensitivity_audit(
        img, geom.qx, geom.qy, valid, q_unit="nm^-1", q_windows=((0.22, 0.85), (0.12, 0.85)), sigma=sigma
    )
    audit_c = run_window_sensitivity_audit(
        img, q_unit="nm^-1", q_windows=((0.22, 0.85), (0.12, 0.85)), sigma=sigma, geometry_cache=cache
    )
    json.dumps(audit_c, allow_nan=False)
    for wu, wc in zip(audit_u["windows"], audit_c["windows"]):
        assert wc["q_win_invariant"] == pytest.approx(wu["q_win_invariant"], rel=1e-12)
        assert wc["q_star_direct"] == pytest.approx(wu["q_star_direct"], abs=1e-10)


def test_analytical_jacobian_c2_and_transverse_gaussian_accuracy() -> None:
    # Verify C2-bridge + 4-lobe seeded vs unseeded and shared vs independent widths
    angles = np.linspace(1.0, 359.0, 180)
    y = evaluate_c2_bridge_four_lobe(
        angles,
        c0=14.0,
        c2=5.5,
        chi_sym_deg=4.2,
        alpha_deg=36.5,
        lobe_amplitudes=(31.0, 29.0, 30.5, 28.5),
        lobe_sigmas_deg=(9.5, 10.2, 9.5, 10.2),
    )
    fit_unseeded = fit_c2_bridge_four_lobe(angles, y, shared_width=False)
    fit_seeded = fit_c2_bridge_four_lobe(
        angles, y, shared_width=False, initial_chi_sym_deg=4.0, initial_alpha_deg=36.0
    )
    assert fit_unseeded["success"] is True and fit_seeded["success"] is True
    assert fit_seeded["c0"] == pytest.approx(14.0, abs=1e-3)
    assert fit_seeded["c2"] == pytest.approx(5.5, abs=1e-3)
    assert fit_seeded["chi_sym_deg"] == pytest.approx(4.2, abs=1e-3)
    assert fit_seeded["alpha_deg"] == pytest.approx(36.5, abs=1e-3)
    assert fit_unseeded["alpha_deg"] == pytest.approx(fit_seeded["alpha_deg"], abs=1e-3)
    assert fit_seeded["rmse"] < 1e-6

    # Verify include_flat_comparison=False skips flat baseline cleanly
    fit_fast = fit_c2_bridge_four_lobe(
        angles, y, initial_chi_sym_deg=4.0, initial_alpha_deg=36.0, include_flat_comparison=False
    )
    assert fit_fast["success"] is True
    assert fit_fast["flat_baseline_comparison"] is None
    assert fit_fast["alpha_deg"] == pytest.approx(36.5, abs=0.1)

    # Verify 1D transverse Gaussian fit with analytical Jacobian recovers exact parameters
    x_perp = np.linspace(-0.25, 0.25, 31)
    true_sig = 0.042
    y_perp = 3.5 + 50.0 * np.exp(-0.5 * ((x_perp - 0.012) / true_sig) ** 2)
    gfit = _fit_1d_transverse_gaussian(x_perp, y_perp, np.full_like(x_perp, 25, dtype=np.int64), q_perp_max=0.25, min_r2=0.95)
    assert gfit["valid"] is True
    assert gfit["baseline"] == pytest.approx(3.5, abs=1e-5)
    assert gfit["amplitude"] == pytest.approx(50.0, abs=1e-5)
    assert gfit["center_q_perp"] == pytest.approx(0.012, abs=1e-5)
    assert gfit["sigma_q_perp"] == pytest.approx(true_sig, abs=1e-5)
    assert gfit["fwhm_q_perp"] == pytest.approx(true_sig * _FWHM_PER_SIGMA, abs=1e-5)


def test_measure_frame_cached_matches_uncached_and_modules_stay_qt_free() -> None:
    geom = _make_synthetic_geometry((121, 131), center_xy=(65.0, 60.0), dq=0.015)
    img = _make_butterfly_image(geom, q_star=0.38)
    cache = build_geometry_cache(geom, beamstop_radius_px=8.0)

    res_u = measure_frame(img, geom.qx, geom.qy, valid_mask=cache.valid_mask, q_unit="nm^-1", radial_bins=48, angular_bins=36)
    res_c = measure_frame(
        img, geom.qx, geom.qy, valid_mask=cache.valid_mask, q_unit="nm^-1", radial_bins=48, angular_bins=36, geometry_cache=cache
    )
    json.dumps(res_c["summary"], allow_nan=False)
    assert res_c["summary"]["q_win_invariant"] == pytest.approx(res_u["summary"]["q_win_invariant"], rel=1e-12)
    assert res_c["summary"]["a4"] == pytest.approx(res_u["summary"]["a4"], abs=1e-12)
    assert res_c["summary"]["intensity"]["median"] == pytest.approx(res_u["summary"]["intensity"]["median"], abs=1e-12)
    assert res_c["summary"]["in_plane_intensity_tensor"]["anisotropy"] == pytest.approx(
        res_u["summary"]["in_plane_intensity_tensor"]["anisotropy"], abs=1e-12
    )
    assert np.allclose(res_c["polar"]["mean"], res_u["polar"]["mean"], equal_nan=True)
    assert np.allclose(res_c["polar"]["std"], res_u["polar"]["std"], equal_nan=True)

    # Verify Qt-free invariant in a clean subprocess (pytest-qt may preload PySide6 in-process)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import butterfly_saxs.geometry, butterfly_saxs.saxs_physics, "
                "butterfly_saxs.auto_advisor, butterfly_saxs.report_measurements; "
                "loaded = [m for m in ('PySide6', 'PyQt5', 'PyQt6') if m in sys.modules]; "
                "assert not loaded, f'Qt modules imported: {loaded}'"
            ),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Qt-free check failed: {proc.stderr}"
