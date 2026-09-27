from __future__ import annotations

import csv

import numpy as np
import pytest

from butterfly_saxs.azimuthal_analysis import (
    AzimuthalPeak,
    evaluate_azimuthal_model,
    export_azimuthal_csv_bundle,
    fit_azimuthal_peaks,
    measure_azimuthal_profile,
)


def _synthetic_annulus(model: str = "pseudo_voigt"):
    axis = np.linspace(-1.2, 1.2, 501)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    angle = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    peaks = (
        AzimuthalPeak(center_deg=358.0, amplitude=7.5, fwhm_deg=18.0, eta=0.32 if model == "pseudo_voigt" else None),
        AzimuthalPeak(center_deg=149.0, amplitude=4.0, fwhm_deg=27.0, eta=0.32 if model == "pseudo_voigt" else None),
    )
    image = evaluate_azimuthal_model(angle, baseline=2.1, peaks=peaks, model=model, eta=0.32)
    image += 0.15 * q
    return image, qx, qy, q


@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "pseudo_voigt", "von_mises"])
def test_periodic_multi_peak_fit_recovers_peak_across_zero(model):
    image, qx, qy, q = _synthetic_annulus(model)
    profile = measure_azimuthal_profile(
        image, qx=qx, qy=qy, q_window=(0.45, 0.8), q_unit="nm^-1", n_bins=360
    )
    fit = fit_azimuthal_peaks(
        profile,
        model=model,
        max_peaks=4,
        min_separation_deg=30.0,
        min_height_fraction=0.12,
        initial_fwhm_deg=25.0,
        fwhm_bounds_deg=(4.0, 80.0),
        center_window_deg=32.0,
        eta=0.32,
    )
    assert fit.success, fit.message
    assert fit.model_name == model
    assert len(fit.peaks) == 2
    centers = np.asarray([peak.center_deg for peak in fit.peaks])
    distance_to_zero_peak = np.abs((centers - 358.0 + 180.0) % 360.0 - 180.0)
    distance_to_other = np.abs((centers - 149.0 + 180.0) % 360.0 - 180.0)
    assert np.min(distance_to_zero_peak) < 2.0
    assert np.min(distance_to_other) < 2.0
    assert fit.rmse < 0.05
    assert np.all(np.isfinite(fit.residual[np.isfinite(profile.intensity)]))
    assert np.all(np.isnan(fit.residual[~np.isfinite(profile.intensity)]))
    assert profile.q_unit == "nm^-1"
    assert profile.q_min == 0.45 and profile.q_max == 0.8


def test_masked_angular_gap_stays_unmeasured_and_is_not_used_in_fit():
    image, qx, qy, q = _synthetic_annulus()
    angle = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    valid = np.ones(image.shape, dtype=bool)
    valid &= ~((angle < 14.0) | (angle > 346.0))
    profile = measure_azimuthal_profile(
        image, qx=qx, qy=qy, q_window=(0.45, 0.8), valid_mask=valid,
        q_unit="pixel-q", n_bins=360,
    )
    masked_bins = (profile.angle_deg < 14.0) | (profile.angle_deg > 346.0)
    assert np.all(profile.counts[masked_bins] == 0)
    assert np.all(np.isnan(profile.intensity[masked_bins]))
    assert np.all(profile.coverage[masked_bins] == 0.0)
    assert "masked_pixels_in_q_window" in profile.flags
    fit = fit_azimuthal_peaks(profile, min_height_fraction=0.12)
    assert fit.success
    assert "incomplete_pixel_coverage" in fit.flags
    assert np.all(np.isnan(fit.residual[masked_bins]))
    assert profile.q_unit == "pixel-q"


def test_profile_is_unweighted_per_pixel_mean_and_valid_mask_is_positive():
    image = np.full((4, 8), 2.0)
    qx = np.tile(np.linspace(0.4, 0.7, 8), (4, 1))
    qy = np.zeros_like(qx)
    image[0, 0] = 10.0
    valid = np.ones_like(image, dtype=bool)
    valid[0, 0] = False
    profile = measure_azimuthal_profile(
        image, qx=qx, qy=qy, q_window=(0.4, 0.7), valid_mask=valid,
        n_bins=36, q_unit="A^-1",
    )
    occupied = np.flatnonzero(profile.counts)
    assert occupied.size == 1
    assert profile.intensity[occupied[0]] == 2.0
    assert profile.counts[occupied[0]] == 31
    assert profile.geometry_counts[occupied[0]] == 32
    assert profile.q_unit == "A^-1"


def test_zero_q_beam_centre_is_excluded_when_window_starts_at_zero():
    axis = np.linspace(-1.0, 1.0, 101)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = np.ones_like(q)
    image[50, 50] = 1.0e9
    profile = measure_azimuthal_profile(
        image, qx=qx, qy=qy, q_window=(0.0, 1.4), n_bins=360
    )
    assert profile.geometry_counts.sum() == np.count_nonzero((q > 0.0) & (q <= 1.4))
    assert profile.counts.sum() == np.count_nonzero((q > 0.0) & (q <= 1.4))
    assert np.nanmax(profile.intensity) == 1.0


def test_odd_angular_bin_geometry_counts_use_the_same_bins_as_intensity():
    angle = np.deg2rad([10.0, 30.0, 50.0, 190.0, 210.0, 230.0])
    qx = (0.5 * np.cos(angle))[None, :]
    qy = (0.5 * np.sin(angle))[None, :]
    image = np.ones_like(qx)

    profile = measure_azimuthal_profile(
        image, qx=qx, qy=qy, q_window=(0.4, 0.6), n_bins=9,
    )

    assert np.array_equal(profile.counts, profile.geometry_counts)
    occupied = profile.geometry_counts > 0
    assert np.all(profile.coverage[occupied] == 1.0)
    assert np.all(profile.coverage[~occupied] == 0.0)


def test_odd_angular_bin_coverage_stays_bounded_with_masked_and_nonfinite_pixels():
    angle = np.deg2rad([10.0, 30.0, 50.0, 190.0, 210.0, 230.0])
    qx = (0.5 * np.cos(angle))[None, :]
    qy = (0.5 * np.sin(angle))[None, :]
    image = np.ones_like(qx)
    image[0, 2] = np.nan
    valid = np.ones_like(image, dtype=bool)
    valid[0, 4] = False

    profile = measure_azimuthal_profile(
        image, qx=qx, qy=qy, q_window=(0.4, 0.6), valid_mask=valid, n_bins=9,
    )

    assert np.all(profile.counts <= profile.geometry_counts)
    assert np.all((0.0 <= profile.coverage) & (profile.coverage <= 1.0))


def test_invalid_q_domains_shapes_and_fit_settings_are_actionable():
    image = np.ones((5, 5))
    grid = np.arange(5, dtype=float)
    qx, qy = np.meshgrid(grid, grid)
    with pytest.raises(ValueError, match="increasing"):
        measure_azimuthal_profile(image, qx=qx, qy=qy, q_window=(1.0, 0.0))
    with pytest.raises(ValueError, match="nonnegative"):
        measure_azimuthal_profile(image, qx=qx, qy=qy, q_window=(-1.0, 0.5))
    with pytest.raises(ValueError, match="no finite q-map pixels"):
        measure_azimuthal_profile(image, qx=qx, qy=qy, q_window=(20.0, 30.0))
    with pytest.raises(ValueError, match="broadcast to the image shape"):
        measure_azimuthal_profile(image, qx=np.ones((2, 2)), qy=qy, q_window=(0.0, 2.0))
    profile = measure_azimuthal_profile(image, qx=qx, qy=qy, q_window=(0.1, 4.0))
    with pytest.raises(ValueError, match="model must be"):
        fit_azimuthal_peaks(profile, model="ellipse")


def test_export_bundle_contains_profile_residuals_units_and_peak_parameters(tmp_path):
    image, qx, qy, q = _synthetic_annulus()
    profile = measure_azimuthal_profile(
        image, qx=qx, qy=qy, q_window=(0.45, 0.8), q_unit="nm^-1", source="frame-17"
    )
    fit = fit_azimuthal_peaks(profile, min_height_fraction=0.12, eta=0.32)
    profile_path, peaks_path = export_azimuthal_csv_bundle(profile, fit, tmp_path / "ring.csv")
    with profile_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with peaks_path.open(encoding="utf-8-sig", newline="") as stream:
        peak_rows = list(csv.DictReader(stream))
    assert len(rows) == len(profile.angle_deg)
    assert rows[0]["q_unit"] == "nm^-1"
    assert rows[0]["source"] == "frame-17"
    assert all("fitted_intensity" in row and "residual" in row for row in rows)
    assert len(peak_rows) == 2
    assert peak_rows[0]["fit_success"] == "1"
    assert {"center_deg", "amplitude", "fwhm_deg", "baseline", "rmse"} <= set(peak_rows[0])
