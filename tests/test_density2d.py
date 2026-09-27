from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from butterfly_saxs.density2d import (
    Density2DProfile,
    analyze_density2d,
    export_density2d_bundle,
    fit_density_fluctuation,
    measure_density2d_profile,
)


def _q_grid(size: int = 501, bound: float = 0.22):
    axis = np.linspace(-bound, bound, size)
    return np.meshgrid(axis, axis)


def test_ornstein_zernike_candidate_recovers_length_from_measured_2d_sector():
    qx, qy = _q_grid()
    q = np.hypot(qx, qy)
    xi = 42.0
    image = 3.0 + 120.0 / (1.0 + np.square(q * xi))
    angle = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    valid = ~((q >= 0.04) & (q <= 0.055) & (np.abs((angle + 180.0) % 360.0 - 180.0) <= 18.0))

    result = analyze_density2d(
        image,
        qx=qx,
        qy=qy,
        q_window=(0.005, 0.18),
        azimuth_center_deg=0.0,
        azimuth_half_width_deg=18.0,
        n_q=120,
        valid_mask=valid,
        q_unit="nm^-1",
        source="frame-12",
    )
    profile = result.profile
    oz = next(fit for fit in result.fits if fit.model_name == "ornstein_zernike")
    power = next(fit for fit in result.fits if fit.model_name == "power_law")
    masked_bins = (profile.geometry_counts > 0) & (profile.counts == 0)

    assert profile.source == "frame-12"
    assert profile.n_supported_bins > 100
    assert "masked_pixels_in_selected_sector" in profile.flags
    assert power.success and np.isfinite(power.parameters["alpha"])
    assert oz.success, oz.message
    assert abs(float(oz.parameters["xi"]) - xi) < 2.5
    assert oz.parameters["xi_unit"] == "nm"
    assert masked_bins.any()
    assert np.all(np.isnan(oz.residual[masked_bins]))
    assert "empirical_models_do_not_determine_unique_structure" in result.flags


def test_power_law_fit_uses_only_positive_observed_2d_bins():
    qx, qy = _q_grid()
    q = np.hypot(qx, qy)
    with np.errstate(divide="ignore", invalid="ignore"):
        image = 17.0 * np.power(q, -2.4)
    image[q == 0.0] = np.nan
    profile = measure_density2d_profile(
        image, qx=qx, qy=qy, q_window=(0.01, 0.18), n_q=120,
        azimuth_center_deg=45.0, azimuth_half_width_deg=18.0, q_unit="nm^-1",
    )

    fit = fit_density_fluctuation(profile, models=("power_law",))[0]

    assert fit.success
    assert abs(float(fit.parameters["alpha"]) - 2.4) < 0.04
    assert fit.fit_space == "log_intensity"
    assert fit.n_points == profile.n_supported_bins
    assert np.all(np.isnan(fit.residual[~np.isfinite(profile.intensity)]))


def test_zero_q_pixel_is_not_assigned_to_any_azimuth_sector():
    axis = np.linspace(-0.1, 0.1, 101)
    qx, qy = np.meshgrid(axis, axis)
    image = np.ones_like(qx)
    profile = measure_density2d_profile(
        image, qx=qx, qy=qy, q_window=(0.0, 0.1),
        azimuth_center_deg=0.0, azimuth_half_width_deg=15.0, n_q=20,
    )
    central_bin = int(np.argmin(profile.q))

    assert profile.counts[central_bin] > 0
    assert profile.geometry_counts.sum() == int(np.count_nonzero(
        (np.hypot(qx, qy) > 0.0)
        & (np.hypot(qx, qy) <= 0.1)
        & (np.abs((np.degrees(np.arctan2(qy, qx)) + 180.0) % 360.0 - 180.0) <= 15.0)
    ))
    assert "zero_q_pixels_excluded_due_to_undefined_azimuth" in profile.flags


def test_missing_sector_data_and_uncalibrated_q_do_not_claim_physical_xi():
    qx, qy = _q_grid(301)
    q = np.hypot(qx, qy)
    image = 4.0 + 80.0 / (1.0 + np.square(q * 20.0))
    no_pixels = np.zeros_like(image, dtype=bool)
    empty_profile = measure_density2d_profile(
        image, qx=qx, qy=qy, q_window=(0.01, 0.18), valid_mask=no_pixels,
        q_unit="pixel-q",
    )
    empty_fits = fit_density_fluctuation(empty_profile)

    assert "no_measured_pixels_in_selected_sector" in empty_profile.flags
    assert all(not fit.success and "no_measured_bins" in fit.flags for fit in empty_fits)
    assert all(np.all(np.isnan(fit.fitted_intensity)) for fit in empty_fits)

    physical = analyze_density2d(
        image, qx=qx, qy=qy, q_window=(0.01, 0.18), q_unit="pixel-q",
        models=("ornstein_zernike",),
    ).fits[0]
    assert physical.success
    assert "xi" not in physical.parameters
    assert "xi_candidate" not in physical.parameters
    assert "physical_correlation_length_unavailable_without_calibrated_q" in physical.flags


def test_flat_ornstein_zernike_candidate_marks_unidentified_xi():
    q = np.linspace(0.1, 0.9, 40)
    profile = Density2DProfile(
        q=q,
        intensity=np.full(q.shape, 10.0),
        counts=np.full(q.shape, 4, dtype=int),
        geometry_counts=np.full(q.shape, 4, dtype=int),
        coverage=np.ones(q.shape),
        q_min=float(q.min()),
        q_max=float(q.max()),
        azimuth_center_deg=0.0,
        azimuth_half_width_deg=18.0,
        q_unit="nm^-1",
    )

    fit = fit_density_fluctuation(profile, models=("ornstein_zernike",))[0]

    assert fit.success
    assert "xi_candidate" in fit.parameters
    assert "xi" not in fit.parameters
    assert "oz_amplitude_near_zero_xi_unidentified" in fit.flags


def test_export_is_strict_and_requires_explicit_overwrite(tmp_path):
    qx, qy = _q_grid(201)
    q = np.hypot(qx, qy)
    image = 2.0 + 10.0 / (1.0 + np.square(q * 8.0))
    result = analyze_density2d(
        image, qx=qx, qy=qy, q_window=(0.01, 0.16), q_unit="nm^-1", source="frame-a",
    )
    csv_path, json_path = export_density2d_bundle(
        result.profile, result.fits, tmp_path / "sector.csv"
    )
    with json_path.open(encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=lambda value: pytest.fail(f"invalid JSON number: {value}"))
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert payload["schema"] == "butterfly_saxs.density2d.v1"
    assert payload["profile"]["source"] == "frame-a"
    assert len(rows) == len(result.profile.q)
    assert rows[0]["q_unit"] == "nm^-1"
    assert {"power_law_residual", "ornstein_zernike_residual"} <= set(rows[0])
    with pytest.raises(FileExistsError, match="already exist"):
        export_density2d_bundle(result.profile, result.fits, csv_path)
    export_density2d_bundle(result.profile, result.fits, csv_path, overwrite=True)


def test_invalid_density_sector_inputs_are_actionable():
    qx, qy = _q_grid(31)
    image = np.ones_like(qx)
    with pytest.raises(ValueError, match="increasing"):
        measure_density2d_profile(image, qx=qx, qy=qy, q_window=(0.2, 0.1))
    with pytest.raises(ValueError, match="no finite q-map pixels"):
        measure_density2d_profile(image, qx=qx, qy=qy, q_window=(2.0, 3.0))
    with pytest.raises(ValueError, match="broadcast to the image shape"):
        measure_density2d_profile(image, qx=np.ones((2, 2)), qy=qy, q_window=(0.0, 1.0))
    profile = measure_density2d_profile(image, qx=qx, qy=qy, q_window=(0.01, 0.2))
    with pytest.raises(ValueError, match="at least one density model"):
        fit_density_fluctuation(profile, models=())
