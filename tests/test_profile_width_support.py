from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.observables import (
    AngularSpectrum,
    _linear_fwhm,
    _periodic_fwhm,
    measure_four_lobe_peaks,
)


def test_periodic_fwhm_recovers_wrapped_gaussian_from_measured_coordinates() -> None:
    period = 2.0 * np.pi
    n = 240
    step = period / n
    regular = -np.pi + np.arange(n) * step
    angles = regular + 0.25 * step * np.sin(np.arange(n) * 0.73)
    sigma = 0.24
    centre = np.pi - 0.12
    half_width = sigma * np.sqrt(2.0 * np.log(2.0))
    exact_crossings = np.array((centre - half_width, centre + half_width - period, centre))
    keep = np.ones(angles.size, dtype=bool)
    for crossing in exact_crossings:
        keep &= np.abs(angles - crossing) > 0.7 * step
    x = np.sort(np.concatenate((angles[keep], exact_crossings)))
    distance = np.angle(np.exp(1j * (x - centre)))
    baseline = 1.0
    y = baseline + 10.0 * np.exp(-0.5 * (distance / sigma) ** 2)
    index = int(np.argmin(np.abs(x - centre)))

    width = _periodic_fwhm(x, y, index, baseline)

    assert width == pytest.approx(2.0 * half_width, abs=2e-5)


def test_linear_fwhm_interpolates_both_finite_half_height_crossings() -> None:
    x = np.arange(5, dtype=float)
    y = np.array((0.0, 1.0, 4.0, 1.0, 0.0))

    assert _linear_fwhm(x, y, 2, 0.0) == pytest.approx(4.0 / 3.0)


def test_periodic_fwhm_is_nan_when_mask_gap_removes_a_crossing_bracket() -> None:
    n = 360
    period = 2.0 * np.pi
    x = -np.pi + (np.arange(n, dtype=float) + 0.5) * period / n
    baseline = 1.0
    centre = 0.05
    sigma = 0.20
    distance = np.angle(np.exp(1j * (x - centre)))
    y = baseline + 10.0 * np.exp(-0.5 * (distance / sigma) ** 2)
    index = int(np.argmax(y))
    half_crossing = centre + sigma * np.sqrt(2.0 * np.log(2.0))
    missing = int(np.searchsorted(x, half_crossing, side="left"))
    y[missing] = np.nan

    assert np.isnan(_periodic_fwhm(x, y, index, baseline))
    spectrum = AngularSpectrum(
        angle=x,
        intensity=y,
        counts=np.isfinite(y).astype(int),
        candidate_counts=np.ones(n, dtype=int),
        coverage=np.isfinite(y).astype(float),
        q_min=0.1,
        q_max=0.2,
        q_center=0.15,
    )
    peaks = measure_four_lobe_peaks(spectrum, expected=1, snr_threshold=0.1)
    assert len(peaks) == 1
    assert np.isfinite(peaks[0].angle)
    assert np.isnan(peaks[0].fwhm)


def test_linear_fwhm_is_nan_when_peak_is_truncated_at_window_edge() -> None:
    x = np.arange(5, dtype=float)
    y = np.array((4.0, 3.0, 2.0, 1.0, 0.0))

    assert np.isnan(_linear_fwhm(x, y, 0, 0.0))
