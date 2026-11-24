"""Azimuthal spectrum, four-lobe peak localization, and apparent tilt observables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from ._observables_types import (
    APPARENT_FLAGS,
    AngularSpectrum,
    LobeMetrics,
    _bin_mean,
    _extract_maps,
    _flags_with_q_unit,
    _get_field,
    _q_limits,
    _q_unit,
)


def measure_angular_spectrum(
    frame: Any,
    qmap: Any,
    q_window: Any = None,
    *,
    q_range: Any = None,
    n_bins: int = 360,
    statistic: str = "mean",
    mask: Any = None,
    smooth_sigma: float = 0.0,
) -> AngularSpectrum:
    """Measure a masked angular spectrum over a q window.

    The profile is a per-pixel mean by default, so a wedge with fewer pixels
    does not look artificially weak.  Use ``statistic='sum'`` for integrated
    intensity.  No symmetry or mirror completion is applied.
    """

    if int(n_bins) < 8:
        raise ValueError("n_bins must be at least 8")
    if statistic not in {"mean", "sum"}:
        raise ValueError("statistic must be 'mean' or 'sum'")
    values, q, angle, valid = _extract_maps(frame, qmap, mask)
    q_unit = _q_unit(qmap)
    q_min, q_max = _q_limits(q, q_window, q_range)
    candidate = np.isfinite(values) & np.isfinite(q) & (q >= q_min) & (q <= q_max)
    selected = candidate & valid
    n_bins = int(n_bins)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx_all = np.clip(np.digitize(angle[candidate], edges, right=False) - 1, 0, n_bins - 1)
    idx_valid = np.clip(np.digitize(angle[selected], edges, right=False) - 1, 0, n_bins - 1)
    candidate_counts = np.bincount(idx_all, minlength=n_bins).astype(int)
    profile, counts = _bin_mean(values[selected], idx_valid, n_bins, statistic)
    if smooth_sigma:
        profile = _smooth_supported_angular_profile(profile, counts > 0, sigma=float(smooth_sigma))
    coverage = np.divide(counts, candidate_counts, out=np.zeros(n_bins, dtype=float), where=candidate_counts > 0)
    return AngularSpectrum(
        angle=centres,
        intensity=profile,
        counts=counts,
        candidate_counts=candidate_counts,
        coverage=coverage,
        q_min=q_min,
        q_max=q_max,
        q_center=0.5 * (q_min + q_max),
        statistic=statistic,
        flags=_flags_with_q_unit(APPARENT_FLAGS, q_unit),
        q_unit=q_unit,
    )


angular_spectrum = measure_angular_spectrum
azimuthal_profile = measure_angular_spectrum
extract_azimuthal_spectrum = measure_angular_spectrum


def _robust_noise(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    noise = 1.4826 * mad
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = float(np.std(finite, ddof=1))
    return noise


def _circular_supported_runs(support: np.ndarray) -> list[np.ndarray]:
    """Return contiguous supported angular-bin runs without crossing gaps.

    The angular coordinate is periodic, so the first and last bins belong to
    the same run when both are supported.  A run is deliberately kept as an
    explicit index array: callers can smooth or estimate noise within that
    run without ever borrowing a value across a masked detector gap.
    """

    support = np.asarray(support, dtype=bool).ravel()
    n_bins = int(support.size)
    if n_bins == 0 or not np.any(support):
        return []
    if np.all(support):
        return [np.arange(n_bins, dtype=int)]
    starts = np.flatnonzero(support & ~np.roll(support, 1))
    runs: list[np.ndarray] = []
    for start in starts.tolist():
        indices: list[int] = []
        index = int(start)
        while support[index]:
            indices.append(index)
            index = (index + 1) % n_bins
            if index == int(start):
                break
        runs.append(np.asarray(indices, dtype=int))
    return runs


def _smooth_supported_angular_profile(
    profile: np.ndarray,
    support: np.ndarray,
    *,
    sigma: float = 1.0,
) -> np.ndarray:
    """Smooth only within contiguous supported angular bins.

    ``gaussian_filter1d(..., mode='wrap')`` is valid for a complete angular
    profile, but it silently leaks intensity across a masked gap.  This helper
    uses periodic wrapping only for a fully supported profile and otherwise
    processes each contiguous run independently.
    """

    profile = np.asarray(profile, dtype=float).ravel()
    support = np.asarray(support, dtype=bool).ravel() & np.isfinite(profile)
    smoothed = np.full(profile.shape, np.nan, dtype=float)
    if not np.any(support):
        return smoothed
    if gaussian_filter1d is None or not np.isfinite(float(sigma)) or float(sigma) <= 0.0:
        smoothed[support] = profile[support]
        return smoothed
    runs = _circular_supported_runs(support)
    if len(runs) == 1 and runs[0].size == profile.size:
        smoothed[:] = gaussian_filter1d(profile, float(sigma), mode="wrap")
        return smoothed
    for run in runs:
        values = profile[run]
        if values.size == 1:
            smoothed[run] = values
        else:
            # ``nearest`` remains inside the supported run.  It never samples
            # the opposite side of a missing/masked angular interval.
            smoothed[run] = gaussian_filter1d(values, float(sigma), mode="nearest")
    return smoothed


def _supported_angular_noise(
    profile: np.ndarray,
    support: np.ndarray,
) -> tuple[float, str, dict[str, int]]:
    """Estimate detector noise from local angular changes, not lobe height.

    The MAD of an entire angular profile measures the physical four-lobe
    contrast as if it were detector noise.  Robust adjacent differences are
    therefore preferred; a local three-point detrended residual is a fallback
    for short supported runs.  Differences are added across the periodic
    boundary only when that boundary is actually supported.
    """

    profile = np.asarray(profile, dtype=float).ravel()
    support = np.asarray(support, dtype=bool).ravel() & np.isfinite(profile)
    runs = _circular_supported_runs(support)
    differences: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for run in runs:
        values = profile[run]
        if values.size >= 2:
            differences.append(np.diff(values))
            if run.size == profile.size:
                differences.append(np.asarray([values[0] - values[-1]], dtype=float))
        if values.size >= 3:
            residuals.append(values[1:-1] - 0.5 * (values[:-2] + values[2:]))
    diff_values = (
        np.concatenate(differences).astype(float, copy=False)
        if differences
        else np.asarray([], dtype=float)
    )
    residual_values = (
        np.concatenate(residuals).astype(float, copy=False)
        if residuals
        else np.asarray([], dtype=float)
    )
    if diff_values.size >= 2:
        noise = _robust_noise(diff_values) / np.sqrt(2.0)
        if np.isfinite(noise) and noise > np.finfo(float).eps:
            return float(noise), "adjacent_difference_mad", {
                "difference_count": int(diff_values.size),
                "detrended_residual_count": int(residual_values.size),
            }
    # Preserve a finite floor for the peak detector without falling back to
    # the full profile MAD: that would re-introduce the physical lobe contrast
    # as detector noise.  A constant or locally linear profile has no local
    # prominence and therefore remains safely peak-free at this floor.
    return float(np.finfo(float).eps), "local_difference_floor_short_or_constant", {
        "difference_count": int(diff_values.size),
        "detrended_residual_count": int(residual_values.size),
    }


def _wrap_distance(angle: np.ndarray | float, centre: float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.asarray(angle) - float(centre)))))


def _quadratic_peak(x: np.ndarray, y: np.ndarray, index: int, period: float | None = None) -> tuple[float, float]:
    """Sub-bin parabolic peak estimate; returns x and y."""

    n = len(y)
    index = int(index)
    if n < 3:
        return float(x[index]), float(y[index])
    if period is not None:
        period_val = float(period)
        if not np.isfinite(period_val) or period_val <= 0.0:
            raise ValueError("period must be finite and positive")
        im = (index - 1) % n
        ip = (index + 1) % n
        ym, y0, yp = float(y[im]), float(y[index]), float(y[ip])
        dx = float((float(x[index]) - float(x[im])) % period_val)
        if not np.isfinite(dx) or dx <= 0.0 or dx >= 0.5 * period_val:
            dx = period_val / float(n)
    else:
        if index <= 0 or index >= n - 1:
            return float(x[index]), float(y[index])
        ym, y0, yp = float(y[index - 1]), float(y[index]), float(y[index + 1])
        dx = float(x[index] - x[index - 1])
    if not (np.isfinite(ym) and np.isfinite(y0) and np.isfinite(yp)):
        return float(x[index]), y0
    denom = ym - 2.0 * y0 + yp
    if not np.isfinite(denom) or abs(denom) <= np.finfo(float).eps:
        return float(x[index]), y0
    offset = 0.5 * (ym - yp) / denom
    offset = float(np.clip(offset, -0.5, 0.5))
    xp = float(x[index] + offset * dx)
    ypval = float(y0 - 0.25 * (ym - yp) * offset)
    if period is not None:
        period_val = float(period)
        if abs(period_val - 2.0 * np.pi) <= 1e-9:
            xp = float(np.angle(np.exp(1j * xp)))
        elif float(np.nanmin(x)) < 0.0:
            xp = float((xp + 0.5 * period_val) % period_val - 0.5 * period_val)
        else:
            xp = float(xp % period_val)
    return xp, ypval


def _periodic_fwhm(x: np.ndarray, y: np.ndarray, index: int, baseline: float, period: float = 2.0 * np.pi) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    finite = np.isfinite(y_values)
    if not np.any(finite) or not np.isfinite(y_values[index]):
        return float("nan")
    y0 = float(y_values[index])
    half = baseline + 0.5 * (y0 - baseline)
    n = len(y_values)
    ext_y = np.tile(y_values, 3)
    ext_x = np.concatenate((x_values - period, x_values, x_values + period))
    centre = n + index
    if ext_y[centre] < half:
        return float("nan")
    left = centre
    while left > 0 and np.isfinite(ext_y[left - 1]) and ext_y[left - 1] >= half:
        left -= 1
    right = centre
    while right < len(ext_y) - 1 and np.isfinite(ext_y[right + 1]) and ext_y[right + 1] >= half:
        right += 1
    if left == 0 or right == len(ext_y) - 1:
        return float("nan")
    left_bracket = (left - 1, left)
    right_bracket = (right, right + 1)
    bracket_indices = (*left_bracket, *right_bracket)
    if not np.all(np.isfinite(ext_y[list(bracket_indices)])) or not np.all(
        np.isfinite(ext_x[list(bracket_indices)])
    ):
        return float("nan")
    # Linear crossing at each half-height boundary.
    def crossing(i0: int, i1: int) -> float:
        y0_, y1_ = ext_y[i0], ext_y[i1]
        if y1_ == y0_:
            return float(ext_x[i0])
        return float(ext_x[i0] + (half - y0_) * (ext_x[i1] - ext_x[i0]) / (y1_ - y0_))

    width = crossing(*right_bracket) - crossing(*left_bracket)
    return float(max(0.0, width)) if np.isfinite(width) else float("nan")


def _linear_fwhm(
    x: np.ndarray,
    y: np.ndarray,
    index: int,
    baseline: float,
    *,
    support_mask: np.ndarray | None = None,
) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    finite = np.isfinite(y_values)
    if not np.any(finite) or not np.isfinite(y_values[index]):
        return float("nan")
    half = baseline + 0.5 * (float(y_values[index]) - baseline)
    if y_values[index] < half:
        return float("nan")
    left = int(index)
    while left > 0 and np.isfinite(y_values[left - 1]) and y_values[left - 1] >= half:
        left -= 1
    right = int(index)
    while right < len(y_values) - 1 and np.isfinite(y_values[right + 1]) and y_values[right + 1] >= half:
        right += 1
    if left == 0 or right == len(y_values) - 1:
        return float("nan")
    bracket_indices = (left - 1, left, right, right + 1)
    if not np.all(np.isfinite(y_values[list(bracket_indices)])) or not np.all(
        np.isfinite(x_values[list(bracket_indices)])
    ):
        return float("nan")
    if support_mask is not None:
        measured = np.asarray(support_mask, dtype=bool)
        if measured.shape != y_values.shape or not np.all(measured[left - 1 : right + 2]):
            return float("nan")

    def cross(i0: int, i1: int) -> float:
        if y_values[i1] == y_values[i0]:
            return float(x_values[i0])
        return float(
            x_values[i0]
            + (half - y_values[i0]) * (x_values[i1] - x_values[i0]) / (y_values[i1] - y_values[i0])
        )

    width = cross(right, right + 1) - cross(left - 1, left)
    return float(max(0.0, width)) if np.isfinite(width) else float("nan")


def _refine_symmetric_lobes(
    spectrum: AngularSpectrum | Mapping[str, Any],
    lobes: Sequence[LobeMetrics],
    *,
    reference_axis_deg: float = 0.0,
) -> list[LobeMetrics]:
    """Refine observed candidates with one robust four-envelope angle model.

    The fit estimates a common apparent lobe angle and width from the finite,
    actually observed angular bins.  It never inserts a missing lobe: each
    returned record still corresponds to one detected candidate and retains
    that candidate's coverage and pixel count.
    """

    if least_squares is None or len(lobes) < 2:
        return list(lobes)
    angle = np.asarray(_get_field(spectrum, ("angle", "azimuth")), dtype=float)
    intensity = np.asarray(_get_field(spectrum, ("intensity", "profile")), dtype=float)
    counts = np.asarray(_get_field(spectrum, ("counts",)), dtype=float)
    coverage = np.asarray(_get_field(spectrum, ("coverage",)), dtype=float)
    finite = np.isfinite(angle) & np.isfinite(intensity) & (counts > 0)
    if np.count_nonzero(finite) < 12:
        return list(lobes)

    x = angle[finite]
    y = intensity[finite]
    weights = np.clip(coverage[finite], 0.0, 1.0)
    baseline = float(np.percentile(y, 10.0))
    amplitude = max(float(np.percentile(y, 99.0) - baseline), np.finfo(float).eps)
    scale = max(float(np.std(y)), np.finfo(float).eps)
    reference = np.deg2rad(float(reference_axis_deg))
    acute = [
        abs((float(lobe.angle) - reference + np.pi / 2.0) % np.pi - np.pi / 2.0)
        for lobe in lobes
    ]
    phi0 = float(np.clip(np.median(acute), 0.0, np.pi / 2.0))

    def model(candidate: np.ndarray) -> np.ndarray:
        offset, amplitude_plus, amplitude_minus, phi, sigma = candidate

        def envelope(centre: float) -> np.ndarray:
            delta = np.angle(np.exp(1j * (x - centre)))
            return np.exp(-0.5 * (delta / sigma) ** 2)

        return (
            offset
            + amplitude_plus * (envelope(reference + phi) + envelope(reference + np.pi + phi))
            + amplitude_minus * (envelope(reference - phi) + envelope(reference + np.pi - phi))
        )

    lower = np.asarray(
        [float(np.min(y) - amplitude), 0.0, 0.0, 0.0, np.deg2rad(1.0)],
        dtype=float,
    )
    upper = np.asarray(
        [float(np.max(y)), 10.0 * amplitude, 10.0 * amplitude, np.pi / 2.0, np.deg2rad(40.0)],
        dtype=float,
    )
    initial = np.asarray(
        [baseline, amplitude, amplitude, phi0, np.deg2rad(10.0)],
        dtype=float,
    )
    fit_candidates = []
    for phi_start in (phi0, 0.5 * phi0, min(np.pi / 2.0, 1.5 * phi0)):
        candidate_initial = initial.copy()
        candidate_initial[3] = max(lower[3], min(upper[3], phi_start))
        try:
            candidate_fit = least_squares(
                lambda candidate: (model(candidate) - y) * weights / scale,
                candidate_initial,
                bounds=(lower, upper),
                loss="cauchy",
                f_scale=0.5,
                max_nfev=500,
            )
        except (FloatingPointError, ValueError):
            continue
        if candidate_fit.success and np.all(np.isfinite(candidate_fit.x)):
            fit_candidates.append(candidate_fit)
    if not fit_candidates:
        return list(lobes)
    fit = min(fit_candidates, key=lambda candidate: (float(candidate.cost), tuple(candidate.x)))

    phi = float(fit.x[3])
    sigma = float(fit.x[4])
    centres = np.asarray(
        (
            reference + phi,
            reference - phi,
            reference + np.pi + phi,
            reference + np.pi - phi,
        ),
        dtype=float,
    )
    centres = np.angle(np.exp(1j * centres))
    assignments: dict[int, list[LobeMetrics]] = {}
    for lobe in lobes:
        index = int(np.argmin(_wrap_distance(centres, lobe.angle)))
        assignments.setdefault(index, []).append(lobe)

    overlap = bool(2.0 * phi <= 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)
    refined: list[LobeMetrics] = []
    for centre_index, candidates in assignments.items():
        candidates = sorted(candidates, key=lambda item: (item.valid, item.snr), reverse=True)
        for rank, lobe in enumerate(candidates):
            lobe.angle = float(centres[centre_index])
            lobe.fwhm = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)
            lobe.refinement = "symmetric_cauchy"
            if rank > 0:
                lobe.valid = False
                lobe.reason = "duplicate_symmetric_lobe_candidate"
                lobe.flags = tuple(dict.fromkeys(lobe.flags + (lobe.reason,)))
            elif overlap:
                lobe.valid = False
                lobe.reason = "overlapping_lobes_unresolved"
                lobe.flags = tuple(dict.fromkeys(lobe.flags + (lobe.reason,)))
            elif lobe.valid:
                lobe.reason = "accepted"
            refined.append(lobe)
    return sorted(refined, key=lambda item: item.angle)


def measure_four_lobe_peaks(
    spectrum: AngularSpectrum | Mapping[str, Any],
    *,
    expected: int = 4,
    min_prominence: float | None = None,
    snr_threshold: float = 2.0,
    min_distance_fraction: float = 0.12,
    symmetric_refine: bool = False,
    reference_axis_deg: float = 0.0,
) -> list[LobeMetrics]:
    """Find up to four angular lobes without completing missing quadrants."""

    angle = np.asarray(_get_field(spectrum, ("angle", "azimuth")), dtype=float)
    y = np.asarray(_get_field(spectrum, ("intensity", "profile")), dtype=float)
    counts = np.asarray(_get_field(spectrum, ("counts",)), dtype=float)
    coverage = np.asarray(_get_field(spectrum, ("coverage",)), dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return []
    baseline = float(np.nanpercentile(y, 10.0))
    noise = _robust_noise(y - baseline)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.nanstd(y)), np.finfo(float).eps)
    prominence = float(min_prominence) if min_prominence is not None else float(snr_threshold * noise)
    # Circular extension means lobes crossing -pi/pi are not discarded.
    if find_peaks is None:  # pragma: no cover
        candidates = np.asarray([int(np.nanargmax(y))])
    else:
        n = len(y)
        extended = np.tile(np.where(finite, y, baseline), 3)
        distance = max(1, int(round(n * float(min_distance_fraction))))
        candidates, props = find_peaks(extended, prominence=prominence, distance=distance)
        candidates = candidates[(candidates >= n) & (candidates < 2 * n)] - n
        # Deduplicate modulo n if a very broad peak is reported twice.
        unique: list[int] = []
        for candidate in sorted(candidates.tolist(), key=lambda i: float(y[i]), reverse=True):
            if all(_wrap_distance(angle[candidate], angle[i]) > (2.0 * np.pi / n) for i in unique):
                unique.append(int(candidate))
        candidates = np.asarray(unique, dtype=int)
    candidates = candidates[np.argsort(y[candidates])[::-1]] if len(candidates) else candidates
    candidates = candidates[: int(expected)]
    out: list[LobeMetrics] = []
    n = len(y)
    for index in candidates:
        index = int(index)
        peak_angle, peak_intensity = _quadratic_peak(angle, y, index, period=2.0 * np.pi)
        snr = (peak_intensity - baseline) / noise if noise else float("inf")
        fwhm = _periodic_fwhm(angle, y, index, baseline)
        # Integrate the positive lobe area over one angular-bin period.
        half = baseline + 0.5 * (peak_intensity - baseline)
        dist = _wrap_distance(angle, peak_angle)
        selected = np.isfinite(y) & (y >= half) & (dist <= max(fwhm / 2.0 if np.isfinite(fwhm) else np.inf, np.pi / n))
        if np.count_nonzero(selected) > 1:
            relative = np.angle(np.exp(1j * (angle[selected] - peak_angle)))
            order = np.argsort(relative)
            area = float(
                np.trapezoid(
                    np.maximum(y[selected][order] - baseline, 0.0),
                    relative[order],
                )
            )
        else:
            area = 0.0
        is_valid = bool(np.isfinite(snr) and snr >= snr_threshold)
        out.append(
            LobeMetrics(
                angle=float(peak_angle),
                intensity=float(peak_intensity),
                baseline=baseline,
                snr=float(snr),
                fwhm=float(fwhm),
                area=area,
                index=index,
                coverage=float(coverage[index]) if index < len(coverage) else 0.0,
                n_pixels=int(counts[index]) if index < len(counts) else 0,
                valid=is_valid,
                reason="accepted" if is_valid else "low_snr",
            )
        )
    out = sorted(out, key=lambda item: item.angle)
    return (
        _refine_symmetric_lobes(
            spectrum,
            out,
            reference_axis_deg=reference_axis_deg,
        )
        if symmetric_refine and int(expected) == 4
        else out
    )


find_four_lobe_peaks = measure_four_lobe_peaks
measure_four_peaks = measure_four_lobe_peaks


def apparent_lamellar_tilt(
    lobes: Sequence[LobeMetrics | Mapping[str, Any]],
    *,
    draw_axis_deg: float = 90.0,
) -> tuple[float, float]:
    """Return the directly observed lobe tilt ``phi`` and its robust spread.

    A SAXS reflection and its Friedel counterpart describe the same unoriented
    line, so every lobe angle is reduced modulo 180 degrees before measuring
    its absolute deviation from the draw axis.  In the ideal four-point case
    this equals half the peak-to-peak separation across the meridian described
    by Grubb et al.; no ellipse-axis angle or structural ``alpha`` is used.
    """

    draw_axis = np.deg2rad(float(draw_axis_deg))
    values: list[float] = []
    for lobe in lobes:
        if _get_field(lobe, ("valid",), True) is False:
            continue
        angle = _get_field(lobe, ("angle", "azimuth"))
        if angle is None or not np.isfinite(float(angle)):
            continue
        # Wrap an unoriented line difference onto [-pi/2, pi/2).
        delta = (float(angle) - draw_axis + np.pi / 2.0) % np.pi - np.pi / 2.0
        values.append(abs(delta))
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    centre = float(np.median(array))
    mad = float(np.median(np.abs(array - centre)))
    return float(np.degrees(centre)), float(np.degrees(1.4826 * mad))


__all__ = [
    "angular_spectrum",
    "apparent_lamellar_tilt",
    "azimuthal_profile",
    "extract_azimuthal_spectrum",
    "find_four_lobe_peaks",
    "measure_angular_spectrum",
    "measure_four_lobe_peaks",
    "measure_four_peaks",
    "_circular_supported_runs",
    "_linear_fwhm",
    "_periodic_fwhm",
    "_quadratic_peak",
    "_refine_symmetric_lobes",
    "_robust_noise",
    "_smooth_supported_angular_profile",
    "_supported_angular_noise",
    "_wrap_distance",
]
