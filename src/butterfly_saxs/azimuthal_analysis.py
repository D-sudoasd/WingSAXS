"""Periodic peak analysis of angular intensity profiles from 2-D SAXS images.

Profiles are per-pixel means over an explicitly selected q annulus.  Masked,
non-finite, and geometrically unsupported bins remain distinguishable; fitting
uses only measured bins and never fills a detector gap with synthetic data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .observables import measure_angular_spectrum


_MODEL_ALIASES = {
    "gauss": "gaussian",
    "gaussian": "gaussian",
    "lorentz": "lorentzian",
    "lorentzian": "lorentzian",
    "pseudovoigt": "pseudo_voigt",
    "pseudo-voigt": "pseudo_voigt",
    "pseudo_voigt": "pseudo_voigt",
    "vm": "von_mises",
    "vonmises": "von_mises",
    "von-mises": "von_mises",
    "von_mises": "von_mises",
}


@dataclass(frozen=True, slots=True)
class AzimuthalProfile:
    """Measured angular means and sampling coverage for one q annulus."""

    angle_deg: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    geometry_counts: np.ndarray
    coverage: np.ndarray
    q_min: float
    q_max: float
    q_unit: str = "unknown"
    statistic: str = "mean"
    source: str | None = None
    flags: tuple[str, ...] = ()

    @property
    def q_center(self) -> float:
        return 0.5 * (self.q_min + self.q_max)

    @property
    def q_width(self) -> float:
        return self.q_max - self.q_min

    @property
    def n_supported_bins(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.intensity)))


@dataclass(frozen=True, slots=True)
class AzimuthalPeak:
    """One fitted periodic peak; amplitude is peak height above baseline."""

    center_deg: float
    amplitude: float
    fwhm_deg: float
    eta: float | None = None
    kappa: float | None = None


@dataclass(frozen=True, slots=True)
class AzimuthalFitResult:
    """Joint fit to the observed bins of one angular profile."""

    model_name: str
    peaks: tuple[AzimuthalPeak, ...]
    baseline: float
    model: np.ndarray
    residual: np.ndarray
    success: bool
    message: str
    rmse: float
    r_squared: float
    flags: tuple[str, ...] = ()
    initial_centres_deg: tuple[float, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)


def _finite_pair(value: Any, name: str) -> tuple[float, float]:
    try:
        pair = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numbers") from exc
    if len(pair) != 2 or not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain two finite numbers")
    if pair[0] >= pair[1]:
        raise ValueError(f"{name} must be increasing")
    return pair


def measure_azimuthal_profile(
    data: Any,
    *,
    qx: Any,
    qy: Any,
    q_window: Any,
    valid_mask: Any = None,
    q_unit: str = "unknown",
    n_bins: int = 360,
    statistic: str = "mean",
    source: str | None = None,
) -> AzimuthalProfile:
    """Measure intensity versus azimuth over a fixed calibrated q annulus.

    ``valid_mask=True`` means a pixel is usable.  ``geometry_counts`` counts
    finite q-map pixels before masks and intensity validity are applied;
    ``counts`` counts measured pixels.  Thus coverage is an observation
    fraction, not a fit weight or uncertainty estimate.
    """

    image = np.asarray(data, dtype=float)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("data must be a non-empty two-dimensional image")
    q_min, q_max = _finite_pair(q_window, "q_window")
    if q_min < 0.0:
        raise ValueError("q_window lower bound must be nonnegative for q magnitude")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError("n_bins must be an integer >= 8")
    n_bins = int(n_bins)
    if n_bins < 8:
        raise ValueError("n_bins must be an integer >= 8")
    statistic = str(statistic).strip().lower()
    if statistic not in {"mean", "sum"}:
        raise ValueError("statistic must be 'mean' or 'sum'")

    try:
        qx_array, qy_array = np.broadcast_arrays(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
        qx_array = np.broadcast_to(qx_array, image.shape)
        qy_array = np.broadcast_to(qy_array, image.shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("qx and qy must broadcast to the image shape") from exc
    q = np.hypot(qx_array, qy_array)
    # Azimuth is undefined at the beam centre (q=0), so those pixels do not
    # contribute to any angular bin even when the numeric q interval includes 0.
    geometry = np.isfinite(qx_array) & np.isfinite(qy_array) & np.isfinite(q) & (q > 0.0)
    geometry &= (q >= q_min) & (q <= q_max)
    if not np.any(geometry):
        raise ValueError("q_window contains no finite q-map pixels")

    if valid_mask is None:
        usable = np.ones(image.shape, dtype=bool)
    else:
        try:
            usable = np.broadcast_to(np.asarray(valid_mask, dtype=bool), image.shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("valid_mask must broadcast to the image shape") from exc
    # Reuse the app's canonical fixed-q angular integration for observed
    # intensity/counts.  Recompute the denominator from finite q geometry so
    # non-finite detector values are reported as lost coverage as well.
    qmap = {"qx": qx_array, "qy": qy_array, "q_unit": str(q_unit or "unknown")}
    spectrum = measure_angular_spectrum(
        image,
        qmap,
        (q_min, q_max),
        n_bins=n_bins,
        statistic=statistic,
        mask=~usable | ~(q > 0.0),
    )
    angle_deg = np.mod(np.degrees(spectrum.angle), 360.0)
    order = np.argsort(angle_deg)
    angle_deg = angle_deg[order]
    intensity = np.asarray(spectrum.intensity, dtype=float)[order]
    counts = np.asarray(spectrum.counts, dtype=np.int64)[order]
    # Match measure_angular_spectrum's exact [-pi, pi) edges before applying
    # the same centre sort.  A separate [0, 2pi) grid differs by half a bin
    # for odd n_bins and can report coverage above one.
    geometry_angle = np.mod(np.arctan2(qy_array, qx_array) + np.pi, 2.0 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    geometry_bin = np.digitize(geometry_angle[geometry], edges, right=False) - 1
    geometry_bin = np.clip(geometry_bin, 0, n_bins - 1)
    geometry_counts = np.bincount(geometry_bin, minlength=n_bins).astype(np.int64)[order]
    coverage = np.divide(
        counts, geometry_counts, out=np.zeros(n_bins, dtype=float), where=geometry_counts > 0,
    )
    flags: list[str] = []
    if np.any(geometry & ~usable):
        flags.append("masked_pixels_in_q_window")
    if np.any(geometry & ~np.isfinite(image)):
        flags.append("nonfinite_intensity_in_q_window")
    if not np.any(counts):
        flags.append("no_measured_pixels_in_q_window")
    return AzimuthalProfile(
        angle_deg=angle_deg,
        intensity=intensity,
        counts=counts,
        geometry_counts=geometry_counts,
        coverage=coverage,
        q_min=float(spectrum.q_min),
        q_max=float(spectrum.q_max),
        q_unit=str(spectrum.q_unit or q_unit or "unknown"),
        statistic=statistic,
        source=None if source is None else str(source),
        flags=tuple(dict.fromkeys((*flags, *spectrum.flags))),
    )


def _normalise_model(model: Any) -> str:
    name = str(model or "pseudo_voigt").strip().lower().replace(" ", "_")
    normalized = _MODEL_ALIASES.get(name)
    if normalized is None:
        raise ValueError("model must be gaussian, lorentzian, pseudo_voigt, or von_mises")
    return normalized


def _positive_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _angular_distance(angle_deg: np.ndarray, center_deg: float) -> np.ndarray:
    return (np.asarray(angle_deg, dtype=float) - float(center_deg) + 180.0) % 360.0 - 180.0


def _peak_shape(
    angle_deg: np.ndarray,
    center_deg: float,
    fwhm_deg: float,
    model_name: str,
    eta: float,
) -> np.ndarray:
    delta = _angular_distance(angle_deg, center_deg)
    width = max(float(fwhm_deg), 1e-9)
    if model_name == "gaussian":
        sigma = width / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return np.exp(-0.5 * (delta / sigma) ** 2)
    if model_name == "lorentzian":
        return 1.0 / (1.0 + (2.0 * delta / width) ** 2)
    if model_name == "pseudo_voigt":
        gaussian = _peak_shape(angle_deg, center_deg, width, "gaussian", eta)
        lorentzian = _peak_shape(angle_deg, center_deg, width, "lorentzian", eta)
        return (1.0 - eta) * gaussian + eta * lorentzian
    # Choose kappa from the exact periodic half-height definition, rather
    # than the narrow-peak approximation used by some legacy implementations.
    denominator = 1.0 - math.cos(math.radians(min(width, 360.0) / 2.0))
    kappa = math.log(2.0) / max(denominator, 1e-15)
    return np.exp(kappa * (np.cos(np.deg2rad(delta)) - 1.0))


def evaluate_azimuthal_model(
    angle_deg: Any,
    *,
    baseline: float,
    peaks: tuple[AzimuthalPeak, ...] | list[AzimuthalPeak],
    model: str,
    eta: float = 0.5,
) -> np.ndarray:
    """Evaluate a periodic background-plus-peaks model at arbitrary angles."""

    angles = np.asarray(angle_deg, dtype=float)
    normalized = _normalise_model(model)
    eta_value = float(eta)
    if not np.isfinite(eta_value) or not 0.0 <= eta_value <= 1.0:
        raise ValueError("eta must be finite and between 0 and 1")
    result = np.full(angles.shape, float(baseline), dtype=float)
    for peak in peaks:
        result += peak.amplitude * _peak_shape(
            angles, peak.center_deg, peak.fwhm_deg, normalized,
            peak.eta if peak.eta is not None else eta_value,
        )
    return result


def _detect_peak_seeds(
    profile: AzimuthalProfile,
    *,
    max_peaks: int,
    min_separation_deg: float,
    min_height_fraction: float,
) -> tuple[list[int], float]:
    values = np.asarray(profile.intensity, dtype=float)
    supported = np.isfinite(values) & (np.asarray(profile.counts) > 0)
    indices = np.flatnonzero(supported)
    if indices.size < 3:
        return [], float("nan")
    baseline = float(np.percentile(values[supported], 20.0))
    # Smoothing is used only to place initial centres.  Candidate centres must
    # have observed neighbours, and the optimizer below consumes raw means.
    valid_weight = supported.astype(float)
    smooth_num = gaussian_filter1d(np.where(supported, values, 0.0), 1.0, mode="wrap")
    smooth_den = gaussian_filter1d(valid_weight, 1.0, mode="wrap")
    smooth = np.divide(smooth_num, smooth_den, out=np.full_like(smooth_num, np.nan), where=smooth_den > 1e-8)
    neighbor_support = supported & np.roll(supported, 1) & np.roll(supported, -1)
    maximum = float(np.nanmax(smooth[supported]) - baseline)
    if not np.isfinite(maximum) or maximum <= 0.0:
        return [], baseline
    threshold = baseline + float(min_height_fraction) * maximum
    candidates = [
        int(index)
        for index in range(values.size)
        if neighbor_support[index]
        and np.isfinite(smooth[index])
        and smooth[index] >= smooth[(index - 1) % values.size]
        and smooth[index] > smooth[(index + 1) % values.size]
        and smooth[index] >= threshold
    ]
    candidates.sort(key=lambda index: (-float(smooth[index]), index))
    chosen: list[int] = []
    step_deg = 360.0 / values.size
    for index in candidates:
        if len(chosen) >= max_peaks:
            break
        separated = all(
            abs((index - other + values.size / 2.0) % values.size - values.size / 2.0) * step_deg
            >= min_separation_deg
            for other in chosen
        )
        if separated:
            chosen.append(index)
    return sorted(chosen, key=lambda index: float(profile.angle_deg[index])), baseline


def fit_azimuthal_peaks(
    profile: AzimuthalProfile,
    *,
    model: str = "pseudo_voigt",
    max_peaks: int = 4,
    min_separation_deg: float = 20.0,
    min_height_fraction: float = 0.08,
    initial_fwhm_deg: float = 35.0,
    fwhm_bounds_deg: tuple[float, float] = (2.0, 180.0),
    center_window_deg: float = 45.0,
    eta: float = 0.5,
    max_nfev: int = 5000,
) -> AzimuthalFitResult:
    """Detect and jointly fit periodic peaks to the observed angular bins.

    A robust soft-L1 optimizer limits the effect of isolated intensity
    outliers.  The returned residual is NaN for bins with no measured support.
    Fit parameters are empirical descriptors of this profile, not a 3-D
    structural solution.
    """

    model_name = _normalise_model(model)
    max_peaks = _positive_int(max_peaks, "max_peaks", 1)
    max_nfev = _positive_int(max_nfev, "max_nfev", 1)
    min_separation_deg = float(min_separation_deg)
    min_height_fraction = float(min_height_fraction)
    initial_fwhm_deg = float(initial_fwhm_deg)
    center_window_deg = float(center_window_deg)
    eta = float(eta)
    width_min, width_max = _finite_pair(fwhm_bounds_deg, "fwhm_bounds_deg")
    if width_min <= 0.0 or width_max > 360.0:
        raise ValueError("fwhm_bounds_deg must lie within (0, 360]")
    if not np.isfinite(min_separation_deg) or not 0.0 < min_separation_deg <= 180.0:
        raise ValueError("min_separation_deg must be finite and in (0, 180]")
    if not np.isfinite(min_height_fraction) or not 0.0 <= min_height_fraction <= 1.0:
        raise ValueError("min_height_fraction must be finite and between 0 and 1")
    if not width_min <= initial_fwhm_deg <= width_max:
        raise ValueError("initial_fwhm_deg must be within fwhm_bounds_deg")
    if not np.isfinite(center_window_deg) or not 0.0 < center_window_deg <= 180.0:
        raise ValueError("center_window_deg must be finite and in (0, 180]")
    if not np.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be finite and between 0 and 1")

    x = np.asarray(profile.angle_deg, dtype=float)
    y = np.asarray(profile.intensity, dtype=float)
    counts = np.asarray(profile.counts)
    if x.ndim != 1 or y.shape != x.shape or counts.shape != x.shape or x.size < 8:
        raise ValueError("profile angle, intensity, and counts must be matching 1-D arrays with at least 8 bins")
    observed = np.isfinite(x) & np.isfinite(y) & (counts > 0)
    geometry_counts = np.asarray(profile.geometry_counts)
    coverage_flags: list[str] = []
    if np.any((geometry_counts > 0) & (counts < geometry_counts)):
        coverage_flags.append("incomplete_pixel_coverage")
    if np.any(geometry_counts == 0):
        coverage_flags.append("angular_bins_without_q_support")
    baseline_seed = float(np.percentile(y[observed], 20.0)) if np.any(observed) else float("nan")
    seeds, detected_baseline = _detect_peak_seeds(
        profile,
        max_peaks=max_peaks,
        min_separation_deg=min_separation_deg,
        min_height_fraction=min_height_fraction,
    )
    if np.isfinite(detected_baseline):
        baseline_seed = detected_baseline

    empty_model = np.full(x.shape, baseline_seed if np.isfinite(baseline_seed) else np.nan, dtype=float)
    empty_residual = np.full(x.shape, np.nan, dtype=float)
    if np.any(observed) and np.isfinite(baseline_seed):
        empty_residual[observed] = y[observed] - empty_model[observed]
    settings = {
        "model": model_name,
        "max_peaks": max_peaks,
        "min_separation_deg": min_separation_deg,
        "min_height_fraction": min_height_fraction,
        "initial_fwhm_deg": initial_fwhm_deg,
        "fwhm_bounds_deg": [width_min, width_max],
        "center_window_deg": center_window_deg,
        "eta": eta if model_name == "pseudo_voigt" else None,
        "loss": "soft_l1",
        "observed_bins": int(np.count_nonzero(observed)),
        "weighting": "unweighted angular-bin means",
    }

    if not np.any(observed):
        return AzimuthalFitResult(
            model_name, (), baseline_seed, empty_model, empty_residual, False,
            "No measured angular bins are available in this q annulus.",
            float("nan"), float("nan"), tuple((*coverage_flags, "no_measured_angular_bins")), (), settings,
        )
    if not seeds:
        return AzimuthalFitResult(
            model_name, (), baseline_seed, empty_model, empty_residual, False,
            "No angular peaks met the current height and support criteria.",
            float("nan"), float("nan"), tuple((*coverage_flags, "no_peak_candidates")), (), settings,
        )

    xo = x[observed]
    yo = y[observed]
    n_peaks = len(seeds)
    n_parameters = 1 + 3 * n_peaks
    initial = [baseline_seed]
    lower = [-np.inf]
    upper = [np.inf]
    intensity_range = max(float(np.nanmax(yo) - np.nanmin(yo)), np.finfo(float).eps)
    amp_upper = max(10.0 * intensity_range, float(np.nanmax(np.abs(yo))) * 10.0, 1.0)
    for seed in seeds:
        center = float(x[seed])
        amp = max(float(y[seed] - baseline_seed), intensity_range * 0.05, np.finfo(float).eps)
        initial.extend((center, amp, initial_fwhm_deg))
        lower.extend((center - center_window_deg, 0.0, width_min))
        upper.extend((center + center_window_deg, amp_upper, width_max))
    initial = np.asarray(initial, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    initial = np.minimum(np.maximum(initial, lower_array + np.where(np.isfinite(lower_array), 1e-12, 0.0)),
                         upper_array - np.where(np.isfinite(upper_array), 1e-12, 0.0))

    def evaluate(parameters: np.ndarray, angles: np.ndarray) -> np.ndarray:
        result = np.full(angles.shape, float(parameters[0]), dtype=float)
        for peak_index in range(n_peaks):
            offset = 1 + peak_index * 3
            center, amplitude, width = parameters[offset : offset + 3]
            result += amplitude * _peak_shape(angles, center, width, model_name, eta)
        return result

    flags: list[str] = list(coverage_flags)
    if xo.size < n_parameters + 1:
        flags.append("insufficient_observed_bins_for_fit")
        return AzimuthalFitResult(
            model_name, (), baseline_seed, empty_model, empty_residual, False,
            f"Need at least {n_parameters + 1} observed angular bins for {n_peaks} fitted peaks; found {xo.size}.",
            float("nan"), float("nan"), tuple(flags),
            tuple(float(x[index]) for index in seeds), settings,
        )

    try:
        fit = least_squares(
            lambda parameters: evaluate(parameters, xo) - yo,
            initial,
            bounds=(lower_array, upper_array),
            loss="soft_l1",
            f_scale=max(intensity_range * 0.02, np.finfo(float).eps),
            max_nfev=max_nfev,
        )
        model_values = evaluate(fit.x, x)
        residual = np.full(x.shape, np.nan, dtype=float)
        residual[observed] = y[observed] - model_values[observed]
        ordinary_error = residual[observed]
        rmse = float(np.sqrt(np.mean(ordinary_error**2)))
        total = float(np.sum((yo - np.mean(yo)) ** 2))
        r_squared = float(1.0 - np.sum(ordinary_error**2) / total) if total > np.finfo(float).eps else float("nan")
        peaks: list[AzimuthalPeak] = []
        bound_tolerance = 1e-5
        for peak_index in range(n_peaks):
            offset = 1 + peak_index * 3
            center, amplitude, width = (float(item) for item in fit.x[offset : offset + 3])
            if abs(width - width_min) <= bound_tolerance * max(width_min, 1.0) or abs(width - width_max) <= bound_tolerance * max(width_max, 1.0):
                flags.append(f"peak_{peak_index + 1}_width_at_bound")
            if abs(center - lower_array[offset]) <= bound_tolerance or abs(center - upper_array[offset]) <= bound_tolerance:
                flags.append(f"peak_{peak_index + 1}_center_at_bound")
            kappa = None
            if model_name == "von_mises":
                denominator = 1.0 - math.cos(math.radians(min(width, 360.0) / 2.0))
                kappa = float(math.log(2.0) / max(denominator, 1e-15))
            peaks.append(AzimuthalPeak(
                center_deg=float(center % 360.0),
                amplitude=max(0.0, amplitude),
                fwhm_deg=width,
                eta=eta if model_name == "pseudo_voigt" else None,
                kappa=kappa,
            ))
        for left_index, left in enumerate(peaks):
            for right in peaks[left_index + 1 :]:
                separation = abs((left.center_deg - right.center_deg + 180.0) % 360.0 - 180.0)
                if separation < min_separation_deg:
                    flags.append("fitted_peaks_below_minimum_separation")
                    break
        success = bool(fit.success and np.all(np.isfinite(fit.x)) and np.all(np.isfinite(model_values)))
        if not success:
            flags.append("optimizer_did_not_converge")
        return AzimuthalFitResult(
            model_name=model_name,
            peaks=tuple(peaks),
            baseline=float(fit.x[0]),
            model=model_values,
            residual=residual,
            success=success,
            message=str(fit.message),
            rmse=rmse,
            r_squared=r_squared,
            flags=tuple(dict.fromkeys(flags)),
            initial_centres_deg=tuple(float(x[index]) for index in seeds),
            settings=settings,
        )
    except (ValueError, FloatingPointError, RuntimeError) as exc:
        flags.append("fit_failed")
        return AzimuthalFitResult(
            model_name, (), baseline_seed, empty_model, empty_residual, False,
            f"Peak fit failed: {exc}", float("nan"), float("nan"), tuple(flags),
            tuple(float(x[index]) for index in seeds), settings,
        )


def export_azimuthal_csv_bundle(
    profile: AzimuthalProfile,
    fit: AzimuthalFitResult,
    profile_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write a profile/residual CSV and a component-parameter CSV.

    Existing members of the bundle are preserved unless ``overwrite=True`` is
    explicitly supplied.  Both destinations are checked before either file is
    opened so a collision cannot leave a half-written bundle.
    """

    destination = Path(profile_path)
    if destination.suffix.lower() != ".csv":
        destination = destination.with_suffix(".csv")
    peaks_path = destination.with_name(f"{destination.stem}_peaks.csv")
    if not overwrite and (destination.exists() or peaks_path.exists()):
        raise FileExistsError("one or both azimuthal CSV outputs already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "angle_deg", "intensity", "count", "geometry_count", "coverage",
            "fitted_intensity", "residual", "q_min", "q_max", "q_unit",
            "source", "profile_flags", "fit_model", "fit_success",
        ])
        for index, angle in enumerate(profile.angle_deg):
            writer.writerow([
                _csv_number(angle),
                _csv_number(profile.intensity[index]),
                int(profile.counts[index]),
                int(profile.geometry_counts[index]),
                _csv_number(profile.coverage[index]),
                _csv_number(fit.model[index]) if np.isfinite(fit.model[index]) else "",
                _csv_number(fit.residual[index]) if np.isfinite(fit.residual[index]) else "",
                _csv_number(profile.q_min), _csv_number(profile.q_max), profile.q_unit,
                profile.source or "", ";".join(profile.flags), fit.model_name, int(fit.success),
            ])
    with peaks_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "peak_id", "center_deg", "amplitude", "fwhm_deg", "eta", "kappa",
            "baseline", "rmse", "r_squared", "fit_success", "fit_message", "fit_flags",
            "q_min", "q_max", "q_unit", "source", "fit_model",
        ])
        for index, peak in enumerate(fit.peaks, start=1):
            writer.writerow([
                index, _csv_number(peak.center_deg), _csv_number(peak.amplitude),
                _csv_number(peak.fwhm_deg), _csv_number(peak.eta), _csv_number(peak.kappa),
                _csv_number(fit.baseline), _csv_number(fit.rmse), _csv_number(fit.r_squared),
                int(fit.success), fit.message, ";".join(fit.flags),
                _csv_number(profile.q_min), _csv_number(profile.q_max), profile.q_unit,
                profile.source or "", fit.model_name,
            ])
        if not fit.peaks:
            writer.writerow([
                "", "", "", "", "", "", _csv_number(fit.baseline),
                _csv_number(fit.rmse), _csv_number(fit.r_squared), int(fit.success),
                fit.message, ";".join(fit.flags), _csv_number(profile.q_min),
                _csv_number(profile.q_max), profile.q_unit, profile.source or "", fit.model_name,
            ])
    return destination, peaks_path


def _csv_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return format(number, ".12g") if np.isfinite(number) else ""


__all__ = [
    "AzimuthalFitResult",
    "AzimuthalPeak",
    "AzimuthalProfile",
    "evaluate_azimuthal_model",
    "export_azimuthal_csv_bundle",
    "fit_azimuthal_peaks",
    "measure_azimuthal_profile",
]
