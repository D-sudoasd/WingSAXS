"""Sector-resolved low-q analysis measured directly from 2-D SAXS images.

The sector profile contains means of observed image pixels only. Empty or
masked q bins remain missing; the empirical fits do not fill detector gaps or
establish a unique structural mechanism.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .settings import canonical_q_unit


@dataclass(frozen=True, slots=True)
class Density2DProfile:
    """Measured radial intensity means in one angular sector of a 2-D image."""

    q: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    geometry_counts: np.ndarray
    coverage: np.ndarray
    q_min: float
    q_max: float
    azimuth_center_deg: float
    azimuth_half_width_deg: float
    q_unit: str = "unknown"
    source: str | None = None
    flags: tuple[str, ...] = ()

    @property
    def n_supported_bins(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.intensity) & (self.counts > 0)))


@dataclass(frozen=True, slots=True)
class Density2DFit:
    """One empirical candidate fit to measured bins in a sector profile."""

    model_name: str
    success: bool
    message: str
    parameters: Mapping[str, float | str]
    fitted_intensity: np.ndarray
    residual: np.ndarray
    n_points: int
    q_min: float | None = None
    q_max: float | None = None
    r_squared: float | None = None
    fit_space: str = "intensity"
    rmse: float | None = None
    rss: float | None = None
    aic: float | None = None
    flags: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return "candidate" if self.success else "failed"


@dataclass(frozen=True, slots=True)
class Density2DResult:
    """The measured sector profile and all requested model candidates."""

    profile: Density2DProfile
    fits: tuple[Density2DFit, ...]
    flags: tuple[str, ...] = ("empirical_models_do_not_determine_unique_structure",)


def _finite_pair(value: Any, name: str) -> tuple[float, float]:
    try:
        pair = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain two finite numbers") from exc
    if len(pair) != 2 or not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain two finite numbers")
    if pair[0] >= pair[1]:
        raise ValueError(f"{name} must be increasing")
    return pair


def _positive_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return parsed


def _circular_distance_deg(angle: np.ndarray, center: float) -> np.ndarray:
    return (angle - center + 180.0) % 360.0 - 180.0


def measure_density2d_profile(
    data: Any,
    *,
    qx: Any,
    qy: Any,
    q_window: Any,
    azimuth_center_deg: float = 0.0,
    azimuth_half_width_deg: float = 18.0,
    n_q: int = 120,
    valid_mask: Any = None,
    q_unit: str = "unknown",
    source: Any = None,
) -> Density2DProfile:
    """Measure the mean intensity in linear q bins inside a selected sector.

    Angles are measured counter-clockwise in q space: 0 degrees is +qx and
    90 degrees is +qy. ``valid_mask=True`` denotes a usable detector pixel.
    """

    image = np.asarray(data, dtype=float)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("data must be a non-empty two-dimensional image")
    q_min, q_max = _finite_pair(q_window, "q_window")
    if q_min < 0.0:
        raise ValueError("q_window lower bound must be non-negative")
    n_q = _positive_int(n_q, "n_q", 4)
    center = float(azimuth_center_deg)
    half_width = float(azimuth_half_width_deg)
    if not np.isfinite(center):
        raise ValueError("azimuth_center_deg must be finite")
    if not np.isfinite(half_width) or not 0.0 < half_width <= 180.0:
        raise ValueError("azimuth_half_width_deg must be finite and in (0, 180]")

    try:
        qx_map, qy_map = np.broadcast_arrays(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
        qx_map = np.broadcast_to(qx_map, image.shape)
        qy_map = np.broadcast_to(qy_map, image.shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("qx and qy must broadcast to the image shape") from exc
    if valid_mask is None:
        usable = np.ones(image.shape, dtype=bool)
    else:
        try:
            usable = np.broadcast_to(np.asarray(valid_mask, dtype=bool), image.shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("valid_mask must broadcast to the image shape") from exc

    q_radius = np.hypot(qx_map, qy_map)
    angle = np.mod(np.degrees(np.arctan2(qy_map, qx_map)), 360.0)
    finite_geometry = np.isfinite(qx_map) & np.isfinite(qy_map) & np.isfinite(q_radius)
    in_window = (q_radius >= q_min) & (q_radius <= q_max)
    in_sector = np.abs(_circular_distance_deg(angle, center % 360.0)) <= half_width
    zero_q_in_window = finite_geometry & (q_radius == 0.0) & (q_min <= 0.0 <= q_max)
    # q=0 has no defined azimuth, so assigning it to one selected sector would
    # introduce an orientation-dependent direct-beam contribution.
    geometry = finite_geometry & (q_radius > 0.0) & in_window & in_sector
    if not np.any(geometry):
        raise ValueError("selected q window and angular sector contain no finite q-map pixels")

    edges = np.linspace(q_min, q_max, n_q + 1, dtype=float)
    q_centers = 0.5 * (edges[:-1] + edges[1:])
    radial_bin = np.searchsorted(edges, q_radius[geometry], side="right") - 1
    radial_bin = np.clip(radial_bin, 0, n_q - 1)
    geometry_counts = np.bincount(radial_bin, minlength=n_q).astype(np.int64)

    selected_rows, selected_cols = np.nonzero(geometry)
    measured_values = image[selected_rows, selected_cols]
    measured = usable[selected_rows, selected_cols] & np.isfinite(measured_values)
    counts = np.bincount(radial_bin[measured], minlength=n_q).astype(np.int64)
    sums = np.bincount(
        radial_bin[measured], weights=measured_values[measured], minlength=n_q
    ).astype(float)
    intensity = np.full(n_q, np.nan, dtype=float)
    np.divide(sums, counts, out=intensity, where=counts > 0)
    coverage = np.divide(
        counts,
        geometry_counts,
        out=np.zeros(n_q, dtype=float),
        where=geometry_counts > 0,
    )

    flags: list[str] = []
    if np.any(zero_q_in_window):
        flags.append("zero_q_pixels_excluded_due_to_undefined_azimuth")
    if np.any(~usable[selected_rows, selected_cols]):
        flags.append("masked_pixels_in_selected_sector")
    if np.any(~np.isfinite(measured_values)):
        flags.append("nonfinite_intensity_in_selected_sector")
    if not np.any(measured):
        flags.append("no_measured_pixels_in_selected_sector")
    return Density2DProfile(
        q=q_centers,
        intensity=intensity,
        counts=counts,
        geometry_counts=geometry_counts,
        coverage=coverage,
        q_min=q_min,
        q_max=q_max,
        azimuth_center_deg=center % 360.0,
        azimuth_half_width_deg=half_width,
        q_unit=str(q_unit or "unknown"),
        source=None if source is None else str(source),
        flags=tuple(flags),
    )


def evaluate_density2d_model(
    q: Any,
    model: str,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    """Evaluate one supported empirical model at arbitrary q values."""

    q_values = np.asarray(q, dtype=float)
    if model == "power_law":
        prefactor = float(parameters["prefactor"])
        alpha = float(parameters["alpha"])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return prefactor * np.power(q_values, -alpha)
    if model == "ornstein_zernike":
        amplitude = float(parameters["I0"])
        background = float(parameters["background"])
        xi = float(parameters["_xi_q_inverse"])
        return amplitude / (1.0 + np.square(q_values * xi)) + background
    raise ValueError("model must be 'power_law' or 'ornstein_zernike'")


def _empty_fit(
    profile: Density2DProfile,
    model: str,
    message: str,
    flags: tuple[str, ...],
    *,
    n_points: int = 0,
    q: np.ndarray | None = None,
) -> Density2DFit:
    empty = np.full(np.asarray(profile.q).shape, np.nan, dtype=float)
    q_values = np.asarray(q if q is not None else [], dtype=float)
    q_values = q_values[np.isfinite(q_values)]
    return Density2DFit(
        model_name=model,
        success=False,
        message=message,
        parameters={},
        fitted_intensity=empty,
        residual=empty.copy(),
        n_points=int(n_points),
        q_min=float(np.min(q_values)) if q_values.size else None,
        q_max=float(np.max(q_values)) if q_values.size else None,
        flags=flags,
    )


def _goodness(observed: np.ndarray, predicted: np.ndarray, k: int) -> tuple[float, float, float, float]:
    residual = observed - predicted
    rss = max(float(np.sum(np.square(residual))), np.finfo(float).tiny)
    centered = observed - float(np.mean(observed))
    total = float(np.sum(np.square(centered)))
    r_squared = 1.0 if total == 0.0 and rss <= np.finfo(float).tiny else (0.0 if total == 0.0 else 1.0 - rss / total)
    rmse = float(math.sqrt(rss / observed.size))
    aic = float(observed.size * math.log(rss / observed.size) + 2 * k)
    return rss, r_squared, rmse, aic


def _fit_power_law(profile: Density2DProfile, minimum: int) -> Density2DFit:
    q_all = np.asarray(profile.q, dtype=float)
    intensity_all = np.asarray(profile.intensity, dtype=float)
    use = np.isfinite(q_all) & (q_all > 0.0) & np.isfinite(intensity_all) & (intensity_all > 0.0)
    q, intensity = q_all[use], intensity_all[use]
    flags: list[str] = []
    if np.count_nonzero(np.isfinite(q_all) & np.isfinite(intensity_all)) > q.size:
        flags.append("nonpositive_q_or_intensity_excluded")
    if q.size < minimum:
        return _empty_fit(
            profile, "power_law", f"at least {minimum} positive measured bins are required",
            tuple(flags + ["insufficient_points"]), n_points=int(q.size), q=q,
        )
    if np.unique(q).size < 2:
        return _empty_fit(
            profile, "power_law", "measured q values do not span a range",
            tuple(flags + ["insufficient_q_range"]), n_points=int(q.size), q=q,
        )

    log_q = np.log(q)
    log_i = np.log(intensity)
    slope, intercept = np.polyfit(log_q, log_i, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return _empty_fit(
            profile, "power_law", "log-space linear fit returned non-finite parameters",
            tuple(flags + ["fit_failed"]), n_points=int(q.size), q=q,
        )
    alpha = float(-slope)
    prefactor = float(math.exp(float(np.clip(intercept, -700.0, 700.0))))
    parameters: dict[str, float | str] = {
        "prefactor": prefactor,
        "alpha": alpha,
        "intercept_log": float(intercept),
    }
    fitted_valid = evaluate_density2d_model(q, "power_law", parameters)
    if not np.all(np.isfinite(fitted_valid)):
        return _empty_fit(
            profile, "power_law", "power-law predictions are non-finite",
            tuple(flags + ["fit_failed"]), n_points=int(q.size), q=q,
        )
    fitted = np.full(q_all.shape, np.nan, dtype=float)
    residual = np.full(q_all.shape, np.nan, dtype=float)
    fitted[use] = fitted_valid
    residual[use] = intensity - fitted_valid
    log_predicted = np.log(fitted_valid)
    log_rss, log_r2, _, log_aic = _goodness(log_i, log_predicted, 2)
    linear_rss, _, linear_rmse, _ = _goodness(intensity, fitted_valid, 2)
    q_low, q_high = float(np.min(q)), float(np.max(q))
    log_decades = float(math.log10(q_high / q_low)) if q_low > 0.0 else 0.0
    if log_decades < 0.5:
        flags.append("less_than_half_log_decade")
    flags.append("empirical_power_law_exponent_only")
    parameters.update({"log_decades": log_decades, "rss_log_intensity": log_rss, "rss_intensity": linear_rss})
    return Density2DFit(
        model_name="power_law",
        success=True,
        message="power law fitted to positive measured bins in log-log space",
        parameters=parameters,
        fitted_intensity=fitted,
        residual=residual,
        n_points=int(q.size),
        q_min=q_low,
        q_max=q_high,
        r_squared=log_r2,
        fit_space="log_intensity",
        rmse=linear_rmse,
        rss=linear_rss,
        aic=log_aic,
        flags=tuple(flags),
    )


def _fit_ornstein_zernike(profile: Density2DProfile, minimum: int) -> Density2DFit:
    q_all = np.asarray(profile.q, dtype=float)
    intensity_all = np.asarray(profile.intensity, dtype=float)
    use = np.isfinite(q_all) & (q_all > 0.0) & np.isfinite(intensity_all)
    q, intensity = q_all[use], intensity_all[use]
    if q.size < minimum:
        return _empty_fit(
            profile, "ornstein_zernike", f"at least {minimum} finite measured bins are required",
            ("insufficient_points",), n_points=int(q.size), q=q,
        )
    if np.unique(q).size < 3:
        return _empty_fit(
            profile, "ornstein_zernike", "measured q values do not span a range",
            ("insufficient_q_range",), n_points=int(q.size), q=q,
        )

    q_reference = float(np.median(q))
    intensity_scale = max(float(np.ptp(intensity)), float(np.max(np.abs(intensity))) * 1e-8, 1e-12)
    amplitude0 = max(float(np.max(intensity) - np.min(intensity)), intensity_scale * 0.1, 1e-12)
    background0 = float(np.min(intensity))

    # Fit q_ref * xi as a positive dimensionless parameter. This keeps the
    # optimizer well-scaled for both nm^-1 and Angstrom^-1 coordinate maps.
    def predict(free: np.ndarray) -> np.ndarray:
        amplitude, q_reference_xi, background = free
        ratio = q / q_reference
        return amplitude / (1.0 + np.square(ratio * q_reference_xi)) + background

    try:
        from scipy.optimize import least_squares

        fit = least_squares(
            lambda free: (predict(free) - intensity) / intensity_scale,
            x0=np.asarray([amplitude0, 1.0, background0], dtype=float),
            bounds=(np.asarray([0.0, 1e-10, -np.inf]), np.asarray([np.inf, 1e10, np.inf])),
            max_nfev=10000,
            method="trf",
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return _empty_fit(
            profile, "ornstein_zernike", f"fit failed: {type(exc).__name__}: {exc}",
            ("fit_failed",), n_points=int(q.size), q=q,
        )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return _empty_fit(
            profile, "ornstein_zernike", str(fit.message),
            ("fit_failed",), n_points=int(q.size), q=q,
        )

    amplitude, q_reference_xi, background = (float(value) for value in fit.x)
    xi_q_inverse = q_reference_xi / q_reference
    parameters: dict[str, float | str] = {
        "I0": amplitude,
        "background": background,
        "q_reference": q_reference,
        "q_reference_times_xi": q_reference_xi,
        "_xi_q_inverse": xi_q_inverse,
    }
    canonical_unit = canonical_q_unit(profile.q_unit)
    diagnostics: list[str] = []
    q_xi = q * xi_q_inverse
    if float(np.max(q_xi)) < 1.0:
        diagnostics.append("oz_turnover_not_bracketed_at_higher_q")
    elif float(np.min(q_xi)) > 1.0:
        diagnostics.append("oz_turnover_not_bracketed_at_lower_q")

    jacobian = np.asarray(fit.jac, dtype=float)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.size:
        tolerance = float(np.max(singular_values)) * max(jacobian.shape) * np.finfo(float).eps
        jacobian_rank = int(np.count_nonzero(singular_values > tolerance))
        condition = (
            float(np.max(singular_values) / np.min(singular_values))
            if np.min(singular_values) > 0.0 else math.inf
        )
    else:
        jacobian_rank, condition = 0, math.inf
    parameters["jacobian_rank"] = float(jacobian_rank)
    parameters["jacobian_condition"] = condition
    if jacobian_rank < 3 or condition > 1e10:
        diagnostics.append("oz_parameters_weakly_identified_jacobian")
    amplitude_tolerance = max(float(np.max(np.abs(intensity))) * 1e-8, 1e-12)
    if amplitude <= amplitude_tolerance:
        diagnostics.append("oz_amplitude_near_zero_xi_unidentified")

    if canonical_unit == "nm⁻¹":
        parameters["xi_candidate"] = xi_q_inverse
        parameters["xi_unit"] = "nm"
    elif canonical_unit == "Å⁻¹":
        parameters["xi_candidate"] = xi_q_inverse
        parameters["xi_unit"] = "Å"
    # Only expose xi as an identified fit parameter when the finite q range
    # samples the turnover and the fit has nonzero curvature information.
    if not diagnostics and "xi_candidate" in parameters:
        parameters["xi"] = parameters["xi_candidate"]
    elif "xi_candidate" in parameters:
        parameters["xi_status"] = "weakly_identified_candidate"
    predicted_valid = predict(fit.x)
    if not np.all(np.isfinite(predicted_valid)):
        return _empty_fit(
            profile, "ornstein_zernike", "Ornstein-Zernike predictions are non-finite",
            ("fit_failed",), n_points=int(q.size), q=q,
        )
    fitted = np.full(q_all.shape, np.nan, dtype=float)
    residual = np.full(q_all.shape, np.nan, dtype=float)
    fitted[use] = predicted_valid
    residual[use] = intensity - predicted_valid
    rss, r_squared, rmse, aic = _goodness(intensity, predicted_valid, 3)
    flags: list[str] = ["correlation_length_model_dependent", *diagnostics]
    if "xi_candidate" not in parameters:
        flags.append("physical_correlation_length_unavailable_without_calibrated_q")
    if q_reference_xi <= 1.01e-10 or q_reference_xi >= 0.99e10:
        flags.append("q_xi_at_optimizer_bound")
    parameters.update({"rss": rss, "aic": aic})
    return Density2DFit(
        model_name="ornstein_zernike",
        success=True,
        message="Ornstein-Zernike intensity model fitted to finite measured bins",
        parameters=parameters,
        fitted_intensity=fitted,
        residual=residual,
        n_points=int(q.size),
        q_min=float(np.min(q)),
        q_max=float(np.max(q)),
        r_squared=r_squared,
        fit_space="intensity",
        rmse=rmse,
        rss=rss,
        aic=aic,
        flags=tuple(flags),
    )


def fit_density_fluctuation(
    profile: Density2DProfile,
    *,
    models: tuple[str, ...] | list[str] = ("power_law", "ornstein_zernike"),
    min_power_law_points: int = 3,
    min_ornstein_zernike_points: int = 4,
) -> tuple[Density2DFit, ...]:
    """Fit requested empirical candidates independently to measured bins."""

    if not isinstance(profile, Density2DProfile):
        raise TypeError("profile must be a Density2DProfile")
    minimum_power = _positive_int(min_power_law_points, "min_power_law_points", 3)
    minimum_oz = _positive_int(min_ornstein_zernike_points, "min_ornstein_zernike_points", 4)
    selected = tuple(str(model).strip().lower() for model in models)
    if not selected:
        raise ValueError("at least one density model must be selected")
    if len(set(selected)) != len(selected) or any(model not in {"power_law", "ornstein_zernike"} for model in selected):
        raise ValueError("models may contain 'power_law' and/or 'ornstein_zernike' once each")
    if profile.n_supported_bins == 0:
        return tuple(
            _empty_fit(profile, model, "no measured intensity bins are available", ("no_measured_bins",))
            for model in selected
        )
    functions = {
        "power_law": lambda: _fit_power_law(profile, minimum_power),
        "ornstein_zernike": lambda: _fit_ornstein_zernike(profile, minimum_oz),
    }
    return tuple(functions[model]() for model in selected)


def analyze_density2d(
    data: Any,
    *,
    qx: Any,
    qy: Any,
    q_window: Any,
    azimuth_center_deg: float = 0.0,
    azimuth_half_width_deg: float = 18.0,
    n_q: int = 120,
    valid_mask: Any = None,
    q_unit: str = "unknown",
    source: Any = None,
    models: tuple[str, ...] | list[str] = ("power_law", "ornstein_zernike"),
) -> Density2DResult:
    """Measure one 2-D q sector and fit the selected empirical models."""

    profile = measure_density2d_profile(
        data,
        qx=qx,
        qy=qy,
        q_window=q_window,
        azimuth_center_deg=azimuth_center_deg,
        azimuth_half_width_deg=azimuth_half_width_deg,
        n_q=n_q,
        valid_mask=valid_mask,
        q_unit=q_unit,
        source=source,
    )
    fits = fit_density_fluctuation(profile, models=models)
    flags = ["empirical_models_do_not_determine_unique_structure"]
    if profile.flags:
        flags.extend(profile.flags)
    if any(not fit.success for fit in fits):
        flags.append("one_or_more_model_fits_failed")
    return Density2DResult(profile, fits, tuple(flags))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    return str(value)


def export_density2d_bundle(
    profile: Density2DProfile,
    fits: tuple[Density2DFit, ...] | list[Density2DFit],
    csv_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write sector evidence as a strict JSON record and a wide CSV table."""

    target = Path(csv_path)
    if target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv")
    json_target = target.with_suffix(".json")
    targets = (target, json_target)
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("one or both density 2-D outputs already exist")
    if any(path.exists() and path.is_dir() for path in targets):
        raise IsADirectoryError("an export output path is a directory")

    fit_by_name = {fit.model_name: fit for fit in fits}
    fields = [
        "q", "q_unit", "intensity", "counts", "geometry_counts", "coverage",
        "azimuth_center_deg", "azimuth_half_width_deg", "source",
    ]
    for model_name in ("power_law", "ornstein_zernike"):
        fields.extend((f"{model_name}_status", f"{model_name}_fitted_intensity", f"{model_name}_residual"))
    csv_records: list[dict[str, Any]] = []
    for index, q_value in enumerate(np.asarray(profile.q, dtype=float)):
        row: dict[str, Any] = {
            "q": _csv_value(q_value),
            "q_unit": profile.q_unit,
            "intensity": _csv_value(profile.intensity[index]),
            "counts": int(profile.counts[index]),
            "geometry_counts": int(profile.geometry_counts[index]),
            "coverage": _csv_value(profile.coverage[index]),
            "azimuth_center_deg": profile.azimuth_center_deg,
            "azimuth_half_width_deg": profile.azimuth_half_width_deg,
            "source": profile.source or "",
        }
        for model_name in ("power_law", "ornstein_zernike"):
            fit = fit_by_name.get(model_name)
            row[f"{model_name}_status"] = fit.status if fit else "not_requested"
            row[f"{model_name}_fitted_intensity"] = _csv_value(fit.fitted_intensity[index]) if fit else ""
            row[f"{model_name}_residual"] = _csv_value(fit.residual[index]) if fit else ""
        csv_records.append(row)

    payload = {
        "schema": "butterfly_saxs.density2d.v1",
        "scope": "2-D image sector; no standalone 1-D input",
        "interpretation": "Empirical power-law and Ornstein-Zernike candidates do not determine a unique structural mechanism.",
        "profile": {
            "q": profile.q,
            "intensity": profile.intensity,
            "counts": profile.counts,
            "geometry_counts": profile.geometry_counts,
            "coverage": profile.coverage,
            "q_window": [profile.q_min, profile.q_max],
            "q_unit": profile.q_unit,
            "azimuth_center_deg": profile.azimuth_center_deg,
            "azimuth_half_width_deg": profile.azimuth_half_width_deg,
            "source": profile.source,
            "flags": profile.flags,
        },
        "fits": [
            {
                "model": fit.model_name,
                "status": fit.status,
                "success": fit.success,
                "message": fit.message,
                "parameters": fit.parameters,
                "q_min": fit.q_min,
                "q_max": fit.q_max,
                "n_points": fit.n_points,
                "fit_space": fit.fit_space,
                "r_squared": fit.r_squared,
                "rmse": fit.rmse,
                "rss": fit.rss,
                "aic": fit.aic,
                "fitted_intensity": fit.fitted_intensity,
                "residual": fit.residual,
                "flags": fit.flags,
            }
            for fit in fits
        ],
    }

    target.parent.mkdir(parents=False, exist_ok=True)
    temporary_paths: list[Path] = []
    reservations: list[Path] = []
    try:
        for output in targets:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", delete=False, dir=target.parent,
                prefix=f".{output.stem}.", suffix=".tmp",
            ) as handle:
                temporary_paths.append(Path(handle.name))
        with temporary_paths[0].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_records)
        with temporary_paths[1].open("w", encoding="utf-8", newline="") as stream:
            json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

        if overwrite:
            for temporary, output in zip(temporary_paths, targets):
                os.replace(temporary, output)
        else:
            try:
                for output in targets:
                    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
                    os.close(descriptor)
                    reservations.append(output)
            except FileExistsError:
                raise FileExistsError("one or both density 2-D outputs already exist") from None
            for temporary, output in zip(temporary_paths, targets):
                os.replace(temporary, output)
                if output in reservations:
                    reservations.remove(output)
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        for reserved in reservations:
            try:
                reserved.unlink(missing_ok=True)
            except OSError:
                pass
    return targets


def _csv_value(value: Any) -> float | str:
    number = float(value)
    return number if np.isfinite(number) else ""


__all__ = [
    "Density2DFit",
    "Density2DProfile",
    "Density2DResult",
    "analyze_density2d",
    "evaluate_density2d_model",
    "export_density2d_bundle",
    "fit_density_fluctuation",
    "measure_density2d_profile",
]
