"""Measurements for anisotropic two-dimensional SAXS patterns.

This module orchestrates the standard angular, four-lobe, radial/curvature/azimuthal
ridge, and symmetric double-ellipse measurement chain while re-exporting the
public types and measurement helpers from ``_observables_*``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._observables_angular import (
    _circular_supported_runs as _circular_supported_runs,
    _linear_fwhm as _linear_fwhm,
    _periodic_fwhm as _periodic_fwhm,
    _quadratic_peak as _quadratic_peak,
    _refine_symmetric_lobes as _refine_symmetric_lobes,
    _robust_noise as _robust_noise,
    _smooth_supported_angular_profile as _smooth_supported_angular_profile,
    _supported_angular_noise as _supported_angular_noise,
    _wrap_distance as _wrap_distance,
    angular_spectrum,
    apparent_lamellar_tilt,
    azimuthal_profile,
    extract_azimuthal_spectrum,
    find_four_lobe_peaks,
    measure_angular_spectrum,
    measure_four_lobe_peaks,
    measure_four_peaks,
)
from ._observables_ellipse import (
    _assign_symmetric_lobe_branches,
    _ellipse_center_hint,
    _empty_double_ellipse as _empty_double_ellipse,
    _infer_ridge_q_unit as _infer_ridge_q_unit,
    _initial_to_canonical as _initial_to_canonical,
    _quadrant_labels as _quadrant_labels,
    _ridge_components as _ridge_components,
    _ridge_source as _ridge_source,
    _ridge_weights as _ridge_weights,
    _ridge_xy as _ridge_xy,
    _symmetry_diagnostics as _symmetry_diagnostics,
    _write_fit_branch_ids,
    fit_double_ellipse,
    fit_symmetric_double_ellipse,
    fit_symmetric_ellipses,
)
from ._observables_ridges import (
    _CurvatureField as _CurvatureField,
    _annotate_azimuthal_tracks as _annotate_azimuthal_tracks,
    _annotate_ridge_continuity as _annotate_ridge_continuity,
    _azimuthal_peak_ridges as _azimuthal_peak_ridges,
    _continuous_radial_points as _continuous_radial_points,
    _curvature_point_for_sector as _curvature_point_for_sector,
    _measure_lobe_radial_observables,
    _radial_noise_floor as _radial_noise_floor,
    _radial_peak as _radial_peak,
    _radial_peak_candidates as _radial_peak_candidates,
    _radial_profile_baseline_noise as _radial_profile_baseline_noise,
    _reject_ridge_point as _reject_ridge_point,
    _supported_radial_local_maxima as _supported_radial_local_maxima,
    _surface_curvature_field as _surface_curvature_field,
    extract_ridge_points,
    extract_surface_curvature_ridges,
    measure_radial_profile,
    measure_radial_ridges,
    radial_profile,
    radial_ridge,
)
from ._observables_types import (
    APPARENT_FLAGS,
    _SPACING_CENTER_REL_TOL as _SPACING_CENTER_REL_TOL,
    AngularSpectrum,
    DoubleEllipseFit,
    EllipseGeometry,
    LobeMetrics,
    ObservableSet,
    RadialProfile,
    RidgePoint,
    RidgeTrack,
    _MappingResult as _MappingResult,
    _array_field as _array_field,
    _bin_mean as _bin_mean,
    _extract_maps as _extract_maps,
    _flags_with_q_unit,
    _get_field as _get_field,
    _normalized_q_unit as _normalized_q_unit,
    _q_limits as _q_limits,
    _q_to_nm_inverse_scale as _q_to_nm_inverse_scale,
    _q_unit,
    _q_unit_spacing_flag as _q_unit_spacing_flag,
    ellipse_radius,
)
from .cancellation import raise_if_cancelled


def measure_observables(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    n_angular_bins: int = 360,
    n_ridge_angles: int = 72,
    n_radial_bins: int = 256,
    fit_ellipse: bool = True,
    ellipse_parameters: Any = None,
    ellipse_residual: str = "sampson",
    ellipse_multistart: int = 7,
    mask: Any = None,
    ridge_method: str = "radial_peak",
    ridge_snr_threshold: float | None = None,
    ridge_min_peak_fraction: float = 0.0,
    ridge_min_coverage: float = 0.0,
    draw_axis_deg: float = 90.0,
    curvature_sigma: float = 2.0,
    curvature_percentile: float = 25.0,
    curvature_normal_step: float = 1.0,
    p4_quality_thresholds: Mapping[str, Any] | None = None,
    butterfly_options: Mapping[str, Any] | None = None,
    cancel_event: Any = None,
) -> ObservableSet:
    """Run the standard angular, lobe, ridge, and ellipse measurement chain."""

    raise_if_cancelled(cancel_event, "observables:start")
    if str(ridge_method).replace("-", "_") == "butterfly_curvature":
        from .butterfly import measure_butterfly_observables

        return measure_butterfly_observables(
            frame, qmap, q_window, mask=mask, options=butterfly_options,
            draw_axis_deg=draw_axis_deg, ellipse_parameters=ellipse_parameters,
            multistart=ellipse_multistart, fit_ellipse=fit_ellipse,
            n_angular_bins=n_angular_bins, n_radial_bins=n_radial_bins,
            snr_threshold=ridge_snr_threshold, cancel_event=cancel_event,
        )
    q_unit = _q_unit(qmap)
    angular = measure_angular_spectrum(frame, qmap, q_window, n_bins=n_angular_bins, mask=mask)
    lobes = measure_four_lobe_peaks(
        angular,
        symmetric_refine=True,
        reference_axis_deg=float(draw_axis_deg) - 90.0,
    )
    ridge = measure_radial_ridges(
        frame,
        qmap,
        q_window,
        n_angles=n_ridge_angles,
        n_bins=n_radial_bins,
        lobe_metrics=lobes,
        mask=mask,
        ridge_method=ridge_method,
        ridge_snr_threshold=ridge_snr_threshold,
        ridge_min_peak_fraction=ridge_min_peak_fraction,
        ridge_min_coverage=ridge_min_coverage,
        curvature_sigma=curvature_sigma,
        curvature_percentile=curvature_percentile,
        curvature_normal_step=curvature_normal_step,
        cancel_event=cancel_event,
    )
    lobe_radial_profiles, lobe_radial_peaks = _measure_lobe_radial_observables(
        frame,
        qmap,
        q_window,
        lobes,
        n_radial_bins=n_radial_bins,
        # Keep the shared ridge threshold as the acceptance contract for the
        # companion radial reflection.  A finite q_star is therefore never
        # silently retained below the user-selected SNR floor.
        snr_threshold=(
            float(ridge_snr_threshold)
            if ridge_snr_threshold is not None
            else 2.0
        ),
        min_coverage=ridge_min_coverage,
        mask=mask,
        cancel_event=cancel_event,
    )
    # Azimuthal maxima carry identities from observed q-wise tracks.  Lobe
    # centres are a separate angular summary and must not overwrite those
    # branch IDs (or create an implied symmetric counterpart).
    if str(ridge_method).lower().replace("-", "_") != "azimuthal_peak":
        center_hint, _ = _ellipse_center_hint(ellipse_parameters)
        _assign_symmetric_lobe_branches(
            ridge,
            lobes,
            reference_axis_deg=float(draw_axis_deg) - 90.0,
            center=center_hint,
        )
    ellipse = (
        fit_symmetric_double_ellipse(
            ridge,
            parameters=ellipse_parameters,
            residual=ellipse_residual,
            multistart=ellipse_multistart,
            reference_axis_deg=float(draw_axis_deg) - 90.0,
            q_unit=q_unit,
            strict_symmetry=True,
            cancel_event=cancel_event,
        )
        if fit_ellipse
        else None
    )
    if ellipse is not None:
        _write_fit_branch_ids(ridge, ellipse)
        from .p4_quality import evaluate_p4_ellipse_quality

        ellipse.quality = evaluate_p4_ellipse_quality(
            ridge,
            ellipse,
            thresholds=p4_quality_thresholds,
        )
    phi_app_deg, phi_app_std_deg = apparent_lamellar_tilt(lobes, draw_axis_deg=draw_axis_deg)
    return ObservableSet(
        angular=angular,
        lobes=lobes,
        ridge=ridge,
        ellipse=ellipse,
        phi_app_deg=phi_app_deg,
        phi_app_std_deg=phi_app_std_deg,
        phi_app_mad_deg=phi_app_std_deg,
        draw_axis_deg=float(draw_axis_deg),
        flags=_flags_with_q_unit(APPARENT_FLAGS, q_unit),
        q_unit=q_unit,
        lobe_radial_profiles=lobe_radial_profiles,
        lobe_radial_peaks=lobe_radial_peaks,
    )


__all__ = [
    "APPARENT_FLAGS",
    "AngularSpectrum",
    "LobeMetrics",
    "RadialProfile",
    "RidgePoint",
    "RidgeTrack",
    "EllipseGeometry",
    "DoubleEllipseFit",
    "ObservableSet",
    "measure_angular_spectrum",
    "angular_spectrum",
    "azimuthal_profile",
    "extract_azimuthal_spectrum",
    "measure_four_lobe_peaks",
    "find_four_lobe_peaks",
    "measure_four_peaks",
    "apparent_lamellar_tilt",
    "measure_radial_profile",
    "radial_profile",
    "measure_radial_ridges",
    "radial_ridge",
    "extract_ridge_points",
    "extract_surface_curvature_ridges",
    "ellipse_radius",
    "fit_symmetric_double_ellipse",
    "fit_symmetric_ellipses",
    "fit_double_ellipse",
    "measure_observables",
]
