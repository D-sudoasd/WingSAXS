"""Compatibility facade re-exporting ``butterfly_saxs.physics`` public and tested APIs."""

from __future__ import annotations

from .physics import (
    _fit_1d_transverse_gaussian as _fit_1d_transverse_gaussian,
    _radial_metrics_from_1d_profile as _radial_metrics_from_1d_profile,
    compute_azimuthal_harmonics_and_order,
    evaluate_c2_bridge_four_lobe,
    fit_arrhenius_rate,
    fit_c2_bridge_four_lobe,
    fit_power_law_coarsening,
    measure_perpendicular_streak_width,
    measure_radial_coarsening_observables,
    run_window_sensitivity_audit,
    unwrap_periodic_angles_deg,
)
from .physics._common import (
    _FWHM_PER_SIGMA as _FWHM_PER_SIGMA,
    _GAS_CONSTANT_KJ_MOL_K as _GAS_CONSTANT_KJ_MOL_K,
    _KJ_MOL_PER_EV as _KJ_MOL_PER_EV,
    _bin_azimuthal_from_2d as _bin_azimuthal_from_2d,
    _finite as _finite,
    _q_unit_info as _q_unit_info,
    _resolve_2d_context as _resolve_2d_context,
    _validate_q_window as _validate_q_window,
)

__all__ = [
    "compute_azimuthal_harmonics_and_order",
    "evaluate_c2_bridge_four_lobe",
    "fit_arrhenius_rate",
    "fit_c2_bridge_four_lobe",
    "fit_power_law_coarsening",
    "measure_perpendicular_streak_width",
    "measure_radial_coarsening_observables",
    "run_window_sensitivity_audit",
    "unwrap_periodic_angles_deg",
]
