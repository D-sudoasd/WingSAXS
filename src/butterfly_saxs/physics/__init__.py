"""Quantitative 2D SAXS butterfly physics observables, models, and kinetics subpackage."""

from __future__ import annotations

from ._common import unwrap_periodic_angles_deg
from .audit import run_window_sensitivity_audit
from .harmonics import (
    compute_azimuthal_harmonics_and_order,
    evaluate_c2_bridge_four_lobe,
    fit_c2_bridge_four_lobe,
)
from .kinetics import fit_arrhenius_rate, fit_power_law_coarsening
from .radial import (
    _radial_metrics_from_1d_profile as _radial_metrics_from_1d_profile,
    measure_radial_coarsening_observables,
)
from .streak import (
    _fit_1d_transverse_gaussian as _fit_1d_transverse_gaussian,
    measure_perpendicular_streak_width,
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
