"""Backward-compatible re-export shim for ``butterfly_saxs.lamellar.adapter``."""

from __future__ import annotations

from .lamellar.adapter import (
    _direction_for_group,
    _ellipse_components,
    _ellipse_parameter_records,
    _ellipse_payload,
    _exact_integer,
    _execution_failure,
    _group_populations,
    _metadata_base,
    _parameter_value,
    _period_record,
    _q_unit_from_source,
    _radial_peak_direction_support,
    _radial_peaks,
    _source_identity,
    _source_mapping,
    _stable_seed,
    observed_direction_support,
)

__all__ = [
    "observed_direction_support",
    "_direction_for_group",
    "_ellipse_components",
    "_ellipse_parameter_records",
    "_ellipse_payload",
    "_exact_integer",
    "_execution_failure",
    "_group_populations",
    "_metadata_base",
    "_parameter_value",
    "_period_record",
    "_q_unit_from_source",
    "_radial_peak_direction_support",
    "_radial_peaks",
    "_source_identity",
    "_source_mapping",
    "_stable_seed",
]
