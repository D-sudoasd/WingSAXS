"""Scoring helpers for repeated-trial uncertainty validation.

Coverage is reported both conditional on an available interval and over every
attempted trial.  Missing intervals remain in the all-trial denominator; finite
candidate values are summarized independently of solver or quality status.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


DEFAULT_PARAMETERS = ("a", "b", "axis_ratio")


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _quantile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "minimum": None, "maximum": None}
    numeric = np.asarray(values, dtype=float)
    return {
        "n": int(numeric.size),
        "median": float(np.median(numeric)),
        "minimum": float(np.min(numeric)),
        "maximum": float(np.max(numeric)),
    }


def _interval_bounds(value: Any) -> tuple[float | None, float | None]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None, None
    try:
        array = np.asarray(value, dtype=object)
    except Exception:
        return None, None
    if array.ndim != 1 or array.size != 2:
        return None, None
    low, high = (_finite_float(array[0]), _finite_float(array[1]))
    if low is None or high is None or high < low:
        return None, None
    return low, high


def _wilson_95(successes: int, trials: int) -> list[float] | None:
    if trials <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [max(0.0, float(center - half_width)), min(1.0, float(center + half_width))]


def summarize_uncertainty_trials(
    records: Sequence[Mapping[str, Any]],
    *,
    parameters: Sequence[str] = DEFAULT_PARAMETERS,
) -> dict[str, Any]:
    """Summarize candidate error, interval coverage and identifiability signals.

    Each row may contain ``truth``, ``candidate``, ``intervals``,
    ``parameter_status``, ``condition`` and ``execution_status`` mappings or
    scalar fields.  Finite candidates are counted even when ``solver_success``
    is false or a parameter status is ``candidate``.  A missing, non-finite,
    malformed or reversed interval is unavailable and still counts as a trial
    without coverage in the all-trial denominator.
    """

    rows = tuple(_mapping(row) for row in records)
    execution_counts = Counter(str(row.get("execution_status", "unspecified")) for row in rows)
    solver_counts = Counter(
        "unspecified" if row.get("solver_success") is None else str(bool(row.get("solver_success")))
        for row in rows
    )
    conditions = [
        number
        for row in rows
        for number in [_finite_float(row.get("condition", _mapping(row.get("candidate")).get("condition")))]
        if number is not None and number >= 0.0
    ]

    parameter_summaries: dict[str, Any] = {}
    for parameter in parameters:
        finite_candidates = 0
        missing_candidates = 0
        error_values: list[float] = []
        interval_available = 0
        interval_contains = 0
        reference_truth_missing = 0
        interval_bounds_missing = 0
        interval_widths: list[float] = []
        relative_widths: list[float] = []
        candidate_statuses: Counter[str] = Counter()
        confidence_statuses: Counter[str] = Counter()

        for row in rows:
            truth_map = _mapping(row.get("truth"))
            candidate_map = _mapping(row.get("candidate"))
            truth = _finite_float(truth_map.get(parameter))
            candidate = _finite_float(candidate_map.get(parameter))
            if candidate is None:
                missing_candidates += 1
            else:
                finite_candidates += 1
                if truth is not None:
                    error_values.append(candidate - truth)

            parameter_status = _mapping(row.get("parameter_status"))
            status = parameter_status.get(parameter)
            if status is not None:
                candidate_statuses[str(status)] += 1
            confidence = _mapping(row.get("confidence")).get(parameter)
            if confidence is not None:
                confidence_statuses[str(confidence)] += 1

            interval_map = _mapping(row.get("intervals"))
            raw_interval = interval_map.get(parameter)
            low, high = _interval_bounds(raw_interval)
            if truth is None:
                reference_truth_missing += 1
                continue
            if low is None or high is None:
                interval_bounds_missing += 1
                continue
            interval_available += 1
            width = high - low
            interval_widths.append(width)
            if candidate is not None and abs(candidate) > np.finfo(float).eps:
                relative_widths.append(width / abs(candidate))
            if low <= truth <= high:
                interval_contains += 1

        error_array = np.asarray(error_values, dtype=float)
        parameter_summaries[str(parameter)] = {
            "trial_count": len(rows),
            "finite_candidate_count": finite_candidates,
            "missing_candidate_count": missing_candidates,
            "candidate_truth_pair_count": int(error_array.size),
            "candidate_bias": float(np.mean(error_array)) if error_array.size else None,
            "candidate_mae": float(np.mean(np.abs(error_array))) if error_array.size else None,
            "candidate_rmse": float(np.sqrt(np.mean(np.square(error_array)))) if error_array.size else None,
            "interval_attempt_count": len(rows),
            "interval_available_count": interval_available,
            "interval_unavailable_count": len(rows) - interval_available,
            "interval_missing_count": interval_bounds_missing,
            "reference_truth_missing_count": reference_truth_missing,
            "interval_contains_truth_count": interval_contains,
            "coverage_fraction_available_intervals": (
                float(interval_contains / interval_available) if interval_available else None
            ),
            "coverage_fraction_all_trials": (
                float(interval_contains / len(rows)) if rows else None
            ),
            "coverage_wilson_95_available_intervals": _wilson_95(interval_contains, interval_available),
            "coverage_wilson_95_all_trials": _wilson_95(interval_contains, len(rows)),
            "interval_width": _quantile_summary(interval_widths),
            "relative_interval_width": _quantile_summary(relative_widths),
            "candidate_status_counts": dict(sorted(candidate_statuses.items())),
            "confidence_counts": dict(sorted(confidence_statuses.items())),
        }

    return {
        "trial_count": len(rows),
        "execution_status_counts": dict(sorted(execution_counts.items())),
        "solver_success_counts": dict(sorted(solver_counts.items())),
        "condition_number": _quantile_summary(conditions),
        "parameters": parameter_summaries,
        "coverage_denominator_definitions": {
            "coverage_fraction_available_intervals": "intervals with finite ordered bounds and finite truth",
            "coverage_fraction_all_trials": "all attempted trials; unavailable intervals count as not covered",
        },
    }


def summarize_calibration_sensitivity(
    baseline_candidate: Mapping[str, Any] | None,
    variants: Sequence[Mapping[str, Any]],
    *,
    parameters: Sequence[str] = DEFAULT_PARAMETERS,
) -> dict[str, Any]:
    """Summarize paired candidate shifts under explicitly named perturbations."""

    baseline = _mapping(baseline_candidate)
    summaries: dict[str, Any] = {}
    for variant in variants:
        variant = _mapping(variant)
        candidate = _mapping(variant.get("candidate"))
        per_parameter: dict[str, Any] = {}
        for parameter in parameters:
            base_value = _finite_float(baseline.get(parameter))
            varied_value = _finite_float(candidate.get(parameter))
            delta = varied_value - base_value if base_value is not None and varied_value is not None else None
            relative_delta = (
                delta / abs(base_value)
                if delta is not None and base_value is not None and abs(base_value) > np.finfo(float).eps
                else None
            )
            per_parameter[str(parameter)] = {
                "baseline": base_value,
                "perturbed": varied_value,
                "delta": delta,
                "relative_delta": relative_delta,
                "paired_finite": delta is not None,
            }
        summaries[str(variant.get("label", f"variant_{len(summaries)}"))] = {
            "execution_status": str(variant.get("execution_status", "unspecified")),
            "solver_success": variant.get("solver_success"),
            "perturbation": variant.get("perturbation"),
            "parameters": per_parameter,
        }
    return {
        "variant_count": len(variants),
        "baseline_candidate": {
            str(parameter): _finite_float(baseline.get(parameter)) for parameter in parameters
        },
        "variants": summaries,
        "interpretation": "deterministic shifts under declared synthetic calibration perturbations; not instrument uncertainty estimates",
    }


__all__ = [
    "DEFAULT_PARAMETERS",
    "summarize_calibration_sensitivity",
    "summarize_uncertainty_trials",
]
