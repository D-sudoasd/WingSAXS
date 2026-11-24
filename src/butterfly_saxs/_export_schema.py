"""Canonical column schemas and per-frame row builders shared by batch exporters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import json
import math
from typing import Any

import numpy as np

from .batch import BatchRunResult, FrameFitResult, FrameRef, _json_safe
from .csv_utils import safe_csv_cell


_MISSING = object()
_FLAG_NAMES = (
    "scientific_flags",
    "flags",
    "quality_flags",
    "fit_flags",
    "warnings",
)

CANONICAL_FRAME_SUMMARY_COLUMNS = [
    "frame_index",
    "frame_id",
    "path",
    "frame_selector",
    "dataset",
    "time",
    "status",
    "quality_status",
    "confidence",
    "confidence_reason",
    "quality_diagnostic",
    "arc_sides",
    "q_star_from_arcs",
    "L_from_observed_radius_nm",
    "q_star_source",
    "observed_arc_q_median",
    "radial_hint_q",
    "radial_arc_comparison_q_unit",
    "radial_hint_selection_status",
    "radial_hint_reason",
    "observed_arc_q_source",
    "radial_arc_comparison_status",
    "radial_hint_to_observed_arc_ratio",
    "radial_arc_comparison_signed_relative_difference",
    "ellipse_kind",
    "a",
    "b",
    "axis_ratio",
    "theta_deg",
    "center_qx",
    "center_qy",
    "ellipticity",
    "eccentricity",
    "Ln_from_minor_axis_nm",
    "Lz_from_draw_axis_nm",
    "L_from_major_axis_nm",
    "Ln_candidate_from_minor_axis_nm",
    "Lz_candidate_from_draw_axis_nm",
    "L_candidate_from_major_axis_nm",
    "fit_status",
    "fit_success",
    "full2d_status",
    "full2d_success",
    "rmse",
    "weighted_rmse",
    "ndata",
    "sampled_n",
    "sample_rmse",
    "nfev",
    "condition_number",
    "error",
    "diagnostic",
    "warm_start_from",
    "elapsed_s",
    "resumed",
    "flags",
    "scientific_flags",
    "parameters_json",
    "frame_metadata_json",
]

PARAMETER_BASE_COLUMNS = [
    "frame_index",
    "frame_id",
    "path",
    "frame_selector",
    "dataset",
    "time",
    "status",
    "parameter",
    "value",
    "stderr",
    "uncertainty",
    "fixed",
    "unit",
    "flags",
    "bound_flags",
    "scientific_flags",
    "candidate_value",
    "publication_status",
    "identifiability_status",
    "identifiability_reason",
    "parameter_source",
    "confidence",
    "confidence_reason",
]

STREAMING_PARAMETER_COLUMNS = [
    *PARAMETER_BASE_COLUMNS,
    "interval_low",
    "interval_high",
    "interval_kind",
]

CANONICAL_RIDGE_COLUMNS = [
    "frame_index",
    "frame_id",
    "path",
    "frame_selector",
    "dataset",
    "time",
    "status",
    "error",
    "warm_start_from",
    "elapsed_s",
    "resumed",
    "point_index",
    "branch",
    "qx",
    "qy",
    "q",
    "q_star",
    "angle",
    "angle_deg",
    "intensity",
    "baseline",
    "snr",
    "radial_fwhm",
    "azimuthal_fwhm",
    "area",
    "coverage",
    "n_pixels",
    "valid",
    "accepted",
    "method",
    "reason",
    "flags",
    "support",
    "score",
    "point_score",
    "trajectory_id",
    "branch_id",
    "q_unit",
    "quadrant",
    "quadrant_pair",
    "branch_assignment_source",
    "symmetry_flags",
    "point_id",
    "arc_id",
    "side",
    "pixel_x",
    "pixel_y",
    "localization_sigma_q",
    "normal_fwhm_q",
    "normal_qx",
    "normal_qy",
    "q_normal_step",
    "scale_stability",
    "topology_flags",
    "projection_qx",
    "projection_qy",
    "normal_residual_q",
    "curvature_seed_qx",
    "curvature_seed_qy",
    "curvature_seed_pixel_x",
    "curvature_seed_pixel_y",
    "profile_shift_q",
    "profile_shift_pixel",
    "profile_refinement_applied",
    "profile_refinement_reason",
    "uncertainty_source",
]

CANONICAL_LOBE_COLUMNS = [
    "frame_index",
    "frame_id",
    "path",
    "frame_selector",
    "dataset",
    "time",
    "status",
    "error",
    "warm_start_from",
    "elapsed_s",
    "resumed",
    "measurement_kind",
    "lobe_index",
    "angle_rad",
    "angle_deg",
    "angle_unit",
    "q_star",
    "q_unit",
    "intensity",
    "baseline",
    "snr",
    "fwhm",
    "fwhm_deg",
    "radial_fwhm",
    "azimuthal_fwhm",
    "area",
    "coverage",
    "n_pixels",
    "valid",
    "accepted",
    "method",
    "reason",
    "refinement",
    "flags",
    "point_id",
    "q",
    "qx",
    "qy",
    "pixel_x",
    "pixel_y",
    "smoothed_intensity",
    "landmark_json",
]


def _value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _sequence(value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes)):
        return [] if value is None else [value]
    if isinstance(value, Mapping):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _scalar(value: Any) -> Any:
    """Convert numpy/Python scalar values for CSV without coercing arrays."""

    if value is None or isinstance(value, (str, bool, int)):
        return safe_csv_cell(value)
    if isinstance(value, float):
        return safe_csv_cell(value) if math.isfinite(value) else ""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _scalar(item())
        except Exception:  # pragma: no cover
            pass
    if isinstance(value, (list, tuple, Mapping)):
        return _json_text(value)
    return (
        safe_csv_cell(value)
        if isinstance(value, (str, int, float, bool))
        else _json_text(value)
    )


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return vars(value)
    return None


def _result_mapping(item: FrameFitResult) -> Mapping[str, Any] | None:
    return _as_mapping(item.result)


def _frame_results(batch: Any) -> list[FrameFitResult]:
    if isinstance(batch, BatchRunResult):
        return list(batch.frame_results)
    if hasattr(batch, "frame_results"):
        return list(batch.frame_results)
    result: list[FrameFitResult] = []
    for index, item in enumerate(batch or []):
        if isinstance(item, FrameFitResult):
            result.append(item)
        elif isinstance(item, Mapping):
            frame = item.get(
                "frame", item.get("frame_ref", item.get("path", f"frame_{index}"))
            )
            frame_ref = frame if isinstance(frame, FrameRef) else FrameRef(frame)
            result.append(
                FrameFitResult(
                    frame=frame_ref,
                    result=item.get("result", item.get("fit_result")),
                    status=item.get("status", "ok"),
                    error=item.get("error"),
                    diagnostic=item.get("diagnostic"),
                    warm_start_from=item.get("warm_start_from", item.get("lineage")),
                )
            )
        else:
            metadata = _value(item, "metadata", default={})
            if not isinstance(metadata, Mapping):
                metadata = {}
            source = metadata.get("path", metadata.get("source", f"frame_{index}"))
            frame_ref = FrameRef(
                source,
                frame_id=_value(item, "frame_id", "id", default=None),
                time=_value(item, "timestamp", "time", default=metadata.get("time")),
                metadata=metadata,
            )
            result.append(FrameFitResult(frame=frame_ref, result=item))
    return result


def _frame_base(item: FrameFitResult, index: int) -> dict[str, Any]:
    frame = item.frame
    return {
        "frame_index": index,
        "frame_id": _scalar(frame.frame_id),
        "path": str(frame.path),
        "frame_selector": _scalar(frame.frame_selector),
        "dataset": _scalar(frame.dataset_id or None),
        "time": _scalar(frame.time),
        "status": item.status,
        "error": _scalar(item.error) if item.error else "",
        "diagnostic": _scalar(item.diagnostic) if item.diagnostic else "",
        "warm_start_from": _scalar(item.warm_start_from) if item.warm_start_from else "",
        "elapsed_s": _scalar(item.elapsed_s),
        "resumed": bool(item.resumed),
    }


def _empty_parameter_source(value: Any) -> bool:
    if value is None or value is _MISSING:
        return True
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return not value
    try:
        return len(value) == 0
    except (TypeError, AttributeError):
        return False


def _parameter_source(value: Any) -> tuple[Any, list[Any]]:
    """Return the preferred parameter source and metadata contexts."""

    contexts: list[Any] = [value]
    if isinstance(value, Mapping):
        full2d = _value(value, "full2d", default=_MISSING)
        nested = _value(full2d, "parameters", "params", default=_MISSING)
        for name in ("parameters", "params"):
            if name in value and not _empty_parameter_source(value[name]):
                source = value[name]
                if not _empty_parameter_source(nested):
                    try:
                        is_nested_alias = _json_safe(source) == _json_safe(nested)
                    except (TypeError, ValueError):
                        is_nested_alias = source is nested
                    if is_nested_alias:
                        return source, [full2d, value]
                return source, contexts
    else:
        own = vars(value) if hasattr(value, "__dict__") else {}
        for name in ("parameters", "params"):
            if name in own and not _empty_parameter_source(own[name]):
                return own[name], contexts

    source = _value(value, "parameters", "params", default=_MISSING)
    if not _empty_parameter_source(source):
        full2d = _value(value, "full2d", default=_MISSING)
        nested = _value(full2d, "parameters", "params", default=_MISSING)
        if not _empty_parameter_source(nested) and source is nested:
            return nested, [full2d, value]
        return source, contexts

    full2d = _value(value, "full2d", default=_MISSING)
    if full2d is not _MISSING and full2d is not None:
        nested = _value(full2d, "parameters", "params", default=_MISSING)
        if not _empty_parameter_source(nested):
            return nested, [full2d, value]
    source = _value(value, "values", default=_MISSING)
    if not _empty_parameter_source(source):
        return source, contexts
    source = _value(value, "observables", default=_MISSING)
    if not _empty_parameter_source(source):
        return source, contexts
    ellipse = _value(value, "ellipse_fit", "ellipse", default=None)
    nested = _value(ellipse, "parameters", "params", default=_MISSING)
    if not _empty_parameter_source(nested):
        return nested, [ellipse, value]
    return _MISSING, contexts


def _first_context_value(contexts: Sequence[Any], names: Sequence[str]) -> Any:
    for context in contexts:
        candidate = _value(context, *names, default=_MISSING)
        if candidate is not _MISSING and candidate is not None:
            return candidate
    return _MISSING


def _context_parameter_value(
    contexts: Sequence[Any], names: Sequence[str], name: Any
) -> Any:
    candidate = _first_context_value(contexts, names)
    if candidate is _MISSING:
        return _MISSING
    if isinstance(candidate, Mapping):
        return candidate.get(name, _MISSING)
    return candidate


def _result_flags(value: Any) -> Any:
    """Prefer flags beside the selected parameter source."""

    _, contexts = _parameter_source(value)
    flags = _first_context_value(contexts, _FLAG_NAMES)
    return None if flags is _MISSING else flags


def _quality_confidence(value: Any) -> tuple[Any, Any]:
    """Return confidence and its explanation from result or butterfly quality."""

    butterfly = _value(value, "butterfly", default=None)
    if butterfly is None:
        butterfly = _value(
            _value(value, "observables", default={}), "butterfly", default=None
        )
    quality = _value(butterfly, "quality", default=None)
    if quality is None:
        quality = _value(value, "quality", default=None)
    confidence = _value(value, "confidence", default=_MISSING)
    if confidence is _MISSING:
        confidence = _value(quality, "confidence", "confidence_level", default="")
    reason = _value(value, "confidence_reason", default=_MISSING)
    if reason is _MISSING:
        reason = _value(quality, "confidence_reason", "confidence_basis", default="")
    return confidence, reason


def _eccentricity_from_ratio(value: Any) -> float | None:
    """Keep a missing/non-finite ellipse estimate missing during export."""

    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return math.sqrt(max(0.0, 1.0 - ratio**2)) if math.isfinite(ratio) else None


def _parameters(value: Any) -> list[dict[str, Any]]:
    """Normalise parameter sources while retaining fit diagnostics."""

    source, contexts = _parameter_source(value)
    if source is _MISSING or source is None:
        return []

    spec_items = getattr(source, "spec_items", None)
    if callable(spec_items):
        try:
            iterable = list(spec_items())
        except Exception:  # pragma: no cover - defensive for custom sets
            iterable = []
    elif isinstance(source, Mapping):
        iterable = list(source.items())
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        iterable = []
        for item in source:
            if isinstance(item, Mapping):
                name = item.get("name", item.get("parameter", item.get("key")))
                if name is not None:
                    iterable.append((name, item))
    else:
        source_mapping = _as_mapping(source)
        iterable = list(source_mapping.items()) if source_mapping is not None else []

    rows: list[dict[str, Any]] = []
    geometry_units = _value(value, "geometry_parameter_units", default={})
    if not isinstance(geometry_units, Mapping):
        geometry_units = {}
    confidence, confidence_reason = _quality_confidence(value)
    confidence_cell = (
        _json_text(confidence)
        if isinstance(confidence, (Mapping, list, tuple, set, frozenset))
        else (_scalar(confidence) if confidence is not _MISSING else "")
    )
    confidence_reason_cell = (
        _json_text(confidence_reason)
        if isinstance(confidence_reason, (Mapping, list, tuple, set, frozenset))
        else (_scalar(confidence_reason) if confidence_reason is not _MISSING else "")
    )
    for name, spec in iterable:
        parameter_name = str(name)
        if np.isscalar(spec) or isinstance(spec, str):
            value_field = spec
            stderr = uncertainty = fixed = unit = flags = bound_flags = _MISSING
        else:
            spec_mapping = spec if isinstance(spec, Mapping) else _as_mapping(spec)
            if spec_mapping is not None:
                value_field = _value(
                    spec_mapping, "value", "val", "estimate", default=None
                )
                stderr = _value(spec_mapping, "stderr", default=_MISSING)
                uncertainty = _value(
                    spec_mapping,
                    "uncertainty",
                    "sigma",
                    "error",
                    "std",
                    default=_MISSING,
                )
                fixed = _value(spec_mapping, "fixed", "is_fixed", default=_MISSING)
                unit = _value(spec_mapping, "unit", default=_MISSING)
                flags = _value(
                    spec_mapping, "flags", "scientific_flags", default=_MISSING
                )
                bound_flags = _value(spec_mapping, "bound_flags", default=_MISSING)
                if fixed is _MISSING and (
                    "vary" in spec_mapping or "expr" in spec_mapping
                ):
                    fixed = spec_mapping.get("expr") is None and not bool(
                        spec_mapping.get("vary", True)
                    )
                if flags is _MISSING and spec_mapping.get("expr") is not None:
                    flags = {"tied": True}
            else:
                value_field = getattr(spec, "value", spec)
                stderr = getattr(spec, "stderr", _MISSING)
                uncertainty = getattr(spec, "uncertainty", _MISSING)
                fixed = getattr(spec, "is_fixed", _MISSING)
                if fixed is _MISSING and hasattr(spec, "vary"):
                    fixed = not bool(getattr(spec, "vary"))
                unit = getattr(spec, "unit", _MISSING)
                flags = getattr(spec, "flags", _MISSING)
                if flags is _MISSING and getattr(spec, "is_tied", False):
                    flags = {"tied": True}
                bound_flags = getattr(spec, "bound_flags", _MISSING)

        if stderr is _MISSING:
            stderr = _context_parameter_value(contexts, ("stderr",), name)
        if uncertainty is _MISSING:
            uncertainty = stderr
        if stderr is _MISSING:
            stderr = uncertainty
        if fixed is _MISSING:
            fixed = _context_parameter_value(contexts, ("fixed", "is_fixed"), name)
        if unit is _MISSING:
            unit = _context_parameter_value(contexts, ("units", "unit"), name)
        if unit is _MISSING:
            unit = geometry_units.get(str(name), _MISSING)
        if flags is _MISSING:
            flags = _context_parameter_value(
                contexts,
                ("flags", "scientific_flags", "quality_flags", "fit_flags"),
                name,
            )
        if bound_flags is _MISSING:
            bound_flags = _context_parameter_value(contexts, ("bound_flags",), name)

        rows.append(
            {
                "parameter": parameter_name,
                "value": _scalar(value_field),
                "candidate_value": _scalar(value_field),
                "publication_status": "not_assessed",
                "identifiability_status": "not_assessed",
                "identifiability_reason": "",
                "parameter_source": "candidate_only",
                "confidence": confidence_cell,
                "confidence_reason": confidence_reason_cell,
                "stderr": _scalar(stderr) if stderr is not _MISSING else "",
                "uncertainty": (
                    _scalar(uncertainty) if uncertainty is not _MISSING else ""
                ),
                "fixed": _scalar(fixed) if fixed is not _MISSING else "",
                "unit": _scalar(unit) if unit is not _MISSING else "",
                "flags": (
                    _json_text(flags)
                    if isinstance(flags, (Mapping, list, tuple, set, frozenset))
                    else (_scalar(flags) if flags is not _MISSING else "")
                ),
                "bound_flags": (
                    _json_text(bound_flags)
                    if isinstance(bound_flags, (Mapping, list, tuple, set, frozenset))
                    else (_scalar(bound_flags) if bound_flags is not _MISSING else "")
                ),
            }
        )
    ellipse = _value(value, "ellipse_fit", "ellipse", default={})
    evidence = _value(ellipse, "quantitative_parameters", default={})
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if evidence:
        aliases = {
            "semi_major": "a",
            "semi_minor": "b",
            "axes_ratio": "axis_ratio",
            "ellipse_axis_tilt_deg": "theta_deg",
            "angle_deg": "theta_deg",
            "eccentricity": "axis_ratio",
            "ellipticity": "axis_ratio",
        }
        for row in rows:
            name = row["parameter"]
            check = evidence.get(aliases.get(name, name))
            if not isinstance(check, Mapping):
                continue
            candidate_value = check.get("candidate_value")
            if candidate_value is None:
                candidate_value = row["value"]
            if name in ("eccentricity", "ellipticity"):
                candidate_value = _eccentricity_from_ratio(candidate_value)
            row["candidate_value"] = _scalar(candidate_value)
            row["identifiability_status"] = check.get("status", "undetermined")
            row["identifiability_reason"] = check.get("reason", "")
            quantitative_value = check.get("value")
            if name in ("eccentricity", "ellipticity"):
                quantitative_value = _eccentricity_from_ratio(quantitative_value)
            check_status = str(check.get("status", "undetermined")).strip().casefold()
            value_is_finite = (
                not isinstance(quantitative_value, bool)
                and isinstance(quantitative_value, (int, float, np.number))
                and math.isfinite(float(quantitative_value))
            )
            if check_status in {"available", "estimate", "candidate"} and value_is_finite:
                row["value"] = _scalar(quantitative_value)
                row["publication_status"] = (
                    "available" if check_status == "available" else "not_assessed"
                )
                row["parameter_source"] = (
                    "quantitative" if check_status == "available" else "candidate"
                )
            else:
                row["value"] = ""
                row["publication_status"] = "not_assessed"
                row["parameter_source"] = "candidate_only"
            parameter_confidence = check.get(
                "confidence", check.get("confidence_level", _MISSING)
            )
            if parameter_confidence is not _MISSING:
                row["confidence"] = (
                    _json_text(parameter_confidence)
                    if isinstance(
                        parameter_confidence, (Mapping, list, tuple, set, frozenset)
                    )
                    else _scalar(parameter_confidence)
                )
            parameter_confidence_reason = check.get("confidence_reason", _MISSING)
            if parameter_confidence_reason is not _MISSING:
                row["confidence_reason"] = (
                    _json_text(parameter_confidence_reason)
                    if isinstance(
                        parameter_confidence_reason,
                        (Mapping, list, tuple, set, frozenset),
                    )
                    else _scalar(parameter_confidence_reason)
                )
            interval = check.get("interval")
            if (
                isinstance(interval, (tuple, list))
                and len(interval) == 2
                and name not in ("eccentricity", "ellipticity")
            ):
                row["interval_low"], row["interval_high"] = interval
            row["interval_kind"] = check.get("interval_kind", "")
            if not row.get("unit"):
                row["unit"] = (
                    "dimensionless"
                    if name in ("eccentricity", "ellipticity")
                    else check.get("unit", "")
                )
    shape_available = all(
        isinstance(evidence.get(name), Mapping)
        and evidence[name].get("status") == "available"
        for name in ("a", "b", "axis_ratio", "theta_deg")
    )
    if not shape_available:
        for row in rows:
            if row["parameter"] not in {
                "L_N",
                "L_z",
                "Ln_from_minor_axis_nm",
                "Lz_from_draw_axis_nm",
                "L_from_major_axis_nm",
            }:
                continue
            if row.get("candidate_value") in (None, ""):
                row["candidate_value"] = row["value"]
            row["value"] = ""
            row["publication_status"] = "not_assessed"
            row["identifiability_status"] = "undetermined"
            row["identifiability_reason"] = "ellipse_shape_not_quantitatively_available"
            row["parameter_source"] = "candidate_only"
    return rows


def _quality_summary(value: Any) -> dict[str, Any]:
    """Flatten status and confidence labels while preserving detailed source JSON."""

    butterfly = _value(value, "butterfly", default=None)
    if butterfly is None:
        butterfly = _value(
            _value(value, "observables", default={}), "butterfly", default=None
        )
    quality = _value(butterfly, "quality", default=None)
    if quality is None:
        quality = _value(value, "quality", default=None)
    confidence, confidence_reason = _quality_confidence(value)
    diagnostic = _value(
        quality,
        "diagnostic",
        "reason",
        default=_value(value, "quality_diagnostic", "diagnostic", default=""),
    )
    return {
        "quality_status": _scalar(
            _value(
                quality, "status", default=_value(value, "quality_status", default="")
            )
        ),
        "confidence": (
            _json_text(confidence)
            if isinstance(confidence, (Mapping, list, tuple, set, frozenset))
            else (_scalar(confidence) if confidence is not _MISSING else "")
        ),
        "confidence_reason": (
            _json_text(confidence_reason)
            if isinstance(confidence_reason, (Mapping, list, tuple, set, frozenset))
            else (_scalar(confidence_reason) if confidence_reason is not _MISSING else "")
        ),
        "quality_diagnostic": (
            _json_text(diagnostic)
            if isinstance(diagnostic, (Mapping, list, tuple, set, frozenset))
            else (_scalar(diagnostic) if diagnostic is not _MISSING else "")
        ),
    }


def _radial_arc_summary(value: Any) -> dict[str, Any]:
    """Flatten radial-hint and observed-arc comparison data for frame CSVs."""

    geometry = _as_mapping(_value(value, "geometry_parameters", default=None)) or {}
    butterfly = _value(value, "butterfly", default=None)
    if butterfly is None:
        butterfly = _value(
            _value(value, "observables", default={}), "butterfly", default=None
        )
    candidate = _value(butterfly, "candidate_fit", default={})
    candidate_parameters = _value(
        candidate, "parameters", "parameter_values", default={}
    )
    comparison = _value(geometry, "radial_arc_comparison", default=_MISSING)
    if comparison is _MISSING:
        comparison = _value(candidate, "radial_arc_comparison", default={})

    def geometry_or_candidate(name: str, *, parameter_value: bool = False) -> Any:
        result_value = _value(geometry, name, default=_MISSING)
        if result_value is not _MISSING:
            return result_value
        if parameter_value:
            result_value = _value(candidate_parameters, name, default=_MISSING)
            if result_value is not _MISSING:
                return result_value
        return _value(candidate, name, default=None)

    q_unit = _value(comparison, "q_unit", default=_MISSING)
    if q_unit is _MISSING:
        q_unit = _value(candidate, "q_unit", default=_MISSING)
    if q_unit is _MISSING:
        q_unit = _value(
            _value(value, "geometry_metrics", default={}), "q_unit", default=""
        )
    return {
        "observed_arc_q_median": _scalar(
            geometry_or_candidate("observed_arc_q_median", parameter_value=True)
        ),
        "radial_hint_q": _scalar(
            geometry_or_candidate("radial_hint_q", parameter_value=True)
        ),
        "radial_arc_comparison_q_unit": _scalar(q_unit),
        "radial_hint_selection_status": _scalar(
            geometry_or_candidate("radial_hint_selection_status")
        ),
        "radial_hint_reason": _scalar(geometry_or_candidate("radial_hint_reason")),
        "observed_arc_q_source": _scalar(
            geometry_or_candidate("observed_arc_q_source")
        ),
        "radial_arc_comparison_status": _scalar(
            _value(comparison, "status", default=None)
        ),
        "radial_hint_to_observed_arc_ratio": _scalar(
            _value(comparison, "radial_hint_to_observed_arc_ratio", default=None)
        ),
        "radial_arc_comparison_signed_relative_difference": _scalar(
            _value(comparison, "signed_relative_difference", default=None)
        ),
    }


def build_frame_summary_row(item: FrameFitResult, index: int) -> dict[str, Any]:
    """Build one canonical ``frame_summary.csv`` row shared by batch and streaming exporters."""

    row = _frame_base(item, index)
    result = item.result
    mapping = _result_mapping(item)
    params = _parameters(result)
    if mapping:
        for key, value in mapping.items():
            key_text = str(key)
            if key_text in {
                "parameters",
                "params",
                "ridge_points",
                "ellipse_fit",
                "ellipse",
            }:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[key_text] = _scalar(value)
            elif key_text in _FLAG_NAMES:
                row[key_text] = _json_text(value)
        if "flags" not in row:
            row["flags"] = _json_text(_value(result, "flags", default=None))
        if "scientific_flags" not in row:
            row["scientific_flags"] = _json_text(
                _value(result, "scientific_flags", default=None)
            )
    for section_name, prefix in (("metrics", "fit"), ("full2d", "full2d")):
        section = _value(result, section_name, default=None)
        section_mapping = _as_mapping(section)
        if not section_mapping:
            continue
        for metric_name in (
            "status",
            "success",
            "rmse",
            "weighted_rmse",
            "ndata",
            "sampled_n",
            "sample_rmse",
            "nfev",
            "condition_number",
            "condition",
        ):
            metric = _value(section_mapping, metric_name, default=_MISSING)
            if metric is _MISSING or not (metric is None or np.isscalar(metric)):
                continue
            if metric_name in {"status", "success"}:
                output_name = f"{prefix}_{metric_name}"
            elif metric_name == "condition":
                output_name = "condition_number"
            else:
                output_name = metric_name
            row.setdefault(output_name, _scalar(metric))
    for parameter in params:
        name = parameter["parameter"]
        row.setdefault(name, parameter["value"])
    geometry = _as_mapping(_value(result, "geometry_parameters", default=None))
    if geometry:
        for name in (
            "a",
            "b",
            "axis_ratio",
            "theta_deg",
            "center_qx",
            "center_qy",
            "ellipticity",
            "eccentricity",
            "Ln_from_minor_axis_nm",
            "Lz_from_draw_axis_nm",
            "L_from_major_axis_nm",
        ):
            if name in geometry:
                row[name] = _scalar(geometry[name])
    for name, value in _quality_summary(result).items():
        row.setdefault(name, value)
    row.setdefault("arc_sides", _scalar(_value(result, "arc_sides", default=None)))
    row.setdefault(
        "q_star_from_arcs",
        _scalar(
            _value(geometry, "q_star_from_arcs", default=None)
            or _value(result, "q_star_from_arcs", default=None)
        ),
    )
    row.setdefault(
        "L_from_observed_radius_nm",
        _scalar(
            _value(geometry, "L_from_observed_radius_nm", default=None)
            or _value(result, "L_from_observed_radius_nm", default=None)
        ),
    )
    row.setdefault(
        "q_star_source",
        _scalar(
            _value(geometry, "q_star_source", default=None)
            or _value(result, "q_star_source", default=None)
        ),
    )
    row.update(_radial_arc_summary(result))
    row.setdefault(
        "ellipse_kind",
        _scalar(
            _value(result, "ellipse_kind", default=None)
            or _value(geometry, "ellipse_kind", default=None)
        ),
    )
    row.setdefault(
        "Ln_candidate_from_minor_axis_nm",
        _scalar(
            _value(geometry, "Ln_candidate_from_minor_axis_nm", default=None)
            or _value(result, "Ln_candidate_from_minor_axis_nm", default=None)
        ),
    )
    row.setdefault(
        "Lz_candidate_from_draw_axis_nm",
        _scalar(
            _value(geometry, "Lz_candidate_from_draw_axis_nm", default=None)
            or _value(result, "Lz_candidate_from_draw_axis_nm", default=None)
        ),
    )
    row.setdefault(
        "L_candidate_from_major_axis_nm",
        _scalar(
            _value(geometry, "L_candidate_from_major_axis_nm", default=None)
            or _value(result, "L_candidate_from_major_axis_nm", default=None)
        ),
    )
    row["parameters_json"] = _json_text(params)
    row["frame_metadata_json"] = _json_text(item.frame.metadata)
    return row


def _frame_summary_rows(results: Sequence[FrameFitResult]) -> list[dict[str, Any]]:
    return [build_frame_summary_row(item, index) for index, item in enumerate(results)]


def _ridge_points_from_sequence(points: Any) -> list[dict[str, Any]]:
    if points is None:
        return []
    tolist = getattr(points, "tolist", None)
    if callable(tolist):
        try:
            points = tolist()
        except Exception:  # pragma: no cover
            pass
    if isinstance(points, Mapping):
        return [dict(points)]
    if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        rows: list[dict[str, Any]] = []
        for index, point in enumerate(points):
            if isinstance(point, Mapping):
                rows.append({str(key): _scalar(value) for key, value in point.items()})
            elif (point_mapping := _as_mapping(point)) is not None:
                row = {
                    str(key): _scalar(value)
                    for key, value in point_mapping.items()
                    if key != "metadata"
                }
                metadata = point_mapping.get("metadata")
                if isinstance(metadata, Mapping):
                    for key in (
                        "quadrant",
                        "quadrant_pair",
                        "branch_assignment_source",
                        "symmetry_flags",
                    ):
                        if key in metadata:
                            row[key] = _scalar(metadata[key])
                rows.append(row)
            elif isinstance(point, Sequence) and not isinstance(point, (str, bytes)):
                row = {"point_index": index}
                for axis, coordinate in enumerate(point):
                    row[
                        "q_x"
                        if axis == 0
                        else "q_y"
                        if axis == 1
                        else f"coordinate_{axis}"
                    ] = _scalar(coordinate)
                rows.append(row)
            else:
                rows.append({"point_index": index, "value": _scalar(point)})
        return rows
    return [{"value": _scalar(points)}]


def _ridge_points(value: Any) -> list[dict[str, Any]]:
    points = _value(value, "ridge_points", "ridges", "ridge", default=None)
    if points is None:
        return []
    if isinstance(points, Mapping):
        rows: list[dict[str, Any]] = []
        for branch, branch_points in points.items():
            for row in _ridge_points_from_sequence(branch_points):
                row.setdefault("branch", branch)
                rows.append(row)
        return rows
    return _ridge_points_from_sequence(points)


def _ellipse_fit(value: Any) -> Any:
    return _value(value, "ellipse_fit", "ellipse", default=None)


def _lobe_radial_profiles(value: Any) -> Any:
    direct = _value(value, "lobe_radial_profiles", "radial_profiles", default=None)
    if direct is not None:
        return direct
    observables = _value(value, "observables", "measurements", default=None)
    return _value(observables, "lobe_radial_profiles", "radial_profiles", default=None)


def _lobe_radial_peaks(value: Any) -> Any:
    direct = _value(value, "lobe_radial_peaks", "radial_peaks", default=None)
    if direct is not None:
        return direct
    observables = _value(value, "observables", "measurements", default=None)
    return _value(observables, "lobe_radial_peaks", "radial_peaks", default=None)


def _lobe_angular(value: Any) -> Any:
    direct = _value(value, "lobes", "lobe_metrics", "angular_lobes", default=None)
    if direct is not None:
        return direct
    observables = _value(value, "observables", "measurements", default=None)
    return _value(observables, "lobes", "lobe_metrics", "angular_lobes", default=None)


def _butterfly_evidence(value: Any) -> Any:
    direct = _value(value, "butterfly", default=None)
    if isinstance(direct, Mapping) and direct:
        return direct
    observables = _value(value, "observables", default=None)
    nested = _value(observables, "butterfly", default=None)
    if isinstance(nested, Mapping):
        return nested
    return direct if direct is not None else nested if nested is not None else {}


def _angle_pair(value: Any, angle_deg: Any = None) -> tuple[float | None, float | None]:
    try:
        if value is None and angle_deg is not None:
            degree = float(angle_deg)
            return float(np.deg2rad(degree)), degree
        if value is not None:
            radians = float(value)
            return radians, float(np.degrees(radians))
    except (TypeError, ValueError, OverflowError):
        return None, None
    return None, None


def _lobe_measurement_rows(item: FrameFitResult, index: int) -> list[dict[str, Any]]:
    """Flatten all direct angular and radial lobe scalar observables."""

    base = _frame_base(item, index)
    rows: list[dict[str, Any]] = []
    angular = _sequence(_lobe_angular(item.result))
    for lobe_index, lobe in enumerate(angular):
        angle = _value(lobe, "angle", "azimuth", default=None)
        angle_deg_value = _value(lobe, "angle_deg", "azimuth_deg", default=None)
        angle, angle_deg = _angle_pair(angle, angle_deg_value)
        fwhm = _value(lobe, "fwhm", "azimuthal_fwhm", default=None)
        try:
            fwhm_deg = float(np.degrees(float(fwhm)))
        except (TypeError, ValueError):
            fwhm_deg = None
        rows.append(
            {
                **base,
                "measurement_kind": "angular",
                "lobe_index": lobe_index,
                "angle_rad": angle,
                "angle_deg": angle_deg,
                "angle_unit": "rad",
                "q_star": None,
                "q_unit": None,
                "intensity": _value(lobe, "intensity", default=None),
                "baseline": _value(lobe, "baseline", default=None),
                "snr": _value(lobe, "snr", default=None),
                "fwhm": fwhm,
                "fwhm_deg": fwhm_deg,
                "radial_fwhm": None,
                "azimuthal_fwhm": fwhm,
                "area": _value(lobe, "area", default=None),
                "coverage": _value(lobe, "coverage", default=None),
                "n_pixels": _value(lobe, "n_pixels", default=None),
                "valid": _value(lobe, "valid", default=None),
                "accepted": _value(lobe, "valid", "accepted", default=None),
                "method": "angular_profile",
                "reason": _value(lobe, "reason", default=None),
                "refinement": _value(lobe, "refinement", default=None),
                "flags": _json_text(_value(lobe, "flags", default=())),
            }
        )
    radial = _sequence(_lobe_radial_peaks(item.result))
    for lobe_index, peak in enumerate(radial):
        angle = _value(peak, "angle", "azimuth", default=None)
        angle, angle_deg = _angle_pair(angle, _value(peak, "angle_deg", default=None))
        rows.append(
            {
                **base,
                "measurement_kind": "radial",
                "lobe_index": lobe_index,
                "angle_rad": angle,
                "angle_deg": angle_deg,
                "angle_unit": "rad",
                "q_star": _value(peak, "q_star", "q", "q_position", default=None),
                "q_unit": _value(peak, "q_unit", default=None),
                "intensity": _value(peak, "intensity", default=None),
                "baseline": _value(peak, "baseline", default=None),
                "snr": _value(peak, "snr", default=None),
                "fwhm": None,
                "fwhm_deg": None,
                "radial_fwhm": _value(peak, "radial_fwhm", default=None),
                "azimuthal_fwhm": _value(peak, "azimuthal_fwhm", default=None),
                "area": _value(peak, "area", default=None),
                "coverage": _value(peak, "coverage", default=None),
                "n_pixels": _value(peak, "n_pixels", default=None),
                "valid": _value(peak, "valid", default=None),
                "accepted": _value(peak, "accepted", "valid", default=None),
                "method": _value(peak, "method", default="radial_peak"),
                "reason": _value(peak, "reason", default=None),
                "refinement": None,
                "flags": _json_text(_value(peak, "flags", default=())),
            }
        )
    butterfly = _butterfly_evidence(item.result)
    landmarks = (
        butterfly.get("peak_landmarks", {}) if isinstance(butterfly, Mapping) else {}
    )
    if isinstance(landmarks, Mapping):
        raw = landmarks.get("raw_global_max")
        entries = [("raw_pixel_maximum", raw)] if isinstance(raw, Mapping) else []
        entries += [
            ("supported_lobe_pixel", point)
            for point in landmarks.get("peaks", [])
            if isinstance(point, Mapping)
        ]
        for peak_index, (kind, point) in enumerate(entries):
            rows.append(
                {
                    **base,
                    "measurement_kind": kind,
                    "lobe_index": peak_index,
                    "q_star": None,
                    "q": point.get("q"),
                    "q_unit": landmarks.get("q_unit"),
                    "qx": point.get("qx"),
                    "qy": point.get("qy"),
                    "angle_deg": point.get("chi_deg"),
                    "angle_unit": "deg",
                    "pixel_x": point.get("pixel_x"),
                    "pixel_y": point.get("pixel_y"),
                    "point_id": _value(
                        point, "peak_id", "point_id", "id", default="G"
                    ),
                    "intensity": point.get("raw_intensity"),
                    "smoothed_intensity": point.get("smoothed_intensity"),
                    "valid": True,
                    "accepted": None if kind == "raw_pixel_maximum" else True,
                    "method": landmarks.get("method_version"),
                    "reason": (
                        "literal maximum in supplied valid domain"
                        if kind == "raw_pixel_maximum"
                        else "supported lobe maximum"
                    ),
                    "flags": _json_text(point.get("flags", [])),
                    "landmark_json": _json_text(point),
                }
            )
    return rows


def _fit_audit(value: Any) -> dict[str, Any]:
    """Collect multistart/sample-vs-full objective fields for export."""

    direct = _value(value, "fit_audit", default=None)
    if isinstance(direct, Mapping):
        return dict(direct)
    full2d = _value(value, "full2d", default=None)
    source = full2d if full2d is not None else value
    audit_fields = (
        "sample_cost",
        "full_cost",
        "selection_objective",
        "candidate_solutions",
        "selected_start_index",
        "multistart_count",
        "symmetry",
    )
    return {
        name: _value(source, name, default=None)
        for name in audit_fields
        if _value(source, name, default=None) is not None
    }


def _batch_ellipse_row(item: FrameFitResult, index: int) -> dict[str, Any]:
    """Build one shared compact ellipse/butterfly sidecar row."""

    from .pipeline import _jsonable

    butterfly = _jsonable(_butterfly_evidence(item.result), array_summary=True)
    return {
        **_frame_base(item, index),
        "ellipse_fit": _json_safe(_ellipse_fit(item.result)),
        "fit_audit": _json_safe(_fit_audit(item.result)),
        "lobe_angular": _json_safe(_lobe_angular(item.result)),
        "lobe_radial_profiles": _json_safe(_lobe_radial_profiles(item.result)),
        "lobe_radial_peaks": _json_safe(_lobe_radial_peaks(item.result)),
        "peak_landmarks": _json_safe(_value(butterfly, "peak_landmarks", default={})),
        "butterfly": _json_safe(butterfly),
    }


__all__ = [
    "CANONICAL_FRAME_SUMMARY_COLUMNS",
    "CANONICAL_LOBE_COLUMNS",
    "CANONICAL_RIDGE_COLUMNS",
    "PARAMETER_BASE_COLUMNS",
    "STREAMING_PARAMETER_COLUMNS",
    "_MISSING",
    "_as_mapping",
    "_batch_ellipse_row",
    "_butterfly_evidence",
    "_ellipse_fit",
    "_fit_audit",
    "_frame_base",
    "_frame_results",
    "_frame_summary_rows",
    "_json_text",
    "_lobe_angular",
    "_lobe_measurement_rows",
    "_lobe_radial_peaks",
    "_lobe_radial_profiles",
    "_parameter_source",
    "_parameters",
    "_quality_confidence",
    "_quality_summary",
    "_radial_arc_summary",
    "_result_flags",
    "_result_mapping",
    "_ridge_points",
    "_scalar",
    "_sequence",
    "_value",
    "build_frame_summary_row",
]
