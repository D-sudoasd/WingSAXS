"""Qt-free batch and evolution record extraction, flattening, and failure classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from ..butterfly_quality import classify_ellipse_publication, ellipse_candidate_is_limited
from ._common import _read

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None

_BATCH_STATUS_KEYS: dict[str, str] = {
    "ready": "status.ready",
    "ok": "status.batch_ok",
    "warning": "status.batch_warning",
    "warn": "status.batch_warning",
    "failed": "status.batch_failed",
    "skipped": "status.batch_skipped",
}

_BATCH_TABLE_HEADERS: tuple[str, ...] = (
    "header.frame",
    "header.status",
    "header.quality",
    "header.a",
    "header.b",
    "header.axis_ratio",
    "header.theta_deg",
    "header.arcs",
    "header.ln",
    "header.lz",
    "header.l_radial",
    "header.rmse",
    "header.flags",
)

_BATCH_SUCCESS_STATUSES = frozenset({"ok", "success", "completed", "warning", "warn"})
_BATCH_LIMITED_STATUSES = frozenset({"warning", "warn"})


def _sequence(value: Any) -> list[Any]:
    """Convert optional NumPy/iterable profile fields without truth testing arrays."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _result_value(result: Any, names: tuple[str, ...], default: Any = None) -> Any:
    value = _read(result, names, default)
    return default if value is None else value


def _is_butterfly_trace_result(result: Any) -> bool:
    """Recognize successful arc tracing before ellipse evaluation."""

    analysis = _read(result, ("analysis",), {})
    butterfly = _read(result, ("butterfly",), None)
    if not isinstance(butterfly, Mapping):
        observables = _read(result, ("observables", "measurements"), None)
        butterfly = _read(observables, ("butterfly",), None)
    settings = _read(butterfly, ("settings",), {})
    nested = _read(analysis, ("butterfly",), {})
    stage = _read(settings, ("stage",), _read(nested, ("stage",), None))
    method = str(_read(analysis, ("ridge_method",), "") or "").strip().lower().replace("-", "_")
    points = _read(butterfly, ("points",), None)
    return (
        method == "butterfly_curvature"
        and str(stage or "").lower() == "trace"
        and isinstance(points, list)
    )


def _batch_scalar(
    source: Any,
    names: tuple[str, ...],
    *,
    include_candidate: bool = False,
) -> float | None:
    """Read one finite geometry/export scalar from nested batch records."""

    if not isinstance(source, Mapping):
        return None
    for name in names:
        value = source.get(name)
        if isinstance(value, Mapping):
            value_keys = (
                ("value", "val", "candidate_value") if include_candidate else ("value", "val")
            )
            value = next(
                (value.get(key) for key in value_keys if value.get(key) not in (None, "")),
                None,
            )
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _batch_parameter_metadata(
    sources: Sequence[Any],
    names: tuple[str, ...],
) -> dict[str, str]:
    """Read confidence and reporting state attached to a quantitative row."""

    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for name in names:
            value = source.get(name)
            if isinstance(value, Mapping) and any(
                value.get(key) not in (None, "")
                for key in ("status", "confidence", "publication_status")
            ):
                return {
                    key: str(value.get(key) or "").strip()
                    for key in ("status", "confidence", "publication_status")
                }
    return {}


def _batch_candidate_scalar(
    source: Any,
    candidate_names: tuple[str, ...],
    parameter_names: tuple[str, ...],
    *,
    include_plain_parameter: bool = False,
) -> float | None:
    """Read an explicit candidate field or a nested candidate value."""

    value = _batch_scalar(source, candidate_names, include_candidate=True)
    if value is not None or not isinstance(source, Mapping):
        return value
    if include_plain_parameter:
        value = _batch_scalar(source, parameter_names, include_candidate=True)
        if value is not None:
            return value
    for name in parameter_names:
        parameter = source.get(name)
        if isinstance(parameter, Mapping) and any(
            parameter.get(key) not in (None, "") for key in ("candidate_value", "candidate")
        ):
            return _batch_scalar(source, (name,), include_candidate=True)
    return None


def _batch_record_geometry(record: Mapping[str, Any]) -> dict[str, Any]:
    """Collect public ellipse fields for the batch review table."""

    candidate_sources: list[Any] = [record.get("candidate_geometry_parameters")]
    sources: list[Any] = [
        record.get("geometry_parameters"),
        record.get("parameters"),
        record.get("ellipse_fit"),
        record.get("metrics"),
        record.get("quantitative_parameters"),
    ]
    butterfly = record.get("butterfly")
    if isinstance(butterfly, Mapping):
        candidate_sources.append(butterfly.get("candidate_geometry_parameters"))
        sources.extend(
            (
                butterfly.get("candidate_fit"),
                butterfly.get("quantitative_parameters"),
                butterfly,
            )
        )
    observables = record.get("observables")
    if isinstance(observables, Mapping):
        sources.extend((observables.get("ellipse"), observables.get("butterfly"), observables))
    a = b = ratio = theta = ln = lz = l_radial = rmse = None
    a_candidate = b_candidate = ratio_candidate = theta_candidate = None
    ln_candidate = lz_candidate = l_major_candidate = None
    for source in sources:
        if a is None:
            a = _batch_scalar(source, ("a", "semi_major"))
        if b is None:
            b = _batch_scalar(source, ("b", "semi_minor"))
        if ratio is None:
            ratio = _batch_scalar(source, ("axis_ratio", "b_over_a"))
        if theta is None:
            theta = _batch_scalar(source, ("theta_deg", "angle_deg"))
        if ln is None:
            ln = _batch_scalar(source, ("Ln_from_minor_axis_nm", "L_N", "Ln_nm"))
        if lz is None:
            lz = _batch_scalar(source, ("Lz_from_draw_axis_nm", "L_z", "Lz_nm"))
        if ln_candidate is None:
            ln_candidate = _batch_scalar(source, ("Ln_candidate_from_minor_axis_nm",))
        if lz_candidate is None:
            lz_candidate = _batch_scalar(source, ("Lz_candidate_from_draw_axis_nm",))
        if l_major_candidate is None:
            l_major_candidate = _batch_scalar(source, ("L_candidate_from_major_axis_nm",))
        if l_radial is None:
            l_radial = _batch_scalar(source, ("L_from_observed_radius_nm",))
        if rmse is None:
            rmse = _batch_scalar(source, ("rmse", "geometry_rmse", "residual_rms"))
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        if ln_candidate is None:
            ln_candidate = _batch_candidate_scalar(
                source,
                ("Ln_candidate_from_minor_axis_nm",),
                ("Ln_from_minor_axis_nm", "L_N", "Ln"),
            )
        if lz_candidate is None:
            lz_candidate = _batch_candidate_scalar(
                source,
                ("Lz_candidate_from_draw_axis_nm",),
                ("Lz_from_draw_axis_nm", "L_z", "Lz"),
            )
        if l_major_candidate is None:
            l_major_candidate = _batch_candidate_scalar(
                source,
                ("L_candidate_from_major_axis_nm",),
                ("L_from_major_axis_nm", "L_major"),
            )
    for source in candidate_sources:
        if not isinstance(source, Mapping):
            continue
        if a_candidate is None:
            a_candidate = _batch_candidate_scalar(
                source,
                ("a_candidate", "semi_major_candidate"),
                ("a", "semi_major"),
                include_plain_parameter=True,
            )
        if b_candidate is None:
            b_candidate = _batch_candidate_scalar(
                source,
                ("b_candidate", "semi_minor_candidate"),
                ("b", "semi_minor"),
                include_plain_parameter=True,
            )
        if ratio_candidate is None:
            ratio_candidate = _batch_candidate_scalar(
                source,
                ("axis_ratio_candidate", "b_over_a_candidate"),
                ("axis_ratio", "b_over_a"),
                include_plain_parameter=True,
            )
        if theta_candidate is None:
            theta_candidate = _batch_candidate_scalar(
                source,
                ("theta_deg_candidate", "angle_deg_candidate"),
                ("theta_deg", "angle_deg"),
                include_plain_parameter=True,
            )
        if ln_candidate is None:
            ln_candidate = _batch_candidate_scalar(
                source,
                ("Ln_candidate_from_minor_axis_nm",),
                ("Ln_from_minor_axis_nm", "L_N", "Ln"),
                include_plain_parameter=True,
            )
        if lz_candidate is None:
            lz_candidate = _batch_candidate_scalar(
                source,
                ("Lz_candidate_from_draw_axis_nm",),
                ("Lz_from_draw_axis_nm", "L_z", "Lz"),
                include_plain_parameter=True,
            )
        if l_major_candidate is None:
            l_major_candidate = _batch_candidate_scalar(
                source,
                ("L_candidate_from_major_axis_nm",),
                ("L_from_major_axis_nm",),
                include_plain_parameter=True,
            )
    if ratio_candidate is None and a_candidate not in (None, 0.0) and b_candidate is not None:
        try:
            ratio_candidate = float(b_candidate) / float(a_candidate)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio_candidate = None
    if ratio is None and a not in (None, 0.0) and b is not None:
        try:
            ratio = float(b) / float(a)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = None
    arcs = record.get("arc_sides")
    if not arcs:
        for source in sources:
            if isinstance(source, Mapping) and source.get("arc_sides"):
                arcs = source.get("arc_sides")
                break
    if not arcs:
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            quality_map = source.get("quality")
            metrics = (
                quality_map.get("metrics")
                if isinstance(quality_map, Mapping)
                else source.get("metrics")
            )
            counts = metrics.get("side_counts") if isinstance(metrics, Mapping) else None
            if isinstance(counts, Mapping) and counts:
                present = sum(
                    1
                    for value in counts.values()
                    if isinstance(value, (int, float)) and value > 0
                )
                arcs = f"{present}/4"
                break
    quality = record.get("quality_status")
    if not quality:
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            nested = source.get("quality")
            if isinstance(nested, Mapping):
                quality = nested.get("status", nested.get("engineering_status"))
            if quality:
                break
            quality = source.get("quality_status")
            if quality:
                break
    flag_tokens: list[str] = []
    flags = record.get("flags", ())
    if isinstance(flags, Mapping):
        flags = flags.get("flags", ())
    if isinstance(flags, str):
        flag_tokens.extend(part.strip() for part in flags.split(",") if part.strip())
    else:
        flag_tokens.extend(str(item) for item in (flags or ()) if item)
    for source in (*sources, *candidate_sources):
        if not isinstance(source, Mapping):
            continue
        nested = source.get("flags")
        if isinstance(nested, str):
            flag_tokens.extend(part.strip() for part in nested.split(",") if part.strip())
        elif nested:
            flag_tokens.extend(str(item) for item in nested if item)
        bound_flags = source.get("bound_flags", source.get("bound_status"))
        if isinstance(bound_flags, Mapping):
            if bound_flags.get("axis_ratio"):
                flag_tokens.append("axis_ratio_at_bound")
            if any(value is True for value in bound_flags.values()):
                flag_tokens.append("parameter_at_explicit_bound")
    flags_text = ", ".join(dict.fromkeys(flag_tokens))
    quality_text = str(quality).upper() if quality else None
    ellipse_limited = ellipse_candidate_is_limited(flag_tokens)
    ellipse_kind = classify_ellipse_publication(
        quality_status=quality_text,
        axis_ratio=ratio if ratio is not None else ratio_candidate,
        flags=flag_tokens,
    )
    ln_metadata = _batch_parameter_metadata(
        sources, ("Ln_from_minor_axis_nm", "Ln", "L_N", "Ln_nm")
    )
    lz_metadata = _batch_parameter_metadata(
        sources, ("Lz_from_draw_axis_nm", "Lz", "L_z", "Lz_nm")
    )
    shape_metadata = {
        "a": _batch_parameter_metadata(sources, ("a", "semi_major")),
        "b": _batch_parameter_metadata(sources, ("b", "semi_minor")),
        "axis_ratio": _batch_parameter_metadata(sources, ("axis_ratio", "b_over_a")),
        "theta_deg": _batch_parameter_metadata(sources, ("theta_deg", "angle_deg")),
    }
    shape_values = {"a": a, "b": b, "axis_ratio": ratio, "theta_deg": theta}
    shape_candidates = {
        "a": a_candidate,
        "b": b_candidate,
        "axis_ratio": ratio_candidate,
        "theta_deg": theta_candidate,
    }
    for name, value in shape_values.items():
        metadata = shape_metadata[name]
        candidate_only = (
            metadata.get("status", "").lower() in {"candidate", "unavailable"}
            or metadata.get("confidence", "").lower() in {"limited", "unavailable"}
            or metadata.get("publication_status", "").lower() == "not_assessed"
            or ellipse_limited
            or ellipse_kind != "ellipse"
        )
        if candidate_only and value is not None:
            if shape_candidates[name] is None:
                shape_candidates[name] = value
            shape_values[name] = None
    a, b, ratio, theta = (
        shape_values["a"],
        shape_values["b"],
        shape_values["axis_ratio"],
        shape_values["theta_deg"],
    )
    a_candidate, b_candidate, ratio_candidate, theta_candidate = (
        shape_candidates["a"],
        shape_candidates["b"],
        shape_candidates["axis_ratio"],
        shape_candidates["theta_deg"],
    )
    ln_candidate_only = (
        ln_metadata.get("status", "").lower() in {"candidate", "unavailable"}
        or ln_metadata.get("confidence", "").lower() in {"limited", "unavailable"}
        or ln_metadata.get("publication_status", "").lower() == "not_assessed"
        or ellipse_limited
        or ellipse_kind != "ellipse"
    )
    if ln_candidate_only and ln is not None:
        if ln_candidate is None:
            ln_candidate = ln
        ln = None
    lz_candidate_only = (
        lz_metadata.get("status", "").lower() in {"candidate", "unavailable"}
        or lz_metadata.get("confidence", "").lower() in {"limited", "unavailable"}
        or lz_metadata.get("publication_status", "").lower() == "not_assessed"
        or ellipse_limited
        or ellipse_kind != "ellipse"
    )
    if lz_candidate_only and lz is not None:
        if lz_candidate is None:
            lz_candidate = lz
        lz = None
    ln_state = ln_metadata.get("status", "")
    ln_confidence = ln_metadata.get("confidence", "")
    lz_state = lz_metadata.get("status", "")
    lz_confidence = lz_metadata.get("confidence", "")
    if ln_candidate is not None and not ln_state:
        ln_state = "candidate"
    if ln_candidate is not None and not ln_confidence:
        ln_confidence = "limited"
    if lz_candidate is not None and not lz_state:
        lz_state = "candidate"
    if lz_candidate is not None and not lz_confidence:
        lz_confidence = "limited"
    return {
        "a": a,
        "a_candidate": a_candidate,
        "a_status": shape_metadata["a"].get(
            "status", "candidate" if a_candidate is not None else ""
        ),
        "a_confidence": shape_metadata["a"].get(
            "confidence", "limited" if a_candidate is not None else ""
        ),
        "b": b,
        "b_candidate": b_candidate,
        "b_status": shape_metadata["b"].get(
            "status", "candidate" if b_candidate is not None else ""
        ),
        "b_confidence": shape_metadata["b"].get(
            "confidence", "limited" if b_candidate is not None else ""
        ),
        "axis_ratio": ratio,
        "axis_ratio_candidate": ratio_candidate,
        "axis_ratio_status": shape_metadata["axis_ratio"].get(
            "status", "candidate" if ratio_candidate is not None else ""
        ),
        "axis_ratio_confidence": shape_metadata["axis_ratio"].get(
            "confidence", "limited" if ratio_candidate is not None else ""
        ),
        "theta_deg": theta,
        "theta_deg_candidate": theta_candidate,
        "theta_deg_status": shape_metadata["theta_deg"].get(
            "status", "candidate" if theta_candidate is not None else ""
        ),
        "theta_deg_confidence": shape_metadata["theta_deg"].get(
            "confidence", "limited" if theta_candidate is not None else ""
        ),
        "arcs": arcs,
        "quality": quality_text,
        "ellipse_kind": ellipse_kind,
        "ellipse_limited": ellipse_limited,
        "ln": ln,
        "lz": lz,
        "ln_candidate": ln_candidate,
        "ln_status": ln_state,
        "ln_confidence": ln_confidence,
        "lz_candidate": lz_candidate,
        "lz_status": lz_state,
        "lz_confidence": lz_confidence,
        "l_major_candidate": l_major_candidate,
        "l_radial": l_radial,
        "rmse": rmse,
        "flags": flags_text,
    }


def _result_has_failure(result: Any) -> bool:
    """Return whether a result carries an explicit failure condition."""

    if _read(result, ("cancelled",), False) is True:
        return True
    if _is_butterfly_trace_result(result):
        flags = _read(
            _read(result, ("metrics", "statistics", "summary"), {}),
            ("flags", "flag"),
            _read(result, ("flags", "flag"), []),
        )
        flags = [flags] if isinstance(flags, str) else list(flags or ())
        return any(
            any(
                token in str(flag).lower()
                for token in ("no_engine", "exception", "input_failed", "observables_failed")
            )
            for flag in flags
        )

    try:
        from ..batch import _quality_failure_reason

        if _quality_failure_reason(result) is not None:
            return True
    except Exception:
        pass
    status = str(_read(result, ("status", "solver_status"), "") or "").lower()
    if status in {"fail", "failed", "error"}:
        return True
    quality = str(_read(result, ("quality_status",), "") or "").upper()
    if quality == "FAIL":
        return True
    metrics = _read(result, ("metrics", "statistics", "summary"), {})
    if isinstance(metrics, Mapping) and metrics.get("success") is False:
        return True
    ellipse = _read(result, ("ellipse_fit", "ellipse", "ellipse_result"), None)
    ellipse_status = str(_read(ellipse, ("quality_status", "status"), "") or "").upper()
    ellipse_quality = _read(ellipse, ("quality",), {})
    nested_quality_status = str(
        _read(ellipse_quality, ("status", "engineering_status"), "") or ""
    ).upper()
    if ellipse_status in {"FAIL", "FAILED", "INVALID"} or nested_quality_status in {
        "FAIL",
        "FAILED",
        "INVALID",
    }:
        return True
    flags = _read(metrics, ("flags", "flag"), _read(result, ("flags", "flag"), []))
    if isinstance(flags, str):
        flags = [flags]
    return any(
        any(
            token in str(flag).lower()
            for token in ("failed", "error", "invalid", "no_engine", "exception")
        )
        for flag in (flags or [])
    )


def _batch_warning_has_explicit_failure(record: Mapping[str, Any]) -> bool:
    """Find execution/input failures that a warning lifecycle must not hide."""

    for name in (
        "error",
        "traceback",
        "read_error",
        "input_error",
        "optimizer_error",
        "solver_error",
        "exception",
    ):
        if record.get(name) not in (None, ""):
            return True

    failure_statuses = {"fail", "failed", "error", "exception", "invalid"}
    for name in (
        "read_status",
        "input_status",
        "optimizer_status",
        "solver_status",
        "fit_status",
        "io_status",
    ):
        if str(record.get(name, "") or "").strip().casefold() in failure_statuses:
            return True
    if record.get("success") is False or record.get("optimizer_success") is False:
        return True

    for stage_name in ("optimizer", "solver", "full2d", "metrics"):
        stage = record.get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        if stage.get("success") is False:
            return True
        if str(stage.get("status", "") or "").strip().casefold() in failure_statuses:
            return True

    flags = record.get("flags", ())
    if isinstance(flags, Mapping):
        flags = flags.get("flags", ())
    if isinstance(flags, str):
        flags = (flags,)
    operational_failure_tokens = (
        "optimizer_failed",
        "solver_failed",
        "intensity_fit_failed",
        "read_failed",
        "input_failed",
        "decode_failed",
        "no_engine",
        "exception",
    )
    return any(
        token in str(flag).casefold()
        for flag in (flags or ())
        for token in operational_failure_tokens
    )


def _batch_records_have_failure(records: Any) -> bool:
    """Return whether any batch record is explicitly unsuccessful."""

    for record in _sequence(records):
        if isinstance(record, Mapping):
            status = str(record.get("status", "ok") or "ok").casefold()
            if status not in _BATCH_SUCCESS_STATUSES:
                return True
            if status in _BATCH_LIMITED_STATUSES:
                if _batch_warning_has_explicit_failure(record):
                    return True
                continue
        if _result_has_failure(record):
            return True
    return False


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
        return "—" if not math.isfinite(number) else f"{number:.6g}"
    except (TypeError, ValueError):
        return str(value)


def _numeric(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_finite(value: Any) -> bool:
    try:
        return bool(_np.isfinite(value)) if _np is not None else value == value
    except (TypeError, ValueError):
        return False


def _flatten_evolution_record(record: Any, index: int) -> dict[str, Any]:
    """Flatten batch ``parameters`` specs into scalar evolution columns."""

    if isinstance(record, Mapping):
        row: dict[str, Any] = dict(record)
    else:
        row = {"frame": index, "value": record}

    def scalar_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return next(
                (
                    value.get(key)
                    for key in ("value", "val", "candidate_value", "candidate", "initial", "best")
                    if value.get(key) not in (None, "")
                ),
                None,
            )
        return value

    def flatten_parameter_block(
        block: Any,
        *,
        overwrite: bool,
        candidate_fallback: bool = False,
    ) -> None:
        if not isinstance(block, Mapping):
            return
        for name, value in block.items():
            scalar = scalar_value(value)
            key = str(name)
            current = row.get(key)
            empty_or_nonfinite = current in (None, "") or (
                isinstance(current, (int, float)) and not _is_finite(current)
            )
            if overwrite or key not in row or empty_or_nonfinite:
                row[str(name)] = scalar
                if candidate_fallback and scalar not in (None, ""):
                    row.setdefault(f"{name}_status", "candidate")
                    row.setdefault(f"{name}_confidence", "limited")
                    row.setdefault(f"{name}_publication_status", "not_assessed")
            if isinstance(value, Mapping):
                for metadata in ("status", "confidence", "publication_status", "reason"):
                    metadata_value = value.get(metadata)
                    if metadata_value not in (None, ""):
                        row[f"{name}_{metadata}"] = metadata_value

    flatten_parameter_block(row.pop("parameters", None), overwrite=True)
    flatten_parameter_block(row.get("quantitative_parameters"), overwrite=False)
    flatten_parameter_block(row.get("geometry_parameters"), overwrite=False)
    flatten_parameter_block(
        row.get("candidate_geometry_parameters"),
        overwrite=False,
        candidate_fallback=True,
    )
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        for name, value in metrics.items():
            row.setdefault(str(name), scalar_value(value))
    return row


def _evolution_numeric_keys(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> list[str]:
    ignored = {"time", "time_s", "frame", "index", "status", "path"}
    return [
        str(key)
        for key in keys
        if key not in ignored
        and not str(key).endswith(("_status", "_confidence", "_publication_status", "_reason"))
        and any(_is_finite(_numeric(row.get(key), float("nan"))) for row in rows)
    ]


def _classify_evolution_row(row: Mapping[str, Any], y_key: str) -> str:
    """Classify one flattened evolution row as ``measured``, ``limited``, or ``failed``."""

    status = str(row.get(f"{y_key}_status", row.get("status", "")) or "").strip().lower()
    confidence = str(
        row.get(f"{y_key}_confidence", row.get("confidence", "")) or ""
    ).strip().lower()
    publication = str(
        row.get(f"{y_key}_publication_status", row.get("publication_status", "")) or ""
    ).strip().lower()
    quality = str(row.get("quality_status", row.get("quality", "")) or "").strip().lower()
    frame_status = str(row.get("status", "") or "").strip().lower()
    if frame_status in _BATCH_LIMITED_STATUSES:
        return "failed" if _batch_warning_has_explicit_failure(row) else "limited"
    if (
        frame_status in {"failed", "fail", "error", "invalid", "skipped"}
        or status in {"failed", "fail", "error", "invalid", "skipped"}
        or quality in {"failed", "fail", "error", "invalid"}
    ):
        return "failed"
    if (
        status in {"warning", "warn", "candidate", "estimate", "unavailable"}
        or confidence in {"limited", "unavailable"}
        or publication == "not_assessed"
        or quality in {"warning", "warn"}
    ):
        return "limited"
    return "measured"


__all__ = [
    "_BATCH_LIMITED_STATUSES",
    "_BATCH_STATUS_KEYS",
    "_BATCH_SUCCESS_STATUSES",
    "_BATCH_TABLE_HEADERS",
    "_batch_candidate_scalar",
    "_batch_parameter_metadata",
    "_batch_record_geometry",
    "_batch_records_have_failure",
    "_batch_scalar",
    "_batch_warning_has_explicit_failure",
    "_classify_evolution_row",
    "_evolution_numeric_keys",
    "_flatten_evolution_record",
    "_format_metric",
    "_is_butterfly_trace_result",
    "_is_finite",
    "_numeric",
    "_result_has_failure",
    "_result_value",
    "_sequence",
]
