"""Fit session review, candidate snapshot, comparison state, and manual-evidence helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ._batch_records import _result_has_failure, _result_value
from ._common import _read, json_safe
from .qt_compat import QT_AVAILABLE, QtWidgets

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None


def _new_fit_session() -> dict[str, Any]:
    """Return the small, JSON-friendly manual-review session state."""

    return {
        "manual_status": "unreviewed",
        "reviewed_by": "",
        "reviewed_at": None,
        "review_notes": "",
        "optimize_before": None,
        "optimize_after": None,
        "accepted_parameters": None,
        "snapshots": [],
    }


def _utc_timestamp() -> str:
    """Return an explicit UTC timestamp for human-review audit fields."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _full2d_model_context(result: Any) -> dict[str, Any]:
    """Extract the actual Optimize fit context from service and legacy results."""

    metrics = _result_value(result, ("metrics", "statistics", "summary"), {})
    if not isinstance(metrics, Mapping):
        metrics = {}

    def metric_or_root(metric_name: str, root_names: tuple[str, ...]) -> Any:
        value = metrics.get(metric_name)
        if value is not None:
            return value
        return _result_value(result, root_names, None)

    success = metric_or_root("success", ("pixel_model_success", "success"))
    explicit_status = _result_value(
        result, ("status", "solver_status", "pixel_model_status"), None
    )
    if explicit_status in (None, ""):
        explicit_status = _read(metrics, ("status", "solver_status"), None)
    if explicit_status not in (None, "") and str(explicit_status).lower() != "unknown":
        status = str(explicit_status)
    elif success is not None:
        if isinstance(success, str):
            normalized = success.strip().lower()
            if normalized in {"false", "fail", "failed", "no", "0"}:
                success = False
            elif normalized in {"true", "success", "ok", "yes", "1"}:
                success = True
        status = "success" if bool(success) else "failed"
    else:
        status = "unknown"

    parameters = _result_value(result, ("parameters", "fitted_parameters", "params"), None)
    if isinstance(parameters, Mapping):
        parameters = {
            str(name): (
                _read(value, ("value", "val", "initial"), None)
                if isinstance(value, Mapping)
                else value
            )
            for name, value in parameters.items()
        }
    diagnostics = {
        "rmse": metric_or_root("rmse", ("pixel_model_rmse", "full2d_rmse", "rmse")),
        "weighted_rmse": metric_or_root(
            "weighted_rmse", ("pixel_model_weighted_rmse", "weighted_rmse")
        ),
        "condition_number": metric_or_root(
            "condition_number",
            ("pixel_model_condition_number", "condition_number"),
        ),
        "bound_flags": deepcopy(
            metric_or_root("bound_flags", ("pixel_model_bound_flags", "bound_flags"))
        ),
        "effective_bounds": deepcopy(
            metric_or_root(
                "effective_bounds", ("pixel_model_effective_bounds", "effective_bounds")
            )
        ),
        "bound_flag_intensity_scale": metric_or_root(
            "bound_flag_intensity_scale",
            ("pixel_model_bound_flag_intensity_scale", "bound_flag_intensity_scale"),
        ),
    }
    reference_axis_deg = metric_or_root(
        "reference_axis_deg", ("pixel_model_reference_axis_deg", "reference_axis_deg")
    )
    return {
        "status": status,
        "success": success,
        "parameters": parameters,
        "reference_axis_deg": reference_axis_deg,
        "diagnostics": diagnostics,
    }


def fit_state_signature(window: Any) -> str:
    """Return the small deterministic identity of the current fit inputs."""

    state = {
        "parameters": window.parameter_model.parameter_dict(),
        "analysis": window.analysis_settings,
        "input": {
            "path": window._source_path,
            "frame": window._frame,
            "dataset": window._dataset,
        },
        "poni": window._poni_path,
        "mask": {
            "path": window._mask_path,
            "frame": window._mask_frame,
            "dataset": window._mask_dataset,
        },
        "rois": window._roi_specs,
        "input_records": window._loaded_input_records,
    }
    return json.dumps(
        json_safe(state),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def is_geometry_only_result(window: Any, result: Any = None) -> bool:
    """Identify geometry actions that intentionally omit intensity fit."""

    candidate = window._last_result if result is None else result
    action = _result_value(
        candidate,
        ("geometry_action", "geometry_stage", "analysis_stage"),
        None,
    )
    if str(action or "").strip().lower() in {
        "remeasure",
        "refine",
        "geometry",
        "geometry_only",
    }:
        return True
    stage = _result_value(candidate, ("stage",), None)
    if str(stage or "").strip().lower() in {
        "geometry",
        "geometry_only",
        "measurement",
    }:
        return True
    model_status = _result_value(candidate, ("model_status",), None)
    if str(model_status or "").strip().lower() in {
        "unfitted_preview",
        "geometry_only",
        "not_run",
        "unfitted",
    }:
        return True
    return str(getattr(window, "_last_result_kind", "")).lower() in {
        "measure_geometry",
        "refine_geometry",
    }


def result_has_fit_images(
    window: Any,
    result: Any,
    *,
    include_failed: bool = False,
) -> bool:
    """Return whether one result can support visual human review."""

    if (
        result is None
        or (_result_has_failure(result) and not include_failed)
        or window._is_geometry_only_result(result)
    ):
        return False
    observed = _result_value(result, ("observed", "data", "image"), window._observed)
    model = _result_value(
        result,
        ("model", "model_image", "predicted", "fit", "intensity", "simulation"),
        result if not isinstance(result, Mapping) else None,
    )
    if observed is None or model is None:
        return False
    if _np is None:
        return True
    try:
        observed_array = _np.asarray(observed)
        model_array = _np.asarray(model)
    except (TypeError, ValueError):
        return False
    return (
        observed_array.ndim == 2
        and observed_array.size > 0
        and model_array.shape == observed_array.shape
    )


def current_result_is_reviewable(window: Any) -> bool:
    """Require a successful current result, not a stale displayed image."""

    return (
        window._last_result_signature is not None
        and window._last_result_signature == window._fit_state_signature()
        and window._result_has_fit_images(window._last_result)
    )


def capture_fit_context(
    window: Any,
    parameters: Mapping[str, Any] | None = None,
    *,
    include_arrays: bool = True,
) -> dict[str, Any]:
    """Capture the current fit context before a background request."""

    if parameters is None:
        parameters = window.parameter_model.parameter_dict()
    input_context: dict[str, Any] = {
        "path": window._source_path,
        "frame": window._frame,
        "dataset": window._dataset,
    }
    mask_context: dict[str, Any] = {
        "path": window._mask_path,
        "frame": window._mask_frame,
        "dataset": window._mask_dataset,
    }
    qmap_context: dict[str, Any] = {}
    if include_arrays:
        input_context["data"] = deepcopy(window._observed)
        mask_context["file"] = deepcopy(window._file_mask)
        mask_context["external"] = deepcopy(window._external_mask)
        qmap_context = {
            "qx": deepcopy(window._qx),
            "qy": deepcopy(window._qy),
            "value": deepcopy(window._qmap),
        }
    context: dict[str, Any] = {
        "parameters": deepcopy(parameters),
        "input": input_context,
        "frame": window._frame,
        "dataset": window._dataset,
        "poni": deepcopy(window._poni_path),
        "mask": mask_context,
        "analysis": deepcopy(window.analysis_settings),
        "roi": {
            "exclusion": deepcopy(window._exclusion_roi),
            "specs": deepcopy(window._roi_specs),
        },
    }
    if qmap_context:
        context["qmap"] = qmap_context
    return context


def fit_context_for_project(context: Any) -> Any:
    """Remove detector-sized arrays from a persisted fit context."""

    if not isinstance(context, Mapping):
        return None
    clean = deepcopy(dict(context))
    input_context = clean.get("input")
    if isinstance(input_context, Mapping):
        input_context = dict(input_context)
        input_context.pop("data", None)
        clean["input"] = input_context
    mask_context = clean.get("mask")
    if isinstance(mask_context, Mapping):
        mask_context = dict(mask_context)
        for key in ("file", "external", "data", "array"):
            mask_context.pop(key, None)
        clean["mask"] = mask_context
    clean.pop("qmap", None)
    return json_safe(clean)


def fit_result_summary(result: Any) -> dict[str, Any]:
    """Keep review-relevant result metadata without detector images."""

    if not isinstance(result, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for key in ("status", "solver_status", "quality_status", "flags", "q_unit"):
        value = result.get(key)
        if value is not None:
            summary[key] = deepcopy(value)
    observed = _result_value(result, ("observed", "data", "image"), None)
    if observed is not None:
        try:
            summary["shape"] = [int(value) for value in _np.asarray(observed).shape]
        except (AttributeError, TypeError, ValueError):
            pass
    metrics = _read(result, ("metrics", "statistics", "summary"), None)
    if isinstance(metrics, Mapping):
        def metadata_value(value: Any) -> Any:
            if _np is not None and isinstance(value, _np.ndarray):
                return None
            if isinstance(value, Mapping):
                return {str(key): metadata_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [metadata_value(item) for item in value]
            return deepcopy(value)

        summary["metrics"] = {
            str(key): metadata_value(value) for key, value in metrics.items()
        }
    return json_safe(summary)


def normalise_fit_session(source: Any) -> dict[str, Any]:
    """Load a schema-1/2 session while treating absent fields as empty."""

    session = _new_fit_session()
    if not isinstance(source, Mapping):
        return session
    status = str(source.get("manual_status", "unreviewed") or "unreviewed").lower()
    session["manual_status"] = (
        status if status in {"unreviewed", "accepted", "rejected"} else "unreviewed"
    )
    session["reviewed_by"] = str(source.get("reviewed_by", source.get("reviewer", "")) or "")
    session["reviewed_at"] = source.get("reviewed_at")
    session["review_notes"] = str(source.get("review_notes", "") or "")
    for key in ("optimize_before", "optimize_after"):
        value = source.get(key)
        session[key] = deepcopy(value) if isinstance(value, Mapping) else None
    accepted = source.get("accepted_parameters")
    session["accepted_parameters"] = (
        deepcopy(accepted) if isinstance(accepted, Mapping) else None
    )
    snapshots = source.get("snapshots", [])
    if isinstance(snapshots, Iterable) and not isinstance(snapshots, (str, bytes, Mapping)):
        for snapshot in snapshots:
            if not isinstance(snapshot, Mapping):
                continue
            parameters = snapshot.get("parameters")
            if not isinstance(parameters, Mapping):
                continue
            item: dict[str, Any] = {
                "order": len(session["snapshots"]) + 1,
                "note": str(snapshot.get("note", "") or ""),
                "created_at": snapshot.get("created_at"),
                "parameters": deepcopy(parameters),
            }
            if isinstance(snapshot.get("context"), Mapping):
                item["context"] = deepcopy(snapshot["context"])
            session["snapshots"].append(item)
    return session


def fit_session_for_project(session: Mapping[str, Any]) -> dict[str, Any]:
    """Return fit-session JSON without detector-sized in-memory data."""

    persisted = _new_fit_session()
    for key in ("manual_status", "reviewed_by", "reviewed_at", "review_notes"):
        persisted[key] = deepcopy(session.get(key, persisted[key]))
    for key in ("optimize_before", "optimize_after"):
        persisted[key] = fit_context_for_project(session.get(key))
    accepted = session.get("accepted_parameters")
    persisted["accepted_parameters"] = (
        deepcopy(accepted) if isinstance(accepted, Mapping) else None
    )
    snapshots = session.get("snapshots", [])
    if isinstance(snapshots, list):
        for index, snapshot in enumerate(snapshots):
            if not isinstance(snapshot, Mapping) or not isinstance(
                snapshot.get("parameters"), Mapping
            ):
                continue
            item: dict[str, Any] = {
                "order": index + 1,
                "note": str(snapshot.get("note", "") or ""),
                "created_at": snapshot.get("created_at"),
                "parameters": deepcopy(snapshot["parameters"]),
            }
            if isinstance(snapshot.get("context"), Mapping):
                item["context"] = fit_context_for_project(snapshot["context"])
            persisted["snapshots"].append(item)
    return json_safe(persisted)


def sync_fit_session_controls(window: Any, *, preserve_edits: bool = False) -> None:
    """Refresh the compact right-dock controls from session state."""

    if not hasattr(window, "manual_status_label"):
        return
    status = str(window._fit_session.get("manual_status", "unreviewed"))
    window.manual_status_label.setText(window._tr(f"manual.{status}"))
    if not preserve_edits:
        window.reviewer_edit.setText(str(window._fit_session.get("reviewed_by", "") or ""))
        window.review_notes_edit.setText(str(window._fit_session.get("review_notes", "") or ""))
    selected = window.snapshot_combo.currentData() if window.snapshot_combo.count() else None
    window.snapshot_combo.blockSignals(True)
    window.snapshot_combo.clear()
    snapshots = window._fit_session.get("snapshots", [])
    if isinstance(snapshots, list):
        for index, snapshot in enumerate(snapshots):
            if not isinstance(snapshot, Mapping):
                continue
            note = str(snapshot.get("note", "") or "")
            label = window._tr(
                "snapshot.label",
                index=index + 1,
                note=note or window._tr("snapshot.no_note"),
            )
            window.snapshot_combo.addItem(label, index)
    if window.snapshot_combo.count():
        if isinstance(selected, int) and 0 <= selected < window.snapshot_combo.count():
            window.snapshot_combo.setCurrentIndex(selected)
        else:
            window.snapshot_combo.setCurrentIndex(window.snapshot_combo.count() - 1)
    window._refresh_snapshot_item_tooltips()
    window.snapshot_combo.blockSignals(False)
    window.restore_snapshot_button.setEnabled(bool(window.snapshot_combo.count()))
    window.restore_before_optimize_button.setEnabled(
        isinstance(window._fit_session.get("optimize_before"), Mapping)
    )
    reviewable = window._current_result_is_reviewable()
    window.accept_current_button.setEnabled(reviewable)
    window.reject_current_button.setEnabled(reviewable)
    if hasattr(window, "export_evidence_action"):
        window.export_evidence_action.setEnabled(reviewable)


def mark_manual_unreviewed(window: Any, *, clear_candidate: bool = True) -> None:
    """Invalidate a previous human decision after a state edit."""

    if window._fit_session_restore_active:
        return
    if hasattr(window, "lamellar_page"):
        window.lamellar_page.invalidate()
    window._fit_session["manual_status"] = "unreviewed"
    window._fit_session["reviewed_by"] = ""
    window._fit_session["reviewed_at"] = None
    window._fit_session["review_notes"] = ""
    window._fit_session["accepted_parameters"] = None
    window._last_result_signature = None
    window._last_model_diagnostic_signature = None
    window._last_result_input_records = None
    if clear_candidate:
        window._fit_session["optimize_after"] = None
    window._sync_fit_session_controls()


def review_candidate(window: Any, status: str) -> bool:
    """Apply an explicit Accept/Reject decision to the current fit."""

    reviewer = window.reviewer_edit.text().strip()
    if not reviewer:
        window._set_status("status.reviewer_required", flags="review_required")
        return False
    if not window._current_result_is_reviewable():
        window._set_status("status.review_result_required", flags="review_required")
        return False
    window._fit_session["manual_status"] = status
    window._fit_session["reviewed_by"] = reviewer
    window._fit_session["reviewed_at"] = _utc_timestamp()
    window._fit_session["review_notes"] = window.review_notes_edit.text().strip()
    window._fit_session["accepted_parameters"] = (
        deepcopy(window.parameter_model.parameter_dict()) if status == "accepted" else None
    )
    window._sync_fit_session_controls()
    window._set_status(
        "status.manual_accepted" if status == "accepted" else "status.manual_rejected",
        flags=f"manual_{status}",
    )
    return True


def manual_evidence_result(window: Any) -> dict[str, Any]:
    """Build the exporter payload from the committed UI result."""

    if isinstance(window._last_result, Mapping):
        payload = dict(window._last_result)
    elif window._last_result is not None:
        payload = {"model": window._last_result}
    else:
        raise ValueError("No Preview/Optimize result is available")
    observed = _result_value(payload, ("observed", "data", "image"), window._observed)
    model = _result_value(
        payload,
        ("model", "predicted", "fit", "intensity", "simulation"),
        None,
    )
    residual = _result_value(payload, ("residual", "difference", "resid"), None)
    if residual is None and observed is not None and model is not None and _np is not None:
        observed_array = _np.asarray(observed, dtype=float)
        model_array = _np.asarray(model, dtype=float)
        if observed_array.shape == model_array.shape:
            residual = observed_array - model_array
    payload["observed"] = observed
    payload["model"] = model
    payload["residual"] = residual
    payload["parameters"] = deepcopy(window.parameter_model.parameter_dict())
    payload.setdefault("qx", window._qx)
    payload.setdefault("qy", window._qy)
    payload["q_unit"] = window._active_q_unit(payload)
    payload.setdefault("valid_mask", _read(window._qmap, ("valid_mask", "valid"), None))
    payload.setdefault("external_mask", window._external_mask)
    payload.setdefault("ridge_points", window._fit_ridge_points)
    payload.setdefault("ellipses", window._observed_fit_ellipses)
    return payload


def manual_evidence_context(window: Any) -> dict[str, Any]:
    """Return arrays/selectors needed by the seven-file exporter."""

    return {
        "source": window._source_path,
        "frame": window._frame,
        "dataset": window._dataset,
        "poni": window._poni_path,
        "mask_path": window._mask_path,
        "roi": deepcopy(window._roi_specs),
        "analysis": deepcopy(window.analysis_settings),
        "result_kind": window._last_result_kind,
        "qmap": window._qmap,
        "qx": window._qx,
        "qy": window._qy,
        "q_unit": window._active_q_unit(window._last_result),
        "valid_mask": _read(window._qmap, ("valid_mask", "valid"), None),
        "external_mask": window._external_mask,
        "current_model_ellipses": deepcopy(window._model_ellipses),
        "fit_input_records": deepcopy(window._last_result_input_records),
    }


def export_manual_evidence(
    window: Any,
    path: str | Path | bool | None = None,
    *,
    force: bool = False,
) -> bool:
    """Export the current fit as exactly seven auditable files."""

    if not window._current_result_is_reviewable():
        window._set_status("status.evidence_stale", flags="evidence_stale")
        return False
    status = str(window._fit_session.get("manual_status", "unreviewed"))
    if status in {"accepted", "rejected"} and (
        window.reviewer_edit.text().strip()
        != str(window._fit_session.get("reviewed_by", "") or "")
        or window.review_notes_edit.text().strip()
        != str(window._fit_session.get("review_notes", "") or "")
    ):
        window._set_status("status.review_fields_changed", flags="review_required")
        return False
    if isinstance(path, bool) or path is None:
        if not QT_AVAILABLE:
            return False
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            window,
            window._tr("dialog.evidence_folder"),
            "",
        )
        if not chosen:
            return False
        path = chosen
    try:
        from ..manual_evidence import export_manual_fit

        written = export_manual_fit(
            window._manual_evidence_result(),
            Path(path),
            context=window._manual_evidence_context(),
            review={
                "manual_status": status,
                "reviewed_by": window._fit_session.get("reviewed_by"),
                "reviewed_at": window._fit_session.get("reviewed_at"),
                "review_notes": window._fit_session.get("review_notes", ""),
            },
            force=bool(force),
        )
    except (OSError, TypeError, ValueError) as exc:
        window._set_status("status.evidence_failed", flags="evidence_error", error=exc)
        return False
    window._last_evidence_paths = dict(written)
    window._set_status(
        "status.evidence_exported",
        flags="evidence_exported",
        path=Path(path),
    )
    return True


def _reset_session_review_fields(session: dict[str, Any]) -> None:
    session["manual_status"] = "unreviewed"
    session["reviewed_by"] = ""
    session["reviewed_at"] = None
    session["review_notes"] = ""
    session["accepted_parameters"] = None
    session["optimize_after"] = None


def restore_before_optimize(window: Any) -> bool:
    """Restore the most recent Optimize-before parameter table."""

    before = window._fit_session.get("optimize_before")
    parameters = before.get("parameters") if isinstance(before, Mapping) else None
    if not isinstance(parameters, Mapping):
        window._set_status("status.no_optimize_snapshot", flags="snapshot_missing")
        return False
    window._invalidate_pending_work(clear_fit=True)
    window._fit_session_restore_active = True
    try:
        window.parameter_model.set_rows(deepcopy(parameters))
        setter = getattr(window.engine, "set_parameters", None)
        if callable(setter):
            setter(window.parameter_model.parameter_dict())
    finally:
        window._fit_session_restore_active = False
    _reset_session_review_fields(window._fit_session)
    window._sync_fit_session_controls()
    window._refresh_model_overlay()
    window._set_status("status.restored_before_optimize", flags="snapshot_restored")
    return True


def save_snapshot(window: Any, note: str | None = None) -> bool:
    """Append a detached, ordered parameter snapshot with a note."""

    if isinstance(note, bool) or note is None:
        note = window.snapshot_note_edit.text()
    note = str(note).strip()
    if not note:
        window._set_status("status.snapshot_note_required", flags="snapshot_note_required")
        return False
    snapshots = window._fit_session.setdefault("snapshots", [])
    snapshot = {
        "order": len(snapshots) + 1,
        "note": note,
        "created_at": _utc_timestamp(),
        "parameters": deepcopy(window.parameter_model.parameter_dict()),
        "context": window._capture_fit_context(parameters={}, include_arrays=False),
    }
    snapshots.append(snapshot)
    window._sync_fit_session_controls()
    window.snapshot_combo.setCurrentIndex(window.snapshot_combo.count() - 1)
    window.snapshot_note_edit.clear()
    window._set_status("status.snapshot_saved", index=snapshot["order"], note=note)
    return True


def restore_snapshot(window: Any, index: int | None = None) -> bool:
    """Restore one saved snapshot and clear views from the old fit."""

    if isinstance(index, bool):
        index = None
    snapshots = window._fit_session.get("snapshots", [])
    if not isinstance(snapshots, list) or not snapshots:
        window._set_status("status.no_snapshot", flags="snapshot_missing")
        return False
    if index is None:
        selected = window.snapshot_combo.currentData()
        index = int(selected) if selected is not None else window.snapshot_combo.currentIndex()
    try:
        snapshot = snapshots[int(index)]
    except (IndexError, TypeError, ValueError):
        window._set_status("status.snapshot_invalid", flags="snapshot_invalid")
        return False
    parameters = snapshot.get("parameters") if isinstance(snapshot, Mapping) else None
    if not isinstance(parameters, Mapping):
        window._set_status("status.snapshot_no_parameters", flags="snapshot_invalid")
        return False
    window._invalidate_pending_work(clear_fit=True)
    window._fit_session_restore_active = True
    try:
        window.parameter_model.set_rows(deepcopy(parameters))
        setter = getattr(window.engine, "set_parameters", None)
        if callable(setter):
            setter(window.parameter_model.parameter_dict())
    finally:
        window._fit_session_restore_active = False
    _reset_session_review_fields(window._fit_session)
    window._sync_fit_session_controls()
    window.snapshot_combo.setCurrentIndex(int(index))
    window._refresh_model_overlay()
    note = str(snapshot.get("note", "") or "")
    window._set_status(
        "status.snapshot_restored",
        flags="snapshot_restored",
        index=int(index) + 1,
        note=note,
    )
    return True


__all__ = [
    "_full2d_model_context",
    "_new_fit_session",
    "_utc_timestamp",
    "capture_fit_context",
    "current_result_is_reviewable",
    "export_manual_evidence",
    "fit_context_for_project",
    "fit_result_summary",
    "fit_session_for_project",
    "fit_state_signature",
    "is_geometry_only_result",
    "manual_evidence_context",
    "manual_evidence_result",
    "mark_manual_unreviewed",
    "normalise_fit_session",
    "restore_before_optimize",
    "restore_snapshot",
    "result_has_fit_images",
    "review_candidate",
    "save_snapshot",
    "sync_fit_session_controls",
]
