"""Project document payload construction, validation, and atomic rollback orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from ._batch_records import _result_value
from ._common import _read, confirm_save_path, json_safe
from .project_document import ProjectDocumentController
from .qt_compat import QT_AVAILABLE, QtWidgets

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None

_DIRECT_DOCUMENT_FIELDS: tuple[str, ...] = (
    "_source_path",
    "_legacy_geometry",
    "_frame",
    "_dataset",
    "_mask_frame",
    "_mask_dataset",
    "_poni_path",
    "_mask_path",
    "_observed",
    "_qx",
    "_qy",
    "_qmap",
    "_file_mask",
    "_external_mask",
    "_last_result",
    "_last_result_signature",
    "_last_model_diagnostic_signature",
    "_last_result_kind",
    "_geometry_only_result",
    "_last_error",
    "_display_scale",
    "_display_percentile",
    "_project_path",
    "_config_path",
    "_auto_scale_initial",
    "_batch_cancel_event",
    "measurement_observables",
    "evolution_y_key",
)

_COPIED_DOCUMENT_FIELDS: tuple[str, ...] = (
    "_exclusion_roi",
    "_roi_specs",
    "_last_result_input_records",
    "_pending_input_records",
    "_loaded_input_records",
    "_last_evidence_paths",
    "_fit_ridge_points",
    "_rejected_ridge_points",
    "_observed_fit_ellipses",
    "_model_ellipses",
    "last_metrics",
    "_analysis_settings",
    "evolution_records",
    "_evolution_rows",
    "batch_frames",
    "_fit_session",
    "_metric_display",
    "_status_values",
)


def _resolve_project_path(value: Any, base: Path) -> Any:
    """Compatibility alias for the Qt-free project document boundary."""

    return ProjectDocumentController.resolve_path(value, base)


def _resolve_project_frame(value: Any, base: Path) -> Any:
    """Compatibility alias for the Qt-free project document boundary."""

    return ProjectDocumentController.resolve_frame(value, base)


def snapshot_project_value(value: Any) -> Any:
    """Copy project metadata while retaining detector-array ownership."""

    if _np is not None and isinstance(value, _np.ndarray):
        return value
    if isinstance(value, Mapping):
        return {key: snapshot_project_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_project_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_project_value(item) for item in value)
    if isinstance(value, set):
        return {snapshot_project_value(item) for item in value}
    try:
        return deepcopy(value)
    except Exception:
        return value


def snapshot_engine_document(engine: Any) -> dict[str, Any]:
    """Snapshot engine state via its explicit ``snapshot_state`` contract."""

    snapshot_fn = getattr(engine, "snapshot_state", None)
    if callable(snapshot_fn):
        return {"state": snapshot_fn()}
    # Lightweight test doubles without snapshot_state may carry only public
    # attributes; never inspect private ButterflyAnalysisService internals.
    engine_dict = getattr(engine, "__dict__", None)
    if isinstance(engine_dict, Mapping):
        public_attrs = {
            key: snapshot_project_value(val)
            for key, val in engine_dict.items()
            if not str(key).startswith("_") and not callable(val)
        }
        if public_attrs:
            return {"public_attrs": public_attrs}
    return {}


def restore_engine_document(engine: Any, snapshot: Mapping[str, Any]) -> None:
    """Restore engine state via its explicit ``restore_state`` contract."""

    if "state" in snapshot:
        restore_fn = getattr(engine, "restore_state", None)
        if callable(restore_fn):
            restore_fn(snapshot["state"])
        return
    public_attrs = snapshot.get("public_attrs")
    if isinstance(public_attrs, Mapping):
        for name, value in public_attrs.items():
            try:
                setattr(engine, name, value)
            except Exception:
                continue


def project_to_dict(window: Any) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "parameters": window.parameter_model.parameter_dict(),
        "analysis": json_safe(window.analysis_settings),
        "display": json_safe(window.display_settings),
        "input": window._source_path,
        "poni": window._poni_path,
        "frame": window._frame,
        "dataset": window._dataset,
        "mask": window._mask_path,
        "mask_frame": window._mask_frame,
        "mask_dataset": window._mask_dataset,
        "roi_exclusion": json_safe(window._exclusion_roi),
        "rois": list(window._roi_specs),
        "fit_session": window._fit_session_for_project(),
        "lamellar_view": window.lamellar_page.document(),
        "legacy_geometry": window._legacy_geometry,
        "local_measurements": window.local_measurement_page.document(),
        "azimuthal_view": window.azimuthal_page.document(),
        "density2d_view": window.density2d_page.document(),
        "batch": {
            "mode": window.batch_mode_combo.currentData(),
            "stage": window.batch_stage_combo.currentData(),
            "full2d": window.batch_stage_combo.currentData() == "full2d",
            "stream": window.batch_stream_check.isChecked(),
            "frames": json_safe(list(window.batch_frames)),
            "manifest": window.batch_manifest_edit.text() or None,
            "checkpoint": window.batch_checkpoint_edit.text() or None,
            "resume": window.batch_resume_check.isChecked(),
            "series": window.batch_series_edit.text().strip() or None,
            "start": (
                None if window.batch_start_spin.value() < 0 else window.batch_start_spin.value()
            ),
            "stop": (
                None if window.batch_stop_spin.value() < 0 else window.batch_stop_spin.value()
            ),
            "stride": window.batch_stride_spin.value(),
            "range": window.batch_range_edit.text().strip() or None,
            "output": window.batch_output_edit.text() or None,
        },
        "metadata": {
            "project_path": str(window._project_path) if window._project_path else None,
            "config_path": window._config_path,
            "qmap_shape": (
                list(getattr(window._qmap, "shape", ())) if window._qmap is not None else None
            ),
        },
    }


def snapshot_project_document(window: Any) -> dict[str, Any]:
    """Capture document references before a potentially failing load."""

    snapshot = {
        name: getattr(window, name)
        for name in _DIRECT_DOCUMENT_FIELDS
        if hasattr(window, name)
    }
    snapshot.update(
        {
            name: snapshot_project_value(getattr(window, name))
            for name in _COPIED_DOCUMENT_FIELDS
            if hasattr(window, name)
        }
    )
    snapshot["_displayed_flags_text"] = window._displayed_flags_text
    snapshot["_status_key"] = window._status_key
    snapshot["_fit_session_restore_active"] = window._fit_session_restore_active
    snapshot["ui_parameter_values"] = snapshot_project_value(
        window.parameter_model.parameter_dict()
    )
    snapshot["ui_analysis_settings"] = snapshot_project_value(window.analysis_settings)
    snapshot["ui_display_settings"] = snapshot_project_value(window.display_settings)
    snapshot["lamellar_view"] = window.lamellar_page.snapshot_state()
    if hasattr(window, "butterfly_workbench"):
        snapshot["butterfly_workbench"] = window.butterfly_workbench.snapshot_state()
    snapshot["local_measurements"] = window.local_measurement_page.document()
    snapshot["azimuthal_view"] = window.azimuthal_page.document()
    snapshot["density2d_view"] = window.density2d_page.document()
    snapshot["ui_batch"] = {
        "mode": window.batch_mode_combo.currentData(),
        "stage": window.batch_stage_combo.currentData(),
        "stream": window.batch_stream_check.isChecked(),
        "manifest": window.batch_manifest_edit.text(),
        "checkpoint": window.batch_checkpoint_edit.text(),
        "output": window.batch_output_edit.text(),
        "series": window.batch_series_edit.text(),
        "range": window.batch_range_edit.text(),
        "resume": window.batch_resume_check.isChecked(),
        "start": window.batch_start_spin.value(),
        "stop": window.batch_stop_spin.value(),
        "stride": window.batch_stride_spin.value(),
        "frames": list(window.batch_frames),
    }
    snapshot["_view_active_q_window"] = getattr(window.views, "_active_q_window", None)
    snapshot["engine"] = window._snapshot_engine_document()
    return snapshot


def restore_project_document(window: Any, snapshot: Mapping[str, Any]) -> None:
    parameter_values = snapshot.get("ui_parameter_values")
    if isinstance(parameter_values, Mapping):
        window.parameter_model.set_rows(parameter_values)
    analysis_settings = snapshot.get("ui_analysis_settings")
    if isinstance(analysis_settings, Mapping):
        window.set_analysis_settings(analysis_settings, trigger_preview=False)
    display_settings = snapshot.get("ui_display_settings")
    if isinstance(display_settings, Mapping):
        window.set_display_settings(display_settings)
    batch_state = snapshot.get("ui_batch")
    if isinstance(batch_state, Mapping):
        mode_index = window.batch_mode_combo.findData(batch_state.get("mode"))
        if mode_index >= 0:
            window.batch_mode_combo.setCurrentIndex(mode_index)
        stage_index = window.batch_stage_combo.findData(batch_state.get("stage"))
        if stage_index >= 0:
            window.batch_stage_combo.setCurrentIndex(stage_index)
        window.batch_stream_check.setChecked(bool(batch_state.get("stream")))
        for widget, key in (
            (window.batch_manifest_edit, "manifest"),
            (window.batch_checkpoint_edit, "checkpoint"),
            (window.batch_output_edit, "output"),
            (window.batch_series_edit, "series"),
            (window.batch_range_edit, "range"),
        ):
            widget.setText(str(batch_state.get(key) or ""))
        window.batch_resume_check.setChecked(bool(batch_state.get("resume")))
        window.batch_start_spin.setValue(int(batch_state.get("start", -1)))
        window.batch_stop_spin.setValue(int(batch_state.get("stop", -1)))
        window.batch_stride_spin.setValue(int(batch_state.get("stride", 1)))
        window.set_batch_frames(batch_state.get("frames", ()))
    for name in _DIRECT_DOCUMENT_FIELDS:
        if name in snapshot:
            setattr(window, name, snapshot[name])
    for name in _COPIED_DOCUMENT_FIELDS:
        if name in snapshot:
            setattr(window, name, snapshot_project_value(snapshot[name]))
    window._displayed_flags_text = str(
        snapshot.get("_displayed_flags_text", window._displayed_flags_text)
    )
    window._status_key = str(snapshot.get("_status_key", window._status_key))
    window._fit_session_restore_active = bool(
        snapshot.get("_fit_session_restore_active", window._fit_session_restore_active)
    )
    window._restore_engine_document(snapshot.get("engine", {}))

    window.views.clear_fit()
    result = window._last_result
    observed = _result_value(result, ("observed", "data", "image"), window._observed)
    model = _result_value(
        result,
        ("model", "predicted", "fit", "intensity", "simulation"),
        None,
    )
    residual = _result_value(result, ("residual", "difference", "resid"), None)
    if window._geometry_only_result:
        model = None
        residual = None
    if residual is None and observed is not None and model is not None and _np is not None:
        try:
            observed_array = _np.asarray(observed)
            model_array = _np.asarray(model)
            if observed_array.shape == model_array.shape:
                residual = observed_array - model_array
        except (TypeError, ValueError):
            residual = None
    q_unit = window._active_q_unit(result)
    window._refresh_q_parameter_units(q_unit)
    window.views.set_images(
        observed,
        model,
        residual,
        qx=_result_value(result, ("qx", "qx_nm_inv"), window._qx),
        qy=_result_value(result, ("qy", "qy_nm_inv"), window._qy),
        q_unit=q_unit,
        valid_mask=_result_value(
            result,
            ("valid_mask",),
            _read(window._qmap, ("valid_mask", "valid"), None),
        ),
        external_mask=_result_value(
            result,
            ("mask", "external_mask"),
            window._external_mask,
        ),
    )
    window.views.set_roi(window._roi_specs)
    window.views.set_overlay(
        window._fit_ridge_points,
        window._observed_fit_ellipses,
        model_ellipses=window._model_ellipses,
        rejected_ridge_points=window._rejected_ridge_points,
    )
    window.views.set_display_settings(window._display_scale, window._display_percentile)
    q_window = snapshot.get("_view_active_q_window")
    if q_window is None:
        window.views.set_q_view(full=True)
    else:
        window.views.set_q_view(q_window)
    window._sync_roi_widgets()
    window._sync_fit_session_controls(preserve_edits=True)
    window._render_metric_labels()
    window._render_status()
    if isinstance(window._last_result, Mapping):
        window._update_measurements(window._last_result)
    else:
        window._update_measurements({})
    if window.evolution_records:
        window.plot_evolution(window.evolution_records)
    else:
        window.evolution_table.setRowCount(0)
        window.evolution_parameter_combo.clear()
        if window.evolution_plot is not None:
            window.evolution_plot.clear()
    if "butterfly_workbench" in snapshot and hasattr(window, "butterfly_workbench"):
        window.butterfly_workbench.restore_state(snapshot["butterfly_workbench"])
    if "lamellar_view" in snapshot:
        window.lamellar_page.restore_state(snapshot["lamellar_view"])
    window._sync_image_measurement_pages()

    for key, page in (
        ("local_measurements", window.local_measurement_page),
        ("azimuthal_view", window.azimuthal_page),
        ("density2d_view", window.density2d_page),
    ):
        if key in snapshot:
            page.restore_document(snapshot[key])


def apply_project_document(window: Any, data: Mapping[str, Any], target: Path) -> None:
    """Apply an already parsed/normalized document through the UI seam."""

    project_base = target.parent
    source = data.get("input", data.get("input_path"))
    poni = data.get("poni", data.get("poni_path"))
    if isinstance(poni, str) and not poni.strip():
        poni = None
    mask = data.get("mask", data.get("mask_path"))
    window._invalidate_pending_work(clear_fit=False)
    window._auto_scale_initial = False
    window.parameter_model.set_rows(data.get("parameters", {}))
    analysis = data.get(
        "analysis", data.get("measurement", data.get("analysis_settings", {}))
    )
    if isinstance(analysis, Mapping):
        window.set_analysis_settings(
            analysis,
            trigger_preview=False,
            reset_butterfly_if_missing=True,
        )
    display = data.get("display")
    if isinstance(display, Mapping):
        window.set_display_settings(display)
    frame = data.get("frame")
    dataset = data.get("dataset")
    mask_frame = data.get("mask_frame")
    mask_dataset = data.get("mask_dataset")
    selected_frame = int(frame) if frame is not None else None
    selected_dataset = str(dataset) if dataset is not None else None
    selected_mask_frame = int(mask_frame) if mask_frame is not None else None
    selected_mask_dataset = str(mask_dataset) if mask_dataset is not None else None
    roi = data.get("roi_exclusion")
    rois = data.get("rois")
    batch = data.get("batch")
    if isinstance(batch, Mapping):
        mode_index = window.batch_mode_combo.findData(batch.get("mode", "independent"))
        if mode_index >= 0:
            window.batch_mode_combo.setCurrentIndex(mode_index)
        stage_value = batch.get("stage")
        if stage_value is None and "full2d" in batch:
            stage_value = "full2d" if bool(batch.get("full2d")) else "geometry"
        if stage_value is not None:
            stage_index = window.batch_stage_combo.findData(stage_value)
            if stage_index >= 0:
                window.batch_stage_combo.setCurrentIndex(stage_index)
        window.batch_stream_check.setChecked(bool(batch.get("stream", False)))
        for widget, key in (
            (window.batch_manifest_edit, "manifest"),
            (window.batch_checkpoint_edit, "checkpoint"),
            (window.batch_output_edit, "output"),
        ):
            value = _resolve_project_path(batch.get(key), project_base)
            widget.setText("" if value is None else str(value))
        window.batch_series_edit.setText(str(batch.get("series") or ""))
        window.batch_range_edit.setText(str(batch.get("range") or ""))
        window.batch_resume_check.setChecked(bool(batch.get("resume", False)))
        window.batch_start_spin.setValue(
            int(batch.get("start", -1) if batch.get("start") is not None else -1)
        )
        window.batch_stop_spin.setValue(
            int(batch.get("stop", -1) if batch.get("stop") is not None else -1)
        )
        window.batch_stride_spin.setValue(max(1, int(batch.get("stride", 1))))
        frames = batch.get("frames", data.get("batch_frames", []))
        window.set_batch_frames(
            _resolve_project_frame(item, project_base) for item in (frames or [])
        )
    if source:
        if poni in (None, ""):
            setter = getattr(window.engine, "set_poni", None)
            if callable(setter):
                setter(None)
            window._poni_path = None
            window._legacy_geometry = None
            window._qmap = None
            window._qx = None
            window._qy = None
            window._capture_loaded_input_record("poni", None)
        if not window.open_image(
            source,
            frame=selected_frame,
            dataset=selected_dataset,
            poni=poni,
            external_mask=mask,
            mask_frame=selected_mask_frame,
            mask_dataset=selected_mask_dataset,
        ):
            raise ValueError(f"could not load project input: {source}")
    elif mask:
        if not window.select_mask(
            mask,
            mask_frame=selected_mask_frame,
            mask_dataset=selected_mask_dataset,
        ):
            raise ValueError(f"could not load project mask: {mask}")
    if mask and window._file_mask is None:
        window._load_external_mask(
            mask,
            frame=selected_mask_frame,
            dataset=selected_mask_dataset,
        )
    legacy_geometry = data.get("legacy_geometry")
    if legacy_geometry is not None:
        if not isinstance(legacy_geometry, Mapping):
            raise ValueError("legacy_geometry must be an object")
        window._apply_legacy_geometry(legacy_geometry)
    if rois and isinstance(rois, Iterable):
        window._roi_specs = [dict(spec) for spec in rois if isinstance(spec, Mapping)]
        window._exclusion_roi = window._roi_specs[-1] if window._roi_specs else None
        if window._roi_specs and not window._recompute_external_mask(update_widgets=True):
            raise ValueError("could not apply project ROIs")
    elif roi is not None:
        if not window.set_exclusion_roi(roi):
            raise ValueError("could not apply project ROI")
    if not source and poni and not window.set_poni(poni):
        raise ValueError(f"could not load project PONI: {poni}")
    if not source:
        window._frame = selected_frame
        window._dataset = selected_dataset
        window._mask_frame = selected_mask_frame
        window._mask_dataset = selected_mask_dataset
    setter = getattr(window.engine, "set_parameters", None)
    if callable(setter):
        setter(window.parameter_model.parameter_dict())
    window._fit_session_restore_active = True
    try:
        window._fit_session = window._normalise_fit_session(data.get("fit_session"))
    finally:
        window._fit_session_restore_active = False
    window._sync_fit_session_controls()

    window.lamellar_page.restore_document(
        data.get("lamellar_view"),
        context_signature=window._fit_state_signature(),
        observed=window._observed,
        qx=window._qx,
        qy=window._qy,
    )
    for key, page in (
        ("local_measurements", window.local_measurement_page),
        ("azimuthal_view", window.azimuthal_page),
        ("density2d_view", window.density2d_page),
    ):
        if key in data:
            page.restore_document(data[key])


def save_project(window: Any, path: str | Path | bool | None = None) -> bool:
    if isinstance(path, bool):
        path = None
    if path is None:
        if not QT_AVAILABLE:
            return False
        chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
            window,
            window._tr("dialog.save_project"),
            "",
            window._tr("filter.project"),
        )
        if not chosen:
            return False
        english = window._language.lower().startswith("en")
        confirmed = confirm_save_path(
            window,
            chosen,
            suffix=".json",
            title=window._tr("dialog.save_project"),
            prompt=lambda candidate, _existing: (
                f"Project file already exists. Overwrite?\n{candidate}"
                if english
                else f"项目文件已存在，是否覆盖？\n{candidate}"
            ),
            confirm_if_unchanged=False,
        )
        if confirmed is None:
            return False
        path, _overwrite = confirmed
    try:
        target = window._project_controller.save(path, window.project_to_dict())
    except (OSError, TypeError, ValueError) as exc:
        window._set_status("status.save_failed", flags="error", error=exc)
        return False
    window._project_path = target
    window._set_status("status.saved", name=target.name)
    return True


def load_project(window: Any, path: str | Path | bool | None = None) -> bool:
    if isinstance(path, bool):
        path = None
    if path is None:
        if not QT_AVAILABLE:
            return False
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            window,
            window._tr("dialog.open_project"),
            "",
            window._tr("filter.project"),
        )
        if not chosen:
            return False
        path = chosen
    try:
        target = window._project_controller.load(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        window._set_status("status.load_failed", flags="error", error=exc)
        return False
    window._project_path = target
    window._set_status("status.loaded", name=target.name)
    return True


__all__ = [
    "_resolve_project_frame",
    "_resolve_project_path",
    "apply_project_document",
    "load_project",
    "project_to_dict",
    "restore_engine_document",
    "restore_project_document",
    "save_project",
    "snapshot_engine_document",
    "snapshot_project_document",
    "snapshot_project_value",
]
