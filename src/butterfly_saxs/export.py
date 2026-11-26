"""Auditable exports for batch LamellarSAXS2D analyses.

Exports intentionally retain the full scientific result objects in JSON/NPZ
sidecars while providing compact CSV tables for plotting and downstream
kinetic analysis.  The functions accept :class:`~butterfly_saxs.batch.BatchRunResult`
as well as a plain sequence of ``FrameFitResult`` objects.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import os
import platform
import re
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ._export_schema import (
    CANONICAL_FRAME_SUMMARY_COLUMNS as _CANONICAL_FRAME_SUMMARY_COLUMNS,
    CANONICAL_LOBE_COLUMNS,
    CANONICAL_RIDGE_COLUMNS,
    PARAMETER_BASE_COLUMNS,
    STREAMING_PARAMETER_COLUMNS,
    _MISSING,
    _batch_ellipse_row,
    _frame_base,
    _frame_results,
    _frame_summary_rows,
    _json_text,
    _lobe_measurement_rows,
    _parameters,
    _result_flags,
    _ridge_points,
    _scalar,
    _value,
    build_frame_summary_row,
)
from .batch import (
    BatchRunResult,
    FrameFitResult,
    FrameRef,
    _checkpoint_safe,
    _config_with_file_fingerprints,
    _json_safe,
)
from .figure_support import save_agg_figure
from .file_identity import file_version, sha256_file
from .output_lock import output_lock
from .producer import exporter_identity


def _artifact_digest(path: Path) -> str:
    return sha256_file(path)


def _checkpoint_frame_signature(item: FrameFitResult) -> str | None:
    """Bind exported evidence to a frame's values, excluding execution timing."""
    record = item.to_record(include_result=False)
    try:
        record["result"] = _checkpoint_safe(item.result)
    except RecursionError:
        return None
    record.pop("elapsed_s")
    record.pop("resumed")
    value = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_zip_members(archive: zipfile.ZipFile, members: Iterable[str]) -> None:
    """Read each declared member through EOF so ZIP CRC and size are checked."""

    for member in members:
        try:
            with archive.open(member, "r") as handle:
                while handle.read(1024 * 1024):
                    pass
        except Exception as exc:
            raise ValueError(
                f"previous streamed NPZ member {member!r} failed integrity validation; "
                "re-export the bundle before resuming"
            ) from exc


def _publish_staged_bundle(
    stage: Path,
    targets: Mapping[str, Path],
    *,
    force: bool,
    preserve_existing: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Path]:
    """Publish a complete known bundle with rollback on publication faults."""

    if not stage.exists():
        raise FileNotFoundError(f"staging directory does not exist: {stage}")
    normalized = {str(key): Path(value) for key, value in targets.items()}
    preserved = set(preserve_existing)
    missing = [
        key
        for key, target in normalized.items()
        if key not in preserved and not (stage / target.name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "staged bundle is missing artifacts: " + ", ".join(sorted(missing))
        )
    output = next(iter(normalized.values())).parent
    output.mkdir(parents=True, exist_ok=True)
    marker_target = output / ".bundle.commit.json"
    marker_stage = stage / marker_target.name
    generation = uuid.uuid4().hex
    marker_payload = {
        "schema_version": "lamellarsaxs2d.bundle_commit.v1",
        "generation": generation,
        "complete": True,
        "artifacts": {
            key: {
                "name": target.name,
                "sha256": _artifact_digest(
                    target if key in preserved else stage / target.name
                ),
            }
            for key, target in normalized.items()
        },
    }
    marker_stage.write_text(
        json.dumps(marker_payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    normalized["commit_marker"] = marker_target
    backup = Path(tempfile.mkdtemp(prefix=".bundle-backup-", dir=output))
    old_files: dict[str, bool] = {}
    published: list[Path] = []
    try:
        for key, target in normalized.items():
            old_files[key] = target.exists()
            if key not in preserved and target.exists():
                shutil.copy2(target, backup / target.name)
        for key, target in normalized.items():
            if key in preserved:
                if not target.is_file():
                    raise FileNotFoundError(
                        f"preserved bundle target does not exist: {target}"
                    )
                continue
            os.replace(stage / target.name, target)
            published.append(target)
    except BaseException:
        for key, target in normalized.items():
            if key in preserved:
                continue
            try:
                old = old_files.get(key, False)
                backup_file = backup / target.name
                if old and backup_file.exists():
                    shutil.copy2(backup_file, target)
                elif target in published and target.exists():
                    target.unlink()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
    return normalized


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = list(columns or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["frame_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _scalar(row.get(key, "")) for key in keys})
    return path


def _walk_arrays(
    value: Any,
    prefix: str,
    output: dict[str, Any],
    _active: set[int] | None = None,
) -> None:
    """Collect arrays without descending into scalar metadata or cycles."""

    tolist = getattr(value, "tolist", None)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if callable(tolist) and shape is not None and dtype is not None:
        output[prefix] = value
        return
    if value is None or isinstance(
        value, (str, bytes, int, float, complex, bool, Enum, np.generic, Path)
    ):
        return
    if _active is None:
        _active = set()
    object_id = id(value)
    if object_id in _active:
        return
    _active.add(object_id)
    if isinstance(value, Mapping):
        try:
            for key, item in value.items():
                if key == "object" and (prefix == "qmap" or prefix.endswith("__qmap")):
                    continue
                _walk_arrays(item, f"{prefix}__{_safe_key(key)}", output, _active)
        finally:
            _active.remove(object_id)
        return
    if is_dataclass(value):
        try:
            for item_field in fields(value):
                _walk_arrays(
                    getattr(value, item_field.name),
                    f"{prefix}__{_safe_key(item_field.name)}",
                    output,
                    _active,
                )
        finally:
            _active.remove(object_id)
        return
    if isinstance(value, (list, tuple)):
        try:
            for index, item in enumerate(value):
                _walk_arrays(item, f"{prefix}__{index}", output, _active)
        finally:
            _active.remove(object_id)
        return
    if hasattr(value, "__dict__"):
        try:
            for key, item in vars(value).items():
                if not str(key).startswith("_"):
                    _walk_arrays(item, f"{prefix}__{_safe_key(key)}", output, _active)
        finally:
            _active.remove(object_id)
        return
    _active.remove(object_id)


def _safe_key(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "value"


def _frame_detail_safe(
    value: Any,
    *,
    _active: dict[int, str] | None = None,
    _path: str = "$",
) -> Any:
    """Make a compact, cycle-safe JSON view while leaving arrays in NPZ."""

    if isinstance(value, Enum):
        return {
            "enum_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
            "value": _frame_detail_safe(value.value, _active=_active, _path=f"{_path}.value"),
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _checkpoint_safe(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        try:
            return _frame_detail_safe(value.item(), _active=_active, _path=_path)
        except (TypeError, ValueError):
            return _checkpoint_safe(value)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return _checkpoint_safe(value)

    if _active is None:
        _active = {}
    object_id = id(value)
    if object_id in _active:
        return {"$ref": _active[object_id]}

    if isinstance(value, Mapping):
        _active[object_id] = _path
        try:
            return {
                str(key): _frame_detail_safe(
                    item, _active=_active, _path=f"{_path}.{_safe_key(key)}"
                )
                for key, item in value.items()
            }
        finally:
            _active.pop(object_id, None)
    if isinstance(value, (list, tuple, set, frozenset)):
        _active[object_id] = _path
        try:
            return [
                _frame_detail_safe(item, _active=_active, _path=f"{_path}[{index}]")
                for index, item in enumerate(value)
            ]
        finally:
            _active.pop(object_id, None)
    if is_dataclass(value):
        _active[object_id] = _path
        try:
            converted = {
                item.name: _frame_detail_safe(
                    getattr(value, item.name),
                    _active=_active,
                    _path=f"{_path}.{item.name}",
                )
                for item in fields(value)
            }
            parameters = _value(value, "parameters", default=_MISSING)
            if parameters is not _MISSING and "parameters" not in converted:
                converted["parameters"] = _frame_detail_safe(
                    parameters, _active=_active, _path=f"{_path}.parameters"
                )
            return converted
        finally:
            _active.pop(object_id, None)

    for method_name in ("to_mapping", "to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            if method_name == "to_mapping":
                converted = method(include_arrays=False)
            else:
                converted = method(include_specs=True)
        except TypeError:
            try:
                converted = method()
            except Exception:  # pragma: no cover - compatibility fallback
                continue
        except Exception:  # pragma: no cover - compatibility fallback
            continue
        if converted is value:
            continue
        if isinstance(converted, Mapping):
            _active[object_id] = _path
            try:
                merged = dict(converted)
                parameters = _value(value, "parameters", "params", default=_MISSING)
                if parameters is not _MISSING and "parameters" not in merged:
                    merged["parameters"] = parameters
                for name in (
                    "image", "qmap", "valid_mask", "model", "residual",
                    "model_image", "residual_image", "full2d",
                ):
                    original = _value(value, name, default=_MISSING)
                    if original is not _MISSING:
                        merged[name] = original
                return _frame_detail_safe(merged, _active=_active, _path=_path)
            finally:
                _active.pop(object_id, None)

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        _active[object_id] = _path
        try:
            return _frame_detail_safe(
                {
                    key: item
                    for key, item in attributes.items()
                    if not str(key).startswith("_")
                },
                _active=_active,
                _path=_path,
            )
        finally:
            _active.pop(object_id, None)
    return _checkpoint_safe(value)


def _frame_q_unit(value: Any) -> tuple[Any, str | None]:
    """Return an explicit q unit and the result field that supplied it."""

    for context_name, context in (
        ("result", value),
        ("qmap", _value(value, "qmap", default=None)),
        ("qmap.metadata", _value(_value(value, "qmap", default=None), "metadata", default=None)),
        ("metadata", _value(value, "metadata", default=None)),
        ("analysis_domain", _value(value, "analysis_domain", default=None)),
    ):
        candidate = _value(context, "q_unit", "unit", default=_MISSING)
        if candidate is not _MISSING and candidate is not None:
            return _frame_detail_safe(candidate), context_name
    return None, None


def _frame_detail_record(
    item: FrameFitResult,
    index: int,
    *,
    array_names: Sequence[str] = (),
) -> dict[str, Any]:
    result = item.result
    q_unit, q_unit_source = _frame_q_unit(result)
    qmap = _value(result, "qmap", default=None)
    qmap_metadata = _value(qmap, "metadata", default=None)
    domain = _value(result, "analysis_domain", default=None)
    if domain is None:
        domain = _value(result, "domain_summary", "analysis_domain_summary", default=None)
    summarize_domain = getattr(domain, "to_summary", None)
    if callable(summarize_domain):
        try:
            domain = summarize_domain()
        except Exception:  # pragma: no cover - optional domain summary
            pass
    analysis = _value(result, "analysis", "analysis_settings", "settings", default=None)
    record = {
        **_frame_base(item, index),
        "frame_order": _frame_detail_safe(item.frame.order),
        "frame_source": _frame_detail_safe(item.frame.source),
        "frame_number": item.frame.frame,
        "frame_dataset": _frame_detail_safe(item.frame.dataset),
        "frame_metadata": _frame_detail_safe(item.frame.metadata),
        "traceback": _frame_detail_safe(item.traceback),
        "array_prefix": f"frame_{index:04d}",
        "array_names": list(array_names),
        "q_unit": q_unit,
        "q_unit_source": q_unit_source,
        "qmap_metadata": _frame_detail_safe(qmap_metadata),
        "analysis_settings": _frame_detail_safe(analysis),
        "analysis_domain": _frame_detail_safe(domain),
        "result": _frame_detail_safe(result),
    }
    return _json_safe(record)


def _restore_export_values(
    value: Any, archive: zipfile.ZipFile, prefix: str, *, arrays: bool = True, checkpoint_value: Any = _MISSING
) -> Any:
    """Decode proven detail values needed by native scalar/vector exports."""
    if isinstance(value, Mapping):
        if (set(value) == {"enum_type", "name", "value"}
                and not isinstance(checkpoint_value, Mapping) and checkpoint_value == value["value"]):
            return value["value"]
        member = prefix + ".npy"
        if arrays and value.get("array_omitted") is True and member in archive.namelist():
            with archive.open(member) as handle:
                return np.load(handle, allow_pickle=False)
        excluded = ({"image", "data", "qmap", "analysis_domain", "analysis_arrays", "valid_mask", "model", "residual", "model_image", "residual_image"}
                    if "__" not in prefix else
                    {"model", "residual", "model_image", "residual_image", "sampled_indices"} if prefix.endswith("__full2d") else set())
        return {key: _restore_export_values(item, archive, prefix + "__" + _safe_key(key),
                    arrays=arrays and key not in excluded,
                    checkpoint_value=checkpoint_value.get(key, _MISSING) if isinstance(checkpoint_value, Mapping) else _MISSING)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_export_values(item, archive, f"{prefix}__{index}", arrays=arrays,
                    checkpoint_value=checkpoint_value[index] if isinstance(checkpoint_value, list) and index < len(checkpoint_value) else _MISSING)
                for index, item in enumerate(value)]
    return value


def _contains_omitted_array(value: Any, _visited: set[int] | None = None) -> bool:
    """Detect an explicit checkpoint marker without guessing from absence."""

    if value is None or isinstance(value, (str, bytes, int, float, complex, bool, Enum, np.generic)):
        return False

    if _visited is None:
        _visited = set()
    object_id = id(value)
    if object_id in _visited:
        return False
    _visited.add(object_id)

    if isinstance(value, Mapping):
        for key in ("array_omitted", "arrays_omitted"):
            marker = value.get(key)
            if isinstance(marker, (bool, np.bool_)) and bool(marker):
                return True
        return any(_contains_omitted_array(item, _visited) for item in value.values())
    if is_dataclass(value):
        return any(
            _contains_omitted_array(getattr(value, item_field.name), _visited)
            for item_field in fields(value)
            if not item_field.name.startswith("_")
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_omitted_array(item, _visited) for item in value)
    if hasattr(value, "__dict__"):
        try:
            attributes = vars(value)
        except TypeError:
            return False
        return any(
            _contains_omitted_array(item, _visited)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        )
    return False


def _write_npz(path: Path, results: Sequence[FrameFitResult]) -> Path:
    arrays: dict[str, Any] = {}
    missing_frames: list[int] = []
    missing_frame_ids: list[Any] = []
    missing_frame_paths: list[str] = []
    for index, item in enumerate(results):
        _walk_arrays(item.result, f"frame_{index:04d}", arrays)
        if (
            item.result is None
            or item.status in {"failed", "skipped"}
            or _contains_omitted_array(item.result)
        ):
            missing_frames.append(index)
            missing_frame_ids.append(_scalar(item.frame.frame_id))
            missing_frame_paths.append(str(item.frame.path))
    metadata = {
        "arrays": list(arrays),
        "frame_count": len(results),
        "complete": not missing_frames,
        "missing_frames": missing_frames,
        "missing_frame_ids": missing_frame_ids,
        "missing_frame_paths": missing_frame_paths,
    }
    arrays["__metadata__"] = np.asarray(_json_text(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _write_evolution(path: Path, results: Sequence[FrameFitResult]) -> Path:
    from matplotlib.figure import Figure
    from .visualization import _parameter_evolution_label, plot_parameter_evolution

    parameter_rows = [_parameters(item.result) for item in results]
    keys = list(dict.fromkeys(
        (parameter["parameter"], str(parameter.get("unit") or "").strip())
        for parameters in parameter_rows for parameter in parameters
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not keys:
        fig = Figure(figsize=(9, 5), layout="constrained")
        axis = fig.add_subplot(1, 1, 1)
        axis.text(0.5, 0.5, "No scalar parameter evolution available", ha="center", va="center")
        axis.set_axis_off()
        save_agg_figure(fig, path, clear=True, dpi=160)
        return path
    aliases = {key: f"quantity_{index}" for index, key in enumerate(keys)}
    labels, units = {}, {}
    for (name, unit), alias in aliases.items():
        explicit_units = {name: unit} if unit else None
        rendered = _parameter_evolution_label(name, parameter_units=explicit_units)
        labels[alias], units[alias] = rendered.rsplit(" (", 1)
        units[alias] = units[alias].removesuffix(")")
    rows = []
    has_time = any(item.frame.time is not None for item in results)
    for index, (item, parameters) in enumerate(zip(results, parameter_rows)):
        row = {"frame_index": index, "time": item.frame.time, "status": item.status}
        for parameter in parameters:
            key = (parameter["parameter"], str(parameter.get("unit") or "").strip())
            alias = aliases[key]
            value = parameter.get("value")
            if value in (None, ""):
                value = parameter.get("candidate_value")
            row[alias] = value
            row[f"{alias}_stderr"] = parameter.get("stderr")
            state = item.status
            if parameter.get("publication_status") != "available":
                state = "candidate" if state == "ok" else f"{state}; candidate"
            row[f"{alias}_status"] = state
        rows.append(row)
    from matplotlib._pylab_helpers import Gcf

    fig = plot_parameter_evolution(
        rows, parameters=list(aliases.values()), x_key="time" if has_time else "frame_index",
        parameter_labels=labels, parameter_units=units, output=path, dpi=160,
    )
    Gcf.destroy_fig(fig)
    fig.clear()
    return path


class StreamingBatchExporter:
    """Write a batch bundle while retaining at most one detector frame."""

    _FRAME_COLUMNS = list(_CANONICAL_FRAME_SUMMARY_COLUMNS)
    _PARAMETER_COLUMNS = list(STREAMING_PARAMETER_COLUMNS)
    _RIDGE_COLUMNS = list(CANONICAL_RIDGE_COLUMNS)
    _LOBE_COLUMNS = list(CANONICAL_LOBE_COLUMNS)

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        provenance: Mapping[str, Any] | None = None,
        prefix: str = "",
        force: bool = False,
        resume: bool = False,
        analyzer_identity: Any = None,
    ) -> None:
        self._output_lock = output_lock(output_dir, create=True)
        self._output_lock.__enter__()
        try:
            self._initialize(output_dir, provenance=provenance, prefix=prefix, force=force, resume=resume,
                             analyzer_identity=analyzer_identity)
        except BaseException:
            self._release_output_lock()
            raise

    def _release_output_lock(self) -> None:
        lock = getattr(self, "_output_lock", None)
        self._output_lock = None
        if lock is not None:
            lock.__exit__(None, None, None)

    def _initialize(
        self,
        output_dir: str | os.PathLike[str],
        *,
        provenance: Mapping[str, Any] | None = None,
        prefix: str = "",
        force: bool = False,
        resume: bool = False,
        analyzer_identity: Any = None,
    ) -> None:
        self.output = Path(output_dir)
        self.prefix = str(prefix or "")
        self.force = bool(force)
        self.resume = bool(resume)
        self.provenance = provenance
        self._exporter_identity = exporter_identity()
        from .producer import producer_identity

        self._expected_producer = (None if analyzer_identity is None
                                   else {**producer_identity(), "analyzer": _json_safe(analyzer_identity)})
        self.output.mkdir(parents=True, exist_ok=True)
        self._targets = {
            "frame_summary": self.output / f"{self.prefix}frame_summary.csv",
            "parameters_long": self.output / f"{self.prefix}parameters_long.csv",
            "ridge_points": self.output / f"{self.prefix}ridge_points.csv",
            "lobe_measurements": self.output / f"{self.prefix}lobe_measurements.csv",
            "ellipse_fit": self.output / f"{self.prefix}ellipse_fit.json",
            "ellipse_fit_jsonl": self.output / f"{self.prefix}ellipse_fit.jsonl",
            "frame_details": self.output / f"{self.prefix}frame_details.jsonl",
            "manifest": self.output / f"{self.prefix}manifest.json",
            "provenance": self.output / f"{self.prefix}provenance.json",
            "npz": self.output / f"{self.prefix}results.npz",
            "evolution_png": self.output / f"{self.prefix}evolution.png",
        }
        existing = [path for path in self._targets.values() if path.exists()]
        if existing and not self.force:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Export target(s) already exist; pass force=True to overwrite: {joined}"
            )
        self._stage: Path | None = Path(tempfile.mkdtemp(prefix=".stream-", dir=self.output))
        self._writers: dict[str, Any] = {}
        self._handles: dict[str, Any] = {}
        self._details_handle: Any = None
        self._ellipse_rows: list[dict[str, Any]] = []
        self._compact_results: list[FrameFitResult] = []
        self._frame_signatures: list[str | None] = []
        self._missing_frames: list[int] = []
        self._missing_ids: list[Any] = []
        self._missing_paths: list[str] = []
        self._quality_failed_frames: list[int] = []
        self._quality_warning_frames: list[int] = []
        self._array_names: list[str] = []
        self._touched_frames: set[int] = set()
        self._old_npz: zipfile.ZipFile | None = None
        self._old_npz_metadata: dict[str, Any] | None = None
        self._old_manifest: Mapping[str, Any] | None = None
        self._old_binding_verified = False
        self._old_producer: Any = None
        self._old_details_handle: Any = None
        self._old_details_position = 0
        self._old_details_used = False
        self._verified_targets: dict[str, tuple[Any, str]] = {}
        if self.resume:
            old_npz_path = self._targets["npz"]
            if not old_npz_path.exists():
                shutil.rmtree(self._stage, ignore_errors=True)
                self._stage = None
                raise FileNotFoundError(
                    f"resume stream export requires the previous NPZ bundle: {old_npz_path}"
                )
            try:
                self._old_npz = zipfile.ZipFile(old_npz_path, "r")
                if "__metadata__.npy" not in self._old_npz.namelist():
                    raise ValueError("previous streamed NPZ has no __metadata__ entry")
                with self._old_npz.open("__metadata__.npy", "r") as handle:
                    metadata_value = np.load(io.BytesIO(handle.read()), allow_pickle=False)
                parsed = json.loads(str(metadata_value.item()))
                if not isinstance(parsed, Mapping):
                    raise ValueError("previous streamed NPZ metadata is invalid")
                manifest_path = self._targets["manifest"]
                if manifest_path.exists():
                    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_manifest, Mapping):
                        self._old_manifest = loaded_manifest
                        self._old_producer = loaded_manifest.get("provenance", {}).get("producer_identity")
                if parsed.get("complete") is not True and self._old_manifest is None:
                    raise ValueError(
                        "previous streamed NPZ is incomplete and has no manifest; refusing lossy resume"
                    )
                old_names = set(self._old_npz.namelist())
                missing_names = [
                    name for name in parsed.get("arrays", ())
                    if f"{name}.npy" not in old_names and name not in old_names
                ]
                if missing_names:
                    raise ValueError(
                        "previous streamed NPZ is missing declared arrays: "
                        + ", ".join(str(item) for item in missing_names[:5])
                    )
                declared_members = [
                    str(name) if str(name) in old_names else f"{name}.npy"
                    for name in parsed.get("arrays", ())
                ]
                _verify_zip_members(self._old_npz, declared_members)
                marker_path = self.output / ".bundle.commit.json"
                if marker_path.is_file():
                    proof = json.loads(marker_path.read_text(encoding="utf-8"))
                    if (proof.get("schema_version") != "lamellarsaxs2d.bundle_commit.v1"
                            or proof.get("complete") is not True):
                        raise ValueError("Previous bundle commit marker is invalid; recover its prior generation before resuming.")
                    artifacts = proof.get("artifacts", {})
                    for key in ("npz", "manifest", "provenance", "frame_details"):
                        target = self._targets[key]
                        expected = artifacts.get(key, {})
                        if expected.get("name") != target.name or not target.is_file():
                            raise ValueError(f"Previous bundle {key} does not match its commit marker; recover the prior generation before resuming.")
                        before = file_version(target, target.stat())
                        digest = _artifact_digest(target)
                        if (digest != expected.get("sha256")
                                or file_version(target, target.stat()) != before):
                            raise ValueError(f"Previous bundle {key} does not match its commit marker; recover the prior generation before resuming.")
                        self._verified_targets[key] = (before, digest)
                    self._old_binding_verified = True
                    self._old_details_handle = self._targets["frame_details"].open(encoding="utf-8")
                self._old_npz_metadata = dict(parsed)
            except BaseException:
                if self._old_npz is not None:
                    self._old_npz.close()
                    self._old_npz = None
                shutil.rmtree(self._stage, ignore_errors=True)
                self._stage = None
                raise
        try:
            assert self._stage is not None
            for key, columns in (
                ("frame_summary", self._FRAME_COLUMNS),
                ("parameters_long", self._PARAMETER_COLUMNS),
                ("ridge_points", self._RIDGE_COLUMNS),
                ("lobe_measurements", self._LOBE_COLUMNS),
            ):
                handle = (self._stage / self._targets[key].name).open(
                    "w", newline="", encoding="utf-8"
                )
                self._handles[key] = handle
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                self._writers[key] = writer
            self._details_handle = (
                self._stage / self._targets["frame_details"].name
            ).open("w", encoding="utf-8", newline="\n")
            npz_path = self._stage / self._targets["npz"].name
            self._npz = zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1)
        except BaseException:
            self.abort()
            raise

    def can_restore(self, frame: FrameRef, index: int) -> bool:
        """Retry frames whose checkpoint advanced beyond the published NPZ."""
        del frame
        metadata = self._old_npz_metadata or {}
        proof = (self._old_manifest or {}).get("provenance", {}).get("checkpoint_frame_sha256", [])
        from .producer import calculation_producers_compatible

        return (self._old_npz is not None and index < int(metadata.get("frame_count", 0))
                and self._old_binding_verified
                and (self._expected_producer is None
                     or calculation_producers_compatible(self._old_producer, self._expected_producer))
                and index not in metadata.get("missing_frames", [])
                and index < len(proof) and isinstance(proof[index], str))

    def _original_detail(self, item: FrameFitResult, index: int) -> dict[str, Any]:
        if self._old_details_handle is None:
            raise ValueError("No verified producing frame details are available; rerun the frame before exporting restored values.")
        line = ""
        while self._old_details_position <= index:
            line = self._old_details_handle.readline()
            if not line:
                raise ValueError("Producing frame details ended before the restored frame; recover the prior generation.")
            self._old_details_position += 1
        record = json.loads(line)
        if (not isinstance(record, dict) or record.get("frame_index") != index
                or record.get("path") != str(item.frame.path)
                or record.get("frame_id") != _scalar(item.frame.frame_id)):
            raise ValueError("Producing frame details do not match the restored frame identity.")
        self._old_details_used = True
        return record

    def _close_original_details(self, *, verify: bool = False) -> None:
        handle = self._old_details_handle
        self._old_details_handle = None
        if handle is not None:
            handle.close()
        if verify and self._old_details_used:
            target = self._targets["frame_details"]
            version, digest = self._verified_targets["frame_details"]
            current = file_version(target, target.stat())
            if current != version or (current[-1] is None and _artifact_digest(target) != digest):
                raise ValueError("Producing frame details changed while restoring values; retry after recovering the prior generation.")

    def write(self, item: FrameFitResult) -> None:
        try:
            signature = _checkpoint_frame_signature(item)
            if item.resumed and self.resume:
                from .producer import calculation_producers_compatible

                if (self._expected_producer is not None
                        and not calculation_producers_compatible(self._old_producer, self._expected_producer)):
                    raise ValueError("Previous bundle calculation producer differs from the current analyzer; rerun the frame before retaining its arrays.")
                if not self._old_binding_verified:
                    raise ValueError("No verified producing bundle marker is available; recover its prior generation before restoring frames.")
                proof = (self._old_manifest or {}).get("provenance", {}).get("checkpoint_frame_sha256", [])
                index = len(self._compact_results)
                if not proof or index >= len(proof) or not isinstance(proof[index], str):
                    raise ValueError("No producing frame proof is available for the restored values; use the original software or rerun that frame in a separate output.")
                if proof[index] != signature:
                    raise ValueError("Restored checkpoint values differ from the producing bundle; recover its original checkpoint before resuming.")
            detail = None
            if item.resumed and self.resume:
                assert self._old_npz is not None
                detail = self._original_detail(item, len(self._compact_results))
                item = replace(item, result=_restore_export_values(detail.get("result"), self._old_npz,
                            f"frame_{len(self._compact_results):04d}", checkpoint_value=item.result))
            self._write(item, restored_detail=detail)
            self._frame_signatures.append(signature)
        except BaseException:
            self.abort()
            raise

    def _write(self, item: FrameFitResult, *, restored_detail: dict[str, Any] | None = None) -> None:
        index = len(self._compact_results)
        row = build_frame_summary_row(item, index)
        self._writers["frame_summary"].writerow(
            {key: _scalar(row.get(key, "")) for key in self._FRAME_COLUMNS}
        )
        base = _frame_base(item, index)
        flags = _result_flags(item.result)
        for parameter in _parameters(item.result):
            parameter_row = {
                **{key: _scalar(base.get(key, "")) for key in base},
                **parameter,
                "scientific_flags": _json_text(flags),
            }
            self._writers["parameters_long"].writerow(
                {key: _scalar(value) for key, value in parameter_row.items()}
            )
        for point_index, point in enumerate(_ridge_points(item.result)):
            self._writers["ridge_points"].writerow(
                {
                    **{key: _scalar(base.get(key, "")) for key in base},
                    "point_index": point_index,
                    **point,
                }
            )
        for lobe_row in _lobe_measurement_rows(item, index):
            self._writers["lobe_measurements"].writerow(
                {key: _scalar(lobe_row.get(key, "")) for key in self._LOBE_COLUMNS}
            )
        self._ellipse_rows.append(_batch_ellipse_row(item, index))
        array_prefix = f"frame_{index:04d}"
        arrays: dict[str, Any] = {}
        if not (item.resumed and self._old_npz is not None):
            _walk_arrays(item.result, array_prefix, arrays)
        for name, value in arrays.items():
            with self._npz.open(name + ".npy", "w") as handle:
                np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)
            self._array_names.append(name)
        frame_array_names = list(arrays)
        if item.resumed and self._old_npz is not None:
            prefix = f"{array_prefix}__"
            frame_array_names = sorted(
                set(frame_array_names)
                | {
                    name[:-4]
                    for name in self._old_npz.namelist()
                    if name.startswith(prefix) and name.endswith(".npy")
                }
            )
        if restored_detail is None:
            detail = _frame_detail_record(item, index, array_names=frame_array_names)
        else:
            detail = {**restored_detail, **_frame_base(item, index)}
        self._details_handle.write(
            json.dumps(detail, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )
        self._details_handle.flush()
        if item.result is None or _contains_omitted_array(item.result):
            if not (item.resumed and self._old_npz is not None):
                self._missing_frames.append(index)
                self._missing_ids.append(_scalar(item.frame.frame_id))
                self._missing_paths.append(str(item.frame.path))
        if item.status == "failed":
            self._quality_failed_frames.append(index)
        elif item.status == "warning":
            self._quality_warning_frames.append(index)
        if not item.resumed:
            self._touched_frames.add(index)
        compact = FrameFitResult(
            frame=item.frame,
            result=detail["result"],
            status=item.status,
            error=item.error,
            diagnostic=item.diagnostic,
            traceback=item.traceback,
            warm_start_from=item.warm_start_from,
            elapsed_s=item.elapsed_s,
            resumed=item.resumed,
        )
        self._compact_results.append(compact)
        for handle in self._handles.values():
            handle.flush()

    def _reuse_completed_bundle(self, batch: BatchRunResult | None) -> bool:
        """Verify the producing generation before retaining a no-op resume."""
        if (not isinstance(batch, BatchRunResult) or batch.cancelled or not batch.producer_identity
                or not batch.producer_identity.get("analyzer")
                or not self.resume or self._old_npz is None or self._touched_frames
                or len(self._compact_results) != batch.total_count or not self._old_manifest):
            return False
        previous = self._old_manifest.get("provenance", {})
        if not isinstance(previous, Mapping) or previous.get("producer_identity") != batch.producer_identity:
            return False
        if previous.get("exporter_identity") != self._exporter_identity:
            return False
        expected_results = previous.get("checkpoint_frame_sha256")
        if not expected_results or not all(self._frame_signatures):
            return False
        if expected_results != self._frame_signatures:
            raise ValueError("Restored checkpoint values differ from the producing bundle; recover its original checkpoint before resuming.")
        metadata = self._old_npz_metadata or {}
        if (metadata.get("artifact_complete") is not True or metadata.get("frame_count") != batch.total_count
                or metadata.get("quality_failed_frames", []) != self._quality_failed_frames
                or metadata.get("quality_warning_frames", []) != self._quality_warning_frames):
            return False
        marker = self.output / ".bundle.commit.json"
        try:
            proof = json.loads(marker.read_text(encoding="utf-8"))
            artifacts = proof["artifacts"]
            if (proof.get("schema_version") != "lamellarsaxs2d.bundle_commit.v1"
                    or proof.get("complete") is not True or set(artifacts) != set(self._targets)):
                return False
            for key, target in self._targets.items():
                expected = artifacts[key]
                before = file_version(target, target.stat())
                verified = self._verified_targets.get(key)
                digest = (verified[1] if before[-1] is not None and verified is not None and verified[0] == before
                          else _artifact_digest(target))
                if (expected.get("name") != target.name or digest != expected.get("sha256")
                        or file_version(target, target.stat()) != before):
                    return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return True

    def finalize(self, batch: BatchRunResult | None = None) -> dict[str, Path]:
        try:
            self._close_original_details(verify=True)
            if self.resume and isinstance(batch, BatchRunResult) and batch.cancelled:
                self.abort()
                self._stage = None
                return {**self._targets, "commit_marker": self.output / ".bundle.commit.json"}
            current_producer = batch.producer_identity if isinstance(batch, BatchRunResult) else self._expected_producer
            from .producer import calculation_producers_compatible

            if (self.resume and any(item.resumed for item in self._compact_results)
                    and not calculation_producers_compatible(self._old_producer, current_producer)):
                raise ValueError("Previous bundle calculation producer differs from the resumed checkpoint; rerun its frames before retaining arrays.")
            if self._old_manifest is not None and isinstance(batch, BatchRunResult):
                for key in ("input_hash", "config_hash", "mode"):
                    if self._old_manifest.get(key) != getattr(batch, key):
                        raise ValueError(
                            f"previous streamed bundle {key} does not match the validated resume run"
                        )
            if self._reuse_completed_bundle(batch):
                self.abort()
                self._stage = None
                return {**self._targets, "commit_marker": self.output / ".bundle.commit.json"}
            assert self._stage is not None
            metadata = {
                "arrays": list(self._array_names),
                "frame_count": len(self._compact_results),
                "complete": not self._missing_frames
                and not self._quality_failed_frames
                and not self._quality_warning_frames
                and not (isinstance(batch, BatchRunResult) and batch.cancelled),
                "artifact_complete": not self._missing_frames
                and not (isinstance(batch, BatchRunResult) and batch.cancelled),
                "all_frames_processed": bool(
                    isinstance(batch, BatchRunResult)
                    and not batch.cancelled
                    and len(self._compact_results) == batch.total_count
                ),
                "quality_complete": not self._quality_failed_frames and not self._quality_warning_frames,
                "quality_failed_frames": list(self._quality_failed_frames),
                "quality_warning_frames": list(self._quality_warning_frames),
                "missing_frames": sorted(
                    set(self._missing_frames)
                    | (
                        set(range(len(self._compact_results), batch.total_count))
                        if isinstance(batch, BatchRunResult) and batch.cancelled
                        else set()
                    )
                ),
                "missing_frame_ids": self._missing_ids,
                "missing_frame_paths": self._missing_paths,
            }
            no_op_resume = bool(
                self.resume
                and self._old_binding_verified
                and calculation_producers_compatible(self._old_producer, current_producer)
                and self._old_npz is not None
                and not self._touched_frames
                and not (isinstance(batch, BatchRunResult) and batch.cancelled)
            )
            if self._old_npz is not None and not no_op_resume:
                merged_names: list[str] = []
                for name in self._old_npz.namelist():
                    if name == "__metadata__.npy":
                        continue
                    match = re.match(r"frame_(\d+)__", name)
                    if match and int(match.group(1)) in self._touched_frames:
                        continue
                    if not self._old_binding_verified:
                        raise ValueError("Cannot retain unbound arrays from a previous NPZ; recover its producing commit marker before resuming.")
                    if not calculation_producers_compatible(self._old_producer, current_producer):
                        raise ValueError("Cannot retain arrays from a different calculation producer; rerun the affected frames.")
                    with self._old_npz.open(name, "r") as source_handle:
                        with self._npz.open(name, "w") as target_handle:
                            shutil.copyfileobj(source_handle, target_handle)
                    merged_names.append(name)
                metadata["arrays"] = [name[:-4] for name in merged_names if name.endswith(".npy")] + self._array_names
                self._old_npz.close()
                self._old_npz = None
            elif no_op_resume:
                assert self._old_npz is not None
                self._old_npz.close()
                self._old_npz = None
            metadata_bytes = io.BytesIO()
            np.lib.format.write_array(metadata_bytes, np.asarray(_json_text(metadata)), allow_pickle=False)
            with self._npz.open("__metadata__.npy", "w") as handle:
                handle.write(metadata_bytes.getvalue())
            self._npz.close()
            for handle in self._handles.values():
                handle.close()
            if self._details_handle is not None:
                self._details_handle.close()
                self._details_handle = None
            staged_ellipse = self._stage / self._targets["ellipse_fit"].name
            staged_jsonl = self._stage / self._targets["ellipse_fit_jsonl"].name
            staged_ellipse.write_text(
                json.dumps({"frames": self._ellipse_rows}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            with staged_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
                for row in self._ellipse_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            _write_evolution(self._stage / self._targets["evolution_png"].name, self._compact_results)
            run = batch if isinstance(batch, BatchRunResult) else None
            mode = run.mode if run is not None else "independent"
            input_hash = run.input_hash if run is not None else None
            config_hash = run.config_hash if run is not None else None
            manifest = run.manifest if run is not None else None
            created_at = datetime.now(timezone.utc).isoformat()
            provenance = _provenance_payload(
                created_at=created_at,
                mode=mode,
                input_hash=input_hash,
                config_hash=config_hash,
                source_count=len(self._compact_results),
                user_provenance=self.provenance,
                resolved_config=(run.resolved_config if run is not None else None),
            )
            if run is not None and run.producer_identity is not None:
                provenance["producer_identity"] = run.producer_identity
            provenance["exporter_identity"] = self._exporter_identity
            if (self.resume and self._old_producer != current_producer
                    and any(item.resumed for item in self._compact_results)):
                provenance["restored_calculation_producer_identity"] = self._old_producer
            if no_op_resume:
                old_provenance = (self._old_manifest or {}).get("provenance", {})
                provenance["retained_npz_provenance"] = old_provenance.get("retained_npz_provenance") or {
                    "producer_identity": self._old_producer,
                    "exporter_identity": old_provenance.get("exporter_identity"),
                    "sha256": self._verified_targets["npz"][1],
                }
            provenance["checkpoint_frame_sha256"] = self._frame_signatures if all(self._frame_signatures) else []
            staged_manifest = self._stage / self._targets["manifest"].name
            staged_manifest.write_text(
                json.dumps(
                    {
                        "created_at": created_at,
                        "mode": mode,
                        "input_hash": input_hash,
                        "config_hash": config_hash,
                        "frames": [_json_safe(item.frame.to_dict()) for item in self._compact_results],
                        "user_manifest": _json_safe(manifest),
                        "provenance": provenance,
                        "resolved_config": provenance.get("resolved_batch_config"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
            staged_provenance = self._stage / self._targets["provenance"].name
            staged_provenance.write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            self._close_original_details(verify=True)
            outputs = _publish_staged_bundle(
                self._stage,
                self._targets,
                force=self.force,
                preserve_existing={"npz"} if no_op_resume else frozenset(),
            )
            self._stage = None
            self._release_output_lock()
            return outputs
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        self._close_original_details()
        try:
            npz = getattr(self, "_npz", None)
            if npz is not None:
                npz.close()
        except Exception:
            pass
        old_npz = getattr(self, "_old_npz", None)
        if old_npz is not None:
            try:
                old_npz.close()
            except Exception:
                pass
            self._old_npz = None
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        details_handle = getattr(self, "_details_handle", None)
        if details_handle is not None:
            try:
                details_handle.close()
            except Exception:
                pass
        stage = getattr(self, "_stage", None)
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        self._release_output_lock()


@dataclass
class ExportResult(dict[str, Path]):
    """Dictionary-like output index with a convenient output directory."""

    output_dir: Path = field(default_factory=Path)

    def __post_init__(self) -> None:
        dict.__init__(self)


class _CompatExportMapping(dict[str, Path]):
    """Mapping that keeps legacy iteration keys while exposing new files."""

    def __init__(self, value: Mapping[str, Path], *, hidden: set[str] | frozenset[str] = frozenset()) -> None:
        super().__init__(value)
        self._hidden_iteration_keys = set(hidden)

    def __iter__(self):
        for key in super().__iter__():
            if key not in self._hidden_iteration_keys:
                yield key


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            version = importlib_metadata.version(name)
            return str(version) if version is not None else None
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - broken local metadata
            continue
    return None


def _software_versions() -> dict[str, str | None]:
    try:
        from . import __version__ as package_version
    except Exception:  # pragma: no cover - import-time fallback
        package_version = None
    return {
        "python": platform.python_version(),
        "ButterflySAXS": (
            str(package_version)
            if package_version is not None
            else _distribution_version("butterfly-saxs", "butterfly_saxs")
        ),
        "numpy": _distribution_version("numpy"),
        "scipy": _distribution_version("scipy"),
        "fabio": _distribution_version("fabio"),
        "pyFAI": _distribution_version("pyFAI", "pyfai"),
    }


def _provenance_payload(
    *,
    created_at: str,
    mode: str,
    input_hash: str | None,
    config_hash: str | None,
    source_count: int,
    user_provenance: Mapping[str, Any] | None,
    resolved_config: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at": created_at,
        "tool": "ButterflySAXS",
        "mode": mode,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "source_count": source_count,
    }
    if user_provenance:
        safe = _json_safe(user_provenance)
        if isinstance(safe, Mapping):
            payload.update(safe)
    if resolved_config is not None:
        resolved = _config_with_file_fingerprints(resolved_config)
        if isinstance(resolved, Mapping):
            payload["resolved_batch_config"] = _json_safe(resolved)
            for name in (
                "poni", "poni_path", "mask", "mask_path", "valid_mask",
                "valid_mask_path", "qmap", "external_mask", "rois", "selection",
                "stage", "full2d", "base_dir",
            ):
                if name in resolved:
                    payload.setdefault(name, _json_safe(resolved[name]))
    user_versions = payload.get("versions")
    merged_versions = dict(user_versions) if isinstance(user_versions, Mapping) else {}
    merged_versions.update(_software_versions())
    payload["versions"] = merged_versions
    return payload


def export_batch(
    batch: BatchRunResult | Iterable[FrameFitResult],
    output_dir: str | os.PathLike[str],
    *,
    provenance: Mapping[str, Any] | None = None,
    prefix: str = "",
    force: bool = False,
) -> dict[str, Path]:
    """Publish the complete export bundle while excluding another writer."""
    if isinstance(batch, (str, os.PathLike, Path)) and not isinstance(output_dir, (str, os.PathLike, Path)):
        batch, output_dir = output_dir, batch
    with output_lock(output_dir, create=True):
        return _export_batch_locked(batch, output_dir, provenance=provenance, prefix=prefix, force=force)


def _export_batch_locked(
    batch: BatchRunResult | Iterable[FrameFitResult],
    output_dir: str | os.PathLike[str],
    *,
    provenance: Mapping[str, Any] | None = None,
    prefix: str = "",
    force: bool = False,
) -> dict[str, Path]:
    """Write CSV, JSON/JSONL, NPZ and evolution-plot exports."""

    if isinstance(batch, (str, os.PathLike, Path)) and not isinstance(output_dir, (str, os.PathLike, Path)):
        batch, output_dir = output_dir, batch
    output = Path(output_dir)
    results = _frame_results(batch)
    stem = f"{prefix}" if prefix else ""

    target_map = {
        "frame_summary": output / f"{stem}frame_summary.csv",
        "parameters_long": output / f"{stem}parameters_long.csv",
        "ridge_points": output / f"{stem}ridge_points.csv",
        "lobe_measurements": output / f"{stem}lobe_measurements.csv",
        "ellipse_fit": output / f"{stem}ellipse_fit.json",
        "ellipse_fit_jsonl": output / f"{stem}ellipse_fit.jsonl",
        "frame_details": output / f"{stem}frame_details.jsonl",
        "manifest": output / f"{stem}manifest.json",
        "provenance": output / f"{stem}provenance.json",
        "npz": output / f"{stem}results.npz",
        "evolution_png": output / f"{stem}evolution.png",
    }
    existing_targets = [path for path in target_map.values() if path.exists()]
    if existing_targets and not force:
        paths = ", ".join(str(path) for path in existing_targets)
        raise FileExistsError(
            f"Export target(s) already exist; pass force=True to overwrite: {paths}"
        )
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=output))
    try:
        staged = {key: stage / target.name for key, target in target_map.items()}
        _write_csv(
            staged["frame_summary"],
            _frame_summary_rows(results),
            columns=list(_CANONICAL_FRAME_SUMMARY_COLUMNS),
        )

        parameter_rows: list[dict[str, Any]] = []
        for index, item in enumerate(results):
            base = _frame_base(item, index)
            result_flags = _result_flags(item.result)
            for parameter in _parameters(item.result):
                parameter_rows.append(
                    {
                        **base,
                        **parameter,
                        "scientific_flags": _json_text(result_flags),
                    }
                )
        _write_csv(
            staged["parameters_long"],
            parameter_rows,
            columns=list(PARAMETER_BASE_COLUMNS),
        )

        ridge_rows: list[dict[str, Any]] = []
        for index, item in enumerate(results):
            for point_index, point in enumerate(_ridge_points(item.result)):
                ridge_rows.append(
                    {
                        **_frame_base(item, index),
                        "point_index": point_index,
                        **point,
                    }
                )
        _write_csv(staged["ridge_points"], ridge_rows)

        lobe_rows: list[dict[str, Any]] = []
        for index, item in enumerate(results):
            lobe_rows.extend(_lobe_measurement_rows(item, index))
        _write_csv(
            staged["lobe_measurements"],
            lobe_rows,
            columns=list(CANONICAL_LOBE_COLUMNS),
        )

        ellipse_rows = [
            _batch_ellipse_row(item, index) for index, item in enumerate(results)
        ]
        staged["ellipse_fit"].write_text(
            json.dumps({"frames": ellipse_rows}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with staged["ellipse_fit_jsonl"].open("w", encoding="utf-8", newline="\n") as handle:
            for row in ellipse_rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        with staged["frame_details"].open("w", encoding="utf-8", newline="\n") as handle:
            for index, item in enumerate(results):
                arrays: dict[str, Any] = {}
                _walk_arrays(item.result, f"frame_{index:04d}", arrays)
                detail = _frame_detail_record(item, index, array_names=list(arrays))
                handle.write(
                    json.dumps(detail, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    + "\n"
                )

        if isinstance(batch, BatchRunResult):
            batch_manifest = batch.manifest
            input_hash = batch.input_hash
            config_hash = batch.config_hash
            mode = batch.mode
        else:
            batch_manifest = None
            input_hash = None
            config_hash = None
            mode = "independent"
        created_at = datetime.now(timezone.utc).isoformat()
        provenance_value = _provenance_payload(
            created_at=created_at,
            mode=mode,
            input_hash=input_hash,
            config_hash=config_hash,
            source_count=len(results),
            user_provenance=provenance,
            resolved_config=(batch.resolved_config if isinstance(batch, BatchRunResult) else None),
        )
        manifest_value = {
            "created_at": created_at,
            "mode": mode,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "frames": [_json_safe(item.frame.to_dict()) for item in results],
            "user_manifest": _json_safe(batch_manifest),
            "provenance": provenance_value,
        }
        staged["manifest"].write_text(
            json.dumps(manifest_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staged["provenance"].write_text(
            json.dumps(provenance_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        _write_npz(staged["npz"], results)
        _write_evolution(staged["evolution_png"], results)
        published = _publish_staged_bundle(stage, target_map, force=force)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    return _CompatExportMapping(
        published,
        hidden={"lobe_measurements", "commit_marker"},
    )


export_results = export_batch
write_exports = export_batch
export_batch_results = export_batch


__all__ = [
    "ExportResult",
    "StreamingBatchExporter",
    "export_batch",
    "export_batch_results",
    "export_results",
    "write_exports",
]
