"""Native LamellarSAXS2D single and batch result bundle loading.

The loader is deliberately strict: it recognizes the package's JSON/NPZ
sidecars and never guesses from arbitrary CSV files or array names.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .lamellar import _q_unit_from_source, _source_identity
from .lamellar_utils import _copy_metadata, _read

_NATIVE_SINGLE_HINTS = {
    "observed",
    "image",
    "data",
    "qmap",
    "qx",
    "qy",
    "observables",
    "ellipse_fit",
    "lobe_radial_peaks",
    "metadata",
    "flags",
}
_NATIVE_SCHEMA_PREFIXES = ("lamellarsaxs2d.", "butterflysaxs.")
_NATIVE_BATCH_SCHEMA_PREFIXES = ("lamellarsaxs2d.batch.", "butterflysaxs.batch.")
_DETECTOR_ARRAY_FIELDS = frozenset(
    {
        "image",
        "observed",
        "data",
        "I",
        "qx",
        "qy",
        "q",
        "valid_mask",
        "model",
        "residual",
        "model_image",
        "residual_image",
        "finite_mask",
        "fit_valid_mask",
        "detector_valid_mask",
        "external_valid_mask",
        "q_window_mask",
        "roi_exclusion_mask",
        "weight_valid_mask",
        "sampled_valid_mask",
    }
)


def _native_json_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    schema = str(value.get("schema_version", "") or "")
    if schema.startswith(_NATIVE_SCHEMA_PREFIXES):
        return True
    keys = set(str(key) for key in value)
    return len(keys & _NATIVE_SINGLE_HINTS) >= 3 and bool(
        keys
        & {
            "observed",
            "image",
            "data",
            "observables",
            "ellipse_fit",
            "lobe_radial_peaks",
        }
    )


def _batch_prefix(path: Path) -> str | None:
    """Return the native export prefix for a manifest companion name."""

    suffix = "manifest.json"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else None


def _native_batch_manifest(path: Path, value: Any, npz_path: Path | None) -> bool:
    """Recognize a native batch manifest and its same-prefix companions."""

    prefix = _batch_prefix(path)
    if prefix is None or not isinstance(value, Mapping) or not isinstance(value.get("frames"), list):
        return False
    expected_npz = path.with_name(prefix + "results.npz")
    if npz_path is None or npz_path != expected_npz or not npz_path.is_file():
        return False
    ellipse_path = path.with_name(prefix + "ellipse_fit.json")
    provenance_path = path.with_name(prefix + "provenance.json")
    if not ellipse_path.is_file():
        return False
    schema = str(value.get("schema_version", "") or "").casefold()
    if schema.startswith(_NATIVE_BATCH_SCHEMA_PREFIXES):
        return True
    if not provenance_path.is_file():
        return False
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        try:
            provenance = _load_json(provenance_path)
        except ValueError:
            return False
    tool = str(provenance.get("tool", "") or "").casefold()
    return tool in {"butterflysaxs", "butterflysaxs2d", "lamellarsaxs2d"} and bool(
        provenance.get("versions")
    )


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid native JSON bundle: {path}: {exc}") from exc


def _npz_keys(path: Path) -> list[str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return list(archive.files)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid native NPZ bundle: {path}: {exc}") from exc


def _npz_metadata(path: Path) -> dict[str, Any]:
    """Read only the small native archive receipt, never detector arrays."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if "__metadata__" not in archive.files:
                return {}
            value = np.asarray(archive["__metadata__"])
            if value.ndim != 0:
                return {}
            parsed = json.loads(str(value.item()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid native NPZ metadata: {path}: {exc}") from exc
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _first_key(keys: Sequence[str], names: Sequence[str]) -> str | None:
    for name in names:
        if name in keys:
            return name
    return None


def _validate_array_shapes(
    archive: Any, key_map: Mapping[str, str], *, path: Path
) -> None:
    image_key = key_map.get("observed")
    qx_key = key_map.get("qx")
    qy_key = key_map.get("qy")
    if image_key is None:
        raise ValueError(f"native NPZ bundle has no observed/image array: {path}")
    image = np.asarray(archive[image_key])
    if image.ndim != 2:
        raise ValueError(f"native observed array must be two-dimensional: {path}")
    for label, key in (("qx", qx_key), ("qy", qy_key)):
        if key is not None and np.asarray(archive[key]).shape != image.shape:
            raise ValueError(
                f"native {label} shape does not match observed array: {path}"
            )


def _attach_single_npz(
    source: dict[str, Any], path: Path, *, embed: bool = True
) -> dict[str, Any]:
    keys = _npz_keys(path)
    key_map = {
        "observed": _first_key(keys, ("observed", "image", "data", "I")),
        "qx": _first_key(keys, ("qx", "qx_nm_inv")),
        "qy": _first_key(keys, ("qy", "qy_nm_inv")),
        "q": _first_key(keys, ("q", "q_nm_inv")),
    }
    if key_map["observed"] is None:
        raise ValueError(f"native NPZ bundle has no observed/image array: {path}")
    with np.load(path, allow_pickle=False) as archive:
        _validate_array_shapes(
            archive, {k: v for k, v in key_map.items() if v is not None}, path=path
        )
        if embed:
            source["observed"] = np.array(archive[key_map["observed"]], copy=True)
            if key_map["qx"] is not None:
                source["qx"] = np.array(archive[key_map["qx"]], copy=True)
            if key_map["qy"] is not None:
                source["qy"] = np.array(archive[key_map["qy"]], copy=True)
            if key_map["q"] is not None:
                source.setdefault("q", np.array(archive[key_map["q"]], copy=True))
            q_unit_key = _first_key(keys, ("q_unit", "q_unit_name"))
            if q_unit_key is not None:
                try:
                    source.setdefault(
                        "q_unit", str(np.asarray(archive[q_unit_key]).item())
                    )
                except Exception:
                    pass
        source["array_path"] = os.fspath(path)
        source["npz_path"] = os.fspath(path)
        source["array_keys"] = dict(key_map)
    return source


def _batch_npz_sources(
    frames: Sequence[Any], npz_path: Path, base: Path
) -> list[dict[str, Any]]:
    keys = _npz_keys(npz_path)
    result: list[dict[str, Any]] = []
    for index, item in enumerate(frames):
        frame = dict(item) if isinstance(item, Mapping) else {"frame_index": index}
        raw_index = _read(frame, ("frame_index", "index"), None)
        frame_index = index if raw_index is None else int(raw_index)
        prefix = f"frame_{frame_index:04d}__"
        key_map = {
            "observed": _first_key(
                keys,
                (prefix + "image", prefix + "observed", prefix + "data", prefix + "I"),
            ),
            "qx": _first_key(
                keys,
                (
                    prefix + "qmap__qx",
                    prefix + "qmap__qx_nm_inv",
                    prefix + "qx",
                    prefix + "qx_nm_inv",
                ),
            ),
            "qy": _first_key(
                keys,
                (
                    prefix + "qmap__qy",
                    prefix + "qmap__qy_nm_inv",
                    prefix + "qy",
                    prefix + "qy_nm_inv",
                ),
            ),
            "q": _first_key(
                keys,
                (
                    prefix + "qmap__q",
                    prefix + "qmap__q_nm_inv",
                    prefix + "q",
                    prefix + "q_nm_inv",
                ),
            ),
        }
        source = dict(frame)
        source.setdefault("frame_index", frame_index)
        if key_map["observed"] is not None:
            source["array_path"] = os.fspath(npz_path)
            source["npz_path"] = os.fspath(npz_path)
            source["array_keys"] = dict(key_map)
            source["array_validation"] = "deferred_to_selected_frame"
        else:
            # Failed/skipped frames are still sources.  Their selector and
            # status remain inspectable even though no detector array exists.
            source["array_error"] = "missing observed array in native NPZ"
        result.append(source)
    return result


def _resolve_native_pair(path: Path) -> tuple[Path | None, Path | None]:
    if path.is_dir():
        native_batches: list[tuple[Path, Path]] = []
        for manifest in sorted(path.glob("*manifest.json")):
            npz = manifest.with_name(_batch_prefix(manifest) + "results.npz")
            if not npz.is_file():
                continue
            try:
                value = _load_json(manifest)
            except ValueError:
                continue
            if _native_batch_manifest(manifest, value, npz):
                native_batches.append((manifest, npz))
        if len(native_batches) > 1:
            choices = ", ".join(manifest.name for manifest, _ in native_batches)
            raise ValueError(
                f"Multiple native batch bundles were found in {path}: {choices}. "
                "Open one manifest JSON explicitly."
            )
        if native_batches:
            return native_batches[0]
        json_candidates = sorted(path.glob("*.json"))
        for candidate in json_candidates:
            if _batch_prefix(candidate) is not None or candidate.name in {
                "provenance.json",
                "frame_summary.json",
                "ellipse_fit.json",
                ".bundle.commit.json",
            }:
                continue
            npz = candidate.with_suffix(".npz")
            if npz.is_file():
                return candidate, npz
        return None, None
    if path.suffix.lower() == ".json":
        prefix = _batch_prefix(path)
        npz = path.with_name(prefix + "results.npz") if prefix is not None else path.with_suffix(".npz")
        return path, npz if npz.is_file() else None
    if path.suffix.lower() == ".npz":
        manifest = (
            path.with_name(path.name[: -len("results.npz")] + "manifest.json")
            if path.name.endswith("results.npz")
            else None
        )
        same_stem_json = path.with_suffix(".json")
        json_path = manifest if manifest is not None and manifest.is_file() else same_stem_json
        return json_path if json_path.is_file() else None, path
    return None, None


def _indexed_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    """Read compact per-frame records without opening any detector arrays."""

    records: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"expected a JSON object on line {line_number}")
                if value.get("frame_index") is None:
                    continue
                records[int(value["frame_index"])] = dict(value)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid native frame details: {path}: {exc}") from exc
    return records


def _merge_batch_details(
    sources: list[dict[str, Any]], manifest_path: Path, *, npz_path: Path | None
) -> None:
    """Attach frame details and legacy fit summaries while preserving identity."""

    prefix = _batch_prefix(manifest_path) or ""
    details = _indexed_jsonl(manifest_path.with_name(prefix + "frame_details.jsonl"))
    sidecar = manifest_path.with_name(prefix + "ellipse_fit.json")
    try:
        payload = _load_json(sidecar) if sidecar.is_file() else {}
    except ValueError as exc:
        raise ValueError(f"invalid native ellipse-fit sidecar: {sidecar}: {exc}") from exc
    rows = payload.get("frames") if isinstance(payload, Mapping) else None
    sidecar_by_index = {
        int(row["frame_index"]): dict(row)
        for row in (rows if isinstance(rows, list) else ())
        if isinstance(row, Mapping) and row.get("frame_index") is not None
    }
    for source in sources:
        try:
            index = int(source.get("frame_index"))
        except (TypeError, ValueError):
            continue
        detail = details.get(index)
        row = sidecar_by_index.get(index, {})
        if detail is not None:
            result = detail.get("result")
            if isinstance(result, Mapping):
                source.update(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in _DETECTOR_ARRAY_FIELDS
                    }
                )
                # Keep the complete compact frame record for consumers that
                # need to inspect array descriptors or nested fit diagnostics.
                source["native_result"] = dict(result)
            # Frame-level execution status and provenance take precedence over
            # nested result keys with the same name.
            source.update({key: value for key, value in detail.items() if key != "result"})

        frame = source.pop("_manifest_frame", None)
        if isinstance(frame, Mapping):
            for name in ("path", "frame_id", "frame", "dataset", "time", "order", "source"):
                if name in frame:
                    source[name] = frame[name]
            if "metadata" in frame:
                source["manifest_metadata"] = frame["metadata"]
                source.setdefault("frame_metadata", frame["metadata"])
            if "frame" in frame:
                source.setdefault("frame_selector", frame["frame"])
            if "dataset" in frame:
                source.setdefault("dataset_selector", frame["dataset"])

        if row:
            for name in (
                "ellipse_fit",
                "lobe_radial_peaks",
                "lobe_radial_profiles",
                "lobe_angular",
            ):
                if row.get(name) is not None:
                    target_name = "lobes" if name == "lobe_angular" else name
                    source.setdefault(target_name, row[name])
            if detail is None:
                for name in ("status", "error", "diagnostic", "elapsed_s"):
                    if row.get(name) is not None:
                        source.setdefault(name, row[name])

        settings = source.get("analysis_settings")
        analysis = source.get("analysis")
        if isinstance(settings, Mapping):
            merged_analysis = dict(analysis) if isinstance(analysis, Mapping) else {}
            merged_analysis.update(settings)
            source["analysis"] = merged_analysis
            if settings.get("draw_axis_deg") is not None:
                source["draw_axis_deg"] = settings["draw_axis_deg"]
        identity = source.get("source_identity")
        merged_identity = dict(identity) if isinstance(identity, Mapping) else {}
        for name in ("path", "frame_id", "frame", "frame_selector", "dataset", "dataset_selector", "time", "timestamp"):
            if source.get(name) is not None:
                merged_identity.setdefault(name, source[name])
        if merged_identity:
            source["source_identity"] = merged_identity


def _with_missing_slots(
    sources: list[dict[str, Any]], metadata: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Retain missing or not-yet-processed frame positions in the receipt."""

    indices: set[int] = set()
    try:
        indices.update(int(index) for index in metadata.get("missing_frames", ()))
        frame_count = max(0, int(metadata.get("frame_count", len(sources))))
    except (TypeError, ValueError):
        frame_count = len(sources)
    expected_count = max(frame_count, max(indices, default=-1) + 1, len(sources))
    by_index = {
        int(source.get("frame_index", position)): source
        for position, source in enumerate(sources)
    }
    for index in range(expected_count):
        if index in by_index:
            continue
        by_index[index] = {
            "frame_index": index,
            "frame_id": "",
            "status": "not_run" if index in indices or index >= frame_count else "missing",
            "error": "No frame record was written for this native batch position.",
            "array_error": "No observed array is available for this frame position.",
            "array_keys": {},
        }
    return [by_index[index] for index in sorted(by_index)]


def load_lamellar_sources(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load only native single/batch result bundles.

    Single-frame exports embed the selected arrays.  Batch exports retain a
    validated ``npz_path`` and logical ``array_keys`` mapping so callers can
    load one selected frame lazily without materialising every detector frame.
    CSV files and arbitrary JSON/NPZ files are rejected instead of guessed.
    """

    target = Path(path)
    json_path, npz_path = _resolve_native_pair(target)
    if json_path is None and npz_path is None:
        raise ValueError(f"not a native LamellarSAXS2D result bundle: {target}")
    summary: Any = _load_json(json_path) if json_path is not None else None
    native_batch = bool(
        json_path is not None
        and _native_batch_manifest(json_path, summary, npz_path)
    )
    if summary is not None and not (
        _native_json_mapping(summary)
        or native_batch
    ):
        raise ValueError(f"JSON is not a recognized native result bundle: {json_path}")
    if isinstance(summary, Mapping) and isinstance(summary.get("frames"), list):
        frames = list(summary["frames"])
        if native_batch and json_path is not None and npz_path is not None:
            sources = _batch_npz_sources(frames, npz_path, json_path.parent)
            for source, frame in zip(sources, frames, strict=True):
                source["_manifest_frame"] = frame
            _merge_batch_details(sources, json_path, npz_path=npz_path)
            sources = _with_missing_slots(sources, _npz_metadata(npz_path))
            return [
                _normalise_loaded_source(source, json_path.parent) for source in sources
            ]
        sources: list[dict[str, Any]] = []
        for index, item in enumerate(frames):
            if isinstance(item, Mapping):
                nested = (
                    item.get("result")
                    if isinstance(item.get("result"), Mapping)
                    else item
                )
                source = dict(nested)
                for key in (
                    "frame_index",
                    "frame_id",
                    "path",
                    "frame_selector",
                    "dataset",
                    "time",
                ):
                    if key in item and key not in source:
                        source[key] = item[key]
            else:
                source = {"frame_index": index}
            sources.append(source)
        if npz_path is not None:
            # A sidecar results.npz is preferred for a directory-level bundle.
            array_sources = _batch_npz_sources(sources, npz_path, json_path.parent)
            for source, arrays in zip(sources, array_sources, strict=False):
                source.update(
                    {
                        key: value
                        for key, value in arrays.items()
                        if key in {"array_path", "npz_path", "array_keys"}
                    }
                )
        return [
            _normalise_loaded_source(source, json_path.parent) for source in sources
        ]
    source = dict(summary) if isinstance(summary, Mapping) else {}
    if npz_path is not None:
        source = _attach_single_npz(source, npz_path, embed=True)
    elif not _native_json_mapping(source):
        raise ValueError(f"native single bundle has no usable summary: {target}")
    return [
        _normalise_loaded_source(
            source, json_path.parent if json_path is not None else target.parent
        )
    ]


def _normalise_loaded_source(source: Mapping[str, Any], base: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    array_names = {
        "observed",
        "image",
        "data",
        "I",
        "qx",
        "qy",
        "q",
        "qx_nm_inv",
        "qy_nm_inv",
        "q_nm_inv",
    }
    for key, value in source.items():
        if key in array_names and isinstance(value, np.ndarray):
            result[str(key)] = np.array(value, copy=True)
        else:
            result[str(key)] = _copy_metadata(value)
    if result.get("array_path") and not Path(str(result["array_path"])).is_absolute():
        result["array_path"] = os.fspath((base / str(result["array_path"])).resolve())
    if result.get("npz_path") and not Path(str(result["npz_path"])).is_absolute():
        result["npz_path"] = os.fspath((base / str(result["npz_path"])).resolve())
    if "source_identity" not in result:
        result["source_identity"] = _source_identity(result, result)
    if "q_unit" not in result:
        result["q_unit"] = _q_unit_from_source(result)
    return result


__all__ = ["load_lamellar_sources"]
