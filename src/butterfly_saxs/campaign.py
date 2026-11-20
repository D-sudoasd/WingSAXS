"""Workflow presets, manifest discovery, and multi-sample campaign orchestration.

This module is intentionally Qt-free and standard-library-only at import time
so ``bsaxs --help``, ``bsaxs describe``, and ``bsaxs doctor`` stay fast and
import zero heavy scientific packages.  NumPy, h5py, batch execution, and
reporting modules are imported lazily only inside the functions that execute
them.
"""

from __future__ import annotations

import copy
import csv
import inspect
import json
import math
import os
import re
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .cancellation import AnalysisCancelled
from .csv_utils import safe_csv_cell
from .file_identity import stream_digest
from .path_utils import IMAGE_SUFFIXES, is_supported_image_path
from .project import ProjectConfig, ProjectConfigError, _toml_value
from .settings import deep_merge_mapping, strict_int


CAMPAIGN_CONFIG_SCHEMA = "wingsaxs.campaign_config.v1"
CAMPAIGN_RUN_SCHEMA = "wingsaxs.campaign_run.v1"
MANIFEST_INIT_SCHEMA = "wingsaxs.manifest_init.v1"

# Explicit, documented parameter bundles for common butterfly-SAXS workflows.
# Omitting ``--preset`` leaves all historical defaults untouched.
WORKFLOW_PRESETS: dict[str, dict[str, Any]] = {
    "annular": {
        "ridge_method": "butterfly_curvature",
        "butterfly": {
            "stage": "evaluate",
            "trace_method": "annular_peak",
            "resamples": 0,
            "sensitivity": True,
            "annular_radial_bins": 40,
            "annular_angle_bins": 72,
        },
        "ellipse": {
            "preset": "standard",
        },
    },
    "curvature": {
        "ridge_method": "butterfly_curvature",
        "butterfly": {
            "stage": "evaluate",
            "trace_method": "curvature",
            "resamples": 0,
            "sensitivity": True,
        },
        "ellipse": {
            "preset": "standard",
        },
    },
    "sector": {
        "ridge_method": "butterfly_curvature",
        "butterfly": {
            "stage": "evaluate",
            "trace_method": "radial_sector",
            "resamples": 0,
            "sensitivity": True,
            "sector_width_deg": 10.0,
            "sector_step_deg": 5.0,
        },
        "ellipse": {
            "preset": "standard",
        },
    },
    "butterfly_pub": {
        "ridge_method": "butterfly_curvature",
        "butterfly": {
            "stage": "evaluate",
            "trace_method": "annular_peak",
            "resamples": 32,
            "sensitivity": True,
            "annular_radial_bins": 48,
            "annular_angle_bins": 90,
        },
        "ellipse": {
            "preset": "standard",
            "residual": "geometric",
            "multistart": 9,
        },
    },
}

WORKFLOW_PRESET_DESCRIPTIONS: dict[str, str] = {
    "annular": (
        "Fixed-q annular angular-peak trajectory tracking with symmetric double-ellipse "
        "evaluation (40 radial rings, 72 angular bins, fast 0-resample screening)."
    ),
    "curvature": (
        "Surface-curvature connected butterfly arc tracing with symmetric double-ellipse "
        "evaluation (0 resamples)."
    ),
    "sector": (
        "Fixed-chi radial-sector I(q) peak tracing (10 deg sector width, 5 deg step) "
        "with symmetric double-ellipse evaluation."
    ),
    "butterfly_pub": (
        "Publication-oriented annular butterfly evaluation with finer sampling "
        "(48 rings x 90 angles), geometric ellipse residual, 9 multistarts, and "
        "32 image-level empirical uncertainty resamples."
    ),
}

_NATURAL_PART = re.compile(r"(\d+)")
_HDF5_SUFFIXES = frozenset({".h5", ".hdf5", ".hdf"})
_IGNORED_DEFAULT_FILENAMES = frozenset(
    {
        "manifest.csv",
        "frame_summary.csv",
        "readme.txt",
        "notes.txt",
    }
)

_TEMP_C_PATTERN = re.compile(
    r"(?:^|[_\-\s])T?(?P<value>\d{2,4}(?:\.\d+)?)\s*(?:degC|C)(?=$|[_\-\s.])",
    re.IGNORECASE,
)
_RUN_PATTERN = re.compile(r"(?:^|[_\-\s])(?P<run>run[_\-]?\d+)(?=$|[_\-\s.])", re.IGNORECASE)
_EXPLICIT_FRAME_PATTERN = re.compile(
    r"(?:^|[_\-\s])(?:frame|frm|f)[_\-]?(?P<idx>\d+)(?=$|[_\-\s.])",
    re.IGNORECASE,
)
_TRAILING_INDEX_PATTERN = re.compile(r"[_\-](?P<idx>\d+)$")
_SAFE_DIR_CHAR = re.compile(r"[^A-Za-z0-9._-]+")


def canonical_workflow_preset(name: str) -> str:
    """Validate and return a canonical workflow preset name."""

    token = str(name or "").strip().lower().replace("-", "_")
    if token not in WORKFLOW_PRESETS:
        supported = ", ".join(sorted(WORKFLOW_PRESETS))
        raise ValueError(f"unknown workflow preset {name!r}; expected one of: {supported}")
    return token


def workflow_preset_analysis(name: str) -> dict[str, Any]:
    """Return a fresh deep copy of one named workflow preset mapping."""

    return copy.deepcopy(WORKFLOW_PRESETS[canonical_workflow_preset(name)])


def _natural_key(value: str) -> list[tuple[int, Any]]:
    parts = _NATURAL_PART.split(value)
    key: list[tuple[int, Any]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.casefold()))
    return key


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def _parse_filename_metadata(path: Path) -> dict[str, Any]:
    """Extract safe, non-fabricated metadata tokens from a frame filename.

    Missing values stay ``None``; timestamps are never invented from filenames
    or filesystem mtimes.
    """

    stem = path.stem
    temperature_c: float | None = None
    temp_match = _TEMP_C_PATTERN.search(stem)
    if temp_match is not None:
        raw_temp = temp_match.group("value")
        # Require explicit temperature marker (e.g., 350C or 375degC); reject
        # single-digit false hits or implausible values.
        try:
            value = float(raw_temp)
            if math.isfinite(value):
                temperature_c = value
        except ValueError:
            temperature_c = None

    run_id: str | None = None
    run_match = _RUN_PATTERN.search(stem)
    if run_match is not None:
        run_id = run_match.group("run").lower().replace("-", "_")

    parsed_frame_index: int | None = None
    explicit_frame = _EXPLICIT_FRAME_PATTERN.search(stem)
    if explicit_frame is not None:
        parsed_frame_index = int(explicit_frame.group("idx"))
    else:
        trailing = _TRAILING_INDEX_PATTERN.search(stem)
        if trailing is not None:
            parsed_frame_index = int(trailing.group("idx"))

    # Derive a clean sample identifier by stripping trailing frame/run/temp
    # markers when a meaningful non-numeric prefix exists.
    tokens = [tok for tok in re.split(r"[_\-\s]+", stem) if tok]
    sample_tokens: list[str] = []
    for tok in tokens:
        if _TEMP_C_PATTERN.fullmatch(tok) or _RUN_PATTERN.fullmatch(tok):
            break
        if _EXPLICIT_FRAME_PATTERN.fullmatch(tok):
            break
        if tok.isdigit() and sample_tokens:
            break
        sample_tokens.append(tok)
    sample: str | None = "_".join(sample_tokens) if sample_tokens else None
    if sample is not None and sample.isdigit():
        sample = None

    return {
        "sample": sample,
        "temperature_c": temperature_c,
        "run_id": run_id,
        "parsed_frame_index": parsed_frame_index,
    }


def _file_sha256(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.expanduser().resolve(strict=False)
    cached = cache.get(resolved)
    if cached is not None:
        return cached
    with resolved.open("rb") as handle:
        _, digest = stream_digest(handle)
    cache[resolved] = digest
    return digest


def _inspect_h5_dataset(path: Path, dataset: str | None) -> tuple[str, tuple[int, ...]]:
    """Inspect an HDF5 file read-only to resolve its 2D/3D frame dataset."""

    import h5py

    with h5py.File(path, "r") as handle:
        if dataset is not None:
            if dataset not in handle:
                raise ValueError(f"HDF5 dataset {dataset!r} not found in {path}")
            obj = handle[dataset]
            if not isinstance(obj, h5py.Dataset):
                raise ValueError(f"HDF5 path {dataset!r} in {path} is not a dataset")
            shape = tuple(int(dim) for dim in obj.shape)
            if len(shape) not in {2, 3}:
                raise ValueError(
                    f"HDF5 dataset {dataset!r} in {path} must be 2D or 3D, got shape {shape!r}"
                )
            return dataset.lstrip("/"), shape

        candidates: list[tuple[str, tuple[int, ...]]] = []

        def visitor(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.dtype.kind not in {"f", "i", "u"}:
                return
            if obj.ndim in {2, 3}:
                candidates.append((name, tuple(int(dim) for dim in obj.shape)))

        handle.visititems(visitor)
        if not candidates:
            raise ValueError(f"no 2D/3D numeric dataset found in HDF5 file {path}")
        if len(candidates) == 1:
            return candidates[0]
        # Prefer well-known primary SAXS intensity datasets if uniquely matched.
        preferred_names = {
            "data",
            "intensity",
            "abs2d",
            "i_abs_2d",
            "images",
            "entry/data/data",
            "entry/data/i_abs_2d",
        }
        preferred = [
            item
            for item in candidates
            if item[0].casefold() in preferred_names
            or item[0].rsplit("/", 1)[-1].casefold() in {"data", "intensity", "abs2d", "i_abs_2d"}
        ]
        if len(preferred) == 1:
            return preferred[0]
        names = ", ".join(item[0] for item in candidates)
        raise ValueError(
            f"ambiguous HDF5 datasets in {path}: {names}; specify --dataset explicitly"
        )


def _format_manifest_path(path: Path, relative_to: Path | None) -> str:
    resolved = path.expanduser().resolve(strict=False)
    if relative_to is not None:
        base = relative_to.expanduser().resolve(strict=False)
        try:
            return os.path.relpath(resolved, base).replace("\\", "/")
        except ValueError:
            # Different drive letters on Windows: fall back to absolute path.
            pass
    return str(resolved)


def discover_manifest(
    input_dir: str | os.PathLike[str],
    *,
    pattern: str = "*",
    recursive: bool = False,
    dataset: str | None = None,
    parse_metadata: bool = True,
    compute_sha256: bool = False,
    relative_to: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Discover image files or HDF5 stacks in natural order for a manifest.

    Parameters
    ----------
    input_dir:
        Directory of detector frames or a single supported image/HDF5 file.
    pattern:
        Glob pattern applied when ``input_dir`` is a directory (default ``"*"``).
    recursive:
        When ``True``, descend recursively into subdirectories.
    dataset:
        Optional HDF5 dataset path for multi-frame ``.h5``/``.hdf5`` stacks.
    parse_metadata:
        When ``True``, extract safe non-fabricated filename metadata
        (``sample``, ``temperature_c``, ``run_id``, ``parsed_frame_index``).
        Missing timestamps are never invented.
    compute_sha256:
        When ``True``, record per-file SHA-256 hex digest (computed once per
        unique source file).
    relative_to:
        Optional base directory used to emit portable relative ``path`` values.
    """

    source = Path(input_dir).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"input path does not exist: {source}")

    rel_base = Path(relative_to).expanduser() if relative_to is not None else None
    discovered_files: list[Path] = []
    if source.is_file():
        if not is_supported_image_path(source):
            supported = ", ".join(sorted(IMAGE_SUFFIXES))
            raise ValueError(
                f"unsupported image format {source.suffix or '<none>'!r}; supported: {supported}"
            )
        discovered_files = [source]
    elif source.is_dir():
        iterator = source.rglob(pattern) if recursive else source.glob(pattern)
        for item in iterator:
            if not item.is_file():
                continue
            if not is_supported_image_path(item):
                continue
            if pattern == "*" and item.name.casefold() in _IGNORED_DEFAULT_FILENAMES:
                continue
            discovered_files.append(item)
        discovered_files.sort(key=lambda p: _natural_key(p.relative_to(source).as_posix()))
    else:
        raise ValueError(f"unsupported input path: {source}")

    sha_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def unique_frame_id(candidate: str) -> str:
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        suffix = 2
        while f"{candidate}_{suffix}" in used_ids:
            suffix += 1
        chosen = f"{candidate}_{suffix}"
        used_ids.add(chosen)
        return chosen

    for file_path in discovered_files:
        formatted_path = _format_manifest_path(file_path, rel_base)
        meta = (
            _parse_filename_metadata(file_path)
            if parse_metadata
            else {
                "sample": None,
                "temperature_c": None,
                "run_id": None,
                "parsed_frame_index": None,
            }
        )
        digest = _file_sha256(file_path, sha_cache) if compute_sha256 else None

        if file_path.suffix.casefold() in _HDF5_SUFFIXES:
            ds_name, shape = _inspect_h5_dataset(file_path, dataset)
            is_stack = len(shape) == 3
            n_frames = shape[0] if is_stack else 1
            for frame_idx in range(n_frames):
                frame_id = unique_frame_id(
                    f"{file_path.stem}_f{frame_idx:04d}" if is_stack else file_path.stem
                )
                row: dict[str, Any] = {
                    "order": len(rows),
                    "path": formatted_path,
                    "frame_id": frame_id,
                    "dataset": ds_name,
                    "frame": frame_idx if is_stack else None,
                    "time": None,
                }
                if parse_metadata:
                    row["sample"] = meta["sample"]
                    row["temperature_c"] = meta["temperature_c"]
                    row["run_id"] = meta["run_id"]
                    row["parsed_frame_index"] = frame_idx if is_stack else meta["parsed_frame_index"]
                if compute_sha256:
                    row["sha256"] = digest
                rows.append(row)
        else:
            row = {
                "order": len(rows),
                "path": formatted_path,
                "frame_id": unique_frame_id(file_path.stem),
                "dataset": dataset,
                "frame": None,
                "time": None,
            }
            if parse_metadata:
                row["sample"] = meta["sample"]
                row["temperature_c"] = meta["temperature_c"]
                row["run_id"] = meta["run_id"]
                row["parsed_frame_index"] = meta["parsed_frame_index"]
            if compute_sha256:
                row["sha256"] = digest
            rows.append(row)

    return rows


def _atomic_write_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def write_manifest_csv(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Path:
    """Write validated manifest rows to CSV without overwriting unless forced.

    Optional selector/time columns (``dataset``, ``frame``, ``time``) are only
    emitted when populated so downstream :func:`butterfly_saxs.batch.build_frame_refs`
    never mistakes blank cells for malformed frame selectors or fabricated times.
    """

    destination = Path(output_path).expanduser()
    if destination.exists() and not force:
        raise FileExistsError(f"输出已存在，未覆盖：{destination}（需要 --force）")
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("manifest contains no rows to write")

    for index, row in enumerate(materialized):
        if not row.get("path"):
            raise ValueError(f"manifest row {index} is missing a 'path' value")

    core_columns = ["order", "path", "frame_id"]
    optional_standard = [
        "dataset",
        "frame",
        "time",
        "sample",
        "temperature_c",
        "run_id",
        "parsed_frame_index",
        "sha256",
    ]
    columns = list(core_columns)
    for col in optional_standard:
        if any(row.get(col) not in (None, "") for row in materialized):
            columns.append(col)
    for row in materialized:
        for key in row:
            if key not in columns and any(r.get(key) not in (None, "") for r in materialized):
                columns.append(str(key))

    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for index, row in enumerate(materialized):
        out_row: dict[str, Any] = {}
        for col in columns:
            value = row.get(col)
            if col == "order" and value is None:
                value = index
            if col == "frame_id" and not value:
                value = Path(str(row["path"])).stem
            out_row[col] = safe_csv_cell(value)
        writer.writerow(out_row)

    _atomic_write_text(destination, buffer.getvalue())
    return destination


def _normalize_sample_entry(entry: Any, index: int) -> dict[str, Any]:
    if isinstance(entry, (str, os.PathLike)):
        path = Path(os.fspath(entry))
        name = path.stem if path.is_file() or path.suffix else path.name
        if not name:
            name = f"sample_{index + 1:02d}"
        if path.suffix.casefold() in {".csv", ".json"}:
            return {"name": name, "manifest": os.fspath(entry)}
        return {"name": name, "input_dir": os.fspath(entry)}
    if isinstance(entry, Mapping):
        item = copy.deepcopy(dict(entry))
        name = str(item.get("name") or "").strip()
        if not name:
            for key in ("manifest", "input_dir", "inputs", "path"):
                candidate = item.get(key)
                if isinstance(candidate, (str, os.PathLike)):
                    p = Path(os.fspath(candidate))
                    name = p.stem if p.suffix else p.name
                    if name:
                        break
        if not name:
            name = f"sample_{index + 1:02d}"
        item["name"] = name
        return item
    raise ValueError(f"unsupported campaign sample entry at index {index}: {type(entry).__name__}")


def _validate_beamstop_radius_px(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("beamstop_radius_px must be a finite non-negative number or 'auto'")
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if token.casefold() == "auto":
            return "auto"
        try:
            numeric = float(token)
        except ValueError as exc:
            raise ValueError(
                f"beamstop_radius_px must be a finite non-negative number or 'auto', got {value!r}"
            ) from exc
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"beamstop_radius_px must be a finite non-negative number or 'auto', got {value!r}"
            ) from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"beamstop_radius_px must be a finite non-negative number or 'auto', got {value!r}"
        )
    return float(numeric)


def init_campaign_config(
    samples: Sequence[Any],
    output_path: str | os.PathLike[str],
    *,
    preset: str = "annular",
    force: bool = False,
    poni: str | os.PathLike[str] | None = None,
    mask: str | os.PathLike[str] | None = None,
    q_window: Sequence[float] | None = None,
    report: bool = False,
    journal_style: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    beamstop_radius_px: float | str | None = None,
    max_workers: int = 1,
    deduplicate_identical_frames: bool = False,
) -> Path:
    """Create a validated multi-sample campaign TOML file.

    Refuses to overwrite an existing file unless ``force=True``.
    """

    canonical_preset = canonical_workflow_preset(preset)
    workers = strict_int(max_workers, "max_workers", minimum=1)
    validated_beamstop = _validate_beamstop_radius_px(beamstop_radius_px)
    destination = Path(output_path).expanduser()
    if destination.exists() and not force:
        raise FileExistsError(f"输出已存在，未覆盖：{destination}（需要 --force）")

    normalized_samples = [
        _normalize_sample_entry(entry, idx) for idx, entry in enumerate(samples)
    ]
    if not normalized_samples:
        raise ValueError("campaign requires at least one sample entry")

    seen_names: set[str] = set()
    for sample in normalized_samples:
        name = sample["name"]
        if name in seen_names:
            raise ValueError(f"duplicate campaign sample name: {name!r}")
        seen_names.add(name)
        if "preset" in sample and sample["preset"] is not None:
            sample["preset"] = canonical_workflow_preset(str(sample["preset"]))
        if "beamstop_radius_px" in sample and sample["beamstop_radius_px"] is not None:
            sample["beamstop_radius_px"] = _validate_beamstop_radius_px(
                sample["beamstop_radius_px"]
            )

    lines: list[str] = [
        "# WingSAXS multi-sample campaign configuration",
        "",
        "[campaign]",
        f"schema_version = {_toml_value(CAMPAIGN_CONFIG_SCHEMA)}",
        f"preset = {_toml_value(canonical_preset)}",
        f"report = {_toml_value(bool(report))}",
    ]
    if workers > 1:
        lines.append(f"max_workers = {_toml_value(workers)}")
    if deduplicate_identical_frames:
        lines.append(f"deduplicate_identical_frames = {_toml_value(True)}")
    if journal_style is not None and str(journal_style).strip() and str(journal_style) != "default":
        lines.append(f"journal_style = {_toml_value(str(journal_style))}")
    lines.append("")

    defaults: dict[str, Any] = {}
    if poni is not None:
        defaults["poni"] = os.fspath(poni)
    if mask is not None:
        defaults["mask"] = os.fspath(mask)
    if q_window is not None:
        bounds = [float(v) for v in q_window]
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("q_window must be a finite [q_min, q_max] pair with q_min < q_max")
        defaults["q_window"] = bounds
    if validated_beamstop is not None:
        defaults["beamstop_radius_px"] = validated_beamstop
    if defaults:
        lines.append("[defaults]")
        for key, value in defaults.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")

    if metadata:
        lines.append("[metadata]")
        for key, value in metadata.items():
            if value is not None:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")

    for sample in normalized_samples:
        lines.append("[[samples]]")
        for key, value in sample.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    _atomic_write_text(destination, text)
    return destination


def _resolve_optional_path(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    text = os.fspath(value)
    if not str(text).strip():
        return None
    candidate = Path(text).expanduser()
    return os.fspath(candidate if candidate.is_absolute() else base_dir / candidate)


def load_campaign_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a campaign TOML file, resolving relative paths."""

    source = Path(path).expanduser()
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ProjectConfigError(f"找不到活动配置：{source}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProjectConfigError(f"活动配置 TOML 无法解析：{source}（{exc}）") from exc
    except OSError as exc:
        raise ProjectConfigError(f"无法读取活动配置：{source}（{exc}）") from exc

    if not isinstance(raw, Mapping):
        raise ProjectConfigError("活动配置根节点必须是 TOML 表")

    campaign_section = dict(raw.get("campaign") or {})
    defaults_section = dict(raw.get("defaults") or {})
    global_analysis = dict(raw.get("analysis") or {})
    raw_samples = raw.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ProjectConfigError("活动配置必须包含非空的 [[samples]] 列表")

    raw_preset = campaign_section.get("preset", defaults_section.get("preset", "annular"))
    campaign_preset = canonical_workflow_preset(str(raw_preset)) if raw_preset is not None else "annular"

    base_dir = source.parent.resolve(strict=False)
    samples: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, entry in enumerate(raw_samples):
        if not isinstance(entry, Mapping):
            raise ProjectConfigError(f"samples[{idx}] 必须是 TOML 表")
        merged = copy.deepcopy(defaults_section)
        merged.update(copy.deepcopy(dict(entry)))
        name = str(merged.get("name") or "").strip()
        if not name:
            raise ProjectConfigError(f"samples[{idx}] 缺少非空 name 字段")
        if name in seen_names:
            raise ProjectConfigError(f"活动配置包含重复样品名称：{name!r}")
        seen_names.add(name)

        sample_preset_raw = merged.get("preset", campaign_preset)
        sample_preset = canonical_workflow_preset(str(sample_preset_raw))

        # Resolve paths relative to the campaign TOML directory.
        manifest_path = _resolve_optional_path(merged.get("manifest"), base_dir)
        input_dir_path = _resolve_optional_path(merged.get("input_dir"), base_dir)
        raw_inputs = merged.get("inputs")
        resolved_inputs: list[str] = []
        if isinstance(raw_inputs, (str, os.PathLike)):
            resolved_inputs = [_resolve_optional_path(raw_inputs, base_dir) or os.fspath(raw_inputs)]
        elif isinstance(raw_inputs, Sequence):
            for item in raw_inputs:
                resolved_inputs.append(_resolve_optional_path(item, base_dir) or os.fspath(item))
        if not manifest_path and not input_dir_path and not resolved_inputs:
            raise ProjectConfigError(
                f"样品 {name!r} 必须提供 manifest、input_dir 或 inputs 之一"
            )

        poni_path = _resolve_optional_path(merged.get("poni", merged.get("poni_path")), base_dir)
        mask_path = _resolve_optional_path(merged.get("mask", merged.get("mask_path")), base_dir)
        valid_mask_path = _resolve_optional_path(
            merged.get("valid_mask", merged.get("valid_mask_path")), base_dir
        )

        # Merge preset -> global [analysis] -> per-sample [analysis] -> explicit sample knobs.
        analysis = workflow_preset_analysis(sample_preset)
        if global_analysis:
            analysis = deep_merge_mapping(analysis, global_analysis)
        sample_analysis = merged.get("analysis")
        if isinstance(sample_analysis, Mapping):
            analysis = deep_merge_mapping(analysis, sample_analysis)
        for key in (
            "q_window",
            "q_min",
            "q_max",
            "dataset",
            "frame",
            "mask_dataset",
            "mask_frame",
            "draw_axis_deg",
            "ridge_method",
            "ellipse",
            "butterfly",
            "beamstop_radius_px",
            "uncertainty_dataset",
            "include_physics",
            "intensity_mode",
            "bright_axis_convention",
        ):
            if key in merged and merged[key] is not None:
                if isinstance(merged[key], Mapping) and isinstance(analysis.get(key), Mapping):
                    analysis[key] = deep_merge_mapping(analysis[key], merged[key])
                else:
                    analysis[key] = copy.deepcopy(merged[key])
        sample_beamstop_raw = merged.get("beamstop_radius_px", analysis.get("beamstop_radius_px"))
        sample_beamstop = (
            _validate_beamstop_radius_px(sample_beamstop_raw)
            if sample_beamstop_raw is not None
            else None
        )
        if sample_beamstop is not None:
            analysis["beamstop_radius_px"] = sample_beamstop
        if mask_path is not None:
            analysis["mask"] = mask_path
        if valid_mask_path is not None:
            analysis["valid_mask"] = valid_mask_path

        sample_dedup = bool(
            merged.get(
                "deduplicate_identical_frames",
                merged.get(
                    "deduplicate_frames",
                    campaign_section.get(
                        "deduplicate_identical_frames",
                        campaign_section.get("deduplicate_frames", False),
                    ),
                ),
            )
        )

        samples.append(
            {
                "name": name,
                "preset": sample_preset,
                "manifest": manifest_path,
                "input_dir": input_dir_path,
                "inputs": resolved_inputs,
                "pattern": str(merged.get("pattern", "*")),
                "recursive": bool(merged.get("recursive", False)),
                "dataset": merged.get("dataset"),
                "frame": merged.get("frame"),
                "poni": poni_path,
                "mask": mask_path,
                "valid_mask": valid_mask_path,
                "mask_dataset": merged.get("mask_dataset"),
                "mask_frame": merged.get("mask_frame"),
                "mode": str(merged.get("mode", "independent")),
                "full2d": bool(merged.get("full2d", False)),
                "stream": bool(merged.get("stream", False)),
                "series": merged.get("series"),
                "start": merged.get("start"),
                "stop": merged.get("stop"),
                "stride": int(merged.get("stride", 1)),
                "frame_range": merged.get("frame_range", merged.get("range")),
                "beamstop_radius_px": sample_beamstop,
                "deduplicate_identical_frames": sample_dedup,
                "analysis": analysis,
                "metadata": copy.deepcopy(dict(merged.get("metadata") or {})),
            }
        )

    campaign_max_workers = strict_int(
        campaign_section.get("max_workers", 1), "campaign.max_workers", minimum=1
    )
    campaign_dedup = bool(
        campaign_section.get(
            "deduplicate_identical_frames",
            campaign_section.get("deduplicate_frames", False),
        )
    )
    default_beamstop_raw = defaults_section.get("beamstop_radius_px")
    default_beamstop = (
        _validate_beamstop_radius_px(default_beamstop_raw)
        if default_beamstop_raw is not None
        else None
    )

    return {
        "schema_version": str(campaign_section.get("schema_version", CAMPAIGN_CONFIG_SCHEMA)),
        "config_path": str(source.resolve(strict=False)),
        "preset": campaign_preset,
        "output_root": _resolve_optional_path(campaign_section.get("output_root"), base_dir),
        "report": bool(campaign_section.get("report", False)),
        "journal_style": campaign_section.get("journal_style"),
        "max_workers": campaign_max_workers,
        "deduplicate_identical_frames": campaign_dedup,
        "beamstop_radius_px": default_beamstop,
        "metadata": copy.deepcopy(dict(raw.get("metadata") or {})),
        "samples": samples,
    }


def _safe_sample_dirname(name: str) -> str:
    cleaned = _SAFE_DIR_CHAR.sub("_", name.strip()).strip("._")
    return cleaned or "sample"


def _run_sample_batch(
    sample: Mapping[str, Any],
    sample_output_dir: Path,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute one sample batch cleanly via the shared WingSAXS batch/export pipeline."""

    from . import batch as batch_module
    from . import export as export_module
    from . import pipeline as pipeline_module

    manifest = sample.get("manifest")
    inputs: Any = list(sample.get("inputs") or [])
    if not manifest and not inputs and sample.get("input_dir"):
        discovered = discover_manifest(
            sample["input_dir"],
            pattern=str(sample.get("pattern", "*")),
            recursive=bool(sample.get("recursive", False)),
            dataset=sample.get("dataset"),
            parse_metadata=True,
        )
        if not discovered:
            raise ValueError(f"no supported frames discovered in {sample['input_dir']}")
        manifest = discovered

    poni = sample.get("poni")
    full2d = bool(sample.get("full2d", False))
    sample_beamstop = sample.get("beamstop_radius_px")
    sample_dedup = bool(sample.get("deduplicate_identical_frames", False))
    analysis = copy.deepcopy(dict(sample.get("analysis") or {}))
    analysis["stage"] = "full2d" if full2d else "geometry"
    if sample_beamstop is not None:
        analysis["beamstop_radius_px"] = sample_beamstop
    config = ProjectConfig(
        poni_path=poni,
        output_dir=os.fspath(sample_output_dir),
        full2d=full2d,
        analysis=analysis,
        metadata=sample.get("metadata"),
    )

    geometry_cache: dict[Any, Any] = {}

    def analyze_for_sample(frame_ref: Any, initial_parameters: Any = None, cfg: Any = None) -> Any:
        source = getattr(frame_ref, "path", frame_ref)
        selected_frame = sample.get("frame")
        if selected_frame is None:
            selected_frame = getattr(frame_ref, "frame_selector", None)
        selected_dataset = sample.get("dataset")
        if selected_dataset is None:
            selected_dataset = getattr(frame_ref, "dataset", None)
            if selected_dataset is None:
                selected_dataset = getattr(frame_ref, "dataset_id", None) or None
        return pipeline_module.analyze_frame(
            source,
            poni=poni,
            config=cfg,
            full2d=full2d,
            initial_parameters=initial_parameters,
            frame=selected_frame,
            dataset=selected_dataset,
            mask=sample.get("mask"),
            mask_frame=sample.get("mask_frame"),
            mask_dataset=sample.get("mask_dataset"),
            valid_mask=sample.get("valid_mask"),
            geometry_cache=geometry_cache,
            beamstop_radius_px=sample_beamstop,
        )

    stream = bool(sample.get("stream", False))
    stream_writer = (
        export_module.StreamingBatchExporter(
            sample_output_dir,
            provenance={
                "command": "bsaxs campaign",
                "sample": sample["name"],
                "preset": sample["preset"],
                "full2d": full2d,
                "stream": True,
            },
            force=force,
            analyzer_identity="butterfly_saxs.pipeline.analyze_frame",
        )
        if stream
        else None
    )

    def _on_progress(update: Mapping[str, Any]) -> None:
        if progress is not None:
            progress(
                f"[{sample['name']}] {update.get('completed', 0)}/{update.get('total', 0)}"
            )

    try:
        run = batch_module.run_batch(
            inputs,
            analyze_for_sample,
            mode=str(sample.get("mode", "independent")),
            config=config,
            manifest=manifest,
            series=sample.get("series"),
            start=sample.get("start"),
            stop=sample.get("stop"),
            stride=int(sample.get("stride", 1)),
            frame_range=sample.get("frame_range"),
            progress=_on_progress if progress is not None else None,
            result_sink=None if stream_writer is None else stream_writer.write,
            retain_results=stream_writer is None,
            analyzer_identity="butterfly_saxs.pipeline.analyze_frame",
            deduplicate_identical_frames=sample_dedup,
            beamstop_radius_px=sample_beamstop,
        )
        exports = (
            stream_writer.finalize(run)
            if stream_writer is not None
            else export_module.export_batch(
                run,
                sample_output_dir,
                provenance={
                    "command": "bsaxs campaign",
                    "sample": sample["name"],
                    "preset": sample["preset"],
                    "full2d": full2d,
                },
                force=force,
            )
        )
    except BaseException:
        if stream_writer is not None:
            stream_writer.abort()
        raise

    n_frames = len(run.frame_results)
    n_success = sum(frame.status == "ok" for frame in run.frame_results)
    n_warning = sum(frame.status == "warning" for frame in run.frame_results)
    n_failed = len(run.failures)
    warning_reason = next(
        (
            reason
            for frame in run.frame_results
            if frame.result is not None
            if (reason := batch_module._quality_warning_reason(frame.result)) is not None
        ),
        None,
    )
    if n_failed > 0 or run.cancelled:
        status = "failed" if n_success == 0 and n_warning == 0 else "warning"
    elif n_warning > 0 or warning_reason is not None:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "n_frames": n_frames,
        "n_success": n_success,
        "n_warning": n_warning,
        "n_failed": n_failed,
        "warning_reason": warning_reason,
        "outputs": {str(k): str(v) for k, v in exports.items()},
    }


_DEFAULT_RUN_SAMPLE_BATCH = _run_sample_batch


def _format_sample_record(
    sample: Mapping[str, Any],
    sample_dir: Path,
    outcome: Mapping[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": sample["name"],
        "preset": sample["preset"],
        "status": str(outcome.get("status", "ok")),
        "output_dir": str(sample_dir),
        "n_frames": int(outcome.get("n_frames", 0)),
        "n_success": int(outcome.get("n_success", 0)),
        "n_warning": int(outcome.get("n_warning", 0)),
        "n_failed": int(outcome.get("n_failed", 0)),
        "outputs": dict(outcome.get("outputs") or {}),
        "error": None,
        "elapsed_s": round(float(elapsed_s), 6),
    }
    if outcome.get("warning_reason") is not None:
        record["warning_reason"] = str(outcome["warning_reason"])
    return record


def _format_sample_error_record(
    sample: Mapping[str, Any],
    sample_dir: Path,
    error_type: str,
    error_message: str,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "name": sample["name"],
        "preset": sample["preset"],
        "status": "failed",
        "output_dir": str(sample_dir),
        "n_frames": 0,
        "n_success": 0,
        "n_warning": 0,
        "n_failed": 0,
        "outputs": {},
        "error": {
            "type": error_type,
            "message": error_message,
        },
        "elapsed_s": round(float(elapsed_s), 6),
    }


def _format_sample_skipped_record(
    sample: Mapping[str, Any],
    sample_dir: Path,
) -> dict[str, Any]:
    return {
        "name": sample["name"],
        "preset": sample["preset"],
        "status": "skipped",
        "output_dir": str(sample_dir),
        "n_frames": 0,
        "n_success": 0,
        "n_warning": 0,
        "n_failed": 0,
        "outputs": {},
        "error": None,
        "elapsed_s": 0.0,
    }


def _run_sample_worker(
    sample: dict[str, Any],
    sample_dir_str: str,
    force: bool,
) -> dict[str, Any]:
    """Top-level Windows spawn-safe worker executing one independent sample."""

    sample_dir = Path(sample_dir_str)
    sample_start = time.monotonic()
    try:
        outcome = _run_sample_batch(
            sample,
            sample_dir,
            force=force,
            progress=None,
        )
        return _format_sample_record(
            sample, sample_dir, outcome, time.monotonic() - sample_start
        )
    except Exception as exc:  # noqa: BLE001 - isolate sample failure inside worker
        return _format_sample_error_record(
            sample,
            sample_dir,
            type(exc).__name__,
            str(exc),
            time.monotonic() - sample_start,
        )


def run_campaign(
    campaign_toml_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    force: bool = False,
    continue_on_error: bool = True,
    report: bool | None = None,
    journal_style: str | None = None,
    progress: Callable[[str], None] | None = None,
    max_workers: int | None = None,
    beamstop_radius_px: float | str | None = None,
    deduplicate_identical_frames: bool | None = None,
) -> dict[str, Any]:
    """Execute a multi-sample campaign TOML with per-sample failure isolation.

    Returns a strict-JSON-safe summary dict and writes ``campaign_summary.json``
    under ``output_root``.  Pass ``max_workers > 1`` to execute independent
    samples concurrently via :class:`concurrent.futures.ProcessPoolExecutor`
    while preserving deterministic sample summary ordering and per-sample
    failure isolation.
    """

    started = time.monotonic()
    config = load_campaign_config(campaign_toml_path)
    effective_workers = strict_int(
        config.get("max_workers", 1) if max_workers is None else max_workers,
        "max_workers",
        minimum=1,
    )
    override_beamstop = (
        _validate_beamstop_radius_px(beamstop_radius_px)
        if beamstop_radius_px is not None
        else None
    )
    out_root = Path(output_root).expanduser()
    if out_root.exists() and any(out_root.iterdir()) and not force:
        raise FileExistsError(f"输出目录已有内容，未覆盖：{out_root}（需要 --force）")
    out_root.mkdir(parents=True, exist_ok=True)

    # Pre-assign deterministic collision-free output directories in manifest order.
    sample_jobs: list[tuple[int, dict[str, Any], Path]] = []
    used_dirs: set[str] = set()
    for idx, raw_sample in enumerate(config["samples"]):
        sample = copy.deepcopy(raw_sample)
        if override_beamstop is not None:
            sample["beamstop_radius_px"] = override_beamstop
            sample_analysis = dict(sample.get("analysis") or {})
            sample_analysis["beamstop_radius_px"] = override_beamstop
            sample["analysis"] = sample_analysis
        if deduplicate_identical_frames is not None:
            sample["deduplicate_identical_frames"] = bool(deduplicate_identical_frames)
        base_dirname = _safe_sample_dirname(sample["name"])
        dirname = base_dirname
        suffix = 2
        while dirname in used_dirs:
            dirname = f"{base_dirname}_{suffix}"
            suffix += 1
        used_dirs.add(dirname)
        sample_jobs.append((idx, sample, out_root / dirname))

    sample_results: list[dict[str, Any]] = []
    use_parallel = effective_workers > 1 and len(sample_jobs) > 1 and continue_on_error

    if use_parallel:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

        results_by_index: dict[int, dict[str, Any]] = {}
        # Use real ProcessPoolExecutor for production runs; gracefully fall back
        # to ThreadPoolExecutor when _run_sample_batch is monkeypatched in tests
        # (spawned child processes would not inherit in-process monkeypatches).
        use_processes = _run_sample_batch is _DEFAULT_RUN_SAMPLE_BATCH
        pool_workers = min(effective_workers, len(sample_jobs))

        if use_processes:
            with ProcessPoolExecutor(max_workers=pool_workers) as executor:
                future_map = {
                    executor.submit(_run_sample_worker, sample, os.fspath(sample_dir), force): (
                        idx,
                        sample,
                        sample_dir,
                    )
                    for idx, sample, sample_dir in sample_jobs
                }
                for fut in as_completed(future_map):
                    idx, sample, sample_dir = future_map[fut]
                    try:
                        record = fut.result()
                    except Exception as exc:  # noqa: BLE001 - isolate worker crash
                        record = _format_sample_error_record(
                            sample, sample_dir, type(exc).__name__, str(exc), 0.0
                        )
                    results_by_index[idx] = record
                    if progress is not None:
                        progress(f"[{sample['name']}] {record['status']}")
        else:
            def _run_threaded(sample: dict[str, Any], sample_dir: Path) -> dict[str, Any]:
                sample_start = time.monotonic()
                try:
                    outcome = _run_sample_batch(
                        sample, sample_dir, force=force, progress=progress
                    )
                    return _format_sample_record(
                        sample, sample_dir, outcome, time.monotonic() - sample_start
                    )
                except AnalysisCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - isolate sample failure
                    return _format_sample_error_record(
                        sample,
                        sample_dir,
                        type(exc).__name__,
                        str(exc),
                        time.monotonic() - sample_start,
                    )

            with ThreadPoolExecutor(max_workers=pool_workers) as executor:
                future_map = {
                    executor.submit(_run_threaded, sample, sample_dir): (idx, sample, sample_dir)
                    for idx, sample, sample_dir in sample_jobs
                }
                for fut in as_completed(future_map):
                    idx, sample, sample_dir = future_map[fut]
                    try:
                        record = fut.result()
                    except AnalysisCancelled:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        record = _format_sample_error_record(
                            sample, sample_dir, type(exc).__name__, str(exc), 0.0
                        )
                    results_by_index[idx] = record

        sample_results = [results_by_index[idx] for idx in range(len(sample_jobs))]
    else:
        halted = False
        for _, sample, sample_dir in sample_jobs:
            if halted:
                sample_results.append(_format_sample_skipped_record(sample, sample_dir))
                continue

            sample_start = time.monotonic()
            try:
                outcome = _run_sample_batch(
                    sample,
                    sample_dir,
                    force=force,
                    progress=progress,
                )
                record = _format_sample_record(
                    sample, sample_dir, outcome, time.monotonic() - sample_start
                )
                sample_results.append(record)
                if record["status"] == "failed" and not continue_on_error:
                    halted = True
            except AnalysisCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate sample failure
                sample_results.append(
                    _format_sample_error_record(
                        sample,
                        sample_dir,
                        type(exc).__name__,
                        str(exc),
                        time.monotonic() - sample_start,
                    )
                )
                if not continue_on_error:
                    halted = True

    n_samples = len(sample_results)
    n_success = sum(item["status"] == "ok" for item in sample_results)
    n_warning = sum(item["status"] == "warning" for item in sample_results)
    n_failed = sum(item["status"] == "failed" for item in sample_results)
    n_skipped = sum(item["status"] == "skipped" for item in sample_results)

    if n_failed == n_samples and n_samples > 0:
        overall_status = "failed"
    elif n_failed > 0 or n_skipped > 0:
        overall_status = "partial_failure"
    elif n_warning > 0:
        overall_status = "warning"
    else:
        overall_status = "ok"

    exit_code = 0 if overall_status == "ok" else 1

    should_report = config["report"] if report is None else bool(report)
    effective_style = journal_style if journal_style not in (None, "default") else config.get("journal_style")
    report_summary: dict[str, Any] | None = None
    if should_report and (n_success + n_warning) > 0:
        try:
            from .report import build_analysis_report

            report_kwargs: dict[str, Any] = {"force": force}
            if effective_style not in (None, "default"):
                sig = inspect.signature(build_analysis_report)
                if "journal_style" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    report_kwargs["journal_style"] = str(effective_style)
            report_result = build_analysis_report(out_root, **report_kwargs)
            report_summary = {
                "status": str(report_result.get("status", "completed")),
                "outputs": _json_safe(report_result.get("outputs", {})),
                "exit_code": int(report_result.get("exit_code", 0)),
            }
            if report_summary["exit_code"] != 0:
                exit_code = 1
        except Exception as exc:  # noqa: BLE001 - preserve completed sample outputs
            report_summary = {
                "status": "incomplete",
                "error": str(exc),
                "outputs": {},
            }
            exit_code = 1

    summary_path = out_root / "campaign_summary.json"
    summary: dict[str, Any] = _json_safe(
        {
            "schema_version": CAMPAIGN_RUN_SCHEMA,
            "result_type": "campaign",
            "config_path": config["config_path"],
            "output_root": str(out_root.resolve(strict=False)),
            "preset": config["preset"],
            "continue_on_error": bool(continue_on_error),
            "max_workers": effective_workers,
            "status": overall_status,
            "exit_code": exit_code,
            "n_samples": n_samples,
            "n_success": n_success,
            "n_warning": n_warning,
            "n_failed": n_failed,
            "n_skipped": n_skipped,
            "elapsed_s": round(time.monotonic() - started, 6),
            "samples": sample_results,
            "outputs": {"campaign_summary": str(summary_path.resolve(strict=False))},
            "scientific_acceptance": False,
        }
    )
    if override_beamstop is not None or config.get("beamstop_radius_px") is not None:
        summary["beamstop_radius_px"] = (
            override_beamstop if override_beamstop is not None else config.get("beamstop_radius_px")
        )
    if report_summary is not None:
        summary["analysis_report"] = report_summary

    _atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return summary


__all__ = [
    "CAMPAIGN_CONFIG_SCHEMA",
    "CAMPAIGN_RUN_SCHEMA",
    "MANIFEST_INIT_SCHEMA",
    "WORKFLOW_PRESETS",
    "WORKFLOW_PRESET_DESCRIPTIONS",
    "canonical_workflow_preset",
    "discover_manifest",
    "init_campaign_config",
    "load_campaign_config",
    "run_campaign",
    "workflow_preset_analysis",
    "write_manifest_csv",
]
