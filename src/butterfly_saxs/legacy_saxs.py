"""Read-only inspection and migration support for SAXSAnalyzer artifacts.

Legacy measurements remain source evidence.  This module never maps an old
``Ln_nm``/``Lz_nm`` value into WingSAXS observables; it preserves the original
document and hashes, and only reconstructs the legacy flat-detector q map when
the complete SAXSAnalyzer Fusion v2 scalar geometry is available.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

from .geometry import GeometryMaps
from .io import LoadedImage, load_image


RECEIPT_SCHEMA = "butterfly_saxs.legacy_saxs_receipt.v1"
_LEGACY_GEOMETRY_KEYS = ("px_mm", "dist_mm", "wl_A", "cx", "cy")
_LEGACY_OPTIONAL_GEOMETRY_KEYS = {"px_x_mm", "px_y_mm", "q_unit"}
_ZERO_TRANSFORM_KEYS = {"rot1", "rot2", "rot3", "rotation", "rotation_deg"}
_GEOMETRY_TRANSFORM_KEYS = _ZERO_TRANSFORM_KEYS | {"distortion", "spline_file"}
_MAX_BUNDLE_FILES = 4096
_SKIP_BUNDLE_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
_TEXT_SUFFIXES = {".json", ".csv", ".txt", ".dat", ".md", ".yaml", ".yml"}


class LegacySaxsError(ValueError):
    """A selected legacy file cannot be inspected or migrated safely."""


@dataclass(frozen=True, slots=True)
class LegacyFileEvidence:
    """Identity and filesystem state for one original artifact."""

    path: Path
    relative_path: str | None
    role: str
    present: bool
    size_bytes: int | None = None
    sha256: str | None = None
    kind: str = "file"


@dataclass(frozen=True, slots=True)
class LegacyFusionSession:
    """Unmodified Fusion session metadata plus a conservative geometry check."""

    version: Any
    data_path: Path | None
    image_present: bool
    geometry: Mapping[str, Any]
    geometry_values: Mapping[str, float] | None
    geometry_supported: bool
    geometry_message: str
    pattern_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LegacySaxsInspection:
    """Read-only view of a Fusion session, evidence file, or result directory."""

    path: Path
    artifact_kind: str
    format: str
    source_sha256: str | None
    source_size_bytes: int | None
    document: Any = None
    raw_text: str | None = None
    preview_text: str = ""
    session: LegacyFusionSession | None = None
    files: tuple[LegacyFileEvidence, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LegacySaxsLoadedImage:
    """A detector frame and q map derived from a legacy Fusion v2 session."""

    inspection: LegacySaxsInspection
    image: LoadedImage
    qmap: GeometryMaps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LegacySaxsError(f"legacy geometry field {name!r} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LegacySaxsError(f"legacy geometry field {name!r} must be numeric") from exc
    if not math.isfinite(result):
        raise LegacySaxsError(f"legacy geometry field {name!r} must be finite")
    return result


def _validate_legacy_geometry(geometry: Mapping[str, Any]) -> dict[str, float]:
    """Validate scalar Fusion geometry before any q coordinates are derived."""

    allowed = (
        set(_LEGACY_GEOMETRY_KEYS)
        | _LEGACY_OPTIONAL_GEOMETRY_KEYS
        | _GEOMETRY_TRANSFORM_KEYS
    )
    unsupported = [
        key
        for key, value in geometry.items()
        if key not in allowed and value not in (None, "", 0, 0.0, "0", "0.0")
    ]
    transformed = [
        key
        for key in _GEOMETRY_TRANSFORM_KEYS
        if key in geometry and geometry[key] not in (None, "", 0, 0.0, "0", "0.0")
    ]
    if unsupported:
        raise LegacySaxsError(
            "legacy geometry contains unsupported fields: " + ", ".join(sorted(unsupported))
        )
    if transformed:
        raise LegacySaxsError(
            "legacy scalar geometry includes unsupported detector transforms: "
            + ", ".join(sorted(transformed))
        )

    values = {key: _finite_float(geometry.get(key), key) for key in _LEGACY_GEOMETRY_KEYS}
    for key in ("px_mm", "dist_mm", "wl_A"):
        if values[key] <= 0:
            raise LegacySaxsError(f"legacy geometry field {key!r} must be positive")

    rect_values = {
        key: geometry.get(key)
        for key in ("px_x_mm", "px_y_mm")
        if geometry.get(key) not in (None, "")
    }
    if rect_values:
        if len(rect_values) != 2:
            raise LegacySaxsError(
                "legacy rectangular pixel geometry is incomplete; both px_x_mm and px_y_mm are required"
            )
        pixel_x = _finite_float(rect_values["px_x_mm"], "px_x_mm")
        pixel_y = _finite_float(rect_values["px_y_mm"], "px_y_mm")
        if not (
            math.isclose(pixel_x, pixel_y, rel_tol=1e-12)
            and math.isclose(pixel_x, values["px_mm"], rel_tol=1e-12)
        ):
            raise LegacySaxsError(
                "legacy geometry uses rectangular pixels; the scalar compatibility map only supports square pixels"
            )
    q_unit = str(geometry.get("q_unit") or "nm^-1").strip().casefold()
    if q_unit not in {"nm^-1", "nm-1", "1/nm"}:
        raise LegacySaxsError(
            f"legacy geometry q_unit {q_unit!r} is not the nm^-1 unit produced by the checked scalar map"
        )
    return values


def _legacy_session_from_document(
    document: Any, source_path: Path
) -> tuple[LegacyFusionSession | None, list[str]]:
    if not isinstance(document, Mapping):
        return None, []
    if not (
        "data_path" in document
        and isinstance(document.get("geometry"), Mapping)
        and not {"project", "paths", "stages"}.intersection(document)
    ):
        return None, []

    raw_data_path = document.get("data_path")
    data_path: Path | None
    if isinstance(raw_data_path, (str, os.PathLike)) and str(raw_data_path).strip():
        candidate = Path(raw_data_path).expanduser()
        if not candidate.is_absolute():
            candidate = source_path.parent / candidate
        data_path = candidate.resolve()
    else:
        data_path = None

    geometry = dict(document.get("geometry", {}))
    warnings: list[str] = []
    values: dict[str, float] | None = None
    message = ""
    try:
        values = _validate_legacy_geometry(geometry)
        message = (
            "Supported as SAXSAnalyzer Fusion v2 flat-detector geometry: square pixels, "
            "zero rotations, integer pixel centers; q map follows the archived "
            "pixels_to_q_vectors_rectangular convention."
        )
    except LegacySaxsError as exc:
        values = None
        message = str(exc)
        warnings.append(message)

    image_present = data_path is not None and data_path.is_file()
    if data_path is None:
        warnings.append("Session has no data_path; select the original 2D image separately.")
    elif not image_present:
        warnings.append(
            f"Referenced 2D image is missing: {data_path}. Restore it or select a matching image before loading."
        )

    meta = document.get("meta")
    version = meta.get("version") if isinstance(meta, Mapping) else None
    if version != 2:
        warnings.append(
            f"Session metadata version is {version!r}; scalar geometry conversion is only enabled for Fusion v2."
        )
        values = None
        message = "Only Fusion v2 scalar geometry has a checked pixel and axis convention."

    session = LegacyFusionSession(
        version=version,
        data_path=data_path,
        image_present=image_present,
        geometry=geometry,
        geometry_values=values,
        geometry_supported=values is not None,
        geometry_message=message,
        pattern_options=(
            dict(document.get("pattern_options", {}))
            if isinstance(document.get("pattern_options"), Mapping)
            else {}
        ),
    )
    return session, warnings


def _read_json_text(path: Path) -> tuple[str | None, Any, str | None]:
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return None, None, f"Could not read UTF-8 text: {exc}"
    try:
        document = json.loads(
            raw_text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return raw_text, None, f"JSON was kept as raw text because it is not strict JSON: {exc}"
    return raw_text, document, None


def _npz_preview(path: Path) -> str:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return f"NPZ could not be inspected safely: {exc}"
    return "NPZ arrays (names only; values are not imported):\n" + "\n".join(names)


def _record_file(
    path: Path,
    *,
    role: str,
    relative_path: str | None = None,
    include_hash: bool = True,
) -> LegacyFileEvidence:
    if not path.is_file():
        return LegacyFileEvidence(path, relative_path, role, False, kind="missing")
    try:
        size = path.stat().st_size
        digest = _sha256(path) if include_hash else None
    except OSError as exc:
        return LegacyFileEvidence(path, relative_path, role, False, kind=f"unreadable: {exc}")
    return LegacyFileEvidence(path, relative_path, role, True, size, digest)


def _bundle_records(root: Path) -> tuple[LegacyFileEvidence, ...]:
    found: list[LegacyFileEvidence] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in _SKIP_BUNDLE_DIRS
            and not (current_path / name).is_symlink()
            and depth < 8
        )
        for name in sorted(filenames):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            found.append(
                _record_file(path.resolve(), role="bundle_file", relative_path=relative)
            )
            if len(found) > _MAX_BUNDLE_FILES:
                raise LegacySaxsError(
                    f"Selected folder contains more than {_MAX_BUNDLE_FILES} files; choose a narrower results folder."
                )
    return tuple(found)


def _bundle_primary_document(root: Path, files: tuple[LegacyFileEvidence, ...]) -> tuple[str | None, Any]:
    preferred = ("provenance.json", "manifest.json", "pattern_evidence.json", "grubb_evidence.json")
    by_name = {item.path.name.casefold(): item.path for item in files}
    selected = next((by_name[name] for name in preferred if name in by_name), None)
    if selected is None:
        return None, None
    raw_text, document, _warning = _read_json_text(selected)
    return raw_text, document


def inspect_legacy_saxs(path: str | os.PathLike[str]) -> LegacySaxsInspection:
    """Inspect one legacy session/evidence file or a selected results folder.

    Source files are only read. The returned `document` is the parsed legacy
    JSON mapping as-is; CSV values, result names and scientific quantities are
    not normalized into WingSAXS measurement fields.
    """

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        files = _bundle_records(source)
        raw_text, document = _bundle_primary_document(source, files)
        preview = [f"Read-only legacy result folder: {source}", f"Files: {len(files)}"]
        preview.extend(
            f"{item.relative_path} | {item.size_bytes} bytes | sha256={item.sha256}"
            for item in files
        )
        warnings = [] if files else ["The selected folder contains no regular files."]
        return LegacySaxsInspection(
            source,
            "legacy_evidence_bundle",
            "directory",
            None,
            None,
            document=document,
            raw_text=raw_text,
            preview_text="\n".join(preview),
            files=files,
            warnings=tuple(warnings),
        )
    if not source.is_file():
        raise FileNotFoundError(f"Legacy SAXS source does not exist: {source}")

    digest = _sha256(source)
    size = source.stat().st_size
    suffix = source.suffix.casefold()
    warnings: list[str] = []
    document: Any = None
    raw_text: str | None = None
    session: LegacyFusionSession | None = None
    format_name = suffix.lstrip(".") or "unknown"

    if suffix == ".json":
        raw_text, document, warning = _read_json_text(source)
        if warning:
            warnings.append(warning)
        session, session_warnings = _legacy_session_from_document(document, source)
        warnings.extend(session_warnings)
        artifact_kind = "legacy_fusion_session" if session else "legacy_json_evidence"
        preview = raw_text if raw_text is not None else "JSON text is unavailable."
    elif suffix in _TEXT_SUFFIXES:
        try:
            raw_text = source.read_text(encoding="utf-8-sig")
            preview = raw_text[:12000]
            if len(raw_text) > 12000:
                preview += "\n… preview truncated; the source file remains unchanged."
        except (OSError, UnicodeError) as exc:
            warnings.append(f"Could not decode text preview: {exc}")
            preview = f"Text file ({size} bytes); preview unavailable."
        artifact_kind = "legacy_table_or_text"
    elif suffix == ".npz":
        preview = _npz_preview(source)
        artifact_kind = "legacy_array_evidence"
    elif suffix == ".npy":
        try:
            array = np.load(source, mmap_mode="r", allow_pickle=False)
            preview = f"NumPy array: shape={array.shape}, dtype={array.dtype}; values were not copied."
            del array
        except (OSError, ValueError) as exc:
            preview = f"NumPy array could not be inspected safely: {exc}"
            warnings.append(preview)
        artifact_kind = "legacy_array_evidence"
    else:
        preview = f"Binary source: {suffix or '<no extension>'}, {size} bytes."
        artifact_kind = "legacy_binary_evidence"

    files = [_record_file(source, role="selected_artifact", include_hash=False)]
    if session is not None and session.data_path is not None:
        files.append(
            _record_file(
                session.data_path,
                role="referenced_2d_image",
                include_hash=False,
            )
        )
    if session is not None:
        summary = [
            f"Legacy SAXSAnalyzer Fusion session v{session.version}",
            f"Session: {source}",
            f"Session sha256: {digest}",
            f"2D image: {session.data_path if session.data_path else 'not specified'}",
            f"Image available: {session.image_present}",
            f"Geometry: {session.geometry_message}",
            "Legacy reported measurements are displayed as archived values; they are not converted to WingSAXS q*, L_N, or L_z.",
            "",
            "Original session metadata:",
            preview,
        ]
        preview = "\n".join(summary)
    return LegacySaxsInspection(
        source,
        artifact_kind,
        format_name,
        digest,
        size,
        document=document,
        raw_text=raw_text,
        preview_text=preview,
        session=session,
        files=tuple(files),
        warnings=tuple(warnings),
    )


def legacy_fusion_geometry_map(
    geometry: Mapping[str, Any], shape: tuple[int, int] | list[int]
) -> GeometryMaps:
    """Build the same flat q vectors used by SAXSAnalyzer Fusion v2.

    The archived implementation defines ``dx = x-cx``, ``dy = -(y-cy)``,
    square pixel pitch in mm, ``dist_mm``, and wavelength in Angstrom. Its
    q-magnitude is ``4*pi*sin(atan2(r_mm, dist_mm)/2)/wl_A * 10`` in nm^-1.
    This explicit scalar mapping avoids guessing pyFAI pixel-origin or axis
    conventions. It does not represent detector rotations or distortion.
    """

    try:
        if len(shape) != 2:
            raise ValueError
        rows, cols = (int(shape[0]), int(shape[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise LegacySaxsError(f"legacy geometry needs a 2-D detector shape, got {shape!r}") from exc
    if rows < 1 or cols < 1:
        raise LegacySaxsError(f"legacy geometry needs a non-empty 2-D detector shape, got {shape!r}")
    values = _validate_legacy_geometry(geometry)
    px_mm, distance_mm, wavelength_a = values["px_mm"], values["dist_mm"], values["wl_A"]

    y, x = np.indices((rows, cols), dtype=np.float64)
    dx = x - values["cx"]
    dy = -(y - values["cy"])
    radius_pix = np.hypot(dx, dy)
    radius_mm = radius_pix * px_mm
    theta = 0.5 * np.arctan2(radius_mm, distance_mm)
    q_nm_inv = (4.0 * np.pi * np.sin(theta) / wavelength_a) * 10.0
    inverse_radius = np.divide(
        1.0,
        radius_pix,
        out=np.zeros_like(radius_pix),
        where=radius_pix > 0,
    )
    qx = q_nm_inv * dx * inverse_radius
    qy = q_nm_inv * dy * inverse_radius
    chi = np.arctan2(qy, qx)
    finite = np.isfinite(q_nm_inv) & np.isfinite(qx) & np.isfinite(qy) & np.isfinite(chi)
    payload = {"shape": [rows, cols], "geometry": values, "formula": "saxsanalyzer-fusion-v2-flat-q-v1"}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    metadata = {
        "q_unit": "nm^-1",
        "geometry_source": "SAXSAnalyzer Fusion v2 scalar session fields",
        "geometry_model": "flat detector, square pixels, zero rotation",
        "pixel_center_convention": "integer detector indices are pixel centers (legacy convention)",
        "chi_convention": {"zero_degrees": "+qx", "ninety_degrees": "+qy", "positive_direction": "counter-clockwise in qx/qy"},
        "source_geometry": dict(geometry),
        "scientific_flags": ["legacy_scalar_geometry", "distortion_and_tilt_not_represented"],
    }
    return GeometryMaps(
        q_nm_inv=q_nm_inv,
        chi_rad=chi,
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=finite,
        metadata=metadata,
        fingerprint=fingerprint,
    )


def load_legacy_saxs_image(inspection: LegacySaxsInspection) -> LegacySaxsLoadedImage:
    """Load the selected v2 session image and its explicitly checked q map."""

    session = inspection.session
    if session is None:
        raise LegacySaxsError("Select a SAXSAnalyzer Fusion session JSON to load its image.")
    if not session.geometry_supported or session.geometry_values is None:
        raise LegacySaxsError(
            session.geometry_message
            or "Legacy session geometry is incomplete; inspect the session and supply a verified calibration."
        )
    if session.data_path is None or not session.image_present:
        raise LegacySaxsError(
            f"Legacy session image is unavailable: {session.data_path or 'data_path is empty'}. Restore or select the original image."
        )
    try:
        image = load_image(session.data_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise LegacySaxsError(
            f"Could not load legacy 2D image {session.data_path}: {exc}"
        ) from exc
    qmap = legacy_fusion_geometry_map(session.geometry_values, image.shape)
    image.metadata["legacy_saxs_session"] = {
        "session_path": str(inspection.path),
        "session_sha256": inspection.source_sha256,
        "geometry_fingerprint": qmap.fingerprint,
        "geometry_interpretation": "legacy Fusion v2 flat scalar geometry; no PONI transformations inferred",
        "legacy_measurements_converted": False,
    }
    return LegacySaxsLoadedImage(inspection=inspection, image=image, qmap=qmap)


def _receipt_file_records(inspection: LegacySaxsInspection) -> list[LegacyFileEvidence]:
    records = list(inspection.files)
    if inspection.path.is_file() and not any(item.path == inspection.path for item in records):
        records.insert(0, _record_file(inspection.path, role="selected_artifact"))
    elif inspection.path.is_file():
        records = [
            _record_file(item.path, role=item.role, relative_path=item.relative_path)
            if item.present
            else item
            for item in records
        ]
    if inspection.session and inspection.session.data_path is not None:
        if not any(item.path == inspection.session.data_path for item in records):
            records.append(
                _record_file(
                    inspection.session.data_path,
                    role="referenced_2d_image",
                )
            )
        else:
            records = [
                _record_file(item.path, role=item.role, relative_path=item.relative_path)
                if item.path == inspection.session.data_path and item.present
                else item
                for item in records
            ]
    return records


def compatibility_receipt(inspection: LegacySaxsInspection) -> dict[str, Any]:
    """Build a strict JSON receipt while leaving every original file untouched."""

    files = _receipt_file_records(inspection)
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": inspection.artifact_kind,
        "source": {
            "path": str(inspection.path),
            "format": inspection.format,
            "sha256": inspection.source_sha256,
            "size_bytes": inspection.source_size_bytes,
        },
        "source_files": [
            {
                "path": str(item.path),
                "relative_path": item.relative_path,
                "role": item.role,
                "present": item.present,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "kind": item.kind,
            }
            for item in files
        ],
        "original_metadata": inspection.document,
        "original_text": inspection.raw_text,
        "warnings": list(inspection.warnings),
        "measurement_interpretation": (
            "Legacy SAXSAnalyzer values, including any Ln_nm or Lz_nm fields, are preserved as reported. "
            "No legacy value is recalculated or mapped to WingSAXS q*, L_N, or L_z observables."
        ),
        "image_geometry_migration": (
            {
                "status": "legacy_flat_scalar_geometry_checked" if inspection.session.geometry_supported else "unsupported",
                "message": inspection.session.geometry_message,
                "raw_geometry": dict(inspection.session.geometry),
                "q_unit": "nm^-1" if inspection.session.geometry_supported else None,
            }
            if inspection.session is not None
            else None
        ),
    }


def write_compatibility_receipt(
    inspection: LegacySaxsInspection,
    target: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a hash manifest/receipt, never over an original source."""

    output = Path(target).expanduser().resolve()
    protected = {inspection.path.resolve()}
    if inspection.path.is_dir():
        try:
            output.relative_to(inspection.path.resolve())
        except ValueError:
            pass
        else:
            raise LegacySaxsError(
                "Receipt target is inside the selected legacy results folder; choose a separate output folder to keep the evidence bundle unchanged."
            )
    if inspection.session and inspection.session.data_path is not None:
        protected.add(inspection.session.data_path.resolve())
    protected.update(item.path.resolve() for item in inspection.files)
    if output in protected:
        raise LegacySaxsError("Receipt target resolves to a legacy source file; choose a separate output path.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Receipt already exists: {output}")

    payload = compatibility_receipt(inspection)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


__all__ = [
    "LegacyFileEvidence",
    "LegacyFusionSession",
    "LegacySaxsError",
    "LegacySaxsInspection",
    "LegacySaxsLoadedImage",
    "RECEIPT_SCHEMA",
    "compatibility_receipt",
    "inspect_legacy_saxs",
    "legacy_fusion_geometry_map",
    "load_legacy_saxs_image",
    "write_compatibility_receipt",
]
