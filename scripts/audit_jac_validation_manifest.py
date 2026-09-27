"""Audit a JAC real-data/user-evidence manifest without judging scientific results.

The audit checks the manifest contract, reports required metadata gaps, and
verifies explicitly supplied file paths and optional SHA-256 values. It never
changes a pending status or infers experiment validity, fit accuracy, human
agreement, or independent-user success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "wingsaxs.jac.experimental_validation.v1"
REQUIRED_DATASET_ROLES = {"primary_sequence", "independent_dataset"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestAuditError(ValueError):
    """Raised when the manifest cannot be structurally audited."""


def _reject_nonfinite(value: str) -> None:
    raise ManifestAuditError(f"non-finite JSON value is not allowed: {value}")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except OSError as exc:
        raise ManifestAuditError(f"cannot read manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestAuditError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestAuditError("manifest root must be a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ManifestAuditError(
            f"schema_version must be {SCHEMA_VERSION!r}; found {raw.get('schema_version')!r}"
        )

    datasets = raw.get("datasets")
    if not isinstance(datasets, list):
        raise ManifestAuditError("datasets must be a JSON array")
    if any(not isinstance(item, dict) for item in datasets):
        raise ManifestAuditError("each datasets entry must be a JSON object")
    roles = [item.get("role") for item in datasets]
    if any(not isinstance(role, str) or not role.strip() for role in roles):
        raise ManifestAuditError("each dataset needs a non-empty string role")
    missing_roles = sorted(REQUIRED_DATASET_ROLES - set(roles))
    if missing_roles:
        raise ManifestAuditError(f"datasets is missing required roles: {', '.join(missing_roles)}")
    ids = [item.get("dataset_id") for item in datasets]
    if any(not isinstance(item_id, str) or not item_id.strip() for item_id in ids):
        raise ManifestAuditError("each dataset needs a non-empty string dataset_id")
    if len(ids) != len(set(ids)):
        raise ManifestAuditError("dataset_id values must be unique")

    sessions = raw.get("independent_user_sessions")
    if not isinstance(sessions, list) or len(sessions) < 2:
        raise ManifestAuditError("at least two independent_user_sessions are required")
    if any(not isinstance(item, dict) for item in sessions):
        raise ManifestAuditError("each independent_user_sessions entry must be a JSON object")
    session_ids = [item.get("session_id") for item in sessions]
    if any(not isinstance(session_id, str) or not session_id.strip() for session_id in session_ids):
        raise ManifestAuditError("each user session needs a non-empty string session_id")
    if len(session_ids) != len(set(session_ids)):
        raise ManifestAuditError("each user session needs a unique session_id")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_pointer(parent: str, key: str | int) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}"


def _file_references(node: Any, pointer: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            found.append((pointer or "/", node))
        for key, value in node.items():
            if key in {"path", "sha256"}:
                continue
            found.extend(_file_references(value, _json_pointer(pointer, key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_file_references(value, _json_pointer(pointer, index)))
    return found


def _audit_file_reference(
    pointer: str, reference: dict[str, Any], base_dir: Path
) -> dict[str, Any]:
    supplied_path = reference.get("path")
    expected_hash = reference.get("sha256")
    result: dict[str, Any] = {
        "json_pointer": pointer,
        "supplied_path": supplied_path,
        "declared_sha256": expected_hash,
        "resolved_path": None,
        "actual_sha256": None,
        "status": "pending_path",
        "errors": [],
    }
    if supplied_path is None or supplied_path == "":
        if expected_hash not in (None, ""):
            result["status"] = "invalid_reference"
            result["errors"].append("sha256 is present but path is empty")
        return result
    if not isinstance(supplied_path, str):
        result["status"] = "invalid_reference"
        result["errors"].append("path must be a string or null")
        return result

    candidate = Path(supplied_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    resolved_path = candidate.resolve(strict=False)
    result["resolved_path"] = str(resolved_path)
    if not resolved_path.is_file():
        result["status"] = "missing_file"
        result["errors"].append("referenced file does not exist")
        return result

    try:
        actual_hash = _sha256_file(resolved_path)
    except OSError as exc:
        result["status"] = "unreadable_file"
        result["errors"].append(f"cannot read referenced file: {exc}")
        return result
    result["actual_sha256"] = actual_hash

    if expected_hash in (None, ""):
        result["status"] = "exists_unhashed"
    elif not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
        result["status"] = "invalid_sha256"
        result["errors"].append("declared sha256 must be 64 lowercase hexadecimal characters")
    elif expected_hash != actual_hash:
        result["status"] = "sha256_mismatch"
        result["errors"].append("declared sha256 does not match file contents")
    else:
        result["status"] = "exists_hash_match"
    return result


def _get_value(mapping: dict[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _dataset_metadata_gaps(dataset: dict[str, Any]) -> list[str]:
    required = (
        "sample.material_description",
        "instrument.facility",
        "instrument.instrument_name",
        "instrument.detector",
        "instrument.wavelength_angstrom",
        "instrument.sample_to_detector_distance_mm",
        "acquisition.state_variable",
        "acquisition.state_unit",
        "frame_manifest.path",
        "mask.path",
        "mask.polarity",
        "mask.creation_record.path",
        "corrections.status",
        "corrections.record.path",
        "background.status",
        "background.method",
        "background.record.path",
        "uncertainty_validation.method",
        "uncertainty_validation.analysis_record.path",
        "uncertainty_validation.calibration_sensitivity.analysis_record.path",
        "manual_comparison.reference_method",
        "manual_comparison.consensus_record.path",
        "manual_comparison.analysis_record.path",
        "analysis.wingsaxs_version",
        "analysis.configuration.path",
        "analysis.export.path",
    )
    gaps = [field for field in required if not _is_filled(_get_value(dataset, field))]
    raw_frames = dataset.get("raw_frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        gaps.append("raw_frames (at least one frame record)")
    elif any(not isinstance(item, dict) or not _is_filled(item.get("path")) for item in raw_frames):
        gaps.append("raw_frames[].path")
    if not _is_filled(_get_value(dataset, "manual_comparison.trace_records")):
        gaps.append("manual_comparison.trace_records")
    for status_field in (
        "corrections.status",
        "background.status",
        "uncertainty_validation.status",
        "uncertainty_validation.calibration_sensitivity.status",
        "manual_comparison.status",
    ):
        status_value = _get_value(dataset, status_field)
        if not isinstance(status_value, str) or status_value.strip().casefold() == "pending":
            if status_field not in gaps:
                gaps.append(status_field)
    for field in (
        "uncertainty_validation.coverage_by_parameter",
        "uncertainty_validation.median_interval_width_by_parameter",
        "uncertainty_validation.eligible_observation_count_by_parameter",
        "uncertainty_validation.calibration_sensitivity.perturbations",
    ):
        if not _is_filled(_get_value(dataset, field)):
            gaps.append(field)
    source_type = _get_value(dataset, "calibration.source_type")
    calibration_source_fields = {
        "poni": "calibration.poni.path",
        "embedded_q_map": "calibration.q_map.path",
        "other_verified_source": "calibration.other_verified_source.path",
    }
    if isinstance(source_type, str) and source_type in calibration_source_fields:
        source_field = calibration_source_fields[source_type]
        if not _is_filled(_get_value(dataset, source_field)):
            gaps.append(source_field)
        if not _is_filled(_get_value(dataset, "calibration.verification_record.path")):
            gaps.append("calibration.verification_record.path")
    elif not isinstance(source_type, str) or source_type not in {"pixel-q", "unknown"}:
        gaps.append("calibration.source_type (poni, embedded_q_map, other_verified_source, pixel-q, or unknown)")
    q_unit = _get_value(dataset, "calibration.q_unit")
    allowed_q_units = {
        "nm^-1",
        "Å^-1",
        "angstrom^-1",
        "A^-1",
        "1/nm",
        "1/Å",
        "pixel-q",
        "unknown",
    }
    if not isinstance(q_unit, str) or q_unit.strip() not in allowed_q_units:
        gaps.append("calibration.q_unit (nm^-1, Å^-1, pixel-q, or unknown)")
    if dataset.get("role") == "primary_sequence":
        for field in (
            "repeatability.design",
            "repeatability.analysis_record.path",
            "repeatability.status",
        ):
            if not _is_filled(_get_value(dataset, field)):
                gaps.append(field)
        repeatability_status = _get_value(dataset, "repeatability.status")
        if not isinstance(repeatability_status, str) or repeatability_status.strip().casefold() == "pending":
            if "repeatability.status" not in gaps:
                gaps.append("repeatability.status")
        if not _is_filled(_get_value(dataset, "repeatability.repeat_groups")):
            gaps.append("repeatability.repeat_groups")
    return gaps


def _user_session_metadata_gaps(session: dict[str, Any]) -> list[str]:
    required = (
        "independent_of_development",
        "software_version",
        "software_source.path",
        "environment.operating_system",
        "environment.python_version",
        "environment.installation_method",
        "protocol_document.path",
        "started_at",
        "completed_at",
        "artifacts.single_frame_result.path",
        "artifacts.sequence_export.path",
        "artifacts.session_log.path",
        "human_review.success",
        "human_review.review_record.path",
        "user_success",
    )
    gaps = [field for field in required if not _is_filled(_get_value(session, field))]
    if session.get("independent_of_development") is not True:
        field = "independent_of_development (must be explicitly verified true)"
        if field not in gaps:
            gaps.append(field)
    tasks = session.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        gaps.append("tasks")
    elif any(not isinstance(item, dict) or item.get("status") != "complete" for item in tasks):
        gaps.append("tasks[].status")
    return gaps


def audit_manifest(manifest_path: str | Path, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Return an audit-only report for a dataset and user-session manifest."""
    path = Path(manifest_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise ManifestAuditError(f"manifest file does not exist: {path}")
    manifest = _read_manifest(path)
    root = Path(base_dir).expanduser().resolve(strict=False) if base_dir else path.parent
    if not root.is_dir():
        raise ManifestAuditError(f"base directory does not exist: {root}")

    references = [
        _audit_file_reference(pointer, reference, root)
        for pointer, reference in _file_references(manifest)
    ]
    datasets = []
    for dataset in manifest["datasets"]:
        if not isinstance(dataset, dict):
            raise ManifestAuditError("each datasets entry must be a JSON object")
        datasets.append(
            {
                "dataset_id": dataset.get("dataset_id"),
                "role": dataset.get("role"),
                "declared_status": dataset.get("status", "pending"),
                "raw_frame_record_count": len(dataset.get("raw_frames", []))
                if isinstance(dataset.get("raw_frames"), list)
                else None,
                "metadata_gaps": _dataset_metadata_gaps(dataset),
            }
        )
    sessions = []
    for session in manifest["independent_user_sessions"]:
        if not isinstance(session, dict):
            raise ManifestAuditError("each independent_user_sessions entry must be a JSON object")
        sessions.append(
            {
                "session_id": session.get("session_id"),
                "declared_status": session.get("status", "pending"),
                "declared_user_success": session.get("user_success"),
                "metadata_gaps": _user_session_metadata_gaps(session),
            }
        )

    reference_errors = [
        item for item in references if item["errors"]
    ]
    metadata_gap_count = sum(
        len(item["metadata_gaps"]) for item in [*datasets, *sessions]
    )
    pending_reference_count = sum(
        item["status"] in {"pending_path", "exists_unhashed"} for item in references
    )
    pending_status_count = int(manifest.get("manifest_status", "pending") != "complete")
    pending_status_count += sum(item["status"] != "complete" for item in manifest["datasets"])
    pending_status_count += sum(
        item["status"] != "complete" for item in manifest["independent_user_sessions"]
    )
    warning_count = (
        len(reference_errors)
        + pending_reference_count
        + metadata_gap_count
        + pending_status_count
    )
    try:
        manifest_sha256 = _sha256_file(path)
    except OSError as exc:
        raise ManifestAuditError(f"cannot hash manifest: {exc}") from exc

    return {
        "report_type": "wingsaxs.jac.experimental_validation_manifest_audit.v1",
        "audit_only": True,
        "manifest": {
            "path": str(path),
            "sha256": manifest_sha256,
            "declared_status": manifest.get("manifest_status", "pending"),
            "base_directory": str(root),
        },
        "audit_status": (
            "reference_errors_present"
            if reference_errors
            else "pending_evidence"
            if warning_count
            else "no_reported_manifest_warnings"
        ),
        "evidence_completeness": "pending" if warning_count else "requires_human_review",
        "counts": {
            "dataset_count": len(datasets),
            "user_session_count": len(sessions),
            "file_reference_count": len(references),
            "pending_or_unhashed_reference_count": pending_reference_count,
            "pending_declared_status_count": pending_status_count,
            "reference_error_count": len(reference_errors),
            "metadata_gap_count": metadata_gap_count,
            "warning_count": warning_count,
        },
        "datasets": datasets,
        "independent_user_sessions": sessions,
        "file_references": references,
        "scientific_measurement_status": "NOT_ASSESSED",
        "repeatability_status": "NOT_ASSESSED",
        "manual_agreement_status": "NOT_ASSESSED",
        "independent_user_success_status": "NOT_ASSESSED",
        "scope": [
            "manifest structure and required metadata gaps",
            "existence of explicitly referenced files",
            "SHA-256 comparison when a digest is supplied",
        ],
        "does_not_establish": [
            "scientific validity or measurement accuracy",
            "repeatability or uncertainty calibration",
            "manual-trace agreement",
            "independence or success of any user session",
            "JAC submission readiness or acceptance",
        ],
    }


def _write_report(report: dict[str, Any], output: Path, manifest_path: Path, force: bool) -> None:
    destination = output.expanduser().resolve(strict=False)
    if destination == manifest_path.expanduser().resolve(strict=False):
        raise ManifestAuditError("report output must not overwrite the input manifest")
    referenced_paths = {
        Path(item["resolved_path"]).resolve(strict=False)
        for item in report["file_references"]
        if isinstance(item.get("resolved_path"), str)
    }
    if destination in referenced_paths:
        raise ManifestAuditError("report output must not overwrite a referenced input or evidence file")
    if destination.exists() and not force:
        raise ManifestAuditError(f"output already exists; pass --force to replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    destination.write_text(payload, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="completed or pending JAC validation manifest")
    parser.add_argument("--base-dir", type=Path, help="root for relative file references")
    parser.add_argument("--output", type=Path, help="write the audit report to a new JSON file")
    parser.add_argument("--force", action="store_true", help="replace an existing report output")
    args = parser.parse_args(argv)

    try:
        report = audit_manifest(args.manifest, base_dir=args.base_dir)
        if args.output:
            _write_report(report, args.output, args.manifest, args.force)
        else:
            print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    except (ManifestAuditError, OSError, TypeError, ValueError) as exc:
        error_report = {
            "report_type": "wingsaxs.jac.experimental_validation_manifest_audit.v1",
            "audit_only": True,
            "audit_status": "input_error",
            "evidence_completeness": "not_audited",
            "error": str(exc),
            "scientific_measurement_status": "NOT_ASSESSED",
            "repeatability_status": "NOT_ASSESSED",
            "manual_agreement_status": "NOT_ASSESSED",
            "independent_user_success_status": "NOT_ASSESSED",
        }
        print(json.dumps(error_report, ensure_ascii=True, allow_nan=False))
        print(f"manifest audit error: {exc}", file=sys.stderr)
        return 2
    return 1 if report["counts"]["warning_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
