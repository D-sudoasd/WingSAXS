from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_jac_validation_manifest import audit_manifest, main


TEMPLATE = Path(__file__).parents[1] / "examples" / "validation" / "jac_experimental_validation_template.json"


def _write_manifest(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def test_shipped_template_is_pending_and_audit_makes_no_scientific_claims() -> None:
    report = audit_manifest(TEMPLATE)

    assert report["audit_only"] is True
    assert report["manifest"]["declared_status"] == "pending"
    assert report["counts"]["dataset_count"] == 2
    assert report["counts"]["user_session_count"] == 2
    assert {item["declared_status"] for item in report["datasets"]} == {"pending"}
    assert {item["declared_status"] for item in report["independent_user_sessions"]} == {"pending"}
    assert {item["declared_user_success"] for item in report["independent_user_sessions"]} == {None}
    assert report["scientific_measurement_status"] == "NOT_ASSESSED"
    assert report["repeatability_status"] == "NOT_ASSESSED"
    assert report["manual_agreement_status"] == "NOT_ASSESSED"
    assert report["independent_user_success_status"] == "NOT_ASSESSED"
    assert report["counts"]["reference_error_count"] == 0
    assert report["evidence_completeness"] == "pending"
    assert report["counts"]["warning_count"] > 0
    assert all(item["raw_frame_record_count"] == 0 for item in report["datasets"])
    assert all(
        item["status"] == "pending_path" and item["supplied_path"] is None
        for item in report["file_references"]
    )
    assert any(
        item["json_pointer"] == "/datasets/0/mask/creation_record"
        for item in report["file_references"]
    )


def test_audit_checks_referenced_paths_and_sha_without_editing_manifest(tmp_path: Path) -> None:
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    raw_file = tmp_path / "frame-manifest.csv"
    raw_file.write_text("frame_id,path\n001,frame_001.edf\n", encoding="utf-8")
    expected = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    document["datasets"][0]["frame_manifest"] = {"path": raw_file.name, "sha256": expected}
    document["datasets"][1]["calibration"]["poni"] = {
        "path": "missing.poni",
        "sha256": None,
    }
    manifest = _write_manifest(tmp_path, document)
    original = manifest.read_bytes()

    report = audit_manifest(manifest, base_dir=tmp_path)

    assert report["counts"]["reference_error_count"] == 1
    frame_ref = next(item for item in report["file_references"] if item["supplied_path"] == raw_file.name)
    assert frame_ref["status"] == "exists_hash_match"
    missing_ref = next(item for item in report["file_references"] if item["supplied_path"] == "missing.poni")
    assert missing_ref["status"] == "missing_file"
    assert manifest.read_bytes() == original
    assert report["scientific_measurement_status"] == "NOT_ASSESSED"


def test_cli_returns_one_for_missing_references_and_writes_audit_report(
    tmp_path: Path, capsys
) -> None:
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    document["datasets"][0]["frame_manifest"] = {"path": "absent.csv", "sha256": None}
    manifest = _write_manifest(tmp_path, document)
    report_path = tmp_path / "audit.json"

    code = main([str(manifest), "--base-dir", str(tmp_path), "--output", str(report_path)])

    assert code == 1
    assert capsys.readouterr().out == ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["audit_status"] == "reference_errors_present"
    assert report["scientific_measurement_status"] == "NOT_ASSESSED"


def test_pending_template_returns_warning_without_marking_failure(capsys) -> None:
    code = main([str(TEMPLATE)])

    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["audit_status"] == "pending_evidence"
    assert report["evidence_completeness"] == "pending"
    assert report["scientific_measurement_status"] == "NOT_ASSESSED"


def test_cli_rejects_manifest_missing_a_required_dataset_role(tmp_path: Path, capsys) -> None:
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    document["datasets"][1]["role"] = "primary_sequence"
    manifest = _write_manifest(tmp_path, document)

    code = main([str(manifest)])

    assert code == 2
    captured = capsys.readouterr()
    assert "missing required roles" in captured.err
    error_report = json.loads(captured.out)
    assert error_report["audit_status"] == "input_error"
    assert error_report["scientific_measurement_status"] == "NOT_ASSESSED"


def test_output_cannot_overwrite_existing_or_declared_input_references(
    tmp_path: Path, capsys
) -> None:
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    protected = tmp_path / "raw-image.bin"
    protected.write_bytes(b"raw input must remain unchanged")
    original = protected.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    document["datasets"][0]["frame_manifest"] = {"path": protected.name, "sha256": digest}
    document["datasets"][1]["frame_manifest"] = {"path": "not-yet-created.csv", "sha256": None}
    manifest = _write_manifest(tmp_path, document)

    existing_code = main(
        [str(manifest), "--base-dir", str(tmp_path), "--output", str(protected), "--force"]
    )
    existing_output = capsys.readouterr()
    missing_target = tmp_path / "not-yet-created.csv"
    missing_code = main(
        [str(manifest), "--base-dir", str(tmp_path), "--output", str(missing_target), "--force"]
    )
    missing_output = capsys.readouterr()

    assert existing_code == 2
    assert "must not overwrite a referenced input" in existing_output.err
    assert json.loads(existing_output.out)["audit_status"] == "input_error"
    assert protected.read_bytes() == original
    assert missing_code == 2
    assert "must not overwrite a referenced input" in missing_output.err
    assert json.loads(missing_output.out)["audit_status"] == "input_error"
    assert not missing_target.exists()


def test_audit_reports_actionable_error_for_unhashable_dataset_role(tmp_path: Path) -> None:
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    document["datasets"][1]["role"] = ["independent_dataset"]
    manifest = _write_manifest(tmp_path, document)

    with pytest.raises(ValueError, match="non-empty string role"):
        audit_manifest(manifest)


def test_audit_reports_actionable_error_for_unhashable_dataset_id(tmp_path: Path) -> None:
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    document["datasets"][1]["dataset_id"] = {"invalid": "id"}
    manifest = _write_manifest(tmp_path, document)

    with pytest.raises(ValueError, match="non-empty string dataset_id"):
        audit_manifest(manifest)


def test_audit_reports_actionable_error_for_missing_user_session_id(tmp_path: Path) -> None:
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    document["independent_user_sessions"][0]["session_id"] = None
    manifest = _write_manifest(tmp_path, document)

    with pytest.raises(ValueError, match="non-empty string session_id"):
        audit_manifest(manifest)


def test_invalid_calibration_source_is_reported_as_metadata_gap(tmp_path: Path) -> None:
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    document["datasets"][0]["calibration"]["source_type"] = []
    manifest = _write_manifest(tmp_path, document)

    report = audit_manifest(manifest)

    gaps = report["datasets"][0]["metadata_gaps"]
    assert any(item.startswith("calibration.source_type") for item in gaps)
