from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from butterfly_saxs.legacy_saxs import (
    LegacySaxsError,
    compatibility_receipt,
    inspect_legacy_saxs,
    legacy_fusion_geometry_map,
    load_legacy_saxs_image,
    write_compatibility_receipt,
)


def _session(path, *, data_path="detector.csv", version=2):
    document = {
        "meta": {"version": version, "application": "SAXSAnalyzer"},
        "data_path": data_path,
        "geometry": {
            "px_mm": 0.1,
            "dist_mm": 100.0,
            "wl_A": 1.54,
            "cx": 2,
            "cy": 1,
        },
        "pattern_options": {"q_min": 0.02, "q_max": 0.7},
        "legacy_results": {"Ln_nm": 42.5, "Lz_nm": 111.0},
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def test_v2_session_load_preserves_image_and_reconstructs_legacy_q_convention(tmp_path):
    source = tmp_path / "detector.csv"
    pixels = np.arange(20, dtype=np.int32).reshape(4, 5)
    np.savetxt(source, pixels, delimiter=",")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    session_path = tmp_path / "sample.saxs_fusion_session.json"
    session_document = _session(session_path)
    session_hash = hashlib.sha256(session_path.read_bytes()).hexdigest()

    inspection = inspect_legacy_saxs(session_path)
    assert inspection.artifact_kind == "legacy_fusion_session"
    assert inspection.session is not None
    assert inspection.session.geometry_supported
    assert inspection.session.image_present
    loaded = load_legacy_saxs_image(inspection)
    np.testing.assert_array_equal(loaded.image.data, pixels)
    assert loaded.image.metadata["legacy_saxs_session"]["legacy_measurements_converted"] is False
    assert loaded.image.metadata["legacy_saxs_session"]["session_sha256"] == session_hash

    # Archive convention: x-right and y-up, with integer indices at pixel centers.
    assert loaded.qmap.q_nm_inv[1, 2] == pytest.approx(0.0)
    assert loaded.qmap.qx_nm_inv[1, 3] > 0
    assert loaded.qmap.qy_nm_inv[0, 2] > 0
    assert loaded.qmap.qy_nm_inv[2, 2] < 0
    radius_mm = 0.1
    expected_q = 4 * np.pi * np.sin(0.5 * np.arctan2(radius_mm, 100.0)) / 1.54 * 10
    assert loaded.qmap.q_nm_inv[0, 2] == pytest.approx(expected_q)
    assert loaded.qmap.metadata["geometry_source"] == "SAXSAnalyzer Fusion v2 scalar session fields"

    receipt = compatibility_receipt(inspection)
    assert receipt["original_metadata"] == session_document
    assert receipt["image_geometry_migration"]["status"] == "legacy_flat_scalar_geometry_checked"
    assert "No legacy value is recalculated" in receipt["measurement_interpretation"]
    assert receipt["source_files"][0]["sha256"] == session_hash
    image_record = next(item for item in receipt["source_files"] if item["role"] == "referenced_2d_image")
    assert image_record["sha256"] == source_hash
    assert hashlib.sha256(session_path.read_bytes()).hexdigest() == session_hash
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_incomplete_or_non_v2_session_does_not_enable_geometry(tmp_path):
    image_path = tmp_path / "detector.csv"
    image_path.write_text("1,2\n3,4\n", encoding="utf-8")
    session_path = tmp_path / "unsupported.json"
    _session(session_path, version=1)
    inspection = inspect_legacy_saxs(session_path)
    assert inspection.session is not None
    assert not inspection.session.geometry_supported
    assert inspection.session.image_present
    assert any("only enabled for Fusion v2" in warning for warning in inspection.warnings)
    with pytest.raises(LegacySaxsError, match="Fusion v2 scalar geometry"):
        load_legacy_saxs_image(inspection)

    with pytest.raises(LegacySaxsError, match="positive"):
        legacy_fusion_geometry_map(
            {"px_mm": 0, "dist_mm": 100, "wl_A": 1.5, "cx": 0, "cy": 0},
            (2, 2),
        )


@pytest.mark.parametrize(
    "geometry_update, message",
    [
        ({"px_x_mm": 0.1, "px_y_mm": 0.2}, "rectangular pixels"),
        ({"px_x_mm": 0.1}, "incomplete"),
        ({"q_unit": "A^-1"}, "nm^-1"),
        ({"detector_tilt": 1.0}, "detector_tilt"),
    ],
)
def test_v2_geometry_blocks_unrepresented_pixel_pitch_units_and_transforms(
    tmp_path, geometry_update, message
):
    session_path = tmp_path / "unsupported-geometry.json"
    document = _session(session_path)
    document["geometry"].update(geometry_update)
    session_path.write_text(json.dumps(document), encoding="utf-8")
    inspection = inspect_legacy_saxs(session_path)
    assert inspection.session is not None
    assert not inspection.session.geometry_supported
    assert message.casefold() in inspection.session.geometry_message.casefold()


def test_missing_session_image_is_actionable_and_receipt_retains_missing_path(tmp_path):
    session_path = tmp_path / "missing.saxs_fusion_session.json"
    _session(session_path, data_path="gone.csv")
    inspection = inspect_legacy_saxs(session_path)
    assert inspection.session is not None
    assert inspection.session.geometry_supported
    assert not inspection.session.image_present
    assert any("Referenced 2D image is missing" in warning for warning in inspection.warnings)
    receipt = compatibility_receipt(inspection)
    missing = next(item for item in receipt["source_files"] if item["role"] == "referenced_2d_image")
    assert missing["present"] is False
    assert missing["path"].endswith("gone.csv")
    with pytest.raises(LegacySaxsError, match="unavailable"):
        load_legacy_saxs_image(inspection)


def test_receipt_refuses_overwriting_any_source_and_adding_file_to_bundle(tmp_path):
    (tmp_path / "detector.csv").write_text("1,2\n3,4\n", encoding="utf-8")
    session_path = tmp_path / "session.json"
    _session(session_path)
    inspection = inspect_legacy_saxs(session_path)
    with pytest.raises(LegacySaxsError, match="resolves to a legacy source"):
        write_compatibility_receipt(inspection, session_path, overwrite=True)

    bundle = tmp_path / "results"
    bundle.mkdir()
    evidence = bundle / "pattern_evidence.json"
    evidence.write_text('{"reported": {"Ln_nm": 7.0}}', encoding="utf-8")
    bundle_inspection = inspect_legacy_saxs(bundle)
    assert bundle_inspection.artifact_kind == "legacy_evidence_bundle"
    with pytest.raises(LegacySaxsError, match="inside the selected legacy results folder"):
        write_compatibility_receipt(bundle_inspection, bundle / "inside.json")
    out = tmp_path / "bundle-receipt.json"
    written = write_compatibility_receipt(bundle_inspection, out)
    exported = json.loads(written.read_text(encoding="utf-8"))
    assert exported["source_files"][0]["sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert exported["original_metadata"] == {"reported": {"Ln_nm": 7.0}}


def test_text_and_numpy_evidence_are_described_without_conversion(tmp_path):
    npz_path = tmp_path / "full2d_fit_evidence.npz"
    np.savez(npz_path, intensity=np.arange(6).reshape(2, 3), fit_parameters=np.array([1.0, 2.0]))
    npz_inspection = inspect_legacy_saxs(npz_path)
    assert npz_inspection.artifact_kind == "legacy_array_evidence"
    assert "intensity" in npz_inspection.preview_text
    assert "values are not imported" in npz_inspection.preview_text

    csv_path = tmp_path / "pattern_evidence.csv"
    csv_path.write_text("q,Ln_nm\n0.1,20\n", encoding="utf-8")
    csv_inspection = inspect_legacy_saxs(csv_path)
    assert csv_inspection.artifact_kind == "legacy_table_or_text"
    assert "Ln_nm" in csv_inspection.preview_text
    receipt = compatibility_receipt(csv_inspection)
    assert receipt["source"]["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert "No legacy value is recalculated" in receipt["measurement_interpretation"]
