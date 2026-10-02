from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from butterfly_saxs import annotation_pack, batch, csv_utils, p3_gate, preflight_context
from butterfly_saxs.p4_validation import _load_r0_rows


_DEFAULT_FIELD_LIMIT = 128 * 1024
_NOTES = 'observed first lobe\r\nsecond lobe, "partial"\nthird line\rfourth line'


@pytest.fixture(autouse=True)
def reset_csv_field_limit():
    # Exercise every reader independently, including after another reader has
    # already expanded the process-wide limit in a previous test.
    original = csv.field_size_limit(_DEFAULT_FIELD_LIMIT)
    try:
        yield
    finally:
        csv.field_size_limit(original)


@pytest.fixture
def large_metadata() -> str:
    value = json.dumps(
        {"diagnostics": {"observed_profile": [-3.125, 1.25e-9, 0.0] * 10_000}},
        ensure_ascii=False,
    )
    assert len(value) > _DEFAULT_FIELD_LIMIT
    return value


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_shared_reader_preserves_large_fields_and_line_endings(large_metadata: str) -> None:
    expected = {
        "scientific_metadata": large_metadata,
        "notes": _NOTES,
        "signed_value": "-3.125000000000000e-09",
        "unicode": "实测强度 Δq",
    }
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(expected))
    writer.writeheader()
    writer.writerow(expected)
    handle.seek(0)

    assert csv_utils.read_csv_rows(handle) == [expected]
    assert csv.field_size_limit() >= len(large_metadata)


def test_shared_reader_handles_narrow_platform_field_limit(
    monkeypatch: pytest.MonkeyPatch, large_metadata: str
) -> None:
    original = csv.field_size_limit
    attempts = []
    platform_maximum = (1 << 31) - 1

    def narrow_field_size_limit(new_limit=None):
        if new_limit is None:
            return original()
        attempts.append(new_limit)
        if new_limit > platform_maximum:
            raise OverflowError("Python int too large to convert to C long")
        return original(new_limit)

    monkeypatch.setattr(csv, "field_size_limit", narrow_field_size_limit)
    handle = io.StringIO(newline="")
    csv.writer(handle).writerows([["metadata"], [large_metadata]])
    handle.seek(0)

    assert csv_utils.read_csv_rows(handle) == [{"metadata": large_metadata}]
    assert attempts[0] == sys.maxsize
    assert _DEFAULT_FIELD_LIMIT < original() <= platform_maximum


def test_shared_reader_does_not_restore_a_lower_limit() -> None:
    assert csv_utils.read_csv_rows(io.StringIO("value\n1\n")) == [{"value": "1"}]
    expanded = csv.field_size_limit()
    assert csv_utils.read_csv_rows(io.StringIO("value\n2\n")) == [{"value": "2"}]
    assert csv.field_size_limit() == expanded


@pytest.mark.parametrize(
    "reader", [preflight_context.parse_manifest_file, annotation_pack._read_manifest_file]
)
def test_manifest_readers_preserve_large_metadata_and_multiline_notes(
    tmp_path: Path, large_metadata: str, reader
) -> None:
    expected = {
        "path": "frame.npy",
        "scientific_metadata": large_metadata,
        "notes": _NOTES,
        "time": "-1.25e-9",
    }
    manifest = _write_csv(tmp_path / "manifest.csv", [expected])
    original = manifest.read_bytes()

    assert reader(manifest) == [expected]
    assert manifest.read_bytes() == original


def test_batch_manifest_preserves_large_metadata_and_multiline_notes(
    tmp_path: Path, large_metadata: str
) -> None:
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        [{"path": "frame.npy", "scientific_metadata": large_metadata, "notes": _NOTES}],
    )

    refs = batch.build_frame_refs([], manifest=manifest)

    assert len(refs) == 1
    assert refs[0].path == tmp_path / "frame.npy"
    assert refs[0].metadata["scientific_metadata"] == large_metadata
    assert refs[0].metadata["notes"] == _NOTES


@pytest.fixture
def annotation_evidence(tmp_path: Path, large_metadata: str):
    blind_ids = [f"blind_{index:03d}" for index in range(1, 9)]
    image_hashes = {}
    files = {}
    for blind_id in blind_ids:
        image = tmp_path / f"{blind_id}.png"
        image.write_bytes(f"image-{blind_id}".encode())
        files[f"{blind_id}_png"] = image.name
        image_hashes[blind_id] = hashlib.sha256(image.read_bytes()).hexdigest()
    source_hash = "a" * 64
    for key in ("annotation_manifest", "annotator_a", "annotator_b", "consensus_review"):
        rows = [
            {
                "blind_id": blind_id,
                "role": "pilot",
                "source_path_relative_package": f"{blind_id}.npy",
                "selector": "default",
                "sha256": source_hash,
                "selection_reason": "fixed pilot frame",
                "valid_area": "[[0,0],[10,0],[10,10],[0,10]]",
                "beamstop": "[]",
                "streak": "[]",
                "overlap": "[]",
                "lobe_center_x": "5.0",
                "lobe_center_y": "6.0",
                "ridge_points": "[[4,5],[5,6],[6,5]]",
                "software": "manual-tool",
                "software_version": "1",
                "coordinate_system": annotation_pack.ANNOTATION_COORDINATE_SYSTEM,
                "image_version": image_hashes[blind_id],
                "annotation_time": "2026-08-28T12:00:00+08:00",
                "annotator": key,
                "reviewer": "expert",
                "review_time": "2026-08-28T13:00:00+08:00",
                "consensus_status": "accepted",
                "scientific_metadata": large_metadata if index == 0 else "{}",
                "notes": _NOTES,
            }
            for index, blind_id in enumerate(blind_ids)
        ]
        path = _write_csv(tmp_path / f"{key}.csv", rows)
        files[key] = path.name
    annotation = {
        "files": files,
        "blind_image_hashes": image_hashes,
        "immutable_output_hashes": {
            files["annotation_manifest"]: hashlib.sha256(
                (tmp_path / files["annotation_manifest"]).read_bytes()
            ).hexdigest()
        },
        "input_hashes": [
            {"sha256_before": source_hash, "sha256_after": source_hash, "unchanged": True}
        ],
        "input": {"read_only": True},
        "human_evidence": {"mode": "two_independent_annotators"},
    }
    status = tmp_path / "annotation_status.json"
    status.write_text(json.dumps(annotation), encoding="utf-8")
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "schema_version": "lamellarsaxs2d.pilot_evidence.v1",
                "status": "complete",
                "frame_count": 8,
                "blind_ids": blind_ids,
                "annotation_status_sha256": hashlib.sha256(status.read_bytes()).hexdigest(),
                "consensus_sha256": hashlib.sha256(
                    (tmp_path / files["consensus_review"]).read_bytes()
                ).hexdigest(),
                "frame_results": [
                    {"blind_id": blind_id, "consensus_status": "accepted"}
                    for blind_id in blind_ids
                ],
                "reviewed_by": "expert",
                "reviewed_at": "2026-08-28T14:00:00+08:00",
            }
        ),
        encoding="utf-8",
    )
    pilot_value = {
        "source": pilot.name,
        "sha256": hashlib.sha256(pilot.read_bytes()).hexdigest(),
        "status": "complete",
        "frame_count": 8,
    }
    return status, annotation, pilot_value


def test_p3_annotation_accepts_complete_large_csv_evidence(annotation_evidence) -> None:
    status, annotation, _ = annotation_evidence

    assert p3_gate._annotation_csv_evidence_complete(status, annotation)


def test_p3_pilot_accepts_large_consensus_csv(annotation_evidence) -> None:
    status, _, pilot_value = annotation_evidence

    assert p3_gate._pilot_source_complete(status.parent / "thresholds.json", pilot_value, status)


def test_p4_preserves_large_csv_metadata(annotation_evidence, large_metadata: str) -> None:
    status, annotation, _ = annotation_evidence

    rows = _load_r0_rows(status.parent / annotation["files"]["annotation_manifest"])

    assert len(rows) == 8
    assert rows[0]["scientific_metadata"] == large_metadata
    assert all(row["notes"] == _NOTES for row in rows)
    assert rows[0]["lobe_center_x"] == "5.0"


@pytest.mark.parametrize("module", [preflight_context, annotation_pack])
def test_manifest_csv_errors_keep_domain_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module
) -> None:
    manifest = _write_csv(tmp_path / "manifest.csv", [{"path": "frame.npy"}])

    def malformed_reader(lines):
        raise csv.Error("malformed CSV record")

    monkeypatch.setattr(module, "read_csv_rows", malformed_reader)
    if module is preflight_context:
        reader = module.parse_manifest_file
        error = ValueError
    else:
        reader = module._read_manifest_file
        error = module.AnnotationPackError
    with pytest.raises(error, match="manifest.*malformed CSV record") as exc_info:
        reader(manifest)
    assert isinstance(exc_info.value.__cause__, csv.Error)


def test_p3_csv_errors_still_fail_evidence_checks(
    annotation_evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, annotation, pilot_value = annotation_evidence

    def malformed_reader(lines):
        raise csv.Error("malformed CSV record")

    monkeypatch.setattr(p3_gate, "read_csv_rows", malformed_reader)

    assert not p3_gate._annotation_csv_evidence_complete(status, annotation)
    assert not p3_gate._pilot_source_complete(
        status.parent / "thresholds.json", pilot_value, status
    )
