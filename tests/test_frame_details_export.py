from __future__ import annotations

from enum import Enum
import importlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest

from butterfly_saxs.batch import BatchRunResult, FrameFitResult, FrameRef, _checkpoint_safe
from butterfly_saxs.export import StreamingBatchExporter, export_batch
from butterfly_saxs.pipeline import PipelineResult
from butterfly_saxs.validation import AnalysisDomain


class DetectorOrientation(Enum):
    HORIZONTAL = "horizontal"


def _pipeline_result() -> PipelineResult:
    image = np.arange(16, dtype=float).reshape(4, 4)
    qx = np.linspace(-1, 1, 16).reshape(4, 4)
    qy = qx.T.copy()
    q = np.hypot(qx, qy)
    metadata: dict[str, object] = {"orientation": DetectorOrientation.HORIZONTAL}
    metadata["self"] = metadata
    domain = AnalysisDomain(
        image_shape=image.shape,
        q_window=(0.1, 1.4),
        finite_mask=np.ones(image.shape, dtype=bool),
        detector_valid_mask=np.ones(image.shape, dtype=bool),
        external_valid_mask=np.ones(image.shape, dtype=bool),
        q_window_mask=np.ones(image.shape, dtype=bool),
        roi_exclusion_mask=np.zeros(image.shape, dtype=bool),
        weight_valid_mask=np.ones(image.shape, dtype=bool),
        fit_valid_mask=np.ones(image.shape, dtype=bool),
        sampled_valid_mask=np.ones(image.shape, dtype=bool),
        weight_kind="none",
    )
    return PipelineResult(
        image=image,
        qmap={
            "qx": qx,
            "qy": qy,
            "q": q,
            "q_unit": "nm^-1",
            "metadata": metadata,
        },
        observables={
            "butterfly": {
                "points": [{"qx": 0.25, "qy": 0.5, "intensity": 8.0}],
                "arcs": [{"side": "upper", "points": [[0.2, 0.3], [0.4, 0.5]]}],
                "profiles": [{"q": [0.2, 0.3], "intensity": [4.0, 6.0]}],
                "diagnostics": {"support_fraction": 0.75, "reason": "observed"},
                "candidate_solutions": [{"a": 2.75, "status": "candidate"}],
                "uncertainty": {"available": False, "reason": "resampling_not_run"},
                "sensitivity": {"q_window": [{"q_min": 0.1, "a": 2.8}]},
            }
        },
        ridges=[{"q": 0.42, "qx": 0.3, "qy": 0.29, "q_unit": "nm^-1"}],
        ellipse_fit={
            "parameters": {"a": {"value": 2.75, "candidate_value": 2.75}},
            "candidate_solutions": [{"semi_major": 2.75, "status": "candidate"}],
        },
        full2d={
            "diagnostics": {"cost": 1.25, "success": False, "reason": "boundary_candidate"},
            "uncertainty": {"available": False, "reason": "not_requested"},
            "sensitivity": {"starts": 4},
            "candidate_solutions": [{"semi_major": 2.75, "finite": True}],
            "model": image + 1,
        },
        metadata={"source": "frame.edf", "q_unit": "nm^-1"},
        analysis={"q_min": 0.1, "q_max": 1.4, "method": "butterfly_curvature"},
        analysis_domain=domain,
        analysis_arrays={"fit_residual": image - 1},
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _records() -> list[FrameFitResult]:
    return [
        FrameFitResult(
            FrameRef("frame.edf", frame_id="frame-0", time=1.5, metadata={"detector": "D1"}),
            result=_pipeline_result(),
            status="warning",
            diagnostic="finite candidate retained",
            warm_start_from="frame-previous",
        ),
        FrameFitResult(
            FrameRef("missing.edf", frame_id="frame-1", time=2.0),
            result=None,
            status="failed",
            error="read failed",
        ),
    ]


def test_regular_export_retains_frame_science_and_npz_array_links(tmp_path: Path) -> None:
    records = _records()
    outputs = export_batch(records, tmp_path / "regular")

    assert "frame_details" in outputs
    assert "frame_details" in set(outputs)
    details = _read_jsonl(outputs["frame_details"])
    assert len(details) == 2
    frame, failed = details
    assert frame["frame_id"] == "frame-0"
    assert frame["status"] == "warning"
    assert frame["frame_metadata"] == {"detector": "D1"}
    assert frame["array_prefix"] == "frame_0000"
    assert "frame_0000__image" in frame["array_names"]
    assert frame["q_unit"] == "nm^-1"
    assert frame["q_unit_source"] == "qmap"
    assert frame["analysis_settings"]["method"] == "butterfly_curvature"
    assert frame["analysis_domain"]["fit_pixel_count"] == 16

    scientific = frame["result"]
    butterfly = scientific["observables"]["butterfly"]
    assert butterfly["arcs"][0]["side"] == "upper"
    assert butterfly["profiles"][0]["intensity"] == [4.0, 6.0]
    assert butterfly["diagnostics"]["support_fraction"] == 0.75
    assert butterfly["candidate_solutions"][0]["a"] == 2.75
    assert butterfly["uncertainty"]["available"] is False
    assert butterfly["sensitivity"]["q_window"][0]["a"] == 2.8
    assert scientific["full2d"]["diagnostics"]["cost"] == 1.25
    assert scientific["full2d"]["candidate_solutions"][0]["semi_major"] == 2.75
    assert scientific["ellipse_fit"]["parameters"]["a"]["candidate_value"] == 2.75
    assert frame["qmap_metadata"]["orientation"]["name"] == "HORIZONTAL"
    assert frame["qmap_metadata"]["self"]["$ref"] == "$"
    assert failed["status"] == "failed"
    assert failed["error"] == "read failed"
    assert failed["result"] is None

    with np.load(outputs["npz"], allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["frame_0000__image"], records[0].result.image)
        np.testing.assert_array_equal(bundle["frame_0000__qmap__q"], records[0].result.qmap["q"])


def test_stream_details_match_regular_export_and_are_written_per_frame(tmp_path: Path) -> None:
    records = _records()
    regular = export_batch(records, tmp_path / "regular")
    stream = StreamingBatchExporter(tmp_path / "stream")
    stream.write(records[0])
    staged_details = stream._stage / stream._targets["frame_details"].name
    assert len(staged_details.read_text(encoding="utf-8").splitlines()) == 1
    stream.write(records[1])
    streamed = stream.finalize(BatchRunResult(records, total_count=2, processed_count=2))

    assert _read_jsonl(streamed["frame_details"]) == _read_jsonl(regular["frame_details"])


def test_resumed_stream_keeps_finite_candidates_and_original_array_links(tmp_path: Path) -> None:
    original = FrameFitResult(
        FrameRef("frame.edf", frame_id="frame-0"),
        result={
            "image": np.arange(9, dtype=float).reshape(3, 3),
            "parameters": {"a": {"value": 3.1, "candidate_value": 3.1, "status": "candidate"}},
            "full2d": {
                "diagnostics": {"cost": 2.0, "reason": "boundary_solution"},
                "candidate_solutions": [{"semi_major": 3.1, "finite": True}],
            },
            "analysis": {"q_min": 0.2},
        },
        status="warning",
    )
    output = tmp_path / "stream"
    first = BatchRunResult(
        [original], input_hash="input", config_hash="config", mode="warm_start",
        total_count=1, processed_count=1,
    )
    writer = StreamingBatchExporter(output)
    writer.write(original)
    writer.finalize(first)

    resumed_item = FrameFitResult(
        original.frame,
        result=_checkpoint_safe(original.result),
        status=original.status,
        resumed=True,
    )
    resumed_run = BatchRunResult(
        [resumed_item], input_hash="input", config_hash="config", mode="warm_start",
        total_count=1, processed_count=1,
    )
    resumed_writer = StreamingBatchExporter(output, force=True, resume=True)
    resumed_writer.write(resumed_item)
    outputs = resumed_writer.finalize(resumed_run)
    detail = _read_jsonl(outputs["frame_details"])[0]

    assert detail["resumed"] is True
    assert detail["result"]["parameters"]["a"]["candidate_value"] == 3.1
    assert detail["result"]["full2d"]["candidate_solutions"][0]["semi_major"] == 3.1
    assert "frame_0000__image" in detail["array_names"]
    with np.load(outputs["npz"], allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["frame_0000__image"], original.result["image"])


def test_stream_resume_rejects_corrupt_declared_npz_member(tmp_path: Path) -> None:
    record = FrameFitResult(
        FrameRef("frame.edf", frame_id="frame-0"),
        result={"image": np.arange(9, dtype=float).reshape(3, 3)},
    )
    output = tmp_path / "stream"
    writer = StreamingBatchExporter(output)
    writer.write(record)
    writer.finalize(BatchRunResult([record], total_count=1, processed_count=1))
    npz_path = output / "results.npz"

    with zipfile.ZipFile(npz_path, "r") as archive:
        member_name = "frame_0000__image.npy"
        start_dir = archive.start_dir
    payload = bytearray(npz_path.read_bytes())
    filename_offset = payload.find(member_name.encode("utf-8"), start_dir)
    central_header_offset = filename_offset - 46
    assert payload[central_header_offset : central_header_offset + 4] == b"PK\x01\x02"
    crc_offset = central_header_offset + 16
    crc = struct.unpack_from("<I", payload, crc_offset)[0]
    struct.pack_into("<I", payload, crc_offset, crc ^ 1)
    npz_path.write_bytes(payload)
    corrupted_bundle = npz_path.read_bytes()

    with pytest.raises(ValueError, match="frame_0000__image.npy.*integrity validation"):
        StreamingBatchExporter(output, force=True, resume=True)

    assert npz_path.read_bytes() == corrupted_bundle


def test_stream_resume_replaces_npz_arrays_for_frame_index_above_9999(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "stream"
    output.mkdir()
    npz_path = output / "results.npz"
    np.savez_compressed(
        npz_path,
        frame_10000__image=np.array([1.0]),
        __metadata__=np.asarray(json.dumps({
            "arrays": ["frame_10000__image"],
            "frame_count": 10001,
            "complete": False,
        })),
    )
    (output / "manifest.json").write_text(
        json.dumps({"input_hash": "input", "config_hash": "config", "mode": "warm_start"}),
        encoding="utf-8",
    )
    exporter_module = importlib.import_module("butterfly_saxs.export")
    monkeypatch.setattr(
        exporter_module,
        "_write_evolution",
        lambda path, _results: Path(path).write_bytes(b"plot"),
    )

    writer = StreamingBatchExporter(output, force=True, resume=True)
    placeholder = FrameFitResult(
        FrameRef("placeholder.edf", frame_id="placeholder"), status="skipped"
    )
    writer._compact_results = [placeholder] * 10000
    writer.write(
        FrameFitResult(
            FrameRef("frame_10000.edf", frame_id="frame-10000"),
            result={"image": np.array([2.0])},
        )
    )
    run = BatchRunResult(
        [], input_hash="input", config_hash="config", mode="warm_start",
        total_count=10001, processed_count=10001,
    )
    outputs = writer.finalize(run)

    with np.load(outputs["npz"], allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["frame_10000__image"], np.array([2.0]))
