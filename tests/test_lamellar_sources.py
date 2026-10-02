from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.batch import BatchRunResult, FrameFitResult, FrameRef
from butterfly_saxs.export import StreamingBatchExporter, export_batch
from butterfly_saxs.lamellar import build_lamellar_scene
from butterfly_saxs.lamellar_sources import load_lamellar_sources
from butterfly_saxs.ui.lamellar_state import source_images


def _frames(root: Path) -> list[FrameFitResult]:
    image = np.arange(16, dtype=float).reshape(4, 4)
    qx, qy = np.meshgrid(np.linspace(-0.3, 0.3, 4), np.linspace(-0.3, 0.3, 4))
    result = {
        "status": "result-status-must-not-replace-frame-status",
        "image": image,
        "qmap": {"qx": qx, "qy": qy, "q_unit": "nm^-1"},
        "ellipse_fit": {
            "status": "fit-candidate",
            "parameters": {"a": 0.2, "b": 0.1, "axis_ratio": 0.5, "theta_deg": 20.0},
            "q_unit": "nm^-1",
        },
        "lobe_radial_peaks": [
            {"q_star": 0.2, "angle_deg": 35.0, "q_unit": "nm^-1", "valid": True, "branch_id": 0},
            {"q_star": 0.2, "angle_deg": 215.0, "q_unit": "nm^-1", "valid": True, "branch_id": 0},
        ],
        "analysis": {"ridge_method": "butterfly_curvature", "draw_axis_deg": 37.0},
        "metadata": {"detector": "fixture"},
    }
    first = FrameFitResult(
        frame=FrameRef(
            root / "frame-0.edf",
            frame_id="first",
            time=1.5,
            frame=7,
            dataset="entry/data",
            metadata={"time_unit": "s", "sample": "steel-A"},
        ),
        result=result,
        status="warning",
        diagnostic="finite ellipse candidate retained",
    )
    failed = FrameFitResult(
        frame=FrameRef(
            root / "frame-1.edf",
            frame_id="second",
            time=3.0,
            frame=8,
            dataset="entry/data",
            metadata={"time_unit": "s"},
        ),
        result={"analysis": {"draw_axis_deg": 37.0}},
        status="failed",
        error="synthetic read failure",
    )
    return [first, failed]


def _export(root: Path, *, streaming: bool, prefix: str) -> Path:
    frames = _frames(root)
    if streaming:
        exporter = StreamingBatchExporter(root, prefix=prefix)
        for frame in frames:
            exporter.write(frame)
        exporter.finalize(frames)
    else:
        export_batch(frames, root, prefix=prefix)
    return root


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("prefix", ["", "repeat_"])
def test_regular_and_streamed_native_batches_round_trip_from_directory(
    tmp_path: Path, streaming: bool, prefix: str
) -> None:
    folder = tmp_path / f"batch-{streaming}-{prefix or 'plain'}"
    folder.mkdir()
    _export(folder, streaming=streaming, prefix=prefix)

    sources = load_lamellar_sources(folder)

    assert [source["frame_index"] for source in sources] == [0, 1]
    first, failed = sources
    assert first["frame_id"] == "first"
    assert first["path"] == str(folder / "frame-0.edf")
    assert first["frame"] == first["frame_selector"] == 7
    assert first["dataset"] == first["dataset_selector"] == "entry/data"
    assert first["time"] == 1.5
    assert first["frame_metadata"]["time_unit"] == "s"
    assert first["frame_metadata"]["sample"] == "steel-A"
    assert first["status"] == "warning"
    assert first["diagnostic"] == "finite ellipse candidate retained"
    assert first["ellipse_fit"]["parameters"]["theta_deg"] == 20.0
    assert first["analysis_settings"]["draw_axis_deg"] == 37.0
    assert first["analysis"]["draw_axis_deg"] == 37.0
    assert first["draw_axis_deg"] == 37.0
    assert first["q_unit"] == "nm^-1"
    scene = build_lamellar_scene(
        first,
        {"period_source": "radial", "selected_branch": 0, "layer_count": 1, "stack_count": 1},
    )
    normal = scene.orientations[0][:, 2]
    assert np.degrees(np.arctan2(normal[1], normal[0])) == pytest.approx(88.0)

    # Batch sources link detector arrays by NPZ key and load them only when a
    # selected frame requests its image.
    assert not any(name in first for name in ("image", "observed", "qx", "qy"))
    assert first["array_keys"]["observed"] == "frame_0000__image"
    observed, qx, qy = source_images(first)
    np.testing.assert_array_equal(observed, np.arange(16).reshape(4, 4))
    np.testing.assert_array_equal(qx, np.meshgrid(np.linspace(-0.3, 0.3, 4), np.linspace(-0.3, 0.3, 4))[0])
    assert qy.shape == observed.shape

    assert failed["status"] == "failed"
    assert failed["error"] == "synthetic read failure"
    assert "observed" not in failed.get("array_keys", {})
    assert failed["array_error"] == "missing observed array in native NPZ"


@pytest.mark.parametrize("streaming", [False, True])
def test_prefixed_batch_can_be_opened_by_manifest_or_results_npz(
    tmp_path: Path, streaming: bool
) -> None:
    folder = tmp_path / "named"
    folder.mkdir()
    _export(folder, streaming=streaming, prefix="sample_")

    from_manifest = load_lamellar_sources(folder / "sample_manifest.json")
    from_npz = load_lamellar_sources(folder / "sample_results.npz")

    assert len(from_manifest) == len(from_npz) == 2
    assert from_manifest[0]["frame_id"] == from_npz[0]["frame_id"] == "first"
    assert from_manifest[0]["array_path"] == str(folder / "sample_results.npz")


def test_legacy_batch_without_frame_details_uses_ellipse_sidecar(tmp_path: Path) -> None:
    folder = tmp_path / "legacy"
    folder.mkdir()
    _export(folder, streaming=False, prefix="old_")
    (folder / "old_frame_details.jsonl").unlink()

    sources = load_lamellar_sources(folder)

    assert sources[0]["ellipse_fit"]["parameters"]["axis_ratio"] == 0.5
    assert sources[0]["lobe_radial_peaks"][0]["q_star"] == 0.2
    assert sources[0]["status"] == "warning"
    assert sources[1]["status"] == "failed"
    assert sources[1]["error"] == "synthetic read failure"


def test_cancelled_native_batch_retains_unprocessed_frame_slots(tmp_path: Path) -> None:
    folder = tmp_path / "cancelled"
    folder.mkdir()
    first = _frames(tmp_path)[0]
    run = BatchRunResult(
        [first],
        cancelled=True,
        processed_count=1,
        total_count=3,
    )
    exporter = StreamingBatchExporter(folder, prefix="sample_")
    exporter.write(first)
    exporter.finalize(run)

    sources = load_lamellar_sources(folder)

    assert [source["frame_index"] for source in sources] == [0, 1, 2]
    assert sources[0]["status"] == "warning"
    assert [sources[index]["status"] for index in (1, 2)] == ["not_run", "not_run"]
    assert all("path" not in sources[index] for index in (1, 2))


def test_multiple_valid_native_batches_require_explicit_manifest(tmp_path: Path) -> None:
    _export(tmp_path, streaming=True, prefix="first_")
    _export(tmp_path, streaming=False, prefix="second_")

    with pytest.raises(ValueError, match="Multiple native batch bundles.*manifest JSON explicitly"):
        load_lamellar_sources(tmp_path)

    assert len(load_lamellar_sources(tmp_path / "second_manifest.json")) == 2
