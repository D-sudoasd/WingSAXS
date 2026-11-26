from __future__ import annotations

import json
from pathlib import Path

import threading

import numpy as np

from butterfly_saxs.batch import FrameFitResult, FrameRef, run_batch
from butterfly_saxs.export import StreamingBatchExporter, _walk_arrays, export_batch
from butterfly_saxs.geometry import GeometryMaps
from butterfly_saxs.report import build_analysis_report


def test_npz_completeness_lists_missing_frames_and_keeps_success_arrays(
    tmp_path: Path,
) -> None:
    successful_path = tmp_path / "successful-frame.tif"
    failed_path = tmp_path / "failed-frame.tif"
    resultless_path = tmp_path / "resultless-frame.tif"
    successful = FrameFitResult(
        frame=FrameRef(successful_path, frame_id="ok"),
        result={"image": np.arange(6, dtype=float).reshape(2, 3)},
    )
    failed = FrameFitResult(
        frame=FrameRef(failed_path, frame_id="failed"),
        result={"partial_image": np.ones((2, 2), dtype=float)},
        status="failed",
        error="fit failed",
    )
    resultless = FrameFitResult(
        frame=FrameRef(resultless_path, frame_id="missing"),
        result=None,
    )

    outputs = export_batch([successful, failed, resultless], tmp_path / "exports")

    with np.load(outputs["npz"], allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["__metadata__"]))
        np.testing.assert_array_equal(
            arrays["frame_0000__image"],
            np.arange(6, dtype=float).reshape(2, 3),
        )
    assert metadata == {
        "arrays": [
            "frame_0000__image",
            "frame_0001__partial_image",
        ],
        "frame_count": 3,
        "complete": False,
        "missing_frames": [1, 2],
        "missing_frame_ids": ["failed", "missing"],
        "missing_frame_paths": [str(failed_path), str(resultless_path)],
    }


def test_qmap_object_arrays_are_omitted_from_npz_and_stream_resume_report_succeeds(
    tmp_path: Path,
) -> None:
    y, x = np.indices((6, 8), dtype=float)
    qx = (x - 3.5) * 0.1
    qy = (y - 2.5) * 0.1
    q = np.hypot(qx, qy)
    chi = np.arctan2(qy, qx)
    valid_mask = np.ones((6, 8), dtype=bool)
    valid_mask[:, :2] = False
    geom = GeometryMaps(
        q_nm_inv=q,
        chi_rad=chi,
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=valid_mask,
    )
    qmap = {
        "object": geom,
        "qx": qx,
        "qy": qy,
        "q": q,
        "chi": chi,
        "valid_mask": valid_mask,
        "q_unit": "nm^-1",
    }

    direct_arrays: dict[str, object] = {}
    _walk_arrays(qmap, "qmap", direct_arrays)
    assert {"qmap__qx", "qmap__qy", "qmap__q", "qmap__chi", "qmap__valid_mask"} <= set(direct_arrays)
    assert not any(key.startswith("qmap__object") for key in direct_arrays)

    # Regular export omits redundant qmap["object"] arrays while keeping qmap coords & image/mask.
    direct_out = tmp_path / "direct"
    frame = FrameFitResult(
        frame=FrameRef(tmp_path / "f0.tif", frame_id="f0"),
        result={"image": x + y + 1.0, "qmap": qmap, "valid_mask": valid_mask},
    )
    direct_files = export_batch([frame], direct_out)
    with np.load(direct_files["npz"], allow_pickle=False) as archive:
        assert not any("qmap__object" in name for name in archive.files)
        np.testing.assert_array_equal(archive["frame_0000__qmap__qx"], qx)
        np.testing.assert_array_equal(archive["frame_0000__qmap__qy"], qy)
        np.testing.assert_array_equal(archive["frame_0000__qmap__q"], q)
        np.testing.assert_array_equal(archive["frame_0000__image"], x + y + 1.0)
        np.testing.assert_array_equal(archive["frame_0000__valid_mask"], valid_mask)

    # Streaming partial batch -> checkpoint resume -> build_analysis_report stays valid.
    paths = []
    for idx in range(2):
        path = tmp_path / f"frame_{idx}.dat"
        path.write_text(str(idx), encoding="utf-8")
        paths.append(path)
    stream_out = tmp_path / "stream"
    checkpoint = tmp_path / "checkpoint.json"
    stop = threading.Event()
    calls: list[str] = []

    def analyze(ref: FrameRef):
        calls.append(ref.path.name)
        idx = int(ref.path.read_text(encoding="utf-8"))
        if len(calls) == 1:
            stop.set()
        return {
            "image": (x + y + 1.0) * (idx + 1.0),
            "qmap": qmap,
            "valid_mask": valid_mask,
            "parameters": {"a": 0.3},
            "ellipse_fit": {"q_unit": "nm^-1", "parameters": {"a": 0.3}},
            "metrics": {"success": True},
        }

    writer1 = StreamingBatchExporter(stream_out, analyzer_identity="qmap-dedup-test")
    first = run_batch(
        paths,
        analyze,
        checkpoint=checkpoint,
        analyzer_identity="qmap-dedup-test",
        retain_results=False,
        result_sink=writer1.write,
        cancel_event=stop,
    )
    writer1.finalize(first)
    assert first.cancelled and first.processed_count == 1

    calls.clear()
    writer2 = StreamingBatchExporter(stream_out, force=True, resume=True, analyzer_identity="qmap-dedup-test")
    resumed = run_batch(
        paths,
        analyze,
        checkpoint=checkpoint,
        resume=True,
        analyzer_identity="qmap-dedup-test",
        retain_results=False,
        result_sink=writer2.write,
        restore_frame=writer2.can_restore,
    )
    assert resumed[0].resumed and not resumed[1].resumed
    assert calls == ["frame_1.dat"]
    outputs = writer2.finalize(resumed)

    with np.load(outputs["npz"], allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"]))
        assert metadata["complete"] is True
        assert not any("qmap__object" in name for name in archive.files)
        for prefix in ("frame_0000", "frame_0001"):
            for suffix in ("qmap__qx", "qmap__qy", "qmap__q", "image", "valid_mask"):
                assert f"{prefix}__{suffix}" in archive.files

    details = [json.loads(line) for line in outputs["frame_details"].read_text(encoding="utf-8").splitlines()]
    assert all(not any("qmap__object" in k for k in rec["array_names"]) for rec in details)

    report = build_analysis_report(stream_out, radial_bins=4, angular_bins=6, formats=("png",), dpi=50)
    assert report["counts"]["frames"] == 2
    assert report["counts"]["reported"] == 2
    assert report["counts"]["incomplete"] == 0
    catalog_text = (Path(report["outputs"]["summary"]).parent / "array_catalog.csv").read_text(encoding="utf-8")
    assert "qmap__object" not in catalog_text
    assert "frame_0000__image" in catalog_text
    assert "frame_0001__image" in catalog_text

