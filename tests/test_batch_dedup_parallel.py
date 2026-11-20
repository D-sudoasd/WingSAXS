from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from butterfly_saxs.batch import (
    FrameFitResult,
    FrameRef,
    compute_frame_pixel_sha256,
    read_checkpoint,
    run_batch,
)
from butterfly_saxs.project import ProjectConfig


def _top_level_picklable_analyzer(
    frame: FrameRef, config: Any = None
) -> dict[str, Any]:
    """Top-level Windows spawn-safe analyzer for ProcessPoolExecutor tests."""

    if "fail" in frame.id:
        raise RuntimeError(f"intentional failure for {frame.id}")
    arr = np.load(frame.path, allow_pickle=False)
    bs_val = None
    if isinstance(config, dict):
        bs_val = config.get("beamstop_radius_px")
        if bs_val is None and isinstance(config.get("analysis"), dict):
            bs_val = config["analysis"].get("beamstop_radius_px")
    elif config is not None and hasattr(config, "analysis"):
        bs_val = config.analysis.get("beamstop_radius_px")
    return {
        "status": "ok",
        "success": True,
        "parameters": {
            "mean": float(np.mean(arr)),
            "sum": float(np.sum(arr)),
            "worker_pid": int(os.getpid()),
            "beamstop_radius_px": bs_val,
        },
        "metadata": {"frame_id": frame.id, "path": str(frame.path)},
    }


def test_run_batch_deduplicate_identical_frames_preserves_order_and_provenance(
    tmp_path: Path,
) -> None:
    pattern_a = np.arange(36, dtype=np.float32).reshape(6, 6)
    pattern_b = pattern_a.copy()
    pattern_b[2, 3] += 1.0  # 1-pixel difference must NEVER be deduplicated

    files = [
        tmp_path / "frame_00.npy",
        tmp_path / "frame_01.npy",
        tmp_path / "frame_02.npy",
        tmp_path / "frame_03.npy",
        tmp_path / "frame_04.npy",
    ]
    np.save(files[0], pattern_a)
    np.save(files[1], pattern_a)  # duplicate of frame_00
    np.save(files[2], pattern_b)  # unique
    np.save(files[3], pattern_a)  # duplicate of frame_00
    np.save(files[4], pattern_b)  # duplicate of frame_02

    calls: list[str] = []

    def counting_analyzer(frame: FrameRef) -> dict[str, Any]:
        calls.append(frame.id)
        arr = np.load(frame.path, allow_pickle=False)
        return {
            "status": "ok",
            "success": True,
            "parameters": {"mean": float(np.mean(arr))},
            "metadata": {"frame_id": frame.id, "path": str(frame.path)},
        }

    # Opt-in default False runs every frame and omits dedup keys in to_record().
    default_run = run_batch(files, counting_analyzer, deduplicate_identical_frames=False)
    assert len(calls) == 5
    assert "is_duplicate_frame" not in default_run[0].to_record()

    calls.clear()
    dedup_run = run_batch(files, counting_analyzer, deduplicate_identical_frames=True)
    # Only the 2 unique pixel arrays are analyzed; all 5 rows are retained in exact order.
    assert calls == ["frame_00", "frame_02"]
    assert len(dedup_run.frame_results) == 5
    assert [item.frame.id for item in dedup_run.frame_results] == [
        "frame_00",
        "frame_01",
        "frame_02",
        "frame_03",
        "frame_04",
    ]

    r0, r1, r2, r3, r4 = dedup_run.frame_results
    assert r0.is_duplicate_frame is False
    assert r0.duplicate_of_frame_id is None
    assert isinstance(r0.pixel_sha256, str) and len(r0.pixel_sha256) == 64

    assert r1.is_duplicate_frame is True
    assert r1.duplicate_of_frame_id == "frame_00"
    assert r1.pixel_sha256 == r0.pixel_sha256
    assert r1.result["metadata"]["frame_id"] == "frame_01"
    assert r1.result["metadata"]["is_duplicate_frame"] is True
    assert r1.result["metadata"]["duplicate_of_frame_id"] == "frame_00"
    assert r1.parameters["mean"] == pytest.approx(r0.parameters["mean"])

    assert r2.is_duplicate_frame is False
    assert r2.duplicate_of_frame_id is None
    assert r2.pixel_sha256 != r0.pixel_sha256
    assert r2.parameters["mean"] != pytest.approx(r0.parameters["mean"])

    assert r3.is_duplicate_frame is True
    assert r3.duplicate_of_frame_id == "frame_00"

    assert r4.is_duplicate_frame is True
    assert r4.duplicate_of_frame_id == "frame_02"
    assert r4.pixel_sha256 == r2.pixel_sha256

    # Round-trip serialization and strict JSON compatibility.
    rec1 = r1.to_record()
    assert rec1["is_duplicate_frame"] is True
    assert rec1["duplicate_of_frame_id"] == "frame_00"
    assert rec1["pixel_sha256"] == r0.pixel_sha256
    restored = FrameFitResult.from_record(rec1)
    assert restored.is_duplicate_frame is True
    assert restored.duplicate_of_frame_id == "frame_00"
    assert restored.pixel_sha256 == r0.pixel_sha256
    json.dumps(dedup_run.to_records(), allow_nan=False)


def test_deduplicate_h5_matches_pixel_content_even_when_file_metadata_differs(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    pixels = np.linspace(1.0, 25.0, 25, dtype=np.float32).reshape(5, 5)
    h5_a = tmp_path / "scan_001.h5"
    h5_b = tmp_path / "scan_002.h5"
    with h5py.File(h5_a, "w") as handle:
        handle.attrs["timestamp"] = "2026-11-20T00:00:01Z"
        handle.create_dataset("entry/data/I_abs_2d", data=pixels)
    with h5py.File(h5_b, "w") as handle:
        handle.attrs["timestamp"] = "2026-11-20T00:05:99Z-different-header"
        handle.create_dataset("entry/data/I_abs_2d", data=pixels)

    ref_a = FrameRef(h5_a, frame_id="h5_a", dataset="entry/data/I_abs_2d")
    ref_b = FrameRef(h5_b, frame_id="h5_b", dataset="entry/data/I_abs_2d")
    assert compute_frame_pixel_sha256(ref_a) == compute_frame_pixel_sha256(ref_b)

    called: list[str] = []

    def analyzer(frame: FrameRef) -> dict[str, Any]:
        called.append(frame.id)
        return {"status": "ok", "parameters": {"a": 1.0}, "metadata": {"frame_id": frame.id}}

    run = run_batch([ref_a, ref_b], analyzer, deduplicate_identical_frames=True)
    assert called == ["h5_a"]
    assert run[0].is_duplicate_frame is False
    assert run[1].is_duplicate_frame is True
    assert run[1].duplicate_of_frame_id == "h5_a"


def test_run_batch_parallel_process_pool_windows_spawn_dedup_isolation_and_resume(
    tmp_path: Path,
) -> None:
    arr0 = np.full((8, 8), 10.0, dtype=np.float32)
    arr1 = np.full((8, 8), 20.0, dtype=np.float32)
    arr3 = np.full((8, 8), 30.0, dtype=np.float32)

    refs = []
    for name, arr in (
        ("f00_ok", arr0),
        ("f01_ok", arr1),
        ("f02_fail", arr3),
        ("f03_dup0", arr0),
        ("f04_ok", arr3),
    ):
        path = tmp_path / f"{name}.npy"
        np.save(path, arr)
        refs.append(FrameRef(path, frame_id=name))

    checkpoint = tmp_path / "parallel_checkpoint.json"
    emitted_order: list[str] = []
    run = run_batch(
        refs,
        _top_level_picklable_analyzer,
        mode="independent",
        max_workers=2,
        deduplicate_identical_frames=True,
        beamstop_radius_px=32.5,
        checkpoint=checkpoint,
        result_sink=lambda item: emitted_order.append(item.frame.id),
    )

    assert [item.frame.id for item in run.frame_results] == [
        "f00_ok",
        "f01_ok",
        "f02_fail",
        "f03_dup0",
        "f04_ok",
    ]
    assert emitted_order == ["f00_ok", "f01_ok", "f02_fail", "f03_dup0", "f04_ok"]
    assert [item.status for item in run.frame_results] == ["ok", "ok", "failed", "ok", "ok"]
    assert "RuntimeError: intentional failure for f02_fail" in (run[2].error or "")
    assert run[3].is_duplicate_frame is True
    assert run[3].duplicate_of_frame_id == "f00_ok"
    assert run[0].parameters["beamstop_radius_px"] == pytest.approx(32.5)
    # ProcessPoolExecutor actually ran in worker child processes (PID differs from parent).
    assert run[0].parameters["worker_pid"] != os.getpid()

    saved = read_checkpoint(checkpoint)
    assert saved["processed_count"] == 5
    assert len(saved["frames"]) == 5
    json.dumps(saved, allow_nan=False)

    # Resuming with max_workers=1 reuses completed frames without changing config_hash.
    resumed_calls: list[str] = []

    def resume_analyzer(frame: FrameRef, config: Any = None) -> dict[str, Any]:
        del config
        resumed_calls.append(frame.id)
        return {"status": "ok", "parameters": {"retried": True}}

    resumed = run_batch(
        refs,
        resume_analyzer,
        mode="independent",
        max_workers=1,
        deduplicate_identical_frames=True,
        beamstop_radius_px=32.5,
        checkpoint=checkpoint,
        resume=True,
    )
    # Only the previously failed frame is re-executed.
    assert resumed_calls == ["f02_fail"]
    assert [item.resumed for item in resumed.frame_results] == [True, True, False, True, True]
    assert [item.status for item in resumed.frame_results] == ["ok", "ok", "ok", "ok", "ok"]


def test_run_batch_warm_start_falls_back_sequential_and_closure_falls_back_threads(
    tmp_path: Path,
) -> None:
    files = []
    for idx in range(4):
        path = tmp_path / f"frame_{idx:02d}.npy"
        np.save(path, np.full((6, 6), float(idx + 1), dtype=np.float32))
        files.append(path)

    # warm_start with max_workers > 1 must fall back to sequential and propagate seeds.
    seen_seeds: list[Any] = []

    def warm_analyzer(frame: FrameRef, initial: Any = None) -> dict[str, Any]:
        seen_seeds.append(initial)
        return {"status": "ok", "parameters": {"step": frame.id}}

    warm_run = run_batch(files, warm_analyzer, mode="warm_start", max_workers=4)
    assert seen_seeds == [
        None,
        {"step": "frame_00"},
        {"step": "frame_01"},
        {"step": "frame_02"},
    ]
    assert [item.frame.id for item in warm_run.frame_results] == [
        "frame_00",
        "frame_01",
        "frame_02",
        "frame_03",
    ]

    # Non-picklable local closure with mode="independent", max_workers=2 uses ThreadPoolExecutor.
    def closure_analyzer(frame: FrameRef) -> dict[str, Any]:
        arr = np.load(frame.path, allow_pickle=False)
        return {"status": "ok", "parameters": {"val": float(arr[0, 0])}}

    thread_run = run_batch(files, closure_analyzer, mode="independent", max_workers=2)
    assert [item.parameters["val"] for item in thread_run.frame_results] == [1.0, 2.0, 3.0, 4.0]


def test_run_batch_forwards_beamstop_radius_px_without_mutating_config(
    tmp_path: Path,
) -> None:
    path = tmp_path / "single.npy"
    np.save(path, np.ones((6, 6), dtype=np.float32))
    cfg = ProjectConfig(analysis={"q_min": 0.1})
    seen: list[Any] = []

    def capture_config(frame: FrameRef, config: Any = None) -> dict[str, Any]:
        del frame
        seen.append(config.analysis.get("beamstop_radius_px"))
        return {"status": "ok", "parameters": {"a": 1.0}}

    run_a = run_batch([path], capture_config, config=cfg, beamstop_radius_px=45.0)
    run_b = run_batch([path], capture_config, config=cfg, beamstop_radius_px=None)

    assert seen == [45.0, None]
    assert "beamstop_radius_px" not in cfg.analysis
    assert run_a.config_hash != run_b.config_hash
