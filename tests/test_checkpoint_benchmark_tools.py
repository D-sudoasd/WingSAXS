"""Focused tests for the checkpoint fixture and replay helpers."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark_checkpoint_writes import (
    _default_indices,
    _parse_indices,
    _strict_equal,
    _worker,
    prepare_fixture,
)


def _checkpoint(path: Path, records: list[dict]) -> dict:
    metadata = {
        "version": 1,
        "updated_at": "2026-10-14T00:00:00+00:00",
        "mode": "independent",
        "input_hash": "input-digest",
        "config_hash": "config-digest",
        "producer_identity": {"source": "frozen", "analyzer": {"implementation": "fixture"}},
        "config": {"q_window": [0.1, 1.0]},
        "config_file_fingerprints": {},
        "selection": {},
        "cancelled": False,
        "processed_count": len(records),
        "total_count": len(records),
        "elapsed_s": 12.5,
        "frames": records,
        "frame_count": len(records),
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def test_prepare_streams_records_to_bound_fixture_and_binds_hashes(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.json"
    fixture = tmp_path / "private" / "checkpoint.jsonl"
    records = [
        {"frame_id": "S01", "result": {"value": -0.0, "flags": [True, None]}},
        {"frame_id": "S02", "result": {"value": 1.2345678901234567, "flags": [False]}},
    ]
    expected = _checkpoint(source, records)

    metadata_path = prepare_fixture(source, fixture, expected_frames=2)

    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert rows == expected["frames"]
    assert rows[0]["result"]["value"].hex() == (-0.0).hex()
    assert metadata["record_count"] == 2
    assert metadata["checkpoint_metadata"] == {key: value for key, value in expected.items() if key != "frames"}
    assert metadata["source_stat_before"] == metadata["source_stat_after"]
    assert metadata["source_checkpoint_size_bytes"] == source.stat().st_size
    with pytest.raises(FileExistsError):
        prepare_fixture(source, fixture, expected_frames=2)


def test_prepare_rejects_incomplete_checkpoint_without_publishing_fixture(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.json"
    fixture = tmp_path / "checkpoint.jsonl"
    _checkpoint(source, [{"frame_id": "only"}])

    with pytest.raises(ValueError, match="expected 2 frame records"):
        prepare_fixture(source, fixture, expected_frames=2)

    assert not fixture.exists()
    assert not fixture.with_name(fixture.name + ".metadata.json").exists()


def test_default_selection_spans_sequence_and_explicit_indices_keep_order() -> None:
    selected = _default_indices(121)
    assert len(selected) == 24
    assert selected[0] == 0 and selected[-1] == 120
    assert selected == sorted(set(selected))
    assert _parse_indices("1,6,121", 121) == [0, 5, 120]
    with pytest.raises(ValueError, match="ascending order"):
        _parse_indices("6,1", 121)


def test_strict_equivalence_distinguishes_signed_zero_and_bool_from_integer() -> None:
    _strict_equal({"value": -0.0}, {"value": -0.0})
    with pytest.raises(AssertionError, match="float.hex"):
        _strict_equal({"value": -0.0}, {"value": 0.0})
    with pytest.raises(AssertionError, match="type differs"):
        _strict_equal({"value": True}, {"value": 1})


def test_worker_commits_each_prefix_and_reads_back_final_snapshot(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "source.json"
    fixture = tmp_path / "fixture.jsonl"
    records = [{"frame_id": f"S{index:02d}", "result": {"value": float(index)}} for index in (1, 2)]
    _checkpoint(source_checkpoint, records)
    prepare_fixture(source_checkpoint, fixture, expected_frames=2)
    output = tmp_path / "case"
    output.mkdir()
    metrics_path = output / "worker-metrics.json"
    measurements_path = output / "worker-measurements.jsonl"

    result = _worker(SimpleNamespace(
        source=Path(__file__).resolve().parents[1],
        revision="test-source",
        fixture=fixture,
        mode="prefix",
        indices=None,
        frames=None,
        output_dir=output,
        worker_metrics=metrics_path,
        worker_measurements=measurements_path,
    ))

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    measurements = [json.loads(line) for line in measurements_path.read_text(encoding="utf-8").splitlines()]
    assert result == 0
    assert metrics["write_count"] == metrics["expected_write_count"] == 3
    assert metrics["strict_readback_equivalence"] is True
    assert len(measurements) == 3
    assert [row["selected_frames_committed"] for row in measurements] == [1, 2, 2]
    assert all(row["wall_seconds"] >= 0 and row["file_size_bytes"] > 0 for row in measurements)


def test_batch_replay_uses_each_stored_result_and_preserves_identity_flags(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "source.json"
    fixture = tmp_path / "fixture.jsonl"
    records = []
    for index in (1, 2):
        # Natural filename order is deliberately opposite to the checkpoint's
        # manifest order; the replay must retain the saved frame sequence.
        input_path = tmp_path / ("frame10.tif" if index == 1 else "frame2.tif")
        input_path.write_bytes(f"frame-{index}".encode("ascii"))
        records.append({
            "frame": {
                "path": str(input_path.resolve()), "frame_id": f"F{index}", "time": float(index),
                "metadata": {"sample": "S01"}, "order": index, "source": "entry",
                "dataset": "/entry/data/I_abs_2d", "frame": index - 1,
            },
            "status": "warning", "error": None, "diagnostic": "quality_status=FAIL",
            "traceback": None, "warm_start_from": None, "elapsed_s": 0.5, "resumed": True,
            "result": {"quality_status": "FAIL", "parameters": {"a": float(index)}},
        })
    _checkpoint(source_checkpoint, records)
    prepare_fixture(source_checkpoint, fixture, expected_frames=2)
    output = tmp_path / "batch-case"
    output.mkdir()
    metrics_path = output / "worker-metrics.json"
    measurements_path = output / "worker-measurements.jsonl"

    result = _worker(SimpleNamespace(
        source=Path(__file__).resolve().parents[1], revision="test-source", fixture=fixture,
        mode="batch-replay", indices=None, frames=None, output_dir=output,
        worker_metrics=metrics_path, worker_measurements=measurements_path,
    ))

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    measurements = [json.loads(line) for line in measurements_path.read_text(encoding="utf-8").splitlines()]
    assert result == 0
    assert metrics["workflow_scope"].startswith("batch.run_batch")
    assert metrics["scientific_fit_performed"] is False
    assert metrics["report_or_package_performed"] is False
    assert metrics["normal_run_batch_input_and_config_fingerprinting"] is True
    assert metrics["write_count"] == metrics["checkpoint_payload_count"] == 3
    assert metrics["readback_verification"]["frame_identity_order_equal"] is True
    assert metrics["readback_verification"]["stable_status_and_diagnostic_flags_equal"] is True
    assert metrics["readback_verification"]["source_resumed_count"] == 2
    assert metrics["readback_verification"]["target_resumed_count"] == 0
    assert [row["write_phase"] for row in measurements] == ["frame_prefix", "frame_prefix", "final_snapshot"]
    assert all("payload_construction" in row for row in measurements)


def test_instrumentation_times_only_outer_recursive_conversion():
    from butterfly_saxs import batch
    from scripts.benchmark_checkpoint_writes import _instrument_batch_writer

    # Timing changes alone must not add one clock/counter event per scalar.
    raw = {"profile": [0.0, -0.0, 1.0] * 1000}
    original = batch._checkpoint_safe
    import os
    original_fdopen = os.fdopen
    expected = original(raw)
    restore, stages = _instrument_batch_writer(batch)
    try:
        _strict_equal(batch._checkpoint_safe(raw), expected)
        assert stages()["checkpoint_safe_calls"] == 1
    finally:
        restore()
    assert batch._checkpoint_safe is original
    assert os.fdopen is original_fdopen


@pytest.mark.parametrize("timing", ["outer", "stages"])
def test_partial_resume_replay_preserves_later_records_and_commit_count(tmp_path, timing):
    from scripts.benchmark_checkpoint_writes import sample_fixture

    fixture = sample_fixture(tmp_path / "sample", frames=5, profiles=1, points=8)
    output = tmp_path / "resume"
    output.mkdir()
    metrics_path = output / "metrics.json"
    measurements_path = output / "measurements.jsonl"
    result = _worker(SimpleNamespace(
        source=Path(__file__).resolve().parents[1], revision="test-source", fixture=fixture,
        mode="batch-replay", indices=None, frames=None, output_dir=output,
        timing=timing, resume_every=2,
        worker_metrics=metrics_path, worker_measurements=measurements_path,
    ))
    assert result == 0
    metrics = json.loads(metrics_path.read_text())
    assert metrics["write_count"] == 4  # Three missing frames plus final snapshot.
    assert metrics["readback_verification"]["target_resumed_count"] == 2
    assert metrics["strict_readback_equivalence"] is True
    assert metrics["retained_record_reconstruction_wall_seconds_sum"] >= 0
    rows = [json.loads(line) for line in measurements_path.read_text().splitlines()]
    if timing == "stages":
        stages = metrics["instrumented_stage_totals"]
        assert stages["fsync_calls"] == stages["flush_calls"] == stages["file_close_calls"] == 4
        assert stages["file_write_calls"] == 8
        assert sum(row["retained_record_reconstruction"]["stages"].get(
            "checkpoint_safe_calls", 0) for row in rows) > 0
    else:
        assert metrics["instrumented_stage_totals"] == {}
