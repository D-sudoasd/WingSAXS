from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.sequence_diagnostics import (
    SCHEMA_VERSION,
    compute_frame_content_hashes,
    diagnose_sequence,
    write_sequence_diagnostics,
)


def test_compute_frame_content_hashes_arrays_and_files(tmp_path: Path) -> None:
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    b = a.copy()
    c = a.copy()
    c[3, 4] += 1.0

    hashes = compute_frame_content_hashes([a, b, c])
    assert len(hashes) == 3
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]

    # Separate files with identical pixel arrays produce identical hashes
    p1 = tmp_path / "frame_0001.npy"
    p2 = tmp_path / "frame_0002_copy.npy"
    p3 = tmp_path / "frame_0003_diff.npy"
    np.save(p1, a)
    np.save(p2, b)
    np.save(p3, c)

    file_hashes = compute_frame_content_hashes([p1, p2, p3])
    assert file_hashes == hashes

    # Multi-frame stack with explicit frame_indices
    stack_path = tmp_path / "stack.npy"
    np.save(stack_path, np.stack([a, b, c], axis=0))
    stack_hashes = compute_frame_content_hashes(stack_path, frame_indices=[0, 1, 2])
    assert stack_hashes == hashes


def test_diagnose_sequence_clean_run_has_no_warnings() -> None:
    records = [
        {
            "frame_id": f"f{i:03d}",
            "status": "ok",
            "timestamp_s": float(i * 60),
            "intensity_unit": "cm^-1",
            "q_unit": "nm^-1",
            "axis_ratio": 0.35,
            "q_star": 0.42 - 0.01 * i,
            "friedel_bisector_deg": 91.0 + 0.3 * i,
            "coverage": 0.92,
        }
        for i in range(6)
    ]
    hashes = [f"hash_{i}" for i in range(6)]
    diag = diagnose_sequence(
        records,
        frame_hashes=hashes,
        q_window=(0.11, 1.35),
        nominal_draw_axis_deg=90.0,
    )
    json.dumps(diag, allow_nan=False)

    assert diag["schema_version"] == SCHEMA_VERSION
    assert diag["n_frames"] == 6
    assert diag["n_failed"] == 0
    assert diag["warnings"] == []
    assert diag["remediation"] == []
    assert diag["duplicates"]["detected"] is False
    assert diag["duplicates"]["unique_ratio"] == pytest.approx(1.0)
    assert diag["timestamps"]["chronology_ok"] is True
    assert diag["calibration"]["mixed_intensity_units"] is False
    assert diag["axis_ratio_collapse"]["bound_collapse_rate"] == pytest.approx(0.0)
    assert diag["window_edge_pileup"]["low_q_edge_hits"] == 0
    assert diag["window_edge_pileup"]["high_q_edge_hits"] == 0
    assert diag["orientation_and_regimes"]["orthogonal_flip_detected"] is False
    assert diag["coverage"]["coverage_drop_count"] == 0


def test_diagnose_sequence_detects_all_seven_anomaly_classes_without_mutation() -> None:
    records = [
        {
            "frame_id": "f000",
            "status": "ok",
            "timestamp_s": 0.0,
            "intensity_unit": "cm^-1",
            "q_unit": "nm^-1",
            "axis_ratio_b_over_a": 0.001,
            "q_star": 0.35,
            "friedel_bisector_deg": 91.0,
            "coverage": 0.92,
        },
        {
            "frame_id": "f001",
            "status": "ok",
            "timestamp_s": 60.0,
            "intensity_unit": "cm^-1",
            "q_unit": "nm^-1",
            "axis_ratio_b_over_a": 0.001,
            "q_star": 0.2569,  # low-q edge pile-up inside [0.25, 1.35]
            "friedel_bisector_deg": 92.0,
            "coverage": 0.91,
        },
        {
            "frame_id": "f002",
            "status": "ok",
            "timestamp_s": 60.0,  # stalled timestamp (dt=0)
            "intensity_unit": "cm^-1",
            "q_unit": "nm^-1",
            "axis_ratio_b_over_a": 0.001,
            "q_star": 0.2569,
            "friedel_bisector_deg": 91.5,
            "coverage": 0.91,
        },
        {
            "frame_id": "f003",
            "status": "ok",
            "timestamp_s": 500.0,  # large gap
            "intensity_unit": "raw_counts",  # mixed intensity unit
            "q_unit": "pixel-q",  # uncalibrated q unit
            "axis_ratio_b_over_a": 0.30,
            "q_star": 1.34,  # high-q edge pile-up
            "friedel_bisector_deg": 3.0,  # orthogonal flip + abrupt jump
            "coverage": 0.25,  # coverage drop
        },
        {
            "frame_id": "f004",
            "status": "failed",
            "success": False,
            "timestamp_s": 450.0,  # non-monotonic backward timestamp
            "intensity_unit": "cm^-1",
            "q_unit": "nm^-1",
            "axis_ratio_b_over_a": float("nan"),
            "q_star": None,
            "friedel_bisector_deg": None,
            "coverage": 0.90,
        },
        {
            "frame_id": "f005",
            "status": "ok",
            "timestamp_s": 560.0,
            "intensity_unit": "cm^-1",
            "q_unit": "nm^-1",
            "axis_ratio_b_over_a": 0.0015,
            "q_star": 0.32,
            "friedel_bisector_deg": 5.0,  # second orthogonal flipped frame
            "coverage": 0.92,
        },
    ]
    snapshot = copy.deepcopy(records)
    # Frames 1 & 2 are frozen duplicates (same hash)
    hashes = ["h0", "h_dup", "h_dup", "h3", "h4", "h5"]

    diag = diagnose_sequence(
        records,
        frame_hashes=hashes,
        q_window=(0.25, 1.35),
        nominal_draw_axis_deg=90.0,
    )
    # Must never mutate input records or interpolate failed frames
    assert records[4]["q_star"] is None
    assert math.isnan(records[4]["axis_ratio_b_over_a"])
    assert records[0] == snapshot[0]

    # Strict JSON safety (zero NaN/Inf)
    encoded = json.dumps(diag, allow_nan=False)
    assert "NaN" not in encoded

    assert diag["n_frames"] == 6
    assert diag["n_failed"] == 1
    assert diag["failed_indices"] == [4]
    assert diag["per_frame"][4]["status"] == "failed"
    assert diag["per_frame"][4]["q_star"] is None
    assert diag["per_frame"][4]["axis_ratio"] is None

    # (a) Duplicates
    assert diag["duplicates"]["detected"] is True
    assert diag["duplicates"]["n_unique"] == 5
    assert diag["duplicates"]["duplicate_frame_count"] == 1
    assert diag["duplicates"]["unique_ratio"] == pytest.approx(5.0 / 6.0)
    assert diag["duplicates"]["duplicate_clusters"][0]["indices"] == [1, 2]
    assert diag["duplicates"]["frozen_runs"][0]["length"] == 2

    # (b) Timestamps
    assert diag["timestamps"]["chronology_ok"] is False
    assert 2 in diag["timestamps"]["stalled_indices"]
    assert 4 in diag["timestamps"]["non_monotonic_indices"]
    assert any(gap["index"] == 3 for gap in diag["timestamps"]["gaps"])

    # (c) Mixed calibration provenance
    assert diag["calibration"]["mixed_intensity_units"] is True
    assert diag["calibration"]["mixed_q_units"] is True
    assert diag["calibration"]["raw_count_frame_indices"] == [3]
    assert diag["calibration"]["uncalibrated_q_frame_indices"] == [3]

    # (d) Axis ratio collapse (<= 0.002): 4 out of 5 finite frames
    assert diag["axis_ratio_collapse"]["bound_collapse_count"] == 4
    assert diag["axis_ratio_collapse"]["evaluated_count"] == 5
    assert diag["axis_ratio_collapse"]["bound_collapse_rate"] == pytest.approx(0.8)

    # (e) Window edge pile-ups
    assert diag["window_edge_pileup"]["low_q_edge_indices"] == [1, 2]
    assert diag["window_edge_pileup"]["low_q_onset_index"] == 1
    assert diag["window_edge_pileup"]["high_q_edge_indices"] == [3]

    # (f) Orientation flips & transitions
    assert diag["orientation_and_regimes"]["orthogonal_flip_detected"] is True
    assert diag["orientation_and_regimes"]["orthogonal_flip_indices"] == [3, 5]
    assert any(j["index"] == 3 for j in diag["orientation_and_regimes"]["orientation_jumps"])

    # (g) Coverage drops
    assert diag["coverage"]["coverage_drop_indices"] == [3]

    expected_warnings = {
        "duplicate_frames_detected",
        "timestamp_anomaly",
        "mixed_intensity_units",
        "uncalibrated_pixel_q",
        "axis_ratio_at_bound",
        "low_q_edge_hit",
        "high_q_edge_hit",
        "orthogonal_axis_flip",
        "abrupt_orientation_transition",
        "coverage_drop",
    }
    assert expected_warnings.issubset(set(diag["warnings"]))
    remediation_flags = {item["flag"] for item in diag["remediation"]}
    assert expected_warnings.issubset(remediation_flags)


def test_write_sequence_diagnostics_creates_json_and_csv(tmp_path: Path) -> None:
    records = [
        {"frame_id": "=unsafe_formula", "timestamp_s": 0.0, "axis_ratio": 0.001, "q_star": 0.255},
        {"frame_id": "frame_001", "timestamp_s": 60.0, "axis_ratio": 0.30, "q_star": 0.45},
    ]
    diag = diagnose_sequence(records, frame_hashes=["h1", "h1"], q_window=(0.25, 1.35))
    paths = write_sequence_diagnostics(diag, tmp_path)

    assert set(paths.keys()) == {"json", "csv"}
    assert paths["json"].name == "sequence_diagnostics.json"
    assert paths["csv"].name == "sequence_diagnostics.csv"
    assert paths["json"].exists()
    assert paths["csv"].exists()

    parsed = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["n_frames"] == 2

    with paths["csv"].open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["frame_id"] == "'=unsafe_formula"
    assert rows[1]["is_duplicate"] == "True"

    with pytest.raises(FileExistsError):
        write_sequence_diagnostics(diag, tmp_path, force=False)
