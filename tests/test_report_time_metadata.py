from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from butterfly_saxs.batch import BatchRunResult, FrameFitResult, FrameRef
from butterfly_saxs.export import export_batch
from butterfly_saxs.report import build_analysis_report


def _measured_result(scale: float) -> dict:
    y, x = np.indices((8, 8), dtype=float)
    return {
        "image": scale * (x + y + 1),
        "qmap": {"qx": (x - 3.5) * 0.1, "qy": (y - 3.5) * 0.1, "q_unit": "nm^-1"},
        "ellipse_fit": {
            "q_unit": "nm^-1",
            "parameters": {"a": 0.3, "b": 0.2, "theta_deg": 0.0},
        },
        "lobe_radial_peaks": [
            {"q_star": 0.2, "angle_deg": 30.0, "valid": True}
        ],
    }


def test_report_uses_user_manifest_time_units_by_order_and_keeps_failed_slot(
    tmp_path: Path,
) -> None:
    # The original manifest is intentionally in a different order from the
    # exported frames. Acquisition `order` aligns its per-frame units.
    source_manifest = {
        "metadata": {"time_unit": "s"},
        "frames": [
            {"path": "frame-c.edf", "frame_id": "third", "time": 2, "order": 2,
             "time_unit": "ms"},
            {"path": "frame-a.edf", "frame_id": "first", "time": 0, "order": 0,
             "time_unit": "us"},
            {"path": "frame-b.edf", "frame_id": "failed", "time": 1, "order": 1},
        ],
    }
    source_manifest_path = tmp_path / "input-manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    results = [
        FrameFitResult(
            FrameRef("frame-a.edf", frame_id="first", time=0, order=0,
                     metadata={"time_unit": "ns"}),
            _measured_result(1.0),
        ),
        FrameFitResult(
            FrameRef("frame-b.edf", frame_id="failed", time=1, order=1),
            None,
            status="failed",
            error="frame read failed",
        ),
        FrameFitResult(
            FrameRef("frame-c.edf", frame_id="third", time=2, order=2),
            _measured_result(2.0),
        ),
    ]
    run = BatchRunResult(
        results,
        manifest=str(source_manifest_path),
        processed_count=3,
        total_count=3,
    )
    export_batch(run, tmp_path)

    report = build_analysis_report(
        tmp_path, radial_bins=4, angular_bins=6, formats=("png",), dpi=36
    )

    assert [frame["frame_id"] for frame in report["frames"]] == [
        "first", "failed", "third"
    ]
    # Exported per-frame metadata takes precedence; otherwise each source
    # frame's unit is matched by order, then the source manifest common unit.
    assert [frame["time_unit"] for frame in report["frames"]] == [
        "ns", "s", "ms"
    ]
    assert report["frames"][1]["status"] == "failed"
    summary_path = Path(report["outputs"]["summary"]).parent / "frame_measurements.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        measurements = list(csv.DictReader(handle))
    assert [row["frame_index"] for row in measurements] == ["0", "1", "2"]
    assert [row["time_unit"] for row in measurements] == ["ns", "s", "ms"]


def test_report_resume_rebuilds_when_referenced_manifest_time_unit_changes(
    tmp_path: Path,
) -> None:
    source_manifest_path = tmp_path / "input-manifest.json"
    source_manifest_path.write_text(
        json.dumps({
            "time_unit": "ms",
            "frames": [
                {"path": "frame-a.edf", "frame_id": "first", "time": 0, "order": 0},
                {"path": "frame-b.edf", "frame_id": "second", "time": 1, "order": 1},
            ],
        }),
        encoding="utf-8",
    )
    results = [
        FrameFitResult(
            FrameRef(f"frame-{name}.edf", frame_id=frame_id, time=index, order=index),
            _measured_result(float(index + 1)),
        )
        for index, (name, frame_id) in enumerate((("a", "first"), ("b", "second")))
    ]
    export_batch(
        BatchRunResult(results, manifest=str(source_manifest_path), total_count=2),
        tmp_path,
    )
    options = {"radial_bins": 4, "angular_bins": 6, "formats": ("png",), "dpi": 36}

    first = build_analysis_report(tmp_path, **options)
    summary_path = Path(first["outputs"]["summary"])
    measurements_path = summary_path.parent / "frame_measurements.csv"
    original_mtime = measurements_path.stat().st_mtime_ns
    assert [frame["time_unit"] for frame in first["frames"]] == ["ms", "ms"]
    assert "__source_manifest__" in first["source_state"]
    assert "__source_manifest__" not in first["sources"]

    source_manifest_path.write_text(
        json.dumps({
            "time_unit": "s",
            "frames": [
                {"path": "frame-a.edf", "frame_id": "first", "time": 0, "order": 0},
                {"path": "frame-b.edf", "frame_id": "second", "time": 1, "order": 1},
            ],
        }),
        encoding="utf-8",
    )
    resumed = build_analysis_report(tmp_path, resume=True, **options)

    assert resumed["operation_status"] == "completed"
    assert [frame["time_unit"] for frame in resumed["frames"]] == ["s", "s"]
    assert measurements_path.stat().st_mtime_ns > original_mtime
    with measurements_path.open(encoding="utf-8", newline="") as handle:
        measurements = list(csv.DictReader(handle))
    assert [row["time_unit"] for row in measurements] == ["s", "s"]
