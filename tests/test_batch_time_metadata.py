from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.batch import BatchRunResult, FrameFitResult, build_frame_refs
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
        "lobe_radial_peaks": [{"q_star": 0.2, "angle_deg": 30.0, "valid": True}],
    }


@pytest.mark.parametrize(
    "common_unit",
    [
        {"time_unit": "ms"},
        {"metadata": {"time_unit": "ms"}},
    ],
    ids=["top-level-common-unit", "metadata-common-unit"],
)
def test_manifest_common_time_unit_is_embedded_in_native_export(
    tmp_path: Path, common_unit: dict,
) -> None:
    source_manifest_path = tmp_path / "input-manifest.json"
    source_manifest_path.write_text(
        json.dumps({
            **common_unit,
            "frames": [
                {
                    "path": "frame-a.edf",
                    "frame_id": "metadata-override",
                    "time": 0,
                    "order": 0,
                    "metadata": {"time_unit": "ns"},
                },
                {
                    "path": "frame-b.edf",
                    "frame_id": "row-override",
                    "time": 1,
                    "order": 1,
                    "time_unit": "us",
                },
                {"path": "frame-c.edf", "frame_id": "inherited", "time": 2, "order": 2},
                {"path": "frame-d.edf", "frame_id": "failed", "time": 3, "order": 3},
            ],
        }),
        encoding="utf-8",
    )

    refs = build_frame_refs([], manifest=source_manifest_path)
    assert [ref.metadata.get("time_unit") for ref in refs] == ["ns", "us", "ms", "ms"]

    results = [
        FrameFitResult(refs[0], _measured_result(1.0)),
        FrameFitResult(refs[1], _measured_result(2.0)),
        FrameFitResult(refs[2], _measured_result(3.0)),
        FrameFitResult(refs[3], None, status="failed", error="frame read failed"),
    ]
    export_batch(
        BatchRunResult(
            results,
            manifest=str(source_manifest_path),
            processed_count=4,
            total_count=4,
        ),
        tmp_path / "native-export",
    )

    # Report generation must be self-contained after input-manifest removal.
    source_manifest_path.unlink()
    report = build_analysis_report(
        tmp_path / "native-export",
        radial_bins=4,
        angular_bins=6,
        formats=("png",),
        dpi=36,
    )

    assert [frame["time_unit"] for frame in report["frames"]] == ["ns", "us", "ms", "ms"]
    assert report["frames"][3]["status"] == "failed"
