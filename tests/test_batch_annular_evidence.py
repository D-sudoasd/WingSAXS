from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.batch import run_batch
from butterfly_saxs.export import StreamingBatchExporter, export_batch


def _annular_butterfly() -> dict[str, object]:
    return {
        "trace_method": "annular_peak",
        "method_version": "butterfly-annular-trajectory-v1.1",
        "q_unit": "nm^-1",
        "measurement_summary": {
            "method_version": "annular-direct-measurements-v1",
            "q_unit": "nm^-1", "ellipse_required": False,
            "rows": [{"status": "missing", "chi_deg": None}],
        },
        "annular_peaks": {
            "method_version": "butterfly-annular-trajectory-v1.1",
            "q_unit": "nm^-1",
            "q_window": [0.2, 0.4],
            "settings": {
                "effective_radial_bins": 2,
                "angle_bins": 8,
                "trajectory_selection_method": "adjacent_annulus_dynamic_path",
            },
            "q_edges": np.array([0.2, 0.3, 0.4], dtype=np.float32),
            "annuli": [
                {
                    "annulus_index": 4,
                    "profile_id": "annulus-4",
                    "q_center": 0.35,
                    "status": "tracked",
                    "reason": "partial_or_missing_lobe_support",
                    "candidates": [
                        {
                            "point_id": "annulus-4:peak-7",
                            "annulus_index": 4,
                            "quadrant": "QI",
                            "accepted": False,
                            "reason": "weaker_trajectory_candidate",
                            "trajectory_confidence": "low",
                            "trajectory_ambiguous": True,
                        }
                    ],
                    "selected_peaks": [],
                    "trajectory_selection": {
                        "selected_count": 0,
                        "low_confidence_count": 0,
                        "ambiguous_count": 0,
                    },
                }
            ],
        },
        "profiles": {
            "annulus-4": {
                "profile_axis": "azimuthal",
                "q_center": 0.35,
                "angle_deg": np.array([0.0, 45.0, 90.0, 135.0]),
                "raw_intensity": np.array([1.0, 2.5, 1.2, 0.8]),
                "coverage": np.array([1.0, 0.75, 1.0, 0.5]),
            }
        },
        "diagnostics": {
            "trajectory_selection": {
                "selection_method": "adjacent_annulus_dynamic_path",
                "n_rejected_candidates": 1,
                "n_ambiguous_trajectories": 1,
            }
        },
    }


@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "streaming"])
@pytest.mark.parametrize("placement", ["direct", "nested", "invalid_direct", "empty_direct"])
def test_batch_sidecar_keeps_annular_evidence_and_npz_arrays(
    tmp_path: Path, streaming: bool, placement: str,
) -> None:
    frame = tmp_path / "frame.npy"
    frame.write_bytes(b"synthetic detector frame")
    butterfly = _annular_butterfly()
    result = {"parameters": {"a": 1.0}}
    if placement == "direct":
        result["butterfly"] = butterfly
        array_prefix = "frame_0000__butterfly"
    else:
        result["observables"] = {"butterfly": butterfly}
        array_prefix = "frame_0000__observables__butterfly"
        if placement == "invalid_direct":
            result["butterfly"] = "invalid direct payload"
        elif placement == "empty_direct":
            result["butterfly"] = {}
    output = tmp_path / ("stream" if streaming else "ordinary")

    if streaming:
        writer = StreamingBatchExporter(output)
        run = run_batch(
            [frame],
            lambda _frame: result,
            result_sink=writer.write,
            retain_results=False,
        )
        outputs = writer.finalize(run)
    else:
        run = run_batch([frame], lambda _frame: result)
        outputs = export_batch(run, output)

    rows = [
        json.loads(line)
        for line in outputs["ellipse_fit_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    evidence = rows[0]["butterfly"]
    assert evidence["measurement_summary"] == butterfly["measurement_summary"]
    annular = evidence["annular_peaks"]
    candidate = annular["annuli"][0]["candidates"][0]

    assert annular["method_version"] == "butterfly-annular-trajectory-v1.1"
    assert annular["q_unit"] == "nm^-1"
    assert annular["q_window"] == [0.2, 0.4]
    assert annular["settings"]["trajectory_selection_method"] == "adjacent_annulus_dynamic_path"
    assert annular["annuli"][0]["annulus_index"] == 4
    assert annular["annuli"][0]["profile_id"] == "annulus-4"
    assert candidate["point_id"] == "annulus-4:peak-7"
    assert candidate["accepted"] is False
    assert candidate["reason"] == "weaker_trajectory_candidate"
    assert candidate["trajectory_confidence"] == "low"
    assert evidence["diagnostics"]["trajectory_selection"]["n_rejected_candidates"] == 1

    assert annular["q_edges"] == {
        "shape": [3],
        "dtype": "float32",
        "min": pytest.approx(0.2),
        "max": pytest.approx(0.4),
    }
    profile = evidence["profiles"]["annulus-4"]
    assert profile["raw_intensity"] == {
        "shape": [4],
        "dtype": "float64",
        "min": 0.8,
        "max": 2.5,
    }

    with np.load(outputs["npz"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays[f"{array_prefix}__annular_peaks__q_edges"],
            np.array([0.2, 0.3, 0.4], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            arrays[f"{array_prefix}__profiles__annulus-4__raw_intensity"],
            np.array([1.0, 2.5, 1.2, 0.8]),
        )
