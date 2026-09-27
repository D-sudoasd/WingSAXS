from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compare_butterfly_trajectory_methods import (  # noqa: E402
    METHODS,
    _cartesian_slice_points,
    analyze_variant,
    score_extraction,
)
from butterfly_saxs.benchmark_arcs import generate_arc_case  # noqa: E402


def test_scoring_uses_visible_truth_and_retains_null_detections() -> None:
    axis = np.linspace(-1.0, 1.0, 41)
    qx, qy = np.meshgrid(axis, axis)
    invalid = np.zeros(qx.shape, dtype=bool)
    invalid[:, 4:8] = True
    truth = {
        "actual_observable_arcs": [{
            "arc_id": "plus_upper",
            "branch": "plus",
            "branch_id": 0,
            "side": "upper",
            "polyline_q": [[-0.8, 0.0], [0.8, 0.0]],
        }]
    }
    points = [{
        "point_id": "p0", "qx": 0.25, "qy": 0.0, "branch_id": 0,
        "side": "upper", "accepted": True,
    }]
    result, rows = score_extraction(
        points, truth, qx, qy, invalid, (0.0, 0.9), 0.05, 1.5
    )
    assert result["visible_arc_count"] == 1
    assert rows[0]["truth_role_match"] is True
    assert rows[0]["point_to_truth_sample_cloud_distance_pixels"] < 0.5
    assert result["accepted_point_count"] == 1

    null_result, null_rows = score_extraction(
        points, {"actual_observable_arcs": []}, qx, qy,
        np.zeros(qx.shape, dtype=bool), (0.0, 0.9), 0.05, 1.5,
    )
    assert null_result["accepted_point_count"] == 1
    assert null_result["visible_arc_count"] == 0
    assert null_result["median_point_to_visible_truth_sample_cloud_distance_pixels"] is None
    assert null_rows[0]["point_to_truth_sample_cloud_distance_pixels"] is None


def test_four_methods_receive_one_identical_missing_branch_image() -> None:
    case = generate_arc_case("missing_branch", seed=506, shape=(72, 72))
    invalid = np.asarray(case["mask"], dtype=bool).copy()
    records, point_rows = analyze_variant(
        case_id="missing_branch",
        truth=case["truth"],
        image=case["image"],
        qmap=case["qmap"],
        invalid=invalid,
        q_window=(0.15, 0.95),
    )
    assert [row["method"] for row in records] == list(METHODS)
    assert len({row["same_input_sha256"] for row in records}) == 1
    assert all(row["execution_status"] == "complete" for row in records)
    assert all(row["generator_arc_count"] == 2 for row in records)
    assert set(point_rows) == set(METHODS)
    assert all(row["method_options"] == METHODS[row["method"]]["options"] for row in records)


def test_cartesian_slice_does_not_bridge_a_masked_peak() -> None:
    axis = np.linspace(-0.5, 0.5, 101)
    qx, qy = np.meshgrid(axis, axis)
    image = np.exp(-0.5 * ((qx - 0.2) / 0.035) ** 2) * np.exp(-0.5 * (qy / 0.12) ** 2)
    invalid = np.zeros(qx.shape, dtype=bool)
    invalid[:, 67:74] = True

    unmasked_points, _ = _cartesian_slice_points(
        image, qx, qy, np.zeros(qx.shape, dtype=bool), (0.05, 0.7), float(axis[1] - axis[0])
    )
    points, diagnostics = _cartesian_slice_points(
        image, qx, qy, invalid, (0.05, 0.7), float(axis[1] - axis[0])
    )

    assert any(point["accepted"] and abs(point["qx"] - 0.2) < 0.05 for point in unmasked_points)
    assert diagnostics["supported_run_count"] > 0
    assert not any(
        point["accepted"] and 0.14 <= point["qx"] <= 0.26
        for point in points
    )


def test_cartesian_slice_rejects_non_rectilinear_q_map() -> None:
    axis = np.linspace(-0.5, 0.5, 41)
    qx, qy = np.meshgrid(axis, axis)
    qy = qy + 0.1 * qx
    image = np.ones(qx.shape, dtype=float)
    invalid = np.zeros(qx.shape, dtype=bool)

    try:
        _cartesian_slice_points(image, qx, qy, invalid, (0.05, 0.7), 0.025)
    except ValueError as exc:
        assert "q y" in str(exc) or "qy" in str(exc)
    else:
        raise AssertionError("non-rectilinear q map must not be presented as a fixed-qy scan")
