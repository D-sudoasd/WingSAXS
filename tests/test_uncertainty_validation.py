from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from butterfly_saxs.uncertainty_validation import (
    summarize_calibration_sensitivity,
    summarize_uncertainty_trials,
)


def test_coverage_keeps_all_trials_and_finite_failed_candidates() -> None:
    records = [
        {
            "execution_status": "complete",
            "solver_success": True,
            "truth": {"a": 2.0},
            "candidate": {"a": 2.1},
            "intervals": {"a": [1.8, 2.2]},
            "parameter_status": {"a": "estimate"},
            "confidence": {"a": "moderate"},
            "condition": 4.0,
        },
        {
            "execution_status": "complete",
            "solver_success": False,
            "truth": {"a": 2.0},
            "candidate": {"a": 2.4},
            "intervals": {"a": [2.1, 2.6]},
            "parameter_status": {"a": "candidate"},
            "confidence": {"a": "limited"},
            "condition": 20.0,
        },
        {
            "execution_status": "complete",
            "solver_success": False,
            "truth": {"a": 2.0},
            "candidate": {"a": 1.9},
            "intervals": {"a": [None, None]},
            "parameter_status": {"a": "candidate"},
            "condition": 1.0e9,
        },
        {
            "execution_status": "error",
            "solver_success": False,
            "truth": {},
            "candidate": {"a": 1.7},
            "intervals": {"a": [1.0, 2.0]},
        },
    ]

    result = summarize_uncertainty_trials(records, parameters=("a",))
    metric = result["parameters"]["a"]
    assert result["trial_count"] == 4
    assert result["execution_status_counts"] == {"complete": 3, "error": 1}
    assert result["solver_success_counts"] == {"False": 3, "True": 1}
    assert metric["finite_candidate_count"] == 4
    assert metric["candidate_truth_pair_count"] == 3
    assert metric["candidate_bias"] == pytest.approx(0.4 / 3.0)
    assert metric["interval_attempt_count"] == 4
    assert metric["interval_available_count"] == 2
    assert metric["interval_unavailable_count"] == 2
    assert metric["interval_missing_count"] == 1
    assert metric["reference_truth_missing_count"] == 1
    assert metric["interval_contains_truth_count"] == 1
    assert metric["coverage_fraction_available_intervals"] == pytest.approx(0.5)
    assert metric["coverage_fraction_all_trials"] == pytest.approx(0.25)
    assert metric["coverage_wilson_95_all_trials"] == pytest.approx([0.0456, 0.6994], abs=0.001)
    assert metric["candidate_status_counts"] == {"candidate": 2, "estimate": 1}
    assert result["condition_number"]["maximum"] == pytest.approx(1.0e9)


@pytest.mark.parametrize(
    "raw_interval",
    ["1.0", [1.0], [1.0, 2.0, 3.0], [2.0, 1.0], [float("nan"), 2.0], None],
)
def test_malformed_intervals_are_unavailable_not_scored(raw_interval) -> None:
    result = summarize_uncertainty_trials(
        [{"truth": {"axis_ratio": 0.4}, "candidate": {"axis_ratio": 0.42},
          "intervals": {"axis_ratio": raw_interval}}],
        parameters=("axis_ratio",),
    )
    metric = result["parameters"]["axis_ratio"]
    assert metric["interval_available_count"] == 0
    assert metric["interval_missing_count"] == 1
    assert metric["coverage_fraction_available_intervals"] is None
    assert metric["coverage_fraction_all_trials"] == 0.0


def test_calibration_sensitivity_retains_unpaired_failure() -> None:
    result = summarize_calibration_sensitivity(
        {"a": 2.0, "axis_ratio": 0.4},
        [
            {"label": "q_plus", "candidate": {"a": 2.02, "axis_ratio": 0.4},
             "solver_success": True, "perturbation": {"qcal_scale": 1.01}},
            {"label": "center_fail", "candidate": {}, "solver_success": False},
        ],
        parameters=("a", "axis_ratio"),
    )
    assert result["variants"]["q_plus"]["parameters"]["a"]["delta"] == pytest.approx(0.02)
    assert result["variants"]["q_plus"]["parameters"]["a"]["relative_delta"] == pytest.approx(0.01)
    assert result["variants"]["center_fail"]["parameters"]["a"]["paired_finite"] is False
    assert "not instrument uncertainty" in result["interpretation"]


def test_resampling_error_does_not_erase_baseline_candidate_or_truth(monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "scripts" / "validate_butterfly_uncertainty.py"
    spec = importlib.util.spec_from_file_location("uncertainty_validation_runner_test", source)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    monkeypatch.setattr(
        runner,
        "_fit_summary",
        lambda *_args, **_kwargs: {
            "candidate": {"a": 0.72, "b": 0.288, "axis_ratio": 0.4, "theta_deg": 5.0},
            "solver_success": True,
            "measurement_status": "supported",
            "parameter_status": {"a": "estimate"},
            "confidence": {"a": "moderate"},
            "condition": 10.0,
            "topology_failed": False,
            "missing_observed_sides": [],
        },
    )

    def fail_resampling(*_args, **_kwargs):
        raise RuntimeError("resampling unavailable")

    monkeypatch.setattr(runner, "resample_butterfly", fail_resampling)
    trial = runner._run_trial("ellipse_ratio_400", 123, 24, 4, 100)
    assert trial["execution_status"] == "complete"
    assert trial["truth"]["axis_ratio"] == pytest.approx(0.4)
    assert trial["candidate"]["a"] == pytest.approx(0.72)
    assert trial["resampling"]["status"] == "error"
    assert trial["intervals"] == {}
