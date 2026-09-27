from __future__ import annotations

import json

import numpy as np
import pytest

from butterfly_saxs import benchmark_initialization as benchmark
from butterfly_saxs.batch import FrameRef


def test_initialization_campaign_includes_q_jump_failure_and_noise_control() -> None:
    frames = benchmark.build_initialization_sequence(shape=(64, 64), seed=23)
    assert [frame["frame_id"] for frame in frames] == [
        "pre_00", "pre_01", "pre_02", "jump_03", "all_masked_failure",
        "post_05", "post_06", "noise_only",
    ]
    before = frames[2]
    jump = frames[3]
    assert before["sequence_role"] == "signal_before_jump"
    assert jump["sequence_role"] == "abrupt_q_jump"
    assert jump["structural_q0_nm_inv"] < before["structural_q0_nm_inv"]
    assert frames[4]["sequence_role"] == "all_pixels_masked_failure"
    assert np.all(frames[4]["mask"])
    assert frames[5]["structural_q0_nm_inv"] is not None
    assert frames[-1]["sequence_role"] == "noise_only_control"
    assert frames[-1]["structural_q0_nm_inv"] is None


def test_known_arc_control_has_supported_candidate_truth_and_ordered_failure() -> None:
    frames = benchmark.build_arc_control_sequence(shape=(64, 64), seed=20260927)
    assert [frame["frame_id"] for frame in frames] == [
        "arc_00", "arc_01", "arc_02", "arc_all_masked_failure", "arc_04", "arc_noise_only",
    ]
    assert [frame["truth"].get("axis_ratio") for frame in frames[:3]] == [0.4, 0.38, 0.2]
    assert frames[2]["sequence_role"] == "known_arc_ratio_jump"
    assert np.all(frames[3]["mask"])
    assert frames[4]["sequence_role"] == "known_arc_signal_after_jump"
    assert frames[-1]["truth"] is None
    assert np.isfinite(frames[0]["intensity_noisy"]).all()


def test_known_arc_pipeline_passes_seed_then_reseeds_from_current_observed_arcs(
    tmp_path,
) -> None:
    frames = benchmark.build_arc_control_sequence(shape=(64, 64), seed=20260927)[:2]
    inputs = {}
    masks = {}
    for frame in frames:
        frame_id = frame["frame_id"]
        inputs[frame_id] = tmp_path / f"{frame_id}.npy"
        masks[frame_id] = tmp_path / f"{frame_id}_mask.npy"
        np.save(inputs[frame_id], frame["intensity_noisy"], allow_pickle=False)
        np.save(masks[frame_id], frame["mask"], allow_pickle=False)
    qmap_object = frames[0]["qmap"]
    qmap = {
        name: qmap_object[name] for name in ("qx", "qy", "q", "q_unit")
    }
    config = benchmark.benchmark_sequence.pipeline_analysis_config(q_window=(0.05, 1.1))
    config["analysis"]["draw_axis_deg"] = 90.0
    config["analysis"]["max_nfev"] = 300
    config["analysis"]["ellipse"]["multistart"] = 1

    rows, calls, _run = benchmark._run_order(
        frames,
        inputs,
        masks,
        qmap,
        config,
        mode="warm_start",
        direction="forward",
        campaign="known_arc",
        frame_order=(0, 1),
    )
    assert rows["arc_00"]["warm_start_eligible"] is True
    assert rows["arc_01"]["warm_start_from"] == "arc_00"
    assert calls[1]["initial_parameter_count"] >= 4
    assert benchmark._seed_usage(calls)["calls_with_initial_parameters"] == 1
    received_seed = calls[1]["initial_parameters"]
    starts = rows["arc_01"]["effective_optimizer_starts"]
    assert starts
    actual_start = starts[0]
    assert actual_start["a"] != pytest.approx(received_seed["a"])
    assert actual_start["axis_ratio"] != pytest.approx(received_seed["axis_ratio"])
    assert actual_start["theta_deg"] != pytest.approx(received_seed["theta_deg"])


def test_seed_use_summary_labels_unexercised_initialization() -> None:
    result = benchmark._seed_usage(
        [
            {"frame_id": "first", "initial_parameter_count": 0},
            {"frame_id": "second", "initial_parameter_count": 0},
        ]
    )
    assert result["calls_with_initial_parameters"] == 0
    assert result["status"] == "not_exercised_no_eligible_seed"


def test_run_order_keeps_failure_position_and_previous_seed_source(tmp_path, monkeypatch) -> None:
    shape = (8, 8)
    frames = [
        {"frame_id": "first", "sequence_role": "signal"},
        {"frame_id": "failure", "sequence_role": "failure"},
        {"frame_id": "last", "sequence_role": "signal"},
    ]
    inputs = {}
    masks = {}
    for frame in frames:
        frame_id = frame["frame_id"]
        inputs[frame_id] = tmp_path / f"{frame_id}.npy"
        masks[frame_id] = tmp_path / f"{frame_id}_mask.npy"
        np.save(inputs[frame_id], np.ones(shape), allow_pickle=False)
        np.save(
            masks[frame_id],
            np.full(shape, frame_id == "failure", dtype=bool),
            allow_pickle=False,
        )
    qmap = {
        "qx": np.ones(shape),
        "qy": np.ones(shape),
        "q": np.ones(shape),
        "q_unit": "nm^-1",
    }
    seen_initials = []

    def fake_analyze(_image, *, mask, initial_parameters=None, **_kwargs):
        seen_initials.append(initial_parameters)
        if np.all(mask):
            raise ValueError("all detector pixels are masked")
        return {
            "parameters": {"a": 1.0, "b": 0.5},
            "success": True,
            "observed": np.ones(2),
        }

    monkeypatch.setattr(benchmark, "analyze_frame", fake_analyze)
    rows, calls, run = benchmark._run_order(
        frames,
        inputs,
        masks,
        qmap,
        {"analysis": {}},
        mode="warm_start",
        frame_order=(0, 1, 2),
    )

    assert run["processing_order"] == ["first", "failure", "last"]
    assert [rows[frame["frame_id"]]["status"] for frame in frames] == [
        "ok", "failed", "ok",
    ]
    assert rows["failure"]["warm_start_from"] == "first"
    assert rows["last"]["warm_start_from"] == "first"
    assert seen_initials == [None, {"a": 1.0, "b": 0.5}, {"a": 1.0, "b": 0.5}]
    assert [call["frame_id"] for call in calls] == ["first", "failure", "last"]


def test_reverse_run_rows_align_by_frame_id_and_lineage_names_source(tmp_path, monkeypatch) -> None:
    shape = (8, 8)
    frames = [{"frame_id": name, "sequence_role": "signal"} for name in ("first", "middle", "last")]
    inputs = {}
    masks = {}
    for frame in frames:
        frame_id = frame["frame_id"]
        inputs[frame_id] = tmp_path / f"{frame_id}.npy"
        masks[frame_id] = tmp_path / f"{frame_id}_mask.npy"
        np.save(inputs[frame_id], np.ones(shape), allow_pickle=False)
        np.save(masks[frame_id], np.zeros(shape, dtype=bool), allow_pickle=False)

    def fake_analyze(_image, *, initial_parameters=None, **_kwargs):
        if initial_parameters is None:
            return {"parameters": {"a": 1.0, "b": 0.5}, "success": True}
        return {"parameters": {"a": 1.0, "b": 0.5}, "success": True}

    monkeypatch.setattr(benchmark, "analyze_frame", fake_analyze)
    rows, _calls, run = benchmark._run_order(
        frames,
        inputs,
        masks,
        {"qx": np.ones(shape), "qy": np.ones(shape), "q": np.ones(shape), "q_unit": "nm^-1"},
        {"analysis": {}},
        mode="warm_start",
        frame_order=(2, 1, 0),
    )
    assert run["processing_order"] == ["last", "middle", "first"]
    assert list(rows) == ["last", "middle", "first"]
    assert rows["middle"]["warm_start_from"] == "last"
    assert rows["first"]["warm_start_from"] == "middle"
    assert FrameRef(inputs["first"], frame_id="first").key != FrameRef(
        inputs["last"], frame_id="last"
    ).key


@pytest.mark.parametrize(
    ("result", "expected_exit"),
    [
        (
            {
                "说明": "合成完成",
                "runs": {"independent": {"status_counts": {"ok": 2}}},
                "integrity": {"saved_inputs_unchanged": True},
            },
            0,
        ),
        ({"runs": {"forward": {"status_counts": {"warning": 1}}}}, 1),
        ({"runs": {"forward": {"status_counts": {"failed": 1}}}}, 1),
        ({"runs": {"forward": {"status_counts": {"ok": 1}, "cancelled": True}}}, 1),
    ],
)
def test_cli_json_output_and_completion_exit_status(
    tmp_path, monkeypatch, capsys, result, expected_exit
) -> None:
    monkeypatch.setattr(
        benchmark,
        "run_initialization_campaign",
        lambda *args, **kwargs: result,
    )

    exit_code = benchmark.main(["--output", str(tmp_path / "campaign")])

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed == result
    captured.out.encode("ascii")
    assert captured.err == ""
    assert exit_code == expected_exit


def test_cli_runner_input_error_emits_json_and_exits_two(tmp_path, monkeypatch, capsys) -> None:
    def fail_runner(*_args, **_kwargs):
        raise ValueError("invalid synthetic shape 参数")

    monkeypatch.setattr(benchmark, "run_initialization_campaign", fail_runner)

    exit_code = benchmark.main(["--output", str(tmp_path / "campaign")])

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "error",
        "error": "invalid synthetic shape 参数",
    }
    captured.out.encode("ascii")
    assert "invalid synthetic shape" in captured.err
    assert exit_code == 2


def test_cli_argument_error_emits_json_and_exits_two(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--size", "not-an-integer"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert "--size" in payload["error"]
    captured.out.encode("ascii")
    assert "invalid int value" in captured.err
    assert error.value.code == 2
