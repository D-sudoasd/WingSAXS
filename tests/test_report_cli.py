from __future__ import annotations

import json
import sys
from types import ModuleType

import numpy as np

from butterfly_saxs import cli
from butterfly_saxs.cli import build_parser, main
from butterfly_saxs.cli_contract import REPORT_SCHEMA, agent_manifest


def test_report_command_passes_options_and_emits_strict_summary(monkeypatch, tmp_path, capsys):
    calls = []
    result = {
        "schema_version": REPORT_SCHEMA,
        "status": "completed",
        "counts": {"frames": 2, "samples": 1},
        "outputs": {"index": str(tmp_path / "report" / "index.html")},
        "exit_code": 0,
    }

    def build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return result

    monkeypatch.setattr(cli, "_build_analysis_report", build)
    assert main([
        "report", str(tmp_path), "--resume", "--radial-bins", "24",
        "--angular-bins", "48", "--formats", "png", "svg", "--dpi", "240",
    ]) == 0

    assert calls == [(str(tmp_path), {
        "resume": True,
        "force": False,
        "radial_bins": 24,
        "angular_bins": 48,
        "formats": ["png", "svg"],
        "dpi": 240,
        "lamellar_settings": None,
    })]
    payload = json.loads(capsys.readouterr().out)
    assert payload == result
    assert payload["schema_version"] == REPORT_SCHEMA
    json.dumps(payload, allow_nan=False)


def test_report_builder_progress_is_sent_to_stderr(monkeypatch, capsys):
    report_module = ModuleType("butterfly_saxs.report")

    def build(output_dir, **kwargs):
        kwargs["progress"]("report progress")
        return {"schema_version": REPORT_SCHEMA, "exit_code": 0}

    report_module.build_analysis_report = build
    monkeypatch.setitem(sys.modules, "butterfly_saxs.report", report_module)
    result = cli._build_analysis_report("batch", formats=("png",))
    captured = capsys.readouterr()
    assert result["schema_version"] == REPORT_SCHEMA
    assert captured.out == ""
    assert captured.err == "report progress\n"


def test_report_options_are_validated_during_parse_before_any_work(monkeypatch, tmp_path, capsys):
    called = False

    def build(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid report settings must fail before report generation")

    monkeypatch.setattr(cli, "_build_analysis_report", build)
    assert main(["report", str(tmp_path), "--radial-bins", "1"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "usage_error"
    assert not called


def test_batch_report_options_fail_before_frame_analysis(monkeypatch, tmp_path, capsys):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    called = False

    def analyze(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid report settings must fail before fitting")

    monkeypatch.setattr(cli, "analyze_frame", analyze)
    assert main([
        "batch", str(source), "-o", str(tmp_path / "output"), "--report",
        "--report-angular-bins", "1",
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "usage_error"
    assert not called


def test_batch_report_grid_product_is_rejected_before_frame_analysis(monkeypatch, tmp_path, capsys):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    called = False

    def analyze(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("oversized report grid must fail before fitting")

    monkeypatch.setattr(cli, "analyze_frame", analyze)
    assert main([
        "batch", str(source), "-o", str(tmp_path / "output"), "--report",
        "--report-radial-bins", "1024", "--report-angular-bins", "1024",
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "value_error"
    assert "1000000" in payload["error"]["message"]
    assert not called


def test_describe_catalog_exposes_report_artifacts_and_full_workflow(capsys):
    assert main(["describe"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    report = next(command for command in catalog["commands"] if command["name"] == "report")
    assert report["stdout"] == REPORT_SCHEMA
    assert "results.npz" in report["artifacts"]["inputs"]
    assert "radial_profiles.csv" in report["artifacts"]["outputs"]["tables"]
    assert {
        "lamellar_parameters.csv",
        "lamellar_directions.csv",
        "lamellar_period_by_angle.csv",
        "lamellar_changes.csv",
    }.issubset(report["artifacts"]["outputs"]["tables"])
    morphology = report["artifacts"]["lamellar_morphology"]
    assert morphology["defaults"] == {"mode": "multi", "period_source": "ellipse"}
    assert "not inferred from ellipse theta" in morphology["interpretation"]
    assert "outputs" in report["artifacts"]["outputs"]["entry_points"]
    assert any("--stream --report --package" in command for command in catalog["recommended_agent_workflow"])
    assert agent_manifest() == catalog


def test_batch_generates_report_after_exports_before_package_and_bounds_stdout(
    monkeypatch, tmp_path, capsys
):
    from butterfly_saxs import delivery

    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    output = tmp_path / "output"
    events = []
    monkeypatch.setattr(
        cli,
        "analyze_frame",
        lambda *args, **kwargs: {"parameters": {"axis_ratio": 0.42}, "image": np.ones((8, 8))},
    )

    def build(output_dir, **kwargs):
        events.append("report")
        assert (output_dir / "results.npz").is_file()
        assert kwargs == {
            "resume": False,
            "force": False,
            "radial_bins": 32,
            "angular_bins": 36,
            "formats": ["png", "pdf"],
            "dpi": 200,
            "lamellar_settings": None,
        }
        return {
            "status": "completed",
            "counts": {"frames": 1, "plots": 4},
            "outputs": {"index": str(output_dir / "report" / "index.html")},
            "exit_code": 0,
            "large_internal_detail": "must not be copied to batch stdout",
        }

    def package(*args, **kwargs):
        events.append("package")
        return {
            "status": "completed",
            "operation_status": "completed",
            "counts": {},
            "archive": {},
            "outputs": {},
            "next_action": None,
            "exit_code": 0,
        }

    monkeypatch.setattr(cli, "_build_analysis_report", build)
    monkeypatch.setattr(delivery, "package_batch", package)
    assert main([
        "batch", str(source), "-o", str(output), "--report", "--package",
        "--report-radial-bins", "32", "--report-angular-bins", "36",
        "--report-formats", "png", "pdf", "--report-dpi", "200",
    ]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert events == ["report", "package"]
    assert payload["analysis_report"] == {
        "status": "completed",
        "counts": {"frames": 1, "plots": 4},
        "outputs": {"index": str(output / "report" / "index.html")},
    }
    assert "large_internal_detail" not in captured.out


def test_batch_report_failure_keeps_exports_and_offers_resume_command(monkeypatch, tmp_path, capsys):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    output = tmp_path / "output"
    monkeypatch.setattr(
        cli,
        "analyze_frame",
        lambda *args, **kwargs: {"parameters": {"axis_ratio": 0.42}, "image": np.ones((8, 8))},
    )

    def fail(*args, **kwargs):
        raise RuntimeError("report renderer failed")

    monkeypatch.setattr(cli, "_build_analysis_report", fail)
    assert main(["batch", str(source), "-o", str(output), "--report"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert (output / "results.npz").is_file()
    assert payload["analysis_report"]["status"] == "incomplete"
    assert payload["analysis_report"]["error"] == "report renderer failed"
    assert payload["analysis_report"]["next_command"] == [
        "bsaxs", "report", str(output), "--resume",
    ]


def test_batch_report_resume_command_preserves_custom_recipe(monkeypatch, tmp_path, capsys):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    output = tmp_path / "output"
    monkeypatch.setattr(
        cli,
        "analyze_frame",
        lambda *args, **kwargs: {"parameters": {"axis_ratio": 0.42}, "image": np.ones((8, 8))},
    )
    monkeypatch.setattr(
        cli,
        "_build_analysis_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("report renderer failed")),
    )
    assert main([
        "batch", str(source), "-o", str(output), "--report",
        "--report-radial-bins", "40", "--report-angular-bins", "48",
        "--report-formats", "svg", "tiff", "--report-dpi", "300",
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["analysis_report"]["next_command"] == [
        "bsaxs", "report", str(output), "--resume",
        "--radial-bins", "40", "--angular-bins", "48",
        "--formats", "svg", "tiff", "--dpi", "300",
    ]


def test_report_loads_nested_json_lamellar_settings_and_passes_normalized_values(
    monkeypatch, tmp_path, capsys
):
    settings_path = tmp_path / "lamellar.json"
    settings_path.write_text(
        json.dumps({
            "lamellar": {
                "mode": "single",
                "period_source": "ellipse",
                "thickness_ratio": 0.62,
                "width_ratio": 3.5,
                "depth_ratio": 1.7,
            }
        }),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        cli,
        "_build_analysis_report",
        lambda output_dir, **kwargs: calls.append(kwargs) or {"exit_code": 0},
    )

    assert main(["report", str(tmp_path), "--lamellar-settings", str(settings_path)]) == 0
    assert calls[0]["lamellar_settings"]["mode"] == "single"
    assert calls[0]["lamellar_settings"]["period_source"] == "ellipse"
    assert calls[0]["lamellar_settings"]["thickness_ratio"] == 0.62
    assert calls[0]["lamellar_settings"]["width_ratio"] == 3.5
    assert calls[0]["lamellar_settings"]["depth_ratio"] == 1.7
    assert calls[0]["lamellar_settings"]["seed"] == 0
    json.dumps(json.loads(capsys.readouterr().out), allow_nan=False)


def test_partial_lamellar_settings_use_report_defaults(monkeypatch, tmp_path, capsys):
    settings_path = tmp_path / "lamellar.json"
    settings_path.write_text('{"settings": {"thickness_ratio": 0.6}}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        cli,
        "_build_analysis_report",
        lambda output_dir, **kwargs: calls.append(kwargs) or {"exit_code": 0},
    )

    assert main(["report", str(tmp_path), "--lamellar-settings", str(settings_path)]) == 0
    settings = calls[0]["lamellar_settings"]
    assert settings["mode"] == "multi"
    assert settings["period_source"] == "ellipse"
    assert settings["thickness_ratio"] == 0.6
    assert settings["width_ratio"] == 4.0
    json.dumps(json.loads(capsys.readouterr().out), allow_nan=False)


def test_batch_loads_toml_settings_before_fitting_and_forwards_them(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    settings_path = tmp_path / "lamellar.toml"
    settings_path.write_text(
        'mode = "multi"\nperiod_source = "ellipse"\n'
        "stack_count = 18\nthickness_ratio = 0.7\n",
        encoding="utf-8",
    )
    analyzed = []
    reports = []
    monkeypatch.setattr(
        cli,
        "analyze_frame",
        lambda *args, **kwargs: analyzed.append(True)
        or {"parameters": {"axis_ratio": 0.42}, "image": np.ones((8, 8))},
    )
    monkeypatch.setattr(
        cli,
        "_build_analysis_report",
        lambda output_dir, **kwargs: reports.append(kwargs) or {"exit_code": 0},
    )

    assert main([
        "batch", str(source), "-o", str(tmp_path / "output"), "--report",
        "--report-lamellar-settings", str(settings_path),
    ]) == 0
    assert analyzed
    settings = reports[0]["lamellar_settings"]
    assert settings["mode"] == "multi"
    assert settings["period_source"] == "ellipse"
    assert settings["stack_count"] == 18
    assert settings["thickness_ratio"] == 0.7
    json.dumps(json.loads(capsys.readouterr().out), allow_nan=False)


def test_invalid_batch_lamellar_settings_fail_before_frame_analysis(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    settings_path = tmp_path / "invalid.json"
    settings_path.write_text('{"thickness_ratio": 1.0}', encoding="utf-8")
    analyzed = False

    def analyze(*args, **kwargs):
        nonlocal analyzed
        analyzed = True
        raise AssertionError("invalid lamellar settings must fail before fitting")

    monkeypatch.setattr(cli, "analyze_frame", analyze)
    assert main([
        "batch", str(source), "-o", str(tmp_path / "output"), "--report",
        "--report-lamellar-settings", str(settings_path),
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "value_error"
    assert "thickness_ratio" in payload["error"]["message"]
    assert not analyzed


def test_unknown_nested_lamellar_field_fails_before_batch_fit(monkeypatch, tmp_path, capsys):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    settings_path = tmp_path / "typo.json"
    settings_path.write_text(
        json.dumps({
            "analysis": {"q_min": 0.1},
            "lamellar_settings": {"thickness_ratios": 0.74},
        }),
        encoding="utf-8",
    )
    analyzed = False

    def analyze(*args, **kwargs):
        nonlocal analyzed
        analyzed = True
        raise AssertionError("unknown lamellar fields must fail before fitting")

    monkeypatch.setattr(cli, "analyze_frame", analyze)
    assert main([
        "batch", str(source), "-o", str(tmp_path / "output"), "--report",
        "--report-lamellar-settings", str(settings_path),
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    message = payload["error"]["message"]
    assert payload["error"]["code"] == "value_error"
    assert "thickness_ratios" in message
    assert "thickness_ratio" in message
    assert not analyzed


def test_batch_report_recovery_command_retains_lamellar_settings_path(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8)))
    output = tmp_path / "output"
    settings_path = tmp_path / "lamellar.toml"
    settings_path.write_text('mode = "multi"\n', encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "analyze_frame",
        lambda *args, **kwargs: {"parameters": {"axis_ratio": 0.42}, "image": np.ones((8, 8))},
    )
    monkeypatch.setattr(
        cli,
        "_build_analysis_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("report renderer failed")),
    )
    assert main([
        "batch", str(source), "-o", str(output), "--report",
        "--report-lamellar-settings", str(settings_path),
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["analysis_report"]["next_command"] == [
        "bsaxs", "report", str(output), "--resume",
        "--lamellar-settings", str(settings_path),
    ]


def test_report_command_is_available_in_help() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert "report" in choices
