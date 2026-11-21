from __future__ import annotations

import ast
import json
from pathlib import Path

from butterfly_saxs.cli import build_parser, main
from butterfly_saxs.cli_contract import (
    AGENT_MANIFEST_SCHEMA,
    CLI_ERROR_SCHEMA,
    INSPECT_SCHEMA,
    agent_manifest,
)
from butterfly_saxs.errors import PipelineError


def test_project_reports_warning_candidates_as_completed_not_failed_or_certified(monkeypatch, capsys):
    import butterfly_saxs.cli as cli_module
    from butterfly_saxs.batch import BatchRunResult, FrameFitResult

    run = BatchRunResult(
        [FrameFitResult("partial.npy", result={"parameters": {"a": 0.7}}, status="warning")],
        mode="independent", input_hash="input", config_hash="config",
    )
    monkeypatch.setattr(cli_module, "run_project", lambda *args, **kwargs: run)
    assert main(["project", "project.toml"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["n_warning"] == report["n_completed"] == 1
    assert report["n_success"] == report["n_failed"] == 0
    assert report["frames"][0]["result"]["parameters"]["a"] == 0.7


def test_cli_module_keeps_numpy_off_the_import_path() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "butterfly_saxs" / "cli.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "numpy"
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("numpy")
            assert module != "pipeline"


def test_describe_is_the_default_agent_catalog(
    capsys,
) -> None:
    assert main([]) == 0
    default_report = json.loads(capsys.readouterr().out)
    assert main(["describe"]) == 0
    named_report = json.loads(capsys.readouterr().out)

    assert default_report == named_report == agent_manifest()
    assert default_report["schema_version"] == AGENT_MANIFEST_SCHEMA
    assert default_report["ok"] is True
    names = {item["name"] for item in default_report["commands"]}
    assert {"describe", "doctor", "inspect", "analyze", "batch", "preflight"} <= names
    assert default_report["defaults"]["recommended_ridge_method"] == "butterfly_curvature"
    assert any("scientific acceptance" in item.lower() for item in default_report["invariants"])
    json.dumps(default_report, allow_nan=False)


def test_cli_help_includes_describe_and_doctor() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {"describe", "doctor", "inspect", "analyze", "batch", "synthetic", "gui"} <= set(choices)


def test_describe_one_command_retains_its_contract_without_unrelated_details(capsys) -> None:
    full = agent_manifest()
    assert main(["describe", "batch"]) == 0
    selected = json.loads(capsys.readouterr().out)
    assert selected == agent_manifest("batch")
    assert selected["commands"] == [item for item in full["commands"] if item["name"] == "batch"]
    assert selected["invariants"] == full["invariants"]
    assert selected["exit_codes"] == full["exit_codes"]
    assert "recommended_agent_workflow" not in selected
    assert len(json.dumps(selected)) < len(json.dumps(full)) / 2


def test_describe_unknown_command_emits_json_error(capsys) -> None:
    assert set(agent_manifest("describe")["commands"][0]["exit_codes"]) == {"0", "2"}
    assert main(["describe", "unknown"]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["schema_version"] == CLI_ERROR_SCHEMA
    assert "unknown catalog command" in error["error"]["message"]


def test_scoped_catalog_covers_every_supported_cli_command() -> None:
    choices = build_parser()._subparsers._group_actions[0].choices
    assert {item["name"] for item in agent_manifest()["commands"]} == set(choices)
    for command in choices:
        selected = agent_manifest(command)
        assert [item["name"] for item in selected["commands"]] == [command]


def test_inspect_without_input_emits_json_error_envelope(capsys) -> None:
    assert main(["inspect"]) == 2
    captured = capsys.readouterr()
    assert "错误：" in captured.err
    payload = json.loads(captured.out)
    assert payload["schema_version"] == CLI_ERROR_SCHEMA
    assert payload["ok"] is False
    assert payload["exit_code"] == 2
    assert payload["error"]["type"] == "PipelineError"
    assert payload["error"]["code"] == "pipeline_error"
    assert payload["agent"]["scientific_acceptance"] is False
    json.dumps(payload, allow_nan=False)


def test_usage_error_is_json_on_stdout(capsys) -> None:
    assert main(["inspect", "--definitely-not-a-flag"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == CLI_ERROR_SCHEMA
    assert payload["error"]["code"] == "usage_error"
    assert payload["ok"] is False


def test_inspect_report_carries_agent_contract(tmp_path, capsys) -> None:
    import numpy as np

    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8), dtype=float))
    output = tmp_path / "inspect.json"
    assert main(["inspect", str(source), "-o", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert report == saved
    assert report["schema_version"] == INSPECT_SCHEMA
    assert report["result_type"] == "inspect"
    assert report["agent"]["scientific_acceptance"] is False
    assert report["agent"]["pixel_q"] is True
    assert any("poni" in item.lower() for item in report["agent"]["next"])
    assert "butterfly_curvature" in " ".join(report["agent"]["next"])


def test_analyze_annotation_keeps_payload_and_denies_acceptance(
    monkeypatch, capsys
) -> None:
    import butterfly_saxs.cli as cli_module

    class Result:
        def to_mapping(self):
            return {
                "ellipse_fit": {"status": "ok", "success": True},
                "full2d": {"status": "failed", "success": False},
                "flags": {"uncalibrated_pixel_q": True},
            }

    monkeypatch.setattr(cli_module, "analyze_frame", lambda *args, **kwargs: Result())
    assert main(["analyze", "frame.npy", "--full2d"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["full2d"]["status"] == "failed"
    assert report["schema_version"] == "lamellarsaxs2d.analysis_summary.v1"
    assert report["agent"]["scientific_acceptance"] is False
    assert report["agent"]["quality_gate_reason"]
    assert report["agent"]["pixel_q"] is True


def test_analyze_warn_exits_one_without_discarding_measurement(monkeypatch, capsys) -> None:
    import butterfly_saxs.cli as cli_module

    class Result:
        def to_mapping(self):
            return {
                "butterfly": {
                    "settings": {"stage": "evaluate"},
                    "candidate_fit": {"success": True, "status": "ok"},
                    "quality": {"status": "WARN", "scientific_status": "NOT_ACCEPTED"},
                    "points": [{"accepted": True}],
                },
                "ellipse_fit": {"success": True, "status": "ok"},
            }

    monkeypatch.setattr(cli_module, "analyze_frame", lambda *args, **kwargs: Result())
    assert main(["analyze", "frame.npy"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["butterfly"]["candidate_fit"]["success"] is True
    assert report["agent"]["quality_gate_reason"] == "butterfly.quality.status=WARN"
    assert report["agent"]["exit_code"] == 1


def test_pipeline_error_is_stdlib_importable() -> None:
    assert issubclass(PipelineError, RuntimeError)
    assert PipelineError("x").args == ("x",)


def test_lazy_import_invariant_for_help_describe_and_doctor() -> None:
    import os
    import subprocess
    import sys

    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_src
    heavy = ("numpy", "scipy", "matplotlib", "pyFAI", "PySide6", "h5py", "fabio")

    for cli_args in (
        ["--help"],
        ["describe"],
        ["describe", "inspect"],
        ["describe", "analyze"],
        ["describe", "batch"],
        ["describe", "advise"],
        ["describe", "campaign"],
        ["describe", "init-manifest"],
        ["doctor", "--help"],
    ):
        probe = (
            "import json, sys\n"
            "from butterfly_saxs.cli import main\n"
            f"argv = {cli_args!r}\n"
            "try:\n"
            "    main(argv)\n"
            "except SystemExit:\n"
            "    pass\n"
            f"loaded = [m for m in {heavy!r} if m in sys.modules]\n"
            "sys.stderr.write('LOADED=' + json.dumps(loaded) + '\\n')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        marker = [line for line in proc.stderr.splitlines() if line.startswith("LOADED=")]
        assert marker, (cli_args, proc.stderr, proc.stdout)
        loaded = json.loads(marker[-1].split("=", 1)[1])
        assert loaded == [], (cli_args, loaded)

    # Importing CLI, contract, campaign, project, and doctor modules stays stdlib-only.
    import_probe = (
        "import json, sys\n"
        "from butterfly_saxs import campaign, cli, cli_contract, doctor, project\n"
        f"loaded = [m for m in {heavy!r} if m in sys.modules]\n"
        "sys.stderr.write('LOADED=' + json.dumps(loaded) + '\\n')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", import_probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    marker = [line for line in proc.stderr.splitlines() if line.startswith("LOADED=")]
    assert marker and json.loads(marker[-1].split("=", 1)[1]) == []


def test_cli_init_manifest_command_writes_csv_and_enforces_overwrite(tmp_path, capsys) -> None:
    import numpy as np

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    np.save(frames_dir / "NO50S_350C_0001.npy", np.ones((8, 8), dtype=float))
    np.save(frames_dir / "NO50S_350C_0002.npy", np.ones((8, 8), dtype=float))
    out_csv = tmp_path / "manifest.csv"

    assert main(["init-manifest", "--input", str(frames_dir), "--output", str(out_csv), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "wingsaxs.manifest_init.v1"
    assert payload["n_frames"] == 2
    assert out_csv.is_file()
    json.dumps(payload, allow_nan=False, ensure_ascii=True)

    # Refuses overwrite without --force (exit 2).
    assert main(["init-manifest", "--input", str(frames_dir), "--output", str(out_csv)]) == 2
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["error"]["code"] == "output_exists"

    # Succeeds with --force.
    assert main(["init-manifest", "--input", str(frames_dir), "--output", str(out_csv), "--force"]) == 0
    capsys.readouterr()


def test_cli_advise_command_forwards_contract_and_handles_missing_module(
    monkeypatch, tmp_path, capsys
) -> None:
    import butterfly_saxs.cli as cli_module

    calls = []

    def fake_advise(image_path, **kwargs):
        calls.append((image_path, kwargs))
        return {
            "schema_version": "wingsaxs.advise.v1",
            "beamstop": {"recommended_mask_radius_px": 14},
            "q_window": [0.12, 1.35],
            "symmetry": {"friedel_bisector_deg": 90.2, "orthogonal_flip_detected": False},
            "parameters": {"recommended_workflow": "annular"},
            "warnings": [],
            "remediation": [],
        }

    monkeypatch.setattr(cli_module, "advise_frame", fake_advise)
    out_json = tmp_path / "advice.json"
    assert main([
        "advise",
        "--image", "sample.h5",
        "--poni", "geom.poni",
        "--mask", "mask.npy",
        "--image-dataset", "entry/data/data",
        "--image-frame-index", "3",
        "--nominal-draw-axis-deg", "88.5",
        "--output", str(out_json),
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls == [(
        "sample.h5",
        {
            "poni_path": "geom.poni",
            "mask_path": "mask.npy",
            "image_dataset": "entry/data/data",
            "image_frame_index": 3,
            "mask_dataset": None,
            "mask_frame_index": None,
            "nominal_draw_axis_deg": 88.5,
            "bright_axis_convention": "draw",
        },
    )]
    assert payload["schema_version"] == "wingsaxs.advise.v1"
    assert payload["agent"]["scientific_acceptance"] is False
    assert out_json.is_file()
    json.dumps(payload, allow_nan=False, ensure_ascii=True)

    # Explicit --bright-axis-convention reference forwards cleanly and is documented in describe.
    assert main([
        "advise",
        "--image", "sample.h5",
        "--nominal-draw-axis-deg", "90",
        "--bright-axis-convention", "reference",
        "--json",
    ]) == 0
    capsys.readouterr()
    assert calls[-1][1]["bright_axis_convention"] == "reference"
    assert calls[-1][1]["nominal_draw_axis_deg"] == 90.0
    assert "bright_axis_convention" in agent_manifest("advise")["commands"][0]["options"]

    # Invalid --bright-axis-convention fails with usage error (exit 2).
    assert main(["advise", "--image", "sample.h5", "--bright-axis-convention", "invalid"]) == 2
    bad = json.loads(capsys.readouterr().out)
    assert bad["error"]["code"] == "usage_error"

    # Existing output refuses overwrite without --force.
    assert main(["advise", "--image", "sample.h5", "--output", str(out_json)]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["error"]["code"] == "output_exists"


def test_cli_campaign_command_forwards_flags_and_preserves_exit_code(
    monkeypatch, tmp_path, capsys
) -> None:
    import butterfly_saxs.cli as cli_module

    calls = []

    def fake_run(config_path, output_root, **kwargs):
        calls.append((config_path, output_root, kwargs))
        return {
            "schema_version": "wingsaxs.campaign_run.v1",
            "status": "partial_failure",
            "exit_code": 1,
            "n_samples": 2,
            "n_success": 1,
            "n_warning": 0,
            "n_failed": 1,
            "n_skipped": 0,
            "samples": [],
        }

    monkeypatch.setattr(cli_module, "run_campaign", fake_run)
    assert main([
        "campaign",
        "--config", str(tmp_path / "camp.toml"),
        "--output", str(tmp_path / "out"),
        "--force",
        "--json",
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "wingsaxs.campaign_run.v1"
    assert payload["n_failed"] == 1
    assert calls[0][0] == str(tmp_path / "camp.toml")
    assert calls[0][1] == str(tmp_path / "out")
    assert calls[0][2]["force"] is True
    assert calls[0][2]["continue_on_error"] is True
    json.dumps(payload, allow_nan=False, ensure_ascii=True)


def test_cli_preset_on_analyze_and_batch_preserves_legacy_when_omitted(
    monkeypatch, tmp_path, capsys
) -> None:
    import numpy as np
    import butterfly_saxs.cli as cli_module

    seen_configs = []

    class DummyResult:
        def to_mapping(self):
            return {"ellipse_fit": {"status": "ok", "success": True}}

    def fake_analyze(*args, **kwargs):
        seen_configs.append(kwargs.get("config"))
        return DummyResult()

    monkeypatch.setattr(cli_module, "analyze_frame", fake_analyze)

    # Omitting --preset preserves None config (100% legacy).
    assert main(["analyze", "frame.npy"]) == 0
    capsys.readouterr()
    assert seen_configs[-1] is None

    # --preset annular populates workflow preset; explicit CLI flags override it.
    assert main(["analyze", "frame.npy", "--preset", "annular", "--butterfly-resamples", "8"]) == 0
    capsys.readouterr()
    cfg = seen_configs[-1]
    assert cfg is not None
    assert cfg.analysis["ridge_method"] == "butterfly_curvature"
    assert cfg.analysis["butterfly"]["trace_method"] == "annular_peak"
    assert cfg.analysis["butterfly"]["resamples"] == 8

    # Batch also supports --preset.
    src = tmp_path / "f1.npy"
    np.save(src, np.ones((8, 8), dtype=float))
    seen_configs.clear()
    monkeypatch.setattr(
        cli_module,
        "analyze_frame",
        lambda *a, **kw: seen_configs.append(kw.get("config"))
        or {"parameters": {"axis_ratio": 0.4}, "image": np.ones((8, 8))},
    )
    assert main(["batch", str(src), "-o", str(tmp_path / "batch_out"), "--preset", "sector"]) == 0
    capsys.readouterr()
    batch_cfg = seen_configs[-1]
    assert batch_cfg.analysis["ridge_method"] == "butterfly_curvature"
    assert batch_cfg.analysis["butterfly"]["trace_method"] == "radial_sector"


def test_cli_inspect_recommend_supports_native_and_fallback(monkeypatch, capsys) -> None:
    import butterfly_saxs.cli as cli_module

    # Case 1: inspect_frame natively supports recommend=True.
    native_calls = []

    def native_inspect(source, *, recommend=False, nominal_draw_axis_deg=90.0, **kwargs):
        native_calls.append((source, recommend, nominal_draw_axis_deg))
        return {
            "shape": [8, 8],
            "q_unit": "pixel-q",
            "advisor": {"recommended_workflow": "annular"},
        }

    monkeypatch.setattr(cli_module, "inspect_frame", native_inspect)
    assert main(["inspect", "frame.npy", "--recommend", "--nominal-draw-axis-deg", "85"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert native_calls == [("frame.npy", True, 85.0)]
    assert report["advisor"]["recommended_workflow"] == "annular"

    # Case 2: legacy inspect_frame without recommend parameter falls back to advise_frame.
    def legacy_inspect(source, *, poni=None, config=None, frame=None, dataset=None,
                       mask=None, mask_frame=None, mask_dataset=None, valid_mask=None):
        del poni, config, frame, dataset, mask, mask_frame, mask_dataset, valid_mask
        return {"shape": [8, 8], "q_unit": "pixel-q"}

    monkeypatch.setattr(cli_module, "inspect_frame", legacy_inspect)
    monkeypatch.setattr(
        cli_module,
        "advise_frame",
        lambda image_path, **kw: {"source": image_path, "recommended_workflow": "curvature"},
    )
    assert main(["inspect", "frame.npy", "--recommend"]) == 0
    report2 = json.loads(capsys.readouterr().out)
    assert report2["advisor"]["recommended_workflow"] == "curvature"


def test_cli_report_journal_style_forwards_non_default_and_preserves_default(
    monkeypatch, tmp_path, capsys
) -> None:
    import sys
    from types import ModuleType

    calls = []
    fake_report_mod = ModuleType("butterfly_saxs.report")

    def fake_build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return {"schema_version": "wingsaxs.analysis_report.v1", "exit_code": 0}

    fake_report_mod.build_analysis_report = fake_build
    monkeypatch.setitem(sys.modules, "butterfly_saxs.report", fake_report_mod)

    # Default omits journal_style for backward compatibility.
    assert main(["report", str(tmp_path)]) == 0
    capsys.readouterr()
    assert "journal_style" not in calls[-1][1]

    # Explicit non-default style is forwarded cleanly.
    assert main(["report", str(tmp_path), "--journal-style", "nature"]) == 0
    capsys.readouterr()
    assert calls[-1][1]["journal_style"] == "nature"


def test_describe_exposes_v21_cli_options() -> None:
    inspect_opts = agent_manifest("inspect")["commands"][0]["options"]
    assert "beamstop_radius_px" in inspect_opts

    analyze_opts = agent_manifest("analyze")["commands"][0]["options"]
    assert {
        "beamstop_radius_px",
        "uncertainty_dataset",
        "include_physics",
        "intensity_mode",
    } <= set(analyze_opts)
    assert analyze_opts["intensity_mode"] == ["calibrated", "raw"]

    batch_opts = agent_manifest("batch")["commands"][0]["options"]
    assert {"beamstop_radius_px", "deduplicate_frames", "workers"} <= set(batch_opts)

    campaign_opts = agent_manifest("campaign")["commands"][0]["options"]
    assert {"beamstop_radius_px", "deduplicate_frames", "workers"} <= set(campaign_opts)


def test_cli_inspect_and_analyze_v21_options_forward_and_validate(monkeypatch, capsys) -> None:
    import butterfly_saxs.cli as cli_module

    # 1. inspect forwards --beamstop-radius-px (both 'auto' and numeric float).
    inspect_calls: list[dict] = []

    def native_inspect(source, *, beamstop_radius_px=None, **kwargs):
        inspect_calls.append({"source": source, "beamstop_radius_px": beamstop_radius_px})
        return {"shape": [8, 8], "q_unit": "pixel-q"}

    monkeypatch.setattr(cli_module, "inspect_frame", native_inspect)
    assert main(["inspect", "frame.npy", "--beamstop-radius-px", "auto"]) == 0
    capsys.readouterr()
    assert inspect_calls[-1]["beamstop_radius_px"] == "auto"

    assert main(["inspect", "frame.npy", "--beamstop-radius-px", "18.5"]) == 0
    capsys.readouterr()
    assert inspect_calls[-1]["beamstop_radius_px"] == 18.5

    # Legacy inspect_frame without beamstop_radius_px degrades gracefully with stderr warning.
    def legacy_inspect(source, *, poni=None, config=None, frame=None, dataset=None,
                       mask=None, mask_frame=None, mask_dataset=None, valid_mask=None):
        del poni, config, frame, dataset, mask, mask_frame, mask_dataset, valid_mask
        return {"shape": [8, 8], "q_unit": "pixel-q"}

    monkeypatch.setattr(cli_module, "inspect_frame", legacy_inspect)
    assert main(["inspect", "frame.npy", "--beamstop-radius-px", "12"]) == 0
    captured = capsys.readouterr()
    assert "beamstop_radius_px" in captured.err
    assert json.loads(captured.out)["shape"] == [8, 8]

    # 2. analyze forwards v2.1 options when supported and omits them when unset.
    analyze_calls: list[dict] = []

    class DummyResult:
        def to_mapping(self):
            return {"ellipse_fit": {"status": "ok", "success": True}}

    def native_analyze(source, **kwargs):
        analyze_calls.append({"source": source, **kwargs})
        return DummyResult()

    monkeypatch.setattr(cli_module, "analyze_frame", native_analyze)
    assert main(["analyze", "frame.npy"]) == 0
    capsys.readouterr()
    for omitted in ("beamstop_radius_px", "uncertainty_dataset", "include_physics", "intensity_mode"):
        assert omitted not in analyze_calls[-1]

    assert main([
        "analyze",
        "frame.npy",
        "--beamstop-radius-px", "16",
        "--uncertainty-dataset", "auto",
        "--include-physics",
        "--intensity-mode", "raw",
    ]) == 0
    capsys.readouterr()
    last = analyze_calls[-1]
    assert last["beamstop_radius_px"] == 16.0
    assert last["uncertainty_dataset"] == "auto"
    assert last["include_physics"] is True
    assert last["intensity_mode"] == "raw"

    # Legacy analyze_frame without new kwargs skips unsupported ones gracefully.
    def legacy_analyze(source, *, poni=None, config=None, full2d=False, frame=None,
                       dataset=None, mask=None, mask_frame=None, mask_dataset=None,
                       valid_mask=None, output=None, force=False):
        del source, poni, config, full2d, frame, dataset, mask, mask_frame, mask_dataset, valid_mask, output, force
        return DummyResult()

    monkeypatch.setattr(cli_module, "analyze_frame", legacy_analyze)
    assert main(["analyze", "frame.npy", "--beamstop-radius-px", "auto", "--include-physics"]) == 0
    captured = capsys.readouterr()
    assert "beamstop_radius_px" in captured.err
    assert "include_physics" in captured.err

    # 3. Invalid CLI values produce strict JSON usage errors (exit 2).
    for bad_argv in (
        ["inspect", "frame.npy", "--beamstop-radius-px", "-3"],
        ["analyze", "frame.npy", "--beamstop-radius-px", "nan"],
        ["analyze", "frame.npy", "--beamstop-radius-px", "invalid"],
        ["analyze", "frame.npy", "--intensity-mode", "uncalibrated"],
        ["batch", "frame.npy", "--workers", "0"],
        ["campaign", "camp.toml", "--workers", "-2"],
    ):
        assert main(bad_argv) == 2
        err = json.loads(capsys.readouterr().out)
        assert err["schema_version"] == CLI_ERROR_SCHEMA
        assert err["error"]["code"] == "usage_error"


def test_cli_batch_and_campaign_v21_options_forward_and_pickle(monkeypatch, tmp_path, capsys) -> None:
    import pickle
    import numpy as np
    import butterfly_saxs.batch as batch_module
    import butterfly_saxs.cli as cli_module

    # Picklability check for Windows spawn pools.
    adapter = cli_module._BatchFrameAnalyzer(poni="g.poni", beamstop_radius_px="auto")
    adapter._geometry_cache["k"] = object()
    restored = pickle.loads(pickle.dumps(adapter))
    assert restored.poni == "g.poni"
    assert restored.beamstop_radius_px == "auto"
    assert restored._geometry_cache == {}

    # Batch forwarding: inspect kwargs passed to run_batch and analyze_frame.
    src = tmp_path / "f1.npy"
    np.save(src, np.ones((8, 8), dtype=float))
    analyze_seen: list[dict] = []
    run_batch_seen: list[dict] = []
    real_run_batch = batch_module.run_batch

    def spy_run_batch(frames, analyzer, **kwargs):
        run_batch_seen.append(dict(kwargs))
        # Strip v2.1 kwargs if current run_batch doesn't yet define them.
        accepted, _ = cli_module._filter_supported_kwargs(real_run_batch, kwargs)
        return real_run_batch(frames, analyzer, **accepted)

    def spy_analyze(path, **kwargs):
        analyze_seen.append({"path": str(path), **kwargs})
        return {"parameters": {"axis_ratio": 0.42}, "image": np.ones((8, 8))}

    monkeypatch.setattr(batch_module, "run_batch", spy_run_batch)
    monkeypatch.setattr(cli_module, "analyze_frame", spy_analyze)

    assert main([
        "batch",
        str(src),
        "-o", str(tmp_path / "batch_out"),
        "--beamstop-radius-px", "20",
        "--deduplicate-frames",
        "--workers", "4",
    ]) == 0
    capsys.readouterr()
    assert run_batch_seen[-1]["deduplicate_identical_frames"] is True
    assert run_batch_seen[-1]["max_workers"] == 4
    assert run_batch_seen[-1]["beamstop_radius_px"] == 20.0
    assert analyze_seen[-1]["beamstop_radius_px"] == 20.0
    assert analyze_seen[-1]["config"].analysis["beamstop_radius_px"] == 20.0

    # Campaign forwarding with **kwargs runner and legacy runner.
    campaign_calls: list[dict] = []

    def native_campaign(config_path, output_root, **kwargs):
        campaign_calls.append({"config_path": config_path, "output_root": output_root, **kwargs})
        return {
            "schema_version": "wingsaxs.campaign_run.v1",
            "status": "ok",
            "exit_code": 0,
            "n_samples": 1,
            "n_success": 1,
            "n_warning": 0,
            "n_failed": 0,
            "n_skipped": 0,
            "samples": [],
        }

    monkeypatch.setattr(cli_module, "run_campaign", native_campaign)
    assert main([
        "campaign",
        "--config", str(tmp_path / "camp.toml"),
        "--output", str(tmp_path / "camp_out"),
        "--beamstop-radius-px", "auto",
        "--deduplicate-frames",
        "--workers", "3",
    ]) == 0
    capsys.readouterr()
    assert campaign_calls[-1]["beamstop_radius_px"] == "auto"
    assert campaign_calls[-1]["deduplicate_identical_frames"] is True
    assert campaign_calls[-1]["max_workers"] == 3

    def legacy_campaign(config_path, output_root, *, force=False, continue_on_error=True,
                        report=None, journal_style=None, progress=None):
        del config_path, output_root, force, continue_on_error, report, journal_style, progress
        return {
            "schema_version": "wingsaxs.campaign_run.v1",
            "status": "ok",
            "exit_code": 0,
            "n_samples": 0,
            "n_success": 0,
            "n_warning": 0,
            "n_failed": 0,
            "n_skipped": 0,
            "samples": [],
        }

    monkeypatch.setattr(cli_module, "run_campaign", legacy_campaign)
    assert main([
        "campaign",
        "--config", str(tmp_path / "camp.toml"),
        "--output", str(tmp_path / "camp_out2"),
        "--beamstop-radius-px", "14",
        "--deduplicate-frames",
        "--workers", "2",
    ]) == 0
    captured = capsys.readouterr()
    assert "max_workers" in captured.err
    assert json.loads(captured.out)["status"] == "ok"
