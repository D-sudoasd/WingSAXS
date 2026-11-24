"""Structural contracts for workflow, CLI, preflight, delivery, export, and producer hashing."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from matplotlib.figure import Figure
import numpy as np
import pytest

from butterfly_saxs import (
    _export_schema,
    cli,
    delivery,
    delivery_bindings,
    delivery_dependencies,
    delivery_pairs,
    export,
    file_identity,
    figure_support,
    path_utils,
    preflight,
    preflight_artifacts,
    preflight_checks,
    preflight_context,
    preflight_geometry,
    preflight_hash,
    preflight_io,
    preflight_report,
    preflight_uncertainty,
    producer,
    serialization,
)
from butterfly_saxs.batch import run_batch


def test_cli_import_and_describe_stay_lazy_and_lightweight() -> None:
    src_root = Path(cli.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root)
    code = """
import json
import sys
import butterfly_saxs.cli as cli

heavy = ('numpy', 'scipy', 'matplotlib', 'pyFAI', 'h5py', 'fabio', 'PySide6')
eager_handlers = (
    'butterfly_saxs._cli.single',
    'butterfly_saxs._cli.batch_cmd',
    'butterfly_saxs._cli.delivery_cmd',
    'butterfly_saxs._cli.reporting_cmd',
)
loaded_on_import = [m for m in (*heavy, *eager_handlers) if m in sys.modules]
assert not loaded_on_import, f"unexpected modules on import: {loaded_on_import}"

rc = cli.main(['describe', 'batch'])
assert rc == 0
loaded_after_describe = [m for m in heavy if m in sys.modules]
assert not loaded_after_describe, f"heavy modules after describe: {loaded_after_describe}"
unneeded_handlers = [
    m for m in (
        'butterfly_saxs._cli.batch_cmd',
        'butterfly_saxs._cli.delivery_cmd',
        'butterfly_saxs._cli.reporting_cmd',
    )
    if m in sys.modules
]
assert not unneeded_handlers, f"unrelated CLI handlers imported: {unneeded_handlers}"
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["tool"]["name"] == "WingSAXS"
    assert any(item["name"] == "batch" for item in payload["commands"])


def test_cli_monkeypatch_seams_propagate_to_subpackage_handlers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_inspect(source: object, **kwargs: object) -> dict[str, object]:
        calls["inspect"] = (source, kwargs)
        return {"shape": [8, 8], "q_unit": "nm^-1"}

    def fake_report(output_dir: object, **kwargs: object) -> dict[str, object]:
        calls["report"] = (output_dir, kwargs)
        return {"schema_version": "wingsaxs.analysis_report.v1", "exit_code": 0}

    monkeypatch.setattr(cli, "inspect_frame", fake_inspect)
    monkeypatch.setattr(cli, "_build_analysis_report", fake_report)

    sample = tmp_path / "frame.npy"
    sample.write_bytes(b"placeholder")
    assert cli.main(["inspect", str(sample)]) == 0
    inspect_out = json.loads(capsys.readouterr().out)
    assert inspect_out["shape"] == [8, 8]
    assert inspect_out["q_unit"] == "nm^-1"
    assert calls["inspect"][0] == str(sample)  # type: ignore[index]

    assert cli.main(["report", str(tmp_path), "--dpi", "200"]) == 0
    report_out = json.loads(capsys.readouterr().out)
    assert report_out["exit_code"] == 0
    assert calls["report"][0] == str(tmp_path)  # type: ignore[index]
    assert calls["report"][1]["dpi"] == 200  # type: ignore[index]


def test_recursive_producer_and_preflight_hashing_classify_subpackages(
    tmp_path: Path,
) -> None:
    # Build a synthetic package tree mirroring calculation vs non-calculation layout.
    (tmp_path / "pipeline.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "_primitives.py").write_text("P = 1\n", encoding="utf-8")
    (tmp_path / "_observables_ridges.py").write_text("R = 1\n", encoding="utf-8")
    (tmp_path / "physics").mkdir()
    (tmp_path / "physics" / "metrics.py").write_text("M = 1\n", encoding="utf-8")
    (tmp_path / "tracing").mkdir()
    (tmp_path / "tracing" / "arcs.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "observables").mkdir()
    (tmp_path / "observables" / "core.py").write_text("C = 1\n", encoding="utf-8")

    # Non-calculation subpackages and top-level presentation/export modules.
    for pkg in ("ui", "lamellar", "preflight", "delivery", "_cli"):
        (tmp_path / pkg).mkdir()
        (tmp_path / pkg / "mod.py").write_text("IGNORED = 1\n", encoding="utf-8")
    (tmp_path / "export.py").write_text("E = 1\n", encoding="utf-8")
    (tmp_path / "_export_schema.py").write_text("S = 1\n", encoding="utf-8")
    (tmp_path / "report_figures.py").write_text("F = 1\n", encoding="utf-8")

    calc_files = dict(producer.iter_calculation_source_files(tmp_path, presentation=False))
    assert set(calc_files) == {
        "_observables_ridges.py",
        "_primitives.py",
        "observables/core.py",
        "physics/metrics.py",
        "pipeline.py",
        "tracing/arcs.py",
    }

    pres_files = dict(producer.iter_calculation_source_files(tmp_path, presentation=True))
    assert "lamellar/mod.py" in pres_files
    assert "report_figures.py" in pres_files
    assert "ui/mod.py" not in pres_files
    assert "preflight/mod.py" not in pres_files
    assert "delivery/mod.py" not in pres_files
    assert "_cli/mod.py" not in pres_files

    tree_before = preflight_hash.source_tree_sha256(tmp_path)
    (tmp_path / "_cli" / "mod.py").write_text("IGNORED = 2\n", encoding="utf-8")
    (tmp_path / "ui" / "mod.py").write_text("IGNORED = 2\n", encoding="utf-8")
    assert preflight_hash.source_tree_sha256(tmp_path) == tree_before

    (tmp_path / "physics" / "metrics.py").write_text("M = 2\n", encoding="utf-8")
    assert preflight_hash.source_tree_sha256(tmp_path) != tree_before


def test_producer_self_pin_and_predecessor_checkpoint_compatibility() -> None:
    assert producer._REVIEWED_CHECKPOINT_BUILDER_SHA256 is not None
    current = producer.producer_identity()
    assert current["sources"]["producer.py"] == producer._REVIEWED_CHECKPOINT_BUILDER_SHA256
    assert "export.py" not in current["sources"]
    assert "_export_schema.py" not in current["sources"]

    exp_id = producer.exporter_identity()
    assert "export.py" in exp_id["sources"]
    assert "_export_schema.py" in exp_id["sources"]

    # Pre-refactor checkpoint identities must remain compatible with the reviewed builder.
    old_batch, old_producer = next(iter(producer._PREVIOUS_COMPACT_CHECKPOINT_SOURCES))
    previous = {
        "sources": {
            **current["sources"],
            "batch.py": old_batch,
            "producer.py": old_producer,
        },
        "versions": dict(current["versions"]),
    }
    assert producer.calculation_producers_compatible(previous, current) is True


def test_preflight_and_delivery_subpackage_shim_identity_and_monkeypatching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from butterfly_saxs.delivery import (
        bindings as d_bindings,
        dependencies as d_dependencies,
        pairs as d_pairs,
    )
    from butterfly_saxs.preflight import (
        artifacts as p_artifacts,
        checks as p_checks,
        context as p_context,
        geometry as p_geometry,
        hash as p_hash,
        io as p_io,
        report as p_report,
        uncertainty as p_uncertainty,
    )

    assert preflight_artifacts is p_artifacts
    assert preflight_checks is p_checks
    assert preflight_context is p_context
    assert preflight_geometry is p_geometry
    assert preflight_hash is p_hash
    assert preflight_io is p_io
    assert preflight_report is p_report
    assert preflight_uncertainty is p_uncertainty

    assert delivery_bindings is d_bindings
    assert delivery_dependencies is d_dependencies
    assert delivery_pairs is d_pairs
    assert callable(preflight.run_preflight)
    assert callable(delivery.package_batch)

    # Verify monkeypatching preflight.load_image intercepts preflight execution.
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    img_path = pkg / "frame_0001.npy"
    np.save(img_path, np.ones((8, 8), dtype=np.float32))

    sentinel_calls: list[Path] = []
    real_loader = preflight.load_image

    def spy_loader(path: Path, **kwargs: object) -> object:
        sentinel_calls.append(Path(path))
        return real_loader(path, **kwargs)

    monkeypatch.setattr(preflight, "load_image", spy_loader)
    report = preflight.run_preflight(pkg)
    assert sentinel_calls == [img_path.resolve()]
    assert report["result_type"] == "preflight"


def test_shared_export_schema_parity_between_batch_and_streaming(tmp_path: Path) -> None:
    assert export._CANONICAL_FRAME_SUMMARY_COLUMNS == _export_schema.CANONICAL_FRAME_SUMMARY_COLUMNS
    assert export._parameters is _export_schema._parameters

    frame_path = tmp_path / "sample.npy"
    np.save(frame_path, np.ones((8, 8), dtype=np.float32))
    manifest = [{"path": str(frame_path), "time": 12.5, "frame_id": "f0"}]
    result_payload = {
        "q_unit": "nm^-1",
        "parameters": {
            "a": {"value": 0.42, "stderr": 0.01, "unit": "nm^-1"},
        },
        "ellipse_fit": {
            "status": "ok",
            "quality_status": "PASS",
            "a": 0.42,
            "b": 0.21,
            "axis_ratio": 0.5,
            "theta_deg": 35.0,
        },
        "ridges": [
            {"angle_deg": 35.0, "q": 0.30, "qx": 0.24, "qy": 0.17, "valid": True, "snr": 8.0}
        ],
        "observables": {
            "lobes": [{"lobe": 0, "angle_deg": 35.0, "valid": True, "snr": 9.0}]
        },
    }
    run_mem = run_batch([], lambda *_args, **_kwargs: result_payload, manifest=manifest)
    out_batch = tmp_path / "out_batch"
    export.export_batch(run_mem, out_batch)

    out_stream = tmp_path / "out_stream"
    streamer = export.StreamingBatchExporter(out_stream)
    run_stream = run_batch(
        [],
        lambda *_args, **_kwargs: result_payload,
        manifest=manifest,
        result_sink=streamer.write,
        retain_results=False,
    )
    streamer.finalize(run_stream)

    expected_stream_schemas = {
        "frame_summary.csv": _export_schema.CANONICAL_FRAME_SUMMARY_COLUMNS,
        "parameters_long.csv": _export_schema.STREAMING_PARAMETER_COLUMNS,
        "ridge_points.csv": _export_schema.CANONICAL_RIDGE_COLUMNS,
        "lobe_measurements.csv": _export_schema.CANONICAL_LOBE_COLUMNS,
    }
    for name, stream_cols in expected_stream_schemas.items():
        with (out_batch / name).open(encoding="utf-8", newline="") as hb:
            reader_b = csv.DictReader(hb)
            assert reader_b.fieldnames is not None
            batch_rows = list(reader_b)
        with (out_stream / name).open(encoding="utf-8", newline="") as hs:
            reader_s = csv.DictReader(hs)
            assert tuple(reader_s.fieldnames or ()) == tuple(stream_cols)
            stream_rows = list(reader_s)
        assert len(batch_rows) == len(stream_rows) == 1
        shared_cols = [
            col for col in stream_cols if col in batch_rows[0] and col != "elapsed_s"
        ]
        assert len(shared_cols) >= 10, f"too few shared columns for {name}"
        for col in shared_cols:
            assert stream_rows[0][col] == batch_rows[0][col], f"{name} mismatch on {col}"


def test_shared_infrastructure_helpers_atomic_and_strict(tmp_path: Path) -> None:
    target_json = tmp_path / "strict.json"
    serialization.write_json_atomic(
        target_json,
        {"finite": 1.5, "non_finite": float("nan")},
    )
    parsed = json.loads(target_json.read_text(encoding="utf-8"))
    assert parsed == {"finite": 1.5, "non_finite": None}
    assert file_identity.sha256_file(target_json) == hashlib.sha256(
        target_json.read_bytes()
    ).hexdigest()

    # Staging cleanup on BaseException (e.g. KeyboardInterrupt) leaves no directory behind.
    aborted_dir = tmp_path / "aborted_bundle"
    with pytest.raises(KeyboardInterrupt):
        with path_utils.atomic_stage_dir(aborted_dir) as stage:
            (stage / "partial.txt").write_text("temp\n", encoding="utf-8")
            raise KeyboardInterrupt("simulated interrupt")
    assert not aborted_dir.exists()
    assert not list(tmp_path.glob(".aborted_bundle*"))

    # Overwrite rollback preserves previous published bundle if stage publish fails.
    live_dir = tmp_path / "live_bundle"
    with path_utils.atomic_stage_dir(live_dir) as stage:
        (stage / "marker.txt").write_text("v1\n", encoding="utf-8")
    assert (live_dir / "marker.txt").read_text(encoding="utf-8") == "v1\n"
    with path_utils.atomic_stage_dir(live_dir, overwrite=True) as stage:
        (stage / "marker.txt").write_text("v2\n", encoding="utf-8")
    assert (live_dir / "marker.txt").read_text(encoding="utf-8") == "v2\n"

    # save_agg_figure renders without importing matplotlib.pyplot.
    fig = Figure(figsize=(2.0, 2.0), dpi=72)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1])
    png_path = tmp_path / "plot.png"
    figure_support.save_agg_figure(fig, png_path, clear=True, format="png", dpi=72)
    assert png_path.is_file() and png_path.stat().st_size > 0

    # atomic_stage_dir(overwrite=True) refuses to clobber non-directory targets or strand backups.
    existing_file = tmp_path / "not_a_directory"
    existing_file.write_text("keep-me\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        with path_utils.atomic_stage_dir(existing_file, overwrite=True) as stage:
            (stage / "marker.txt").write_text("v1\n", encoding="utf-8")
    assert existing_file.is_file()
    assert existing_file.read_text(encoding="utf-8") == "keep-me\n"
    assert not list(tmp_path.glob(".not_a_directory*"))


def test_q_unit_and_enum_serialization_contracts() -> None:
    from enum import Enum, IntEnum
    from butterfly_saxs import manual_evidence, p4_quality

    for phys in ("nm^-1", "nm⁻¹", "1/nm", "A^-1", "Å⁻¹", "angstrom-1", "nm⁻1", b"nm^-1"):
        assert p4_quality._physical_q_unit(phys) is True, f"expected physical for {phys!r}"
    for non_phys in ("pixel-q", "pixel", "unknown", "", None):
        assert p4_quality._physical_q_unit(non_phys) is False, f"expected non-physical for {non_phys!r}"

    class StatusEnum(Enum):
        READY = "ready"
        WEIGHTED = (1, float("nan"))

    class CodeEnum(IntEnum):
        OK = 2

    assert serialization.json_safe(StatusEnum.READY) == "ready"
    assert serialization.json_safe(StatusEnum.WEIGHTED) == [1, None]
    assert serialization.json_safe(CodeEnum.OK) == 2
    assert manual_evidence._json_safe({"status": StatusEnum.READY, "code": CodeEnum.OK}) == {
        "status": "ready",
        "code": 2,
    }


def test_scripts_readme_and_validate_arcs_recursive_hashes() -> None:
    repo_root = Path(cli.__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    readme = (scripts_dir / "README.md").read_text(encoding="utf-8")
    for script in sorted(scripts_dir.iterdir()):
        if script.is_file() and script.name != "README.md":
            assert script.name in readme, f"{script.name} missing from scripts/README.md"

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_butterfly_arcs",
        scripts_dir / "validate_butterfly_arcs.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hashes = module.code_hashes()
    assert "src/butterfly_saxs/cli.py" in hashes
    assert "src/butterfly_saxs/_cli/common.py" in hashes
    assert "src/butterfly_saxs/preflight/runner.py" in hashes
    assert "src/butterfly_saxs/delivery/package.py" in hashes
