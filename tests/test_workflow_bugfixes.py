"""Regression tests for Phase 2 workflow, CLI, service, export, reporting, and scripts bugfixes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest


def _synthetic_frame(shape: tuple[int, int] = (48, 48), *, scale: float = 1.0) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    qx = (xx - cx) * 0.02
    qy = (yy - cy) * 0.02
    r = np.hypot(qx, qy)
    theta = np.arctan2(qy, qx)
    ring = np.exp(-0.5 * ((r - 0.25) / 0.04) ** 2) * (1.0 + 0.6 * np.cos(2.0 * theta))
    return (10.0 + 100.0 * scale * ring).astype(np.float32)


def _write_minimal_poni(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "poni_version: 2",
                "Detector: Detector",
                'Detector_config: {"pixel1": 0.000172, "pixel2": 0.000172, "max_shape": [48, 48]}',
                "Distance: 1.5",
                "Poni1: 0.004128",
                "Poni2: 0.004128",
                "Rot1: 0.0",
                "Rot2: 0.0",
                "Rot3: 0.0",
                "Wavelength: 1e-10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# 1. serialization.py & pipeline.py
# ---------------------------------------------------------------------------


def test_serialization_deterministic_sets_and_dataclass_type_guard():
    from butterfly_saxs.serialization import json_safe

    @dataclass
    class _Sample:
        x: int = 1

    assert json_safe({"b", "a", "c"}) == ["a", "b", "c"]
    assert json_safe(frozenset({3, 1, 2})) == [1, 2, 3]
    # Passing a dataclass class itself (not an instance) must return repr string, not class dict.
    encoded_cls = json_safe(_Sample)
    assert isinstance(encoded_cls, str)
    assert json_safe(_Sample(4)) == {"x": 4}


def test_pipeline_jsonable_nonfinite_arrays_and_string_arrays():
    from butterfly_saxs.pipeline import _jsonable

    @dataclass
    class _Cfg:
        value: int = 2

    arr = np.array([[1.0, float("nan")], [float("inf"), -float("inf")]], dtype=float)
    expanded = _jsonable(arr, array_summary=False)
    assert expanded == [[1.0, None], [None, None]]
    json.dumps(expanded, allow_nan=False)

    str_arr = np.array(["alpha", "beta"])
    summary = _jsonable(str_arr, array_summary=True)
    assert summary["shape"] == [2]
    assert "finite_count" not in summary

    assert _jsonable({"z", "a"}) == ["a", "z"]
    assert isinstance(_jsonable(_Cfg), str)


def test_pipeline_run_project_bounded_parallel_independent(tmp_path):
    from butterfly_saxs.pipeline import _BatchFrameAnalyzer, _run_project_bounded
    from butterfly_saxs.project import ProjectConfig
    import pickle

    frames = []
    for idx in range(2):
        fpath = tmp_path / f"f{idx}.npy"
        np.save(fpath, _synthetic_frame(scale=1.0 + 0.1 * idx))
        frames.append(str(fpath))

    # Verify _BatchFrameAnalyzer is picklable for spawn-based ProcessPoolExecutor.
    analyzer = _BatchFrameAnalyzer(
        poni=None,
        full2d=False,
        destination=None,
        force=True,
        resume=False,
        stem_counts={"f0": 1, "f1": 1},
        beamstop_radius_px=None,
        uncertainty_dataset=None,
        include_physics=False,
        intensity_mode="raw",
        bright_axis_convention="vertical",
    )
    roundtripped = pickle.loads(pickle.dumps(analyzer))
    assert isinstance(roundtripped, _BatchFrameAnalyzer)

    cfg = ProjectConfig(
        input_paths=frames,
        output_dir=str(tmp_path / "out"),
        q_unit="pixel-q",
        analysis={
            "ridge_method": "radial_peak",
            "batch_mode": "independent",
            "max_workers": 2,
        },
    )
    run = _run_project_bounded(cfg, force=True)
    assert len(run.frame_results) == 2


# ---------------------------------------------------------------------------
# 2. batch.py & producer.py
# ---------------------------------------------------------------------------


def test_batch_pixel_hash_distinguishes_embedded_qmap_and_uncertainty(tmp_path):
    import h5py
    from butterfly_saxs.batch import FrameRef, compute_frame_pixel_sha256

    image = _synthetic_frame((24, 24))
    qx1 = np.linspace(-0.5, 0.5, 24)[None, :] * np.ones((24, 1))
    qy1 = np.linspace(-0.5, 0.5, 24)[:, None] * np.ones((1, 24))
    qx2 = qx1 * 1.5
    sigma1 = np.full_like(image, 1.0)
    sigma2 = np.full_like(image, 2.5)

    npz1 = tmp_path / "a.npz"
    npz2 = tmp_path / "b.npz"
    npz3 = tmp_path / "c.npz"
    np.savez(npz1, intensity=image, qx=qx1, qy=qy1, q_unit="nm^-1", sigma=sigma1)
    np.savez(npz2, intensity=image, qx=qx2, qy=qy1, q_unit="nm^-1", sigma=sigma1)
    np.savez(npz3, intensity=image, qx=qx1, qy=qy1, q_unit="nm^-1", sigma=sigma2)

    h1 = compute_frame_pixel_sha256(FrameRef(path=npz1, dataset="intensity"))
    h2 = compute_frame_pixel_sha256(FrameRef(path=npz2, dataset="intensity"))
    h3 = compute_frame_pixel_sha256(FrameRef(path=npz3, dataset="intensity"))
    assert h1 is not None and h2 is not None and h3 is not None
    assert h1 != h2, "distinct embedded qmap must not collide in dedup hash"
    assert h1 != h3, "distinct embedded uncertainty must not collide in dedup hash"

    # Also verify HDF5 auto-discovered sibling uncertainty dataset (<dataset>_sigma).
    h5_a = tmp_path / "u1.h5"
    h5_b = tmp_path / "u2.h5"
    for h5_path, sigma in ((h5_a, sigma1), (h5_b, sigma2)):
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("data", data=image)
            handle.create_dataset("data_sigma", data=sigma)
    assert compute_frame_pixel_sha256(FrameRef(path=h5_a, dataset="data")) != compute_frame_pixel_sha256(
        FrameRef(path=h5_b, dataset="data")
    )


def test_batch_dedup_does_not_reuse_checkpoint_array_omitted_stub(tmp_path):
    from butterfly_saxs import batch

    image = _synthetic_frame((16, 16))
    p0 = tmp_path / "frame_0.npy"
    p1 = tmp_path / "frame_1.npy"
    np.save(p0, image)
    np.save(p1, image)
    checkpoint = tmp_path / "checkpoint.json"

    calls: list[str] = []

    def analyze_fn(frame):
        calls.append(frame.path.name)
        return {
            "image": image.copy(),
            "parameters": {"a": 0.25},
            "metrics": {"success": True},
        }

    # Run 1: process [p0, p1] with stop after p0 so checkpoint has frame 0 with array_omitted=True.
    import threading

    stop = threading.Event()

    def analyze_stop_after_first(frame):
        stop.set()
        return analyze_fn(frame)

    batch.run_batch(
        [p0, p1],
        analyze_stop_after_first,
        checkpoint=checkpoint,
        analyzer_identity="stub-guard-v1",
        deduplicate_identical_frames=True,
        cancel_event=stop,
    )
    assert calls == ["frame_0.npy"]

    # Resume [p0, p1]: frame 0 is resumed from checkpoint (with omitted array),
    # so frame 1 MUST NOT reuse frame 0's array_omitted stub via dedup_cache!
    calls.clear()
    run2 = batch.run_batch(
        [p0, p1],
        analyze_fn,
        checkpoint=checkpoint,
        resume=True,
        analyzer_identity="stub-guard-v1",
        deduplicate_identical_frames=True,
    )
    assert run2[0].resumed is True
    assert run2[1].resumed is False
    assert calls == ["frame_1.npy"]
    assert isinstance(run2[1].result["image"], np.ndarray)


def test_batch_manifest_mixed_order_and_nan_time_sorting(tmp_path):
    from butterfly_saxs.batch import discover_frames

    for name in ("c.npy", "a.npy", "b.npy", "d.npy"):
        np.save(tmp_path / name, np.zeros((4, 4), dtype=np.float32))

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,order,time\n"
        "c.npy,2,nan\n"
        "a.npy,,5.0\n"
        "b.npy,1,1.0\n"
        "d.npy,,2.0\n",
        encoding="utf-8",
    )
    refs = discover_frames([], manifest=manifest)
    assert [ref.path.name for ref in refs[:2]] == ["b.npy", "c.npy"]
    assert set(ref.path.name for ref in refs[2:]) == {"a.npy", "d.npy"}


def test_batch_write_checkpoint_cleans_temp_on_base_exception(tmp_path, monkeypatch):
    from butterfly_saxs import batch

    target = tmp_path / "checkpoint.json"

    def interrupt_replace(src, dst):
        raise KeyboardInterrupt("simulated interrupt")

    monkeypatch.setattr(batch.os, "replace", interrupt_replace)
    with pytest.raises(KeyboardInterrupt):
        batch.write_checkpoint(target, {"version": 1, "frames": []})
    assert list(tmp_path.glob("*.tmp")) == []


def test_producer_self_pin_and_batch_hash_match():
    from butterfly_saxs import batch, producer

    batch_bytes = Path(batch.__file__).read_bytes()
    prod_bytes = Path(producer.__file__).read_bytes()
    assert hashlib.sha256(batch_bytes).hexdigest() in producer._COMPACT_CHECKPOINT_BATCH_SHA256
    assert producer._reviewed_checkpoint_builder_hash(prod_bytes) == hashlib.sha256(prod_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 3. export.py
# ---------------------------------------------------------------------------


def test_streaming_exporter_preserves_canonical_summary_columns(tmp_path):
    from butterfly_saxs.batch import BatchRunResult, FrameFitResult, FrameRef
    from butterfly_saxs.export import StreamingBatchExporter, export_batch

    frame_path = tmp_path / "f0.npy"
    np.save(frame_path, np.ones((8, 8), dtype=np.float32))
    ref = FrameRef(path=frame_path, metadata={"temperature_c": 120.0})
    result = {
        "parameters": {"a": 0.31, "b": 0.14},
        "geometry_parameters": {
            "ellipse_kind": "butterfly",
            "a": 0.31,
            "b": 0.14,
            "axis_ratio": 0.45,
            "theta_deg": 12.5,
            "q_star_from_arcs": 0.28,
            "L_from_observed_radius_nm": 22.4,
            "q_star_source": "observed_arcs",
            "Ln_from_minor_axis_nm": 44.8,
        },
        "metrics": {"success": True, "rmse": 0.02, "r2": 0.99},
    }
    record = FrameFitResult(
        frame=ref,
        status="ok",
        result=result,
    )
    run = BatchRunResult(frame_results=[record], processed_count=1, total_count=1)

    batch_dir = tmp_path / "batch_out"
    stream_dir = tmp_path / "stream_out"
    export_batch(run, batch_dir, force=True)

    exporter = StreamingBatchExporter(stream_dir, force=True)
    exporter.write(record)
    exporter.finalize(run)

    with (batch_dir / "frame_summary.csv").open(encoding="utf-8", newline="") as handle:
        batch_rows = list(csv.DictReader(handle))
    with (stream_dir / "frame_summary.csv").open(encoding="utf-8", newline="") as handle:
        stream_rows = list(csv.DictReader(handle))

    assert len(batch_rows) == 1 and len(stream_rows) == 1
    for required_col in (
        "ellipse_kind",
        "a",
        "b",
        "axis_ratio",
        "theta_deg",
        "q_star_from_arcs",
        "L_from_observed_radius_nm",
        "Ln_from_minor_axis_nm",
        "frame_metadata_json",
    ):
        assert required_col in stream_rows[0], f"missing column {required_col}"
        assert stream_rows[0][required_col] == batch_rows[0][required_col]


def test_export_write_evolution_does_not_force_switch_backend(tmp_path, monkeypatch):
    import matplotlib
    from butterfly_saxs import export

    def forbid_use(*args, **kwargs):
        raise AssertionError("matplotlib.use must not be forced inside export._write_evolution")

    monkeypatch.setattr(matplotlib, "use", forbid_use)
    out_png = tmp_path / "evolution.png"
    export._write_evolution(out_png, [])
    assert out_png.is_file() and out_png.stat().st_size > 0


def test_export_publish_staged_bundle_cleans_stage_on_base_exception(tmp_path, monkeypatch):
    from butterfly_saxs import export

    stage = tmp_path / ".stage"
    stage.mkdir()
    (stage / "a.txt").write_text("hi", encoding="utf-8")
    target = tmp_path / "bundle"
    target.mkdir()

    def interrupt_replace(src, dst):
        raise KeyboardInterrupt("boom")

    monkeypatch.setattr(export.os, "replace", interrupt_replace)
    with pytest.raises(KeyboardInterrupt):
        export._publish_staged_bundle(stage, {"a": target / "a.txt"}, force=True)
    assert not stage.exists()


# ---------------------------------------------------------------------------
# 4. service.py & Cross-Seam Contract 1 (snapshot_state / restore_state)
# ---------------------------------------------------------------------------


def test_service_is_qt_free():
    code = (
        "import sys, butterfly_saxs.service; "
        "qt = [m for m in sys.modules if m.startswith(('PySide', 'PyQt'))]; "
        "assert not qt, qt"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_service_set_poni_none_resets_qmap_and_failed_poni_keeps_cache(tmp_path):
    from butterfly_saxs import service
    from butterfly_saxs.service import ButterflyAnalysisService

    img_path = tmp_path / "frame.npy"
    np.save(img_path, _synthetic_frame((48, 48)))
    poni_path = _write_minimal_poni(tmp_path / "cal.poni")

    svc = ButterflyAnalysisService()
    svc.load_image(img_path)
    assert svc._qmap is not None and service._read(svc._qmap, ("q_unit",), None) == "pixel-q"

    svc.set_poni(poni_path)
    assert svc._qmap is not None and service._read(svc._qmap, ("q_unit",), "nm^-1") != "pixel-q"
    svc._geometry_cache[((48, 48), "sentinel")] = object()

    with pytest.raises(Exception):
        svc.set_poni(tmp_path / "missing.poni")
    assert ((48, 48), "sentinel") in svc._geometry_cache

    svc.set_poni(None)
    assert svc.poni_path is None
    assert svc._poni is None
    assert svc._qmap is not None and service._read(svc._qmap, ("q_unit",), None) == "pixel-q"


def test_service_snapshot_restore_batch_isolation_and_warm_start_candidate_fit(tmp_path):
    from butterfly_saxs.service import ButterflyAnalysisService

    img0 = tmp_path / "f0.npy"
    img1 = tmp_path / "f1.npy"
    np.save(img0, _synthetic_frame((48, 48), scale=1.0))
    np.save(img1, _synthetic_frame((48, 48), scale=1.5))

    svc = ButterflyAnalysisService()
    svc.load_image(img0)
    svc.set_analysis_settings({"ridge_method": "radial_peak"})
    snap = svc.snapshot_state()

    # Running batch() on [img0, img1] in warm_start mode exercises butterfly.candidate_fit seed
    # and must NOT clobber current loaded state on the service!
    run = svc.batch(
        payload={
            "frames": [str(img0), str(img1)],
            "mode": "warm_start",
            "analysis": {"ridge_method": "butterfly_curvature"},
        }
    )
    assert len(run["results"]) == 2
    assert svc._loaded is not None and Path(svc._loaded.source) == img0

    svc.load_image(img1)
    svc.set_analysis_settings({"ridge_method": "azimuthal_peak"})
    assert Path(svc._loaded.source) == img1

    svc.restore_state(snap)
    assert svc._loaded is not None and Path(svc._loaded.source) == img0
    assert svc._analysis_settings["ridge_method"] == "radial_peak"


def test_service_measure_geometry_observed_none_unified_schema(monkeypatch):
    from butterfly_saxs.service import ButterflyAnalysisService

    svc = ButterflyAnalysisService()
    monkeypatch.setattr(
        svc,
        "preview",
        lambda **_kw: {
            "ellipse_fit": {
                "parameters": {
                    "a": 0.3,
                    "b": 0.15,
                    "axis_ratio": 0.5,
                    "theta_deg": 12.0,
                },
                "q_unit": "pixel-q",
                "success": True,
            },
            "parameters": {},
        },
    )
    out = svc.measure_geometry(analysis_settings={"ridge_method": "radial_peak"})
    assert "geometry_parameter_units" in out
    assert out["geometry_parameters"]["theta_deg"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# 5. butterfly_figure.py Cross-Seam Contract 2 (journal_style)
# ---------------------------------------------------------------------------


def test_butterfly_figure_journal_style_contract(tmp_path):
    from butterfly_saxs.butterfly_figure import export_butterfly_figure

    img = _synthetic_frame((48, 48))
    yy, xx = np.indices((48, 48), dtype=float)
    qx = (xx - 23.5) * 0.02
    qy = (yy - 23.5) * 0.02
    result = {"points": [], "arcs": [], "diagnostics": {}, "q_window": [0.1, 0.45]}

    default_dir = tmp_path / "fig_default"
    export_butterfly_figure(
        default_dir,
        observed=img,
        qx=qx,
        qy=qy,
        result=result,
        q_unit="pixel-q",
        dpi=100,
    )
    default_settings = json.loads((default_dir / "settings.json").read_text(encoding="utf-8"))
    assert "journal_style" not in default_settings

    nature_dir = tmp_path / "fig_nature"
    export_butterfly_figure(
        nature_dir,
        observed=img,
        qx=qx,
        qy=qy,
        result=result,
        q_unit="pixel-q",
        dpi=100,
        journal_style="nature",
    )
    nature_settings = json.loads((nature_dir / "settings.json").read_text(encoding="utf-8"))
    assert nature_settings["journal_style"] == "nature"
    assert (nature_dir / "butterfly_figure.png").is_file()

    with pytest.raises(ValueError):
        export_butterfly_figure(
            tmp_path / "fig_bad",
            observed=img,
            qx=qx,
            qy=qy,
            result=result,
            q_unit="pixel-q",
            journal_style="nonexistent_journal_preset",
        )


# ---------------------------------------------------------------------------
# 6. manual_evidence.py & annotation_pack.py
# ---------------------------------------------------------------------------


def test_manual_evidence_and_annotation_pack_no_pyplot_backend_switch(tmp_path, monkeypatch):
    import matplotlib
    from butterfly_saxs import annotation_pack, manual_evidence

    def forbid_use(*args, **kwargs):
        raise AssertionError("matplotlib.use must not be forced during headless OO rendering")

    monkeypatch.setattr(matplotlib, "use", forbid_use)
    img = _synthetic_frame((32, 32))
    qx = np.linspace(-0.5, 0.5, 32)[None, :] * np.ones((32, 1))
    qy = np.linspace(-0.5, 0.5, 32)[:, None] * np.ones((1, 32))
    mask = np.ones((32, 32), dtype=bool)

    png_bytes = manual_evidence._render_png(
        img, qx, qy, mask, title="obs", q_unit="pixel-q", vmin=0.0, vmax=120.0
    )
    assert png_bytes.startswith(b"\x89PNG")

    blind_path = tmp_path / "blind.png"
    annotation_pack._write_blind_png(blind_path, img, 0.0, 120.0)
    assert blind_path.is_file() and blind_path.read_bytes().startswith(b"\x89PNG")

    digest = annotation_pack._hash_inline({"x": float("nan"), "y": [1.0, float("inf")]})
    assert isinstance(digest, str) and len(digest) == 64


# ---------------------------------------------------------------------------
# 7. cli.py
# ---------------------------------------------------------------------------


def test_cli_lazy_import_loads_zero_scientific_modules():
    code = (
        "import sys, butterfly_saxs.cli; "
        "heavy = [m for m in ('numpy', 'scipy', 'matplotlib', 'pyFAI', 'h5py', 'fabio', 'PySide6') "
        "if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_cli_range_and_slice_overrides_and_no_stray_output_dir(tmp_path):
    from butterfly_saxs import cli

    frames = []
    for idx in range(3):
        p = tmp_path / f"s{idx}.npy"
        np.save(p, _synthetic_frame((32, 32), scale=1.0 + 0.1 * idx))
        frames.append(str(p))

    proj_range = tmp_path / "proj_range.toml"
    proj_range.write_text(
        "[project]\nq_unit = \"pixel-q\"\n[inputs]\n"
        f"files = {json.dumps(frames)}\n"
        "[analysis]\n"
        'ridge_method = "radial_peak"\n'
        'frame_range = "0:3"\n',
        encoding="utf-8",
    )
    out1 = tmp_path / "out_slice_override"
    rc = cli.main(["batch", "--config", str(proj_range), "--start", "0", "--stop", "2", "--output", str(out1)])
    assert rc in (0, 1)

    proj_slice = tmp_path / "proj_slice.toml"
    proj_slice.write_text(
        "[project]\nq_unit = \"pixel-q\"\n[inputs]\n"
        f"files = {json.dumps(frames)}\n"
        "[analysis]\n"
        'ridge_method = "radial_peak"\n'
        "start = 0\nstop = 3\nstride = 1\n",
        encoding="utf-8",
    )
    out2 = tmp_path / "out_range_override"
    rc2 = cli.main(["batch", "--config", str(proj_slice), "--range", "0:2", "--output", str(out2)])
    assert rc2 in (0, 1)

    missing_out = tmp_path / "should_not_exist"
    rc_err = cli.main(
        [
            "batch",
            "--manifest",
            str(tmp_path / "does_not_exist_manifest.csv"),
            "--output",
            str(missing_out),
        ]
    )
    assert rc_err == 2
    assert not missing_out.exists()

    conflict_out = tmp_path / "conflict_should_not_exist"
    rc_conflict = cli.main(
        [
            "batch",
            frames[0],
            "--range",
            "0:1",
            "--start",
            "0",
            "--output",
            str(conflict_out),
        ]
    )
    assert rc_conflict == 2
    assert not conflict_out.exists()


# ---------------------------------------------------------------------------
# 8. project.py
# ---------------------------------------------------------------------------


def test_project_toml_escaping_quoting_and_no_metadata_leak(tmp_path):
    from butterfly_saxs.project import ProjectConfig, load_project, save_project

    cfg = ProjectConfig(
        input_paths=["a.npy"],
        output_dir="results",
        analysis={
            "ridge_method": "radial_peak",
            "note": "line1\r\nline2\t\x08end",
            "count": np.int64(7),
            "scale": np.float64(1.25),
            "bad=key": "ok",
            "中文键": "值",
        },
        export={"write_plots": False},
        metadata={"sample": "PE"},
    )
    toml_path = tmp_path / "proj.toml"
    save_project(cfg, toml_path)

    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    assert raw["analysis"]["note"] == "line1\r\nline2\t\x08end"
    assert raw["analysis"]["bad=key"] == "ok"
    assert raw["analysis"]["中文键"] == "值"
    assert raw["analysis"]["count"] == 7
    assert raw["analysis"]["scale"] == pytest.approx(1.25)

    reloaded = load_project(toml_path)
    assert "ridge_method" not in reloaded.metadata
    assert "write_plots" not in reloaded.metadata
    assert "bad=key" not in reloaded.metadata
    assert reloaded.metadata.get("sample") == "PE"


# ---------------------------------------------------------------------------
# 9. doctor.py
# ---------------------------------------------------------------------------


def test_doctor_pep440_version_and_extra_marker_filtering(tmp_path, monkeypatch):
    from butterfly_saxs import doctor

    assert doctor._version_satisfies("2.0", ">=2.0.0") is True
    assert doctor._version_satisfies("1.9.0", "~=1.4.2") is False
    assert doctor._version_satisfies("1.4.5", "~=1.4.2") is True
    assert doctor._version_satisfies("2.0.0rc1", ">=2.0.0") is False

    # Force fallback branch by masking packaging in sys.modules.
    monkeypatch.setitem(sys.modules, "packaging", None)
    monkeypatch.setitem(sys.modules, "packaging.specifiers", None)
    monkeypatch.setitem(sys.modules, "packaging.version", None)
    assert doctor._version_satisfies("2.0", ">=2.0.0") is True
    assert doctor._version_satisfies("1.9.0", "~=1.4.2") is False
    assert doctor._version_satisfies("1.4.5", "~=1.4.2") is True
    assert doctor._version_satisfies("2.0.0rc1", ">=2.0.0") is False

    # Test metadata fallback filtering of `; extra == ...` markers.
    sample_reqs = [
        "numpy>=1.24",
        'pytest>=8.0; extra == "dev"',
        'PySide6>=6.6; extra == "ui"',
    ]
    monkeypatch.setattr(doctor.importlib.metadata, "requires", lambda _dist: sample_reqs)
    core_only = doctor._declared_specifiers(tmp_path, require_ui=False)
    with_ui = doctor._declared_specifiers(tmp_path, require_ui=True)
    assert "numpy" in core_only and "pytest" not in core_only and "pyside6" not in core_only
    assert "pyside6" in with_ui and "pytest" not in with_ui


# ---------------------------------------------------------------------------
# 10. preflight_uncertainty.py & preflight_report.py
# ---------------------------------------------------------------------------


def test_preflight_uncertainty_reraises_preflight_error(tmp_path):
    import h5py
    from butterfly_saxs import preflight_uncertainty

    class _CustomPreflightError(RuntimeError):
        pass

    h5 = tmp_path / "ref.h5"
    with h5py.File(h5, "w") as handle:
        handle.create_dataset("data", data=np.ones((4, 4), dtype=np.float32))

    def fail_read_file_record(*_args, **_kwargs):
        raise _CustomPreflightError("tampered during read")

    with pytest.raises(_CustomPreflightError, match="tampered during read"):
        preflight_uncertainty.uncertainty_provenance(
            [{"header": {"UncertaintyHDF5": "ref.h5"}}],
            tmp_path,
            [],
            resolve_path=lambda pkg, decl, **_kw: Path(pkg) / str(decl),
            display_path=lambda p, _pkg: p.name,
            read_file_record=fail_read_file_record,
            json_safe=lambda x: x,
            error_type=_CustomPreflightError,
        )


def test_preflight_report_uses_package_version(tmp_path):
    import butterfly_saxs
    from butterfly_saxs.preflight import run_preflight

    image = tmp_path / "img.npy"
    np.save(image, np.arange(16, dtype=np.float32).reshape(4, 4))
    report = run_preflight(tmp_path, image_glob="img.npy", q_window=(0.0, 2.0))
    assert report["tool"]["name"] == "WingSAXS"
    assert report["tool"]["version"] == butterfly_saxs.__version__


# ---------------------------------------------------------------------------
# 11. output_lock.py, campaign.py, validation.py
# ---------------------------------------------------------------------------


def test_output_lock_does_not_misclassify_non_contention_oserror(tmp_path, monkeypatch):
    from butterfly_saxs import output_lock

    def fail_enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(output_lock, "_acquire_file_lock", fail_enospc)
    with pytest.raises(OSError) as excinfo:
        with output_lock.output_lock(tmp_path / "out", create=True):
            pass
    assert excinfo.value.errno == errno.ENOSPC
    assert "locked by another process" not in str(excinfo.value)


def test_validation_exit_code_strict_int_and_canonical_q_unit(tmp_path):
    from butterfly_saxs.preflight import run_preflight
    from butterfly_saxs.validation import ResultSchemaError, normalise_q_arrays, validate_result_schema

    qx = np.zeros((4, 4), dtype=float)
    qy = np.zeros((4, 4), dtype=float)
    q = np.zeros((4, 4), dtype=float)
    for alias in ("nm^-1", "nm^−1", "nm_inv", "1/nm", "angstrom_inv"):
        _qx, _qy, _q, norm = normalise_q_arrays(qx, qy, q, alias)
        assert norm["q_unit"] == "nm^-1"

    image = tmp_path / "img.npy"
    np.save(image, np.arange(12, dtype=np.float32).reshape(3, 4))
    valid_report = run_preflight(tmp_path, image_glob="img.npy", q_window=(0.0, 2.0))
    validate_result_schema(valid_report)

    bad_bool = json.loads(json.dumps(valid_report))
    bad_bool["status"]["exit_code"] = True
    with pytest.raises(ResultSchemaError, match="exit_code"):
        validate_result_schema(bad_bool)

    bad_float = json.loads(json.dumps(valid_report))
    bad_float["status"]["exit_code"] = 1.0
    with pytest.raises(ResultSchemaError, match="exit_code"):
        validate_result_schema(bad_float)


def test_campaign_propagates_cancel_strict_stride_and_hdf5_frame_filter(tmp_path, monkeypatch):
    import h5py
    from butterfly_saxs import campaign
    from butterfly_saxs.cancellation import AnalysisCancelled

    h5 = tmp_path / "stack.h5"
    with h5py.File(h5, "w") as handle:
        stack = np.stack([_synthetic_frame((32, 32), scale=1.0 + 0.1 * i) for i in range(3)], axis=0)
        handle.create_dataset("data", data=stack)

    # Boolean stride must be rejected.
    bad_toml = tmp_path / "bad_campaign.toml"
    bad_toml.write_text(
        "[campaign]\npreset = \"annular\"\n"
        "[[samples]]\nname = \"s1\"\n"
        f"inputs = [{json.dumps(str(h5))}]\n"
        "dataset = \"data\"\nstride = true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stride"):
        campaign.run_campaign(bad_toml, tmp_path / "camp_bad")

    # Explicit sample["frame"] must select only that single frame from multi-frame HDF5.
    good_toml = tmp_path / "good_campaign.toml"
    good_toml.write_text(
        "[campaign]\npreset = \"annular\"\n"
        "[analysis]\nridge_method = \"radial_peak\"\n"
        "[[samples]]\nname = \"s1\"\n"
        f"inputs = [{json.dumps(str(h5))}]\n"
        "dataset = \"data\"\nframe = 1\n",
        encoding="utf-8",
    )
    summary = campaign.run_campaign(good_toml, tmp_path / "camp_single", force=True, report=False)
    assert summary["samples"][0]["n_frames"] == 1

    # AnalysisCancelled raised inside sample worker must propagate, never be recorded as failed sample.
    def raise_cancelled(*_args, **_kwargs):
        raise AnalysisCancelled("user cancelled campaign")

    monkeypatch.setattr(campaign, "_run_sample_batch", raise_cancelled)
    with pytest.raises(AnalysisCancelled):
        campaign.run_campaign(good_toml, tmp_path / "camp_cancel", force=True, report=False)


# ---------------------------------------------------------------------------
# 12. report.py, comparison_report.py, publication_export.py
# ---------------------------------------------------------------------------


def test_report_prefix_matching_and_comparison_report_force_guard(tmp_path):
    from butterfly_saxs import comparison_report, report

    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / "sample_ab_frame_summary.csv").write_text("frame_index,status\n0,ok\n", encoding="utf-8")
    # Prefix "sample_a" must NOT falsely match "sample_ab_frame_summary.csv".
    with pytest.raises(FileNotFoundError):
        report.build_analysis_report(folder, prefix="sample_a")

    comp_dir = tmp_path / "comp"
    comp_dir.mkdir()
    (comp_dir / "comparison_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        comparison_report.build_comparison_report([], comp_dir, force=False)


def test_publication_export_cancel_and_caption_mapping_scene(tmp_path):
    from butterfly_saxs import publication_export
    from butterfly_saxs.publication_models import PublicationFigureSpec

    def bad_cancel():
        raise ValueError("broken cancel callback")

    with pytest.raises(ValueError, match="broken cancel callback"):
        publication_export._check_cancel(bad_cancel)

    with pytest.raises(TypeError):
        publication_export._check_cancel("not-an-event")

    scene = {
        "status": "schematic",
        "length_unit": "nm",
        "populations": [{"branch_id": 27, "period": 12.5}],
        "settings": {"period_source": "custom_harmonic"},
        "stack_ids": [0, 1],
        "vertices": [[[0.0, 0.0, 0.0]]],
        "source_identity": {"frame": "f0"},
    }
    spec = PublicationFigureSpec()
    text = publication_export._caption(scene, spec)
    assert "custom_harmonic" in text
    assert "27: 12.5 nm" in text

    csv_path = tmp_path / "params.csv"
    publication_export._write_parameter_csv(
        csv_path,
        [{"source_index": 0, "meta": {"bad": float("nan")}}],
    )
    assert "null" in csv_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 13. scripts/
# ---------------------------------------------------------------------------


def test_scripts_bugfixes(tmp_path):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # 1. capture_readme_ui must be importable without executing top-level UI capture.
    cap_mod = importlib.import_module("capture_readme_ui")
    assert callable(getattr(cap_mod, "main", None))

    # 2. audit_jac_validation_manifest handles missing "status" and Unicode q_unit.
    audit_mod = importlib.import_module("audit_jac_validation_manifest")
    roles = sorted(audit_mod.REQUIRED_DATASET_ROLES)
    units = ["nm⁻¹", "Å⁻¹"]
    datasets = [
        {
            "dataset_id": f"d{idx}",
            "role": role,
            "calibration": {"source_type": "pixel-q", "q_unit": units[idx % len(units)]},
            "raw_frames": [],
        }
        for idx, role in enumerate(roles)
    ]
    manifest = tmp_path / "jac_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "wingsaxs.jac.experimental_validation.v1",
                "datasets": datasets,
                "independent_user_sessions": [{"session_id": "u1"}, {"session_id": "u2"}],
            }
        ),
        encoding="utf-8",
    )
    audited = audit_mod.audit_manifest(manifest)
    assert audited["datasets"][0]["declared_status"] == "pending"
    assert not any("q_unit" in gap for gap in audited["datasets"][0]["metadata_gaps"])

    # 3. validate_butterfly_arcs._package_version returns None for missing packages.
    val_mod = importlib.import_module("validate_butterfly_arcs")
    assert val_mod._package_version("definitely_nonexistent_package_xyz_12345") is None

    # 4. render_p3_overviews handles non-default case counts (2 T1, 2 T2) without IndexError.
    p3_mod = importlib.import_module("render_p3_overviews")
    t1_dir = tmp_path / "t1"
    t2_dir = tmp_path / "t2"
    r0_dir = tmp_path / "r0"
    blind_dir = r0_dir / "blind_payload"
    for d in (t1_dir, t2_dir, blind_dir):
        d.mkdir(parents=True)

    qx = np.linspace(-0.5, 0.5, 16)[None, :] * np.ones((16, 1))
    qy = np.linspace(-0.5, 0.5, 16)[:, None] * np.ones((1, 16))
    q = np.hypot(qx, qy)
    img = np.ones((16, 16), dtype=float)
    mask = np.zeros((16, 16), dtype=bool)

    t1_cases = []
    for idx in range(2):
        np.savez(t1_dir / f"c{idx}.npz", intensity=img, mask=mask, qx=qx, qy=qy, q=q)
        t1_cases.append({"name": f"c{idx}", "npz": f"c{idx}.npz"})
    t1_manifest = t1_dir / "manifest.json"
    t1_manifest.write_text(json.dumps({"cases": t1_cases}), encoding="utf-8")

    t2_cases = []
    for idx in range(2):
        np.savez(
            t2_dir / f"t2_{idx}.npz",
            real_space_density=img,
            intensity_noiseless=img,
            mask=mask,
            qx=qx,
            qy=qy,
            q=q,
            projection_reference=np.array([[0.1, 0.1], [-0.1, -0.1]]),
        )
        t2_cases.append({"category": f"cat{idx}", "npz_file": f"t2_{idx}.npz"})
    t2_manifest = t2_dir / "manifest.json"
    t2_manifest.write_text(json.dumps({"cases": t2_cases}), encoding="utf-8")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    for idx in range(8):
        fig = Figure(figsize=(1, 1))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.imshow(img)
        fig.savefig(blind_dir / f"blind_{idx}.png", dpi=20)
    r0_status = r0_dir / "status.json"
    r0_status.write_text(json.dumps({"status": "ready", "input_hashes": [{"unchanged": True}]}), encoding="utf-8")

    out_dir = tmp_path / "p3_out"
    qc = p3_mod.render_overviews(t1_manifest, t2_manifest, r0_status, out_dir)
    assert qc["t1"]["case_count"] == 2
    assert qc["t2"]["case_count"] == 2
    assert (out_dir / "t1_overview.png").is_file()
    assert (out_dir / "t2_overview.png").is_file()
