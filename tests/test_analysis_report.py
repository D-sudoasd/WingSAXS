from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from butterfly_saxs.batch import FrameFitResult, FrameRef
from butterfly_saxs.csv_utils import read_csv_rows
from butterfly_saxs.export import export_batch
from butterfly_saxs.report import build_analysis_report


def _batch(folder: Path, *, failed: bool = True, prefix: str = "", time_unit: str = "") -> Path:
    y, x = np.indices((6, 8), dtype=float)
    qx, qy = (x - 3.5) * 0.1, (y - 2.5) * 0.1
    mask = np.ones(x.shape, dtype=bool)
    mask[:, :2] = False
    result = {
        "image": x + y + 1,
        "qmap": {"qx": qx, "qy": qy, "q_unit": "nm^-1"},
        "valid_mask": mask,
        "ellipse_fit": {"q_unit": "nm^-1", "parameters": {"a": 0.3}, "candidate_solutions": [{"a": 0.31, "status": "candidate"}]},
        "butterfly": {"profiles": [{"point_id": "P1", "offset_q": [-1., 0., 1.], "raw_intensity": [1., 2., 1.], "fit_intensity": [1.1, 1.9, 1.1]}], "diagnostics": {"support": "partial"}},
    }
    metadata = {"time_unit": time_unit} if time_unit else {}
    frames = [FrameFitResult(FrameRef("frame1.edf", frame_id="one", time=0., metadata=metadata), result)]
    if failed:
        frames.append(FrameFitResult(FrameRef("frame2.edf", frame_id="missing", time=1., metadata=metadata), None, status="failed", error="image unreadable"))
    frames.append(FrameFitResult(FrameRef("frame3.edf", frame_id="three", time=2., metadata=metadata), {**result, "image": 2 * result["image"]}))
    export_batch(frames, folder, prefix=prefix)
    return folder


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return read_csv_rows(handle)


def test_sequence_direction_summary_uses_axial_statistics_without_inventing_axis():
    from butterfly_saxs.report import _direction_summary

    summary = _direction_summary([
        {"branch_id": 0, "angle_deg": 179.0},
        {"branch_id": 0, "angle_deg": 1.0},
        {"branch_id": 1, "angle_deg": 0.0},
        {"branch_id": 1, "angle_deg": 90.0},
    ])
    wrapped, degenerate = summary
    assert min(abs(wrapped["axial_mean_deg"]), abs(wrapped["axial_mean_deg"] - 180)) < 1e-10
    assert wrapped["angle_max_deg"] - wrapped["angle_min_deg"] == pytest.approx(2)
    assert wrapped["count"] == 2
    assert degenerate["axial_mean_deg"] is None
    assert degenerate["reason"] == "undefined_axial_mean"


def test_unknown_lamellar_setting_reports_the_typo_before_reading_batch(tmp_path):
    with pytest.raises(ValueError, match="Unknown lamellar settings: thickness_ratios"):
        build_analysis_report(tmp_path, lamellar_settings={"lamellar": {"thickness_ratios": 0.74}})


def test_report_preserves_failed_frame_and_measured_arrays(tmp_path):
    folder = _batch(tmp_path / "batch")
    archive_before = (folder / "results.npz").read_bytes()
    report = build_analysis_report(folder, radial_bins=8, angular_bins=12, formats=("png",), dpi=72)
    target = folder / "figures" / "analysis_report"
    assert report["counts"]["frames"] == 3
    assert report["counts"]["reported"] == 2
    assert report["counts"]["incomplete"] == 1
    assert report["exit_code"] == 1
    assert report["frames"][1]["frame_id"] == "missing"
    assert report["frames"][1]["report_status"] == "incomplete"
    assert (folder / "results.npz").read_bytes() == archive_before
    radial = _rows(target / "radial_profiles.csv")
    assert len(radial) == 16
    assert all(row["q_center"] and row["q_unit"] for row in radial)
    measured = _rows(target / "frame_measurements.csv")
    assert len(measured) == 3
    assert json.loads(measured[0]["pixel_counts"])["observed"] == 36
    assert measured[1]["intensity"] == ""
    catalog = _rows(target / "array_catalog.csv")
    assert any(row["array_key"] == "frame_0000__image" for row in catalog)
    assert any(row["candidate_index"] == "0" for row in _rows(target / "ellipse_candidates.csv"))
    normals = _rows(target / "normal_profiles.csv")
    assert len(normals) == 6
    assert normals[0]["point_id"] == "P1"
    assert float(normals[0]["residual"]) == pytest.approx(-0.1)
    assert (target / "index.html").is_file()
    assert list(target.rglob("*.png"))
    json.dumps(report, allow_nan=False)


def test_noop_resume_and_changed_configuration(tmp_path):
    folder = _batch(tmp_path / "batch", failed=False)
    first = build_analysis_report(folder, radial_bins=4, angular_bins=6, formats=("png",), dpi=50)
    target = Path(first["outputs"]["summary"]).parent
    before = {path.relative_to(target): (path.stat().st_mtime_ns, path.read_bytes()) for path in target.rglob("*") if path.is_file()}
    reused = build_analysis_report(folder, resume=True, radial_bins=4, angular_bins=6, formats=("png",), dpi=50)
    assert reused["operation_status"] == "reused"
    after = {path.relative_to(target): (path.stat().st_mtime_ns, path.read_bytes()) for path in target.rglob("*") if path.is_file()}
    assert before == after
    with pytest.raises(FileExistsError):
        build_analysis_report(folder)
    changed = build_analysis_report(folder, resume=True, radial_bins=5, angular_bins=6, formats=("png",), dpi=50)
    assert changed["operation_status"] == "completed"
    assert len(_rows(target / "radial_profiles.csv")) == 10


def test_parent_reports_and_prefixed_exports(tmp_path):
    from butterfly_saxs.delivery import package_batch

    _batch(tmp_path / "one", failed=False)
    _batch(tmp_path / "two", failed=False, prefix="repeat_")
    report = build_analysis_report(tmp_path, radial_bins=4, angular_bins=6, formats=("png",), dpi=40)
    assert report["counts"]["samples"] == 2
    assert report["counts"]["reported"] == 4
    assert Path(report["outputs"]["index"]).name == "analysis_reports.html"
    assert (tmp_path / "two" / "figures" / "repeat_analysis_report" / "index.html").is_file()
    measurements = _rows(tmp_path / "collection_measurements.csv")
    assert len(measurements) == 4
    assert {(row["sample"], row["export_prefix"]) for row in measurements} == {("one", ""), ("two", "repeat_")}
    assert all(row["q_unit"] == "nm⁻¹" for row in measurements)
    collection_files = [tmp_path / name for name in (
        "analysis_reports.html", "analysis_report_summary.json", "collection_parameters.csv", "collection_measurements.csv")]
    before = [(path.stat().st_mtime_ns, path.read_bytes()) for path in collection_files]
    build_analysis_report(tmp_path, resume=True, radial_bins=4, angular_bins=6, formats=("png",), dpi=40)
    assert before == [(path.stat().st_mtime_ns, path.read_bytes()) for path in collection_files]
    package = package_batch(tmp_path)
    assert package["dependencies"]["issues"] == []
    assert all(path.name in package["files"] for path in collection_files)
    with zipfile.ZipFile(tmp_path / "delivery.zip") as archive:
        assert all(path.name in archive.namelist() for path in collection_files)
        assert "analysis_reports.html" in archive.read("delivery_index.html").decode("utf-8")


@pytest.mark.parametrize("options", [{"radial_bins": 0}, {"angular_bins": True}, {"dpi": -1}, {"formats": []}, {"prefix": "../"}])
def test_invalid_settings_do_not_write(tmp_path, options):
    with pytest.raises(ValueError):
        build_analysis_report(tmp_path, **options)
    assert list(tmp_path.iterdir()) == []


def test_report_package_includes_tables_and_local_dependencies(tmp_path):
    from butterfly_saxs.delivery import package_batch

    folder = _batch(tmp_path / "batch", failed=False)
    build_analysis_report(folder, radial_bins=4, angular_bins=6, formats=("png",), dpi=40)
    package = package_batch(folder, archive=False)
    assert package["dependencies"]["issues"] == []
    assert "figures/analysis_report/frame_measurements.csv" in package["files"]
    assert "frame_details.jsonl" in package["files"]


def test_cancelled_native_export_keeps_unprocessed_positions(tmp_path):
    folder = _batch(tmp_path / "batch", failed=False)
    archive = folder / "results.npz"
    with np.load(archive, allow_pickle=False) as original:
        arrays = {key: original[key] for key in original.files}
    metadata = json.loads(str(arrays["__metadata__"].item()))
    metadata.update(complete=False, all_frames_processed=False, missing_frames=[2, 3])
    arrays["__metadata__"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(archive, **arrays)
    report = build_analysis_report(folder, radial_bins=4, angular_bins=6, formats=("png",), dpi=40)
    assert report["counts"]["frames"] == 4
    assert report["counts"]["reported"] == 2
    assert report["frames"][2]["status"] == "not_processed"
    assert report["frames"][2]["frame_id"] == ""
    assert report["exit_code"] == 1


def test_explicit_time_unit_is_retained_in_report_tables(tmp_path):
    folder = _batch(tmp_path / "batch", time_unit="ms")
    report = build_analysis_report(folder, radial_bins=4, angular_bins=6, formats=("png",), dpi=40)
    target = Path(report["outputs"]["summary"]).parent
    assert all(frame["time_unit"] == "ms" for frame in report["frames"])
    assert all(row["time_unit"] == "ms" for row in _rows(target / "frame_measurements.csv"))
    assert all(row["time_unit"] == "ms" for row in _rows(target / "radial_profiles.csv"))


def test_parameter_changes_keep_candidate_estimates_separate():
    from butterfly_saxs.report import _parameter_trends

    rows = [
        {"frame_index": 0, "parameter": "a", "unit": "nm^-1", "value": "", "candidate_value": 0.3},
        {"frame_index": 1, "parameter": "a", "unit": "nm^-1", "value": 0.4, "candidate_value": ""},
        {"frame_index": 2, "parameter": "a", "unit": "nm^-1", "value": "", "candidate_value": 0.5},
    ]
    trends = {row["value_field"]: row for row in _parameter_trends(rows)}
    assert trends["value"]["n_finite"] == 1
    assert trends["candidate_value"]["n_finite"] == 2
    assert trends["candidate_value"]["change"] == pytest.approx(0.2)


def test_lamellar_figure_failure_keeps_intensity_report_and_fit_data(tmp_path, monkeypatch):
    from butterfly_saxs import lamellar_report_figures

    folder = _batch(tmp_path / "batch", failed=False)
    records_path = folder / "frame_details.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["result"]["ellipse_fit"].update(
            a=0.3, b=0.15, theta_deg=25., center=[0., 0.],
            success=True, status="ok", reference_axis_deg=0.)
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def fail_drawing(*args, **kwargs):
        raise ValueError("scene drawing failed")

    monkeypatch.setattr(lamellar_report_figures, "render_lamellar_report_frame", fail_drawing)
    report = build_analysis_report(folder, radial_bins=4, angular_bins=6, formats=("png",), dpi=40)
    assert report["counts"]["reported"] == 2
    assert report["counts"]["incomplete"] == 0
    assert report["counts"]["lamellar_incomplete"] == 2
    assert report["exit_code"] == 1
    for frame in report["frames"]:
        assert frame["report_status"] == "completed"
        assert frame["figures"]
        assert frame["lamellar_error"] == "scene drawing failed"
        assert "lamellar_analysis" in frame["data"]


def test_report_revalidates_artifact_contents_and_producer(tmp_path, monkeypatch):
    import os
    from butterfly_saxs import report as report_module

    folder = _batch(tmp_path / "batch", failed=False)
    options = {"radial_bins": 4, "angular_bins": 6, "formats": ("png",), "dpi": 30}
    first = build_analysis_report(folder, **options)
    target = Path(first["outputs"]["summary"]).parent / "radial_profiles.csv"
    original = target.read_bytes()
    info = target.stat()
    target.write_bytes(original.replace(b"mean", b"xxxx", 1))
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    repaired = build_analysis_report(folder, resume=True, **options)
    assert repaired["operation_status"] == "completed"
    assert target.read_bytes() == original
    monkeypatch.setattr(report_module, "producer_identity", lambda **_: {"source": "new renderer"})
    regenerated = build_analysis_report(folder, resume=True, **options)
    assert regenerated["operation_status"] == "completed"
    assert regenerated["producer_identity"] == {"source": "new renderer"}


def test_source_digest_detects_rewrite_even_with_restored_size_and_mtime(tmp_path):
    import os
    from butterfly_saxs.report import _source_state

    source = tmp_path / "frame_summary.csv"
    source.write_bytes(b"a,1\n")
    verified = {}
    before = _source_state(tmp_path, "", verified)
    info = source.stat()
    source.write_bytes(b"a,2\n")
    os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert _source_state(tmp_path, "", verified) != before


@pytest.mark.parametrize("revision_available", [False, True])
def test_source_proof_reuse_requires_a_write_sensitive_revision(tmp_path, monkeypatch, revision_available):
    import os
    from butterfly_saxs import report

    source = tmp_path / "frame_summary.csv"
    source.write_bytes(b"a,1\n")
    revision = [1]
    original_digest = report.hashlib.file_digest
    reads = []

    def digest(*args):
        reads.append(args[0].name)
        return original_digest(*args)

    monkeypatch.setattr(report.hashlib, "file_digest", digest)
    monkeypatch.setattr(report, "file_version", lambda path, info: (info.st_size, info.st_mtime_ns,
                                                                 revision[0] if revision_available else None))
    verified = {}
    first = report._source_state(tmp_path, "", verified)
    assert report._source_state(tmp_path, "", verified) == first
    assert len(reads) == (1 if revision_available else 2)
    info = source.stat()
    source.write_bytes(b"a,2\n")
    os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns))
    revision[0] += 1
    assert report._source_state(tmp_path, "", verified) != first
    assert len(reads) == (2 if revision_available else 3)


def test_changed_report_status_cannot_be_promoted_by_receipt_reuse(tmp_path):
    folder = _batch(tmp_path / "batch", failed=True)
    options = {"radial_bins": 4, "angular_bins": 6, "formats": ("png",), "dpi": 30}
    first = build_analysis_report(folder, **options)
    assert first["exit_code"] == 1
    path = Path(first["outputs"]["summary"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(exit_code=0, status="ready")
    payload["counts"]["warning_frames"] = 0
    path.write_text(json.dumps(payload))
    second = build_analysis_report(folder, resume=True, **options)
    assert second["operation_status"] == "completed"
    assert second["exit_code"] == 1


def test_analysis_report_propagates_saxs_physics_and_enforces_raw_unit_honesty(tmp_path):
    y, x = np.indices((10, 12), dtype=float)
    qx, qy = (x - 5.5) * 0.1, (y - 4.5) * 0.1
    img = 3.0 + 25.0 * np.exp(-0.5 * ((np.hypot(qx, qy) - 0.35) / 0.08) ** 2)
    mask = np.ones(img.shape, dtype=bool)
    shared_phys = {
        "q_unit": "nm^-1",
        "harmonics": {
            "status": "estimated",
            "a2": 0.15,
            "profile": {"angle_deg": [0.0] * 36, "intensity": [1.0] * 36, "counts": [10] * 36},
        },
        "radial": {
            "full_azimuth": {
                "I_bg_estimate": 3.1,
                "q_centers": [0.1] * 24,
                "I_q": [4.0] * 24,
            }
        },
        "c2_bridge_fit": {
            "success": True,
            "alpha_deg": 32.5,
            "lobe_fwhm_deg": 18.0,
            "chi_sym_deg": 90.2,
            "r_squared": 0.985,
            "observed": [1.0] * 36,
            "model": [1.0] * 36,
            "flat_baseline_comparison": {"delta_bic": 14.2},
        },
        "streak": {
            "status": "estimated",
            "summary": {
                "total_valid_slices": 8,
                "linear": {"W_nm": 12.4, "mosaic_spread_deg": 3.1},
                "quadrature": {"W_nm": 13.8, "mosaic_spread_deg": 2.9},
            },
            "wings": [{"wing_angle_deg": 32.5, "slices": [{"q_center": 0.3, "fwhm_q_perp": 0.5}]}],
        },
    }
    res_cal = {
        "image": img,
        "qmap": {"qx": qx, "qy": qy, "q_unit": "nm^-1"},
        "valid_mask": mask,
        "ellipse_fit": {"q_unit": "nm^-1", "parameters": {"a": 0.35}},
        "analysis": {"intensity_mode": "calibrated"},
        "saxs_physics": {
            **shared_phys,
            "intensity_mode": "calibrated",
            "q_win_invariant": 0.42,
            "q_win_bg_subtracted": 0.31,
            "q_win_stat_stderr": 0.012,
        },
    }
    res_raw = {
        "image": img,
        "qmap": {"qx": qx, "qy": qy, "q_unit": "nm^-1"},
        "valid_mask": mask,
        "ellipse_fit": {"q_unit": "nm^-1", "parameters": {"a": 0.35}},
        "analysis": {"intensity_mode": "raw"},
        "saxs_physics": {
            **shared_phys,
            "intensity_mode": "raw",
            "q_win_invariant": None,
            "q_win_bg_subtracted": 999.0,  # report must strictly suppress in raw mode
            "q_win_stat_stderr": 999.0,
        },
    }
    folder = tmp_path / "batch_phys"
    export_batch(
        [
            FrameFitResult(FrameRef("f0.npy", frame_id="cal", time=0.0), res_cal),
            FrameFitResult(FrameRef("f1.npy", frame_id="raw", time=1.0), res_raw),
        ],
        folder,
    )
    out = build_analysis_report(folder, radial_bins=4, angular_bins=6, formats=("png",), dpi=30)
    assert out["operation_status"] == "completed"
    report_dir = Path(out["outputs"]["summary"]).parent

    rows = _rows(report_dir / "frame_measurements.csv")
    assert len(rows) == 2
    assert rows[0]["intensity_mode"] == "calibrated"
    assert float(rows[0]["I_bg_estimate"]) == pytest.approx(3.1)
    assert float(rows[0]["q_win_bg_subtracted"]) == pytest.approx(0.31)
    assert float(rows[0]["q_win_stat_stderr"]) == pytest.approx(0.012)
    assert float(rows[0]["c2_alpha_deg"]) == pytest.approx(32.5)
    assert float(rows[0]["c2_delta_bic"]) == pytest.approx(14.2)
    assert float(rows[0]["streak_W_linear_nm"]) == pytest.approx(12.4)
    assert float(rows[0]["streak_W_quad_nm"]) == pytest.approx(13.8)
    assert float(rows[0]["streak_mosaic_deg"]) == pytest.approx(3.1)
    assert int(rows[0]["streak_n_valid_slices"]) == 8

    assert rows[1]["intensity_mode"] == "raw"
    assert rows[1]["q_win_invariant"] in ("", None)
    assert rows[1]["q_win_bg_subtracted"] in ("", None)
    assert rows[1]["q_win_stat_stderr"] in ("", None)

    diag_rows = _rows(report_dir / "fit_diagnostics_long.csv")
    fields = {row["field"] for row in diag_rows}
    assert "saxs_physics.c2_bridge_fit.alpha_deg" in fields
    assert "saxs_physics.streak.summary.linear.W_nm" in fields
    assert not any(f.startswith("saxs_physics.harmonics.profile") for f in fields)
    assert not any("I_q[" in f or ".slices[" in f for f in fields)


def test_build_analysis_report_reuses_identical_crc_coords_and_handles_geometry_change(
    tmp_path, monkeypatch
):
    import shutil
    from collections import Counter

    import butterfly_saxs.report as report_mod
    import butterfly_saxs.report_measurements as rm_mod

    y, x = np.indices((10, 12), dtype=float)
    qx_a, qy_a = (x - 5.5) * 0.1, (y - 4.5) * 0.1
    mask_a = np.ones((10, 12), dtype=bool)
    mask_a[:2, :2] = False

    qx_b, qy_b = (x - 4.5) * 0.12, (y - 3.5) * 0.12
    mask_b = np.ones((10, 12), dtype=bool)
    mask_b[-2:, -2:] = False

    def make_result(idx, qx, qy, mask):
        q = np.hypot(qx, qy)
        img = 2.0 + 15.0 * np.exp(-0.5 * ((q - 0.35) / 0.1) ** 2) * (1.0 + 0.1 * idx)
        return {
            "image": img,
            "qmap": {"qx": qx, "qy": qy, "q_unit": "nm^-1"},
            "valid_mask": mask,
            "ellipse_fit": {"q_unit": "nm^-1", "parameters": {"a": 0.35}},
        }

    records = [
        FrameFitResult(FrameRef("f0.edf", frame_id="f0", time=0.0), make_result(0, qx_a, qy_a, mask_a)),
        FrameFitResult(FrameRef("f1.edf", frame_id="f1", time=1.0), make_result(1, qx_a, qy_a, mask_a)),
        FrameFitResult(FrameRef("f2.edf", frame_id="f2", time=2.0), make_result(2, qx_b, qy_b, mask_b)),
        FrameFitResult(FrameRef("f3.edf", frame_id="f3", time=3.0), make_result(3, qx_b, qy_b, mask_b)),
    ]
    cached_folder = tmp_path / "cached_batch"
    uncached_folder = tmp_path / "uncached_batch"
    export_batch(records, cached_folder)
    shutil.copytree(cached_folder, uncached_folder)

    loaded_counts: Counter[str] = Counter()
    real_np_load = report_mod.np.load

    class _ArchiveSpy:
        def __init__(self, inner):
            self._inner = inner
            self.zip = inner.zip
            self.files = inner.files

        def __getitem__(self, key):
            loaded_counts[key] += 1
            return self._inner[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

    def spied_np_load(file, *args, **kwargs):
        inner = real_np_load(file, *args, **kwargs)
        if Path(file) == cached_folder / "results.npz":
            return _ArchiveSpy(inner)
        return inner

    hypot_calls = 0
    real_hypot = report_mod.np.hypot

    class _HypotWrappedNp:
        load = staticmethod(spied_np_load)

        def __getattr__(self, name):
            if name == "hypot":
                def wrapped(*args, **kwargs):
                    nonlocal hypot_calls
                    hypot_calls += 1
                    return real_hypot(*args, **kwargs)

                return wrapped
            return getattr(np, name)

    seen_flags: list[tuple[bool, bool, bool, bool]] = []
    real_measure_frame = rm_mod.measure_frame

    def wrapped_measure_frame(image, qx, qy, **kwargs):
        valid_mask = kwargs.get("valid_mask")
        q_mag = kwargs.get("q_mag")
        assert q_mag is not None
        seen_flags.append((
            qx.flags.writeable,
            qy.flags.writeable,
            bool(valid_mask.flags.writeable) if valid_mask is not None else False,
            q_mag.flags.writeable,
        ))
        return real_measure_frame(image, qx, qy, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(report_mod, "np", _HypotWrappedNp())
        patch.setattr(rm_mod, "measure_frame", wrapped_measure_frame)
        cached_rep = build_analysis_report(cached_folder, radial_bins=6, angular_bins=8, formats=("png",), dpi=40)

    assert hypot_calls == 2
    assert seen_flags == [(False, False, False, False)] * 4
    assert loaded_counts["frame_0000__qmap__qx"] == 1
    assert loaded_counts["frame_0001__qmap__qx"] == 0
    assert loaded_counts["frame_0002__qmap__qx"] == 1
    assert loaded_counts["frame_0003__qmap__qx"] == 0
    assert loaded_counts["frame_0000__valid_mask"] == 1
    assert loaded_counts["frame_0001__valid_mask"] == 0
    assert loaded_counts["frame_0002__valid_mask"] == 1
    assert loaded_counts["frame_0003__valid_mask"] == 0

    with monkeypatch.context() as patch:
        patch.setattr(report_mod, "_zip_member_signature", lambda *_a, **_kw: None)
        uncached_rep = build_analysis_report(uncached_folder, radial_bins=6, angular_bins=8, formats=("png",), dpi=40)

    cached_dir = Path(cached_rep["outputs"]["summary"]).parent
    uncached_dir = Path(uncached_rep["outputs"]["summary"]).parent
    for fname in ("frame_measurements.csv", "radial_profiles.csv"):
        c_rows = _rows(cached_dir / fname)
        u_rows = _rows(uncached_dir / fname)
        assert len(c_rows) == len(u_rows)
        for c_row, u_row in zip(c_rows, u_rows, strict=True):
            assert list(c_row) == list(u_row)
            for key in c_row:
                cv, uv = c_row[key], u_row[key]
                try:
                    cf, uf = float(cv), float(uv)
                except (TypeError, ValueError):
                    assert cv == uv
                else:
                    assert abs(cf - uf) <= 1e-12

