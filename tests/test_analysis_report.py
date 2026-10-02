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
