from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from butterfly_saxs.batch import FrameFitResult, FrameRef
from butterfly_saxs.delivery import package_batch
from butterfly_saxs.export import export_batch
from butterfly_saxs.report import build_analysis_report


def _ridges(branch_id: int = 0) -> list[dict[str, object]]:
    """Two opposite points retained from an observed ridge trace."""
    angle = np.deg2rad(28.0 if branch_id == 0 else 116.0)
    # Keep measured ridge q distinct from the fitted ellipse prediction so the
    # report must retain both values and their signed residual.
    qx, qy = 0.40 * np.cos(angle), 0.40 * np.sin(angle)
    return [
        {"qx": float(qx), "qy": float(qy), "branch_id": branch_id, "valid": True,
         "accepted": True, "source": "butterfly_curvature", "point_id": "R1"},
        {"qx": float(-qx), "qy": float(-qy), "branch_id": branch_id, "valid": True,
         "accepted": True, "source": "butterfly_curvature", "point_id": "R2"},
    ]


def _frame_result(
    *,
    q_unit: str = "nm^-1",
    b: float = 0.2,
    candidate: bool = False,
    branch: int | None = 0,
    draw_axis_deg: float = 0.0,
) -> dict[str, object]:
    axis = np.linspace(-0.5, 0.5, 18)
    qx, qy = np.meshgrid(axis, axis)
    image = np.ones(qx.shape, dtype=float)
    if branch is not None:
        angle = np.deg2rad(28.0 if branch == 0 else 116.0)
        for sign in (-1.0, 1.0):
            cx, cy = sign * 0.40 * np.cos(angle), sign * 0.40 * np.sin(angle)
            image += 12.0 * np.exp(-((qx - cx) ** 2 + (qy - cy) ** 2) / 0.006)
    b_record = (
        {"value": None, "candidate_value": b, "status": "candidate"}
        if candidate
        else {"value": b, "status": "available"}
    )
    ratio = b / 0.42
    ratio_record = (
        {"value": None, "candidate_value": ratio, "status": "candidate"}
        if candidate
        else {"value": ratio, "status": "available"}
    )
    fitted_members = [
        {"branch_id": 0, "a": 0.42, "b": b, "axis_ratio": ratio,
         "theta_deg": 28.0, "center": [0.0, 0.0]},
        {"branch_id": 1, "a": 0.42, "b": b, "axis_ratio": ratio,
         "theta_deg": 116.0, "center": [0.0, 0.0]},
    ]
    result: dict[str, object] = {
        "image": image,
        "qmap": {"qx": qx, "qy": qy, "q_unit": q_unit},
        "valid_mask": np.ones(qx.shape, dtype=bool),
        "analysis": {"draw_axis_deg": draw_axis_deg},
        "ellipse_fit": {
            "q_unit": q_unit,
            "center": [0.0, 0.0],
            "status": "candidate" if candidate else "available",
            "ellipses": fitted_members,
            "parameters": {
                "a": {"value": 0.42, "status": "available", "unit": q_unit},
                "b": b_record,
                "theta_deg": {"value": 28.0 if branch != 1 else 116.0,
                              "status": "available", "unit": "degree"},
                "axis_ratio": ratio_record,
            },
            "quantitative_parameters": {
                "a": {"value": 0.42, "status": "available", "unit": q_unit},
                "b": b_record,
                "theta_deg": {"value": 28.0 if branch != 1 else 116.0,
                              "status": "available", "unit": "degree"},
                "axis_ratio": ratio_record,
            },
            "candidate_solutions": [],
        },
    }
    if branch is not None:
        result["ridges"] = _ridges(branch)
    return result


def _export_batch(folder: Path, *, include_failure: bool = True) -> Path:
    records = [
        FrameFitResult(
            FrameRef(folder / "nm.edf", frame_id="nm", time=0,
                     metadata={"time_unit": "s"}),
            _frame_result(q_unit="nm^-1", b=0.2),
        ),
        FrameFitResult(
            FrameRef(folder / "pixel.edf", frame_id="pixel-q", time=1,
                     metadata={"time_unit": "s"}),
            _frame_result(q_unit="pixel-q", b=0.1),
        ),
        FrameFitResult(
            FrameRef(folder / "candidate.edf", frame_id="candidate", time=2,
                     metadata={"time_unit": "s"}),
            _frame_result(q_unit="nm^-1", b=0.15, candidate=True),
        ),
        # This frame has only observed branch 1. No second branch is supplied
        # through synthetic radial peak rows or inferred from the image.
        FrameFitResult(
            FrameRef(folder / "single-branch.edf", frame_id="single-branch", time=3,
                     metadata={"time_unit": "s"}),
            _frame_result(q_unit="nm^-1", b=0.25, branch=1),
        ),
    ]
    if include_failure:
        records.append(
            FrameFitResult(
                FrameRef(folder / "failed.edf", frame_id="failed", time=4,
                         metadata={"time_unit": "s"}),
                None,
                status="failed",
                error="unreadable detector frame",
            )
        )
    export_batch(records, folder)
    return folder


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _frame_analysis(target: Path, frame: dict[str, object]) -> dict[str, object]:
    return json.loads((target / str(frame["data"]["lamellar_analysis"])).read_text(encoding="utf-8"))


def test_report_exports_ellipse_morphology_with_units_evidence_and_geometry(tmp_path: Path) -> None:
    folder = _export_batch(tmp_path / "sample")
    archive = folder / "results.npz"
    archive_before = archive.read_bytes()

    report = build_analysis_report(
        folder,
        radial_bins=4,
        angular_bins=8,
        formats=("png",),
        dpi=35,
        lamellar_settings={"layer_count": 2, "stack_count": 2},
    )
    target = Path(report["outputs"]["summary"]).parent
    frames = {str(frame["frame_id"]): frame for frame in report["frames"]}

    assert report["counts"] == {
        "frames": 5,
        "reported": 4,
        "incomplete": 1,
        "lamellar_scenes": 4,
        "lamellar_incomplete": 0,
        "warning_frames": 1,
    }
    assert report["settings"]["lamellar_settings"]["mode"] == "multi"
    assert report["settings"]["lamellar_settings"]["period_source"] == "ellipse"
    assert (folder / "results.npz").read_bytes() == archive_before

    parameters = _rows(target / "lamellar_parameters.csv")
    period_name = "Ln_from_minor_axis_nm"
    nm_period = next(row for row in parameters if row["frame_id"] == "nm" and row["parameter"] == period_name)
    assert float(nm_period["value"]) == pytest.approx(2.0 * np.pi / 0.2)
    assert nm_period["unit"] == "nm"

    candidate_period = next(row for row in parameters if row["frame_id"] == "candidate" and row["parameter"] == period_name)
    assert candidate_period["value"] == ""
    assert float(candidate_period["candidate_value"]) == pytest.approx(2.0 * np.pi / 0.15)
    assert candidate_period["status"] == "candidate"

    assert not any(row["frame_id"] == "pixel-q" and row["parameter"] == period_name and row["unit"] == "nm"
                   and (row["value"] or row["candidate_value"]) for row in parameters)

    directions = _rows(target / "lamellar_directions.csv")
    single_branch = [row for row in directions if row["frame_id"] == "single-branch"]
    assert single_branch
    assert {row["branch_id"] for row in single_branch} == {"1"}
    assert all(row["source"].startswith("ridges.") for row in single_branch)
    assert len(single_branch) == 2
    assert all(row["support_count"] == "1" for row in single_branch)
    assert not any("lobe_radial_peaks" in str(row) for row in single_branch)

    pixel_angles = [row for row in _rows(target / "lamellar_period_by_angle.csv") if row["frame_id"] == "pixel-q"]
    assert pixel_angles
    assert all(row["period_nm"] == "" for row in pixel_angles)
    assert all(row["q_unit"] == "pixel-q" for row in pixel_angles)

    nm_analysis = _frame_analysis(target, frames["nm"])
    assert nm_analysis["scene"]["draw_axis_deg"] == pytest.approx(0.0)
    assert nm_analysis["scene"]["source_identity"]["frame_id"] == "nm"
    reference_period = 2.0 * np.pi / 0.2
    nm_scene_reference = next(
        row for row in nm_analysis["scene"]["parameter_sources"]
        if row["unit"] == "nm"
        and row.get("branch_id") in (None, "")
        and row.get("value") is not None
        and float(row["value"]) == pytest.approx(reference_period)
    )
    assert nm_scene_reference["value"] == pytest.approx(reference_period)
    nm_population = next(population for population in nm_analysis["scene"]["populations"]
                         if population["branch_id"] == 0)
    nm_directional_model = next(
        row for row in nm_analysis["directional_rows"]
        if row["branch_id"] == 0
        and row["observed_support"]
        and row["angle_deg"] == pytest.approx(nm_population["angle_deg"])
    )
    assert nm_population["period"] == pytest.approx(2.0 * np.pi / nm_directional_model["q_radius"])
    assert nm_population["period"] != pytest.approx(reference_period)
    nm_directional_source = next(
        row for row in nm_analysis["scene"]["parameter_sources"]
        if row.get("branch_id") == 0
        and row["unit"] == "nm"
        and row.get("value") is not None
        and float(row["value"]) == pytest.approx(nm_population["period"])
    )
    assert nm_directional_source["value"] == pytest.approx(nm_population["period"])

    # The CSV keeps the retained ridge observation separate from the fitted
    # ellipse prediction at the same direction. The residual is observed q
    # minus model q, not a replacement measurement.
    nm_observed = next(
        row for row in directions
        if row["frame_id"] == "nm"
        and row["branch_id"] == "0"
        and row["source"].startswith("ridges.")
    )
    observed_q = float(nm_observed["q_radius"])
    model_q = float(nm_observed["model_q_radius"])
    assert observed_q == pytest.approx(np.hypot(float(nm_observed["qx"]), float(nm_observed["qy"])))
    assert model_q == pytest.approx(nm_directional_model["q_radius"])
    assert model_q == pytest.approx(0.42)
    assert observed_q == pytest.approx(0.40)
    assert float(nm_observed["residual_q"]) == pytest.approx(observed_q - model_q)
    assert float(nm_observed["residual_q"]) == pytest.approx(-0.02)
    assert float(nm_observed["model_period_nm"]) == pytest.approx(2.0 * np.pi / model_q)
    assert nm_observed["model_geometry_status"] == "available"
    assert "lamellar_geometry" in frames["nm"]["data"]
    with np.load(target / str(frames["nm"]["data"]["lamellar_geometry"]), allow_pickle=False) as geometry:
        assert geometry["centers"].shape == (2, 3)
        assert geometry["vertices"].shape == (2, 8, 3)
        # Only branch 0 has retained observed ridge support in this frame.
        assert set(geometry["branch_ids"].tolist()) == {0}

    candidate_geometry = frames["candidate"]["data"].get("lamellar_geometry")
    assert candidate_geometry
    candidate_analysis = _frame_analysis(target, frames["candidate"])
    assert candidate_analysis["scene"]["status"] == "candidate"
    candidate_reference = next(
        row for row in candidate_analysis["scene"]["parameter_sources"]
        if row["unit"] == "nm"
        and row.get("branch_id") in (None, "")
        and row.get("candidate_value") is not None
        and float(row["candidate_value"]) == pytest.approx(2.0 * np.pi / 0.15)
    )
    assert candidate_reference["value"] is None
    assert candidate_reference["candidate_value"] == pytest.approx(2.0 * np.pi / 0.15)
    candidate_population = next(population for population in candidate_analysis["scene"]["populations"]
                                if population["branch_id"] == 0)
    candidate_directional_source = next(
        row for row in candidate_analysis["scene"]["parameter_sources"]
        if row.get("branch_id") == 0
        and row["unit"] == "nm"
        and row.get("candidate_value") is not None
        and float(row["candidate_value"]) == pytest.approx(candidate_population["period"])
    )
    assert candidate_directional_source["value"] is None

    single_branch_geometry = frames["single-branch"]["data"].get("lamellar_geometry")
    assert single_branch_geometry
    with np.load(target / str(single_branch_geometry), allow_pickle=False) as geometry:
        assert geometry["centers"].shape == (2, 3)
        assert set(geometry["branch_ids"].tolist()) == {1}

    failed = frames["failed"]
    assert failed["report_status"] == "incomplete"
    assert "lamellar_geometry" not in failed["data"]
    assert "lamellar_analysis" in failed["data"]

    frame_figures = {Path(path).name for path in frames["nm"]["figures"].values()}
    assert "lamellar_structure.png" in frame_figures
    assert "lamellar_periods.png" in frame_figures
    assert (target / "index.html").is_file()

    before = {path.relative_to(target): (path.stat().st_mtime_ns, path.read_bytes())
              for path in target.rglob("*") if path.is_file()}
    reused = build_analysis_report(
        folder, resume=True, radial_bins=4, angular_bins=8, formats=("png",), dpi=35,
        lamellar_settings={"layer_count": 2, "stack_count": 2},
    )
    assert reused["operation_status"] == "reused"
    assert before == {path.relative_to(target): (path.stat().st_mtime_ns, path.read_bytes())
                     for path in target.rglob("*") if path.is_file()}

    changed = build_analysis_report(
        folder, resume=True, radial_bins=4, angular_bins=8, formats=("png",), dpi=35,
        lamellar_settings={"layer_count": 3, "stack_count": 2},
    )
    assert changed["operation_status"] == "completed"
    changed_nm = next(frame for frame in changed["frames"] if frame["frame_id"] == "nm")
    with np.load(target / str(changed_nm["data"]["lamellar_geometry"]), allow_pickle=False) as geometry:
        assert geometry["centers"].shape == (3, 3)
        assert set(geometry["branch_ids"].tolist()) == {0}


def test_report_uses_radial_peak_fallback_without_mixing_ridge_table(tmp_path: Path) -> None:
    folder = tmp_path / "radial-only"
    result = _frame_result(q_unit="nm^-1", b=0.2, branch=None)
    angle = np.deg2rad(28.0)
    qx, qy = np.meshgrid(np.linspace(-0.5, 0.5, 18), np.linspace(-0.5, 0.5, 18))
    image = np.asarray(result["image"], dtype=float)
    for sign in (-1.0, 1.0):
        cx, cy = sign * 0.40 * np.cos(angle), sign * 0.40 * np.sin(angle)
        image += 12.0 * np.exp(-((qx - cx) ** 2 + (qy - cy) ** 2) / 0.006)
    result["image"] = image
    result["lobe_radial_peaks"] = [
        {
            "branch_id": 0,
            "angle_deg": 28.0,
            "q_star": 0.40,
            "q_unit": "nm^-1",
            "method": "radial_lobe_peak",
            "valid": True,
            "accepted": True,
            "status": "available",
        }
    ]
    export_batch(
        [
            FrameFitResult(
                FrameRef(folder / "radial-only.edf", frame_id="radial-only", time=0,
                         metadata={"time_unit": "s"}),
                result,
            )
        ],
        folder,
    )

    report = build_analysis_report(
        folder,
        radial_bins=3,
        angular_bins=6,
        formats=("png",),
        dpi=30,
        lamellar_settings={"layer_count": 1, "stack_count": 1},
    )
    target = Path(report["outputs"]["summary"]).parent
    frame = report["frames"][0]
    assert frame["frame_id"] == "radial-only"
    assert frame["lamellar_scene_status"] not in {"unavailable", "stale"}
    assert frame["lamellar_report_status"] == "completed"

    analysis = _frame_analysis(target, frame)
    assert analysis["observed_directions"] == []
    assert len(analysis["observed_radial_peaks"]) == 1
    assert analysis["observed_radial_peaks"][0]["source"] == "lobe_radial_peaks"
    support_sources = analysis["scene"]["metadata"]["direction_support"]
    assert len(support_sources) == 1
    assert support_sources[0]["source"] == "lobe_radial_peaks"
    support_values: dict[int, list[bool]] = {}
    for row in analysis["directional_rows"]:
        support_values.setdefault(int(row["branch_id"]), []).append(bool(row["observed_support"]))
    support_by_branch = {branch: any(values) for branch, values in support_values.items()}
    assert support_by_branch == {0: True, 1: False}

    # A radial observation remains in its own table. It is never re-labelled
    # as an observed ridge point, while the fitted q radius stays separate.
    assert _rows(target / "lamellar_directions.csv") == []
    radial_rows = _rows(target / "lamellar_radial_peaks.csv")
    assert len(radial_rows) == 1
    radial = radial_rows[0]
    assert radial["frame_id"] == "radial-only"
    assert radial["branch_id"] == "0"
    assert radial["comparison_branch_id"] == "0"
    assert float(radial["q"]) == pytest.approx(0.40)
    assert float(radial["observed_q"]) == pytest.approx(0.40)
    assert float(radial["model_q_radius"]) == pytest.approx(0.42)
    assert float(radial["observed_minus_model_q"]) == pytest.approx(-0.02)
    assert radial["comparison_q_unit"] == "nm^-1"
    assert radial["comparison_status"] == "compared"

    geometry_path = frame["data"].get("lamellar_geometry")
    assert geometry_path
    with np.load(target / str(geometry_path), allow_pickle=False) as geometry:
        assert geometry["centers"].shape == (1, 3)
        assert set(geometry["branch_ids"].tolist()) == {0}
    frame_figures = {Path(path).name for path in frame["figures"].values()}
    assert "lamellar_structure.png" in frame_figures
    assert "lamellar_periods.png" in frame_figures
    assert "lamellar_sequence_morphology.png" in {
        Path(path).name for path in report["sequence_figures"].values()
    }
    assert (target / "index.html").is_file()


def test_report_reads_flattened_ellipse_jsonl_when_frame_details_are_absent(tmp_path: Path) -> None:
    folder = _export_batch(tmp_path / "legacy", include_failure=False)
    # Simulate an older export in an isolated temporary fixture. The native
    # NPZ, observed image and ellipse_fit.jsonl remain untouched.
    (folder / "frame_details.jsonl").unlink()

    report = build_analysis_report(
        folder,
        radial_bins=3,
        angular_bins=6,
        formats=("png",),
        dpi=30,
        lamellar_settings={"layer_count": 1, "stack_count": 1},
    )
    target = Path(report["outputs"]["summary"]).parent
    parameters = _rows(target / "lamellar_parameters.csv")
    nm_period = next(row for row in parameters if row["frame_id"] == "nm" and row["parameter"] == "Ln_from_minor_axis_nm")
    assert float(nm_period["value"]) == pytest.approx(2.0 * np.pi / 0.2)
    assert "ellipse_fit.jsonl" in report["sources"]


def test_collection_report_and_zip_include_lamellar_parameter_table(tmp_path: Path) -> None:
    first = _export_batch(tmp_path / "sample-a", include_failure=False)
    second = _export_batch(tmp_path / "sample-b", include_failure=False)
    for folder in (first, second):
        build_analysis_report(
            folder, radial_bins=3, angular_bins=6, formats=("png",), dpi=25,
            lamellar_settings={"layer_count": 1, "stack_count": 1},
        )

    collection = build_analysis_report(
        tmp_path, resume=True, radial_bins=3, angular_bins=6, formats=("png",), dpi=25,
        lamellar_settings={"layer_count": 1, "stack_count": 1},
    )
    table = tmp_path / "collection_lamellar_parameters.csv"
    assert table.is_file()
    rows = _rows(table)
    assert len(rows) >= 2
    assert {row["sample"] for row in rows} == {"sample-a", "sample-b"}
    assert Path(collection["outputs"]["collection_lamellar_parameters"]) == table

    package = package_batch(tmp_path)
    assert package["dependencies"]["issues"] == []
    assert "collection_lamellar_parameters.csv" in package["files"]
    with zipfile.ZipFile(tmp_path / "delivery.zip") as archive:
        assert "collection_lamellar_parameters.csv" in archive.namelist()
        assert archive.testzip() is None
