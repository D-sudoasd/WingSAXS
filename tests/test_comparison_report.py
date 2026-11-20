from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.batch import FrameFitResult, FrameRef
from butterfly_saxs.comparison_report import COMPARISON_SCHEMA, build_comparison_report
from butterfly_saxs.delivery import package_batch
from butterfly_saxs.export import export_batch
from butterfly_saxs.journal_style import inches_to_mm
from butterfly_saxs.report import build_analysis_report
from butterfly_saxs.report_figures import _frame_overview, render_frame_report, render_sequence_report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_entries():
    return [
        {
            "sample": "02_350C_04h_2026_run16",
            "temperature_C": 350.0,
            "calibration_mode": "calibrated",
            "intensity_unit": "cm^-1",
            "q_unit": "nm^-1",
            "measurements": [
                {"frame_index": 0, "time_h": 0.1, "status": "ok", "q_unit": "nm^-1", "L_star_direct_nm": 13.5, "q_win_invariant": 0.42, "a4": 0.05, "wing_half_opening_deg": 37.8},
                {"frame_index": 1, "time_h": 1.0, "status": "ok", "q_unit": "nm^-1", "L_star_direct_nm": 13.6, "q_win_invariant": 0.85, "a4": 0.14, "wing_half_opening_deg": 38.0},
                {"frame_index": 2, "time_h": 2.0, "status": "ok", "q_unit": "nm^-1", "L_star_direct_nm": 13.5, "q_win_invariant": 1.15, "a4": 0.20, "wing_half_opening_deg": 38.0},
            ],
        },
        {
            "sample": "12_425C_02h_2026_run18",
            "temperature_C": 425.0,
            "calibration_mode": "calibrated",
            "intensity_unit": "cm^-1",
            "q_unit": "nm^-1",
            "measurements": [
                {"frame_index": 0, "time_h": 0.1, "status": "ok", "q_unit": "nm^-1", "L_star_direct_nm": 19.6, "q_win_invariant": 1.80, "a4": 0.30, "wing_half_opening_deg": 39.2},
                {"frame_index": 1, "time_h": 1.0, "status": "failed", "q_unit": "nm^-1"},
                {"frame_index": 2, "time_h": 2.0, "status": "ok", "q_unit": "nm^-1", "q_win_invariant": 3.40, "a4": 0.62, "wing_half_opening_deg": 39.5},
            ],
            "parameters": [
                {"frame_index": 2, "parameter": "L_star_direct_nm", "value": None, "candidate_value": 21.8, "unit": "nm", "status": "candidate"},
            ],
        },
        {
            "sample": "06_375_to_350_raw_run03",
            "temperature_C": 375.0,
            "calibration_mode": "raw_counts",
            "intensity_unit": "raw_counts",
            "q_unit": "nm^-1",
            "measurements": [
                {"frame_index": 0, "time_h": 0.1, "status": "ok", "q_unit": "nm^-1", "L_star_direct_nm": 16.5, "q_win_invariant": 9500.0, "a4": 0.22, "wing_half_opening_deg": 37.9},
                {"frame_index": 1, "time_h": 1.0, "status": "ok", "q_unit": "nm^-1", "L_star_direct_nm": 16.6, "q_win_invariant": 9600.0, "a4": 0.23, "wing_half_opening_deg": 38.0},
            ],
        },
        {
            "sample": "uncalibrated_pixelq_check",
            "calibration_mode": "raw_counts",
            "intensity_unit": "raw_counts",
            "q_unit": "pixel-q",
            "measurements": [
                {"frame_index": 0, "time_h": 0.1, "status": "ok", "q_unit": "pixel-q", "q_star": 42.0, "a4": 0.18},
            ],
        },
    ]


def test_build_comparison_report_separates_calibration_estimates_candidates_and_pixel_q(tmp_path):
    summary = build_comparison_report(
        _sample_entries(),
        tmp_path / "comparison",
        journal_style="nature",
        formats=("pdf", "svg", "png"),
        dpi=120,
    )
    assert summary["schema_version"] == COMPARISON_SCHEMA
    assert summary["journal_style"] == "nature"
    assert summary["sample_count"] == 4
    assert set(summary["calibrated_samples"]) == {"02_350C_04h_2026_run16", "12_425C_02h_2026_run18"}
    assert set(summary["raw_count_samples"]) == {"06_375_to_350_raw_run03", "uncalibrated_pixelq_check"}

    out_dir = tmp_path / "comparison"
    ep_rows = {row["sample"]: row for row in _read_csv(out_dir / "cross_sample_endpoint_summary.csv")}
    assert len(ep_rows) == 4

    # Calibrated sample keeps Q_win and separates estimate vs candidate L*.
    r18 = ep_rows["12_425C_02h_2026_run18"]
    assert float(r18["L_star_late_estimate_nm"]) == pytest.approx(19.6)
    assert float(r18["L_star_late_candidate_nm"]) == pytest.approx(21.8)
    assert float(r18["Q_win_late_estimate"]) == pytest.approx(3.40)
    assert int(r18["n_failed"]) == 1

    # Raw-count sample MUST NOT populate absolute calibrated Q_win_late_estimate!
    r03 = ep_rows["06_375_to_350_raw_run03"]
    assert r03["calibration_mode"] == "raw_counts"
    assert r03["Q_win_late_estimate"] == ""
    assert float(r03["L_star_late_estimate_nm"]) == pytest.approx(16.6)

    # Pixel-q sample MUST NEVER produce L_star_nm!
    pix = ep_rows["uncalibrated_pixelq_check"]
    assert pix["L_star_late_estimate_nm"] == ""
    assert pix["L_star_late_candidate_nm"] == ""

    # Verify panel_b Source Data strictly excludes raw-count runs.
    qwin_source = _read_csv(out_dir / "figures" / "comparison_evolution_overview_source_data" / "panel_b_Q_win_calibrated.csv")
    assert qwin_source
    assert all(row["calibration_mode"] == "calibrated" for row in qwin_source)
    assert "06_375_to_350_raw_run03" not in {row["sample"] for row in qwin_source}

    # Verify figures, manifests, HTML, Markdown, and strict JSON summary.
    for fig_name in ("comparison_evolution_overview", "comparison_qc_completeness"):
        for ext in ("pdf", "svg", "png"):
            assert (out_dir / "figures" / f"{fig_name}.{ext}").is_file()
        m_doc = json.loads((out_dir / "figures" / f"{fig_name}_manifest.json").read_text(encoding="utf-8"))
        assert m_doc["size_mm"][0] == pytest.approx(183.0, abs=0.05)

    html_text = (out_dir / "comparison_index.html").read_text(encoding="utf-8")
    assert "WingSAXS Cross-Sample Comparison Report" in html_text
    assert str(tmp_path) not in html_text  # Portable relative links only!
    md_text = (out_dir / "comparison_summary.md").read_text(encoding="utf-8")
    assert "cross_sample_endpoint_summary.csv" in md_text
    json.dumps(json.loads((out_dir / "comparison_summary.json").read_text(encoding="utf-8")), allow_nan=False)


def test_report_figures_journal_style_suppresses_flat_zero_center_and_uses_nature_size(tmp_path):
    summaries = [
        {"frame_index": i, "time": float(i), "time_unit": "s", "status": "ok", "q_unit": "nm^-1", "intensity_sum": 10.0 + i, "intensity_mean": 2.0 + i}
        for i in range(3)
    ]
    radial = [[{"q_min": 0.1, "q_max": 0.2, "q_center": 0.15, "mean": 2.0 + i, "count": 4}] for i in range(3)]
    angular = [[{"angle_min_deg": -90.0, "angle_max_deg": 90.0, "angle_center_deg": 0.0, "mean": 1.0 + i, "count": 4}] for i in range(3)]
    parameters = [
        {"frame_index": 0, "parameter": "cx", "value": 0.0, "unit": "nm^-1", "status": "ok"},
        {"frame_index": 1, "parameter": "cx", "value": 0.0, "unit": "nm^-1", "status": "ok"},
        {"frame_index": 2, "parameter": "cx", "value": 0.0, "unit": "nm^-1", "status": "ok"},
        {"frame_index": 0, "parameter": "cy", "value": 0.0, "unit": "nm^-1", "status": "ok"},
        {"frame_index": 1, "parameter": "cy", "value": 0.0, "unit": "nm^-1", "status": "ok"},
        {"frame_index": 2, "parameter": "cy", "value": 0.0, "unit": "nm^-1", "status": "ok"},
        {"frame_index": 0, "parameter": "a", "value": 0.30, "unit": "nm^-1", "status": "ok"},
        {"frame_index": 1, "parameter": "a", "value": None, "candidate_value": 0.33, "unit": "nm^-1", "status": "candidate"},
        {"frame_index": 2, "parameter": "a", "value": 0.35, "unit": "nm^-1", "status": "ok"},
    ]

    # Legacy mode (journal_style=None) preserves cx/cy backward compatibility.
    legacy_paths = render_sequence_report(
        tmp_path / "legacy",
        summaries=summaries,
        radial_profiles=radial,
        angular_profiles=angular,
        parameter_rows=parameters,
        formats=("png",),
        dpi=80,
    )
    assert "sequence_parameter_cx_nm_1.png" in legacy_paths
    assert "sequence_parameter_cy_nm_1.png" in legacy_paths
    assert "sequence_parameter_a_nm_1.png" in legacy_paths

    # Journal style omits degenerate flat cx=cy=0 zero lines while retaining real parameter 'a'.
    nature_paths = render_sequence_report(
        tmp_path / "nature",
        summaries=summaries,
        radial_profiles=radial,
        angular_profiles=angular,
        parameter_rows=parameters,
        formats=("svg",),
        dpi=100,
        journal_style="nature",
    )
    assert "sequence_parameter_cx_nm_1.svg" not in nature_paths
    assert "sequence_parameter_cy_nm_1.svg" not in nature_paths
    assert "sequence_parameter_a_nm_1.svg" in nature_paths

    # Frame overview in Nature journal style uses 183 mm width and inferno colormap.
    y, x = np.indices((16, 16), dtype=float)
    qx, qy = (x - 7.5) * 0.1, (y - 7.5) * 0.1
    img = 1.0 + np.exp(-((qx / 0.3) ** 2 + (qy / 0.3) ** 2))
    meas = {"summary": {"frame_index": 0, "status": "ok", "q_unit": "nm^-1"}, "radial_rows": radial[0], "angular_rows": angular[0], "polar": {}}
    from butterfly_saxs.journal_style import get_journal_preset

    fig = _frame_overview(img, qx, qy, "nm^-1", meas, None, "Test", preset=get_journal_preset("nature"))
    assert inches_to_mm(fig.get_size_inches()[0]) == pytest.approx(183.0, abs=0.05)
    assert fig.axes[0].collections[0].cmap.name == "inferno"
    fig.clear()

    frame_paths = render_frame_report(
        tmp_path / "frame_nature",
        image=img,
        qx=qx,
        qy=qy,
        valid_mask=np.ones_like(img, dtype=bool),
        q_unit="nm^-1",
        measurement=meas,
        formats=("png", "svg"),
        dpi=90,
        journal_style="nature",
    )
    assert "frame_0000_overview.png" in frame_paths
    assert "frame_0000_overview.svg" in frame_paths


def _make_batch(folder: Path, scale: float) -> Path:
    y, x = np.indices((8, 8), dtype=float)
    qx, qy = (x - 3.5) * 0.1, (y - 3.5) * 0.1
    mask = np.ones(x.shape, dtype=bool)
    mask[:, :1] = False
    result = {
        "image": scale * (x + y + 1.0),
        "qmap": {"qx": qx, "qy": qy, "q_unit": "nm^-1"},
        "valid_mask": mask,
        "ellipse_fit": {"q_unit": "nm^-1", "parameters": {"a": 0.3 * scale, "cx": 0.0, "cy": 0.0}},
    }
    frames = [
        FrameFitResult(FrameRef("f0.edf", frame_id="f0", time=0.0, metadata={"time_unit": "h"}), result),
        FrameFitResult(FrameRef("f1.edf", frame_id="f1", time=1.0, metadata={"time_unit": "h"}), {**result, "image": 1.4 * result["image"]}),
    ]
    export_batch(frames, folder)
    return folder


def test_build_analysis_report_collection_mode_with_journal_style_and_delivery_compatibility(tmp_path):
    with pytest.raises(ValueError, match="Unknown journal preset"):
        build_analysis_report(tmp_path / "does_not_exist", journal_style="not_a_journal")

    _make_batch(tmp_path / "sample_350C", 1.0)
    _make_batch(tmp_path / "sample_425C", 1.5)

    combined = build_analysis_report(
        tmp_path,
        radial_bins=4,
        angular_bins=6,
        formats=("png", "svg"),
        dpi=60,
        journal_style="nature",
    )
    assert combined["counts"]["samples"] == 2
    assert "comparison_report" in combined
    assert Path(combined["outputs"]["comparison_index"]).is_file()
    assert Path(combined["outputs"]["comparison_endpoint_summary"]).is_file()
    assert Path(combined["outputs"]["comparison_evolution_long"]).is_file()
    assert (tmp_path / "comparison_report" / "figures" / "comparison_evolution_overview.svg").is_file()

    # Verify delivery packaging accepts all relative HTML links without any dependency issues.
    package = package_batch(tmp_path, archive=False)
    assert package["dependencies"]["issues"] == []
    json.dumps(combined, allow_nan=False)
