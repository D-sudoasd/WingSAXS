from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.batch import FrameFitResult, FrameRef
from butterfly_saxs.comparison_report import (
    COMPARISON_SCHEMA,
    KINETICS_SYNTHESIS_SCHEMA,
    build_comparison_report,
    extract_time_anchor_rows,
    synthesize_campaign_kinetics,
)
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


def test_extract_time_anchor_rows_matches_target_time_and_enforces_boundaries():
    entries = [
        {
            "sample": "02_350C_04h_2026_run16",
            "temperature_C": 350.0,
            "calibration_mode": "calibrated",
            "q_unit": "nm^-1",
            "measurements": [
                {"frame_index": 0, "frame_id": "r16_00", "source_file": "r16_00.h5", "elapsed_h": 0.5, "status": "ok", "L_star_direct_nm": 12.1, "L_star_kratky_nm": 8.0, "Q_win_sub": 1.0, "a4": 0.08},
                {"frame_index": 1, "frame_id": "r16_01", "source_file": "r16_01.h5", "elapsed_h": 2.02, "status": "ok", "L_star_direct_nm": 12.9, "L_star_kratky_nm": 8.6, "Q_win_sub": 2.9, "a4": 0.12},
                {"frame_index": 2, "frame_id": "r16_02", "source_file": "r16_02.h5", "elapsed_h": 4.0, "status": "ok", "L_star_direct_nm": 14.8, "L_star_kratky_nm": 9.9, "Q_win_sub": 4.2, "a4": 0.18},
            ],
        },
        {
            "sample": "09_400C_short_quarantined",
            "temperature_C": 400.0,
            "calibration_mode": "raw_counts",
            "q_unit": "nm^-1",
            "quarantined": True,
            "measurements": [
                {"frame_index": 0, "frame_id": "r08_00", "elapsed_h": 0.15, "status": "ok", "L_star_direct_nm": 20.0, "Q_win_sub": 5000.0, "a4": 0.4},
            ],
        },
        {
            "sample": "raw_2h_run",
            "temperature_C": 375.0,
            "calibration_mode": "raw_counts",
            "q_unit": "nm^-1",
            "measurements": [
                {"frame_index": 0, "frame_id": "raw_02h", "elapsed_h": 1.98, "status": "ok", "L_star_direct_nm": 17.0, "L_star_kratky_nm": 10.1, "Q_win_sub": 8888.0, "a4": 0.21},
            ],
        },
        {
            "sample": "pixelq_2h_run",
            "calibration_mode": "raw_counts",
            "q_unit": "pixel-q",
            "measurements": [
                {"frame_index": 0, "frame_id": "pix_02h", "elapsed_h": 2.0, "status": "ok", "q_unit": "pixel-q", "q_star": 35.0, "q_star_kratky": 42.0, "a4": 0.30},
            ],
        },
    ]

    anchors = extract_time_anchor_rows(entries, target_time_h=2.0, tolerance_h=0.5)
    by_name = {r["sample"]: r for r in anchors}
    json.dumps(anchors, allow_nan=False)

    # 4 h run selects the t=2.02 h frame (NOT the 4.0 h endpoint!) and preserves source identity.
    r16 = by_name["02_350C_04h_2026_run16"]
    assert r16["anchor_matched"] is True
    assert r16["anchor_frame_index"] == 1
    assert r16["anchor_frame_id"] == "r16_01"
    assert r16["anchor_source_file"] == "r16_01.h5"
    assert r16["anchor_time_h"] == pytest.approx(2.02)
    assert r16["L_star_direct_nm"] == pytest.approx(12.9)
    assert r16["L_star_kratky_nm"] == pytest.approx(8.6)
    assert r16["Q_win_sub"] == pytest.approx(2.9)

    # Short 0.15 h run falls outside tolerance -> unmatched, observables None!
    r08 = by_name["09_400C_short_quarantined"]
    assert r08["anchor_matched"] is False
    assert r08["anchor_status"] == "outside_tolerance"
    assert r08["L_star_direct_nm"] is None
    assert r08["Q_win_sub"] is None

    # Raw 2 h run keeps geometric L* but MUST NEVER populate calibrated Q_win_sub!
    raw2 = by_name["raw_2h_run"]
    assert raw2["anchor_matched"] is True
    assert raw2["L_star_direct_nm"] == pytest.approx(17.0)
    assert raw2["Q_win_sub"] is None
    assert raw2["Q_win"] is None

    # Pixel-q run MUST NEVER populate nm lengths!
    pix = by_name["pixelq_2h_run"]
    assert pix["anchor_matched"] is True
    assert pix["L_star_direct_nm"] is None
    assert pix["L_star_kratky_nm"] is None


def test_synthesize_campaign_kinetics_recovers_power_law_arrhenius_and_deltas():
    R_kj = 8.31446261815324e-3
    ea_q_true = 60.0  # kJ/mol
    ea_l_true = 40.0  # kJ/mol
    n_dir_true = 0.06
    n_krat_true = 0.14
    times = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 4.0, 8.0], dtype=float)
    T_ref_K = 400.0 + 273.15

    entries = []
    for temp_c, beamtime, suffix, scale_l, cal_mode in (
        (350.0, "2026", "r16", 1.0, "calibrated"),
        (375.0, "2026", "r02", 1.0, "calibrated"),
        (400.0, "2026", "r19", 1.0, "calibrated"),
        (400.0, "2025", "r10", 1.05, "calibrated"),
        (425.0, "2026", "r18", 1.0, "calibrated"),
        (450.0, "2026", "r06", 1.0, "calibrated"),
        (400.0, "2026", "raw_ctrl", 1.02, "raw_counts"),
    ):
        T_K = temp_c + 273.15
        k_q = 2.0 * np.exp(-(ea_q_true / R_kj) * (1.0 / T_K - 1.0 / T_ref_K))
        l_anchor = 22.0 * scale_l * np.exp(-(ea_l_true / R_kj) * (1.0 / T_K - 1.0 / T_ref_K))
        lk_anchor = 14.0 * scale_l * np.exp(-(ea_l_true / R_kj) * (1.0 / T_K - 1.0 / T_ref_K))

        meas = []
        for idx, th in enumerate(times):
            # First frame (t=0.02 h) is pre-incubation with A4=0.01 and outlier L*; should be gated out!
            pre_inc = idx == 0
            l_dir = 80.0 if pre_inc else float(l_anchor * (th / 2.0) ** n_dir_true)
            l_krat = 60.0 if pre_inc else float(lk_anchor * (th / 2.0) ** n_krat_true)
            q_val = float(12.0 * (1.0 - np.exp(-k_q * th)))
            if cal_mode != "calibrated":
                q_val *= 5000.0  # wild raw counts that must NEVER enter calibrated kinetics!
            meas.append({
                "frame_index": idx,
                "elapsed_h": float(th),
                "status": "ok",
                "q_unit": "nm^-1",
                "L_star_direct_nm": l_dir,
                "L_star_kratky_nm": l_krat,
                "Q_win_sub": q_val,
                "a4": 0.01 if pre_inc else 0.35,
                "wing_half_opening_deg": 40.0,
                "lobe_fwhm_deg": 22.0,
            })
        entries.append({
            "sample": f"{int(temp_c)}C_{beamtime}_{suffix}",
            "temperature_C": temp_c,
            "beamtime": beamtime,
            "calibration_mode": cal_mode,
            "q_unit": "nm^-1",
            "measurements": meas,
        })

    syn = synthesize_campaign_kinetics(entries, target_time_h=2.0, post_incubation_min_a4=0.05)
    assert syn["schema_version"] == KINETICS_SYNTHESIS_SCHEMA
    json.dumps(syn, allow_nan=False)

    # Direct vs Kratky coarsening exponents strictly separated and recovered after post-incubation filter!
    dir_map = {r["sample"]: r for r in syn["coarsening_fits"]["direct"]}
    krat_map = {r["sample"]: r for r in syn["coarsening_fits"]["kratky"]}
    assert dir_map["400C_2026_r19"]["n"] == pytest.approx(n_dir_true, abs=1e-3)
    assert krat_map["400C_2026_r19"]["n"] == pytest.approx(n_krat_true, abs=1e-3)
    assert dir_map["400C_2026_r19"]["t_onset_h"] == pytest.approx(0.05)

    # Calibrated-only precipitation kinetics exclude raw_counts sample!
    precip_map = {r["sample"]: r for r in syn["precipitation_kinetics"]}
    assert precip_map["400C_2026_raw_ctrl"]["status"] == "excluded_uncalibrated"
    assert precip_map["400C_2026_raw_ctrl"]["dQ_win_dt_per_h"] is None
    assert precip_map["400C_2026_raw_ctrl"]["inv_t50_h_inv"] is None

    # Arrhenius regressions recover Ea_kJ_mol, Ea_eV, 95% CI, and R^2, excluding raw run from Q_win fits!
    arr_inv_t50 = syn["arrhenius_fits"]["inv_t50"]
    assert arr_inv_t50["success"] is True
    assert "400C_2026_raw_ctrl" not in arr_inv_t50["samples_used"]
    assert arr_inv_t50["Ea_kJ_mol"] == pytest.approx(ea_q_true, rel=0.08)
    assert arr_inv_t50["Ea_eV"] == pytest.approx(arr_inv_t50["Ea_kJ_mol"] / 96.48533212, rel=1e-4)
    assert isinstance(arr_inv_t50["Ea_ci_kJ_mol"], list) and len(arr_inv_t50["Ea_ci_kJ_mol"]) == 2
    assert arr_inv_t50["r_squared"] > 0.99

    arr_dq = syn["arrhenius_fits"]["early_dQ_win_dt"]
    assert arr_dq["success"] is True
    assert "400C_2026_raw_ctrl" not in arr_dq["samples_used"]
    assert arr_dq["Ea_kJ_mol"] == pytest.approx(ea_q_true, rel=0.12)
    assert arr_dq["r_squared"] > 0.98

    arr_l_dir = syn["arrhenius_fits"]["L_star_direct_at_target_time"]
    assert arr_l_dir["success"] is True
    assert arr_l_dir["Ea_kJ_mol"] == pytest.approx(ea_l_true, rel=0.05)

    # Pairwise cross-beamtime and replicate deltas never mix calibrated Q_win with raw counts!
    pairs = {(p["sample_a"], p["sample_b"]): p for p in syn["pairwise_agreement"]}
    cb = pairs[("400C_2026_r19", "400C_2025_r10")]
    assert cb["pair_kind"] == "cross_beamtime"
    assert cb["comparison_basis"] == "time_anchor"
    assert cb["q_win_comparable"] is True
    assert cb["deltas"]["Q_win"]["delta"] is not None
    assert cb["deltas"]["L_star_direct_nm"]["rel_diff_pct"] == pytest.approx(100.0 * 0.05 / 1.025, abs=0.2)

    cal_vs_raw = pairs[("400C_2026_r19", "400C_2026_raw_ctrl")]
    assert cal_vs_raw["q_win_comparable"] is False
    assert cal_vs_raw["deltas"]["Q_win"]["delta"] is None
    assert cal_vs_raw["deltas"]["L_star_direct_nm"]["delta"] is not None


def test_build_comparison_report_target_time_kinetics_and_zero_cjk_tofu(tmp_path):
    entries = _sample_entries()
    entries.append({
        "sample": "05_375C_02h_转350C_1h_循环标记_2026_run04",
        "temperature_C": 375.0,
        "calibration_mode": "raw_counts",
        "q_unit": "nm^-1",
        "measurements": [
            {"frame_index": 0, "time_h": 0.1, "status": "ok", "L_star_direct_nm": 17.0, "a4": 0.19},
            {"frame_index": 1, "time_h": 2.0, "status": "ok", "L_star_direct_nm": 17.1, "a4": 0.20},
        ],
    })
    entries.append({
        "sample": "14_475C_02h_2026_run22_设定500C",
        "temperature_C": 475.0,
        "calibration_mode": "calibrated",
        "q_unit": "nm^-1",
        "measurements": [
            {"frame_index": 0, "time_h": 0.1, "status": "ok", "L_star_kratky_nm": 28.0, "Q_win_sub": 6.0, "a4": 0.30},
            {"frame_index": 1, "time_h": 2.0, "status": "ok", "L_star_kratky_nm": 38.4, "Q_win_sub": 16.0, "a4": 0.35},
        ],
    })

    out_dir = tmp_path / "cmp_v2"
    summary = build_comparison_report(
        entries,
        out_dir,
        journal_style="nature",
        formats=("svg", "png"),
        dpi=100,
        target_time_h=2.0,
    )
    assert summary["target_time_h"] == pytest.approx(2.0)
    assert (out_dir / "cross_sample_time_anchor_summary.csv").is_file()
    assert (out_dir / "kinetics_and_scaling_summary.json").is_file()
    assert summary["kinetics_summary_file"] == "kinetics_and_scaling_summary.json"

    kin_raw = (out_dir / "kinetics_and_scaling_summary.json").read_text(encoding="utf-8")
    kin_doc = json.loads(kin_raw)
    json.dumps(kin_doc, allow_nan=False)
    assert kin_doc["target_time_h"] == pytest.approx(2.0)

    # Original CJK sample names preserved losslessly in CSV tables.
    ep_names = {r["sample"] for r in _read_csv(out_dir / "cross_sample_endpoint_summary.csv")}
    assert "05_375C_02h_转350C_1h_循环标记_2026_run04" in ep_names
    assert "14_475C_02h_2026_run22_设定500C" in ep_names

    # ZERO CJK tofu characters in generated Matplotlib SVG figures!
    for svg_name in ("comparison_evolution_overview.svg", "comparison_qc_completeness.svg"):
        svg_text = (out_dir / "figures" / svg_name).read_text(encoding="utf-8")
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in svg_text), f"CJK glyph leaked into {svg_name}"

    # HTML links remain strictly relative and include new artifacts.
    html_text = (out_dir / "comparison_index.html").read_text(encoding="utf-8")
    assert str(tmp_path) not in html_text
    assert "cross_sample_time_anchor_summary.csv" in html_text
    assert "kinetics_and_scaling_summary.json" in html_text
