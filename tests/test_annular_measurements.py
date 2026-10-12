import numpy as np

from butterfly_saxs.annular_measurements import summarize_annular_measurements


def _trace():
    return {
        "q_unit": "pixel-q",
        "branch_pairing": {"reference_axis_deg": 15.0},
        "annular_peaks": {"annuli": [
            {"annulus_index": index, "q_center": 1.0 + index, "q_min": 0.5 + index, "q_max": 1.5 + index}
            for index in range(3)
        ]},
        "points": [
            {"point_id": f"{index}:{quadrant}", "annulus_index": index,
             "quadrant": quadrant, "q_annulus": 1.0 + index, "chi_deg": angle,
             "accepted": True, "intensity": 4.0, "prominence": prominence,
             "azimuthal_fwhm_deg": 20.0, "coverage": 0.9, "snr": 6.0}
            for index, quadrant, angle, prominence in (
                (0, "QI", 359.0, 3.0), (1, "QI", 1.0, 3.0),
                (0, "QIII", 181.0, 1.0), (1, "QIII", 181.0, 1.0),
            )
        ],
    }


def test_direct_measurements_retain_gaps_and_pair_asymmetry_without_ellipse():
    trace = _trace()
    trace["candidate_fit"] = {"success": False}
    summary = summarize_annular_measurements(trace)
    assert len(summary["rows"]) == 12
    assert summary["ellipse_required"] is False
    assert summary["q_unit"] == "pixel-q"
    assert summary["q_annulus_is_radial_peak"] is False
    assert "Ln_nm" not in summary and "q_star" not in summary
    missing = next(row for row in summary["rows"] if row["annulus_index"] == 2 and row["quadrant"] == "QI")
    assert missing["status"] == "missing" and missing["chi_deg"] is None
    pair = summary["opposite_pairs"][0]
    assert pair["opposition_error_deg"] == 2.0
    assert pair["prominence_asymmetry"] == 0.5
    assert pair["q_difference_independent"] is False
    trajectory = summary["trajectories"][0]
    assert trajectory["chi_median_deg"] == 0.0
    assert trajectory["chi_span_deg"] == 2.0
    assert trajectory["n_missing_annuli"] == 1
    assert trajectory["trajectory_evidence_status"] == "local_candidates_only"
    assert trajectory["n_supported_arc_points"] == 0


def test_rejected_peaks_do_not_fill_missing_quadrants():
    trace = _trace()
    trace["points"][2]["accepted"] = False
    summary = summarize_annular_measurements(trace)
    assert summary["opposite_pairs"][0]["status"] == "missing_counterpart"
    assert summary["opposite_pairs"][0]["prominence_asymmetry"] is None


def test_widths_at_gaps_stay_unavailable_and_overlapping_spans_are_labelled():
    trace = _trace()
    trace["points"][0]["azimuthal_fwhm_deg"] = np.nan
    trace["points"][1]["azimuthal_fwhm_deg"] = 0.0
    trace["points"][2]["azimuthal_fwhm_deg"] = 200.0
    trace["points"].append({**trace["points"][2], "quadrant": "QII", "chi_deg": 155.0})
    rows = summarize_annular_measurements(trace)["rows"]
    unavailable = [row for row in rows if row["quadrant"] == "QI" and row["status"] == "observed"]
    assert all(row["azimuthal_fwhm_deg"] is None for row in unavailable)
    overlapping = next(row for row in rows if row["quadrant"] == "QIII" and row["annulus_index"] == 0)
    assert overlapping["azimuthal_fwhm_deg"] == 200.0
    assert "angular_half_height_spans_overlap" in overlapping["flags"]
