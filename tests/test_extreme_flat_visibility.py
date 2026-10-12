import numpy as np
import pytest

from butterfly_saxs.butterfly_quality import (
    classify_ellipse_publication,
    unpublished_ellipse_shape,
)
from butterfly_saxs.candidate_geometry import _candidate_curves


@pytest.mark.parametrize(
    "flag",
    [
        "axis_ratio_at_bound",
        "annular_outer_window_truncated",
        "major_axis_exceeds_observed_extent",
        "minor_axis_not_resolved_against_localization",
    ],
)
def test_finite_extreme_flat_curve_remains_visible_with_limitations(flag):
    candidate = {
        "success": True,
        "status": "ok",
        "a": 1.28,
        "b": 0.01536,
        "axis_ratio": 0.012,
        "theta_deg": 40.0,
        "center_qx": 0.0,
        "center_qy": 0.0,
        "reference_axis_deg": 0.0,
        "flags": [flag],
    }
    result = {"candidate_fit": candidate, "quality": {"status": "WARN", "flags": [flag]}}

    state, curves, metadata = _candidate_curves(result)

    assert state == "candidate"
    assert len(curves) == 2
    assert flag in metadata["flags"]
    assert "constrained ellipse candidate" in metadata["candidate_label"]
    assert all(np.all(np.isfinite(curve["points"])) for curve in curves)
    assert not unpublished_ellipse_shape(quality_status="WARN", axis_ratio=0.012, flags=[flag])


@pytest.mark.parametrize("ratio", [0.0, -0.012, 1.1])
def test_invalid_axis_ratio_is_not_a_valid_flat_ellipse(ratio):
    assert classify_ellipse_publication(quality_status="WARN", axis_ratio=ratio) == "fail"


@pytest.mark.parametrize(
    ("quality_flags", "bound_status", "expected_flag"),
    [
        ([], {"a": True}, None),
        ("minor_axis_not_resolved_against_localization", {}, "minor_axis_not_resolved_against_localization"),
    ],
)
def test_measurement_table_marks_limited_shape_and_derived_values_as_candidates(
    qtbot, quality_flags, bound_status, expected_flag
):
    from butterfly_saxs.ui.main_window import MainWindow

    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    window._update_measurements(
        {
            "ellipse_fit": {
                "success": True,
                "a": 1.28,
                "b": 0.01536,
                "axis_ratio": 0.012,
                "theta_deg": 40.0,
                "Ln_from_minor_axis_nm": 83.0,
                "Lz_from_draw_axis_nm": 41.0,
                "L_from_major_axis_nm": 27.0,
                "bound_status": bound_status,
                "quality": {"status": "WARN", "flags": quality_flags},
            },
            "observables": {},
        }
    )

    assert window.ellipse_table.item(0, 1).text() == "≈1.28"
    assert window.ellipse_table.item(2, 1).text() == "≈0.012"
    assert window.ellipse_table.item(5, 1).text() == "≈83"
    assert window.ellipse_table.item(6, 1).text() == "≈41"
    assert window.ellipse_table.item(7, 1).text() == "≈27"
    candidate_tip = window.ellipse_table.item(2, 1).toolTip()
    assert "constrained candidate" in candidate_tip
    if expected_flag:
        assert expected_flag in candidate_tip
    window.close()


def test_unsuccessful_solver_values_are_diagnostic_only(qtbot):
    from butterfly_saxs.ui.main_window import MainWindow

    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    window._update_measurements(
        {
            "quality": {"status": "FAIL", "flags": "poor_match"},
            "ellipse_fit": {
                "success": False,
                "status": "failed",
                "a": 0.27,
                "b": 0.095,
                "axis_ratio": 0.35,
                "theta_deg": 32.0,
                "Ln_from_minor_axis_nm": 125.0,
            },
            "observables": {},
        }
    )

    assert window.ellipse_table.item(0, 1).text() == "≈0.27"
    assert window.ellipse_table.item(5, 1).text() == "≈125"
    assert "diagnostics only" in window.ellipse_table.item(0, 1).toolTip()
    window.close()
