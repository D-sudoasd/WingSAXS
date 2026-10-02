"""Numerical contracts for ellipse-derived apparent lamellar measurements."""

from __future__ import annotations

import json
import math

import pytest

from butterfly_saxs.lamellar_analysis import analyze_lamellar_morphology


def _ellipse_source(
    *,
    q_unit: str = "nm^-1",
    a: float = 1.0,
    b: float = 0.2,
    theta: float = 0.0,
    reference: float = 0.0,
    draw: float | None = None,
    center: tuple[float, float] = (0.0, 0.0),
    branches: tuple[int, ...] = (0, 1),
) -> dict:
    ellipse_fit = {
        "status": "ok",
        "success": True,
        "q_unit": q_unit,
        "reference_axis_deg": reference,
        "a": a,
        "b": b,
        "axis_ratio": b / a,
        "theta_deg": theta,
        "center": center,
        "ellipses": [
            {
                "branch_id": branch,
                "a": a,
                "b": b,
                "axis_ratio": b / a,
                "theta_deg": reference + (theta if branch == 0 else -theta),
                "center": center,
            }
            for branch in branches
        ],
    }
    source = {"ellipse_fit": ellipse_fit, "q_unit": q_unit}
    if draw is not None:
        source["draw_axis_deg"] = draw
    return source


def _parameter(result: dict, name: str, branch_id=None) -> dict:
    return next(
        row
        for row in result["parameter_rows"]
        if row["parameter"] == name and row["branch_id"] == branch_id
    )


def _direction(result: dict, branch_id: int, angle: float) -> dict:
    return next(
        row
        for row in result["directional_rows"]
        if row["branch_id"] == branch_id
        and math.isclose(row["angle_deg"] % 360.0, angle % 360.0, abs_tol=1e-10)
    )


def _ridge_point(branch: int, qx: float, qy: float, *, valid=True, accepted=True) -> dict:
    return {
        "branch_id": branch,
        "qx": qx,
        "qy": qy,
        "valid": valid,
        "accepted": accepted,
        "q_unit": "nm^-1",
        "support": 0.8,
        "n_pixels": 4,
    }


def test_exact_periods_and_explicit_zero_draw_axis() -> None:
    result = analyze_lamellar_morphology(_ellipse_source(draw=0.0))

    assert result["reference_axis_deg"] == 0.0
    assert result["draw_axis_deg"] == 0.0
    assert _parameter(result, "Ln_from_minor_axis_nm")["candidate_value"] == pytest.approx(
        2.0 * math.pi / 0.2
    )
    assert _parameter(result, "L_from_major_axis_nm")["candidate_value"] == pytest.approx(
        2.0 * math.pi / 1.0
    )
    assert _parameter(result, "Lz_from_draw_axis_nm", 0)["candidate_value"] == pytest.approx(
        2.0 * math.pi / 1.0
    )
    at_zero = _direction(result, 0, 0.0)
    assert at_zero["q_radius"] == pytest.approx(1.0)
    assert at_zero["period_nm"] == pytest.approx(2.0 * math.pi)
    assert at_zero["relative_period"] == pytest.approx(0.2)
    json.dumps(result, allow_nan=False)


def test_global_member_angles_and_lz_follow_rotated_reference_and_draw_axes() -> None:
    source = _ellipse_source(theta=20.0, reference=30.0)
    source["analysis"] = {"draw_axis_deg": 120.0}
    result = analyze_lamellar_morphology(source)

    members = {row["branch_id"]: row for row in result["ellipse"]["members"]}
    assert members[0]["angle_deg"] == pytest.approx(50.0)
    assert members[1]["angle_deg"] == pytest.approx(10.0)
    assert result["draw_axis_deg"] == pytest.approx(120.0)
    assert _direction(result, 0, 120.0)["q_radius"] == pytest.approx(
        1.0 * 0.2 / math.sqrt((0.2 * math.cos(math.radians(70.0))) ** 2 + math.sin(math.radians(70.0)) ** 2)
    )
    assert _parameter(result, "Lz_from_draw_axis_nm", 0)["candidate_value"] == pytest.approx(
        2.0 * math.pi / _direction(result, 0, 120.0)["q_radius"]
    )
    assert _direction(result, 1, 90.0)["q_radius"] != pytest.approx(
        _direction(result, 0, 90.0)["q_radius"]
    )


def test_angstrom_and_inverse_nanometre_axes_give_equivalent_periods() -> None:
    nm = analyze_lamellar_morphology(_ellipse_source(q_unit="nm^-1", a=1.0, b=0.2))
    angstrom = analyze_lamellar_morphology(_ellipse_source(q_unit="Å^-1", a=0.1, b=0.02))

    for name in ("Ln_from_minor_axis_nm", "L_from_major_axis_nm"):
        assert _parameter(angstrom, name)["candidate_value"] == pytest.approx(
            _parameter(nm, name)["candidate_value"]
        )
    assert _direction(angstrom, 0, 90.0)["period_nm"] == pytest.approx(
        _direction(nm, 0, 90.0)["period_nm"]
    )


def test_undetermined_geometry_and_periods_remain_candidates() -> None:
    source = _ellipse_source()
    source["ellipse_fit"]["quantitative_parameters"] = {
        name: {
            "status": "undetermined",
            "value": None,
            "candidate_value": value,
            "reason": "insufficient_observed_sides",
        }
        for name, value in (("a", 1.0), ("b", 0.2), ("axis_ratio", 0.2), ("theta_deg", 0.0))
    }
    result = analyze_lamellar_morphology(source)

    a = _parameter(result, "a")
    assert a["status"] == "undetermined"
    assert a["value"] is None
    assert a["candidate_value"] == pytest.approx(1.0)
    assert a["reason"] == "insufficient_observed_sides"
    ln = _parameter(result, "Ln_from_minor_axis_nm")
    assert ln["status"] == "candidate"
    assert ln["value"] is None
    assert ln["candidate_value"] == pytest.approx(2.0 * math.pi / 0.2)
    assert all("candidate_model" in row["status"] for row in result["directional_rows"])


@pytest.mark.parametrize(
    ("q_unit", "center"),
    (("nm^-1", (0.05, 0.0)), ("pixel-q", (0.0, 0.0)), ("nm^-1", None)),
)
def test_unphysical_or_non_origin_geometry_never_gets_physical_periods(q_unit, center) -> None:
    source = _ellipse_source(q_unit=q_unit)
    if center is None:
        source["ellipse_fit"].pop("center")
        source["ellipse_fit"].pop("center_qx", None)
        source["ellipse_fit"].pop("center_qy", None)
        for member in source["ellipse_fit"]["ellipses"]:
            member.pop("center")
    else:
        source["ellipse_fit"]["center"] = center
        for member in source["ellipse_fit"]["ellipses"]:
            member["center"] = center
    result = analyze_lamellar_morphology(source)

    assert _parameter(result, "Ln_from_minor_axis_nm")["candidate_value"] is None
    assert all(row["period_nm"] is None for row in result["directional_rows"])
    if q_unit == "pixel-q" and center == (0.0, 0.0):
        assert all(row["relative_period"] is not None for row in result["directional_rows"])
    else:
        assert all(row["relative_period"] is None for row in result["directional_rows"])
    if center is not None:
        assert any(row["q_radius"] is not None for row in result["directional_rows"])
    else:
        assert all(row["q_radius"] is None for row in result["directional_rows"])


def test_only_retained_observed_branch_points_support_their_model_branch() -> None:
    source = _ellipse_source()
    source["ridge_points"] = [
        _ridge_point(0, 0.8, 0.0),
        _ridge_point(0, 0.0, 0.18),
        _ridge_point(1, 0.0, -0.18, valid=False),
        _ridge_point(1, -0.8, 0.0, accepted=False),
    ]
    source["lobe_radial_peaks"] = [
        {"method": "radial_peak", "branch_id": 1, "angle_deg": 180.0, "q_star": 0.8, "q_unit": "nm^-1"}
    ]
    result = analyze_lamellar_morphology(source)

    assert result["observed_support_source"] == "accepted_observed_ridge_points"
    assert len(result["observed_directions"]) == 2
    assert {row["branch_id"] for row in result["observed_directions"]} == {0}
    branch_0 = next(row for row in result["model_curves"] if row["branch_id"] == 0)
    branch_1 = next(row for row in result["model_curves"] if row["branch_id"] == 1)
    assert branch_0["observed_support"] is True
    assert branch_0["support_count"] == 2
    assert branch_1["observed_support"] is False
    assert branch_1["status"] == "candidate_model_no_observed_support"
    assert all(not row["observed_support"] for row in result["directional_rows"] if row["branch_id"] == 1)


def test_radial_peaks_support_model_branch_without_becoming_ridge_observations() -> None:
    source = _ellipse_source()
    source["lobe_radial_peaks"] = [
        {
            "method": "radial_peak",
            "branch_id": 0,
            "angle_deg": 0.0,
            "q_star": 0.4,
            "q_unit": "nm^-1",
            "status": "candidate",
        },
        {
            "method": "radial_peak",
            "branch_id": 0,
            "angle_deg": 8.0,
            "q_star": 0.42,
            "q_unit": "nm^-1",
        },
    ]

    result = analyze_lamellar_morphology(source)
    branch_0 = next(row for row in result["model_curves"] if row["branch_id"] == 0)
    branch_1 = next(row for row in result["model_curves"] if row["branch_id"] == 1)
    direction = _direction(result, 0, 0.0)

    assert result["observed_support_source"] == "retained_lobe_radial_peaks"
    assert result["observed_support_count"] == 2
    assert result["observed_directions"] == []
    assert branch_0["observed_support"] is True
    assert branch_0["support_count"] == 2
    assert branch_0["support_source"] == "retained_lobe_radial_peaks"
    assert branch_0["support_statuses"] == ["available", "candidate"]
    assert branch_1["observed_support"] is False
    assert direction["observed_support"] is True
    assert direction["support_source"] == "retained_lobe_radial_peaks"
    assert result["observed_radial_peaks"][0]["q"] == pytest.approx(0.4)
    assert result["observed_radial_peaks"][0]["comparisons"][0]["model_q_radius"] == pytest.approx(1.0)
    assert result["observed_radial_peaks"][0]["comparisons"][0]["observed_minus_model_q"] == pytest.approx(-0.6)


def test_observed_and_explicit_member_units_are_converted_for_comparison() -> None:
    source = _ellipse_source(q_unit="nm^-1")
    source["ellipse_fit"]["ellipses"][0].update(
        {"a": 0.1, "b": 0.02, "axis_ratio": 0.2, "q_unit": "Å^-1"}
    )
    source["ridge_points"] = [
        {
            **_ridge_point(0, 0.4, 0.0),
            "q_unit": "Å^-1",
            "status": "candidate",
        }
    ]
    source["lobe_radial_peaks"] = [
        {"method": "radial_peak", "branch_id": 0, "angle_deg": 0.0, "q_star": 0.4, "q_unit": "Å^-1"}
    ]

    result = analyze_lamellar_morphology(source)
    observed = result["observed_directions"][0]
    peak_comparison = result["observed_radial_peaks"][0]["comparisons"][0]
    member = next(row for row in result["ellipse"]["members"] if row["branch_id"] == 0)

    assert member["q_unit"] == "Å^-1"
    assert observed["status"] == "candidate"
    assert observed["observation_kind"] == "retained_observed_ridge"
    assert observed["q_unit"] == "Å^-1"
    assert observed["comparison_q_unit"] == "Å^-1"
    assert observed["model_q_radius"] == pytest.approx(0.1)
    assert observed["model_period_nm"] == pytest.approx(2.0 * math.pi)
    assert observed["residual_q"] == pytest.approx(0.3)
    assert observed["residual_q_unit"] == "Å^-1"
    assert peak_comparison["q_unit"] == "Å^-1"
    assert peak_comparison["model_q_radius"] == pytest.approx(0.1)
    assert peak_comparison["observed_minus_model_q"] == pytest.approx(0.3)


def test_incommensurate_observed_units_keep_point_without_model_comparison() -> None:
    source = _ellipse_source(q_unit="nm^-1")
    source["ridge_points"] = [
        {**_ridge_point(0, 0.4, 0.0), "q_unit": "detector-unit"}
    ]

    observed = analyze_lamellar_morphology(source)["observed_directions"][0]

    assert observed["q_radius"] == pytest.approx(0.4)
    assert observed["q_unit"] == "detector-unit"
    assert observed["model_q_radius"] is None
    assert observed["model_period_nm"] == pytest.approx(2.0 * math.pi)
    assert observed["residual_q"] is None
    assert observed["comparison_q_unit"] is None
    assert observed["comparison_reason"] == "q_units_incommensurate_or_unknown"


def test_failed_envelope_keeps_direct_radial_measurements_but_no_ellipse_tables() -> None:
    source = {
        "status": "failed",
        "error": "detector frame could not be processed",
        "result": {
            "q_unit": "nm^-1",
            "ridge_points": [_ridge_point(0, 0.4, 0.0)],
            "lobe_radial_peaks": [
                {"method": "radial_peak", "angle_deg": 45.0, "q_star": 0.25, "q_unit": "nm^-1"}
            ],
        },
    }
    result = analyze_lamellar_morphology(source)

    assert result["status"] == "failed"
    assert result["execution_status"] == "failed"
    assert result["parameter_rows"] == []
    assert result["directional_rows"] == []
    assert result["observed_directions"][0]["q_radius"] == pytest.approx(0.4)
    assert result["observed_radial_peaks"][0]["q"] == pytest.approx(0.25)
    assert "detector frame" in result["execution_reason"]


def test_failed_source_keeps_finite_ellipse_only_as_candidate() -> None:
    source = {"status": "failed", "result": _ellipse_source()["ellipse_fit"]}
    result = analyze_lamellar_morphology(source)

    assert result["status"] == "candidate"
    assert result["source_status"] == "ok"
    assert result["execution_status"] == "failed"
    assert _parameter(result, "a")["value"] is None
    assert _parameter(result, "a")["candidate_value"] == pytest.approx(1.0)
    assert _parameter(result, "Ln_from_minor_axis_nm")["value"] is None
    assert _parameter(result, "Ln_from_minor_axis_nm")["candidate_value"] == pytest.approx(
        2.0 * math.pi / 0.2
    )
    assert all(row["geometry_status"] == "candidate" for row in result["directional_rows"])


def test_near_circle_does_not_claim_an_identifiable_ellipse_direction() -> None:
    source = _ellipse_source(a=1.0, b=0.98, theta=37.0)
    source["ellipse_fit"]["quantitative_parameters"] = {
        name: {"status": "available", "value": value}
        for name, value in (("a", 1.0), ("b", 0.98), ("axis_ratio", 0.98), ("theta_deg", 37.0))
    }
    result = analyze_lamellar_morphology(source)

    assert result["ellipse"]["axis_direction_status"] == "not_identified_near_circle"
    assert result["ellipse"]["axis_direction_reason"] == "near_circular_ellipse_axis_unidentifiable"
    assert "near_circular_ellipse_axis_unidentifiable" in result["flags"]
    assert any("unique three-dimensional" in assumption for assumption in result["assumptions"])
    assert all(member["status"] == "candidate" for member in result["ellipse"]["members"])
    assert all(row["status"].startswith("candidate_model") for row in result["directional_rows"])
    assert all(
        row["status"].startswith("candidate") and row["value"] is None
        for row in result["parameter_rows"]
        if row["parameter"] == "Lz_from_draw_axis_nm"
    )


def test_missing_ellipse_returns_empty_geometry_but_not_missing_source_reason() -> None:
    result = analyze_lamellar_morphology({"ellipse_fit": {"status": "failed", "message": "no fit"}})

    assert result["status"] == "failed"
    assert result["ellipse"] is not None
    assert result["parameter_rows"]
    assert all(row["candidate_value"] is None for row in result["parameter_rows"])


def test_ellipse_members_use_available_quantitative_values_over_optimizer_candidates() -> None:
    source = _ellipse_source(theta=22.0, reference=30.0)
    source["ellipse_fit"]["quantitative_parameters"] = {
        "a": {"status": "available", "value": 0.5, "candidate_value": 0.55},
        "b": {"status": "available", "value": 0.1, "candidate_value": 0.11},
        "axis_ratio": {"status": "available", "value": 0.2, "candidate_value": 0.2},
        "theta_deg": {"status": "available", "value": 20.0, "candidate_value": 22.0},
    }
    source["ridge_points"] = [_ridge_point(0, 0.4, 0.0)]
    result = analyze_lamellar_morphology(source)

    assert _parameter(result, "a")["value"] == pytest.approx(0.5)
    assert _parameter(result, "a")["candidate_value"] is None
    assert _parameter(result, "Ln_from_minor_axis_nm")["value"] == pytest.approx(
        2.0 * math.pi / 0.1
    )
    assert _parameter(result, "Ln_from_minor_axis_nm")["candidate_value"] is None
    assert _parameter(result, "Lz_from_draw_axis_nm", 0)["value"] is not None
    assert _parameter(result, "Lz_from_draw_axis_nm", 0)["candidate_value"] is None
    assert all(
        row["candidate_value"] is None
        for row in result["parameter_rows"]
        if row["value"] is not None
    )
    members = {row["branch_id"]: row for row in result["ellipse"]["members"]}
    # Quantitative summary values do not rewrite explicit per-member geometry.
    assert members[0]["a"] == pytest.approx(1.0)
    assert members[1]["a"] == pytest.approx(1.0)
    assert members[0]["angle_deg"] == pytest.approx(52.0)
    assert members[1]["angle_deg"] == pytest.approx(8.0)


def test_explicit_member_angles_are_preserved_without_shared_theta_reconstruction() -> None:
    source = _ellipse_source(theta=19.0, reference=44.0)
    source["ellipse_fit"]["ellipses"][0]["theta_deg"] = 28.0
    source["ellipse_fit"]["ellipses"][0]["angle_deg"] = 28.0
    source["ellipse_fit"]["ellipses"][1]["theta_deg"] = 116.0
    source["ellipse_fit"]["ellipses"][1]["angle_deg"] = 116.0
    source["ellipse_fit"]["quantitative_parameters"] = {
        name: {"status": "available", "value": value}
        for name, value in (("a", 1.0), ("b", 0.2), ("axis_ratio", 0.2), ("theta_deg", 19.0))
    }

    result = analyze_lamellar_morphology(source)
    members = {row["branch_id"]: row for row in result["ellipse"]["members"]}

    assert members[0]["angle_deg"] == pytest.approx(28.0)
    assert members[1]["angle_deg"] == pytest.approx(116.0)
    for branch_id, angle in ((0, 28.0), (1, 116.0)):
        model = _direction(result, branch_id, 0.0)
        assert model["q_radius"] == pytest.approx(
            1.0 * 0.2
            / math.sqrt(
                (0.2 * math.cos(math.radians(angle))) ** 2
                + math.sin(math.radians(angle)) ** 2
            )
        )
