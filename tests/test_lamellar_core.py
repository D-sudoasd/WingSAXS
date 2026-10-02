from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.lamellar import (
    LamellarScene,
    LamellarSettings,
    build_lamellar_scene,
    load_lamellar_sources,
)
from butterfly_saxs.lamellar_adapter import observed_direction_support


def _radial_source(*, q_unit: str = "nm^-1") -> dict:
    return {
        "source_identity": {
            "source": "frame_001.cbf",
            "frame": 3,
            "dataset": "entry/data",
        },
        "q_unit": q_unit,
        "draw_axis_deg": 90.0,
        "lobe_radial_peaks": [
            {
                "angle_deg": 30.0,
                "q_star": 0.2,
                "q_unit": q_unit,
                "valid": True,
                "branch_id": 0,
            },
            {
                "angle_deg": 210.0,
                "q_star": 0.2,
                "q_unit": q_unit,
                "valid": True,
                "branch_id": 0,
            },
            {
                "angle_deg": 150.0,
                "q_star": 0.1,
                "q_unit": q_unit,
                "valid": True,
                "branch_id": 1,
            },
            {
                "angle_deg": 330.0,
                "q_star": 0.1,
                "q_unit": q_unit,
                "valid": True,
                "branch_id": 1,
            },
        ],
    }


def _ridge_point(angle_deg: float, *, branch_id: int | None = 0, q: float = 0.2) -> dict:
    angle = np.deg2rad(angle_deg)
    return {
        "qx": float(q * np.cos(angle)),
        "qy": float(q * np.sin(angle)),
        "q_unit": "nm^-1",
        "valid": True,
        "accepted": True,
        "branch_id": branch_id,
        "status": "observed",
    }


def test_settings_round_trip_and_validation() -> None:
    settings = LamellarSettings.from_mapping(
        {"layer_count": 3, "stack_count": 2, "mode": "multi"}
    )
    assert settings.to_dict()["layer_count"] == 3
    assert settings.to_dict()["mode"] == "multi"
    with pytest.raises(ValueError, match="thickness_ratio"):
        LamellarSettings(thickness_ratio=1.0)
    with pytest.raises(ValueError, match="selected_branch"):
        LamellarSettings(selected_branch=2)


def test_radial_scene_uses_physical_period_and_contract_shapes() -> None:
    scene = build_lamellar_scene(_radial_source(), {"layer_count": 2, "stack_count": 4})
    assert isinstance(scene, LamellarScene)
    assert scene.status == "schematic"
    assert scene.length_unit == "nm"
    assert scene.centers.shape == (4, 3)
    assert scene.sizes.shape == (4, 3)
    assert scene.orientations.shape == (4, 3, 3)
    assert scene.vertices.shape == (4, 8, 3)
    assert scene.stack_ids.shape == (4,)
    assert scene.branch_ids.shape == (4,)
    assert scene.colors.shape == (4, 4)
    assert scene.metadata["source_identity"]["frame"] == 3
    assert scene.metadata["populations"][0]["period"] == pytest.approx(
        2.0 * np.pi / 0.2
    )
    assert np.all(np.isfinite(scene.vertices))
    np.testing.assert_allclose(
        scene.vertices[0, 0],
        scene.centers[0]
        + scene.orientations[0]
        @ np.array(
            [-scene.sizes[0, 0] / 2, -scene.sizes[0, 1] / 2, -scene.sizes[0, 2] / 2]
        ),
    )


def test_different_population_length_units_require_branch_selection() -> None:
    source = _radial_source(q_unit="unknown")
    source["lobe_radial_peaks"][0]["q_unit"] = "nm^-1"
    source["lobe_radial_peaks"][1]["q_unit"] = "nm^-1"

    combined = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 4})

    assert combined.status == "unavailable"
    assert not combined.metadata["available"]
    assert "mixed_length_units" in combined.metadata["flags"]
    assert "select" in combined.message.lower() or "选择" in combined.message
    assert combined.centers.shape == (0, 3)
    assert [item["length_unit"] for item in combined.metadata["populations"]] == ["nm", "relative"]
    assert [item["period"] for item in combined.metadata["populations"]] == pytest.approx([2.0 * np.pi / .2, 1.0])

    physical = build_lamellar_scene(source, {"selected_branch": 0, "layer_count": 1, "stack_count": 4})
    relative = build_lamellar_scene(source, {"selected_branch": 1, "layer_count": 1, "stack_count": 4})
    assert physical.metadata["available"] and physical.length_unit == "nm"
    assert relative.metadata["available"] and relative.length_unit == "relative"
    assert set(physical.branch_ids) == {0}
    assert set(relative.branch_ids) == {1}


def test_draw_axis_rotation_and_opposite_support_are_explicit() -> None:
    source = _radial_source()
    source["draw_axis_deg"] = 0.0
    scene = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 1})
    # 30° measured q plus (90 - draw_axis) display rotation gives 120°.
    normal = scene.orientations[0][:, 2]
    assert np.degrees(np.arctan2(normal[1], normal[0])) == pytest.approx(120.0)
    assert "inconsistent_opposite_support" not in scene.metadata["flags"]
    assert scene.metadata["populations"][0]["support_indices"] == [0, 1]


def test_unknown_q_is_relative_and_annulus_or_invalid_peaks_are_rejected() -> None:
    source = {
        "lobe_radial_peaks": [
            {
                "angle_deg": 20,
                "q": 2.0,
                "q_unit": "unknown",
                "method": "azimuthal_peak",
                "valid": True,
            },
            {"angle_deg": 20, "q_star": 0.0, "q_unit": "unknown", "valid": True},
            {"angle_deg": 20, "q_star": 1.0, "q_unit": "unknown", "valid": False},
        ]
    }
    scene = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 1})
    assert scene.status == "unavailable"
    assert scene.centers.shape == (0, 3)
    assert scene.length_unit == "relative"
    assert "no_valid_lobe_radial_peaks" in scene.metadata["flags"]

    relative = build_lamellar_scene(
        {"lobe_radial_peaks": [{"angle_deg": 20, "q_star": 2.0, "valid": True}]},
        {"layer_count": 1, "stack_count": 1},
    )
    assert relative.status == "schematic"
    assert relative.length_unit == "relative"
    assert relative.metadata["parameter_sources"][0]["value"] == pytest.approx(1.0)


def test_historic_json_safe_ridgepoint_uses_only_radial_lobe_metadata() -> None:
    source = {
        "q_unit": "nm^-1",
        "lobe_radial_peaks": [
            {
                "qx": 0.2 * np.cos(np.deg2rad(30)),
                "qy": 0.2 * np.sin(np.deg2rad(30)),
                "q_unit": "nm^-1",
                "component": 0,
                "metadata": {
                    "accepted": True,
                    "angle": np.deg2rad(30),
                    "method": "radial_peak",
                    "flags": ["lobe_radial_peak"],
                },
            },
            {
                "qx": 0.2 * np.cos(np.deg2rad(210)),
                "qy": 0.2 * np.sin(np.deg2rad(210)),
                "q_unit": "nm^-1",
                "component": 0,
                "metadata": {
                    "accepted": True,
                    "angle": np.deg2rad(210),
                    "source_method": "radial_peak_in_observed_lobe_sector",
                    "flags": ["lobe_radial_peak"],
                },
            },
        ],
    }
    scene = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 1})
    assert scene.metadata["available"]
    assert scene.metadata["populations"][0]["period"] == pytest.approx(
        2.0 * np.pi / 0.2, rel=1e-6
    )


def test_failed_or_nested_cancelled_sources_are_blank_but_manual_mode_survives() -> (
    None
):
    failed = _radial_source()
    failed["status"] = "failed"
    scene = build_lamellar_scene(failed)
    assert scene.status == "unavailable"
    assert not scene.metadata["available"]
    nested = {"result": {"status": "cancelled", **_radial_source()}}
    nested_scene = build_lamellar_scene(nested)
    assert nested_scene.status == "unavailable"
    manual = build_lamellar_scene(
        {"status": "failed"},
        {"period_source": "manual", "layer_count": 1, "stack_count": 1},
    )
    assert manual.status == "manual"
    assert manual.metadata["available"]


def test_unknown_native_q_mismatch_is_candidate_with_both_support_values() -> None:
    source = {
        "q_unit": "unknown",
        "lobe_radial_peaks": [
            {
                "angle_deg": 30.0,
                "q_star": 0.2,
                "q_unit": "unknown",
                "branch_id": 0,
                "valid": True,
            },
            {
                "angle_deg": 210.0,
                "q_star": 0.4,
                "q_unit": "unknown",
                "branch_id": 0,
                "valid": True,
            },
        ],
    }
    scene = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 1})
    assert scene.status == "candidate"
    assert "inconsistent_opposite_support" in scene.metadata["flags"]
    assert scene.metadata["populations"][0]["support_q_values"] == pytest.approx(
        [0.2, 0.4]
    )
    mixed = {
        "lobe_radial_peaks": [
            {
                "angle_deg": 30.0,
                "q_star": 0.2,
                "q_unit": "unknown",
                "branch_id": 0,
                "valid": True,
            },
            {
                "angle_deg": 210.0,
                "q_star": 0.4,
                "q_unit": "pixel-q",
                "branch_id": 0,
                "valid": True,
            },
        ]
    }
    mixed_scene = build_lamellar_scene(mixed, {"layer_count": 1, "stack_count": 1})
    assert mixed_scene.status == "candidate"
    assert "mixed_q_unit_support" in mixed_scene.metadata["flags"]


def test_manual_mode_is_available_without_source() -> None:
    scene = build_lamellar_scene(
        {},
        {
            "period_source": "manual",
            "manual_period": 2.5,
            "manual_angle_deg": 15.0,
            "manual_unit": "relative",
            "layer_count": 2,
            "stack_count": 1,
        },
    )
    assert scene.status == "manual"
    assert scene.length_unit == "relative"
    assert scene.metadata["populations"][0]["status"] == "manual"
    assert scene.centers.shape == (2, 3)


def test_manual_crossed_direction_field_is_explicit_and_fills_requested_slots() -> None:
    scene = build_lamellar_scene(
        {},
        {
            "period_source": "manual",
            "mode": "multi",
            "manual_period": 1.0,
            "manual_angle_deg": 30.0,
            "manual_second_orientation": True,
            "manual_second_angle_deg": 120.0,
            "layer_count": 1,
            "stack_count": 12,
            "position_jitter_pct": 70.0,
            "seed": 4,
        },
    )

    assert len(scene.centers) == 12
    assert set(scene.stack_ids.tolist()) == set(range(12))
    assert np.bincount(scene.branch_ids, minlength=2).tolist() == [6, 6]
    assert not np.array_equal(scene.branch_ids, np.arange(12) % 2)
    assert [item["angle_deg"] for item in scene.metadata["populations"]] == [30., 120.]
    assert any("not fitted populations" in item for item in scene.metadata["assumptions"])

    selected_b = build_lamellar_scene(
        {},
        {
            "period_source": "manual",
            "mode": "multi",
            "manual_second_orientation": True,
            "manual_second_angle_deg": 120.0,
            "selected_branch": 1,
            "layer_count": 1,
            "stack_count": 12,
        },
    )
    assert len(selected_b.centers) == 12
    assert set(selected_b.branch_ids.tolist()) == {1}
    assert selected_b.metadata["populations"][0]["angle_deg"] == pytest.approx(120.)

    single_direction = build_lamellar_scene(
        {},
        {
            "period_source": "manual",
            "mode": "multi",
            "manual_second_orientation": False,
            "layer_count": 1,
            "stack_count": 12,
        },
    )
    assert set(single_direction.stack_ids.tolist()) == set(range(12))


def test_manual_position_grid_uses_largest_period_when_reference_is_smaller() -> None:
    scene = build_lamellar_scene(
        {},
        {
            "period_source": "manual",
            "mode": "multi",
            "manual_period": 10.0,
            "reference_period": 1.0,
            "manual_second_orientation": True,
            "layer_count": 1,
            "stack_count": 12,
            "width_ratio": 3.0,
            "depth_ratio": 1.0,
            "thickness_ratio": 0.25,
            "position_jitter_pct": 70.0,
            "seed": 13,
        },
    )
    centres = scene.centers[::1]  # one layer per packet in this fixture
    distances = np.linalg.norm(centres[:, None, :2] - centres[None, :, :2], axis=2)
    distances += np.eye(len(centres)) * 1.e9
    enclosing_diameter = np.linalg.norm((30.0, 10.0, 2.5))

    assert scene.metadata["reference_period"] == pytest.approx(10.0)
    assert len(centres) == 12
    assert float(np.min(distances)) >= enclosing_diameter - 1.e-8
    assert LamellarSettings().position_jitter_pct == pytest.approx(24.0)


@pytest.mark.parametrize("status", ["estimate", "candidate"])
def test_ellipse_scene_preserves_parameter_evidence_over_raw_solver_alias(status):
    source = _radial_source()
    source["ellipse_fit"] = {
        "a": 0.2, "b": 0.1, "theta_deg": 10.0,
        "center": [0.0, 0.0], "q_unit": "nm^-1", "status": "ok",
        "parameters": {"a": 0.2, "b": 0.1, "theta_deg": 10.0},
        "quantitative_parameters": {
            "b": {"value": 0.1, "candidate_value": 0.1, "status": status}
        },
    }
    scene = build_lamellar_scene(
        source, {"period_source": "ellipse", "layer_count": 1, "stack_count": 1}
    )
    assert scene.status == "candidate"
    radius = 0.2 * 0.1 / np.hypot(
        0.1 * np.cos(np.deg2rad(20.0)),
        0.2 * np.sin(np.deg2rad(20.0)),
    )
    assert scene.metadata["populations"][0]["period"] == pytest.approx(
        2.0 * np.pi / radius
    )
    minor_axis = next(
        row
        for row in scene.metadata["parameter_sources"]
        if row["name"] == "period_from_ellipse_minor_axis"
    )
    assert minor_axis["value"] is None
    assert minor_axis["candidate_value"] == pytest.approx(2.0 * np.pi / 0.1)


def test_ellipse_requires_origin_and_keeps_candidate_formal_value_null() -> None:
    base = _radial_source()
    base["ellipse_fit"] = {"b": 0.1, "center": [0.01, 0.0], "q_unit": "nm^-1"}
    unavailable = build_lamellar_scene(
        base, {"period_source": "ellipse", "layer_count": 1, "stack_count": 1}
    )
    assert unavailable.status == "unavailable"
    assert "spacing_unavailable_nonzero_center" in unavailable.metadata["flags"]

    candidate = _radial_source()
    candidate["ellipse_fit"] = {
        "a": 0.2,
        "b": {"candidate_value": 0.1, "status": "candidate"},
        "theta_deg": 10.0,
        "center": [0.0, 0.0],
        "q_unit": "nm^-1",
        "parameters": {"a": 0.2, "b": 0.1, "theta_deg": 10.0},
        "quantitative_parameters": {"b": {"candidate_value": 0.1, "status": "candidate"}},
    }
    scene = build_lamellar_scene(
        candidate, {"period_source": "ellipse", "layer_count": 1, "stack_count": 1}
    )
    assert scene.status == "candidate"
    assert scene.metadata["populations"][0]["period"] < 2.0 * np.pi / 0.1
    source_row = next(
        item
        for item in scene.metadata["parameter_sources"]
        if item["name"] == "period_from_ellipse_minor_axis"
    )
    assert source_row["value"] is None
    assert source_row["status"] == "candidate"
    b_row = next(
        item for item in scene.metadata["parameter_sources"] if item["name"] == "b"
    )
    assert b_row["value"] is None
    assert b_row["candidate_value"] == pytest.approx(0.1)


def test_ellipse_only_scene_uses_accepted_observed_directions_and_directional_radius() -> None:
    source = {
        "q_unit": "nm^-1",
        "draw_axis_deg": 90.0,
        "ridges": [_ridge_point(30.0), _ridge_point(210.0)],
        "ellipse_fit": {
            "a": 0.2,
            "b": 0.1,
            "theta_deg": 10.0,
            "reference_axis_deg": 0.0,
            "center": [0.0, 0.0],
            "q_unit": "nm^-1",
            "parameters": {"a": 0.2, "b": 0.1, "theta_deg": 10.0},
        },
    }
    scene = build_lamellar_scene(
        source,
        {"period_source": "ellipse", "selected_branch": 0, "layer_count": 1, "stack_count": 1},
    )

    angle = np.deg2rad(30.0 - 10.0)
    radius = 0.2 * 0.1 / np.hypot(0.1 * np.cos(angle), 0.2 * np.sin(angle))
    assert scene.status == "schematic"
    assert scene.metadata["populations"][0]["period"] == pytest.approx(2.0 * np.pi / radius)
    assert scene.metadata["populations"][0]["angle_deg"] == pytest.approx(30.0)
    assert np.rad2deg(np.arctan2(scene.orientations[0, 1, 2], scene.orientations[0, 0, 2])) == pytest.approx(30.0)
    assert not np.isclose(scene.metadata["populations"][0]["period"], 2.0 * np.pi / 0.1)


def test_radial_fallback_direction_provenance_is_not_labelled_as_ridge_points() -> None:
    source = _radial_source()
    source["ellipse_fit"] = {
        "a": 0.2,
        "b": 0.1,
        "theta_deg": 10.0,
        "center": [0.0, 0.0],
        "q_unit": "nm^-1",
        "parameters": {"a": 0.2, "b": 0.1, "theta_deg": 10.0},
    }

    scene = build_lamellar_scene(
        source, {"period_source": "ellipse", "layer_count": 1, "stack_count": 2}
    )

    support_row = next(
        row
        for row in scene.metadata["parameter_sources"]
        if row["name"] == "observed_direction_support"
    )
    assert scene.status == "schematic"
    assert {row["source"] for row in scene.metadata["direction_support"]} == {
        "lobe_radial_peaks"
    }
    assert {population["orientation_source"] for population in scene.metadata["populations"]} == {
        "retained_lobe_radial_peaks"
    }
    assert support_row["source"] == "retained_lobe_radial_peaks"
    assert "ellipse_direction_from_radial_lobe_peaks" in scene.metadata["flags"]
    assert "ellipse_direction_from_observed_ridge_points" not in scene.metadata["flags"]


def test_observed_direction_support_restores_only_explicit_fit_labels_from_unresolved_source_ids() -> None:
    points = [_ridge_point(30.0, branch_id=-1), _ridge_point(210.0, branch_id=-1)]
    for point in points:
        point["branch_assignment_source"] = "reference_quadrant"
        point["metadata"] = {"quadrant_pair": "QI+QIII"}
    source = {
        "ridges": points,
        "ellipse_fit": {
            "branch_assignment_values": np.asarray([0, 1]),
            "branch_assignment_indices": np.asarray([0, 1]),
        },
    }

    rows = observed_direction_support(source)

    assert [row["branch_id"] for row in rows] == [0, 1]
    assert [row["source_branch_id"] for row in rows] == [-1, -1]
    assert all(row["branch_id_source"] == "ellipse_fit.branch_assignment" for row in rows)
    assert all(row["branch_assignment_source"] == "reference_quadrant" for row in rows)


def test_observed_direction_support_does_not_promote_quadrant_or_truncate_branch_ids() -> None:
    points = [_ridge_point(30.0, branch_id=None), _ridge_point(210.0, branch_id=3.7)]
    points[0]["metadata"] = {"quadrant_pair": "QI+QIII"}
    source = {"ridges": points}

    rows = observed_direction_support(source)

    assert [row["branch_id"] for row in rows] == [None, None]
    assert [row["angle_deg"] for row in rows] == pytest.approx([30.0, -150.0])


def test_ellipse_mode_keeps_a_coherent_unassigned_direction_without_fake_branches() -> None:
    source = {
        "q_unit": "nm^-1",
        "ridges": [_ridge_point(0.0, branch_id=-1), _ridge_point(180.0, branch_id=-1)],
        "ellipse_fit": {
            "a": 0.2,
            "b": 0.1,
            "reference_axis_deg": 0.0,
            "center": [0.0, 0.0],
            "q_unit": "nm^-1",
            "ellipses": [
                {"a": 0.2, "b": 0.1, "angle_deg": 28.0, "center": [0.0, 0.0]},
                {"a": 0.2, "b": 0.1, "angle_deg": -28.0, "center": [0.0, 0.0]},
            ],
        },
    }

    scene = build_lamellar_scene(
        source, {"period_source": "ellipse", "layer_count": 1, "stack_count": 2}
    )

    assert scene.status == "schematic"
    assert set(scene.branch_ids.tolist()) == {-1}
    assert scene.metadata["populations"][0]["branch_label"] == "Observed direction"
    assert scene.metadata["populations"][0]["fitted_member_branch_id"] is None


def test_ambiguous_unassigned_ellipse_direction_does_not_discard_explicit_branch() -> None:
    source = {
        "q_unit": "nm^-1",
        "ridges": [
            _ridge_point(28.0, branch_id=0),
            _ridge_point(208.0, branch_id=0),
            _ridge_point(45.0, branch_id=-1),
            _ridge_point(225.0, branch_id=-1),
            _ridge_point(135.0, branch_id=-1),
            _ridge_point(315.0, branch_id=-1),
        ],
        "ellipse_fit": {
            "a": 0.2,
            "b": 0.1,
            "center": [0.0, 0.0],
            "q_unit": "nm^-1",
            "ellipses": [
                {"a": 0.2, "b": 0.1, "angle_deg": 28.0, "center": [0.0, 0.0]},
                {"a": 0.2, "b": 0.1, "angle_deg": 116.0, "center": [0.0, 0.0]},
            ],
        },
    }

    scene = build_lamellar_scene(
        source, {"period_source": "ellipse", "layer_count": 1, "stack_count": 2}
    )

    assert scene.status == "schematic"
    assert set(scene.branch_ids.tolist()) == {0}
    assert "ambiguous_unassigned_direction_retained" in scene.metadata["flags"]
    assert any(
        item["branch_id"] == -1 and item["status"] == "unavailable"
        for item in scene.metadata["populations"]
    )


def test_ellipse_directional_period_uses_explicit_member_angles_and_relative_pixel_q() -> None:
    source = {
        "q_unit": "pixel^-1",
        "ridges": [
            {**_ridge_point(28.0, branch_id=0), "q_unit": "pixel^-1"},
            {**_ridge_point(208.0, branch_id=0), "q_unit": "pixel^-1"},
        ],
        "ellipse_fit": {
            "a": 0.2,
            "b": 0.1,
            "theta_deg": 3.0,
            "reference_axis_deg": 0.0,
            "center": [0.0, 0.0],
            "q_unit": "pixel^-1",
            "ellipses": [
                {"a": 0.2, "b": 0.1, "angle_deg": 28.0, "center": [0.0, 0.0]},
                {"a": 0.2, "b": 0.1, "angle_deg": 116.0, "center": [0.0, 0.0]},
            ],
        },
    }

    scene = build_lamellar_scene(
        source,
        {"period_source": "ellipse", "selected_branch": 0, "reference_period": 10.0, "layer_count": 1, "stack_count": 1},
    )

    assert scene.status == "schematic"
    assert scene.length_unit == "relative"
    assert scene.metadata["populations"][0]["period"] == pytest.approx(10.0)
    assert scene.metadata["populations"][0]["fitted_member_branch_id"] == 0
    assert scene.metadata["populations"][0]["q_unit"] == "pixel^-1"
    assert scene.metadata["parameter_sources"][-1]["unit"] == "pixel^-1"


def test_single_explicit_ellipse_member_matches_only_its_observed_branch() -> None:
    source = _radial_source()
    source["ellipse_fit"] = {
        "a": 0.2,
        "b": 0.1,
        "theta_deg": 10.0,
        "center": [0.0, 0.0],
        "q_unit": "nm^-1",
        "ellipses": [
            {
                "branch_id": 1,
                "a": 0.2,
                "b": 0.1,
                "angle_deg": 150.0,
                "center": [0.0, 0.0],
            }
        ],
    }

    wrong_branch = build_lamellar_scene(
        source,
        {"period_source": "ellipse", "selected_branch": 0, "layer_count": 1, "stack_count": 1},
    )
    matching_branch = build_lamellar_scene(
        source,
        {"period_source": "ellipse", "selected_branch": 1, "layer_count": 1, "stack_count": 1},
    )

    assert wrong_branch.status == "unavailable"
    assert "matches observed branch 0" in wrong_branch.message
    assert wrong_branch.metadata["direction_support"]
    assert matching_branch.status == "schematic"
    assert matching_branch.metadata["populations"][1]["fitted_member_branch_id"] == 1
    assert matching_branch.metadata["populations"][1]["period"] == pytest.approx(
        2.0 * np.pi / 0.2
    )


def test_nested_zero_draw_axis_is_not_replaced_by_default() -> None:
    scene = build_lamellar_scene(
        {"observables": {"draw_axis_deg": 0.0}},
        {
            "period_source": "manual",
            "manual_period": 1.0,
            "manual_angle_deg": 30.0,
            "layer_count": 1,
            "stack_count": 1,
        },
    )

    assert scene.metadata["draw_axis_deg"] == pytest.approx(0.0)
    assert np.rad2deg(np.arctan2(scene.orientations[0, 1, 2], scene.orientations[0, 0, 2])) == pytest.approx(120.0)


def test_scene_is_deterministic_without_mutating_source() -> None:
    source = _radial_source()
    before = json.loads(json.dumps(source))
    settings = {
        "layer_count": 3,
        "stack_count": 2,
        "mode": "multi",
        "spread_deg": 8.0,
        "seed": 42,
    }
    first = build_lamellar_scene(source, settings)
    second = build_lamellar_scene(source, settings)
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.orientations, second.orientations)
    np.testing.assert_array_equal(first.vertices, second.vertices)
    assert source == before


def test_interlayer_spacing_jitter_is_bounded_reproducible_and_explicitly_assumed() -> None:
    source = _radial_source()
    settings = {
        "period_source": "manual",
        "manual_period": 2.0,
        "manual_angle_deg": 30.0,
        "layer_count": 6,
        "stack_count": 1,
        "spacing_jitter_pct": 35.0,
        "seed": 23,
    }
    first = build_lamellar_scene(source, settings)
    repeated = build_lamellar_scene(source, settings)
    np.testing.assert_array_equal(first.centers, repeated.centers)

    normal = first.orientations[0, :, 2]
    projected = first.centers @ normal
    gaps = np.diff(projected)
    assert np.all(gaps >= 2.0 * 0.65 - 1e-12)
    assert np.all(gaps <= 2.0 * 1.35 + 1e-12)
    assert np.ptp(gaps) > 1e-6
    assert any("not fitted uncertainty" in item for item in first.assumptions)


def test_interlayer_spacing_jitter_validation_and_zero_preserves_regular_stack() -> None:
    regular = build_lamellar_scene(
        {},
        {
            "period_source": "manual",
            "manual_period": 3.0,
            "layer_count": 5,
            "stack_count": 1,
            "spacing_jitter_pct": 0.0,
        },
    )
    normal = regular.orientations[0, :, 2]
    gaps = np.diff(regular.centers @ normal)
    np.testing.assert_allclose(gaps, np.full(4, 3.0))

    with pytest.raises(ValueError, match="spacing_jitter_pct"):
        LamellarSettings(spacing_jitter_pct=50.1)
    with pytest.raises(ValueError, match="minimum layer gap"):
        LamellarSettings(thickness_ratio=0.9, spacing_jitter_pct=10.1)


def test_multi_stack_count_is_total_and_fixed_slots_survive_missing_branch() -> None:
    both = build_lamellar_scene(
        _radial_source(),
        {
            "mode": "multi",
            "layer_count": 1,
            "stack_count": 12,
            "reference_period": 100.0,
            "seed": 7,
        },
    )
    assert len(both.centers) == 12
    assert set(both.stack_ids.tolist()) == set(range(12))
    assert (
        len(
            {(round(float(row[0]), 6), round(float(row[1]), 6)) for row in both.centers}
        )
        > 4
    )
    only_a = _radial_source()
    only_a["lobe_radial_peaks"] = only_a["lobe_radial_peaks"][:2]
    one = build_lamellar_scene(
        only_a,
        {
            "mode": "multi",
            "layer_count": 1,
            "stack_count": 12,
            "reference_period": 100.0,
            "seed": 7,
        },
    )
    assert len(one.centers) == 6
    assert set(one.stack_ids.tolist()) == {0, 2, 4, 6, 8, 10}
    both_by_id = {
        int(stack): center
        for stack, center in zip(both.stack_ids, both.centers)
        if int(stack) % 2 == 0
    }
    one_by_id = {
        int(stack): center for stack, center in zip(one.stack_ids, one.centers)
    }
    for stack in one_by_id:
        np.testing.assert_allclose(one_by_id[stack], both_by_id[stack])


def test_native_single_loader_embeds_arrays_and_batch_loader_is_lazy(
    tmp_path: Path,
) -> None:
    observed = np.arange(12, dtype=float).reshape(3, 4)
    qx = np.tile(np.arange(4, dtype=float), (3, 1))
    qy = np.tile(np.arange(3, dtype=float)[:, None], (1, 4))
    summary = {
        "q_unit": "nm^-1",
        "observables": {"lobe_radial_peaks": []},
        "metadata": {"path": "frame.npy"},
        "flags": {},
    }
    summary_path = tmp_path / "frame.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(tmp_path / "frame.npz", image=observed, qx=qx, qy=qy)
    loaded = load_lamellar_sources(summary_path)
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0]["observed"], observed)
    np.testing.assert_array_equal(loaded[0]["qx"], qx)
    assert loaded[0]["source_identity"]["source"] == "frame.npy"

    manifest = {
        "schema_version": "lamellarsaxs2d.batch.v1",
        "frames": [
            {"frame_index": 0, "frame_id": "a"},
            {"frame_index": 1, "frame_id": "b"},
            {"frame_index": 2, "frame_id": "failed", "status": "failed"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "ellipse_fit.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "ellipse_fit": {"b": 0.1},
                        "lobe_radial_peaks": [
                            {"angle_deg": 30, "q_star": 0.2, "valid": True}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        tmp_path / "results.npz",
        **{
            "frame_0000__image": observed,
            "frame_0000__qmap__qx": qx,
            "frame_0000__qmap__qy": qy,
            "frame_0001__image": observed + 1,
            "frame_0001__qmap__qx": qx,
            "frame_0001__qmap__qy": qy,
        },
    )
    batch = load_lamellar_sources(tmp_path)
    assert len(batch) == 3
    assert "observed" not in batch[0]
    assert batch[0]["array_keys"]["observed"] == "frame_0000__image"
    assert batch[0]["lobe_radial_peaks"][0]["q_star"] == pytest.approx(0.2)
    assert Path(batch[0]["npz_path"]).resolve() == (tmp_path / "results.npz").resolve()
    assert batch[2]["frame_id"] == "failed"
    assert "array_error" in batch[2]


def test_loader_rejects_csv_and_unrecognized_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("q,value\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="native"):
        load_lamellar_sources(csv_path)
    json_path = tmp_path / "random.json"
    json_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError, match="native"):
        load_lamellar_sources(json_path)


def test_existing_native_batch_bundle_contract_if_available() -> None:
    bundle = Path(
        r"G:\LamellarSAXS2D_butterfly_upgrade_20260906\results\batch_resume_release"
    )
    if not bundle.is_dir():
        pytest.skip("recorded native batch fixture is not mounted")
    sources = load_lamellar_sources(bundle)
    assert len(sources) >= 2
    assert all(source.get("array_keys", {}).get("observed") for source in sources)
    assert all(source.get("lobe_radial_peaks") is not None for source in sources)


def test_manual_stack_count_is_the_requested_total():
    scene = build_lamellar_scene(
        {}, {"period_source": "manual", "mode": "multi", "stack_count": 12}
    )
    assert len(np.unique(scene.stack_ids)) == 12
    assert len(scene.centers) == 12 * 8
