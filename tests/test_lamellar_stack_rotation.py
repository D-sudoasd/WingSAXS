from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.lamellar import LamellarSettings, build_lamellar_scene


def _source(*angles: float, draw_axis_deg: float = 90.0) -> dict:
    return {
        "draw_axis_deg": draw_axis_deg,
        "q_unit": "nm^-1",
        "lobe_radial_peaks": [
            {
                "angle_deg": angle,
                "q_star": 0.2,
                "q_unit": "nm^-1",
                "branch_id": branch,
                "valid": True,
            }
            for branch, angle in enumerate(angles)
        ],
    }


def _axial_wrap(angle_deg: float) -> float:
    return (angle_deg + 90.0) % 180.0 - 90.0


def test_signed_stack_rotation_sets_mirrored_stack_axes_and_preserves_normal_gap() -> None:
    scene = build_lamellar_scene(
        _source(40.0, 140.0),
        {
            "mode": "single",
            "layer_count": 2,
            "stack_count": 2,
            "stack_rotation_deg": -10.0,
        },
    )
    period = 2.0 * np.pi / 0.2
    expected_alpha = {0: 10.0, 1: -10.0}

    for branch in (0, 1):
        indices = np.flatnonzero(scene.branch_ids == branch)
        first, second = indices
        normal = scene.orientations[first, :, 2]
        step = scene.centers[second] - scene.centers[first]
        step_angle = np.degrees(np.arctan2(step[1], step[0]))

        assert float(np.dot(step, normal)) == pytest.approx(period)
        assert _axial_wrap(step_angle - 90.0) == pytest.approx(expected_alpha[branch])

    populations = {row["branch_id"]: row for row in scene.metadata["populations"]}
    assert populations[0]["stack_rotation_alpha_deg"] == pytest.approx(10.0)
    assert populations[1]["stack_rotation_alpha_deg"] == pytest.approx(-10.0)
    alpha_record = next(
        row
        for row in scene.metadata["parameter_sources"]
        if row["name"] == "stack_rotation_deg"
    )
    assert alpha_record["source"] == "lamellar_settings"
    assert alpha_record["status"] == "assumed"
    assert "not inferred from the ellipse" in alpha_record["reason"]


def test_stack_rotation_does_not_fabricate_an_unobserved_branch_or_zero_tilt() -> None:
    one_branch = build_lamellar_scene(
        _source(90.0),
        {
            "mode": "multi",
            "layer_count": 2,
            "stack_count": 4,
            "stack_rotation_deg": -10.0,
        },
    )
    assert set(one_branch.branch_ids.tolist()) == {0}
    assert {row["branch_id"] for row in one_branch.metadata["populations"]} == {0}
    assert one_branch.metadata["populations"][0]["stack_rotation_alpha_deg"] == 0.0
    assert len(one_branch.centers) == 4


def test_manual_normals_are_not_relabelled_as_measured_by_stack_rotation() -> None:
    scene = build_lamellar_scene({}, {
        "period_source": "manual", "manual_angle_deg": 40.0,
        "stack_rotation_deg": -10.0, "layer_count": 2, "stack_count": 1,
    })
    population = scene.metadata["populations"][0]
    assert "manual normal" in population["orientation_assumption"]
    assert "measured lamellar normal" not in population["orientation_assumption"]
    assert population["status"] == "manual"


def test_stack_axis_rotation_stays_constant_when_normal_gaps_jitter() -> None:
    scene = build_lamellar_scene(
        _source(40.0),
        {
            "layer_count": 5,
            "stack_count": 1,
            "spacing_jitter_pct": 20.0,
            "stack_rotation_deg": -10.0,
            "seed": 19,
        },
    )
    normal = scene.orientations[0, :, 2]
    tangent = scene.orientations[0, :, 0]
    steps = np.diff(scene.centers, axis=0)
    normal_gaps = steps @ normal

    assert np.ptp(normal_gaps) > 0.0
    np.testing.assert_allclose(steps @ tangent / normal_gaps, np.tan(np.deg2rad(60.0)))


def test_none_keeps_legacy_geometry_and_settings_round_trip() -> None:
    source = _source(40.0, 140.0)
    legacy = build_lamellar_scene(
        source,
        {"layer_count": 3, "stack_count": 2, "lateral_shift_ratio": 0.25},
    )
    explicit_none = build_lamellar_scene(
        source,
        {
            "layer_count": 3,
            "stack_count": 2,
            "lateral_shift_ratio": 0.25,
            "stack_rotation_deg": None,
        },
    )
    np.testing.assert_array_equal(explicit_none.centers, legacy.centers)
    np.testing.assert_array_equal(explicit_none.orientations, legacy.orientations)
    settings = LamellarSettings.from_mapping({"stack_rotation_deg": -10.0})
    assert LamellarSettings.from_mapping(settings.to_dict()) == settings
    assert LamellarSettings().stack_rotation_deg is None
    positional = LamellarSettings(
        "single", "radial", 8, 12, 0.25, 4.0, 4.0, 0.0, 0.0, 24.0, 0.2, 13.0
    )
    assert positional.out_of_plane_deg == pytest.approx(13.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -90.0, 90.0])
def test_stack_rotation_requires_a_finite_angle_inside_ninety_degrees(value: float) -> None:
    with pytest.raises(ValueError, match="stack_rotation_deg"):
        LamellarSettings(stack_rotation_deg=value)


def test_stack_rotation_rejects_conflicting_manual_lateral_shift() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        LamellarSettings(stack_rotation_deg=-10.0, lateral_shift_ratio=0.1)


def test_singular_rotation_retains_frame_slot_and_allows_next_frame() -> None:
    import threading
    from butterfly_saxs.ui.lamellar_page import _build_scenes

    sources = [_source(40.0), _source(0.0), _source(140.0)]
    for index, source in enumerate(sources):
        source["frame_id"] = f"frame-{index}"
    scenes = _build_scenes(sources, {
        "layer_count": 2, "stack_count": 1, "stack_rotation_deg": 0.0,
    }, threading.Event())
    assert len(scenes) == 3
    assert scenes[0].centers.size > 0
    assert scenes[2].centers.size > 0
    failed = scenes[1]
    assert failed.status == "unavailable"
    assert "approach 90°" in failed.message
    assert "stack_rotation_singular_geometry" in failed.metadata["flags"]
    assert failed.source_identity["frame_id"] == "frame-1"
    assert any(row["name"] == "stack_rotation_deg" and row["status"] == "assumed"
               for row in failed.metadata["parameter_sources"])


def test_rotation_retains_intrinsic_shear_with_spread_and_out_of_plane_tilt() -> None:
    scene = build_lamellar_scene(_source(40.0), {
        "mode": "multi", "layer_count": 5, "stack_count": 4,
        "stack_rotation_deg": -10.0, "spacing_jitter_pct": 20.0,
        "spread_deg": 15.0, "out_of_plane_deg": 30.0, "seed": 19,
    })
    for stack in np.unique(scene.stack_ids):
        indices = np.flatnonzero(scene.stack_ids == stack)
        normal = scene.orientations[indices[0], :, 2]
        tangent = scene.orientations[indices[0], :, 0]
        steps = np.diff(scene.centers[indices], axis=0)
        gaps = steps @ normal
        period = 2.0 * np.pi / 0.2
        assert np.all(gaps >= 0.8 * period)
        assert np.all(gaps <= 1.2 * period)
        np.testing.assert_allclose(steps @ tangent / gaps, np.tan(np.deg2rad(60.0)))
    assert any("baseline source-plane geometry" in item for item in scene.metadata["assumptions"])


def test_rotation_export_preserves_assumption_and_geometry(tmp_path) -> None:
    import json
    from butterfly_saxs.lamellar_export import export_lamellar_scene

    scene = build_lamellar_scene(_source(40.0), {
        "layer_count": 2, "stack_count": 1, "stack_rotation_deg": -10.0,
    })
    outputs = export_lamellar_scene(scene, tmp_path / "rotation", language="en")
    settings = json.loads(outputs["settings"].read_text(encoding="utf-8"))
    assert settings["scene_settings"]["stack_rotation_deg"] == -10.0
    evidence = outputs["parameter_sources"].read_text(encoding="utf-8")
    assert "stack_rotation_deg" in evidence
    assert "assumed" in evidence
    assert "not inferred from the ellipse" in evidence
    with np.load(outputs["scene_arrays"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["centers"], scene.centers)
