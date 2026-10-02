from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs.lamellar_report_figures import (
    _analysis_text,
    _period_figure,
    _sequence_figure,
    render_lamellar_report_frame,
    render_lamellar_sequence_report,
)


def _scene(*, status: str = "schematic", with_geometry: bool = True) -> SimpleNamespace:
    vertices = np.empty((0, 8, 3), dtype=float)
    if with_geometry:
        signs = np.asarray(
            [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ],
            dtype=float,
        )
        vertices = np.asarray([signs * np.asarray((1.0, 2.0, 0.3)) / 2.0])
    return SimpleNamespace(
        vertices=vertices,
        centers=np.asarray([[0.0, 0.0, 0.0]]) if with_geometry else np.empty((0, 3)),
        sizes=np.asarray([[1.0, 2.0, 0.3]]) if with_geometry else np.empty((0, 3)),
        orientations=np.eye(3)[None, :, :] if with_geometry else np.empty((0, 3, 3)),
        stack_ids=np.asarray([0]) if with_geometry else np.empty((0,), dtype=int),
        branch_ids=np.asarray([0]) if with_geometry else np.empty((0,), dtype=int),
        colors=np.asarray([[0.2, 0.45, 0.72, 0.84]]) if with_geometry else np.empty((0, 4)),
        bounds=np.asarray([[-1.0, -1.5, -1.0], [1.0, 1.5, 1.0]]),
        length_unit="nm",
        metadata={"q_unit": "nm^-1", "frame_id": "frame-0"},
        status=status,
        message="",
        source_identity="frame-0",
        assumptions=[],
        settings={},
        parameter_sources=[],
        populations=[],
        scientific_boundary="schematic geometry",
        draw_axis_deg=90.0,
        reference_period=None,
    )


def _analysis(*, observed_directions=None, candidate=False, q_unit="nm^-1"):
    candidate_row = {
        "parameter": "Ln_from_minor_axis_nm",
        "value": None,
        "candidate_value": 3.4 if candidate else None,
        "unit": "nm",
        "status": "candidate" if candidate else "estimate",
        "branch_id": 0,
    }
    return {
        "schema_version": "lamellar-analysis.v1",
        "status": "candidate" if candidate else "ok",
        "source_status": "candidate" if candidate else "ok",
        "source_reason": "ellipse is weakly constrained" if candidate else "",
        "q_unit": q_unit,
        "reference_axis_deg": 0.0,
        "ellipse": {"a": 2.0, "b": 1.2, "axis_ratio": 0.6, "theta_deg": 12.0},
        "parameter_rows": [
            {"parameter": "a", "value": 2.0, "unit": q_unit, "status": "estimate"},
            {"parameter": "b", "value": 1.2, "unit": q_unit, "status": "estimate"},
            {"parameter": "axis_ratio", "value": 0.6, "unit": "", "status": "estimate"},
            {"parameter": "ellipse_axis_tilt_deg", "value": 12.0, "unit": "degree", "status": "estimate"},
            candidate_row,
            {"parameter": "Lz_from_draw_axis_nm", "value": 4.1, "unit": "nm", "status": "estimate", "branch_id": 0},
        ],
        "directional_rows": [
            {"branch_id": branch, "angle_deg": angle, "q_radius": 2.0 / (1 + branch * 0.2), "q_unit": q_unit, "period_nm": 2.8 / (1 + branch * 0.2), "relative_period": None, "status": "candidate" if candidate else "estimate", "observed_support": branch == 0 and angle in (0, 90)}
            for branch in (0, 1)
            for angle in (0.0, 90.0, 180.0, 270.0)
        ],
        "observed_directions": observed_directions if observed_directions is not None else [
            {"branch_id": 0, "angle_deg": 30.0, "q_radius": 1.8, "q_unit": q_unit, "period_nm": 3.5, "status": "accepted", "source": "ridge", "reason": "", "support_count": 16}
        ],
        "model_curves": [
            {
                "branch_id": branch,
                "qx": np.cos(np.deg2rad(np.linspace(0.0, 360.0, 37))) * (2.0 / (1 + branch * 0.2)),
                "qy": np.sin(np.deg2rad(np.linspace(0.0, 360.0, 37))) * (2.0 / (1 + branch * 0.2)),
                "angle_deg": np.linspace(0.0, 360.0, 37),
                "q_radius": 2.0 / (1.0 + branch * 0.2) + 0.1 * np.cos(np.deg2rad(np.linspace(0.0, 360.0, 37))),
                "q_unit": q_unit,
                "status": "candidate" if candidate else "fitted",
                "observed_support": branch == 0,
                "support_count": 2 if branch == 0 else 0,
                "source": "ellipse",
            }
            for branch in (0, 1)
        ],
    }


def test_frame_report_separates_full_model_curves_from_actual_observed_points(tmp_path):
    analysis = _analysis(candidate=True)
    figure = _period_figure(analysis, title="Frame 0 morphology")
    figure.canvas.draw()
    polar = next(axis for axis in figure.axes if axis.name == "polar")
    renderer = figure.canvas.get_renderer()
    title_box = figure._suptitle.get_window_extent(renderer)
    axis_title_boxes = [axis.title.get_window_extent(renderer) for axis in figure.axes if axis.title.get_text()]
    assert all(not title_box.overlaps(box) for box in axis_title_boxes)
    footer_box = figure.texts[-1].get_window_extent(renderer)
    assert footer_box.y1 < min(axis.get_tightbbox(renderer).y0 for axis in figure.axes)
    assert footer_box.x0 >= 0 and footer_box.x1 <= figure.bbox.x1
    assert polar.get_xlabel() == ""
    observed_scatter = next(collection for collection in polar.collections if len(collection.get_offsets()) == 1)
    assert np.asarray(observed_scatter.get_offsets()).shape == (1, 2)
    assert len(polar.lines) == 2  # only the two fitted model trajectories are lines
    assert "nm" in figure.axes[-1].get_ylabel()
    text = "\n".join(
        [item.get_text() for item in figure.texts]
        + [item.get_text() for axis in figure.axes for item in axis.texts]
    )
    assert "candidate" in text
    assert "q-space ellipse-axis angle" in text
    assert "Thickness, in-plane width and stack depth are schematic assumptions" in text
    labels = {item.get_text() for item in polar.get_legend().get_texts()}
    assert "candidate ellipse model · branch 0 · measured support present" in labels
    assert "candidate ellipse model · branch 1 · no measured support" in labels
    figure.clear()

    observed = np.arange(36, dtype=float).reshape(6, 6)
    qx, qy = np.meshgrid(np.linspace(-1, 1, 6), np.linspace(-1, 1, 6))
    valid = np.ones_like(observed, dtype=bool)
    valid[0, 0] = False
    paths = render_lamellar_report_frame(
        tmp_path,
        analysis=analysis,
        scene=_scene(),
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        formats=("png", "svg"),
        dpi=100,
        title="Frame 0 morphology",
    )
    assert set(paths) == {
        "lamellar_structure.png",
        "lamellar_structure.svg",
        "lamellar_periods.png",
        "lamellar_periods.svg",
    }
    assert all(path.exists() and path.stat().st_size > 500 for path in paths.values())
    periods_svg = paths["lamellar_periods.svg"].read_text(encoding="utf-8")
    assert "Directional period (nm)" in periods_svg
    assert "Retained observed ridge" in periods_svg
    assert "candidate" in periods_svg


def test_frame_report_does_not_export_a_blank_combined_scene_or_label_pixel_q_as_nm(tmp_path):
    analysis = _analysis(q_unit="pixel-q")
    analysis["directional_rows"] = [
        {"branch_id": 0, "angle_deg": angle, "q_radius": 2.0, "q_unit": "pixel-q", "period_nm": 3.1, "relative_period": 1.0, "status": "estimate"}
        for angle in (0.0, 90.0, 180.0, 270.0)
    ]
    paths = render_lamellar_report_frame(
        tmp_path,
        analysis=analysis,
        scene=_scene(with_geometry=False),
        observed=None,
        qx=None,
        qy=None,
        formats=("svg",),
    )
    assert set(paths) == {"lamellar_periods.svg"}
    svg = paths["lamellar_periods.svg"].read_text(encoding="utf-8")
    assert "Relative period (dimensionless)" in svg
    assert "Directional period (nm)" not in svg


def test_single_frame_overlay_converts_physical_q_units_and_reports_incompatible_units():
    analysis = _analysis()
    analysis["observed_directions"] = [
        {"branch_id": 0, "angle_deg": 45.0, "q_radius": 0.4, "q_unit": "Å^-1", "status": "observed", "support_count": 3}
    ]
    curve = analysis["model_curves"][0]
    curve["q_unit"] = "Å^-1"
    curve["q_radius"] = np.full(len(curve["angle_deg"]), 0.4)
    figure = _period_figure(analysis, title="Unit conversion")
    figure.canvas.draw()
    polar = next(axis for axis in figure.axes if axis.name == "polar")
    observed = polar.collections[0].get_offsets()
    assert np.asarray(observed)[0, 1] == pytest.approx(4.0)
    assert polar.lines[0].get_ydata()[0] == pytest.approx(4.0)
    figure.clear()

    incompatible = _analysis()
    incompatible["directional_rows"] = []
    incompatible["observed_directions"] = [
        {"branch_id": 0, "angle_deg": 45.0, "q_radius": 400.0, "q_unit": "pixel-q", "status": "observed", "support_count": 3}
    ]
    for item in incompatible["model_curves"]:
        item["q_unit"] = "pixel-q"
    figure = _period_figure(incompatible, title="Incompatible q units")
    polar = next(axis for axis in figure.axes if axis.name == "polar")
    assert not polar.lines
    assert not polar.collections
    text_ax = next(axis for axis in figure.axes if axis.name != "polar" and axis.get_title(loc="left") == "Ellipse parameters and source status")
    notes = "\n".join(item.get_text() for item in text_ax.texts)
    assert "Not overlaid" in notes
    assert "observed ridge points in q unit 'pixel-q'" in notes
    figure.clear()


def test_sequence_report_uses_declared_time_keeps_candidates_and_leaves_missing_directions_empty(tmp_path):
    first = _analysis()
    first["parameter_rows"][4]["value"] = 3.2
    first["parameter_rows"].append({"parameter": "Lz_from_draw_axis_nm", "value": 4.2, "unit": "nm", "status": "estimate", "branch_id": 1})
    second = _analysis(candidate=True, observed_directions=[])
    # The second frame has no measured ridge directions; the geometry can still
    # supply candidate parameter values but cannot create observed branch points.
    second["parameter_rows"] = [
        {"parameter": "Ln_from_minor_axis_nm", "value": None, "candidate_value": 3.6, "unit": "nm", "status": "candidate"},
        {"parameter": "Lz_from_draw_axis_nm", "value": 4.3, "candidate_value": None, "unit": "nm", "status": "candidate", "branch_id": 0},
        {"parameter": "Lz_from_draw_axis_nm", "value": 4.5, "candidate_value": None, "unit": "nm", "status": "candidate", "branch_id": 1},
        {"parameter": "axis_ratio", "value": 0.58, "unit": "", "status": "estimate"},
        {"parameter": "theta_deg", "value": 14.0, "unit": "degree", "status": "estimate"},
    ]
    frames = [
        {"frame_id": "frame-0", "time_s": 0.0, "q_unit": "nm^-1", "status": "ok", "lamellar_analysis": first},
        {"frame_id": "frame-1", "time_s": 2.0, "q_unit": "nm^-1", "status": "candidate", "lamellar_analysis": second},
    ]
    figure = _sequence_figure(frames, title="Lamellar sequence")
    assert figure is not None
    figure.canvas.draw()
    observed_axis = next(axis for axis in figure.axes if axis.get_title(loc="left") == "Retained observed q-direction angles")
    assert len(observed_axis.collections) == 1
    assert np.asarray(observed_axis.collections[0].get_offsets()).shape == (1, 2)
    assert observed_axis.get_xlabel() == "Time (s)"
    assert observed_axis.get_xlim()[0] <= 0.0 <= observed_axis.get_xlim()[1]
    lz_axis = next(axis for axis in figure.axes if axis.get_title(loc="left") == "Conditional draw-axis period Lz")
    lz_legend = {item.get_text() for item in lz_axis.get_legend().get_texts()}
    assert "Estimate · branch 0" in lz_legend
    assert "Candidate value · branch 0" in lz_legend
    assert "Estimate · branch 1" in lz_legend
    assert "Candidate value · branch 1" in lz_legend

    renderer = figure.canvas.get_renderer()
    title_box = figure._suptitle.get_window_extent(renderer)
    footer_box = figure.texts[-1].get_window_extent(renderer) if figure.texts else None
    assert title_box.x0 >= 0 and title_box.x1 <= figure.bbox.x1
    assert title_box.y1 <= figure.bbox.y1
    if footer_box is not None:
        assert footer_box.x0 >= 0 and footer_box.x1 <= figure.bbox.x1
        assert footer_box.y0 >= 0
    for axis in figure.axes:
        axis_title = axis.title.get_window_extent(renderer)
        assert axis_title.y0 >= 0
        assert axis_title.y1 <= figure.bbox.y1
        tight_box = axis.get_tightbbox(renderer)
        assert tight_box.x0 >= 0 and tight_box.x1 <= figure.bbox.x1
    assert all(not title_box.overlaps(axis.title.get_window_extent(renderer)) for axis in figure.axes if axis.title.get_text())
    figure.clear()

    paths = render_lamellar_sequence_report(tmp_path, frames=frames, formats=("svg",), dpi=100)
    assert set(paths) == {"lamellar_sequence_morphology.svg"}
    svg = paths["lamellar_sequence_morphology.svg"].read_text(encoding="utf-8")
    assert "Time (s)" in svg
    assert "Candidate value" in svg
    assert "Retained observed q-direction angles" in svg


def test_report_rejects_a_valid_mask_with_the_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="same two-dimensional shape"):
        render_lamellar_report_frame(
            tmp_path,
            analysis=_analysis(),
            scene=_scene(),
            observed=np.ones((4, 4)),
            qx=np.ones((4, 4)),
            qy=np.ones((4, 4)),
            valid_mask=np.ones((3, 3), dtype=bool),
            formats=("png",),
        )


def test_sequence_can_plot_explicit_axial_mean_and_range_summaries():
    analysis = _analysis(observed_directions=[])
    analysis["observed_direction_summary"] = [
        {"branch_id": 1, "axial_mean_deg": 2.0, "angle_min_deg": 178.0, "angle_max_deg": 10.0, "count": 8, "status": "observed"},
        {"branch_id": 0, "axial_mean_deg": 42.0, "angle_min_deg": 38.0, "angle_max_deg": 47.0, "count": 5, "status": "candidate"},
        {"branch_id": 0, "axial_mean_deg": 82.0, "angle_min_deg": 78.0, "angle_max_deg": 88.0, "count": 4, "status": "mixed"},
    ]
    figure = _sequence_figure(
        [{"time_s": 0.0, "q_unit": "nm^-1", "lamellar_analysis": analysis}],
        title="Axial direction summary",
    )
    assert figure is not None
    observed_axis = next(axis for axis in figure.axes if axis.get_title(loc="left") == "Mean observed q-direction angle and range")
    labels = {item.get_text() for item in observed_axis.get_legend().get_texts()}
    observed_label = "Axial mean and observed range · branch 1 · status: observed · full directions retained per frame"
    candidate_label = "Candidate-like mean and observed range · branch 0 · status: candidate · full directions retained per frame"
    mixed_label = "Mixed-status mean and observed range · branch 0 · status: mixed · full directions retained per frame"
    assert {observed_label, candidate_label, mixed_label} <= labels
    containers_by_label = {container.get_label(): container for container in observed_axis.containers}
    assert len(containers_by_label) == 3
    observed = containers_by_label[observed_label]
    candidate_line = containers_by_label[candidate_label].lines[0]
    mixed_line = containers_by_label[mixed_label].lines[0]
    assert candidate_line.get_marker() == "D"
    assert candidate_line.get_markerfacecolor() == "none"
    assert mixed_line.get_marker() == "s"
    assert mixed_line.get_markerfacecolor() == "none"
    bar_segments = observed.lines[2][0].get_segments()
    assert np.sort(np.asarray(bar_segments[0])[:, 1]) == pytest.approx([-2.0, 10.0])
    figure.clear()


def test_raw_provisional_undetermined_and_mixed_direction_statuses_keep_distinct_markers():
    analysis = _analysis(observed_directions=[
        {"branch_id": 0, "angle_deg": 12.0, "status": "provisional"},
        {"branch_id": 0, "angle_deg": 22.0, "status": "undetermined"},
        {"branch_id": 0, "angle_deg": 32.0, "status": "mixed"},
    ])
    figure = _sequence_figure(
        [{"time_s": 0.0, "q_unit": "nm^-1", "lamellar_analysis": analysis}],
        title="Direction status markers",
    )
    assert figure is not None
    observed_axis = next(axis for axis in figure.axes if axis.get_title(loc="left") == "Retained observed q-direction angles")
    collections = {collection.get_label(): collection for collection in observed_axis.collections}
    provisional = collections["Candidate-like ridge · branch 0 · status: provisional"]
    undetermined = collections["Candidate-like ridge · branch 0 · status: undetermined"]
    mixed = collections["Mixed-status ridge · branch 0 · status: mixed"]
    provisional_vertices = provisional.get_paths()[0].vertices
    undetermined_vertices = undetermined.get_paths()[0].vertices
    mixed_vertices = mixed.get_paths()[0].vertices
    assert np.max(np.abs(provisional_vertices)) == pytest.approx(np.sqrt(0.5))
    assert np.max(np.abs(undetermined_vertices)) == pytest.approx(np.sqrt(0.5))
    assert np.max(np.abs(mixed_vertices)) == pytest.approx(0.5)
    figure.clear()


def test_available_formal_value_is_not_styled_as_candidate_and_q_unit_text_is_ascii():
    analysis = _analysis()
    ellipse_period = next(row for row in analysis["parameter_rows"] if row["parameter"] == "Ln_from_minor_axis_nm")
    ellipse_period.update(value=3.4, candidate_value=4.1, status="available")
    major_axis = next(row for row in analysis["parameter_rows"] if row["parameter"] == "a")
    major_axis["unit"] = "nm⁻¹"

    lines = _analysis_text(analysis)
    assert "Ellipse semi-major axis a: 2 nm^-1" in lines
    assert "Conditional normal period Lₙ · branch 0: 3.4 nm" in lines
    assert not any("candidate Conditional normal period" in line for line in lines)


def test_radial_peak_fallback_is_unit_converted_and_keeps_unknown_branch_candidate_status():
    analysis = _analysis(observed_directions=[])
    analysis["model_curves"][0]["support_source"] = "lobe_radial_peaks"
    analysis["observed_radial_peaks"] = [
        {
            "branch_id": None,
            "angle_deg": 45.0,
            "q": 0.4,
            "q_unit": "Å^-1",
            "q_nm_inv": 4.0,
            "status": "undetermined",
            "source": "lobe_radial_peaks",
        }
    ]
    figure = _period_figure(analysis, title="Radial peak fallback")
    figure.canvas.draw()
    polar = next(axis for axis in figure.axes if axis.name == "polar")
    labels = {item.get_text() for item in polar.get_legend().get_texts()}
    radial_label = "Measured radial peak · branch unspecified · status: undetermined · lobe_radial_peaks"
    assert radial_label in labels
    assert "ellipse model · branch 0 · radial-peak support present" in labels
    radial_scatter = next(collection for collection in polar.collections if collection.get_label() == radial_label)
    assert np.asarray(radial_scatter.get_offsets())[0] == pytest.approx([np.deg2rad(45.0), 4.0])
    triangle = radial_scatter.get_paths()[0].vertices
    assert len(np.unique(triangle, axis=0)) == 3
    assert radial_scatter.get_facecolors().size == 0
    assert radial_scatter.get_edgecolors()[0, 3] == pytest.approx(1.0)
    figure.clear()

    ridge_analysis = _analysis(observed_directions=[
        {"branch_id": 0, "angle_deg": 30.0, "q_radius": 1.5, "q_unit": "nm^-1", "status": "accepted"}
    ])
    ridge_analysis["observed_radial_peaks"] = analysis["observed_radial_peaks"]
    ridge_figure = _period_figure(ridge_analysis, title="Ridge takes precedence")
    ridge_polar = next(axis for axis in ridge_figure.axes if axis.name == "polar")
    ridge_labels = {item.get_text() for item in ridge_polar.get_legend().get_texts()}
    assert not any("radial peak" in label.casefold() for label in ridge_labels)
    assert any(label.startswith("Retained observed ridge") for label in ridge_labels)
    ridge_figure.clear()
