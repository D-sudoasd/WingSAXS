from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.report_figures import (
    _fit_payloads,
    _fit_support_figure,
    _frame_fit_figure,
    _frame_overview,
    _intensity_norm,
    _local_profile_figure,
    _parameter_label,
    _profile_heatmap,
    _time_axis,
    render_frame_report,
    render_sequence_report,
)


def _frame_case():
    rows, cols = 24, 30
    q = np.linspace(-0.3, 0.3, cols)
    qx, qy = np.meshgrid(q, np.linspace(-0.24, 0.24, rows))
    # Slightly skewed coordinates exercise pcolormesh on an actual q grid.
    qx = qx + 0.015 * qy
    qy = qy + 0.008 * qx
    image = 0.2 + np.exp(-((qx / 0.12) ** 2 + (qy / 0.07) ** 2))
    valid = np.ones(image.shape, dtype=bool)
    valid[:2, :3] = False

    q_edges = np.linspace(0.02, 0.22, 6)
    angle_edges = np.linspace(-180, 180, 9)
    polar_mean = np.arange(40, dtype=float).reshape(5, 8) + 0.2
    polar_count = np.full((5, 8), 4.0)
    polar_coverage = np.full((5, 8), 0.75)
    polar_mean[1, 3] = np.nan
    polar_count[1, 3] = 0
    polar_coverage[1, 3] = 0
    radial_rows = [
        {
            "q_min": q_edges[i], "q_max": q_edges[i + 1],
            "q_center": (q_edges[i] + q_edges[i + 1]) / 2,
            "mean": 5.0 + i, "sum": 20.0 + i, "std": 1.0,
            "sem": 0.5, "count": 4, "coverage": 0.8,
        }
        for i in range(5)
    ]
    radial_rows[2]["mean"] = None
    angular_rows = [
        {
            "angle_min_deg": angle_edges[i], "angle_max_deg": angle_edges[i + 1],
            "angle_center_deg": (angle_edges[i] + angle_edges[i + 1]) / 2,
            "mean": 3.0 + i, "sum": 12.0 + i, "std": 0.8,
            "sem": 0.4, "count": 4, "coverage": 0.9,
        }
        for i in range(8)
    ]
    measurement = {
        "summary": {
            "frame_index": 7,
            "time_s": 1.25,
            "status": "ok",
            "q_unit": "nm^-1",
            "intensity_mean": float(np.mean(image[valid])),
        },
        "radial_rows": radial_rows,
        "angular_rows": angular_rows,
        "polar": {
            "mean": polar_mean,
            "sum": np.nan_to_num(polar_mean) * polar_count,
            "count": polar_count,
            "candidate_count": np.ones((5, 8)),
            "coverage": polar_coverage,
            "q_edges": q_edges,
            "angle_edges": angle_edges,
        },
    }
    return image, qx, qy, valid, measurement


def _sequence_case():
    q_edges = np.array([0.02, 0.06, 0.1, 0.14])
    angle_edges = np.array([-180, -90, 0, 90, 180])

    def radial(offset):
        return [
            {
                "q_min": q_edges[i], "q_max": q_edges[i + 1],
                "q_center": (q_edges[i] + q_edges[i + 1]) / 2,
                "mean": offset + i, "sem": 0.2, "std": 0.4,
                "count": 4, "coverage": 0.75,
            }
            for i in range(3)
        ]

    def angular(offset):
        return [
            {
                "angle_min_deg": angle_edges[i], "angle_max_deg": angle_edges[i + 1],
                "angle_center_deg": (angle_edges[i] + angle_edges[i + 1]) / 2,
                "mean": offset + i, "sem": 0.1, "count": 4, "coverage": 0.8,
            }
            for i in range(4)
        ]

    def tensor(scale):
        return {
            "status": "ok",
            "tensor": np.array([[0.8, 0.1], [0.1, 0.2]]) * scale,
            "anisotropy": 0.6,
            "principal_axis_deg": 9.2,
        }
    summaries = [
        {"frame_index": i, "time": float(i), "time_unit": "s", "status": "ok", "q_unit": "nm^-1", "intensity_sum": 10 + i, "intensity_mean": 2 + i, "in_plane_intensity_tensor": tensor(i + 1)}
        for i in range(3)
    ]
    radial_profiles = [radial(1), [], radial(3)]
    radial_profiles[2][1]["mean"] = None
    angular_profiles = [angular(2), [], angular(4)]
    parameter_rows = [
        {"frame_index": 0, "time": 0.0, "time_unit": "s", "parameter": "a", "value": 0.12, "stderr": 0.01, "unit": "nm^-1", "status": "ok"},
        {"frame_index": 1, "time": 1.0, "time_unit": "s", "parameter": "a", "value": None, "stderr": None, "unit": "nm^-1", "status": "failed"},
        {"frame_index": 2, "time": 2.0, "time_unit": "s", "parameter": "a", "value": 0.14, "stderr": 0.015, "unit": "nm^-1", "status": "ok"},
    ]
    return summaries, radial_profiles, angular_profiles, parameter_rows


def test_frame_report_exports_valid_formats_and_uses_direct_figures(tmp_path):
    import matplotlib
    import matplotlib.pyplot as plt

    backend_before = matplotlib.get_backend()
    figures_before = plt.get_fignums()
    image, qx, qy, valid, measurement = _frame_case()
    original_valid = valid.copy()
    model = image * 0.92
    paths = render_frame_report(
        tmp_path,
        image=image,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        q_unit="nm^-1",
        measurement=measurement,
        fit_record={"status": "candidate"},
        model=model,
        formats=("png", "svg", "pdf"),
        dpi=110,
    )

    assert len(paths) == 6
    assert all(path.exists() and path.stat().st_size > 500 for path in paths.values())
    assert paths["frame_0007_overview.png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths["frame_0007_overview.svg"].read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert paths["frame_0007_full2d_fit.pdf"].read_bytes().startswith(b"%PDF-")
    assert np.array_equal(valid, original_valid)
    assert matplotlib.get_backend() == backend_before
    assert plt.get_fignums() == figures_before

    overview = _frame_overview(image, qx, qy, "nm^-1", measurement, None, "")
    assert len(overview.axes) >= 5
    q_map = overview.axes[0]
    assert q_map.get_xlabel() == r"$q_x$ (nm$^{-1}$)"
    assert q_map.get_ylabel() == r"$q_y$ (nm$^{-1}$)"
    assert q_map.collections
    assert not q_map.images  # q maps retain the measured curvilinear coordinates.
    assert any("Descriptive SEM" in text.get_text() for text in overview.axes[1].get_legend().texts)
    overview.clear()


def test_frame_report_rejects_incompatible_shapes_and_polar_bins(tmp_path):
    image, qx, qy, valid, measurement = _frame_case()
    with pytest.raises(ValueError, match="same shape as image"):
        render_frame_report(tmp_path, image=image, qx=qx[:, :-1], qy=qy, valid_mask=valid, q_unit="nm^-1", measurement=measurement, formats=("png",))

    bad = dict(measurement)
    bad["polar"] = dict(measurement["polar"], mean=np.ones((4, 8)))
    with pytest.raises(ValueError, match="polar mean must have shape"):
        render_frame_report(tmp_path, image=image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", measurement=bad, formats=("png",))


def test_tiff_is_a_supported_output_format(tmp_path):
    image, qx, qy, valid, measurement = _frame_case()
    paths = render_frame_report(tmp_path, image=image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", measurement=measurement, formats=("tiff",), dpi=80)
    data = next(iter(paths.values())).read_bytes()
    assert data[:4] in {b"II*\x00", b"MM\x00*"}


def test_full2d_report_residual_uses_observed_minus_model_sign():
    image, qx, qy, valid, _ = _frame_case()
    model = image * 0.8
    source_residual = model - image  # The native batch NPZ convention.
    figure = _frame_fit_figure(image, qx, qy, valid, "nm^-1", model, source_residual, {}, "")
    residual_values = np.ma.compressed(np.ma.asarray(figure.axes[2].collections[0].get_array()))
    assert residual_values.size
    assert np.all(residual_values > 0)
    assert "observed − model" in figure.axes[2].get_title()
    figure.clear()


def test_native_local_profiles_and_fit_support_are_exported_without_refitting(tmp_path):
    image, qx, qy, valid, measurement = _frame_case()
    profile = {
        "point_id": "point-0007",
        "valid": True,
        "model": "single_gaussian",
        "reason": "accepted",
        "offset_q": [-0.2, 0.0, 0.2],
        "raw_intensity": [1.0, 3.0, 1.5],
        "fit_intensity": [1.2, 2.8, 1.4],
        "residual": [-0.2, 0.2, 0.1],
        "background": 0.5,
        "localization_sigma_q": 0.03,
    }
    fit_record = {
        "result": {
            "butterfly": {"profiles": {"point-0007": profile}},
            "ellipse_fit": {
                "point_diagnostics": [{
                    "point_id": "point-0007", "arc_id": 3, "used": True,
                    "normal_residual_q": 0.02, "projection_residual_q": -0.01,
                    "distance_q": 0.025, "localization_sigma_q": 0.03,
                }],
                "arc_diagnostics": [{
                    "arc_id": 3, "normal_residual_q_rms": 0.02,
                    "projection_rmse_q": 0.025,
                    "normal_residual_localization_ratio_rms": 0.67,
                    "support_endpoint_fraction": 0.1,
                    "manual_endpoint_fraction": 0.0,
                    "support_gap_count": 1, "infeasible_projection_count": 0,
                }],
            },
        }
    }
    profiles, point_rows, arc_rows = _fit_payloads(fit_record, measurement)
    assert len(profiles) == 1
    local = _local_profile_figure(profiles, 0, 1, q_unit="nm^-1", title="Frame 7")
    assert np.array_equal(local.axes[0].collections[0].get_offsets()[:, 1], np.asarray(profile["raw_intensity"]))
    assert np.array_equal(local.axes[0].lines[0].get_ydata(), np.asarray(profile["fit_intensity"]))
    assert len(local.axes) == 1
    local.canvas.draw()
    footer = next(text for text in local.texts if "normal_profiles.csv" in text.get_text())
    assert footer.get_window_extent().y1 < local.axes[0].get_window_extent().y0
    local.clear()

    support = _fit_support_figure(point_rows, arc_rows, "nm^-1", "Frame 7")
    assert "residual" in support.axes[0].get_title().lower()
    assert "nm^-1" in support.axes[1].get_ylabel()
    assert "dimensionless" in support.axes[3].get_ylabel()
    assert "fraction" in support.axes[4].get_ylabel()
    support.clear()

    paths = render_frame_report(tmp_path, image=image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", measurement=measurement, fit_record=fit_record, formats=("png", "svg"))
    assert "frame_0007_local_profiles_001.png" in paths
    assert "frame_0007_local_profile_residuals_001.svg" in paths
    assert "frame_0007_fit_support.png" in paths
    profile_svg = paths["frame_0007_local_profiles_001.svg"].read_text(encoding="utf-8")
    assert "normal_profiles.csv" in profile_svg
    assert "single_gaussian" in profile_svg


def test_profile_pages_omit_deferred_or_unaligned_measurements(tmp_path):
    image, qx, qy, valid, measurement = _frame_case()
    profiles = {
        "observed": {
            "point_id": "observed",
            "offset_q": [-1.0, 0.0, 1.0],
            "raw_intensity": [0.2, 1.1, 0.4],
            "fit_intensity": [0.3, 1.0, 0.4],
            "residual": [-0.1, 0.1, 0.0],
        },
        "deferred": {
            "point_id": "deferred",
            "offset_q": [-1.0, 0.0, 1.0],
            "raw_intensity": [],
            "fit_intensity": [0.3, 1.0, 0.4],
        },
        "unaligned": {
            "point_id": "unaligned",
            "offset_q": [-1.0, 0.0, 1.0],
            "raw_intensity": [0.2, 1.1],
        },
    }
    fit_record = {"result": {"butterfly": {"profiles": profiles}}}
    retained, _point_rows, _arc_rows = _fit_payloads(fit_record, measurement)
    assert len(retained) == 3  # The figure filter does not discard source diagnostics.
    paths = render_frame_report(
        tmp_path,
        image=image,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        q_unit="nm^-1",
        measurement=measurement,
        fit_record=fit_record,
        formats=("svg",),
    )
    profile_path = paths["frame_0007_local_profiles_001.svg"]
    source = profile_path.read_text(encoding="utf-8")
    assert "observed" in source
    assert "deferred" not in source
    assert "unaligned" not in source

    invalid_only = {"result": {"butterfly": {"profiles": {"deferred": profiles["deferred"], "unaligned": profiles["unaligned"]}}}}
    no_profile_paths = render_frame_report(
        tmp_path / "invalid-only",
        image=image,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        q_unit="nm^-1",
        measurement=measurement,
        fit_record=invalid_only,
        formats=("png",),
    )
    assert not any("local_profiles" in name or "local_profile_residuals" in name for name in no_profile_paths)


def test_sequence_heatmap_preserves_missing_frames_and_bins():
    summaries, radial, _, _ = _sequence_case()
    figure = _profile_heatmap(
        summaries,
        radial,
        kind="radial",
        unit="nm^-1",
        x_values=np.array([0.0, 1.0, 2.0]),
        x_label="Time (s)",
    )
    ax = figure.axes[0]
    mesh = ax.collections[0]
    values = np.ma.asarray(mesh.get_array())
    assert ax.get_xlabel() == r"$q$ (nm$^{-1}$)"
    assert ax.get_ylabel() == "Time (s)"
    assert np.ma.getmaskarray(values).sum() >= 4  # the empty frame row and missing bin remain gaps.
    figure.clear()


def test_sequence_report_exports_unit_grouped_trends_and_parameter_rows(tmp_path):
    summaries, radial, angular, parameters = _sequence_case()
    paths = render_sequence_report(
        tmp_path,
        summaries=summaries,
        radial_profiles=radial,
        angular_profiles=angular,
        parameter_rows=parameters,
        formats=("png", "svg"),
        dpi=100,
    )
    assert "sequence_radial_evolution_nm_1.png" in paths
    assert "sequence_angular_evolution_nm_1.svg" in paths
    assert "sequence_intensity_anisotropy_nm_1.png" in paths
    assert "sequence_parameter_a_nm_1.svg" in paths
    assert len(paths) == 8
    assert all(path.exists() and path.stat().st_size > 500 for path in paths.values())
    assert paths["sequence_radial_evolution_nm_1.svg"].read_text(encoding="utf-8").find("Time (s)") >= 0


def test_sequence_uses_frame_order_when_time_is_incomplete_and_checks_lengths(tmp_path):
    summaries, radial, angular, parameters = _sequence_case()
    summaries[1] = {**summaries[1], "time": None}
    paths = render_sequence_report(
        tmp_path,
        summaries=summaries,
        radial_profiles=radial,
        angular_profiles=angular,
        parameter_rows=parameters,
        formats=("svg",),
    )
    assert "sequence_parameter_a_nm_1.svg" in paths
    assert "Frame index" in paths["sequence_radial_evolution_nm_1.svg"].read_text(encoding="utf-8")
    assert "Frame index" in paths["sequence_parameter_a_nm_1.svg"].read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="must match summaries length"):
        render_sequence_report(tmp_path, summaries=summaries, radial_profiles=radial[:-1], angular_profiles=angular, parameter_rows=parameters, formats=("png",))


def test_time_axis_uses_only_declared_units_and_rejects_mixed_or_partial_units():
    seconds, seconds_label, used_time = _time_axis([{"time_s": 0.0}, {"time_s": 1.0}])
    assert np.array_equal(seconds, [0.0, 1.0])
    assert seconds_label == "Time (s)"
    assert used_time

    milliseconds, ms_label, used_time = _time_axis([{"time": 0.0, "time_unit": "ms"}, {"time": 5.0, "time_unit": "ms"}])
    assert np.array_equal(milliseconds, [0.0, 5.0])
    assert ms_label == "Time (ms)"
    assert used_time

    _, unspecified_label, used_time = _time_axis([{"time": 0.0}, {"time": 1.0}])
    assert unspecified_label == "Time (unit unspecified)"
    assert used_time

    for rows in (
        [{"time": 0.0, "time_unit": "ms"}, {"time": 1.0, "time_unit": "s"}],
        [{"time": 0.0, "time_unit": "ms"}, {"time": 1.0}],
    ):
        indices, label, used_time = _time_axis(rows)
        assert np.array_equal(indices, [0.0, 1.0])
        assert label.startswith("Frame index")
        assert not used_time


def test_empty_numeric_unit_groups_are_skipped_and_supported_unit_keeps_failed_frame_gap(tmp_path):
    summaries = [
        {"frame_index": 0, "time": 0.0, "time_unit": "ms", "q_unit": "pixel-q", "status": "warning"},
        {"frame_index": 1, "time": 1.0, "time_unit": "ms", "q_unit": "unknown", "status": "failed"},
        {"frame_index": 2, "time": 2.0, "time_unit": "ms", "q_unit": "pixel_q", "status": "warning"},
    ]
    radial = [
        {"q_center": [2.0, 3.0], "mean": [0.5, 0.2], "count": [10, 8]},
        [],
        {"q_center": [2.0, 3.0], "mean": [0.7, 0.1], "count": [9, 7]},
    ]
    angular = [[], [], []]
    paths = render_sequence_report(
        tmp_path,
        summaries=summaries,
        radial_profiles=radial,
        angular_profiles=angular,
        parameter_rows=[],
        formats=("png",),
        dpi=100,
    )
    assert set(paths) == {"sequence_radial_evolution_pixel_q.png"}

    figure = _profile_heatmap(summaries, radial, kind="radial", unit="pixel-q", x_values=np.array([0.0, 1.0, 2.0]), x_label="Time (ms)")
    mesh = figure.axes[0].collections[0]
    values = np.ma.asarray(mesh.get_array())
    assert values.shape == (3, 2)
    assert np.ma.getmaskarray(values)[1].all()
    figure.clear()


def test_candidate_value_is_plotted_but_string_parameters_and_empty_physical_units_are_skipped(tmp_path):
    summaries = [{"frame_index": 0, "time": 0.0, "time_unit": "s", "q_unit": "pixel-q", "status": "warning"}]
    parameters = [
        {"frame_index": 0, "parameter": "candidate_radius", "value": None, "candidate_value": 3.2, "unit": "nm", "status": "warning"},
        {"frame_index": 0, "parameter": "q_unit", "value": "pixel-q", "candidate_value": "pixel-q", "unit": "unit unspecified"},
        {"frame_index": 0, "parameter": "q_star_source", "value": "observed", "candidate_value": "observed", "unit": "unit unspecified"},
        {"frame_index": 0, "parameter": "Ln_from_minor_axis_nm", "value": None, "candidate_value": None, "unit": "nm"},
    ]
    paths = render_sequence_report(
        tmp_path,
        summaries=summaries,
        radial_profiles=[[]],
        angular_profiles=[[]],
        parameter_rows=parameters,
        formats=("svg",),
    )
    assert set(paths) == {"sequence_parameter_candidate_radius_nm.svg"}
    source = paths["sequence_parameter_candidate_radius_nm.svg"].read_text(encoding="utf-8")
    assert "Candidate value only" in source


def test_time_series_footers_are_reserved_and_radial_sem_note_is_outside_plot(tmp_path):
    image, qx, qy, valid, measurement = _frame_case()
    overview = _frame_overview(image, qx, qy, "nm^-1", measurement, None, "")
    overview.canvas.draw()
    footer = next(text for text in overview.texts if "descriptive within-bin SEM" in text.get_text())
    assert "not fit uncertainty" in footer.get_text()
    assert "Polar mean: Log color scale" in footer.get_text()
    assert not any("Band is within-bin" in text.get_text() for text in overview.axes[1].texts)
    assert footer.get_window_extent().y1 < min(axis.get_window_extent().y0 for axis in overview.axes)
    overview.clear()

    summaries, radial, angular, _ = _sequence_case()
    heatmap = _profile_heatmap(summaries, radial, kind="radial", unit="nm^-1", x_values=np.array([0.0, 1.0, 2.0]), x_label="Time (s)")
    heatmap.canvas.draw()
    assert heatmap.axes[-1].get_ylabel() == "Bin mean intensity (input units)"
    assert any("Linear color scale" in text.get_text() for text in heatmap.texts)
    note = heatmap.texts[0]
    assert note.get_window_extent().y1 < heatmap.axes[0].get_window_extent().y0
    heatmap.clear()

    log_summaries = [
        {"frame_index": 0, "time_s": 0.0, "q_unit": "pixel-q"},
        {"frame_index": 1, "time_s": 1.0, "q_unit": "pixel-q"},
    ]
    log_profiles = [
        {"q_center": [1.0, 2.0], "mean": [0.001, 1.0]},
        {"q_center": [1.0, 2.0], "mean": [0.01, 0.8]},
    ]
    log_figure = _profile_heatmap(log_summaries, log_profiles, kind="radial", unit="pixel-q", x_values=np.array([0.0, 1.0]), x_label="Time (s)")
    log_figure.canvas.draw()
    assert log_figure.axes[-1].get_ylabel() == "Bin mean intensity (input units)"
    assert any("Log color scale; values above the 99.5th percentile are clipped." in text.get_text() for text in log_figure.texts)
    log_figure.clear()


def test_parameter_labels_keep_radial_period_and_conditional_ellipse_spacing_distinct():
    assert _parameter_label("Ln_nm", "nm") == "Radial peak period 2π/q* (nm)"
    assert _parameter_label("Ln_from_minor_axis_nm", "nm") == "Conditional ellipse normal spacing Ln (nm)"
    assert _parameter_label("Lz_from_draw_axis_nm", "nm") == "Conditional ellipse draw-axis spacing Lz (nm)"
    assert _parameter_label("L_from_observed_radius_nm", "nm") == "Observed ring period 2π/q* (nm)"


def test_profile_page_title_does_not_overlap_panel_titles(tmp_path):
    from butterfly_saxs.report_figures import _local_profile_figure

    profile = {"point_id": "ridge-017669f9dc460fb9", "model": "single_gaussian",
               "offset_q": [-1.0, 0.0, 1.0], "raw_intensity": [1.0, 2.0, 1.0],
               "fit_intensity": [1.0, 1.9, 1.0]}
    figure = _local_profile_figure([profile] * 8, 0, 5, q_unit="pixel-q", title="Frame 0: fixture_0")
    figure.canvas.draw()
    page_title = figure._suptitle.get_window_extent()
    assert all(axis.title.get_window_extent().y1 < page_title.y0 for axis in figure.axes)
    figure.savefig(tmp_path / "profiles.png", dpi=100, bbox_inches="tight")
    figure.clear()


def test_log_normalization_is_labeled_explicitly():
    norm, label = _intensity_norm(np.geomspace(1e-4, 1.0, 100))
    assert norm.__class__.__name__ == "LogNorm"
    assert "log color scale" in label
    assert "99.5th percentile" in label
