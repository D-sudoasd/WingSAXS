"""Inspect actual plotted values, status markers and uncertainty geometry."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.path import Path
import numpy as np
import pytest

from butterfly_saxs.visualization import plot_parameter_evolution


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _markers(axis):
    return {
        item.get_label(): item
        for item in axis.collections
        if isinstance(item, PathCollection)
    }


def test_retains_finite_warning_failed_and_unassessed_estimates():
    rows = [
        {"time_s": 0, "status": "ok", "theta_deg": 10},
        {"time_s": 2, "status": "warning", "theta_deg": 11},
        {"time_s": 3, "status": "failed", "theta_deg": 12},
        {"time_s": 4, "status": "partial", "theta_deg": 13},
        {"time_s": 5, "status": "recovered", "theta_deg": 14},
        {"time_s": 6, "status": "bad_fit", "theta_deg": 15},
        {"time_s": 7, "theta_deg": 16},
    ]
    fig = plot_parameter_evolution(rows, parameters=("theta_deg",))
    axis = fig.axes[0]
    markers = _markers(axis)
    for row in rows:
        status = row.get("status", "unspecified").replace("_", " ")
        assert markers[f"Estimate ({status})"].get_offsets().tolist() == [
            [row["time_s"], row["theta_deg"]]
        ]
    assert axis.get_ylabel() == "Apparent ellipse axis tilt (deg)"
    assert axis.get_xlabel() == "Time (s)"
    assert len({markers[f"Estimate ({s})"].get_paths()[0].vertices.tobytes() for s in ("ok", "warning", "failed")}) == 3


def test_missing_values_break_line_and_remain_at_axes_relative_positions():
    rows = [
        {"time_s": 0, "status": "ok", "a": 10},
        {"time_s": 1, "status": "failed", "a": None},
        {"time_s": 2, "status": "ok", "a": 12},
        {"time_s": 3, "status": "ok", "a": float("inf")},
        {"time_s": 4, "status": "warning", "a": 14},
    ]
    axis = plot_parameter_evolution(rows, parameters=("a",)).axes[0]
    line = next(line for line in axis.lines if line.get_label() == "_sequence")
    np.testing.assert_equal(line.get_xdata(), [0, 1, 2, 3, 4])
    np.testing.assert_equal(line.get_ydata(), [10, np.nan, 12, np.nan, 14])
    segments = list(line.get_path().iter_segments(remove_nans=True))
    assert [code for _, code in segments] == [Path.MOVETO] * 3
    markers = _markers(axis)
    for status, x in (("failed", 1), ("ok", 3)):
        missing = markers[f"No finite estimate ({status})"]
        assert missing.get_offsets().tolist() == [[x, 0.025]]
        assert missing.get_offset_transform() == axis.get_xaxis_transform()
    # The bottom-strip ticks must not become spurious zero-valued measurements.
    assert axis.get_ylim()[0] > 0


def test_parameter_status_overrides_frame_status_independently_on_each_panel():
    rows = [
        {
            "time_s": 0, "status": "ok", "a": 10, "axis_ratio": 0.5,
            "a_status": "candidate", "axis_ratio_status": "available",
        },
        {"time_s": 1, "status": "warning", "a": 11, "axis_ratio": 0.6},
        {
            "time_s": 2, "status": "ok", "a": None, "axis_ratio": 0.7,
            "a_status": "unavailable", "axis_ratio_status": "candidate",
        },
    ]
    fig = plot_parameter_evolution(rows, parameters=("a", "axis_ratio"))
    radii = _markers(fig.axes[0])
    ratios = _markers(fig.axes[1])
    assert radii["Estimate (candidate)"].get_offsets().tolist() == [[0, 10]]
    assert radii["Estimate (warning)"].get_offsets().tolist() == [[1, 11]]
    assert radii["No finite estimate (unavailable)"].get_offsets().tolist() == [[2, 0.025]]
    assert ratios["Estimate (available)"].get_offsets().tolist() == [[0, 0.5]]
    assert ratios["Estimate (warning)"].get_offsets().tolist() == [[1, 0.6]]
    assert ratios["Estimate (candidate)"].get_offsets().tolist() == [[2, 0.7]]
    assert (
        ratios["Estimate (available)"].get_paths()[0].vertices.tobytes()
        != ratios["Estimate (candidate)"].get_paths()[0].vertices.tobytes()
    )
    assert "Estimate (ok)" not in radii
    assert "Estimate (ok)" not in ratios


@pytest.mark.parametrize("missing", [None, "", "invalid", np.nan, np.inf, -np.inf])
def test_incomplete_time_uses_one_explicit_frame_coordinate_system(missing):
    rows = [
        {"time_s": 100, "a": 10},
        {"time_s": missing, "a": 11},
        {"time_s": 900, "a": 12},
    ]
    axis = plot_parameter_evolution(rows, parameters=("a",)).axes[0]
    np.testing.assert_equal(axis.lines[0].get_xdata(), [0, 1, 2])
    assert axis.get_xlabel() == "Frame index (0-based row order; time incomplete)"
    np.testing.assert_equal(axis.get_xticks(), np.floor(axis.get_xticks()))


def test_finite_times_preserve_original_order_and_custom_x_is_honest():
    rows = [{"time_s": time, "a": 10} for time in [5.0, 1.0, 1.0]]
    axis = plot_parameter_evolution(rows, parameters=("a",)).axes[0]
    np.testing.assert_equal(axis.lines[0].get_xdata(), [5.0, 1.0, 1.0])
    custom = plot_parameter_evolution(rows, parameters=("a",), x_key="strain").axes[0]
    assert "strain incomplete" in custom.get_xlabel()
    assert "Time" not in custom.get_xlabel()


def test_bars_only_for_reported_finite_nonnegative_stderr():
    errors = [0.2, None, np.nan, -0.4, np.inf, "invalid", 0.0]
    rows = [
        {"time_s": index, "a": 5 + index, "a_stderr": error}
        for index, error in enumerate(errors)
    ]
    # The alternate field remains usable when the first is unavailable.
    rows.append({"time_s": 7, "a": 12, "a_stderr": None, "stderr_a": 0.3})
    axis = plot_parameter_evolution(rows, parameters=("a",)).axes[0]
    bars = [item for item in axis.containers if isinstance(item, ErrorbarContainer)]
    assert len(bars) == 1
    segments = bars[0].lines[2][0].get_segments()
    assert len(segments) == 3
    np.testing.assert_allclose(segments[0], [[0, 4.8], [0, 5.2]])
    np.testing.assert_allclose(segments[1], [[6, 11], [6, 11]])
    np.testing.assert_allclose(segments[2], [[7, 11.7], [7, 12.3]])
    assert len(_markers(axis)["Estimate (unspecified)"].get_offsets()) == 8


def test_no_stderr_does_not_create_any_errorbar_container():
    axis = plot_parameter_evolution(
        [{"time_s": 0, "a": 1}, {"time_s": 1, "a": 2}], parameters=("a",)
    ).axes[0]
    assert not axis.containers
    assert "Reported standard error" not in axis.get_legend_handles_labels()[1]


def test_known_units_and_explicit_overrides_do_not_guess_physical_q_units():
    parameters = ("a", "axis_ratio", "Ln_nm", "arbitrary_scale")
    row = dict.fromkeys(parameters, 1.0)
    fig = plot_parameter_evolution([row], parameters=parameters)
    assert [axis.get_ylabel() for axis in fig.axes] == [
        "Ellipse semi-major axis a (unit unspecified)",
        "Axis ratio b/a (dimensionless)",
        "Radial peak spacing 2π/q* (nm)",
        "arbitrary scale (unit unspecified)",
    ]
    figure = plot_parameter_evolution(
        [row], parameters=("a",),
        parameter_labels={"a": "Major-axis q radius"},
        parameter_units={"a": "pixel-q"},
    )
    assert figure.axes[0].get_ylabel() == "Major-axis q radius (pixel-q)"
    assert figure.axes[0].lines[0].get_ydata().tolist() == [1.0]


@pytest.mark.parametrize("rows, message", [([], "No frames"), ([{"a": None}], "No finite estimates")])
def test_empty_and_missing_series_are_explicit(rows, message):
    axis = plot_parameter_evolution(rows, parameters=("a",)).axes[0]
    assert [text.get_text() for text in axis.texts] == [message]
    assert not axis.lines


def test_render_representative_status_and_uncertainty_figure(tmp_path):
    rows = [
        {"time_s": 0, "status": "ok", "theta_deg": 10, "theta_deg_stderr": 0.2, "axis_ratio": 0.6},
        {"time_s": 1, "status": "warning", "theta_deg": 11, "axis_ratio": 0.62},
        {"time_s": None, "status": "failed", "theta_deg": None, "axis_ratio": None},
        {"time_s": 3, "status": "failed", "theta_deg": 15, "axis_ratio": 0.7},
        {"time_s": 4, "status": "ok", "theta_deg": 12, "theta_deg_stderr": 0.4, "axis_ratio": 0.64},
    ]
    target = tmp_path / "evolution.png"
    fig = plot_parameter_evolution(
        rows, parameters=("theta_deg", "axis_ratio"), output=target, dpi=120,
    )
    fig.canvas.draw()
    assert target.exists() and target.stat().st_size > 10000
    assert len(fig.axes) == 2
    assert len(fig.axes[0].get_legend_handles_labels()[1]) == 5
