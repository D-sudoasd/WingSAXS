"""Regression tests for Phase 2 core physics, geometry, fitting, IO, and analysis bugfixes."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import threading
import warnings

import numpy as np
from PIL import Image
import pytest
import tifffile

from butterfly_saxs.arc_geometry import _parse_points, _project_point_to_support, fit_arc_ellipses
from butterfly_saxs.arc_support import freeze_observed_support
from butterfly_saxs.auto_advisor import (
    _find_circular_peaks,
    _refine_reflection_bisector,
    advise_frame,
    advise_from_arrays,
)
from butterfly_saxs.azimuthal_analysis import (
    AzimuthalPeak,
    AzimuthalProfile,
    _detect_peak_seeds,
    evaluate_azimuthal_model,
    export_azimuthal_csv_bundle,
    fit_azimuthal_peaks,
)
from butterfly_saxs.cancellation import AnalysisCancelled
from butterfly_saxs.density2d import (
    Density2DProfile,
    evaluate_density2d_model,
    export_density2d_bundle,
    fit_density_fluctuation,
)
from butterfly_saxs.ellipse import (
    EllipseGeometry,
    _canonical_parameter_mapping,
    _multistart_parameter_sets,
    ellipse_points,
    fit_ellipse,
)
from butterfly_saxs.geometry import GeometryMaps, build_geometry_cache
from butterfly_saxs.intensity import (
    _robust_cost,
    double_ellipse_intensity,
    fit_double_ellipse_intensity,
    parameter_values,
)
from butterfly_saxs.io import load_image
from butterfly_saxs.legacy_saxs import legacy_fusion_geometry_map
from butterfly_saxs.local_measurement import (
    export_local_measurements_json,
    extract_line_profile,
    measure_point,
    measure_roi_orientation,
)
from butterfly_saxs.observables import (
    RidgePoint,
    _normalized_q_unit,
    _quadratic_peak,
    _reject_ridge_point,
    _symmetry_diagnostics,
    fit_symmetric_double_ellipse,
    measure_angular_spectrum,
    measure_radial_ridges,
)
from butterfly_saxs.parameters import ParameterSet, ParameterSpec
from butterfly_saxs.public_ellipse import canonical_ellipse_payload
from butterfly_saxs.saxs_physics import (
    fit_c2_bridge_four_lobe,
    fit_power_law_coarsening,
    measure_perpendicular_streak_width,
    run_window_sensitivity_audit,
)
from butterfly_saxs.synthetic import make_butterfly_sequence


# ---------------------------------------------------------------------------
# 1. intensity.py
# ---------------------------------------------------------------------------


def test_intensity_radial_fwhm_expanded_and_popped() -> None:
    params = parameter_values({"a": 0.4, "radial_fwhm": 0.12})
    assert "radial_fwhm" not in params
    assert params["radial_sigma"] == pytest.approx(0.12 / (2.0 * math.sqrt(2.0 * math.log(2.0))))
    assert params["radial_gamma"] == pytest.approx(0.06)

    grid = np.linspace(-0.8, 0.8, 33)
    qx, qy = np.meshgrid(grid, grid)
    qmap = {"qx": qx, "qy": qy}
    truth = parameter_values(
        {
            "a": 0.45,
            "b": 0.32,
            "theta": 0.2,
            "amplitude_plus": 40.0,
            "amplitude_minus": 40.0,
            "radial_sigma": 0.04,
            "eta": 0.0,
            "background": 1.0,
        }
    )
    img = double_ellipse_intensity(qx, qy, truth)

    # Case (a): ParameterSet with radial_fwhm=0.20 (vary=True) and NO radial_sigma/radial_gamma keys
    pset_only_fwhm = ParameterSet(
        {
            "a": ParameterSpec(0.43, min=0.2, max=0.7),
            "b": ParameterSpec(0.30, min=0.1, max=0.65),
            "theta": ParameterSpec(0.2, vary=False),
            "lobe_angle": ParameterSpec(truth["lobe_angle"], vary=False),
            "angular_width": ParameterSpec(truth["angular_width"], vary=False),
            "amplitude_plus": ParameterSpec(35.0, min=0.1, max=200.0),
            "amplitude_minus": ParameterSpec(35.0, min=0.1, max=200.0),
            "radial_fwhm": ParameterSpec(0.20, min=0.02, max=0.5, vary=True),
            "eta": ParameterSpec(0.0, vary=False),
            "background": ParameterSpec(1.0, min=0.0, max=20.0),
            "background_slope": ParameterSpec(0.0, vary=False),
            "background_curvature": ParameterSpec(0.0, vary=False),
            "background_amplitude": ParameterSpec(0.0, vary=False),
            "background_width": ParameterSpec(1.0, vary=False),
        }
    )
    res_a = fit_double_ellipse_intensity(img, qmap, initial=pset_only_fwhm)
    assert res_a.success
    assert "radial_fwhm" not in res_a.parameters
    assert "radial_sigma" in res_a.parameters
    assert "radial_gamma" in res_a.parameters
    assert res_a.parameters.resolve()["radial_sigma"] == pytest.approx(0.04, rel=0.03)
    assert math.isfinite(res_a.condition_number) and res_a.condition_number < 1e8

    # Case (b): ParameterSet with radial_fwhm=0.20 (vary=True) alongside stale radial_sigma=0.10 (vary=False)
    pset_both = ParameterSet(
        {
            "a": ParameterSpec(0.43, min=0.2, max=0.7),
            "b": ParameterSpec(0.30, min=0.1, max=0.65),
            "theta": ParameterSpec(0.2, vary=False),
            "lobe_angle": ParameterSpec(truth["lobe_angle"], vary=False),
            "angular_width": ParameterSpec(truth["angular_width"], vary=False),
            "amplitude_plus": ParameterSpec(35.0, min=0.1, max=200.0),
            "amplitude_minus": ParameterSpec(35.0, min=0.1, max=200.0),
            "radial_fwhm": ParameterSpec(0.20, min=0.02, max=0.5, vary=True),
            "radial_sigma": ParameterSpec(0.10, vary=False),
            "radial_gamma": ParameterSpec(0.10, vary=False),
            "eta": ParameterSpec(0.0, vary=False),
            "background": ParameterSpec(1.0, min=0.0, max=20.0),
            "background_slope": ParameterSpec(0.0, vary=False),
            "background_curvature": ParameterSpec(0.0, vary=False),
            "background_amplitude": ParameterSpec(0.0, vary=False),
            "background_width": ParameterSpec(1.0, vary=False),
        }
    )
    res_b = fit_double_ellipse_intensity(img, qmap, initial=pset_both)
    assert res_b.success
    assert "radial_fwhm" not in res_b.parameters
    assert "radial_sigma" in res_b.parameters
    assert res_b.parameters.resolve()["radial_sigma"] == pytest.approx(0.04, rel=0.03)
    assert math.isfinite(res_b.condition_number) and res_b.condition_number < 1e8


def test_intensity_alias_radian_degree_conflict_raises() -> None:
    with pytest.raises(ValueError, match="not both"):
        parameter_values({"tilt": 0.2, "theta_deg": 45.0})


def test_intensity_autoscale_respects_parameter_set_and_explicit_bounds() -> None:
    grid = np.linspace(-0.8, 0.8, 25)
    qx, qy = np.meshgrid(grid, grid)
    qmap = {"qx": qx, "qy": qy}
    truth = parameter_values(
        {
            "a": 0.4,
            "b": 0.3,
            "theta": 0.0,
            "amplitude_plus": 500.0,
            "amplitude_minus": 500.0,
            "radial_sigma": 0.07,
            "eta": 0.0,
            "background": 100.0,
        }
    )
    img = double_ellipse_intensity(qx, qy, truth)
    pset = ParameterSet(
        {
            "a": ParameterSpec(0.4, vary=False),
            "b": ParameterSpec(0.3, vary=False),
            "theta": ParameterSpec(0.0, vary=False),
            "lobe_angle": ParameterSpec(truth["lobe_angle"], vary=False),
            "angular_width": ParameterSpec(truth["angular_width"], vary=False),
            "amplitude_plus": ParameterSpec(10.0, min=0.0, max=50.0),
            "amplitude_minus": ParameterSpec(10.0, min=0.0, max=50.0),
            "radial_sigma": ParameterSpec(0.07, vary=False),
            "radial_gamma": ParameterSpec(0.05, vary=False),
            "eta": ParameterSpec(0.0, vary=False),
            "background": ParameterSpec(10.0, min=0.0, max=30.0),
            "background_slope": ParameterSpec(0.0, vary=False),
            "background_curvature": ParameterSpec(0.0, vary=False),
            "background_amplitude": ParameterSpec(0.0, vary=False),
            "background_width": ParameterSpec(1.0, vary=False),
        }
    )
    # auto_scale_initial must clamp within ParameterSet finite bounds rather than raising
    res = fit_double_ellipse_intensity(img, qmap, initial=pset, auto_scale_initial=True)
    resolved_clamped = res.parameters.resolve()
    assert resolved_clamped["amplitude_plus"] <= 50.0 + 1e-9
    assert resolved_clamped["background"] <= 30.0 + 1e-9

    # Explicit bounds wider than ParameterSet spec bounds must not raise on resolution
    res_wide = fit_double_ellipse_intensity(
        img,
        qmap,
        initial=pset,
        bounds={
            "amplitude_plus": (0.0, 1000.0),
            "amplitude_minus": (0.0, 1000.0),
            "background": (0.0, 500.0),
        },
        auto_scale_initial=True,
    )
    assert res_wide.success
    wide_vals = parameter_values(res_wide.parameters)
    assert wide_vals["amplitude_plus"] == pytest.approx(500.0, rel=0.05)


def test_intensity_narrow_bounds_no_clip_inversion() -> None:
    grid = np.linspace(-0.8, 0.8, 21)
    qx, qy = np.meshgrid(grid, grid)
    qmap = {"qx": qx, "qy": qy}
    truth = parameter_values({"a": 0.4, "axis_ratio": 0.7, "amp_upper": 20.0, "amp_lower": 20.0, "bg0": 1.0})
    img = double_ellipse_intensity(qx, qy, truth)
    fixed_all_but_a = [k for k in truth if k != "a"]
    res = fit_double_ellipse_intensity(
        img,
        qmap,
        initial=truth,
        fixed=fixed_all_but_a,
        bounds={"a": (0.4, 0.4 + 1e-13)},
    )
    assert res.success
    assert 0.4 <= res.values["a"] <= 0.4 + 1e-13


def test_intensity_zero_free_params_uses_robust_cost() -> None:
    grid = np.linspace(-0.6, 0.6, 17)
    qx, qy = np.meshgrid(grid, grid)
    qmap = {"qx": qx, "qy": qy}
    truth = parameter_values({"a": 0.35, "axis_ratio": 0.7, "amp_upper": 25.0, "amp_lower": 25.0, "bg0": 2.0})
    img = double_ellipse_intensity(qx, qy, truth)
    img[0, 0] += 5000.0  # strong outlier so linear cost != soft_l1 cost
    all_keys = list(truth.keys())
    res_lin = fit_double_ellipse_intensity(img, qmap, initial=truth, fixed=all_keys, robust_loss="linear")
    res_rob = fit_double_ellipse_intensity(img, qmap, initial=truth, fixed=all_keys, robust_loss="soft_l1", f_scale=1.0)
    obs_scale = float(np.std(img))
    expected_rob = _robust_cost((res_rob.residual / obs_scale).ravel(), "soft_l1", 1.0)
    assert res_rob.cost == pytest.approx(expected_rob)
    assert res_rob.cost < res_lin.cost


# ---------------------------------------------------------------------------
# 2. ellipse.py
# ---------------------------------------------------------------------------


def test_ellipse_canonical_mapping_order_independent_alias_conflict() -> None:
    with pytest.raises(ValueError, match="duplicate ellipse parameter aliases"):
        _canonical_parameter_mapping({"cx": 0.0, "center_x": 0.1})
    with pytest.raises(ValueError, match="duplicate ellipse parameter aliases"):
        _canonical_parameter_mapping({"center_x": 0.1, "cx": 0.0})


def test_ellipse_multistart_saxs_scale_positive_and_skips_infeasible_tied() -> None:
    base = ParameterSet(
        {
            "a": ParameterSpec(0.20, min=1e-6, max=np.inf),
            "axis_ratio": ParameterSpec(0.65, min=0.1, max=1.0),
            "theta": ParameterSpec(0.1, min=-math.pi, max=math.pi),
            "cx": ParameterSpec(0.01, min=-np.inf, max=np.inf),
            "cy": ParameterSpec(-0.01, min=-np.inf, max=np.inf),
        }
    )
    starts = _multistart_parameter_sets(base, 6)
    assert len(starts) == 6
    for pset in starts:
        vals = pset.resolve()
        # Positive scale param 'a' must stay strictly SAXS-scale, never collapse near eps (~1e-16)
        assert 0.05 <= vals["a"] <= 0.50
        # Unbounded center must stay near SAXS scale, not jump +-2.0
        assert abs(vals["cx"]) < 0.5 and abs(vals["cy"]) < 0.5

    # Tied parameter with finite bounds: infeasible starts must be skipped cleanly instead of raising
    tied_base = ParameterSet(
        {
            "a": ParameterSpec(0.20, min=1e-6, max=np.inf),
            "b_tied": ParameterSpec(0.10, expr="0.5 * a", min=0.09, max=0.11),
            "axis_ratio": ParameterSpec(0.70, min=0.2, max=1.0),
            "theta": ParameterSpec(0.0, min=-math.pi, max=math.pi),
            "cx": ParameterSpec(0.0, vary=False),
            "cy": ParameterSpec(0.0, vary=False),
        }
    )
    tied_starts = _multistart_parameter_sets(tied_base, 5)
    assert len(tied_starts) >= 1
    for pset in tied_starts:
        resolved = pset.resolve()
        assert 0.09 <= resolved["b_tied"] <= 0.11


def test_ellipse_fit_forwards_residual_to_run_fit_and_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    import butterfly_saxs.ellipse as ellipse_mod

    geom = EllipseGeometry(cx=0.0, cy=0.0, a=0.4, axis_ratio=0.625, theta=0.2)
    angles = np.linspace(0.2, math.pi - 0.2, 24)
    pts = ellipse_points(geom, angles)
    captured: dict[str, str] = {}

    orig_run_fit = ellipse_mod._run_fit
    orig_coverage = ellipse_mod._coverage

    def spy_run_fit(*args, **kwargs):
        captured["run_fit_residual"] = kwargs.get("residual", "MISSING")
        return orig_run_fit(*args, **kwargs)

    def spy_coverage(*args, **kwargs):
        captured["coverage_residual"] = kwargs.get("residual", "MISSING")
        return orig_coverage(*args, **kwargs)

    monkeypatch.setattr(ellipse_mod, "_run_fit", spy_run_fit)
    monkeypatch.setattr(ellipse_mod, "_coverage", spy_coverage)

    res = fit_ellipse(pts, residual="geometric")
    assert res.success
    assert captured["run_fit_residual"] == "geometric"
    assert captured["coverage_residual"] == "geometric"


# ---------------------------------------------------------------------------
# 3. arc_geometry.py & arc_support.py
# ---------------------------------------------------------------------------


def test_arc_geometry_project_to_support_uses_rectangle_intervals() -> None:
    # Unit circle: point is at (0.0, 1.05) -> unconstrained closest point is t=pi/2 (0.0, 1.0).
    # Support rectangle spans qx in [0.25, 0.60], qy in [0.70, 1.10]: excludes (0, 1) but
    # intersects the upper arc near t = arccos(0.25) (~1.318 rad).
    geom = EllipseGeometry(cx=0.0, cy=0.0, a=1.0, axis_ratio=1.0, theta=0.0)
    rect = {
        "frame_origin_q": [0.0, 0.0],
        "tangent_q": [1.0, 0.0],
        "normal_q": [0.0, 1.0],
        "tangent_bounds": [0.25, 0.60],
        "normal_bounds": [0.70, 1.10],
    }
    proj = _project_point_to_support(np.array([0.0, 1.05]), geom, "upper", None, (rect,))
    assert proj["projection_valid"] is True
    assert proj["support_infeasible"] is False
    assert proj["support_violation_q"] == 0.0
    px, py = geom.point(proj["t"])
    assert 0.25 - 1e-6 <= px <= 0.60 + 1e-6
    assert 0.70 - 1e-6 <= py <= 1.10 + 1e-6


def test_arc_geometry_infeasible_penalty_reflects_rectangle_violation() -> None:
    # Ellipse passes directly through (0.0, 1.0), so unconstrained distance is 0,
    # while the support rectangle sits strictly outside the ellipse at [0.2, 0.5] x [1.2, 1.5].
    geom = EllipseGeometry(cx=0.0, cy=0.0, a=1.0, axis_ratio=1.0, theta=0.0)
    disjoint_rect = {
        "frame_origin_q": [0.0, 0.0],
        "tangent_q": [1.0, 0.0],
        "normal_q": [0.0, 1.0],
        "tangent_bounds": [0.2, 0.5],
        "normal_bounds": [1.2, 1.5],
    }
    proj = _project_point_to_support(np.array([0.0, 1.0]), geom, "upper", None, (disjoint_rect,))
    assert proj["projection_valid"] is False
    assert proj["support_infeasible"] is True
    # Penalty must reflect the rectangle violation (~0.28), not collapse to 1e-12
    assert proj["support_violation_q"] > 0.15
    assert proj["objective_distance"] > 0.15


def test_arc_geometry_parse_points_implicit_point_id_alignment() -> None:
    # Points lack explicit "point_id" (only implicit list index or "id"), relying on
    # fallback alignment between arc_support._point_id and arc_geometry._parse_points
    geom0 = EllipseGeometry(cx=0.0, cy=0.0, a=0.5, axis_ratio=0.6, theta=0.2)
    geom1 = EllipseGeometry(cx=0.0, cy=0.0, a=0.5, axis_ratio=0.6, theta=-0.2)
    ts_u = np.linspace(0.4, math.pi - 0.4, 6)
    ts_l = np.linspace(math.pi + 0.4, 2.0 * math.pi - 0.4, 6)
    records: list[dict[str, float | int | str | bool]] = []
    for arc_id, branch_id, side, geom, ts in (
        (0, 0, "upper", geom0, ts_u),
        (1, 0, "lower", geom0, ts_l),
        (2, 1, "upper", geom1, ts_u),
        (3, 1, "lower", geom1, ts_l),
    ):
        for x, y in ellipse_points(geom, ts):
            records.append(
                {
                    "qx": float(x),
                    "qy": float(y),
                    "branch_id": branch_id,
                    "side": side,
                    "arc_id": arc_id,
                    "accepted": True,
                    "localization_sigma_q": 0.02,
                }
            )
    arcs = [
        {"arc_id": 0, "branch_id": 0, "side": "upper", "localization_sigma_q": 0.02},
        {"arc_id": 1, "branch_id": 0, "side": "lower", "localization_sigma_q": 0.02},
        {"arc_id": 2, "branch_id": 1, "side": "upper", "localization_sigma_q": 0.02},
        {"arc_id": 3, "branch_id": 1, "side": "lower", "localization_sigma_q": 0.02},
    ]
    frozen_meta = freeze_observed_support(records, arcs)
    parsed = _parse_points(records, support_registry=frozen_meta["support_registry"])
    parsed_pts = parsed[0]
    diagnostics = parsed[5]
    assert len(parsed_pts) == 24
    assert all(d["used"] and not d["excluded"] for d in diagnostics)
    fit = fit_arc_ellipses(records, multistart=1)
    assert fit["fit"].success is True


def test_arc_support_refreeze_clears_stale_rectangles_and_unifies_arc_ids() -> None:
    # Mixed int 1 and str "1" arc_ids must unify without split-brain,
    # and re-freezing an excluded point must clear stale support rectangles.
    pts = [
        {
            "point_id": "p0",
            "qx": 0.10,
            "qy": 0.30,
            "branch_id": 0,
            "side": "upper",
            "arc_id": 1,
            "accepted": True,
            "localization_sigma_q": 0.03,
        },
        {
            "point_id": "p1",
            "qx": 0.15,
            "qy": 0.29,
            "branch_id": 0,
            "side": "upper",
            "arc_id": "1",
            "accepted": True,
            "localization_sigma_q": 0.03,
        },
        {
            "point_id": "p2",
            "qx": 0.20,
            "qy": 0.27,
            "branch_id": 0,
            "side": "upper",
            "arc_id": 1,
            "accepted": True,
            "localization_sigma_q": 0.03,
        },
    ]
    arcs = [{"arc_id": 1, "branch_id": 0, "side": "upper", "localization_sigma_q": 0.03}]
    freeze_observed_support(pts, arcs)
    assert all(p.get("support_rectangles") for p in pts)
    # p1 (arc_id="1") was unified into the middle of [p0, p1, p2], touching both segments
    assert len(pts[1]["support_rectangles"]) == 2
    assert pts[0]["support_component_ids"] == pts[1]["support_component_ids"] == [0]

    # Now mark p2 excluded and re-freeze: its stale support_rectangles must be cleared
    pts[2]["accepted"] = False
    freeze_observed_support(pts, arcs)
    assert pts[2].get("support_rectangles") == []
    assert pts[2].get("support_components") == []


# ---------------------------------------------------------------------------
# 4. observables.py
# ---------------------------------------------------------------------------


def test_observables_cancellation_propagates_double_ellipse_and_azimuthal_ridge() -> None:
    grid = np.linspace(-0.8, 0.8, 25)
    qx, qy = np.meshgrid(grid, grid)
    img = np.ones_like(qx)
    qmap = {"qx": qx, "qy": qy, "q_unit": "nm^-1"}
    geom = EllipseGeometry(cx=0.0, cy=0.0, a=0.4, axis_ratio=0.7, theta=0.0)
    pts = np.vstack([ellipse_points(geom, np.linspace(0.2, math.pi - 0.2, 12))])
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(AnalysisCancelled):
        fit_symmetric_double_ellipse(pts, cancel_event=cancel)

    with pytest.raises(AnalysisCancelled):
        measure_radial_ridges(img, qmap, ridge_method="azimuthal_peak", cancel_event=cancel)


def test_observables_mismatched_valid_mask_raises() -> None:
    qx, qy = np.meshgrid(np.linspace(-0.5, 0.5, 16), np.linspace(-0.5, 0.5, 16))
    img = np.ones((16, 16))
    bad_qmap = {"qx": qx, "qy": qy, "valid_mask": np.ones((8, 8), dtype=bool)}
    with pytest.raises(ValueError, match="shape"):
        measure_angular_spectrum(img, bad_qmap)


def test_observables_angular_spectrum_no_gap_fabrication_and_pi_clip() -> None:
    # Construct grid including exact angle == +pi (qy=0, qx<0) and a masked 60-deg gap
    y, x = np.mgrid[-30:31, -30:31]
    qx = x.astype(float) * 0.02
    qy = y.astype(float) * 0.02
    q = np.hypot(qx, qy)
    ang_deg = np.degrees(np.arctan2(qy, qx))
    img = 10.0 + 5.0 * np.exp(-0.5 * ((q - 0.35) / 0.05) ** 2)
    mask = np.ones_like(img, dtype=bool)
    mask[(ang_deg >= 30.0) & (ang_deg <= 90.0)] = False  # 60-degree masked gap

    spec = measure_angular_spectrum(
        img,
        {"qx": qx, "qy": qy, "valid_mask": mask},
        q_window=(0.25, 0.45),
        n_bins=72,
        smooth_sigma=1.5,
    )
    # Every zero-count bin in the masked gap must stay NaN even when smooth_sigma > 0
    empty_bins = spec.counts == 0
    assert np.any(empty_bins)
    assert np.all(np.isnan(spec.intensity[empty_bins]))
    assert np.all(np.isfinite(spec.intensity[~empty_bins]))


def test_observables_symmetry_diagnostics_multi_annulus_and_empty() -> None:
    # Empty / all-unassigned labels must return None cleanly without RuntimeWarning
    pts_empty = np.array([[0.2, 0.1], [-0.2, -0.1]], dtype=float)
    labels_unassigned = np.array([-1, -1], dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        empty_diag = _symmetry_diagnostics(
            pts_empty,
            labels_unassigned,
            center=(0.0, 0.0),
            reference_axis_deg=0.0,
            azimuthal=True,
        )
    assert empty_diag["central_symmetry"]["matched_pair_count"] == 0
    assert empty_diag["central_symmetry"]["median_deviation_q"] is None

    # Two distinct annuli (q~0.2 and q~0.6) along the same QI/QIII ray on branch 0:
    # opposite points must pair within the SAME radial shell, yielding zero deviation rather than ~0.4
    pts_multi = np.array(
        [
            [0.2, 0.05],
            [-0.2, -0.05],
            [0.6, 0.15],
            [-0.6, -0.15],
        ],
        dtype=float,
    )
    labels_multi = np.array([0, 0, 0, 0], dtype=int)
    diag = _symmetry_diagnostics(
        pts_multi,
        labels_multi,
        center=(0.0, 0.0),
        reference_axis_deg=0.0,
        azimuthal=True,
    )
    assert diag["central_symmetry"]["matched_pair_count"] == 2
    assert diag["central_symmetry"]["median_deviation_q"] < 1e-9
    assert diag["central_symmetry"]["median_q_difference"] < 1e-9


def test_observables_reject_ridge_point_fail_closes_q_star() -> None:
    cand = RidgePoint(
        angle=0.5,
        q=0.35,
        q_star=0.35,
        lamellar_spacing=2.0 * math.pi / 0.35,
        intensity=12.0,
        snr=1.5,
        valid=True,
        accepted=True,
        q_unit="nm^-1",
    )
    rejected = _reject_ridge_point(cand, "low_snr")
    assert rejected.valid is False
    assert rejected.accepted is False
    assert math.isnan(rejected.q_star)
    assert math.isnan(rejected.lamellar_spacing)
    assert rejected.q == pytest.approx(0.35)
    assert rejected.reason == "low_snr"


def test_observables_normalized_q_unit_superscript_minus() -> None:
    assert _normalized_q_unit("nm⁻1") == "nm^-1"
    assert _normalized_q_unit("Å⁻¹") == "a^-1"


def test_observables_quadratic_peak_periodic_seam_and_period() -> None:
    angles = np.linspace(0.0, 360.0, 36, endpoint=False)  # 10 deg spacing
    true_peak = 358.0
    delta = ((angles - true_peak + 180.0) % 360.0) - 180.0
    values = 100.0 - 0.5 * (delta**2)
    est, _ = _quadratic_peak(angles, values, 0, period=360.0)
    err = abs(((est - true_peak + 180.0) % 360.0) - 180.0)
    assert err < 0.2


# ---------------------------------------------------------------------------
# 5. auto_advisor.py
# ---------------------------------------------------------------------------


def test_auto_advisor_large_pure_noise_frame_not_false_ring() -> None:
    rng = np.random.default_rng(42)
    n = 512
    img = 100.0 + rng.normal(0.0, 5.0, size=(n, n))
    coords = (np.arange(n, dtype=float) - n / 2.0) * 0.005
    qx, qy = np.meshgrid(coords, coords)
    q_map = np.hypot(qx, qy)
    chi_deg = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    valid_mask = np.ones_like(img, dtype=bool)
    advice = advise_from_arrays(img, q_map, chi_deg, valid_mask, q_unit="nm^-1")
    assert advice["symmetry"]["morphology_class"] == "low_signal"


def test_auto_advisor_reflection_bisector_unbiased_at_cardinal_and_oblique() -> None:
    angles = np.arange(0.5, 360.0, 1.0)
    for true_axis in (90.0, 89.9, 45.0):
        # Symmetric four-lobe butterfly profile centered on true_axis
        profile = np.full_like(angles, 10.0)
        for shift in (-30.0, 30.0, 150.0, 210.0):
            d = ((angles - (true_axis + shift) + 180.0) % 360.0) - 180.0
            profile += 80.0 * np.exp(-0.5 * (d / 12.0) ** 2)
        est = _refine_reflection_bisector(angles, profile, true_axis)
        err = abs(((est - true_axis + 90.0) % 180.0) - 90.0)
        assert err < 0.5, f"axis={true_axis}, est={est}, err={err}"


def test_auto_advisor_masked_gap_does_not_fabricate_narrow_fwhm() -> None:
    angles = np.arange(0.5, 360.0, 1.0)
    delta = ((angles - 90.0 + 180.0) % 360.0) - 180.0
    profile = 10.0 + 100.0 * np.exp(-0.5 * (delta / 25.0) ** 2)
    # Mask a detector gap right next to the peak before intensity falls to half-height
    profile[(angles >= 94.0) & (angles <= 125.0)] = np.nan
    peaks = _find_circular_peaks(angles, profile, min_prominence_frac=0.12)
    assert len(peaks) >= 1
    assert math.isnan(peaks[0]["fwhm_deg"])


def test_auto_advisor_advise_frame_rejects_mismatched_cache_shape(tmp_path: Path) -> None:
    img_path = tmp_path / "frame64.tif"
    tifffile.imwrite(img_path, np.ones((64, 64), dtype=np.float32))
    coords = (np.arange(32, dtype=float) - 16.0) * 0.02
    qx_small, qy_small = np.meshgrid(coords, coords)
    geom_small = GeometryMaps(
        q_nm_inv=np.hypot(qx_small, qy_small),
        chi_rad=np.mod(np.arctan2(qy_small, qx_small), 2.0 * math.pi),
        qx_nm_inv=qx_small,
        qy_nm_inv=qy_small,
        valid_mask=np.ones((32, 32), dtype=bool),
    )
    small_cache = build_geometry_cache(geom_small)
    with pytest.raises(ValueError, match="shape"):
        advise_frame(img_path, geometry_cache=small_cache)


# ---------------------------------------------------------------------------
# 6. saxs_physics.py
# ---------------------------------------------------------------------------


def test_saxs_physics_coarsening_rejects_constant_time() -> None:
    res = fit_power_law_coarsening([5.0, 5.0, 5.0, 5.0], [10.0, 11.0, 12.0, 13.0])
    assert res["success"] is False
    assert res["status"] == "unavailable"
    assert "degenerate_time_span_for_coarsening_fit" in res["flags"]


def test_saxs_physics_c2_bridge_robust_loss_and_bright_axis_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angles = np.linspace(0.0, 360.0, 180, endpoint=False)
    # 4-lobe profile with alpha=30 deg around chi_sym=0 (reference/equator axis)
    chi_sym = 0.0
    alpha = 30.0
    sigma = 10.0
    y = 10.0 + 15.0 * np.cos(2.0 * np.deg2rad(angles - chi_sym))
    for mu in (chi_sym - alpha, chi_sym + alpha, chi_sym + 180.0 - alpha, chi_sym + 180.0 + alpha):
        d = ((angles - mu + 180.0) % 360.0) - 180.0
        y += 60.0 * np.exp(-0.5 * (d / sigma) ** 2)
    y_out = y.copy()
    y_out[10] += 80.0  # outlier spike

    res = fit_c2_bridge_four_lobe(
        angles,
        y_out,
        loss="cauchy",
        nominal_draw_axis_deg=90.0,
        bright_axis_convention="reference",
    )
    assert res["success"] is True
    assert abs(((res["chi_sym_deg"] - 0.0 + 90.0) % 180.0) - 90.0) < 3.0

    # Verify run_window_sensitivity_audit forwards bright_axis_convention
    import butterfly_saxs.saxs_physics as sp_mod

    seen: list[str] = []
    orig_fit = sp_mod.fit_c2_bridge_four_lobe

    def spy_fit(*args, **kwargs):
        seen.append(str(kwargs.get("bright_axis_convention")))
        return orig_fit(*args, **kwargs)

    monkeypatch.setattr(sp_mod, "fit_c2_bridge_four_lobe", spy_fit)
    coords = np.linspace(-0.6, 0.6, 65)
    qx, qy = np.meshgrid(coords, coords)
    q = np.hypot(qx, qy)
    img = 5.0 + 50.0 * np.exp(-0.5 * ((q - 0.3) / 0.05) ** 2)
    run_window_sensitivity_audit(
        img,
        qx=qx,
        qy=qy,
        q_windows=((0.2, 0.45),),
        bright_axis_convention="draw",
        n_radial_bins=16,
        n_azimuthal_bins=36,
    )
    assert seen and all(conv == "draw" for conv in seen)


def test_saxs_physics_streak_width_rotated_detector_pixel_dq() -> None:
    # Build a 90-degree rotated (qx, qy) grid where qx varies along rows (axis 0)
    # and qy varies along cols (axis 1), breaking naive diff(qx, axis=1) == 0.
    coords = np.linspace(-0.8, 0.8, 121)
    qy, qx = np.meshgrid(coords, coords)  # rotated 90 deg
    true_sigma0 = 0.04
    mosaic_sigma_slope = 0.04  # slight positive mosaic broadening so slope > 0
    true_fwhm0 = true_sigma0 * 2.0 * math.sqrt(2.0 * math.log(2.0))
    sigma_local = np.hypot(true_sigma0, mosaic_sigma_slope * np.maximum(qy, 0.0))
    img = 2.0 + 100.0 * np.exp(-0.5 * (qx / sigma_local) ** 2) * np.where(qy > 0.0, 1.0, 0.0)
    res = measure_perpendicular_streak_width(
        img,
        qx=qx,
        qy=qy,
        q_window=(0.20, 0.65),
        q_unit="nm^-1",
        wing_angles_deg=(90.0,),
        n_q_slices=6,
    )
    assert res["status"] == "estimated"
    assert res["summary"]["total_valid_slices"] >= 3
    assert res["summary"]["quadrature"]["dq_perp_0"] == pytest.approx(true_fwhm0, rel=0.15)


# ---------------------------------------------------------------------------
# 7. local_measurement.py
# ---------------------------------------------------------------------------


def test_local_measurement_diagonal_line_profile_chord_distance() -> None:
    coords = np.arange(64, dtype=float) * 0.01
    qx, qy = np.meshgrid(coords, coords)
    # 22.5-degree-like slope: (0, 0) -> (40, 17)
    start, end = (0, 0), (40, 17)
    vx, vy = float(end[0] - start[0]), float(end[1] - start[1])
    true_px_len = math.hypot(vx, vy)
    true_q_len = true_px_len * 0.01
    # Gaussian peak orthogonal to the line at midpoint with known sigma in q
    ux, uy = vx / true_px_len, vy / true_px_len
    proj_q = qx * ux + qy * uy
    sigma_q = 0.05
    img = 5.0 + 100.0 * np.exp(-0.5 * ((proj_q - 0.5 * true_q_len) / sigma_q) ** 2)

    prof = extract_line_profile(img, qx, qy, start, end, q_unit="nm^-1")
    assert prof.pixel_distance[-1] == pytest.approx(true_px_len, rel=1e-9)
    assert prof.q_distance[-1] == pytest.approx(true_q_len, rel=1e-9)
    expected_fwhm = sigma_q * 2.0 * math.sqrt(2.0 * math.log(2.0))
    assert prof.fwhm.status == "estimated"
    assert prof.fwhm.value == pytest.approx(expected_fwhm, rel=0.05)


def test_local_measurement_single_positive_pixel_roi_unavailable() -> None:
    coords = np.arange(32, dtype=float) * 0.02
    qx, qy = np.meshgrid(coords, coords)
    img = np.zeros((32, 32), dtype=float)
    img[16, 16] = 100.0  # single positive pixel inside a multi-pixel ROI strip
    roi = measure_roi_orientation(img, qx, qy, (10, 16), (22, 16), half_width_pixels=3.0, q_unit="nm^-1")
    assert roi.status == "unavailable"
    assert roi.reason == "zero_spatial_variance"
    assert roi.orientation_deg is None
    assert roi.principal_vector is None


def test_local_measurement_rejects_bool_patch_radius() -> None:
    coords = np.arange(8, dtype=float) * 0.1
    qx, qy = np.meshgrid(coords, coords)
    img = np.ones((8, 8), dtype=float)
    with pytest.raises(ValueError, match="patch_radius"):
        measure_point(img, qx, qy, 3, 3, patch_radius=True)


def test_local_measurement_json_export_atomic_on_nan_failure(tmp_path: Path) -> None:
    target = tmp_path / "measurements.json"
    original = '{"existing": true}\n'
    target.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError):
        export_local_measurements_json(target, [{"bad": float("nan")}], overwrite=True)
    assert target.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# 8. density2d.py & azimuthal_analysis.py
# ---------------------------------------------------------------------------


def test_density2d_power_law_underflow_fails_cleanly() -> None:
    # Construct profile where clipped exp(intercept) * q**(-alpha) underflows to 0.0 in float64
    q = np.array([1e10, 2e10, 4e10, 8e10], dtype=float)
    intensity = np.array([1e-100, 1e-130, 1e-160, 1e-190], dtype=float)
    profile = Density2DProfile(
        q=q,
        intensity=intensity,
        counts=np.array([10, 10, 10, 10]),
        geometry_counts=np.array([10, 10, 10, 10]),
        coverage=np.ones(4),
        q_min=float(q[0]),
        q_max=float(q[-1]),
        q_unit="nm⁻¹",
        azimuth_center_deg=0.0,
        azimuth_half_width_deg=15.0,
        source="underflow_test",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        (fit,) = fit_density_fluctuation(profile, models=("power_law",))
    assert fit.success is False
    assert "fit_failed" in fit.flags


def test_density2d_evaluate_oz_from_exported_json_parameters_and_nested_dir(tmp_path: Path) -> None:
    q = np.linspace(0.05, 1.0, 30)
    intensity = 100.0 / (1.0 + np.square(q * 4.0)) + 2.5
    profile = Density2DProfile(
        q=q,
        intensity=intensity,
        counts=np.full(q.shape, 20, dtype=int),
        geometry_counts=np.full(q.shape, 20, dtype=int),
        coverage=np.ones_like(q),
        q_min=float(q[0]),
        q_max=float(q[-1]),
        q_unit="nm⁻¹",
        azimuth_center_deg=0.0,
        azimuth_half_width_deg=15.0,
        source="oz_test",
    )
    fits = fit_density_fluctuation(profile, models=("ornstein_zernike",))
    assert fits[0].success

    nested_csv = tmp_path / "level1" / "level2" / "density_out.csv"
    csv_path, json_path = export_density2d_bundle(profile, fits, nested_csv)
    assert csv_path.exists() and json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    exported_params = payload["fits"][0]["parameters"]
    assert "_xi_q_inverse" not in exported_params
    re_eval = evaluate_density2d_model(q, "ornstein_zernike", exported_params)
    assert np.allclose(re_eval, fits[0].fitted_intensity, rtol=1e-6, atol=1e-6)


def test_azimuthal_detect_seeds_skips_nan_angles_and_c2_csv_roundtrip(tmp_path: Path) -> None:
    angles = np.linspace(0.0, 360.0, 72, endpoint=False)
    y = 10.0 + 20.0 * np.cos(2.0 * np.deg2rad(angles - 90.0))
    for mu in (60.0, 120.0, 240.0, 300.0):
        d = ((angles - mu + 180.0) % 360.0) - 180.0
        y += 50.0 * np.exp(-0.5 * (d / 12.0) ** 2)

    # Inject a NaN angle at a local maximum: _detect_peak_seeds must ignore it
    angles_nan = angles.copy()
    angles_nan[12] = np.nan
    y_nan = y.copy()
    y_nan[12] = 500.0
    prof_nan = AzimuthalProfile(
        angle_deg=angles_nan,
        intensity=y_nan,
        counts=np.full(angles.shape, 10, dtype=int),
        geometry_counts=np.full(angles.shape, 10, dtype=int),
        coverage=np.ones_like(angles),
        q_min=0.2,
        q_max=0.4,
        q_unit="nm⁻¹",
    )
    seeds, _ = _detect_peak_seeds(prof_nan, max_peaks=4, min_separation_deg=20.0, min_height_fraction=0.08)
    assert 12 not in seeds

    # c2_bridge_four_lobe CSV bundle must include c2 and chi_sym_deg for full reconstruction
    prof = AzimuthalProfile(
        angle_deg=angles,
        intensity=y,
        counts=np.full(angles.shape, 10, dtype=int),
        geometry_counts=np.full(angles.shape, 10, dtype=int),
        coverage=np.ones_like(angles),
        q_min=0.2,
        q_max=0.4,
        q_unit="nm⁻¹",
    )
    fit = fit_azimuthal_peaks(prof, model="c2_bridge_four_lobe")
    assert fit.success
    _, peaks_csv = export_azimuthal_csv_bundle(prof, fit, tmp_path / "az_out.csv")
    with peaks_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["c2"] != "" and rows[0]["chi_sym_deg"] != ""
    rebuilt_peaks = [
        AzimuthalPeak(
            center_deg=float(r["center_deg"]),
            amplitude=float(r["amplitude"]),
            fwhm_deg=float(r["fwhm_deg"]),
        )
        for r in rows
    ]
    rebuilt = evaluate_azimuthal_model(
        angles,
        baseline=float(rows[0]["baseline"]),
        peaks=rebuilt_peaks,
        model="c2_bridge_four_lobe",
        c2=float(rows[0]["c2"]),
        chi_sym_deg=float(rows[0]["chi_sym_deg"]),
    )
    assert np.allclose(rebuilt, fit.model, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 9. public_ellipse.py, io.py, parameters.py, geometry.py, legacy_saxs.py, synthetic.py
# ---------------------------------------------------------------------------


def test_public_ellipse_canonical_payload_reads_geometry_maps_q_unit() -> None:
    coords = (np.arange(16, dtype=float) - 8.0) * 0.05
    qx, qy = np.meshgrid(coords, coords)
    geom = GeometryMaps(
        q_nm_inv=np.hypot(qx, qy),
        chi_rad=np.mod(np.arctan2(qy, qx), 2.0 * math.pi),
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=np.ones((16, 16), dtype=bool),
    )
    fit = {
        "success": True,
        "a": 0.5,
        "b": 0.25,
        "theta_deg": 0.0,
        "center": (0.0, 0.0),
    }
    payload = canonical_ellipse_payload(fit, n_points=20, qmap=geom)
    assert payload["q_unit"] == "nm^-1"
    assert payload["L_N"] == pytest.approx(2.0 * math.pi / 0.25)


def test_io_npz_bytes_q_unit_png_frame0_nan_mask_and_tiff_single_frame(tmp_path: Path) -> None:
    # 1. NPZ with np.bytes_ q_unit
    npz_path = tmp_path / "bytes_unit.npz"
    arr = np.ones((8, 8), dtype=np.float32)
    np.savez(npz_path, image=arr, qx=arr, qy=arr, q_unit=np.bytes_("nm^-1"))
    loaded_npz = load_image(npz_path, include_qmap=True)
    assert loaded_npz.qmap is not None
    assert loaded_npz.qmap["q_unit"] == "nm^-1"

    # 2. Single-frame PNG accepts frame=0 and rejects frame=1
    png_path = tmp_path / "single.png"
    Image.fromarray(np.full((8, 8), 42, dtype=np.uint8)).save(png_path)
    loaded_png = load_image(png_path, frame=0)
    assert loaded_png.data.shape == (8, 8)
    assert loaded_png.frame == 0
    with pytest.raises(Exception):
        load_image(png_path, frame=1)

    # 3. Float NaN in valid_mask -> False (invalid); in external_mask -> True (masked out)
    vm = np.ones((8, 8), dtype=float)
    vm[0, 0] = np.nan
    loaded_vm = load_image(png_path, valid_mask=vm)
    assert loaded_vm.valid_mask is not None
    assert bool(loaded_vm.valid_mask[0, 0]) is False
    assert bool(loaded_vm.valid_mask[1, 1]) is True

    em = np.zeros((8, 8), dtype=float)
    em[0, 1] = np.nan
    loaded_em = load_image(png_path, external_mask=em)
    assert loaded_em.valid_mask is not None
    assert bool(loaded_em.valid_mask[0, 1]) is False
    assert bool(loaded_em.valid_mask[1, 1]) is True

    # 4. Multi-frame TIFF single-frame page extraction
    tiff_path = tmp_path / "stack.tif"
    stack = np.stack([np.full((8, 8), i * 10, dtype=np.uint16) for i in range(4)], axis=0)
    tifffile.imwrite(tiff_path, stack, photometric="minisblack")
    loaded_tiff = load_image(tiff_path, frame=2)
    assert loaded_tiff.frame == 2
    assert np.all(loaded_tiff.data == 20.0)


def test_parameters_atomic_rollback_and_allow_tied() -> None:
    pset = ParameterSet(
        {
            "a": ParameterSpec(0.5, min=0.1, max=1.0),
            "b": ParameterSpec(0.25, expr="0.5 * a", min=0.05, max=0.5),
            "c": ParameterSpec(1.0, min=0.0, max=2.0),
        }
    )
    # Failed update_values must roll back earlier successful assignments atomically
    with pytest.raises(ValueError):
        pset.update_values({"a": 0.8, "c": 99.0})
    assert pset["a"].value == pytest.approx(0.5)
    assert pset["c"].value == pytest.approx(1.0)

    # Failed __delitem__ (dependency violation) must restore the deleted parameter
    with pytest.raises(ValueError):
        del pset["a"]
    assert "a" in pset
    assert pset["a"].value == pytest.approx(0.5)

    # allow_tied=True updates free parameters and skips tied ones without raising
    pset.update_values({"a": 0.6, "b": 999.0}, allow_tied=True)
    resolved = pset.resolve()
    assert resolved["a"] == pytest.approx(0.6)
    assert resolved["b"] == pytest.approx(0.3)


def test_geometry_and_legacy_chi_normalization() -> None:
    qx = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    qy = np.array([[-1.0, -1.0], [1.0, 1.0]])
    raw_chi_deg = np.degrees(np.arctan2(qy, qx))  # contains negative angles (-135, -45)
    gmap = GeometryMaps(
        q_nm_inv=np.hypot(qx, qy),
        chi_rad=np.mod(np.arctan2(qy, qx), 2.0 * math.pi),
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=np.ones((2, 2), dtype=bool),
        chi_deg=raw_chi_deg,
    )
    assert float(np.min(gmap.chi_deg)) >= 0.0
    assert float(np.max(gmap.chi_deg)) < 360.0

    legacy_map = legacy_fusion_geometry_map(
        {
            "cx": 8.0,
            "cy": 8.0,
            "dist_mm": 1500.0,
            "wl_A": 1.24,
            "px_mm": 0.172,
        },
        (16, 16),
    )
    assert float(np.min(legacy_map.chi_rad)) >= 0.0
    assert float(np.max(legacy_map.chi_rad)) < 2.0 * math.pi + 1e-12
    assert float(np.min(legacy_map.chi_deg)) >= 0.0
    assert float(np.max(legacy_map.chi_deg)) < 360.0


def test_synthetic_multipoint_and_ndarray_evolution() -> None:
    seq_list = make_butterfly_sequence(
        n_frames=4,
        shape=(16, 16),
        evolution={"a": [0.30, 0.32, 0.45, 0.50]},
        seed=7,
    )
    assert [p["a"] for p in seq_list.parameters] == pytest.approx([0.30, 0.32, 0.45, 0.50])

    seq_arr = make_butterfly_sequence(
        n_frames=4,
        shape=(16, 16),
        evolution={"a": np.array([0.31, 0.36, 0.41, 0.46])},
        seed=7,
    )
    assert [p["a"] for p in seq_arr.parameters] == pytest.approx([0.31, 0.36, 0.41, 0.46])

    # Length 2 still interpolates start..stop
    seq_pair = make_butterfly_sequence(
        n_frames=4,
        shape=(16, 16),
        evolution={"a": (0.30, 0.60)},
        seed=7,
    )
    assert [p["a"] for p in seq_pair.parameters] == pytest.approx([0.30, 0.40, 0.50, 0.60])

    # Mismatched length raises ValueError instead of silently discarding intermediate points
    with pytest.raises(ValueError, match="length"):
        make_butterfly_sequence(
            n_frames=4,
            shape=(16, 16),
            evolution={"a": [0.30, 0.40, 0.50]},
            seed=7,
        )
