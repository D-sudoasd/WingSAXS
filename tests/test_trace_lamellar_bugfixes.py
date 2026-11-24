"""Regression tests for Butterfly/Ridge/Peak tracing and Lamellar bugfixes."""

from __future__ import annotations

import csv
import json
import math
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from butterfly_saxs import annular_measurements
from butterfly_saxs import annular_trace
from butterfly_saxs import butterfly
from butterfly_saxs import butterfly_diagnostics
from butterfly_saxs import butterfly_quality
from butterfly_saxs import butterfly_ridge
from butterfly_saxs import butterfly_settings
from butterfly_saxs import butterfly_uncertainty
from butterfly_saxs import lamellar
from butterfly_saxs import lamellar_adapter
from butterfly_saxs import lamellar_analysis
from butterfly_saxs import lamellar_export
from butterfly_saxs import lamellar_models
from butterfly_saxs import lamellar_render
from butterfly_saxs import lamellar_report_figures
from butterfly_saxs import lamellar_scattering
from butterfly_saxs import lamellar_sources
from butterfly_saxs import lamellar_utils
from butterfly_saxs import peak_landmarks
from butterfly_saxs import ridge_profiles
from butterfly_saxs import sector_peaks
from butterfly_saxs import sector_trace
from butterfly_saxs.cancellation import AnalysisCancelled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_qmap(
    shape: tuple[int, int] = (48, 48),
    q_max: float = 0.6,
    *,
    q_unit: str = "nm^-1",
    angle_deg: float = 0.0,
) -> dict[str, object]:
    ny, nx = shape
    xs = np.linspace(-q_max, q_max, nx)
    ys = np.linspace(-q_max, q_max, ny)
    gx, gy = np.meshgrid(xs, ys)
    rad = math.radians(angle_deg)
    qx = gx * math.cos(rad) - gy * math.sin(rad)
    qy = gx * math.sin(rad) + gy * math.cos(rad)
    q = np.hypot(qx, qy)
    return {
        "qx": qx,
        "qy": qy,
        "q": q,
        "q_unit": q_unit,
    }


def _make_butterfly_image(
    qmap: dict[str, object],
    *,
    a: float = 0.42,
    b: float = 0.22,
    theta_deg: float = 28.0,
    sigma: float = 0.028,
) -> np.ndarray:
    qx = np.asarray(qmap["qx"], dtype=float)
    qy = np.asarray(qmap["qy"], dtype=float)
    image = np.full_like(qx, 2.0, dtype=float)
    for sign in (1.0, -1.0):
        ang = math.radians(sign * theta_deg)
        u = qx * math.cos(ang) + qy * math.sin(ang)
        v = -qx * math.sin(ang) + qy * math.cos(ang)
        dist = np.abs(np.hypot(u / a, v / b) - 1.0) * math.sqrt(a * b)
        image += 80.0 * np.exp(-0.5 * (dist / sigma) ** 2)
    return image


def _simple_scene(*, length_unit: str = "nm", status: str = "schematic") -> lamellar_models.LamellarScene:
    unit_name = "nm^-1" if length_unit == "nm" else "pixel-q"
    source = {
        "q_unit": unit_name,
        "lobe_radial_peaks": [
            {"method": "radial_peak", "branch_id": 0, "angle_deg": 25.0, "q_star": 0.4, "q_unit": unit_name},
            {"method": "radial_peak", "branch_id": 1, "angle_deg": 115.0, "q_star": 0.5, "q_unit": unit_name},
        ],
    }
    scene = lamellar.build_lamellar_scene(
        source,
        {"mode": "single", "period_source": "radial", "layer_count": 3, "stack_count": 2},
    )
    if status != scene.status:
        scene.metadata["status"] = status
    return scene


# ---------------------------------------------------------------------------
# A.1 ridge_profiles.py
# ---------------------------------------------------------------------------


def test_ridge_profile_bilinear_sampling_blocks_masked_sentinel_leakage() -> None:
    qmap = _make_qmap((33, 33), q_max=0.5)
    image = np.full((33, 33), 10.0, dtype=float)
    valid = np.ones((33, 33), dtype=bool)
    # Place extreme masked sentinel pixels directly next to the profile sample track.
    image[16, 17] = 1.0e12
    image[17, 17] = float("nan")
    valid[16, 17] = False
    valid[17, 17] = False

    ctx = ridge_profiles.build_profile_context(image, qmap, mask=~valid)
    qx_arr = np.asarray(qmap["qx"], dtype=float)
    qy_arr = np.asarray(qmap["qy"], dtype=float)
    prof = ridge_profiles.extract_normal_profile(
        image,
        qmap,
        {
            "pixel_y": 16.0,
            "pixel_x": 16.25,
            "qx": float(qx_arr[16, 16]),
            "qy": float(qy_arr[16, 16]),
            "normal_qx": 0.0,
            "normal_qy": 1.0,
        },
        mask=~valid,
        options={"_profile_context": ctx},
    )
    raw = np.asarray(prof["raw_intensity"], dtype=float)
    finite = raw[np.isfinite(raw)]
    assert finite.size > 0
    assert np.allclose(finite, 10.0, atol=1e-6)
    assert float(np.max(finite)) < 20.0


def test_ridge_profile_fit_models_supports_symmetric_positive_and_negative_slopes() -> None:
    offsets = np.linspace(-0.08, 0.08, 25)
    sigma = 0.018
    step = float(offsets[1] - offsets[0])
    for true_slope in (-250.0, 250.0):
        signal = 50.0 + true_slope * offsets + 120.0 * np.exp(-0.5 * (offsets / sigma) ** 2)
        fit = ridge_profiles._fit_profile_models(
            offsets, signal, noise=0.5, step=step, delta_bic=4.0
        )
        assert fit["model"] in {"single_gaussian", "ambiguous_single_double"}
        assert fit["peaks"]
        assert abs(fit["peaks"][0]["center_q"]) < 0.004
        assert float(np.nanmax(np.abs(np.asarray(fit["fit"], dtype=float) - signal))) < 1.0


def test_wang2007_vertical_slice_check_rotated_grid_and_upper_boundary() -> None:
    qmap = _make_qmap((61, 61), q_max=0.5, angle_deg=37.0)
    qx = np.asarray(qmap["qx"], dtype=float)
    image = 5.0 + 100.0 * np.exp(-0.5 * ((qx - 0.20) / 0.04) ** 2)
    valid = np.ones_like(image, dtype=bool)
    res = ridge_profiles.wang2007_vertical_slice_check(
        image,
        qmap,
        (0.05, 0.45),
        mask=~valid,
        reference_axis_deg=90.0,
        options={"wang_slice_bins": 20, "wang_min_pixels": 5},
    )
    assert "insufficient_binned_profile" not in res["applicability"]["reasons"]
    assert int(np.count_nonzero(np.asarray(res["counts"], dtype=int) > 0)) >= 5

    # Sparse mask leaving fewer than 5 finite bins must flag insufficient_binned_profile.
    qy = np.asarray(qmap["qy"], dtype=float)
    q = np.asarray(qmap["q"], dtype=float)
    in_strip = (q >= 0.05) & (q <= 0.45) & (np.abs(qy) <= 0.03)
    strip_indices = np.argwhere(in_strip)
    sparse_valid = np.zeros_like(valid)
    for r, c in strip_indices[:3]:
        sparse_valid[int(r), int(c)] = True
    sparse = ridge_profiles.wang2007_vertical_slice_check(
        image,
        qmap,
        (0.05, 0.45),
        mask=~sparse_valid,
        reference_axis_deg=90.0,
        options={"wang_slice_bins": 40, "wang_min_pixels": 1},
    )
    assert "insufficient_binned_profile" in sparse["applicability"]["reasons"]


# ---------------------------------------------------------------------------
# A.2 butterfly_ridge.py
# ---------------------------------------------------------------------------


def test_butterfly_ridge_nms_and_graph_arcs_handle_nonfinite_and_multiscale() -> None:
    options = butterfly_ridge._normalise_options(
        {
            "nms_radius_factor": 0.5,
            "graph_radius_factor": 1.5,
            "graph_tangent_angle_deg": 45.0,
            "graph_normal_factor": 1.5,
        }
    )
    q_step = 0.01
    candidates = [
        {"point_id": "bad", "qx": float("nan"), "qy": 0.1, "scale": 2.0, "score": 100.0, "accepted": True, "branch_id": 0, "normal_qx": 1.0, "normal_qy": 0.0, "tangent_qx": 0.0, "tangent_qy": 1.0},
        {"point_id": "p0", "qx": 0.10, "qy": 0.00, "scale": 1.0, "score": 10.0, "accepted": True, "branch_id": 0, "normal_qx": 1.0, "normal_qy": 0.0, "tangent_qx": 0.0, "tangent_qy": 1.0},
        {"point_id": "p1", "qx": 0.10, "qy": 0.04, "scale": 5.0, "score": 10.0, "accepted": True, "branch_id": 0, "normal_qx": 1.0, "normal_qy": 0.0, "tangent_qx": 0.0, "tangent_qy": 1.0},
        {"point_id": "p2", "qx": 0.10, "qy": 0.08, "scale": 1.0, "score": 10.0, "accepted": True, "branch_id": 0, "normal_qx": 1.0, "normal_qy": 0.0, "tangent_qx": 0.0, "tangent_qy": 1.0},
    ]
    kept = butterfly_ridge._nms(candidates, options, q_step)
    assert all(math.isfinite(p["qx"]) and math.isfinite(p["qy"]) for p in kept)

    # Symmetric max(scale_i, scale_j) + dynamic reach links p0 <-> p1 <-> p2 into one component.
    groups, _edges = butterfly_ridge._graph_arcs(kept, options, q_step)
    assert len(groups) == 1
    assert len(groups[0]) == 3

    # Asymmetric parallel-track ("echelon") jump where a coarse endpoint's scale
    # would excuse an offset exceeding the fine endpoint's own normal corridor
    # must NOT stitch unrelated radial populations together.
    echelon_opts = butterfly_ridge._normalise_options(
        {
            "graph_radius_factor": 3.5,
            "graph_tangent_angle_deg": 65.0,
            "graph_normal_factor": 1.8,
        }
    )
    echelon = [
        {
            "point_id": "outer",
            "qx": -0.4542,
            "qy": 0.2221,
            "scale": 3.6,
            "accepted": True,
            "branch_id": 1,
            "tangent_qx": 0.9955,
            "tangent_qy": 0.0952,
            "normal_qx": 0.0952,
            "normal_qy": -0.9955,
        },
        {
            "point_id": "spur",
            "qx": -0.3811,
            "qy": 0.1578,
            "scale": 1.2,
            "accepted": True,
            "branch_id": 1,
            "tangent_qx": 0.9292,
            "tangent_qy": -0.3697,
            "normal_qx": -0.3697,
            "normal_qy": -0.9292,
        },
    ]
    echelon_groups, echelon_edges = butterfly_ridge._graph_arcs(
        echelon, echelon_opts, 0.01227
    )
    assert echelon_edges == []
    assert len(echelon_groups) == 2


def test_butterfly_ridge_ordered_component_avoids_chord_leaps_and_uturns() -> None:
    pts = [
        {"qx": 0.00, "qy": 0.000},
        {"qx": 0.01, "qy": 0.001},
        {"qx": 0.02, "qy": 0.000},
        {"qx": 0.03, "qy": 0.001},
    ]
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    ordered = butterfly_ridge._ordered_component([0, 1, 2, 3], edges, pts)
    assert ordered in ([0, 1, 2, 3], [3, 2, 1, 0])


def test_butterfly_ridge_refine_radial_peak_respects_usable_neighbors() -> None:
    # Build fine-bin samples around q_bin=0.30 where the center bin and its right neighbor
    # have >=4 samples (usable), while the left neighbor bin has only 1 sample with a huge outlier.
    radii_list = (
        [0.26] * 5
        + [0.28] * 5
        + [0.292] * 1  # single-pixel outlier in left neighbor bin -> counts < 4 (not usable!)
        + [0.300] * 6
        + [0.308] * 6
        + [0.320] * 5
        + [0.340] * 5
    )
    weights_list = (
        [10.0] * 5
        + [10.0] * 5
        + [1.0e9] * 1
        + [50.0] * 6
        + [48.0] * 6
        + [10.0] * 5
        + [10.0] * 5
    )
    refined = butterfly_ridge._refine_radial_peak_q(
        np.asarray(radii_list, dtype=float),
        np.asarray(weights_list, dtype=float),
        0.300,
        0.10,
        0.50,
        n_bins=15,
        coarse_bins=40,
    )
    assert abs(refined - 0.300) < 0.012


def test_butterfly_ridge_axis_separation_scale_invariant_and_exclude_after_ring_fill() -> None:
    qmap = _make_qmap((52, 52), q_max=0.6)
    image = _make_butterfly_image(qmap)
    t1 = butterfly_ridge.trace_butterfly_ridges(
        image, qmap, (0.12, 0.55), options={"smoothing_scales": [1.2, 2.0]}
    )
    t2 = butterfly_ridge.trace_butterfly_ridges(
        image * 1.0e4, qmap, (0.12, 0.55), options={"smoothing_scales": [1.2, 2.0]}
    )
    s1 = [p["axis_separation"] for p in t1["points"] if p.get("axis_separation") is not None]
    s2 = [p["axis_separation"] for p in t2["points"] if p.get("axis_separation") is not None]
    assert s1 and s2
    assert float(np.median(s1)) == pytest.approx(float(np.median(s2)), rel=0.05)

    accepted = [p for p in t1["points"] if p.get("accepted")]
    assert accepted
    target_id = accepted[0]["point_id"]
    t_excl = butterfly_ridge.trace_butterfly_ridges(
        image,
        qmap,
        (0.12, 0.55),
        options={"smoothing_scales": [1.2, 2.0]},
        edits=[{"type": "exclude_point", "point_id": target_id}],
    )
    matching = [p for p in t_excl["points"] if p["point_id"] == target_id]
    assert matching and matching[0]["accepted"] is False
    assert matching[0]["reason"] == "excluded_point_edit"


def test_butterfly_ridge_sparse_ring_fill_unit_normals_and_shifted_center() -> None:
    qmap = _make_qmap((48, 48), q_max=0.6)
    qx = np.asarray(qmap["qx"], dtype=float)
    qy = np.asarray(qmap["qy"], dtype=float)
    cx, cy = 0.05, -0.04
    r = np.hypot(qx - cx, qy - cy)
    image = 2.0 + 90.0 * np.exp(-0.5 * ((r - 0.30) / 0.025) ** 2)
    valid = (r >= 0.12) & (r <= 0.52)
    points: list[dict[str, object]] = []
    arcs: list[dict[str, object]] = []
    opts = butterfly_ridge._normalise_options({"center_qx": cx, "center_qy": cy, "reference_axis_deg": 0.0})
    diag = butterfly_ridge._fill_sparse_first_order_ring(
        points,
        qx=qx,
        qy=qy,
        q=np.hypot(qx, qy),
        intensity=image,
        valid=valid,
        hint=0.30,
        options=opts,
        q_step=0.02,
        signature="sig0",
        arcs=arcs,
    )
    assert diag["added"] > 0
    for point in points:
        n_norm = math.hypot(float(point["normal_qx"]), float(point["normal_qy"]))
        t_norm = math.hypot(float(point["tangent_qx"]), float(point["tangent_qy"]))
        assert n_norm == pytest.approx(1.0, abs=1e-6)
        assert t_norm == pytest.approx(1.0, abs=1e-6)
        dx, dy = float(point["qx"]) - cx, float(point["qy"]) - cy
        d_norm = math.hypot(dx, dy)
        assert (float(point["normal_qx"]) * dx + float(point["normal_qy"]) * dy) / d_norm == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# A.3 butterfly.py
# ---------------------------------------------------------------------------


def test_butterfly_sensitivity_handles_dict_window_negative_arc_id_radial_sector_and_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qmap = _make_qmap((40, 40), q_max=0.6)
    image = _make_butterfly_image(qmap)
    trace_stub = {
        "points": [
            {"arc_id": -1, "accepted": True, "branch_id": 0, "side": "upper", "qx": 0.2, "qy": 0.2},
            {"arc_id": 0, "accepted": True, "branch_id": 0, "side": "upper", "qx": 0.25, "qy": 0.2},
        ]
    }
    calls: list[dict[str, object]] = []

    def fake_analyze(img, qm, win, **kwargs):
        opts = kwargs.get("options") or {}
        calls.append({"win": win, "opts": dict(opts)})
        if len(calls) == 1:
            raise ValueError("synthetic variant domain error")
        return {"candidate_fit": {"success": True, "a": 0.4, "b": 0.2, "axis_ratio": 0.5, "theta_deg": 25.0}}

    monkeypatch.setattr(butterfly, "analyze_butterfly", fake_analyze)
    sens = butterfly._sensitivity(
        image,
        qmap,
        {"q_min": 0.1, "q_max": 0.5},
        mask=None,
        options={"trace_method": "radial_sector", "smoothing_sigma_bins": 1.2, "sector_width_deg": 12.0},
        parameters=None,
        reference=0.0,
        multistart=1,
        cancel_event=None,
        trace=trace_stub,
    )
    variant_names = {r["variant"] for r in sens["records"]}
    assert "sector_0.75" in variant_names
    assert "sector_1.25" in variant_names
    assert sens["records"][0]["success"] is False
    assert "synthetic variant domain error" in sens["records"][0].get("error", "")
    held_ids = [item["arc_id"] for item in sens["held_out_arcs"]]
    assert -1 not in held_ids
    assert 0 in held_ids


def test_butterfly_recipe_sha256_accepts_ndarray_counts() -> None:
    qmap = _make_qmap((36, 36), q_max=0.6)
    image = _make_butterfly_image(qmap)
    counts = np.full((36, 36), 50.0, dtype=float)
    res = butterfly.analyze_butterfly(
        image,
        qmap,
        (0.12, 0.52),
        options={"stage": "trace", "counts": counts, "smoothing_scales": [1.5]},
    )
    assert isinstance(res["recipe_sha256"], str)
    assert len(res["recipe_sha256"]) == 64


# ---------------------------------------------------------------------------
# A.4 butterfly_quality / diagnostics / settings / annular / sector
# ---------------------------------------------------------------------------


def test_butterfly_diagnostics_prefers_post_fit_candidate_rows() -> None:
    trace = {"arc_diagnostics": [{"arc_id": 0, "source": "pre"}]}
    candidate = {"arc_diagnostics": [{"arc_id": 0, "source": "post"}]}
    bundle = butterfly_diagnostics.arc_diagnostic_rows(trace, candidate)
    assert len(bundle["rows"]) == 1
    assert bundle["rows"][0]["source"] == "post"


def test_butterfly_quality_handles_none_fields_and_reads_both_seed_keys() -> None:
    for seed_key in ("seed_matches", "seed_actions"):
        trace = {
            "points": [
                {"accepted": True, "branch_id": 0, "side": "upper", "qx": 0.2, "qy": 0.1, "normal_ok": True, "support_component_count": None, "support_components": None},
                {"accepted": True, "branch_id": 0, "side": "lower", "qx": -0.2, "qy": -0.1, "normal_ok": True},
                {"accepted": True, "branch_id": 1, "side": "upper", "qx": -0.2, "qy": 0.1, "normal_ok": True},
                {"accepted": True, "branch_id": 1, "side": "lower", "qx": 0.2, "qy": -0.1, "normal_ok": True},
            ],
            "arc_diagnostics": [
                {"arc_id": 0, "support_status": "frozen", "support_component_count": None, "support_components": None, "topology_flags": None}
            ],
            "diagnostics": {
                seed_key: [{"matched": False}],
                "topology_flags": None,
            },
        }
        candidate = {
            "success": True,
            "a": 0.4,
            "b": 0.2,
            "axis_ratio": 0.5,
            "theta_deg": 25.0,
            "bound_flags": None,
            "diagnostics": {"profile_residual_rms": None},
        }
        uncertainty = {"status": "ok", "intervals": {"a": [0.38, 0.42]}}
        sensitivity = {
            "completed": True,
            "records": [],
            "parameter_ranges": {"a": [None, 0.45]},
            "held_out_arcs": [],
        }
        ev = butterfly_quality.evaluate_arc_evidence(
            trace, candidate, uncertainty=uncertainty, sensitivity=sensitivity
        )
        assert "seed_not_supported_by_observed_arc" in ev["quality"]["flags"]


def test_annular_measurements_handles_none_q_annulus() -> None:
    trace = {
        "q_unit": "nm^-1",
        "annular_peaks": {
            "annuli": [
                {"annulus_index": 0, "q_center": None},
                {"annulus_index": 1, "q_center": 0.335},
            ]
        },
        "points": [
            {"accepted": True, "quadrant": "QI", "annulus_index": 0, "chi_deg": 25.0, "q_annulus": 0.25},
            {"accepted": True, "quadrant": "QI", "annulus_index": 1, "chi_deg": 26.0, "q_annulus": 0.335},
        ],
    }
    summary = annular_measurements.summarize_annular_measurements(trace)
    qi_traj = next(item for item in summary["trajectories"] if item["quadrant"] == "QI")
    assert qi_traj["q_min_observed"] == pytest.approx(0.335)
    assert qi_traj["q_max_observed"] == pytest.approx(0.335)


def test_butterfly_settings_allows_auto_seed_and_rejects_np_bool() -> None:
    norm = butterfly_settings.normalize_butterfly_settings(
        {"edits": [{"type": "seed", "qx": 0.1, "qy": 0.2, "branch_id": None}, {"type": "seed", "qx": -0.1, "qy": 0.2, "branch_id": -1}]}
    )
    assert len(norm["edits"]) == 2
    assert norm["edits"][0]["branch_id"] is None
    assert norm["edits"][1]["branch_id"] == -1
    with pytest.raises(ValueError):
        butterfly_settings.normalize_butterfly_settings({"resamples": np.bool_(True)})


def test_annular_and_sector_traces_emit_seed_matches_and_honor_min_arc_points() -> None:
    qmap = _make_qmap((44, 44), q_max=0.6)
    image = _make_butterfly_image(qmap)
    ann = annular_trace.trace_butterfly_annuli(
        image, qmap, (0.15, 0.50), options={"annular_radial_bins": 12, "annular_angle_bins": 48, "annular_min_arc_points": 5}
    )
    assert "seed_matches" in ann["diagnostics"]
    assert "seed_actions" in ann["diagnostics"]
    for arc in ann["arcs"]:
        if arc["valid"]:
            assert arc["accepted_points"] >= 5
        elif arc["accepted_points"] < 5:
            assert arc["valid"] is False

    sec = sector_trace.trace_butterfly_sector_peaks(
        image, qmap, (0.15, 0.50), options={"sector_step_deg": 30.0, "sector_width_deg": 18.0}
    )
    assert "seed_matches" in sec["diagnostics"]
    assert "seed_actions" in sec["diagnostics"]


def test_sector_peaks_representative_pixel_matches_localized_q_star() -> None:
    qmap = _make_qmap((48, 48), q_max=0.6)
    q = np.asarray(qmap["q"], dtype=float)
    target_q = 0.315
    image = 2.0 + 200.0 * np.exp(-0.5 * ((q - target_q) / 0.025) ** 2)
    bundle = sector_peaks.measure_sector_peaks(
        image, qmap, (0.10, 0.55), options={"sector_step_deg": 90.0, "sector_width_deg": 20.0, "n_radial_bins": 16}
    )
    sec0 = bundle["sectors"][0]
    peak = sec0["selected_peak"]
    assert peak is not None
    assert abs(peak["pixel_q"] - peak["q_star"]) <= 0.03


# ---------------------------------------------------------------------------
# A.5 butterfly_uncertainty.py
# ---------------------------------------------------------------------------


def test_butterfly_uncertainty_axial_interval_wraps_across_0_180() -> None:
    samples = [178.0, 178.5, 179.0, 179.5, 0.5, 1.0, 1.5, 2.0]
    interval = butterfly_uncertainty._axial_interval_deg(samples)
    assert interval is not None
    low, high = interval
    assert high - low < 6.0


def test_butterfly_3side_resampling_refit_does_not_demand_missing_4th_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qmap = _make_qmap((40, 40), q_max=0.6)
    image = _make_butterfly_image(qmap)
    three_sides_points = [
        {"point_id": "p0", "accepted": True, "valid": True, "arc_id": 0, "branch_id": 0, "side": "upper", "qx": 0.25, "qy": 0.15},
        {"point_id": "p1", "accepted": True, "valid": True, "arc_id": 1, "branch_id": 0, "side": "lower", "qx": -0.25, "qy": -0.15},
        {"point_id": "p2", "accepted": True, "valid": True, "arc_id": 2, "branch_id": 1, "side": "upper", "qx": -0.25, "qy": 0.15},
    ]
    fake_trace = {
        "points": three_sides_points,
        "arcs": [],
        "profiles": {},
        "diagnostics": {},
        "method_version": "test",
    }
    monkeypatch.setattr(butterfly_ridge, "trace_butterfly_ridges", lambda *a, **kw: fake_trace)
    monkeypatch.setattr(
        butterfly,
        "_fit_trace",
        lambda *a, **kw: {"success": True, "a": 0.4, "b": 0.2, "axis_ratio": 0.5, "theta_deg": 25.0},
    )
    captured_replicates: list[dict[str, object]] = []

    def fake_resample(img, qm, *, mask, q_window, refit, options, cancel_event=None):
        rep = refit(np.asarray(img, dtype=float), {})
        captured_replicates.append(rep)
        return {"status": "ok", "intervals": {}, "coverage_calibrated": False}

    monkeypatch.setattr(butterfly_uncertainty, "resample_butterfly", fake_resample)
    butterfly.analyze_butterfly(
        image,
        qmap,
        (0.12, 0.52),
        options={"stage": "evaluate", "resamples": 4, "sensitivity": False},
    )
    assert captured_replicates
    assert captured_replicates[0]["topology_failed"] is False


def test_butterfly_uncertainty_sensitivity_propagates_qmap_mask_and_cancellation() -> None:
    qmap = _make_qmap((24, 24), q_max=0.5)
    qmap_mask = np.zeros((24, 24), dtype=bool)
    qmap_mask[0, 0] = True
    qmap["mask"] = qmap_mask
    qmap["valid_mask"] = np.ones((24, 24), dtype=bool)
    qmap["valid_mask"][0, 1] = False

    copied = butterfly_uncertainty._copy_qmap_with_perturbation(
        qmap, (24, 24), center_delta_px=(0.0, 0.0), qcal_scale=1.0
    )
    assert bool(copied["mask"][0, 0]) is True
    assert bool(copied["mask"][0, 1]) is True
    assert bool(copied["valid_mask"][0, 0]) is False
    assert bool(copied["valid_mask"][0, 1]) is False

    seen_masks: list[np.ndarray] = []

    def record_refit(img: np.ndarray, overrides: dict[str, object]) -> dict[str, object]:
        seen_masks.append(np.asarray(overrides["mask"], dtype=bool))
        return {"candidate_fit": {"success": True, "a": 0.4, "b": 0.2, "axis_ratio": 0.5, "theta_deg": 25.0}}

    butterfly_uncertainty.evaluate_refit_sensitivity(
        np.ones((24, 24)),
        qmap,
        record_refit,
        q_window=(0.1, 0.4),
    )
    assert seen_masks
    assert bool(seen_masks[0][0, 0]) is True
    assert bool(seen_masks[0][0, 1]) is True

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AnalysisCancelled):
        butterfly_uncertainty.evaluate_refit_sensitivity(
            np.ones((24, 24)),
            qmap,
            record_refit,
            q_window=(0.1, 0.4),
            cancel_event=cancel,
        )


# ---------------------------------------------------------------------------
# A.6 peak_landmarks.py
# ---------------------------------------------------------------------------


def test_peak_landmarks_no_edge_cliff_peak_and_no_smoothing_bleed_and_bool_rejected() -> None:
    qmap = _make_qmap((64, 64), q_max=0.6)
    qx = np.asarray(qmap["qx"], dtype=float)
    qy = np.asarray(qmap["qy"], dtype=float)
    q = np.asarray(qmap["q"], dtype=float)
    angle = np.degrees(np.arctan2(qy, qx)) % 360.0
    # Flat isotropic ring with a 60-degree masked wedge: zero true angular peaks!
    ring = 5.0 + 100.0 * np.exp(-0.5 * ((q - 0.30) / 0.03) ** 2)
    valid = (q >= 0.15) & (q <= 0.45) & ~((angle >= 60.0) & (angle <= 120.0))

    res = peak_landmarks.compute_peak_landmarks(
        ring, qx, qy, valid_mask=valid, q_window=(0.15, 0.45), q_unit="nm^-1"
    )
    ang_prof = res["profiles"]["angular"]
    for val, sup in zip(ang_prof["intensity_smoothed"], ang_prof["supported"]):
        if not sup:
            assert val is None
    for peak in res.get("peaks", []):
        ang = float(peak["angular_peak_deg"]) % 360.0
        assert not (50.0 <= ang <= 130.0)

    with pytest.raises(ValueError):
        peak_landmarks.compute_peak_landmarks(
            ring, qx, qy, valid_mask=valid, q_window=(0.15, 0.45), options={"angular_bins": np.bool_(True)}
        )
    with pytest.raises(ValueError):
        peak_landmarks.compute_peak_landmarks(
            ring, qx, qy, valid_mask=valid, q_window=(0.15, 0.45), options={"minimum_prominence": True}
        )


# ---------------------------------------------------------------------------
# B.7 lamellar_report_figures.py
# ---------------------------------------------------------------------------


def test_lamellar_report_figures_fixes_axial_wrap_q_unit_support_and_analysis_text() -> None:
    expanded = lamellar_report_figures._curve_records(
        {"branch_id": 0, "angle_deg": [0.0, 90.0], "q_radius": [0.4, 0.3], "supported": [True, False], "observed_support_count": [3, 0]}
    )
    assert lamellar_report_figures._curve_has_observed_support(expanded) is True

    lines = lamellar_report_figures._analysis_text(
        {"status": "candidate", "q_unit": "nm^-1", "ellipse": {"a": 0.4, "b": 0.2, "theta_deg": 25.0}}
    )
    joined = "\n".join(lines)
    assert "candidate" in joined.lower()
    assert "degree" in joined.lower()

    frames = [
        {
            "frame_index": 0,
            "q_unit": None,
            "lamellar_analysis": {
                "q_unit": "nm^-1",
                "parameter_rows": [
                    {"parameter": "Ln_from_minor_axis_nm", "value": 15.0, "unit": "nm", "status": "available"},
                ],
                "observed_direction_summary": [
                    {
                        "branch_id": 0,
                        "axial_mean_deg": 91.0 - 1.0e-14,
                        "angle_min_deg": 91.0,
                        "angle_max_deg": 91.0 - 2.0e-14,
                        "count": 1,
                        "status": "observed_summary",
                    }
                ],
            },
        },
        {
            "frame_index": 1,
            "q_unit": "unknown",
            "lamellar_analysis": {
                "q_unit": "nm^-1",
                "parameter_rows": [
                    {"parameter": "Ln_from_minor_axis_nm", "value": 16.0, "unit": "nm", "status": "available"},
                ],
                "observed_direction_summary": [
                    {
                        "branch_id": 0,
                        "axial_mean_deg": 92.0,
                        "angle_min_deg": 90.0,
                        "angle_max_deg": 94.0,
                        "count": 3,
                        "status": "observed_summary",
                    }
                ],
            },
        },
    ]
    fig = lamellar_report_figures._sequence_figure(frames, title="Sequence")
    assert fig is not None
    assert len(fig.axes) == 2
    obs_ax = fig.axes[1]
    for col in obs_ax.collections:
        segments = getattr(col, "get_segments", lambda: [])()
        for seg in segments:
            if len(seg) == 2:
                assert abs(float(seg[1][1]) - float(seg[0][1])) < 10.0


# ---------------------------------------------------------------------------
# B.8 lamellar_scattering.py
# ---------------------------------------------------------------------------


def test_lamellar_scattering_dotted_stem_aliasing_zero_intensity_and_excl_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _simple_scene(length_unit="nm")
    sim = lamellar_scattering.simulate_projected_density_fft(scene, shape=(128, 16))
    assert any("fewer than two raster pixels" in w for w in sim.diagnostics["warnings"])

    npz1, json1, png1 = lamellar_scattering.export_lamellar_scattering(
        sim, tmp_path / "sample.v1.npz", overwrite=False
    )
    assert npz1.name == "sample.v1.npz"
    assert json1.name == "sample.v1.json"
    assert png1.name == "sample.v1.png"
    assert all(p.is_file() for p in (npz1, json1, png1))

    npz2, json2, png2 = lamellar_scattering.export_lamellar_scattering(
        sim, tmp_path / "run_1.5", overwrite=False
    )
    assert (npz2.name, json2.name, png2.name) == ("run_1.5.npz", "run_1.5.json", "run_1.5.png")

    zero_res = lamellar_scattering.LamellarScatteringResult(
        projected_density=np.zeros((8, 8), dtype=float),
        qx=np.linspace(-1.0, 1.0, 8),
        qy=np.linspace(-1.0, 1.0, 8),
        intensity=np.zeros((8, 8), dtype=float),
        bounds_xy=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        pixel_size=(0.25, 0.25),
        length_unit="nm",
        diagnostics={},
        assumptions=(),
        scene_status="schematic",
        scene_flags=(),
    )
    fig = lamellar_scattering.render_lamellar_scattering(zero_res)
    assert isinstance(fig.canvas, FigureCanvasAgg)

    def _no_link(src, dst):
        raise OSError("hard links unsupported")

    monkeypatch.setattr(os, "link", _no_link)
    out3 = lamellar_scattering.export_lamellar_scattering(
        sim, tmp_path / "exfat_case.v1", overwrite=False
    )
    assert all(p.is_file() for p in out3)
    with pytest.raises(FileExistsError):
        lamellar_scattering.export_lamellar_scattering(
            sim, tmp_path / "exfat_case.v1", overwrite=False
        )


# ---------------------------------------------------------------------------
# B.9 lamellar_adapter.py
# ---------------------------------------------------------------------------


def test_lamellar_adapter_radial_peaks_theta_rad_q_unit_and_direction_group() -> None:
    source = {
        "lobe_radial_peaks": [
            {"method": "radial_peak", "angle_deg": 20.0, "q_star": 1.0, "q_unit": "nm^-1", "valid": True, "accepted": False},
            {"method": "radial_peak", "angle_deg": 25.0, "q_star": 1.0, "q_unit": "nm^-1", "q_star_Ainv": 0.1},
            {"method": "radial_peak", "angle_deg": 35.0, "q_star_nm_inv": 0.0, "q_star_Ainv": 0.2},
        ]
    }
    peaks = lamellar_adapter._radial_peaks(source)
    assert len(peaks) == 2
    assert peaks[0]["angle_deg"] == pytest.approx(25.0)
    assert peaks[0]["q_unit"] == "nm^-1"
    assert peaks[0]["q"] == pytest.approx(1.0)
    assert peaks[0]["q_nm_inv"] == pytest.approx(1.0)
    assert peaks[1]["q_unit"] == "Å^-1"
    assert peaks[1]["q"] == pytest.approx(0.2)
    assert peaks[1]["q_nm_inv"] == pytest.approx(2.0)

    unit = lamellar_adapter._q_unit_from_source(
        {}, [{"q_unit": "unknown"}, {"q_unit": "nm^-1"}]
    )
    assert unit == "nm^-1"

    records = lamellar_adapter._ellipse_parameter_records(
        {"q_unit": "nm^-1", "parameters": {"theta": 0.5, "theta_deg": 28.6}}
    )
    by_name = {row["name"]: row for row in records}
    assert by_name["theta"]["unit"] == "rad"
    assert by_name["theta_deg"]["unit"] == "degree"

    _, inc1 = lamellar_adapter._direction_for_group([{"angle_deg": 0.0}, {"angle_deg": 14.0}, {"angle_deg": -14.0}])
    _, inc2 = lamellar_adapter._direction_for_group([{"angle_deg": 14.0}, {"angle_deg": 0.0}, {"angle_deg": -14.0}])
    assert inc1 is True and inc2 is True
    ang, _ = lamellar_adapter._direction_for_group([{"angle_deg": 1.0e9}])
    assert math.isfinite(ang)


# ---------------------------------------------------------------------------
# B.10 lamellar_export.py
# ---------------------------------------------------------------------------


def test_lamellar_export_dict_scene_stale_preservation_csv_nan_and_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _simple_scene(length_unit="nm")
    dict_scene = scene.to_mapping()
    stale_scene = SimpleNamespace(
        status="stale",
        message="",
        source_identity="frame_real_07",
        length_unit="nm",
        metadata={"status": "stale", "source_identity": "frame_real_07"},
        vertices=np.full((1, 8, 3), 1.0e6, dtype=float),
        bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
    )
    nan_scene = SimpleNamespace(
        status="schematic",
        length_unit="nm",
        vertices=np.full((1, 8, 3), float("nan"), dtype=float),
    )

    bounds = lamellar_export._sequence_bounds([dict_scene, stale_scene, nan_scene], length_unit="nm")
    assert np.all(np.isfinite(bounds))
    assert float(np.max(np.abs(bounds))) < 1.0e4

    csv_path = tmp_path / "params.csv"
    lamellar_export._write_parameter_csv(
        csv_path, [{"name": "p", "value": float("nan"), "other": float("inf"), "ok": 1.5}]
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["value"] == ""
    assert rows[0]["other"] == ""
    assert rows[0]["ok"] == "1.5"

    arrs = lamellar_export._scene_arrays({"populations": [(0, 12.5)]})
    assert math.isnan(float(arrs["population_angle_deg"][0]))

    out_dir = tmp_path / "seq_export"
    result = lamellar_export.export_lamellar_sequence(
        [dict_scene, stale_scene], out_dir, language="en"
    )
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert manifest["frames"][1]["render_status"] == "stale"
    assert manifest["frames"][1]["source_identity"] == "frame_real_07"

    def _interrupt(*args, **kwargs):
        raise KeyboardInterrupt("simulated interrupt")

    monkeypatch.setattr(lamellar_export, "_write_scene_artifacts", _interrupt)
    abort_dir = tmp_path / "aborted_export"
    with pytest.raises(KeyboardInterrupt):
        lamellar_export.export_lamellar_scene(scene, abort_dir)
    assert not list(tmp_path.glob(".aborted_export.staging-*"))


# ---------------------------------------------------------------------------
# B.11 lamellar_sources.py
# ---------------------------------------------------------------------------


def test_lamellar_sources_case_insensitive_manifest_and_bytes_q_unit(tmp_path: Path) -> None:
    assert lamellar_sources._batch_prefix(Path("run_Manifest.json")) == "run_"
    assert lamellar_sources._batch_prefix(Path("Manifest.json")) == ""

    manifest_path = tmp_path / "Manifest.json"
    npz_path = tmp_path / "results.npz"
    ellipse_path = tmp_path / "ellipse_fit.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "lamellarsaxs2d.batch.v1", "frames": [{"frame_index": 0}]}),
        encoding="utf-8",
    )
    ellipse_path.write_text(json.dumps({"frames": []}), encoding="utf-8")
    np.savez(npz_path, frame_0000__image=np.ones((4, 4), dtype=float))

    resolved_json, resolved_npz = lamellar_sources._resolve_native_pair(tmp_path)
    assert resolved_json is not None and resolved_json.name.casefold() == "manifest.json"
    assert resolved_npz is not None and resolved_npz.name.casefold() == "results.npz"

    single_npz = tmp_path / "single.npz"
    np.savez(single_npz, observed=np.ones((4, 4), dtype=float), q_unit=np.asarray(b"nm^-1"))
    attached = lamellar_sources._attach_single_npz({}, single_npz, embed=True)
    assert attached["q_unit"] == "nm^-1"


# ---------------------------------------------------------------------------
# B.12 lamellar_render.py
# ---------------------------------------------------------------------------


def test_lamellar_render_edge_cases_and_no_legend_accumulation() -> None:
    scene = _simple_scene(length_unit="nm")

    fig = Figure(figsize=(4, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    bounds = np.array([[0.0, 0.0, 0.0], [4.545454545454545e-10, 4.545454545454545e-10, 1.0]])
    lamellar_render._draw_scale_bar(ax, bounds, scene, "en")

    assert lamellar_render._colors(scene, 0).shape == (0, 4)
    assert lamellar_render._colors({"branch_ids": []}, 3).shape == (3, 4)

    assert lamellar_render._image_extent(np.array(1.0), np.array(2.0), (4, 4)) is None
    qx = np.tile(np.linspace(-1.0, 1.0, 4), (4, 1))
    qy = np.tile(np.linspace(-2.0, 2.0, 4)[:, None], (1, 4))
    qx[0, :] = np.nan
    qy[:, 0] = np.nan
    ext = lamellar_render._image_extent(qx, qy, (4, 4))
    assert ext is not None and all(math.isfinite(v) for v in ext)

    assert lamellar_render._image_to_array(np.zeros((4, 4, 5), dtype=np.uint8)) is None
    assert lamellar_render._image_to_array(np.ones((4, 4), dtype=np.complex128)) is None

    redraw_fig = Figure(figsize=(6, 5))
    FigureCanvasAgg(redraw_fig)
    redraw_ax = redraw_fig.add_subplot(111)
    lamellar_render.render_lamellar_2d(scene, ax=redraw_ax, language="en")
    lamellar_render.render_lamellar_2d(scene, ax=redraw_ax, language="en")
    lamellar_render.render_lamellar_2d(scene, ax=redraw_ax, language="en")
    assert len(redraw_fig.legends) <= 1
    assert len(redraw_fig.texts) <= 1


# ---------------------------------------------------------------------------
# B.13 lamellar_analysis.py, lamellar.py, lamellar_models.py, lamellar_utils.py
# ---------------------------------------------------------------------------


def test_lamellar_utils_and_models_bool_and_none_safety() -> None:
    assert lamellar_utils._normalise_q_unit("nm⁻1") == "nm^-1"
    assert lamellar_utils._q_scale_to_nm("nm⁻1") == 1.0
    assert lamellar_utils._finite(True) is None
    assert lamellar_utils._finite(np.bool_(False)) is None

    with pytest.raises(ValueError):
        lamellar_models.LamellarSettings(selected_branch=True)
    with pytest.raises(ValueError):
        lamellar_models.LamellarSettings(spread_deg=True)

    scene = _simple_scene()
    scene.metadata["draw_axis_deg"] = None
    scene.metadata["reference_period"] = None
    assert scene.draw_axis_deg == 90.0
    assert scene.reference_period == 1.0


def test_lamellar_analysis_member_id_negative_axes_ellipse_pair_near_circle_and_estimate() -> None:
    assert lamellar_analysis._finite(np.bool_(True)) is None

    source = {
        "q_unit": "nm^-1",
        "ellipse_fit": {
            "status": "ok",
            "q_unit": "nm^-1",
            "a": 0.5,
            "b": 0.25,
            "theta_deg": 20.0,
            "center": (0.0, 0.0),
            "ellipse_pair": [
                {"branch_id": np.int64(1), "a": 0.5, "b": 0.25, "theta_deg": -20.0, "q_unit": "unknown"},
                {"branch_id": np.int64(0), "a": 0.5, "b": 0.25, "theta_deg": 20.0, "q_unit": ""},
            ],
        },
        "ridge_points": [
            {"branch_id": 1, "qx": 0.4, "qy": -0.1, "valid": True, "accepted": True, "q_unit": "nm^-1"},
            {"branch_id": -1, "qx": 0.0, "qy": 0.25, "valid": True, "accepted": True, "q_unit": "nm^-1"},
        ],
    }
    res = lamellar_analysis.analyze_lamellar_morphology(source)
    members_by_id = {m["branch_id"]: m for m in res["ellipse"]["members"]}
    assert members_by_id[1]["angle_deg"] == pytest.approx(-20.0)
    assert members_by_id[0]["angle_deg"] == pytest.approx(20.0)
    assert members_by_id[1]["q_unit"] == "nm^-1"
    assert members_by_id[1]["observed_support"] is True
    assert members_by_id[0]["observed_support"] is False
    unassigned_obs = [row for row in res["observed_directions"] if row["branch_id"] == -1]
    assert unassigned_obs and unassigned_obs[0]["model_q_radius"] is not None

    neg = lamellar_analysis.analyze_lamellar_morphology(
        {"q_unit": "nm^-1", "ellipse_fit": {"a": -1.0, "b": 0.2, "theta_deg": 10.0, "center": (0.0, 0.0)}}
    )
    assert neg["model_curves"] == []
    assert neg["status"] == "unavailable"

    near_circ = lamellar_analysis.analyze_lamellar_morphology(
        {
            "q_unit": "nm^-1",
            "ellipse_fit": {
                "a": 1.0,
                "b": 0.99,
                "axis_ratio": 0.99,
                "theta_deg": 35.0,
                "center": (0.0, 0.0),
                "quantitative_parameters": {
                    k: {"status": "available", "value": v}
                    for k, v in (("a", 1.0), ("b", 0.99), ("axis_ratio", 0.99), ("theta_deg", 35.0))
                },
            },
        }
    )
    theta_row = next(r for r in near_circ["parameter_rows"] if r["parameter"] == "ellipse_axis_tilt_deg")
    assert theta_row["status"] == "candidate"
    assert theta_row["value"] is None
    assert theta_row["candidate_value"] == pytest.approx(35.0)

    est = lamellar_analysis.analyze_lamellar_morphology(
        {
            "q_unit": "nm^-1",
            "ellipse_fit": {
                "a": 1.0,
                "b": 0.2,
                "axis_ratio": 0.2,
                "theta_deg": 20.0,
                "center": (0.0, 0.0),
                "quantitative_parameters": {
                    "a": {"status": "estimate", "value": 1.0},
                    "b": {"status": "estimate", "value": 0.2},
                    "axis_ratio": {"status": "estimate", "value": 0.2},
                    "theta_deg": {"status": "estimate", "value": 20.0},
                },
            },
        }
    )
    a_row = next(r for r in est["parameter_rows"] if r["parameter"] == "a")
    assert a_row["value"] is None
    assert a_row["candidate_value"] == pytest.approx(1.0)

    no_theta = lamellar_analysis.analyze_lamellar_morphology(
        {"q_unit": "nm^-1", "ellipse_fit": {"a": 1.0, "b": 1.0, "center": (0.0, 0.0)}}
    )
    ln_row = next(r for r in no_theta["parameter_rows"] if r["parameter"] == "Ln_from_minor_axis_nm")
    assert ln_row["reason"] != "spacing_unavailable_missing_ellipse_center"


def test_lamellar_scene_single_mode_rng_unknown_unit_shadowing_and_nonphysical_mismatch() -> None:
    two_branch = {
        "q_unit": "nm^-1",
        "lobe_radial_peaks": [
            {"method": "radial_peak", "branch_id": 0, "angle_deg": 30.0, "q_star": 0.4, "q_unit": "nm^-1"},
            {"method": "radial_peak", "branch_id": 1, "angle_deg": 30.0, "q_star": 0.4, "q_unit": "nm^-1"},
        ],
    }
    scene = lamellar.build_lamellar_scene(
        two_branch,
        {"mode": "single", "period_source": "radial", "layer_count": 2, "stack_count": 1, "spread_deg": 20.0, "seed": 7},
    )
    ori_b0 = scene.orientations[scene.branch_ids == 0][0]
    ori_b1 = scene.orientations[scene.branch_ids == 1][0]
    assert not np.allclose(ori_b0, ori_b1)

    shadow_src = {
        "q_unit": "nm^-1",
        "ridge_points": [
            {"branch_id": 0, "qx": 0.4, "qy": 0.0, "valid": True, "accepted": True, "q_unit": "nm^-1"},
        ],
        "ellipse_fit": {
            "a": 0.5,
            "b": 0.25,
            "theta_deg": 15.0,
            "center": (0.0, 0.0),
            "q_unit": "unknown",
        },
    }
    shadow_scene = lamellar.build_lamellar_scene(
        shadow_src,
        {"mode": "single", "period_source": "ellipse", "layer_count": 2, "stack_count": 1},
    )
    assert shadow_scene.length_unit == "nm"

    mismatch_src = {
        "lobe_radial_peaks": [
            {"method": "radial_peak", "branch_id": 0, "angle_deg": 20.0, "q_star": 50.0, "q_unit": "pixel"},
            {"method": "radial_peak", "branch_id": 1, "angle_deg": 110.0, "q_star": 0.5, "q_unit": "arb"},
        ],
    }
    mismatch_scene = lamellar.build_lamellar_scene(
        mismatch_src,
        {"mode": "single", "period_source": "radial", "layer_count": 2, "stack_count": 1},
    )
    assert mismatch_scene.status == "unavailable"
    assert "mixed_q_unit_support" in mismatch_scene.metadata["flags"]
