from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import struct

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from butterfly_saxs.composite_figures import (
    CompositeFigureBuilder,
    FigureSpec,
    PUBLICATION_FIGURE_SCHEMA,
    PanelSpec,
    plot_azimuthal_fit_panel,
    plot_evolution_panel,
    plot_polar_q_chi_panel,
    plot_radial_kratky_panel,
    plot_saxs_2d_panel,
    plot_streak_width_panel,
    render_sample_physics_dashboard,
)
from butterfly_saxs.journal_style import inches_to_mm


def _png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png_bytes[16:24])
    return width, height


def _synthetic_saxs_2d():
    ny, nx = 48, 48
    q = np.linspace(-1.0, 1.0, nx)
    qx, qy = np.meshgrid(q, np.linspace(-1.0, 1.0, ny))
    r = np.hypot(qx, qy)
    chi = np.arctan2(qy, qx)
    # Four-lobe butterfly scattering around q* = 0.4 nm^-1 and wings at +/- 38 deg from 90 deg.
    radial = np.exp(-0.5 * ((r - 0.4) / 0.12) ** 2)
    lobe_angles = np.deg2rad([52.0, 128.0, -128.0, -52.0])
    angular = sum(np.exp(-0.5 * ((np.angle(np.exp(1j * (chi - a)))) / np.deg2rad(16.0)) ** 2) for a in lobe_angles)
    image = 2.0 + 120.0 * radial * (0.2 + angular)
    valid_mask = np.ones_like(image, dtype=bool)
    valid_mask[r < 0.11] = False  # beamstop
    valid_mask[22:25, :] = False  # detector module gap
    return image, qx, qy, valid_mask


def test_composite_builder_exact_nature_mm_dimensions_and_headless_safety():
    backend_before = matplotlib.get_backend()
    fignums_before = plt.get_fignums()

    spec = FigureSpec(
        preset="nature",
        width="double",
        height_mm=120.0,
        nrows=2,
        ncols=2,
        panels=(
            PanelSpec("a", 0, 0),
            PanelSpec("b", 0, 1),
            PanelSpec("c", 1, 0, style_1d=True),
            PanelSpec("d", 1, 1, style_1d=True),
        ),
    )
    builder = CompositeFigureBuilder.from_spec(spec)
    w_in, h_in = builder.figure.get_size_inches()
    assert inches_to_mm(w_in) == pytest.approx(183.0, abs=0.05)
    assert inches_to_mm(h_in) == pytest.approx(120.0, abs=0.05)
    assert set(builder.panels) == {"a", "b", "c", "d"}
    assert builder.panels["c"].spines["top"].get_visible() is False
    assert builder.panels["c"].spines["right"].get_visible() is False

    single = CompositeFigureBuilder(preset="nature", width="single", height_mm=80.0)
    sw_in, sh_in = single.figure.get_size_inches()
    assert inches_to_mm(sw_in) == pytest.approx(89.0, abs=0.05)
    assert inches_to_mm(sh_in) == pytest.approx(80.0, abs=0.05)

    assert matplotlib.get_backend() == backend_before
    assert plt.get_fignums() == fignums_before
    builder.figure.clear()
    single.figure.clear()


def test_plot_saxs_2d_and_polar_panels_preserve_mask_gaps_and_overlays():
    image, qx, qy, valid_mask = _synthetic_saxs_2d()
    builder = CompositeFigureBuilder(preset="nature", width="double", height_mm=86.0, nrows=1, ncols=2, dpi=150)
    ax2d = builder.add_panel("a", 0, 0)
    axpol = builder.add_panel("b", 0, 1)

    ridge_pts = np.array([[0.25, 0.32], [-0.25, 0.32], [-0.25, -0.32], [0.25, -0.32]])
    res2d = plot_saxs_2d_panel(
        ax2d,
        image,
        qx,
        qy,
        valid_mask=valid_mask,
        q_unit="nm^-1",
        intensity_unit="cm^-1",
        norm_mode="asinh",
        ridge_points=ridge_pts,
        rays_deg=[52.0, 128.0, 232.0, 308.0],
        bisector_deg=90.0,
        q_circles=[{"q": 0.11, "label": "Beamstop"}, {"q": 0.4, "label": r"$q^*$"}],
        scalebar_q=0.5,
        title="425 °C (2 h)",
    )
    assert res2d["mesh"].cmap.name == "inferno"
    assert res2d["mesh"].cmap.get_bad()[0] > 0.8
    assert r"\mathrm{nm}^{-1}" in ax2d.get_xlabel()
    leg = ax2d.get_legend()
    assert leg is not None
    assert leg.get_frame_on() is True

    # Polar (q, chi) panel with missing masked azimuths and ridge overlay.
    q_edges = np.linspace(0.11, 1.0, 21)
    chi_edges = np.linspace(-180.0, 180.0, 37)
    polar_z = np.full((20, 36), 5.0)
    polar_z[6:10, :] = 80.0
    polar_z[:, 15:18] = np.nan  # masked module gap
    chi_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    ridge_q = np.full_like(chi_centers, 0.42)
    ridge_q[15:18] = np.nan

    res_pol = plot_polar_q_chi_panel(
        axpol,
        polar_z,
        q_edges,
        chi_edges,
        q_unit="nm^-1",
        ridge_chi_deg=chi_centers,
        ridge_q=ridge_q,
        q_cutoffs=[{"q": 0.25, "label": "Legacy 0.25"}],
        bisector_deg=90.0,
        title="Unwrapped polar map",
    )
    assert res_pol["mesh"].cmap.name == "inferno"
    mesh_arr = np.ma.asarray(res_pol["mesh"].get_array())
    assert np.ma.getmaskarray(mesh_arr).sum() >= 60

    builder.figure.canvas.draw()
    builder.figure.clear()

    with pytest.raises(ValueError, match="same shape"):
        bad_builder = CompositeFigureBuilder(preset="nature", width="single", height_mm=70.0)
        plot_saxs_2d_panel(bad_builder.add_panel("a", 0, 0), image, qx[:, :-1], qy)


def test_evolution_panel_enforces_estimate_candidate_and_nan_gap_encoding():
    builder = CompositeFigureBuilder(preset="nature", width="single", height_mm=72.0, dpi=150)
    ax = builder.add_panel("a", 0, 0)

    x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    y = [13.5, 15.0, None, 17.8, 19.2, 20.5]
    statuses = ["estimate", "estimate", "failed", "candidate", "candidate", "estimate"]

    info = plot_evolution_panel(
        ax,
        [
            {
                "label": "425 °C",
                "x": x,
                "y": y,
                "yerr": [0.4, 0.5, None, 0.8, 0.9, 0.6],
                "status": statuses,
                "color": "#0072B2",
            }
        ],
        xlabel="Time (h)",
        ylabel=r"$L^*$",
        y_unit="nm",
        log_log_inset=True,
        inset_ref_slopes=[1.0 / 3.0],
    )
    assert info["series_summary"][0] == {
        "label": "425 °C",
        "n_total": 6,
        "n_estimate": 3,
        "n_candidate": 2,
        "n_failed_or_missing": 1,
    }
    assert info["inset_ax"] is not None

    solid_lines = [line for line in ax.lines if line.get_linestyle() == "-"]
    dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(solid_lines) == 1
    assert len(dashed_lines) == 1

    # Estimate line: filled markers, explicit NaN breaks at failed (idx 2) and candidate (idx 3, 4) positions!
    est_line = solid_lines[0]
    est_y = np.asarray(est_line.get_ydata(), dtype=float)
    assert np.isfinite(est_y[0]) and np.isfinite(est_y[1]) and np.isfinite(est_y[5])
    assert np.isnan(est_y[2]) and np.isnan(est_y[3]) and np.isnan(est_y[4])
    assert est_line.get_markerfacecolor() == "#0072B2"

    # Candidate line: hollow white markers, explicit NaN breaks at estimate & failed positions!
    cand_line = dashed_lines[0]
    cand_y = np.asarray(cand_line.get_ydata(), dtype=float)
    assert np.isfinite(cand_y[3]) and np.isfinite(cand_y[4])
    assert np.isnan(cand_y[0]) and np.isnan(cand_y[1]) and np.isnan(cand_y[2]) and np.isnan(cand_y[5])
    assert cand_line.get_markerfacecolor() == "#FFFFFF"

    builder.figure.canvas.draw()
    builder.figure.clear()


def test_radial_kratky_azimuthal_and_streak_width_panels():
    builder = CompositeFigureBuilder(preset="nature", width="double", height_mm=130.0, nrows=2, ncols=2, dpi=150)
    ax_rad = builder.add_panel("a", 0, 0)
    ax_az = builder.add_panel("b", 0, 1)
    ax_sw_phys = builder.add_panel("c", 1, 0)
    ax_sw_pix = builder.add_panel("d", 1, 1)

    q = np.linspace(0.11, 1.2, 80)
    iq = 1.5 + 80.0 * np.exp(-0.5 * ((q - 0.32) / 0.07) ** 2)
    rk = plot_radial_kratky_panel(
        ax_rad,
        [{"label": "425 °C", "q": q, "intensity": iq, "q_star_direct": 0.32, "q_star_kratky": 0.35}],
        mode="both",
        q_unit="nm^-1",
        intensity_unit="cm^-1",
        q_cutoffs=[{"q": 0.11, "label": "Beamstop", "shade_below": True}, {"q": 0.25, "label": "Old 0.25"}],
    )
    assert rk["ax_intensity"] is ax_rad
    assert rk["ax_kratky"] is not None

    chi = np.linspace(-180.0, 180.0, 91)
    bridge = 10.0 + 4.0 * np.cos(2.0 * np.deg2rad(chi))
    lobes = [
        25.0 * np.exp(-0.5 * (((chi - mu + 180) % 360 - 180) / 14.0) ** 2)
        for mu in (52.0, 128.0, -128.0, -52.0)
    ]
    fit_total = bridge + sum(lobes)
    obs_az = fit_total + np.sin(np.deg2rad(chi * 3.0)) * 0.4
    obs_az[40:44] = np.nan  # masked gap
    az_res = plot_azimuthal_fit_panel(
        ax_az,
        chi,
        obs_az,
        fit_total=fit_total,
        c2_bridge=bridge,
        lobe_components=lobes,
        bisector_deg=90.0,
        half_opening_deg=38.0,
        metrics={"r2": 0.992, "half_opening_deg": 38.0, "lobe_fwhm_deg": 28.4},
        intensity_unit="cm^-1",
    )
    assert az_res["ax_resid"] is not None

    q_slices = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    dq_perp = 0.05 + np.deg2rad(10.0) * q_slices
    sw_phys = plot_streak_width_panel(
        ax_sw_phys,
        q_slices,
        dq_perp,
        dq_perp_err=np.full_like(q_slices, 0.008),
        linear_fit={"dq_perp_0": 0.05, "delta_alpha_deg": 10.0, "intercept_err": 0.005, "slope_err": 0.01, "r2": 0.98},
        quadrature_fit={"dq_perp_0": 0.06, "delta_alpha_deg": 9.6, "r2": 0.97},
        q_unit="nm^-1",
    )
    assert sw_phys["linear"]["W_apparent_nm"] == pytest.approx(2.0 * np.pi / 0.05)
    assert any("W_{\\mathrm{app}}" in t.get_text() for t in ax_sw_phys.texts)

    # Pixel-q MUST NEVER compute nm for apparent coherence W!
    sw_pix = plot_streak_width_panel(
        ax_sw_pix,
        q_slices * 100.0,
        dq_perp * 100.0,
        linear_fit={"dq_perp_0": 5.0, "delta_alpha_deg": 10.0, "r2": 0.98},
        q_unit="pixel-q",
    )
    assert sw_pix["linear"]["W_apparent_nm"] is None
    assert not any("nm" in t.get_text() for t in ax_sw_pix.texts)

    builder.figure.canvas.draw()
    builder.figure.clear()


def test_export_publication_figure_writes_vector_raster_source_data_and_sha256_manifest(tmp_path):
    builder = CompositeFigureBuilder(preset="nature", width="double", height_mm=100.0, nrows=1, ncols=2, dpi=300)
    ax_a = builder.add_panel("a", 0, 0, style_1d=True)
    ax_b = builder.add_panel("b", 0, 1, style_1d=True)
    ax_a.plot([0, 1, 2], [1.0, 2.0, 3.0], color="#0072B2")
    ax_a.set_xlabel("Time (h)")
    ax_a.set_ylabel(r"$L^*$ ($\mathrm{nm}$)")
    ax_b.plot([0, 1, 2], [0.1, 0.4, 0.7], color="#D55E00")
    ax_b.set_xlabel("Time (h)")
    ax_b.set_ylabel(r"$A_4$")

    builder.register_source_data(
        "panel_a",
        [
            {"time_h": 0.0, "L_star_nm": 13.5, "note": "=SUM(A1:A2)"},
            {"time_h": 1.0, "L_star_nm": float("nan"), "note": "missing"},
        ],
    )
    builder.register_source_data("panel_b", {"time_h": [0.0, 1.0, 2.0], "A4": [0.1, 0.4, 0.7]})

    stem = tmp_path / "fig1_butterfly"
    manifest = builder.export(
        stem,
        formats=("pdf", "svg", "png"),
        dpi=300,
        metadata={"title": "Test Nature Figure", "nan_field": float("nan")},
    )
    builder.figure.clear()

    assert manifest["schema_version"] == PUBLICATION_FIGURE_SCHEMA
    assert manifest["size_mm"][0] == pytest.approx(183.0, abs=0.05)
    assert manifest["size_mm"][1] == pytest.approx(100.0, abs=0.05)

    pdf_path = tmp_path / "fig1_butterfly.pdf"
    svg_path = tmp_path / "fig1_butterfly.svg"
    png_path = tmp_path / "fig1_butterfly.png"
    manifest_path = tmp_path / "fig1_butterfly_manifest.json"
    source_a = tmp_path / "fig1_butterfly_source_data" / "panel_a.csv"
    source_b = tmp_path / "fig1_butterfly_source_data" / "panel_b.csv"

    for p in (pdf_path, svg_path, png_path, manifest_path, source_a, source_b):
        assert p.is_file() and p.stat().st_size > 10

    assert pdf_path.read_bytes().startswith(b"%PDF-")
    svg_text = svg_path.read_text(encoding="utf-8")
    assert svg_text.lstrip().startswith("<?xml")
    assert "<text" in svg_text  # svg.fonttype="none" keeps editable text elements!

    # Verify exact pixel dimensions match 183 mm x 100 mm at 300 dpi (within 2 px rounding).
    px_w, px_h = _png_dimensions(png_path.read_bytes())
    assert abs(px_w - round(183.0 / 25.4 * 300)) <= 2
    assert abs(px_h - round(100.0 / 25.4 * 300)) <= 2

    # Verify Source Data CSV formula safety and empty NaN cells.
    with source_a.open(encoding="utf-8", newline="") as handle:
        rows_a = list(csv.DictReader(handle))
    assert rows_a[0]["note"] == "'=SUM(A1:A2)"
    assert rows_a[1]["L_star_nm"] == ""

    # Verify strict JSON manifest (allow_nan=False) and SHA-256 digests match every file.
    manifest_raw = manifest_path.read_text(encoding="utf-8")
    parsed = json.loads(manifest_raw)
    json.dumps(parsed, allow_nan=False)
    assert parsed["metadata"]["nan_field"] is None
    for rel_path, expected_sha in parsed["sha256"].items():
        actual_sha = hashlib.sha256((tmp_path / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha
    assert set(parsed["sha256"]) == {
        "fig1_butterfly.pdf",
        "fig1_butterfly.svg",
        "fig1_butterfly.png",
        "fig1_butterfly_source_data/panel_a.csv",
        "fig1_butterfly_source_data/panel_b.csv",
    }


def test_2d_and_polar_panels_clean_colorbar_ticks_wrapped_rays_and_intensity_modes():
    image, qx, qy, valid_mask = _synthetic_saxs_2d()
    # Low-count raw intensity scale (~0.03..3.0) reproducing BL19B2 raw colorbar challenge.
    low_img = 0.03 + 2.97 * ((image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image)))

    builder = CompositeFigureBuilder(preset="nature", width="double", height_mm=86.0, nrows=1, ncols=2, dpi=120)
    ax2d = builder.add_panel("a", 0, 0)
    axpol = builder.add_panel("b", 0, 1)

    res2d = plot_saxs_2d_panel(
        ax2d,
        low_img,
        qx,
        qy,
        valid_mask=valid_mask,
        q_unit="nm^-1",
        intensity_mode="raw_fallback",
        norm_mode="asinh",
        bisector_deg=2.0,
        half_opening_deg=38.0,
        title="2D SAXS (0.15 h, raw fb)",
    )
    assert res2d["rays_deg"] == pytest.approx([324.0, 40.0, 144.0, 220.0])
    assert len(res2d["colorbar_ticks"]) >= 2
    norm2d = res2d["norm"]
    pos2d = [float(norm2d(t)) for t in res2d["colorbar_ticks"]]
    assert all(pos2d[i + 1] - pos2d[i] >= 0.16 - 1e-6 for i in range(len(pos2d) - 1))
    cbar2d_label = res2d["colorbar"].ax.get_ylabel()
    assert "counts (raw fallback)" in cbar2d_label
    assert "cm" not in cbar2d_label.lower()

    # Polar map over [0, 360] with seam-crossing rays and raw intensity mode.
    q_edges = np.linspace(0.11, 1.0, 21)
    chi_edges = np.linspace(0.0, 360.0, 37)
    polar_z = np.full((20, 36), 0.05)
    polar_z[6:10, :] = 2.8
    res_pol = plot_polar_q_chi_panel(
        axpol,
        polar_z,
        q_edges,
        chi_edges,
        q_unit="nm^-1",
        intensity_mode="raw_counts",
        bisector_deg=2.0,
        half_opening_deg=38.0,
    )
    assert res_pol["rays_deg"] == pytest.approx([324.0, 40.0, 144.0, 220.0])
    assert sorted(res_pol["drawn_rays_deg"]) == pytest.approx([40.0, 144.0, 220.0, 324.0])
    assert all(0.0 <= ang <= 360.0 for ang in res_pol["drawn_rays_deg"])
    cbar_pol_label = res_pol["colorbar"].ax.get_ylabel()
    assert "counts (raw)" in cbar_pol_label
    assert "cm" not in cbar_pol_label.lower()
    assert len(res_pol["colorbar_ticks"]) >= 2

    builder.figure.canvas.draw()
    builder.figure.clear()


def _synthetic_dashboard_inputs(*, mixed: bool = False, q_unit: str = "nm^-1"):
    image, qx, qy, valid_mask = _synthetic_saxs_2d()
    q_edges = np.linspace(0.11, 1.0, 21)
    chi_edges = np.linspace(0.0, 360.0, 37)
    polar_z = np.full((20, 36), 4.0)
    polar_z[6:10, :] = 90.0
    q_grid = np.linspace(0.11, 1.0, 40)
    iq1 = 1.5 + 30.0 * np.exp(-0.5 * ((q_grid - 0.35) / 0.07) ** 2)
    iq2 = 1.5 + 80.0 * np.exp(-0.5 * ((q_grid - 0.30) / 0.06) ** 2)
    chi_az = np.linspace(0.0, 360.0, 73)
    bridge = 10.0 + 4.0 * np.cos(2.0 * np.deg2rad(chi_az - 2.0))
    lobes = [
        20.0 * np.exp(-0.5 * (((chi_az - mu + 180.0) % 360.0 - 180.0) / 14.0) ** 2)
        for mu in (324.0, 40.0, 144.0, 220.0)
    ]
    fit_total = bridge + sum(lobes)
    obs_az = fit_total + 0.3 * np.sin(np.deg2rad(chi_az * 3.0))

    rows = [
        {
            "frame_index": 0,
            "elapsed_h": 0.2,
            "status": "ok",
            "q_unit": q_unit,
            "calibration_mode": "calibrated",
            "L_star_direct_nm": 18.0 if q_unit == "nm^-1" else None,
            "L_star_kratky_nm": 12.0 if q_unit == "nm^-1" else None,
            "Q_win_sub": 2.5,
            "a4": 0.18,
            "wing_half_opening_deg": 38.0,
            "lobe_fwhm_deg": 24.0,
            "c2_r2": 0.92,
        },
        {
            "frame_index": 1,
            "elapsed_h": 1.0,
            "status": "ok",
            "q_unit": q_unit,
            "calibration_mode": "calibrated",
            "L_star_direct_nm": 20.5 if q_unit == "nm^-1" else None,
            "L_star_kratky_nm": 14.0 if q_unit == "nm^-1" else None,
            "Q_win_sub": 8.0,
            "a4": 0.45,
            "wing_half_opening_deg": 39.5,
            "lobe_fwhm_deg": 22.0,
            "c2_r2": 0.60,  # below 0.75 gate -> alpha/FWHM suppressed
        },
        {
            "frame_index": 2,
            "elapsed_h": 2.0,
            "status": "ok",
            "q_unit": q_unit,
            "calibration_mode": "raw_fallback" if mixed else "calibrated",
            "L_star_direct_nm": 22.0 if q_unit == "nm^-1" else None,
            "L_star_kratky_nm": 15.2 if q_unit == "nm^-1" else None,
            "Q_win_sub": 9999.0 if mixed else 12.4,
            "a4": 0.56,
            "wing_half_opening_deg": 40.2,
            "lobe_fwhm_deg": 20.5,
            "c2_r2": 0.95,
        },
    ]

    radial_curves = [
        {"label": "0.20 h", "q": q_grid, "intensity": iq1, "q_star_direct": 0.35, "calibration_mode": "calibrated"},
        {"label": "1.00 h", "q": q_grid, "intensity": iq2, "q_star_direct": 0.30, "calibration_mode": "calibrated"},
    ]
    if mixed:
        radial_curves.append(
            {"label": "2.00 h raw", "q": q_grid, "intensity": iq2 * 100.0, "calibration_mode": "raw_fallback"}
        )

    cache = {
        "q_unit": q_unit,
        "endpoint_intensity_mode": "raw_fallback" if mixed else "calibrated",
        "q_cutoffs": [{"q": 0.11, "label": "Beamstop", "shade_below": True}],
        "endpoint_2d": {
            "image": image,
            "qx": qx,
            "qy": qy,
            "valid_mask": valid_mask,
            "beamstop_q": 0.11,
            "bisector_deg": 2.0,
            "half_opening_deg": 38.0,
        },
        "endpoint_polar": {
            "polar_intensity": polar_z,
            "q_coords": q_edges,
            "chi_coords_deg": chi_edges,
            "bisector_deg": 2.0,
            "half_opening_deg": 38.0,
        },
        "radial_curves": radial_curves,
        "endpoint_azimuthal": {
            "chi_deg": chi_az,
            "observed": obs_az,
            "fit_total": fit_total,
            "c2_bridge": bridge,
            "lobe_components": lobes,
            "bisector_deg": 2.0,
            "half_opening_deg": 38.0,
            "r2": 0.991,
            "lobe_fwhm_deg": 22.4,
        },
    }
    return rows, cache


def test_render_sample_physics_dashboard_exports_6_panels_source_data_and_respects_modes(tmp_path):
    rows, cache = _synthetic_dashboard_inputs(mixed=False)
    stem = tmp_path / "dash_cal"
    manifest = render_sample_physics_dashboard(
        rows,
        cache,
        stem,
        sample_label="14_475C_02h_2026_run22_设定500C",
        intensity_mode="calibrated",
        journal_style="nature",
        formats=("pdf", "svg", "png"),
        dpi=150,
    )
    assert manifest["size_mm"][0] == pytest.approx(183.0, abs=0.05)
    assert manifest["intensity_mode"] == "calibrated"
    assert manifest["endpoint_intensity_mode"] == "calibrated"
    assert manifest["panel_summary"]["panel_d"]["q_win_plotted"] is True
    assert manifest["panel_summary"]["panel_e"]["n_r2_gated"] == 1

    # Strict JSON and SHA-256 verification.
    json.dumps(manifest, allow_nan=False)
    for rel_path, expected_sha in manifest["sha256"].items():
        actual_sha = hashlib.sha256((tmp_path / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha
    assert (tmp_path / "dash_cal.pdf").is_file()
    assert (tmp_path / "dash_cal.png").is_file()
    svg_text = (tmp_path / "dash_cal.svg").read_text(encoding="utf-8")
    # Zero CJK tofu glyphs inside SVG!
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in svg_text)

    # Mixed mode: panel c filters raw curve out, panel d drops raw Q_win, panel a/b/f tagged raw_fallback.
    rows_m, cache_m = _synthetic_dashboard_inputs(mixed=True)
    stem_m = tmp_path / "dash_mixed"
    man_m = render_sample_physics_dashboard(
        rows_m,
        cache_m,
        stem_m,
        sample_label="11_425C_02h_2026_run15",
        intensity_mode="mixed",
        formats=("svg",),
        dpi=100,
    )
    assert man_m["endpoint_intensity_mode"] == "raw_fallback"
    assert man_m["panel_summary"]["panel_c"]["calibrated_only_filtered"] is True
    assert man_m["panel_summary"]["panel_c"]["excluded_raw_curves"] == 1
    with (tmp_path / "dash_mixed_source_data" / "panel_d_spacing_invariant.csv").open(encoding="utf-8") as fh:
        d_rows = list(csv.DictReader(fh))
    assert d_rows[0]["Q_win_calibrated"] == "2.5"
    assert d_rows[2]["Q_win_calibrated"] == ""  # raw fallback frame excluded!

    # Pixel-q + raw-counts mode: no nm lengths and no calibrated Q_win plotted.
    rows_p, cache_p = _synthetic_dashboard_inputs(mixed=False, q_unit="pixel-q")
    for r in rows_p:
        r["calibration_mode"] = "raw_counts"
    cache_p["endpoint_intensity_mode"] = "raw_counts"
    man_p = render_sample_physics_dashboard(
        rows_p,
        cache_p,
        tmp_path / "dash_pixelq",
        intensity_mode="raw_counts",
        formats=("png",),
        dpi=100,
    )
    assert man_p["panel_summary"]["panel_d"]["physical_q"] is False
    assert man_p["panel_summary"]["panel_d"]["q_win_plotted"] is False


def test_pub_polish_modules_are_qt_free_and_pyplot_free():
    src_root = Path(__file__).resolve().parents[1] / "src" / "butterfly_saxs"
    forbidden_prefixes = ("matplotlib.pyplot", "PyQt", "PySide", "qtpy")
    for name in ("journal_style.py", "composite_figures.py", "comparison_report.py"):
        tree = ast.parse((src_root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), f"{name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_prefixes), f"{name} imports from {node.module}"
