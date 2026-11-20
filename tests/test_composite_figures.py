from __future__ import annotations

import csv
import hashlib
import json
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
    export_publication_figure,
    plot_azimuthal_fit_panel,
    plot_evolution_panel,
    plot_polar_q_chi_panel,
    plot_radial_kratky_panel,
    plot_saxs_2d_panel,
    plot_streak_width_panel,
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
