from __future__ import annotations

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import AsinhNorm, LogNorm, Normalize
from matplotlib.figure import Figure
import numpy as np
import pytest

from butterfly_saxs.journal_style import (
    JOURNAL_PRESETS,
    JournalPreset,
    apply_journal_style,
    format_axis_label,
    format_math_unit,
    get_journal_preset,
    get_okabe_ito_palette,
    get_residual_colormap,
    get_saxs_colormap,
    get_thermal_palette,
    inches_to_mm,
    is_physical_q_unit,
    journal_rc_context,
    make_saxs_norm,
    mm_to_inches,
    q_to_period_nm,
    style_1d_axes,
    style_2d_axes,
    style_panel_label,
)


def test_journal_presets_define_required_targets_and_nature_defaults():
    assert set(JOURNAL_PRESETS) == {"nature", "nature_comm", "science", "macromolecules"}
    nature = JOURNAL_PRESETS["nature"]
    assert isinstance(nature, JournalPreset)
    assert nature.single_col_mm == pytest.approx(89.0)
    assert nature.mid_col_mm == pytest.approx(120.0)
    assert nature.double_col_mm == pytest.approx(183.0)
    assert nature.max_height_mm == pytest.approx(247.0)
    assert nature.font_family == "sans-serif"
    assert list(nature.sans_serif_fonts[:3]) == ["Arial", "Helvetica", "DejaVu Sans"]
    assert 6.5 <= nature.axis_label_size <= 7.0
    assert nature.tick_label_size == pytest.approx(6.0)
    assert nature.panel_label_size == pytest.approx(8.0)
    assert nature.panel_label_weight == "bold"
    assert nature.panel_label_case == "lower"
    assert 0.5 <= nature.spine_linewidth <= 0.6
    assert nature.tick_direction == "out"
    assert nature.tick_length == pytest.approx(2.5)
    assert nature.hide_top_right_spines_1d is True
    assert nature.pdf_fonttype == 42
    assert nature.svg_fonttype == "none"
    assert nature.dpi == 600
    assert nature.saxs_cmap in {"inferno", "magma", "cividis"}
    assert nature.to_dict()["name"] == "nature"


def test_get_journal_preset_aliases_and_errors():
    assert get_journal_preset("nature") is JOURNAL_PRESETS["nature"]
    assert get_journal_preset("Nature Comm") is JOURNAL_PRESETS["nature_comm"]
    assert get_journal_preset("nature-communications") is JOURNAL_PRESETS["nature_comm"]
    assert get_journal_preset("SCIENCE") is JOURNAL_PRESETS["science"]
    assert get_journal_preset("Macromolecules") is JOURNAL_PRESETS["macromolecules"]
    preset_obj = JOURNAL_PRESETS["nature"]
    assert get_journal_preset(preset_obj) is preset_obj

    with pytest.raises(ValueError, match="Unknown journal preset"):
        get_journal_preset("nonexistent_journal")
    with pytest.raises(ValueError, match="non-empty string"):
        get_journal_preset("")


def test_mm_to_inches_and_figsize_conversions():
    assert mm_to_inches(25.4) == pytest.approx(1.0)
    assert mm_to_inches(89.0) == pytest.approx(89.0 / 25.4)
    assert inches_to_mm(mm_to_inches(183.0)) == pytest.approx(183.0)
    w_in, h_in = JOURNAL_PRESETS["nature"].figsize_inches("double", height_mm=110.0)
    assert inches_to_mm(w_in) == pytest.approx(183.0)
    assert inches_to_mm(h_in) == pytest.approx(110.0)

    for bad in (0.0, -5.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            mm_to_inches(bad)


def test_journal_rc_context_and_apply_style_restore_cleanly():
    original_pdf = matplotlib.rcParams["pdf.fonttype"]
    original_svg = matplotlib.rcParams["svg.fonttype"]
    original_tick = matplotlib.rcParams["xtick.direction"]

    with journal_rc_context("nature") as preset:
        assert preset.name == "nature"
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["svg.fonttype"] == "none"
        assert matplotlib.rcParams["xtick.direction"] == "out"
        assert matplotlib.rcParams["xtick.major.size"] == pytest.approx(2.5)
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["axes.spines.right"] is False

    assert matplotlib.rcParams["pdf.fonttype"] == original_pdf
    assert matplotlib.rcParams["svg.fonttype"] == original_svg
    assert matplotlib.rcParams["xtick.direction"] == original_tick

    snapshot = dict(matplotlib.rcParams)
    try:
        applied = apply_journal_style("science")
        assert applied.name == "science"
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["svg.fonttype"] == "none"
    finally:
        matplotlib.rcParams.update(snapshot)


def test_palettes_are_colorblind_safe_valid_hex_and_non_aliasing():
    oi1 = get_okabe_ito_palette()
    oi2 = get_okabe_ito_palette()
    assert len(oi1) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in oi1)
    oi1.append("#123456")
    assert len(oi2) == 8

    for n in (1, 2, 3, 4, 5, 6, 9, 14):
        pal = get_thermal_palette(n)
        assert len(pal) == n
        assert all(c.startswith("#") and len(c) == 7 for c in pal)
        if n > 1:
            assert len(set(pal)) == n

    for bad in (0, -2, True):
        with pytest.raises(ValueError):
            get_thermal_palette(bad)


def test_format_math_unit_and_physical_q_gating():
    assert format_math_unit("nm^-1") == r"$\mathrm{nm}^{-1}$"
    assert format_math_unit("1/nm") == r"$\mathrm{nm}^{-1}$"
    assert format_math_unit("nm⁻¹") == r"$\mathrm{nm}^{-1}$"
    assert format_math_unit("cm^-1") == r"$\mathrm{cm}^{-1}$"
    assert format_math_unit("A^-1") == r"$\mathrm{\AA}^{-1}$"
    assert format_math_unit("Å^-1") == r"$\mathrm{\AA}^{-1}$"
    assert format_math_unit("cm^-1 nm^-3") == r"$\mathrm{cm}^{-1}\,\mathrm{nm}^{-3}$"
    assert format_math_unit("pixel-q") == "pixel-q"
    assert format_math_unit(r"$\mathrm{nm}^{-1}$") == r"$\mathrm{nm}^{-1}$"
    assert format_axis_label(r"$q$", "nm^-1") == r"$q$ ($\mathrm{nm}^{-1}$)"

    assert is_physical_q_unit("nm^-1") is True
    assert is_physical_q_unit("Å^-1") is True
    assert is_physical_q_unit("pixel-q") is False
    assert is_physical_q_unit("unknown") is False

    assert q_to_period_nm(0.25, "nm^-1") == pytest.approx(2.0 * np.pi / 0.25)
    assert q_to_period_nm(0.025, "A^-1") == pytest.approx(2.0 * np.pi / 0.25)
    assert q_to_period_nm(0.25, "pixel-q") is None
    assert q_to_period_nm(0.0, "nm^-1") is None


def test_saxs_colormap_blocks_diverging_and_sets_neutral_mask_color():
    cmap = get_saxs_colormap("inferno", preset="nature")
    bad_rgba = cmap.get_bad()
    # Neutral light gray #D9DDE2 is well above dark black (> 0.8 luminance), avoiding dark-on-dark traps.
    assert bad_rgba[0] > 0.8 and bad_rgba[1] > 0.8 and bad_rgba[2] > 0.8

    for bad_name in ("PuOr", "RdBu", "RdBu_r", "coolwarm", "jet", "seismic"):
        with pytest.raises(ValueError, match="diverging or non-perceptual"):
            get_saxs_colormap(bad_name)

    resid_cmap = get_residual_colormap(preset="nature")
    assert resid_cmap.name == "RdBu_r"


def test_make_saxs_norm_asinh_log_and_linear():
    rng = np.random.default_rng(7)
    data = np.exp(rng.normal(loc=1.0, scale=1.5, size=(32, 32)))
    data[0, 0] = np.nan

    asinh_norm = make_saxs_norm(data, mode="asinh")
    assert isinstance(asinh_norm, AsinhNorm)
    assert asinh_norm.vmin < asinh_norm.vmax
    assert asinh_norm.linear_width > 0.0

    log_norm = make_saxs_norm(data, mode="log")
    assert isinstance(log_norm, LogNorm)
    assert 0.0 < log_norm.vmin < log_norm.vmax

    lin_norm = make_saxs_norm(data, mode="linear", vmin=1.0, vmax=20.0)
    assert isinstance(lin_norm, Normalize)
    assert lin_norm.vmin == pytest.approx(1.0)
    assert lin_norm.vmax == pytest.approx(20.0)

    # Constant data should never crash with degenerate bounds.
    const_norm = make_saxs_norm(np.full((8, 8), 5.0), mode="asinh")
    assert const_norm.vmin < const_norm.vmax

    with pytest.raises(ValueError, match="strictly positive"):
        make_saxs_norm(np.zeros((4, 4)), mode="log")
    with pytest.raises(ValueError, match="mode must be"):
        make_saxs_norm(data, mode="invalid_mode")


def test_style_panel_label_and_axes_helpers():
    fig = Figure(figsize=(4.0, 3.0), dpi=100)
    FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    style_1d_axes(ax1, preset="nature")
    assert ax1.spines["top"].get_visible() is False
    assert ax1.spines["right"].get_visible() is False
    assert ax1.spines["left"].get_linewidth() == pytest.approx(0.55)

    style_2d_axes(ax2, preset="nature", equal_aspect=True)
    assert ax2.spines["top"].get_visible() is True
    assert ax2.spines["right"].get_visible() is True

    txt_a = style_panel_label(ax1, "a", preset="nature")
    assert txt_a.get_text() == "a"
    assert txt_a.get_fontsize() == pytest.approx(8.0)
    assert txt_a.get_fontweight() in {"bold", 700}

    txt_b = style_panel_label(ax2, "b", preset="nature", inside=True)
    assert txt_b.get_bbox_patch() is not None
    fig.clear()
