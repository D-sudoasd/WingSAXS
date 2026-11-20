"""Publication-grade journal typography, dimensions, palettes, and SAXS norms.

This module is strictly Qt-free and headless-safe.  It defines publication presets
(Nature, Nature Communications, Science, ACS Macromolecules), perceptually
uniform sequential colormaps for positive SAXS intensities with clean neutral
masked-gap colors, colorblind-safe Okabe-Ito and thermal palettes, mathtext unit
formatting, and ``AsinhNorm`` / ``LogNorm`` helpers.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import re
from typing import Any

import matplotlib
from matplotlib.colors import AsinhNorm, Colormap, LinearSegmentedColormap, LogNorm, Normalize
import numpy as np


MM_PER_INCH = 25.4

_OKABE_ITO_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
)

OKABE_ITO_NAMED: dict[str, str] = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "bluish_green": "#009E73",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "black": "#000000",
    "yellow": "#F0E442",
}

# High-contrast, CVD-safe thermal progression on white background:
# deep navy -> blue -> teal -> ochre -> vermilion -> crimson.
_THERMAL_STOPS: tuple[str, ...] = (
    "#1F4E79",
    "#0072B2",
    "#009E73",
    "#D99B00",
    "#D55E00",
    "#9E1B32",
)

_THERMAL_CURATED: dict[int, tuple[str, ...]] = {
    1: ("#0072B2",),
    2: ("#0072B2", "#D55E00"),
    3: ("#1F4E79", "#D99B00", "#9E1B32"),
    4: ("#1F4E79", "#009E73", "#D99B00", "#9E1B32"),
    5: ("#1F4E79", "#0072B2", "#009E73", "#D55E00", "#9E1B32"),
    6: _THERMAL_STOPS,
}

# Diverging colormaps are strictly inappropriate for non-negative SAXS intensity
# maps; reserve them exclusively for signed residual maps.
_DISALLOWED_POSITIVE_SAXS_CMAPS: frozenset[str] = frozenset({
    "puor",
    "puor_r",
    "rdbu",
    "rdbu_r",
    "bwr",
    "bwr_r",
    "seismic",
    "seismic_r",
    "coolwarm",
    "coolwarm_r",
    "piyg",
    "piyg_r",
    "prgn",
    "prgn_r",
    "brbg",
    "brbg_r",
    "rdgy",
    "rdgy_r",
    "rdylbu",
    "rdylbu_r",
    "rdylgn",
    "rdylgn_r",
    "spectral",
    "spectral_r",
    "jet",
    "jet_r",
    "rainbow",
    "rainbow_r",
    "hsv",
    "hsv_r",
})

_DEFAULT_SANS_FONTS: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")


@dataclass(frozen=True)
class JournalPreset:
    """Declarative typography, geometry, and colormap rules for a target journal."""

    name: str
    display_name: str
    single_col_mm: float
    mid_col_mm: float
    double_col_mm: float
    max_height_mm: float
    font_family: str = "sans-serif"
    sans_serif_fonts: tuple[str, ...] = _DEFAULT_SANS_FONTS
    base_fontsize: float = 6.5
    axis_label_size: float = 7.0
    axis_title_size: float = 7.0
    tick_label_size: float = 6.0
    legend_fontsize: float = 6.0
    panel_label_size: float = 8.0
    panel_label_weight: str = "bold"
    panel_label_case: str = "lower"
    spine_linewidth: float = 0.55
    line_linewidth: float = 1.1
    tick_direction: str = "out"
    tick_length: float = 2.5
    tick_width: float = 0.55
    dpi: int = 600
    pdf_fonttype: int = 42
    ps_fonttype: int = 42
    svg_fonttype: str = "none"
    saxs_cmap: str = "inferno"
    polar_cmap: str = "inferno"
    residual_cmap: str = "RdBu_r"
    mask_color: str = "#D9DDE2"
    background_color: str = "#FFFFFF"
    grid_alpha: float = 0.18
    grid_linewidth: float = 0.4
    hide_top_right_spines_1d: bool = True

    def width_mm(self, width: str | float = "double") -> float:
        """Resolve a column width name ('single', 'mid', 'double') or numeric mm."""
        if isinstance(width, (int, float)) and not isinstance(width, bool):
            value = float(width)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Figure width in mm must be positive and finite, got {width!r}")
            return value
        key = str(width).strip().lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "single": self.single_col_mm,
            "single_col": self.single_col_mm,
            "1col": self.single_col_mm,
            "mid": self.mid_col_mm,
            "medium": self.mid_col_mm,
            "1.5col": self.mid_col_mm,
            "double": self.double_col_mm,
            "double_col": self.double_col_mm,
            "2col": self.double_col_mm,
            "full": self.double_col_mm,
        }
        if key not in mapping:
            raise ValueError(
                f"Unknown width specification {width!r}; expected 'single', 'mid', 'double', or a positive mm value"
            )
        return mapping[key]

    def figsize_inches(
        self,
        width: str | float = "double",
        *,
        height_mm: float | None = None,
        aspect: float = 0.62,
    ) -> tuple[float, float]:
        """Return ``(width_in, height_in)`` honoring journal column width and max height."""
        w_mm = self.width_mm(width)
        if height_mm is None:
            if not math.isfinite(aspect) or aspect <= 0.0:
                raise ValueError(f"aspect must be positive and finite, got {aspect!r}")
            h_mm = min(w_mm * float(aspect), self.max_height_mm)
        else:
            h_mm = float(height_mm)
            if not math.isfinite(h_mm) or h_mm <= 0.0:
                raise ValueError(f"height_mm must be positive and finite, got {height_mm!r}")
        return mm_to_inches(w_mm), mm_to_inches(h_mm)

    def rc_params(self) -> dict[str, Any]:
        """Build the complete Matplotlib ``rcParams`` dictionary for this preset."""
        return {
            "font.family": self.font_family,
            "font.sans-serif": list(self.sans_serif_fonts),
            "font.size": self.base_fontsize,
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": self.axis_label_size,
            "axes.titlesize": self.axis_title_size,
            "axes.titleweight": "normal",
            "axes.linewidth": self.spine_linewidth,
            "axes.edgecolor": "#222222",
            "axes.facecolor": self.background_color,
            "axes.unicode_minus": False,
            "axes.spines.top": not self.hide_top_right_spines_1d,
            "axes.spines.right": not self.hide_top_right_spines_1d,
            "xtick.direction": self.tick_direction,
            "ytick.direction": self.tick_direction,
            "xtick.major.size": self.tick_length,
            "ytick.major.size": self.tick_length,
            "xtick.minor.size": self.tick_length * 0.6,
            "ytick.minor.size": self.tick_length * 0.6,
            "xtick.major.width": self.tick_width,
            "ytick.major.width": self.tick_width,
            "xtick.minor.width": self.tick_width * 0.8,
            "ytick.minor.width": self.tick_width * 0.8,
            "xtick.labelsize": self.tick_label_size,
            "ytick.labelsize": self.tick_label_size,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "lines.linewidth": self.line_linewidth,
            "lines.markersize": 3.5,
            "legend.fontsize": self.legend_fontsize,
            "legend.frameon": True,
            "legend.framealpha": 0.90,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#CCCCCC",
            "legend.borderpad": 0.3,
            "legend.handlelength": 1.4,
            "legend.labelspacing": 0.25,
            "figure.facecolor": self.background_color,
            "figure.edgecolor": self.background_color,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "savefig.facecolor": self.background_color,
            "savefig.edgecolor": self.background_color,
            "pdf.fonttype": self.pdf_fonttype,
            "ps.fonttype": self.ps_fonttype,
            "svg.fonttype": self.svg_fonttype,
            "grid.alpha": self.grid_alpha,
            "grid.linewidth": self.grid_linewidth,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the preset."""
        payload = asdict(self)
        payload["sans_serif_fonts"] = list(self.sans_serif_fonts)
        return payload


JOURNAL_PRESETS: dict[str, JournalPreset] = {
    "nature": JournalPreset(
        name="nature",
        display_name="Nature",
        single_col_mm=89.0,
        mid_col_mm=120.0,
        double_col_mm=183.0,
        max_height_mm=247.0,
        base_fontsize=6.5,
        axis_label_size=7.0,
        axis_title_size=7.0,
        tick_label_size=6.0,
        legend_fontsize=6.0,
        panel_label_size=8.0,
        panel_label_weight="bold",
        panel_label_case="lower",
        spine_linewidth=0.55,
        tick_length=2.5,
        tick_width=0.55,
        dpi=600,
        saxs_cmap="inferno",
        polar_cmap="inferno",
    ),
    "nature_comm": JournalPreset(
        name="nature_comm",
        display_name="Nature Communications",
        single_col_mm=88.0,
        mid_col_mm=120.0,
        double_col_mm=180.0,
        max_height_mm=225.0,
        base_fontsize=6.5,
        axis_label_size=7.0,
        axis_title_size=7.0,
        tick_label_size=6.0,
        legend_fontsize=6.0,
        panel_label_size=8.0,
        panel_label_weight="bold",
        panel_label_case="lower",
        spine_linewidth=0.55,
        tick_length=2.5,
        tick_width=0.55,
        dpi=600,
        saxs_cmap="inferno",
        polar_cmap="inferno",
    ),
    "science": JournalPreset(
        name="science",
        display_name="Science",
        single_col_mm=55.0,
        mid_col_mm=120.0,
        double_col_mm=175.0,
        max_height_mm=233.0,
        base_fontsize=6.5,
        axis_label_size=7.0,
        axis_title_size=7.0,
        tick_label_size=6.0,
        legend_fontsize=6.0,
        panel_label_size=8.0,
        panel_label_weight="bold",
        panel_label_case="upper",
        spine_linewidth=0.5,
        tick_length=2.5,
        tick_width=0.5,
        dpi=600,
        saxs_cmap="inferno",
        polar_cmap="inferno",
    ),
    "macromolecules": JournalPreset(
        name="macromolecules",
        display_name="Macromolecules",
        single_col_mm=84.7,
        mid_col_mm=120.0,
        double_col_mm=177.8,
        max_height_mm=240.0,
        base_fontsize=7.0,
        axis_label_size=7.0,
        axis_title_size=7.0,
        tick_label_size=6.0,
        legend_fontsize=6.0,
        panel_label_size=8.0,
        panel_label_weight="bold",
        panel_label_case="lower",
        spine_linewidth=0.6,
        tick_length=2.5,
        tick_width=0.6,
        dpi=600,
        saxs_cmap="cividis",
        polar_cmap="cividis",
    ),
}

_PRESET_ALIASES: dict[str, str] = {
    "nature": "nature",
    "nature_materials": "nature",
    "nature_comm": "nature_comm",
    "nature_communications": "nature_comm",
    "natcomm": "nature_comm",
    "science": "science",
    "macromolecules": "macromolecules",
    "acs": "macromolecules",
    "acs_macromolecules": "macromolecules",
}


def get_journal_preset(name: str | JournalPreset) -> JournalPreset:
    """Resolve a journal preset name or return an existing ``JournalPreset``."""
    if isinstance(name, JournalPreset):
        return name
    if not isinstance(name, str) or not name.strip():
        supported = ", ".join(sorted(JOURNAL_PRESETS))
        raise ValueError(f"Journal preset name must be a non-empty string; supported: {supported}")
    normalized = re.sub(r"[\s\-]+", "_", name.strip().lower())
    canonical = _PRESET_ALIASES.get(normalized, normalized)
    if canonical not in JOURNAL_PRESETS:
        supported = ", ".join(sorted(JOURNAL_PRESETS))
        raise ValueError(f"Unknown journal preset {name!r}; supported presets: {supported}")
    return JOURNAL_PRESETS[canonical]


def mm_to_inches(mm: float) -> float:
    """Convert millimeters to inches, validating a strictly positive finite length."""
    if isinstance(mm, bool):
        raise ValueError("mm must be a positive real number, not bool")
    value = float(mm)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"mm must be a strictly positive finite number, got {mm!r}")
    return value / MM_PER_INCH


def inches_to_mm(inches: float) -> float:
    """Convert inches to millimeters, validating a strictly positive finite length."""
    if isinstance(inches, bool):
        raise ValueError("inches must be a positive real number, not bool")
    value = float(inches)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"inches must be a strictly positive finite number, got {inches!r}")
    return value * MM_PER_INCH


def apply_journal_style(preset: str | JournalPreset = "nature") -> JournalPreset:
    """Apply journal typography and export settings to ``matplotlib.rcParams``."""
    resolved = get_journal_preset(preset)
    matplotlib.rcParams.update(resolved.rc_params())
    return resolved


@contextmanager
def journal_rc_context(
    preset: str | JournalPreset = "nature",
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Iterator[JournalPreset]:
    """Temporarily apply a journal preset without mutating process-global rcParams."""
    resolved = get_journal_preset(preset)
    params = resolved.rc_params()
    if overrides:
        params.update(overrides)
    with matplotlib.rc_context(params):
        yield resolved


def get_okabe_ito_palette() -> list[str]:
    """Return a fresh list of the 8 colorblind-safe Okabe-Ito hex colors."""
    return list(_OKABE_ITO_PALETTE)


def get_thermal_palette(n: int) -> list[str]:
    """Return ``n`` high-contrast, colorblind-safe thermal progression hex colors."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if n in _THERMAL_CURATED:
        return list(_THERMAL_CURATED[n])
    cmap = LinearSegmentedColormap.from_list("wingsaxs_thermal", list(_THERMAL_STOPS), N=max(256, n))
    positions = np.linspace(0.0, 1.0, n)
    colors: list[str] = []
    for pos in positions:
        r, g, b, _a = cmap(float(pos))
        colors.append(f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}")
    return colors


_EXACT_UNIT_MAP: dict[str, str] = {
    "nm^-1": r"$\mathrm{nm}^{-1}$",
    "nm-1": r"$\mathrm{nm}^{-1}$",
    "1/nm": r"$\mathrm{nm}^{-1}$",
    "nm⁻¹": r"$\mathrm{nm}^{-1}$",
    "nm−1": r"$\mathrm{nm}^{-1}$",
    "cm^-1": r"$\mathrm{cm}^{-1}$",
    "cm-1": r"$\mathrm{cm}^{-1}$",
    "1/cm": r"$\mathrm{cm}^{-1}$",
    "cm⁻¹": r"$\mathrm{cm}^{-1}$",
    "cm−1": r"$\mathrm{cm}^{-1}$",
    "a^-1": r"$\mathrm{\AA}^{-1}$",
    "a-1": r"$\mathrm{\AA}^{-1}$",
    "1/a": r"$\mathrm{\AA}^{-1}$",
    "å^-1": r"$\mathrm{\AA}^{-1}$",
    "å-1": r"$\mathrm{\AA}^{-1}$",
    "1/å": r"$\mathrm{\AA}^{-1}$",
    "å⁻¹": r"$\mathrm{\AA}^{-1}$",
    "å−1": r"$\mathrm{\AA}^{-1}$",
    "angstrom^-1": r"$\mathrm{\AA}^{-1}$",
    "m^-1": r"$\mathrm{m}^{-1}$",
    "1/m": r"$\mathrm{m}^{-1}$",
    "s^-1": r"$\mathrm{s}^{-1}$",
    "1/s": r"$\mathrm{s}^{-1}$",
    "min^-1": r"$\mathrm{min}^{-1}$",
    "1/min": r"$\mathrm{min}^{-1}$",
    "h^-1": r"$\mathrm{h}^{-1}$",
    "1/h": r"$\mathrm{h}^{-1}$",
    "kj/mol": r"$\mathrm{kJ\,mol}^{-1}$",
    "kj mol^-1": r"$\mathrm{kJ\,mol}^{-1}$",
    "um": r"$\mu\mathrm{m}$",
    "µm": r"$\mu\mathrm{m}$",
    "μm": r"$\mu\mathrm{m}$",
    "deg": r"$^\circ$",
    "degree": r"$^\circ$",
    "degrees": r"$^\circ$",
    "°": r"$^\circ$",
    "degc": r"$^\circ\mathrm{C}$",
    "°c": r"$^\circ\mathrm{C}$",
    "pixel-q": "pixel-q",
    "pixel_q": "pixel-q",
    "pixelq": "pixel-q",
    "pixel": "pixel-q",
}

_SUPERSCRIPT_TRANSLATION = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "−": "-",
    "⁺": "+",
})

_TOKEN_POWER_RE = re.compile(r"^([A-Za-zÅåµμ]+)\^?([+-]?\d+(?:\.\d+)?)$")


def format_math_unit(unit: str) -> str:
    """Convert ASCII/Unicode scientific unit strings into clean Matplotlib mathtext.

    Examples:
        ``"nm^-1"`` -> ``r"$\\mathrm{nm}^{-1}$"``
        ``"cm^-1"`` -> ``r"$\\mathrm{cm}^{-1}$"``
        ``"cm^-1 nm^-3"`` -> ``r"$\\mathrm{cm}^{-1}\\,\\mathrm{nm}^{-3}$"``
        ``"pixel-q"`` -> ``"pixel-q"``
    """
    if unit is None:
        return ""
    raw = str(unit).strip()
    if not raw:
        return ""
    if raw.startswith("$") and raw.endswith("$") and len(raw) >= 2:
        return raw

    compact = re.sub(r"\s+", " ", raw)
    lookup_key = compact.lower()
    if lookup_key in _EXACT_UNIT_MAP:
        return _EXACT_UNIT_MAP[lookup_key]
    compact_nospace = lookup_key.replace(" ", "")
    if compact_nospace in _EXACT_UNIT_MAP:
        return _EXACT_UNIT_MAP[compact_nospace]

    # Translate Unicode superscripts into ASCII exponents and parse compound tokens.
    normalized = compact.translate(_SUPERSCRIPT_TRANSLATION)
    tokens = [tok for tok in re.split(r"[\s·*]+", normalized) if tok]
    if not tokens:
        return raw

    math_parts: list[str] = []
    any_math = False
    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower in _EXACT_UNIT_MAP and _EXACT_UNIT_MAP[tok_lower].startswith("$"):
            math_parts.append(_EXACT_UNIT_MAP[tok_lower].strip("$"))
            any_math = True
            continue
        match = _TOKEN_POWER_RE.match(tok)
        if match:
            base, exp = match.group(1), match.group(2)
            if base in {"Å", "å"}:
                base_tex = r"\mathrm{\AA}"
            elif base.startswith(("µ", "μ")):
                base_tex = r"\mu\mathrm{" + base[1:] + "}"
            else:
                base_tex = rf"\mathrm{{{base}}}"
            math_parts.append(f"{base_tex}^{{{exp}}}")
            any_math = True
        else:
            if tok in {"Å", "å"}:
                math_parts.append(r"\mathrm{\AA}")
                any_math = True
            else:
                math_parts.append(rf"\mathrm{{{tok}}}")

    if any_math:
        return "$" + r"\,".join(math_parts) + "$"
    return compact


def format_axis_label(quantity: str, unit: str | None = None) -> str:
    """Format an axis label as ``<quantity> (<unit>)`` with proper mathtext units."""
    qty = str(quantity).strip()
    if not unit:
        return qty
    formatted_unit = format_math_unit(unit)
    if not formatted_unit or formatted_unit.lower() in {"dimensionless", "none", "unit unspecified"}:
        return qty
    return f"{qty} ({formatted_unit})"


def is_physical_q_unit(unit: Any) -> bool:
    """Return True only for calibrated physical reciprocal-length units."""
    text = str(unit or "").strip().lower().replace(" ", "")
    return text in {
        "nm^-1",
        "nm-1",
        "1/nm",
        "nm⁻¹",
        "nm−1",
        "a^-1",
        "a-1",
        "1/a",
        "å^-1",
        "å-1",
        "1/å",
        "å⁻¹",
        "å−1",
        "angstrom^-1",
    }


def q_to_period_nm(q_value: float, q_unit: Any) -> float | None:
    """Convert calibrated physical ``q`` to real-space period ``2*pi/q`` in nm.

    Returns ``None`` for pixel-q, unknown units, or non-positive/non-finite ``q``.
    """
    try:
        q = float(q_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(q) or q <= 0.0:
        return None
    text = str(q_unit or "").strip().lower().replace(" ", "")
    if text in {"nm^-1", "nm-1", "1/nm", "nm⁻¹", "nm−1"}:
        return float(2.0 * math.pi / q)
    if text in {"a^-1", "a-1", "1/a", "å^-1", "å-1", "1/å", "å⁻¹", "å−1", "angstrom^-1"}:
        # 2*pi / (q [A^-1]) gives Angstroms; divide by 10 for nm.
        return float((2.0 * math.pi / q) / 10.0)
    return None


def style_panel_label(
    ax: Any,
    label: str,
    *,
    preset: str | JournalPreset = "nature",
    x: float | None = None,
    y: float | None = None,
    inside: bool = False,
    fontsize: float | None = None,
    fontweight: str | None = None,
    color: str = "#111111",
    transform_case: bool = False,
    **kwargs: Any,
) -> Any:
    """Place a crisp publication panel letter (e.g., bold lowercase ``a`` for Nature)."""
    resolved = get_journal_preset(preset)
    text = str(label).strip()
    if transform_case and len(text) == 1 and text.isalpha():
        text = text.upper() if resolved.panel_label_case == "upper" else text.lower()

    if inside:
        px = 0.035 if x is None else float(x)
        py = 0.965 if y is None else float(y)
        ha = kwargs.pop("ha", "left")
        va = kwargs.pop("va", "top")
        bbox = kwargs.pop(
            "bbox",
            {
                "facecolor": "#FFFFFF",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 1.4,
            },
        )
    else:
        px = -0.11 if x is None else float(x)
        py = 1.04 if y is None else float(y)
        ha = kwargs.pop("ha", "left")
        va = kwargs.pop("va", "bottom")
        bbox = kwargs.pop("bbox", None)

    return ax.text(
        px,
        py,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=resolved.panel_label_size if fontsize is None else float(fontsize),
        fontweight=resolved.panel_label_weight if fontweight is None else fontweight,
        color=color,
        bbox=bbox,
        clip_on=False,
        zorder=kwargs.pop("zorder", 10),
        **kwargs,
    )


def style_1d_axes(
    ax: Any,
    *,
    preset: str | JournalPreset = "nature",
    grid: bool = False,
) -> None:
    """Style a 1D Cartesian axes per journal conventions (clean open top/right spines)."""
    resolved = get_journal_preset(preset)
    if resolved.hide_top_right_spines_1d:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for name in ("left", "bottom", "top", "right"):
        if name in ax.spines:
            ax.spines[name].set_linewidth(resolved.spine_linewidth)
            ax.spines[name].set_color("#222222")
    ax.tick_params(
        which="both",
        direction=resolved.tick_direction,
        width=resolved.tick_width,
        labelsize=resolved.tick_label_size,
        colors="#222222",
    )
    ax.tick_params(which="major", length=resolved.tick_length)
    ax.tick_params(which="minor", length=resolved.tick_length * 0.6)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, alpha=resolved.grid_alpha, linewidth=resolved.grid_linewidth, color="#888888")
    else:
        ax.grid(False)


def style_2d_axes(
    ax: Any,
    *,
    preset: str | JournalPreset = "nature",
    equal_aspect: bool = False,
) -> None:
    """Style a 2D map axes with crisp 4-sided frame and outward ticks."""
    resolved = get_journal_preset(preset)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(resolved.spine_linewidth)
        spine.set_color("#222222")
    ax.tick_params(
        which="both",
        direction=resolved.tick_direction,
        width=resolved.tick_width,
        labelsize=resolved.tick_label_size,
        colors="#222222",
    )
    ax.tick_params(which="major", length=resolved.tick_length)
    ax.tick_params(which="minor", length=resolved.tick_length * 0.6)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def get_saxs_colormap(
    name: str | None = None,
    *,
    preset: str | JournalPreset = "nature",
    mask_color: str | None = None,
) -> Colormap:
    """Return a sequential perceptual colormap for positive SAXS maps with neutral masked color."""
    resolved = get_journal_preset(preset)
    cmap_name = resolved.saxs_cmap if name is None else str(name).strip()
    if cmap_name.lower() in _DISALLOWED_POSITIVE_SAXS_CMAPS:
        raise ValueError(
            f"Colormap {cmap_name!r} is diverging or non-perceptual and must not be used for "
            "positive SAXS intensity maps; use 'inferno', 'magma', 'cividis', or 'viridis'."
        )
    bad = resolved.mask_color if mask_color is None else mask_color
    return matplotlib.colormaps[cmap_name].with_extremes(bad=bad)


def get_residual_colormap(
    name: str | None = None,
    *,
    preset: str | JournalPreset = "nature",
    mask_color: str | None = None,
) -> Colormap:
    """Return a diverging colormap reserved for signed residuals (observed - model)."""
    resolved = get_journal_preset(preset)
    cmap_name = resolved.residual_cmap if name is None else str(name).strip()
    bad = resolved.mask_color if mask_color is None else mask_color
    return matplotlib.colormaps[cmap_name].with_extremes(bad=bad)


def make_saxs_norm(
    data: Any,
    mode: str = "asinh",
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.5,
    linear_width: float | None = None,
) -> Normalize:
    """Construct a robust Matplotlib normalization for positive SAXS intensities.

    Parameters
    ----------
    data:
        Observed intensity array (may contain ``NaN``/``Inf`` or be a masked array).
    mode:
        ``"asinh"`` (default, dynamic-range compression smooth through low intensity),
        ``"log"`` (strictly positive logarithmic scale), or ``"linear"``.
    vmin, vmax:
        Optional explicit display limits.
    """
    norm_mode = str(mode).strip().lower()
    if norm_mode not in {"asinh", "log", "linear"}:
        raise ValueError(f"mode must be 'asinh', 'log', or 'linear', got {mode!r}")

    masked = np.ma.array(data, copy=False)
    raw = np.asarray(np.ma.getdata(masked), dtype=np.float64)
    mask = np.asarray(np.ma.getmaskarray(masked), dtype=bool)
    finite = raw[(~mask) & np.isfinite(raw)]

    if norm_mode == "log":
        pos = finite[finite > 0.0]
        if vmin is None:
            if not pos.size:
                raise ValueError("LogNorm requires at least one strictly positive finite value")
            lo = max(float(np.percentile(pos, low_percentile)), np.finfo(np.float64).tiny)
        else:
            lo = float(vmin)
            if not math.isfinite(lo) or lo <= 0.0:
                raise ValueError(f"LogNorm requires positive finite vmin, got {vmin!r}")
        if vmax is None:
            if not pos.size:
                raise ValueError("LogNorm requires at least one strictly positive finite value")
            hi = float(np.percentile(pos, high_percentile))
        else:
            hi = float(vmax)
            if not math.isfinite(hi) or hi <= 0.0:
                raise ValueError(f"LogNorm requires positive finite vmax, got {vmax!r}")
        if hi <= lo:
            hi = max(lo * 10.0, lo + 1e-6)
        return LogNorm(vmin=lo, vmax=hi, clip=True)

    if finite.size:
        lo = float(np.percentile(finite, low_percentile)) if vmin is None else float(vmin)
        hi = float(np.percentile(finite, high_percentile)) if vmax is None else float(vmax)
    else:
        lo = 0.0 if vmin is None else float(vmin)
        hi = 1.0 if vmax is None else float(vmax)

    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"vmin and vmax must be finite, got vmin={lo!r}, vmax={hi!r}")
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1.0)
        lo, hi = lo - pad, hi + pad

    if norm_mode == "linear":
        return Normalize(vmin=lo, vmax=hi, clip=True)

    # mode == "asinh"
    if linear_width is None:
        pos_mag = np.abs(finite[finite > 0.0]) if finite.size else np.asarray([], dtype=np.float64)
        if pos_mag.size:
            lw = float(np.percentile(pos_mag, 20.0))
        else:
            lw = 0.1 * (hi - lo)
        lw = max(lw, (hi - lo) * 1e-4, 1e-12)
    else:
        lw = float(linear_width)
        if not math.isfinite(lw) or lw <= 0.0:
            raise ValueError(f"linear_width must be strictly positive and finite, got {linear_width!r}")
    return AsinhNorm(linear_width=lw, vmin=lo, vmax=hi, clip=True)


def _clean_tick_float(val: float) -> float:
    cleaned = float(f"{float(val):.12g}")
    return 0.0 if cleaned == 0.0 else cleaned


def _eval_norm_position(norm: Any, val: float) -> float | None:
    try:
        raw = norm(float(val))
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        return None
    if np.ma.isMaskedArray(raw):
        if bool(np.ma.is_masked(raw)):
            return None
        raw = np.ma.getdata(raw)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size != 1:
        return None
    pos = float(arr.reshape(-1)[0])
    if not math.isfinite(pos) or pos < -1e-6 or pos > 1.0 + 1e-6:
        return None
    return min(1.0, max(0.0, pos))


def format_clean_tick_label(val: float) -> str:
    """Format a colorbar or axis tick value compactly without noisy float artifacts.

    Examples:
        ``0.0`` -> ``"0"``
        ``0.030000000000000004`` -> ``"0.03"``
        ``1000.0`` -> ``"1e3"``
        ``0.001`` -> ``"1e-3"``
    """
    if isinstance(val, bool):
        raise ValueError("val must be a real number, not bool")
    try:
        fval = float(val)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"val must be numeric, got {val!r}") from exc
    if not math.isfinite(fval):
        return ""
    abs_v = abs(fval)
    if abs_v < 1e-15:
        return "0"
    sign = "-" if fval < 0.0 else ""

    rounded_fixed = round(abs_v, 4)
    if 0.01 <= rounded_fixed < 1000.0 and abs_v >= 0.01 - 1e-12:
        body = f"{rounded_fixed:.4f}".rstrip("0").rstrip(".")
        return f"{sign}{body}"

    exp = math.floor(math.log10(abs_v))
    mant = abs_v / (10.0 ** exp)
    mant_r = round(mant, 3)
    if mant_r >= 10.0:
        mant_r = round(mant_r / 10.0, 3)
        exp += 1
    if abs(mant_r - round(mant_r)) < 1e-9:
        mant_str = str(int(round(mant_r)))
    else:
        mant_str = f"{mant_r:.3f}".rstrip("0").rstrip(".")
    return f"{sign}{mant_str}e{exp}"


def _linear_nice_ticks(
    norm: Any,
    lo: float,
    hi: float,
    *,
    min_visual_gap: float,
    max_ticks: int,
) -> list[tuple[float, float]]:
    span = hi - lo
    if span <= 0.0:
        pos = _eval_norm_position(norm, lo)
        return [(_clean_tick_float(lo), pos)] if pos is not None else []
    min_step = span * max(min_visual_gap, 1.0 / max(max_ticks, 1))
    base_exp = math.floor(math.log10(max(min_step, 1e-30)))
    eps = 1e-9 * span

    for exp in range(base_exp - 1, base_exp + 4):
        for mult in (1.0, 2.0, 2.5, 5.0):
            step = mult * (10.0 ** exp)
            if step <= 0.0 or not math.isfinite(step):
                continue
            k_start = math.ceil((lo - eps) / step)
            k_end = math.floor((hi + eps) / step)
            count = k_end - k_start + 1
            if count < 1 or count > max_ticks:
                continue
            pts: list[tuple[float, float]] = []
            ok = True
            for k in range(k_start, k_end + 1):
                tv = _clean_tick_float(k * step)
                if tv < lo - eps or tv > hi + eps:
                    continue
                tv_clamped = min(hi, max(lo, tv))
                pos = _eval_norm_position(norm, tv_clamped)
                if pos is None:
                    ok = False
                    break
                if pts and abs(pos - pts[-1][1]) < min_visual_gap - 1e-9:
                    ok = False
                    break
                pts.append((_clean_tick_float(tv_clamped), pos))
            if ok and len(pts) >= 2:
                return pts
    return []


def format_clean_colorbar_ticks(
    norm: Any,
    vmin: float,
    vmax: float,
    *,
    min_visual_gap: float = 0.16,
    max_ticks: int = 5,
) -> list[float]:
    """Select crisp, non-overlapping numeric colorbar ticks filtered by normalized visual spacing.

    Generates human-friendly ``1-2-5`` decades and subticks (with linear step fallback on
    narrow ranges) and enforces ``|norm(t_i) - norm(t_j)| >= min_visual_gap`` so ``AsinhNorm``
    and ``LogNorm`` colorbars never suffer colliding bottom tick labels.
    """
    if isinstance(vmin, bool) or isinstance(vmax, bool):
        raise ValueError("vmin and vmax must be finite real numbers, not bool")
    if isinstance(min_visual_gap, bool):
        raise ValueError("min_visual_gap must be a positive real number, not bool")
    if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks < 1:
        raise ValueError(f"max_ticks must be an integer >= 1, got {max_ticks!r}")

    lo = float(vmin)
    hi = float(vmax)
    gap = float(min_visual_gap)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"vmin and vmax must be finite, got vmin={vmin!r}, vmax={vmax!r}")
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError(f"min_visual_gap must be positive and finite, got {min_visual_gap!r}")
    if hi < lo:
        lo, hi = hi, lo
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        pos = _eval_norm_position(norm, lo)
        return [_clean_tick_float(lo)] if pos is not None else []

    # Pure linear Normalize: prefer uniform nice linear 1-2-2.5-5 step ticks.
    if type(norm) is Normalize:
        lin_pts = _linear_nice_ticks(norm, lo, hi, min_visual_gap=gap, max_ticks=max_ticks)
        if lin_pts:
            return [val for val, _ in lin_pts]

    span = hi - lo
    eps = 1e-12 * max(span, abs(lo), abs(hi), 1.0)
    tier0_raw: list[float] = []
    tier1_raw: list[tuple[float, int]] = []
    tier2_raw: list[tuple[float, int]] = []

    if lo <= 0.0 <= hi:
        tier0_raw.append(0.0)

    def _add_side(side_lo: float, side_hi: float, sign: float) -> None:
        if side_hi <= 0.0:
            return
        exp_max = math.ceil(math.log10(side_hi))
        if side_lo > 0.0:
            exp_min = math.floor(math.log10(side_lo)) - 1
        else:
            lw = getattr(norm, "linear_width", None)
            if lw is not None and not isinstance(lw, bool):
                try:
                    lw_f = float(lw)
                except (TypeError, ValueError):
                    lw_f = 0.0
            else:
                lw_f = 0.0
            ref_lo = min(side_hi, lw_f) * 0.1 if (math.isfinite(lw_f) and lw_f > 0.0) else side_hi * 1e-4
            exp_min = math.floor(math.log10(max(ref_lo, 1e-30))) - 1
        exp_min = max(exp_min, exp_max - 12)
        for exp in range(exp_min, exp_max + 1):
            base = 10.0 ** exp
            v1 = _clean_tick_float(sign * base)
            if lo - eps <= v1 <= hi + eps:
                tier0_raw.append(min(hi, max(lo, v1)) if (v1 < lo or v1 > hi) else v1)
            for mult, prio in ((5.0, 2), (2.0, 1)):
                vm = _clean_tick_float(sign * mult * base)
                if lo - eps <= vm <= hi + eps:
                    tier1_raw.append((min(hi, max(lo, vm)) if (vm < lo or vm > hi) else vm, prio))
            v3 = _clean_tick_float(sign * 3.0 * base)
            if lo - eps <= v3 <= hi + eps:
                tier2_raw.append((min(hi, max(lo, v3)) if (v3 < lo or v3 > hi) else v3, 0))

    if hi > 0.0:
        _add_side(max(lo, 0.0), hi, 1.0)
    if lo < 0.0:
        _add_side(max(-hi, 0.0), -lo, -1.0)

    def _dedup_eval(items: list[tuple[float, int]]) -> list[tuple[float, float, int]]:
        seen: dict[float, tuple[float, float, int]] = {}
        for val, prio in items:
            cval = _clean_tick_float(val)
            if cval in seen:
                continue
            pos = _eval_norm_position(norm, cval)
            if pos is None:
                continue
            seen[cval] = (cval, pos, prio)
        return list(seen.values())

    tier0 = _dedup_eval([(v, 3) for v in tier0_raw])
    tier1 = _dedup_eval(tier1_raw)
    tier2 = _dedup_eval(tier2_raw)

    # Step 1: Filter pure decades (and 0) cleanly from top down, preferring 0 over sub-linear-width decades.
    selected: list[tuple[float, float]] = []
    zero_entry = next(((v, p) for v, p, _ in tier0 if v == 0.0), None)
    if zero_entry is not None:
        selected.append(zero_entry)

    non_zero_t0 = sorted(((v, p) for v, p, _ in tier0 if v != 0.0), key=lambda item: item[1], reverse=True)
    for val, pos in non_zero_t0:
        if all(abs(pos - sp) >= gap - 1e-9 for _, sp in selected):
            selected.append((val, pos))
    selected.sort(key=lambda item: item[0])

    # If pure decades alone exceed max_ticks, thin uniformly with integer stride keeping upper range.
    if len(selected) > max_ticks:
        if max_ticks == 1:
            selected = [min(selected, key=lambda item: abs(item[1] - 0.5))]
        else:
            stride = math.ceil((len(selected) - 1) / float(max_ticks - 1))
            thinned = selected[::-stride][::-1]
            if len(thinned) > max_ticks:
                thinned = thinned[-max_ticks:]
            selected = thinned

    def _insert_widest(pool: list[tuple[float, float, int]]) -> None:
        while len(selected) < max_ticks:
            best: tuple[float, int, float, float] | None = None
            for val, pos, prio in pool:
                if any(math.isclose(val, sv, rel_tol=0.0, abs_tol=1e-12) for sv, _ in selected):
                    continue
                if selected:
                    min_dist = min(abs(pos - sp) for _, sp in selected)
                else:
                    min_dist = 1.0 - abs(pos - 0.5)
                if min_dist < gap - 1e-9:
                    continue
                score = (round(min_dist, 6), prio, -abs(pos - 0.5), pos)
                if best is None or score > best[:4]:
                    best = (*score[:3], pos)
                    best_item = (val, pos)
            if best is None:
                break
            selected.append(best_item)
            selected.sort(key=lambda item: item[0])

    # Step 2: Insert 1-2-5 subticks (2, 5) into widest visual gaps.
    _insert_widest(tier1)

    # Step 3: If wide gaps remain where neither 2 nor 5 fit (e.g. ~2-decade log/asinh spans), allow 3*10^e.
    if len(selected) < min(max_ticks, 4):
        _insert_widest(tier2)

    # Step 4: Narrow-range fallback when fewer than 2 ticks were found from decades/subticks.
    if len(selected) < 2 and max_ticks >= 2:
        lin_pts = _linear_nice_ticks(norm, lo, hi, min_visual_gap=gap, max_ticks=max_ticks)
        if len(lin_pts) >= 2:
            selected = lin_pts

    # Final safety verification of strict visual spacing and sorting.
    selected.sort(key=lambda item: item[0])
    verified: list[tuple[float, float]] = []
    for val, pos in selected:
        if verified and abs(pos - verified[-1][1]) < gap - 1e-9:
            continue
        verified.append((val, pos))
        if len(verified) >= max_ticks:
            break
    return [val for val, _ in verified]


def wrap_butterfly_ray_angles_deg(
    bisector_deg: float,
    half_opening_deg: float,
) -> tuple[float, float, float, float]:
    """Compute exact ``[0, 360)`` wrapped 4-lobe butterfly ray angles across the ``0/360`` seam.

    Returns ``((bisector - alpha) % 360, (bisector + alpha) % 360,
    (bisector + 180 - alpha) % 360, (bisector + 180 + alpha) % 360)``.
    """
    if isinstance(bisector_deg, bool) or isinstance(half_opening_deg, bool):
        raise ValueError("bisector_deg and half_opening_deg must be finite real numbers, not bool")
    b = float(bisector_deg)
    a = float(half_opening_deg)
    if not (math.isfinite(b) and math.isfinite(a)):
        raise ValueError(
            f"bisector_deg and half_opening_deg must be finite, got ({bisector_deg!r}, {half_opening_deg!r})"
        )
    if a < 0.0:
        raise ValueError(f"half_opening_deg must be non-negative, got {half_opening_deg!r}")

    def _wrap360(angle: float) -> float:
        wrapped = float(angle) % 360.0
        if wrapped >= 360.0 or math.isclose(wrapped, 360.0, rel_tol=0.0, abs_tol=1e-12):
            wrapped = 0.0
        return 0.0 if wrapped == 0.0 else wrapped

    return (
        _wrap360(b - a),
        _wrap360(b + a),
        _wrap360(b + 180.0 - a),
        _wrap360(b + 180.0 + a),
    )


__all__ = [
    "JOURNAL_PRESETS",
    "JournalPreset",
    "MM_PER_INCH",
    "OKABE_ITO_NAMED",
    "apply_journal_style",
    "format_axis_label",
    "format_clean_colorbar_ticks",
    "format_clean_tick_label",
    "format_math_unit",
    "get_journal_preset",
    "get_okabe_ito_palette",
    "get_residual_colormap",
    "get_saxs_colormap",
    "get_thermal_palette",
    "inches_to_mm",
    "is_physical_q_unit",
    "journal_rc_context",
    "make_saxs_norm",
    "mm_to_inches",
    "q_to_period_nm",
    "style_1d_axes",
    "style_2d_axes",
    "style_panel_label",
    "wrap_butterfly_ray_angles_deg",
]
