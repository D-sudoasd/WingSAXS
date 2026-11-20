"""Declarative multi-panel publication figure composer and reusable SAXS panels.

Strictly Qt-free and headless ``Agg``-safe: figures are created via Matplotlib's
OO API with ``FigureCanvasAgg`` directly, never touching ``matplotlib.pyplot``
or process-global backend state.

Scientific invariants enforced here:
- Positive 2D SAXS maps use perceptually uniform sequential colormaps with clean
  neutral masked-gap coloring (never diverging maps on positive intensity).
- ``estimate``, ``candidate``, and ``failed``/``missing`` states remain visually
  distinct: estimates use solid lines + filled markers; candidates use dashed
  lines + hollow markers; failed/missing points create explicit NaN breaks and
  are never interpolated across.
- Real-space lengths in nm (``L* = 2*pi/q*``, apparent ``W = 2*pi/dq_perp_0``)
  are only reported when calibrated physical ``q`` units are supplied.
- Publication export writes vector PDF (fonttype 42), editable SVG (fonttype
  ``"none"``), 600 dpi PNG, Nature Source Data CSVs, and a strict-JSON SHA-256
  manifest (``allow_nan=False``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from .csv_utils import safe_csv_cell
from .journal_style import (
    JournalPreset,
    format_axis_label,
    format_math_unit,
    get_journal_preset,
    get_okabe_ito_palette,
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
from .serialization import json_safe


PUBLICATION_FIGURE_SCHEMA = "wingsaxs.publication_figure.v1"

_ESTIMATE_STATUSES: frozenset[str] = frozenset({
    "estimate",
    "estimated",
    "ok",
    "accepted",
    "measured",
    "success",
    "recovered",
    "valid",
    "ready",
    "published",
})

_CANDIDATE_STATUSES: frozenset[str] = frozenset({
    "candidate",
    "candidate_only",
    "warning_candidate",
    "provisional",
    "warn",
    "warning",
})

_FAILED_STATUSES: frozenset[str] = frozenset({
    "failed",
    "failure",
    "error",
    "missing",
    "invalid",
    "unavailable",
    "not_processed",
    "bad_fit",
    "cancelled",
    "canceled",
    "rejected",
    "omitted",
})


def _to_float_array(values: Any) -> np.ndarray:
    """Convert a sequence/array to float64, mapping None/non-numeric entries to NaN."""
    if values is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    flat_in = arr.ravel()
    flat_out = out.ravel()
    for idx, item in enumerate(flat_in):
        if item is None or isinstance(item, bool):
            continue
        try:
            num = float(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(num):
            flat_out[idx] = num
    return out


def _safe_coordinate_mesh(
    qx: np.ndarray, qy: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep curvilinear q coordinates valid around masked/non-finite cells without fabricating data."""
    valid_coords = np.isfinite(qx) & np.isfinite(qy)
    if not np.any(valid_coords):
        raise ValueError("qx and qy contain no finite coordinates")
    if np.all(valid_coords):
        return qx, qy, values
    from scipy.ndimage import distance_transform_edt

    nearest = distance_transform_edt(~valid_coords, return_distances=False, return_indices=True)
    safe_x = qx[tuple(nearest)]
    safe_y = qy[tuple(nearest)]
    safe_values = np.where(valid_coords, values, np.nan)
    return safe_x, safe_y, safe_values


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def _classify_point_status(status: Any, *, default_kind: str = "estimate") -> str:
    if status is None or str(status).strip() == "":
        return default_kind
    key = str(status).strip().lower().replace(" ", "_").replace("-", "_")
    if key in _FAILED_STATUSES:
        return "failed"
    if key in _CANDIDATE_STATUSES or "candidate" in key:
        return "candidate"
    if key in _ESTIMATE_STATUSES:
        return "estimate"
    return default_kind


def _slug_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    return cleaned or "panel"


def _normalize_source_table(table: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """Normalize either row-oriented records or column-oriented arrays into CSV rows."""
    if table is None:
        return [], []
    if isinstance(table, Mapping):
        if not table:
            return [], []
        columns = [str(key) for key in table]
        col_arrays: dict[str, list[Any]] = {}
        max_len = 0
        for key, col in table.items():
            if isinstance(col, (str, bytes)) or np.isscalar(col):
                col_arrays[str(key)] = [col]
                max_len = max(max_len, 1)
            else:
                seq = list(col)
                col_arrays[str(key)] = seq
                max_len = max(max_len, len(seq))
        rows: list[dict[str, Any]] = []
        for idx in range(max_len):
            row: dict[str, Any] = {}
            for col_name in columns:
                seq = col_arrays[col_name]
                if len(seq) == 1 and max_len > 1:
                    row[col_name] = seq[0]
                elif idx < len(seq):
                    row[col_name] = seq[idx]
                else:
                    row[col_name] = None
            rows.append(row)
        return columns, rows

    if isinstance(table, Sequence) and not isinstance(table, (str, bytes)):
        rows_list = [dict(item) for item in table if isinstance(item, Mapping)]
        if not rows_list:
            return [], []
        columns = list(dict.fromkeys(str(k) for row in rows_list for k in row))
        return columns, rows_list

    raise TypeError(f"Unsupported source_data table type: {type(table).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PanelSpec:
    """Declarative specification for a panel inside a ``FigureSpec``."""

    label: str
    row: int | slice
    col: int | slice
    projection: str | None = None
    style_1d: bool = False
    label_inside: bool = False
    label_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FigureSpec:
    """Declarative multi-panel publication figure specification."""

    preset: str | JournalPreset = "nature"
    width: str | float = "double"
    height_mm: float | None = None
    nrows: int = 1
    ncols: int = 1
    width_ratios: Sequence[float] | None = None
    height_ratios: Sequence[float] | None = None
    wspace: float | None = None
    hspace: float | None = None
    dpi: int | None = None
    layout: str | None = "constrained"
    panels: Sequence[PanelSpec] = ()


class CompositeFigureBuilder:
    """Declarative, headless-safe composer for Nature-grade multi-panel SAXS figures."""

    def __init__(
        self,
        *,
        preset: str | JournalPreset = "nature",
        width: str | float = "double",
        height_mm: float | None = None,
        nrows: int = 1,
        ncols: int = 1,
        width_ratios: Sequence[float] | None = None,
        height_ratios: Sequence[float] | None = None,
        dpi: int | None = None,
        layout: str | None = "constrained",
        gridspec_kw: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(nrows, bool) or not isinstance(nrows, int) or nrows < 1:
            raise ValueError(f"nrows must be an integer >= 1, got {nrows!r}")
        if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
            raise ValueError(f"ncols must be an integer >= 1, got {ncols!r}")
        self.preset: JournalPreset = get_journal_preset(preset)
        self.width_mm: float = self.preset.width_mm(width)
        if height_mm is None:
            default_aspect = min(1.35, max(0.45, 0.55 * (nrows / ncols)))
            self.height_mm = min(self.width_mm * default_aspect, self.preset.max_height_mm)
        else:
            h_mm = float(height_mm)
            if not math.isfinite(h_mm) or h_mm <= 0.0:
                raise ValueError(f"height_mm must be positive and finite, got {height_mm!r}")
            self.height_mm = h_mm
        self.dpi: int = int(self.preset.dpi if dpi is None else dpi)
        if self.dpi <= 0:
            raise ValueError(f"dpi must be a positive integer, got {dpi!r}")

        figsize = (mm_to_inches(self.width_mm), mm_to_inches(self.height_mm))
        with journal_rc_context(self.preset):
            self.figure = Figure(
                figsize=figsize,
                dpi=self.dpi,
                facecolor=self.preset.background_color,
                edgecolor=self.preset.background_color,
                layout=layout,
            )
            FigureCanvasAgg(self.figure)
            gs_kwargs: dict[str, Any] = dict(gridspec_kw or {})
            if width_ratios is not None:
                gs_kwargs["width_ratios"] = list(width_ratios)
            if height_ratios is not None:
                gs_kwargs["height_ratios"] = list(height_ratios)
            self.gridspec = self.figure.add_gridspec(nrows, ncols, **gs_kwargs)
        self.panels: dict[str, Any] = {}
        self.source_data: dict[str, Any] = {}

    @classmethod
    def from_spec(cls, spec: FigureSpec) -> CompositeFigureBuilder:
        """Instantiate a builder and pre-populate all declared ``PanelSpec`` axes."""
        gs_kw: dict[str, Any] = {}
        if spec.wspace is not None:
            gs_kw["wspace"] = spec.wspace
        if spec.hspace is not None:
            gs_kw["hspace"] = spec.hspace
        builder = cls(
            preset=spec.preset,
            width=spec.width,
            height_mm=spec.height_mm,
            nrows=spec.nrows,
            ncols=spec.ncols,
            width_ratios=spec.width_ratios,
            height_ratios=spec.height_ratios,
            dpi=spec.dpi,
            layout=spec.layout,
            gridspec_kw=gs_kw or None,
        )
        for panel in spec.panels:
            builder.add_panel(
                panel.label,
                panel.row,
                panel.col,
                projection=panel.projection,
                style_1d=panel.style_1d,
                label_inside=panel.label_inside,
                **dict(panel.label_kwargs),
            )
        return builder

    def add_panel(
        self,
        label: str,
        row: int | slice,
        col: int | slice,
        *,
        projection: str | None = None,
        style_1d: bool = False,
        label_inside: bool = False,
        **label_kwargs: Any,
    ) -> Any:
        """Add a labeled subplot at ``gridspec[row, col]`` and return its ``Axes``."""
        with journal_rc_context(self.preset):
            subplot_kw: dict[str, Any] = {}
            if projection is not None:
                subplot_kw["projection"] = projection
            ax = self.figure.add_subplot(self.gridspec[row, col], **subplot_kw)
            if style_1d and projection is None:
                style_1d_axes(ax, preset=self.preset)
            if label:
                style_panel_label(
                    ax,
                    label,
                    preset=self.preset,
                    inside=label_inside,
                    **label_kwargs,
                )
        key = str(label) if label else f"panel_{len(self.panels)}"
        self.panels[key] = ax
        return ax

    def add_subgrid(
        self,
        row: int | slice,
        col: int | slice,
        nrows: int,
        ncols: int,
        **kwargs: Any,
    ) -> Any:
        """Subdivide a grid cell into a nested GridSpec (e.g., 2D gallery or main+residual)."""
        return self.gridspec[row, col].subgridspec(nrows, ncols, **kwargs)

    def register_source_data(self, panel_name: str, data: Any) -> None:
        """Attach tabular source data for Nature Source Data CSV export."""
        self.source_data[str(panel_name)] = data

    def build(self) -> Figure:
        """Return the composed Matplotlib ``Figure``."""
        return self.figure

    def export(
        self,
        output_stem: str | Path,
        *,
        formats: Sequence[str] = ("pdf", "svg", "png"),
        dpi: int | None = None,
        source_data: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        bbox_inches: str | None = None,
    ) -> dict[str, Any]:
        """Export the composed figure with vector/raster files, Source Data CSVs, and manifest."""
        merged_source = dict(self.source_data)
        if source_data:
            merged_source.update(source_data)
        return export_publication_figure(
            self.figure,
            output_stem,
            formats=formats,
            dpi=self.dpi if dpi is None else int(dpi),
            source_data=merged_source or None,
            metadata=metadata,
            preset=self.preset,
            bbox_inches=bbox_inches,
        )


def plot_saxs_2d_panel(
    ax: Any,
    image: Any,
    qx: Any,
    qy: Any,
    *,
    valid_mask: Any | None = None,
    q_unit: str = "nm^-1",
    norm_mode: str = "asinh",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    mask_color: str | None = None,
    ridge_points: Any | None = None,
    rays_deg: Sequence[float] | None = None,
    bisector_deg: float | None = None,
    q_circles: Sequence[float | Mapping[str, Any]] | None = None,
    q_lim: float | tuple[float, float] | None = None,
    scalebar_q: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    intensity_unit: str | None = None,
    title: str | None = None,
    legend_loc: str = "upper right",
    annotation: str | None = None,
    preset: str | JournalPreset = "nature",
) -> dict[str, Any]:
    """Render a calibrated 2D SAXS intensity map with clean mask gaps and high-contrast overlays."""
    resolved = get_journal_preset(preset)
    obs = np.asarray(image, dtype=np.float64)
    qx_arr = np.asarray(qx, dtype=np.float64)
    qy_arr = np.asarray(qy, dtype=np.float64)
    if obs.ndim != 2 or min(obs.shape, default=0) < 2:
        raise ValueError("image must be a 2D array with shape at least (2, 2)")
    if qx_arr.shape != obs.shape or qy_arr.shape != obs.shape:
        raise ValueError("qx and qy must have the same shape as image")
    if valid_mask is None:
        valid = np.isfinite(obs)
    else:
        mask_arr = np.asarray(valid_mask, dtype=bool)
        if mask_arr.shape != obs.shape:
            raise ValueError("valid_mask must have the same shape as image")
        valid = mask_arr & np.isfinite(obs)

    masked_img = np.where(valid, obs, np.nan)
    safe_qx, safe_qy, safe_img = _safe_coordinate_mesh(qx_arr, qy_arr, masked_img)
    saxs_cmap = get_saxs_colormap(cmap, preset=resolved, mask_color=mask_color)
    norm = make_saxs_norm(safe_img, mode=norm_mode, vmin=vmin, vmax=vmax)

    with journal_rc_context(resolved):
        mesh = ax.pcolormesh(
            safe_qx,
            safe_qy,
            np.ma.masked_invalid(safe_img),
            shading="auto",
            cmap=saxs_cmap,
            norm=norm,
            rasterized=True,
        )
        style_2d_axes(ax, preset=resolved, equal_aspect=True)
        ax.set_facecolor(resolved.mask_color if mask_color is None else mask_color)

        valid_xy = np.isfinite(safe_qx) & np.isfinite(safe_qy) & valid
        if np.any(valid_xy):
            x_min, x_max = float(np.min(safe_qx[valid_xy])), float(np.max(safe_qx[valid_xy]))
            y_min, y_max = float(np.min(safe_qy[valid_xy])), float(np.max(safe_qy[valid_xy]))
        else:
            x_min, x_max = float(np.nanmin(safe_qx)), float(np.nanmax(safe_qx))
            y_min, y_max = float(np.nanmin(safe_qy)), float(np.nanmax(safe_qy))

        if q_lim is not None:
            if isinstance(q_lim, (int, float)) and not isinstance(q_lim, bool):
                lim = abs(float(q_lim))
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                r_max = lim
            else:
                lo_lim, hi_lim = float(q_lim[0]), float(q_lim[1])
                ax.set_xlim(lo_lim, hi_lim)
                ax.set_ylim(lo_lim, hi_lim)
                r_max = max(abs(lo_lim), abs(hi_lim))
        else:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            r_max = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

        legend_handles: list[Any] = []

        # Concentric q-reference circles (e.g., beamstop cutoff, q* ring).
        if q_circles:
            for idx, item in enumerate(q_circles):
                if isinstance(item, Mapping):
                    radius = float(item["q"])
                    c_label = item.get("label")
                    c_color = str(item.get("color", "#FFFFFF"))
                    c_ls = str(item.get("ls", item.get("linestyle", "--")))
                    c_lw = float(item.get("lw", item.get("linewidth", 0.75)))
                else:
                    radius = float(item)
                    c_label = None
                    c_color = "#FFFFFF"
                    c_ls = "--"
                    c_lw = 0.75
                if not math.isfinite(radius) or radius <= 0.0:
                    continue
                circle = Circle(
                    (0.0, 0.0),
                    radius,
                    fill=False,
                    edgecolor=c_color,
                    linestyle=c_ls,
                    linewidth=c_lw,
                    alpha=0.9,
                    zorder=4,
                )
                ax.add_patch(circle)
                if c_label:
                    legend_handles.append(
                        Line2D([0], [0], color=c_color, ls=c_ls, lw=c_lw, label=str(c_label))
                    )

        # Butterfly wing ray overlays.
        if rays_deg:
            ray_added = False
            for angle in rays_deg:
                ang = float(angle)
                if not math.isfinite(ang):
                    continue
                rad = math.radians(ang)
                x0, y0 = 0.12 * r_max * math.cos(rad), 0.12 * r_max * math.sin(rad)
                x1, y1 = 0.92 * r_max * math.cos(rad), 0.92 * r_max * math.sin(rad)
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color="#00E5FF",
                    linestyle="--",
                    linewidth=0.85,
                    alpha=0.95,
                    zorder=5,
                )
                ray_added = True
            if ray_added:
                legend_handles.append(
                    Line2D([0], [0], color="#00E5FF", ls="--", lw=0.85, label="Wing rays")
                )

        # Friedel symmetry bisector overlay.
        if bisector_deg is not None and math.isfinite(float(bisector_deg)):
            rad = math.radians(float(bisector_deg))
            dx, dy = 0.92 * r_max * math.cos(rad), 0.92 * r_max * math.sin(rad)
            ax.plot(
                [-dx, dx],
                [-dy, dy],
                color="#F0E442",
                linestyle="-.",
                linewidth=0.9,
                alpha=0.95,
                zorder=5,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#F0E442",
                    ls="-.",
                    lw=0.9,
                    label=rf"$\chi_{{\mathrm{{sym}}}} = {float(bisector_deg):.1f}^\circ$",
                )
            )

        # Observed ridge points overlay.
        if ridge_points is not None:
            if isinstance(ridge_points, Mapping):
                ridge_groups = list(ridge_points.items())
            else:
                ridge_groups = [("Observed ridge", ridge_points)]
            palette = ["#00E5FF", "#F0E442", "#56B4E9", "#FFFFFF"]
            for g_idx, (g_label, coords) in enumerate(ridge_groups):
                pts = np.asarray(coords, dtype=np.float64)
                if pts.ndim == 2 and pts.shape[1] == 2:
                    finite_pts = pts[np.all(np.isfinite(pts), axis=1)]
                    if finite_pts.size:
                        edge = palette[g_idx % len(palette)]
                        ax.scatter(
                            finite_pts[:, 0],
                            finite_pts[:, 1],
                            s=9.0,
                            facecolors="none",
                            edgecolors=edge,
                            linewidths=0.7,
                            zorder=6,
                        )
                        legend_handles.append(
                            Line2D(
                                [0],
                                [0],
                                linestyle="none",
                                marker="o",
                                markersize=3.5,
                                markerfacecolor="none",
                                markeredgecolor=edge,
                                markeredgewidth=0.8,
                                label=str(g_label),
                            )
                        )

        # Crisp high-contrast scalebar in bottom-left.
        if scalebar_q is not None and math.isfinite(float(scalebar_q)) and float(scalebar_q) > 0.0:
            sb_len = float(scalebar_q)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_span = xlim[1] - xlim[0]
            y_span = ylim[1] - ylim[0]
            bx0 = xlim[0] + 0.06 * x_span
            by0 = ylim[0] + 0.08 * y_span
            # High-contrast backing pill so text & bar are never dark over dark mask gaps.
            pill = Rectangle(
                (bx0 - 0.025 * x_span, by0 - 0.03 * y_span),
                sb_len + 0.05 * x_span,
                0.115 * y_span,
                facecolor="#FFFFFF",
                edgecolor="#CCCCCC",
                linewidth=0.4,
                alpha=0.90,
                zorder=7,
            )
            ax.add_patch(pill)
            ax.plot(
                [bx0, bx0 + sb_len],
                [by0, by0],
                color="#111111",
                linewidth=1.8,
                solid_capstyle="butt",
                zorder=8,
            )
            unit_str = format_math_unit(q_unit)
            sb_text = f"{sb_len:g} {unit_str}".strip()
            ax.text(
                bx0 + 0.5 * sb_len,
                by0 + 0.022 * y_span,
                sb_text,
                ha="center",
                va="bottom",
                fontsize=max(resolved.tick_label_size - 0.3, 5.0),
                color="#111111",
                fontweight="medium",
                zorder=8,
            )

        if annotation:
            ax.text(
                0.04,
                0.96,
                str(annotation),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=resolved.tick_label_size,
                color="#111111",
                bbox={
                    "facecolor": "#FFFFFF",
                    "edgecolor": "#CCCCCC",
                    "linewidth": 0.4,
                    "alpha": 0.90,
                    "pad": 1.8,
                },
                zorder=8,
            )

        # High-contrast legend box (fixed corner to avoid expensive QuadMesh path collision).
        if legend_handles:
            loc = "upper right" if legend_loc == "best" else legend_loc
            leg = ax.legend(
                handles=legend_handles,
                loc=loc,
                frameon=True,
                facecolor="#FFFFFF",
                framealpha=0.92,
                edgecolor="#CCCCCC",
                fontsize=max(resolved.legend_fontsize - 0.3, 5.0),
                borderpad=0.3,
                handlelength=1.2,
                labelspacing=0.2,
            )
            leg.set_zorder(9)
            for txt in leg.get_texts():
                txt.set_color("#111111")

        ax.set_xlabel(format_axis_label(r"$q_x$", q_unit), fontsize=resolved.axis_label_size)
        ax.set_ylabel(format_axis_label(r"$q_y$", q_unit), fontsize=resolved.axis_label_size)
        if title:
            ax.set_title(str(title), fontsize=resolved.axis_title_size, pad=3.0)

        cbar = None
        if colorbar:
            cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, shrink=0.88)
            label_text = colorbar_label
            if label_text is None:
                label_text = format_axis_label(r"$I$", intensity_unit) if intensity_unit else "Intensity (a.u.)"
            cbar.set_label(label_text, fontsize=resolved.tick_label_size, labelpad=2.0)
            cbar.ax.tick_params(labelsize=max(resolved.tick_label_size - 0.5, 5.0), width=resolved.tick_width)
            cbar.outline.set_linewidth(resolved.spine_linewidth)

    return {
        "axes": ax,
        "mesh": mesh,
        "norm": norm,
        "colorbar": cbar,
        "q_unit": q_unit,
    }


def plot_polar_q_chi_panel(
    ax: Any,
    polar_intensity: Any,
    q_coords: Any,
    chi_coords_deg: Any,
    *,
    q_unit: str = "nm^-1",
    norm_mode: str = "asinh",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    mask_color: str | None = None,
    ridge_chi_deg: Any | None = None,
    ridge_q: Any | None = None,
    q_cutoffs: Sequence[float | Mapping[str, Any]] | None = None,
    bisector_deg: float | None = None,
    orientation: str = "chi_x",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    intensity_unit: str | None = None,
    title: str | None = None,
    preset: str | JournalPreset = "nature",
) -> dict[str, Any]:
    """Render an unwrapped polar ``(q, chi)`` map with optional ``q*`` ridge and cutoffs."""
    resolved = get_journal_preset(preset)
    z = np.asarray(polar_intensity, dtype=np.float64)
    if z.ndim != 2 or min(z.shape, default=0) < 1:
        raise ValueError("polar_intensity must be a non-empty 2D array with shape (n_q, n_chi)")
    q_arr = np.asarray(q_coords, dtype=np.float64).reshape(-1)
    chi_arr = np.asarray(chi_coords_deg, dtype=np.float64).reshape(-1)

    nq, nchi = z.shape
    if q_arr.size == nq + 1:
        q_edges = q_arr
    elif q_arr.size == nq:
        q_edges = _edges_from_centers(q_arr)
    else:
        raise ValueError(f"q_coords length ({q_arr.size}) must be n_q ({nq}) or n_q + 1 ({nq + 1})")

    if chi_arr.size == nchi + 1:
        chi_edges = chi_arr
    elif chi_arr.size == nchi:
        chi_edges = _edges_from_centers(chi_arr)
    else:
        raise ValueError(f"chi_coords_deg length ({chi_arr.size}) must be n_chi ({nchi}) or n_chi + 1 ({nchi + 1})")

    if orientation not in {"chi_x", "q_x"}:
        raise ValueError(f"orientation must be 'chi_x' or 'q_x', got {orientation!r}")

    saxs_cmap = get_saxs_colormap(cmap or resolved.polar_cmap, preset=resolved, mask_color=mask_color)
    norm = make_saxs_norm(z, mode=norm_mode, vmin=vmin, vmax=vmax)

    with journal_rc_context(resolved):
        if orientation == "chi_x":
            mesh = ax.pcolormesh(
                chi_edges,
                q_edges,
                np.ma.masked_invalid(z),
                shading="auto",
                cmap=saxs_cmap,
                norm=norm,
                rasterized=True,
            )
            ax.set_xlabel(format_axis_label(r"Azimuth $\chi$", "deg"), fontsize=resolved.axis_label_size)
            ax.set_ylabel(format_axis_label(r"$q$", q_unit), fontsize=resolved.axis_label_size)
            ax.set_xlim(float(chi_edges[0]), float(chi_edges[-1]))
            ax.set_ylim(float(q_edges[0]), float(q_edges[-1]))
        else:
            mesh = ax.pcolormesh(
                q_edges,
                chi_edges,
                np.ma.masked_invalid(z.T),
                shading="auto",
                cmap=saxs_cmap,
                norm=norm,
                rasterized=True,
            )
            ax.set_xlabel(format_axis_label(r"$q$", q_unit), fontsize=resolved.axis_label_size)
            ax.set_ylabel(format_axis_label(r"Azimuth $\chi$", "deg"), fontsize=resolved.axis_label_size)
            ax.set_xlim(float(q_edges[0]), float(q_edges[-1]))
            ax.set_ylim(float(chi_edges[0]), float(chi_edges[-1]))

        style_2d_axes(ax, preset=resolved, equal_aspect=False)
        ax.set_facecolor(resolved.mask_color if mask_color is None else mask_color)

        legend_handles: list[Any] = []

        if ridge_chi_deg is not None and ridge_q is not None:
            r_chi = _to_float_array(ridge_chi_deg)
            r_q = _to_float_array(ridge_q)
            if r_chi.size == r_q.size and r_chi.size > 0:
                if orientation == "chi_x":
                    ax.plot(r_chi, r_q, color="#00E5FF", linewidth=1.0, zorder=5)
                else:
                    ax.plot(r_q, r_chi, color="#00E5FF", linewidth=1.0, zorder=5)
                legend_handles.append(Line2D([0], [0], color="#00E5FF", lw=1.0, label=r"Observed $q^*(\chi)$"))

        if q_cutoffs:
            for idx, item in enumerate(q_cutoffs):
                if isinstance(item, Mapping):
                    q_val = float(item["q"])
                    c_label = item.get("label")
                    c_color = str(item.get("color", "#FFFFFF"))
                    c_ls = str(item.get("ls", item.get("linestyle", "--")))
                else:
                    q_val = float(item)
                    c_label = rf"$q = {q_val:g}$" if idx == 0 else None
                    c_color = "#FFFFFF"
                    c_ls = "--"
                if not math.isfinite(q_val):
                    continue
                if orientation == "chi_x":
                    ax.axhline(q_val, color=c_color, linestyle=c_ls, linewidth=0.85, alpha=0.95, zorder=5)
                else:
                    ax.axvline(q_val, color=c_color, linestyle=c_ls, linewidth=0.85, alpha=0.95, zorder=5)
                if c_label:
                    legend_handles.append(Line2D([0], [0], color=c_color, ls=c_ls, lw=0.85, label=str(c_label)))

        if bisector_deg is not None and math.isfinite(float(bisector_deg)):
            b_deg = float(bisector_deg)
            if orientation == "chi_x":
                ax.axvline(b_deg, color="#F0E442", linestyle=":", linewidth=0.9, alpha=0.95, zorder=5)
            else:
                ax.axhline(b_deg, color="#F0E442", linestyle=":", linewidth=0.9, alpha=0.95, zorder=5)
            legend_handles.append(
                Line2D([0], [0], color="#F0E442", ls=":", lw=0.9, label=rf"$\chi_{{\mathrm{{sym}}}} = {b_deg:.1f}^\circ$")
            )

        if legend_handles:
            leg = ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                facecolor="#FFFFFF",
                framealpha=0.92,
                edgecolor="#CCCCCC",
                fontsize=max(resolved.legend_fontsize - 0.3, 5.0),
            )
            leg.set_zorder(9)
            for txt in leg.get_texts():
                txt.set_color("#111111")

        if title:
            ax.set_title(str(title), fontsize=resolved.axis_title_size, pad=3.0)

        cbar = None
        if colorbar:
            cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, shrink=0.88)
            label_text = colorbar_label
            if label_text is None:
                label_text = format_axis_label(r"$I$", intensity_unit) if intensity_unit else "Intensity (a.u.)"
            cbar.set_label(label_text, fontsize=resolved.tick_label_size, labelpad=2.0)
            cbar.ax.tick_params(labelsize=max(resolved.tick_label_size - 0.5, 5.0), width=resolved.tick_width)
            cbar.outline.set_linewidth(resolved.spine_linewidth)

    return {
        "axes": ax,
        "mesh": mesh,
        "norm": norm,
        "colorbar": cbar,
    }


def plot_evolution_panel(
    ax: Any,
    series: Sequence[Mapping[str, Any]],
    *,
    xlabel: str = "Time (h)",
    ylabel: str = "",
    x_unit: str | None = None,
    y_unit: str | None = None,
    title: str | None = None,
    palette: Sequence[str] | None = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    show_missing_ticks: bool = True,
    log_log_inset: bool = False,
    inset_bounds: tuple[float, float, float, float] = (0.55, 0.14, 0.40, 0.38),
    inset_ref_slopes: Sequence[float] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    preset: str | JournalPreset = "nature",
) -> dict[str, Any]:
    """Plot multi-series trajectories with strict estimate/candidate/missing visual encoding.

    Visual contract:
    - ``estimate`` points: solid line (``"-"``) and filled markers.
    - ``candidate`` points: dashed line (``"--"``) and hollow markers (``facecolor="white"``).
    - ``failed`` / ``missing`` / non-finite points: forced to ``NaN`` so Matplotlib
      leaves an explicit break and never interpolates across failed frames.
    """
    resolved = get_journal_preset(preset)
    series_list = list(series)
    if palette is None:
        colors = get_thermal_palette(max(1, len(series_list))) if len(series_list) > 1 else get_okabe_ito_palette()
    else:
        colors = list(palette)
    markers = ("o", "s", "^", "D", "v", "P", "X", "h")

    summaries: list[dict[str, Any]] = []
    any_candidate = False
    any_missing = False
    inset_payload: list[dict[str, Any]] = []

    with journal_rc_context(resolved):
        for idx, entry in enumerate(series_list):
            label = str(entry.get("label", f"Series {idx + 1}"))
            x = _to_float_array(entry.get("x"))
            y_raw = _to_float_array(entry.get("y"))
            if x.size != y_raw.size:
                raise ValueError(f"Series {label!r}: x length ({x.size}) != y length ({y_raw.size})")

            default_kind = _classify_point_status(entry.get("kind", "estimate"), default_kind="estimate")
            raw_statuses = entry.get("status")
            if raw_statuses is None:
                statuses = [default_kind if (np.isfinite(xv) and np.isfinite(yv)) else "failed" for xv, yv in zip(x, y_raw)]
            else:
                if len(raw_statuses) != x.size:
                    raise ValueError(f"Series {label!r}: status length ({len(raw_statuses)}) != x length ({x.size})")
                statuses = [_classify_point_status(s, default_kind=default_kind) for s in raw_statuses]

            finite_xy = np.isfinite(x) & np.isfinite(y_raw)
            is_est = np.asarray([s == "estimate" for s in statuses], dtype=bool) & finite_xy
            is_cand = np.asarray([s == "candidate" for s in statuses], dtype=bool) & finite_xy
            is_fail = (~is_est) & (~is_cand)

            # Explicit NaN arrays guarantee breaks across failed/missing/opposite-status frames.
            y_est = np.where(is_est, y_raw, np.nan)
            y_cand = np.where(is_cand, y_raw, np.nan)

            color = str(entry.get("color") or colors[idx % len(colors)])
            marker = str(entry.get("marker") or markers[idx % len(markers)])
            lw = float(entry.get("linewidth", resolved.line_linewidth))
            ms = float(entry.get("markersize", 3.4))
            open_symbol = bool(entry.get("open_symbol", False))

            # Uncertainty shading (only where finite on valid points).
            y_low_arr = _to_float_array(entry.get("y_low")) if entry.get("y_low") is not None else None
            y_high_arr = _to_float_array(entry.get("y_high")) if entry.get("y_high") is not None else None
            yerr_arr = _to_float_array(entry.get("yerr")) if entry.get("yerr") is not None else None
            if y_low_arr is not None and y_high_arr is not None and y_low_arr.size == x.size and y_high_arr.size == x.size:
                band_valid = (is_est | is_cand) & np.isfinite(y_low_arr) & np.isfinite(y_high_arr)
                if np.any(band_valid):
                    ax.fill_between(
                        x,
                        np.where(band_valid, y_low_arr, np.nan),
                        np.where(band_valid, y_high_arr, np.nan),
                        where=band_valid,
                        color=color,
                        alpha=0.18,
                        linewidth=0.0,
                        zorder=1,
                    )
            elif yerr_arr is not None and yerr_arr.size == x.size:
                band_valid = (is_est | is_cand) & np.isfinite(yerr_arr) & (yerr_arr >= 0.0)
                if np.any(band_valid):
                    ax.fill_between(
                        x,
                        np.where(band_valid, y_raw - yerr_arr, np.nan),
                        np.where(band_valid, y_raw + yerr_arr, np.nan),
                        where=band_valid,
                        color=color,
                        alpha=0.18,
                        linewidth=0.0,
                        zorder=1,
                    )

            series_labeled = False
            if np.any(is_est):
                ax.plot(
                    x,
                    y_est,
                    color=color,
                    linestyle=str(entry.get("linestyle", "-")),
                    linewidth=lw,
                    marker=marker,
                    markersize=ms,
                    markerfacecolor="#FFFFFF" if open_symbol else color,
                    markeredgecolor=color,
                    markeredgewidth=0.8 if open_symbol else 0.5,
                    label=label,
                    zorder=3,
                )
                series_labeled = True

            if np.any(is_cand):
                any_candidate = True
                ax.plot(
                    x,
                    y_cand,
                    color=color,
                    linestyle="--",
                    linewidth=max(lw * 0.85, 0.7),
                    marker=marker,
                    markersize=ms,
                    markerfacecolor="#FFFFFF",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    label=(f"{label} (candidate)" if not series_labeled else None),
                    zorder=4,
                )

            if np.any(is_fail & np.isfinite(x)):
                any_missing = True
                if show_missing_ticks:
                    fail_x = x[is_fail & np.isfinite(x)]
                    ax.scatter(
                        fail_x,
                        np.full(fail_x.size, 0.03),
                        transform=ax.get_xaxis_transform(),
                        marker="|",
                        color="#888888",
                        s=24,
                        linewidths=0.7,
                        zorder=2,
                    )

            inset_payload.append({
                "x": x,
                "y_est": y_est,
                "y_cand": y_cand,
                "color": color,
                "marker": marker,
                "open_symbol": open_symbol,
            })
            summaries.append({
                "label": label,
                "n_total": int(x.size),
                "n_estimate": int(np.count_nonzero(is_est)),
                "n_candidate": int(np.count_nonzero(is_cand)),
                "n_failed_or_missing": int(np.count_nonzero(is_fail)),
            })

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(format_axis_label(xlabel, x_unit) if x_unit else xlabel, fontsize=resolved.axis_label_size)
        ax.set_ylabel(format_axis_label(ylabel, y_unit) if y_unit else ylabel, fontsize=resolved.axis_label_size)
        if title:
            ax.set_title(str(title), fontsize=resolved.axis_title_size, pad=3.0)
        style_1d_axes(ax, preset=resolved, grid=True)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if any_candidate:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="#444444",
                        linestyle="--",
                        linewidth=0.9,
                        marker="o",
                        markersize=3.2,
                        markerfacecolor="#FFFFFF",
                        markeredgecolor="#444444",
                        label="Candidate only",
                    )
                )
                labels.append("Candidate only")
            if any_missing and show_missing_ticks:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="#888888",
                        linestyle="none",
                        marker="|",
                        markersize=5.0,
                        label="Failed / missing",
                    )
                )
                labels.append("Failed / missing")
            if handles:
                ax.legend(
                    handles=handles,
                    labels=labels,
                    loc=legend_loc,
                    frameon=True,
                    facecolor="#FFFFFF",
                    framealpha=0.90,
                    edgecolor="#CCCCCC",
                    fontsize=resolved.legend_fontsize,
                )

        inset_ax = None
        if log_log_inset:
            inset_ax = ax.inset_axes(list(inset_bounds))
            pos_xs: list[float] = []
            pos_ys: list[float] = []
            for item in inset_payload:
                xv = item["x"]
                ye = item["y_est"]
                yc = item["y_cand"]
                m_est = np.isfinite(xv) & np.isfinite(ye) & (xv > 0.0) & (ye > 0.0)
                if np.any(m_est):
                    inset_ax.plot(
                        np.where(m_est, xv, np.nan),
                        np.where(m_est, ye, np.nan),
                        color=item["color"],
                        linestyle="-",
                        linewidth=0.8,
                        marker=item["marker"],
                        markersize=2.2,
                        markerfacecolor="#FFFFFF" if item["open_symbol"] else item["color"],
                        markeredgecolor=item["color"],
                    )
                    pos_xs.extend(xv[m_est].tolist())
                    pos_ys.extend(ye[m_est].tolist())
                m_cand = np.isfinite(xv) & np.isfinite(yc) & (xv > 0.0) & (yc > 0.0)
                if np.any(m_cand):
                    inset_ax.plot(
                        np.where(m_cand, xv, np.nan),
                        np.where(m_cand, yc, np.nan),
                        color=item["color"],
                        linestyle="--",
                        linewidth=0.7,
                        marker=item["marker"],
                        markersize=2.2,
                        markerfacecolor="#FFFFFF",
                        markeredgecolor=item["color"],
                    )
                    pos_xs.extend(xv[m_cand].tolist())
                    pos_ys.extend(yc[m_cand].tolist())
            if pos_xs and pos_ys:
                inset_ax.set_xscale("log")
                inset_ax.set_yscale("log")
                if inset_ref_slopes:
                    x_ref = np.geomspace(min(pos_xs), max(pos_xs), 20)
                    y_anchor = float(np.median(pos_ys))
                    x_anchor = float(np.median(pos_xs))
                    for slope in inset_ref_slopes:
                        s_val = float(slope)
                        y_ref = y_anchor * (x_ref / x_anchor) ** s_val
                        inset_ax.plot(x_ref, y_ref, color="#666666", linestyle=":", linewidth=0.7)
                        inset_ax.text(
                            x_ref[-1],
                            y_ref[-1],
                            f"$n={s_val:.2g}$",
                            fontsize=max(resolved.tick_label_size - 1.0, 4.5),
                            color="#444444",
                            ha="right",
                            va="bottom",
                        )
            style_1d_axes(inset_ax, preset=resolved, grid=False)
            inset_ax.tick_params(labelsize=max(resolved.tick_label_size - 1.0, 4.5), length=1.8)
            inset_ax.set_facecolor("#FFFFFF")

    return {
        "axes": ax,
        "inset_ax": inset_ax,
        "series_summary": summaries,
    }


def plot_radial_kratky_panel(
    ax: Any,
    curves: Sequence[Mapping[str, Any]],
    *,
    mode: str = "both",
    q_unit: str = "nm^-1",
    intensity_unit: str | None = None,
    q_cutoffs: Sequence[float | Mapping[str, Any]] | None = None,
    show_q_star: bool = True,
    log_y: bool = True,
    palette: Sequence[str] | None = None,
    title: str | None = None,
    legend_loc: str = "best",
    preset: str | JournalPreset = "nature",
) -> dict[str, Any]:
    """Plot overlaid radial ``I(q)`` and/or Kratky ``q^2 I(q)`` curves with ``q*`` markers & cutoffs.

    ``ax`` may be either a single Matplotlib ``Axes`` or a 2-tuple ``(ax_iq, ax_kratky)``.
    When ``mode="both"`` with a single ``Axes``, ``I(q)`` is drawn on the primary axis and
    ``q^2 I(q)`` on a clean right-hand twin axis.
    """
    resolved = get_journal_preset(preset)
    plot_mode = str(mode).strip().lower()
    if plot_mode not in {"intensity", "kratky", "both"}:
        raise ValueError(f"mode must be 'intensity', 'kratky', or 'both', got {mode!r}")

    curve_list = list(curves)
    if palette is None:
        colors = get_thermal_palette(max(1, len(curve_list))) if len(curve_list) > 1 else get_okabe_ito_palette()
    else:
        colors = list(palette)

    if isinstance(ax, tuple) and len(ax) == 2:
        ax_iq, ax_k = ax
        split_axes = True
    else:
        split_axes = False
        ax_iq = ax if plot_mode in {"intensity", "both"} else None
        ax_k = ax if plot_mode == "kratky" else None

    with journal_rc_context(resolved):
        if plot_mode == "both" and not split_axes:
            ax_k = ax_iq.twinx()

        iq_positive_any = False
        for idx, curve in enumerate(curve_list):
            label = str(curve.get("label", f"Curve {idx + 1}"))
            q = _to_float_array(curve.get("q"))
            intensity = _to_float_array(curve.get("intensity", curve.get("I")))
            if curve.get("kratky") is not None:
                kratky = _to_float_array(curve.get("kratky"))
            else:
                kratky = np.where(np.isfinite(q) & np.isfinite(intensity), (q ** 2) * intensity, np.nan)
            color = str(curve.get("color") or colors[idx % len(colors)])
            lw = float(curve.get("linewidth", resolved.line_linewidth))
            q_star_status = _classify_point_status(curve.get("q_star_status", "estimate"))
            sem = _to_float_array(curve.get("sem", curve.get("yerr"))) if curve.get("sem", curve.get("yerr")) is not None else None

            if ax_iq is not None and intensity.size == q.size:
                if log_y:
                    y_plot = np.where(np.isfinite(intensity) & (intensity > 0.0), intensity, np.nan)
                else:
                    y_plot = intensity
                if np.any(np.isfinite(y_plot)):
                    iq_positive_any = iq_positive_any or bool(np.any(y_plot[np.isfinite(y_plot)] > 0.0))
                ls_i = str(curve.get("linestyle", "-"))
                ax_iq.plot(q, y_plot, color=color, linestyle=ls_i, linewidth=lw, label=label, zorder=3)
                if sem is not None and sem.size == q.size:
                    band = np.isfinite(q) & np.isfinite(y_plot) & np.isfinite(sem) & (sem >= 0.0)
                    if log_y:
                        band &= (y_plot - sem) > 0.0
                    if np.any(band):
                        ax_iq.fill_between(
                            q,
                            np.where(band, y_plot - sem, np.nan),
                            np.where(band, y_plot + sem, np.nan),
                            where=band,
                            color=color,
                            alpha=0.16,
                            linewidth=0.0,
                            zorder=1,
                        )
                if show_q_star:
                    q_star_i = curve.get("q_star_direct", curve.get("q_star"))
                    if q_star_i is not None and math.isfinite(float(q_star_i)):
                        qs = float(q_star_i)
                        valid_i = np.isfinite(q) & np.isfinite(y_plot)
                        if np.any(valid_i):
                            yv = float(np.interp(qs, q[valid_i], y_plot[valid_i]))
                            ax_iq.scatter(
                                [qs],
                                [yv],
                                s=24,
                                marker="v",
                                facecolors="#FFFFFF" if q_star_status == "candidate" else color,
                                edgecolors=color,
                                linewidths=0.9,
                                zorder=5,
                            )

            if ax_k is not None and kratky.size == q.size:
                ls_k = "--" if (plot_mode == "both" and not split_axes) else str(curve.get("linestyle", "-"))
                k_label = label if (split_axes or plot_mode == "kratky") else f"{label} ($q^2 I$)"
                ax_k.plot(q, kratky, color=color, linestyle=ls_k, linewidth=lw, label=k_label, zorder=3)
                if show_q_star:
                    q_star_k = curve.get("q_star_kratky", curve.get("q_star") if plot_mode == "kratky" else None)
                    if q_star_k is not None and math.isfinite(float(q_star_k)):
                        qsk = float(q_star_k)
                        valid_k = np.isfinite(q) & np.isfinite(kratky)
                        if np.any(valid_k):
                            yk = float(np.interp(qsk, q[valid_k], kratky[valid_k]))
                            ax_k.scatter(
                                [qsk],
                                [yk],
                                s=24,
                                marker="D",
                                facecolors="#FFFFFF" if q_star_status == "candidate" else color,
                                edgecolors=color,
                                linewidths=0.9,
                                zorder=5,
                            )

        # Low-q / window cutoff indicators.
        target_axes = [a for a in (ax_iq, ax_k) if a is not None]
        if plot_mode == "both" and not split_axes:
            target_axes = [ax_iq]
        if q_cutoffs:
            for target_ax in target_axes:
                for item in q_cutoffs:
                    if isinstance(item, Mapping):
                        q_cut = float(item["q"])
                        c_label = item.get("label")
                        c_color = str(item.get("color", "#555555"))
                        c_ls = str(item.get("ls", item.get("linestyle", "--")))
                        shade_below = bool(item.get("shade_below", False))
                    else:
                        q_cut = float(item)
                        c_label = None
                        c_color = "#555555"
                        c_ls = "--"
                        shade_below = False
                    if not math.isfinite(q_cut):
                        continue
                    target_ax.axvline(q_cut, color=c_color, linestyle=c_ls, linewidth=0.8, alpha=0.85, zorder=2)
                    if shade_below:
                        x0 = target_ax.get_xlim()[0]
                        if q_cut > x0:
                            target_ax.axvspan(x0, q_cut, color="#E0E4EA", alpha=0.45, zorder=0)
                    if c_label:
                        target_ax.text(
                            q_cut,
                            0.97,
                            f" {c_label}",
                            transform=target_ax.get_xaxis_transform(),
                            ha="left",
                            va="top",
                            fontsize=max(resolved.tick_label_size - 0.5, 5.0),
                            color=c_color,
                        )

        q_axis_label = format_axis_label(r"$q$", q_unit)
        i_unit_fmt = format_math_unit(intensity_unit) if intensity_unit else ""
        if ax_iq is not None:
            if log_y and iq_positive_any:
                ax_iq.set_yscale("log")
            ax_iq.set_xlabel(q_axis_label, fontsize=resolved.axis_label_size)
            ax_iq.set_ylabel(
                rf"$I(q)$ ({i_unit_fmt})" if i_unit_fmt else r"$I(q)$ (a.u.)",
                fontsize=resolved.axis_label_size,
            )
            style_1d_axes(ax_iq, preset=resolved, grid=True)
            if title:
                ax_iq.set_title(str(title), fontsize=resolved.axis_title_size, pad=3.0)
            handles, labels = ax_iq.get_legend_handles_labels()
            if handles:
                ax_iq.legend(handles, labels, loc=legend_loc, fontsize=resolved.legend_fontsize)

        if ax_k is not None:
            if split_axes or plot_mode == "kratky":
                ax_k.set_xlabel(q_axis_label, fontsize=resolved.axis_label_size)
                style_1d_axes(ax_k, preset=resolved, grid=True)
                if title and plot_mode == "kratky":
                    ax_k.set_title(str(title), fontsize=resolved.axis_title_size, pad=3.0)
                handles_k, labels_k = ax_k.get_legend_handles_labels()
                if handles_k:
                    ax_k.legend(handles_k, labels_k, loc=legend_loc, fontsize=resolved.legend_fontsize)
            else:
                ax_k.spines["top"].set_visible(False)
                ax_k.spines["right"].set_visible(True)
                ax_k.spines["right"].set_linewidth(resolved.spine_linewidth)
                ax_k.tick_params(
                    axis="y",
                    direction=resolved.tick_direction,
                    length=resolved.tick_length,
                    width=resolved.tick_width,
                    labelsize=resolved.tick_label_size,
                )
            q_unit_fmt = format_math_unit(q_unit)
            if i_unit_fmt and is_physical_q_unit(q_unit):
                k_ylabel = rf"$q^2 I(q)$ ({i_unit_fmt} $\cdot$ {q_unit_fmt}$^2$)"
            else:
                k_ylabel = r"$q^2 I(q)$ (a.u.)"
            ax_k.set_ylabel(k_ylabel, fontsize=resolved.axis_label_size)

    return {
        "ax_intensity": ax_iq,
        "ax_kratky": ax_k,
    }


def plot_azimuthal_fit_panel(
    ax_main: Any,
    chi_deg: Any,
    observed: Any,
    *,
    fit_total: Any | None = None,
    c2_bridge: Any | None = None,
    lobe_components: Sequence[Any] | None = None,
    residual: Any | None = None,
    ax_resid: Any | None = None,
    bisector_deg: float | None = None,
    half_opening_deg: float | None = None,
    metrics: Mapping[str, Any] | None = None,
    intensity_unit: str | None = None,
    title: str | None = None,
    preset: str | JournalPreset = "nature",
) -> dict[str, Any]:
    """Plot observed azimuthal ``I(chi)``, ``C2`` bridge + 4-lobe fit, and a dedicated residual strip."""
    resolved = get_journal_preset(preset)
    chi = _to_float_array(chi_deg)
    obs = _to_float_array(observed)
    if chi.size != obs.size:
        raise ValueError(f"chi_deg length ({chi.size}) must match observed length ({obs.size})")

    fit_arr = _to_float_array(fit_total) if fit_total is not None else None
    bridge_arr = _to_float_array(c2_bridge) if c2_bridge is not None else None
    if residual is not None:
        resid_arr = _to_float_array(residual)
    elif fit_arr is not None and fit_arr.size == obs.size:
        resid_arr = np.where(np.isfinite(obs) & np.isfinite(fit_arr), obs - fit_arr, np.nan)
    else:
        resid_arr = None

    palette = get_okabe_ito_palette()
    c_obs = palette[0]
    c_fit = palette[1]
    c_bridge = palette[2]
    c_lobe = palette[3]

    with journal_rc_context(resolved):
        valid_obs = np.isfinite(chi) & np.isfinite(obs)
        if np.any(valid_obs):
            ax_main.scatter(
                chi[valid_obs],
                obs[valid_obs],
                s=8.5,
                color=c_obs,
                alpha=0.85,
                linewidths=0.0,
                label="Observed",
                zorder=3,
            )

        if bridge_arr is not None and bridge_arr.size == chi.size:
            ax_main.plot(
                chi,
                bridge_arr,
                color=c_bridge,
                linestyle="--",
                linewidth=0.95,
                label=r"$C_2$ bridge",
                zorder=4,
            )

        if lobe_components:
            for l_idx, comp in enumerate(lobe_components):
                comp_arr = _to_float_array(comp)
                if comp_arr.size != chi.size:
                    continue
                base = bridge_arr if (bridge_arr is not None and bridge_arr.size == chi.size) else np.zeros_like(chi)
                ax_main.plot(
                    chi,
                    base + comp_arr,
                    color=c_lobe,
                    linestyle=":",
                    linewidth=0.75,
                    alpha=0.85,
                    label=("4-lobe components" if l_idx == 0 else None),
                    zorder=4,
                )

        if fit_arr is not None and fit_arr.size == chi.size:
            ax_main.plot(
                chi,
                fit_arr,
                color=c_fit,
                linestyle="-",
                linewidth=1.25,
                label=r"$C_2$ + 4-lobe fit",
                zorder=5,
            )

        if bisector_deg is not None and math.isfinite(float(bisector_deg)):
            b = float(bisector_deg)
            ax_main.axvline(b, color="#444444", linestyle="-.", linewidth=0.75, alpha=0.8, zorder=2)
            if half_opening_deg is not None and math.isfinite(float(half_opening_deg)):
                a = abs(float(half_opening_deg))
                ax_main.axvline(b - a, color="#888888", linestyle=":", linewidth=0.7, alpha=0.75, zorder=2)
                ax_main.axvline(b + a, color="#888888", linestyle=":", linewidth=0.7, alpha=0.75, zorder=2)

        if metrics:
            lines: list[str] = []
            if metrics.get("r2") is not None and math.isfinite(float(metrics["r2"])):
                lines.append(rf"$R^2 = {float(metrics['r2']):.3f}$")
            if metrics.get("half_opening_deg") is not None and math.isfinite(float(metrics["half_opening_deg"])):
                lines.append(rf"$\alpha = {float(metrics['half_opening_deg']):.1f}^\circ$")
            if metrics.get("lobe_fwhm_deg") is not None and math.isfinite(float(metrics["lobe_fwhm_deg"])):
                lines.append(rf"$\Delta\chi = {float(metrics['lobe_fwhm_deg']):.1f}^\circ$")
            if lines:
                ax_main.text(
                    0.03,
                    0.95,
                    "\n".join(lines),
                    transform=ax_main.transAxes,
                    ha="left",
                    va="top",
                    fontsize=max(resolved.tick_label_size - 0.3, 5.0),
                    color="#111111",
                    bbox={
                        "facecolor": "#FFFFFF",
                        "edgecolor": "#CCCCCC",
                        "linewidth": 0.4,
                        "alpha": 0.90,
                        "pad": 1.6,
                    },
                    zorder=8,
                )

        i_label = format_axis_label(r"$I(\chi)$", intensity_unit) if intensity_unit else r"$I(\chi)$ (a.u.)"
        ax_main.set_ylabel(i_label, fontsize=resolved.axis_label_size)
        if title:
            ax_main.set_title(str(title), fontsize=resolved.axis_title_size, pad=3.0)
        style_1d_axes(ax_main, preset=resolved, grid=True)
        handles, labels = ax_main.get_legend_handles_labels()
        if handles:
            ax_main.legend(handles, labels, loc="upper right", fontsize=max(resolved.legend_fontsize - 0.3, 5.0))

        # Dedicated residual strip below main plot.
        if resid_arr is not None:
            if ax_resid is None:
                divider = make_axes_locatable(ax_main)
                ax_resid = divider.append_axes("bottom", size="30%", pad=0.06, sharex=ax_main)
            ax_main.tick_params(labelbottom=False)
            ax_resid.axhline(0.0, color="#666666", linestyle="-", linewidth=0.6, zorder=1)
            valid_r = np.isfinite(chi) & np.isfinite(resid_arr)
            if np.any(valid_r):
                ax_resid.plot(
                    chi,
                    resid_arr,
                    color=palette[3],
                    linewidth=0.85,
                    marker="o",
                    markersize=1.8,
                    zorder=3,
                )
            ax_resid.set_xlabel(format_axis_label(r"Azimuth $\chi$", "deg"), fontsize=resolved.axis_label_size)
            ax_resid.set_ylabel(r"$\Delta I$", fontsize=resolved.tick_label_size)
            style_1d_axes(ax_resid, preset=resolved, grid=True)
            ax_resid.tick_params(labelsize=max(resolved.tick_label_size - 0.4, 5.0))
        else:
            ax_main.set_xlabel(format_axis_label(r"Azimuth $\chi$", "deg"), fontsize=resolved.axis_label_size)

    return {
        "ax_main": ax_main,
        "ax_resid": ax_resid,
    }


def plot_streak_width_panel(
    ax: Any,
    q: Any,
    dq_perp: Any,
    *,
    dq_perp_err: Any | None = None,
    valid_mask: Any | None = None,
    linear_fit: Mapping[str, Any] | None = None,
    quadrature_fit: Mapping[str, Any] | None = None,
    q_unit: str = "nm^-1",
    label: str | None = None,
    color: str | None = None,
    title: str | None = None,
    preset: str | JournalPreset = "nature",
) -> dict[str, Any]:
    """Plot perpendicular butterfly streak width ``dq_perp(q)`` vs ``q`` with linear/quadrature fits.

    Extracts and annotates intrinsic mosaic spread ``Delta alpha`` (deg) and apparent
    lateral coherence ``W = 2*pi/dq_perp(0)`` (nm, strictly gated on calibrated physical
    ``q_unit`` with explicit apparent/model-dependent framing).
    """
    resolved = get_journal_preset(preset)
    q_arr = _to_float_array(q)
    w_arr = _to_float_array(dq_perp)
    if q_arr.size != w_arr.size:
        raise ValueError(f"q length ({q_arr.size}) must match dq_perp length ({w_arr.size})")

    err_arr = _to_float_array(dq_perp_err) if dq_perp_err is not None else None
    if err_arr is not None and err_arr.size != q_arr.size:
        raise ValueError("dq_perp_err must match q length")

    if valid_mask is None:
        valid = np.isfinite(q_arr) & np.isfinite(w_arr)
    else:
        vm = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if vm.size != q_arr.size:
            raise ValueError("valid_mask must match q length")
        valid = vm & np.isfinite(q_arr) & np.isfinite(w_arr)

    palette = get_okabe_ito_palette()
    c_pts = color or palette[0]
    c_lin = palette[1]
    c_quad = palette[2]
    extracted_summary: dict[str, Any] = {"linear": None, "quadrature": None}

    with journal_rc_context(resolved):
        if np.any(valid):
            if err_arr is not None and np.any(valid & np.isfinite(err_arr)):
                ax.errorbar(
                    q_arr[valid],
                    w_arr[valid],
                    yerr=np.where(np.isfinite(err_arr[valid]), err_arr[valid], 0.0),
                    fmt="o",
                    color=c_pts,
                    markersize=3.6,
                    elinewidth=0.75,
                    capsize=1.8,
                    label=label or r"Observed $\Delta q_\perp(q)$",
                    zorder=4,
                )
            else:
                ax.scatter(
                    q_arr[valid],
                    w_arr[valid],
                    s=16,
                    color=c_pts,
                    label=label or r"Observed $\Delta q_\perp(q)$",
                    zorder=4,
                )

        invalid_finite = (~valid) & np.isfinite(q_arr) & np.isfinite(w_arr)
        if np.any(invalid_finite):
            ax.scatter(
                q_arr[invalid_finite],
                w_arr[invalid_finite],
                s=16,
                facecolors="none",
                edgecolors="#888888",
                linewidths=0.75,
                label="Excluded slice",
                zorder=3,
            )

        q_valid = q_arr[valid]
        q_grid_default = (
            np.linspace(0.0, float(np.max(q_valid)) * 1.05, 80)
            if q_valid.size
            else np.linspace(0.0, 1.0, 50)
        )

        annotation_lines: list[str] = []

        def _extract_fit(
            fit_map: Mapping[str, Any], *, is_quad: bool, fit_color: str, fit_ls: str, fit_name: str
        ) -> dict[str, Any]:
            intercept = fit_map.get("dq_perp_0", fit_map.get("intercept"))
            slope_rad = fit_map.get("delta_alpha_rad", fit_map.get("slope"))
            delta_alpha_deg = fit_map.get("delta_alpha_deg")
            if slope_rad is None and delta_alpha_deg is not None and math.isfinite(float(delta_alpha_deg)):
                slope_rad = math.radians(float(delta_alpha_deg))
            if delta_alpha_deg is None and slope_rad is not None and math.isfinite(float(slope_rad)):
                delta_alpha_deg = math.degrees(float(slope_rad))

            qg = _to_float_array(fit_map["q_grid"]) if fit_map.get("q_grid") is not None else q_grid_default
            if fit_map.get("y_fit") is not None:
                yf = _to_float_array(fit_map["y_fit"])
            elif intercept is not None and slope_rad is not None:
                b0 = float(intercept)
                m1 = float(slope_rad)
                if is_quad:
                    yf = np.sqrt(np.maximum(b0 ** 2 + (m1 * qg) ** 2, 0.0))
                else:
                    yf = b0 + m1 * qg
            else:
                yf = np.asarray([], dtype=np.float64)

            if yf.size == qg.size and yf.size > 0:
                ax.plot(qg, yf, color=fit_color, linestyle=fit_ls, linewidth=1.15, label=fit_name, zorder=5)
                ci_low = _to_float_array(fit_map["ci_low"]) if fit_map.get("ci_low") is not None else None
                ci_high = _to_float_array(fit_map["ci_high"]) if fit_map.get("ci_high") is not None else None
                if ci_low is None or ci_high is None:
                    b_err = fit_map.get("intercept_err", fit_map.get("dq_perp_0_err"))
                    m_err = fit_map.get("slope_err", fit_map.get("delta_alpha_rad_err"))
                    if b_err is not None and m_err is not None and intercept is not None and slope_rad is not None:
                        b0, m1 = float(intercept), float(slope_rad)
                        sb, sm = abs(float(b_err)), abs(float(m_err))
                        if is_quad:
                            denom = np.maximum(yf, 1e-12)
                            sigma = np.sqrt(((b0 * sb) ** 2 + ((m1 * (qg ** 2)) * sm) ** 2)) / denom
                        else:
                            sigma = np.sqrt(sb ** 2 + (qg * sm) ** 2)
                        ci_low = np.maximum(yf - 1.96 * sigma, 0.0)
                        ci_high = yf + 1.96 * sigma
                if ci_low is not None and ci_high is not None and ci_low.size == qg.size and ci_high.size == qg.size:
                    ax.fill_between(qg, ci_low, ci_high, color=fit_color, alpha=0.16, linewidth=0.0, zorder=2)

            w_nm: float | None = None
            if is_physical_q_unit(q_unit):
                if fit_map.get("W_nm") is not None and math.isfinite(float(fit_map["W_nm"])):
                    w_nm = float(fit_map["W_nm"])
                elif fit_map.get("coherence_W_nm") is not None and math.isfinite(float(fit_map["coherence_W_nm"])):
                    w_nm = float(fit_map["coherence_W_nm"])
                elif intercept is not None and math.isfinite(float(intercept)) and float(intercept) > 0.0:
                    w_nm = q_to_period_nm(float(intercept), q_unit)

            parts: list[str] = []
            if delta_alpha_deg is not None and math.isfinite(float(delta_alpha_deg)):
                parts.append(rf"$\Delta\alpha = {float(delta_alpha_deg):.1f}^\circ$")
            if w_nm is not None and math.isfinite(w_nm):
                parts.append(rf"$W_{{\mathrm{{app}}}} = {w_nm:.1f}\,\mathrm{{nm}}$")
            if fit_map.get("r2") is not None and math.isfinite(float(fit_map["r2"])):
                parts.append(rf"$R^2 = {float(fit_map['r2']):.3f}$")
            if parts:
                prefix = "Quad" if is_quad else "Lin"
                annotation_lines.append(f"{prefix}: " + ", ".join(parts))

            return {
                "dq_perp_0": None if intercept is None else float(intercept),
                "delta_alpha_deg": None if delta_alpha_deg is None else float(delta_alpha_deg),
                "W_apparent_nm": w_nm,
                "r2": None if fit_map.get("r2") is None else float(fit_map["r2"]),
            }

        if linear_fit is not None:
            extracted_summary["linear"] = _extract_fit(
                linear_fit, is_quad=False, fit_color=c_lin, fit_ls="-", fit_name="Linear fit"
            )
        if quadrature_fit is not None:
            extracted_summary["quadrature"] = _extract_fit(
                quadrature_fit, is_quad=True, fit_color=c_quad, fit_ls="--", fit_name="Quadrature fit"
            )

        if annotation_lines:
            ax.text(
                0.04,
                0.96,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=max(resolved.tick_label_size - 0.3, 5.0),
                color="#111111",
                bbox={
                    "facecolor": "#FFFFFF",
                    "edgecolor": "#CCCCCC",
                    "linewidth": 0.4,
                    "alpha": 0.90,
                    "pad": 1.6,
                },
                zorder=8,
            )

        ax.set_xlabel(format_axis_label(r"$q$", q_unit), fontsize=resolved.axis_label_size)
        ax.set_ylabel(format_axis_label(r"$\Delta q_\perp$", q_unit), fontsize=resolved.axis_label_size)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        if title:
            ax.set_title(str(title), fontsize=resolved.axis_title_size, pad=3.0)
        style_1d_axes(ax, preset=resolved, grid=True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", fontsize=max(resolved.legend_fontsize - 0.3, 5.0))

    return {
        "axes": ax,
        **extracted_summary,
    }


def export_publication_figure(
    fig: Figure,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
    source_data: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    preset: str | JournalPreset = "nature",
    bbox_inches: str | None = None,
) -> dict[str, Any]:
    """Export a publication figure to PDF/SVG/PNG + Nature Source Data CSVs + SHA-256 manifest.

    Every exported artifact is hashed with SHA-256 and recorded in ``<stem>_manifest.json``
    using strict JSON (``allow_nan=False``).
    """
    resolved = get_journal_preset(preset)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError(f"dpi must be a positive integer, got {dpi!r}")

    if isinstance(formats, (str, bytes)):
        formats = (str(formats),)
    clean_formats: list[str] = []
    for item in formats:
        fmt = str(item).strip().lower().lstrip(".")
        if fmt not in {"pdf", "svg", "png", "tiff"}:
            raise ValueError(f"Unsupported format {item!r}; allowed: pdf, svg, png, tiff")
        if fmt not in clean_formats:
            clean_formats.append(fmt)
    if not clean_formats:
        raise ValueError("formats must contain at least one output format")

    stem_path = Path(output_stem)
    out_dir = stem_path.parent
    stem_name = stem_path.name
    if not stem_name:
        raise ValueError("output_stem must include a file stem name")
    out_dir.mkdir(parents=True, exist_ok=True)

    if fig.canvas is None:
        FigureCanvasAgg(fig)

    width_in, height_in = (float(v) for v in fig.get_size_inches())
    files_manifest: dict[str, dict[str, Any]] = {}
    sha_map: dict[str, str] = {}
    figure_paths: dict[str, str] = {}

    meta_title = str((metadata or {}).get("title") or stem_name)
    meta_desc = str((metadata or {}).get("description") or "WingSAXS publication figure")

    with journal_rc_context(resolved):
        for fmt in clean_formats:
            dest = out_dir / f"{stem_name}.{fmt}"
            save_kwargs: dict[str, Any] = {
                "format": fmt,
                "facecolor": fig.get_facecolor(),
                "edgecolor": "none",
            }
            if bbox_inches is not None:
                save_kwargs["bbox_inches"] = bbox_inches
            if fmt in {"png", "tiff"}:
                save_kwargs["dpi"] = dpi
            if fmt == "png":
                save_kwargs["metadata"] = {
                    "Title": meta_title,
                    "Description": meta_desc,
                    "Software": "butterfly_saxs",
                }
            elif fmt == "pdf":
                save_kwargs["metadata"] = {
                    "Title": meta_title,
                    "Subject": meta_desc,
                    "Creator": "butterfly_saxs",
                }
            elif fmt == "svg":
                save_kwargs["metadata"] = {
                    "Title": meta_title,
                    "Description": meta_desc,
                    "Creator": "butterfly_saxs",
                }
            fig.savefig(dest, **save_kwargs)
            rel = dest.relative_to(out_dir).as_posix()
            digest = _sha256_file(dest)
            files_manifest[rel] = {
                "path": rel,
                "kind": "figure",
                "format": fmt,
                "bytes": dest.stat().st_size,
                "sha256": digest,
            }
            sha_map[rel] = digest
            figure_paths[fmt] = str(dest)

    source_data_files: list[str] = []
    source_data_dir: Path | None = None
    if source_data:
        source_data_dir = out_dir / f"{stem_name}_source_data"
        source_data_dir.mkdir(parents=True, exist_ok=True)
        for panel_name, table in source_data.items():
            columns, rows = _normalize_source_table(table)
            csv_name = f"{_slug_filename(panel_name)}.csv"
            csv_path = source_data_dir / csv_name
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                if columns:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({col: safe_csv_cell(row.get(col)) for col in columns})
                else:
                    handle.write("")
            rel = csv_path.relative_to(out_dir).as_posix()
            digest = _sha256_file(csv_path)
            files_manifest[rel] = {
                "path": rel,
                "kind": "source_data",
                "panel": str(panel_name),
                "rows": len(rows),
                "columns": columns,
                "bytes": csv_path.stat().st_size,
                "sha256": digest,
            }
            sha_map[rel] = digest
            source_data_files.append(rel)

    manifest_path = out_dir / f"{stem_name}_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": PUBLICATION_FIGURE_SCHEMA,
        "stem": stem_name,
        "preset": resolved.name,
        "dpi": int(dpi),
        "size_inches": [round(width_in, 6), round(height_in, 6)],
        "size_mm": [round(inches_to_mm(width_in), 4), round(inches_to_mm(height_in), 4)],
        "formats": clean_formats,
        "files": files_manifest,
        "sha256": sha_map,
        "source_data_files": source_data_files,
        "metadata": json_safe(dict(metadata or {})),
    }
    manifest_text = json.dumps(json_safe(manifest), ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "figure_paths": figure_paths,
        "source_data_dir": str(source_data_dir) if source_data_dir is not None else None,
    }


__all__ = [
    "CompositeFigureBuilder",
    "FigureSpec",
    "PUBLICATION_FIGURE_SCHEMA",
    "PanelSpec",
    "export_publication_figure",
    "plot_azimuthal_fit_panel",
    "plot_evolution_panel",
    "plot_polar_q_chi_panel",
    "plot_radial_kratky_panel",
    "plot_saxs_2d_panel",
    "plot_streak_width_panel",
]
