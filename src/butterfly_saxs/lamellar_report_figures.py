"""Backward-compatible re-export shim for ``butterfly_saxs.lamellar.report_figures``."""

from __future__ import annotations

from .lamellar.report_figures import (
    _analysis_text as _analysis_text,
    _curve_has_observed_support as _curve_has_observed_support,
    _curve_records as _curve_records,
    _parameter_rows as _parameter_rows,
    _period_figure as _period_figure,
    _sequence_figure as _sequence_figure,
    render_lamellar_report_frame,
    render_lamellar_sequence_report,
)

__all__ = [
    "render_lamellar_report_frame",
    "render_lamellar_sequence_report",
]
