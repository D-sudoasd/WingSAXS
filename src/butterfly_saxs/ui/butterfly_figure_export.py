"""Pure task helpers for the butterfly measurement-figure worker."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._common import unique_export_dir


def new_figure_export_target(parent: str | Path) -> Path:
    """Return a new child directory without creating or overwriting anything."""

    return unique_export_dir(parent, "butterfly-figure")


def run_butterfly_figure_export(*, payload: Mapping[str, Any], **_: Any) -> dict[str, Path]:
    """Execute a detached figure snapshot without touching Qt widgets."""

    from ..butterfly_figure import export_butterfly_figure

    progress_callback = payload.get("progress")

    def report_progress(percent: int, phase: str) -> None:
        if callable(progress_callback):
            progress_callback({"percent": int(percent), "phase": str(phase)})

    kwargs: dict[str, Any] = {
        "observed": payload["observed"],
        "qx": payload["qx"],
        "qy": payload["qy"],
        "valid_mask": payload.get("valid_mask"),
        "result": payload["result"],
        "q_unit": str(payload.get("q_unit", "unknown")),
        "context": payload.get("context"),
        "display_scale": str(payload.get("display_scale", "log1p")),
        "width_mm": float(payload["width_mm"]),
        "dpi": int(payload.get("dpi", 600)),
        "cancel_event": payload["cancel_event"],
        "progress": report_progress,
    }
    if payload.get("model") is not None:
        kwargs["model"] = payload["model"]
    journal_style = payload.get("journal_style")
    if journal_style is not None:
        kwargs["journal_style"] = str(journal_style)
        try:
            return export_butterfly_figure(payload["target"], **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument 'journal_style'" not in str(exc):
                raise
            kwargs.pop("journal_style", None)
    return export_butterfly_figure(payload["target"], **kwargs)


__all__ = ["new_figure_export_target", "run_butterfly_figure_export"]
