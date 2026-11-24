"""Shared lightweight helpers for the optional WingSAXS Qt workbench.

Importing this module is safe on headless installations without PySide6: Qt
widgets are touched only inside interactive dialog helpers such as
:func:`confirm_save_path`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..serialization import json_safe
from ..settings import canonical_q_unit

if TYPE_CHECKING:
    from .qt_compat import QT_AVAILABLE

_SUPERSCRIPT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("⁻¹", "^-1"),
    ("⁻²", "^-2"),
    ("⁻³", "^-3"),
    ("¹", "^1"),
    ("²", "^2"),
    ("³", "^3"),
)


def _read(source: Any, key: str | Sequence[str], default: Any = None) -> Any:
    """Read one attribute or mapping entry using a single name or fallback names."""

    if isinstance(key, str):
        if isinstance(source, Mapping):
            return source.get(key, default)
        return getattr(source, key, default)
    if isinstance(source, Mapping):
        for name in key:
            if name in source:
                return source[name]
        return default
    for name in key:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def unique_export_dir(parent: str | Path, stem: str, *, resolve: bool = True) -> Path:
    """Return a non-existing child directory without creating or overwriting files."""

    base = Path(parent).expanduser()
    root = base.resolve() if resolve else base
    candidate = root / stem
    index = 2
    while candidate.exists():
        candidate = root / f"{stem}-{index}"
        index += 1
    return candidate


def normalize_save_path(path: str | Path, suffix: str | None = None) -> Path:
    """Normalize a save target so it carries ``suffix`` (case-insensitively)."""

    target = Path(path)
    if suffix:
        normalized = suffix if suffix.startswith(".") else f".{suffix}"
        if target.suffix.lower() != normalized.lower():
            target = target.with_suffix(normalized)
    return target


def confirm_save_path(
    parent: Any,
    chosen: str | Path,
    *,
    suffix: str | None = None,
    extra_targets: Sequence[Path] | Callable[[Path], Sequence[Path]] = (),
    title: str = "",
    prompt: str | Callable[[Path, list[Path]], str] = "",
    confirm_if_unchanged: bool = True,
) -> tuple[Path, bool] | None:
    """Normalize ``chosen`` and ask for explicit overwrite permission when needed.

    Returns ``(target, overwrite)`` when the write may proceed, or ``None`` when
    the user declines overwriting an existing target/companion file.
    """

    raw_path = Path(chosen)
    target = normalize_save_path(raw_path, suffix)
    companions = extra_targets(target) if callable(extra_targets) else extra_targets
    seen: list[Path] = []
    for candidate in (target, *companions):
        candidate_path = Path(candidate)
        if candidate_path not in seen:
            seen.append(candidate_path)
    existing = [candidate for candidate in seen if candidate.exists()]
    needs_prompt = bool(existing) and (confirm_if_unchanged or target != raw_path)
    if needs_prompt:
        from .qt_compat import QtWidgets, require_qt

        require_qt()
        message = (
            prompt(target, existing)
            if callable(prompt)
            else (prompt or f"File already exists. Overwrite?\n{target}")
        )
        answer = QtWidgets.QMessageBox.question(
            parent,
            title,
            message,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return None
    return target, bool(existing)


def is_qt_valid(obj: Any) -> bool:
    """Return True when ``obj`` is non-None and its underlying C++ QObject is alive."""

    if obj is None:
        return False
    from .qt_compat import is_qt_valid as _is_qt_valid

    return _is_qt_valid(obj)


def safe_single_shot(delay_ms: int, receiver: Any, callback: Callable[[], None]) -> None:
    """Schedule a single-shot timer that skips invocation if ``receiver`` is deleted."""

    if receiver is None:
        return
    from .qt_compat import safe_single_shot as _safe_single_shot

    _safe_single_shot(delay_ms, receiver, callback)


def __getattr__(name: str) -> Any:
    if name == "QT_AVAILABLE":
        from .qt_compat import QT_AVAILABLE

        return QT_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def display_q_unit(value: Any) -> str:
    """Format a q-unit string consistently for ASCII-friendly spinbox/plot labels."""

    canonical = canonical_q_unit(value)
    if canonical == "nm⁻¹":
        return "nm^-1"
    if canonical == "Å⁻¹":
        return "A^-1"
    if canonical == "pixel-q":
        return "pixel-q"
    text = str(value or "unknown").replace("−", "-")
    for superscript, replacement in _SUPERSCRIPT_REPLACEMENTS:
        text = text.replace(superscript, replacement)
    return text


def _raw_q_unit_key(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return (
        text.replace(" ", "")
        .replace("−", "-")
        .replace("⁻", "-")
        .replace("¹", "1")
        .replace("^", "")
    )


def same_q_unit(first: Any, second: Any) -> bool:
    """Return True when two q-unit strings refer to the same canonical unit."""

    canon_first = canonical_q_unit(first)
    canon_second = canonical_q_unit(second)
    if canon_first != "unknown" or canon_second != "unknown":
        return canon_first == canon_second
    return _raw_q_unit_key(first) == _raw_q_unit_key(second)


__all__ = [
    "QT_AVAILABLE",
    "_read",
    "confirm_save_path",
    "display_q_unit",
    "is_qt_valid",
    "json_safe",
    "normalize_save_path",
    "same_q_unit",
    "safe_single_shot",
    "unique_export_dir",
]
