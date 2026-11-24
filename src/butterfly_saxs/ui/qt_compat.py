"""Optional Qt bindings used by the WingSAXS workbench.

The scientific/core modules deliberately do not depend on Qt.  This module is
the single import boundary for the optional UI stack, so importing
``butterfly_saxs.ui`` (or any of its non-Qt helpers) remains safe on a headless
machine where PySide6 is not installed.
"""

from __future__ import annotations

from typing import Any, Callable

QT_AVAILABLE = False
QT_IMPORT_ERROR: Exception | None = None
_shiboken: Any = None

try:  # pragma: no cover - the fallback is covered on minimal installations
    from PySide6 import QtCore, QtGui, QtWidgets
    try:
        import shiboken6 as _shiboken  # type: ignore[no-redef]
    except Exception:
        _shiboken = None

    QT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the host install
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    QT_IMPORT_ERROR = exc


def is_qt_valid(obj: Any) -> bool:
    """Return True when ``obj`` is non-None and its underlying C++ QObject is alive."""

    if obj is None:
        return False
    if not QT_AVAILABLE or _shiboken is None:
        return True
    if isinstance(obj, QtCore.QObject):
        try:
            return bool(_shiboken.isValid(obj))
        except Exception:
            return False
    return True


def safe_single_shot(delay_ms: int, receiver: Any, callback: Callable[[], None]) -> None:
    """Schedule a single-shot timer that skips invocation if ``receiver`` is deleted."""

    if not QT_AVAILABLE or not is_qt_valid(receiver):
        return

    def _run() -> None:
        if not is_qt_valid(receiver):
            return
        callback()

    if isinstance(receiver, QtCore.QObject):
        QtCore.QTimer.singleShot(int(delay_ms), receiver, _run)
    else:
        QtCore.QTimer.singleShot(int(delay_ms), _run)


def require_qt() -> None:
    """Raise an actionable error when a Qt-only entry point is requested."""

    if not QT_AVAILABLE:
        detail = f": {QT_IMPORT_ERROR}" if QT_IMPORT_ERROR else ""
        raise RuntimeError(
            "The WingSAXS workbench requires the optional UI dependencies "
            "(PySide6 and pyqtgraph)" + detail
        )


def qt_enum(name: str, fallback: Any = None) -> Any:
    """Return a Qt enum by dotted path without importing Qt at module import.

    A couple of small helpers use this for compatibility with PySide6 minor
    releases whose enum aliases moved from ``Qt`` to ``Qt.ItemDataRole``.
    """

    if not QT_AVAILABLE:
        return fallback
    current: Any = QtCore.Qt
    for part in name.split("."):
        current = getattr(current, part)
    return current


__all__ = [
    "QT_AVAILABLE",
    "QT_IMPORT_ERROR",
    "QtCore",
    "QtGui",
    "QtWidgets",
    "is_qt_valid",
    "qt_enum",
    "require_qt",
    "safe_single_shot",
]
