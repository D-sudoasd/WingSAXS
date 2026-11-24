"""Dependency-light scalar, reciprocal-unit, q-window, and gradient primitives.

This leaf module intentionally avoids top-level NumPy, SciPy, Matplotlib, and
Qt imports so configuration and CLI entrypoints can validate scalar settings
without loading scientific or GUI stacks. Array helpers import NumPy lazily on
demand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

_QX_ALIASES = ("qx", "q_x", "x", "qx_map")
_QY_ALIASES = ("qy", "q_y", "y", "qy_map")
_NM_INV_TOKENS = frozenset(
    {
        "1/nm",
        "nm^-1",
        "nm-1",
        "nm**-1",
        "nm_inv",
        "q_nm_inv",
    }
)
_ANGSTROM_INV_TOKENS = frozenset(
    {
        "1/a",
        "a^-1",
        "a-1",
        "a**-1",
        "a_inv",
        "1/angstrom",
        "angstrom^-1",
        "angstrom-1",
        "angstrom**-1",
        "angstrom_inv",
    }
)


def is_bool(value: Any) -> bool:
    """Return True for Python ``bool`` and NumPy ``bool_`` without importing NumPy."""

    if isinstance(value, bool):
        return True
    cls = type(value)
    return cls.__module__ == "numpy" and cls.__name__ in {"bool", "bool_"}


def _unwrap_scalar(value: Any) -> Any:
    """Unwrap zero-dimensional NumPy/scalar wrappers when present."""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return value
    if getattr(value, "ndim", 0) != 0:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, IndexError):
            return value
    return value


def finite_float(value: Any, default: Any = None) -> Any:
    """Convert ``value`` to a finite ``float``, rejecting ``bool`` and ``np.bool_``."""

    if is_bool(value):
        return default
    unwrapped = _unwrap_scalar(value)
    if is_bool(unwrapped):
        return default
    if isinstance(unwrapped, (bytes, bytearray)):
        return default
    try:
        result = float(unwrapped)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def strict_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse an integer without accepting booleans or truncating decimals."""

    if is_bool(value):
        raise ValueError(f"{name} must be an integer")
    unwrapped = _unwrap_scalar(value)
    if is_bool(unwrapped):
        raise ValueError(f"{name} must be an integer")
    if isinstance(unwrapped, float):
        if not math.isfinite(unwrapped) or not unwrapped.is_integer():
            raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(unwrapped)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(unwrapped, str):
        text = unwrapped.strip()
        if not text or not text.lstrip("+-").isdigit():
            raise ValueError(f"{name} must be an integer")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def _decode_unit_text(unit: Any) -> str:
    unwrapped = _unwrap_scalar(unit)
    if unwrapped is None:
        return "unknown"
    if isinstance(unwrapped, (bytes, bytearray)):
        text = unwrapped.decode("utf-8", errors="replace").strip()
    else:
        text = str(unwrapped).strip()
    return text or "unknown"


def normalize_q_unit(unit: Any) -> str:
    """Normalize reciprocal-unit spelling without inventing a missing unit."""

    text = _decode_unit_text(unit).casefold()
    return (
        text.replace(" ", "")
        .replace("⁻¹", "^-1")
        .replace("⁻", "^-")
        .replace("−", "-")
        .replace("å", "a")
        .replace("Å", "a")
    )


def q_scale_to_nm_inv(unit: Any) -> float | None:
    """Return multiplier converting ``unit`` to ``nm^-1``, or ``None`` if uncalibrated."""

    normalized = normalize_q_unit(unit)
    if normalized in _NM_INV_TOKENS:
        return 1.0
    if normalized in _ANGSTROM_INV_TOKENS:
        return 10.0
    return None


def _mapping_field(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def parse_q_window(
    q_window: Any,
    q: Any = None,
    *,
    allow_none: bool = False,
    non_negative: bool = False,
    empty_fallback: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Parse a ``(q_min, q_max)`` pair or mapping, optionally falling back to ``q``."""

    if q_window is None:
        if allow_none and q is None:
            return None
        if q is not None:
            import numpy as np

            arr = np.asarray(q, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                return float(np.min(finite)), float(np.max(finite))
            if empty_fallback is not None:
                return empty_fallback
            raise ValueError("q map contains no finite values")
        raise ValueError("q_window must be a (q_min, q_max) pair")

    if isinstance(q_window, Mapping):
        low_raw = _mapping_field(q_window, ("q_min", "min", "low", "start"))
        high_raw = _mapping_field(q_window, ("q_max", "max", "high", "stop"))
        if low_raw is None or high_raw is None:
            raise ValueError("q_window mapping must provide min/max")
    else:
        if isinstance(q_window, (str, bytes, bytearray)):
            raise ValueError("q_window must be a (q_min, q_max) pair")
        try:
            low_raw, high_raw = q_window
        except Exception as exc:
            raise ValueError("q_window must be a (q_min, q_max) pair") from exc

    low = finite_float(low_raw)
    high = finite_float(high_raw)
    if low is None or high is None:
        raise ValueError("q_window limits must be finite numbers")
    if non_negative and low < 0.0:
        raise ValueError("q_window lower bound must be non-negative")
    if high <= low:
        raise ValueError("q_window must contain finite max > min")
    return low, high


def safe_gradient(array: Any, axis: int) -> np.ndarray:
    """Finite-difference gradient returning zeros along 0- or 1-pixel axes."""

    import numpy as np

    arr = np.asarray(array, dtype=float)
    if arr.shape[axis] < 2:
        return np.zeros_like(arr, dtype=float)
    return np.asarray(np.gradient(arr, axis=axis), dtype=float)


def _lookup_array(obj: Any, names: Sequence[str]) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return None
    for name in names:
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if val is not None:
            return val
    return None


def representative_q_step(qmap: Any, qy: Any = None) -> float | None:
    """Return a robust physical q step using both detector axes, or ``None``."""

    import numpy as np

    if qy is None:
        qx_raw = _lookup_array(qmap, _QX_ALIASES)
        qy_raw = _lookup_array(qmap, _QY_ALIASES)
    else:
        qx_raw, qy_raw = qmap, qy
    if qx_raw is None or qy_raw is None:
        return None
    try:
        qx_arr, qy_arr = np.broadcast_arrays(
            np.asarray(qx_raw, dtype=float),
            np.asarray(qy_raw, dtype=float),
        )
    except (TypeError, ValueError):
        return None
    if qx_arr.ndim != 2:
        return None
    qx_y, qx_x = safe_gradient(qx_arr, 0), safe_gradient(qx_arr, 1)
    qy_y, qy_x = safe_gradient(qy_arr, 0), safe_gradient(qy_arr, 1)
    steps = np.concatenate((np.hypot(qx_x, qy_x).ravel(), np.hypot(qx_y, qy_y).ravel()))
    finite = steps[np.isfinite(steps) & (steps > np.finfo(float).eps)]
    return float(np.median(finite)) if finite.size else None


__all__ = [
    "finite_float",
    "is_bool",
    "normalize_q_unit",
    "parse_q_window",
    "q_scale_to_nm_inv",
    "representative_q_step",
    "safe_gradient",
    "strict_int",
]
