"""Shared Qt-free scalar, reciprocal-unit, geometry, and scene lookup helpers for lamellar."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .._primitives import (
    finite_float,
    normalize_q_unit as _normalise_q_unit,
    q_scale_to_nm_inv as _q_scale_to_nm,
)

_MISSING = object()
_CORNER_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ),
    dtype=float,
)
_BLUE = np.asarray((0.20, 0.45, 0.72, 0.82), dtype=float)
_ORANGE = np.asarray((0.92, 0.47, 0.18, 0.82), dtype=float)


def _read(value: Any, names: Sequence[str], default: Any = None) -> Any:
    """Read the first non-None entry from ``names`` on a mapping, object, or 0-arg callable."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            candidate = value[name]
        elif hasattr(value, name):
            try:
                candidate = getattr(value, name)
            except Exception:  # pragma: no cover - defensive property access
                continue
        else:
            continue
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                pass
        if candidate is not None:
            return candidate
    return default


def _get(value: Any, name: str, default: Any = None) -> Any:
    """Single-key mapping or attribute reader with positional ``default``."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _get_any(value: Any, *names: str, default: Any = None) -> Any:
    """Variadic mapping/attribute reader returning the first non-None match or ``default``."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            candidate = value[name]
        else:
            try:
                candidate = getattr(value, name)
            except (AttributeError, TypeError, ValueError):
                continue
        if candidate is not None:
            return candidate
    return default


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_mapping", None)
    if callable(converter):
        try:
            converted = converter(include_arrays=False)
        except TypeError:
            converted = converter()
        if isinstance(converted, Mapping):
            return converted
    converter = getattr(value, "as_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return converted
    if hasattr(value, "__dict__"):
        try:
            return vars(value)
        except TypeError:
            return None
    return None


def _finite(value: Any, default: float | None = None) -> float | None:
    return finite_float(value, default)


def _finite_value_or_candidate(value: Any) -> float | None:
    if isinstance(value, Mapping):
        mapped = value.get("value")
        value = mapped if mapped is not None else value.get("candidate_value")
    return finite_float(value, None)


def _finite_positive(value: Any) -> float | None:
    result = _finite(value)
    return result if result is not None and result > 0.0 else None


def _is_physical_unit(unit: Any) -> bool:
    return _q_scale_to_nm(unit) is not None


def _copy_metadata(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _copy_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    try:
        return copy.deepcopy(value)
    except Exception:  # pragma: no cover - unusual third-party metadata
        return str(value)


def _axial_wrap_deg(angle_deg: float) -> float:
    """Wrap an unoriented in-plane direction into ``[-90, 90)`` degrees."""

    return float((float(angle_deg) + 90.0) % 180.0 - 90.0)


def _axial_distance_deg(first: float, second: float) -> float:
    difference = (float(second) - float(first) + 90.0) % 180.0 - 90.0
    return abs(difference)


def _origin_centered_xy(
    cx: float | None,
    cy: float | None,
    a: float | None,
    b: float | None,
    *,
    rel_tol: float = 1.0e-8,
) -> bool:
    if None in (cx, cy, a, b):
        return False
    assert cx is not None and cy is not None and a is not None and b is not None
    if a <= 0.0 or b <= 0.0:
        return False
    tolerance = rel_tol * max(1.0, abs(a), abs(b))
    return math.hypot(cx, cy) <= tolerance


def _sequence(value: Any, *, unwrap_0d: bool = False) -> list[Any]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            if unwrap_0d:
                try:
                    return [value.item()]
                except (TypeError, ValueError):
                    return []
            return []
        return list(value)
    if isinstance(value, Sequence):
        return list(value)
    if unwrap_0d and getattr(value, "ndim", None) == 0:
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return [item()]
            except (TypeError, ValueError):
                return []
    return []


def scene_status(scene: Any) -> str:
    value = _get(scene, "status", default="schematic")
    return str(value or "schematic").strip().lower()


def scene_source_identity(scene: Any, *, fallback: str = "Lamellar scene") -> Any:
    value = _get(scene, "source_identity", default=None)
    if value not in (None, ""):
        return value
    metadata = _get(scene, "metadata", default={})
    if isinstance(metadata, Mapping):
        for key in ("frame_id", "source_id", "id", "name", "path", "source"):
            if metadata.get(key) not in (None, ""):
                return metadata[key]
    return fallback


def _convex_hull_2d(points: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Return the counter-clockwise 2D convex hull of ``points``."""

    arr = np.asarray(points, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        arr = arr[:, :2]
    else:
        arr = arr.reshape((-1, 2))
    values = np.unique(arr, axis=0)
    if len(values) < 3:
        if strict:
            raise ValueError("a projected slab polygon must have nonzero area")
        return values
    values = values[np.lexsort((values[:, 1], values[:, 0]))]
    scale = max(float(np.ptp(values[:, 0])), float(np.ptp(values[:, 1])), 1.0)
    tolerance = 0.0 if strict else 1.0e-12 * scale * scale

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        a = first - origin
        b = second - origin
        return float(a[0] * b[1] - a[1] * b[0])

    lower: list[np.ndarray] = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in values[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if strict and len(hull) < 3:
        raise ValueError("a projected slab polygon must have nonzero area")
    return hull


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if is_dataclass(value):
        try:
            return _jsonable(asdict(value))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _jsonable(enum_value)
    return str(value)


__all__ = [
    "_BLUE",
    "_CORNER_SIGNS",
    "_MISSING",
    "_ORANGE",
    "_as_mapping",
    "_axial_distance_deg",
    "_axial_wrap_deg",
    "_convex_hull_2d",
    "_copy_metadata",
    "_finite",
    "_finite_positive",
    "_finite_value_or_candidate",
    "_get",
    "_get_any",
    "_is_physical_unit",
    "_jsonable",
    "_normalise_q_unit",
    "_origin_centered_xy",
    "_q_scale_to_nm",
    "_read",
    "_sequence",
    "scene_source_identity",
    "scene_status",
]
