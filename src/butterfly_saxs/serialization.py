"""Small, dependency-light serializers shared by public result boundaries.

Scientific fitting code is allowed to keep NumPy values and non-finite
diagnostics while it is running.  Values crossing a JSON boundary use
``json_safe`` so that non-finite numbers become JSON ``null`` and NumPy
containers/scalars become ordinary Python values.  The conversion is a
recursive copy; it never mutates the numerical inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a strict-JSON-compatible copy of ``value``.

    ``numpy`` is intentionally not imported here.  Array/scalar duck typing
    keeps this helper safe to import from low-level modules and still handles
    NumPy values without introducing an import cycle.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [json_safe(item) for item in value]
        return sorted(converted, key=lambda item: (type(item).__name__, repr(item)))
    if isinstance(value, type):
        return repr(value)
    if is_dataclass(value):
        return {
            item.name: json_safe(getattr(value, item.name))
            for item in fields(value)
        }

    # NumPy arrays and scalars both expose one of these methods.  Convert
    # through the returned built-in value and recurse so NaN/Inf is normalized
    # in arrays as well as at scalar leaves.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            converted = tolist()
        except Exception:  # pragma: no cover - unusual array-like object
            converted = value
        if converted is not value:
            return json_safe(converted)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except Exception:  # pragma: no cover - unusual scalar-like object
            converted = value
        if converted is not value:
            return json_safe(converted)

    # Parameter/result objects in the public seams commonly expose one of
    # these explicit mapping methods.  Keep this fallback after array/scalar
    # conversion so an ndarray is never stringified.
    for method_name in ("to_dict", "as_dict", "to_mapping"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            try:
                converted = method(include_specs=True)
            except TypeError:
                converted = method()
        except Exception:  # pragma: no cover - defensive compatibility path
            continue
        if converted is not value:
            return json_safe(converted)

    if hasattr(value, "__dict__"):
        try:
            return json_safe(
                {
                    key: item
                    for key, item in vars(value).items()
                    if not str(key).startswith("_")
                }
            )
        except Exception:  # pragma: no cover - defensive compatibility path
            pass
    return repr(value)


def dumps_strict_json(
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    separators: tuple[str, str] | None = None,
    trailing_newline: bool = True,
) -> str:
    """Serialize ``value`` as strict JSON (``allow_nan=False``) after ``json_safe``."""

    text = json.dumps(
        json_safe(value),
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
        separators=separators,
        allow_nan=False,
    )
    return text + "\n" if trailing_newline else text


def write_text_atomic(
    path: str | os.PathLike[str] | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write ``text`` through a sibling temporary file and atomic ``os.replace``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def write_json_atomic(
    path: str | os.PathLike[str] | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> Path:
    """Serialize ``payload`` strictly (``allow_nan=False``) and replace ``path`` atomically."""

    text = dumps_strict_json(
        payload,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        trailing_newline=True,
    )
    return write_text_atomic(path, text)


# Descriptive alias for callers that want to make the strict boundary
# explicit in their code without copying the implementation.
strict_jsonable = json_safe


__all__ = [
    "dumps_strict_json",
    "json_safe",
    "strict_jsonable",
    "write_json_atomic",
    "write_text_atomic",
]
