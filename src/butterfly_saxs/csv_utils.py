"""Lossless CSV reading and boundary encoding for scientific metadata."""

from __future__ import annotations

import csv
import math
import re
import sys
from collections.abc import Iterable, Iterator
from typing import Any


_NUMERIC_TEXT = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)


def iter_csv_rows(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Iterate dictionaries without the default 128 KiB per-field ceiling.

    Nested scientific metadata can legitimately exceed the standard CSV
    limit. Raise it to a platform-supported value rather than truncating the
    data. Some Python builds use a narrower C integer, so retry smaller values
    on overflow. This process-wide setting is intentionally never restored:
    restoring it could break another CSV reader running in a worker thread.

    Supply a file opened with ``newline=""`` or a newline-preserving StringIO
    so quoted multiline fields retain their original line endings. Parsing
    errors propagate to the caller's existing diagnostic handling.
    """

    limit = sys.maxsize
    current_limit = csv.field_size_limit()
    while current_limit < limit:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:  # pragma: no cover - platform dependent
            limit //= 10
    return iter(csv.DictReader(lines))


def read_csv_rows(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Read all rows losslessly; use iter_csv_rows for bounded-memory readers."""

    return list(iter_csv_rows(lines))


def safe_csv_cell(value: Any) -> Any:
    """Protect formula-like text while preserving numeric values/types.

    Only text values beginning with ``=``/``+``/``-``/``@`` are considered
    dangerous.  A string that is a valid signed number remains numeric text,
    and Python/numpy numeric values remain numeric for downstream readers.
    """

    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return safe_csv_cell(item())
        except Exception:  # pragma: no cover
            pass
    if isinstance(value, str):
        # Spreadsheet engines also evaluate formulas after ignorable leading
        # whitespace, control characters, or a UTF-8 BOM.  Keep the original
        # text for display, but inspect the normalized prefix.
        candidate = value.lstrip(" \t\r\n\v\f").lstrip("\ufeff").lstrip()
        if candidate.startswith(("=", "+", "-", "@")) and not _NUMERIC_TEXT.fullmatch(candidate):
            return "'" + value
        return value
    return value


__all__ = ["iter_csv_rows", "read_csv_rows", "safe_csv_cell"]
