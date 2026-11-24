"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.preflight.artifacts`."""

from __future__ import annotations

import sys as _sys

from .preflight import artifacts as _impl
from .preflight.artifacts import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
