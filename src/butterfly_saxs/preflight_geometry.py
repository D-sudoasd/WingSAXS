"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.preflight.geometry`."""

from __future__ import annotations

import sys as _sys

from .preflight import geometry as _impl
from .preflight.geometry import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
