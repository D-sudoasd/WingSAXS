"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.preflight.hash`."""

from __future__ import annotations

import sys as _sys

from .preflight import hash as _impl
from .preflight.hash import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
