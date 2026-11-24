"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.preflight.checks`."""

from __future__ import annotations

import sys as _sys

from .preflight import checks as _impl
from .preflight.checks import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
