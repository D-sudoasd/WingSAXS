"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.preflight.report`."""

from __future__ import annotations

import sys as _sys

from .preflight import report as _impl
from .preflight.report import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
