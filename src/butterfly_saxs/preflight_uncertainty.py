"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.preflight.uncertainty`."""

from __future__ import annotations

import sys as _sys

from .preflight import uncertainty as _impl
from .preflight.uncertainty import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
