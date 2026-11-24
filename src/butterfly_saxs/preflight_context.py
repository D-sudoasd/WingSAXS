"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.preflight.context`."""

from __future__ import annotations

import sys as _sys

from .preflight import context as _impl
from .preflight.context import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
