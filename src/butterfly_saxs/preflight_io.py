"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.preflight.io`."""

from __future__ import annotations

import sys as _sys

from .preflight import io as _impl
from .preflight.io import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
