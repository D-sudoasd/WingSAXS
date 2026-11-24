"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.delivery.dependencies`."""

from __future__ import annotations

import sys as _sys

from .delivery import dependencies as _impl
from .delivery.dependencies import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
