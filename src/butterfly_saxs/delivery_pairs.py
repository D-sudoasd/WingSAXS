"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.delivery.pairs`."""

from __future__ import annotations

import sys as _sys

from .delivery import pairs as _impl
from .delivery.pairs import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
