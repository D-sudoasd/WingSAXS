"""Backward-compatible shim; implementation moved to :mod:`butterfly_saxs.delivery.bindings`."""

from __future__ import annotations

import sys as _sys

from .delivery import bindings as _impl
from .delivery.bindings import *  # noqa: F401,F403

_sys.modules[__name__] = _impl
