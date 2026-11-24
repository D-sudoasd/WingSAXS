"""Delivery packaging, link dependency resolution, and binding verification."""

from __future__ import annotations

import os

from .bindings import verify_delivery_bindings
from .dependencies import collect_html_dependencies, local_path_problem
from .package import (
    SCHEMA,
    _atomic_text,
    _json,
    _package_batch_locked,
    _render,
    _write_source,
    package_batch,
)
from .pairs import (
    check_archive_pair,
    check_planned_pair,
    plan_image_members,
    reuse_image_archive,
)

__all__ = [
    "SCHEMA",
    "_atomic_text",
    "_json",
    "_package_batch_locked",
    "_render",
    "_write_source",
    "check_archive_pair",
    "check_planned_pair",
    "collect_html_dependencies",
    "local_path_problem",
    "os",
    "package_batch",
    "plan_image_members",
    "reuse_image_archive",
    "verify_delivery_bindings",
]
