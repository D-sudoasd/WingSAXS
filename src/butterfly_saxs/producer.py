"""Content identities for calculation and presentation reuse, without a cache."""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform


def producer_identity(*, presentation: bool = False) -> dict:
    """Bind reuse to installed source bytes and relevant dependency versions.

    Presentation-only edits must not invalidate calculated detector results.
    Unknown/new modules are included conservatively in the calculation set.
    No identity is reused between calls or stored outside the existing receipts.
    """
    folder = Path(__file__).parent
    sources = {}
    excluded = {"__main__.py", "export.py", "output_lock.py", "font_support.py",
                "visualization.py", "figure_support.py", "fit_overlays.py", "lamellar.py"}
    prefixes = ("report", "delivery", "benchmark", "publication", "lamellar_", "preflight", "doctor")
    for path in sorted(folder.glob("*.py")):
        if not presentation and (path.name in excluded or path.name.startswith(prefixes)
                                 or "figure" in path.stem):
            continue
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {"python": platform.python_version()}
    names = ("numpy", "scipy", "pyFAI", "fabio", "h5py", "tifffile", "PyYAML", "pillow")
    if presentation:
        names += ("matplotlib",)
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"sources": sources, "versions": versions}


def exporter_identity() -> dict:
    """Bind serialized artifacts and the native evolution plot to their writer.

    Keep this separate from calculation identity so a writer change can
    regenerate tables from verified fits without repeating scientific work.
    """
    identity = producer_identity()
    folder = Path(__file__).parent
    for name in ("export.py", "visualization.py", "font_support.py", "figure_support.py"):
        path = folder / name
        if path.is_file():
            identity["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("matplotlib", "pillow"):
        try:
            identity["versions"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            identity["versions"][name] = None
    return identity
