"""Content identities for calculation and presentation reuse, without a cache."""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform
from collections.abc import Mapping


# Only the reviewed pretty-JSON -> compact-JSON storage update may restore
# these older calculation identities. All other source bytes, dependencies
# and analyzer evidence must still match. A later batch edit fails closed
# until its compatibility has been established separately.
_COMPACT_CHECKPOINT_BATCH_SHA256 = frozenset({
    "ef6238070ec330b5e585270692c60a74329e3752cc13582f31897be525237e0d",
    "4e424ac36a0006acdb1883aa11b3d4b571ff4284d26cf0fcaf05b52426d9b572",
})
_PRETTY_CHECKPOINT_SOURCES = frozenset({
    ("9d2451115cdd219bf26118b62f930a9ba350dc6237ee408d6fd81fa0c7a3c0fa", "e87ce353aaf1b07ef85c13243a9e1ffc7fd60f0034d03582a160c787fa891544"),
    ("bb12be937d5b105fd81714923ae9413f3388ab45d97c1b7578461b489cb7c9e3", "ff5cc51d19ca4438b0073b83e0741cc43791ded7656556b70259e515fc713049"),
})
_CHECKPOINT_COMPATIBILITY_SOURCE_SHA256 = "d7be191e8545fc04510b1a768dec86a55eb01a4e7a7ab0aab7131b6bafb81fbb"


def _reviewed_checkpoint_builder_hash(source: bytes) -> str | None:
    """Pin the compatibility builder without a self-referential source hash.

    Only the digest declaration is omitted. Any change to identity collection,
    compatibility rules, or their allowed source hashes disables the legacy
    upgrade until the new builder has been reviewed and pinned explicitly.
    """
    lines = source.replace(b"\r\n", b"\n").splitlines(keepends=True)
    prefix = b"_CHECKPOINT_COMPATIBILITY_SOURCE_SHA256 = "
    if sum(line.startswith(prefix) for line in lines) != 1:
        return None
    reviewed = b"".join(line for line in lines if not line.startswith(prefix))
    if hashlib.sha256(reviewed).hexdigest() != _CHECKPOINT_COMPATIBILITY_SOURCE_SHA256:
        return None
    return hashlib.sha256(source).hexdigest()


_REVIEWED_CHECKPOINT_BUILDER_SHA256 = _reviewed_checkpoint_builder_hash(Path(__file__).read_bytes())


def calculation_producers_compatible(previous: object, current: object) -> bool:
    """Accept an exact producer or the explicitly verified storage-only upgrade.

    Actual source hashes remain in new receipts; this does not rewrite or
    pretend that new code has the old identity. ``producer.py`` itself only
    constructs/checks these identities and contains no scientific computation.
    Exporter/presentation identities keep their independent strict checks.
    """
    if previous == current:
        return True
    if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
        return False
    old_sources, new_sources = previous.get("sources"), current.get("sources")
    if not isinstance(old_sources, Mapping) or not isinstance(new_sources, Mapping):
        return False
    if (_REVIEWED_CHECKPOINT_BUILDER_SHA256 is None
            or new_sources.get("producer.py") != _REVIEWED_CHECKPOINT_BUILDER_SHA256):
        return False
    new_batch = new_sources.get("batch.py")
    old_pair = (old_sources.get("batch.py"), old_sources.get("producer.py"))
    if not isinstance(new_batch, str) or new_batch not in _COMPACT_CHECKPOINT_BATCH_SHA256:
        return False
    if not all(isinstance(value, str) for value in old_pair) or old_pair not in _PRETTY_CHECKPOINT_SOURCES:
        return False
    sources = dict(new_sources)
    for name in ("batch.py", "producer.py"):
        sources[name] = old_sources[name]
    return previous == {**current, "sources": sources}


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
