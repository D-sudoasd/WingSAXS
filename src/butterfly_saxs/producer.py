"""Content identities for calculation and presentation reuse, without a cache."""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform
from collections.abc import Mapping


# The reviewed compact writer and scalar-list copy specialization preserve
# checkpoint values and durability. Every accepted predecessor and current
# batch/builder is source-pinned; other sources, dependencies and analyzers
# must match exactly. Unknown later edits continue to fail closed.
_COMPACT_CHECKPOINT_BATCH_SHA256 = frozenset({
    "2c5aab4a065f3e1328eee1beccb8cbe918f13446f79657a7d81a5fecc856acfe",
    "8647f887723d2bc4d9e6e425d95f896fcff9ae34314f2bf85a513823fdd610c4",
})
_PREVIOUS_COMPACT_CHECKPOINT_SOURCES = frozenset({
    ('88db55573794b6f98ea0eedea998cbe12aafba0c4cdc7ab9be90931fe22af9e9', 'fa7bc9250dd8af849a0a45980e3537e651c1f4f68a66c50cd619a688701df145'),
    ('56e387c44681532539585d4051b7f8e61b4819276561dff5fa8009642ab21d77', '0ff804228c2cb7e06f00f759fd2b892aae5e1138b6dd03fd5fb13b8600859d35'),
    ('d7597a2daa437577c3bab935825405cde87d53642f782231031541cfe7341428', '869afa78dfa39e57f9a11b7612a6183a89fd42e154e028654884214890652cf1'),
    ('251a28278620099c46a0f6a28423faafb1ff6c9eb9d55307076622124eee599a', '8a9f78ef594741b8ae99aeefddc271a28cdd1608b2b46bb16176f4fe775434a4'),

    ("ef6238070ec330b5e585270692c60a74329e3752cc13582f31897be525237e0d", "ece30c4a0b985fb0f62b1d7b1bd580fc41c6f09de15af0f276c5dbc9544a655d"),
    ("4e424ac36a0006acdb1883aa11b3d4b571ff4284d26cf0fcaf05b52426d9b572", "e479cd669d4b3eb88c89dd5ff1caf972945f14aab0cb957c22752b6559bfe62a"),
})
_PRETTY_CHECKPOINT_SOURCES = frozenset({
    ("9d2451115cdd219bf26118b62f930a9ba350dc6237ee408d6fd81fa0c7a3c0fa", "e87ce353aaf1b07ef85c13243a9e1ffc7fd60f0034d03582a160c787fa891544"),
    ("bb12be937d5b105fd81714923ae9413f3388ab45d97c1b7578461b489cb7c9e3", "ff5cc51d19ca4438b0073b83e0741cc43791ded7656556b70259e515fc713049"),
})
_CHECKPOINT_COMPATIBILITY_SOURCE_SHA256 = "4cd39da993c9f9ad42df5847fdd261a39ce790c5d88b7d9027916ad357e519ea"


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
    """Accept an exact producer or the explicitly verified checkpoint-only upgrade.

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
    if not all(isinstance(value, str) for value in old_pair) or old_pair not in (_PRETTY_CHECKPOINT_SOURCES | _PREVIOUS_COMPACT_CHECKPOINT_SOURCES):
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
