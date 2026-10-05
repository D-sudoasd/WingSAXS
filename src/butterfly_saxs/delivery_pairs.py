"""Read-only member planning and links in a DATA-first, IMAGE-overwrite view.

This is a narrow seam for delivery scripts with an already-bound artifact
inventory. No files are rewritten, no inventories are extended, and no archive
is created. Planned SHA-256 values are *expectations*, not fresh verification.
The archive writer must still verify source bytes against them while copying.
Only effective HTML bodies and ZIP metadata are read; non-HTML DATA payloads
are neither decompressed nor rehashed. CSS/JavaScript-generated URLs and remote
resources are outside the explicit HTML-attribute check.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
from typing import Any
from urllib.parse import unquote, urlsplit
import zipfile

from .delivery_dependencies import _HTML_SUFFIXES, _References, local_path_problem
from .file_identity import file_version, stream_digest as _stream_digest


@dataclass(frozen=True)
class _Member:
    read: Callable[[], bytes]


def _issue(report: dict[str, Any], source: str, reference: str, reason: str) -> None:
    item = {"source": source, "reference": reference, "reason": reason}
    if item not in report["issues"]:
        report["issues"].append(item)


def _report() -> dict[str, Any]:
    return {
        "status": "blocked", "issues": [], "external": [], "html_checked": [],
        "historical_html": [], "overridden_html": [], "archive_directories": [],
        "content_hashes_verified": False, "effective_member_count": 0,
    }


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    report["status"] = "blocked" if report["issues"] else "ready"
    report["html_checked"].sort()
    return report


def _relative_name(value: str | Path) -> str:
    if not isinstance(value, (str, Path)):
        raise ValueError("member paths must be strings or Paths")
    name = str(value)
    # ZIP paths are case-sensitive, portable POSIX names. Do not silently
    # normalize aliases, traversal, Windows drive names, or staging files.
    if (not name or "\\" in name or "\x00" in name or ":" in name
            or any(part in {"", ".", ".."} or part.startswith(".")
                   for part in name.split("/"))):
        raise ValueError(f"unsafe relative member path: {name!r}")
    return name


def _sample_name(value: str) -> str:
    name = _relative_name(value)
    if "/" in name:
        raise ValueError("sample_id must be a single directory name")
    return name


def _is_html(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in _HTML_SUFFIXES


def _historical_names(values: Iterable[str], report: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for value in values:
        try:
            name = _relative_name(value)
            if not _is_html(name):
                raise ValueError("historical exclusions must name HTML files")
        except ValueError:
            _issue(report, str(value), str(value), "invalid_historical_html")
            continue
        names.add(name)
    return names


def _zip_members(
    archive: zipfile.ZipFile, sample_id: str, report: dict[str, Any],
) -> dict[str, _Member]:
    members: dict[str, _Member] = {}
    seen: set[str] = set()
    folded: dict[str, str] = {}
    for info in archive.infolist():
        original = info.orig_filename
        name = original[:-1] if original.endswith("/") else original
        try:
            _relative_name(name)
        except ValueError:
            _issue(report, str(archive.filename), original, "unsafe_archive_member")
            continue
        if name in seen:
            _issue(report, str(archive.filename), name, "duplicate_archive_member")
        seen.add(name)
        previous = folded.setdefault(name.casefold(), name)
        if previous != name:
            _issue(report, str(archive.filename), name, "case_colliding_member")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
            _issue(report, str(archive.filename), name, "nonregular_archive_member")
            continue
        prefix, separator, relative = name.partition("/")
        if prefix != sample_id or (not separator and not info.is_dir()):
            _issue(report, str(archive.filename), name, "unexpected_sample_member")
            continue
        if info.is_dir():
            if relative and relative not in report["archive_directories"]:
                report["archive_directories"].append(relative)
            continue
        if mode == stat.S_IFDIR:
            _issue(report, str(archive.filename), name, "nonregular_archive_member")
            continue
        members[relative] = _Member(lambda info=info: archive.read(info))
    return members


def _open_archive(
    stack: ExitStack, path: Path, sample_id: str, report: dict[str, Any],
) -> dict[str, _Member]:
    try:
        archive = stack.enter_context(zipfile.ZipFile(path, "r"))
        return _zip_members(archive, sample_id, report)
    except (OSError, ValueError, zipfile.BadZipFile):
        _issue(report, str(path), str(path), "unreadable_archive")
        return {}


def _local_member(root: Path, name: str, report: dict[str, Any]) -> _Member | None:
    path = root / name
    try:
        problem = local_path_problem(root, path)
        if problem:
            _issue(report, name, name, problem)
            return None
        if not path.exists():
            _issue(report, name, name, "missing_file")
            return None
        if not stat.S_ISREG(path.stat().st_mode):
            _issue(report, name, name, "nonregular_local_member")
            return None
    except (OSError, ValueError):
        _issue(report, name, name, "unreadable_file")
        return None

    def read() -> bytes:
        # Repeat safety checking before reads; a plan is not a filesystem lock.
        if local_path_problem(root, path):
            raise ValueError("local HTML path changed after preflight")
        return path.read_bytes()

    return _Member(read)


def _root(path: Path) -> Path:
    lexical = Path(os.path.abspath(path))
    # Resolving first would conceal a symlink root. Ancestors are also checked.
    if any(part.is_symlink() for part in (lexical, *lexical.parents)):
        raise ValueError("sample_root must not contain symlink components")
    if not lexical.is_dir():
        raise ValueError("sample_root must be an existing directory")
    return lexical


def _overlay(
    data: Mapping[str, _Member], image: Mapping[str, _Member], report: dict[str, Any],
) -> dict[str, _Member]:
    effective = {**data, **image}
    report["effective_member_count"] = len(effective)
    report["overridden_html"] = sorted(name for name in data.keys() & image.keys() if _is_html(name))
    folded: dict[str, str] = {}
    for name in effective:
        previous = folded.setdefault(name.casefold(), name)
        if previous != name:
            _issue(report, name, previous, "case_colliding_member")
    # Extraction must also work on case-insensitive filesystems. A regular
    # file A blocks both an explicit a/ directory and any implicit a/child
    # parent, even when the conflicting names come from different archives.
    explicit_directories = set(report["archive_directories"])
    for name in (*effective, *sorted(explicit_directories)):
        directories = list(PurePosixPath(name).parents)
        if name in explicit_directories:
            directories.append(PurePosixPath(name))
        for directory in directories:
            conflicting_file = folded.get(directory.as_posix().casefold())
            if conflicting_file is not None:
                _issue(report, name, conflicting_file, "file_directory_conflict")
    return effective


def _references(source: str, member: _Member, report: dict[str, Any]) -> list[tuple[str, str]]:
    parser = _References()
    try:
        parser.feed(member.read().decode("utf-8-sig"))
        parser.close()
    except (OSError, UnicodeError, ValueError, RuntimeError, zipfile.BadZipFile, NotImplementedError):
        _issue(report, source, source, "unreadable_html")
        return []
    report["html_checked"].append(source)
    for reference, reason in parser.unsupported:
        _issue(report, source, reference, reason)
    if any(reason == "html_base_url" for _, reason in parser.unsupported):
        return []
    resolved: list[tuple[str, str]] = []
    for reference in dict.fromkeys(parser.references):
        try:
            parts = urlsplit(reference.strip())
            if parts.scheme or parts.netloc:
                if parts.scheme.lower() == "file" or len(parts.scheme) == 1:
                    _issue(report, source, reference, "absolute_path")
                else:
                    report["external"].append({"source": source, "reference": reference})
                continue
            decoded = unquote(parts.path, encoding="utf-8", errors="strict").replace("\\", "/")
        except (ValueError, UnicodeError):
            _issue(report, source, reference, "invalid_url")
            continue
        if "\x00" in decoded:
            _issue(report, source, reference, "invalid_url")
            continue
        if not decoded:
            continue
        if decoded.startswith("/") or ":" in decoded:
            _issue(report, source, reference, "absolute_path")
            continue
        target = posixpath.normpath(posixpath.join(posixpath.dirname(source), decoded))
        if target == ".." or target.startswith("../"):
            _issue(report, source, reference, "outside_sample")
            continue
        try:
            _relative_name(target)
        except ValueError:
            _issue(report, source, reference, "hidden_or_staging_path")
            continue
        resolved.append((reference, target))
    return resolved


def _check_historical(names: set[str], effective: Mapping[str, _Member], report: dict[str, Any]) -> None:
    report["historical_html"] = sorted(names & effective.keys())
    for name in sorted(names - effective.keys()):
        _issue(report, name, name, "historical_html_not_found")


def _check_links(
    effective: Mapping[str, _Member], historical: set[str], report: dict[str, Any],
) -> None:
    _check_historical(historical, effective, report)
    for name in sorted(effective):
        if not _is_html(name) or name in historical:
            continue
        for reference, target in _references(name, effective[name], report):
            if target not in effective:
                _issue(report, name, reference, "missing_file")


def _inventory_records(
    artifact_inventory: Iterable[Mapping[str, Any]], report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    folded: dict[str, str] = {}
    for record in artifact_inventory:
        if not isinstance(record, Mapping):
            _issue(report, "artifact_inventory", "", "invalid_artifact_record")
            continue
        try:
            name = _relative_name(record["path"])
            size, sha = record["size_bytes"], record["sha256"]
            if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                    or not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha) is None):
                raise ValueError("invalid inventory metadata")
        except (KeyError, TypeError, ValueError):
            _issue(report, "artifact_inventory", str(record.get("path", "")), "invalid_artifact_record")
            continue
        normalized = {"path": name, "size_bytes": size, "sha256": sha.lower()}
        previous = folded.setdefault(name.casefold(), name)
        if previous != name:
            _issue(report, "artifact_inventory", name, "case_colliding_member")
        if name in inventory and inventory[name] != normalized:
            _issue(report, "artifact_inventory", name, "conflicting_artifact_record")
        inventory[name] = normalized

    return inventory


def plan_image_members(
    sample_root: Path,
    initial_members: Iterable[str | Path],
    artifact_inventory: Iterable[Mapping[str, Any]],
    *,
    data_zip: Path | None = None,
    historical_html: Iterable[str] = (),
) -> dict[str, Any]:
    """Extend an explicit IMAGE allowlist with registered HTML dependencies.

    All paths are sample-relative, including exact ``historical_html``
    exclusions. Every initial or added member requires a valid inventory record
    (path, non-negative integer size_bytes, 64-hex sha256), a regular local file,
    and matching size. Unknown files are never registered or copied. A generated
    manifest exception, if needed, belongs in the caller with its own explicit
    record. DATA ZIP members satisfy links without being recopied into IMAGE.

    ``members`` contains expected inventory records, ``included`` contains only
    additions, and ``status`` is ready only when issues is empty. Nothing here
    certifies content hashes: retain the writer's streaming SHA verification.
    An invalid root raises ValueError; ordinary preflight failures are reported.
    """
    root = _root(sample_root)
    sample_id = _sample_name(root.name)
    report = _report()
    inventory = _inventory_records(artifact_inventory, report)

    image: dict[str, _Member] = {}

    def add(name: str, source: str, reference: str) -> bool:
        if name in image:
            return True
        expected = inventory.get(name)
        if expected is None:
            _issue(report, source, reference, "unregistered_artifact")
            return False
        member = _local_member(root, name, report)
        if member is None:
            return False
        try:
            if (root / name).stat().st_size != expected["size_bytes"]:
                _issue(report, source, reference, "size_mismatch")
                return False
        except OSError:
            _issue(report, source, reference, "unreadable_file")
            return False
        image[name] = member
        return True

    initial: set[str] = set()
    for value in initial_members:
        try:
            name = _relative_name(value)
        except ValueError:
            _issue(report, str(value), str(value), "unsafe_local_member")
            continue
        initial.add(name)
        add(name, name, name)
    historical = _historical_names(historical_html, report)
    with ExitStack() as stack:
        data = {} if data_zip is None else _open_archive(stack, data_zip, sample_id, report)
        effective = _overlay(data, image, report)
        queue = deque(sorted(name for name in effective if _is_html(name)))
        visited: set[str] = set()
        while queue:
            name = queue.popleft()
            if name in visited or name in historical:
                continue
            visited.add(name)
            for reference, target in _references(name, effective[name], report):
                if target in effective:
                    continue
                if add(target, name, reference):
                    effective[target] = image[target]
                    if _is_html(target):
                        queue.append(target)
        effective = _overlay(data, image, report)
        _check_historical(historical, effective, report)
    report["members"] = [inventory[name] for name in sorted(image)]
    report["included"] = sorted(image.keys() - initial)
    return _finish(report)


def check_planned_pair(
    data_zip: Path,
    sample_root: Path,
    image_members: Iterable[str | Path],
    *,
    historical_html: Iterable[str] = (),
) -> dict[str, Any]:
    """Check live links before compression using DATA + local IMAGE members.

    This checks membership and links, not inventory provenance or hashes. Call
    ``plan_image_members`` for that metadata preflight and stop if it is blocked.
    Member and historical paths are sample-relative; IMAGE wins exact matches.
    """
    root = _root(sample_root)
    sample_id = _sample_name(root.name)
    report = _report()
    image: dict[str, _Member] = {}
    for value in image_members:
        try:
            name = _relative_name(value)
        except ValueError:
            _issue(report, str(value), str(value), "unsafe_local_member")
            continue
        member = _local_member(root, name, report)
        if member is not None:
            image[name] = member
    historical = _historical_names(historical_html, report)
    with ExitStack() as stack:
        data = _open_archive(stack, data_zip, sample_id, report)
        _check_links(_overlay(data, image, report), historical, report)
    return _finish(report)


def check_archive_pair(
    data_zip: Path,
    image_zip: Path,
    *,
    sample_id: str,
    historical_html: Iterable[str] = (),
) -> dict[str, Any]:
    """Check a finished single-sample pair without extracting or hashing DATA.

    Both archives must use the explicit ``sample_id/relative/path`` layout.
    Historical exclusions remain exact *sample-relative* HTML paths. Invalid
    entries, duplicate members and cross-sample links are failures, even if a
    second archive would hide them. Old HTML bodies replaced by IMAGE are never
    read or parsed. This is link validation, not full ZIP CRC/hash verification.
    """
    sample_id = _sample_name(sample_id)
    report = _report()
    historical = _historical_names(historical_html, report)
    with ExitStack() as stack:
        data = _open_archive(stack, data_zip, sample_id, report)
        image = _open_archive(stack, image_zip, sample_id, report)
        _check_links(_overlay(data, image, report), historical, report)
    return _finish(report)


def _file_signature(path: Path) -> tuple[int | None, ...]:
    return file_version(path, path.stat())


def reuse_image_archive(
    sample_root: Path,
    image_zip: Path,
    planned_members: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify and reuse an existing immutable IMAGE ZIP without recompression.

    Unlike link preflight, this reads/hashes every planned local source and every
    archived member in bounded chunks, checks ZIP CRCs while reading, and hashes
    the archive itself. The exact member set, sizes and SHA-256 values must match
    the plan. A disagreement returns ``status='blocked'``; nothing is replaced,
    rewritten or deleted. Success returns the usual archive record with
    ``status='reused'`` and ``content_hashes_verified=True``. DATA is not opened.

    Source bytes are checked after the archived members, so a source changed
    while inspecting the ZIP cannot pass on timestamps alone. Keep sources
    frozen during finalization: version checks are not a filesystem lock or
    proof against hostile concurrent rewrites. Existing writers remain
    responsible for new archives.
    """
    root = _root(sample_root)
    sample_id = _sample_name(root.name)
    path = Path(image_zip)
    report: dict[str, Any] = {
        "status": "blocked", "issues": [], "archive_directories": [],
        "path": str(path), "content_hashes_verified": False,
        "source_hash_checks": 0, "member_hash_checks": 0,
    }
    records = list(planned_members)
    inventory = _inventory_records(records, report)
    if not inventory:
        _issue(report, str(path), str(path), "empty_member_plan")
    if len(records) != len(inventory):
        _issue(report, str(path), str(path), "duplicate_or_invalid_planned_member")
    report["member_count"] = len(inventory)
    if report["issues"]:
        return report

    signatures: dict[str, tuple[int | None, ...]] = {}
    for name, expected in inventory.items():
        if _local_member(root, name, report) is None:
            continue
        source = root / name
        try:
            before = _file_signature(source)
            if before[2] != expected["size_bytes"]:
                _issue(report, name, name, "size_mismatch")
                continue
            signatures[name] = before
        except (OSError, ValueError):
            _issue(report, name, name, "unreadable_file")
    if report["issues"]:
        return report

    try:
        absolute = Path(os.path.abspath(path))
        if any(part.is_symlink() for part in (absolute, *absolute.parents)):
            _issue(report, str(path), str(path), "symlink_path")
            return report
        if not stat.S_ISREG(path.stat().st_mode):
            _issue(report, str(path), str(path), "nonregular_archive_member")
            return report
        before = _file_signature(path)
        with zipfile.ZipFile(path, "r") as archive:
            _zip_members(archive, sample_id, report)
            expected_names = {f"{sample_id}/{name}" for name in inventory}
            if set(archive.namelist()) != expected_names:
                _issue(report, str(path), str(path), "archive_members_mismatch")
            if report["issues"]:
                return report
            for info in archive.infolist():
                name = info.filename.partition("/")[2]
                if info.file_size != inventory[name]["size_bytes"]:
                    _issue(report, str(path), name, "archive_size_mismatch")
            if report["issues"]:
                return report
            for info in archive.infolist():
                name = info.filename.partition("/")[2]
                try:
                    with archive.open(info, "r") as handle:
                        size, digest = _stream_digest(handle)
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, NotImplementedError):
                    _issue(report, str(path), name, "unreadable_archive_member")
                    continue
                if size != inventory[name]["size_bytes"] or digest != inventory[name]["sha256"]:
                    _issue(report, str(path), name, "archive_hash_mismatch")
                else:
                    report["member_hash_checks"] += 1
        if report["issues"]:
            return report
        for name, expected in inventory.items():
            source = root / name
            try:
                signature = _file_signature(source)
                with source.open("rb") as handle:
                    source_size, source_digest = _stream_digest(handle)
                if (
                    signatures[name] != signature or signature != _file_signature(source)
                    or local_path_problem(root, source)
                ):
                    _issue(report, name, name, "source_changed_during_verification")
                elif source_size != expected["size_bytes"] or source_digest != expected["sha256"]:
                    _issue(report, name, name, "source_hash_mismatch")
                else:
                    report["source_hash_checks"] += 1
            except (OSError, ValueError):
                _issue(report, name, name, "unreadable_file")
        if report["issues"]:
            return report
        with path.open("rb") as handle:
            size, digest = _stream_digest(handle)
        if before != _file_signature(path) or path.is_symlink():
            _issue(report, str(path), str(path), "archive_changed_during_verification")
        for name, signature in signatures.items():
            try:
                changed = signature != _file_signature(root / name) or local_path_problem(root, root / name)
            except (OSError, ValueError):
                changed = True
            if changed:
                _issue(report, name, name, "source_changed_during_verification")
        if not report["issues"]:
            report.update(status="reused", size_bytes=size, sha256=digest, content_hashes_verified=True)
    except (OSError, ValueError, zipfile.BadZipFile):
        _issue(report, str(path), str(path), "unreadable_archive")
    return report
