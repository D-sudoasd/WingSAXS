"""Collect referenced local files without rewriting scientific HTML or data."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

_HTML_SUFFIXES = {".html", ".htm"}
_URL_ATTRIBUTES = {
    "a": {"href"}, "area": {"href"}, "link": {"href"}, "img": {"src"},
    "script": {"src"}, "iframe": {"src"}, "object": {"data"}, "embed": {"src"},
    "audio": {"src"}, "video": {"src", "poster"}, "source": {"src"}, "input": {"src"},
    "image": {"href", "xlink:href"}, "use": {"href", "xlink:href"},
}


def local_path_problem(root: Path, path: Path) -> str | None:
    """Check a lexical path before opening bytes, including symlink ancestors."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return "outside_output_root"
    current = root
    for part in parts:
        if part.startswith("."):
            return "hidden_or_staging_path"
        current /= part
        if current.is_symlink():
            return "symlink_path"
    if not path.resolve().is_relative_to(root):
        return "outside_output_root"
    return None


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []
        self.unsupported: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if not value:
                continue
            if tag == "base" and name == "href":
                # A base URL changes every relative link, even ones preceding
                # it. Do not silently package the wrong files or rewrite it.
                self.unsupported.append((value, "html_base_url"))
            elif name in _URL_ATTRIBUTES.get(tag, set()):
                self.references.append(value)
            elif tag in {"img", "source"} and name == "srcset":
                # Keep a conservative boundary around data URLs (which may
                # themselves contain commas) and escaped/complex candidates.
                if "data:" in value.lower():
                    self.unsupported.append((value, "complex_srcset"))
                else:
                    self.references.extend(candidate.strip().split()[0]
                                           for candidate in value.split(",") if candidate.strip())

    handle_startendtag = handle_starttag


_ACTIONS = {
    "missing_file": "Restore this referenced file, then rerun package --resume.",
    "generated_delivery_reference": "Link to the native manifest/data or delivery_index.html, not to a generated receipt or enclosing ZIP.",
    "unreadable_file": "Make this referenced output readable as a regular file, then rerun package --resume.",
    "outside_output_root": "Export the dependency inside the output root and use a relative link to it.",
    "absolute_path": "Use a portable relative link to a file inside the output root.",
    "symlink_path": "Export a regular copy inside the output root and link to that copy.",
    "hidden_or_staging_path": "Finish the export into a non-hidden output path and update the link.",
    "directory_reference": "Link to a concrete exported file rather than a directory.",
    "html_base_url": "Export this gallery with document-relative links and without a base URL.",
    "complex_srcset": "Use ordinary local src/srcset references for the exported gallery.",
    "unreadable_html": "Save a readable UTF-8 HTML export, then rerun package --resume.",
    "invalid_url": "Correct the invalid URL in the exported HTML, then rerun package --resume.",
}


def collect_html_dependencies(
    root: Path,
    initial_files: Iterable[str],
    record: Callable[[Path], str],
    *,
    provided_files: Iterable[str] = (),
    reserved_files: Iterable[str] = (),
) -> dict[str, Any]:
    """Follow explicit HTML URL attributes recursively within the output root.

    Remote URLs are reported but never fetched. Only named local dependencies
    are included; directories, hidden/staging files and symlinks are not swept
    into delivery. CSS/JavaScript runtime-generated links are not interpreted.
    """
    initial = set(initial_files)
    provided = set(provided_files)
    # ZIP names are case-sensitive even when packaging on Windows. Only the
    # canonical generated index spelling is guaranteed after extraction.
    reserved = {path.casefold() for path in reserved_files} | {path.casefold() for path in provided}
    queue = deque(sorted(path for path in initial if Path(path).suffix.lower() in _HTML_SUFFIXES))
    visited: set[str] = set()
    included: set[str] = set()
    issues: list[dict[str, str]] = []
    external: list[dict[str, str]] = []
    seen_references: set[tuple[str, str]] = set()

    def issue(source: str, reference: str, reason: str) -> None:
        issues.append({"source": source, "reference": reference, "reason": reason,
                       "action": _ACTIONS[reason]})

    while queue:
        source = queue.popleft()
        if source in visited:
            continue
        visited.add(source)
        parser = _References()
        try:
            parser.feed((root / source).read_text(encoding="utf-8-sig"))
            parser.close()
        except (OSError, UnicodeError, ValueError):
            issue(source, source, "unreadable_html")
            continue
        for reference, reason in parser.unsupported:
            issue(source, reference, reason)
        # With a base URL we cannot interpret the relative references below
        # using the document directory without changing their meaning.
        if any(reason == "html_base_url" for _, reason in parser.unsupported):
            continue
        for reference in parser.references:
            key = source, reference
            if key in seen_references:
                continue
            seen_references.add(key)
            try:
                parts = urlsplit(reference.strip())
                if parts.scheme or parts.netloc:
                    if parts.scheme.lower() in {"file"} or len(parts.scheme) == 1:
                        issue(source, reference, "absolute_path")
                    else:
                        external.append({"source": source, "reference": reference})
                    continue
                decoded = unquote(parts.path, encoding="utf-8", errors="strict").replace("\\", "/")
            except (ValueError, UnicodeError):
                issue(source, reference, "invalid_url")
                continue
            if "\x00" in decoded:
                issue(source, reference, "invalid_url")
                continue
            if not decoded:  # Fragment/query-only links stay in this document.
                continue
            if decoded.startswith("/") or (len(decoded) >= 2 and decoded[1] == ":"):
                issue(source, reference, "absolute_path")
                continue
            # abspath normalizes .. lexically before any file can be opened;
            # percent-encoded traversal receives exactly the same treatment.
            target = Path(os.path.abspath(root / Path(source).parent / decoded))
            try:
                problem = local_path_problem(root, target)
                if problem:
                    issue(source, reference, problem)
                    continue
                relative = target.relative_to(root).as_posix()
                # The new landing page is generated after discovery. Never
                # read/copy an older generation into its own replacement.
                if relative in provided:
                    continue
                if relative.casefold() in reserved:
                    issue(source, reference, "generated_delivery_reference")
                    continue
                if not target.exists():
                    issue(source, reference, "missing_file")
                    continue
                if not target.is_file():
                    issue(source, reference, "directory_reference")
                    continue
            except (OSError, ValueError):
                issue(source, reference, "unreadable_file")
                continue
            try:
                relative = record(target)
            except (OSError, ValueError):
                issue(source, reference, "unreadable_file")
                continue
            if relative not in initial:
                included.add(relative)
            if target.suffix.lower() in _HTML_SUFFIXES and relative not in visited:
                queue.append(relative)
    return {"included": sorted(included), "issues": issues, "external": external}
