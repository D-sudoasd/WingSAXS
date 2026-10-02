"""Finish and resume delivery of existing batch exports without fitting again.

The package is a navigable copy of existing evidence. Warnings remain warnings;
packaging does not classify scientific quality or rewrite source data.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import quote
import zipfile

from .csv_utils import iter_csv_rows
from .delivery_dependencies import collect_html_dependencies, local_path_problem

SCHEMA = "wingsaxs.delivery.v1"
_ARTIFACTS = {
    "frame_summary": "frame_summary.csv",
    "parameters": "parameters_long.csv",
    "ridge_points": "ridge_points.csv",
    "lobe_measurements": "lobe_measurements.csv",
    "fit_details": "ellipse_fit.json",
    "fit_records": "ellipse_fit.jsonl",
    "manifest": "manifest.json",
    "provenance": "provenance.json",
    "arrays": "results.npz",
    "evolution": "evolution.png",
}
_REQUIRED = {"frame_summary", "parameters", "manifest", "arrays", "evolution"}
_INDEX = "delivery_index.html"
_SUMMARY = "delivery_summary.json"
_ARCHIVE = "delivery.zip"
_CONTENTS_KEYS = ("schema_version", "status", "counts", "samples", "files", "dependencies")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_if_changed(path: Path, text: str) -> None:
    # Receipts can themselves be hashed by a parent delivery. A no-op resume
    # must not change their bytes or mtime just to report an execution action.
    if path.is_file() and path.read_bytes() == text.encode("utf-8"):
        return
    _atomic_text(path, text)


def _snapshot(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _changed(relative: str) -> ValueError:
    return ValueError(f"Output changed while packaging: {relative}; retry package --resume after export finishes.")


def _zip_info(relative: str, size: int = 0) -> zipfile.ZipInfo:
    # Package identity depends on contents, not export mtimes, local time zone,
    # host permissions or when an interrupted finalization was retried.
    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.file_size = size
    info.compress_type = (zipfile.ZIP_STORED if Path(relative).suffix.lower() in
                          {".npz", ".png", ".pdf", ".jpg", ".jpeg", ".zip"}
                          else zipfile.ZIP_DEFLATED)
    return info


def _write_source(bundle: zipfile.ZipFile, source: Path, relative: str, record: Mapping[str, Any]) -> None:
    # Hash the bytes actually copied into the ZIP, not a second read that could
    # disagree with them after a concurrent exporter replaces the source.
    digest = hashlib.sha256()
    info = _zip_info(relative, record["bytes"])
    with source.open("rb") as source_handle, bundle.open(info, "w") as output:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != record["sha256"]:
        raise _changed(relative)


def _previous_archive(report: Mapping[str, Any]) -> dict[str, Any]:
    archive = report.get("archive", {})
    if not isinstance(archive, dict):
        return {}
    if archive.get("status") == "skipped":
        archive = archive.get("previous", {})
    if not isinstance(archive, dict) or not archive.get("sha256"):
        return {}
    # Old v1 receipts bound the ZIP to the report's source signature. New ones
    # keep that binding on the archive, even when an index-only run follows.
    return {"sha256": archive["sha256"], "bytes": archive.get("bytes"),
            "source_signature": archive.get("source_signature", report.get("source_signature"))}


def _export_inventory(root: Path) -> tuple[list[Path], set[str]]:
    summaries = sorted(path for path in root.rglob("*frame_summary.csv")
                       if not any(part.startswith(".") for part in path.relative_to(root).parts))
    files: set[str] = set()
    for summary in summaries:
        prefix = summary.name.removesuffix("frame_summary.csv")
        folder = summary.parent
        files.update(path.relative_to(root).as_posix() for name in _ARTIFACTS.values()
                     if (path := folder / f"{prefix}{name}").is_file())
        figures = folder / "figures"
        if figures.is_dir():
            files.update(path.relative_to(root).as_posix() for path in figures.rglob("*")
                         if path.is_file() and not any(part.startswith(".") for part in path.relative_to(figures).parts))
    return summaries, files


def _link(path: str, label: str) -> str:
    return f'<a href="{quote(path, safe="/")}">{html.escape(label)}</a>'


def _render(report: Mapping[str, Any]) -> str:
    sections = []
    navigation = []
    dependency_issues = report.get("dependencies", {}).get("issues", [])
    dependency_notice = ""
    if dependency_issues:
        entries = "".join(
            f'<li>{html.escape(item["source"])} → {html.escape(item["reference"])}: '
            f'{html.escape(item["reason"])}. {html.escape(item["action"])}</li>'
            for item in dependency_issues
        )
        dependency_notice = ('<section class="warn"><h2>Delivery incomplete: linked files need attention</h2>'
                             f'<ul>{entries}</ul><p>Available outputs remain linked below.</p></section>')
    for number, sample in enumerate(report["samples"]):
        name = html.escape(sample["sample"])
        navigation.append(f'<li><a href="#sample-{number}">{name}</a> '
                          f'({len(sample["frames"])} frames)</li>')
        links = " · ".join(_link(path, role.replace("_", " "))
                           for role, path in sample["artifacts"].items())
        missing = ", ".join(sample["missing"])
        warnings = f'<p class="warn">Missing outputs: {html.escape(missing)}</p>' if missing else ""
        figure_links = "".join(f'<li>{_link(path, Path(path).name)}</li>' for path in sample["figures"])
        rows = []
        for frame in sample["frames"]:
            cells = [frame.get(key, "") for key in
                     ("csv_record", "frame_index", "frame_id", "path", "status", "quality_status", "diagnostic", "error")]
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>")
        evolution = sample["artifacts"].get("evolution")
        preview = (f'<a href="{quote(evolution, safe="/")}"><img loading="lazy" '
                   f'src="{quote(evolution, safe="/")}" alt="Parameter evolution for {name}"></a>') if evolution else ""
        sections.append(f'<section id="sample-{number}"><h2>{name}</h2><p>{links}</p>{warnings}'
                        f'{preview}<ul>{figure_links}</ul><details><summary>Frame-to-data lookup '
                        f'({len(sample["frames"])})</summary><p>Use frame_index / frame_id in the CSV and JSON files; '
                        'CSV record is zero-based record order; it does not replace missing frame metadata. '
                        'Original diagnostics and uncertainty columns remain in the linked data.</p>'
                        '<div class="scroll"><table><thead><tr><th>CSV record</th><th>Frame index</th><th>Frame ID</th><th>Source</th>'
                        '<th>Run status</th><th>Quality</th><th>Diagnostic</th><th>Error</th></tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody></table></div></details></section>')
    counts = report["counts"]
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>WingSAXS batch results</title><style>'
            'body{font:16px system-ui,sans-serif;max-width:1150px;margin:auto;padding:24px;color:#172333;background:#fafcff}'
            'a{color:#075a9e}section{padding:20px;margin:24px 0;background:white;border:1px solid #dbe4ec;border-radius:8px}'
            'img{max-width:100%;max-height:720px;object-fit:contain}table{border-collapse:collapse;width:100%}'
            'td,th{border:1px solid #dbe4ec;padding:8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}'
            'td{white-space:pre-wrap}.scroll{overflow:auto}.warn{color:#8b4800}summary{cursor:pointer}</style>'
            f'<h1>WingSAXS batch results</h1><p>{counts["samples"]} samples · {counts["frames"]} frames</p>'
            '<p>Open each sample for its tables, native arrays, fit details, plots and source-frame lookup. '
            'Run/quality warnings are retained; this delivery does not confer scientific acceptance.</p>'
            f'{dependency_notice}<nav><ul>{"".join(navigation)}</ul></nav>{"".join(sections)}</html>\n')


def package_batch(
    root: str | os.PathLike[str],
    *,
    archive: bool = True,
    resume: bool = False,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Index one export or a directory of sample exports and optionally ZIP it.

    Known export filenames (including their optional prefix) and files under
    each export's ``figures/`` are included, along with explicit local HTML
    dependencies inside the output root. Raw inputs and unrelated siblings
    are not swept into the archive. Resume rebuilds only changed delivery files.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"batch output directory does not exist: {root}")
    targets = [root / _INDEX, root / _SUMMARY] + ([root / _ARCHIVE] if archive else [])
    if not (resume or force) and any(path.exists() for path in targets):
        raise FileExistsError("Delivery already exists; use --resume to finish/reuse it, or --force to rebuild.")
    old = {}
    if (root / _SUMMARY).is_file():
        try:
            old = json.loads((root / _SUMMARY).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    if resume and not force and any(path.exists() for path in targets):
        if not isinstance(old, dict) or old.get("schema_version") != SCHEMA:
            raise ValueError("Existing delivery is not recognizable; choose --force only for generated delivery files.")
    samples: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, tuple[int, int, int, int, int]] = {}
    summaries, initial_inventory = _export_inventory(root)

    def verify_sources(*, inventory: bool = False) -> None:
        for relative, snapshot in snapshots.items():
            path = root / relative
            if local_path_problem(root, path) or not path.is_file() or _snapshot(path) != snapshot:
                raise _changed(relative)
        if inventory:
            current_summaries, current_inventory = _export_inventory(root)
            if current_summaries != summaries or current_inventory != initial_inventory:
                raise ValueError("Output inventory changed while packaging; retry package --resume after export finishes.")

    def record(path: Path) -> str:
        if local_path_problem(root, path):
            raise ValueError(f"Delivery source must be a regular file inside the output directory: {path}")
        relative = path.relative_to(root).as_posix()
        if relative not in files:
            if progress:
                progress(f"Indexing {relative}")
            if local_path_problem(root, path) or not path.is_file():
                raise _changed(relative)
            snapshots[relative] = _snapshot(path)
            files[relative] = {"bytes": snapshots[relative][2], "sha256": _digest(path)}
            if _snapshot(path) != snapshots[relative]:
                raise _changed(relative)
        return relative

    if not summaries:
        raise ValueError("No frame_summary.csv exports found. Point package at a batch output or its sample parent directory.")
    for summary in summaries:
        # Read navigation records only after taking the same source snapshot
        # used by the archive. Otherwise a changed CSV can be hashed correctly
        # while its old frame statuses remain in the navigation.
        record(summary)
        prefix = summary.name.removesuffix("frame_summary.csv")
        folder = summary.parent
        relative_folder = folder.relative_to(root).as_posix()
        sample_name = (root.name if relative_folder == "." else relative_folder) + (f"/{prefix.rstrip('_')}" if prefix else "")
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            # Discard unneeded large metadata columns after each record.
            # Keep package RAM proportional to the small navigation index,
            # rather than reconstructing a whole streamed batch in memory.
            frames = [{"csv_record": index, **{key: row.get(key, "") for key in
                       ("frame_index", "frame_id", "path", "status", "quality_status", "diagnostic", "error")}}
                      for index, row in enumerate(iter_csv_rows(handle))]
        artifacts = {role: record(folder / f"{prefix}{name}") for role, name in _ARTIFACTS.items()
                     if (folder / f"{prefix}{name}").is_file()}
        figures = []
        figure_dir = folder / "figures"
        if figure_dir.is_dir():
            figures = [record(path) for path in sorted(figure_dir.rglob("*"))
                       if path.is_file() and not any(part.startswith(".") for part in path.relative_to(figure_dir).parts)]
        samples.append({"sample": sample_name, "frames": frames, "artifacts": artifacts,
                        "figures": figures, "missing": sorted(_REQUIRED - artifacts.keys())})
    dependencies = collect_html_dependencies(
        root, tuple(files), record, provided_files=(_INDEX,),
        reserved_files=(_SUMMARY, _ARCHIVE, "delivery_contents.json"),
    )
    verify_sources()
    statuses = Counter(frame["status"] or "unknown" for sample in samples for frame in sample["frames"])
    quality_warnings = sum(str(frame["quality_status"]).upper() in {"WARN", "WARNING", "FAIL", "FAILED", "INVALID"}
                           for sample in samples for frame in sample["frames"])
    missing = sum(len(sample["missing"]) for sample in samples)
    warning = missing or quality_warnings or any(not sample["frames"] for sample in samples) or any(status not in {"ok", "success", "recovered"} for status in statuses)
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "incomplete" if dependencies["issues"] else "ready_with_warnings" if warning else "ready",
        "operation_status": "completed",
        "scientific_acceptance": False,
        "counts": {"samples": len(samples), "frames": sum(statuses.values()), "frame_statuses": dict(statuses),
                   "quality_warning_frames": quality_warnings, "missing_artifacts": missing},
        "samples": samples, "files": files, "dependencies": dependencies,
    }
    index_text = _render(report)
    index_digest = hashlib.sha256(index_text.encode("utf-8")).hexdigest()
    contents = {key: report[key] for key in _CONTENTS_KEYS}
    signature = hashlib.sha256(_json({"contents": contents, "index_sha256": index_digest}).encode("utf-8")).hexdigest()
    previous_archive = _previous_archive(old) if isinstance(old, dict) else {}
    legacy_signature = hashlib.sha256(_json({"files": files, "index_sha256": index_digest}).encode("utf-8")).hexdigest()
    if (previous_archive.get("source_signature") == legacy_signature
            and {key: old.get(key) for key in _CONTENTS_KEYS} == contents):
        # Do not recompress a pre-existing v1 delivery just to upgrade receipt
        # identity. All embedded contents must match, not only its file hashes.
        signature = legacy_signature
    report["source_signature"] = signature
    report["index_sha256"] = index_digest
    report["archive"] = {"status": "skipped", "reason": "archive creation disabled"}
    if not archive and previous_archive:
        report["archive"]["previous"] = previous_archive
    report["outputs"] = {"index": str(root / _INDEX), "summary": str(root / _SUMMARY)}
    index_changed = not (root / _INDEX).is_file() or _digest(root / _INDEX) != index_digest
    verify_sources()
    # Establish a recovery receipt before replacing the index. An interruption
    # must not leave a new index beside an old receipt claiming completion.
    # Keep the previous archive's own binding available for a possible reuse.
    if not isinstance(old, dict) or old.get("schema_version") != SCHEMA or index_changed:
        pending = {**report, "operation_status": "indexing"}
        if archive:
            pending["archive"] = {**previous_archive, "status": "pending"}
        _atomic_text(root / _SUMMARY, _json(pending))
    # Index recovery precedes archive work, so a packaging interruption still
    # leaves a useful landing page. Only generated delivery files are replaced.
    if index_changed:
        _atomic_text(root / _INDEX, index_text)
    if archive:
        destination = root / _ARCHIVE
        reuse = (resume and not force and previous_archive.get("source_signature") == signature
                 and previous_archive.get("sha256")
                 and destination.is_file() and _digest(destination) == previous_archive["sha256"])
        if reuse:
            report["archive"] = {**previous_archive, "status": "reused"}
        else:
            # Save an honest partial receipt first, so interrupted ZIP creation
            # can be retried with --resume without touching analysis outputs.
            report["operation_status"] = "packaging"
            report["archive"] = {"status": "pending"}
            _atomic_text(root / _SUMMARY, _json(report))
            fd, temporary = tempfile.mkstemp(prefix=".delivery-", suffix=".zip", dir=root)
            os.close(fd)
            try:
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
                    bundle.writestr(_zip_info(_INDEX), index_text)
                    # Archive-contained receipt is relocatable and does not
                    # claim a circular checksum of its own enclosing ZIP.
                    bundle.writestr(_zip_info("delivery_contents.json"), _json({**contents, "source_signature": signature}))
                    for number, (relative, source_record) in enumerate(files.items(), start=1):
                        if progress:
                            progress(f"Packaging {number}/{len(files)}: {relative}")
                        source = root / relative
                        if local_path_problem(root, source) or not source.is_file() or _snapshot(source) != snapshots[relative]:
                            raise _changed(relative)
                        _write_source(bundle, source, relative, source_record)
                verify_sources()
                # Save the completed candidate's binding BEFORE publishing the
                # ZIP. If the last receipt write fails, resume recognizes that
                # already-published ZIP without repeating compression.
                report["archive"] = {"status": "pending", "sha256": _digest(Path(temporary)),
                                     "bytes": Path(temporary).stat().st_size, "source_signature": signature}
                _atomic_text(root / _SUMMARY, _json(report))
                verify_sources(inventory=True)
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
            report["archive"]["status"] = "created"
        report["outputs"]["archive"] = str(destination)
    verify_sources(inventory=not archive or bool(reuse))
    report["operation_status"] = "completed"
    report["exit_code"] = 1 if warning or dependencies["issues"] else 0
    report["next_action"] = ("Resolve the linked-file issues shown in the index, then rerun package --resume; available outputs are preserved."
                             if dependencies["issues"] else
                             "Review missing outputs and recorded frame diagnostics in the index; rerun package --resume after adding outputs."
                             if warning else ("Open delivery_index.html, or extract delivery.zip and open the same index."
                                              if archive else "Open delivery_index.html to browse the existing outputs."))
    # created/reused describes this invocation, not a different artifact.
    # Persist a stable receipt so downstream hash bindings survive no-op runs.
    persisted = report
    if report["archive"]["status"] == "reused":
        persisted = {**report, "archive": {**report["archive"], "status": "created"}}
        # Already-bound legacy receipts need not be migrated just to add
        # optional bindings. Preserve their exact bytes (including formatting)
        # when the completed receipt is equivalent to this verified result.
        previous = {**old, "index_sha256": old.get("index_sha256", index_digest),
                    "archive": {**old.get("archive", {}), "status": "created",
                                "source_signature": previous_archive["source_signature"]}}
        if not index_changed and previous == persisted:
            return report
    _write_if_changed(root / _SUMMARY, _json(persisted))
    return report
