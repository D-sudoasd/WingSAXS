"""Audit two complete benchmark bundles without loading a whole sequence."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image


# These describe execution/proof generation, rather than scientific values.
_OPERATIONAL = {"created_at", "updated_at", "elapsed_s", "resumed", "generation",
                "producer_identity", "checkpoint_frame_sha256", "source_state",
                "artifact_sha256", "receipt_sha256", "exporter_identity"}
_BUNDLE_COMMIT_ARTIFACTS = {
    "ellipse_fit": "ellipse_fit.json",
    "ellipse_fit_jsonl": "ellipse_fit.jsonl",
    "evolution_png": "evolution.png",
    "frame_details": "frame_details.jsonl",
    "frame_summary": "frame_summary.csv",
    "lobe_measurements": "lobe_measurements.csv",
    "manifest": "manifest.json",
    "npz": "results.npz",
    "parameters_long": "parameters_long.csv",
    "provenance": "provenance.json",
    "ridge_points": "ridge_points.csv",
}


def _digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _normalized(value, root):
    if isinstance(value, dict):
        return {key: _normalized(item, root) for key, item in value.items() if key not in _OPERATIONAL}
    if isinstance(value, list):
        return [_normalized(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<OUTPUT>").replace(root.as_posix(), "<OUTPUT>")
    return value


def _inventory(root):
    return {path.relative_to(root).as_posix(): path for path in root.rglob("*")
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)}


def _svg_contents(path):
    """Compare actual vector/raster data, ignoring dates and random SVG IDs."""
    root = ET.parse(path).getroot()
    ids = {node.attrib["id"]: f"id_{index}" for index, node in enumerate(root.iter()) if "id" in node.attrib}

    def references(value):
        return re.sub(r"(?<=#)[A-Za-z_][\w:.-]*", lambda match: ids.get(match[0], match[0]), value)

    def contents(node):
        return (node.tag, sorted((key, ids[value] if key == "id" else references(value))
                                 for key, value in node.attrib.items()), references(node.text or ""),
                [contents(child) for child in node if child.tag != "{http://purl.org/dc/elements/1.1/}date"])

    return contents(root)


def _pdf_contents(path):
    """Compare decoded drawing commands, resources, images, fonts and pages.

    Requires pypdf (available in the bundled document runtime). Object numbers,
    stream compression and creation dates do not describe plot content.
    """
    from pypdf import PdfReader
    from pypdf.generic import StreamObject

    def contents(value, active=None, location="$"):
        value = value.get_object() if hasattr(value, "get_object") else value
        active = {} if active is None else active
        if isinstance(value, (dict, list)):
            if id(value) in active:
                return ("reference", active[id(value)])
            active[id(value)] = location
            try:
                if isinstance(value, dict):
                    omitted = {"/Length", "/Filter", "/DecodeParms"} if isinstance(value, StreamObject) else set()
                    data = [(str(key), contents(item, active, location + str(key)))
                            for key, item in sorted(value.items()) if key not in omitted]
                    if isinstance(value, StreamObject):
                        data.append(("decoded_stream_sha256", hashlib.sha256(value.get_data()).hexdigest()))
                    return data
                return [contents(item, active, f"{location}[{index}]") for index, item in enumerate(value)]
            finally:
                active.pop(id(value))
        if isinstance(value, bytes):
            return ("bytes", hashlib.sha256(value).hexdigest())
        return str(value)

    reader = PdfReader(path, strict=True)
    pages = []
    for page in reader.pages:
        properties = {key: value for key, value in page.items() if key not in {"/Parent", "/Contents"}}
        pages.append((contents(properties), page.get_contents().get_data()))
    metadata = {key: value for key, value in (reader.metadata or {}).items() if key not in {"/CreationDate", "/ModDate"}}
    return contents(metadata), pages


def _verify_delivery(root):
    receipt = json.loads((root / "delivery_summary.json").read_text(encoding="utf-8"))
    path = root / "delivery.zip"
    assert receipt["operation_status"] == "completed"
    assert _digest(path) == receipt["archive"]["sha256"], "ZIP SHA differs from its receipt"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "duplicate ZIP member"
        assert set(names) == set(receipt["files"]) | {"delivery_index.html", "delivery_contents.json"}
        verified_files = {}
        for name in names:
            with archive.open(name) as handle:
                # Streaming to EOF validates CRC as well as the member SHA.
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if name == "delivery_index.html":
                assert digest == receipt["index_sha256"]
                assert digest == _digest(root / name), "published index differs from the ZIP"
            elif name != "delivery_contents.json":
                assert digest == receipt["files"][name]["sha256"], f"member SHA differs: {name}"
                assert _digest(root / name) == digest, f"published source differs: {name}"
                verified_files[name] = digest
        contents = json.loads(archive.read("delivery_contents.json"))
        assert contents["files"] == receipt["files"]
        assert contents["source_signature"] == receipt["source_signature"]
        for key, value in contents.items():
            if key not in {"files", "source_signature"}:
                assert value == receipt.get(key), f"embedded delivery metadata differs: {key}"

    # The native bundle commit marker is deliberately hidden from the public
    # delivery inventory. Validate its small proof separately and bind every
    # declared artifact to both the package receipt and the source SHA already
    # checked above (including the large NPZ, without hashing it a second time).
    marker_path = root / ".bundle.commit.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert isinstance(marker, dict), "native bundle commit marker must be an object"
    assert marker.get("schema_version") == "lamellarsaxs2d.bundle_commit.v1", "unsupported native bundle commit marker"
    assert marker.get("complete") is True, "native bundle commit marker is incomplete"
    artifacts = marker.get("artifacts")
    assert isinstance(artifacts, dict), "native bundle commit artifacts must be an object"
    assert set(artifacts) == set(_BUNDLE_COMMIT_ARTIFACTS), "native bundle commit artifact inventory differs"
    verified_artifacts = {}
    for key, expected_name in _BUNDLE_COMMIT_ARTIFACTS.items():
        entry = artifacts[key]
        assert isinstance(entry, dict) and set(entry) == {"name", "sha256"}, f"invalid native artifact proof: {key}"
        name, digest = entry["name"], entry["sha256"]
        # Fixed schema-v1 artifact names reject absolute paths, traversal and
        # other path aliases before consulting receipt/source inventories.
        assert name == expected_name, f"invalid native artifact path: {key}"
        assert isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), f"invalid native artifact SHA: {key}"
        assert name in receipt["files"], f"native artifact is absent from delivery receipt: {name}"
        assert name in verified_files, f"native artifact was not verified in the package: {name}"
        assert digest == receipt["files"][name]["sha256"] == verified_files[name], f"native artifact SHA differs: {name}"
        verified_artifacts[key] = {"name": name, "sha256": digest}

    semantic = {key: value for key, value in receipt.items()
                if key not in {"files", "archive", "source_signature", "index_sha256"}}
    semantic["archive"] = {key: value for key, value in receipt["archive"].items()
                           if key not in {"sha256", "bytes", "source_signature"}}
    return {"members": len(names), "member_names": sorted(names),
            "bytes": path.stat().st_size, "sha256": receipt["archive"]["sha256"],
            "counts": receipt["counts"], "dependencies": receipt["dependencies"], "semantic_receipt": semantic,
            "native_commit": {"schema_version": marker["schema_version"], "complete": True,
                              "marker_sha256": _digest(marker_path), "artifacts": verified_artifacts}}


def compare(before: Path, after: Path, *, allow_overview_legend_changes: bool = False) -> dict:
    before, after = before.resolve(), after.resolve()
    started = time.perf_counter()
    result = {"passed": True, "before": str(before), "after": str(after), "failures": [],
              "operational_fields_excluded": sorted(_OPERATIONAL), "arrays": 0, "array_values": 0,
              "maximum_absolute_error": 0.0, "csv_rows": 0, "json_records": 0,
              "diagnostic_timing_values_excluded": 0,
              "png_identical": 0, "png_legend_changes": [],
              "overview_legend_changes_allowed": bool(allow_overview_legend_changes), "vector_figures": 0}
    result["vector_check"] = "SVG geometry/text/embedded rasters; PDF decoded commands/resources/fonts/images/pages"
    result["comparison_script_sha256"] = _digest(Path(__file__))
    result["python"] = sys.version
    result["validation_versions"] = {name: metadata.version(name) for name in ("numpy", "Pillow", "pypdf")}
    first, second = _inventory(before), _inventory(after)
    result["files"] = len(first)
    if set(first) != set(second):
        result["failures"].append({"inventory": sorted(set(first) ^ set(second))})
    csv.field_size_limit(2**31 - 1)
    for name in sorted(set(first) & set(second)):
        left, right = first[name], second[name]
        try:
            if name in {"delivery.zip", "delivery_summary.json"}:
                continue
            if left.suffix == ".npz":
                with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
                    assert set(a.files) == set(b.files), "NPZ member sets differ"
                    for key in a.files:
                        x, y = a[key], b[key]
                        assert x.dtype == y.dtype and x.shape == y.shape, f"NPZ shape/dtype: {key}"
                        # Native NPY payloads must match bit for bit, including
                        # NaN/Inf placement. This is stronger than tolerance.
                        assert x.tobytes() == y.tobytes(), f"NPZ values differ: {key}"
                        result["arrays"] += 1
                        result["array_values"] += x.size
            elif left.suffix == ".csv":
                with left.open(encoding="utf-8-sig", newline="") as a, right.open(encoding="utf-8-sig", newline="") as b:
                    rows_a, rows_b = csv.DictReader(a), csv.DictReader(b)
                    assert rows_a.fieldnames == rows_b.fieldnames, "CSV columns differ"
                    for index, (x, y) in enumerate(zip(rows_a, rows_b, strict=True)):
                        # The long diagnostic table represents run timing as
                        # a value row rather than an elapsed_s column. Retain
                        # its field, frame, type and every other cell.
                        if (x.get("field") == y.get("field")
                                and x.get("field", "").endswith((".diagnostics.elapsed_s", ".trace_diagnostics.elapsed_s"))):
                            x["value"] = y["value"] = "<EXECUTION_SECONDS>"
                            result["diagnostic_timing_values_excluded"] += 1
                        # Most long-table rows are already identical. Avoid
                        # recursively copying every cell in multi-million-row
                        # full batches; differing rows retain the same checks.
                        assert x == y or _normalized(x, before) == _normalized(y, after), f"CSV row {index} differs"
                        result["csv_rows"] += 1
            elif left.suffix in {".json", ".jsonl"}:
                with left.open(encoding="utf-8") as a, right.open(encoding="utf-8") as b:
                    if left.suffix == ".json":
                        records = [(json.load(a), json.load(b))]
                    else:
                        records = ((json.loads(x), json.loads(y)) for x, y in zip(a, b, strict=True))
                    for x, y in records:
                        if name.endswith("report_summary.json"):
                            # Only the report builder's top-level receipt
                            # version changed. Scientific method tags remain.
                            x.pop("method_version", None)
                            y.pop("method_version", None)
                        assert _normalized(x, before) == _normalized(y, after), "scientific JSON differs"
                        result["json_records"] += 1
            elif left.suffix == ".png":
                with Image.open(left) as a, Image.open(right) as b:
                    assert a.size == b.size, "PNG dimensions differ"
                    identical = a.convert("RGBA").tobytes() == b.convert("RGBA").tobytes()
                if identical:
                    result["png_identical"] += 1
                else:
                    assert name.endswith("_overview.png"), "unexpected image pixel change"
                    result["png_legend_changes"].append(name)
                    assert allow_overview_legend_changes, "overview image pixel changes are disabled"
            elif left.suffix in {".svg", ".pdf"}:
                reader = _svg_contents if left.suffix == ".svg" else _pdf_contents
                try:
                    assert reader(left) == reader(right), "vector drawing commands/resources differ"
                except Exception as error:
                    raise ValueError(f"Vector comparison failed: {error}") from error
                result["vector_figures"] += 1
            else:
                a, b = left.read_text(encoding="utf-8"), right.read_text(encoding="utf-8")
                assert _normalized(a, before) == _normalized(b, after), "text/index differs"
        except (AssertionError, ValueError, OSError, KeyError) as error:
            result["failures"].append({"file": name, "error": str(error)})
    for label, root in (("before", before), ("after", after)):
        try:
            result[label + "_delivery"] = _verify_delivery(root)
        except (AssertionError, ValueError, OSError, KeyError) as error:
            result["failures"].append({"bundle": label, "error": str(error)})
    if "before_delivery" in result and "after_delivery" in result:
        if result["before_delivery"]["member_names"] != result["after_delivery"]["member_names"]:
            result["failures"].append({"delivery_members": "Package member inventories differ"})
        if (_normalized(result["before_delivery"]["semantic_receipt"], before)
                != _normalized(result["after_delivery"]["semantic_receipt"], after)):
            result["failures"].append({"delivery_metadata": "Package receipt/content semantics differ"})
    result["passed"] = not result["failures"]
    result["verification_wall_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-overview-legend-changes", action="store_true",
                        help="Diagnostic only: allow pixel changes in overview PNGs while recording them in the receipt")
    args = parser.parse_args()
    if util.find_spec("pypdf") is None:
        parser.error("PDF verification requires pypdf. Use the bundled document Python runtime or install pypdf in this verification environment.")
    if args.output.exists():
        parser.error("Refusing to overwrite comparison evidence")
    result = compare(args.before, args.after, allow_overview_legend_changes=args.allow_overview_legend_changes)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "files", "arrays", "csv_rows", "failures", "verification_wall_seconds")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
