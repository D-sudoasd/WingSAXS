"""Audit two complete benchmark bundles without loading a whole sequence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image


# These describe execution/proof generation, rather than scientific values.
_OPERATIONAL = {"created_at", "updated_at", "elapsed_s", "resumed", "generation",
                "producer_identity", "checkpoint_frame_sha256", "source_state",
                "artifact_sha256", "method_version", "receipt_sha256"}


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


def _verify_delivery(root):
    receipt = json.loads((root / "delivery_summary.json").read_text(encoding="utf-8"))
    path = root / "delivery.zip"
    assert receipt["operation_status"] == "completed"
    assert _digest(path) == receipt["archive"]["sha256"], "ZIP SHA differs from its receipt"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "duplicate ZIP member"
        assert set(names) == set(receipt["files"]) | {"delivery_index.html", "delivery_contents.json"}
        for name in names:
            with archive.open(name) as handle:
                # Streaming to EOF validates CRC as well as the member SHA.
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if name == "delivery_index.html":
                assert digest == receipt["index_sha256"]
            elif name != "delivery_contents.json":
                assert digest == receipt["files"][name]["sha256"], f"member SHA differs: {name}"
                assert _digest(root / name) == digest, f"published source differs: {name}"
        contents = json.loads(archive.read("delivery_contents.json"))
        assert contents["files"] == receipt["files"]
        assert contents["source_signature"] == receipt["source_signature"]
    return {"members": len(names), "bytes": path.stat().st_size, "sha256": receipt["archive"]["sha256"],
            "counts": receipt["counts"], "dependencies": receipt["dependencies"]}


def compare(before: Path, after: Path) -> dict:
    before, after = before.resolve(), after.resolve()
    started = time.perf_counter()
    result = {"passed": True, "before": str(before), "after": str(after), "failures": [],
              "operational_fields_excluded": sorted(_OPERATIONAL), "arrays": 0, "array_values": 0,
              "maximum_absolute_error": 0.0, "csv_rows": 0, "json_records": 0,
              "png_identical": 0, "png_legend_changes": [], "vector_figures": 0}
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
                        assert _normalized(x, before) == _normalized(y, after), f"CSV row {index} differs"
                        result["csv_rows"] += 1
            elif left.suffix in {".json", ".jsonl"}:
                with left.open(encoding="utf-8") as a, right.open(encoding="utf-8") as b:
                    if left.suffix == ".json":
                        records = [(json.load(a), json.load(b))]
                    else:
                        records = ((json.loads(x), json.loads(y)) for x, y in zip(a, b, strict=True))
                    for x, y in records:
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
            elif left.suffix in {".svg", ".pdf"}:
                expected = b"%PDF" if left.suffix == ".pdf" else b"<?xml"
                with left.open("rb") as a, right.open("rb") as b:
                    assert a.read(5).startswith(expected[:5]) and b.read(5).startswith(expected[:5])
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
    result["passed"] = not result["failures"]
    result["verification_wall_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite comparison evidence")
    result = compare(args.before, args.after)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "files", "arrays", "csv_rows", "failures", "verification_wall_seconds")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
