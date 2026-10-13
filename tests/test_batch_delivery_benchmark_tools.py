from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from PIL import Image
import pytest

from scripts import compare_batch_delivery


_ARTIFACTS = {
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _isolated_compare(monkeypatch):
    monkeypatch.setattr(compare_batch_delivery.metadata, "version", lambda _: "fixture-version")
    monkeypatch.setattr(compare_batch_delivery, "_verify_delivery", lambda _: {
        "members": 0, "member_names": [], "bytes": 0, "sha256": "fixture",
        "counts": {}, "dependencies": {}, "semantic_receipt": {},
    })


def _image_pair(tmp_path: Path, name: str):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    Image.new("RGB", (3, 2), (200, 20, 10)).save(before / name)
    Image.new("RGB", (3, 2), (10, 20, 200)).save(after / name)
    return before, after


def test_overview_pixel_change_fails_by_default_and_is_recorded(tmp_path, monkeypatch):
    _isolated_compare(monkeypatch)
    before, after = _image_pair(tmp_path, "frame_0000_overview.png")

    receipt = compare_batch_delivery.compare(before, after)

    assert receipt["passed"] is False
    assert receipt["overview_legend_changes_allowed"] is False
    assert receipt["png_legend_changes"] == ["frame_0000_overview.png"]
    assert any(item.get("error") == "overview image pixel changes are disabled"
               for item in receipt["failures"])


def test_explicit_overview_diagnostic_option_is_recorded_and_allows_only_overviews(tmp_path, monkeypatch):
    _isolated_compare(monkeypatch)
    before, after = _image_pair(tmp_path, "frame_0000_overview.png")

    receipt = compare_batch_delivery.compare(before, after, allow_overview_legend_changes=True)

    assert receipt["passed"] is True
    assert receipt["overview_legend_changes_allowed"] is True
    assert receipt["png_legend_changes"] == ["frame_0000_overview.png"]


def test_non_overview_pixel_change_fails_even_with_overview_diagnostic_option(tmp_path, monkeypatch):
    _isolated_compare(monkeypatch)
    before, after = _image_pair(tmp_path, "frame_0000_fit_support.png")

    receipt = compare_batch_delivery.compare(before, after, allow_overview_legend_changes=True)

    assert receipt["passed"] is False
    assert receipt["overview_legend_changes_allowed"] is True
    assert any(item.get("error") == "unexpected image pixel change" for item in receipt["failures"])


@pytest.mark.parametrize("change", ["identical", "execution", "output_root", "scientific"])
def test_csv_fast_path_retains_scientific_and_operational_checks(tmp_path, monkeypatch, change):
    _isolated_compare(monkeypatch)
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    left = "frame_id,a,elapsed_s,path\nf0,1.25,12,source.h5\n"
    right = left
    if change == "execution":
        right = left.replace(",12,", ",99,")
    elif change == "scientific":
        right = left.replace("1.25", "2.5")
    elif change == "output_root":
        left = left.replace("source.h5", str(before / "source.h5"))
        right = right.replace("source.h5", str(after / "source.h5"))
    (before / "values.csv").write_text(left, encoding="utf-8")
    (after / "values.csv").write_text(right, encoding="utf-8")
    receipt = compare_batch_delivery.compare(before, after)
    assert receipt["passed"] is (change != "scientific")
    if change == "scientific":
        assert receipt["failures"][0]["error"] == "CSV row 0 differs"


def test_cli_receipt_records_explicit_overview_diagnostic_option(tmp_path, monkeypatch):
    _isolated_compare(monkeypatch)
    monkeypatch.setattr(compare_batch_delivery.util, "find_spec", lambda _: object())
    before, after = _image_pair(tmp_path, "frame_0000_overview.png")
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", [
        "compare_batch_delivery.py", str(before), str(after), "--output", str(output),
        "--allow-overview-legend-changes",
    ])

    assert compare_batch_delivery.main() == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["passed"] is True
    assert receipt["overview_legend_changes_allowed"] is True
    assert receipt["png_legend_changes"] == ["frame_0000_overview.png"]


def _write_delivery_bundle(root: Path):
    root.mkdir()
    contents_by_name = {name: f"fixture:{name}".encode() for name in _ARTIFACTS.values()}
    for name, content in contents_by_name.items():
        (root / name).write_bytes(content)
    index = b"<html>fixture</html>"
    (root / "delivery_index.html").write_bytes(index)
    files = {name: {"sha256": _sha256(content)} for name, content in contents_by_name.items()}
    source_signature = "fixture-source"
    embedded = json.dumps({"files": files, "source_signature": source_signature}).encode("utf-8")
    archive_path = root / "delivery.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents_by_name.items():
            archive.writestr(name, content)
        archive.writestr("delivery_index.html", index)
        archive.writestr("delivery_contents.json", embedded)
    receipt = {
        "operation_status": "completed",
        "archive": {"sha256": _file_sha256(archive_path),
                    "bytes": archive_path.stat().st_size, "status": "created",
                    "source_signature": source_signature},
        "files": files,
        "index_sha256": _sha256(index),
        "source_signature": source_signature,
        "counts": {},
        "dependencies": {},
    }
    (root / "delivery_summary.json").write_text(json.dumps(receipt), encoding="utf-8")
    marker = {
        "schema_version": "lamellarsaxs2d.bundle_commit.v1",
        "complete": True,
        "generation": "fixture-generation",
        "artifacts": {key: {"name": name, "sha256": files[name]["sha256"]}
                      for key, name in _ARTIFACTS.items()},
    }
    marker_path = root / ".bundle.commit.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    return marker_path, marker


def test_delivery_verification_checks_hidden_native_commit_marker(tmp_path):
    marker_path, _ = _write_delivery_bundle(tmp_path / "bundle")

    receipt = compare_batch_delivery._verify_delivery(marker_path.parent)

    assert receipt["native_commit"]["complete"] is True
    assert receipt["native_commit"]["marker_sha256"] == _file_sha256(marker_path)
    assert receipt["native_commit"]["artifacts"]["npz"]["name"] == "results.npz"


@pytest.mark.parametrize("damage", ["missing_marker", "missing_artifact", "invalid_path", "bad_sha", "incomplete"])
def test_delivery_verification_rejects_invalid_native_commit_marker(tmp_path, damage):
    marker_path, marker = _write_delivery_bundle(tmp_path / "bundle")
    if damage == "missing_marker":
        marker_path.unlink()
        error = OSError
        message = r"\.bundle\.commit"
    else:
        if damage == "missing_artifact":
            marker["artifacts"].pop("npz")
            message = "native bundle commit artifact inventory differs"
        elif damage == "invalid_path":
            marker["artifacts"]["npz"]["name"] = "../results.npz"
            message = "invalid native artifact path"
        elif damage == "bad_sha":
            marker["artifacts"]["npz"]["sha256"] = "0" * 64
            message = "native artifact SHA differs"
        else:
            marker["complete"] = False
            message = "native bundle commit marker is incomplete"
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        error = AssertionError

    with pytest.raises(error, match=message):
        compare_batch_delivery._verify_delivery(marker_path.parent)


def test_hard_stop_records_measured_checkpoint_and_startup_identity(tmp_path):
    source = Path(__file__).resolve().parents[1]
    driver = source / "scripts" / "benchmark_batch_delivery.py"
    manifest, config = tmp_path / "manifest.json", tmp_path / "config.json"
    output, metrics = tmp_path / "bundle", tmp_path / "hard-stop.metrics.json"
    manifest.write_text("{}", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    child = r"""
import runpy
from pathlib import Path
import sys

source, manifest, config, output, metrics = map(Path, sys.argv[1:])
sys.path.insert(0, str(source / "src"))
from butterfly_saxs import batch, cli

def fake_main(command):
    checkpoint = Path(command[command.index("--checkpoint") + 1])
    batch.write_checkpoint(checkpoint, {
        "processed_count": 1,
        "frames": [{"status": "warning", "result": {"probe": 0.25}}],
    })
    raise AssertionError("durable checkpoint did not trigger the real hard stop")

cli.main = fake_main
sys.argv = [
    str(source / "scripts" / "benchmark_batch_delivery.py"),
    "--source", str(source), "--manifest", str(manifest), "--config", str(config),
    "--output", str(output), "--hard-stop-after", "1", "--worker", "--metrics", str(metrics),
]
runpy.run_path(sys.argv[0], run_name="__main__")
"""

    completed = subprocess.run(
        [sys.executable, "-c", child, str(source), str(manifest), str(config), str(output), str(metrics)],
        cwd=source, capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 99, completed.stderr
    checkpoint = output / "checkpoint.json"
    assert checkpoint.is_file()
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["frames"][0]["result"]["probe"] == 0.25
    record = json.loads(metrics.read_text(encoding="utf-8"))
    assert record["injected_abrupt_exit"] is True
    assert record["phase"] == "after_durable_checkpoint_before_bundle_publish"
    assert isinstance(record["source_sha256"], dict) and record["source_sha256"]
    assert record["benchmark_driver_sha256"] == _file_sha256(driver)
    assert record["python"] and record["platform"]
    assert isinstance(record["thread_environment"], dict)
    assert record["cpu_seconds"] >= 0
    assert record["process_cpu_seconds"] >= record["cpu_seconds"]
    assert record["cli_wall_seconds"] >= 0
    checkpoint_stage = record["stages_inclusive"]["checkpoint_write"]
    assert checkpoint_stage["calls"] == 1
    assert checkpoint_stage["seconds"] >= 0
