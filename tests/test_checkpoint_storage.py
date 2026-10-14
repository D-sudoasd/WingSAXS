from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from butterfly_saxs import batch, producer
from butterfly_saxs.export import StreamingBatchExporter


def _legacy_identity(current):
    old = copy.deepcopy(current)
    old_batch, old_producer = sorted(producer._PRETTY_CHECKPOINT_SOURCES)[0]
    old["sources"].update({"batch.py": old_batch, "producer.py": old_producer})
    return old


def test_compact_checkpoint_preserves_legacy_json_values_and_float_bits(tmp_path):
    payload = {"version": 1, "frame_id": "实测帧", "frames": [{
        "result": {"profile": [0., -0., np.nextafter(1., 2.), 1e-300, None],
                   "candidate": None, "flags": ["warning"], "observed": True},
    }]}
    expected = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    path = tmp_path / "checkpoint.json"
    batch.write_checkpoint(path, payload)
    compact = path.read_text(encoding="utf-8")
    assert len(compact.encode("utf-8")) < len(expected.encode("utf-8"))
    assert compact.endswith("\n") and "\n " not in compact
    assert batch.read_checkpoint(path) == json.loads(expected)
    actual = batch.read_checkpoint(path)["frames"][0]["result"]["profile"]
    assert [value.hex() for value in actual[:-1]] == [float(value).hex() for value in payload["frames"][0]["result"]["profile"][:-1]]
    path.write_text(expected, encoding="utf-8")
    assert batch.read_checkpoint(path) == payload


def test_storage_upgrade_is_pinned_to_the_reviewed_batch_source():
    raw = Path(batch.__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest() in producer._COMPACT_CHECKPOINT_BATCH_SHA256
    current = {**producer.producer_identity(), "analyzer": "fixture v1"}
    assert producer.calculation_producers_compatible(_legacy_identity(current), current)
    changed = copy.deepcopy(current)
    changed["sources"]["batch.py"] = hashlib.sha256(raw + b"\n# future edit\n").hexdigest()
    assert not producer.calculation_producers_compatible(_legacy_identity(current), changed)


@pytest.mark.parametrize("change", ["input_module", "dependency", "analyzer", "old_batch", "old_builder", "new_builder", "unbound", "malformed"])
def test_storage_upgrade_keeps_all_other_identity_checks(change):
    current = {**producer.producer_identity(), "analyzer": "fixture v1"}
    old = _legacy_identity(current)
    if change == "input_module":
        old["sources"]["pipeline.py"] = "different calculation"
    elif change == "dependency":
        old["versions"]["numpy"] = "different dependency"
    elif change == "analyzer":
        old["analyzer"] = "another analyzer"
    elif change == "old_batch":
        old["sources"]["batch.py"] = "unknown older implementation"
    elif change == "old_builder":
        old["sources"]["producer.py"] = "unknown identity implementation"
    elif change == "new_builder":
        current["sources"]["producer.py"] = "unreviewed identity implementation"
    elif change == "unbound":
        old = None
    else:
        old["sources"]["batch.py"] = []
    assert not producer.calculation_producers_compatible(old, current)


def test_builder_source_pin_rejects_an_unreviewed_rule_change():
    source = Path(producer.__file__).read_bytes()
    assert producer._reviewed_checkpoint_builder_hash(source) == hashlib.sha256(source).hexdigest()
    changed = source.replace(b"return previous == {**current,", b"return previous != {**current,")
    assert changed != source
    assert producer._reviewed_checkpoint_builder_hash(changed) is None


@pytest.mark.parametrize("partial", [False, True])
def test_legacy_checkpoint_and_native_arrays_restore_without_refitting(tmp_path, monkeypatch, partial):
    paths = [tmp_path / f"frame_{index}.dat" for index in range(2)]
    for index, path in enumerate(paths):
        path.write_text(str(index))
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    current = producer.producer_identity()
    legacy = _legacy_identity(current)
    stop = threading.Event()

    def analyze(frame):
        if partial:
            stop.set()
        return {"image": np.full((3, 4), int(frame.path.read_text()) + .5),
                "parameters": {"a": .4}, "metrics": {"success": False}}

    with monkeypatch.context() as patch:
        patch.setattr(producer, "producer_identity", lambda: copy.deepcopy(legacy))
        writer = StreamingBatchExporter(output, analyzer_identity="fixture v1")
        run = batch.run_batch(paths, analyze, checkpoint=checkpoint, analyzer_identity="fixture v1",
                              retain_results=False, result_sink=writer.write, cancel_event=stop)
        writer.finalize(run)
        if partial:
            # A later durable checkpoint can lead the last committed native
            # bundle. The second frame must then be retried, despite its fit.
            batch.run_batch(paths, analyze, checkpoint=checkpoint, analyzer_identity="fixture v1")
    old = batch.read_checkpoint(checkpoint)
    # The original v1 pretty representation is still a valid restart input.
    checkpoint.write_text(json.dumps(old, indent=2), encoding="utf-8")
    with np.load(output / "results.npz", allow_pickle=False) as archive:
        expected_arrays = {name: archive[name].copy() for name in archive.files if name != "__metadata__"}
    old_npz_sha = hashlib.sha256((output / "results.npz").read_bytes()).hexdigest()
    old_exporter = json.loads((output / "provenance.json").read_text())["exporter_identity"]
    writer = StreamingBatchExporter(output, resume=True, force=True, analyzer_identity="fixture v1")
    fitted = []

    def retry_missing(frame):
        assert partial and frame.path == paths[1], "storage-only upgrade refitted a delivered frame"
        fitted.append(frame.key)
        return analyze(frame)

    restored = batch.run_batch(paths, retry_missing,
                               checkpoint=checkpoint, resume=True, analyzer_identity="fixture v1",
                               retain_results=False, result_sink=writer.write, restore_frame=writer.can_restore)
    assert restored[0].resumed and restored[1].resumed is (not partial)
    assert len(fitted) == int(partial)
    assert restored.producer_identity == {**current, "analyzer": "fixture v1"}
    writer.finalize(restored)
    with np.load(output / "results.npz", allow_pickle=False) as archive:
        for name, expected in expected_arrays.items():
            assert archive[name].dtype == expected.dtype
            assert archive[name].shape == expected.shape
            assert archive[name].tobytes() == expected.tobytes()
    assert batch.read_checkpoint(checkpoint)["producer_identity"] == restored.producer_identity
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["restored_calculation_producer_identity"] == {**legacy, "analyzer": "fixture v1"}
    if not partial:
        assert hashlib.sha256((output / "results.npz").read_bytes()).hexdigest() == old_npz_sha
        assert provenance["retained_npz_provenance"] == {
            "producer_identity": {**legacy, "analyzer": "fixture v1"},
            "exporter_identity": old_exporter, "sha256": old_npz_sha,
        }


@pytest.mark.parametrize("boundary", ["fsync", "replace"])
def test_real_process_exit_preserves_a_whole_checkpoint_generation(tmp_path, boundary):
    checkpoint = tmp_path / "checkpoint.json"
    prior = {"version": 1, "processed_count": 1, "frames": [{"result": {"a": -0.}}]}
    checkpoint.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    code = """
import os, sys
from butterfly_saxs import batch
boundary = sys.argv[2]
original = getattr(batch.os, boundary)
def stop(*args, **kwargs):
    result = original(*args, **kwargs)
    os._exit(99)
setattr(batch.os, boundary, stop)
batch.write_checkpoint(sys.argv[1], {"version": 1, "processed_count": 2,
    "frames": [{"result": {"a": -0.}}, {"result": {"a": 1.0000000000000002}}]})
"""
    env = {**os.environ, "PYTHONPATH": str(Path(batch.__file__).resolve().parents[1])}
    stopped = subprocess.run([sys.executable, "-c", code, str(checkpoint), boundary],
                             env=env, capture_output=True, timeout=30)
    assert stopped.returncode == 99, stopped.stderr.decode(errors="replace")
    record = batch.read_checkpoint(checkpoint)
    assert record["processed_count"] == (1 if boundary == "fsync" else 2)
    assert len(record["frames"]) == record["processed_count"]
    assert record["frames"][0]["result"]["a"].hex() == (-0.).hex()
    batch.write_checkpoint(checkpoint, record)
    assert batch.read_checkpoint(checkpoint) == record
