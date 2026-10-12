from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os

import numpy as np
import pytest

from butterfly_saxs import batch, producer
from butterfly_saxs.export import StreamingBatchExporter
from butterfly_saxs.output_lock import output_lock


def _sources(tmp_path):
    paths = [tmp_path / f"frame_{index}.dat" for index in range(2)]
    for index, path in enumerate(paths):
        path.write_bytes(str(index).encode())
    return paths


def _result(frame):
    return {"image": np.ones((3, 4)), "parameters": {"a": 0.4}, "metrics": {"success": False}}


def _stream_run(paths, output, checkpoint, *, resume=False):
    writer = StreamingBatchExporter(output, resume=resume, force=resume)
    run = batch.run_batch(paths, _result, checkpoint=checkpoint, resume=resume,
                          result_sink=writer.write, retain_results=False, analyzer_identity="fixture v1",
                          restore_frame=writer.can_restore)
    writer.finalize(run)
    return run, writer


def test_checkpoint_does_not_expand_detector_arrays_before_omitting_them(tmp_path):
    class Detector(np.ndarray):
        def tolist(self):
            pytest.fail("checkpoint expanded an array it immediately omits")

    image = np.arange(12.).reshape(3, 4).view(Detector)
    paths = _sources(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    run = batch.run_batch(paths, lambda _: {"image": image, "parameters": {"a": 0.4}},
                          checkpoint=checkpoint)
    assert run[0].result["image"] is image
    record = json.loads(checkpoint.read_text())["frames"][0]["result"]["image"]
    assert record["array_omitted"] is True
    assert record["min"] == 0. and record["max"] == 11.


def test_noop_stream_resume_preserves_verified_artifacts_including_warnings(tmp_path):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    first, writer = _stream_run(paths, output, checkpoint)
    assert first.frame_results[0].status == "warning"
    before = {path.name: (path.stat().st_mtime_ns, path.read_bytes())
              for path in output.iterdir() if path.is_file() and path.name != ".wingsaxs.lock"}
    resumed, second_writer = _stream_run(paths, output, checkpoint, resume=True)
    assert all(item.resumed for item in resumed)
    after = {path.name: (path.stat().st_mtime_ns, path.read_bytes())
             for path in output.iterdir() if path.is_file() and path.name != ".wingsaxs.lock"}
    assert after == before
    # Repeated cleanup after successful publication must never mean cwd.
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    writer.abort()
    second_writer.abort()
    assert sentinel.read_text() == "keep"
    with ThreadPoolExecutor() as pool:
        assert pool.submit(_can_lock, output).result()


def _can_lock(output):
    with output_lock(output):
        return True


def test_tampered_table_is_regenerated_instead_of_noop_reused(tmp_path):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    target = output / "parameters_long.csv"
    original = target.read_bytes()
    info = target.stat()
    target.write_bytes(original.replace(b"0.4", b"0.9"))
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    _stream_run(paths, output, checkpoint, resume=True)
    assert target.read_bytes() == original


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_rejects_changed_or_unbound_calculation_producer(tmp_path, monkeypatch, legacy):
    paths = _sources(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    monkeypatch.setattr(producer, "producer_identity", lambda: {"source": "one"})
    batch.run_batch(paths, _result, checkpoint=checkpoint)
    if legacy:
        payload = json.loads(checkpoint.read_text())
        del payload["producer_identity"]
        checkpoint.write_text(json.dumps(payload))
    else:
        monkeypatch.setattr(producer, "producer_identity", lambda: {"source": "two"})
    with pytest.raises(ValueError, match="producer/dependency identity"):
        batch.run_batch(paths, lambda _: pytest.fail("mismatched producer fitted a frame"),
                        checkpoint=checkpoint, resume=True)


def test_presentation_edit_does_not_invalidate_calculation_but_dependencies_do(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "__file__", str(tmp_path / "producer.py"))
    numeric = tmp_path / "observables.py"
    numeric.write_text("science=1")
    figure = tmp_path / "report_figures.py"
    figure.write_text("figure=1")
    before = producer.producer_identity()
    before_report = producer.producer_identity(presentation=True)
    figure.write_text("figure=2")
    assert producer.producer_identity() == before
    assert producer.producer_identity(presentation=True) != before_report
    numeric.write_text("science=2")
    assert producer.producer_identity() != before
    monkeypatch.setattr(producer.metadata, "version", lambda _: "changed")
    assert producer.producer_identity()["versions"] != before["versions"]


def test_modified_checkpoint_values_cannot_reuse_unrelated_producing_evidence(tmp_path):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    original = (output / "parameters_long.csv").read_bytes()
    payload = json.loads(checkpoint.read_text())
    payload["frames"][0]["result"]["parameters"]["a"] = 0.9
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Restored checkpoint values"):
        _stream_run(paths, output, checkpoint, resume=True)
    assert (output / "parameters_long.csv").read_bytes() == original


def test_partial_resume_checks_restored_values_before_fitting_remaining_frames(tmp_path):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    cancel = __import__("threading").Event()

    def first_analyzer(frame):
        cancel.set()
        return _result(frame)

    writer = StreamingBatchExporter(output)
    first = batch.run_batch(paths, first_analyzer, checkpoint=checkpoint, analyzer_identity="fixture v1",
                            cancel_event=cancel, result_sink=writer.write, retain_results=False)
    writer.finalize(first)
    original = (output / "parameters_long.csv").read_bytes()
    payload = json.loads(checkpoint.read_text())
    payload["frames"][0]["result"]["parameters"]["a"] = 0.9
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Restored checkpoint values"):
        _stream_run(paths, output, checkpoint, resume=True)
    assert (output / "parameters_long.csv").read_bytes() == original


def test_analyzer_identity_change_invalidates_checkpoint(tmp_path):
    paths = _sources(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    batch.run_batch(paths, _result, checkpoint=checkpoint, analyzer_identity="producer one")
    with pytest.raises(ValueError, match="producer/dependency identity"):
        batch.run_batch(paths, _result, checkpoint=checkpoint, resume=True, analyzer_identity="producer two")


def test_checkpoint_ahead_of_published_npz_retries_only_missing_evidence(tmp_path):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    cancel = __import__("threading").Event()

    def first_analyzer(frame):
        cancel.set()
        return _result(frame)

    writer = StreamingBatchExporter(output)
    first = batch.run_batch(paths, first_analyzer, checkpoint=checkpoint, analyzer_identity="fixture v1",
                            cancel_event=cancel, result_sink=writer.write, retain_results=False)
    writer.finalize(first)
    # Simulate a later process whose checkpoint reached frame two before its
    # staging bundle was published. The prior valid bundle stays available.
    batch.run_batch(paths, _result, checkpoint=checkpoint, analyzer_identity="fixture v1", retain_results=False)
    resumed, _ = _stream_run(paths, output, checkpoint, resume=True)
    assert [item.resumed for item in resumed] == [True, False]
    with np.load(output / "results.npz", allow_pickle=False) as archive:
        assert {"frame_0000__image", "frame_0001__image"} <= set(archive.files)
        assert json.loads(str(archive["__metadata__"]))["artifact_complete"] is True


def test_direct_resume_without_producing_manifest_fails_closed(tmp_path):
    output = tmp_path / "bundle"
    original = batch.FrameFitResult(batch.FrameRef("frame.dat"),
                                    result={"image": np.ones((2, 2)), "parameters": {"a": 0.4}})
    writer = StreamingBatchExporter(output)
    writer.write(original)
    writer.finalize(batch.BatchRunResult([original], total_count=1, processed_count=1))
    before = (output / "results.npz").read_bytes()
    (output / "manifest.json").unlink()
    (output / ".bundle.commit.json").unlink()
    resumed_writer = StreamingBatchExporter(output, resume=True, force=True)
    assert not resumed_writer.can_restore(original.frame, 0)
    resumed = batch.FrameFitResult(original.frame, result=batch._checkpoint_safe(original.result), resumed=True)
    with pytest.raises(ValueError, match="No producing frame proof"):
        resumed_writer.write(resumed)
    assert (output / "results.npz").read_bytes() == before
