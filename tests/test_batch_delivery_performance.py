from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os

import numpy as np
import pytest

from butterfly_saxs import batch, export, producer
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
    writer = StreamingBatchExporter(output, resume=resume, force=resume, analyzer_identity="fixture v1")
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


def test_cli_wrapper_and_exporter_changes_bind_the_correct_reuse_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "__file__", str(tmp_path / "producer.py"))
    wrapper, writer = tmp_path / "cli.py", tmp_path / "export.py"
    wrapper.write_text("dataset='original'")
    writer.write_text("schema='original'")
    calculation, exported = producer.producer_identity(), producer.exporter_identity()
    writer.write_text("schema='changed'")
    assert producer.producer_identity() == calculation
    assert producer.exporter_identity() != exported
    wrapper.write_text("dataset='changed'")
    assert producer.producer_identity() != calculation


def test_writer_change_refreshes_artifacts_from_verified_fits_without_refitting(tmp_path, monkeypatch):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    pixels = (output / "results.npz").read_bytes()
    original = export._write_evolution
    rendered = []

    def render(*args):
        rendered.append(args[0])
        return original(*args)

    monkeypatch.setattr(export, "exporter_identity", lambda: {"writer": "changed"})
    monkeypatch.setattr(export, "_write_evolution", render)
    monkeypatch.setattr(__import__(__name__, fromlist=["_result"]), "_result",
                        lambda _: pytest.fail("writer change repeated a fit"))
    run, _ = _stream_run(paths, output, checkpoint, resume=True)
    assert all(item.resumed for item in run)
    assert len(rendered) == 1
    assert (output / "results.npz").read_bytes() == pixels
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["exporter_identity"] == {"writer": "changed"}


def test_current_checkpoint_cannot_retain_another_producers_pixels_with_equal_summaries(tmp_path, monkeypatch):
    paths = _sources(tmp_path)
    image = np.arange(12.).reshape(3, 4)
    monkeypatch.setattr(producer, "producer_identity", lambda: {"sources": {"science": "old"}, "versions": {}})
    monkeypatch.setitem(globals(), "_result", lambda _: {"image": image, "parameters": {"a": 0.4}})
    output = tmp_path / "old_bundle"
    _stream_run(paths, output, tmp_path / "old_checkpoint.json")
    monkeypatch.setattr(producer, "producer_identity", lambda: {"sources": {"science": "current"}, "versions": {}})
    changed_image = image[:, ::-1].copy()
    assert batch._checkpoint_safe(changed_image) == batch._checkpoint_safe(image)
    monkeypatch.setitem(globals(), "_result", lambda _: {"image": changed_image, "parameters": {"a": 0.4}})
    current_checkpoint = tmp_path / "current_checkpoint.json"
    _stream_run(paths, tmp_path / "current_bundle", current_checkpoint)
    resumed, _ = _stream_run(paths, output, current_checkpoint, resume=True)
    assert not any(item.resumed for item in resumed)
    with np.load(output / "results.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["frame_0000__image"], changed_image)
        np.testing.assert_array_equal(arrays["frame_0001__image"], changed_image)


def test_direct_writer_rejects_cross_producer_checkpoint_before_publication(tmp_path):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    original = (output / "results.npz").read_bytes()
    writer = StreamingBatchExporter(output, resume=True, force=True)
    run = batch.run_batch(paths, _result, checkpoint=checkpoint, resume=True,
                          analyzer_identity="fixture v1", result_sink=writer.write, retain_results=False)
    run.producer_identity = {"source": "another producer", "analyzer": "fixture v1"}
    with pytest.raises(ValueError, match="Previous bundle calculation producer"):
        writer.finalize(run)
    assert (output / "results.npz").read_bytes() == original


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
    with pytest.raises(ValueError, match="No verified producing bundle marker"):
        resumed_writer.write(resumed)
    assert (output / "results.npz").read_bytes() == before


def test_missing_marker_cannot_promote_old_pixels_to_a_new_generation(tmp_path):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    (output / ".bundle.commit.json").unlink()
    before = (output / "results.npz").read_bytes()
    writer = StreamingBatchExporter(output, resume=True, force=True)
    assert not writer.can_restore(batch.FrameRef(paths[0]), 0)
    restored = batch.read_checkpoint(checkpoint)["frames"][0]
    item = batch.FrameFitResult.from_record(restored)
    item.resumed = True
    with pytest.raises(ValueError, match="No verified producing bundle marker"):
        writer.write(item)
    assert (output / "results.npz").read_bytes() == before
    assert not (output / ".bundle.commit.json").exists()


def test_partial_resume_preserves_rich_metadata_and_full_fit_vectors(tmp_path, monkeypatch):
    from enum import IntEnum
    import threading

    class Orientation(IntEnum):
        TOP = 1

    paths = _sources(tmp_path)
    result = {"image": np.ones((3, 4)), "parameters": {"a": 0.4},
              "analysis_arrays": {"fit_valid_mask": np.ones((3, 4), dtype=bool)},
              "full2d": {"sampled_indices": np.arange(12)},
              "qmap": {"metadata": {"orientation": Orientation.TOP}},
              "ellipse_fit": {"a": 0.4, "residual": np.array([0.1, 0.2]), "covariance": np.eye(2),
                              "metadata": {"enum_type": "literal", "name": "TOP", "value": 1},
                              "orientation": Orientation.TOP}}
    monkeypatch.setitem(globals(), "_result", lambda _: result)
    cancel = threading.Event()
    writer = StreamingBatchExporter(tmp_path / "bundle", analyzer_identity="fixture v1")

    def analyzer(frame):
        cancel.set()
        return result

    checkpoint = tmp_path / "checkpoint.json"
    run = batch.run_batch(paths, analyzer, checkpoint=checkpoint, cancel_event=cancel,
                          analyzer_identity="fixture v1", result_sink=writer.write, retain_results=False)
    writer.finalize(run)
    output = tmp_path / "bundle"
    detail = json.loads((output / "frame_details.jsonl").read_text().splitlines()[0])
    ellipse = json.loads((output / "ellipse_fit.jsonl").read_text().splitlines()[0])
    original_load = np.load

    def load(*args, **kwargs):
        assert "analysis_arrays" not in str(getattr(args[0], "name", "")), "unneeded detector mask was materialized"
        assert "sampled_indices" not in str(getattr(args[0], "name", "")), "unneeded full2d indices were materialized"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    resumed, _ = _stream_run(paths, output, checkpoint, resume=True)
    assert [item.resumed for item in resumed] == [True, False]
    restored_detail = json.loads((output / "frame_details.jsonl").read_text().splitlines()[0])
    assert restored_detail["result"] == detail["result"]
    assert restored_detail["qmap_metadata"] == detail["qmap_metadata"]
    restored_ellipse = json.loads((output / "ellipse_fit.jsonl").read_text().splitlines()[0])
    for row in (ellipse, restored_ellipse):
        row.pop("elapsed_s")
        row.pop("resumed")
    assert restored_ellipse == ellipse
    with np.load(output / "results.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["frame_0000__analysis_arrays__fit_valid_mask"], result["analysis_arrays"]["fit_valid_mask"])


@pytest.mark.parametrize("revision_available", [False, True])
def test_original_detail_mutation_during_restore_never_publishes_a_new_generation(tmp_path, monkeypatch, revision_available):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    pixels = (output / "results.npz").read_bytes()
    marker = (output / ".bundle.commit.json").read_bytes()
    if not revision_available:
        original_version = export.file_version
        monkeypatch.setattr(export, "file_version", lambda path, info: (*original_version(path, info)[:-1], None))
    writer = StreamingBatchExporter(output, resume=True, force=True, analyzer_identity="fixture v1")

    def sink(item):
        writer.write(item)
        if len(writer._compact_results) == 1:
            target = output / "frame_details.jsonl"
            info = target.stat()
            target.write_bytes(target.read_bytes().replace(b'"a":0.4', b'"a":0.9'))
            os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))

    run = batch.run_batch(paths, _result, checkpoint=checkpoint, resume=True, analyzer_identity="fixture v1",
                          result_sink=sink, retain_results=False, restore_frame=writer.can_restore)
    with pytest.raises(ValueError, match="Producing frame details changed"):
        writer.finalize(run)
    assert (output / "results.npz").read_bytes() == pixels
    assert (output / ".bundle.commit.json").read_bytes() == marker


@pytest.mark.parametrize("during_hash", [False, True])
def test_cancelled_resume_preserves_delivered_generation_and_checkpoint_fits(tmp_path, during_hash):
    import threading

    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    before = {path.name: path.read_bytes() for path in output.iterdir()
              if path.is_file() and path.name != ".wingsaxs.lock"}
    prior = json.loads(checkpoint.read_text())
    cancel = threading.Event()
    if during_hash:
        cancel.set()

    def progress(update):
        if update.get("phase") == "analyze":
            cancel.set()

    writer = StreamingBatchExporter(output, resume=True, force=True, analyzer_identity="fixture v1")
    run = batch.run_batch(paths, lambda _: pytest.fail("canceled resume fitted a frame"),
                          checkpoint=checkpoint, resume=True, analyzer_identity="fixture v1",
                          result_sink=writer.write, retain_results=False, restore_frame=writer.can_restore,
                          cancel_event=cancel, progress=progress)
    writer.finalize(run)
    assert run.cancelled
    assert {path.name: path.read_bytes() for path in output.iterdir()
            if path.is_file() and path.name != ".wingsaxs.lock"} == before
    current = json.loads(checkpoint.read_text())
    assert [item["result"] for item in current["frames"]] == [item["result"] for item in prior["frames"]]
    assert all(item["status"] == "warning" for item in current["frames"])
    resumed, _ = _stream_run(paths, output, checkpoint, resume=True)
    assert all(item.resumed for item in resumed)


def test_cancelled_resume_after_retry_keeps_prior_tail_and_published_arrays(tmp_path):
    import threading

    paths = _sources(tmp_path)
    paths.append(tmp_path / "frame_2.dat")
    paths[-1].write_bytes(b"2")
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    before = {path.name: path.read_bytes() for path in output.iterdir()
              if path.is_file() and path.name != ".wingsaxs.lock"}
    payload = json.loads(checkpoint.read_text())
    payload["frames"][1].update(status="failed", error="RuntimeError: retry", traceback="injected exception")
    checkpoint.write_text(json.dumps(payload))
    cancel = threading.Event()

    def analyzer(frame):
        cancel.set()
        return _result(frame)

    writer = StreamingBatchExporter(output, resume=True, force=True, analyzer_identity="fixture v1")
    run = batch.run_batch(paths, analyzer, checkpoint=checkpoint, resume=True, analyzer_identity="fixture v1",
                          result_sink=writer.write, retain_results=False, restore_frame=writer.can_restore,
                          cancel_event=cancel)
    writer.finalize(run)
    assert run.cancelled
    assert [item.resumed for item in run] == [True, False, True]
    assert all(item["result"] is not None for item in json.loads(checkpoint.read_text())["frames"])
    assert {path.name: path.read_bytes() for path in output.iterdir()
            if path.is_file() and path.name != ".wingsaxs.lock"} == before


def test_detail_mutation_after_copy_started_is_rejected_before_publication(tmp_path, monkeypatch):
    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    before = (output / "results.npz").read_bytes()
    marker = (output / ".bundle.commit.json").read_bytes()
    monkeypatch.setattr(export, "exporter_identity", lambda: {"writer": "regenerate"})
    original = export._write_evolution

    def changed(*args):
        target = output / "frame_details.jsonl"
        target.write_bytes(target.read_bytes().replace(b'"a":0.4', b'"a":0.9'))
        return original(*args)

    monkeypatch.setattr(export, "_write_evolution", changed)
    with pytest.raises(ValueError, match="Producing frame details changed"):
        _stream_run(paths, output, checkpoint, resume=True)
    assert (output / "results.npz").read_bytes() == before
    assert (output / ".bundle.commit.json").read_bytes() == marker


def test_retained_array_copy_failure_preserves_prior_generation_and_releases_lock(tmp_path, monkeypatch):
    import threading
    import zipfile

    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    cancel = threading.Event()
    writer = StreamingBatchExporter(output, analyzer_identity="fixture v1")

    def analyzer(frame):
        cancel.set()
        return _result(frame)

    run = batch.run_batch(paths, analyzer, checkpoint=checkpoint, analyzer_identity="fixture v1",
                          cancel_event=cancel, result_sink=writer.write, retain_results=False)
    writer.finalize(run)
    before = (output / "results.npz").read_bytes()
    marker = (output / ".bundle.commit.json").read_bytes()
    original = zipfile.ZipFile.open

    def failing(archive, name, mode="r", *args, **kwargs):
        if mode == "w" and name == "frame_0000__image.npy":
            raise OSError("injected retained-member copy failure")
        return original(archive, name, mode, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(zipfile.ZipFile, "open", failing)
        with pytest.raises(OSError, match="retained-member copy failure"):
            _stream_run(paths, output, checkpoint, resume=True)
    assert (output / "results.npz").read_bytes() == before
    assert (output / ".bundle.commit.json").read_bytes() == marker
    with ThreadPoolExecutor() as pool:
        assert pool.submit(_can_lock, output).result()
    resumed, _ = _stream_run(paths, output, checkpoint, resume=True)
    assert [item.resumed for item in resumed] == [True, False]


def test_cancel_and_durable_progress_preserve_checkpoint_ahead_of_native_arrays(tmp_path, monkeypatch):
    import threading

    paths = _sources(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    batch.run_batch(paths, _result, checkpoint=checkpoint, analyzer_identity="fixture v1")
    before = json.loads(checkpoint.read_text())
    snapshots = []
    original_write = batch.write_checkpoint

    def saved(path, payload):
        snapshots.append(json.loads(json.dumps(payload)))
        return original_write(path, payload)

    monkeypatch.setattr(batch, "write_checkpoint", saved)
    cancel = threading.Event()

    def analyzer(frame):
        cancel.set()
        return _result(frame)

    run = batch.run_batch(paths, analyzer, checkpoint=checkpoint, resume=True, analyzer_identity="fixture v1",
                          restore_frame=lambda frame, index: False, cancel_event=cancel)
    assert run.cancelled and run[1].status == "skipped"
    assert len(snapshots) == 2
    for snapshot in snapshots:
        assert len(snapshot["frames"]) == 2
        assert snapshot["processed_count"] == 2
        assert [row["result"] for row in snapshot["frames"]] == [row["result"] for row in before["frames"]]
        assert all(row["status"] == "warning" for row in snapshot["frames"])


def test_cancelled_unbound_resume_preserves_old_files_without_promoting_them(tmp_path):
    import threading

    paths = _sources(tmp_path)
    output, checkpoint = tmp_path / "bundle", tmp_path / "checkpoint.json"
    _stream_run(paths, output, checkpoint)
    (output / ".bundle.commit.json").unlink()
    before = {path.name: path.read_bytes() for path in output.iterdir()
              if path.is_file() and path.name != ".wingsaxs.lock"}
    cancel = threading.Event()

    def progress(update):
        if update.get("phase") == "analyze":
            cancel.set()

    writer = StreamingBatchExporter(output, resume=True, force=True, analyzer_identity="fixture v1")
    run = batch.run_batch(paths, lambda _: pytest.fail("canceled resume fitted a frame"),
                          checkpoint=checkpoint, resume=True, analyzer_identity="fixture v1",
                          result_sink=writer.write, retain_results=False, restore_frame=writer.can_restore,
                          cancel_event=cancel, progress=progress)
    writer.finalize(run)
    assert run.cancelled and all(item.status == "skipped" for item in run)
    assert not (output / ".bundle.commit.json").exists()
    assert {path.name: path.read_bytes() for path in output.iterdir()
            if path.is_file() and path.name != ".wingsaxs.lock"} == before
    assert all(row["result"] is not None for row in json.loads(checkpoint.read_text())["frames"])
