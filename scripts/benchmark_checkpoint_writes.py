"""Replay real checkpoint records and measure synchronous checkpoint commits.

The ``prepare`` subcommand extracts frame records from an existing pretty-JSON
v1 checkpoint one record at a time.  The ``run`` subcommand replays a selected
sequence through the production ``write_checkpoint`` function in a fresh
process and includes conversion, serialization, flush, fsync, close, and atomic
replace in every write measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_REPLAY_FRAMES = 24
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
        "file_id": value.st_ino,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(text: str, *, source: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc


def _prefix_metadata(lines: list[str]) -> dict[str, Any]:
    fragment = "".join(lines).rstrip()
    if fragment.endswith(","):
        fragment = fragment[:-1]
    parsed = _load_json(fragment + "\n}", source="checkpoint header")
    if not isinstance(parsed, dict):
        raise ValueError("checkpoint header is not an object")
    return parsed


def _suffix_metadata(lines: list[str]) -> dict[str, Any]:
    body = "".join(lines).rstrip()
    if not body:
        return {}
    parsed = _load_json("{\n" + body + "\n}", source="checkpoint trailer")
    if not isinstance(parsed, dict):
        raise ValueError("checkpoint trailer is not an object")
    return parsed


def _validate_checkpoint_metadata(metadata: Mapping[str, Any], frame_count: int,
                                  expected_frames: int) -> None:
    if metadata.get("version") != 1:
        raise ValueError(f"expected checkpoint version 1, got {metadata.get('version')!r}")
    if frame_count != expected_frames:
        raise ValueError(f"expected {expected_frames} frame records, found {frame_count}")
    for key in ("frame_count", "processed_count", "total_count"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != frame_count:
            raise ValueError(f"checkpoint {key}={value!r} does not match {frame_count} records")
    if metadata.get("cancelled") is True:
        raise ValueError("source checkpoint is marked cancelled")


def prepare_fixture(source_checkpoint: Path, fixture: Path, expected_frames: int = 121) -> Path:
    """Extract frame records with memory bounded to one frame record."""
    source_checkpoint = source_checkpoint.resolve()
    fixture = fixture.resolve()
    metadata_path = fixture.with_name(fixture.name + ".metadata.json")
    if fixture == source_checkpoint or metadata_path == source_checkpoint:
        raise ValueError("fixture output must differ from its source checkpoint")
    if fixture.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite fixture or metadata: {fixture}")
    if expected_frames < 1:
        raise ValueError("expected frame count must be positive")

    fixture.parent.mkdir(parents=True, exist_ok=True)
    source_before = _stat(source_checkpoint)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{fixture.name}.", suffix=".tmp", dir=fixture.parent,
    )
    temporary = Path(temporary_name)
    source_digest = hashlib.sha256()
    fixture_digest = hashlib.sha256()
    count = 0
    phase = "header"
    header_lines: list[str] = []
    trailer_lines: list[str] = []
    record_lines: list[str] | None = None
    root_closed = False

    try:
        with source_checkpoint.open("rb") as source, os.fdopen(fd, "wb") as output:
            for raw_line in source:
                source_digest.update(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("checkpoint is not UTF-8") from exc

                if root_closed:
                    if line.strip():
                        raise ValueError("non-whitespace data follows checkpoint root object")
                    continue

                if phase == "header":
                    if line.rstrip("\r\n") == '  "frames": [':
                        header = _prefix_metadata(header_lines)
                        phase = "frames"
                    else:
                        header_lines.append(line)
                    continue

                if phase == "frames":
                    if record_lines is None:
                        stripped = line.rstrip("\r\n")
                        if stripped in ("  ],", "  ]"):
                            phase = "trailer"
                            continue
                        if not line.strip():
                            continue
                        if stripped != "    {":
                            raise ValueError(f"unexpected checkpoint frame-array line: {stripped[:80]!r}")
                        record_lines = [line]
                        continue

                    record_lines.append(line)
                    stripped = line.rstrip("\r\n")
                    if stripped in ("    },", "    }"):
                        if stripped.endswith(","):
                            record_lines[-1] = line[:len(line.rstrip("\r\n")) - 1] + line[len(line.rstrip("\r\n")):]
                        record = _load_json("".join(record_lines), source=f"frame record {count + 1}")
                        if not isinstance(record, dict):
                            raise ValueError(f"frame record {count + 1} is not an object")
                        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=True).encode("utf-8") + b"\n"
                        output.write(encoded)
                        fixture_digest.update(encoded)
                        count += 1
                        record_lines = None
                    continue

                if phase == "trailer":
                    if line.rstrip("\r\n") == "}":
                        trailer = _suffix_metadata(trailer_lines)
                        overlap = set(header).intersection(trailer)
                        if overlap or "frames" in header or "frames" in trailer:
                            raise ValueError(f"duplicate or misplaced top-level checkpoint keys: {sorted(overlap)}")
                        metadata = {**header, **trailer}
                        _validate_checkpoint_metadata(metadata, count, expected_frames)
                        root_closed = True
                    else:
                        trailer_lines.append(line)
                    continue

            if phase != "trailer" or not root_closed or record_lines is not None:
                raise ValueError("checkpoint ended before all frame records and the root object were complete")
            output.flush()
            os.fsync(output.fileno())

        source_after = _stat(source_checkpoint)
        if source_after != source_before:
            raise RuntimeError("source checkpoint changed while it was being read")

        fixture_size = temporary.stat().st_size
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "fixture_format": "one compact UTF-8 JSON frame object per line",
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": source_digest.hexdigest(),
            "source_checkpoint_size_bytes": source_before["size_bytes"],
            "source_stat_before": source_before,
            "source_stat_after": source_after,
            "record_count": count,
            "checkpoint_metadata": metadata,
            "fixture": str(fixture),
            "fixture_sha256": fixture_digest.hexdigest(),
            "fixture_size_bytes": fixture_size,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        # Publish a complete, fsynced fixture first.  The sidecar is the
        # identity/commit record; incomplete outputs remain detectable.
        os.replace(temporary, fixture)
        _write_json_new(metadata_path, metadata)
        return metadata_path
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _default_indices(record_count: int, requested: int = DEFAULT_REPLAY_FRAMES) -> list[int]:
    """Return ascending zero-based indices spanning the complete sequence."""
    if record_count < 1 or requested < 1:
        raise ValueError("record_count and requested frame count must be positive")
    count = min(record_count, requested)
    if count == 1:
        return [0]
    return [round(index * (record_count - 1) / (count - 1)) for index in range(count)]


def _parse_indices(value: str, record_count: int) -> list[int]:
    try:
        indices = [int(item.strip()) - 1 for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--indices must be comma-separated 1-based integers") from exc
    if not indices or any(index < 0 or index >= record_count for index in indices):
        raise ValueError(f"--indices values must be between 1 and {record_count}")
    if indices != sorted(set(indices)):
        raise ValueError("--indices must be unique and in original ascending order")
    return indices


def _strict_equal(expected: Any, actual: Any, path: str = "$") -> None:
    if type(expected) is not type(actual):
        raise AssertionError(f"type differs at {path}: {type(expected).__name__} != {type(actual).__name__}")
    if isinstance(expected, float):
        if expected.hex() != actual.hex():
            raise AssertionError(f"float.hex differs at {path}: {expected.hex()} != {actual.hex()}")
        return
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"object keys differ at {path}")
        for key in expected:
            _strict_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"array length differs at {path}: {len(expected)} != {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _strict_equal(left, right, f"{path}[{index}]")
        return
    if expected != actual:
        raise AssertionError(f"value differs at {path}: {expected!r} != {actual!r}")


def _windows_io() -> dict[str, int] | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(IoCounters)]
    counters = IoCounters()
    if not kernel32.GetProcessIoCounters(kernel32.GetCurrentProcess(), ctypes.byref(counters)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"read_bytes": counters.ReadTransferCount, "write_bytes": counters.WriteTransferCount,
            "read_operations": counters.ReadOperationCount, "write_operations": counters.WriteOperationCount}


def _instrument_batch_writer(batch: Any) -> tuple[Callable[..., Any], Callable[[], dict[str, float]]]:
    """Time conversion, dumps, fsync, replace, and the inclusive writer call."""
    originals = {
        "json_safe": batch._json_safe,
        "checkpoint_safe": batch._checkpoint_safe,
        "json_dumps": batch.json.dumps,
        "fsync": batch.os.fsync,
        "replace": batch.os.replace,
    }
    current: dict[str, float] = {}

    def wrap(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def measured(*args: Any, **kwargs: Any) -> Any:
            wall = time.perf_counter()
            cpu = time.process_time()
            try:
                return function(*args, **kwargs)
            finally:
                current[name + "_wall_seconds"] = current.get(name + "_wall_seconds", 0.0) + time.perf_counter() - wall
                current[name + "_cpu_seconds"] = current.get(name + "_cpu_seconds", 0.0) + time.process_time() - cpu
        return measured

    batch._json_safe = wrap("json_safe", originals["json_safe"])
    batch._checkpoint_safe = wrap("checkpoint_safe", originals["checkpoint_safe"])
    batch.json.dumps = wrap("json_dumps", originals["json_dumps"])
    batch.os.fsync = wrap("fsync", originals["fsync"])
    batch.os.replace = wrap("replace", originals["replace"])

    def restore() -> dict[str, float]:
        batch._json_safe = originals["json_safe"]
        batch._checkpoint_safe = originals["checkpoint_safe"]
        batch.json.dumps = originals["json_dumps"]
        batch.os.fsync = originals["fsync"]
        batch.os.replace = originals["replace"]
        return current.copy()

    return restore, lambda: current.copy()


def _instrument_run_batch(batch: Any) -> tuple[
        Callable[[], dict[str, float]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure run_batch payload construction and each synchronous checkpoint."""
    restore_writer_stages, get_stages = _instrument_batch_writer(batch)
    original_payload = batch._checkpoint_payload
    original_write = batch.write_checkpoint
    payload_events: list[dict[str, Any]] = []
    write_events: list[dict[str, Any]] = []

    def measured_payload(*args: Any, **kwargs: Any) -> Any:
        stages_before = get_stages()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            return original_payload(*args, **kwargs)
        finally:
            wall = time.perf_counter() - wall_started
            cpu = time.process_time() - cpu_started
            stages_after = get_stages()
            payload_events.append({
                "call_index": len(payload_events) + 1,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "writer_stage_delta": {
                    name: stages_after.get(name, 0.0) - stages_before.get(name, 0.0)
                    for name in stages_after.keys() | stages_before.keys()
                },
            })

    def measured_write(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Any:
        io_before = _windows_io()
        stages_before = get_stages()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        result = original_write(path, payload)
        wall = time.perf_counter() - wall_started
        cpu = time.process_time() - cpu_started
        io_after = _windows_io()
        stages_after = get_stages()
        stage_delta = {
            name: stages_after.get(name, 0.0) - stages_before.get(name, 0.0)
            for name in stages_after.keys() | stages_before.keys()
        }
        size_bytes = Path(result).stat().st_size
        event: dict[str, Any] = {
            "write_index": len(write_events) + 1,
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "file_size_bytes": size_bytes,
            "cumulative_file_bytes": sum(item["file_size_bytes"] for item in write_events) + size_bytes,
            "writer_stages": stage_delta,
        }
        if io_before is not None and io_after is not None:
            event["logical_io"] = {
                "read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
                "write_bytes": io_after["write_bytes"] - io_before["write_bytes"],
                "read_operations": io_after["read_operations"] - io_before["read_operations"],
                "write_operations": io_after["write_operations"] - io_before["write_operations"],
            }
        write_events.append(event)
        return result

    batch._checkpoint_payload = measured_payload
    batch.write_checkpoint = measured_write

    def restore() -> dict[str, float]:
        batch._checkpoint_payload = original_payload
        batch.write_checkpoint = original_write
        return restore_writer_stages()

    return restore, payload_events, write_events


def _verify_batch_replay(source_records: list[dict[str, Any]], actual: Mapping[str, Any],
                         run: Any, expected_mode: str) -> dict[str, Any]:
    """Compare stored scientific results and stable identity/quality fields."""
    actual_records = actual.get("frames")
    if not isinstance(actual_records, list) or len(actual_records) != len(source_records):
        raise AssertionError("batch replay changed the frame-record count")
    stable_fields = (
        "status", "error", "diagnostic", "traceback", "warm_start_from",
    )
    source_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    for index, (expected, observed) in enumerate(zip(source_records, actual_records, strict=True)):
        if not isinstance(observed, dict):
            raise AssertionError(f"batch replay frame record {index + 1} is not an object")
        _strict_equal(expected.get("frame"), observed.get("frame"), f"$.frames[{index}].frame")
        _strict_equal(expected.get("result"), observed.get("result"), f"$.frames[{index}].result")
        for name in stable_fields:
            _strict_equal(expected.get(name), observed.get(name), f"$.frames[{index}].{name}")
        source_counts[str(expected.get("status"))] += 1
        actual_counts[str(observed.get("status"))] += 1

    frame_count = len(source_records)
    if actual.get("mode") != expected_mode or run.mode != expected_mode:
        raise AssertionError("batch replay changed the processing mode")
    for name in ("frame_count", "total_count", "processed_count"):
        if actual.get(name) != frame_count:
            raise AssertionError(f"batch replay {name} does not match {frame_count} selected frames")
    if actual.get("cancelled") is not False or run.cancelled:
        raise AssertionError("batch replay was cancelled")
    if run.total_count != frame_count or run.processed_count != frame_count:
        raise AssertionError("run_batch reported an incomplete frame count")
    if not isinstance(actual.get("input_hash"), str) or not actual["input_hash"]:
        raise AssertionError("run_batch did not produce an input fingerprint")
    if not isinstance(actual.get("config_hash"), str) or not actual["config_hash"]:
        raise AssertionError("run_batch did not produce a configuration fingerprint")
    if actual.get("input_hash") != run.input_hash or actual.get("config_hash") != run.config_hash:
        raise AssertionError("checkpoint fingerprints differ from the run_batch result")
    if actual.get("producer_identity") != run.producer_identity:
        raise AssertionError("checkpoint producer identity differs from the run_batch result")
    if actual.get("selection") != run.selection:
        raise AssertionError("checkpoint frame selection differs from the run_batch result")
    if dict(source_counts) != dict(actual_counts):
        raise AssertionError("batch replay changed frame status counts")
    return {
        "frame_records_equal": True,
        "frame_identity_order_equal": True,
        "frame_result_float_hex_equal": True,
        "stable_status_and_diagnostic_flags_equal": True,
        "source_resumed_count": sum(record.get("resumed") is True for record in source_records),
        "target_resumed_count": sum(record.get("resumed") is True for record in actual_records),
        "operational_fields_excluded": ["elapsed_s", "resumed"],
        "source_status_counts": dict(source_counts),
        "target_status_counts": dict(actual_counts),
        "target_frame_count": frame_count,
        "target_total_count": actual["total_count"],
        "target_processed_count": actual["processed_count"],
        "input_fingerprint_present": True,
        "config_fingerprint_present": True,
        "producer_identity_matches_run": True,
    }


def _run_batch_replay(args: argparse.Namespace, batch: Any, source: Path,
                      identity: Mapping[str, Any], fixture_identity: dict[str, Any],
                      records: list[dict[str, Any]]) -> int:
    metadata = fixture_identity["checkpoint_metadata"]
    mode = metadata.get("mode")
    if mode not in {"independent", "warm_start"}:
        raise ValueError(f"unsupported source batch mode: {mode!r}")
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("fixture checkpoint does not contain a mapping config")
    producer = metadata.get("producer_identity")
    if not isinstance(producer, Mapping):
        raise ValueError("fixture checkpoint does not contain producer identity")

    refs = [batch._as_ref(record["frame"]) for record in records]
    result_by_key: dict[str, Any] = {}
    for record, ref in zip(records, refs, strict=True):
        if ref.key in result_by_key:
            raise ValueError(f"fixture contains duplicate frame identity: {ref.key}")
        result_by_key[ref.key] = record.get("result")

    def replay_analyzer(frame: Any, **_keywords: Any) -> Any:
        # Bind each callback strictly to its own FrameRef identity.  Missing
        # keys fail closed; a neighboring result is never substituted.
        try:
            return result_by_key[frame.key]
        except KeyError as exc:
            raise KeyError(f"no stored result for frame identity {frame.key}") from exc

    target = args.output_dir.resolve() / "checkpoint.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing replay checkpoint: {target}")
    restore, payload_events, writes = _instrument_run_batch(batch)
    io_before = _windows_io()
    production_wall_started = time.perf_counter()
    production_cpu_started = time.process_time()
    try:
        run_result = batch.run_batch(
            refs,
            replay_analyzer,
            mode=mode,
            config=config,
            # A FrameRef sequence supplied as ``frames`` alone is naturally
            # filename-sorted by the public resolver.  Reuse the same refs as
            # an explicit manifest so their saved order selectors remain the
            # ordering authority for this replay.
            manifest=refs,
            checkpoint=target,
            retain_results=False,
            analyzer_identity=producer.get("analyzer"),
        )
        production_wall = time.perf_counter() - production_wall_started
        production_cpu = time.process_time() - production_cpu_started
        io_after = _windows_io()
    finally:
        stage_totals = restore()

    if len(payload_events) != len(records) + 1 or len(writes) != len(records) + 1:
        raise AssertionError(
            f"run_batch emitted {len(payload_events)} payloads and {len(writes)} writes; "
            f"expected {len(records) + 1} of each"
        )
    for write, payload_event in zip(writes, payload_events, strict=True):
        write["write_phase"] = (
            "final_snapshot" if write["write_index"] == len(records) + 1 else "frame_prefix"
        )
        write["payload_construction"] = payload_event

    verification_started = time.perf_counter()
    actual_payload = batch.read_checkpoint(target)
    verification = _verify_batch_replay(records, actual_payload, run_result, mode)
    verification_wall = time.perf_counter() - verification_started
    hash_started = time.perf_counter()
    target_sha = _sha256(target)
    target_hash_wall = time.perf_counter() - hash_started

    source_paths = {str(ref.path.resolve()) for ref in refs}
    metrics = {
        **identity,
        "fixture": fixture_identity,
        "mode": "batch-replay",
        "workflow_scope": "batch.run_batch orchestration and checkpoint replay using stored frame results",
        "scientific_fit_performed": False,
        "report_or_package_performed": False,
        "normal_run_batch_input_and_config_fingerprinting": True,
        "source_processing_mode": mode,
        "selected_record_count": len(records),
        "unique_input_path_count": len(source_paths),
        "checkpoint_path": str(target),
        "write_count": len(writes),
        "expected_write_count": len(records) + 1,
        "checkpoint_payload_count": len(payload_events),
        "checkpoint_payload_wall_seconds_sum": sum(item["wall_seconds"] for item in payload_events),
        "checkpoint_payload_cpu_seconds_sum": sum(item["cpu_seconds"] for item in payload_events),
        "checkpoint_wall_seconds_sum": sum(item["wall_seconds"] for item in writes),
        "checkpoint_cpu_seconds_sum": sum(item["cpu_seconds"] for item in writes),
        "production_wall_seconds": production_wall,
        "production_cpu_seconds": production_cpu,
        "logical_io_total": (None if io_before is None or io_after is None else {
            key: io_after[key] - io_before[key] for key in io_before
        }),
        "cumulative_file_bytes": sum(item["file_size_bytes"] for item in writes),
        "final_checkpoint_size_bytes": target.stat().st_size,
        "final_checkpoint_sha256": target_sha,
        "strict_readback_equivalence": True,
        "readback_verification": verification,
        "readback_equivalence_wall_seconds": verification_wall,
        "final_hash_wall_seconds": target_hash_wall,
        "instrumented_stage_totals": stage_totals,
        "batch_run_result": {
            "input_hash": run_result.input_hash,
            "config_hash": run_result.config_hash,
            "producer_identity": run_result.producer_identity,
            "processed_count": run_result.processed_count,
            "total_count": run_result.total_count,
            "cancelled": run_result.cancelled,
            "elapsed_s": run_result.elapsed_s,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_new(args.worker_metrics, metrics)
    with args.worker_measurements.open("x", encoding="utf-8", newline="\n") as handle:
        for item in writes:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


def _read_fixture(fixture: Path, metadata_path: Path, selected: list[int]) -> tuple[
        list[dict[str, Any]], dict[str, Any]]:
    identity = json.loads(metadata_path.read_text(encoding="utf-8"))
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported fixture metadata schema")
    if Path(identity.get("fixture", "")).resolve() != fixture.resolve():
        raise ValueError("fixture metadata points to a different fixture path")
    if not isinstance(identity.get("checkpoint_metadata"), dict):
        raise ValueError("fixture metadata is missing checkpoint metadata")
    before = _stat(fixture)
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    selected_set = set(selected)
    count = 0
    started = time.perf_counter()
    with fixture.open("rb") as handle:
        for count, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if raw_line in (b"\n", b"\r\n"):
                raise ValueError(f"blank fixture record at line {count}")
            if count - 1 in selected_set:
                value = _load_json(raw_line.decode("utf-8"), source=f"fixture record {count}")
                if not isinstance(value, dict):
                    raise ValueError(f"fixture record {count} is not an object")
                records.append(value)
    elapsed = time.perf_counter() - started
    after = _stat(fixture)
    if before != after:
        raise RuntimeError("fixture changed while it was being read")
    if count != identity.get("record_count"):
        raise ValueError(f"fixture contains {count} records; metadata says {identity.get('record_count')}")
    if before["size_bytes"] != identity.get("fixture_size_bytes"):
        raise ValueError("fixture byte size does not match its metadata")
    if digest.hexdigest() != identity.get("fixture_sha256"):
        raise ValueError("fixture SHA-256 does not match its metadata")
    if len(records) != len(selected):
        raise ValueError("not all requested records were read")
    identity["observed_fixture_sha256"] = digest.hexdigest()
    identity["observed_fixture_size_bytes"] = before["size_bytes"]
    identity["fixture_load_wall_seconds"] = elapsed
    identity["fixture_load_bytes"] = before["size_bytes"]
    identity["selected_indices_1based"] = [index + 1 for index in selected]
    return records, identity


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in ("numpy", "scipy", "matplotlib", "fabio", "pyFAI", "tifffile", "PyYAML"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _worker(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    sys.path.insert(0, str(source / "src"))
    from butterfly_saxs import batch

    metadata_path = args.fixture.with_name(args.fixture.name + ".metadata.json")
    fixture_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    record_count = fixture_meta.get("record_count")
    if not isinstance(record_count, int) or record_count < 1:
        raise ValueError("invalid fixture record count")
    if args.mode == "single-snapshot":
        if args.frames is not None or args.indices:
            raise ValueError("single-snapshot mode always uses all fixture records; omit --frames/--indices")
        selected = list(range(record_count))
    elif args.indices:
        selected = _parse_indices(args.indices, record_count)
    elif args.frames is not None:
        if args.frames < 1 or args.frames > record_count:
            raise ValueError(f"--frames must be between 1 and {record_count}")
        selected = list(range(args.frames))
    else:
        selected = _default_indices(record_count)

    identity = {
        "source_path": str(source),
        "source_revision_label": args.revision,
        "source_sha256": {
            path.relative_to(source / "src").as_posix(): _sha256(path)
            for path in sorted((source / "src").rglob("*.py"))
        },
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "dependencies": _versions(),
    }
    records, fixture_identity = _read_fixture(args.fixture.resolve(), metadata_path.resolve(), selected)
    checkpoint_metadata = fixture_identity["checkpoint_metadata"]
    if args.mode == "single-snapshot" and len(records) != record_count:
        raise ValueError("single-snapshot mode requires every checkpoint frame")

    target = args.output_dir.resolve() / "checkpoint.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing replay checkpoint: {target}")
    if args.mode == "batch-replay":
        return _run_batch_replay(args, batch, source, identity, fixture_identity, records)

    payload_metadata = dict(checkpoint_metadata)
    restore, get_stages = _instrument_batch_writer(batch)
    writes: list[dict[str, Any]] = []
    production_wall_start = time.perf_counter()
    production_cpu_start = time.process_time()
    logical_before = _windows_io()
    try:
        if args.mode == "single-snapshot":
            prefixes = [(len(records), "single_snapshot")]
        else:
            prefixes = [(index, "frame_prefix") for index in range(1, len(records) + 1)]
            prefixes.append((len(records), "final_snapshot"))
        for completed, write_phase in prefixes:
            payload = dict(payload_metadata)
            payload["frames"] = records[:completed]
            payload["processed_count"] = completed
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            payload["elapsed_s"] = time.perf_counter() - production_wall_start
            # The normal writer returns only after flush, fsync, close, and
            # replace; per-write time therefore ends at the durable boundary.
            io_before = _windows_io()
            stage_before = get_stages()
            write_cpu_start = time.process_time()
            write_wall_start = time.perf_counter()
            batch.write_checkpoint(target, payload)
            write_wall = time.perf_counter() - write_wall_start
            write_cpu = time.process_time() - write_cpu_start
            io_after = _windows_io()
            stages_after = get_stages()
            stages = {key: stages_after.get(key, 0.0) - stage_before.get(key, 0.0)
                      for key in stages_after.keys() | stage_before.keys()}
            stage_wall = sum(value for key, value in stages.items() if key.endswith("_wall_seconds"))
            size_bytes = target.stat().st_size
            item: dict[str, Any] = {
                "write_index": len(writes) + 1,
                "write_phase": write_phase,
                "selected_frames_committed": completed,
                "last_source_record_index_1based": fixture_identity["selected_indices_1based"][completed - 1],
                "wall_seconds": write_wall,
                "cpu_seconds": write_cpu,
                "other_write_close_directory_seconds": max(0.0, write_wall - stage_wall),
                "file_size_bytes": size_bytes,
                "cumulative_file_bytes": sum(entry["file_size_bytes"] for entry in writes) + size_bytes,
                "stages": stages,
            }
            if io_before is not None and io_after is not None:
                item["logical_io"] = {
                    "read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
                    "write_bytes": io_after["write_bytes"] - io_before["write_bytes"],
                    "read_operations": io_after["read_operations"] - io_before["read_operations"],
                    "write_operations": io_after["write_operations"] - io_before["write_operations"],
                }
            writes.append(item)
        production_wall = time.perf_counter() - production_wall_start
        production_cpu = time.process_time() - production_cpu_start
        logical_after = _windows_io()
    finally:
        stage_totals = restore()

    verify_started = time.perf_counter()
    verified_payload = batch.read_checkpoint(target)
    _strict_equal(payload, verified_payload)
    equivalence_wall = time.perf_counter() - verify_started
    hash_started = time.perf_counter()
    target_sha = _sha256(target)
    target_hash_wall = time.perf_counter() - hash_started

    metrics = {
        **identity,
        "fixture": fixture_identity,
        "mode": args.mode,
        "selected_record_count": len(records),
        "checkpoint_path": str(target),
        "write_count": len(writes),
        "expected_write_count": 1 if args.mode == "single-snapshot" else len(records) + 1,
        "production_wall_seconds": production_wall,
        "production_cpu_seconds": production_cpu,
        "checkpoint_wall_seconds_sum": sum(item["wall_seconds"] for item in writes),
        "checkpoint_cpu_seconds_sum": sum(item["cpu_seconds"] for item in writes),
        "logical_io_total": (None if logical_before is None or logical_after is None else {
            key: logical_after[key] - logical_before[key] for key in logical_before
        }),
        "cumulative_file_bytes": sum(item["file_size_bytes"] for item in writes),
        "final_checkpoint_size_bytes": target.stat().st_size,
        "final_checkpoint_sha256": target_sha,
        "strict_readback_equivalence": True,
        "float_comparison": "float.hex() at every nested value",
        "readback_equivalence_wall_seconds": equivalence_wall,
        "final_hash_wall_seconds": target_hash_wall,
        "instrumented_stage_totals": stage_totals,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_new(args.worker_metrics, metrics)
    with args.worker_measurements.open("x", encoding="utf-8", newline="\n") as handle:
        for item in writes:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


def _run(args: argparse.Namespace) -> int:
    fixture = args.fixture.resolve()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite replay evidence: {output_dir}")
    if not (source / "src" / "butterfly_saxs" / "batch.py").is_file():
        raise FileNotFoundError(f"--source must contain src/butterfly_saxs/batch.py: {source}")
    metadata_path = fixture.with_name(fixture.name + ".metadata.json")
    if not fixture.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"fixture and metadata sidecar are required: {fixture}")
    if args.mode == "single-snapshot" and (args.frames is not None or args.indices):
        raise ValueError("single-snapshot mode always uses all fixture records; omit --frames/--indices")
    output_dir.mkdir(parents=True, exist_ok=False)
    worker_metrics = output_dir / ".worker_metrics.json"
    worker_measurements = output_dir / ".worker_measurements.jsonl"
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0", MPLBACKEND="Agg")
    for name in THREAD_VARIABLES:
        environment[name] = "1"
    command = [
        sys.executable, str(Path(__file__).resolve()), "run",
        "--worker", "--fixture", str(fixture), "--source", str(source),
        "--output-dir", str(output_dir), "--mode", args.mode,
        "--worker-metrics", str(worker_metrics), "--worker-measurements", str(worker_measurements),
        "--revision", args.revision or "",
    ]
    if args.indices:
        command.extend(("--indices", args.indices))
    if args.frames is not None:
        command.extend(("--frames", str(args.frames)))
    outer_started = time.perf_counter()
    with (output_dir / "stdout.log").open("x", encoding="utf-8") as stdout, \
            (output_dir / "stderr.log").open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=False)
    outer_wall = time.perf_counter() - outer_started
    if not worker_metrics.is_file():
        raise RuntimeError(f"checkpoint replay worker failed ({completed.returncode}); see {output_dir / 'stderr.log'}")
    metrics = json.loads(worker_metrics.read_text(encoding="utf-8"))
    metrics.update(outer_process_wall_seconds=outer_wall, exit_code=completed.returncode,
                   command=command, run_mode=args.mode)
    report_path = output_dir / "report.json"
    measurements_path = output_dir / "measurements.jsonl"
    _write_json_new(report_path, metrics)
    os.replace(worker_measurements, measurements_path)
    worker_metrics.unlink()
    print(json.dumps({"report": str(report_path), "measurements": str(measurements_path),
                      "exit_code": completed.returncode}, ensure_ascii=False))
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="stream a v1 checkpoint into a compact JSONL fixture")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--fixture", type=Path, required=True)
    prepare.add_argument("--expected-frames", type=int, default=121)
    run = subparsers.add_parser("run", help="replay real records through the synchronous production writer")
    run.add_argument("--fixture", type=Path, required=True)
    run.add_argument("--source", type=Path, required=True, help="frozen source tree containing src/butterfly_saxs/batch.py")
    run.add_argument("--output-dir", type=Path, required=True, help="new, case-specific evidence directory")
    run.add_argument("--mode", choices=("prefix", "single-snapshot", "batch-replay"), default="prefix")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--frames", type=int, help="use the first N records in prefix or batch-replay mode")
    selection.add_argument("--indices", help="1-based ascending frame record indices, e.g. 1,6,12")
    run.add_argument("--revision", help="descriptive source revision label; file hashes are authoritative")
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--worker-metrics", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--worker-measurements", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            metadata = prepare_fixture(args.checkpoint, args.fixture, args.expected_frames)
            print(json.dumps({"fixture": str(args.fixture.resolve()), "metadata": str(metadata)}, ensure_ascii=False))
            return 0
        if args.worker:
            if args.worker_metrics is None or args.worker_measurements is None:
                parser.error("worker output paths are required")
            return _worker(args)
        return _run(args)
    except (AssertionError, FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"checkpoint benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
