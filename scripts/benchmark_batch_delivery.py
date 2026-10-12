"""Time the real batch -> report -> package CLI in a fresh Python process.

Supply an existing manifest/config; raw data is never created or rewritten.
Metrics and diagnostic profiles live beside, rather than inside, the bundle.
"""
from __future__ import annotations

import argparse
import cProfile
import ctypes
from ctypes import wintypes
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _resources() -> dict:
    if sys.platform != "win32":
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)}

    class Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage",
            )
        ]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    ctypes.windll.kernel32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(IO)]
    process = ctypes.windll.kernel32.GetCurrentProcess()
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    counters = IO()
    ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(memory), memory.cb)
    ctypes.windll.kernel32.GetProcessIoCounters(process, ctypes.byref(counters))
    return {"peak_rss_bytes": memory.PeakWorkingSetSize,
            "io_read_bytes": counters.ReadTransferCount, "io_write_bytes": counters.WriteTransferCount}


def _worker(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(args.source.resolve() / "src"))
    from butterfly_saxs import batch, delivery, export, pipeline, report
    from butterfly_saxs import lamellar_report_figures, report_figures, report_measurements
    from butterfly_saxs.cli import main

    if args.hard_stop_after is not None:
        original_checkpoint = batch.write_checkpoint

        @functools.wraps(original_checkpoint)
        def crash_after_checkpoint(path, payload):
            result = original_checkpoint(path, payload)
            completed = sum(item.get("status") != "skipped" for item in payload.get("frames", ()))
            if completed >= args.hard_stop_after and not payload.get("cancelled"):
                # Real abrupt exit: no Python finally blocks, ZIP close or
                # publication. The prior committed bundle must survive.
                record = {"injected_abrupt_exit": True, "phase": "after_durable_checkpoint_before_bundle_publish",
                          "processed_frames": completed, "checkpoint": str(path), "resources": _resources(),
                          "source": str(args.source.resolve()), "revision_label": args.revision,
                          "manifest_sha256": _digest(args.manifest), "config_sha256": _digest(args.config)}
                args.metrics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                os._exit(99)
            return result

        batch.write_checkpoint = crash_after_checkpoint

    if args.stop_after is not None:
        original_batch = batch.run_batch

        @functools.wraps(original_batch)
        def interrupted(*positional, **keywords):
            event = threading.Event()
            callback = keywords.get("progress")

            def progress(update):
                if callback is not None:
                    callback(update)
                if update.get("phase") == "analyze" and update.get("completed", 0) >= args.stop_after:
                    event.set()

            keywords.update(cancel_event=event, progress=progress)
            return original_batch(*positional, **keywords)

        batch.run_batch = interrupted

    stages: dict[str, dict] = {}

    def instrument(owner, name: str, label: str) -> None:
        function = getattr(owner, name)

        @functools.wraps(function)
        def measured(*positional, **keywords):
            resources = _resources()
            cpu = time.process_time()
            started = time.perf_counter()
            try:
                return function(*positional, **keywords)
            finally:
                elapsed = time.perf_counter() - started
                cpu_elapsed = time.process_time() - cpu
                after = _resources()
                entry = stages.setdefault(label, {"calls": 0, "seconds": 0.0, "cpu_seconds": 0.0,
                                                  "io_read_bytes": 0, "io_write_bytes": 0})
                entry["calls"] += 1
                entry["seconds"] += elapsed
                entry["cpu_seconds"] += cpu_elapsed
                for name in ("io_read_bytes", "io_write_bytes"):
                    entry[name] += after.get(name, 0) - resources.get(name, 0)

        setattr(owner, name, measured)

    for owner, name, label in (
        (pipeline, "analyze_frame", "analysis"),
        (batch, "input_fingerprint", "input_identity"),
        (batch, "_checkpoint_payload", "checkpoint_payload"),
        (batch, "write_checkpoint", "checkpoint_write"),
        (export.StreamingBatchExporter, "write", "frame_export"),
        (export.StreamingBatchExporter, "finalize", "export_finalize"),
        (report, "build_analysis_report", "report"),
        (report, "_source_state", "report_source_validation"),
        (report_measurements, "measure_frame", "report_measurement"),
        (report_figures, "render_frame_report", "report_frame_figures"),
        (report_figures, "render_sequence_report", "report_sequence_figures"),
        (lamellar_report_figures, "render_lamellar_report_frame", "report_lamellar_figures"),
        (lamellar_report_figures, "render_lamellar_sequence_report", "report_lamellar_sequence"),
        (delivery, "package_batch", "package"),
        (delivery, "_write_source", "package_members"),
    ):
        instrument(owner, name, label)
    command = ["batch", "--manifest", str(args.manifest.resolve()), "--config", str(args.config.resolve()),
               "--output", str(args.output.resolve()), "--stream", "--checkpoint",
               str(args.output.resolve() / "checkpoint.json"), "--report", "--package"]
    if args.unattended:
        command += ["--unattended", str(args.unattended.resolve())]
    if args.resume:
        command.append("--resume")
    profiler = cProfile.Profile() if args.profile else None
    started_cpu = time.process_time()
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        code = main(command)
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(args.metrics.with_suffix(".prof")))
        metrics = {"cli_wall_seconds": time.perf_counter() - started,
                   "cpu_seconds": time.process_time() - started_cpu, "stages_inclusive": stages,
                   "process_cpu_seconds": time.process_time(), "cpu_logical_cores": os.cpu_count(),
                   "resources": _resources(), "command": ["python", "-m", "butterfly_saxs", *command],
                   "python": sys.version, "platform": platform.platform(),
                   "manifest_sha256": _digest(args.manifest), "config_sha256": _digest(args.config),
                   "source": str(args.source.resolve()), "resume": args.resume, "profiled": args.profile,
                   "revision_label": args.revision, "stop_after_frames": args.stop_after,
                   "source_sha256": {str(path.relative_to(args.source / "src")): _digest(path)
                                     for path in sorted((args.source / "src").rglob("*.py"))},
                   "thread_environment": {name: os.environ.get(name) for name in (
                       "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unattended", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Diagnostic only; do not compare to an unprofiled run")
    parser.add_argument("--revision", help="Descriptive revision label; actual source hashes are also recorded")
    parser.add_argument("--label", default="", help="Optional suffix to retain repeated resume measurements")
    parser.add_argument("--stop-after", type=int, help="Exercise real cooperative cancellation after N frames")
    parser.add_argument("--hard-stop-after", type=int, help="Abruptly exit after the Nth durable checkpoint, before publishing arrays (fault test only)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metrics", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be >= 1")
    if args.hard_stop_after is not None and (args.hard_stop_after < 1 or args.stop_after is not None):
        parser.error("--hard-stop-after must be >= 1 and cannot be combined with --stop-after")
    if args.label and (Path(args.label).name != args.label or "/" in args.label or "\\" in args.label):
        parser.error("--label must be a file-name component")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = "resume" if args.resume else "first"
    if args.label:
        suffix += "." + args.label
    metrics = args.output.parent / f"{args.output.name}.{suffix}.metrics.json"
    if metrics.exists():
        parser.error(f"Refusing to overwrite benchmark evidence: {metrics}")
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0", MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", "--metrics", str(metrics)]
    started = time.perf_counter()
    with metrics.with_suffix(".stdout.json").open("w", encoding="utf-8") as stdout, \
            metrics.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.perf_counter() - started
    record = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
    record.update(process_wall_seconds=elapsed, exit_code=completed.returncode)
    metrics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(metrics), "process_wall_seconds": elapsed, "exit_code": completed.returncode}))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
