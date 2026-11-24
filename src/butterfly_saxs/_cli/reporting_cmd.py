"""Report, benchmark, annotation-pack, and P3/P4 gate subcommand handlers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any

from .common import (
    _REPORT_DEFAULT_FORMATS,
    _REPORT_MAX_GRID_CELLS,
    cli_module,
    print_json,
)


def validate_report_options(radial_bins: int, angular_bins: int) -> None:
    if radial_bins * angular_bins > _REPORT_MAX_GRID_CELLS:
        raise ValueError(
            "report radial_bins * angular_bins must not exceed "
            f"{_REPORT_MAX_GRID_CELLS}"
        )


def report_resume_command(
    output_dir: str | Path,
    *,
    radial_bins: int,
    angular_bins: int,
    formats: Sequence[str],
    dpi: int,
    lamellar_settings_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    command = ["bsaxs", "report", os.fspath(output_dir), "--resume"]
    if radial_bins != 128:
        command.extend(("--radial-bins", str(radial_bins)))
    if angular_bins != 72:
        command.extend(("--angular-bins", str(angular_bins)))
    if tuple(formats) != _REPORT_DEFAULT_FORMATS:
        command.extend(("--formats", *(str(value) for value in formats)))
    if dpi != 180:
        command.extend(("--dpi", str(dpi)))
    if lamellar_settings_path is not None:
        command.extend(("--lamellar-settings", os.fspath(lamellar_settings_path)))
    return command


def load_lamellar_settings(
    path: str | os.PathLike[str] | None,
) -> dict[str, Any] | None:
    """Load and validate one JSON/TOML lamellar geometry settings mapping."""

    if path is None:
        return None
    source = Path(path).expanduser()
    raw = source.read_text(encoding="utf-8-sig")
    suffix = source.suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(raw)
        elif suffix == ".toml":
            value = tomllib.loads(raw)
        elif suffix:
            raise ValueError("lamellar settings file must use .json or .toml")
        else:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = tomllib.loads(raw)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid lamellar settings file {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("lamellar settings file must contain a mapping")

    from ..lamellar_models import LamellarSettings

    try:
        settings_mapping: Mapping[str, Any] = value
        for key in ("lamellar", "lamellar_settings", "settings"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                settings_mapping = nested
                break
        allowed_fields = set(LamellarSettings().to_dict())
        unknown_fields = sorted(set(settings_mapping) - allowed_fields)
        if unknown_fields:
            supported = ", ".join(sorted(allowed_fields))
            names = ", ".join(str(name) for name in unknown_fields)
            raise ValueError(
                f"unknown lamellar setting(s): {names}; supported fields: {supported}"
            )
        normalized = {"mode": "multi", "period_source": "ellipse"}
        normalized.update(settings_mapping)
        return LamellarSettings.from_mapping(normalized).to_dict()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid lamellar settings in {source}: {exc}") from exc


def build_analysis_report_cmd(
    output_dir: str | Path,
    *,
    resume: bool = False,
    force: bool = False,
    radial_bins: int = 128,
    angular_bins: int = 72,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 180,
    lamellar_settings: Mapping[str, Any] | None = None,
    journal_style: str | None = None,
) -> Mapping[str, Any]:
    """Load the optional report pipeline only when a report is requested."""

    from ..report import build_analysis_report

    options: dict[str, Any] = {
        "resume": resume,
        "force": force,
        "radial_bins": radial_bins,
        "angular_bins": angular_bins,
        "formats": tuple(formats),
        "dpi": dpi,
        "progress": lambda message: print(message, file=sys.stderr, flush=True),
    }
    if lamellar_settings is not None:
        options["lamellar_settings"] = dict(lamellar_settings)
    if (
        journal_style is not None
        and str(journal_style).strip()
        and str(journal_style) != "default"
    ):
        import inspect as _inspect

        try:
            sig = _inspect.signature(build_analysis_report)
            params = sig.parameters
            accepts_style = "journal_style" in params or any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            accepts_style = True
        if accepts_style:
            options["journal_style"] = str(journal_style)
        else:
            print(
                f"警告：当前报告模块尚不支持 --journal-style={journal_style}，已回退默认样式。",
                file=sys.stderr,
            )
    return build_analysis_report(output_dir, **options)


def handle_report(args: argparse.Namespace) -> int:
    validate_report_options(args.radial_bins, args.angular_bins)
    lamellar_settings = load_lamellar_settings(args.lamellar_settings)
    report_kwargs: dict[str, Any] = {
        "resume": args.resume,
        "force": args.force,
        "radial_bins": args.radial_bins,
        "angular_bins": args.angular_bins,
        "formats": args.formats,
        "dpi": args.dpi,
        "lamellar_settings": lamellar_settings,
    }
    journal_style = getattr(args, "journal_style", "default")
    if journal_style is not None and str(journal_style) != "default":
        report_kwargs["journal_style"] = str(journal_style)
    result = cli_module()._build_analysis_report(args.output_dir, **report_kwargs)
    print_json(result)
    return int(result.get("exit_code", 0))


def handle_benchmark(args: argparse.Namespace) -> int:
    from .. import benchmark_t1, benchmark_t2

    output = Path(args.output)
    manifests: dict[str, str] = {}
    if args.suite in {"t1", "all"}:
        destination = output / "t1" if args.suite == "all" else output
        manifest = benchmark_t1.write_evidence_directory(
            destination,
            shape=args.shape,
            seed=args.seed,
            force=args.force,
        )
        manifests["t1"] = manifest.as_posix()
    if args.suite in {"t2", "all"}:
        destination = output / "t2" if args.suite == "all" else output
        manifest = benchmark_t2.write_evidence_directory(
            destination,
            shape=args.shape or benchmark_t2.DEFAULT_SHAPE,
            seed=args.seed,
            force=args.force,
        )
        manifests["t2"] = manifest.as_posix()
    print_json({"suite": args.suite, "manifests": manifests})
    return 0


def handle_annotation_pack(args: argparse.Namespace) -> int:
    from ..annotation_pack import build_annotation_pack

    result = build_annotation_pack(
        args.package,
        args.rt_manifest,
        args.hold_manifest,
        args.output,
        preflight_json=args.preflight,
        poni=args.poni,
        mask=args.mask,
    )
    print_json(
        {
            "output_directory": Path(result["output_directory"]).as_posix(),
            "candidate_count": result["candidate_count"],
            "status": result["status"]["status"],
            "human_consensus": result["status"]["human_consensus"],
        }
    )
    return 0


def handle_p3_status(args: argparse.Namespace) -> int:
    from ..p3_gate import evaluate_p3_gate, write_p3_gate_report

    if args.output:
        report = write_p3_gate_report(
            args.output,
            args.t1_manifest,
            args.t2_manifest,
            args.annotation_status,
            args.thresholds,
            force=args.force,
        )
    else:
        report = evaluate_p3_gate(
            args.t1_manifest,
            args.t2_manifest,
            args.annotation_status,
            args.thresholds,
        )
    print_json(report)
    return int(report["exit_code"])


def handle_p4_evaluate(args: argparse.Namespace) -> int:
    from ..p4_validation import run_p4_engineering

    report = run_p4_engineering(
        t1_manifest=args.t1_manifest,
        t2_manifest=args.t2_manifest,
        thresholds=args.thresholds,
        output=args.output,
        r0_package=args.r0_package,
        r0_manifest=args.r0_manifest,
        poni=args.poni,
        mask=args.mask,
        ridge_method=args.ridge_method,
        run_sensitivity=not args.skip_sensitivity,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print_json(
        {
            "stage": report["stage"],
            "engineering_status": report["engineering_status"],
            "scientific_status": report["scientific_status"],
            "p4_go_no_go": report["p4_go_no_go"],
            "outputs": report["outputs"],
        }
    )
    return 0 if report["p4_go_no_go"] == "GO" else 1
