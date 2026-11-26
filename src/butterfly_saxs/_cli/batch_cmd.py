"""Batch, project, init-manifest, and campaign subcommand handlers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import glob
import os
from pathlib import Path
import sys
from typing import Any

from ..cli_contract import annotate_report
from ..errors import PipelineError
from ..path_utils import filter_supported_image_paths
from ..project import ProjectConfig
from ..settings import deep_merge_mapping
from .common import (
    _BatchFrameAnalyzer,
    analysis_overrides,
    campaign_runner,
    cli_module,
    filter_supported_kwargs,
    load_cli_config,
    pipeline_symbol,
    print_json,
    unattended_source_path,
    with_analysis,
)
from .reporting_cmd import (
    load_lamellar_settings,
    report_resume_command,
    validate_report_options,
)


def handle_batch(args: argparse.Namespace) -> int:
    from ..output_lock import output_lock

    if args.report_lamellar_settings and not args.report:
        raise PipelineError("--report-lamellar-settings requires --report")
    lamellar_settings = load_lamellar_settings(args.report_lamellar_settings)
    if args.report:
        validate_report_options(args.report_radial_bins, args.report_angular_bins)
    config = with_analysis(load_cli_config(args.config), analysis_overrides(args))
    output_dir = Path(args.output or (config.output_dir if config else "results"))
    analysis = config.analysis if config is not None else {}
    manifest = args.manifest or analysis.get("manifest")
    raw_inputs = list(args.inputs)
    if not raw_inputs and config:
        raw_inputs = list(config.input_paths)
    if not raw_inputs and not manifest:
        raise PipelineError(
            "batch 没有输入；请提供路径、--manifest 或 TOML 的 inputs.files"
        )
    if isinstance(manifest, (str, os.PathLike)):
        manifest_candidate = Path(manifest).expanduser()
        if not manifest_candidate.is_file():
            with manifest_candidate.open("rb"):
                pass
    if args.unattended is not None:
        if not args.output and not (config and config.output_dir):
            raise PipelineError(
                "unattended batch requires an explicit output directory"
            )
        poni_precheck = args.poni or (config.poni_path if config else None)
        if poni_precheck is None:
            raise PipelineError("unattended batch requires a PONI calibration")
        package_root = Path(args.unattended).expanduser().resolve(strict=False)
        output_root = output_dir.expanduser().resolve(strict=False)
        if output_root == package_root or output_root.is_relative_to(package_root):
            raise PipelineError("unattended output must be outside the raw package")
    cli_slice_precheck = (
        args.start is not None or args.stop is not None or args.stride is not None
    )
    if args.frame_range is not None and cli_slice_precheck:
        raise PipelineError("frame_range cannot be combined with start/stop/stride")
    resume_precheck = bool(args.resume or analysis.get("resume", False))
    if (
        output_dir.exists()
        and any(path.name != ".wingsaxs.lock" for path in output_dir.iterdir())
        and not args.force
        and not resume_precheck
    ):
        raise FileExistsError(
            f"输出目录已有内容，未覆盖：{output_dir}（需要 --force）"
        )
    with output_lock(output_dir, create=True):
        return _handle_batch_locked(args, config, lamellar_settings)


def _handle_batch_locked(
    args: argparse.Namespace,
    config: ProjectConfig | None,
    lamellar_settings: Any,
) -> int:
    from .. import batch as batch_module
    from .. import export as export_module

    analysis = config.analysis if config is not None else {}
    manifest = args.manifest or analysis.get("manifest")
    inputs = list(args.inputs)
    if not inputs and config:
        inputs = list(config.input_paths)
    if not inputs and not manifest:
        raise PipelineError(
            "batch 没有输入；请提供路径、--manifest 或 TOML 的 inputs.files"
        )
    expanded_inputs: list[str] = []
    for value in inputs:
        text = os.path.expanduser(os.fspath(value))
        if any(char in text for char in "*?[]"):
            expanded_inputs.extend(
                os.fspath(item)
                for item in filter_supported_image_paths(glob.glob(text))
            )
        elif Path(text).is_dir():
            expanded_inputs.extend(
                os.fspath(item)
                for item in filter_supported_image_paths(Path(text).iterdir())
            )
        else:
            expanded_inputs.append(text)
    inputs = expanded_inputs
    if not inputs and not manifest:
        raise PipelineError("batch 输入通配符没有匹配任何文件")
    mode = args.mode or str(
        analysis.get("batch_mode", analysis.get("mode", "independent"))
    )
    checkpoint = args.checkpoint or analysis.get("checkpoint")
    resume = bool(args.resume or analysis.get("resume", False))
    unattended = args.unattended is not None
    series = args.series if args.series is not None else analysis.get("series")
    cli_slice = (
        args.start is not None or args.stop is not None or args.stride is not None
    )
    cli_range = args.frame_range is not None
    if cli_range and cli_slice:
        start = args.start
        stop = args.stop
        stride = args.stride if args.stride is not None else 1
        frame_range = args.frame_range
    elif cli_range:
        start = None
        stop = None
        stride = 1
        frame_range = args.frame_range
    elif cli_slice:
        if analysis.get("frame_range") is not None:
            start = args.start
            stop = args.stop
            stride = args.stride if args.stride is not None else 1
        else:
            start = args.start if args.start is not None else analysis.get("start")
            stop = args.stop if args.stop is not None else analysis.get("stop")
            stride = (
                args.stride if args.stride is not None else analysis.get("stride", 1)
            )
        frame_range = None
    else:
        start = analysis.get("start")
        stop = analysis.get("stop")
        stride = analysis.get("stride", 1)
        frame_range = analysis.get("frame_range")
    output_dir = Path(args.output or (config.output_dir if config else "results"))
    poni = args.poni or (config.poni_path if config else None)
    if unattended:
        if not args.output and not (config and config.output_dir):
            raise PipelineError(
                "unattended batch requires an explicit output directory"
            )
        if poni is None:
            raise PipelineError("unattended batch requires a PONI calibration")
        package_root = Path(args.unattended).expanduser().resolve(strict=False)
        poni = unattended_source_path(poni, package_root)
        output_root = output_dir.expanduser().resolve(strict=False)
        if output_root == package_root or output_root.is_relative_to(package_root):
            raise PipelineError("unattended output must be outside the raw package")
        if checkpoint is None:
            checkpoint = output_dir / "checkpoint.json"
        checkpoint_root = Path(checkpoint).expanduser().resolve(strict=False)
        if not checkpoint_root.is_relative_to(output_root):
            raise PipelineError(
                "unattended checkpoint must be inside the output directory"
            )
    stream = bool(args.stream or unattended)
    if (
        output_dir.exists()
        and any(path.name != ".wingsaxs.lock" for path in output_dir.iterdir())
        and not args.force
        and not resume
    ):
        raise FileExistsError(
            f"输出目录已有内容，未覆盖：{output_dir}（需要 --force）"
        )

    full2d = args.full2d or (config.full2d if config else False)
    path_analysis = dict(config.analysis) if isinstance(config, ProjectConfig) else {}
    for name, value in (
        ("mask", args.mask),
        ("valid_mask", args.valid_mask),
        ("mask_frame", args.mask_frame),
        ("mask_dataset", args.mask_dataset),
        ("beamstop_radius_px", getattr(args, "beamstop_radius_px", None)),
    ):
        if value is not None:
            path_analysis[name] = value
    if unattended:
        for name in ("mask", "valid_mask"):
            source_path = path_analysis.get(name)
            if isinstance(source_path, (str, os.PathLike)):
                path_analysis[name] = unattended_source_path(
                    source_path, package_root
                )
    if cli_slice and not cli_range:
        path_analysis.pop("frame_range", None)
    elif cli_range and not cli_slice:
        for key in ("start", "stop", "stride"):
            path_analysis.pop(key, None)
    for name, value in (
        ("series", args.series),
        ("start", start if cli_slice else None),
        ("stop", stop if cli_slice else None),
        ("stride", stride if cli_slice else None),
        ("frame_range", frame_range if cli_range else None),
    ):
        if value is not None:
            path_analysis[name] = value
    if unattended:
        selected_method = path_analysis.setdefault(
            "ridge_method", "butterfly_curvature"
        )
        if selected_method != "butterfly_curvature":
            raise PipelineError(
                "unattended butterfly analysis requires ridge_method=butterfly_curvature"
            )
        configured_recipe = path_analysis.get("butterfly") or {}
        if not isinstance(configured_recipe, Mapping):
            raise PipelineError("analysis.butterfly must be a mapping")
        butterfly_recipe = dict(configured_recipe)
        butterfly_recipe.setdefault("stage", "evaluate")
        butterfly_recipe.setdefault("trace_method", "annular_peak")
        butterfly_recipe.setdefault("resamples", 0)
        if butterfly_recipe["stage"] != "evaluate":
            raise PipelineError(
                "unattended butterfly analysis requires butterfly.stage=evaluate"
            )
        path_analysis["butterfly"] = butterfly_recipe
    path_analysis["stage"] = "full2d" if full2d else "geometry"
    if isinstance(config, ProjectConfig) and (
        poni != config.poni_path or path_analysis != config.analysis
    ):
        batch_config: Any = ProjectConfig(
            input_paths=config.input_paths,
            poni_path=poni,
            output_dir=config.output_dir,
            q_unit=config.q_unit,
            full2d=full2d,
            analysis=path_analysis,
            export=config.export,
            metadata=config.metadata,
        )
    elif config is None and (poni is not None or path_analysis):
        batch_config = ProjectConfig(poni_path=poni, analysis=path_analysis)
    else:
        batch_config = config

    batch_inputs: Any = inputs
    batch_manifest: Any = manifest
    if args.frame is not None or args.dataset is not None:
        resolved_refs = batch_module.build_frame_refs(
            inputs,
            manifest=manifest,
            allow_mixed_series=series is not None,
        )
        resolved_with_cli_selectors = [
            batch_module.FrameRef(
                ref.path,
                time=ref.time,
                frame_id=ref.frame_id,
                metadata=ref.metadata,
                order=ref.order,
                source=ref.source,
                dataset=args.dataset if args.dataset is not None else ref.dataset,
                frame=args.frame if args.frame is not None else ref.frame,
            )
            for ref in resolved_refs
        ]
        batch_inputs = resolved_with_cli_selectors
        batch_manifest = [
            {**ref.to_dict(), "order": index}
            for index, ref in enumerate(resolved_with_cli_selectors)
        ]
        selector_config = {
            name: value
            for name, value in (("frame", args.frame), ("dataset", args.dataset))
            if value is not None
        }
        if isinstance(batch_config, ProjectConfig):
            batch_config = ProjectConfig(
                input_paths=batch_config.input_paths,
                poni_path=batch_config.poni_path,
                output_dir=batch_config.output_dir,
                q_unit=batch_config.q_unit,
                full2d=full2d,
                analysis=deep_merge_mapping(batch_config.analysis, selector_config),
                export=batch_config.export,
                metadata=batch_config.metadata,
            )
        else:
            batch_config = {
                "analysis": deep_merge_mapping(path_analysis, selector_config)
            }

    preflight_summary: dict[str, Any] | None = None
    if unattended:
        from ..service import ButterflyAnalysisService

        allow_mixed = bool(
            series is not None
            or path_analysis.get(
                "allow_mixed_series", path_analysis.get("independent_series", False)
            )
        )
        selected_refs = batch_module.build_frame_refs(
            batch_inputs,
            manifest=batch_manifest,
            allow_mixed_series=allow_mixed,
        )
        preflight_start, preflight_stop, preflight_stride = start, stop, stride
        if frame_range is not None:
            preflight_start, preflight_stop, preflight_stride = (
                batch_module.parse_frame_range(frame_range)
            )
            if start is not None or stop is not None or stride != 1:
                raise ValueError(
                    "frame_range cannot be combined with start/stop/stride"
                )
        selected_refs = batch_module.select_frame_refs(
            selected_refs,
            series=series,
            start=preflight_start,
            stop=preflight_stop,
            stride=preflight_stride,
        )
        if not selected_refs:
            raise ValueError("batch selection matched no frames")
        mask = path_analysis.get("mask")
        valid_mask = path_analysis.get("valid_mask")
        if mask is not None and valid_mask is not None:
            raise PipelineError(
                "unattended preflight cannot represent mask and valid_mask together"
            )
        q_window = path_analysis.get("q_window")
        if (
            q_window is None
            and path_analysis.get("q_min") is not None
            and path_analysis.get("q_max") is not None
        ):
            q_window = (path_analysis["q_min"], path_analysis["q_max"])
        if q_window is None and (path_analysis.get("q_min") is None) != (
            path_analysis.get("q_max") is None
        ):
            raise PipelineError(
                "unattended preflight requires both q_min and q_max when q_window is absent"
            )
        preflight_dir = output_dir / "preflight"
        preflight_report = ButterflyAnalysisService().preflight(
            args.unattended,
            manifest=[
                {
                    **ref.to_dict(),
                    "path": str(ref.path.expanduser().resolve(strict=False)),
                }
                for ref in selected_refs
            ],
            poni=poni,
            mask=mask if mask is not None else valid_mask,
            mask_convention=(
                "1_valid_0_invalid"
                if valid_mask is not None
                else "0_valid_1_invalid"
            ),
            mask_frame=path_analysis.get("mask_frame"),
            mask_dataset=path_analysis.get("mask_dataset"),
            q_window=q_window,
            context=(
                unattended_source_path(args.preflight_context, package_root)
                if args.preflight_context
                else None
            ),
            output=preflight_dir,
            force=bool(args.force or resume),
        )
        if not isinstance(preflight_report, Mapping):
            raise PipelineError("preflight returned no report")
        preflight_status = preflight_report.get("status")
        if not isinstance(preflight_status, Mapping):
            raise PipelineError("preflight returned no status envelope")
        preflight_color = preflight_status.get("status_color")
        preflight_code = preflight_status.get("exit_code")
        if {"green": 0, "yellow": 1, "red": 2}.get(preflight_color) != preflight_code:
            raise PipelineError("preflight returned an inconsistent status envelope")
        selector = preflight_report.get("selector")
        selected_mask = selector.get("mask") if isinstance(selector, Mapping) else None
        if isinstance(selected_mask, Mapping):
            selected_mask_path = selected_mask.get("path")
            expected_mask = mask if mask is not None else valid_mask
            if expected_mask is None and selected_mask_path is not None:
                raise PipelineError(
                    "preflight context selected a mask absent from the batch recipe"
                )
            if expected_mask is not None:
                if selected_mask_path is None:
                    raise PipelineError("preflight did not apply the batch mask")
                actual = Path(selected_mask_path)
                actual = actual if actual.is_absolute() else package_root / actual
                if actual.resolve(strict=False) != Path(
                    expected_mask
                ).expanduser().resolve(strict=False):
                    raise PipelineError("preflight mask differs from the batch mask")
        geometry = preflight_report.get("geometry")
        if isinstance(geometry, Mapping) and q_window is None:
            observed_window = geometry.get("q_window")
            full_range = geometry.get("q_range")
            if isinstance(observed_window, Mapping) and isinstance(
                full_range, Mapping
            ):
                for edge in ("min", "max"):
                    observed = float(observed_window[edge])
                    expected = float(full_range[edge])
                    if abs(observed - expected) > max(1e-9, 1e-6 * abs(expected)):
                        raise PipelineError(
                            "preflight context q_window differs from the batch recipe"
                        )
        preflight_summary = {
            "status_color": preflight_color,
            "scientific_status": preflight_status.get("scientific_status"),
            "exit_code": preflight_code,
            "report": str(preflight_dir / "preflight.json"),
            "selected_frames": len(selected_refs),
        }
        if preflight_color == "red":
            blocked = annotate_report(
                {
                    "unattended": True,
                    "blocked_stage": "preflight",
                    "preflight": preflight_summary,
                    "n_frames": 0,
                    "n_success": 0,
                    "n_failed": 0,
                    "outputs": {"preflight": str(preflight_dir / "preflight.json")},
                },
                command="batch",
                exit_code=2,
            )
            print_json(blocked)
            return 2

    effective_beamstop = (
        getattr(args, "beamstop_radius_px", None)
        if getattr(args, "beamstop_radius_px", None) is not None
        else path_analysis.get("beamstop_radius_px")
    )
    analyze_for_batch = _BatchFrameAnalyzer(
        poni=poni,
        full2d=full2d,
        frame=args.frame,
        dataset=args.dataset,
        mask=path_analysis.get("mask") if unattended else args.mask,
        mask_frame=path_analysis.get("mask_frame") if unattended else args.mask_frame,
        mask_dataset=path_analysis.get("mask_dataset")
        if unattended
        else args.mask_dataset,
        valid_mask=path_analysis.get("valid_mask") if unattended else args.valid_mask,
        beamstop_radius_px=effective_beamstop,
    )

    optional_batch_kwargs: dict[str, Any] = {}
    if getattr(args, "deduplicate_frames", False):
        optional_batch_kwargs["deduplicate_identical_frames"] = True
    if getattr(args, "workers", None) is not None:
        optional_batch_kwargs["max_workers"] = int(args.workers)
    if effective_beamstop is not None:
        optional_batch_kwargs["beamstop_radius_px"] = effective_beamstop
    accepted_batch_kwargs, unsupported_batch_kwargs = filter_supported_kwargs(
        batch_module.run_batch,
        optional_batch_kwargs,
        aliases={"deduplicate_identical_frames": ("deduplicate_frames",)},
    )
    if unsupported_batch_kwargs:
        warn_names = [
            name for name in unsupported_batch_kwargs if name != "beamstop_radius_px"
        ]
        if warn_names:
            print(
                f"警告：当前 run_batch 尚不支持参数 ({', '.join(warn_names)})，已回退默认执行。",
                file=sys.stderr,
            )

    stream_writer = None
    if stream:
        stream_writer = export_module.StreamingBatchExporter(
            output_dir,
            provenance={"command": "bsaxs batch", "full2d": full2d, "stream": True},
            force=bool(args.force or resume),
            resume=resume,
            analyzer_identity="butterfly_saxs.pipeline.analyze_frame",
        )
    try:
        run = batch_module.run_batch(
            batch_inputs,
            analyze_for_batch,
            mode=mode,
            config=batch_config,
            manifest=batch_manifest,
            checkpoint=checkpoint,
            resume=resume,
            series=series,
            start=start,
            stop=stop,
            stride=stride,
            frame_range=frame_range,
            progress=lambda update: print(
                f"batch progress {update.get('completed', 0)}/{update.get('total', 0)}",
                file=sys.stderr,
                flush=True,
            ),
            result_sink=None if stream_writer is None else stream_writer.write,
            retain_results=stream_writer is None,
            analyzer_identity="butterfly_saxs.pipeline.analyze_frame",
            restore_frame=None if stream_writer is None else stream_writer.can_restore,
            **accepted_batch_kwargs,
        )
        exports = (
            stream_writer.finalize(run)
            if stream_writer is not None
            else export_module.export_batch(
                run,
                output_dir,
                provenance={"command": "bsaxs batch", "full2d": full2d},
                force=bool(args.force or resume),
            )
        )
    except BaseException:
        if stream_writer is not None:
            stream_writer.abort()
        raise

    compact_records = []
    for item in run.frame_results:
        record = item.to_record(include_result=False)
        if hasattr(item.result, "to_mapping"):
            result_mapping = item.result.to_mapping()
            if stream and isinstance(result_mapping, Mapping):
                result_mapping = {
                    key: result_mapping.get(key)
                    for key in (
                        "metadata",
                        "flags",
                        "parameters",
                        "ridges",
                        "ridge_points",
                        "ellipse_fit",
                        "butterfly",
                        "lobe_radial_profiles",
                        "lobe_radial_peaks",
                        "saxs_physics",
                        "full2d",
                        "analysis",
                        "analysis_domain",
                        "valid_mask",
                    )
                    if key in result_mapping
                }
            record["result"] = result_mapping
        elif stream and isinstance(item.result, Mapping):
            record["result"] = {
                key: item.result.get(key)
                for key in (
                    "metadata",
                    "flags",
                    "parameters",
                    "ridges",
                    "ridge_points",
                    "ellipse_fit",
                    "butterfly",
                    "lobe_radial_profiles",
                    "lobe_radial_peaks",
                    "saxs_physics",
                    "full2d",
                    "analysis",
                    "analysis_domain",
                )
                if key in item.result
            }
        else:
            record["result"] = item.to_record()["result"]
        if unattended:
            record.pop("result", None)
        compact_records.append(record)

    report: dict[str, Any] = {
        "mode": run.mode,
        "input_hash": run.input_hash,
        "config_hash": run.config_hash,
        "frames": compact_records,
        "n_frames": len(run.frame_results),
        "n_success": sum(frame.status == "ok" for frame in run.frame_results),
        "n_warning": sum(frame.status == "warning" for frame in run.frame_results),
        "n_completed": len(run.successful),
        "n_failed": len(run.failures),
        "checkpoint": str(run.checkpoint) if run.checkpoint is not None else None,
        "selection": run.selection,
        "cancelled": run.cancelled,
        "elapsed_s": run.elapsed_s,
        "processed_count": run.processed_count,
        "total_count": run.total_count,
        "outputs": {key: str(path) for key, path in exports.items()},
    }
    if unattended:
        report["unattended"] = True
        report["preflight"] = preflight_summary
        report["outputs"]["preflight"] = str(
            output_dir / "preflight" / "preflight.json"
        )
    from ..batch import _quality_warning_reason

    warning_reason = next(
        (
            reason
            for frame in run.frame_results
            if frame.result is not None
            if (reason := _quality_warning_reason(frame.result)) is not None
        ),
        None,
    )
    exit_code = (
        1
        if (
            run.failures
            or run.cancelled
            or warning_reason
            or report["n_warning"]
            or (
                preflight_summary is not None
                and preflight_summary["status_color"] != "green"
            )
        )
        else 0
    )
    if args.report:
        report_resume = bool(resume)
        next_command = report_resume_command(
            output_dir,
            radial_bins=args.report_radial_bins,
            angular_bins=args.report_angular_bins,
            formats=args.report_formats,
            dpi=args.report_dpi,
            lamellar_settings_path=args.report_lamellar_settings,
        )
        try:
            report_kwargs: dict[str, Any] = {
                "resume": report_resume,
                "force": bool(args.force),
                "radial_bins": args.report_radial_bins,
                "angular_bins": args.report_angular_bins,
                "formats": args.report_formats,
                "dpi": args.report_dpi,
                "lamellar_settings": lamellar_settings,
            }
            report_journal_style = getattr(args, "report_journal_style", None)
            if report_journal_style not in (None, "default"):
                report_kwargs["journal_style"] = report_journal_style
            report_result = cli_module()._build_analysis_report(
                output_dir,
                **report_kwargs,
            )
            status = report_result.get(
                "status", report_result.get("operation_status", "completed")
            )
            report_summary = {
                "status": status,
                "counts": report_result.get("counts", {}),
                "outputs": report_result.get("outputs", {}),
            }
            report_exit_code = int(report_result.get("exit_code", 0))
            if report_exit_code != 0:
                report_summary["next_command"] = next_command
                exit_code = 1
            report["analysis_report"] = report_summary
        except Exception as exc:
            report["analysis_report"] = {
                "status": "incomplete",
                "counts": {},
                "outputs": {},
                "error": str(exc),
                "next_command": next_command,
            }
            print(
                f"Analysis exports saved; report incomplete: {exc}. "
                "Use the reported report command to continue without refitting.",
                file=sys.stderr,
            )
            exit_code = 1
    if args.package:
        from ..delivery import package_batch

        try:
            delivery = package_batch(
                output_dir,
                resume=resume,
                force=bool(args.force),
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            report["delivery"] = {
                key: delivery[key]
                for key in (
                    "status",
                    "operation_status",
                    "counts",
                    "archive",
                    "outputs",
                    "next_action",
                )
            }
            if delivery["exit_code"]:
                exit_code = 1
        except (OSError, ValueError) as exc:
            report["delivery"] = {
                "operation_status": "incomplete",
                "error": str(exc),
                "next_command": ["bsaxs", "package", str(output_dir), "--resume"],
            }
            print(
                f"Analysis exports saved; delivery incomplete: {exc}. "
                "Use the reported package command to finish without refitting.",
                file=sys.stderr,
            )
            exit_code = 1
    report = annotate_report(
        report,
        command="batch",
        exit_code=exit_code,
        quality_gate_reason=warning_reason,
    )
    print_json(report)
    return exit_code


def handle_project(args: argparse.Namespace) -> int:
    project_runner = getattr(cli_module(), "run_project", None)
    if project_runner is None:
        project_runner = pipeline_symbol("run_project_bounded")
    run = project_runner(args.config, force=args.force)
    compact_records = []
    for item in run.frame_results:
        has_to_mapping = hasattr(item.result, "to_mapping")
        record = item.to_record(include_result=not has_to_mapping)
        if has_to_mapping:
            record["result"] = item.result.to_mapping()
        compact_records.append(record)
    has_warnings = any(item.status == "warning" for item in run.frame_results)
    if args.legacy_json:
        print_json(compact_records)
        return 1 if run.failures or run.cancelled or has_warnings else 0
    exit_code = 1 if run.failures or run.cancelled or has_warnings else 0
    report = annotate_report(
        {
            "schema_version": "lamellarsaxs2d.project_run.v2",
            "mode": run.mode,
            "input_hash": run.input_hash,
            "config_hash": run.config_hash,
            "frames": compact_records,
            "n_frames": len(run.frame_results),
            "n_success": sum(frame.status == "ok" for frame in run.frame_results),
            "n_warning": sum(frame.status == "warning" for frame in run.frame_results),
            "n_completed": len(run.successful),
            "n_failed": len(run.failures),
            "cancelled": run.cancelled,
            "selection": run.selection,
            "processed_count": run.processed_count,
            "total_count": run.total_count,
            "elapsed_s": run.elapsed_s,
            "checkpoint": (
                str(run.checkpoint) if run.checkpoint is not None else None
            ),
        },
        command="project",
        exit_code=exit_code,
    )
    print_json(report)
    return exit_code


def handle_init_manifest(args: argparse.Namespace) -> int:
    from ..campaign import discover_manifest, write_manifest_csv

    source = args.input_dir or args.input_path
    if not source:
        raise PipelineError("init-manifest 需要输入目录或 HDF5/图像文件（--input）")
    output_path = Path(args.output).expanduser()
    rows = discover_manifest(
        source,
        pattern=args.pattern,
        recursive=bool(args.recursive),
        dataset=args.dataset,
        parse_metadata=bool(args.parse_metadata),
        compute_sha256=bool(args.sha256),
        relative_to=output_path.parent if args.relative else None,
    )
    if not rows:
        raise PipelineError(f"未发现受支持的图像或数据集帧：{source}")
    written = write_manifest_csv(rows, output_path, force=bool(args.force))
    report = annotate_report(
        {
            "ok": True,
            "input": str(Path(source).expanduser().resolve(strict=False)),
            "output": str(written.resolve(strict=False)),
            "n_frames": len(rows),
            "dataset": rows[0].get("dataset") if rows else args.dataset,
            "sha256_computed": bool(args.sha256),
            "parse_metadata": bool(args.parse_metadata),
            "rows_preview": rows[:5],
        },
        command="init-manifest",
        exit_code=0,
    )
    print_json(report)
    return 0


def handle_campaign(args: argparse.Namespace) -> int:
    config_path = args.config or args.config_pos
    if not config_path:
        raise PipelineError("campaign 需要活动 TOML 配置文件（--config）")
    output_root = args.output
    if not output_root:
        from ..campaign import load_campaign_config

        loaded = load_campaign_config(config_path)
        output_root = loaded.get("output_root")
    if not output_root:
        raise PipelineError("campaign 需要输出根目录（--output）")

    runner = campaign_runner()
    kwargs: dict[str, Any] = {
        "force": bool(args.force),
        "continue_on_error": bool(args.continue_on_error),
    }
    if args.report is not None:
        kwargs["report"] = bool(args.report)
    if args.journal_style and args.journal_style != "default":
        kwargs["journal_style"] = str(args.journal_style)

    optional_campaign_kwargs: dict[str, Any] = {}
    if getattr(args, "beamstop_radius_px", None) is not None:
        optional_campaign_kwargs["beamstop_radius_px"] = args.beamstop_radius_px
    if getattr(args, "deduplicate_frames", False):
        optional_campaign_kwargs["deduplicate_identical_frames"] = True
    if getattr(args, "workers", None) is not None:
        optional_campaign_kwargs["max_workers"] = int(args.workers)

    base_accepted, _ = filter_supported_kwargs(runner, kwargs)
    opt_accepted, opt_unsupported = filter_supported_kwargs(
        runner,
        optional_campaign_kwargs,
        aliases={"deduplicate_identical_frames": ("deduplicate_frames",)},
    )
    if opt_unsupported:
        print(
            f"警告：当前 run_campaign 尚不支持参数 ({', '.join(opt_unsupported)})，已回退默认执行。",
            file=sys.stderr,
        )
    call_kwargs = {**base_accepted, **opt_accepted}
    progress_accepted, _ = filter_supported_kwargs(
        runner,
        {"progress": lambda message: print(message, file=sys.stderr, flush=True)},
    )
    call_kwargs.update(progress_accepted)

    result = runner(config_path, output_root, **call_kwargs)
    exit_code = int(
        result.get(
            "exit_code",
            1 if (result.get("n_failed", 0) or result.get("n_warning", 0)) else 0,
        )
    )
    report = annotate_report(result, command="campaign", exit_code=exit_code)
    print_json(report)
    return exit_code
