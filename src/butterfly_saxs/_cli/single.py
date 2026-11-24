"""Single-frame, diagnostics, synthetic, and GUI subcommand handlers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import sys
from typing import Any

from ..cli_contract import agent_manifest, annotate_report
from ..errors import PipelineError
from ..project import ProjectConfig
from ..serialization import write_json_atomic
from .common import (
    advisor_callable,
    analysis_overrides,
    filter_supported_kwargs,
    load_cli_config,
    pipeline_symbol,
    print_json,
    with_analysis,
)


def _pick_input(args: argparse.Namespace, config: ProjectConfig | None) -> Any:
    value = args.input_file or args.input
    if value:
        return value
    if config and config.input_paths:
        return config.input_paths[0]
    raise PipelineError("请提供输入图像，或在 TOML 中填写 inputs.files")


def _write_synthetic(
    array: Any,
    qmap: dict[str, Any],
    output: str | os.PathLike[str],
    *,
    force: bool,
) -> Path:
    import numpy as np

    destination = Path(output)
    if destination.exists() and not force:
        raise FileExistsError(f"输出已存在，未覆盖：{destination}（需要 --force）")
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".npy":
        np.save(destination, array)
    elif suffix == ".npz":
        payload = {
            key: value for key, value in qmap.items() if isinstance(value, np.ndarray)
        }
        if qmap.get("q_unit") is not None:
            payload["q_unit"] = np.asarray(str(qmap["q_unit"]))
        np.savez_compressed(destination, data=array, **payload)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        tifffile.imwrite(destination, array)
    else:
        raise PipelineError("synthetic 输出格式支持 .npy、.npz、.tif/.tiff")
    return destination


def handle_describe(args: argparse.Namespace) -> int:
    report = agent_manifest(getattr(args, "topic", None))
    if getattr(args, "text", False):
        tool = report["tool"]
        print(
            (
                f"{tool['name']} {tool['version']} agent catalog\n"
                "Run commands via `bsaxs <command>`. JSON stdout; exit 0/1/2.\n"
                "success=True is not scientific acceptance. pixel-q is not a period."
            ),
            file=sys.stderr,
        )
    print_json(report)
    return 0


def handle_doctor(args: argparse.Namespace) -> int:
    from ..doctor import main as doctor_main

    argv: list[str] = []
    if args.require_ui:
        argv.append("--require-ui")
    if args.json:
        argv.append("--json")
    if args.quiet:
        argv.append("--quiet")
    return int(doctor_main(argv))


def handle_inspect(args: argparse.Namespace) -> int:
    config = load_cli_config(args.config)
    source = _pick_input(args, config)
    poni = args.poni or (config.poni_path if config else None)
    inspect_fn = pipeline_symbol("inspect_frame")
    inspect_kwargs: dict[str, Any] = {
        "poni": poni,
        "config": config,
        "frame": args.frame,
        "dataset": args.dataset,
        "mask": args.mask,
        "mask_frame": args.mask_frame,
        "mask_dataset": args.mask_dataset,
        "valid_mask": args.valid_mask,
    }
    beamstop_radius = getattr(args, "beamstop_radius_px", None)
    if beamstop_radius is not None:
        accepted_bs, unsupported_bs = filter_supported_kwargs(
            inspect_fn, {"beamstop_radius_px": beamstop_radius}
        )
        inspect_kwargs.update(accepted_bs)
        if unsupported_bs:
            print(
                f"警告：当前 inspect_frame 尚不支持参数 ({', '.join(unsupported_bs)})，已跳过。",
                file=sys.stderr,
            )
    recommend = bool(getattr(args, "recommend", False))
    nominal_axis = getattr(args, "nominal_draw_axis_deg", None)
    used_native_recommend = False
    if recommend:
        import inspect as _inspect

        try:
            sig = _inspect.signature(inspect_fn)
            params = sig.parameters
            accepts_var_kw = any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if accepts_var_kw or "recommend" in params:
                inspect_kwargs["recommend"] = True
                used_native_recommend = True
                if nominal_axis is not None and (
                    accepts_var_kw or "nominal_draw_axis_deg" in params
                ):
                    inspect_kwargs["nominal_draw_axis_deg"] = float(nominal_axis)
        except (TypeError, ValueError):
            pass

    report = inspect_fn(source, **inspect_kwargs)
    if recommend and not used_native_recommend and "advisor" not in report:
        report = dict(report)
        try:
            advisor_fn = advisor_callable()
        except PipelineError as exc:
            report["advisor"] = {
                "available": False,
                "reason": str(exc),
            }
        else:
            advisor_kwargs: dict[str, Any] = {
                "poni_path": poni,
                "mask_path": args.mask,
                "image_dataset": args.dataset,
                "image_frame_index": args.frame,
                "mask_dataset": args.mask_dataset,
                "mask_frame_index": args.mask_frame,
                "nominal_draw_axis_deg": (
                    90.0 if nominal_axis is None else float(nominal_axis)
                ),
            }
            if isinstance(beamstop_radius, (int, float)):
                accepted_adv, _ = filter_supported_kwargs(
                    advisor_fn, {"beamstop_radius_px": float(beamstop_radius)}
                )
                advisor_kwargs.update(accepted_adv)
            report["advisor"] = advisor_fn(source, **advisor_kwargs)
    report = annotate_report(report, command="inspect", exit_code=0)
    if args.output:
        destination = Path(args.output)
        if destination.exists() and not args.force:
            raise FileExistsError(f"输出已存在，未覆盖：{destination}（需要 --force）")
        write_json_atomic(destination, report, sort_keys=False)
    print_json(report)
    return 0


def handle_analyze(args: argparse.Namespace) -> int:
    config = load_cli_config(args.config)
    overrides = analysis_overrides(args)
    figure_output = getattr(args, "figure_output", None)
    if figure_output:
        figure_target = Path(figure_output).expanduser().resolve()
        if figure_target.exists():
            raise FileExistsError(f"科研图目录已存在，未覆盖：{figure_output}")
        if args.output:
            analysis_target = Path(args.output).expanduser().resolve()
            if (
                analysis_target == figure_target
                or figure_target in analysis_target.parents
            ):
                raise PipelineError(
                    "分析输出会占用科研图目录；请为 --output 和 --figure-output 选择独立目录"
                )
            if (
                analysis_target.suffix.lower() in {".json", ".npz", ".csv"}
                and analysis_target in figure_target.parents
            ):
                raise PipelineError("科研图目录不能放在分析输出文件内部；请选择独立目录")
        selected_method = overrides.get(
            "ridge_method", (config.analysis if config else {}).get("ridge_method")
        )
        from ..analysis_config import normalize_ridge_method

        if (
            selected_method is not None
            and normalize_ridge_method(selected_method) != "butterfly_curvature"
        ):
            raise PipelineError("科研图导出需要 --ridge-method butterfly_curvature")
        overrides["ridge_method"] = "butterfly_curvature"
    config = with_analysis(config, overrides)
    source = _pick_input(args, config)
    analyze_fn = pipeline_symbol("analyze_frame")
    analyze_kwargs: dict[str, Any] = {
        "poni": args.poni or (config.poni_path if config else None),
        "config": config,
        "full2d": args.full2d or (config.full2d if config else False),
        "frame": args.frame,
        "dataset": args.dataset,
        "mask": args.mask,
        "mask_frame": args.mask_frame,
        "mask_dataset": args.mask_dataset,
        "valid_mask": args.valid_mask,
        "output": None if figure_output else args.output,
        "force": args.force,
    }
    optional_analyze_kwargs: dict[str, Any] = {}
    if getattr(args, "beamstop_radius_px", None) is not None:
        optional_analyze_kwargs["beamstop_radius_px"] = args.beamstop_radius_px
    if getattr(args, "uncertainty_dataset", None) is not None:
        optional_analyze_kwargs["uncertainty_dataset"] = args.uncertainty_dataset
    if getattr(args, "include_physics", False):
        optional_analyze_kwargs["include_physics"] = True
    if getattr(args, "intensity_mode", None) is not None:
        optional_analyze_kwargs["intensity_mode"] = args.intensity_mode
    accepted_opt, unsupported_opt = filter_supported_kwargs(
        analyze_fn, optional_analyze_kwargs
    )
    analyze_kwargs.update(accepted_opt)
    if unsupported_opt:
        print(
            f"警告：当前 analyze_frame 尚不支持参数 ({', '.join(unsupported_opt)})，已跳过。",
            file=sys.stderr,
        )
    result = analyze_fn(source, **analyze_kwargs)
    if figure_output:
        from ..butterfly_figure import export_butterfly_figure
        from ..pipeline import _coerce_qmap, _result_output_paths

        if not isinstance(result.butterfly, Mapping):
            raise PipelineError("当前分析没有可导出的蝴蝶观测结果")
        if args.output:
            analysis_paths = [
                path.expanduser().resolve()
                for path in _result_output_paths(result, args.output)
            ]
            if any(
                path == figure_target
                or figure_target in path.parents
                or path in figure_target.parents
                for path in analysis_paths
            ):
                raise PipelineError("分析文件与科研图目录冲突；请选择独立目录")
            existing = [path for path in analysis_paths if path.exists()]
            if any(path.is_dir() for path in existing):
                raise PipelineError(
                    "分析输出文件的位置已有目录；科研图未写入，请选择独立目录"
                )
            if any(
                parent.exists() and not parent.is_dir()
                for path in analysis_paths
                for parent in path.parents
            ):
                raise PipelineError(
                    "分析输出的父目录位置已有文件；科研图未写入，请选择独立目录"
                )
            if existing and not args.force:
                raise FileExistsError(f"分析输出已存在，科研图未写入：{existing[0]}")
        coordinates = _coerce_qmap(result.qmap, result.image.shape)
        pixel_fit = result.full2d if isinstance(result.full2d, Mapping) else {}
        pixel_model = pixel_fit.get("model_image", pixel_fit.get("model"))
        comparison_inputs = {} if pixel_model is None else {"model": pixel_model}
        written = export_butterfly_figure(
            figure_output,
            observed=result.image,
            qx=coordinates["qx"],
            qy=coordinates["qy"],
            valid_mask=result.valid_mask,
            result=result.butterfly,
            q_unit=coordinates.get("q_unit", "unknown"),
            context={
                "metadata": result.metadata,
                "analysis": result.analysis,
                "valid_mask_role": "analysis fit domain, including detector/external mask, q-window, ROI and weight validity",
                "analysis_domain": None
                if result.analysis_domain is None
                else result.analysis_domain.to_summary(),
                "pixel_model_source": "full2d fit output"
                if pixel_model is not None
                else None,
                "pixel_model_status": pixel_fit.get("status"),
                "pixel_model_parameters": pixel_fit.get("parameters"),
                "pixel_model_reference_axis_deg": pixel_fit.get("reference_axis_deg"),
                "pixel_model_bound_flags": pixel_fit.get("bound_flags"),
                "pixel_model_condition_number": pixel_fit.get("condition_number"),
                "pixel_model_rmse": pixel_fit.get("rmse"),
                "pixel_model_effective_bounds": pixel_fit.get("effective_bounds"),
            },
            width_mm=args.figure_width,
            dpi=args.figure_dpi,
            **comparison_inputs,
        )
        result.output_paths.extend(str(path) for path in written.values())
        if args.output:
            from ..pipeline import export_result

            analysis_paths = export_result(result, args.output, force=args.force)
            result.output_paths.extend(str(path) for path in analysis_paths)
    report = result.to_mapping()
    from ..batch import _quality_failure_reason, _quality_warning_reason

    quality_reason = _quality_failure_reason(report) or _quality_warning_reason(report)
    exit_code = 1 if quality_reason is not None else 0
    report = annotate_report(
        report,
        command="analyze",
        exit_code=exit_code,
        quality_gate_reason=quality_reason,
    )
    print_json(report)
    return exit_code


def handle_advise(args: argparse.Namespace) -> int:
    image = args.image or args.image_pos
    if not image:
        raise PipelineError("advise 需要输入图像路径（--image）")
    advisor_fn = advisor_callable()
    advise_kwargs: dict[str, Any] = {
        "poni_path": args.poni,
        "mask_path": args.mask,
        "image_dataset": args.image_dataset,
        "image_frame_index": args.image_frame_index,
        "mask_dataset": args.mask_dataset,
        "mask_frame_index": args.mask_frame_index,
        "nominal_draw_axis_deg": float(args.nominal_draw_axis_deg),
        "bright_axis_convention": str(getattr(args, "bright_axis_convention", "draw")),
    }
    advice = advisor_fn(image, **advise_kwargs)
    report = annotate_report(advice, command="advise", exit_code=0)
    if args.output:
        destination = Path(args.output).expanduser()
        if destination.exists() and not args.force:
            raise FileExistsError(f"输出已存在，未覆盖：{destination}（需要 --force）")
        write_json_atomic(destination, report, sort_keys=False)
    print_json(report)
    return 0


def handle_synthetic(args: argparse.Namespace) -> int:
    import numpy as np

    array, qmap = pipeline_symbol("synthetic_butterfly")(
        args.shape,
        q0=args.q0,
        width=args.width,
        ellipticity=args.ellipticity,
        angle_deg=args.angle,
        noise=args.noise,
        seed=args.seed,
        return_qmap=True,
    )
    destination = (
        _write_synthetic(array, qmap, args.output, force=args.force)
        if args.output
        else None
    )
    report = annotate_report(
        {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "output": os.fspath(destination) if destination else None,
            "intensity_min": float(np.min(array)),
            "intensity_max": float(np.max(array)),
            "seed": args.seed,
            "flags": {
                "empirical_model_only": True,
                "mechanism_under_determined": True,
                "forward_simulation_only": True,
                "nonunique_inverse_problem": True,
                "uncalibrated_pixel_q": True,
            },
        },
        command="synthetic",
        exit_code=0,
    )
    print_json(report)
    return 0


def handle_gui(args: argparse.Namespace) -> int:
    config = load_cli_config(args.config)
    return int(
        pipeline_symbol("launch_gui")(
            input_path=args.input,
            poni=args.poni or (config.poni_path if config else None),
            config=config,
            config_path=args.config,
        )
        or 0
    )
