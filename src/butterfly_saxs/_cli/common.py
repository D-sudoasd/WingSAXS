"""Pure-Python CLI option parsers, JSON emitters, and patch-seam resolvers.

Must never import ``numpy``, ``scipy``, ``matplotlib``, ``pyFAI``, ``h5py``,
``fabio``, or ``PySide6`` at module import time.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from ..cli_contract import cli_error_payload
from ..errors import PipelineError
from ..project import ProjectConfig, load_project
from ..settings import deep_merge_mapping

_WORKFLOW_PRESETS = ("annular", "curvature", "sector", "butterfly_pub")
_JOURNAL_STYLES = ("default", "nature", "nature_comm", "science", "macromolecules")
_BRIGHT_AXIS_CONVENTIONS = ("draw", "reference")
_INTENSITY_MODES = ("calibrated", "raw")
_REPORT_FORMATS = ("png", "svg", "pdf", "tiff")
_REPORT_DEFAULT_FORMATS = ("png", "svg", "pdf")
_REPORT_MAX_GRID_CELLS = 1_000_000


def cli_module() -> Any:
    """Return :mod:`butterfly_saxs.cli` so test monkeypatches take precedence."""

    mod = sys.modules.get("butterfly_saxs.cli")
    if mod is not None:
        return mod
    from .. import cli as imported_cli

    return imported_cli


def pipeline_symbol(name: str) -> Any:
    """Return a pipeline symbol lazily, honouring monkeypatches on ``butterfly_saxs.cli``."""

    patched = getattr(cli_module(), name, None)
    if patched is not None:
        return patched
    from .. import pipeline as pipeline_module

    return getattr(pipeline_module, name)


def advisor_callable() -> Any:
    """Return ``advise_frame`` lazily, honouring monkeypatches on ``butterfly_saxs.cli``."""

    patched = getattr(cli_module(), "advise_frame", None)
    if patched is not None:
        return patched
    try:
        from ..auto_advisor import advise_frame as live_advise
    except ModuleNotFoundError as exc:
        raise PipelineError(
            "auto_advisor module is unavailable in this installation"
        ) from exc
    return live_advise


def campaign_runner() -> Any:
    """Return ``run_campaign`` lazily, honouring monkeypatches on ``butterfly_saxs.cli``."""

    patched = getattr(cli_module(), "run_campaign", None)
    if patched is not None:
        return patched
    from ..campaign import run_campaign as live_run_campaign

    return live_run_campaign


def parse_shape(value: str) -> tuple[int, int]:
    pieces = [
        item for item in value.replace("×", "x").replace(",", "x").split("x") if item
    ]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("shape 应为 HxW，例如 256x256")
    try:
        result = tuple(int(item) for item in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape 应为 HxW，例如 256x256") from exc
    if any(item < 4 for item in result):
        raise argparse.ArgumentTypeError("shape 的两个尺寸必须至少为 4")
    return result  # type: ignore[return-value]


def integer_at_least(minimum: int, label: str):
    def parse(value: str) -> int:
        try:
            result = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} 必须是整数") from exc
        if result < minimum:
            raise argparse.ArgumentTypeError(f"{label} 必须至少为 {minimum}")
        return result

    return parse


def parse_beamstop_radius(value: str) -> float | str:
    """Parse ``--beamstop-radius-px`` as a non-negative finite float or ``'auto'``."""

    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        parsed = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "beamstop-radius-px 必须是非负有限数值（像素）或 auto"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            "beamstop-radius-px 必须是非负有限数值（像素）或 auto"
        )
    return parsed


def filter_supported_kwargs(
    func: Any,
    optional_kwargs: Mapping[str, Any],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Filter optional kwargs against ``func``'s signature without heavy imports."""

    if not optional_kwargs:
        return {}, []
    import inspect as _inspect

    try:
        sig = _inspect.signature(func)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        return dict(optional_kwargs), []
    if accepts_var_kw:
        return dict(optional_kwargs), []
    accepted: dict[str, Any] = {}
    unsupported: list[str] = []
    alias_map = aliases or {}
    for key, value in optional_kwargs.items():
        if key in params:
            accepted[key] = value
            continue
        matched = next((alt for alt in alias_map.get(key, ()) if alt in params), None)
        if matched is not None:
            accepted[matched] = value
        else:
            unsupported.append(key)
    return accepted, unsupported


class _BatchFrameAnalyzer:
    """Picklable per-frame adapter for ``batch`` (safe for Windows spawn pools)."""

    def __init__(
        self,
        *,
        poni: Any = None,
        full2d: bool = False,
        frame: int | None = None,
        dataset: str | None = None,
        mask: Any = None,
        mask_frame: int | None = None,
        mask_dataset: str | None = None,
        valid_mask: Any = None,
        beamstop_radius_px: float | str | None = None,
    ) -> None:
        self.poni = poni
        self.full2d = full2d
        self.frame = frame
        self.dataset = dataset
        self.mask = mask
        self.mask_frame = mask_frame
        self.mask_dataset = mask_dataset
        self.valid_mask = valid_mask
        self.beamstop_radius_px = beamstop_radius_px
        self._geometry_cache: dict[Any, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_geometry_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "_geometry_cache" not in self.__dict__:
            self._geometry_cache = {}

    def __call__(
        self,
        frame_ref: Any,
        initial_parameters: Any = None,
        config: Any = None,
    ) -> Any:
        source = getattr(frame_ref, "path", frame_ref)
        selected_frame = self.frame
        if selected_frame is None:
            selected_frame = getattr(frame_ref, "frame_selector", None)
        selected_dataset = self.dataset
        if selected_dataset is None:
            selected_dataset = getattr(frame_ref, "dataset", None)
            if selected_dataset is None:
                selected_dataset = getattr(frame_ref, "dataset_id", None) or None
        analyze_fn = pipeline_symbol("analyze_frame")
        call_kwargs: dict[str, Any] = {
            "poni": self.poni,
            "config": config,
            "full2d": self.full2d,
            "initial_parameters": initial_parameters,
            "frame": selected_frame,
            "dataset": selected_dataset,
            "mask": self.mask,
            "mask_frame": self.mask_frame,
            "mask_dataset": self.mask_dataset,
            "valid_mask": self.valid_mask,
            "geometry_cache": self._geometry_cache,
        }
        if self.beamstop_radius_px is not None:
            accepted, _ = filter_supported_kwargs(
                analyze_fn, {"beamstop_radius_px": self.beamstop_radius_px}
            )
            call_kwargs.update(accepted)
        return analyze_fn(source, **call_kwargs)


def load_cli_config(value: str | None) -> ProjectConfig | None:
    if not value:
        return None
    source = Path(value)
    return load_project(source).resolve_paths(source.parent)


def unattended_source_path(value: str | os.PathLike[str], package_root: Path) -> str:
    """Resolve a CLI source path once for preflight and batch fitting."""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = package_root / candidate
    return str(candidate.resolve(strict=False))


def analysis_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect explicit CLI refinement controls without overriding TOML defaults."""

    mapping: dict[str, Any] = {}
    for argument, key in (
        ("q_min", "q_min"),
        ("q_max", "q_max"),
        ("q_window", "q_window"),
        ("ridge_method", "ridge_method"),
        ("ridge_snr_threshold", "ridge_snr_threshold"),
        ("ridge_min_peak_fraction", "ridge_min_peak_fraction"),
        ("ridge_min_coverage", "ridge_min_coverage"),
        ("full2d_multistart", "full2d_multistart"),
        ("mask", "mask"),
        ("valid_mask", "valid_mask"),
        ("mask_frame", "mask_frame"),
        ("mask_dataset", "mask_dataset"),
        ("uncertainty_dataset", "uncertainty_dataset"),
        ("intensity_mode", "intensity_mode"),
        ("bright_axis_convention", "bright_axis_convention"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            mapping[key] = value
    if getattr(args, "include_physics", None):
        mapping["include_physics"] = True
    ellipse: dict[str, Any] = {}
    for argument, key in (
        ("ellipse_preset", "preset"),
        ("ellipse_ratio_min", "axis_ratio_min"),
        ("ellipse_ratio_max", "axis_ratio_max"),
        ("ellipse_a", "a"),
        ("ellipse_b", "b"),
        ("ellipse_ratio", "axis_ratio"),
        ("ellipse_a_min", "a_min"),
        ("ellipse_a_max", "a_max"),
        ("ellipse_b_min", "b_min"),
        ("ellipse_b_max", "b_max"),
        ("ellipse_angle_min", "theta_min_deg"),
        ("ellipse_angle_max", "theta_max_deg"),
        ("ellipse_fixed_center", "fixed_center"),
        ("ellipse_fixed_a", "fixed_a"),
        ("ellipse_fixed_ratio", "fixed_axis_ratio"),
        ("ellipse_center_qx", "center_qx"),
        ("ellipse_center_qy", "center_qy"),
        ("ellipse_fixed_angle", "fixed_angle"),
        ("ellipse_angle_deg", "angle_deg"),
        ("ellipse_residual", "residual"),
        ("ellipse_multistart", "multistart"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            ellipse[key] = value
    if ellipse:
        mapping["ellipse"] = ellipse
    butterfly = {}
    for argument, key in (
        ("butterfly_stage", "stage"),
        ("butterfly_resamples", "resamples"),
        ("butterfly_seed", "seed"),
        ("butterfly_sensitivity", "sensitivity"),
        ("butterfly_trace_method", "trace_method"),
        ("sector_width", "sector_width_deg"),
        ("sector_step", "sector_step_deg"),
        ("annular_rings", "annular_radial_bins"),
        ("annular_angles", "annular_angle_bins"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            butterfly[key] = value
    if butterfly:
        mapping["butterfly"] = butterfly
    if getattr(args, "butterfly_trace_method", None) is not None:
        selected = mapping.get("ridge_method")
        if selected is not None and selected != "butterfly_curvature":
            raise ValueError(
                "--butterfly-trace-method requires the butterfly workflow; omit --ridge-method or use butterfly_curvature"
            )
        mapping["ridge_method"] = "butterfly_curvature"
    preset = getattr(args, "preset", None)
    if preset is not None:
        from ..campaign import workflow_preset_analysis

        preset_defaults = workflow_preset_analysis(preset)
        mapping = deep_merge_mapping(preset_defaults, mapping)
    return mapping


def with_analysis(config: ProjectConfig | None, overrides: Mapping[str, Any]) -> Any:
    """Return a config carrying explicit CLI overrides with TOML precedence."""

    if not overrides:
        return config
    if config is None:
        return ProjectConfig(analysis=dict(overrides))
    return ProjectConfig(
        input_paths=config.input_paths,
        poni_path=config.poni_path,
        output_dir=config.output_dir,
        q_unit=config.q_unit,
        full2d=config.full2d,
        analysis=deep_merge_mapping(config.analysis, overrides),
        export=config.export,
        metadata=config.metadata,
    )


def add_refinement_options(parser: argparse.ArgumentParser) -> None:
    """Add the common flat-ellipse/ridge controls to a CLI subcommand."""

    parser.add_argument(
        "--preset",
        choices=_WORKFLOW_PRESETS,
        default=None,
        help="工作流参数预设（annular/curvature/sector/butterfly_pub；省略时保持原有默认行为）",
    )
    parser.add_argument(
        "--q-window",
        type=float,
        nargs=2,
        metavar=("Q_MIN", "Q_MAX"),
        help="analysis q window in the active q-map unit",
    )
    parser.add_argument("--q-min", type=float, help="analysis q lower bound")
    parser.add_argument("--q-max", type=float, help="analysis q upper bound")
    parser.add_argument(
        "--ridge-method",
        choices=(
            "radial_peak",
            "azimuthal_peak",
            "surface_curvature",
            "butterfly_curvature",
        ),
        help="observed ridge method, including side-aware butterfly_curvature",
    )
    parser.add_argument(
        "--butterfly-stage", choices=("trace", "evaluate"), default=None
    )
    parser.add_argument(
        "--butterfly-trace-method",
        choices=("curvature", "radial_sector", "annular_peak"),
        help="butterfly observable: fixed-q angular tracks, radial sector peaks, or curvature candidates",
    )
    parser.add_argument(
        "--sector-width",
        type=float,
        help="radial-sector full angular width in degrees (default 10)",
    )
    parser.add_argument(
        "--annular-rings",
        type=int,
        help="number of fixed-q annuli (default 40; limited by pixel q sampling)",
    )
    parser.add_argument(
        "--annular-angles",
        type=int,
        help="angular bins per q annulus (default 72)",
    )
    parser.add_argument(
        "--sector-step",
        type=float,
        help="radial-sector angular sampling step in degrees (default 5)",
    )
    parser.add_argument(
        "--butterfly-resamples",
        type=int,
        default=None,
        help="image-level resampling count; zero skips empirical intervals",
    )
    parser.add_argument("--butterfly-seed", type=int, default=None)
    parser.add_argument(
        "--butterfly-no-sensitivity",
        dest="butterfly_sensitivity",
        action="store_false",
        default=None,
    )
    parser.add_argument("--ridge-snr-threshold", type=float, help="minimum ridge SNR")
    parser.add_argument(
        "--ridge-min-peak-fraction",
        type=float,
        help="minimum valid support fraction for a ridge candidate [0, 1]",
    )
    parser.add_argument(
        "--ridge-min-coverage",
        type=float,
        help="minimum detector coverage for a ridge candidate [0, 1]",
    )
    parser.add_argument(
        "--ellipse-preset",
        choices=("standard", "flat_ellipse", "very_flat_ellipse"),
        help="constrained measured-ellipse preset",
    )
    parser.add_argument(
        "--ellipse-ratio-min", type=float, help="measured ellipse b/a lower bound"
    )
    parser.add_argument(
        "--ellipse-ratio-max", type=float, help="measured ellipse b/a upper bound"
    )
    parser.add_argument(
        "--ellipse-a", type=float, help="measured ellipse a starting value"
    )
    parser.add_argument(
        "--ellipse-b", type=float, help="measured ellipse b starting value"
    )
    parser.add_argument(
        "--ellipse-ratio", type=float, help="measured ellipse b/a starting value"
    )
    parser.add_argument(
        "--ellipse-a-min", type=float, help="measured ellipse a lower bound"
    )
    parser.add_argument(
        "--ellipse-a-max", type=float, help="measured ellipse a upper bound"
    )
    parser.add_argument(
        "--ellipse-b-min", type=float, help="derived measured ellipse b lower bound"
    )
    parser.add_argument(
        "--ellipse-b-max", type=float, help="derived measured ellipse b upper bound"
    )
    parser.add_argument(
        "--ellipse-angle-min",
        type=float,
        help="measured ellipse angle lower bound (deg)",
    )
    parser.add_argument(
        "--ellipse-angle-max",
        type=float,
        help="measured ellipse angle upper bound (deg)",
    )
    parser.add_argument(
        "--ellipse-fixed-center",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fix measured ellipse centre at ellipse-center-qx/qy",
    )
    parser.add_argument(
        "--ellipse-fixed-a",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fix measured ellipse a at its explicit value",
    )
    parser.add_argument(
        "--ellipse-fixed-ratio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fix measured ellipse b/a at its explicit value",
    )
    parser.add_argument(
        "--ellipse-center-qx", type=float, help="measured ellipse centre qx"
    )
    parser.add_argument(
        "--ellipse-center-qy", type=float, help="measured ellipse centre qy"
    )
    parser.add_argument(
        "--ellipse-fixed-angle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fix the measured ellipse angle at ellipse-angle-deg",
    )
    parser.add_argument(
        "--ellipse-angle-deg",
        type=float,
        help="measured ellipse angle initial value (deg)",
    )
    parser.add_argument(
        "--ellipse-residual",
        choices=("sampson", "geometric"),
        help="residual used by measured ellipse fit",
    )
    parser.add_argument(
        "--ellipse-multistart",
        type=int,
        help="deterministic measured ellipse starts",
    )
    parser.add_argument(
        "--full2d-multistart", type=int, help="deterministic full2d starts"
    )


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False))


def emit_error(
    exc: BaseException,
    *,
    exit_code: int,
    command: str | None = None,
    code: str | None = None,
) -> None:
    """Human stderr line plus a strict JSON envelope on stdout for agents."""

    print(f"错误：{exc}", file=sys.stderr)
    print_json(
        cli_error_payload(exc, exit_code=exit_code, command=command, code=code)
    )
