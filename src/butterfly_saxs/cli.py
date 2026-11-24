"""Command line parser and lazy dispatcher for the shared WingSAXS pipeline.

``import butterfly_saxs.cli``, ``bsaxs --help``, and ``bsaxs describe`` must
remain strictly free of scientific and UI imports (``numpy``, ``scipy``,
``matplotlib``, ``pyFAI``, ``h5py``, ``fabio``, ``PySide6``).  Subcommand
handlers live under :mod:`butterfly_saxs._cli` and are imported only inside
:func:`main` after argument parsing succeeds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._cli.common import (
    _BRIGHT_AXIS_CONVENTIONS,
    _INTENSITY_MODES,
    _JOURNAL_STYLES,
    _REPORT_DEFAULT_FORMATS,
    _REPORT_FORMATS,
    _BatchFrameAnalyzer,
    add_refinement_options as _add_refinement_options,
    analysis_overrides as _analysis_overrides,
    emit_error as _emit_error,
    filter_supported_kwargs as _filter_supported_kwargs,
    integer_at_least as _integer_at_least,
    parse_beamstop_radius as _beamstop_radius,
    parse_shape as _shape,
    pipeline_symbol as _pipeline_symbol,
    print_json as _print_json,
)
from .cancellation import AnalysisCancelled
from .cli_contract import usage_error_payload
from .errors import PipelineError
from .project import ProjectConfigError

# Patch seams for tests. ``None`` means "import from target module on demand".
analyze_frame = None
inspect_frame = None
run_project = None
synthetic_butterfly = None
launch_gui = None
advise_frame = None
run_campaign = None

_REPORT_BIN_COUNT = _integer_at_least(2, "report bin 数")
_REPORT_DPI = _integer_at_least(1, "report dpi")


def _build_analysis_report(
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

    from ._cli.reporting_cmd import build_analysis_report_cmd

    return build_analysis_report_cmd(
        output_dir,
        resume=resume,
        force=force,
        radial_bins=radial_bins,
        angular_bins=angular_bins,
        formats=formats,
        dpi=dpi,
        lamellar_settings=lamellar_settings,
        journal_style=journal_style,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsaxs",
        description=(
            "WingSAXS：蝴蝶状二维 SAXS 花样的定量测量与椭圆精修。"
            "无子命令时打印 agent 清单（bsaxs describe）。"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)

    describe_parser = sub.add_parser(
        "describe", help="打印机器可读的命令、退出码与科学边界清单"
    )
    describe_parser.add_argument(
        "topic", nargs="?", help="只返回指定命令的契约，例如 describe batch"
    )
    describe_parser.add_argument(
        "--text",
        action="store_true",
        help="同时在 stderr 打印简短人类可读摘要；stdout 仍是 JSON",
    )

    doctor_parser = sub.add_parser(
        "doctor", help="检查 Python 与依赖（同 bsaxs-doctor）"
    )
    doctor_parser.add_argument(
        "--require-ui",
        action="store_true",
        help="Treat PySide6 and pyqtgraph as required.",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Print a strict machine-readable JSON report.",
    )
    doctor_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print nothing; communicate readiness through the exit code.",
    )

    inspect_parser = sub.add_parser("inspect", help="检查图像、q 空间和基础观测量")
    inspect_parser.add_argument("input", nargs="?", help="CBF/EDF/TIF/NPY/NPZ/HDF5 图像")
    inspect_parser.add_argument(
        "-i", "--input-file", dest="input_file", help="输入图像（input 的显式别名）"
    )
    inspect_parser.add_argument("-c", "--config", help="TOML 项目配置")
    inspect_parser.add_argument("--poni", help="PONI 几何文件")
    inspect_parser.add_argument("--frame", type=int, help="多帧文件中的零基帧索引")
    inspect_parser.add_argument("--dataset", help="HDF5/NPZ 数据集或 TIFF series:N")
    inspect_parser.add_argument("--mask", help="外部掩膜路径（True=无效像素）")
    inspect_parser.add_argument("--mask-frame", type=int, help="多帧掩膜中的零基帧索引")
    inspect_parser.add_argument("--mask-dataset", help="HDF5/NPZ 掩膜数据集或键")
    inspect_parser.add_argument(
        "--valid-mask", dest="valid_mask", help="有效像素掩膜路径（True=有效像素）"
    )
    inspect_parser.add_argument(
        "--recommend",
        action="store_true",
        help="附带智能参数、光束阻挡器与对称性建议（若 auto_advisor 可用）",
    )
    inspect_parser.add_argument(
        "--nominal-draw-axis-deg",
        type=float,
        default=None,
        help="配合 --recommend 使用的名义拉伸轴角度（度，默认 90.0）",
    )
    inspect_parser.add_argument(
        "--beamstop-radius-px",
        type=_beamstop_radius,
        default=None,
        metavar="RADIUS|auto",
        help="可选光束阻挡器排除半径（非负像素数值或 auto；仅在内存中叠加，不修改原始 mask）",
    )
    inspect_parser.add_argument("-o", "--output", help="可选 JSON 输出路径")
    inspect_parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")

    analyze_parser = sub.add_parser("analyze", help="分析单帧并拟合对称双椭圆")
    analyze_parser.add_argument("input", nargs="?", help="输入图像")
    analyze_parser.add_argument(
        "-i", "--input-file", dest="input_file", help="输入图像（input 的显式别名）"
    )
    analyze_parser.add_argument("-c", "--config", help="TOML 项目配置")
    analyze_parser.add_argument("--poni", help="PONI 几何文件")
    analyze_parser.add_argument("--frame", type=int, help="多帧文件中的零基帧索引")
    analyze_parser.add_argument("--dataset", help="HDF5/NPZ 数据集或 TIFF series:N")
    analyze_parser.add_argument("--mask", help="外部掩膜路径（True=无效像素）")
    analyze_parser.add_argument("--mask-frame", type=int, help="多帧掩膜中的零基帧索引")
    analyze_parser.add_argument("--mask-dataset", help="HDF5/NPZ 掩膜数据集或键")
    analyze_parser.add_argument(
        "--valid-mask", dest="valid_mask", help="有效像素掩膜路径（True=有效像素）"
    )
    analyze_parser.add_argument(
        "--beamstop-radius-px",
        type=_beamstop_radius,
        default=None,
        metavar="RADIUS|auto",
        help="可选光束阻挡器排除半径（非负像素数值或 auto；仅在内存中叠加，不修改原始 mask）",
    )
    analyze_parser.add_argument(
        "--uncertainty-dataset",
        default=None,
        metavar="PATH|auto",
        help="HDF5 伴随统计不确定度数据集路径，或 auto 自动探测（例如 /entry/data/uncertainty/statistical）",
    )
    analyze_parser.add_argument(
        "--include-physics",
        action="store_true",
        default=False,
        help="附加扩展 2D SAXS 物理观测量汇总（方位谐波/C2-bridge+4-lobe、径向粗化、Q_win 与条纹展宽）",
    )
    analyze_parser.add_argument(
        "--intensity-mode",
        choices=_INTENSITY_MODES,
        default=None,
        help="强度标定模式（calibrated 或 raw；raw 模式下不报告无物理尺度的 Q_win）",
    )
    analyze_parser.add_argument("-o", "--output", help="JSON/NPZ 文件或输出目录")
    analyze_parser.add_argument(
        "--full2d", action="store_true", help="调用可选的 full2d 精修模块"
    )
    analyze_parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")
    analyze_parser.add_argument(
        "--figure-output",
        help="另存基于测量数组的科研图包到新目录（SVG/PDF/TIFF/PNG 与源数据）",
    )
    analyze_parser.add_argument(
        "--figure-width",
        type=float,
        choices=(89.0, 183.0),
        default=183.0,
        help="科研图画板宽度，单位 mm（默认双栏 183）",
    )
    analyze_parser.add_argument(
        "--figure-dpi",
        type=int,
        choices=(300, 600, 1200),
        default=600,
        help="科研图中栅格内容的分辨率（默认 600 dpi）",
    )
    _add_refinement_options(analyze_parser)

    package_parser = sub.add_parser(
        "package", help="汇总已有批次的数据和图，无需重新拟合或绘图"
    )
    package_parser.add_argument(
        "output_dir", help="单批次输出或包含各样品输出的父目录"
    )
    package_parser.add_argument(
        "--resume", action="store_true", help="续做交付；复用未改变的 ZIP"
    )
    package_parser.add_argument(
        "--force", action="store_true", help="重建已生成的交付索引和 ZIP"
    )
    package_parser.add_argument(
        "--no-archive",
        "--no-archives",
        action="store_true",
        help="只建立导航；明确跳过 ZIP",
    )

    report_parser = sub.add_parser(
        "report", help="从已有批次 NPZ 导出统计和图表，不重新拟合"
    )
    report_parser.add_argument(
        "output_dir", help="单批次输出目录或包含多个样品输出的父目录"
    )
    report_parser.add_argument(
        "--resume", action="store_true", help="续做报告；复用设置和输入未改变的输出"
    )
    report_parser.add_argument("--force", action="store_true", help="重建已有报告")
    report_parser.add_argument(
        "--radial-bins",
        type=_REPORT_BIN_COUNT,
        default=128,
        help="径向汇总箱数（默认 128，至少 2）",
    )
    report_parser.add_argument(
        "--angular-bins",
        type=_REPORT_BIN_COUNT,
        default=72,
        help="角向汇总箱数（默认 72，至少 2）",
    )
    report_parser.add_argument(
        "--formats",
        nargs="+",
        choices=_REPORT_FORMATS,
        default=_REPORT_DEFAULT_FORMATS,
        help="图表格式（可选 png、svg、pdf、tiff；默认 png svg pdf）",
    )
    report_parser.add_argument(
        "--dpi",
        type=_REPORT_DPI,
        default=180,
        help="栅格图分辨率（默认 180）",
    )
    report_parser.add_argument(
        "--lamellar-settings",
        help="JSON/TOML 片层示意参数；椭圆派生周期和绘制假设可在此设置",
    )
    report_parser.add_argument(
        "--journal-style",
        choices=_JOURNAL_STYLES,
        default="default",
        help="期刊出版风格预设（default/nature/nature_comm/science/macromolecules；默认保持原有风格）",
    )

    delivery_check = sub.add_parser(
        "verify-delivery",
        help="只读核对所选结项回执的当前文件绑定，不改写历史或科学状态",
    )
    delivery_check.add_argument(
        "receipt", help="包含顶层 bindings 列表的现有结项 JSON"
    )
    delivery_check.add_argument(
        "--root", help="绑定相对路径的输出根目录；默认回执所在目录"
    )

    batch_parser = sub.add_parser("batch", help="批量分析原位序列")
    batch_parser.add_argument("inputs", nargs="*", help="输入图像或通配符")
    batch_parser.add_argument(
        "-c", "--config", help="TOML 项目配置（可提供 inputs.files）"
    )
    batch_parser.add_argument("--poni", help="PONI 几何文件")
    batch_parser.add_argument("--frame", type=int, help="多帧文件中的零基帧索引")
    batch_parser.add_argument("--dataset", help="HDF5/NPZ 数据集或 TIFF series:N")
    batch_parser.add_argument("--mask", help="外部掩膜路径（True=无效像素）")
    batch_parser.add_argument("--mask-frame", type=int, help="多帧掩膜中的零基帧索引")
    batch_parser.add_argument("--mask-dataset", help="HDF5/NPZ 掩膜数据集或键")
    batch_parser.add_argument(
        "--valid-mask", dest="valid_mask", help="有效像素掩膜路径（True=有效像素）"
    )
    batch_parser.add_argument("-o", "--output", help="输出目录")
    batch_parser.add_argument(
        "--full2d", action="store_true", help="调用可选的 full2d 精修模块"
    )
    batch_parser.add_argument(
        "--mode", choices=("independent", "warm_start"), help="序列拟合模式"
    )
    batch_parser.add_argument(
        "--manifest", help="JSON/CSV 帧清单（含 time/frame_id 等元数据）"
    )
    batch_parser.add_argument("--checkpoint", help="批量检查点 JSON 路径")
    batch_parser.add_argument(
        "--package",
        action="store_true",
        help="导出后继续生成可浏览导航和交付 ZIP；交付失败可单独续做",
    )
    batch_parser.add_argument(
        "--report", action="store_true", help="数据导出后从 NPZ 生成统计和图表报告"
    )
    batch_parser.add_argument(
        "--report-radial-bins",
        type=_REPORT_BIN_COUNT,
        default=128,
        help="--report 的径向汇总箱数（默认 128，至少 2）",
    )
    batch_parser.add_argument(
        "--report-angular-bins",
        type=_REPORT_BIN_COUNT,
        default=72,
        help="--report 的角向汇总箱数（默认 72，至少 2）",
    )
    batch_parser.add_argument(
        "--report-formats",
        nargs="+",
        choices=_REPORT_FORMATS,
        default=_REPORT_DEFAULT_FORMATS,
        help="--report 的图表格式（默认 png svg pdf）",
    )
    batch_parser.add_argument(
        "--report-dpi",
        type=_REPORT_DPI,
        default=180,
        help="--report 的栅格图分辨率（默认 180）",
    )
    batch_parser.add_argument(
        "--report-lamellar-settings",
        help="--report 使用的 JSON/TOML 片层示意参数；拟合前加载并校验",
    )
    batch_parser.add_argument(
        "--resume", action="store_true", help="从已有检查点恢复"
    )
    batch_parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")
    batch_parser.add_argument(
        "--unattended",
        metavar="PACKAGE",
        help="先预检数据包，再以流式导出和自动检查点运行；预检红灯阻止拟合",
    )
    batch_parser.add_argument(
        "--preflight-context", help="无人值守预检的 project_context.yaml/yml"
    )
    batch_parser.add_argument(
        "--stream",
        action="store_true",
        help="逐帧写出参数/脊线和 NPZ，释放已处理帧的 detector 数组",
    )
    batch_parser.add_argument("--series", help="只处理 manifest 中指定的 series/group")
    batch_parser.add_argument(
        "--start", type=int, help="选中有序序列的起始位置（含）"
    )
    batch_parser.add_argument(
        "--stop", type=int, help="选中有序序列的结束位置（含）"
    )
    batch_parser.add_argument("--stride", type=int, default=None, help="有序序列步长")
    batch_parser.add_argument(
        "--range",
        dest="frame_range",
        help="序列范围 START:STOP[:STEP]，STOP 包含在内",
    )
    batch_parser.add_argument(
        "--beamstop-radius-px",
        type=_beamstop_radius,
        default=None,
        metavar="RADIUS|auto",
        help="可选光束阻挡器排除半径（非负像素数值或 auto；仅在内存中叠加，不修改原始 mask）",
    )
    batch_parser.add_argument(
        "--deduplicate-frames",
        action="store_true",
        default=False,
        help="对像素内容 SHA-256 完全相同的重复帧复用计算结果，保留每帧原始位置与来源身份",
    )
    batch_parser.add_argument(
        "--workers",
        type=_integer_at_least(1, "workers"),
        default=None,
        metavar="N",
        help="独立拟合模式并行 worker 数（映射到 max_workers；默认 1，warm_start 自动顺序执行）",
    )
    _add_refinement_options(batch_parser)

    synthetic_parser = sub.add_parser(
        "synthetic", help="生成可重复的蝴蝶状二维测试花样"
    )
    synthetic_parser.add_argument(
        "-o", "--output", help=".npy/.npz/.tif 输出路径；不提供则只打印摘要"
    )
    synthetic_parser.add_argument(
        "--shape", type=_shape, default=(128, 128), help="图像尺寸 HxW"
    )
    synthetic_parser.add_argument(
        "--q0", type=float, default=28.0, help="椭圆特征 q 半径"
    )
    synthetic_parser.add_argument(
        "--width", type=float, default=2.0, help="峰脊宽度"
    )
    synthetic_parser.add_argument(
        "--ellipticity", type=float, default=2.0, help="椭圆长短轴比"
    )
    synthetic_parser.add_argument(
        "--angle", type=float, default=28.0, help="对称椭圆角度（度）"
    )
    synthetic_parser.add_argument(
        "--noise", type=float, default=0.0, help="高斯噪声标准差"
    )
    synthetic_parser.add_argument("--seed", type=int, default=0, help="随机种子")
    synthetic_parser.add_argument(
        "--force", action="store_true", help="允许覆盖已有输出"
    )

    gui_parser = sub.add_parser(
        "gui", help="打开与 CLI 共用 pipeline seam 的交互界面"
    )
    gui_parser.add_argument("input", nargs="?", help="可选输入图像")
    gui_parser.add_argument("-c", "--config", help="可选 TOML 项目配置")
    gui_parser.add_argument("--poni", help="PONI 几何文件")

    project_parser = sub.add_parser("project", help="执行 TOML 项目配置")
    project_parser.add_argument("config", help="项目 TOML 文件")
    project_parser.add_argument(
        "--force", action="store_true", help="允许覆盖已有输出"
    )
    project_parser.add_argument(
        "--legacy-json",
        action="store_true",
        help="以旧版 per-frame JSON 列表输出；默认输出带 schema_version 的批处理 envelope",
    )

    preflight_parser = sub.add_parser(
        "preflight", help="只读检查真实数据包、几何、掩膜、单位与清单"
    )
    preflight_parser.add_argument("package", help="真实数据包根目录")
    preflight_parser.add_argument("--manifest", help="CSV/JSON/TOML 帧清单")
    preflight_parser.add_argument("--poni", help="PONI 几何文件")
    preflight_parser.add_argument("--mask", help="外部掩膜文件")
    preflight_parser.add_argument("--context", help="project_context.yaml/yml")
    preflight_parser.add_argument("--image-glob", help="未使用清单时的图像通配符")
    preflight_parser.add_argument("--frame", type=int, help="图像多帧选择器")
    preflight_parser.add_argument(
        "--dataset", help="图像 HDF5/NPZ 数据集或 TIFF series:N"
    )
    preflight_parser.add_argument("--mask-frame", type=int, help="掩膜多帧选择器")
    preflight_parser.add_argument(
        "--mask-dataset", help="掩膜 HDF5/NPZ 数据集或 TIFF series:N"
    )
    preflight_parser.add_argument(
        "--q-window", type=float, nargs=2, metavar=("Q_MIN", "Q_MAX")
    )
    preflight_parser.add_argument(
        "--mask-convention",
        choices=("0_valid_1_invalid", "1_valid_0_invalid"),
    )
    preflight_parser.add_argument("--correction-state")
    preflight_parser.add_argument("--uncertainty-state")
    preflight_parser.add_argument("-o", "--output", help="预检证据输出目录")
    preflight_parser.add_argument(
        "--force", action="store_true", help="允许覆盖已有预检输出"
    )

    benchmark_parser = sub.add_parser(
        "benchmark", help="生成 P3 的 T1 同模型或 T2 独立 FFT 基准证据"
    )
    benchmark_parser.add_argument(
        "--suite", choices=("t1", "t2", "all"), default="all", help="要生成的基准套件"
    )
    benchmark_parser.add_argument(
        "-o", "--output", required=True, help="新的证据输出目录"
    )
    benchmark_parser.add_argument(
        "--shape", type=_shape, help="可选统一图像尺寸 HxW"
    )
    benchmark_parser.add_argument("--seed", type=int, help="可选统一起始随机种子")
    benchmark_parser.add_argument(
        "--force", action="store_true", help="仅覆盖本命令的目标文件"
    )

    annotation_parser = sub.add_parser(
        "annotation-pack", help="从 R0 清单生成 8 帧只读盲标包，不运行拟合"
    )
    annotation_parser.add_argument("package", help="真实数据包根目录")
    annotation_parser.add_argument(
        "--rt-manifest", required=True, help="室温参考帧清单"
    )
    annotation_parser.add_argument(
        "--hold-manifest", required=True, help="保温序列帧清单"
    )
    annotation_parser.add_argument(
        "--preflight", help="可选原始强度预检 JSON，用于困难帧排序"
    )
    annotation_parser.add_argument(
        "--poni", help="PONI 文件，只记录来源而不参与选择"
    )
    annotation_parser.add_argument(
        "--mask", help="mask 文件，只记录来源而不参与选择"
    )
    annotation_parser.add_argument(
        "-o", "--output", required=True, help="新的盲标包输出目录"
    )

    p3_parser = sub.add_parser("p3-status", help="只读评估 P3 Go/No-Go 证据门")
    p3_parser.add_argument(
        "--t1-manifest", required=True, help="T1 truth_manifest.json"
    )
    p3_parser.add_argument(
        "--t2-manifest", required=True, help="T2 truth_manifest.json"
    )
    p3_parser.add_argument(
        "--annotation-status", required=True, help="annotation_status.json"
    )
    p3_parser.add_argument(
        "--thresholds", required=True, help="阈值 JSON（draft 或 frozen）"
    )
    p3_parser.add_argument("-o", "--output", help="可选门禁报告 JSON")
    p3_parser.add_argument(
        "--force", action="store_true", help="允许覆盖已有门禁报告"
    )

    p4_parser = sub.add_parser(
        "p4-evaluate", help="运行 P4 ridge/lobe/双椭圆工程证据（不冒充科学验收）"
    )
    p4_parser.add_argument(
        "--t1-manifest", required=True, help="T1 truth_manifest.json"
    )
    p4_parser.add_argument(
        "--t2-manifest", required=True, help="T2 truth_manifest.json"
    )
    p4_parser.add_argument(
        "--thresholds", required=True, help="draft/frozen 阈值 JSON"
    )
    p4_parser.add_argument(
        "-o", "--output", required=True, help="新的 P4 证据输出目录"
    )
    p4_parser.add_argument("--r0-package", help="可选 R0 原始数据包根目录")
    p4_parser.add_argument(
        "--r0-manifest", help="可选固定 8 帧 annotation_manifest.csv"
    )
    p4_parser.add_argument("--poni", help="R0 使用的 PONI 文件")
    p4_parser.add_argument("--mask", help="R0 使用的外部 mask")
    p4_parser.add_argument(
        "--ridge-method",
        choices=("radial_peak", "surface_curvature"),
        default="radial_peak",
    )
    p4_parser.add_argument(
        "--skip-sensitivity", action="store_true", help="跳过单个 T1 方法敏感性对照"
    )

    init_manifest_parser = sub.add_parser(
        "init-manifest",
        help="从文件夹或 HDF5 帧栈发现自然排序的帧并生成校验后的 manifest.csv",
    )
    init_manifest_parser.add_argument(
        "input_path",
        nargs="?",
        help="输入图像目录或 HDF5/图像文件（也可用 --input）",
    )
    init_manifest_parser.add_argument(
        "-i",
        "--input",
        dest="input_dir",
        help="输入图像目录或 HDF5/图像文件",
    )
    init_manifest_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出 manifest.csv 路径（与原始数据分开，未提供 --force 时不覆盖）",
    )
    init_manifest_parser.add_argument("--dataset", help="HDF5/NPZ 数据集路径")
    init_manifest_parser.add_argument(
        "--pattern", default="*", help="目录扫描通配符（默认 *）"
    )
    init_manifest_parser.add_argument(
        "--recursive", action="store_true", help="递归扫描子目录"
    )
    init_manifest_parser.add_argument(
        "--no-parse-metadata",
        dest="parse_metadata",
        action="store_false",
        default=True,
        help="关闭文件名元数据解析",
    )
    init_manifest_parser.add_argument(
        "--sha256", action="store_true", help="计算每个源文件的 SHA-256 摘要"
    )
    init_manifest_parser.add_argument(
        "--relative",
        action="store_true",
        help="在 manifest 中记录相对输出目录的路径",
    )
    init_manifest_parser.add_argument(
        "--force", action="store_true", help="允许覆盖已有 manifest"
    )
    init_manifest_parser.add_argument(
        "--json",
        action="store_true",
        help="以严格 JSON 输出摘要（stdout 默认即为严格 JSON）",
    )

    advise_parser = sub.add_parser(
        "advise",
        help="运行单帧智能参数、光束阻挡器与 Friedel 对称性诊断建议（只读，不拟合）",
    )
    advise_parser.add_argument(
        "image_pos", nargs="?", help="输入图像路径（也可用 --image）"
    )
    advise_parser.add_argument(
        "-i", "--image", dest="image", help="输入二维图像路径"
    )
    advise_parser.add_argument("--poni", help="PONI 几何标定文件")
    advise_parser.add_argument("--mask", help="外部掩膜路径（True=无效像素）")
    advise_parser.add_argument(
        "--image-dataset",
        "--dataset",
        dest="image_dataset",
        help="HDF5/NPZ 图像数据集路径",
    )
    advise_parser.add_argument(
        "--image-frame-index",
        "--frame",
        dest="image_frame_index",
        type=int,
        help="多帧图像中的零基帧索引",
    )
    advise_parser.add_argument("--mask-dataset", help="HDF5/NPZ 掩膜数据集路径")
    advise_parser.add_argument(
        "--mask-frame-index",
        "--mask-frame",
        dest="mask_frame_index",
        type=int,
        help="多帧掩膜中的零基帧索引",
    )
    advise_parser.add_argument(
        "--nominal-draw-axis-deg",
        type=float,
        default=90.0,
        help="名义拉伸轴方向（度，默认 90.0）",
    )
    advise_parser.add_argument(
        "--bright-axis-convention",
        choices=_BRIGHT_AXIS_CONVENTIONS,
        default="draw",
        help="预期亮桥/对称轴约定：draw（默认，与拉伸轴一致）或 reference（垂直拉伸轴，适用于赤道翼）",
    )
    advise_parser.add_argument("-o", "--output", help="可选 JSON 输出路径")
    advise_parser.add_argument(
        "--force", action="store_true", help="允许覆盖已有输出"
    )
    advise_parser.add_argument(
        "--json",
        action="store_true",
        help="以严格 JSON 输出诊断建议（stdout 默认即为严格 JSON）",
    )

    campaign_parser = sub.add_parser(
        "campaign",
        help="执行多样品活动 TOML 配置，隔离单样品失败并输出严格 JSON 汇总",
    )
    campaign_parser.add_argument(
        "config_pos", nargs="?", help="活动 TOML 路径（也可用 --config）"
    )
    campaign_parser.add_argument(
        "-c", "--config", dest="config", help="活动 TOML 配置文件路径"
    )
    campaign_parser.add_argument(
        "-o",
        "--output",
        help="多样品输出根目录（未提供 --force 时不覆盖非空目录）",
    )
    campaign_parser.add_argument(
        "--force", action="store_true", help="允许覆盖已有输出"
    )
    campaign_parser.add_argument(
        "--fail-fast",
        "--no-continue-on-error",
        dest="continue_on_error",
        action="store_false",
        default=True,
        help="首个样品失败后停止后续样品（默认隔离失败并继续）",
    )
    campaign_parser.add_argument(
        "--report",
        action="store_true",
        default=None,
        help="完成后生成多样品汇总报告",
    )
    campaign_parser.add_argument(
        "--journal-style",
        choices=_JOURNAL_STYLES,
        default="default",
        help="汇总报告期刊风格预设",
    )
    campaign_parser.add_argument(
        "--beamstop-radius-px",
        type=_beamstop_radius,
        default=None,
        metavar="RADIUS|auto",
        help="可选光束阻挡器排除半径（非负像素数值或 auto；仅在内存中叠加，不修改原始 mask）",
    )
    campaign_parser.add_argument(
        "--deduplicate-frames",
        action="store_true",
        default=False,
        help="对像素内容 SHA-256 完全相同的重复帧复用计算结果，保留每帧原始位置与来源身份",
    )
    campaign_parser.add_argument(
        "--workers",
        type=_integer_at_least(1, "workers"),
        default=None,
        metavar="N",
        help="多样品并行 worker 数（映射到 max_workers；默认 1 顺序执行）",
    )
    campaign_parser.add_argument(
        "--json",
        action="store_true",
        help="以严格 JSON 输出汇总（stdout 默认即为严格 JSON）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = 0 if exc.code is None else int(exc.code)
        if code == 0:
            raise
        _print_json(
            usage_error_payload(
                "invalid command or arguments; run `bsaxs describe` or `bsaxs --help`"
            )
        )
        return 2

    command = args.command or "describe"
    try:
        if command in {"describe", "doctor", "inspect", "analyze", "advise", "synthetic", "gui"}:
            from ._cli import single

            single_handlers = {
                "describe": single.handle_describe,
                "doctor": single.handle_doctor,
                "inspect": single.handle_inspect,
                "analyze": single.handle_analyze,
                "advise": single.handle_advise,
                "synthetic": single.handle_synthetic,
                "gui": single.handle_gui,
            }
            return single_handlers[command](args)
        if command in {"preflight", "package", "verify-delivery"}:
            from ._cli import delivery_cmd

            delivery_handlers = {
                "preflight": delivery_cmd.handle_preflight,
                "package": delivery_cmd.handle_package,
                "verify-delivery": delivery_cmd.handle_verify_delivery,
            }
            return delivery_handlers[command](args)
        if command in {"batch", "project", "init-manifest", "campaign"}:
            from ._cli import batch_cmd

            batch_handlers = {
                "batch": batch_cmd.handle_batch,
                "project": batch_cmd.handle_project,
                "init-manifest": batch_cmd.handle_init_manifest,
                "campaign": batch_cmd.handle_campaign,
            }
            return batch_handlers[command](args)
        if command in {"report", "benchmark", "annotation-pack", "p3-status", "p4-evaluate"}:
            from ._cli import reporting_cmd

            reporting_handlers = {
                "report": reporting_cmd.handle_report,
                "benchmark": reporting_cmd.handle_benchmark,
                "annotation-pack": reporting_cmd.handle_annotation_pack,
                "p3-status": reporting_cmd.handle_p3_status,
                "p4-evaluate": reporting_cmd.handle_p4_evaluate,
            }
            return reporting_handlers[command](args)
        parser.error(f"未知命令：{command}")
    except AnalysisCancelled as exc:
        _emit_error(exc, exit_code=1, command=command, code="cancelled")
        return 1
    except (
        PipelineError,
        ProjectConfigError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        _emit_error(exc, exit_code=2, command=command)
        return 2
    return 2


__all__ = [
    "PipelineError",
    "_BatchFrameAnalyzer",
    "_analysis_overrides",
    "_build_analysis_report",
    "_filter_supported_kwargs",
    "_pipeline_symbol",
    "_print_json",
    "build_parser",
    "main",
]
