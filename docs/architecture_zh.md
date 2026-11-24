# 软件架构与数据流

使用入口：[操作、输入输出、UI 与批处理指南](user_guide_zh.md)；科学量和结构解释边界见[科学量、符号、单位与可解释性边界](scientific_basis_zh.md)。

```text
CBF / EDF / TIF / TIFF / NPY / NPZ / HDF5 (+ 伴随 uncertainty) / CSV/TXT
              + PONI + mask + 可选 sigma/weights / beamstop_radius_px
                         |
                         v
  只读 preflight/ + auto_advisor (bsaxs advise) + init-manifest + 输入 SHA-256
                         |
                         v
  ImageFrame + QMap / SAXSGeometryCache (PONI: nm^-1;
         无 PONI/嵌入 qmap: pixel-q + flag; 内存叠加 beamstop 掩膜)
                         |
          +--------------+--------------+--------------+
          |                             |              |
          v                             v              v
  多尺度观测量与真实 ridge       扩展 2D 物理量       完整二维经验强度场
  (_observables_* / tracing/)   (physics/ &          （可选 --full2d:
  (annular/curvature/sector)    saxs_physics.py:      intensity.py）
          |                     C2-bridge+4-lobe,          |
          +------- 对称双椭圆    Q_win, 粗化, W)            |
          |        bounds/精修          |                  |
          +--------------+--------------+------------------+
                         |
            observed / model / residual / sequence_diagnostics
                         |
       单帧证据包 / 批次演化表 (batch.py, export.py + _export_schema.py) / campaign.py
                         |
    分析报告 (report.py + lamellar/)、出版复合图 (journal_style / composite_figures /
    comparison_report / butterfly_figure / publication_export) 与交付核验 (delivery/)
```

## 包目录布局（`src/butterfly_saxs/`）

```text
src/butterfly_saxs/
├── cli.py + _cli/              # 纯解析与零科学库懒加载入口；按子命令延迟导入 _cli/{common,single,batch_cmd,delivery_cmd,reporting_cmd}
├── service.py / pipeline.py    # 无 Qt 统一分析服务层与单帧/项目执行管线；campaign.py 负责多样品编排与清单发现
├── preflight/ (+ preflight_*.py 兼容薄垫片)
│   └── runner, context, io, geometry, uncertainty, checks, hash, artifacts, report
├── delivery/ (+ delivery_*.py 兼容薄垫片)
│   └── package, dependencies, bindings, pairs
├── _primitives.py              # 跨模块共享的无重依赖数值/角度/安全转换基础原语
├── observables.py (+ _observables_{types,angular,ridges,ellipse}.py)
├── tracing/ (+ butterfly*.py)  # 蝴蝶弧段追踪（annular_peak / radial_sector / curvature）、法向剖面与不确定度评估
├── physics/ (+ saxs_physics.py)# 方位谐波、直接/Kratky 分列径向粗化、标定强度 Q_win、条纹展宽与动力学拟合
├── lamellar/ (+ lamellar_*.py 兼容薄垫片)
│   └── 片层几何模型、椭圆条件表观周期分析、三维示意网格与报告/出版层导出
├── batch.py / export.py / _export_schema.py / producer.py
│   └── 序列执行、流式/批量共享列契约导出与递归子包计算源码身份校验
├── serialization.py / file_identity.py / path_utils.py / figure_support.py
│   └── 共享基础设施：严格 JSON 原子写入、流式 SHA-256 与文件版本证据、原子暂存目录发布、无 pyplot 的 Agg 画布保存
└── ui/                         # 桌面 Qt/PySide6 交互工作台；后台 worker 严格不接触 QWidget
```

## 模块边界

- `cli.py` 与 `_cli/` + `cli_contract.py`：面向用户与自动化的严格 JSON CLI 入口（`allow_nan=False`、ASCII 转义）；`cli.py` 仅负责参数解析、错误信封与延迟分发，`import butterfly_saxs.cli`、`--help`、`describe` 及 `doctor` 严格保持零重型科学/UI 库懒加载（不导入 `numpy`/`scipy`/`matplotlib`/`pyFAI`/`PySide6`/`h5py`/`fabio`）。
- `io.py`：只负责读取原始数组、frame、dataset 和元数据，不做归一化或物理解释；`valid_mask=True` 为有效像素，外部 `mask=True` 为无效像素。显式 HDF5 `dataset` 直接访问数据集（避免全树扫描），并支持可选 `uncertainty_dataset`（含 `"auto"` 探测 `/entry/data/uncertainty/statistical` 等伴随统计不确定度）。
- `validation.py`：定义严格结果 schema 和统一 `analysis-domain`；测量、ridge、ellipse、`full2d` 与导出共享同一最终像素集合。
- `service.py`/`pipeline.py`：把 CLI、Qt 与批处理统一到同一 `analysis` 合同（`service.py` 严格保持无 Qt）；`flat_ellipse` 的 ratio/a/b/angle/center 约束传入观测 ridge 椭圆拟合，`geometry-only` 操作不会隐式启动 `full2d`。支持内存叠加 `beamstop_radius_px`（数值或 `"auto"`，不修改原始 mask 文件）、可选 `uncertainty_dataset`、`include_physics` 与 `intensity_mode`（`calibrated`/`raw`）。
- `preflight/`（及顶层 `preflight_*.py` 兼容重导出垫片）：在拟合前只读核验 package、manifest、PONI、mask、单位、校正/不确定度状态和读写前后 SHA-256，按统一规则递归计算核心计算源码树摘要（`source_tree_sha256`），生成 green/yellow/red 证据。
- `auto_advisor.py`：只读预检与参数建议模块（`bsaxs advise` 与 `bsaxs inspect --recommend`），估计光束阻挡器半径、有效外层 q 截断与推荐 `q_window`、Friedel 对称性/正交翻转风险（支持 `bright_axis_convention="draw"|"reference"`）以及工作流预设；仅提供拟合前诊断，不执行拟合也不构成科学验收。
- `geometry.py`：只通过 PONI/pyFAI 产生 `q/chi/qx/qy`，不重新发明探测器旋转公式；提供 `SAXSGeometryCache`（缓存束心、`pixel_dq`、一维广播 `apply_beamstop_mask` 与有效 q ROI 切片），避免序列分析重复校验与计算全图坐标。
- `physics/` 与 `saxs_physics.py`：提供扩展二维 SAXS 物理观测量与动力学辅助：方位角 Fourier 谐波（`A_2`/`A_4`/`P_2`）与 `C2-bridge + 4-lobe` 拟合、严格分列的直接 `I(q)` 峰与 Kratky `q^2 I(q)` 峰径向粗化、仅在标定强度下计算的窗口受限不变量 `Q_win`（`intensity_mode="raw"` 时强制返回 `null`；支持 `sigma` 统计标准误传播）、垂直条纹宽度与 Grubb/Williamson-Hall 表观横向宽度 `W`、幂律粗化与 Arrhenius 速率拟合，以及窗口敏感性审计（`run_window_sensitivity_audit`）。
- `observables.py`（与 `_primitives.py`、`_observables_*.py`）及 `tracing/`（与 `butterfly*.py`）：提取剖面、lobe、ridge、多方法蝴蝶弧段轨迹与直接派生量，严格保留观测支持度与缺失状态。
- `ellipse.py`：只拟合倒易空间曲线，返回残差、协方差和可辨识性；圆环 `q*` / `L = 2π/q*` 与椭圆半轴 `a`、`b/a`、轴角及条件 `Ln/Lz` 严格分离。
- `intensity.py`：生成和拟合经验二维强度场；当前 `full2d` 是共享双椭圆、角向包络、径向线形和背景的经验模型，不是 Grubb 2016 的完整 3D 取向正演。
- `sequence_diagnostics.py`：只读序列级异常与质量诊断，计算解码像素数组 SHA-256 哈希以检测重复/冻结帧，并标记时间间隔异常、轴比塌缩、q 窗口边界命中、正交翻转和覆盖率跌落；保留全部源帧序号，从不插值或替换失败帧。
- `batch.py`、`export.py` + `_export_schema.py` 与 `producer.py`：FrameRef 的容器 selector 与序列 selector 分开；输入、PONI、mask、beamstop 与不确定度文件内容参与 checkpoint 身份。`export_batch` 与 `StreamingBatchExporter` 共用 `_export_schema.py` 的列契约与行构建器，`--stream` 逐帧写 NPZ 并在 resume 时保留已验证的旧数组。`producer.py` 按 POSIX 相对路径递归哈希核心计算模块与计算子包（`physics/`、`tracing/` 等），同时将 `ui/`、`lamellar/`、`preflight/`、`delivery/`、`_cli/` 及纯展示/导出模块排除在检测器计算身份之外，避免展示层重构使有效计算检查点失效。
- `campaign.py`：工作流预设（`annular`/`curvature`/`sector`/`butterfly_pub`）、自然排序清单发现（`bsaxs init-manifest`，缺失时间戳留空不捏造），以及多样品活动编排（`bsaxs campaign`，支持单样品失败隔离、可选 `--workers` 并行与 `--beamstop-radius-px` / `--deduplicate-frames`）。
- `serialization.py`、`file_identity.py`、`path_utils.py`、`figure_support.py`：统一提供 `write_json_atomic` / `write_text_atomic`（`allow_nan=False`、同目录临时文件 + `os.replace`、`BaseException` 清理）、`sha256_bytes` / `sha256_file`、`atomic_stage_dir`（原子目录发布与覆盖备份回滚）以及无 `pyplot` / 无全局后端切换的 `save_agg_figure`，供报告、图包、盲标包与 P3/P4 门禁复用。
- `report.py`、`lamellar/`、`journal_style.py`、`composite_figures.py`、`comparison_report.py`、`butterfly_figure.py`、`publication_export.py` 与 `delivery/`：严格无 `matplotlib.pyplot` 的报告、出版级复合图与可离线浏览交付打包层；`delivery/`（`package`、`dependencies`、`bindings`、`pairs`）仅索引和核验既有导出证据，不重跑拟合也不赋予科学验收状态。
- `annotation_pack.py`、`manual_evidence.py`、`p3_gate.py`、`p4_quality.py`、`p4_validation.py`：独立于自动验收宣称的盲标包、人工复核证据与 P3/P4 门禁评估层；`scripts/` 下的开发/基准/验证脚本索引见 `scripts/README.md`。
- `ui/`：只管理交互和显示；后台 worker 不接触 QWidget。

## 输出原则

CSV 保存有界、可人工检查的表格；完整 `I/qx/qy/model/residual/fit_valid_mask/sampled_valid_mask` 保存在压缩 NPZ。统一分析域为 `finite(I,qx,qy) AND detector_valid AND NOT external_mask AND NOT ROI AND q_window AND weight_valid`，JSON 同时记录各阶段计数。批次的 `manifest.json`/`provenance.json` 保存输入/config hash、模式和版本；单帧 JSON 保存摘要，不能替代 NPZ 像素证据。执行成功与科学解释是两套状态：`fit_success=True` 不能清除 `mechanism_under_determined`、`apparent_geometry_only` 或 `nonunique_inverse_problem`。
