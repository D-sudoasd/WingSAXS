# 软件架构与数据流

使用入口：[操作、输入输出、UI 与批处理指南](user_guide_zh.md)；科学量和结构解释边界见[科学量、符号、单位与可解释性边界](scientific_basis_zh.md)。

```text
CBF / EDF / TIF / TIFF / NPY / NPZ / HDF5 (+ 伴随 uncertainty) / CSV/TXT
              + PONI + mask + 可选 sigma/weights / beamstop_radius_px
                         |
                         v
  只读 preflight / auto_advisor (bsaxs advise) / init-manifest + 输入 hash
                         |
                         v
  ImageFrame + QMap / SAXSGeometryCache (PONI: nm^-1;
         无 PONI/嵌入 qmap: pixel-q + flag; 内存叠加 beamstop 掩膜)
                         |
          +--------------+--------------+--------------+
          |                             |              |
          v                             v              v
  多尺度观测量与真实 ridge       扩展 2D 物理量       完整二维经验强度场
  (annular/curvature/sector)   (saxs_physics:       （可选 --full2d）
          |                     C2-bridge+4-lobe,          |
          +------- 对称双椭圆    Q_win, 粗化, W)            |
          |        bounds/精修          |                  |
          +--------------+--------------+------------------+
                         |
            observed / model / residual / sequence_diagnostics
                         |
       单帧证据包 / 批次演化表 (batch, dedup, workers) / 多样品 campaign
                         |
    分析报告、期刊复合图与 Source Data (journal_style / composite_figures / comparison_report)
```

## 模块边界

- `cli.py`/`cli_contract.py`：面向用户与自动化的严格 JSON CLI 入口（`allow_nan=False`、ASCII 转义）；`--help`、`describe` 及 `doctor` 模块加载严格保持零重型科学库懒加载（不导入 `numpy`/`scipy`/`matplotlib`/`pyFAI`/`PySide6`/`h5py`/`fabio`）。
- `io.py`：只负责读取原始数组、frame、dataset 和元数据，不做归一化或物理解释；`valid_mask=True` 为有效像素，外部 `mask=True` 为无效像素。显式 HDF5 `dataset` 直接访问数据集（避免全树扫描），并支持可选 `uncertainty_dataset`（含 `"auto"` 探测 `/entry/data/uncertainty/statistical` 等伴随统计不确定度）。
- `validation.py`：定义严格结果 schema 和统一 `analysis-domain`；测量、ridge、ellipse、`full2d` 与导出共享同一最终像素集合。
- `service.py`/`pipeline.py`：把 CLI、Qt 与批处理统一到同一 `analysis` 合同（`service.py` 严格保持无 Qt）；`flat_ellipse` 的 ratio/a/b/angle/center 约束传入观测 ridge 椭圆拟合，`geometry-only` 操作不会隐式启动 `full2d`。支持内存叠加 `beamstop_radius_px`（数值或 `"auto"`，不修改原始 mask 文件）、可选 `uncertainty_dataset`、`include_physics` 与 `intensity_mode`（`calibrated`/`raw`）。
- `preflight.py`：在拟合前只读核验 package、manifest、PONI、mask、单位、校正/不确定度状态和 SHA-256，并生成 green/yellow/red 证据。
- `auto_advisor.py`：只读预检与参数建议模块（`bsaxs advise` 与 `bsaxs inspect --recommend`），估计光束阻挡器半径、有效外层 q 截断与推荐 `q_window`、Friedel 对称性/正交翻转风险（支持 `bright_axis_convention="draw"|"reference"`）以及工作流预设；仅提供拟合前诊断，不执行拟合也不构成科学验收。
- `geometry.py`：只通过 PONI/pyFAI 产生 `q/chi/qx/qy`，不重新发明探测器旋转公式；提供 `SAXSGeometryCache`（缓存束心、`pixel_dq`、一维广播 `apply_beamstop_mask` 与有效 q ROI 切片），避免序列分析重复校验与计算全图坐标。
- `saxs_physics.py`：提供扩展二维 SAXS 物理观测量与动力学辅助：方位角 Fourier 谐波（`A_2`/`A_4`/`P_2`）与 `C2-bridge + 4-lobe` 拟合、严格分列的直接 `I(q)` 峰与 Kratky `q^2 I(q)` 峰径向粗化、仅在标定强度下计算的窗口受限不变量 `Q_win`（`intensity_mode="raw"` 时强制返回 `null`；支持 `sigma` 统计标准误传播）、垂直条纹宽度与 Grubb/Williamson-Hall 表观横向宽度 `W`、幂律粗化与 Arrhenius 速率拟合，以及窗口敏感性审计（`run_window_sensitivity_audit`）。
- `sequence_diagnostics.py`：只读序列级异常与质量诊断，计算解码像素数组 SHA-256 哈希以检测重复/冻结帧，并标记时间间隔异常、轴比塌缩、q 窗口边界命中、正交翻转和覆盖率跌落；保留全部源帧序号，从不插值或替换失败帧。
- `batch.py`/`export.py`：FrameRef 的容器 selector 与序列 selector 分开；输入、PONI、mask、beamstop 与不确定度文件内容参与 checkpoint 身份，`--stream` 逐帧写 NPZ 并在 resume 时保留已验证的旧数组。`run_batch` 支持 `independent`/`warm_start`、失败隔离、opt-in 精确像素内容 SHA-256 去重（`deduplicate_identical_frames` / `--deduplicate-frames`，仅复用比特完全一致帧的计算结果并保留源帧身份与 provenance）与独立模式多进程并行（`max_workers` / `--workers`）；没有跨全序列的 global/shared 联合优化。
- `campaign.py`：工作流预设（`annular`/`curvature`/`sector`/`butterfly_pub`）、自然排序清单发现（`bsaxs init-manifest`，缺失时间戳留空不捏造），以及多样品活动编排（`bsaxs campaign`，支持单样品失败隔离、可选 `--workers` 并行与 `--beamstop-radius-px` / `--deduplicate-frames`）。
- `observables.py`：提取剖面、lobe、ridge 和直接派生量。
- `ellipse.py`：只拟合倒易空间曲线，返回残差、协方差和可辨识性。
- `intensity.py`：生成和拟合经验二维强度场；当前 `full2d` 是共享双椭圆、角向包络、径向线形和背景的经验模型，不是 Grubb 2016 的完整 3D 取向正演。
- `journal_style.py`/`composite_figures.py`/`comparison_report.py`：严格无 Qt、无 `matplotlib.pyplot` 的出版级绘图与跨样品对比层；提供期刊排版预设（`--journal-style {default,nature,nature_comm,science,macromolecules}`）、防重叠 `asinh`/`log` 色条刻度与环绕射线角、声明式多面板 Nature 复合图与 6 面板单样品物理总览（导出 PDF/SVG/高 DPI PNG、Source Data CSV 与 SHA-256 清单），以及跨样品时间锚定对比（`target_time_h`）与动力学/重复性汇总。
- `ui/`：只管理交互和显示；后台 worker 不接触 QWidget。

## 输出原则

CSV 保存有界、可人工检查的表格；完整 `I/qx/qy/model/residual/fit_valid_mask/sampled_valid_mask` 保存在压缩 NPZ。统一分析域为 `finite(I,qx,qy) AND detector_valid AND NOT external_mask AND NOT ROI AND q_window AND weight_valid`，JSON 同时记录各阶段计数。批次的 `manifest.json`/`provenance.json` 保存输入/config hash、模式和版本；单帧 JSON 保存摘要，不能替代 NPZ 像素证据。执行成功与科学解释是两套状态：`fit_success=True` 不能清除 `mechanism_under_determined`、`apparent_geometry_only` 或 `nonunique_inverse_problem`。
