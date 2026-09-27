# 序列拟合的初始化与处理方向比较

本基准检查 WingSAXS 批处理 API 在同一组二维图像上使用独立初始化、正向 warm-start 和反向 warm-start 时，拟合结果是否依赖上一帧参数。它回答的是**数值稳定性和初始化敏感性**，不回答轨迹方法的准确度，也不能替代真实 SAXS 实验数据验证。

## 可复现运行

在仓库根目录执行：

```powershell
py -3.13 scripts/benchmark_initialization.py --output results/validation/jac_initialization
```

程序使用固定随机种子。目标目录中若已有本基准输出，命令会停止以免覆盖；确认需要重新生成时显式添加 `--force`。可用 `--size 128` 快速运行较小的演示矩阵，默认阵列为 `192 × 192`。

有限层片 FFT 序列在当前配置下未产生可用于 warm-start 的拟合候选时，应报告为“未测试到 warm-start 敏感性”，不能把独立拟合结果相同写成 warm-start 稳定。为检查传递本身，另提供一个 64 × 64 的已知椭圆弧补充基准：

```powershell
py -3.13 scripts/benchmark_initialization.py --known-arcs-only --output results/validation/jac_initialization_known_arcs
```

该补充序列使用独立的已知观测弧生成器，轴比为 0.40、0.38、0.20 和 0.22；0.38 到 0.20 是突变。全像素屏蔽帧放在突变与后续有效帧之间，并另有噪声对照。它仍是合成几何测试，不是物理散射正演或真实实验验证。

## 输入序列

输入来自 `benchmark_sequence.generate_oblique_stack_sequence`：有限高斯层片在实空间生成，经过二维 FFT 和功率非相干求和得到强度图。序列包含 6 个有信号帧，名义层间距在 `jump_03` 前从 11.4 nm 变化到 11.9 nm，随后突变至 15.5 nm 并变化到 15.9 nm；两个区段使用相同的倾角和堆栈旋转范围。每个区段的一帧含局部缺瓣 mask。序列另含一个全像素屏蔽的失败帧，以及一个只有噪声的负对照。

`nm^-1` 是合成器基于 1 nm 像素间距定义的模型坐标单位；它不代表探测器 PONI 标定结果。有限层片 FFT 是简化的二维合成源，不是完整三维 Grubb 正演模型。生成器参数中的层间距用于描述输入突变，不能视为每个椭圆拟合参数的唯一反演真值。

## 比较流程

三个分析都调用 `batch.run_batch` 和相同的单帧 `pipeline.analyze_frame` 工作流，并保持图像、mask、q-map、q 窗口、拟合模式及计算预算一致。唯一改变的是帧次序和初始化策略：

1. `independent`：每帧从自身观测独立初始化；
2. `warm_start` 正向：按合成序列顺序传递上一帧的可用候选；
3. `warm_start` 反向：按逆序处理同一组输入。

报告根据稳定 `frame_id` 合并三次结果。它记录逐帧状态、椭圆候选参数、观测 ridge 半径和 `warm_start_from`。全像素屏蔽帧用于检查失败隔离：它保留在序列原位，不能成为下一帧的初值来源。噪声对照单独报告，避免将对照帧与有信号帧的参数差异混为一谈。

## 输出与解释

`results/validation/jac_initialization/` 下的 `campaign_manifest.json` 保存输入帧、生成设置和 SHA-256；`inputs/` 保存强度、mask 与 q-map；`per_frame_comparison.csv` 保存按帧对齐的拟合值、差异及 `effective_optimizer_starts`；`warm_start_calls.csv` 保存传入每次分析的初始参数和 lineage 来源；`summary.json` 与 `report.md` 汇总失败状态和参数差异。CLI 标准输出为 ASCII 转义的严格 JSON；无警告完成返回 0，存在 warning、failed、skipped 或取消返回 1，输入或覆盖错误返回 2，错误说明写入标准错误。

逐参数报告配对有限估计的中位和最大绝对差异。轴角按 180° 周期计算最小角差；不报告角度的相对百分比，因为角度零点会使其失去意义。缺失拟合值不补齐、不插值，并由 `n_paired` 明示可比较帧数。

`initial_parameters` 记录的是前一帧几何参数是否传到 `pipeline.analyze_frame`，而 `effective_optimizer_starts` 记录拟合器实际生成并提交优化的起点。对当前 `butterfly_curvature` 弧拟合路径，若当前帧具有足够的已观测弧点，`fit_arc_ellipses` 会运行代数弧初值估计并覆盖自由参数的起点。因此，已完成的 known-arcs campaign 证明批处理完成了参数传递，但其近零差异不能解释为“不同前帧初值下拟合稳定”；实际优化器起点一致时，该比较检验的是当前帧代数初始化下的处理方向一致性。有限候选、固定或受约束参数仍按各自配置处理。投稿论文仍需用真实实验重复曝光、独立人工描迹或共同支持的文献方法评估测量差异，并报告真实数据中的失败帧和适用范围。
