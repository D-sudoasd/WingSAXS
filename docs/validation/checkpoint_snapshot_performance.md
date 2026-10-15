# 检查点快照构造：重复转换与阶段计时

基线是 PR #9 合并后的 main：`cc0de47e0c1ba51831ae61bd710e115952e08384`。
本环境无法访问原始 121 帧生产记录。以下为明确标记的合成记录回放，未拟合科学数据，
不能据此给出真实 121 帧生产批次的加速比例。

## 成本及修复

`publish(retain_results=False)`、`_checkpoint_payload` 和写入前的 `_json_safe`
原先会重复遍历科学结果。实测中，快照构造和写入前通用转换均有可观成本。
本次保留每次读取、重建历史帧的行为，以保存 analyzer 或 sink 对已有结果的修改；
不引入跨快照缓存。修复包括：

- 数值列表中的内置标量直接完成有限性转换，嵌套对象及标量子类继续走原转换规则。
- 内部 `_CheckpointPayload` 分别转换帧元数据、科学结果和顶层元数据，写入时直接序列化。
  普通公开 `write_checkpoint` 输入仍完整执行通用安全转换。
- 数组描述中的 min/max 若在转为 Python float 时溢出，直接写 `None`，
  与旧版写入前转换后的 `null` 一致。
- 补充当前 batch.py 的 LF/CRLF 精确摘要和 PR #9 原生产者摘要，并重新固定身份构造器。
  未知源码、依赖、analyzer、输入、配置及原生证据仍按原规则拒绝；新收据保留真实新身份。

每帧同步保存、最后一次保存、完整帧记录、version 1、排序键、浮点精度、
flush、fsync、关闭及原子替换均保留。

## 同环境串行回放

Python 3.12，NumPy 2.3.5，SciPy 1.17.0，pyFAI 2026.9.0，FabIO 2026.6.0，
h5py 3.16.0；线程数固定为 1。两边各使用独立新进程、同一 fixture、相同顺序和参数。
合成样例包含 24 帧，每帧 profile 有 20000 个 `float(j)/7` 和一个负零。
连续运行有 25 次同步提交。该文件系统上的 fsync 等待很短，不代表生产磁盘的延迟。

| 阶段，累计墙钟秒 | 基线 | 修复后 |
|---|---:|---:|
| 实际 run_batch 编排 | 4.127695 | 2.706829 |
| `_checkpoint_payload`，包含下一行 | 0.964685 | 0.558373 |
| 其中历史结果 `_checkpoint_safe` | 0.959663 | 0.551909 |
| 通用 `_json_safe`，包括轻量元数据 | 0.856871 | 0.003305 |
| JSON 序列化 | 2.133414 | 2.012070 |
| 文件 write | 0.036239 | 0.040949 |
| flush | 0.000196 | 0.000139 |
| fsync | 0.000103 | 0.000106 |
| close | 0.000391 | 0.000383 |
| 原子 replace | 0.006962 | 0.006469 |
| 同步 writer 总计，包含转换、序列化和提交 | 3.053497 | 2.068805 |

本次编排墙钟减少 34.42%，快照构造减少 42.12%。JSON 序列化本身没有算法变化，
两次的差异包含共享机器负载和缓存波动。嵌套行不能相加；读取 fixture、读回比较、
报告生成和进程启动由原基准另外列出。逻辑文件大小会因时间字段的文字长度略有差异，
科学字段没有减少。

计时器仅计最外层 `_checkpoint_safe` 调用，递归时暂时恢复原函数，避免嵌套累计及
对每个标量加计时开销。`file_write` 包含 Python 文本编码/缓冲行为，不能称为物理磁盘流量。

## 恢复处理

`resume-replay` 先在独立输出中产生检查点，再恢复全部已完成帧；analyzer 一旦被调用即报错。
两边各恢复 24 帧，只提交最后一次完整状态。`cProfile` 对 `restore_checkpoint_item`
记录的累计包含时间分别为 0.006433 和 0.005640 秒，均调用 24 次。
这些恢复回放启用了 profiler，不能与上表未启用 profiler 的连续运行总时间直接比较。

`partial-resume-replay` 将独立检查点前半帧改为显式异常记录，然后仅回放这些帧各自的
原存储结果，后半帧保持可恢复。它不修改原 fixture，不调用科学拟合，保留正常恢复验证。
阶段报告另外区分快照之后、写入之前的历史恢复记录合并时间，以及其中的安全转换时间。

部分恢复样例两边均提交 13 次，启用了 profiler：

| 阶段，累计墙钟秒 | 基线 | 修复后 |
|---|---:|---:|
| 快照后历史恢复记录合并 | 1.918022 | 0.989332 |
| 其中安全转换 | 1.863841 | 0.940137 |
| `restore_checkpoint_item`，234 次 | 0.054170 | 0.048635 |

合并阶段确有重复的科学树转换；恢复资格判断本身耗时较小。这里保留其验证与重新判断，
仅由相同的标量转换优化降低转换成本。合并计时包含从 payload 返回至 writer 入口的轻量编排；
恢复函数的 profiler 时间与合并阶段部分重叠，不能相加。

## 复现

在仓库根目录使用同一解释器，给每次回放分配新的输出目录：

```bash
git worktree add ../WingSAXS-baseline cc0de47e0c1ba51831ae61bd710e115952e08384
python scripts/benchmark_checkpoint_writes.py synthetic --output-dir ../synthetic-checkpoint
python scripts/benchmark_checkpoint_writes.py run --fixture ../synthetic-checkpoint/fixture.jsonl --source ../WingSAXS-baseline --output-dir ../before --mode batch-replay
python scripts/benchmark_checkpoint_writes.py run --fixture ../synthetic-checkpoint/fixture.jsonl --source . --output-dir ../after --mode batch-replay
```

用同一 fixture 和新输出目录，将 mode 分别替换为 `resume-replay`、`partial-resume-replay`
检查全部及部分恢复。后两者报告 `restoration_profile` 并注明 profiler 开销。
`measurements.jsonl` 逐次给出 payload、历史恢复记录合并和 writer 阶段；`report.json`
保存依赖、源文件摘要、fixture 身份、提交次数、阶段合计及严格读回结果。
真实记录可访问时沿用原 `prepare`，用 `--frames 121 --mode batch-replay` 完整回放，
不要用邻帧替代结果，不必重拟合或重建报告包。

## 等价性与故障验证

连续、全部恢复和部分恢复的基线/新版检查点，均逐字段、Python 类型和 `float.hex()` 比较。
只有真实运行时间、更新时间及此次修改的两个源文件身份允许不同：
输入、配置、选择、顺序、结果、状态、诊断及 resumed 标记均一致。
两边所有其它源码摘要和依赖身份相同，新版没有冒用旧生产者身份。
原基准逐帧比较源记录科学结果，保留负零；运行耗时和 resumed 在源记录重放中按其实际含义单列。

复核两份检查点的稳定字段：

```python
import json
from butterfly_saxs.producer import calculation_producers_compatible
from scripts.benchmark_checkpoint_writes import _strict_equal

a = json.load(open('../before/checkpoint.json', encoding='utf-8'))
b = json.load(open('../after/checkpoint.json', encoding='utf-8'))
assert calculation_producers_compatible(a.pop('producer_identity'), b.pop('producer_identity'))
for payload in (a, b):
    payload.pop('updated_at')
    payload.pop('elapsed_s')
    for frame in payload['frames']:
        frame.pop('elapsed_s')
_strict_equal(a, b)
```

验证命令：

```bash
PYTHONPATH=src python -m pytest tests/test_checkpoint_snapshot.py tests/test_checkpoint_storage.py tests/test_checkpoint_benchmark_tools.py tests/test_checkpoint_quality.py tests/test_unattended_batch.py tests/test_batch_export.py tests/test_p2_io_batch_contracts.py tests/test_batch_delivery_performance.py -q
```

结果：149 passed，3 个 Win32 专属元数据测试在 Linux 跳过。
覆盖旧 pretty/compact 检查点的全部精确来源组合、完整/部分原生结果恢复、取消、有效结果复用、
未知源码/依赖/analyzer 拒绝、原 NPZ 字节和生产者证据、公开 writer 与内部快照严格等价、
历史结果修改、非有限值、负零、同步提交失败以及 fsync/replace 后真实子进程突然退出。
导出回归通过；未重新生成真实生产批次的科学交付包。

## 剩余限制

完整历史科学记录仍逐次构造、序列化并同步重写，累计复杂度仍随帧数呈二次增长。
在本样例中，修复后 JSON 序列化约占编排时间的 74%；文件系统同步等待在生产环境中
可能更大。恢复记录判断在本合成数据中占比较小，未修改其规则。
下一步若继续减少历史遍历，需要先确定结果可变性及持久化格式的边界，不能仅按对象身份缓存。
本次没有测得、也没有外推真实 121 帧的全批次性能。
