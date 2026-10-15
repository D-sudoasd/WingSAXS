# 检查点持久化性能与旧版本恢复

`write_checkpoint` 保留 version 1 的 JSON 数据结构、全部字段和浮点精度，
使用无缩进、固定键序的紧凑 JSON。非有限值仍通过原有转换变成 `null`。
文件仍在目标目录创建临时件，完成写入、flush、fsync、关闭和原子替换后才返回；
没有异步写入、延迟提交、减少检查点次数或跳过恢复验证。

## 统计口径

完整批次的 `checkpoint_write` 包含安全转换、JSON 序列化、文件写入及上述同步提交等待。
`checkpoint_payload` 是调用写入前的结果记录构造，两者顺序执行，时间可以相加。
Windows `GetProcessIoCounters` 的字节数是进程逻辑 I/O，不能称作物理磁盘写入量。
每帧提交一次，正常批次最后还提交一次包含最终状态的快照，即 N 帧有 N+1 次提交。

累计完整快照使已有帧被反复序列化和重写。深层 profile/peak/point 列表中的缩进空格
也被重复写出，且 `json.dumps(indent=2)` 使用较慢的递归编码路径。
本修复删除格式开销，不删科学明细；累计快照及安全转换的复杂度仍是剩余成本。

## 旧检查点兼容边界

旧版 pretty JSON 与新版 compact JSON 均由 `read_checkpoint` 读取。
计算生产者保存的真实源码 SHA 不会改写成旧身份。仅允许已列明的旧
`batch.py`/`producer.py` 哈希迁移到经过验证的新检查点写入版本；其它计算源码、
依赖、analyzer、输入内容、配置及 mode 必须保持匹配。

当前 `batch.py` 由精确 SHA 限定；当前身份构造器也由其源码摘要限定，摘要声明本身
从计算中去除以避免自引用，LF/CRLF仅在该构造器自检中归一化。
后续代码变化默认关闭这条旧身份兼容通道，不能自动放行未知旧版或新版。

原生数组恢复仍核验提交标记、成员完整性、帧签名及原始帧明细。
检查点领先原生发布时，缺少可靠原生证据的帧仍须重算。
跨版本恢复会更新生成的元数据，不把新生产者冒充旧生产者；保留整份NPZ时，
`retained_npz_provenance` 保存其原生产者、导出者和已验证 SHA，
`restored_calculation_producer_identity` 记录旧拟合来源。
整包和报告的生产者校验不因计算存储兼容而放宽。

## 可比回放入口

使用已完成的真实检查点，不重新拟合或扫描其大型 NPZ：

```powershell
& <python> scripts/benchmark_checkpoint_writes.py prepare `
  --checkpoint <原始v1检查点.json> --fixture <独立目录/fixture.jsonl>

& <python> scripts/benchmark_checkpoint_writes.py run `
  --fixture <fixture.jsonl> --source <冻结源码目录> `
  --output-dir <独立新目录> --mode prefix
```

prepare逐条读取原检查点，记录来源SHA和读取前后文件状态，保存完整JSON值。
默认取覆盖全序列的24个真实帧，并逐帧提交及最后提交一次。
`--mode single-snapshot` 用全部帧测量一次完整尺寸快照；
`--mode batch-replay` 将同一帧自己的既有结果交给实际 `run_batch`，保留原顺序、
输入/配置验证及检查点构造，测量批次编排总时间。后者没有重新拟合、报告或包装。

两边使用相同fixture、选择、参数、环境、同步提交次数和磁盘位置，串行启动新进程；
数值线程固定为1。记录各次提交的墙钟、CPU、逻辑I/O、文件大小及分段时间。
读回逐字段、类型和 float.hex 比较，包含负零；读取fixture和读回核对时间单列，
同时保留包含启动、加载、验证和写入报告的外层进程耗时。

实际24帧回放、121帧单快照和历史完整生产批次是不同范围，必须分开报告。
依据回放比例推算完整生产批次收益时须标为估算，不能称作重新实测的全流程加速比。
共享机器负载与OS缓存会影响墙钟，不宣称独占机器或冷磁盘结果。

快照构造中重复转换的后续测量、修复及复现实例见
[快照构造性能验证](checkpoint_snapshot_performance.md)。
