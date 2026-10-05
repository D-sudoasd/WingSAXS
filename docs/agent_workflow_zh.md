# Agent 按需执行与验证

本页在需要任务路由或确定验证范围时加载。已知入口、已有 diff 和当前任务中
取得的证据优先；无需先读完本页的所有链接，也无需建立整个仓库的文件地图。

## 从任务到相关文件

先确认工作区已有改动，保留不相关工作。已知模块时，对该文件或目录用 `rg -n`
查相关符号，读取命中附近的实现、调用者和测试；入口未知时才用范围受限的
`rg --files`。当前上下文已有的 AGENTS、代码或配置，内容未改变时直接复用。
只有接口关系不明确、局部检查失败或任务跨模块时才扩大搜索。

| 当前任务 | 优先入口 | 最小有判别力的验证起点 |
| --- | --- | --- |
| 命令发现、JSON 提示、参数路由 | `cli.py`、`cli_contract.py` | `tests/test_cli_agent_contract.py`；涉及执行参数再选 `tests/test_pipeline_cli.py` 中的相关用例 |
| 图像/NPZ、帧和数据集选择、掩膜 | `io.py`、`pipeline._read_frame_bundle` | `tests/test_io_geometry.py`、`tests/test_npz_frame_maps.py`、`tests/test_p2_io_batch_contracts.py` |
| batch、哈希、checkpoint、resume | `batch.py`，相应 CLI/service 调用者 | `tests/test_batch_export.py`、`tests/test_unattended_batch.py`；元数据问题加对应 time-metadata 测试 |
| 轨迹或椭圆拟合 | 出问题的算法模块及其调用者 | 对应算法测试；按需要增加 CLI/service 数值用例、掩膜/弱信号/边界解回归 |
| 已有结果的报告或打包 | `report.py`、`delivery.py`，具体绘图模块 | `tests/test_report_cli.py`、`tests/test_analysis_report.py` 或对应 delivery/绘图测试 |
| service 或 GUI 交互 | `service.py`、相关 `ui/` 控件/worker | 对应 service/UI 测试；只涉及控件时不重新拟合实验数据 |
| 文档或说明修改 | 目标文档及相关接口定义 | 检查链接、命令/字段是否与实现一致；无需全套数值测试 |

表中测试是选择起点，不要求把一行全部执行。对 Python 改动，先 lint 实际改动
文件，再运行与行为有关的用例或测试文件，例如 Windows：

```powershell
& ./.venv-project/Scripts/python.exe -m ruff check src/butterfly_saxs/cli_contract.py
& ./.venv-project/Scripts/python.exe -m pytest -q tests/test_cli_agent_contract.py
```

一次通过的检查已经覆盖同一内容、依赖、配置和执行环境，无需在每个 skill、
review、commit、merge 步骤再跑一遍。在任务上下文保留检查范围、结果和未解决问题；
继续修改时用 diff 确认哪些证据失效。合并产生新内容或冲突时检查受影响部分；
无冲突且内容相同的合并不重新执行同环境的等价检查。

## 扩大验证的条件

- 局部失败：先复现并修复，重跑失败检查；然后扩展至相关调用者。
- 共享数值行为、单位/校准、选择器/掩膜、checkpoint/序列化或多个 CLI/service/UI
  接口发生变化：检查相应接口合同与回归。影响不能可靠局限时，完整 pytest 跑一次。
- 包依赖、支持的 Python/OS、构建元数据、入口或打包资源变化：依赖检查、构建、
  新环境 wheel smoke 和适用平台测试。仅命令说明修改不触发 wheel 重建。
- 明确要求发布、完整验收或实验数据包审计：执行其对应深度路径。只对选定对象
  做 integrity 检查；无变化的同一对象不反复全量扫描。

CI 对运行时代码保留 Ubuntu/Windows × Python 3.11/3.12/3.13 全测试和 wheel smoke；
lint 只执行一次，Ruff 的语法检查取代六次额外 compileall。文档/README/图示改动
不启动该矩阵。main push、PR、`v*` tag 或手动运行触发 CI；新提交取消被替代的
旧运行。

## 数据工作流中的复用

- `bsaxs describe COMMAND` 返回一个命令的契约；参数详情用 `COMMAND --help`。
  当前安装代码未变化就复用已取得的契约。环境已可用时不固定执行 doctor。
- 直接执行所需操作。`inspect` 用于决定分析配置，synthetic 用于复现或算法验证。
  `batch --unattended` 自带预检，无需先单独做一遍同一预检。
- `report` 读取已有 batch 导出，`package --resume` 复用未变化的交付 ZIP；只要输入
  和分析设置未变化，不为重新出图/打包而重读原始图像、预检和拟合。
- NPZ 读取在同一 archive 中完成数据集选择、强度和 q-map 解析。只读取选定强度和
  实际存在的 q-map 字段；共享 2D q-map 和每帧 3D q-map 都保留原来的选择规则。
  `io.load_image(include_qmap=True)` 显式启用；普通 loader 的严格多数据集行为不变。
- batch 输入身份在单次调用内按规范化路径、size/mtime、文件身份和变更时间复用 SHA-256。
  同一容器的帧/数据集/顺序/时间等仍各自参与身份，不缓存拟合结果或 detector 值。
  Windows 支持版本的 `st_ctime` 是创建时间，使用 Win32 `FILE_BASIC_INFO.ChangeTime`
  判断调用内的等长、恢复 mtime 改写；取不到变更时间时逐次读内容，不复用摘要。
  读取期间文件元数据或身份变化会报错；每次新调用及 resume 都重新读取内容，因此同大小、
  同 mtime 的跨调用改写仍使旧 checkpoint 失效。没有磁盘缓存或额外 manifest 要维护。

Windows 时间语义见 [Python stat 文档](https://docs.python.org/3.13/library/os.html#os.stat_result)
和 [FILE_BASIC_INFO 定义](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_basic_info)。

独立读取可以在一个 `functions.exec` 中用 `Promise.allSettled` 批量调用工具并逐项
检查结果。依赖前一步的编辑、批准、验证和收尾保持顺序；不为了并行再重复发现项目。
通用 skills 的相同检查共享已有证据；与任务无关的 skill 不加载，也不修改全局 skills。

## 改造证据（2026-10-05）

基线为 `5b7a114`。在同一 Python 3.13 环境、同一临时输入上分别加载该版本和
改造后的函数，用 `numpy.load` 计数 archive 打开、`Path.open("rb")` 计数容器
哈希读取；两份科学输出和输入身份逐字比较。测试输入为 64×64 的带物理 q-map
环状强度 NPZ，以及 16×256×256 的 float32 NPY 容器，均由程序生成。

| 常见操作 | 改造前 | 改造后 | 保留的判据 |
| --- | --- | --- | --- |
| `inspect` 带 q-map 的 NPZ | 3 次 archive 打开 | 1 次 | 完整科学 JSON 相同 |
| `analyze` 同一 NPZ | 3 次 archive 打开 | 1 次 | 完整结果（含数组摘要）相同 |
| 16 个 selector 的容器身份 | 16 次哈希读取，67,110,912 B | 1 次，4,194,432 B | checkpoint 输入哈希相同；每个 selector 的身份仍参与 |
| 取得 batch 命令契约 | 基线完整清单 9,723 B | `describe batch` 2,734 B | batch 契约、退出码和科学边界保留；输出减少 71.9% |
| 一次运行时代码 CI | 6 次全量 lint + 6 次 compileall | 1 次 lint | 六组完整 pytest、依赖/CLI smoke 与 wheel smoke 保留 |

字节数按 `json.dumps(..., ensure_ascii=True, allow_nan=False)` 计算，不包括缩进。
单次计时中容器哈希从 56.0 ms 到 8.7 ms（约 6.5 倍）；`inspect` 为 92.8/87.9 ms，
`analyze` 为 4.06/4.38 s。小图的拟合耗时主导总时间，未据此宣称整体拟合加速。
稳定收益是消除重复打开、全容器哈希和无关命令上下文。
顶层 AGENTS 从 91 行缩减到 60 行；详细路由和验证条件由本页按需提供。

2026-10-02 的现有执行轨迹包含 504 次外层工具调用（其中 24 次 spawn、88 次
agent 消息、41 次 followup、34 次等待）、16 次全量 Ruff、5 次全量 compileall、
27 次 pytest 命令（6 次完整、20 次聚焦、1 次 collect-only）。207 次可识别的静态
文件/范围读取对应 198 个不同键，CI 配置全文出现 4 次。这些次数用于定位复查
和协调成本；轨迹也有修改后的重测及中断重启，不将全部次数视作可删除的浪费。

永久回归位于 `test_npz_frame_maps.py`（一次打开/每字段一次解析、未选强度不解析、
歧义/帧/掩膜/异常关闭），`test_batch_export.py`（调用内复用、新调用重读、输入
改写失效、读取期间变化、取消），`test_cli_agent_contract.py`（局部契约和 JSON
错误）。无持久缓存、自动 repo map、额外测试选择框架或 agent 状态文件。

验证：相关测试 144 passed；一次完整回归 1609 passed、12 skipped（未挂载历史
fixture、软件图形后端不支持 Qt 3D），用时 888.49 s。最终补充命令目录覆盖后，
重跑相关目录/发现测试 6 passed；共享数值与 I/O 未再修改，不重复完整回归。

最终 review-pro 审查修复了 Windows 调用内恢复 mtime 的缓存失效，以及 describe
未知命令的退出码契约。batch/resume/CLI 回归 67 passed，增加读取期间等长改写和
变更时间不可用时不复用的检查。修复后重放同一 16 帧容器，三次测量均为哈希读取
16→1、输入身份完全相同，中位耗时 57.59→10.35 ms；Win32 额外操作只读取元数据。
