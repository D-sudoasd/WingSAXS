# 批次交付：数据、图与样品导航

`bsaxs package` 汇总**已有**批次输出：生成可在浏览器打开的导航页，以及可直接解压浏览的 ZIP。它不会调用拟合、重新绘图或修改数值、误差和质量诊断。

## 已有结果只补交付

```bash
# 单个样品/序列的 batch 输出
bsaxs package results/batch

# 父目录下已有多个样品的 batch 输出
bsaxs package results/all_samples

# 中断后续做；输入和 ZIP 未变时复用原 ZIP，不重新压缩
bsaxs package results/all_samples --resume

# 只生成导航，明确跳过 ZIP；不会生成空 ZIP
bsaxs package results/all_samples --no-archive
```

同样接受 `--no-archives`。如果跳过打包时已有旧 ZIP，它不会被删除，也不会被列作这一次的交付文件。其已知哈希与来源签名保留在 `archive.previous`，仅用于之后恢复，并不表示本次核验过旧 ZIP。之后用 `--resume` 可以补齐 ZIP；若来源和旧 ZIP 都未改变，会直接复用。若只生成导航期间来源发生变化，不会把旧 ZIP 当成本次结果。

## 新批次自动完成交付

```bash
bsaxs batch "data/frame_*.edf" --poni geometry.poni --stream \
  --checkpoint results/batch/checkpoint.json -o results/batch --package
```

`--package` 在原生 CSV/JSON/NPZ 与演化图导出完成后继续生成导航和 ZIP。若这一步失败，已完成的分析输出保留，JSON 的 `delivery.next_command` 给出单独补做交付的命令，无需重跑拟合。

继续拟合仍使用原有 `bsaxs batch ... --resume`；只补交付则使用 `bsaxs package ... --resume`。

## 交付文件

- `delivery_index.html`：样品列表、数据/图链接、可展开的逐帧来源和诊断；无外部网页依赖
- `delivery_summary.json`：实际输出路径、样品/帧计数、缺失项目、打包状态及下一步
- `delivery.zip`：相同导航页、可移动的 `delivery_contents.json`，以及原样复制的数据和图

在原目录打开 `delivery_index.html`，或解压 ZIP 后打开同名文件。每个样品链接其 `frame_summary.csv`、`parameters_long.csv`、`ridge_points.csv`、`lobe_measurements.csv`、拟合 JSON/JSONL、`manifest.json`、`provenance.json`、`results.npz` 和 `evolution.png` 中实际存在的文件。可选导出前缀也受支持。

额外图包放在相应样品输出目录的 `figures/` 内，会一同链接并收入 ZIP。画廊 HTML 中明确引用的本地文件也会递归收入：例如 `figures/gallery.html` 引用同一输出根目录内的 `../diagnostic_sources/profile.csv`，该 CSV 会保留原相对路径进入 ZIP，继续引用的 HTML 也会处理。明确引用的普通文件不限扩展名，`.dat`、`.h5`、`.poni` 等同样按原字节复制。URL 编码、查询参数、锚点和循环链接受支持，内容不改写。目录外的原始数据及无关文件不会被整目录打包。

引用的文件缺失、越过输出根目录、经过符号链接、位于隐藏/暂存目录或无法读取时，导航和 JSON 会明确报告 `status: incomplete`、具体来源链接和补救动作；其余可用文件仍打包，退出码为 1。补齐文件或修正画廊导出后运行 `--resume`。不会抓取远程 URL；外部引用记录在 `dependencies.external`。跟踪范围是 HTML 的显式 URL 属性（包括常用 `href`、`src`、`data`、`poster` 及普通 `srcset`），不执行 JavaScript，也不解释 CSS 中的额外引用或 JavaScript 动态生成的地址；带 `base` URL 或复杂 data-URL `srcset` 的画廊会提示改用明确的相对链接，不会猜测依赖。不要将原始数据目录作为交付输出目录。多个批次命令同时写同一交付目录时，应等导出结束再运行打包。

逐帧导航保留 `frame_index` 和 `frame_id`，可用于查找表格及拟合记录；`CSV record` 仅表示从零开始的记录次序，旧表缺少帧索引时不会猜测其科学含义。原生 NPZ 数组仍使用原有键名。未知版本的来源元数据原样保留，不因版本字符串变化重做数据。

## 状态怎么读

- 退出 **0**：本次交付完成
- 退出 **1**：可用文件已交付，但仍有缺失/不安全的链接依赖、缺失输出、空样品或原有帧警告/失败；打开导航查看，不应当成程序崩溃
- 退出 **2**：路径、读取、写入或覆盖问题；JSON `error.message` 说明原因
- `archive.status` 为 `created`、`reused`、`skipped` 或中断时的 `pending`；跳过和失败不是同一件事

缺图或有限值的质量警告不会挡住其余数据交付。补上缺失图后再次 `package --resume` 会更新导航和 ZIP。原有质量状态不会被打包改成 PASS；交付成功也不是科学验收。

`--resume` 只重建 WingSAXS 可识别的交付文件；`--force` 则明确重建现有生成物。ZIP 在临时文件中完成后才替换旧版，磁盘错误不会把旧 ZIP 换成半个文件。替换之前保存候选 ZIP 的哈希及来源签名，因此 ZIP 已完成但最后一次回执写入被中断时，重试可以识别并复用它。重试会重新检查来源和现有 ZIP 的内容，损坏或新增文件会触发重建。

未改变的 `--resume` 不改写导航、ZIP 或落盘回执，三者的字节和修改时间保持不变。命令返回的 `archive.status: reused` 表示这一次复用了 ZIP；新生成的落盘 `delivery_summary.json` 对已完成 ZIP 记录 `created`，不为了记录本次动作改写自身哈希。旧版已完成且内容等价的 ZIP 和回执也直接复用，不因升级签名算法或补充可选字段而强制重压缩或修改回执。`index_sha256` 绑定生成导航；`archive.sha256` 绑定 ZIP；`archive.source_signature` 绑定该 ZIP 的来源与内容。ZIP 内部不包含外部回执或自身 ZIP 的哈希，避免循环引用。固定成员时间和权限使相同内容在相同压缩实现下重建时得到相同 ZIP 字节，而不是因文件修改时间变化而产生新哈希。

打包时对实际写入 ZIP 的每个成员流计算 SHA-256，并检查来源在读取导航、收集 HTML 依赖和最终替换期间是否被改动。发现来源变化会拒绝宣称本次交付完成，保留之前的 ZIP；请等写入结束后重试。它不锁定外部导出器，不能代替“先完成导出、再打包”的顺序。

## 多样品流水线的收尾顺序

已有自定义批处理程序时，先生成最终数据、图、HTML 和包内 README，再逐样品运行 `package --resume`。只有来源改变的样品需要重建；不要为了更新根目录登记表而强制重打所有样品。

随后生成包外总导航和说明，最后计算它们及各 ZIP 的哈希并写根目录登记表；如果另有收尾文件要绑定这些登记表，收尾文件必须最后写。一旦修改了被绑定的 README、导航或登记表，就应重算受影响的下游绑定。禁止把包含某个 ZIP 哈希的登记表再打进同一 ZIP，也不要让收尾文件绑定自己的哈希。

`package` 只管理上面列出的三个交付文件，不会猜测或重写自定义生产流水线的根目录验证报告、科学验收记录和收尾登记表。应由该流水线按实际依赖顺序调用它并维护这些绑定，不能把重算哈希当成科学审核通过。

独立 DATA / IMAGE 两包、历史导航和当前结项绑定的接入方式见 [配对交付收尾](paired_delivery_zh.md)。

## 大字段与演化图

CSV 读取支持超过 Python 默认 128 KiB 的科学元数据字段，不裁剪文本、数值精度或带引号的换行。演化图保留有限的警告/失败估计和候选状态；缺值断线，只为实际提供的非负有限标准误画误差棒。参数按数量与单位分别成图，轴比为无量纲；未声明的 q 或强度单位不会猜成物理单位。时间不完整时整条序列退回明确标注的帧序号，不把帧号与秒混在一条坐标轴上。
