# 已有 DATA / IMAGE 配对交付的收尾

有些批处理使用独立的 DATA ZIP 和 IMAGE ZIP，而不是 `bsaxs package` 的单一 ZIP。下面的接口供这种已有编排流程复用；不重新拟合、绘图或改写 DATA ZIP，也不替代来源登记和科学审核。

## 先补齐登记过的依赖，再压缩

`butterfly_saxs.delivery_pairs.plan_image_members` 接收样品目录、原有 IMAGE 成员列表、已有 artifact inventory，以及可选 DATA ZIP。它跟踪 HTML 明确引用的本地文件，例如 `figure_source_map.csv`、`lobe_radial_peaks.csv` 和 `ellipse_candidates.csv`。只允许已登记的来源进入计划；文件存在但没有登记时返回阻断项，不自动补造来源证明。

提供 DATA ZIP 后，已由 DATA 交付的成员不必再复制进 IMAGE。链接预检只读取目录和有效 HTML，不为检查导航读取全部 DATA 数值载荷。计划中的 SHA 是来源登记的期望值，`content_hashes_verified=false`；写入 ZIP 时仍须校验实际流入的成员字节。

`bundle_manifest.json` 若作为包内成员，可在调用计划函数时用当前文件的 SHA 和大小补充一条临时记录；这条记录仅供打包器使用，不能写回该 manifest，否则会形成自我引用。

原有批处理的调用位置是：

1. 完成来源登记和当前导航修改，获得最终 IMAGE 初始成员
2. 对全部样品调用 `plan_image_members` 和 `check_planned_pair`，阻断未登记来源及断链后才开始新压缩
3. 按计划流校验成员并创建 IMAGE；已有不可变 IMAGE 通过 `reuse_image_archive` 核对当前来源、确切成员集、逐成员 SHA/CRC 和 ZIP SHA 后复用。它不会写 ZIP，也不会以重新压缩作为检查手段；不一致则保留旧包并阻断
4. 对生成的两包调用 `check_archive_pair`，再写下游包哈希和当前结项绑定

本仓库不包含旧的、按某次实验写死路径和样品数量的生产脚本。仅升级通用 `package` 不会自动改变那些脚本；应在上述调用点接入配对接口，并先用小型 fixture 验证。

`examples/paired_delivery_adapter.patch` 提供两个已有定制脚本的最小接入补丁：在全部样品写 ZIP 之前准备成员，在最终配对检查时使用覆盖视图，已有不可变 IMAGE 则先校验并复用。补丁针对源 SHA 为 `88edfc9ce0f65c75a9855829e841278686f97c6f9dff2338390e65ecf4ee4f75` 的 `finalize_minimal_delivery.py` 和 `b92e7ae061d1296ff1e7dc536fb4128926eea83434efcc8fb7d1fa621850a76f` 的 `revise_run03_current_delivery.py`；其他版本需重新核对上下文。它没有复制生产路径或整套定制脚本，也不会随安装自动应用到生产环境。

可选的 `test_production_delivery_adapter.py` 在明确提供 `WINGSAXS_DELIVERY_EVIDENCE` 时，对这些源快照的临时副本应用补丁，并使用合成小文件调用实际选择、预检、ZIP 写入与重试函数；不运行原来的 `main/apply/plan` 入口。未提供源快照时该组测试跳过，不能据此宣称真实生产数据已重跑验证。

## 检查实际解压后的页面

`check_planned_pair` 检查 DATA ZIP 加本地 IMAGE 计划；`check_archive_pair` 检查最终两包。两者统一使用“先解压 DATA，再用 IMAGE 覆盖同名文件”的有效视图，只检查最后实际显示的 HTML。旧 DATA 页面已被替换时，其旧链接不应继续充当当前页面的检查对象；当前页面真的缺文件仍会阻断。

重试时，即使当前导航在上一次尝试中已经改好，也仍要把它放进 IMAGE，否则解压后会重新出现 DATA 中的旧页面。保留原导航时可使用 `.html.before`；若历史文件仍用 `.html` 后缀，必须通过 `historical_html` 明确列出准确路径，并保留其字节。接口不会自行忽略整个历史目录、删链接或把所有坏页面视作历史。

## 只核对明确选定的当前结项回执

```bash
bsaxs verify-delivery results/delivery_closeout_receipt.json
# 回执放在别处时，明确提供其中相对路径的根目录
bsaxs verify-delivery closeout.json --root results
```

命令只读取该 JSON 顶层的 `bindings` 列表（每项为相对 `path` 和 `sha256`），流式核对这些文件。退出 0 表示所选绑定与当前文件相符；退出 1 表示缺失、变更或读取中发生变化；退出 2 表示回执结构、路径或输入错误。结果仅输出 JSON，不生成新审计文件、不修复哈希、不递归检查历史对象，也不将既有 `PASS` 字样当成当前内容仍有效的依据。

先修改最终 README/index，再绑定当前结项回执。随后再次修改导航，旧回执就会失效，应由原有收尾流程更新当前绑定；此前失败尝试及历史回执保持原样。新回执明确绑定当前文件时，旧记录中已被取代的导航 SHA 不应使整个交付失败。

绑定一致与科学验收是不同问题。`scientific_acceptance=false`、未审核、边界解、条件性标准误和失败历史不会被这些工具改成通过；已完成的必要交付也不因这些保留的科学限制被重新划成未完成任务。
