# 批次分析报告与数据、图表包

`bsaxs report` 从已有批次导出生成逐帧描述统计、序列图、片层几何分析和可浏览报告。它读取批次保存的图像、q 坐标、有效域、拟合记录与 profile 数组，不重新拟合，也不运行 `full2d`。`bsaxs package` 可在此基础上把现有结果、图表和导航页打成 ZIP，供离线查看或交付。

## 对已有批次生成报告

单个样品或序列：

```bash
bsaxs report results/sample
```

父目录包含多个样品批次时：

```bash
bsaxs report results/all_samples
```

报告默认使用 128 个径向箱、72 个角向箱，以 180 dpi 输出 PNG、SVG 和 PDF。可以调整分箱数、图片格式和栅格图分辨率：

```bash
bsaxs report results/sample \
  --radial-bins 160 --angular-bins 90 \
  --formats png svg pdf --dpi 240
```

格式也可选 `tiff`。对已有报告运行 `--resume`：如果源文件、设置和报告版本都未改变，就复用已有文件；有变化或文件缺失时重新生成。`--force` 会强制重建报告。

## 基于椭圆结果生成片层几何分析

报告默认以已有椭圆拟合结果提供周期尺度，并以已观测 q 方位和支持信息确定片层面内方向。优先使用已接受的脊点方向；没有脊点时使用有效的径向瓣峰，图中分别标示这两类测量，径向峰及其模型比较另存为 `lamellar_radial_peaks.csv`。对观测方向 `β`，报告按对应的拟合分支计算条件方向周期 `2π/R(β)`；椭圆倾角 `theta` 不会被当作片层法向角。`Ln_from_minor_axis = 2π/b` 单独保留为短轴全局参考，不赋给每个观测方向。候选椭圆、边界解、缺失观测和失效帧会保留其原有状态；拟合分支或方向来源不明确时不补造该方向的周期。

无需重新拟合，可以为已有批次重新生成片层报告：

```bash
bsaxs report results/sample --resume --lamellar-settings results/lamellar.toml
```

新批次可在分析后直接使用同一组参数生成报告：

```bash
bsaxs batch "data/sample/*.edf" -o results/sample --report \
  --report-lamellar-settings results/lamellar.toml
```

参数文件接受 JSON 或 TOML；`lamellar`、`lamellar_settings`、`settings` 嵌套表均可。例如：

```toml
[lamellar]
mode = "multi"
period_source = "ellipse"
layer_count = 8
stack_count = 12
thickness_ratio = 0.25
width_ratio = 4.0
depth_ratio = 4.0
spread_deg = 0.0
spacing_jitter_pct = 0.0
position_jitter_pct = 24.0
lateral_shift_ratio = 0.0
out_of_plane_deg = 0.0
seed = 0
```

未列出的绘制参数使用 `LamellarSettings` 当前默认值。`thickness_ratio`、`width_ratio`、`depth_ratio`、层数、堆栈数、方向分散和起伏等控制的是几何示意，不是 SAXS 椭圆直接测得的微观厚度、横向尺寸或取向分布。可以在一个设置文件中统一更改这些假设；CLI 会在启动批次拟合前校验该文件。

每帧报告保存片层参数和方向来源到 `lamellar_analysis.json`；能构造出几何场景时，还会保存 `lamellar_geometry.npz`。长表和图展示可用的片层参数、候选状态、方向证据及其随帧变化。输出包括 `lamellar_parameters.csv`、`lamellar_directions.csv`、`lamellar_period_by_angle.csv`、`lamellar_radial_peaks.csv` 和 `lamellar_changes.csv`；样品集合另外生成 `collection_lamellar_parameters.csv`。方向表同时保留实测 q、对应模型 q、条件模型周期和残差。角度采样表中的 `observed_support` 表示该分支存在已保留的实测点，并不表示每个采样角度都被观测。序列图使用各分支实测方向的轴向均值和范围，完整点记录仍保留在逐帧 JSON 和方向表中。报告图片跟随 `--formats` 支持 PNG、SVG、PDF 和 TIFF，并随 `bsaxs package` 加入交付 ZIP。

物理长度需要已有的物理 q 标定。若 q 仍是像素倒数或来源单位未知，片层长度保持相对单位。椭圆和二维观测方向只能约束当前参数化示意的输入，不能唯一重建真实显微形貌或三维片层结构；报告中的片层几何应与实测强度、拟合参数和绘制假设分别解释。

单个批次的报告写入 `results/sample/figures/analysis_report/`。入口为 `index.html`，`report_summary.json` 记录来源状态、分箱设置、逐帧状态、表格和图表路径。`README.json` 说明统计量定义。传入包含多个批次目录的父目录时，每个批次分别生成报告，并在父目录汇总为 `analysis_reports.html`、`analysis_report_summary.json`、`collection_parameters.csv`、`collection_measurements.csv` 和 `collection_lamellar_parameters.csv`。这些 collection CSV 合并各批次对应的长表，并保留 `sample` 与 `export_prefix` 来源标识；逐帧测量表保留帧索引、帧 ID、运行状态和 q 单位，参数表保留原有帧信息、状态及参数单位。对该父目录运行 `bsaxs package` 时，这些汇总文件、各批次原生结果和 `figures/` 报告文件都会收入 `delivery.zip`。

## 智能体的实际工作顺序

开始前运行 `bsaxs describe`，读取当前 CLI 命令和数据约定；再用 `bsaxs inspect` 检查本次实验的代表图像、PONI、mask 和 q 范围。根据当前数据明确追踪方法与 manifest 元数据；需要限制 q-window 时，在本次命令指定 `--q-window Q_MIN Q_MAX` 或使用本次 TOML 配置。报告分箱设置只控制汇总与绘图，不会替新批次选择实验 q-window，也不会从历史批次复制设置。

之后按完整序列运行 `batch --stream --checkpoint ... --report --package`。先从 `figures/analysis_report/index.html` 查看帧顺序、观测图、片层投影与序列变化，再按需要检索 CSV 长表及原生 `results.npz`；警告、缺口、覆盖率和候选状态仍对应原批次证据。报告或 ZIP 中断后，按上一节命令单独续做相应交付步骤。

## 与序列拟合和最终交付连用

新序列可在批次导出后自动生成统计报告，再生成包含报告的导航页和 ZIP：

```bash
bsaxs batch "data/sample/*.edf" \
  --poni geometry/detector.poni --mask masks/detector.npy \
  --manifest data/sample/sequence.csv \
  --ridge-method butterfly_curvature --butterfly-stage evaluate \
  --mode warm_start --stream \
  --checkpoint results/sample/checkpoint.json \
  -o results/sample \
  --report --report-radial-bins 128 --report-angular-bins 72 \
  --report-formats png svg pdf --report-dpi 180 \
  --package
```

应为当前样品明确指定校准、mask、帧清单和需要的分析参数；例如 q-window 应来自当前实验设置，不会从别的历史批次自动继承。`--stream` 适合较长序列或大型探测器，`--checkpoint` 支持中断后恢复批次分析。报告使用本次批次实际保存的 q map、analysis domain 和帧元数据。`--package` 在报告生成后继续创建可浏览的交付索引和 ZIP；`figures/` 下的报告文件会随批次数据一起收入 ZIP。

如果批次分析已完成而报告或打包中断，无需重跑拟合：

```bash
bsaxs report results/sample --resume
bsaxs package results/sample --resume
```

`batch --report` 失败时已完成的原生批次结果仍保留；CLI JSON 的 `analysis_report.next_command` 会给出单独补做报告的命令。只需补报告时运行 `bsaxs report`，只需生成浏览页和 ZIP 时运行 `bsaxs package`。

## 报告包含什么

每帧概览图包含观测 q 图、径向平均强度、角向平均强度、极坐标平均强度和观测覆盖率。径向和角向曲线仅在有数据支持的位置显示描述性 SEM。存在原始 `full2d` 模型数组时，另生成实测强度、已保存模型与残差图；报告不会据此重新拟合模型。

序列图按采集时间显示径向、角向强度分布、强度与面内二阶矩摘要及可用的拟合参数变化；有 `time_s` 时使用秒，否则采用记录的 `time` 和 `time_unit`，未记录时间单位时标为“单位未注明”。时间值不完整或单位混杂时，改用 0-based 帧序号。不同 q 单位分别处理，不进行自动换算。报告保留警告帧、缺失帧、空箱和无效区；它不会插值缺失值来连接趋势线。

报告目录含有以下长表：

| 文件 | 内容 |
| --- | --- |
| `frame_measurements.csv` | 逐帧强度分布、q 范围、象限计数、最大像素、面内二阶矩及其观测域摘要 |
| `parameter_changes.csv` | 按参数和单位汇总帧间变化；`value_field` 分开标明常规 `value` 与 `candidate_value`，分别统计帧数、首末值、变化量、均值、标准差和范围。参数图将仅有有限候选值的点用空心菱形单独标出 |
| `radial_profiles.csv` | q 箱边界与中心、像素均值/总和/标准差/SEM、像素数和覆盖率 |
| `angular_profiles.csv` | 角度箱边界与中心及相应强度统计和覆盖率 |
| `stored_profile_samples.csv` | 原批次已保存的一维 profile 与拟合向量样本值，以及对应的 NPZ 数组键 |
| `fit_diagnostics_long.csv` | 原拟合记录中 butterfly、ellipse、`full2d`、分析域和 flags 等标量诊断 |
| `ellipse_candidates.csv` | 各椭圆候选解的字段和值 |
| `array_catalog.csv` | 原生 `results.npz` 中逐帧数组的角色、键名、形状和数据类型 |
| `normal_profiles.csv` | 已保存法向剖面的逐点 `offset_q`、`raw_intensity`、`fit_intensity` 和 `residual`（原始值减拟合值），按 `profile_index`、`point_id`、`sample_index` 对齐；并保留 `valid`、`model`、`reason`、`snr`、`normal_fwhm_q`、`localization_sigma_q`、`support_fraction`、`uncertainty_source`。无已保存 profile 的帧对应表格无记录 |

片层几何长表另含 `lamellar_parameters.csv`（周期来源与几何参数及状态）、`lamellar_directions.csv`（观测方位和方向支持记录）、`lamellar_period_by_angle.csv`（按角度采样对应椭圆分支的条件周期 `2π/R(β)`，另标注实测方向支持；采样曲线属于拟合模型）和 `lamellar_changes.csv`（可比较的序列变化）。`Ln_from_minor_axis = 2π/b` 保留为独立短轴参考；分支匹配不明确时不填方向周期。这些表保留输入参数来源，不将厚度、宽度或深度假设标为拟合测量。

逐帧目录还包含 `measurements.json`、`polar_measurements.npz` 和 `fit_details.json`。它们分别保存该帧统计摘要、二维 q-角度分箱数组和原有拟合详情。如果 `butterfly.profiles` 中保存了逐点法向剖面记录，会生成实测强度与已有局部拟合曲线图；有已保存 profile 残差时，另生成 residual 图。有 point/arc fit diagnostics 时，会生成 `frame####_fit_support` 图，显示脊点残差与距离、定位尺度、弧段残差、端点支持比例和支持计数。报告只绘制实际保存的数据，不为缺少 profile 或诊断的帧构造图。原生数组仍在批次的 `results.npz` 中；`array_catalog.csv` 和报告页面将帧与数组键对应起来。新增的原生 `frame_details.jsonl` 保留逐帧完整标量拟合、候选、观测支持、不确定度诊断及实际 profile 的数组引用。

## 读取统计量时的边界

- 报告继承源批次实际采用的有效像素域、mask、ROI 和 q-window。径向公共分箱按精确 q 单位分组，分箱数可调整；不同单位不会为了画在一起而猜测换算关系。
- `SEM` 是箱内像素样本标准差除以像素数平方根，描述空间采样变化。它不包含像素间相关性、校准误差或拟合参数不确定度，也不能替代批次已有的 uncertainty 结果。
- 强度以输入强度单位统计和绘图，不会自动转成绝对强度。只有输入数据本身经过相应校正和标定时，才可按其来源说明作物理解释。
- 面内加权矩和角度是给定图像、q-window 与有效域下的二维描述量，不唯一确定三维层片结构或变形机制。输出中的 `2π/q` 范围代理是倒易空间 q 范围的描述，不能当作识别出的径向反射峰 `q*` 或层片周期。
- 报告展示原有拟合的估计、候选和警告，不会将有限边界解、部分覆盖或失败帧改写成通过，也不会用镜像花瓣或插值补出未观测强度。

完整的批处理交付目录结构见[批次交付：数据、图与样品导航](batch_delivery_zh.md)；参数含义和单位解释见[科学量、符号、单位与可解释性边界](scientific_basis_zh.md)。
