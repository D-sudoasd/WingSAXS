# JAC manuscript and validation

WingSAXS is being evaluated for a **Computer Programs** paper in the
*Journal of Applied Crystallography*. This directory contains the working
manuscript and the evidence needed to interpret its claims.

- [English manuscript](manuscript.md): measurement definitions, implementation,
  comparison strategy and explicit missing experimental evidence.
- [Editable Word manuscript](manuscript.docx): the same draft with native
  Word equations; regenerate after editing the Markdown source.
- [Evidence and figure plan](evidence_plan.md): claim-to-evidence mapping,
  terminology and the proposed six-figure sequence.
- [Numerical results](numerical_results.md): measured results of the local
  development campaigns, with configurations and limitations.
- [Verification record](verification.md): regression checks, benchmark scope
  and document validation.

## Running the validation

Run scripts from the repository root in the installed development environment.
Use a new output directory for each campaign and retain the JSON reports.
The commands below are development runs; their sample counts do not establish
publication-scale coverage or experimental accuracy.

```powershell
py -3.13 scripts/compare_butterfly_trajectory_methods.py --output results/validation/jac_trajectories
py -3.13 scripts/validate_butterfly_uncertainty.py --output results/validation/jac_uncertainty --trials 10 --resamples 32 --shape 64
py -3.13 scripts/benchmark_initialization.py --output results/validation/jac_initialization
py -3.13 scripts/benchmark_initialization.py --known-arcs-only --output results/validation/jac_initialization_known_arcs
```

The [trajectory comparison protocol](../validation/trajectory_method_comparison.md)
defines the paired inputs and support-distance metrics. The
[initialization protocol](../validation/initialization_benchmark.md)
describes the sequence comparison. The uncertainty report separates the
number of attempted images, finite candidates, available intervals and
intervals containing the known reference. Deterministic calibration
perturbations are sensitivity calculations, not measured instrument errors.

To regenerate the interval figure from a completed report:

```powershell
py -3.13 scripts/plot_uncertainty_validation.py results/validation/jac_uncertainty/report.json --output results/validation/jac_uncertainty/intervals
```

The Word exporter is `scripts/export_jac_manuscript.cjs`. It requires the
`docx` and `jszip` Node packages (from the existing runtime or `NODE_PATH`).
Run it after editing the source:

```text
node scripts/export_jac_manuscript.cjs docs/jac/manuscript.md docs/jac/manuscript.docx
```

It regenerates that output document; edit the Markdown source to retain changes.
The two figures are embedded from the local validation outputs referenced by
the source. Keep those outputs when regenerating the illustrated draft; the
exporter reports a missing image instead of silently dropping it.

## Experimental evidence and independent use

Follow the [experimental protocol](../validation/jac_experimental_protocol.md)
and copy the [pending evidence template](../../examples/validation/jac_experimental_validation_template.json)
to a local data directory before filling it. Its audit command checks the
inventory and referenced files, not scientific validity or human success:

```powershell
py -3.13 scripts/audit_jac_validation_manifest.py examples/validation/jac_experimental_validation_template.json --output results/validation/jac_manifest_audit.json
```

The numerical campaigns distinguish localization error, recovered support,
parameter error, empirical interval coverage and initialization dependence.
They retain finite candidates and failed trials. Comparisons of different
trajectory definitions do not assume that the definitions are equivalent.

No experimental dataset or independent-user report was available when this
draft was prepared. The paper therefore does not claim experimental accuracy,
successful independent use or submission readiness. Generated numerical
results remain in the ignored `results/validation/` directory; the documentation
summarizes them without publishing private inputs or local provenance.

## 投稿准备范围

本轮优先补充可执行的定量验证和英文初稿。正文围绕“哪些二维图样参数仍由
观测支持，其变化是否超过不确定度”组织。真实实验、独立参照和两名独立
用户的成功使用必须由实际数据与记录补充，不能由自动测试替代。

基于固定 q 环的角向轨迹、扇区径向峰与曲率轨迹使用不同极值条件。本轮比较
已有应用路径，并加入固定 qy、沿 qx 找峰的切片基线；该基线不是文献中完整
切片分析流程的复现。
界面装饰、通用 SAXS 功能扩张和三维反演不属于本轮升级内容。

期刊要求以 [JAC 作者指南](https://journals.iucr.org/j/services/notesforauthors.html)
为准；当前材料是投稿准备成果，不能保证录用。
