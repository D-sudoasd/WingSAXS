# WingSAXS Repository Scripts

Standalone developer, benchmark, validation, and documentation utilities under `scripts/`.
Run Python scripts with the project virtual environment and `PYTHONPATH=src` (on PowerShell: `$env:PYTHONPATH = "$PWD\src"`).
None of these scripts modify raw inputs, PONI calibrations, or detector masks.

## Benchmarks (`benchmarks`)

| Script | Purpose |
|---|---|
| `benchmark_batch_delivery.py` | Time the end-to-end `bsaxs batch -> report -> package` CLI workflow in a fresh Python process. |
| `benchmark_checkpoint_writes.py` | Replay batch checkpoint records and measure synchronous checkpoint commit latency and throughput. |
| `benchmark_initialization.py` | Run the synthetic `independent` / `forward` / `reverse` initialization stability campaign. |
| `benchmark_sparse_ring.py` | Benchmark the sparse first-order sector kernel before and after index packing. |

## Validation (`validation`)

| Script | Purpose |
|---|---|
| `audit_jac_validation_manifest.py` | Read-only structural audit of a JAC real-data / user-evidence manifest without judging scientific acceptance. |
| `compare_batch_delivery.py` | Audit two complete benchmark/delivery bundles for numerical and artifact parity without loading an entire sequence into memory. |
| `compare_butterfly_trajectory_methods.py` | Compare the three WingSAXS butterfly trace methods (`annular_peak`, `radial_sector`, `curvature`) and a fixed-`qy` baseline on paired images. |
| `validate_butterfly_arcs.py` | Reproducible development/holdout evidence runner for butterfly arc tracing and symmetric-ellipse candidate fits (records recursive package SHA-256 digests). |
| `validate_butterfly_uncertainty.py` | Repeated-trial empirical coverage validation of butterfly resampling intervals on synthetic ground-truth images. |
| `validate_lamellar_sequence.py` | Generate and analyze an independent continuous lamellar SAXS evolution sequence for longitudinal validation. |

## Figures & Docs (`figures/docs`)

| Script | Purpose |
|---|---|
| `capture_readme_ui.py` | Render a deterministic README hero screenshot from the synthetic butterfly pattern. |
| `export_jac_manuscript.cjs` | Node.js exporter (`docx` + `jszip`) converting the JAC manuscript Markdown into an editable `.docx` document. |
| `plot_uncertainty_validation.py` | Plot saved empirical interval coverage summaries without rerunning or altering the underlying analysis. |
| `render_p3_overviews.py` | Render compact, read-only visual overview plates for P3 `T1`/`T2`/`R0` evidence directories. |
