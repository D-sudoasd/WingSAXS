# Numerical validation: local development campaigns

Date: 27 September 2026. These results describe the configurations below,
not experimental accuracy or a general ranking of extraction methods.
Raw numerical evidence remains under the ignored `results/validation/`
directory. The software baseline is 0.5.0 with the added validation tools;
each report records the source hashes used by its run.

## Same-image trajectory comparison

The campaign used six generated cases (`ellipse_ratio_100`, `partial_arcs`,
`missing_branch`, `overlapping_unresolved`, `smooth_nonellipse`, `null`),
two additional Gaussian-noise levels (0 and 0.02), and two mask conditions
(none and a paired wedge). The 24 paired image variants were 96 × 96 pixels,
with a common q window of 0.15–0.95 nm^-1. Each variant was supplied unchanged
to all four methods, including the fixed-qy Cartesian-slice baseline:
96 trace attempts, with no execution errors.

The zero additional-noise condition retains the generator's baseline noise.
The variants share generated cases and are not 24 independent repetitions.
The geometric generator is separate from the fitted empirical intensity
model but is not a physical SAXS forward model.

| Method | Localized points, excluding null controls | Pooled median distance (pixels) | Pooled 95th-percentile distance (pixels) | Mean variant support coverage | Accepted points on null controls |
|---|---:|---:|---:|---:|---:|
| Annular angular peak | 635 | 0.311 | 2.144 | 0.690 | 8 |
| Radial sector peak | 262 | 0.585 | 3.451 | 0.446 | 0 |
| Curvature ridge | 953 | 0.086 | 1.693 | 0.916 | 0 |
| Fixed-qy Cartesian slice | 500 | 0.644 | 2.403 | 0.697 | 0 |

Distances are to the densely sampled visible reference arcs, in units of
median adjacent-pixel q spacing. Support coverage is the fraction of visible
reference samples within 1.5 such pixel spacings of an accepted point, averaged
over variants with a reference. The tolerance is an operational benchmark
choice, not an experimentally validated resolution criterion. Pooled point
statistics are affected by the different sampling densities. The 8 null-control
points are extraction outputs, not 8 falsely accepted butterfly models: this
campaign stops at tracing and does not fit or select an ellipse.

The lower curvature distance in this matrix supports a condition-specific
comparison. It does not establish superiority for other arc widths,
backgrounds, detector sampling or experimental data. The Cartesian baseline
selects peaks at observed detector-pixel coordinates on a rectilinear grid,
without subpixel refinement or interpolation across masks. Differences in
sampling and localization resolution are part of this implemented-method
comparison. The full slice workflow from the literature has not been
reproduced here.

The report, per-point CSV files and saved input arrays allow each value to be
traced to its contributing variant. The accompanying PNG/SVG presents one
explicitly identified example; it is not a visual summary of all conditions.
The local bundle is
`results/validation/trajectory_method_comparison_20260927/run-02/`. The earlier
three-method `run-01/` is retained; its inputs and the original three methods'
results are unchanged in the four-method extension.

## Interval coverage and calibration sensitivity

The curvature pipeline was evaluated on 10 independently generated noisy
64 × 64 images of `ellipse_ratio_400`, with a = 0.72 nm^-1,
b = 0.288 nm^-1 and b/a = 0.4. Each image used 32 residual-block resamples,
a q window of 0.05–1.1 nm^-1 and a maximum of 300 fit evaluations. Seeds
started at 20260927. The report is
`results/validation/uncertainty_coverage_10x32_20260927/report.json`.
The run took 269.54 s and its recorded source hashes were unchanged.

| Parameter | Bias | RMSE | Median empirical interval width | Available intervals | Intervals containing truth |
|---|---:|---:|---:|---:|---:|
| a (nm^-1) | 0.0000152 | 0.000972 | 0.02349 | 10/10 | 10/10 |
| b (nm^-1) | 0.001639 | 0.001862 | 0.01264 | 10/10 | 10/10 |
| b/a | 0.002268 | 0.002369 | 0.009615 | 10/10 | 10/10 |

Both conditional and all-trial containment were 10/10 for these three
parameters. The 95% Wilson interval for each observed containment proportion
is approximately 0.722–1.000. Thus this run does **not** establish nominal
95% coverage; its precision is limited by the 10 outer trials. The empirical
parameter intervals were substantially wider than the between-trial RMSE in
this condition. More trials, resampling-endpoint stability and other noise
conditions are required before treating them as calibrated confidence
intervals. Axis-angle and radial q* coverage were not scored.

The separate partial-arc case returned finite internal fit values but failed
to converge, with a condition number of approximately 2.34 × 10^11. Its
major-axis candidate was 160.78 nm^-1 against the generator's 0.72 nm^-1,
and no uncertainty interval was available. The values and failed status are
retained as diagnostics, **not accepted measurements**. This is one stress
case, not a completed parameter-identifiability map.

A deterministic ±1% q-scale sensitivity calculation changed the fitted a by
−0.997%/+0.982% and b by −0.993%/+0.953%. The ratio changed by less than
0.03% relative. Four additional ±0.5-pixel centre-shift calculations are
retained in the report. These perturbations are assumed diagnostic values,
not measured instrument uncertainties, and calibration variation was not
included in the coverage intervals above.

`intervals.png` and `intervals.svg` in the coverage output directory show each
trial's fitted estimate and interval relative to the known reference. They
are derived directly from the report by `scripts/plot_uncertainty_validation.py`.

## Initialization and sequence direction

The first batch campaign used eight 192 × 192 finite-stack FFT images: six
signal frames with a nominal spacing change from 11.9 to 15.5 nm, one fully
masked frame and one noise-only control. Each processing strategy retained
seven warning frames and one failed frame. The fully masked frame remained
in place. Input and implementation hashes were unchanged during the run.

No fitted frame was eligible to seed a later frame: every recorded
`warm_start_from` was null. Although paired finite parameters agreed exactly,
**this run did not exercise warm-start initialization** and cannot demonstrate
its independence from processing direction. It verifies failure retention
and the observed absence of seed propagation for this configuration. A
supplementary campaign below therefore tests parameter propagation without
relaxing the eligibility criteria.

The separate 64 × 64 known-arc sequence contains four signal frames with
reference axis ratios 0.40, 0.38, 0.20 and 0.22, one fully masked frame after
the jump, and a noise-only control. All three processing strategies retained
five warning frames and one failed frame. Forward processing passed initial
parameters to 5/6 calls and reverse processing to 4/6 calls; these counts
include attempts on the masked frame. Four signal fits were seed-eligible.
The failed frame supplied no later seed and its fitted parameters remained
missing. The noise control was not seed-eligible.

Independent fits returned axis ratios 0.4027475, 0.3802388, 0.2004765 and
0.2209408. The abrupt change was retained. Across the four signal frames,
the maximum absolute forward–reverse differences were 2.84 × 10^-11 nm^-1
for a and 8.77 × 10^-12 for b/a. The maximum axis-ratio difference from
independent fitting was 4.83 × 10^-11. These differences describe the current
initialization policy, not robustness to different effective optimizer starts.
The optimizer's current-frame algebraic initialization replaces the supplied
free-parameter starts: a check on `arc_01` received a = 0.720600 and b/a =
0.402747 from the previous frame, but used a = 0.715203 and b/a = 0.379877 in
its first optimizer candidate. The near-zero differences therefore cannot be
attributed to convergence from those previous-frame starts. This test verifies
batch-to-pipeline propagation and failure retention; effective-start
sensitivity remains untested. Saved inputs and analysis sources were unchanged
during the run. Complete lineage and attempted fits are in
`results/validation/jac_initialization_known_arcs/`.
The effective-start inspection is saved separately in
`results/validation/jac_initialization_seed_effect_audit/effective_optimizer_start_probe.json`.
The sequence runner subsequently gained explicit optimizer-start fields and
CLI status reporting; the original campaign manifest retains its original
source hash rather than attributing those runs to the later reporting code.

## Evidence still needed

This development matrix does not include a complete factorial scan of arc
length, ellipse ratio, tilt, q window, background and noise. The trajectory
matrix also does not yet include dedicated near-circle and streak controls.
Repeated-trial angle coverage, joint uncertainty of changes across frames,
real-data repeatability, independent manual comparison and two independent-user
sessions remain to be demonstrated. None is replaced by software regression
tests or by a finite fitted candidate.
