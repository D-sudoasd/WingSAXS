# Butterfly trajectory-method comparison

This protocol compares three WingSAXS trace methods and one explicitly defined
fixed-qy Cartesian slice baseline on paired synthetic images. Every method
receives identical image pixels, q coordinates, invalid-pixel mask, and q
window for a given case/noise/mask variant. The run archives those inputs and
writes per-point records so the scores can be checked against the trace outputs.

## Methods compared

| Method identifier | Observable extracted | Important distinction |
|---|---|---|
| `annular_peak` | Angular intensity maxima on successive prescribed q annuli, linked across adjacent annuli | The annulus is a sampling coordinate; it is not necessarily a radial reflection peak or the local ridge normal. |
| `radial_sector` | A radial-profile maximum measured in each finite azimuthal sector and linked across sector centers | This is an azimuthal-sector method. It is not the Cartesian x-slice method described in the cited SAXS literature. |
| `curvature` | Multiscale principal-curvature ridge candidates, followed by local profile refinement | The benchmark exercises WingSAXS' implementation. It does not establish equivalence to any published curvature algorithm. |
| `cartesian_slice` | Local maxima of observed intensity along each fixed-qy detector row, scanned in qx | This is a simple baseline on a rectilinear q grid, not a full reproduction of the slice algorithm in Murthy and Grubb (2024) or another published method. |

The fixed-qy baseline requires a rectilinear q grid: qy must be constant across
each detector row, qx constant down columns, and qx strictly monotonic across
columns. For every row, the baseline smooths each contiguous supported run
independently with a Gaussian sigma of 1 detector bin. It calls an observed
local maximum only when its prominence exceeds the larger of four times the
profile-local robust noise estimate or 2% of the run's intensity range; peaks
must be at least 2 bins apart and have width of at least 0.8 bins. Peak height
above the robust 10th-percentile baseline must also exceed four noise scales.
Masked, non-finite, and q-window-excluded pixels split runs. The method does
not interpolate, bridge gaps, link peaks between rows, or impose butterfly
topology. Its output is therefore a fixed-coordinate slice baseline, not a
literature-method equivalence claim.

## Inputs and truth

The runner uses `butterfly_saxs.benchmark_arcs.generate_arc_case`, which
rasterizes known observable arcs onto an image and adds a small baseline
Gaussian noise term. The added noise levels in this protocol are additional
image-intensity noise; a requested extra sigma of zero means that no additional
noise was added, not that the image is noiseless. The optional `wedge` mask
excludes the same detector pixels for all four methods. The run-02 matrix uses
the same six cases, two additional-noise levels, and two masks as run-01 (24
paired image variants). The available cases
include full and partial ellipse arcs, a missing branch, unresolved overlap,
asymmetric or miscentered branches, a warped q map, a smooth non-elliptic
curve, and a background-only negative control.

The truth is the generator-declared observable support. It is not an
independent physical SAXS forward model and does not validate a material
interpretation. The non-elliptic case tests curve localization, not automatic
classification of ellipse versus non-ellipse. The null case has no target
support; accepted points are retained as false detections instead of being
scored as localization errors.

## Scores

The runner reports accepted and rejected point counts, visible truth-arc counts,
branch/side assignment, point-to-support distance, and sampled-support
coverage. Truth polylines are densified at half the median adjacent-pixel q
spacing, clipped to the analysis q window, and filtered by the supplied mask
using the nearest detector pixel. Point distances are nearest-neighbor
distances to that visible sampled support, not exact point-to-segment normal
distances. They are reported in physical q and in units of the median adjacent
pixel q spacing.

For descriptive coverage, the default tolerance is 1.5 median pixel-q spacings.
This threshold is an operational synthetic benchmark setting, not an
instrument-calibrated resolution, scientific acceptance criterion, or
confidence interval. `median_*` and `p95_*` summarize extracted points for
which visible truth support exists. `visible_truth_sample_cloud_coverage` is
the fraction of visible truth samples having an accepted extracted point
within that tolerance. Empty support is reported as missing/undefined; it is
not assigned a zero localization error.

All methods run at `stage=trace`; no ellipse fit or uncertainty resampling is
performed. Method-specific sampling settings are recorded because the methods
use different coordinates and sampling schemes. This comparison is
about their measured trajectories, not a claim that their parameter estimates
are interchangeable or that one method is universally superior.

## Reproduction

From the repository root, run for example:

```text
py -3.13 scripts/compare_butterfly_trajectory_methods.py \
  --output results/validation/trajectory_method_comparison_20260927/run-02 \
  --cases ellipse_ratio_100,partial_arcs,missing_branch,overlapping_unresolved,smooth_nonellipse,null \
  --noise-sigmas 0,0.02 --mask-variants none,wedge
```

The output contains `summary.json`, flat summary CSV files, saved paired input
NPZ files, per-method point CSV files, truth metadata, and a 2 x 2 PNG/SVG
overview with one panel per method. Every panel shows the same image and
generator-declared observable support. Existing generated output files are replaced only when `--force` is
given; unknown files in the target directory prevent replacement. Standard
output contains one strict JSON completion record, while case progress is
written to standard error.

## Scope limits

This matrix is synthetic only. It does not replace experimental validation,
repeat-exposure or independent-user evidence, uncertainty-interval coverage,
calibration-error sensitivity, or parameter-identifiability analysis. The
Cartesian baseline is not a full published slice-method implementation, and
the WingSAXS curvature method is not claimed equivalent to a published
curvature method. Direct comparisons to those algorithms require separately
specified implementations and common test inputs.
