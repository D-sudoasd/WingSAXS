# WingSAXS: measuring butterfly trajectories in small-angle X-ray scattering sequences

**Working draft for Journal of Applied Crystallography, Computer Programs.**
Software baseline: 0.5.0; validation additions developed in this working tree.
Authors, affiliations and corresponding author: **[AUTHOR INPUT REQUIRED]**.
This draft reports software capabilities and numerical validation only. Experimental
and independent-user results are explicitly pending.

## Synopsis

WingSAXS extracts observed trajectories from two-dimensional small-angle X-ray
scattering images and retains their measurement support, empirical uncertainty
and frame history. Its validation separates trajectory definitions, apparent
ellipse geometry and structural interpretation.

## Abstract

Butterfly-shaped small-angle X-ray scattering patterns contain curved intensity
features whose positions change during processing or deformation. Missing
lobes, overlapping reflections and weak intensity complicate both trajectory
extraction and the interpretation of fitted parameters. WingSAXS is a Python
program for inspecting calibrated two-dimensional images, extracting intensity
trajectories and following their apparent geometry through image sequences.
The program preserves measured support and finite constrained estimates,
distinguishes radial peak positions from fixed-radius angular maxima, and
reports limitations alongside fitted quantities. Image-level residual
resampling provides empirical parameter intervals; repeated simulations test
their coverage rather than assuming that percentile intervals are calibrated.
Comparisons use the same intensity array, mask and analysis window to examine
the dependence of trajectory position on the extraction definition. Sequence
tests compare independent initialization with forward and reverse warm starts.
In a 24-variant geometric benchmark, the four extraction paths differed in
both localization and recovered support. Ten repeated noisy-image trials
provided empirical intervals containing the reference a, b and b/a values in
every trial, although the sample size was insufficient to establish nominal
95% coverage. The resulting workflow supports reproducible assessment of
two-dimensional pattern measurements without identifying a fitted ellipse
with a unique three-dimensional lamellar structure. Experimental accuracy
and independent-user reproducibility remain to be established.

**Keywords:** small-angle X-ray scattering; butterfly patterns; intensity
trajectories; uncertainty; image sequences; scientific software.

## 1. Introduction

An in situ scattering experiment records a sequence of intensity fields, but
the scientific comparison is often made between a small number of parameters.
For butterfly patterns, the central measurement problem is to determine which
features remain constrained by the recorded intensity when lobes disappear,
reflections overlap or the signal weakens. A smooth sequence of fitted values
does not by itself demonstrate that each frame contains enough information to
support those values. The measurement record must retain the observed points,
the fit constraints and the frames for which an estimate is unavailable.

Elliptical trajectories already have an established role in the analysis of
oriented scattering patterns. Grubb et al. (2016) showed that distributions of
lamellar-stack properties can produce approximately elliptical reflections
without a unique structural cause. Murthy & Grubb (2024) discuss peak positions from
Cartesian slices and curvature trajectories, including the loss of usefulness
of a slice when it becomes parallel to the reflection contour. They also
distinguish pattern parameterization from non-unique structural modelling.
WingSAXS builds on this distinction: its contribution is a reproducible
measurement workflow and an explicit assessment of extraction and fitting
limitations, rather than the introduction of elliptical scattering geometry.

The question addressed here is whether a change in a measured pattern
parameter is larger than its measurement uncertainty under the support
available in each frame. Answering this question requires three linked tests:
comparison of trajectory definitions on identical images, evaluation of
parameter error and interval coverage, and assessment of initialization
dependence across a sequence. These tests define the scope of the software
evaluation and the experimental evidence required for its application.

## 2. Measurement definitions and implementation

### 2.1. Inputs and calibrated coordinates

The input is a two-dimensional intensity array with detector geometry, an
invalid-pixel mask and, when available, intensity uncertainties. The program
reads CBF, EDF, TIFF, NPY/NPZ and supported HDF5 datasets. PONI calibration is
interpreted through pyFAI (Ashiotis et al., 2015); a supplied physical q map can also define the
coordinates. The scattering-vector convention is

\[
q=4\pi\sin\vartheta/\lambda,
\]

where \(\vartheta\) is half the scattering angle and \(\lambda\) is the
wavelength. Coordinates expressed in pixel-q do not support conversion to
nanometres. A radial intensity peak at \(q^*\), measured in inverse nanometres,
defines the reciprocal-space period \(L=2\pi/q^*\). This quantity is recorded
separately from lengths conditional on an ellipse model.

The analysis domain is the intersection of finite intensity and coordinates,
valid detector pixels, the selected q window and valid weights, with external
masks and excluded regions removed. Missing pixels remain excluded. An
interpolated sample is not an additional detector observation. Inputs,
calibration and masks are preserved; analysis outputs are written separately.

### 2.2. Trajectories are defined by the direction of measurement

Let \(I(\mathbf q)\) be a differentiable intensity field. A local maximum
along a profile direction \(\mathbf t\) satisfies
\(\mathbf t\cdot\nabla I=0\) with negative second derivative along that
profile. Cartesian-slice maxima, radial maxima and angular maxima generally
use different directions. In polar coordinates the latter two satisfy
\(\partial I/\partial q=0\) and \(\partial I/\partial\chi=0\),
respectively. They therefore define different loci unless the local intensity
distribution makes the conditions coincide.

WingSAXS provides radial peak extraction, angular peak extraction on successive
q annuli, and curvature-based ridge extraction. The annular workflow connects
observed angular maxima into trajectories while retaining the profiles,
counts and coverage of individual rings. Its radius is a sampling coordinate,
not a radial peak measurement. The curvature workflow uses local intensity
derivatives and observed support to identify ridge candidates. In the current
implementation the intensity is treated as a height surface over scaled
physical q coordinates. The first and second fundamental forms give its
principal curvatures; the smallest principal curvature, its direction and a
zero of the directional slope define a candidate. Smoothing, intensity
scaling and coordinate scaling are therefore part of the extraction settings
and must accompany a reported trajectory. A numerical
curvature implementation is not presumed identical to every published
curvature construction. The comparator definitions and settings are therefore
recorded with the benchmark.

For directional comparison, the benchmark adds a fixed-qy Cartesian-slice
baseline that locates local intensity peaks along qx on the rectilinear
synthetic grid, retaining detector-pixel coordinates without subpixel
refinement. Peak selection operates within contiguous valid segments,
so masked gaps are not bridged. Its recorded prominence and noise criteria
make the definition reproducible. This baseline tests the fixed-coordinate
measurement direction; it is not a full reproduction of the published
slice-analysis workflow.

The fixed-radius angular profile can shift its maximum when the intensity
envelope varies along a broad arc, even if the geometric centre of that arc
is unchanged. Near overlap, the maximum can instead follow the sum of two
features. These are differences in the measured intensity field, not errors
that can be resolved by imposing agreement between trajectories. The
comparison reports distance to known synthetic arcs together with the amount
of recovered support; low distance alone would reward an extractor that
returns only a small, easy portion of the image.

### 2.3. Apparent geometry and parameter support

Observed trajectory points can be fitted by the supported ellipse
parameterization. The principal reported quantities are the semi-major axis
\(a\), the axis ratio \(b/a\), and the ellipse-axis angle relative to a
declared reference direction. For a two-ellipse model, the branch assignment
and the occupied sides are retained. Missing opposite lobes are not generated
by reflection. Bounds and fixed parameters are explicit fit assumptions.

A finite solution may remain useful even when its major axis is extrapolated
or a parameter reaches a bound. Such a solution is retained as an estimate or
candidate with the corresponding limitation. Conversely, an axis direction
becomes poorly identifiable as the ellipse approaches a circle. A numerical
angle from that case does not establish a preferred direction. The existing
near-circle threshold is a diagnostic heuristic rather than a confidence test.

The quantities \(a\), \(b/a\) and the ellipse-axis angle describe the
two-dimensional pattern. Conditional Ln/Lz estimates require the model
assumptions and physical units documented with the result. Neither these
quantities nor fit convergence establish a unique lamellar tilt, stack
rotation or three-dimensional morphology. The optional full2d model is an
empirical intensity model and is not run implicitly by geometry analysis.

### 2.4. Empirical uncertainty and calibration sensitivity

The image-level uncertainty routine resamples spatial residual blocks and
refits each perturbed image. It reports the 2.5th and 97.5th percentiles of the
eligible fitted values as a central empirical interval. Failed refits and loss
of trajectory topology are counted separately and excluded from those
quantiles. Consequently, an interval can describe only the surviving refits
when extraction frequently fails. A fixed parameter is not a measurement with
zero uncertainty.

Coverage is assessed across independently generated noisy images, using the
known simulated parameter as the reference. For each parameter the report
retains the number of attempted images, finite estimates, available intervals
and intervals containing the reference. Coverage conditional on an available
interval is reported separately from the fraction of all trials yielding an
interval that contains the reference. This distinction prevents unsuccessful
analyses from disappearing from the evaluation.

A common q-scale perturbation \(q'=(1+\epsilon)q\) gives
\(L'=L/(1+\epsilon)\). For small relative errors,
\(\delta L/L\simeq-\epsilon\), while ideal ellipse ratios and angles
are invariant to an isotropic coordinate scaling. The validation distinguishes
this coordinate sensitivity from uncertainty caused by intensity noise.
Detector-centre and geometry errors can have additional effects and require
instrument-specific calibration evidence. Uncertainty shared by frames is
correlated: overlapping or non-overlapping marginal intervals alone are not
a test of a change between frames.

### 2.5. Sequence processing and reproducibility

Each frame is fitted to its own pixels. Warm starts supply initialization from
an eligible preceding fit, with its source frame recorded; they do not add a
temporal smoothness penalty. A subsequent data-derived initializer can replace
free-parameter starts, so supplied parameters and effective optimizer starts
must be distinguished. A failed frame keeps its position and reason in
the exported sequence. The validation compares independent fits with forward
and reverse processing of the same frames, including an abrupt change.
Agreement is evaluated after matching frame identifiers rather than row order.
Disagreement indicates sensitivity to initialization or available support and
is retained for inspection.

The Python package is named `butterfly-saxs`, with the import name
`butterfly_saxs` and command-line entry point `bsaxs`. Python 3.11–3.13 is
supported. NumPy and SciPy provide numerical operations, pyFAI supplies
calibrated geometry, and the optional PySide6 interface provides interactive
inspection. The service layer is independent of Qt. Machine-readable results
and diagnostic exports allow inspection outside the graphical interface.
Hardware and elapsed time belong to each benchmark record; a universal
minimum-memory or performance claim is not made here.

## 3. Numerical evaluation

### 3.1. Comparison on identical images

The comparison uses shared intensity arrays, q coordinates, masks and windows
(Fig. 1).
Geometric arc images provide an explicit localization reference, while the
existing finite real-space lamellar FFT generators provide a separate test of
intensity features. The FFT-generated structural period, noiseless radial
intensity peak and extracted trajectory are distinct references. A geometric
arc test does not validate a lamellar forward model, and an FFT test does not
make every extracted trajectory equal to the structural reciprocal period.

Localization error is reported with support and failure counts for each
method and condition. Pure-noise controls test whether extraction returns
spurious features; ring and streak controls are specified for the extended
protocol. Producing points on a negative control is not
equivalent to accepting a butterfly model; extraction and model-selection
outcomes must be counted separately. Timing comparisons require the same
machine and explicit sampling and optimization budgets.

A development campaign evaluated six geometric cases under two additional
noise levels and two mask conditions, giving 24 paired 96 × 96 pixel variants
and 96 trace attempts. The q window was 0.15–0.95 nm^-1. No trace attempt
raised an execution error. Median distances from accepted points to the
sampled visible reference arcs were 0.311, 0.585, 0.086 and 0.644 pixel spacings
for annular, radial-sector, curvature and Cartesian-slice extraction,
respectively. Mean recovered support at a tolerance of 1.5 pixel spacings was
0.690, 0.446, 0.916 and 0.697.
These pooled distances depend on point sampling density and describe this
generator and configuration; they are not independent-image uncertainty
estimates or a general method ranking.

Annular extraction returned a total of eight accepted points across four
noise-only variants, whereas the other three methods returned none. This is an
extraction-level negative-control result, not a rate of falsely accepted
ellipse models, because fitting was not performed in this comparison. The
per-variant records and numerical definitions are supplied in the companion
`numerical_results.md` report. Additional near-circle and streak controls and
a systematic background scan remain necessary.

### 3.2. Error, coverage and initialization dependence

Parameter bias and root-mean-square error are evaluated against the same
parameter definition used to generate the image. Interval width and empirical
coverage are interpreted jointly: a wide interval may cover the reference
while providing little ability to resolve a change. The number of independent
images determines the precision of a coverage estimate, whereas the number
of image resamples determines how well its percentile endpoints are resolved.
Small development runs are reported as such.

For 10 independent 64 × 64 noisy images with reference values a = 0.72 nm^-1
and b/a = 0.4, 32 image resamples per trial produced available intervals for
a, b and b/a in every trial (Fig. 2). All ten intervals contained the corresponding
reference for each parameter. The 95% Wilson interval for the containment
proportion was 0.722–1.000, so the observed 10/10 result cannot establish
nominal 95% coverage. The RMSE values were 0.000972 nm^-1 for a,
0.001862 nm^-1 for b and 0.002369 for b/a. Median empirical interval widths
were 0.02349 nm^-1, 0.01264 nm^-1 and 0.009615, respectively. This condition
therefore also illustrates that containment must be considered alongside
interval width. Axis-angle and radial peak coverage were not evaluated.

A separate short-arc test retained a finite internal solution but did not
converge; its condition number was approximately 2.34 × 10^11 and no interval
was available. Its a value of 160.78 nm^-1, compared with a reference of
0.72 nm^-1, is retained with the failure record and is not treated as a
measurement. Deterministic ±1% q-scale perturbations of the supported case
changed its fitted semi-axes by approximately 1%, while the relative change
in b/a remained below 0.03%. These perturbations quantify sensitivity to an
assumed coordinate change, not the uncertainty of an experimental calibration.

The sequence comparison examines whether each processing direction follows
the observations at an abrupt transition and whether it retains failures
without substituting a previous estimate. The resulting differences are an
optimization diagnostic, not evidence that one processing direction is the
physical truth. The complete per-frame record belongs in supporting data;
conditions that materially change the conclusion remain in the main text.

In the initial eight-frame finite-stack FFT campaign, all three strategies
retained seven warning frames and one fully masked failure. No result was
eligible to initialize a subsequent frame, and the recorded seed sources
were all absent. Exact agreement between the finite estimates therefore
reflects runs without active seed propagation; it does not establish
warm-start robustness. This distinction is retained in the report rather
than interpreting zero disagreement as a positive validation result.

A supplementary 64 × 64 known-arc sequence exercised parameter propagation
with reference axis ratios 0.40, 0.38, 0.20 and 0.22, a fully masked frame
after the jump, and a noise-only control. Forward and reverse processing
passed initial parameters to five and four of the six calls, respectively,
including masked-frame attempts. Independent fitted ratios were 0.40275,
0.38024, 0.20048 and 0.22094. The maximum forward–reverse difference in b/a
was 8.77 × 10^-12 across the four signal frames. The failed frame retained
missing parameters and never supplied a later seed; the noise control was
also ineligible to seed a fit. Inspection of the actual optimizer starts
showed that the current-frame algebraic initialization subsequently replaced
the supplied free-parameter starts. Thus this control verifies parameter
propagation, failure isolation and preservation of the abrupt change under
the existing initialization policy. It does not demonstrate robustness to
different effective optimizer starting values. Such a test must record the
starts actually used by the optimizer, in addition to batch-level provenance.

## 4. Experimental and independent-user validation

**[EXPERIMENTAL RESULTS REQUIRED.]** No experimental dataset was supplied
for this draft. The primary study should analyse a complete calibrated
sequence with its acquisition variable, background and correction history,
including weak and failed frames. Repeated exposures or independently traced
positions should provide a measurement reference with its own uncertainty.
A separate sample, batch or geometry should then be processed with settings
chosen before inspecting its outcomes. Report any manual intervention and
its time cost; do not replace difficult frames with selected examples.

**[INDEPENDENT-USER RESULTS REQUIRED.]** The experimental protocol and user
record template accompany the software. Two researchers who did not develop
the program should each attempt installation, calibrated single-frame
analysis, sequence processing and export using the public documentation.
Their actual outcomes, assistance and unresolved problems must be recorded.
No independent-user success is claimed in this draft.

## 5. Discussion

The useful unit of output is a measured parameter together with the
observations and assumptions that support it. Retaining a finite candidate
allows a weak or partly obscured frame to remain inspectable, but does not
make that candidate interchangeable with a well-constrained measurement.
Method comparisons and coverage experiments supply the evidence needed to
decide how much weight to give it.

The distinction between trajectory definitions is central to interpreting
method disagreement. If two extractors sample different intensity extrema,
their fitted ellipses need not agree even in a noiseless image. A comparison
should first identify the observable required by the scientific question,
then quantify error relative to that observable. Agreement among methods is
useful corroboration but cannot establish a unique structural interpretation.

The current numerical evaluation addresses controlled image conditions.
Experimental backgrounds, detector response and correlated calibration errors
can differ from the simulation assumptions. Repeat exposures and independent
measurements are therefore required before claiming experimental precision
or interpreting small changes during processing. Additional WAXS or
microstructural evidence is needed only where a structural claim depends on
it; a paper restricted to reliable two-dimensional measurement can maintain
a narrower and testable contribution.

## 6. Availability and reuse

WingSAXS is distributed under the MIT licence at
[the project repository](https://github.com/D-sudoasd/WingSAXS). Installation,
input requirements and interpretation are described in the repository
documentation. The numerical validation scripts retain configurations, seeds,
failure records and provenance in separate output directories. Local
validation artifacts are not automatically published.

**[Before submission: identify the immutable software revision and public
example-data archive, provide complete environment and hardware information,
verify anonymous download and execution, and deposit the source data for
every reported numerical figure. Supply data permissions and availability
statements for experimental inputs.]**

## Acknowledgements and declarations

**[AUTHOR INPUT REQUIRED: funding, contributions, competing interests and
data permissions.]** OpenAI Codex assisted with software development,
validation preparation and drafting. The authors must verify the code,
results, references and final text before submission; author verification
has not been represented as completed here.

## Figures

![Trajectory extraction on a shared synthetic input](../../results/validation/trajectory_method_comparison_20260927/run-02/trajectory_method_comparison.png)

**Figure 1.** Trajectory extraction on the shared synthetic example identified
in the panel headings. All methods receive the same intensity, q coordinates,
mask and q window. Reference arcs specify visible generator geometry;
extracted points specify each algorithm's output. Equal q-axis scales preserve
the geometry. This example illustrates differing sampling directions and
support; the numerical comparison uses all 24 variants rather than this image
alone. Cartesian slicing is the explicitly defined benchmark baseline, not
a full reproduction of a published implementation.

![Empirical parameter intervals across independent noisy images](../../results/validation/uncertainty_coverage_10x32_20260927/intervals.png)

**Figure 2.** Fitted estimates and empirical central 95% intervals for ten
independently generated noisy images, each with 32 spatial residual-block
resamples. Values are expressed relative to the known a, b and b/a references.
All displayed intervals contain the reference in this condition. The limited
number of trials and broad intervals prevent interpreting this containment
as established nominal 95% coverage or experimental precision. Calibration
variation is excluded from these intervals.

## References

Ashiotis, G., Deschildre, A., Nawaz, Z., Wright, J. P., Karkoulis, D.,
Picca, F. E. & Kieffer, J. (2015). The fast azimuthal integration Python
library: pyFAI. *Journal of Applied Crystallography*, **48**, 510–519.
[Publisher article](https://journals.iucr.org/j/issues/2015/02/00/fv5028/).

Grubb, D. T., Murthy, N. S. & Francescangeli, O. (2016). Elliptical
small-angle X-ray scattering patterns from aligned lamellar arrays.
*Journal of Polymer Science Part B: Polymer Physics*, **54**, 308–318.
[Publisher article](https://doi.org/10.1002/polb.23930).

Murthy, N. S. & Grubb, D. T. (2024). Evolution of elliptical SAXS patterns in
aligned systems. *Journal of Applied Crystallography*, **57**, 1127–1136.
[Publisher article](https://journals.iucr.org/j/issues/2024/04/00/tu5052/).

**[Complete implementation references for additional methods from their
verified primary publications before submission. Consult the Grubb 2016 and
2021 primary papers for any detailed structural interpretation added later.]**
