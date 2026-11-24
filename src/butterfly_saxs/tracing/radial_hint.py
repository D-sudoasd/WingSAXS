"""First-order radial I(q) peak hints, trajectory support reconciliation, and sparse ring sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import sys
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from .topology import (
    _assign_point_ids,
    _assign_reference_branches,
    _demote_unresolved_identity_point,
    _point_identity_key,
    _side_from_reference_axis,
)

FIRST_ORDER_PEAK_MIN_BIN_COUNT = 8
FIRST_ORDER_PEAK_MIN_TOTAL_SUPPORTED_BINS = 8
FIRST_ORDER_PEAK_MIN_CONTIGUOUS_RUN_BINS = 8
FIRST_ORDER_PEAK_MIN_EFFECTIVE_SAMPLES = 2.0
FIRST_ORDER_PEAK_NOISE_SIGMA_BINS = 0.8
FIRST_ORDER_PEAK_PROMINENCE_NOISE_MULTIPLE = 3.0
FIRST_ORDER_PEAK_AMBIGUITY_NOISE_MULTIPLE = 1.0
FIRST_ORDER_PEAK_AMBIGUITY_MIN_WIDTH_BINS = 2.0
FIRST_ORDER_PEAK_FAMILY_SPAN = 1.45
RADIAL_HINT_SMOOTHING_FWHM_Q_PIXEL_FRACTION = 0.9
RADIAL_HINT_MAX_SMOOTHING_SIGMA_BINS = 4.0
RADIAL_HINT_CACHE_MAX_BYTES = 64 * 1024 * 1024


def _resolve_prepare_sparse_samples():
    br_mod = sys.modules.get("butterfly_saxs.butterfly_ridge")
    if br_mod is not None:
        return getattr(
            br_mod,
            "_prepare_sparse_first_order_samples",
            _prepare_sparse_first_order_samples,
        )
    return _prepare_sparse_first_order_samples


def _contiguous_profile_runs(usable: np.ndarray) -> list[tuple[int, int]]:
    transitions = np.diff(np.concatenate(([False], usable, [False])).astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if int(stop - start) >= FIRST_ORDER_PEAK_MIN_CONTIGUOUS_RUN_BINS
    ]


def _radial_profile_noise_mad(
    profile: np.ndarray,
    supported_runs: Sequence[tuple[int, int]],
) -> float | None:
    if gaussian_filter1d is None:
        return None
    residuals = [
        profile[start:stop]
        - gaussian_filter1d(profile[start:stop], sigma=FIRST_ORDER_PEAK_NOISE_SIGMA_BINS)
        for start, stop in supported_runs
    ]
    if not residuals:
        return None
    high_pass = np.concatenate(residuals)
    high_pass = high_pass[np.isfinite(high_pass)]
    if high_pass.size == 0:
        return None
    centre = float(np.median(high_pass))
    return float(1.4826 * np.median(np.abs(high_pass - centre)))


def _radial_q_pixel_step(
    q: np.ndarray,
    valid: np.ndarray,
    q_min: float,
    q_max: float,
) -> float | None:
    """Estimate one reciprocal-space pixel from adjacent radial-q samples.

    A radial q map on a detector grid has smaller changes along directions
    tangent to a ring and larger changes across it.  Taking the median of both
    detector-axis differences, then multiplying by ``sqrt(2)``, estimates the
    magnitude of one pixel step for a locally isotropic q grid.  One-dimensional
    profiles do not carry enough geometry for this estimate and return None.
    """

    q_array = np.asarray(q, dtype=float)
    if q_array.ndim != 2 or min(q_array.shape) < 2:
        return None
    selected = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(q_array)
        & (q_array >= float(q_min))
        & (q_array <= float(q_max))
    )
    axis_steps: list[float] = []
    for axis in (0, 1):
        left_slice = [slice(None), slice(None)]
        right_slice = [slice(None), slice(None)]
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)
        left = tuple(left_slice)
        right = tuple(right_slice)
        paired = selected[left] & selected[right]
        differences = np.abs(q_array[right] - q_array[left])
        positive = differences[paired & np.isfinite(differences) & (differences > 0.0)]
        if positive.size:
            axis_steps.append(float(np.median(positive)))
    if len(axis_steps) != 2:
        return None
    step = float(np.sqrt(2.0) * np.median(axis_steps))
    return step if np.isfinite(step) and step > 0.0 else None


def _effective_radial_bin_samples(
    bin_indices: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Return scale-stable Kish effective sample counts for radial bins."""

    maximum = np.zeros(int(n_bins), dtype=float)
    np.maximum.at(maximum, bin_indices, weights)
    scales = maximum[bin_indices]
    scaled = np.divide(
        weights,
        scales,
        out=np.zeros_like(weights, dtype=float),
        where=scales > 0.0,
    )
    total = np.bincount(bin_indices, weights=scaled, minlength=int(n_bins))
    squared_total = np.bincount(bin_indices, weights=scaled * scaled, minlength=int(n_bins))
    return np.divide(
        total * total,
        squared_total,
        out=np.zeros(int(n_bins), dtype=float),
        where=squared_total > 0.0,
    )


class _RadialHintGeometryCache:
    """One-analysis cache for radial bins, with exact geometry/support checks."""

    def __init__(self, max_bytes: int = RADIAL_HINT_CACHE_MAX_BYTES) -> None:
        self.max_bytes = int(max_bytes)
        self._radii: np.ndarray | None = None
        self._selected: np.ndarray | None = None
        self._edges: np.ndarray | None = None
        self._bin_indices: np.ndarray | None = None
        self._counts: np.ndarray | None = None

    def _clear(self) -> None:
        self._radii = None
        self._selected = None
        self._edges = None
        self._bin_indices = None
        self._counts = None

    def get(
        self,
        radii: np.ndarray,
        selected: np.ndarray,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return bin labels/counts, reusing them only for identical inputs."""

        if (
            self._radii is not None
            and self._selected is not None
            and self._edges is not None
            and self._bin_indices is not None
            and self._counts is not None
            and np.array_equal(edges, self._edges)
            and np.array_equal(selected, self._selected)
            and np.array_equal(radii, self._radii)
        ):
            return self._bin_indices, self._counts

        n_bins = int(edges.size - 1)
        bin_indices = np.searchsorted(edges, radii, side="right") - 1
        np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)
        counts = np.bincount(bin_indices, minlength=n_bins)

        if n_bins <= np.iinfo(np.uint8).max + 1:
            compact_indices = bin_indices.astype(np.uint8)
        elif n_bins <= np.iinfo(np.uint16).max + 1:
            compact_indices = bin_indices.astype(np.uint16)
        else:
            compact_indices = bin_indices
        cache_bytes = int(
            radii.nbytes + selected.nbytes + edges.nbytes
            + compact_indices.nbytes + counts.nbytes
        )
        if cache_bytes <= self.max_bytes:
            self._radii = np.asarray(radii, dtype=float).copy()
            self._selected = np.asarray(selected, dtype=bool).copy()
            self._edges = np.asarray(edges, dtype=float).copy()
            self._bin_indices = compact_indices
            self._counts = counts
            for cached in (self._radii, self._selected, self._edges, self._bin_indices, self._counts):
                cached.flags.writeable = False
        else:
            self._clear()
        return bin_indices, counts


def _first_order_q_hint(
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    q_min: float,
    q_max: float,
    *,
    n_bins: int = 96,
    geometry_cache: _RadialHintGeometryCache | None = None,
) -> dict[str, Any]:
    """Select a supported radial peak only when prominence clears profile noise.

    A lower-q family that is visibly present but below the significance limit
    makes the first order ambiguous; it must not cause a brighter harmonic to
    be promoted. Noise-sized one-bin shoulders are ignored.
    """

    summary: dict[str, Any] = {
        "q_star": None,
        "band": None,
        "n_bins": int(n_bins),
        "radial_q_pixel_step": None,
        "profile_smoothing_sigma_bins": 0.0,
        "profile_support_method": "annular_bin_pixel_count",
        "reason": None,
        "selection_status": "not_selected",
        "selection_method": "lowest_significant_supported_radial_peak_family",
        "profile_noise_estimator": "1.4826*MAD(profile - gaussian_filter1d(profile, sigma=0.8 bins))",
        "profile_noise_sigma_bins": FIRST_ORDER_PEAK_NOISE_SIGMA_BINS,
        "minimum_bin_count": FIRST_ORDER_PEAK_MIN_BIN_COUNT,
        "minimum_total_supported_bins": FIRST_ORDER_PEAK_MIN_TOTAL_SUPPORTED_BINS,
        "minimum_contiguous_run_bins": FIRST_ORDER_PEAK_MIN_CONTIGUOUS_RUN_BINS,
        "minimum_effective_samples_per_peak_bin": FIRST_ORDER_PEAK_MIN_EFFECTIVE_SAMPLES,
        "significance_prominence_noise_multiple": FIRST_ORDER_PEAK_PROMINENCE_NOISE_MULTIPLE,
        "ambiguity_prominence_noise_multiple": FIRST_ORDER_PEAK_AMBIGUITY_NOISE_MULTIPLE,
        "ambiguity_min_width_bins": FIRST_ORDER_PEAK_AMBIGUITY_MIN_WIDTH_BINS,
        "peak_family_span": FIRST_ORDER_PEAK_FAMILY_SPAN,
        "n_supported_bins": 0,
        "n_supported_runs": 0,
        "profile_noise_mad": None,
        "effective_noise_floor": None,
        "significance_prominence_threshold": None,
        "ambiguity_prominence_floor": None,
        "candidate_peaks": [],
        "n_significant_peaks": 0,
    }
    selected = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(q)
        & np.isfinite(intensity)
        & (q >= q_min)
        & (q <= q_max)
    )
    if int(np.count_nonzero(selected)) < 200:
        summary["reason"] = "insufficient_pixels"
        summary["selection_status"] = "no_hint"
        return summary
    radii = np.asarray(q[selected], dtype=float)
    weights = np.clip(np.asarray(intensity[selected], dtype=float), 0.0, None)
    edges = np.linspace(float(q_min), float(q_max), int(n_bins) + 1)
    if geometry_cache is None:
        bin_indices = np.searchsorted(edges, radii, side="right") - 1
        np.clip(bin_indices, 0, int(n_bins) - 1, out=bin_indices)
        counts = np.bincount(bin_indices, minlength=int(n_bins))
    else:
        bin_indices, counts = geometry_cache.get(radii, selected, edges)
    summed = np.bincount(bin_indices, weights=weights, minlength=int(n_bins))
    effective_samples = _effective_radial_bin_samples(bin_indices, weights, int(n_bins))
    raw_profile = np.divide(
        summed,
        counts,
        out=np.zeros_like(summed, dtype=float),
        where=counts > 0,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    raw_usable = (
        np.isfinite(raw_profile)
        & (counts >= FIRST_ORDER_PEAK_MIN_BIN_COUNT)
    )
    raw_supported_runs = _contiguous_profile_runs(raw_usable)
    summary["n_raw_supported_bins"] = int(np.count_nonzero(raw_usable))
    summary["n_raw_supported_runs"] = len(raw_supported_runs)
    summary["n_supported_bins"] = int(np.count_nonzero(raw_usable))
    summary["n_supported_runs"] = len(raw_supported_runs)
    if summary["n_raw_supported_bins"] < FIRST_ORDER_PEAK_MIN_TOTAL_SUPPORTED_BINS:
        summary["reason"] = "sparse_radial_profile"
        summary["selection_status"] = "no_hint"
        return summary
    if not raw_supported_runs:
        summary["reason"] = "insufficient_contiguous_support"
        summary["selection_status"] = "no_hint"
        return summary

    # A detector grid samples q on a Cartesian lattice.  Narrow radial bins
    # can then alternate between populated and empty even where the measured
    # annular intensity is continuous; using only those runs can discard a
    # real first-order peak (especially at low detector resolution).  Smooth
    # the binned weighted sums and counts together over approximately one
    # detector-pixel FWHM in q.  This averages observed samples only and keeps
    # the original pixel-level effective-sample check for outlier rejection.
    radial_q_pixel_step = _radial_q_pixel_step(q, valid, q_min, q_max)
    smoothing_sigma_bins = 0.0
    profile = raw_profile
    support_counts = np.asarray(counts, dtype=float)
    if radial_q_pixel_step is not None and gaussian_filter1d is not None:
        bin_width = float(edges[1] - edges[0])
        if np.isfinite(bin_width) and bin_width > 0.0:
            smoothing_sigma_bins = min(
                RADIAL_HINT_MAX_SMOOTHING_SIGMA_BINS,
                RADIAL_HINT_SMOOTHING_FWHM_Q_PIXEL_FRACTION
                * radial_q_pixel_step
                / (2.354820045 * bin_width),
            )
            if smoothing_sigma_bins > 0.0:
                smooth_summed = gaussian_filter1d(
                    summed, smoothing_sigma_bins, mode="constant", cval=0.0
                )
                support_counts = gaussian_filter1d(
                    np.asarray(counts, dtype=float),
                    smoothing_sigma_bins,
                    mode="constant",
                    cval=0.0,
                )
                profile = np.divide(
                    smooth_summed,
                    support_counts,
                    out=np.zeros_like(smooth_summed, dtype=float),
                    where=support_counts > 0.0,
                )
                # Compute exact Kish N_eff for the Gaussian-smoothed bin sums so
                # lattice-empty bins between valid neighbours remain eligible
                # while single-pixel outliers still yield N_eff ~ 1.
                summed_sq = np.bincount(
                    bin_indices, weights=np.square(weights), minlength=int(n_bins)
                )
                kernel_radius = max(1, int(4.0 * smoothing_sigma_bins + 0.5))
                kernel_x = np.arange(-kernel_radius, kernel_radius + 1, dtype=float)
                kernel = np.exp(-0.5 * np.square(kernel_x / smoothing_sigma_bins))
                kernel /= float(np.sum(kernel))
                smooth_sum_w2 = np.convolve(summed_sq, np.square(kernel), mode="same")
                effective_samples = np.divide(
                    np.square(smooth_summed),
                    smooth_sum_w2,
                    out=np.zeros_like(smooth_summed, dtype=float),
                    where=smooth_sum_w2 > 0.0,
                )
                summary["profile_support_method"] = (
                    "q_pixel_resolution_smoothed_annular_bin_counts"
                )
    summary["radial_q_pixel_step"] = (
        float(radial_q_pixel_step)
        if radial_q_pixel_step is not None
        else None
    )
    summary["profile_smoothing_sigma_bins"] = float(smoothing_sigma_bins)
    usable = np.isfinite(profile) & (
        support_counts >= FIRST_ORDER_PEAK_MIN_BIN_COUNT
    )
    summary["n_supported_bins"] = int(np.count_nonzero(usable))
    if summary["n_supported_bins"] < FIRST_ORDER_PEAK_MIN_TOTAL_SUPPORTED_BINS:
        summary["reason"] = "sparse_radial_profile"
        summary["selection_status"] = "no_hint"
        return summary
    supported_runs = _contiguous_profile_runs(usable)
    summary["n_supported_runs"] = len(supported_runs)
    if not supported_runs:
        summary["reason"] = "insufficient_contiguous_support"
        summary["selection_status"] = "no_hint"
        return summary
    if find_peaks is None or peak_widths is None:
        summary["reason"] = "peak_detection_unavailable"
        summary["selection_status"] = "no_hint"
        return summary
    noise_mad = _radial_profile_noise_mad(raw_profile, raw_supported_runs)
    if noise_mad is None:
        summary["reason"] = "noise_estimator_unavailable"
        summary["selection_status"] = "no_hint"
        return summary
    peak_height = float(np.max(profile[usable]))
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        summary["reason"] = "empty_radial_profile"
        summary["selection_status"] = "no_hint"
        return summary
    noise_floor = max(noise_mad, np.finfo(float).eps * max(1.0, abs(peak_height)))
    significance_threshold = FIRST_ORDER_PEAK_PROMINENCE_NOISE_MULTIPLE * noise_floor
    ambiguity_floor = FIRST_ORDER_PEAK_AMBIGUITY_NOISE_MULTIPLE * noise_floor
    summary.update(
        {
            "profile_noise_mad": float(noise_mad),
            "effective_noise_floor": float(noise_floor),
            "significance_prominence_threshold": float(significance_threshold),
            "ambiguity_prominence_floor": float(ambiguity_floor),
        }
    )

    candidate_peaks: list[dict[str, Any]] = []
    for start, stop in supported_runs:
        segment = profile[start:stop]
        local_peaks, properties = find_peaks(segment, prominence=(None, None))
        widths = peak_widths(segment, local_peaks, rel_height=0.5)[0]
        for local_peak, prominence, width in zip(
            local_peaks, properties["prominences"], widths
        ):
            index = int(start + local_peak)
            height = float(profile[index])
            prominence = float(prominence)
            width = float(width)
            candidate_peaks.append(
                {
                    "index": index,
                    "bin_index": index,
                    "q_bin": float(centres[index]),
                    "height": height,
                    "prominence": prominence,
                    "prominence_noise_ratio": float(prominence / noise_floor),
                    "width_bins": width,
                    "effective_samples": float(effective_samples[index]),
                    "support_status": (
                        "eligible"
                        if effective_samples[index] >= FIRST_ORDER_PEAK_MIN_EFFECTIVE_SAMPLES
                        else "outlier_dominated"
                    ),
                }
            )
    summary["candidate_peaks"] = [
        {key: value for key, value in candidate.items() if key != "index"}
        for candidate in candidate_peaks
    ]
    if not candidate_peaks:
        summary["reason"] = "no_supported_local_peak"
        summary["selection_status"] = "no_hint"
        return summary
    eligible_peaks = [
        peak for peak in candidate_peaks if peak["support_status"] == "eligible"
    ]
    summary["n_outlier_dominated_peaks"] = int(len(candidate_peaks) - len(eligible_peaks))
    if not eligible_peaks:
        summary["reason"] = "outlier_dominated_peak_only"
        summary["selection_status"] = "no_hint"
        return summary

    significant_peak_count = sum(
        peak["prominence"] >= significance_threshold for peak in eligible_peaks
    )
    summary["n_significant_peaks"] = int(significant_peak_count)
    families: list[list[dict[str, Any]]] = []
    for candidate in eligible_peaks:
        if (
            not families
            or candidate["q_bin"]
            > FIRST_ORDER_PEAK_FAMILY_SPAN * families[-1][0]["q_bin"]
        ):
            families.append([candidate])
        else:
            families[-1].append(candidate)

    selected_family: list[dict[str, Any]] | None = None
    for family in families:
        significant = [
            peak for peak in family if peak["prominence"] >= significance_threshold
        ]
        if significant:
            selected_family = significant
            break
        ambiguous = [
            peak
            for peak in family
            if peak["prominence"] >= ambiguity_floor
            and peak["width_bins"] >= FIRST_ORDER_PEAK_AMBIGUITY_MIN_WIDTH_BINS
        ]
        if ambiguous:
            summary["reason"] = "ambiguous_lower_q_peak_family"
            summary["selection_status"] = "ambiguous"
            summary["ambiguous_peaks"] = [
                {key: value for key, value in peak.items() if key != "index"}
                for peak in ambiguous
            ]
            return summary
    if selected_family is None:
        summary["reason"] = "no_significant_supported_peak"
        summary["selection_status"] = "no_hint"
        return summary

    selected_indices = np.asarray([peak["index"] for peak in selected_family], dtype=int)
    peak_index = _select_first_order_peak_index(
        centres[selected_indices], profile[selected_indices], selected_indices,
        family_span=FIRST_ORDER_PEAK_FAMILY_SPAN,
    )
    q_bin = float(centres[peak_index])
    q_star = _refine_radial_peak_q(
        radii, weights, q_bin, q_min, q_max, coarse_bins=int(n_bins)
    )
    if not np.isfinite(q_star) or q_star <= 0.0:
        summary["reason"] = "invalid_peak"
        summary["selection_status"] = "no_hint"
        return summary
    low = max(float(q_min), 0.62 * q_star)
    high = min(float(q_max), 1.45 * q_star)
    if high <= low:
        summary.update(
            {
                "q_star": float(q_star),
                "q_star_bin": q_bin,
                "reason": "degenerate_band",
                "selection_status": "selected",
            }
        )
        return summary
    summary.update(
        {
            "q_star": float(q_star),
            "q_star_bin": q_bin,
            "band": [low, high],
            "reason": "ok",
            "selection_status": "selected",
            "selected_peak": next(
                {
                    key: value
                    for key, value in peak.items()
                    if key != "index"
                }
                for peak in selected_family
                if peak["index"] == peak_index
            ),
        }
    )
    return summary


def _circular_angular_span_deg(points: Sequence[Mapping[str, Any]]) -> float:
    """Return the smallest circular span containing the observed arc points."""

    angles = np.asarray(
        [
            math.degrees(math.atan2(float(point["qy"]), float(point["qx"]))) % 360.0
            for point in points
            if np.isfinite(float(point.get("qx", float("nan"))))
            and np.isfinite(float(point.get("qy", float("nan"))))
        ],
        dtype=float,
    )
    if angles.size < 2:
        return 0.0
    angles.sort()
    gaps = np.diff(np.concatenate((angles, angles[:1] + 360.0)))
    return float(max(0.0, 360.0 - float(np.max(gaps))))


def _apply_hint_independent_arc_support(
    first_order: Mapping[str, Any],
    arcs: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    *,
    q_min: float,
    q_max: float,
    radii: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reconcile radial peaks with curvature arcs found before hint weighting.

    Radial candidates remain inspectable. A lower-q family is bypassed only
    when a higher-q family has stronger support in every observed-trajectory
    summary and stronger radial prominence. Conflicting evidence leaves q*
    unset instead of steering NMS with an unverified hint.
    """

    summary = dict(first_order)
    peaks = [dict(peak) for peak in first_order.get("candidate_peaks", [])]
    summary["candidate_peaks"] = peaks
    summary["selection_method"] = "radial_peak_with_hint_independent_observed_arc_support"
    summary["order_assignment"] = "provisional"
    summary["target_definition"] = (
        "radial peak family supported by observed two-dimensional curvature arcs; "
        "not a definitive Bragg-order assignment"
    )
    summary["arc_support_method"] = "unweighted_curvature_candidates_nms_and_valid_arc_topology"
    summary["candidate_families"] = []
    summary["arc_support_comparisons"] = []
    summary["competing_peaks"] = []
    if not peaks:
        summary.update(
            {"selection_status": "no_hint", "q_star": None, "band": None,
             "reason": first_order.get("reason") or "no_radial_peak_candidates"}
        )
        return summary

    eligible = [
        peak for peak in peaks
        if peak.get("support_status") == "eligible"
        and np.isfinite(float(peak.get("q_bin", float("nan"))))
    ]
    eligible.sort(key=lambda peak: float(peak["q_bin"]))
    family_span = float(first_order.get("peak_family_span", FIRST_ORDER_PEAK_FAMILY_SPAN))
    groups: list[list[dict[str, Any]]] = []
    for peak in eligible:
        if not groups or float(peak["q_bin"]) > family_span * float(groups[-1][0]["q_bin"]):
            groups.append([peak])
        else:
            groups[-1].append(peak)

    points_by_id = {
        str(point.get("point_id")): point
        for point in points
        if point.get("point_id") is not None
    }
    bin_width = (float(q_max) - float(q_min)) / max(int(first_order.get("n_bins", 96)), 1)
    pixel_step = first_order.get("radial_q_pixel_step")
    pixel_step = float(pixel_step) if pixel_step is not None else 0.0
    peak_bands: dict[int, tuple[float, float]] = {}
    for peak in eligible:
        peak_width = max(float(peak.get("width_bins", 0.0)), 0.0)
        half_width = max(0.5 * peak_width * bin_width, 0.5 * pixel_step, bin_width)
        peak_q = float(peak["q_bin"])
        peak_bands[id(peak)] = (
            max(float(q_min), peak_q - half_width),
            min(float(q_max), peak_q + half_width),
        )

    significant_threshold = first_order.get("significance_prominence_threshold")
    ambiguity_floor = first_order.get("ambiguity_prominence_floor")
    significant_threshold = float(significant_threshold) if significant_threshold is not None else float("inf")
    ambiguity_floor = float(ambiguity_floor) if ambiguity_floor is not None else float("inf")
    ambiguity_width = float(first_order.get(
        "ambiguity_min_width_bins", FIRST_ORDER_PEAK_AMBIGUITY_MIN_WIDTH_BINS
    ))
    for family_id, family_peaks in enumerate(groups):
        for peak in family_peaks:
            peak["family_id"] = int(family_id)
        family_bands = [peak_bands[id(peak)] for peak in family_peaks]
        support_arcs: list[dict[str, Any]] = []
        supported_point_ids: set[str] = set()
        for arc in arcs:
            if not bool(arc.get("valid", False)):
                continue
            ordered_ids = arc.get("ordered_point_ids", arc.get("point_ids", ()))
            arc_points = [points_by_id[str(point_id)] for point_id in ordered_ids if str(point_id) in points_by_id]
            if len(arc_points) < 2:
                continue
            radii_on_arc = np.asarray(
                [math.hypot(float(point["qx"]), float(point["qy"])) for point in arc_points],
                dtype=float,
            )
            matched = np.asarray(
                [any(low <= radius <= high for low, high in family_bands) for radius in radii_on_arc],
                dtype=bool,
            )
            positions = np.flatnonzero(matched)
            if positions.size == 0:
                continue
            runs = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
            matched_points = [arc_points[int(index)] for index in positions]
            supported_point_ids.update(str(point["point_id"]) for point in matched_points)
            matched_spans = [
                _circular_angular_span_deg([arc_points[int(index)] for index in run])
                for run in runs
                if run.size >= 2
            ]
            support_arcs.append(
                {
                    "arc_id": int(arc.get("arc_id", -1)),
                    "n_arc_points": int(len(arc_points)),
                    "n_family_points": int(positions.size),
                    "n_matched_runs": int(len(runs)),
                    "n_contiguous_matched_runs": int(len(matched_spans)),
                    "has_contiguous_match": bool(matched_spans),
                    "matched_angular_span_deg": float(sum(matched_spans)),
                    "full_arc_angular_span_reference_deg": _circular_angular_span_deg(arc_points),
                    "q_median": float(np.median(radii_on_arc)),
                    "branch_ids": list(arc.get("branch_ids", [])),
                    "side": str(arc.get("side", "unknown")),
                }
            )
        significant_peaks = [
            peak for peak in family_peaks
            if float(peak.get("prominence", 0.0)) >= significant_threshold
        ]
        ambiguous_peaks = [
            peak for peak in family_peaks
            if float(peak.get("prominence", 0.0)) >= ambiguity_floor
            and float(peak.get("width_bins", 0.0)) >= ambiguity_width
        ]
        continuous_support_arcs = [arc for arc in support_arcs if arc["has_contiguous_match"]]
        family_summary: dict[str, Any] = {
            "family_id": int(family_id),
            "q_min": float(min(peak["q_bin"] for peak in family_peaks)),
            "q_max": float(max(peak["q_bin"] for peak in family_peaks)),
            "radial_evidence_status": "significant" if significant_peaks else "ambiguous" if ambiguous_peaks else "below_significance",
            "n_radial_peaks": int(len(family_peaks)),
            "max_prominence": float(max(float(peak.get("prominence", 0.0)) for peak in family_peaks)),
            "n_significant_peaks": int(len(significant_peaks)),
            "n_ambiguous_peaks": int(len(ambiguous_peaks)),
            "n_valid_arcs": int(len(continuous_support_arcs)),
            "n_isolated_arc_hits": int(len(support_arcs) - len(continuous_support_arcs)),
            "n_arc_supported_points": int(len(supported_point_ids)),
            "n_matched_runs": int(sum(arc["n_contiguous_matched_runs"] for arc in support_arcs)),
            "matched_angular_span_deg": float(sum(arc["matched_angular_span_deg"] for arc in support_arcs)),
            "mean_matched_arc_points": float(np.mean([arc["n_family_points"] for arc in continuous_support_arcs])) if continuous_support_arcs else None,
            "mean_matched_arc_span_deg": float(np.mean([arc["matched_angular_span_deg"] for arc in continuous_support_arcs])) if continuous_support_arcs else None,
            "full_arc_angular_span_reference_deg": float(sum(arc["full_arc_angular_span_reference_deg"] for arc in support_arcs)),
            "arc_support_status": "observed_contiguous_arc_segments" if continuous_support_arcs else "isolated_radial_hits_only" if support_arcs else "no_valid_arc_support",
            "supporting_arcs": support_arcs,
            "weaker_observed_support_than_family_ids": [],
        }
        summary["candidate_families"].append(family_summary)

    families = summary["candidate_families"]
    for lower_index, lower in enumerate(families):
        for higher in families[lower_index + 1:]:
            lower_arc_n = int(lower["n_valid_arcs"])
            higher_arc_n = int(higher["n_valid_arcs"])
            lower_span = float(lower["mean_matched_arc_span_deg"] or 0.0)
            higher_span = float(higher["mean_matched_arc_span_deg"] or 0.0)
            lower_points = float(lower["mean_matched_arc_points"] or 0.0)
            higher_points = float(higher["mean_matched_arc_points"] or 0.0)
            higher_dominates = bool(
                lower_arc_n > 0
                and higher_arc_n > 0
                and higher_span > lower_span
                and higher_points >= lower_points
            )
            lower_dominates = bool(
                lower_arc_n > 0
                and higher_arc_n > 0
                and lower_span > higher_span
                and lower_points >= higher_points
            )
            similar_support = bool(
                lower_arc_n == higher_arc_n
                and np.isclose(lower_span, higher_span, rtol=1e-6, atol=1e-6)
                and np.isclose(lower_points, higher_points, rtol=1e-6, atol=1e-6)
            )
            lower_prominence = float(lower["max_prominence"])
            higher_prominence = float(higher["max_prominence"])
            prominence_relation = (
                "higher_q_stronger" if higher_prominence > lower_prominence
                else "lower_q_stronger" if lower_prominence > higher_prominence
                else "similar"
            )
            weaker = bool(higher_dominates and higher_prominence > lower_prominence)
            if weaker:
                lower["weaker_observed_support_than_family_ids"].append(int(higher["family_id"]))
            relation = (
                "higher_q_only_observed_arc_support" if lower_arc_n == 0 and higher_arc_n > 0
                else "lower_q_only_observed_arc_support" if lower_arc_n > 0 and higher_arc_n == 0
                else "higher_q_dominates_observed_support" if higher_dominates
                else "lower_q_dominates_observed_support" if lower_dominates
                else "similar_observed_support" if similar_support
                else "incomparable_observed_support"
            )
            summary["arc_support_comparisons"].append(
                {
                    "lower_family_id": int(lower["family_id"]),
                    "higher_family_id": int(higher["family_id"]),
                    "observed_support_relation": relation,
                    "lower_mean_matched_arc_span_deg": lower["mean_matched_arc_span_deg"],
                    "higher_mean_matched_arc_span_deg": higher["mean_matched_arc_span_deg"],
                    "lower_mean_matched_arc_points": lower["mean_matched_arc_points"],
                    "higher_mean_matched_arc_points": higher["mean_matched_arc_points"],
                    "radial_prominence_relation": prominence_relation,
                    "weaker_observed_support": weaker,
                }
            )

    significant_families = [family for family in families if family["radial_evidence_status"] == "significant"]
    supported_significant = [
        family for family in significant_families
        if int(family["n_valid_arcs"]) > 0
        and not family["weaker_observed_support_than_family_ids"]
    ]
    selected_family = supported_significant[0] if supported_significant else None
    unresolved_lower = [
        family for family in families
        if family["radial_evidence_status"] in {"significant", "ambiguous"}
        and not family["weaker_observed_support_than_family_ids"]
        and int(family["n_valid_arcs"]) == 0
        and (selected_family is None or int(family["family_id"]) < int(selected_family["family_id"]))
    ]
    ambiguous_lower = [
        family for family in families
        if family["radial_evidence_status"] == "ambiguous"
        and not family["weaker_observed_support_than_family_ids"]
        and (selected_family is None or int(family["family_id"]) < int(selected_family["family_id"]))
    ]
    summary["q_star"] = None
    summary["q_star_bin"] = None
    summary["band"] = None
    summary.pop("selected_peak", None)
    summary.pop("ambiguous_peaks", None)
    if unresolved_lower or ambiguous_lower:
        blocking = unresolved_lower or ambiguous_lower
        summary["selection_status"] = "ambiguous"
        summary["reason"] = (
            "lower_q_peak_without_independent_arc_support"
            if unresolved_lower
            else "ambiguous_lower_q_peak_family_with_observed_arc_support"
        )
        blocking_ids = {int(family["family_id"]) for family in blocking}
        summary["ambiguous_peaks"] = [peak for peak in peaks if int(peak.get("family_id", -1)) in blocking_ids]
    elif selected_family is None:
        summary["selection_status"] = "ambiguous" if significant_families else "no_hint"
        summary["reason"] = (
            "no_hint_independent_observed_arc_support"
            if significant_families
            else "no_significant_supported_peak"
        )
    else:
        family_id = int(selected_family["family_id"])
        selected_peaks = [
            peak for peak in eligible
            if int(peak.get("family_id", -1)) == family_id
            and float(peak.get("prominence", 0.0)) >= significant_threshold
        ]
        selected_peak = max(selected_peaks, key=lambda peak: float(peak.get("height", 0.0)))
        q_bin = float(selected_peak["q_bin"])
        q_star = (
            _refine_radial_peak_q(
                np.asarray(radii, dtype=float), np.asarray(weights, dtype=float), q_bin,
                q_min, q_max, coarse_bins=int(first_order.get("n_bins", 96)),
            )
            if radii is not None and weights is not None else q_bin
        )
        if not np.isfinite(q_star) or q_star <= 0.0:
            summary["selection_status"] = "no_hint"
            summary["reason"] = "invalid_observed_arc_supported_peak"
        else:
            summary.update(
                {
                    "q_star": float(q_star),
                    "q_star_bin": q_bin,
                    "band": [max(float(q_min), 0.62 * float(q_star)), min(float(q_max), 1.45 * float(q_star))],
                    "selection_status": "selected",
                    "reason": "ok_observed_arc_supported",
                    "selected_peak": dict(selected_peak),
                    "selected_family_id": family_id,
                }
            )
    selected_id = summary.get("selected_family_id")
    summary["competing_peaks"] = [
        dict(peak) for peak in peaks
        if selected_id is None or int(peak.get("family_id", -1)) != int(selected_id)
    ]
    if selected_id is None:
        summary.pop("selected_family_id", None)
    return summary


def _select_first_order_peak_index(
    peak_q: np.ndarray,
    peak_intensity: np.ndarray,
    peak_indices: np.ndarray,
    *,
    family_span: float = 1.45,
) -> int:
    """Pick the strongest bin in the lowest-q peak family."""

    if peak_indices.size == 1:
        return int(peak_indices[0])
    families: list[list[int]] = [[0]]
    for offset in range(1, int(peak_q.size)):
        root = float(peak_q[families[-1][0]])
        value = float(peak_q[offset])
        if np.isfinite(root) and np.isfinite(value) and value <= family_span * root:
            families[-1].append(offset)
        else:
            families.append([offset])
    family = families[0]
    best = max(family, key=lambda item: float(peak_intensity[item]))
    return int(peak_indices[best])


def _refine_radial_peak_q(
    radii: np.ndarray,
    weights: np.ndarray,
    q_bin: float,
    q_min: float,
    q_max: float,
    *,
    n_bins: int = 25,
    coarse_bins: int = 96,
) -> float:
    """Parabolic sub-bin refinement of I(q)* around a coarse histogram peak."""

    try:
        centre = float(q_bin)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(centre) and centre > 0.0):
        return float("nan")
    coarse_step = (float(q_max) - float(q_min)) / max(int(coarse_bins), 1)
    if not np.isfinite(coarse_step) or coarse_step <= 0.0:
        coarse_step = 0.08 * centre
    # Stay inside the coarse peak's own bin.  A 0.75–1.25 window plus a
    # grid that misses q_bin makes the 3-point interpolate hit its clamp
    # and snap every frame to one of two discrete q* values.
    span = max(float(coarse_step), 0.06 * centre)
    count = max(7, int(n_bins) | 1)
    mid = count // 2
    step = 2.0 * span / count
    low = centre - (mid + 0.5) * step
    high = centre + (mid + 0.5) * step
    low = max(float(q_min), low)
    high = min(float(q_max), high)
    if high <= low:
        return centre
    edges = np.linspace(low, high, count + 1)
    summed, _ = np.histogram(radii, bins=edges, weights=weights)
    counts, _ = np.histogram(radii, bins=edges)
    profile = np.divide(
        summed,
        counts,
        out=np.zeros_like(summed, dtype=float),
        where=counts > 0,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    usable = np.isfinite(profile) & (counts >= 4)
    if int(np.count_nonzero(usable)) < 5:
        return centre
    intensity_q = np.where(usable, profile, 0.0)
    index = int(np.argmin(np.abs(centres - centre)))
    if not usable[index]:
        return centre
    if 0 < index < intensity_q.size - 1:
        left_val = float(intensity_q[index - 1]) if usable[index - 1] else -np.inf
        right_val = float(intensity_q[index + 1]) if usable[index + 1] else -np.inf
        if intensity_q[index] < left_val or intensity_q[index] < right_val:
            # One step toward the higher usable neighbour keeps the interpolate
            # on this ring without a second argmax over the whole window.
            index = index - 1 if left_val > right_val else index + 1
            if not usable[index]:
                return centre
    if index <= 0 or index >= intensity_q.size - 1 or not (usable[index - 1] and usable[index] and usable[index + 1]):
        return float(centres[index])
    y0, y1, y2 = float(intensity_q[index - 1]), float(intensity_q[index]), float(intensity_q[index + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12 or y1 + 1e-12 < max(y0, y2):
        return float(centres[index])
    delta = 0.5 * (y0 - y2) / denom
    delta = max(-0.5, min(0.5, float(delta)))
    bin_step = float(centres[index] - centres[index - 1])
    refined = float(centres[index] + delta * bin_step)
    if not (np.isfinite(refined) and low <= refined <= high):
        return float(centres[index])
    return refined


def _prefer_first_order_scores(
    candidates: Sequence[dict[str, Any]],
    q_star: float | None,
    *,
    sigma_log: float = 0.45,
) -> int:
    """Down-weight harmonic-scale candidates without clipping ellipse tips.

    A hard |q| crop around I(q)* removes the major-axis ends of a flat
    Wang ellipse.  A log-q score weight keeps those tips while making a
    3–4× harmonic lose NMS / max_points competition.
    """

    try:
        hint = float(q_star) if q_star is not None else float("nan")
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(hint) or hint <= 0.0 or not np.isfinite(sigma_log) or sigma_log <= 0.0:
        return 0
    updated = 0
    for point in candidates:
        try:
            radius = float(np.hypot(float(point["qx"]), float(point["qy"])))
            score = float(point.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(radius) and radius > 0.0 and np.isfinite(score)):
            continue
        weight = math.exp(-0.5 * (math.log(radius / hint) / float(sigma_log)) ** 2)
        point["score"] = score * weight
        point["first_order_weight"] = float(weight)
        updated += 1
    return updated


def _axis_deviation_deg(
    qx: float,
    qy: float,
    *,
    center_x: float,
    center_y: float,
    reference_axis_deg: float,
) -> float:
    """Absolute angle from the declared draw-axis line, in degrees."""

    theta = math.radians(float(reference_axis_deg))
    dx = float(qx) - float(center_x)
    dy = float(qy) - float(center_y)
    along = dx * math.cos(theta) + dy * math.sin(theta)
    perpendicular = -dx * math.sin(theta) + dy * math.cos(theta)
    angle = math.degrees(math.atan2(perpendicular, along))
    return min(abs(angle), abs(abs(angle) - 180.0))


def _angular_delta_deg(first: float, second: float) -> float:
    return abs(((float(first) - float(second) + 180.0) % 360.0) - 180.0)


def _sparse_first_order_coverage(
    points: Sequence[Mapping[str, Any]],
    *,
    center_x: float,
    center_y: float,
    reference_axis_deg: float,
) -> dict[str, Any]:
    """True when accepted identity points are too axial to define an ellipse."""

    identity = [
        point
        for point in points
        if bool(point.get("accepted", False)) and _point_identity_key(point) is not None
    ]
    if not identity:
        return {"sparse": True, "n": 0, "max_dev_deg": None}
    deviations = [
        _axis_deviation_deg(
            float(point["qx"]),
            float(point["qy"]),
            center_x=center_x,
            center_y=center_y,
            reference_axis_deg=reference_axis_deg,
        )
        for point in identity
    ]
    max_dev = max(deviations)
    sparse = max_dev < 18.0 or (len(identity) < 8 and max_dev < 25.0)
    return {"sparse": sparse, "n": len(identity), "max_dev_deg": float(max_dev)}


def _prepare_sparse_first_order_samples(
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    *,
    hint: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Pack finite first-order-band pixels for reuse across azimuth sectors."""

    q_lo, q_hi = 0.70 * hint, 1.45 * hint
    qx_array = np.asarray(qx)
    qy_array = np.asarray(qy)
    rel_x = qx_array - float(center_x)
    rel_y = qy_array - float(center_y)
    q_array = (
        np.hypot(rel_x, rel_y)
        if (center_x != 0.0 or center_y != 0.0)
        else np.asarray(q)
    )
    intensity_array = np.asarray(intensity)
    band = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(q_array)
        & np.isfinite(intensity_array)
        & (q_array >= q_lo)
        & (q_array <= q_hi)
    )
    rows, cols = np.nonzero(band)
    return {
        "qx": qx_array[band],
        "qy": qy_array[band],
        "q": q_array[band],
        "intensity": intensity_array[band],
        "angle_deg": np.degrees(np.arctan2(rel_y[band], rel_x[band])),
        "rows": rows,
        "cols": cols,
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
    }


def _sector_first_order_peak(
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    *,
    sector_deg: float,
    halfwidth_deg: float,
    hint: float,
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return one observed radial peak in an azimuth sector, or None."""

    if not (np.isfinite(hint) and hint > 0.0):
        return None
    q_lo, q_hi = 0.70 * hint, 1.45 * hint
    if prepared is None:
        prepared = _prepare_sparse_first_order_samples(qx, qy, q, intensity, valid, hint=hint)
    # ``prepared`` is a compact, C-order subset of the finite q band.  Keeping
    # this subset in the caller lets all sectors reuse the expensive detector
    # angle and validity work without retaining a frame-wide cache.
    sample_qx = prepared["qx"]
    sample_qy = prepared["qy"]
    sample_q = prepared["q"]
    sample_intensity = prepared["intensity"]
    delta = np.abs(
        ((prepared["angle_deg"] - float(sector_deg) + 180.0) % 360.0) - 180.0
    )
    selected = delta <= float(halfwidth_deg)
    if int(np.count_nonzero(selected)) < 12:
        return None
    radii = np.asarray(sample_q[selected], dtype=float)
    values = np.asarray(sample_intensity[selected], dtype=float)
    edges = np.linspace(q_lo, q_hi, 9)
    profile = np.full(edges.size - 1, np.nan, dtype=float)
    counts = np.zeros(edges.size - 1, dtype=int)
    # Clip the right edge into the last bin: np.digitize returns one past the
    # final bin for q == q_hi, which previously discarded those pixels.
    idx = np.clip(np.digitize(radii, edges) - 1, 0, profile.size - 1)
    for bin_i in range(edges.size - 1):
        in_bin = idx == bin_i
        counts[bin_i] = int(np.count_nonzero(in_bin))
        if counts[bin_i] >= 2:
            profile[bin_i] = float(np.nanmedian(values[in_bin]))
    usable = np.isfinite(profile)
    if int(np.count_nonzero(usable)) < 3:
        return None
    peak_i = int(np.nanargmax(np.where(usable, profile, -np.inf)))
    peak = float(profile[peak_i])
    baseline = float(np.nanmedian(profile[usable]))
    if not (np.isfinite(peak) and np.isfinite(baseline) and baseline > 0 and peak >= 1.30 * baseline):
        return None
    q_star = float(0.5 * (edges[peak_i] + edges[peak_i + 1]))
    in_bin = selected & (sample_q >= edges[peak_i])
    if peak_i == profile.size - 1:
        in_bin &= sample_q <= edges[peak_i + 1]
    else:
        in_bin &= sample_q < edges[peak_i + 1]
    if int(np.count_nonzero(in_bin)) < 3:
        in_bin = selected
    median_qx = float(np.nanmedian(sample_qx[in_bin]))
    median_qy = float(np.nanmedian(sample_qy[in_bin]))
    if not (np.isfinite(median_qx) and np.isfinite(median_qy)):
        return None
    rows = prepared["rows"][in_bin]
    cols = prepared["cols"][in_bin]
    nearest = int(np.argmin((sample_qx[in_bin] - median_qx) ** 2 + (sample_qy[in_bin] - median_qy) ** 2))
    return {
        "qx": median_qx,
        "qy": median_qy,
        "q_star": q_star,
        "intensity": peak,
        "contrast": peak / baseline,
        "pixel_x": float(cols[nearest]),
        "pixel_y": float(rows[nearest]),
        "sector_deg": float(sector_deg),
    }


def _next_arc_id(points: Sequence[Mapping[str, Any]], arcs: Sequence[Mapping[str, Any]] | None = None) -> int:
    used = {-1}
    for record in list(points) + list(arcs or ()):
        try:
            used.add(int(record.get("arc_id", -1)))
        except (TypeError, ValueError):
            continue
    return max(used) + 1


def _fill_sparse_first_order_ring(
    points: list[dict[str, Any]],
    *,
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    hint: float | None,
    options: Mapping[str, Any],
    q_step: float,
    signature: str,
    arcs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add observed first-order ring samples when curvature kept only the streak.

    This does not invent quadrants.  A sector is filled only when I(q) still
    has a peak near the first-order hint and no accepted identity point is
    already there.
    """

    summary: dict[str, Any] = {"applied": False, "added": 0, "reason": None, "sectors": []}
    if not bool(options.get("fill_sparse_first_order_ring", True)):
        summary["reason"] = "disabled"
        return summary
    try:
        hint_q = float(hint) if hint is not None else float("nan")
    except (TypeError, ValueError):
        hint_q = float("nan")
    if not (np.isfinite(hint_q) and hint_q > 0.0):
        summary["reason"] = "no_first_order_hint"
        return summary
    center_x = float(options.get("center_qx", 0.0))
    center_y = float(options.get("center_qy", 0.0))
    reference = float(options.get("reference_axis_deg", 0.0))
    coverage = _sparse_first_order_coverage(
        points, center_x=center_x, center_y=center_y, reference_axis_deg=reference
    )
    summary.update(coverage)
    if not coverage["sparse"]:
        summary["reason"] = "identity_coverage_sufficient"
        return summary
    existing_angles = []
    for point in points:
        if not bool(point.get("accepted", False)):
            continue
        existing_angles.append(
            math.degrees(math.atan2(float(point["qy"]) - center_y, float(point["qx"]) - center_x))
        )
    added: list[dict[str, Any]] = []
    ridge_mod = sys.modules.get("butterfly_saxs.butterfly_ridge")
    prepare_fn = (
        getattr(ridge_mod, "_prepare_sparse_first_order_samples", _prepare_sparse_first_order_samples)
        if ridge_mod is not None
        else _prepare_sparse_first_order_samples
    )
    prepared = prepare_fn(
        qx,
        qy,
        q,
        intensity,
        valid,
        hint=hint_q,
        center_x=center_x,
        center_y=center_y,
    )
    for quadrant in (0.0, 90.0, 180.0, 270.0):
        for offset in (30.0, 45.0, 60.0, 75.0):
            sector = reference + quadrant + offset
            if any(_angular_delta_deg(sector, angle) < 12.0 for angle in existing_angles):
                continue
            peak = _sector_first_order_peak(
                qx,
                qy,
                q,
                intensity,
                valid,
                sector_deg=sector,
                halfwidth_deg=7.5,
                hint=hint_q,
                prepared=prepared,
            )
            if peak is None:
                continue
            dx_peak = float(peak["qx"]) - center_x
            dy_peak = float(peak["qy"]) - center_y
            peak_norm = max(math.hypot(dx_peak, dy_peak), 1e-12)
            point = {
                "qx": peak["qx"],
                "qy": peak["qy"],
                "pixel_x": peak["pixel_x"],
                "pixel_y": peak["pixel_y"],
                "intensity": peak["intensity"],
                "snr": peak["contrast"],
                "normal_qx": dx_peak / peak_norm,
                "normal_qy": dy_peak / peak_norm,
                "tangent_qx": -dy_peak / peak_norm,
                "tangent_qy": dx_peak / peak_norm,
                "curvature": float("nan"),
                "scale": 1.0,
                "scale_stability": float("nan"),
                "scale_stable": False,
                "topology_flag": "first_order_ring_sample",
                "topology": "first_order_ring_sample",
                "topology_flags": ["first_order_ring_sample", "side_from_reference_axis"],
                "accepted": True,
                "valid": True,
                "reason": "first_order_ring_sample",
                "arc_id": -1,
                "score": float(peak["contrast"]),
                "q_normal_step": float(q_step),
                "localization_sigma_q": float(q_step),
                "sampling_sigma_q": float(q_step),
                "uncertainty_source": "azimuthal_radial_peak",
            }
            _assign_reference_branches([point], options)
            point["side"] = _side_from_reference_axis(
                float(point["qx"]),
                float(point["qy"]),
                center_x=center_x,
                center_y=center_y,
                reference_axis_deg=reference,
                q_step=q_step,
            )
            if _point_identity_key(point) is None:
                continue
            added.append(point)
            existing_angles.append(
                math.degrees(math.atan2(float(point["qy"]) - center_y, float(point["qx"]) - center_x))
            )
            summary["sectors"].append(
                {
                    "sector_deg": peak["sector_deg"],
                    "q_star": peak["q_star"],
                    "contrast": peak["contrast"],
                }
            )
    if not added:
        summary["reason"] = "no_observed_sector_peak"
        return summary
    _assign_point_ids(added, signature)
    next_arc = _next_arc_id(points, arcs)
    for point in added:
        point["arc_id"] = int(next_arc)
        if arcs is not None:
            arcs.append(
                {
                    "arc_id": int(next_arc),
                    "valid": True,
                    "reason": "first_order_ring_sample",
                    "topology_flags": ["first_order_ring_sample"],
                    "point_ids": [str(point["point_id"])],
                    "identity_resolved": _point_identity_key(point) is not None,
                }
            )
        next_arc += 1
    points.extend(added)
    summary["applied"] = True
    summary["added"] = len(added)
    summary["reason"] = "filled_observed_ring_sectors"
    return summary


def _demote_secondary_radial_population(
    points: Sequence[dict[str, Any]],
    prefer_radius: float | None = None,
) -> dict[str, Any]:
    """Diagnose clearly separated radial populations without losing support.

    A finite positive q* hint may select the nearer population for one-ellipse
    fitting. Without a usable hint, retain both groups and expose the
    ambiguity for review. A continuous wing remains intact regardless of its
    distance from q*.
    """

    accepted = [
        point
        for point in points
        if bool(point.get("accepted", False)) and _point_identity_key(point) is not None
    ]
    radii = []
    for point in accepted:
        try:
            radius = float(np.hypot(float(point["qx"]), float(point["qy"])))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(radius) and radius > 0.0:
            radii.append(radius)
    summary = {
        "split": False,
        "kept": 0,
        "demoted": 0,
        "threshold": None,
        "keep": None,
        "selection_status": "not_split",
        "reason": "no_separated_radial_populations",
        "low_q_population": None,
        "high_q_population": None,
        "flags": [],
    }
    if len(radii) >= 12:
        ordered = np.sort(np.asarray(radii, dtype=float))
        gaps = np.diff(ordered)
        if gaps.size > 0:
            split_at = int(np.argmax(gaps))
            low = ordered[: split_at + 1]
            high = ordered[split_at + 1 :]
            gap = float(gaps[split_at])
            median = float(np.median(ordered))
            can_split = (
                low.size >= 4
                and high.size >= 4
                and median > 0.0
                and gap >= 0.35 * median
                and float(high[0] / max(float(low[-1]), np.finfo(float).eps)) >= 1.8
            )
            if can_split:
                try:
                    hint_radius = float(prefer_radius)
                except (TypeError, ValueError):
                    hint_radius = float("nan")
                hint_valid = bool(np.isfinite(hint_radius) and hint_radius > 0.0)
                threshold = 0.5 * (float(low[-1]) + float(high[0]))
                demoted = 0
                if hint_valid:
                    keep_high = abs(float(np.median(high)) - hint_radius) < abs(
                        float(np.median(low)) - hint_radius
                    )
                    for point in accepted:
                        try:
                            radius = float(np.hypot(float(point["qx"]), float(point["qy"])))
                        except (KeyError, TypeError, ValueError):
                            continue
                        drop = radius < threshold if keep_high else radius >= threshold
                        if drop:
                            _demote_unresolved_identity_point(
                                point, "secondary_radial_population", "mixed_radial_populations"
                            )
                            demoted += 1
                    selection_status = "selected"
                    reason = "nearest_radial_population_to_q_hint"
                    keep = "high_q" if keep_high else "low_q"
                else:
                    selection_status = "ambiguous"
                    reason = "q_hint_unavailable" if prefer_radius is None else "q_hint_invalid"
                    keep = "all_observed"
                summary.update(
                    {
                        "split": True,
                        "kept": int(len(accepted) - demoted),
                        "demoted": int(demoted),
                        "threshold": threshold,
                        "keep": keep,
                        "selection_status": selection_status,
                        "reason": reason,
                        "low_q_population": {
                            "q_min": float(low[0]),
                            "q_max": float(low[-1]),
                            "n_points": int(low.size),
                        },
                        "high_q_population": {
                            "q_min": float(high[0]),
                            "q_max": float(high[-1]),
                            "n_points": int(high.size),
                        },
                        "flags": ["mixed_radial_populations"] if not hint_valid else [],
                    }
                )
    summary["kept"] = int(sum(1 for point in accepted if point.get("accepted")))
    return summary
