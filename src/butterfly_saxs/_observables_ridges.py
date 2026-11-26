"""Radial profiles, curvature/azimuthal ridge extraction, and lobe radial companions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import sys
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks, peak_prominences

from ._observables_angular import (
    _linear_fwhm,
    _periodic_fwhm,
    _quadratic_peak,
    _robust_noise,
    _smooth_supported_angular_profile,
    _supported_angular_noise,
    _wrap_distance,
)
from ._observables_types import (
    APPARENT_FLAGS,
    LobeMetrics,
    RadialProfile,
    RidgePoint,
    RidgeTrack,
    _array_field,
    _bin_mean,
    _extract_maps,
    _flags_with_q_unit,
    _q_limits,
    _q_to_nm_inverse_scale,
    _q_unit,
)
from .cancellation import raise_if_cancelled


def _resolve_radial_profile():
    obs_mod = sys.modules.get("butterfly_saxs.observables")
    if obs_mod is not None:
        return getattr(obs_mod, "measure_radial_profile", measure_radial_profile)
    return measure_radial_profile


def _resolve_radial_ridges():
    obs_mod = sys.modules.get("butterfly_saxs.observables")
    if obs_mod is not None:
        return getattr(obs_mod, "measure_radial_ridges", measure_radial_ridges)
    return measure_radial_ridges


def measure_radial_profile(
    frame: Any,
    qmap: Any,
    angle: float,
    q_window: Any = None,
    *,
    q_range: Any = None,
    n_bins: int = 256,
    sector_width: float = np.deg2rad(4.0),
    statistic: str = "mean",
    mask: Any = None,
) -> RadialProfile:
    """Measure a radial profile in one observed angular sector."""

    if int(n_bins) < 8:
        raise ValueError("n_bins must be at least 8")
    values, q, phi, valid = _extract_maps(frame, qmap, mask)
    q_unit = _q_unit(qmap)
    q_min, q_max = _q_limits(q, q_window, q_range)
    candidate = np.isfinite(values) & np.isfinite(q) & (q >= q_min) & (q <= q_max) & (_wrap_distance(phi, angle) <= float(sector_width) / 2.0)
    selected = candidate & valid
    n = int(n_bins)
    edges = np.linspace(q_min, q_max, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Closed window [q_min, q_max]: digitize would place q == q_max in bin n.
    all_idx = np.clip(np.digitize(q[candidate], edges, right=False) - 1, 0, n - 1)
    valid_idx = np.clip(np.digitize(q[selected], edges, right=False) - 1, 0, n - 1)
    candidate_counts = np.bincount(all_idx, minlength=n).astype(int)
    profile, counts = _bin_mean(values[selected], valid_idx, n, statistic)
    coverage = np.divide(counts, candidate_counts, out=np.zeros_like(profile), where=candidate_counts > 0)
    return RadialProfile(
        angle=float(angle),
        q=centres,
        intensity=profile,
        counts=counts,
        candidate_counts=candidate_counts,
        coverage=coverage,
        q_min=q_min,
        q_max=q_max,
        flags=_flags_with_q_unit(APPARENT_FLAGS, q_unit),
        q_unit=q_unit,
    )


radial_profile = measure_radial_profile


def _radial_profile_baseline_noise(
    q: np.ndarray,
    intensity: np.ndarray,
    finite: np.ndarray,
) -> tuple[float, float]:
    """Estimate baseline and high-frequency profile noise consistently.

    The MAD of all profile values is inflated by a broad physical radial
    peak, which made the same peak's SNR depend on how much of the feature was
    included in the q window.  Adjacent differences on genuinely observed
    neighbouring bins estimate the local high-frequency noise instead; a
    smoothed detrend and finally the profile standard deviation are defensive
    fallbacks for sparse/constant sectors.  Both the continuity candidate and
    single-sector fallback use this exact estimator.
    """

    del q
    values = np.asarray(intensity, dtype=float)
    finite = np.asarray(finite, dtype=bool)
    observed = values[finite]
    if observed.size == 0:
        return float("nan"), float("nan")
    baseline = float(np.nanpercentile(observed, 10.0))
    noise = float("nan")
    adjacent = finite[:-1] & finite[1:]
    if np.any(adjacent):
        differences = values[1:][adjacent] - values[:-1][adjacent]
        noise = _robust_noise(differences) / np.sqrt(2.0)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        if gaussian_filter1d is not None and observed.size > 8:
            filled = np.interp(np.arange(values.size), np.flatnonzero(finite), observed)
            trend = gaussian_filter1d(filled, 2.0, mode="nearest")
            noise = _robust_noise((values - trend)[finite])
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = float(np.nanstd(observed - baseline))
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.finfo(float).eps * max(1.0, np.nanmax(np.abs(observed)))), np.finfo(float).eps)
    return baseline, noise


def _supported_radial_local_maxima(
    values: np.ndarray,
    finite: np.ndarray,
    *,
    dense_profile: bool,
    blocked_bins: np.ndarray | None = None,
) -> np.ndarray:
    """Return measured local maxima with observed support on both sides.

    A candidate must be bracketed by measured samples.  Empty bins known to
    contain geometric support but no measured pixels are treated as masked
    gaps; bins with no geometric pixels may bracket a sparse observed profile.
    Without geometric-support information, dense profiles require adjacent
    measured bins as a conservative fallback.
    """

    signal = np.asarray(values, dtype=float)
    measured = np.asarray(finite, dtype=bool) & np.isfinite(signal)
    indices = np.flatnonzero(measured)
    if indices.size < 3:
        return np.empty(0, dtype=int)
    blocked = None if blocked_bins is None else np.asarray(blocked_bins, dtype=bool)

    maxima: list[int] = []
    position = 0
    while position < indices.size:
        start = position
        end = position
        while end + 1 < indices.size:
            current_index = int(indices[end])
            next_index = int(indices[end + 1])
            if next_index != current_index + 1:
                break
            scale = max(1.0, abs(float(signal[current_index])), abs(float(signal[next_index])))
            tolerance = 8.0 * np.finfo(float).eps * scale
            if abs(float(signal[next_index] - signal[current_index])) > tolerance:
                break
            end += 1
        if start > 0 and end + 1 < indices.size:
            left_index = int(indices[start - 1])
            right_index = int(indices[end + 1])
            first_index = int(indices[start])
            last_index = int(indices[end])
            if blocked is not None:
                gap_has_unmeasured_support = bool(
                    np.any(blocked[left_index + 1 : first_index])
                    or np.any(blocked[last_index + 1 : right_index])
                )
                bracketed = not gap_has_unmeasured_support
            else:
                bracketed = not dense_profile or (
                    first_index - left_index == 1 and right_index - last_index == 1
                )
            plateau_value = float(np.max(signal[indices[start : end + 1]]))
            if (
                bracketed
                and plateau_value > signal[left_index]
                and plateau_value > signal[right_index]
            ):
                maxima.append(int(indices[(start + end) // 2]))
        position = end + 1
    return np.asarray(maxima, dtype=int)


def _radial_peak(
    profile: RadialProfile,
    snr_threshold: float = 2.0,
    min_coverage: float = 0.0,
) -> RidgePoint:
    q = np.asarray(profile.q, dtype=float)
    y = np.asarray(profile.intensity, dtype=float)
    finite = np.isfinite(y) & (np.asarray(profile.counts) > 0)
    if np.count_nonzero(finite) < 3:
        reason = "no_radial_candidate" if not np.any(finite) else "low_coverage"
        flags = list(_flags_with_q_unit(APPARENT_FLAGS, profile.q_unit))
        flags.append(reason)
        return RidgePoint(
            angle=profile.angle,
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            intensity=float("nan"),
            baseline=float("nan"),
            snr=float("nan"),
            radial_fwhm=float("nan"),
            azimuthal_fwhm=float("nan"),
            area=float("nan"),
            coverage=float(np.nanmean(profile.coverage)) if profile.coverage.size else 0.0,
            n_pixels=int(np.sum(profile.counts)),
            valid=False,
            flags=tuple(flags),
            score=0.0,
            reason=reason,
            q_unit=profile.q_unit,
        )
    global_coverage = float(np.sum(profile.counts) / max(1, np.sum(profile.candidate_counts)))
    baseline, noise = _radial_profile_baseline_noise(q, y, finite)
    smooth = y.copy()
    if gaussian_filter1d is not None and np.count_nonzero(finite) > 8:
        fill = np.interp(q, q[finite], y[finite])
        smooth = gaussian_filter1d(fill, 1.0, mode="nearest")
    # Interpolation is useful for smoothing across small gaps, but local
    # maxima are accepted only when measured bins bracket them on both sides.
    # Nearest observed bins may bracket sparse sampling, but never across
    # bins with geometric pixels that were masked or left unmeasured.
    dense_profile = np.count_nonzero(finite) >= max(12, int(np.ceil(0.5 * len(finite))))
    geometry_counts = np.asarray(profile.candidate_counts)
    blocked_bins = (geometry_counts > 0) & ~finite
    local_maxima = _supported_radial_local_maxima(
        smooth,
        finite,
        dense_profile=dense_profile,
        blocked_bins=blocked_bins,
    )
    resolved_local_maximum = bool(local_maxima.size)
    if resolved_local_maximum:
        index = int(local_maxima[np.argmax(smooth[local_maxima])])
    else:
        # Keep the strongest observed sample as an auditable candidate even
        # when it lies at a q-window edge or lacks two-sided support.
        observed_indices = np.flatnonzero(finite)
        index = int(observed_indices[np.argmax(smooth[observed_indices])])
    peak_q, peak_intensity = _quadratic_peak(q, smooth, index)
    snr = (peak_intensity - baseline) / noise if noise else float("inf")
    observed_indices = np.flatnonzero(finite)
    observed_position = int(np.searchsorted(observed_indices, index))
    local_indices = [index]
    candidate_gap = False
    if observed_position > 0:
        left_index = int(observed_indices[observed_position - 1])
        local_indices.append(left_index)
        candidate_gap |= bool(np.any(blocked_bins[left_index + 1 : index]))
    if observed_position + 1 < observed_indices.size:
        right_index = int(observed_indices[observed_position + 1])
        local_indices.append(right_index)
        candidate_gap |= bool(np.any(blocked_bins[index + 1 : right_index]))
    has_two_sided_support = bool(
        observed_position > 0
        and observed_position + 1 < observed_indices.size
        and not candidate_gap
    )
    local_coverage = float(
        np.nanmin(np.asarray(profile.coverage, dtype=float)[local_indices])
    )
    fwhm = _linear_fwhm(q, smooth, index, baseline, support_mask=finite)
    area = float(np.trapezoid(np.maximum(y[finite] - baseline, 0.0), q[finite])) if np.count_nonzero(finite) > 1 else 0.0
    flags = list(_flags_with_q_unit(APPARENT_FLAGS, profile.q_unit))
    if not np.isfinite(fwhm):
        flags.append("radial_fwhm_unavailable")
    unresolved_reason = None
    candidate_edge = None
    if not resolved_local_maximum:
        if index == int(observed_indices[0]) or index == int(observed_indices[-1]):
            unresolved_reason = "unresolved_radial_boundary"
            candidate_edge = "low" if index == int(observed_indices[0]) else "high"
        elif candidate_gap:
            unresolved_reason = "unsupported_radial_gap"
        else:
            unresolved_reason = "no_local_radial_maximum"
    if local_coverage < float(min_coverage):
        flags.append("low_peak_support")
        if unresolved_reason is not None:
            flags.append(unresolved_reason)
        point = RidgePoint(
            angle=profile.angle,
            q=float(peak_q),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            intensity=float(peak_intensity),
            baseline=baseline,
            snr=float(snr),
            radial_fwhm=float(fwhm),
            azimuthal_fwhm=float("nan"),
            area=area,
            coverage=local_coverage,
            n_pixels=int(np.sum(profile.counts[local_indices])),
            valid=False,
            flags=tuple(flags),
            score=0.0,
            reason="low_peak_support",
            q_unit=profile.q_unit,
        )
        point.metadata.update(
            profile_global_coverage=global_coverage,
            peak_coverage=local_coverage,
            peak_bin_index=int(index),
            candidate_q=float(peak_q),
            candidate_edge=candidate_edge,
            resolved_local_maximum=resolved_local_maximum,
            has_two_sided_support=has_two_sided_support,
            support_density="dense" if dense_profile else "sparse",
            radial_fwhm_available=bool(np.isfinite(fwhm)),
        )
        return point
    if unresolved_reason is not None:
        flags.append(unresolved_reason)
        point = RidgePoint(
            angle=profile.angle,
            q=float(peak_q),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            intensity=float(peak_intensity),
            baseline=baseline,
            snr=float(snr),
            radial_fwhm=float(fwhm),
            azimuthal_fwhm=float("nan"),
            area=area,
            coverage=local_coverage,
            n_pixels=int(np.sum(profile.counts[local_indices])),
            valid=False,
            flags=tuple(flags),
            score=0.0,
            reason=unresolved_reason,
            q_unit=profile.q_unit,
        )
        point.metadata.update(
            profile_global_coverage=global_coverage,
            peak_coverage=local_coverage,
            peak_bin_index=int(index),
            candidate_q=float(peak_q),
            candidate_edge=candidate_edge,
            resolved_local_maximum=False,
            has_two_sided_support=has_two_sided_support,
            support_density="dense" if dense_profile else "sparse",
            radial_fwhm_available=bool(np.isfinite(fwhm)),
        )
        return point

    valid = bool(np.isfinite(snr) and snr >= snr_threshold and np.isfinite(peak_q))
    if not valid:
        flags.append("low_snr")
    reason = "accepted" if valid else "low_snr"
    # A low-SNR maximum is a diagnostic candidate, not an accepted radial
    # reflection.  Keep its intensity/SNR for audit but suppress q_star and
    # spacing aliases so downstream exports cannot mistake it for measured
    # structure.
    q_star = float(peak_q) if valid else float("nan")
    score = float(max(0.0, snr) * max(0.0, min(1.0, local_coverage))) if np.isfinite(snr) and valid else 0.0
    point = RidgePoint(
        angle=profile.angle,
        q=q_star,
        q_star=q_star,
        lamellar_spacing=None,
        intensity=float(peak_intensity),
        baseline=baseline,
        snr=float(snr),
        radial_fwhm=float(fwhm),
        azimuthal_fwhm=float("nan"),
        area=area,
        coverage=local_coverage,
        n_pixels=int(np.sum(profile.counts)),
        valid=valid,
        flags=tuple(flags),
        score=score,
        reason=reason,
        q_unit=profile.q_unit,
    )
    point.metadata.update(
        profile_global_coverage=global_coverage,
        peak_coverage=local_coverage,
        peak_bin_index=int(index),
        candidate_q=float(peak_q),
        resolved_local_maximum=True,
        has_two_sided_support=True,
        support_density="dense" if dense_profile else "sparse",
        radial_fwhm_available=bool(np.isfinite(fwhm)),
    )
    return point


def _radial_noise_floor(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    q_range: Any = None,
    mask: Any = None,
) -> float:
    """Estimate one detector-domain noise scale for continuity candidates."""

    values, q, _angle, valid = _extract_maps(frame, qmap, mask)
    q_min, q_max = _q_limits(q, q_window, q_range)
    selected = valid & np.isfinite(values) & (q >= q_min) & (q <= q_max)
    observed = values[selected]
    if observed.size < 8:
        return float("nan")
    median = float(np.median(observed))
    noise = float(np.percentile(observed, 84.0) - median)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        scale = max(float(np.max(np.abs(observed))), 1.0)
        noise = float(np.finfo(float).eps * scale)
    return noise


def _radial_peak_candidates(
    profile: RadialProfile,
    *,
    noise_floor: float,
    top_n: int = 6,
    minimum_score: float = 2.0,
    min_coverage: float = 0.0,
) -> list[RidgePoint]:
    """Return observed local radial peaks eligible for continuity tracking."""

    if find_peaks is None or peak_prominences is None or gaussian_filter1d is None:
        return []
    q = np.asarray(profile.q, dtype=float)
    y = np.asarray(profile.intensity, dtype=float)
    counts = np.asarray(profile.counts)
    finite = np.isfinite(y) & (counts > 0)
    if np.count_nonzero(finite) < 3:
        return []
    fill = np.interp(q, q[finite], y[finite])
    smooth = gaussian_filter1d(fill, 1.0, mode="nearest")
    baseline, profile_noise = _radial_profile_baseline_noise(q, y, finite)
    noise = profile_noise if np.isfinite(profile_noise) and profile_noise > 0.0 else float(noise_floor)
    if not np.isfinite(noise) or noise <= 0.0:
        return []
    peak_indices, _ = find_peaks(smooth, distance=4)
    dense_profile = np.count_nonzero(finite) >= max(12, int(np.ceil(0.5 * len(finite))))
    supported_indices = _supported_radial_local_maxima(
        smooth,
        finite,
        dense_profile=dense_profile,
        blocked_bins=(np.asarray(profile.candidate_counts) > 0) & ~finite,
    )
    peak_indices = peak_indices[np.isin(peak_indices, supported_indices)]
    if peak_indices.size == 0:
        return []
    prominences = peak_prominences(smooth, peak_indices)[0]
    order = np.argsort(prominences, kind="mergesort")[::-1][: int(top_n)]
    global_coverage = float(np.sum(profile.counts) / max(1, np.sum(profile.candidate_counts)))
    area = (
        float(np.trapezoid(np.maximum(y[finite] - baseline, 0.0), q[finite]))
        if np.count_nonzero(finite) > 1
        else 0.0
    )
    candidates: list[RidgePoint] = []
    for position in order:
        index = int(peak_indices[int(position)])
        score = float(prominences[int(position)] / noise)
        if not np.isfinite(score) or score < float(minimum_score):
            continue
        peak_q, peak_intensity = _quadratic_peak(q, smooth, index)
        fwhm = _linear_fwhm(q, smooth, index, baseline, support_mask=finite)
        observed_indices = np.flatnonzero(finite)
        observed_position = int(np.searchsorted(observed_indices, index))
        local_indices = observed_indices[observed_position - 1 : observed_position + 2]
        local_coverage = float(
            np.nanmin(np.asarray(profile.coverage, dtype=float)[local_indices])
        )
        if local_coverage < float(min_coverage):
            continue
        candidates.append(
            RidgePoint(
                angle=profile.angle,
                q=peak_q,
                q_star=peak_q,
                intensity=peak_intensity,
                baseline=baseline,
                snr=score,
                radial_fwhm=fwhm,
                area=area,
                coverage=local_coverage,
                n_pixels=int(np.sum(profile.counts[local_indices])),
                valid=True,
                flags=_flags_with_q_unit(
                    APPARENT_FLAGS
                    + ("radial_continuity_candidate",)
                    + (() if np.isfinite(fwhm) else ("radial_fwhm_unavailable",)),
                    profile.q_unit,
                ),
                score=score,
                reason="accepted",
                q_unit=profile.q_unit,
            )
        )
        candidates[-1].metadata.update(
            profile_global_coverage=global_coverage,
            peak_coverage=local_coverage,
            peak_bin_index=int(index),
        )
    return candidates


def _reject_ridge_point(point: RidgePoint, reason: str) -> RidgePoint:
    candidate_q = float(point.q_star) if np.isfinite(point.q_star) else (float(point.q) if np.isfinite(point.q) else float("nan"))
    point.valid = False
    point.accepted = False
    point.q_star = float("nan")
    point.lamellar_spacing = float("nan")
    point.reason = str(reason)
    point.flags = tuple(dict.fromkeys(point.flags + (point.reason,)))
    if np.isfinite(candidate_q):
        point.metadata.setdefault("candidate_q", candidate_q)
    point.metadata.update(
        flags=point.flags,
        accepted=False,
        reason=point.reason,
    )
    return point


def _continuous_radial_points(
    profiles: Sequence[RadialProfile],
    *,
    noise_floor: float,
    snr_threshold: float = 2.0,
    minimum_track_support: int = 3,
    min_coverage: float = 0.0,
) -> list[RidgePoint]:
    """Select one point per angle from real multi-peak continuity tracks."""

    candidates_by_sector = [
        _radial_peak_candidates(
            profile,
            noise_floor=noise_floor,
            minimum_score=float(snr_threshold),
            min_coverage=min_coverage,
        )
        for profile in profiles
    ]
    nodes: list[tuple[int, RidgePoint]] = [
        (sector, point)
        for sector, candidates in enumerate(candidates_by_sector)
        for point in candidates
    ]
    if not nodes:
        return [
            _reject_ridge_point(
                _radial_peak(
                    profile,
                    snr_threshold=snr_threshold,
                    min_coverage=min_coverage,
                ),
                "no_continuous_candidate",
            )
            for profile in profiles
        ]

    parent = list(range(len(nodes)))
    lookup: dict[int, list[int]] = {}
    for node_index, (sector, _point) in enumerate(nodes):
        lookup.setdefault(sector, []).append(node_index)

    def find(node_index: int) -> int:
        while parent[node_index] != node_index:
            parent[node_index] = parent[parent[node_index]]
            node_index = parent[node_index]
        return node_index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    n_sectors = len(profiles)
    q_step = float(
        np.median(
            [
                np.median(np.diff(profile.q))
                for profile in profiles
                if len(profile.q) > 1
            ]
        )
    )
    for sector in range(n_sectors):
        for left in lookup.get(sector, []):
            for right in lookup.get((sector + 1) % n_sectors, []):
                left_point, right_point = nodes[left][1], nodes[right][1]
                widths = [
                    value
                    for value in (left_point.radial_fwhm, right_point.radial_fwhm)
                    if np.isfinite(value) and value > 0.0
                ]
                width_scale = max(widths) if widths else 0.0
                maximum_jump = max(6.0 * q_step, 0.75 * width_scale)
                if abs(left_point.q - right_point.q) <= maximum_jump:
                    union(left, right)

    components: dict[int, list[int]] = {}
    for node_index in range(len(nodes)):
        components.setdefault(find(node_index), []).append(node_index)
    support = {
        root: len({nodes[node_index][0] for node_index in component})
        for root, component in components.items()
    }

    selected: list[RidgePoint] = []
    for sector, profile in enumerate(profiles):
        eligible = [
            node_index
            for node_index in lookup.get(sector, [])
            if support.get(find(node_index), 0) >= int(minimum_track_support)
        ]
        if eligible:
            chosen = max(
                eligible,
                key=lambda node_index: (
                    support[find(node_index)],
                    nodes[node_index][1].score,
                    -nodes[node_index][1].q,
                ),
            )
            selected.append(nodes[chosen][1])
            continue
        candidates = candidates_by_sector[sector]
        if candidates:
            selected.append(
                _reject_ridge_point(candidates[0], "short_disconnected_track")
            )
        else:
            selected.append(
                _reject_ridge_point(
                    _radial_peak(
                        profile,
                        snr_threshold=snr_threshold,
                        min_coverage=min_coverage,
                    ),
                    "no_continuous_candidate",
                )
            )
    return selected


@dataclass
class _CurvatureField:
    """Detector-grid principal-curvature diagnostics used for ridge tracing."""

    image: np.ndarray
    smooth: np.ndarray
    q: np.ndarray
    angle: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    valid: np.ndarray
    candidate_domain: np.ndarray
    ridge_candidate: np.ndarray
    score: np.ndarray
    lambda_min: np.ndarray
    normal_slope: np.ndarray
    normal_y: np.ndarray
    normal_x: np.ndarray
    offset: np.ndarray
    support: np.ndarray
    q_normal_step: np.ndarray
    q_scale_anisotropy: np.ndarray
    baseline: float
    noise: float
    q_unit: str = "unknown"


def _surface_curvature_field(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    q_range: Any = None,
    mask: Any = None,
    smooth_sigma: float = 2.0,
    curvature_percentile: float = 25.0,
    normal_step: float = 1.0,
) -> _CurvatureField:
    """Calculate a strongly smoothed, mask-normalized 2-D ridge field.

    This follows the topographic construction used by Grubb et al. (2021):
    the ridge normal is the eigenvector of the most negative Hessian
    eigenvalue, and the subpixel ridge position is the zero of the slope in
    that direction.  Derivatives are evaluated on the detector pixel grid;
    accepted points are converted to q coordinates only after localization.
    """

    if gaussian_filter is None or map_coordinates is None:  # pragma: no cover
        raise RuntimeError("surface_curvature requires scipy.ndimage")
    sigma = float(smooth_sigma)
    step = float(normal_step)
    percentile = float(curvature_percentile)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("curvature_sigma must be finite and positive")
    if not np.isfinite(step) or step <= 0 or step > 2.0:
        raise ValueError("curvature_normal_step must be in (0, 2]")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("curvature_percentile must be in [0, 100]")

    raw = _array_field(frame, ("data", "intensity", "image", "values"))
    if raw is None:
        raise ValueError("ImageFrame must expose data/intensity/image/values")
    image = np.asarray(raw, dtype=float)
    if image.ndim != 2:
        raise ValueError("surface_curvature requires a two-dimensional image")
    values, q_flat, angle_flat, valid_flat = _extract_maps(frame, qmap, mask)
    q_min, q_max = _q_limits(q_flat, q_window, q_range)
    q = q_flat.reshape(image.shape)
    angle = angle_flat.reshape(image.shape)
    valid = valid_flat.reshape(image.shape)
    candidate_domain = (
        np.isfinite(image)
        & np.isfinite(q)
        & np.isfinite(angle)
        & (q >= q_min)
        & (q <= q_max)
    )
    valid_domain = candidate_domain & valid
    if not np.any(valid_domain):
        raise ValueError("q window and mask leave no finite pixels for curvature ridge extraction")

    # Normalized convolution prevents masked pixels from being interpreted as
    # zero intensity.  The support field is retained so mask edges cannot be
    # mistaken for physical ridges after smoothing.
    weights = valid_domain.astype(float)
    support = gaussian_filter(weights, sigma=sigma, mode="nearest")
    numerator = gaussian_filter(np.where(valid_domain, image, 0.0), sigma=sigma, mode="nearest")
    smooth = np.divide(
        numerator,
        support,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=support > 1e-8,
    )
    finite_smooth = np.isfinite(smooth) & (support > 1e-8)
    fill = float(np.nanmedian(smooth[finite_smooth])) if np.any(finite_smooth) else 0.0
    derivative_surface = np.where(finite_smooth, smooth, fill)

    grad_y, grad_x = np.gradient(derivative_surface)
    hyy, hyx = np.gradient(grad_y)
    hxy, hxx = np.gradient(grad_x)
    hxy = 0.5 * (hxy + hyx)
    trace = hxx + hyy
    discriminant = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy**2, 0.0))
    lambda_min = 0.5 * (trace - discriminant)

    # Half-angle gives the largest-eigenvalue direction; its perpendicular is
    # the minimum-eigenvalue direction.  Eigenvector sign is irrelevant to the
    # zero-slope and subpixel equations.
    tangent_angle = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
    normal_x = -np.sin(tangent_angle)
    normal_y = np.cos(tangent_angle)
    normal_slope = grad_x * normal_x + grad_y * normal_y

    # Derivatives above intentionally remain on the detector pixel grid.  The
    # local q-map Jacobian below records how one pixel step maps into q space,
    # so curvature results can be compared across detector geometry/scales
    # without pretending that pixel curvature already has reciprocal units.
    qx_grid = q * np.cos(angle)
    qy_grid = q * np.sin(angle)
    dqx_dy, dqx_dx = np.gradient(qx_grid)
    dqy_dy, dqy_dx = np.gradient(qy_grid)
    normal_qx = dqx_dx * normal_x + dqx_dy * normal_y
    normal_qy = dqy_dx * normal_x + dqy_dy * normal_y
    tangent_x = -normal_y
    tangent_y = normal_x
    tangent_qx = dqx_dx * tangent_x + dqx_dy * tangent_y
    tangent_qy = dqy_dx * tangent_x + dqy_dy * tangent_y
    q_normal_step = np.hypot(normal_qx, normal_qy)
    q_tangent_step = np.hypot(tangent_qx, tangent_qy)
    q_scale_min = np.minimum(q_normal_step, q_tangent_step)
    q_scale_max = np.maximum(q_normal_step, q_tangent_step)
    q_scale_anisotropy = np.divide(
        q_scale_max,
        q_scale_min,
        out=np.full_like(q_scale_max, np.nan),
        where=q_scale_min > np.finfo(float).eps,
    )

    yy, xx = np.indices(image.shape, dtype=float)
    minus = map_coordinates(
        derivative_surface,
        [yy - step * normal_y, xx - step * normal_x],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    plus = map_coordinates(
        derivative_surface,
        [yy + step * normal_y, xx + step * normal_x],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    second_along_normal = (minus - 2.0 * derivative_surface + plus) / (step * step)
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = 0.5 * step * (minus - plus) / (minus - 2.0 * derivative_surface + plus)

    # Strong smoothing creates a partial-support halo around masks.  Require
    # nearly complete support and one-pixel distance from the image boundary.
    edge_ok = np.ones(image.shape, dtype=bool)
    margin = max(1, int(np.ceil(step)))
    edge_ok[:margin, :] = edge_ok[-margin:, :] = False
    edge_ok[:, :margin] = edge_ok[:, -margin:] = False
    support_ok = support >= 0.90
    concave_down = (lambda_min < 0.0) & (second_along_normal < 0.0)
    subpixel_ok = np.isfinite(offset) & (np.abs(offset) <= 0.75 * step)
    strength = np.maximum(-lambda_min, 0.0)
    dynamic = float(np.nanmax(derivative_surface[valid_domain]) - np.nanmin(derivative_surface[valid_domain]))
    curvature_floor = max(np.finfo(float).eps, 1e-6 * max(dynamic, np.finfo(float).eps))
    usable_strength = strength[valid_domain & support_ok & concave_down & subpixel_ok]
    adaptive_floor = (
        float(np.nanpercentile(usable_strength, percentile))
        if usable_strength.size
        else float("inf")
    )
    threshold = max(curvature_floor, adaptive_floor)

    baseline = float(np.nanpercentile(derivative_surface[valid_domain], 10.0))
    noise = _robust_noise(derivative_surface[valid_domain] - baseline)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.nanstd(derivative_surface[valid_domain])), np.finfo(float).eps)
    signal = np.maximum(derivative_surface - baseline, 0.0)
    score = strength * signal / max(noise, np.finfo(float).eps)
    ridge_candidate = (
        valid_domain
        & support_ok
        & edge_ok
        & concave_down
        & subpixel_ok
        & (strength >= threshold)
        & np.isfinite(score)
    )

    return _CurvatureField(
        image=image,
        smooth=derivative_surface,
        q=q,
        angle=angle,
        qx=qx_grid,
        qy=qy_grid,
        valid=valid,
        candidate_domain=candidate_domain,
        ridge_candidate=ridge_candidate,
        score=score,
        lambda_min=lambda_min,
        normal_slope=normal_slope,
        normal_y=normal_y,
        normal_x=normal_x,
        offset=offset,
        support=support,
        q_normal_step=q_normal_step,
        q_scale_anisotropy=q_scale_anisotropy,
        baseline=baseline,
        noise=noise,
        q_unit=_q_unit(qmap),
    )


def _curvature_point_for_sector(
    field: _CurvatureField,
    requested_angle: float,
    sector_width: float,
    snr_threshold: float,
    min_coverage: float = 0.0,
) -> RidgePoint:
    """Select and refine the strongest principal-curvature ridge in a sector."""

    sector = _wrap_distance(field.angle, requested_angle) <= float(sector_width) / 2.0
    candidate_pixels = field.candidate_domain & sector
    valid_pixels = candidate_pixels & field.valid
    coverage = float(np.count_nonzero(valid_pixels) / max(1, np.count_nonzero(candidate_pixels)))
    accepted = field.ridge_candidate & sector
    if coverage < float(min_coverage):
        reason = "low_coverage"
        flags = list(APPARENT_FLAGS + ("detector_pixel_principal_curvature", reason))
        return RidgePoint(
            angle=float(requested_angle),
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            coverage=coverage,
            n_pixels=int(np.count_nonzero(valid_pixels)),
            valid=False,
            flags=tuple(flags),
            method="surface_curvature",
            accepted=False,
            reason=reason,
            score=0.0,
            q_unit=field.q_unit,
        )
    if not np.any(accepted):
        reason = "low_coverage" if coverage < 0.5 else "no_curvature_candidate"
        flags = list(APPARENT_FLAGS + ("detector_pixel_principal_curvature", reason))
        return RidgePoint(
            angle=float(requested_angle),
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            coverage=coverage,
            n_pixels=int(np.count_nonzero(valid_pixels)),
            valid=False,
            flags=tuple(flags),
            method="surface_curvature",
            accepted=False,
            reason=reason,
            score=0.0,
            q_unit=field.q_unit,
        )

    flat_candidates = np.flatnonzero(accepted)
    chosen_flat = flat_candidates[int(np.argmax(field.score.flat[flat_candidates]))]
    py, px = np.unravel_index(chosen_flat, field.image.shape)
    delta = float(field.offset[py, px])
    sub_y = float(py + delta * field.normal_y[py, px])
    sub_x = float(px + delta * field.normal_x[py, px])
    coordinates = np.asarray([[sub_y], [sub_x]], dtype=float)
    qx = float(map_coordinates(field.qx, coordinates, order=1, mode="nearest", prefilter=False)[0])
    qy = float(map_coordinates(field.qy, coordinates, order=1, mode="nearest", prefilter=False)[0])
    q_star = float(np.hypot(qx, qy))
    actual_angle = float(np.arctan2(qy, qx))
    peak_intensity = float(map_coordinates(field.smooth, coordinates, order=1, mode="nearest", prefilter=False)[0])
    snr = float((peak_intensity - field.baseline) / max(field.noise, np.finfo(float).eps))
    is_valid = bool(np.isfinite(q_star) and q_star > 0 and np.isfinite(snr) and snr >= snr_threshold)
    reason = "accepted" if is_valid else "low_snr"
    flags = APPARENT_FLAGS + ("detector_pixel_principal_curvature",)
    if not is_valid:
        flags += ("low_snr",)
    return RidgePoint(
        angle=actual_angle,
        q=q_star,
        q_star=q_star,
        lamellar_spacing=None,
        intensity=peak_intensity,
        baseline=field.baseline,
        snr=snr,
        radial_fwhm=float("nan"),
        azimuthal_fwhm=float("nan"),
        area=float("nan"),
        coverage=coverage,
        n_pixels=int(np.count_nonzero(valid_pixels)),
        valid=is_valid,
        flags=flags,
        method="surface_curvature",
        curvature=float(field.lambda_min[py, px]),
        normal_slope=float(field.normal_slope[py, px]),
        support=float(field.support[py, px]),
        pixel_y=sub_y,
        pixel_x=sub_x,
        accepted=is_valid,
        reason=reason,
        score=float(max(0.0, field.score[py, px]) * max(0.0, min(1.0, coverage))) if np.isfinite(field.score[py, px]) else 0.0,
        q_normal_step=float(field.q_normal_step[py, px]),
        q_scale_anisotropy=float(field.q_scale_anisotropy[py, px]),
        q_unit=field.q_unit,
    )


def _annotate_ridge_continuity(
    points: Sequence[RidgePoint],
    requested_angles: np.ndarray,
) -> tuple[float, float, float, tuple[str, ...]]:
    """Attach deterministic circular continuity diagnostics to real points.

    This is intentionally a small diagnostic, not a multi-track tracker.  A
    valid point is connected to the preceding valid angular sample; missing
    sectors create a gap and an unusually large local q change creates a
    jump.  No point is added or mirrored when either condition is observed.
    """

    n_points = len(points)
    if n_points == 0:
        return 0.0, float("nan"), float("nan"), tuple()
    angles = np.asarray(requested_angles, dtype=float).ravel()
    if angles.size != n_points:
        raise ValueError("requested_angles must contain one value per ridge point")
    valid = np.asarray([bool(point.valid) and np.isfinite(point.q) for point in points], dtype=bool)
    order = np.argsort(np.mod(angles, 2.0 * np.pi), kind="mergesort")
    sorted_angles = np.mod(angles[order], 2.0 * np.pi)
    full_deltas = np.mod(np.roll(sorted_angles, -1) - sorted_angles, 2.0 * np.pi)
    positive_deltas = full_deltas[full_deltas > 1e-12]
    expected_step = float(np.median(positive_deltas)) if positive_deltas.size else float("nan")
    valid_positions = [int(pos) for pos, index in enumerate(order) if valid[index]]
    if not valid_positions:
        for point in points:
            point.continuity_score = 0.0
            point.trajectory_id = None
            point.local_q_step = float("nan")
            if "no_valid_ridge" not in point.flags:
                point.flags = tuple(point.flags) + ("no_valid_ridge",)
                point.metadata["flags"] = point.flags
        return 0.0, 0.0, 0.0, ("no_valid_ridge",)

    q_valid = np.asarray([points[index].q for index in order if valid[index]], dtype=float)
    q_scale = float(np.nanmedian(np.abs(q_valid))) if q_valid.size else 0.0
    q_mad = float(np.nanmedian(np.abs(q_valid - np.nanmedian(q_valid)))) if q_valid.size else 0.0
    jump_threshold = max(0.10 * q_scale, 6.0 * q_mad, 1e-12)
    incoming: dict[int, tuple[float, bool, bool]] = {}
    for offset, current_pos in enumerate(valid_positions):
        if len(valid_positions) == 1:
            incoming[current_pos] = (float("nan"), False, False)
            continue
        previous_pos = valid_positions[offset - 1]
        previous_index = int(order[previous_pos])
        current_index = int(order[current_pos])
        angle_gap = float(np.mod(sorted_angles[current_pos] - sorted_angles[previous_pos], 2.0 * np.pi))
        if current_pos <= previous_pos:
            angle_gap = float(
                np.mod(
                    sorted_angles[current_pos] + 2.0 * np.pi - sorted_angles[previous_pos],
                    2.0 * np.pi,
                )
            )
        between_positions = (current_pos - previous_pos - 1) % n_points
        if current_pos <= previous_pos:
            between_positions = n_points - previous_pos - 1 + current_pos
        missing = any(not valid[int(order[(previous_pos + step) % n_points])] for step in range(1, between_positions + 1))
        local_q_step = float(abs(points[current_index].q - points[previous_index].q))
        angle_gap_flag = bool(np.isfinite(expected_step) and angle_gap > 1.75 * expected_step)
        jump_flag = bool(np.isfinite(local_q_step) and local_q_step > jump_threshold)
        incoming[current_pos] = (local_q_step, bool(missing or angle_gap_flag), jump_flag)

    trajectory_id = 0
    for offset, current_pos in enumerate(valid_positions):
        index = int(order[current_pos])
        local_q_step, gap_flag, jump_flag = incoming[current_pos]
        point = points[index]
        point.local_q_step = local_q_step
        point.continuity_score = 0.0 if gap_flag or jump_flag else 1.0
        point.trajectory_id = trajectory_id
        extra_flags = []
        if gap_flag:
            extra_flags.append("continuity_gap")
        if jump_flag:
            extra_flags.append("continuity_jump")
        if extra_flags:
            point.flags = tuple(dict.fromkeys(tuple(point.flags) + tuple(extra_flags)))
            point.metadata["flags"] = point.flags
        if offset > 0 and (gap_flag or jump_flag):
            trajectory_id += 1
            point.trajectory_id = trajectory_id
    for index, point in enumerate(points):
        if not valid[index]:
            point.continuity_score = 0.0
            point.trajectory_id = None
            point.local_q_step = float("nan")

    valid_count = int(np.count_nonzero(valid))
    continuity_values = np.asarray(
        [points[int(order[pos])].continuity_score for pos in valid_positions],
        dtype=float,
    )
    continuity_fraction = float(np.mean(continuity_values > 0.0))
    continuity_score = float(np.mean(continuity_values))
    diagnostics = []
    if any("continuity_gap" in point.flags for point in points):
        diagnostics.append("continuity_gap")
    if any("continuity_jump" in point.flags for point in points):
        diagnostics.append("continuity_jump")
    return float(valid_count / n_points), continuity_fraction, continuity_score, tuple(diagnostics)


def _annotate_azimuthal_tracks(
    points: Sequence[RidgePoint],
    annulus_indices: Mapping[int, Sequence[int]],
    *,
    n_annuli: int,
    q_step: float,
    angle_step: float,
) -> tuple[float, float, float, tuple[str, ...]]:
    """Attach q-wise trajectories and observed branch identities.

    Azimuthal maxima are sampled independently in each q annulus, so the
    radial-sector continuity diagnostic cannot be reused.  A point is linked
    only to a maximum in the immediately preceding *observed* annulus when
    its angular displacement is compatible with one angular bin (or the
    measured peak width).  Branch labels are then obtained from the two
    circular clusters that are actually present modulo pi; no missing
    quadrant is created to complete a pair.
    """

    if not points:
        return 0.0, 0.0, 0.0, ("no_valid_ridge",)
    previous: list[int] = []
    next_trajectory = 0
    ever_observed = False
    continuity_values: list[float] = []
    supported_annuli: set[int] = set()
    for annulus in range(int(n_annuli)):
        current = list(annulus_indices.get(annulus, ()))
        if current:
            supported_annuli.add(annulus)
        used_previous: set[int] = set()
        for point_index in current:
            point = points[int(point_index)]
            best: tuple[float, int] | None = None
            for previous_index in previous:
                if previous_index in used_previous:
                    continue
                prior = points[previous_index]
                widths = [
                    0.5 * float(width)
                    for width in (point.azimuthal_fwhm, prior.azimuthal_fwhm)
                    if np.isfinite(width) and width > 0.0
                ]
                allowed = max([2.5 * float(angle_step), *widths])
                distance = float(_wrap_distance(point.angle, prior.angle))
                candidate = (distance, previous_index)
                if distance <= allowed and (best is None or candidate < best):
                    best = candidate
            if best is None:
                point.trajectory_id = next_trajectory
                next_trajectory += 1
                point.continuity_score = 0.0 if ever_observed else 1.0
                if ever_observed:
                    point.flags = tuple(dict.fromkeys(point.flags + ("continuity_gap",)))
                    point.metadata["flags"] = point.flags
            else:
                point.trajectory_id = points[best[1]].trajectory_id
                point.continuity_score = 1.0
                used_previous.add(best[1])
            point.local_q_step = float(q_step)
            continuity_values.append(float(point.continuity_score))
        if current:
            ever_observed = True
        previous = current

    # Circular two-means in the unoriented angle domain.  A pair at +phi and
    # -phi has distinct centres modulo pi; Friedel counterparts share the
    # same centre.  Degenerate one-cluster data receives branch 0 only.
    angles = np.mod(np.asarray([point.angle for point in points], dtype=float), np.pi)
    if angles.size == 1:
        labels = np.zeros(1, dtype=int)
    else:
        sorted_angles = np.sort(angles)
        centres = np.asarray(
            (sorted_angles[angles.size // 4], sorted_angles[(3 * angles.size) // 4]),
            dtype=float,
        )
        for _ in range(24):
            distance = np.abs(angles[:, None] - centres[None, :])
            distance = np.minimum(distance, np.pi - distance)
            labels = np.argmin(distance, axis=1)
            updated = centres.copy()
            for cluster in (0, 1):
                values = angles[labels == cluster]
                if values.size:
                    updated[cluster] = float(
                        0.5 * np.mod(np.arctan2(np.mean(np.sin(2.0 * values)),
                                                  np.mean(np.cos(2.0 * values))), 2.0 * np.pi)
                    )
            if np.max(np.minimum(np.abs(updated - centres), np.pi - np.abs(updated - centres))) < 1e-10:
                centres = updated
                break
            centres = updated
        separation = float(min(abs(centres[0] - centres[1]), np.pi - abs(centres[0] - centres[1])))
        if separation < max(1.5 * float(angle_step), 1e-6) or np.count_nonzero(labels == 1) == 0:
            labels = np.zeros(angles.size, dtype=int)
        elif centres[1] < centres[0]:
            labels = 1 - labels
    for point, label in zip(points, labels):
        point.branch_id = int(label)
        point.metadata["branch_id"] = int(label)

    valid_annulus_fraction = float(len(supported_annuli) / max(1, int(n_annuli)))
    continuity_fraction = float(np.mean(np.asarray(continuity_values) > 0.0)) if continuity_values else 0.0
    continuity_score = float(np.mean(continuity_values)) if continuity_values else 0.0
    flags: list[str] = []
    if any("continuity_gap" in point.flags for point in points):
        flags.append("continuity_gap")
    if len(supported_annuli) < int(n_annuli):
        flags.append("short_q_annulus_support")
    return valid_annulus_fraction, continuity_fraction, continuity_score, tuple(flags)


def _azimuthal_peak_ridges(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    q_range: Any = None,
    n_annuli: int = 256,
    n_angle_bins: int = 72,
    snr_threshold: float = 2.0,
    min_peak_fraction: float = 0.0,
    min_coverage: float = 0.0,
    min_bin_count: int = 1,
    mask: Any = None,
    diagnostics: dict[str, Any] | None = None,
    cancel_event: Any = None,
) -> tuple[list[RidgePoint], np.ndarray, np.ndarray, float, float, float, tuple[str, ...]]:
    """Extract directly observed angular maxima in sampled q annuli.

    The image is binned once into a q-by-angle accumulator.  This keeps the
    method practical for detector-sized frames while preserving the distinction
    from ``radial_peak``: each point is an angular maximum at a fixed q annulus,
    not a radial maximum in a fixed angular sector.
    """

    n_annuli = int(n_annuli)
    n_angle_bins = int(n_angle_bins)
    if n_annuli < 4:
        raise ValueError("n_bins must be at least 4 for azimuthal_peak")
    if n_angle_bins < 16:
        raise ValueError("n_angles must be at least 16 for azimuthal_peak")
    if not np.isfinite(float(snr_threshold)) or float(snr_threshold) < 0.0:
        raise ValueError("ridge_snr_threshold must be finite and non-negative")
    if not np.isfinite(float(min_peak_fraction)) or not 0.0 <= float(min_peak_fraction) <= 1.0:
        raise ValueError("ridge_min_peak_fraction must be in [0, 1]")
    if not np.isfinite(float(min_coverage)) or not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError("ridge_min_coverage must be in [0, 1]")
    if isinstance(min_bin_count, (bool, np.bool_)) or int(min_bin_count) != min_bin_count or int(min_bin_count) < 1:
        raise ValueError("min_bin_count must be a positive integer")
    min_bin_count = int(min_bin_count)

    values, q, angle, valid = _extract_maps(frame, qmap, mask)
    source_data = _array_field(frame, ("data", "intensity", "image", "values"))
    source_shape = tuple(np.asarray(source_data).shape) if source_data is not None else ()
    if len(source_shape) >= 2:
        source_indices = np.unravel_index(np.arange(values.size), source_shape)
        pixel_y = np.asarray(source_indices[-2], dtype=float)
        pixel_x = np.asarray(source_indices[-1], dtype=float)
    else:
        pixel_y = np.zeros(values.size, dtype=float)
        pixel_x = np.arange(values.size, dtype=float)
    q_unit = _q_unit(qmap)
    q_min, q_max = _q_limits(q, q_window, q_range)
    q_edges = np.linspace(q_min, q_max, n_annuli + 1, dtype=float)
    q_centres = 0.5 * (q_edges[:-1] + q_edges[1:])
    q_step = float(q_edges[1] - q_edges[0])
    angle_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1, dtype=float)
    angle_centres = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    angle_step = float(angle_edges[1] - angle_edges[0])

    if diagnostics is not None:
        diagnostics.clear()

    q_index_float = (
        (q - q_min) / max(q_max - q_min, np.finfo(float).eps) * n_annuli
    )
    q_index = np.floor(np.where(np.isfinite(q_index_float), q_index_float, 0.0)).astype(np.int64)
    q_index = np.clip(q_index, 0, n_annuli - 1)
    angle_index_float = (angle + np.pi) / (2.0 * np.pi) * n_angle_bins
    angle_index = np.floor(
        np.where(np.isfinite(angle_index_float), angle_index_float, 0.0)
    ).astype(np.int64)
    angle_index = np.clip(angle_index, 0, n_angle_bins - 1)
    candidate = (
        np.isfinite(q)
        & np.isfinite(angle)
        & (q >= q_min)
        & (q <= q_max)
    )
    selected = candidate & valid & np.isfinite(values)
    selected_positions = np.flatnonzero(selected)
    flat_candidate = q_index[candidate] * n_angle_bins + angle_index[candidate]
    flat_selected = q_index[selected_positions] * n_angle_bins + angle_index[selected_positions]
    total = n_annuli * n_angle_bins
    candidate_counts = np.bincount(flat_candidate, minlength=total).reshape(n_annuli, n_angle_bins)
    counts = np.bincount(flat_selected, minlength=total).reshape(n_annuli, n_angle_bins)
    sums = np.bincount(
        flat_selected,
        weights=values[selected_positions],
        minlength=total,
    ).reshape(n_annuli, n_angle_bins)
    if selected_positions.size:
        bin_order = np.argsort(flat_selected, kind="stable")
        sorted_selected_positions = selected_positions[bin_order]
    else:
        sorted_selected_positions = np.empty(0, dtype=np.intp)
    flat_counts = counts.ravel()
    bin_ends = np.cumsum(flat_counts)
    bin_starts = bin_ends - flat_counts
    profile = np.divide(
        sums,
        counts,
        out=np.full((n_annuli, n_angle_bins), np.nan, dtype=float),
        where=counts > 0,
    )
    coverage = np.divide(
        counts,
        candidate_counts,
        out=np.zeros_like(profile),
        where=candidate_counts > 0,
    )
    annulus_candidate = np.sum(candidate_counts, axis=1)
    annulus_valid = np.sum(counts, axis=1)
    annulus_coverage = np.divide(
        annulus_valid,
        annulus_candidate,
        out=np.zeros(n_annuli, dtype=float),
        where=annulus_candidate > 0,
    )
    points: list[RidgePoint] = []
    annulus_indices: dict[int, list[int]] = {}
    smoothed_matrix = np.full_like(profile, np.nan, dtype=float)
    annulus_diagnostics: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    rejected_boundary = False
    rejected_support = False
    for annulus in range(n_annuli):
        raise_if_cancelled(cancel_event, "ridges:annulus")
        annulus_diag: dict[str, Any] = {
            "annulus_index": int(annulus),
            "q_min": float(q_edges[annulus]),
            "q_max": float(q_edges[annulus + 1]),
            "q_center": float(q_centres[annulus]),
            "candidate_count": int(annulus_candidate[annulus]),
            "valid_count": int(annulus_valid[annulus]),
            "coverage": float(annulus_coverage[annulus]),
            "baseline": float("nan"),
            "noise": float("nan"),
            "noise_method": None,
            "noise_details": {},
            "supported_bins": [],
            "candidates": [],
            "status": "not_evaluated",
        }
        annulus_diagnostics.append(annulus_diag)
        if annulus_candidate[annulus] <= 0 or annulus_coverage[annulus] < float(min_coverage):
            if annulus_candidate[annulus] > 0:
                rejected_support = True
                annulus_diag["status"] = "rejected_low_annulus_coverage"
            else:
                annulus_diag["status"] = "no_geometry_support"
            continue
        values_angular = profile[annulus]
        finite_profile = np.isfinite(values_angular) & (counts[annulus] > 0)
        support_mask = finite_profile & (coverage[annulus] >= float(min_coverage))
        annulus_diag["supported_bins"] = np.flatnonzero(support_mask).astype(int).tolist()
        if np.count_nonzero(finite_profile) < 3:
            annulus_diag["status"] = "insufficient_angular_support"
            continue
        baseline = float(np.nanpercentile(values_angular[finite_profile], 10.0))
        noise, noise_method, noise_details = _supported_angular_noise(values_angular, support_mask)
        smoothed = _smooth_supported_angular_profile(values_angular, support_mask, sigma=1.0)
        smoothed_matrix[annulus] = smoothed
        annulus_diag.update(
            baseline=float(baseline),
            noise=float(noise),
            noise_method=str(noise_method),
            noise_details=dict(noise_details),
            status="profile_evaluated",
        )
        peak_input = np.where(np.isfinite(smoothed), smoothed, baseline)
        if find_peaks is None:  # pragma: no cover
            peak_indices = np.asarray([int(np.nanargmax(peak_input))])
            peak_prominence = np.asarray([float(np.nanmax(peak_input) - baseline)])
        else:
            extended = np.tile(peak_input, 3)
            peak_indices_extended, properties = find_peaks(
                extended,
                prominence=max(0.0, float(snr_threshold) * noise),
                distance=max(1, int(round(n_angle_bins * 0.02))),
            )
            # Keep the prominence from the corresponding central copy and
            # deduplicate edge peaks modulo the angular period.
            peak_indices_list: list[int] = []
            peak_prominence_list: list[float] = []
            all_prominences = np.asarray(properties.get("prominences", ()), dtype=float)
            for property_index, peak_position_extended in enumerate(peak_indices_extended.tolist()):
                if not (n_angle_bins <= peak_position_extended < 2 * n_angle_bins):
                    continue
                peak = int(peak_position_extended - n_angle_bins)
                if all(
                    _wrap_distance(angle_centres[int(peak)], angle_centres[other]) > angle_step
                    for other in peak_indices_list
                ):
                    peak_indices_list.append(int(peak))
                    if property_index < all_prominences.size:
                        peak_prominence_list.append(float(all_prominences[property_index]))
                    else:
                        peak_prominence_list.append(float(peak_input[int(peak)] - baseline))
            peak_indices = np.asarray(peak_indices_list, dtype=int)
            peak_prominence = np.asarray(peak_prominence_list, dtype=float)
        finite_prominence = peak_prominence[np.isfinite(peak_prominence) & (peak_prominence > 0.0)]
        strongest_prominence = float(np.max(finite_prominence)) if finite_prominence.size else float("nan")
        for peak_position, index in enumerate(peak_indices.tolist()):
            index = int(index)
            prominence = float(peak_prominence[peak_position]) if peak_position < len(peak_prominence) else float(peak_input[index] - baseline)
            height = float(peak_input[index] - baseline)
            candidate_diag: dict[str, Any] = {
                "annulus_index": int(annulus),
                "angular_bin_index": int(index),
                "chi_deg": float(np.degrees(angle_centres[index])),
                "q_center": float(q_centres[annulus]),
                "raw_intensity": float(values_angular[index]) if np.isfinite(values_angular[index]) else float("nan"),
                "smoothed_intensity": float(smoothed[index]) if np.isfinite(smoothed[index]) else float("nan"),
                "baseline": float(baseline),
                "height_above_baseline": float(height),
                "prominence": float(prominence),
                "snr": float("nan"),
                "n_pixels": int(counts[annulus, index]),
                "coverage": float(coverage[annulus, index]),
                "accepted": False,
                "reason": "candidate",
            }
            all_candidates.append(candidate_diag)
            annulus_diag["candidates"].append(candidate_diag)
            neighbours = bool(
                support_mask[(index - 1) % n_angle_bins]
                and support_mask[(index + 1) % n_angle_bins]
            )
            if not support_mask[index] or not neighbours:
                rejected_boundary = True
                candidate_diag["reason"] = "masked_gap_or_boundary"
                continue
            peak_angle, peak_intensity = _quadratic_peak(
                angle_centres,
                smoothed,
                index,
                period=2.0 * np.pi,
            )
            if (
                np.isfinite(strongest_prominence)
                and prominence < float(min_peak_fraction) * strongest_prominence
            ):
                candidate_diag["reason"] = "below_annulus_prominence_fraction"
                continue
            snr = float(max(prominence, peak_intensity - baseline) / max(noise, np.finfo(float).eps))
            candidate_diag["snr"] = float(snr)
            if not np.isfinite(snr) or snr < float(snr_threshold):
                candidate_diag["reason"] = "below_snr_threshold"
                continue
            flat_bin = annulus * n_angle_bins + index
            support_positions = sorted_selected_positions[bin_starts[flat_bin] : bin_ends[flat_bin]]
            bin_values = np.asarray(values[support_positions], dtype=float)
            bin_values = bin_values[np.isfinite(bin_values)]
            excess = np.maximum(bin_values - baseline, 0.0)
            excess_sum = float(np.sum(excess))
            max_excess_fraction = (
                float(np.max(excess) / excess_sum)
                if excess.size and excess_sum > np.finfo(float).eps
                else float("nan")
            )
            n_eff = (
                float(excess_sum * excess_sum / np.sum(excess * excess))
                if excess.size and np.sum(excess * excess) > np.finfo(float).eps
                else float(bin_values.size)
            )
            candidate_diag.update(
                effective_support_pixels=float(n_eff),
                max_excess_fraction=float(max_excess_fraction),
            )
            if int(bin_values.size) < int(min_bin_count):
                rejected_support = True
                candidate_diag["reason"] = "insufficient_bin_support"
                continue
            # A single hot pixel must not become an angular ridge.  This is a
            # conservative diagnostic gate only for a bin that is demonstrably
            # dominated by one positive excess contribution; the raw mean is
            # retained unchanged in the diagnostics.
            if (
                bin_values.size >= 2
                and np.isfinite(n_eff)
                and (n_eff < 2.0 or (np.isfinite(max_excess_fraction) and max_excess_fraction > 0.75))
            ):
                rejected_support = True
                candidate_diag["reason"] = "hot_pixel_dominated"
                continue
            representative_x = representative_y = float("nan")
            if support_positions.size:
                xs, ys = pixel_x[support_positions], pixel_y[support_positions]
                distance = (xs - np.median(xs)) ** 2 + (ys - np.median(ys)) ** 2
                nearest = int(np.argmin(distance))
                representative_x, representative_y = float(xs[nearest]), float(ys[nearest])
            fwhm = _periodic_fwhm(angle_centres, smoothed, index, baseline)
            point = RidgePoint(
                angle=float(peak_angle),
                q=float(q_centres[annulus]),
                # An angular maximum is sampled at a prescribed q annulus;
                # it is not a radial reflection position.  Keep ``q`` for
                # the Cartesian track coordinate, but fail closed for the
                # legacy q_star/lamellar-spacing aliases.
                q_star=float("nan"),
                lamellar_spacing=float("nan"),
                intensity=float(peak_intensity),
                baseline=baseline,
                snr=snr,
                radial_fwhm=float("nan"),
                azimuthal_fwhm=float(fwhm),
                area=float(max(0.0, peak_intensity - baseline) * (fwhm if np.isfinite(fwhm) else angle_step)),
                coverage=float(annulus_coverage[annulus]),
                n_pixels=int(counts[annulus, index]),
                valid=True,
                source="observed",
                flags=_flags_with_q_unit(
                    APPARENT_FLAGS + (
                        "azimuthal_peak",
                        "observed_angular_maximum",
                        "azimuthal_area_approximation",
                        "spacing_unavailable_azimuthal_trajectory",
                    ),
                    q_unit,
                ),
                method="azimuthal_peak",
                support=float(annulus_coverage[annulus]),
                pixel_x=representative_x,
                pixel_y=representative_y,
                accepted=True,
                reason="accepted",
                q_unit=q_unit,
                score=float(max(0.0, snr) * max(0.0, min(1.0, annulus_coverage[annulus]))),
                q_normal_step=q_step,
            )
            point.metadata.update(
                annulus_index=int(annulus),
                annulus_q_min=float(q_edges[annulus]),
                annulus_q_max=float(q_edges[annulus + 1]),
                angular_bin_index=index,
                angular_bin_coverage=float(coverage[annulus, index]),
                prominence=float(prominence),
                raw_intensity=float(values_angular[index]),
                smoothed_intensity=float(smoothed[index]),
                raw_bin_chi=float(angle_centres[index]),
                raw_bin_chi_deg=float(np.degrees(angle_centres[index])),
                refinement="quadratic_on_supported_smoothed_profile",
                representative_pixel_role="median_of_valid_support_pixels",
                support_pixel_count=int(bin_values.size),
                effective_support_pixels=float(n_eff),
                max_excess_fraction=float(max_excess_fraction),
                profile_noise=float(noise),
                profile_noise_method=str(noise_method),
                area_definition="peak_height_times_angular_fwhm_approximation",
            )
            point_index = len(points)
            points.append(point)
            annulus_indices.setdefault(annulus, []).append(point_index)
            candidate_diag.update(
                accepted=True,
                reason="accepted",
                point_index=int(point_index),
                refined_angle=float(peak_angle),
                refined_angle_deg=float(np.degrees(peak_angle)),
            )

    valid_annulus_fraction, continuity_fraction, continuity_score, continuity_flags = _annotate_azimuthal_tracks(
        points,
        annulus_indices,
        n_annuli=n_annuli,
        q_step=q_step,
        angle_step=angle_step,
    )
    flags: list[str] = [
        "azimuthal_peak_ridge",
        "observed_angular_maxima",
        "azimuthal_area_approximation",
    ]
    flags.extend(continuity_flags)
    if rejected_boundary:
        flags.append("masked_gap_or_boundary_peak_rejected")
    if rejected_support:
        flags.append("low_peak_support")
    if not points:
        flags.append("no_azimuthal_peak")
    final_flags = tuple(dict.fromkeys(flags))
    if diagnostics is not None:
        diagnostics.update(
            {
                "schema_version": "azimuthal_peak_diagnostics.v1",
                "q_unit": q_unit,
                "q_edges": np.asarray(q_edges, dtype=float).copy(),
                "q_centers": np.asarray(q_centres, dtype=float).copy(),
                "q_centres": np.asarray(q_centres, dtype=float).copy(),
                "angle_edges_deg": np.degrees(angle_edges).astype(float, copy=True),
                "angle_centers_deg": np.degrees(angle_centres).astype(float, copy=True),
                "angle_centres_deg": np.degrees(angle_centres).astype(float, copy=True),
                "raw_mean": np.asarray(profile, dtype=float).copy(),
                "raw_sum": np.asarray(sums, dtype=float).copy(),
                "counts": np.asarray(counts, dtype=int).copy(),
                "geometry_counts": np.asarray(candidate_counts, dtype=int).copy(),
                "candidate_counts": np.asarray(candidate_counts, dtype=int).copy(),
                "coverage": np.asarray(coverage, dtype=float).copy(),
                "smoothed": np.asarray(smoothed_matrix, dtype=float).copy(),
                "annuli": annulus_diagnostics,
                "per_annulus": annulus_diagnostics,
                "candidates": all_candidates,
                "selected_points": [
                    {
                        "point_index": int(index),
                        "annulus_index": int(point.metadata.get("annulus_index", -1)),
                        "angular_bin_index": int(point.metadata.get("angular_bin_index", -1)),
                        "q": float(point.q),
                        "chi_deg": float(np.degrees(point.angle)),
                        "prominence": float(point.metadata.get("prominence", float("nan"))),
                    }
                    for index, point in enumerate(points)
                ],
                "settings": {
                    "n_annuli": int(n_annuli),
                    "n_angle_bins": int(n_angle_bins),
                    "snr_threshold": float(snr_threshold),
                    "min_peak_fraction": float(min_peak_fraction),
                    "min_coverage": float(min_coverage),
                    "min_bin_count": int(min_bin_count),
                    "smoothing_sigma_bins": 1.0,
                    "noise_definition": "robust_adjacent_difference_or_local_detrended_residual",
                    "smoothing_definition": "within_contiguous_supported_angular_runs_only",
                },
                "flags": final_flags,
                "valid_annulus_fraction": float(valid_annulus_fraction),
                "continuity_fraction": float(continuity_fraction),
                "continuity_score": float(continuity_score),
            }
        )
    return points, np.asarray([point.angle for point in points], dtype=float), np.asarray(
        [point.q for point in points], dtype=float
    ), valid_annulus_fraction, continuity_fraction, continuity_score, final_flags




def _batched_radial_profiles(
    frame: Any,
    qmap: Any,
    angle_array: np.ndarray,
    q_window: Any,
    *,
    q_range: Any,
    n_bins: int,
    sector_width: float,
    mask: Any,
    q_unit: str,
    cancel_event: Any,
    cancel_stage: str,
) -> list[RadialProfile]:
    """Compute uniform/arbitrary-sector radial profiles in a single map pass."""

    n = int(n_bins)
    if n < 8:
        raise ValueError("n_bins must be at least 8")
    values, q, phi, valid = _extract_maps(frame, qmap, mask)
    q_min, q_max = _q_limits(q, q_window, q_range)
    edges = np.linspace(q_min, q_max, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    in_q = np.isfinite(values) & np.isfinite(q) & np.isfinite(phi) & (q >= q_min) & (q <= q_max)
    q_in = q[in_q]
    phi_in = phi[in_q]
    val_in = values[in_q]
    valid_in = valid[in_q]
    bin_in = np.clip(np.digitize(q_in, edges, right=False) - 1, 0, n - 1)

    order = np.argsort(phi_in)
    phi_s = phi_in[order]
    val_s = val_in[order]
    valid_s = valid_in[order]
    bin_s = bin_in[order]
    total_n = int(phi_s.size)

    half_width = float(sector_width) / 2.0
    flags = _flags_with_q_unit(APPARENT_FLAGS, q_unit)
    eps = 1.0e-12
    full_circle_cand: np.ndarray | None = None
    full_circle_prof: np.ndarray | None = None
    full_circle_cnt: np.ndarray | None = None
    full_circle_cov: np.ndarray | None = None

    profiles: list[RadialProfile] = []
    for raw_angle in angle_array:
        raise_if_cancelled(cancel_event, cancel_stage)
        angle_f = float(raw_angle)
        if half_width < 0.0 or total_n == 0:
            candidate_counts = np.zeros(n, dtype=int)
            counts = np.zeros(n, dtype=int)
            profile = np.full(n, np.nan, dtype=float)
            coverage = np.zeros(n, dtype=float)
        elif half_width >= np.pi - 2.0 * eps:
            if full_circle_prof is None:
                cand_full = np.bincount(bin_s, minlength=n).astype(int)
                sel_full = bin_s[valid_s]
                val_full = val_s[valid_s]
                cnt_full = np.bincount(sel_full, minlength=n).astype(int)
                sum_full = np.bincount(sel_full, weights=val_full, minlength=n).astype(float)
                prof_full = np.full(n, np.nan, dtype=float)
                np.divide(sum_full, cnt_full, out=prof_full, where=cnt_full > 0)
                cov_full = np.divide(
                    cnt_full, cand_full, out=np.zeros_like(prof_full), where=cand_full > 0
                )
                full_circle_cand = cand_full
                full_circle_cnt = cnt_full
                full_circle_prof = prof_full
                full_circle_cov = cov_full
            assert full_circle_cand is not None
            assert full_circle_cnt is not None
            assert full_circle_prof is not None
            assert full_circle_cov is not None
            candidate_counts = full_circle_cand.copy()
            counts = full_circle_cnt.copy()
            profile = full_circle_prof.copy()
            coverage = full_circle_cov.copy()
        else:
            centre = ((angle_f + np.pi) % (2.0 * np.pi)) - np.pi
            lo = centre - half_width - eps
            hi = centre + half_width + eps
            if lo < -np.pi:
                s1_end = int(np.searchsorted(phi_s, hi, side="right"))
                s2_start = int(np.searchsorted(phi_s, lo + 2.0 * np.pi, side="left"))
                if s1_end >= s2_start:
                    slices: tuple[tuple[int, int], ...] = ((0, total_n),)
                else:
                    slices = ((0, s1_end), (s2_start, total_n))
            elif hi >= np.pi:
                s1_start = int(np.searchsorted(phi_s, lo, side="left"))
                s2_end = int(np.searchsorted(phi_s, hi - 2.0 * np.pi, side="right"))
                if s2_end >= s1_start:
                    slices = ((0, total_n),)
                else:
                    slices = ((0, s2_end), (s1_start, total_n))
            else:
                s_start = int(np.searchsorted(phi_s, lo, side="left"))
                s_end = int(np.searchsorted(phi_s, hi, side="right"))
                slices = ((s_start, s_end),)

            candidate_counts = np.zeros(n, dtype=int)
            counts = np.zeros(n, dtype=int)
            sums = np.zeros(n, dtype=float)
            for start, stop in slices:
                if stop <= start:
                    continue
                phi_sl = phi_s[start:stop]
                in_sec = _wrap_distance(phi_sl, angle_f) <= half_width
                if not np.any(in_sec):
                    continue
                bin_sl = bin_s[start:stop]
                candidate_counts += np.bincount(bin_sl[in_sec], minlength=n).astype(int)
                sel_mask = in_sec & valid_s[start:stop]
                if np.any(sel_mask):
                    sel_bins = bin_sl[sel_mask]
                    sel_vals = val_s[start:stop][sel_mask]
                    counts += np.bincount(sel_bins, minlength=n).astype(int)
                    sums += np.bincount(sel_bins, weights=sel_vals, minlength=n).astype(float)
            profile = np.full(n, np.nan, dtype=float)
            np.divide(sums, counts, out=profile, where=counts > 0)
            coverage = np.divide(
                counts, candidate_counts, out=np.zeros_like(profile), where=candidate_counts > 0
            )
        profiles.append(
            RadialProfile(
                angle=angle_f,
                q=centres,
                intensity=profile,
                counts=counts,
                candidate_counts=candidate_counts,
                coverage=coverage,
                q_min=q_min,
                q_max=q_max,
                flags=flags,
                q_unit=q_unit,
            )
        )
    return profiles


def measure_radial_ridges(
    frame: Any,
    qmap: Any,
    q_window: Any = None,
    *,
    q_range: Any = None,
    angles: Sequence[float] | None = None,
    n_angles: int = 72,
    sector_width: float | None = None,
    n_bins: int = 256,
    lobe_metrics: Sequence[LobeMetrics] | None = None,
    snr_threshold: float = 2.0,
    ridge_snr_threshold: float | None = None,
    ridge_min_peak_fraction: float = 0.0,
    ridge_min_coverage: float = 0.0,
    ridge_method: str = "radial_peak",
    method: str | None = None,
    mask: Any = None,
    curvature_sigma: float = 2.0,
    curvature_percentile: float = 25.0,
    curvature_normal_step: float = 1.0,
    cancel_event: Any = None,
) -> RidgeTrack:
    """Measure directly observed radial or annular angular maxima.

    If ``angles`` is omitted, uniformly spaced sectors cover the complete
    angular range.  Crucially, this routine never copies or mirrors a point
    from a counterpart quadrant; masked and low-SNR sectors stay invalid.
    With ``ridge_method='azimuthal_peak'``, ``n_bins`` is the number of q
    annuli and ``n_angles`` is the angular bin count.  In that mode
    ``ridge_min_peak_fraction`` retains only angular maxima whose measured
    prominence is at least that fraction of the strongest prominence in the
    same annulus; ``ridge_min_coverage`` independently gates detector
    support.
    """

    raise_if_cancelled(cancel_event, "ridges:start")
    ridge_method = str(method if method is not None else ridge_method).lower().replace("-", "_")
    if ridge_method not in {"radial_peak", "surface_curvature", "curvature", "azimuthal_peak"}:
        raise ValueError("ridge_method must be 'radial_peak', 'surface_curvature' or 'azimuthal_peak'")
    if ridge_snr_threshold is not None:
        snr_threshold = float(ridge_snr_threshold)
    snr_threshold = float(snr_threshold)
    if not np.isfinite(snr_threshold) or snr_threshold < 0.0:
        raise ValueError("ridge_snr_threshold must be finite and non-negative")
    ridge_min_peak_fraction = float(ridge_min_peak_fraction)
    if not np.isfinite(ridge_min_peak_fraction) or not 0.0 <= ridge_min_peak_fraction <= 1.0:
        raise ValueError("ridge_min_peak_fraction must be in [0, 1]")
    ridge_min_coverage = float(ridge_min_coverage)
    if not np.isfinite(ridge_min_coverage) or not 0.0 <= ridge_min_coverage <= 1.0:
        raise ValueError("ridge_min_coverage must be in [0, 1]")
    azimuthal_mode = ridge_method == "azimuthal_peak"
    curvature_mode = ridge_method in {"surface_curvature", "curvature"}
    automatic_angles = angles is None
    if angles is None:
        # Lobe locations are annotations, not a substitute for a densely
        # sampled trajectory.  A four-point-only track cannot constrain the
        # five geometric parameters of the shared-centre double ellipse.
        angles = np.linspace(-np.pi, np.pi, int(n_angles), endpoint=False).tolist()
    angle_array = np.asarray(angles, dtype=float).ravel()
    q_unit = _q_unit(qmap)
    continuity_mode = bool(
        not curvature_mode
        and automatic_angles
        and len(angle_array) >= 12
        and _q_to_nm_inverse_scale(q_unit) is not None
    )
    if not angle_array.size:
        return RidgeTrack(
            points=[],
            angles=angle_array,
            q=np.array([]),
            valid=np.array([], dtype=bool),
            coverage=np.array([]),
            q_unit=q_unit,
        )
    if azimuthal_mode:
        points, observed_angles, q_values, valid_fraction, continuity_fraction, continuity_score, azimuthal_flags = _azimuthal_peak_ridges(
            frame,
            qmap,
            q_window,
            q_range=q_range,
            n_annuli=n_bins,
            n_angle_bins=len(angle_array),
            snr_threshold=snr_threshold,
            min_peak_fraction=ridge_min_peak_fraction,
            min_coverage=ridge_min_coverage,
            mask=mask,
            cancel_event=cancel_event,
        )
        valid = np.asarray([point.valid for point in points], dtype=bool)
        coverage = np.asarray([point.coverage for point in points], dtype=float)
        track_flags = _flags_with_q_unit(
            APPARENT_FLAGS + azimuthal_flags,
            q_unit,
        )
        if lobe_metrics and points:
            # Keep the annulus-local angular width measured from the direct
            # profile.  A nearest lobe is a separate global annotation and
            # must not overwrite this method's observable.
            for point in points:
                raise_if_cancelled(cancel_event, "ridges:azimuthal-annotation")
                distances = [_wrap_distance(point.angle, lobe.angle) for lobe in lobe_metrics]
                nearest_lobe = lobe_metrics[int(np.argmin(distances))]
                point.metadata["nearest_global_lobe_angle"] = float(nearest_lobe.angle)
                point.metadata["nearest_global_lobe_fwhm"] = float(nearest_lobe.fwhm)
        return RidgeTrack(
            points=points,
            angles=observed_angles,
            q=q_values,
            valid=valid,
            coverage=coverage,
            flags=track_flags,
            q_unit=q_unit,
            valid_fraction=float(valid_fraction),
            continuity_fraction=float(continuity_fraction),
            continuity_score=float(continuity_score),
        )
    if sector_width is None:
        sector_width = 2.0 * np.pi / max(1, len(angle_array)) * 1.25
    curvature_field = None
    if curvature_mode:
        curvature_field = _surface_curvature_field(
            frame,
            qmap,
            q_window,
            q_range=q_range,
            mask=mask,
            smooth_sigma=curvature_sigma,
            curvature_percentile=curvature_percentile,
            normal_step=curvature_normal_step,
        )
    profile_fn = _resolve_radial_profile()
    points: list[RidgePoint]
    if continuity_mode:
        if profile_fn is measure_radial_profile:
            profiles = _batched_radial_profiles(
                frame,
                qmap,
                angle_array,
                q_window,
                q_range=q_range,
                n_bins=n_bins,
                sector_width=float(sector_width),
                mask=mask,
                q_unit=q_unit,
                cancel_event=cancel_event,
                cancel_stage="ridges:continuity-sector",
            )
        else:
            profiles = []
            for angle in angle_array:
                raise_if_cancelled(cancel_event, "ridges:continuity-sector")
                profiles.append(profile_fn(
                    frame,
                    qmap,
                    float(angle),
                    q_window,
                    q_range=q_range,
                    n_bins=n_bins,
                    sector_width=sector_width,
                    mask=mask,
                ))
        raise_if_cancelled(cancel_event, "ridges:continuity-noise")
        noise_floor = _radial_noise_floor(
            frame,
            qmap,
            q_window,
            q_range=q_range,
            mask=mask,
        )
        points = _continuous_radial_points(
            profiles,
            noise_floor=noise_floor,
            snr_threshold=snr_threshold,
            min_coverage=ridge_min_coverage,
        )
        raise_if_cancelled(cancel_event, "ridges:continuity-complete")
    elif not curvature_mode and profile_fn is measure_radial_profile:
        batched_profiles = _batched_radial_profiles(
            frame,
            qmap,
            angle_array,
            q_window,
            q_range=q_range,
            n_bins=n_bins,
            sector_width=float(sector_width),
            mask=mask,
            q_unit=q_unit,
            cancel_event=cancel_event,
            cancel_stage="ridges:sector",
        )
        points = [
            _radial_peak(
                radial,
                snr_threshold=snr_threshold,
                min_coverage=ridge_min_coverage,
            )
            for radial in batched_profiles
        ]
    else:
        points = []
        for angle in angle_array:
            raise_if_cancelled(cancel_event, "ridges:sector")
            if curvature_mode:
                assert curvature_field is not None
                point = _curvature_point_for_sector(
                    curvature_field,
                    float(angle),
                    float(sector_width),
                    float(snr_threshold),
                    min_coverage=float(ridge_min_coverage),
                )
            else:
                radial = profile_fn(
                    frame,
                    qmap,
                    float(angle),
                    q_window,
                    q_range=q_range,
                    n_bins=n_bins,
                    sector_width=sector_width,
                    mask=mask,
                )
                point = _radial_peak(
                    radial,
                    snr_threshold=snr_threshold,
                    min_coverage=ridge_min_coverage,
                )
            points.append(point)

    if lobe_metrics:
        for angle, point in zip(angle_array, points):
            distances = [_wrap_distance(float(angle), lobe.angle) for lobe in lobe_metrics]
            nearest = int(np.argmin(distances))
            nearest_lobe = lobe_metrics[nearest]
            point.azimuthal_fwhm = float(nearest_lobe.fwhm)  # type: ignore[misc]
    q_values = np.asarray([point.q for point in points], dtype=float)
    valid = np.asarray([point.valid for point in points], dtype=bool)
    coverage = np.asarray([point.coverage for point in points], dtype=float)
    valid_fraction, continuity_fraction, continuity_score, continuity_flags = _annotate_ridge_continuity(
        points,
        angle_array,
    )
    track_flags = APPARENT_FLAGS + (("detector_pixel_principal_curvature",) if curvature_mode else tuple())
    if continuity_mode:
        track_flags += ("radial_continuity_tracking",)
    track_flags += continuity_flags
    track_flags = _flags_with_q_unit(track_flags, q_unit)
    return RidgeTrack(
        points=points,
        angles=angle_array,
        q=q_values,
        valid=valid,
        coverage=coverage,
        flags=track_flags,
        q_unit=q_unit,
        valid_fraction=valid_fraction,
        continuity_fraction=continuity_fraction,
        continuity_score=continuity_score,
    )


radial_ridge = measure_radial_ridges
extract_ridge_points = measure_radial_ridges


def extract_surface_curvature_ridges(*args: Any, **kwargs: Any) -> RidgeTrack:
    """Explicit public entry point for the principal-curvature ridge mode."""

    kwargs["ridge_method"] = "surface_curvature"
    return _resolve_radial_ridges()(*args, **kwargs)


def _measure_lobe_radial_observables(
    frame: Any,
    qmap: Any,
    q_window: Any,
    lobes: Sequence[LobeMetrics],
    *,
    n_radial_bins: int,
    snr_threshold: float,
    min_coverage: float,
    mask: Any = None,
    cancel_event: Any = None,
) -> tuple[list[RadialProfile], list[RidgePoint]]:
    """Measure radial reflection quantities in each observed lobe sector.

    This companion measurement is intentionally separate from an
    ``azimuthal_peak`` trajectory.  The latter supplies a q-annulus coordinate
    for an angular maximum, while this routine supplies a genuine radial
    q_star (and hence optional spacing proxy) from a narrow, observed angular
    sector.  Invalid/edge peaks are retained with their reason so a caller can
    audit missing support without inventing a value.
    """

    profiles: list[RadialProfile] = []
    peaks: list[RidgePoint] = []
    for lobe in lobes:
        raise_if_cancelled(cancel_event, "lobe-radial:sector")
        if not lobe.valid or not np.isfinite(lobe.angle):
            continue
        fwhm = float(lobe.fwhm) if np.isfinite(lobe.fwhm) and lobe.fwhm > 0.0 else 0.0
        sector_width = max(np.deg2rad(6.0), min(np.deg2rad(30.0), 1.5 * fwhm))
        profile = _resolve_radial_profile()(
            frame,
            qmap,
            float(lobe.angle),
            q_window,
            n_bins=n_radial_bins,
            sector_width=sector_width,
            mask=mask,
        )
        raise_if_cancelled(cancel_event, "lobe-radial:peak")
        peak = _radial_peak(
            profile,
            snr_threshold=snr_threshold,
            min_coverage=min_coverage,
        )
        peak.flags = tuple(dict.fromkeys(peak.flags + ("lobe_radial_peak",)))
        peak.metadata.update(
            flags=peak.flags,
            lobe_angle=float(lobe.angle),
            lobe_sector_width=float(sector_width),
            source_method="radial_peak_in_observed_lobe_sector",
        )
        profiles.append(profile)
        peaks.append(peak)
    raise_if_cancelled(cancel_event, "lobe-radial:complete")
    return profiles, peaks


__all__ = [
    "extract_ridge_points",
    "extract_surface_curvature_ridges",
    "measure_radial_profile",
    "measure_radial_ridges",
    "radial_profile",
    "radial_ridge",
    "_CurvatureField",
    "_annotate_azimuthal_tracks",
    "_annotate_ridge_continuity",
    "_azimuthal_peak_ridges",
    "_continuous_radial_points",
    "_curvature_point_for_sector",
    "_measure_lobe_radial_observables",
    "_radial_noise_floor",
    "_radial_peak",
    "_radial_peak_candidates",
    "_radial_profile_baseline_noise",
    "_reject_ridge_point",
    "_supported_radial_local_maxima",
    "_surface_curvature_field",
]
