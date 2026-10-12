from __future__ import annotations

import json

import numpy as np
import pytest

from butterfly_saxs.observables import RadialProfile, _radial_peak, _radial_peak_candidates


def _profile(
    intensity: np.ndarray,
    *,
    counts: np.ndarray | None = None,
    candidate_counts: np.ndarray | None = None,
) -> RadialProfile:
    values = np.asarray(intensity, dtype=float)
    measured = np.ones(values.size, dtype=bool) if counts is None else np.asarray(counts) > 0
    actual_counts = np.full(values.size, 20, dtype=int) if counts is None else np.asarray(counts, dtype=int)
    geometry_counts = (
        np.full(values.size, 20, dtype=int)
        if candidate_counts is None
        else np.asarray(candidate_counts, dtype=int)
    )
    values = values.copy()
    values[~measured] = np.nan
    return RadialProfile(
        angle=0.0,
        q=np.linspace(0.2534375, 1.3465625, values.size),
        intensity=values,
        counts=actual_counts,
        candidate_counts=geometry_counts,
        coverage=np.divide(
            actual_counts,
            geometry_counts,
            out=np.zeros(values.size, dtype=float),
            where=geometry_counts > 0,
        ),
        q_min=0.25,
        q_max=1.35,
        q_unit="nm^-1",
    )


@pytest.mark.parametrize(
    ("intensity", "edge"),
    [
        (np.linspace(100.0, 1.0, 40), "low"),
        (np.linspace(1.0, 100.0, 40), "high"),
    ],
)
def test_high_snr_monotonic_q_window_edge_is_candidate_not_resolved_peak(
    intensity: np.ndarray,
    edge: str,
) -> None:
    point = _radial_peak(_profile(intensity), snr_threshold=2.0)

    assert point.snr > 2.0
    assert np.isfinite(point.q)  # Retain the boundary-adjacent candidate coordinate.
    assert np.isnan(point.q_star)
    assert np.isnan(point.lamellar_spacing)
    assert not point.valid
    assert not point.accepted
    assert point.reason == "unresolved_radial_boundary"
    assert point.metadata["resolved_local_maximum"] is False
    assert point.metadata["candidate_edge"] == edge
    assert point.as_dict()["metadata"]["candidate_q"] == pytest.approx(point.q)


def test_low_coverage_boundary_keeps_candidate_but_not_accepted_q_or_spacing() -> None:
    intensity = np.linspace(100.0, 1.0, 40)
    counts = np.full(intensity.size, 5, dtype=int)
    point = _radial_peak(
        _profile(intensity, counts=counts),
        snr_threshold=2.0,
        min_coverage=0.5,
    )

    assert point.reason == "low_peak_support"
    assert "unresolved_radial_boundary" in point.flags
    assert np.isfinite(point.q)
    assert np.isnan(point.q_star)
    assert np.isnan(point.lamellar_spacing)
    assert np.isfinite(point.intensity) and np.isfinite(point.snr)
    assert point.metadata["resolved_local_maximum"] is False
    assert point.metadata["candidate_edge"] == "low"


def test_dense_peak_next_to_unmeasured_q_gap_is_not_resolved() -> None:
    q = np.arange(60, dtype=float)
    intensity = 2.0 + 40.0 * np.exp(-0.5 * ((q - 30.0) / 5.0) ** 2)
    counts = np.full(q.size, 20, dtype=int)
    counts[29] = 0
    point = _radial_peak(_profile(intensity, counts=counts), snr_threshold=2.0)

    assert np.isfinite(point.intensity)
    assert np.isfinite(point.snr)
    assert np.isfinite(point.q)
    assert np.isnan(point.q_star)
    assert not point.valid
    assert point.reason == "unsupported_radial_gap"
    assert point.metadata["resolved_local_maximum"] is False
    assert point.metadata["peak_bin_index"] == 30
    assert point.as_dict()["metadata"]["reason"] == "unsupported_radial_gap"


def test_sparse_peak_cannot_be_bracketed_across_a_masked_geometry_gap() -> None:
    intensity = np.full(16, np.nan)
    counts = np.zeros(16, dtype=int)
    geometry_counts = np.zeros(16, dtype=int)
    intensity[[3, 8, 13]] = [2.0, 100.0, 1.0]
    counts[[3, 8, 13]] = 20
    geometry_counts[[3, 7, 8, 13]] = 20
    point = _radial_peak(
        _profile(intensity, counts=counts, candidate_counts=geometry_counts),
        snr_threshold=1.0,
    )

    assert not point.valid and not point.accepted
    assert point.reason == "unsupported_radial_gap"
    assert np.isfinite(point.q)
    assert np.isnan(point.q_star)
    assert np.isnan(point.lamellar_spacing)
    assert np.isfinite(point.intensity) and np.isfinite(point.snr)
    assert point.metadata["resolved_local_maximum"] is False
    assert point.metadata["has_two_sided_support"] is False


def test_bracketed_interior_peak_is_accepted() -> None:
    q = np.arange(60, dtype=float)
    intensity = 2.0 + 40.0 * np.exp(-0.5 * ((q - 30.0) / 4.0) ** 2)
    point = _radial_peak(_profile(intensity), snr_threshold=2.0)

    assert point.valid and point.accepted
    assert point.reason == "accepted"
    assert point.q_star == pytest.approx(_profile(intensity).q[30], abs=0.01)
    assert point.metadata["resolved_local_maximum"] is True
    assert point.as_dict()["metadata"]["resolved_local_maximum"] is True


def test_measured_flat_top_with_lower_bins_on_both_sides_is_an_interior_peak() -> None:
    intensity = np.full(40, 2.0)
    intensity[19:21] = 20.0
    profile = _profile(intensity)

    point = _radial_peak(profile, snr_threshold=2.0)

    assert point.valid and point.accepted
    assert point.q_star == pytest.approx(profile.q[19:21].mean(), abs=0.01)
    assert point.metadata["resolved_local_maximum"] is True


def test_bracketed_sparse_peak_can_be_accepted_without_claiming_fwhm() -> None:
    intensity = np.full(16, np.nan)
    counts = np.zeros(16, dtype=int)
    geometry_counts = np.zeros(16, dtype=int)
    intensity[[3, 8, 13]] = [2.0, 100.0, 1.0]
    counts[[3, 8, 13]] = 20
    geometry_counts[[3, 8, 13]] = 20
    point = _radial_peak(
        _profile(intensity, counts=counts, candidate_counts=geometry_counts),
        snr_threshold=1.0,
    )

    assert point.valid and point.accepted
    assert np.isfinite(point.q_star)
    assert np.isnan(point.radial_fwhm)
    assert "radial_fwhm_unavailable" in point.flags
    assert point.metadata["radial_fwhm_available"] is False
    assert point.metadata["resolved_local_maximum"] is True


def test_masked_half_height_flank_suppresses_fwhm_without_rejecting_q_star() -> None:
    q = np.arange(60, dtype=float)
    intensity = 2.0 + 40.0 * np.exp(-0.5 * ((q - 30.0) / 4.0) ** 2)
    counts = np.full(q.size, 20, dtype=int)
    counts[35] = 0
    point = _radial_peak(_profile(intensity, counts=counts), snr_threshold=2.0)

    assert point.valid and point.accepted
    assert np.isfinite(point.q_star)
    assert np.isnan(point.radial_fwhm)
    assert "radial_fwhm_unavailable" in point.flags
    assert point.metadata["radial_fwhm_available"] is False
    assert point.metadata["resolved_local_maximum"] is True


@pytest.mark.parametrize("masked_gap", [False, True])
def test_continuity_candidates_require_measured_sparse_peak_support(masked_gap: bool) -> None:
    intensity = np.full(16, np.nan)
    counts = np.zeros(16, dtype=int)
    geometry_counts = np.zeros(16, dtype=int)
    intensity[[3, 8, 13]] = [2.0, 100.0, 1.0]
    counts[[3, 8, 13]] = 20
    geometry_counts[[3, 8, 13]] = 20
    if masked_gap:
        geometry_counts[7] = 20
    candidates = _radial_peak_candidates(
        _profile(intensity, counts=counts, candidate_counts=geometry_counts),
        noise_floor=1.0,
        minimum_score=0.1,
    )

    if masked_gap:
        assert candidates == []
    else:
        assert len(candidates) == 1
        assert candidates[0].valid and candidates[0].accepted
        assert np.isfinite(candidates[0].q_star)
        assert np.isnan(candidates[0].radial_fwhm)


def test_sparse_continuity_coverage_gate_includes_observed_brackets() -> None:
    intensity = np.full(16, np.nan)
    counts = np.zeros(16, dtype=int)
    geometry_counts = np.zeros(16, dtype=int)
    intensity[[3, 8, 13]] = [2.0, 100.0, 1.0]
    counts[[3, 8, 13]] = [1, 20, 20]
    geometry_counts[[3, 8, 13]] = 20
    profile = _profile(intensity, counts=counts, candidate_counts=geometry_counts)

    candidates = _radial_peak_candidates(profile, noise_floor=1.0, minimum_score=0.1)
    assert len(candidates) == 1
    assert candidates[0].coverage == pytest.approx(0.05)
    assert _radial_peak_candidates(
        profile, noise_floor=1.0, minimum_score=0.1, min_coverage=0.5,
    ) == []


def test_unresolved_candidate_serializes_without_physical_peak_or_spacing_aliases() -> None:
    from butterfly_saxs.pipeline import _jsonable

    point = _radial_peak(_profile(np.linspace(100.0, 1.0, 40)))
    record = json.loads(json.dumps(_jsonable(point.as_dict()), allow_nan=False))

    assert record["q"] == pytest.approx(point.q)
    assert record["metadata"]["candidate_q"] == pytest.approx(point.q)
    assert record["metadata"]["resolved_local_maximum"] is False
    assert record["valid"] is False and record["accepted"] is False
    for key in ("q_star", "q_star_Ainv", "q_star_nm_inv", "lamellar_spacing", "Ln", "Ln_nm"):
        assert record[key] is None
