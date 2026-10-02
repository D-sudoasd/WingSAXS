from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.report_measurements import measure_frame


def test_radial_profiles_keep_masked_pixels_in_candidate_coverage_and_include_last_edge():
    image = np.array([[1.0, 3.0], [-1.0, 5.0]])
    qx = np.array([[0.0, 1.0], [0.0, 1.0]])
    qy = np.array([[0.0, 0.0], [1.0, 1.0]])
    valid = np.ones(image.shape, dtype=bool)
    valid[1, 0] = False

    result = measure_frame(
        image,
        qx,
        qy,
        valid_mask=valid,
        q_edges=[0.0, 1.0, np.sqrt(2.0)],
        radial_bins=2,
        angular_bins=4,
    )

    radial = result["radial_rows"]
    assert radial[0]["q_min"] == 0.0
    assert radial[1]["q_max"] == pytest.approx(np.sqrt(2.0))
    # q=1 lies on the interior edge and is assigned once to the upper bin;
    # q=sqrt(2) lies on the final edge and remains included.
    assert radial[0]["count"] == 1
    assert radial[1]["candidate_count"] == 3
    assert radial[1]["count"] == 2
    assert radial[1]["coverage"] == pytest.approx(2.0 / 3.0)
    assert radial[1]["mean"] == pytest.approx(4.0)
    assert radial[1]["sum"] == pytest.approx(8.0)
    assert radial[1]["std"] == pytest.approx(np.sqrt(2.0))
    assert radial[1]["sem"] == pytest.approx(1.0)
    assert "not_fit_uncertainty" in radial[1]["sem_kind"]

    assert result["polar"]["mean"].shape == (2, 4)
    assert result["polar"]["q_edges"].tolist() == pytest.approx([0.0, 1.0, np.sqrt(2.0)])
    assert result["summary"]["pixel_counts"]["candidate_qmap"] == 4
    assert result["summary"]["pixel_counts"]["observed"] == 3
    assert result["summary"]["quadrants"]["QI"]["candidate_count"] == 4
    assert result["summary"]["quadrants"]["QI"]["count"] == 3
    assert result["summary"]["maxima"] == {
        "intensity": 5.0,
        "row": 1,
        "column": 1,
        "qx": 1.0,
        "qy": 1.0,
        "q": pytest.approx(np.sqrt(2.0)),
        "angle_deg": pytest.approx(45.0),
    }


def test_angular_and_polar_bins_retain_empty_bins_as_missing():
    image = np.array([[1.0, 2.0, 3.0, 4.0]])
    qx = np.array([[1.0, 0.0, -1.0, 0.0]])
    qy = np.array([[0.0, 1.0, 0.0, -1.0]])

    result = measure_frame(image, qx, qy, q_unit="pixel-q", q_edges=[0.0, 1.0], angular_bins=4)

    angular = result["angular_rows"]
    assert [row["count"] for row in angular] == [0, 1, 1, 2]
    assert angular[0]["mean"] is None
    assert angular[0]["sum"] is None
    assert angular[1]["mean"] == pytest.approx(4.0)
    assert angular[2]["mean"] == pytest.approx(1.0)
    assert angular[3]["mean"] == pytest.approx(2.5)
    assert result["polar"]["count"].tolist() == [[0, 1, 1, 2]]
    assert np.isnan(result["polar"]["mean"][0, 0])
    assert np.isnan(result["polar"]["sum"][0, 0])
    assert result["polar"]["coverage"][0].tolist() == pytest.approx([np.nan, 1.0, 1.0, 1.0], nan_ok=True)
    assert result["polar"]["angle_edges"].tolist() == pytest.approx([-180.0, -90.0, 0.0, 90.0, 180.0])
    assert result["summary"]["quadrants"]["QI"]["sum"] == pytest.approx(3.0)
    assert result["summary"]["quadrants"]["QII"]["sum"] == pytest.approx(3.0)
    assert result["summary"]["quadrants"]["QIV"]["sum"] == pytest.approx(4.0)


def test_summary_reports_negative_intensities_and_descriptive_weighted_axis():
    image = np.array([[2.0, 2.0, -5.0]])
    qx = np.array([[-1.0, 1.0, 0.0]])
    qy = np.array([[0.0, 0.0, 1.0]])

    result = measure_frame(image, qx, qy, q_unit="pixel-q", radial_bins=2, angular_bins=8)

    summary = result["summary"]
    assert summary["intensity"]["count"] == 3
    assert summary["intensity"]["sum"] == pytest.approx(-1.0)
    assert summary["intensity"]["mean"] == pytest.approx(-1.0 / 3.0)
    assert summary["intensity"]["negative_count"] == 1
    assert summary["intensity"]["p50"] == pytest.approx(2.0)
    assert summary["maxima"]["column"] == 0  # deterministic first maximum in row-major order
    populated_radial_bin = next(row for row in result["radial_rows"] if row["count"] == 3)
    assert populated_radial_bin["std"] == pytest.approx(7.0 / np.sqrt(3.0))
    assert populated_radial_bin["sem"] == pytest.approx(7.0 / 3.0)

    tensor = summary["in_plane_intensity_tensor"]
    assert tensor["status"] == "estimated"
    assert tensor["weighting"] == "non-negative observed intensity: max(I, 0)"
    assert tensor["principal_axis_deg"] == pytest.approx(0.0)
    assert tensor["anisotropy"] == pytest.approx(1.0)
    assert tensor["axis_reason"] is None
    assert "unique 3D structure" in tensor["interpretation"]
    assert summary["reciprocal_spacing_proxy"]["status"] == "unavailable"


def test_isotropic_weighted_tensor_reports_zero_anisotropy_and_undefined_axis():
    image = np.ones((1, 4))
    qx = np.array([[1.0, 0.0, -1.0, 0.0]])
    qy = np.array([[0.0, 1.0, 0.0, -1.0]])

    result = measure_frame(image, qx, qy, q_edges=[0.0, 1.0], radial_bins=1, angular_bins=4)
    tensor = result["summary"]["in_plane_intensity_tensor"]

    assert tensor["eigenvalues"] == pytest.approx([0.5, 0.5])
    assert tensor["anisotropy"] == 0.0
    assert tensor["principal_axis_deg"] is None
    assert tensor["axis_reason"] == "eigenvalue_degeneracy"


def test_physical_spacing_proxy_uses_only_declared_reciprocal_units():
    image = np.array([[1.0, 2.0]])
    qx = np.array([[1.0, 0.0]])
    qy = np.array([[0.0, 2.0]])

    physical = measure_frame(image, qx, qy, q_unit="A^-1", q_edges=[0.0, 2.0], radial_bins=1)
    proxy = physical["summary"]["reciprocal_spacing_proxy"]
    assert physical["summary"]["q_unit"] == "Å⁻¹"
    assert proxy["unit"] == "Å"
    assert proxy["minimum"] == pytest.approx(np.pi)
    assert proxy["maximum"] == pytest.approx(2.0 * np.pi)
    assert "not a unique structural length" in proxy["definition"]

    pixel = measure_frame(image, qx, qy, q_unit="pixel-q", q_edges=[0.0, 2.0], radial_bins=1)
    assert pixel["summary"]["reciprocal_spacing_proxy"]["minimum"] is None
    assert pixel["summary"]["reciprocal_spacing_proxy"]["unit"] is None


def test_empty_observed_set_keeps_geometry_counts_but_no_fabricated_measurements():
    image = np.array([[1.0, 2.0]])
    qx = np.array([[0.0, 1.0]])
    qy = np.zeros_like(qx)
    valid = np.zeros_like(image, dtype=bool)

    result = measure_frame(image, qx, qy, valid_mask=valid, q_edges=[0.0, 1.0], radial_bins=1, angular_bins=2)

    assert result["summary"]["intensity"]["count"] == 0
    assert result["summary"]["intensity"]["mean"] is None
    assert result["summary"]["maxima"]["intensity"] is None
    assert result["radial_rows"][0]["count"] == 0
    assert result["radial_rows"][0]["candidate_count"] == 2
    assert result["radial_rows"][0]["mean"] is None
    assert result["radial_rows"][0]["sum"] is None
    assert np.isnan(result["polar"]["mean"]).all()
    assert np.isnan(result["polar"]["sum"]).all()
    assert result["polar"]["count"].sum() == 0
    assert result["polar"]["candidate_count"].sum() == 2


def test_constant_q_map_gets_finite_bins_and_invalid_domains_raise_actionable_errors():
    image = np.ones((2, 2))
    qx = np.ones_like(image)
    qy = np.zeros_like(image)
    result = measure_frame(image, qx, qy, radial_bins=4, angular_bins=2)
    assert sum(row["count"] for row in result["radial_rows"]) == 4
    assert np.all(np.diff(result["polar"]["q_edges"]) > 0.0)

    with pytest.raises(ValueError, match="qx shape"):
        measure_frame(image, qx[:1], qy)
    with pytest.raises(ValueError, match="no finite reciprocal-space coordinates"):
        measure_frame(image, np.full_like(qx, np.nan), qy)
    with pytest.raises(ValueError, match="radial_bins must be a positive integer"):
        measure_frame(image, qx, qy, radial_bins=0)
    with pytest.raises(ValueError, match="q_edges must be finite and strictly increasing"):
        measure_frame(image, qx, qy, q_edges=[0.0, 1.0, 1.0])
