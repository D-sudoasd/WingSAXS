from __future__ import annotations

import json
import csv

import numpy as np
import pytest

from butterfly_saxs.local_measurement import (
    estimate_fwhm,
    export_local_measurements_csv,
    export_local_measurements_json,
    extract_line_profile,
    measure_point,
    measure_roi_orientation,
)


def _maps(shape: tuple[int, int]):
    rows, cols = np.indices(shape, dtype=float)
    return cols * 0.01, rows * 0.01


def test_point_reports_physical_q_spacing_and_masked_patch_statistics():
    data = np.arange(25, dtype=float).reshape(5, 5)
    qx, qy = _maps(data.shape)
    valid = np.ones(data.shape, dtype=bool)
    valid[1, 1] = False

    measured = measure_point(data, qx, qy, 2, 2, valid_mask=valid, q_unit="1/nm", patch_radius=1)

    assert (measured.row, measured.col, measured.intensity) == (2, 2, 12.0)
    assert measured.q == pytest.approx(np.hypot(0.02, 0.02))
    assert measured.spacing == pytest.approx(2 * np.pi / measured.q)
    assert measured.spacing_unit == "nm"
    assert measured.patch_count == 8
    assert measured.patch_median == pytest.approx(12.5)


def test_point_does_not_create_length_from_pixel_q_and_rejects_masked_pick():
    data = np.ones((3, 3))
    qx, qy = _maps(data.shape)
    measured = measure_point(data, qx, qy, 1, 1, q_unit="pixel-q")
    assert measured.spacing is None
    assert measured.spacing_unit is None
    valid = np.ones(data.shape, dtype=bool)
    valid[1, 1] = False
    with pytest.raises(ValueError, match="masked"):
        measure_point(data, qx, qy, 1, 1, valid_mask=valid, q_unit="nm^-1")


def test_line_profile_keeps_mask_gaps_and_fwhm_uses_one_observed_segment():
    cols = np.arange(21, dtype=float)
    data = np.tile(np.exp(-((cols - 15) / 2.0) ** 2), (3, 1))
    qx, qy = _maps(data.shape)
    valid = np.ones(data.shape, dtype=bool)
    valid[1, 3] = False

    profile = extract_line_profile(data, qx, qy, (0, 1), (20, 1), valid_mask=valid, q_unit="nm^-1")

    assert profile.cols.tolist() == list(range(21))
    assert not profile.valid[3]
    assert np.isnan(profile.intensity[3])
    assert profile.q_unit == "nm⁻¹"
    assert profile.axis_unit == "nm⁻¹"
    assert profile.fwhm.value is not None
    assert profile.fwhm.length_proxy == pytest.approx(2 * np.pi / profile.fwhm.value)
    assert profile.fwhm.length_unit == "nm"


def test_diagonal_profile_samples_each_raster_pixel_once():
    data = np.ones((8, 9))
    qx, qy = _maps(data.shape)

    profile = extract_line_profile(data, qx, qy, (0, 0), (8, 7), q_unit="unknown")

    pixels = list(zip(profile.rows.tolist(), profile.cols.tolist()))
    assert pixels[0] == (0, 0)
    assert pixels[-1] == (7, 8)
    assert len(pixels) == len(set(pixels))
    assert profile.axis_unit == "pixel"


def test_fwhm_does_not_bridge_a_mask_gap_on_one_side_of_peak():
    x = np.linspace(0, 10, 101)
    y = np.exp(-((x - 5.0) / 0.8) ** 2)
    valid = np.ones(x.shape, dtype=bool)
    valid[56] = False

    result = estimate_fwhm(x, y, valid=valid, x_unit="nm^-1")

    assert result.value is None
    assert result.status == "unavailable"
    assert result.reason == "half_height_not_crossed_on_both_sides"
    assert result.length_proxy is None


def test_fwhm_reports_low_support_without_hiding_a_measured_width():
    result = estimate_fwhm([0, 1, 2, 3, 4], [0, 1, 4, 1, 0], x_unit="pixel")

    assert result.value == pytest.approx(4.0 / 3.0)
    assert result.status == "low_support"
    assert "fewer_than_7_contiguous_samples" in result.flags
    assert result.length_proxy is None


def test_roi_orientation_is_intensity_weighted_and_labeled_as_observed():
    rows, cols = np.indices((21, 21), dtype=float)
    theta = np.deg2rad(30.0)
    x = cols - 10.0
    y = rows - 10.0
    qx = 0.01 * (x * np.cos(theta) - y * np.sin(theta))
    qy = 0.01 * (x * np.sin(theta) + y * np.cos(theta))
    data = 0.01 + 100.0 * np.exp(-((x / 5.0) ** 2))

    result = measure_roi_orientation(data, qx, qy, (4, 10), (16, 10), q_unit="nm^-1", half_width_pixels=2)

    assert result.status == "estimated"
    assert result.n_pixels == 65
    assert result.orientation_deg == pytest.approx(30.0, abs=1.0)
    assert result.eigenvalue_ratio > 2.0
    assert "Observed in-plane intensity axis" in result.interpretation
    assert result.weighting == "non-negative observed intensity"


def test_export_preserves_roi_fields_profile_gaps_and_overwrite_contract(tmp_path):
    data = np.tile(np.array([0.0, 1.0, 3.0, 1.0, 0.0]), (3, 1))
    qx, qy = _maps(data.shape)
    valid = np.ones(data.shape, dtype=bool)
    valid[1, 1] = False
    point = measure_point(data, qx, qy, 1, 0, valid_mask=valid, q_unit="nm^-1", source="frame=1")
    line = extract_line_profile(data, qx, qy, (0, 1), (4, 1), valid_mask=valid, q_unit="nm^-1", source="frame=1")
    roi = measure_roi_orientation(data, qx, qy, (0, 1), (4, 1), valid_mask=valid,
                                  q_unit="nm^-1", half_width_pixels=0.5, source="frame=1")
    records = [point, line, roi]

    json_path = export_local_measurements_json(tmp_path / "measurements.json", records)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["schema"] == "butterfly-saxs/local-measurements-v1"
    assert document["measurements"][2]["kind"] == "roi_orientation"
    assert document["measurements"][1]["samples"][1]["intensity"] is None
    assert document["measurements"][1]["samples"][1]["valid"] is False

    csv_path = export_local_measurements_csv(tmp_path / "measurements.csv", records)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert any(row["row_type"] == "roi_orientation" and row["orientation_deg"] for row in rows)
    assert any(row["row_type"] == "profile_sample" and row["valid"] == "False" and row["intensity"] == "" for row in rows)
    assert any(row["row_type"] == "point" and row["row"] == "1" and row["col"] == "0" for row in rows)
    with pytest.raises(FileExistsError):
        export_local_measurements_csv(csv_path, records)
    export_local_measurements_json(json_path, records, overwrite=True)
