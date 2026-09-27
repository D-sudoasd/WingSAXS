from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.local_measurement_page import LocalMeasurementPage


def _frame(value: float = 1.0):
    rows, cols = np.indices((11, 11), dtype=float)
    data = np.full((11, 11), value)
    qx = cols * 0.01
    qy = rows * 0.01
    valid = np.ones(data.shape, dtype=bool)
    return data, qx, qy, valid


def test_page_keeps_only_currently_recomputed_results_after_frame_or_calibration_change(qtbot):
    page = LocalMeasurementPage(language="en")
    qtbot.addWidget(page)
    data, qx, qy, valid = _frame(2.0)
    page.set_data(data, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="scan.edf [frame=0]")
    page._add_point(4, 3)
    assert len(page.measurements) == 1
    original = page.measurements[0]

    updated, qx2, qy2, valid2 = _frame(5.0)
    page.set_data(updated, qx=qx2 * 2.0, qy=qy2 * 2.0, valid_mask=valid2,
                  q_unit="nm^-1", source="scan.edf [frame=0]")

    assert len(page.measurements) == 1
    current = page.measurements[0]
    assert current.intensity == 5.0
    assert current.qx == pytest.approx(2.0 * original.qx)
    assert page.document()["selections"][0]["col"] == 3


def test_page_preserves_valid_mask_argument_and_drops_numeric_results_on_invalidate(qtbot):
    page = LocalMeasurementPage(language="en")
    qtbot.addWidget(page)
    data, qx, qy, valid = _frame()
    valid[4, 4] = False
    before = valid.copy()
    page.set_data(data, qx=qx, qy=qy, valid_mask=valid, q_unit="pixel-q", source="frame-1")
    assert np.array_equal(valid, before)
    page._add_point(4, 4)
    assert not page.measurements
    assert "masked" in page.status.text()

    page._add_point(5, 5)
    assert len(page.measurements) == 1
    page.invalidate("geometry changed")
    assert not page.measurements
    assert page.document()["pending_selections"]
    assert "geometry changed" in page.status.text()


def test_page_line_measurement_retains_mask_gaps_and_exports_json(qtbot, tmp_path):
    page = LocalMeasurementPage(language="en")
    qtbot.addWidget(page)
    data, qx, qy, valid = _frame()
    data[5, :] = np.exp(-((np.arange(11) - 5) / 2.0) ** 2)
    valid[5, 3] = False
    page.set_data(data, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-2")
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("line"))
    page._add_line((0, 5), (10, 5))
    line = page.measurements[0]
    assert np.isnan(line.intensity[3])
    assert not line.valid[3]

    output = page.export_to(tmp_path / "local.json")
    assert output.exists()
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["measurements"][0]["samples"][3]["valid"] is False
    assert document["measurements"][0]["samples"][3]["intensity"] is None


def test_page_restore_recomputes_saved_point_against_loaded_frame(qtbot):
    source = "scan.edf [frame=2]"
    first = LocalMeasurementPage(language="en")
    second = LocalMeasurementPage(language="en")
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    data, qx, qy, valid = _frame(2.0)
    first.set_data(data, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source=source)
    first._add_point(4, 3)
    saved = first.document()

    second.restore_document(saved)
    assert not second.measurements
    changed, qx2, qy2, valid2 = _frame(9.0)
    second.set_data(changed, qx=qx2, qy=qy2, valid_mask=valid2, q_unit="nm^-1", source=source)

    assert len(second.measurements) == 1
    assert second.measurements[0].intensity == 9.0


def test_page_exports_roi_orientation_as_its_own_record(qtbot, tmp_path):
    page = LocalMeasurementPage(language="en")
    qtbot.addWidget(page)
    rows, cols = np.indices((21, 21), dtype=float)
    theta = np.deg2rad(25)
    x, y = cols - 10, rows - 10
    qx = 0.01 * (x * np.cos(theta) - y * np.sin(theta))
    qy = 0.01 * (x * np.sin(theta) + y * np.cos(theta))
    data = np.exp(-(x / 5) ** 2) + 0.01
    valid = np.ones(data.shape, dtype=bool)
    page.set_data(data, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-3")
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("roi"))
    page._add_roi((4, 10), (16, 10))

    assert page.measurements[0].orientation_deg == pytest.approx(25.0, abs=1.0)
    path = page.export_to(tmp_path / "roi.csv")
    text = path.read_text(encoding="utf-8-sig")
    assert "roi_orientation" in text
    assert "non-negative observed intensity" in text


def test_mode_controls_are_visible_and_profile_axis_uses_unscaled_ascii_unit(qtbot):
    page = LocalMeasurementPage(language="zh_CN")
    qtbot.addWidget(page)
    page.resize(1440, 900)
    page.show()
    data, qx, qy, valid = _frame()
    data[5, :] = np.exp(-((np.arange(11) - 5) / 2.0) ** 2)
    page.set_data(data, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-4")
    qtbot.wait(20)

    assert page.patch_radius.isVisible()
    assert page.patch_radius.width() > 0
    assert page.scale_combo.isVisible()
    assert page.scale_combo.width() > 0
    assert not page.roi_half_width.isVisible()

    page.mode_combo.setCurrentIndex(page.mode_combo.findData("roi"))
    assert page.roi_half_width.isVisible()
    assert not page.patch_radius.isVisible()
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("line"))
    page._add_line((0, 5), (10, 5))
    axis = page.profile_plot.getPlotItem().getAxis("bottom")
    assert axis.autoSIPrefix is False
    label = axis.label.toPlainText()
    assert "nm^-1" in label
    assert "mnm" not in label
