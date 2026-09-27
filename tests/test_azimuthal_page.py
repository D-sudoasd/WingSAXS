from __future__ import annotations

import csv

import numpy as np
import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.azimuthal_page import AzimuthalPage  # noqa: E402


def _frame():
    axis = np.linspace(-1.0, 1.0, 321)
    qx, qy = np.meshgrid(axis, axis)
    angle = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    delta0 = (angle - 358.0 + 180.0) % 360.0 - 180.0
    delta1 = (angle - 145.0 + 180.0) % 360.0 - 180.0
    image = 1.0 + 6.0 * np.exp(-0.5 * (delta0 / 9.0) ** 2)
    image += 4.0 * np.exp(-0.5 * (delta1 / 13.0) ** 2)
    return image, qx, qy


def test_page_accepts_current_calibration_analyzes_and_invalidates(qtbot, tmp_path):
    page = AzimuthalPage(language="en")
    qtbot.addWidget(page)
    image, qx, qy = _frame()
    page.set_data(
        image, qx=qx, qy=qy, valid_mask=np.ones(image.shape, dtype=bool),
        q_unit="nm^-1", source="sample.edf [frame=3; dataset=/entry/data]",
    )
    assert page.analyze_button.isEnabled()
    page.q_min_spin.setValue(0.35)
    page.q_max_spin.setValue(0.85)
    page.min_separation_spin.setValue(30.0)
    fit = page.analyze()
    assert fit is not None and fit.success
    assert len(fit.peaks) == 2
    assert page.export_button.isEnabled()
    profile_path, peak_path = page.export_to(tmp_path / "analysis.csv")
    with profile_path.open(encoding="utf-8-sig", newline="") as stream:
        first = next(csv.DictReader(stream))
    assert first["q_unit"] == "nm^-1"
    assert "frame=3" in first["source"]
    assert peak_path.exists()
    with pytest.raises(FileExistsError):
        page.export_to(profile_path)
    page.min_separation_spin.setValue(31.0)
    assert page.profile is None and page.fit_result is None
    assert not page.export_button.isEnabled()
    page.invalidate()
    assert page.profile is None and page.fit_result is None
    assert not page.analyze_button.isEnabled()
    assert not page.export_button.isEnabled()
    assert page.status_label.text() == page._tr("no_data")


def test_page_settings_document_roundtrip_and_language(qtbot):
    page = AzimuthalPage(language="zh_CN")
    qtbot.addWidget(page)
    saved = page.document()
    saved["settings"]["model"] = "gaussian"
    saved["settings"]["n_bins"] = 180
    page.restore_document(saved)
    assert page.model_combo.currentData() == "gaussian"
    assert page.bins_spin.value() == 180
    page.set_language("en")
    assert page.heading.text() == "Azimuthal peak fitting"


def test_page_does_not_restore_q_window_across_different_units(qtbot):
    page = AzimuthalPage(language="en")
    qtbot.addWidget(page)
    saved = page.document()
    saved["q_unit"] = "pixel-q"
    saved["q_window"] = [0.2, 0.4]
    image, qx, qy = _frame()
    page.set_data(
        image, qx=qx, qy=qy, valid_mask=np.ones(image.shape, dtype=bool),
        q_unit="nm^-1", source="sample.edf",
    )
    current = (page.q_min_spin.value(), page.q_max_spin.value())
    page.restore_document(saved)
    assert (page.q_min_spin.value(), page.q_max_spin.value()) == current


def test_preview_uses_calibrated_q_map_contours_and_marks_mask(qtbot, monkeypatch):
    page = AzimuthalPage(language="en")
    qtbot.addWidget(page)
    image, qx, qy = _frame()
    yy, xx = np.indices(image.shape)
    # Introduce a smooth detector-coordinate distortion: constant-q contours
    # are not circles in detector pixel coordinates.
    qx = qx * (1.0 + 0.12 * np.sin(yy / 55.0))
    qy = qy * (1.0 + 0.07 * np.cos(xx / 61.0))
    valid = np.ones(image.shape, dtype=bool)
    valid[160, 260] = False
    page.set_data(image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="warped-map")

    observed: dict[str, np.ndarray] = {}
    original_contour = page.image_axes.contour

    def record_contour(x_grid, y_grid, q_values, **kwargs):
        observed["q"] = np.asarray(q_values).copy()
        return original_contour(x_grid, y_grid, q_values, **kwargs)

    monkeypatch.setattr(page.image_axes, "contour", record_contour)
    page.q_min_spin.setValue(0.45)
    page.q_max_spin.setValue(0.8)
    assert "q" in observed
    stride = max(1, int(np.ceil(max(image.shape) / 600.0)))
    np.testing.assert_allclose(observed["q"], np.hypot(qx, qy)[::stride, ::stride])
    assert len(page.image_axes.images) == 2  # detector image plus visible invalid-pixel overlay
    alpha = page.image_axes.images[0].get_alpha()
    assert alpha[160, 160] < alpha[160, 240]
