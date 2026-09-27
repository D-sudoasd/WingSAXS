from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.density2d_page import Density2DPage


@pytest.fixture
def page(qtbot):
    widget = Density2DPage(language="zh_CN")
    qtbot.addWidget(widget)
    yield widget


def _frame(size: int = 181):
    axis = np.linspace(-0.2, 0.2, size)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = 2.0 + 25.0 / (1.0 + np.square(q * 18.0))
    valid = np.ones_like(image, dtype=bool)
    valid[20:40, 70:90] = False
    return image, qx, qy, valid


def test_density_page_shows_2d_selection_fits_and_restorable_settings(page):
    image, qx, qy, valid = _frame()
    page.resize(1280, 900)
    page.show()
    page.layout().activate()
    qt_widgets = [
        page._labels["models"],
        page.power_law_check,
        page.ornstein_zernike_check,
    ]
    numeric_widgets = [
        page.q_min_spin,
        page.q_max_spin,
        page.azimuth_spin,
        page.half_width_spin,
        page.n_q_spin,
    ]
    assert all(
        not model.geometry().intersects(control.geometry())
        for model in qt_widgets
        for control in numeric_widgets
    )

    page.set_data(
        image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-8"
    )
    page.q_min_spin.setValue(0.005)
    page.q_max_spin.setValue(0.18)
    page.n_q_spin.setValue(48)
    page.canvas.draw()

    assert len(page.image_axes.images) == 2
    assert page.profile is None
    fits = page.analyze()
    assert fits is not None and len(fits) == 2
    assert page.profile is not None and page.profile.source == "frame-8"
    assert all(fit.success for fit in fits)
    assert page.export_button.isEnabled()

    snapshot = page.document()
    page.azimuth_spin.setValue(90.0)
    page.restore_document(snapshot)
    assert page.azimuth_spin.value() == 0.0
    assert page.n_q_spin.value() == 48
    assert page.document()["settings"]["models"] == ["power_law", "ornstein_zernike"]

    page.set_language("en")
    assert page.heading.text() == "2-D low-q sector analysis"
    page.invalidate("Calibration changed")
    assert page.data is None
    assert not page.export_button.isEnabled()
    assert page.status_label.text() == "Calibration changed"


@pytest.mark.parametrize(
    ("control_name", "new_value"),
    [
        ("q_min_spin", 0.01),
        ("q_max_spin", 0.17),
        ("azimuth_spin", 5.0),
        ("half_width_spin", 20.0),
        ("n_q_spin", 60),
        ("power_law_check", False),
        ("ornstein_zernike_check", False),
    ],
)
def test_density_page_invalidates_results_after_analysis_setting_changes(
    page, tmp_path, control_name, new_value
):
    image, qx, qy, valid = _frame()
    page.set_data(image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-8")
    page.q_min_spin.setValue(0.005)
    page.q_max_spin.setValue(0.18)
    assert page.analyze() is not None
    assert page.profile is not None
    assert page.fits
    assert page.export_button.isEnabled()

    control = getattr(page, control_name)
    if control_name.endswith("_check"):
        control.setChecked(new_value)
    else:
        control.setValue(new_value)

    assert page.data is not None
    assert page.profile is None
    assert page.fits == ()
    assert not page.export_button.isEnabled()
    with pytest.raises(ValueError, match="analyzed 2-D sector profile is required"):
        page.export_to(tmp_path / "stale.csv")


def _density_document(page):
    image, qx, qy, valid = _frame()
    page.set_data(image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-8")
    page.q_min_spin.setValue(0.005)
    page.q_max_spin.setValue(0.18)
    page.azimuth_spin.setValue(37.0)
    page.n_q_spin.setValue(72)
    return page.document()


def test_density_page_restore_does_not_apply_q_window_from_different_unit(page, qtbot):
    document = _density_document(page)
    image, qx, qy, valid = _frame()
    target = Density2DPage(language="zh_CN")
    qtbot.addWidget(target)
    target.set_data(image, qx=qx, qy=qy, valid_mask=valid, q_unit="pixel-q", source="frame-8")
    target_q_window = target.q_min_spin.value(), target.q_max_spin.value()

    target.restore_document(document)

    assert (target.q_min_spin.value(), target.q_max_spin.value()) == target_q_window
    assert target.azimuth_spin.value() == 37.0
    assert target.n_q_spin.value() == 72
    assert target.q_unit == "pixel-q"


def test_density_page_restore_before_data_checks_saved_q_unit_when_data_arrives(page, qtbot):
    document = _density_document(page)
    target = Density2DPage(language="zh_CN")
    qtbot.addWidget(target)
    target.restore_document(document)
    image, qx, qy, valid = _frame()

    target.set_data(image, qx=qx, qy=qy, valid_mask=valid, q_unit="pixel-q", source="frame-8")

    default_q_window = target.q_min_spin.value(), target.q_max_spin.value()
    assert default_q_window != tuple(document["settings"]["q_window"])
    assert target.azimuth_spin.value() == 37.0
    assert target.n_q_spin.value() == 72
    assert target._pending_q_window is None
