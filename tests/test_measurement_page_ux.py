"""Headless layout and background-job regressions for measurement pages."""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui import azimuthal_page, density2d_page
from butterfly_saxs.ui.local_measurement_page import LocalMeasurementPage
from butterfly_saxs.ui.qt_compat import QtCore, QtWidgets
from butterfly_saxs.ui.theme import apply_theme


def _frame():
    axis = np.linspace(-0.2, 0.2, 121)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    angle = np.arctan2(qy, qx)
    image = 2.0 + 25.0 / (1.0 + (q * 18.0) ** 2) * (1.0 + 0.5 * np.cos(2.0 * angle))
    valid = np.ones(image.shape, dtype=bool)
    valid[15:25, 40:50] = False
    return image, qx, qy, valid


def _load(page):
    image, qx, qy, valid = _frame()
    page.set_data(image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-3")
    if hasattr(page, "q_min_spin"):
        page.q_min_spin.setValue(0.015)
        page.q_max_spin.setValue(0.18)
    return image, qx, qy, valid


@pytest.mark.parametrize("language", ["en", "zh_CN"])
@pytest.mark.parametrize("page_class,width", [
    (LocalMeasurementPage, 900),
    (azimuthal_page.AzimuthalPage, 980),
    (density2d_page.Density2DPage, 980),
])
def test_measurement_controls_fit_narrow_window(qtbot, page_class, width, language):
    page = page_class(language=language)
    qtbot.addWidget(page)
    apply_theme(page)
    _load(page)
    page.resize(width, 680)
    page.show()
    QtWidgets.QApplication.processEvents()

    assert page.width() == width
    assert page.height() == 680
    controls = page.findChildren(QtWidgets.QAbstractSpinBox) + page.findChildren(QtWidgets.QComboBox)
    for control in controls:
        if not control.isVisible():
            continue
        position = control.mapTo(page, QtCore.QPoint(0, 0))
        assert 0 <= position.x() < page.width()
        assert position.x() + control.width() <= page.width()
        assert position.y() + control.height() <= page.height()
        assert control.accessibleName()
    if isinstance(page, LocalMeasurementPage):
        assert page.image_plot.width() >= 360
        assert page.profile_plot.width() >= 290
        assert page.profile_plot.height() >= 170
    else:
        assert page.canvas.height() >= 240
        assert page.analyze_button.property("role") == "primary"
        assert page.status_label.property("state") == "ready"


@pytest.mark.parametrize("page_class", [azimuthal_page.AzimuthalPage, density2d_page.Density2DPage])
@pytest.mark.parametrize("language", ["en", "zh_CN"])
def test_embedded_analysis_page_preserves_plot_height_and_scrolls_all_settings(qtbot, page_class, language):
    page = page_class(language=language)
    qtbot.addWidget(page)
    apply_theme(page)
    _load(page)
    page.resize(980, 545)
    page.show()
    QtWidgets.QApplication.processEvents()
    assert page.height() == 545
    assert page.canvas.height() >= 240
    controls = page.settings_group.findChildren(QtWidgets.QAbstractSpinBox)
    controls += page.settings_group.findChildren(QtWidgets.QComboBox)
    controls += page.settings_group.findChildren(QtWidgets.QCheckBox)
    for control in controls:
        # Scroll the whole field into view; ensureWidgetVisible on a spin box
        # may expose only its embedded editor's cursor rectangle.
        top = control.mapTo(page.settings_group, QtCore.QPoint(0, 0)).y()
        page.settings_scroll.verticalScrollBar().setValue(top)
        QtWidgets.QApplication.processEvents()
        position = control.mapTo(page.settings_scroll.viewport(), QtCore.QPoint(0, 0))
        rect = QtCore.QRect(position, control.size())
        assert page.settings_scroll.viewport().rect().contains(rect)


@pytest.fixture(params=[azimuthal_page, density2d_page], ids=["azimuthal", "density2d"])
def threaded_page(request, qtbot):
    module = request.param
    page_class = module.AzimuthalPage if module is azimuthal_page else module.Density2DPage
    page = page_class(language="en")
    qtbot.addWidget(page)
    _load(page)
    return module, page


def _hold_job(monkeypatch, module, page, *, failure=False):
    started = threading.Event()
    release = threading.Event()
    original = module._analyze_frame
    captured = {}

    def delayed(parameters, payload=None):
        captured["thread"] = QtCore.QThread.currentThread()
        captured["inputs"] = payload or parameters
        started.set()
        if not release.wait(5.0):
            raise RuntimeError("test did not release worker")
        if failure:
            raise ValueError("invalid selected q window")
        if module is azimuthal_page:
            return original(parameters, payload)
        return original(parameters)

    monkeypatch.setattr(module, "_analyze_frame", delayed)
    return started, release, captured


def test_button_runs_snapshot_off_gui_thread_and_publishes_on_gui_thread(threaded_page, qtbot, monkeypatch):
    module, page = threaded_page
    started, release, captured = _hold_job(monkeypatch, module, page)
    frame_before = page.data.copy()
    callback_threads = []
    page.profileChanged.connect(lambda *_args: callback_threads.append(QtCore.QThread.currentThread()))
    try:
        page.analyze_button.click()
        qtbot.waitUntil(started.is_set)
        assert page.jobs_running()
        assert not page.analyze_button.isEnabled()
        assert not page.export_button.isEnabled()
        assert page.status_label.property("state") == "running"
        assert captured["thread"] != page.thread()
        # A queued event must still run while the numerical task is blocked.
        responsive = []
        QtCore.QTimer.singleShot(0, lambda: responsive.append(True))
        qtbot.waitUntil(lambda: bool(responsive))
        for key, array in captured["inputs"].items():
            if isinstance(array, np.ndarray):
                assert not np.shares_memory(array, getattr(page, key))
        page.data[:] = -100.0
        np.testing.assert_array_equal(captured["inputs"]["data"], frame_before)
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=5000)
        assert page.profile is not None
        assert page.profile.source == "frame-3"
        assert page.export_button.isEnabled()
        assert callback_threads == [page.thread()]
        assert page.status_label.property("state") in ("complete", "warning")
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=6000)


@pytest.mark.parametrize("change", ["settings", "frame", "mask", "invalidate", "shutdown"])
def test_background_result_is_discarded_after_source_or_settings_change(
    threaded_page, qtbot, monkeypatch, change,
):
    module, page = threaded_page
    started, release, _captured = _hold_job(monkeypatch, module, page)
    emitted = []
    page.profileChanged.connect(lambda *_args: emitted.append(True))
    try:
        page.start_analysis()
        qtbot.waitUntil(started.is_set)
        if change == "settings":
            page.q_min_spin.setValue(0.02)
        elif change == "frame":
            image, qx, qy, valid = _frame()
            page.set_data(image * 2.0, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-4")
        elif change == "mask":
            image, qx, qy, valid = _frame()
            valid[:, :80] = False
            page.set_data(image, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-3")
        elif change == "invalidate":
            page.invalidate("Calibration changed")
        else:
            page.shutdown()
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=5000)
        assert page.profile is None
        assert not page.export_button.isEnabled()
        assert not emitted
        if change == "invalidate":
            assert page.status_label.text() == "Calibration changed"
            assert not page.analyze_button.isEnabled()
        elif change != "shutdown":
            assert page.analyze_button.isEnabled()
            assert page.status_label.property("state") == "ready"
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=6000)


def test_background_failure_retains_no_stale_exports_and_enables_retry(threaded_page, qtbot, monkeypatch):
    module, page = threaded_page
    started, release, _captured = _hold_job(monkeypatch, module, page, failure=True)
    try:
        page.start_analysis()
        qtbot.waitUntil(started.is_set)
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=5000)
        assert page.profile is None
        assert not page.export_button.isEnabled()
        assert page.analyze_button.isEnabled()
        assert page.status_label.property("state") == "error"
        assert "invalid selected q window" in page.status_label.text()
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=6000)


def test_stale_worker_error_is_discarded(threaded_page, qtbot, monkeypatch):
    module, page = threaded_page
    started, release, _captured = _hold_job(monkeypatch, module, page, failure=True)
    try:
        page.start_analysis()
        qtbot.waitUntil(started.is_set)
        page.q_min_spin.setValue(0.02)
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=5000)
        assert page.status_label.property("state") == "ready"
        assert "invalid selected q window" not in page.status_label.text()
        assert page.profile is None
        assert not page.export_button.isEnabled()
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=6000)


def test_standalone_close_waits_asynchronously_and_discards_worker_result(threaded_page, qtbot, monkeypatch):
    module, page = threaded_page
    started, release, _captured = _hold_job(monkeypatch, module, page)
    emitted = []
    page.profileChanged.connect(lambda *_args: emitted.append(True))
    page.show()
    try:
        page.start_analysis()
        qtbot.waitUntil(started.is_set)
        assert not page.close()
        assert page.isVisible()
        release.set()
        qtbot.waitUntil(lambda: not page.isVisible(), timeout=5000)
        assert not page.jobs_running()
        assert not emitted
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=6000)


def test_hidden_frame_updates_defer_drawing_until_page_is_shown(threaded_page, qtbot):
    _module, page = threaded_page
    page.resize(980, 680)
    draws = []
    page.canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    page.show()
    qtbot.waitUntil(lambda: bool(draws))
    page.hide()
    QtWidgets.QApplication.processEvents()
    draws.clear()
    image, qx, qy, valid = _frame()
    page.set_data(image * 2.0, qx=qx, qy=qy, valid_mask=valid, q_unit="nm^-1", source="frame-4")
    QtWidgets.QApplication.processEvents()
    assert not draws
    page.show()
    qtbot.waitUntil(lambda: bool(draws))
    assert page.image_axes.images
