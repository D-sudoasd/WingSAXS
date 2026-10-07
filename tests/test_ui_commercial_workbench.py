from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from butterfly_saxs.ui import MainWindow


def test_workspace_context_and_actions_follow_current_input(qtbot):
    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    assert "Open a 2D SAXS image" in window.workspace_context.text()
    assert not window.preview_button.isEnabled()
    assert not window.optimize_button.isEnabled()
    window.set_observed_data([[1.0, 2.0], [3.0, 4.0]])
    assert "2 × 2 px" in window.workspace_context.text()
    observed = np.ones((8, 12))
    window.set_observed_data(observed)
    assert "12 × 8 px" in window.workspace_context.text()
    assert "pixel-q" in window.workspace_context.text()
    assert window.preview_button.isEnabled()
    yy, xx = np.indices(observed.shape)
    window.set_observed_data(observed, qmap={"qx": xx / 10, "qy": yy / 10, "q_unit": "nm^-1"})
    assert "nm^-1" in window.workspace_context.text()
    window.set_language("zh_CN", persist=False)
    assert "内存帧" in window.workspace_context.text()
    window.close()


def test_narrow_butterfly_canvas_keeps_lists_available_without_obscuring_image(qtbot):
    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.resize(1440, 900)
    window.show()
    page = window.butterfly_workbench
    qtbot.waitUntil(lambda: page.frame_rail.isVisible())
    window.resize(980, 680)
    qtbot.waitUntil(lambda: not page.frame_rail.isVisible())
    qtbot.waitUntil(lambda: page.qspace.width() > 540)
    hidden_width = page.qspace.width()
    page.rail_toggle.click()
    assert page.frame_rail.isVisible()
    qtbot.waitUntil(lambda: page.qspace.width() < hidden_width)
    page.rail_toggle.click()
    assert not page.frame_rail.isVisible()
    window.close()


def test_measurement_tables_and_batch_options_remain_reachable_at_supported_minimum(qtbot):
    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.resize(980, 680)
    window.show()
    window.pages.setCurrentWidget(window.measurements_page)
    tabs = window.measurement_table_tabs
    assert tabs.count() == 4
    for index, table in enumerate((window.lobe_table, window.ridge_table, window.ellipse_table, window.radial_table)):
        tabs.setCurrentIndex(index)
        qtbot.waitUntil(table.isVisible)
        assert table.viewport().width() > 440
    window.pages.setCurrentWidget(window.batch_page)
    assert not window.parameters_dock.isVisible()
    scroll = window.batch_options_scroll
    scroll.ensureWidgetVisible(window.batch_output_edit)
    qtbot.wait(1)
    field_rect = window.batch_output_edit.rect()
    position = window.batch_output_edit.mapTo(scroll.viewport(), field_rect.center())
    assert scroll.viewport().rect().contains(position)
    assert window.batch_table.height() > 130
    assert window.open_image_action.shortcut().toString() == "Ctrl+O"
    window.close()


def test_butterfly_controls_fit_narrow_scroll_viewport_in_both_languages(qtbot):
    from PySide6 import QtWidgets

    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.resize(980, 680)
    window.show()
    scroll = window.butterfly_workbench.findChild(QtWidgets.QScrollArea, "butterflyControlsScroll")
    panel = scroll.widget()
    for language in ("en", "zh_CN"):
        window.set_language(language, persist=False)
        qtbot.waitUntil(lambda: panel.width() <= scroll.viewport().width())
        for widget in panel.findChildren(QtWidgets.QWidget):
            if not widget.isVisible() or not isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)):
                continue
            left = widget.mapTo(scroll.viewport(), widget.rect().topLeft()).x()
            right = widget.mapTo(scroll.viewport(), widget.rect().bottomRight()).x()
            assert 0 <= left <= right < scroll.viewport().width(), widget.objectName()
    window.close()


def test_parent_close_waits_for_measurement_worker_and_discards_its_result(qtbot, monkeypatch):
    from butterfly_saxs.ui import azimuthal_page

    started = threading.Event()
    release = threading.Event()

    def analyze_frame(*_args):
        started.set()
        release.wait(10)
        return None, None

    monkeypatch.setattr(azimuthal_page, "_analyze_frame", analyze_frame)
    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.show()
    yy, xx = np.indices((8, 8), dtype=float)
    window.set_observed_data(np.ones((8, 8)), qmap={"qx": xx, "qy": yy, "q_unit": "pixel-q"})
    page = window.azimuthal_page
    try:
        page.start_analysis()
        qtbot.waitUntil(started.is_set)
        assert not window.close()
        assert window.isVisible()
        assert page._closed
        release.set()
        qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)
        assert not page.jobs_running()
        assert page.fit_result is None
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page.jobs_running(), timeout=5000)
        window.close()


def test_failed_measurement_retains_specific_diagnostics_in_visible_help(qtbot):
    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    reason = "analysis_validation_failed:q domain is too close to the origin"
    window._last_result = {"metrics": {"flags": [reason], "success": False}}
    window._set_busy(False, "measure_geometry", result_ok=False)
    assert "result_failed" in window.flags_label.text()
    assert reason in window.flags_label.text()
    assert reason in window.flags_label.toolTip()
    window.close()
