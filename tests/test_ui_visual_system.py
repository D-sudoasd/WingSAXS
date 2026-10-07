from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtGui, QtWidgets

from butterfly_saxs.ui.theme import COLORS, apply_theme
from butterfly_saxs.ui.views import PatternView, ViewGrid


def test_theme_is_window_scoped_and_preserves_application_defaults(qtbot, qapp):
    application_palette = QtGui.QPalette(qapp.palette())
    application_font = QtGui.QFont(qapp.font())
    application_style = qapp.styleSheet()
    window = QtWidgets.QMainWindow()
    other = QtWidgets.QWidget()
    qtbot.addWidget(window)
    qtbot.addWidget(other)
    other_palette = QtGui.QPalette(other.palette())
    central = QtWidgets.QWidget()
    window.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)
    action = QtWidgets.QPushButton("Fit")
    action.setProperty("role", "primary")
    layout.addWidget(action)

    apply_theme(window)
    window.show()

    assert window.palette().color(QtGui.QPalette.ColorRole.Window).name() == COLORS["background"]
    assert action.palette().color(QtGui.QPalette.ColorRole.Button).name() == COLORS["accent"]
    assert action.palette().color(QtGui.QPalette.ColorRole.ButtonText).name() == COLORS["surface"]
    action.setDisabled(True)
    qtbot.wait(1)
    assert action.palette().color(
        QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText,
    ).name() == "#7b8798"
    assert qapp.palette() == application_palette
    assert qapp.font() == application_font
    assert qapp.styleSheet() == application_style
    assert other.palette() == other_palette


def test_image_cards_explain_empty_states_and_restore_them_after_clear(qtbot):
    view = PatternView("Model", language="en")
    qtbot.addWidget(view)
    assert "intensity fit" in view.state_label.text()
    assert view._content_layout.currentWidget() is view.state_label

    view.set_language("zh_CN")
    assert "强度拟合" in view.state_label.text()
    assert view.subtitle_label.text() == "拟合强度"
    view.set_image(np.ones((5, 7)))
    assert view._content_layout.currentWidget() is view.plot
    view.clear_image("仅几何结果")
    assert view._content_layout.currentWidget() is view.state_label
    view.set_language("en")
    assert view.state_label.text() == "仅几何结果"
    view.clear_image()
    assert "intensity fit" in view.state_label.text()


def test_card_styling_preserves_signed_residual_masks_and_q_units(qtbot):
    grid = ViewGrid(language="en")
    qtbot.addWidget(grid)
    apply_theme(grid)
    observed = np.arange(24, dtype=float).reshape(4, 6)
    model = observed + 2.0
    residual = observed - model
    residual[1, 1] = 3.0
    mask = np.zeros_like(observed, dtype=bool)
    mask[0, 0] = True
    original = observed.copy()
    grid.set_images(
        observed, model, residual, external_mask=mask,
        q_extent=(-0.3, 0.3, -0.2, 0.2), q_unit="nm^-1",
    )
    grid.set_display_settings("log1p")

    np.testing.assert_array_equal(observed, original)
    assert np.isnan(grid.observed.raw_image_data[0, 0])
    assert grid.residual.image_data[1, 1] > 0
    assert grid.residual.image_data[1, 2] < 0
    low, high = grid.residual.image_item.getLevels()
    assert low == pytest.approx(-high)
    assert grid.observed.image_item.getLevels() == pytest.approx(grid.model.image_item.getLevels())
    assert grid.overlay.plot.getAxis("bottom").labelText == "qx (nm^-1)"
    assert grid.overlay.image_extent == (-0.3, 0.3, -0.2, 0.2)
    assert grid.observed.plot.backgroundBrush().color().name() == COLORS["surface"]
    grid.clear_fit()
    assert grid.model._content_layout.currentWidget() is grid.model.state_label
    assert grid.residual._content_layout.currentWidget() is grid.residual.state_label
    assert grid.observed._content_layout.currentWidget() is grid.observed.plot


def test_overlay_without_detector_background_remains_visible(qtbot):
    view = PatternView("Overlay", language="en")
    qtbot.addWidget(view)
    view.set_overlay([{"qx": 0.1, "qy": 0.2}], [{"a": 0.2, "b": 0.1}])
    assert view._content_layout.currentWidget() is view.plot
    view.set_language("zh_CN")
    assert view._content_layout.currentWidget() is view.plot
    view.set_overlay([], [])
    assert view._content_layout.currentWidget() is view.state_label
