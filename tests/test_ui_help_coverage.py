"""Exercise real pages, optional plot controls and on-demand dialog help."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets

from butterfly_saxs.ui import MainWindow
from butterfly_saxs.ui.figure_export_dialog import FigureExportDialog
from butterfly_saxs.ui.legacy_saxs_dialog import LegacySaxsDialog
from butterfly_saxs.ui.help import apply_help


def _controls(root):
    seen = set()
    kinds = (QtWidgets.QAbstractButton, QtWidgets.QLineEdit, QtWidgets.QComboBox,
             QtWidgets.QAbstractSpinBox, QtWidgets.QSlider, QtWidgets.QAbstractItemView,
             QtWidgets.QPlainTextEdit, QtWidgets.QGraphicsView)
    for owner in [root, *root.findChildren(QtWidgets.QWidget)]:
        if not type(owner).__module__.startswith("butterfly_saxs."):
            continue
        for name, control in vars(owner).items():
            if isinstance(control, kinds) and id(control) not in seen:
                seen.add(id(control))
                yield f"{type(owner).__name__}.{name}", control


def _assert_help(root):
    controls = list(_controls(root))
    for name, control in controls:
        assert control.toolTip().strip(), name
        assert control.whatsThis().strip(), name
        if isinstance(control, QtWidgets.QComboBox):
            for index in range(control.count()):
                assert control.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole), (name, index)
        if isinstance(control, QtWidgets.QAbstractSpinBox):
            assert control.lineEdit().toolTip() == control.toolTip(), name
    for tabs in root.findChildren(QtWidgets.QTabWidget):
        for index in range(tabs.count()):
            assert tabs.tabToolTip(index).strip(), (tabs.objectName(), index)
    return controls


def test_all_workspace_pages_and_options_have_translated_help_without_data_changes(qtbot):
    window = MainWindow(engine=object(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    before = window.project_to_dict()
    generation = window._generation.current
    chinese = {name: control.toolTip() for name, control in _assert_help(window)}
    window.set_language("en", persist=False)
    english = {name: control.toolTip() for name, control in _assert_help(window)}
    assert chinese.keys() == english.keys()
    assert all(chinese[name] != english[name] for name in chinese)
    assert window.project_to_dict() == before
    assert window._generation.current == generation
    assert "Immediately process" in window.butterfly_workbench.apply_batch_button.toolTip()
    window.close()


@pytest.mark.parametrize("dialog_type", [FigureExportDialog, LegacySaxsDialog])
def test_standalone_dialog_help_refreshes_with_language(qtbot, dialog_type):
    dialog = dialog_type(language="zh_CN")
    qtbot.addWidget(dialog)
    chinese = {name: control.toolTip() for name, control in _assert_help(dialog)}
    dialog.set_language("en")
    english = {name: control.toolTip() for name, control in _assert_help(dialog)}
    assert all(chinese[name] != english[name] for name in chinese)
    dialog.close()


def test_hover_on_embedded_spin_editor_shows_actual_help(qtbot):
    window = MainWindow(engine=object(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    window.show()
    field = window.butterfly_workbench.reference_axis_spin
    editor = field.lineEdit()
    position = editor.rect().center()
    event = QtGui.QHelpEvent(QtCore.QEvent.Type.ToolTip, position, editor.mapToGlobal(position))
    QtWidgets.QApplication.sendEvent(editor, event)
    assert QtWidgets.QToolTip.text() == field.toolTip()
    QtWidgets.QToolTip.hideText()
    window.close()


@pytest.mark.parametrize("method", ["numeric", "sources"])
def test_on_demand_dialogs_have_field_option_and_close_help(qtbot, monkeypatch, method):
    window = MainWindow(engine=object(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    checked = []

    def inspect(dialog):
        kinds = (QtWidgets.QAbstractButton, QtWidgets.QLineEdit, QtWidgets.QComboBox,
                 QtWidgets.QPlainTextEdit, QtWidgets.QTreeWidget)
        fields = [w for w in dialog.findChildren(QtWidgets.QWidget)
                  if isinstance(w, kinds) and w.objectName() and not w.objectName().startswith("qt_")]
        assert fields
        for field in fields:
            assert field.toolTip().strip(), field.objectName()
            if isinstance(field, QtWidgets.QComboBox):
                for i in range(field.count()):
                    assert field.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
        checked.append(dialog.objectName() or "sources")
        return int(QtWidgets.QDialog.DialogCode.Rejected)

    monkeypatch.setattr(QtWidgets.QDialog, "exec", inspect)
    if method == "numeric":
        window.butterfly_workbench._open_numeric_edit_dialog()
    else:
        window.lamellar_page.show_sources()
    assert checked
    window.close()


def test_plot_zoom_icon_and_embedded_context_controls_have_help(qtbot):
    window = MainWindow(engine=object(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    plot = window.views.observed.plot
    if plot is None:
        pytest.skip("optional pyqtgraph unavailable")
    item = plot.getPlotItem()
    assert item.autoBtn.toolTip()
    for owner in (item.ctrl, *item.vb.menu.ctrl):
        for name, field in vars(owner).items():
            if isinstance(field, (QtWidgets.QAbstractButton, QtWidgets.QLineEdit,
                                  QtWidgets.QAbstractSpinBox, QtWidgets.QSlider, QtWidgets.QComboBox)):
                assert field.toolTip(), name
    first = item.autoBtn.toolTip()
    apply_help(window, "en")
    assert item.autoBtn.toolTip() != first
    window.close()


def test_evaluation_help_tracks_repeat_count_without_waiting_for_language_change(qtbot):
    window = MainWindow(engine=object(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    workbench = window.butterfly_workbench
    workbench.evaluation_resamples_combo.setCurrentIndex(workbench.evaluation_resamples_combo.findData(128))
    assert "128" in workbench.evaluate_button.toolTip()
    workbench.evaluation_resamples_combo.setCurrentIndex(workbench.evaluation_resamples_combo.findData(0))
    assert "不重复模拟" in workbench.evaluate_button.toolTip()
    window.close()


def test_help_refresh_preserves_the_actual_failure_reason(qtbot):
    window = MainWindow(engine=object(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    window._displayed_flags_text = "mask_shape_mismatch"
    window._render_metric_labels()
    apply_help(window, "zh_CN")
    assert "mask_shape_mismatch" in window.flags_label.toolTip()
    assert "需要检查的原因" in window.flags_label.toolTip()
    window.set_language("en", persist=False)
    assert "mask_shape_mismatch" in window.flags_label.toolTip()
    window.close()


def test_profile_table_dynamic_description_coexists_with_generic_help(qtbot):
    window = MainWindow(engine=object(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    workbench = window.butterfly_workbench
    for panel in (workbench.normal_profile, workbench.ellipse_diagnostic,
                  workbench.peak_angular_profile, workbench.peak_radial_profile):
        panel.set_series([0.1, 0.2], {"raw": [1.0, 2.0]},
                         x_label="q (1/nm)", y_label="intensity")
        description = panel.table.accessibleDescription()
        assert "q (1/nm)" in description
        apply_help(workbench, "en")
        assert panel.table.accessibleDescription() == description
        assert panel.table.toolTip()
        assert panel.table.whatsThis() == panel.table.toolTip()
        panel.clear()
        empty_description = panel.table.accessibleDescription()
        apply_help(workbench, "en")
        assert panel.table.accessibleDescription() == empty_description
        assert empty_description == "No profile values are available."
    window.close()
