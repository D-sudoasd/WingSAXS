from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import threading

import numpy as np
import pytest

from butterfly_saxs.density2d import Density2DFit, Density2DProfile
from butterfly_saxs.lamellar_models import LamellarScene
from butterfly_saxs.ui.butterfly_figure_export import run_butterfly_figure_export
from butterfly_saxs.ui.project_document import ProjectDocumentController
from butterfly_saxs.ui.qt_compat import QT_AVAILABLE, QtCore, QtWidgets, is_qt_valid

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Windows launchers (codepages CP936/CP437, pyw fallback, deduplicated PYTHONPATH)
# ---------------------------------------------------------------------------


def test_windows_launchers_content_and_crlf() -> None:
    wrapper = (ROOT / "启动_LamellarSAXS2D.cmd").read_bytes()
    main_cmd = (ROOT / "启动_WingSAXS.cmd").read_bytes()
    for content in (wrapper, main_cmd):
        assert b"\n" not in content.replace(b"\r\n", b"")
        assert b"\r" not in content.replace(b"\r\n", b"")

    wrapper_text = wrapper.decode("utf-8")
    assert "chcp 65001" in wrapper_text
    assert wrapper_text.index("chcp 65001") < wrapper_text.index("启动_WingSAXS.cmd")

    main_text = main_cmd.decode("utf-8")
    assert main_text.count("PYTHONPATH=%PROJECT_DIR%src") == 2  # if / else branches only
    assert "PYTHONW_ARGS=pyw" in main_text


@pytest.mark.skipif(os.name != "nt", reason="Windows CMD codepage integration test")
@pytest.mark.parametrize("codepage", [936, 437])
def test_lamellar_cmd_wrapper_resolves_target_under_legacy_codepages(codepage: int) -> None:
    cmd = f'cmd.exe /d /c "chcp {codepage} >nul & call 启动_LamellarSAXS2D.cmd --check"'
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    combined = (result.stdout + result.stderr).lower()
    assert result.returncode == 0, combined
    assert "environment check passed" in combined


# ---------------------------------------------------------------------------
# 2. Pure Python / cross-seam contracts (ProjectDocumentController, journal_style)
# ---------------------------------------------------------------------------


def test_project_document_controller_overwrite_guard(tmp_path: Path) -> None:
    controller = ProjectDocumentController(
        snapshot=lambda: {},
        restore=lambda _: None,
        apply=lambda _doc, _target: None,
    )
    target = tmp_path / "proj.json"
    controller.save(target, {"version": 1}, overwrite=False)
    with pytest.raises(FileExistsError):
        controller.save(target, {"version": 2}, overwrite=False)
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    controller.save(target, {"version": 2}, overwrite=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 2}


def test_butterfly_figure_export_forwards_journal_style_conditionally(monkeypatch, tmp_path: Path) -> None:
    import butterfly_saxs.butterfly_figure as bf_mod

    received: list[dict[str, object]] = []

    def stub_with_style(target, *, journal_style=None, **kwargs):
        received.append({"target": target, "journal_style": journal_style, **kwargs})
        return {"index": Path(target) / "index.html"}

    monkeypatch.setattr(bf_mod, "export_butterfly_figure", stub_with_style)
    base_payload = {
        "target": tmp_path / "out",
        "observed": np.ones((4, 4)),
        "qx": np.zeros((4, 4)),
        "qy": np.zeros((4, 4)),
        "result": {},
        "width_mm": 89.0,
        "dpi": 300,
        "cancel_event": threading.Event(),
        "journal_style": "nature",
    }
    run_butterfly_figure_export(payload=base_payload)
    assert received[-1]["journal_style"] == "nature"

    received.clear()

    def stub_without_style(
        target,
        *,
        observed,
        qx,
        qy,
        valid_mask=None,
        result=None,
        q_unit="unknown",
        context=None,
        display_scale="log1p",
        width_mm=89.0,
        dpi=600,
        cancel_event=None,
        progress=None,
    ):
        del observed, qx, qy, valid_mask, result, q_unit, context, display_scale, width_mm, dpi, cancel_event, progress
        received.append({"target": target})
        return {"index": Path(target) / "index.html"}

    monkeypatch.setattr(bf_mod, "export_butterfly_figure", stub_without_style)
    run_butterfly_figure_export(payload=base_payload)
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Qt-dependent regression tests
# ---------------------------------------------------------------------------

pytestmark_qt = pytest.mark.skipif(not QT_AVAILABLE, reason="PySide6 is unavailable")


@pytestmark_qt
def test_parameter_table_vary_checkbox_recheck_and_derived_theta_deg(qtbot) -> None:
    from butterfly_saxs.ui.models import ParameterTableModel

    model = ParameterTableModel(
        {
            "theta": {"value": 0.0, "vary": True, "minimum": -3.14, "maximum": 3.14},
            "theta_deg": {"value": 0.0, "vary": False, "expression": "degrees(theta)"},
        }
    )
    vary_idx = model.index(0, 4)
    # PySide6 QStyledItemDelegate sends plain int 0 / 2 for CheckStateRole.
    assert model.setData(vary_idx, 0, QtCore.Qt.ItemDataRole.CheckStateRole) is True
    assert model.data(vary_idx, QtCore.Qt.ItemDataRole.CheckStateRole) == QtCore.Qt.CheckState.Unchecked
    assert model.setData(vary_idx, 2, QtCore.Qt.ItemDataRole.CheckStateRole) is True
    assert model.data(vary_idx, QtCore.Qt.ItemDataRole.CheckStateRole) == QtCore.Qt.CheckState.Checked

    emitted: list[tuple[int, list[object]]] = []
    model.dataChanged.connect(lambda tl, _br, roles: emitted.append((tl.row(), list(roles))))
    val_idx = model.index(0, 1)
    assert model.setData(val_idx, math.pi / 4.0, None) is True
    assert all(None not in roles for _, roles in emitted)
    assert any(row == 1 for row, _ in emitted)
    assert float(model.data(model.index(1, 1), QtCore.Qt.ItemDataRole.DisplayRole)) == pytest.approx(45.0)


@pytestmark_qt
def test_main_window_busy_focus_deleted_editor_no_crash(qtbot) -> None:
    import shiboken6
    from butterfly_saxs.ui import MainWindow

    window = MainWindow(auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    # Deleted transient button / cell editor must never crash _restore_busy_focus.
    temp = QtWidgets.QPushButton("ephemeral", window)
    window._remember_busy_focus_widget(temp)
    shiboken6.delete(temp)
    assert not is_qt_valid(temp)
    window._restore_busy_focus()

    # Realistic QTableView delegate edit -> busy -> idle cycle.
    window.pages.setCurrentIndex(1)
    idx = window.parameter_model.index(0, 1)
    window.parameter_table.setFocus()
    window.parameter_table.setCurrentIndex(idx)
    window.parameter_table.edit(idx)
    window._set_busy(True, "preview")
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    window._set_busy(False, "preview")
    window.close()


@pytestmark_qt
def test_main_window_failed_load_project_rolls_back_butterfly_and_uses_service_snapshot(
    qtbot, tmp_path: Path
) -> None:
    from butterfly_saxs.ui import MainWindow

    class _SnapshotEngine:
        def __init__(self) -> None:
            self.token = "initial"
            self.snapshot_calls = 0
            self.restore_calls = 0

        def snapshot_state(self) -> dict[str, str]:
            self.snapshot_calls += 1
            return {"token": self.token}

        def restore_state(self, state: dict[str, str]) -> None:
            self.restore_calls += 1
            self.token = state["token"]

        def load_image(self, path, **_kwargs):
            arr = np.load(path)
            self.token = Path(path).name
            return {"data": arr, "qmap": None, "metadata": {}}

    engine = _SnapshotEngine()
    window = MainWindow(engine=engine, auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)

    path_a = tmp_path / "a.npy"
    path_b = tmp_path / "b.npy"
    np.save(path_a, np.full((24, 24), 10.0, dtype=np.float32))
    np.save(path_b, np.full((24, 24), 999.0, dtype=np.float32))
    assert window.open_image(path_a)
    window.butterfly_workbench._on_edit_requested({"type": "seed", "qx": 0.1, "qy": 0.2})
    assert len(window.butterfly_workbench.edits) == 1

    bad_project = tmp_path / "bad.json"
    bad_project.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "input": str(path_b),
                "parameters": {},
                "legacy_geometry": "NOT_A_MAPPING",
            }
        ),
        encoding="utf-8",
    )
    assert window.load_project(bad_project) is False
    assert engine.snapshot_calls >= 1
    assert engine.restore_calls >= 1
    assert engine.token == "a.npy"
    assert float(np.nanmean(window._observed)) == pytest.approx(10.0)
    assert float(np.nanmean(window.butterfly_workbench.qspace._observed)) == pytest.approx(10.0)
    assert len(window.butterfly_workbench.edits) == 1
    window.close()


@pytestmark_qt
def test_main_window_ridge_spins_wired_dark_plots_preserved_and_evolution_blanks(qtbot) -> None:
    from butterfly_saxs.ui import MainWindow

    window = MainWindow(auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)

    # Ridge spinboxes must propagate changes to analysis_settings.
    window.ridge_snr_threshold_spin.setValue(4.25)
    window.ridge_min_peak_fraction_spin.setValue(0.37)
    window.ridge_min_coverage_spin.setValue(0.62)
    settings = window.analysis_settings
    assert settings["ridge_snr_threshold"] == pytest.approx(4.25)
    assert settings["ridge_min_peak_fraction"] == pytest.approx(0.37)
    assert settings["ridge_min_coverage"] == pytest.approx(0.62)

    # Dark diagnostic plots must NOT be clobbered to white by global style_plot.
    for plot in (
        window.butterfly_workbench.normal_profile.plot,
        window.butterfly_workbench.ellipse_diagnostic.plot,
        window.butterfly_workbench.peak_angular_profile.plot,
        window.butterfly_workbench.peak_radial_profile.plot,
        window.local_measurement_page.image_plot,
    ):
        assert plot is not None
        assert plot.backgroundBrush().color().name().lower() != "#ffffff"

    # Evolution table must render None / NaN as blank strings, not "None" / "nan".
    window.plot_evolution(
        [
            {"frame": 0, "rmse": 0.12, "q_star": None},
            {"frame": 1, "rmse": float("nan"), "q_star": 0.45},
        ]
    )
    headers = [
        window.evolution_table.horizontalHeaderItem(c).text()
        for c in range(window.evolution_table.columnCount())
    ]
    assert window.evolution_table.item(0, headers.index("q_star")).text() == ""
    assert window.evolution_table.item(1, headers.index("rmse")).text() == ""

    # flags_label must retain help explanation after _render_metric_labels.
    window._set_status("status.ready", flags="low_coverage")
    tip = window.flags_label.toolTip()
    assert "\n" in tip and "low_coverage" in tip
    window.close()


@pytestmark_qt
def test_main_window_save_project_confirms_overwrite_after_json_suffix(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    from butterfly_saxs.ui import MainWindow

    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    existing = tmp_path / "existing.json"
    existing.write_text('{"keep": true}', encoding="utf-8")

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_kw: (str(tmp_path / "existing"), "JSON (*.json)"),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_a, **_kw: QtWidgets.QMessageBox.StandardButton.No,
    )
    assert window.save_project() is False
    assert json.loads(existing.read_text(encoding="utf-8")) == {"keep": True}

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_a, **_kw: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    assert window.save_project() is True
    assert "schema_version" in json.loads(existing.read_text(encoding="utf-8"))
    window.close()


@pytestmark_qt
def test_local_measurement_pixel_rect_boundary_note_and_overwrite(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    from butterfly_saxs.ui.local_measurement_page import LocalMeasurementPage

    page = LocalMeasurementPage(language="zh_CN")
    qtbot.addWidget(page)
    rows, cols = np.indices((10, 12), dtype=float)
    data = np.full((10, 12), 5.0)
    page.set_data(data, qx=cols * 0.01, qy=rows * 0.01, q_unit="nm^-1", source="f0")

    rect = page.image_item.mapRectToParent(page.image_item.boundingRect())
    assert rect.x() == pytest.approx(-0.5)
    assert rect.y() == pytest.approx(-0.5)
    assert rect.width() == pytest.approx(12.0)
    assert rect.height() == pytest.approx(10.0)

    assert "空隙" in page.boundary_note.text()
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("roi"))
    assert "ROI 主轴" in page.boundary_note.text()

    page.mode_combo.setCurrentIndex(page.mode_combo.findData("point"))
    page._add_point(3, 4)
    assert "像素 (列, 行)" in page.details.text()

    existing = tmp_path / "local.csv"
    existing.write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_kw: (str(tmp_path / "local"), "CSV (*.csv)"),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_a, **_kw: QtWidgets.QMessageBox.StandardButton.No,
    )
    page._choose_export()
    assert existing.read_text(encoding="utf-8") == "original"

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_a, **_kw: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    page._choose_export()
    assert existing.read_text(encoding="utf-8-sig") != "original"


@pytestmark_qt
def test_lamellar_scattering_dialog_allows_confirmed_overwrite(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    from butterfly_saxs.ui.lamellar_scattering_dialog import LamellarScatteringDialog

    signs = np.asarray(
        [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
        ],
        dtype=float,
    )
    sizes = np.asarray([[2.0, 1.5, 0.6]])
    orientations = np.eye(3)[None, :, :]
    vertices = signs[None, :, :] * (sizes[:, None, :] / 2.0)
    scene = LamellarScene(
        centers=np.zeros((1, 3)),
        sizes=sizes,
        orientations=orientations,
        vertices=vertices,
        stack_ids=np.asarray([0]),
        branch_ids=np.asarray([0]),
        colors=np.asarray([[0.2, 0.4, 0.7, 0.8]]),
        bounds=np.asarray((vertices.reshape(-1, 3).min(axis=0), vertices.reshape(-1, 3).max(axis=0))),
        length_unit="nm",
        metadata={"status": "schematic"},
    )
    dialog = LamellarScatteringDialog(scene, language="en")
    qtbot.addWidget(dialog)

    target = tmp_path / "fft_out.npz"
    target.write_bytes(b"old")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_kw: (str(target), "Compressed arrays (*.npz)"),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_a, **_kw: QtWidgets.QMessageBox.StandardButton.No,
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *_a, **_kw: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_kw: None)
    dialog._export()
    assert target.read_bytes() == b"old"

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_a, **_kw: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    dialog._export()
    assert target.read_bytes() != b"old"
    assert target.with_suffix(".json").exists()


@pytestmark_qt
def test_lamellar_3d_error_isolation_single_camera_signal_and_qml_mouse_area(qtbot) -> None:
    from butterfly_saxs.ui.lamellar_3d import Lamellar3DView

    qml_text = (ROOT / "src" / "butterfly_saxs" / "ui" / "qml" / "LamellarScene.qml").read_text(
        encoding="utf-8"
    )
    assert "import QtQuick 6.8" not in qml_text
    assert "import QtQuick3D 6.8" not in qml_text
    assert "visible: !root.publicationMode" in qml_text

    view = Lamellar3DView()
    qtbot.addWidget(view)
    view.set_language("en")
    # Invalid scene should show the scene error, NOT the hardware-acceleration message.
    view.set_scene(object())
    assert "graphics acceleration" not in view._message.text().lower()
    assert "cannot be displayed" in view._message.text().lower()

    # Subsequent valid scene clears the scene error completely.
    signs = np.asarray(
        [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
        ],
        dtype=float,
    )
    sizes = np.asarray([[2.0, 1.5, 0.6]])
    vertices = signs[None, :, :] * (sizes[:, None, :] / 2.0)
    valid_scene = LamellarScene(
        centers=np.zeros((1, 3)),
        sizes=sizes,
        orientations=np.eye(3)[None, :, :],
        vertices=vertices,
        stack_ids=np.asarray([0]),
        branch_ids=np.asarray([0]),
        colors=np.asarray([[0.2, 0.4, 0.7, 0.8]]),
        bounds=np.asarray((vertices.reshape(-1, 3).min(axis=0), vertices.reshape(-1, 3).max(axis=0))),
        length_unit="nm",
        metadata={"status": "schematic"},
    )
    view.set_scene(valid_scene)
    assert view._scene_error == ""
    assert "cannot be displayed" not in view.error_message.lower()

    # _connect_camera_signal must connect only once even if called repeatedly.
    class _FakeRoot(QtCore.QObject):
        cameraChanged = QtCore.Signal(object)

    fake_root = _FakeRoot()
    emissions: list[dict[str, object]] = []
    view.cameraChanged.connect(lambda state: emissions.append(state))
    view._connect_camera_signal(fake_root)
    view._connect_camera_signal(fake_root)
    fake_root.cameraChanged.emit({"yaw": 30.0, "pitch": 15.0, "distance": 12.0})
    assert len(emissions) == 1


@pytestmark_qt
def test_lamellar_page_focus_unfocus_restore_and_publication_language(qtbot) -> None:
    from butterfly_saxs.ui.lamellar_page import LamellarPage
    from butterfly_saxs.ui.publication_panel import PublicationDialog

    page = LamellarPage(language="en")
    qtbot.addWidget(page)
    page.focus_view("three_d")
    assert page.focus_buttons["three_d"].property("textKey") == "unfocus"
    page.focus_view("three_d")
    assert page.focus_buttons["three_d"].property("textKey") == "focus"

    calls: list[int] = []
    orig = page._queue_build
    page._queue_build = lambda: (calls.append(1), orig())  # type: ignore[assignment]
    snap = page.snapshot_state()
    calls.clear()
    page.restore_state(snap)
    assert len(calls) == 1
    assert page.wait_for_done(200) is True

    dialog = PublicationDialog(page, language="zh_CN")
    qtbot.addWidget(dialog)
    assert "发表画板" in dialog.windowTitle()
    dialog.set_language("en")
    assert dialog.windowTitle() == "Publication artboard"
    assert dialog.export_button.text() == "Export publication figure…"
    assert dialog.wait_for_done(200) is True
    dialog.close()
    page.close()


@pytestmark_qt
def test_views_qspace_summary_workbench_azimuthal_density2d_and_dialogs(
    qtbot, monkeypatch
) -> None:
    from butterfly_saxs.ui.azimuthal_page import AzimuthalPage
    from butterfly_saxs.ui.butterfly_summary import ButterflyQualitySummary
    from butterfly_saxs.ui.butterfly_workbench import ButterflyWorkbench
    from butterfly_saxs.ui.density2d_page import Density2DPage
    from butterfly_saxs.ui.figure_export_dialog import FigureExportDialog
    from butterfly_saxs.ui.qspace import QSpaceView
    from butterfly_saxs.ui.views import ViewGrid
    from butterfly_saxs.ui.workbench import _workflow_lines

    # ViewGrid: set_image(None) clears image; set_language preserves _active_q_window.
    grid = ViewGrid(language="zh_CN")
    qtbot.addWidget(grid)
    rows, cols = np.indices((16, 16), dtype=float)
    qx = (cols - 8.0) * 0.05
    qy = (rows - 8.0) * 0.05
    grid.set_images(np.ones((16, 16)), qx=qx, qy=qy, q_unit="nm^-1")
    grid.set_q_view((0.1, 0.3))
    assert grid._active_q_window == (0.1, 0.3)
    grid.set_language("en")
    assert grid._active_q_window == (0.1, 0.3)
    grid.observed.set_image(None)
    assert grid.observed.image_data is None
    assert grid.observed._content_layout.currentWidget() is grid.observed.state_label

    # QSpaceView: set_q_window refreshes levels; set_data clears stale butterfly/edits.
    qview = QSpaceView()
    qtbot.addWidget(qview)
    img = np.ones((20, 20), dtype=float)
    img[0, 0] = 10000.0
    qview.set_data(img, qx=(np.indices((20, 20))[1] - 10) * 0.1, qy=(np.indices((20, 20))[0] - 10) * 0.1)
    levels_full = qview._levels
    qview.set_q_window((0.0, 0.35))
    assert qview._levels != levels_full
    qview.set_edits([{"type": "seed", "qx": 0.1, "qy": 0.1}])
    qview.set_data(np.ones((20, 20), dtype=float))
    assert qview._edits == []
    assert qview._butterfly == {}

    # ButterflyQualitySummary: annular_peak must preserve load/identify/wait/retry guidance.
    summary = ButterflyQualitySummary(language="en")
    qtbot.addWidget(summary)
    summary.set_state(result=None, data_ready=False, trace_method="annular_peak")
    assert "load" in summary.next_label.text().lower()
    summary.set_state(result=None, data_ready=True, trace_method="annular_peak")
    assert "identify" in summary.next_label.text().lower()
    summary.set_state(result=None, data_ready=True, busy=True, trace_method="annular_peak")
    assert "wait" in summary.next_label.text().lower()
    summary.set_state(result=None, data_ready=True, error="boom", trace_method="annular_peak")
    assert "ellipse residuals" not in summary.next_label.text().lower()
    assert qview._butterfly == {}

    # ButterflyWorkbench: Delete shortcut is WidgetShortcut; legacy banner localizes; NaN rejected.
    wb = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(wb)
    assert wb._point_delete_shortcut.context() == QtCore.Qt.ShortcutContext.WidgetShortcut
    wb.set_legacy_method("radial_peak")
    assert "旧版脊线方法" in wb.legacy_banner.text()
    wb.set_language("en")
    assert "Loaded legacy ridge method" in wb.legacy_banner.text()

    warnings_shown: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _p, _t, msg: warnings_shown.append(str(msg)),
    )
    orig_exec = QtWidgets.QDialog.exec

    def fake_nan_exec(dlg):
        dlg.findChild(QtWidgets.QLineEdit, "butterflyNumericQx").setText("nan")
        dlg.findChild(QtWidgets.QLineEdit, "butterflyNumericQy").setText("0.2")
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_nan_exec)
    wb._open_numeric_edit_dialog()
    monkeypatch.setattr(QtWidgets.QDialog, "exec", orig_exec)
    assert wb.edits == []
    assert warnings_shown

    # AzimuthalPage: set_data blocks spinbox signals so _draw runs once, not 4x.
    az = AzimuthalPage(language="en")
    qtbot.addWidget(az)
    draw_count = []
    orig_draw = az._draw
    az._draw = lambda: (draw_count.append(1), orig_draw())  # type: ignore[assignment]
    az.set_data(
        np.ones((16, 16)),
        qx=qx,
        qy=qy,
        valid_mask=np.ones((16, 16), dtype=bool),
        q_unit="nm^-1",
        source="f0",
    )
    assert len(draw_count) == 1
    assert az.wait_for_done(200) is True

    # Density2DPage: fit.q_min <= 0 must not crash geomspace, and legend localizes.
    d2d = Density2DPage(language="zh_CN")
    qtbot.addWidget(d2d)
    d2d.set_data(np.ones((16, 16)) * 10.0, qx=qx, qy=qy, q_unit="1/nm")
    assert "nm^-1" in d2d.q_min_spin.suffix()
    prof = Density2DProfile(
        q=np.asarray([0.0, 0.05, 0.1, 0.2]),
        intensity=np.asarray([100.0, 40.0, 15.0, 5.0]),
        counts=np.asarray([5, 5, 5, 5]),
        geometry_counts=np.asarray([5, 5, 5, 5]),
        coverage=np.ones(4),
        q_min=0.0,
        q_max=0.2,
        azimuth_center_deg=0.0,
        azimuth_half_width_deg=18.0,
        q_unit="1/nm",
    )
    fit = Density2DFit(
        model_name="power_law",
        success=True,
        message="ok",
        parameters={"prefactor": 1.0, "alpha": 2.0},
        fitted_intensity=np.asarray([100.0, 40.0, 15.0, 5.0]),
        residual=np.zeros(4),
        n_points=4,
        q_min=0.0,
        q_max=0.2,
        r_squared=0.99,
        rmse=0.1,
    )
    d2d.profile = prof
    d2d.fits = (fit,)
    d2d._draw()
    assert "幂律" in d2d._fit_legend(fit)
    assert d2d.wait_for_done(200) is True

    # FigureExportDialog: set_language while running translates action buttons.
    fed = FigureExportDialog(language="zh_CN")
    qtbot.addWidget(fed)
    fed.set_export_running()
    fed.set_language("en")
    assert fed.cancel_button.text() == "Cancel export"
    assert fed.start_button.text() == "Exporting…"
    fed.close()

    # workbench._workflow_lines: no raw English 'unreviewed'/'preview' in zh_CN prose.
    class _StubWin:
        _language = "zh_CN"
        _observed = np.ones((8, 8))
        _source_path = "sample.npy"
        _qmap = {"q_unit": "1/nm"}
        _file_mask = None
        _roi_specs = ()
        _last_result = {"ok": True}
        _last_result_kind = "preview"
        _fit_session = {"manual_status": "unreviewed"}

    body, _warn = _workflow_lines(_StubWin())
    assert "unreviewed" not in body
    assert "preview" not in body
