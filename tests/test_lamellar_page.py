from __future__ import annotations

from copy import deepcopy
import threading
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from butterfly_saxs.ui import lamellar_page as page_module  # noqa: E402


class Stub3D(QtWidgets.QWidget):
    """Page tests isolate the UI controller; actual RHI has separate tests."""

    cameraChanged = QtCore.Signal(object)
    available = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = {"yaw": 25., "pitch": 15., "zoom": 1.}

    def set_scene(self, scene, **kwargs):
        self.scene = scene

    def clear(self, message=""):
        self.scene = None

    def camera_state(self):
        return dict(self.camera)

    def set_camera_state(self, value):
        self.camera.update(value)

    def set_standard_view(self, name):
        self.camera["view"] = name

    def reset_camera(self):
        self.camera = {"yaw": 25., "pitch": 15., "zoom": 1.}

    def set_language(self, language):
        pass

    def set_selected_branch(self, branch):
        self.selected_branch = branch

    def capture_image(self, width=2400, height=1800):
        image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGBA8888)
        image.fill(QtGui.QColor("#90bddd"))
        return image


def source(q=.2, frame=0, *, status="ok", image=True):
    data = {
        "source_identity": {"source": "in-memory", "frame": frame, "dataset": "detector", "id": f"frame-{frame}"},
        "q_unit": "nm^-1", "draw_axis_deg": 90., "status": status,
        "lobe_radial_peaks": [
            {"q_star": q, "angle": np.pi / 3, "valid": True, "q_unit": "nm^-1", "branch_id": 0},
            {"q_star": q, "angle": 4 * np.pi / 3, "valid": True, "q_unit": "nm^-1", "branch_id": 0},
        ],
    }
    if image:
        x, y = np.meshgrid(np.linspace(-.5, .5, 16), np.linspace(-.5, .5, 16))
        data.update(observed=np.ones((16, 16)) * (frame + 1), qx=x, qy=y)
    return data


def ellipse_source(q=.2, frame=0, *, image=True):
    data = source(q, frame, image=image)
    data["ellipse_fit"] = {
        "parameters": {"a": .3, "b": .15, "theta_deg": 0.},
        "center": [0., 0.],
        "q_unit": "nm^-1",
        "status": "available",
    }
    return data


@pytest.fixture
def page(qtbot, monkeypatch):
    monkeypatch.setattr(page_module, "Lamellar3DView", Stub3D)
    widget = page_module.LamellarPage(language="en")
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()
    qtbot.waitUntil(lambda: not widget.jobs_running(), timeout=15000)


def ready(qtbot, page):
    qtbot.waitUntil(lambda: page.current_scene is not None and not page.jobs_running(), timeout=15000)


def test_settings_are_independent_and_undo_restores_geometry(page, qtbot):
    original = source()
    before = deepcopy(original["lobe_radial_peaks"])
    page.set_source(original, context_signature="a")
    ready(qtbot, page)
    vertices = page.current_scene.vertices.copy()
    page.update_settings({**page.settings, "thickness_ratio": .5})
    qtbot.waitUntil(lambda: bool(page.scenes) and page.settings["thickness_ratio"] == .5 and not page.jobs_running(), timeout=15000)
    assert not np.array_equal(vertices, page.current_scene.vertices)
    assert original["lobe_radial_peaks"] == before
    page.undo()
    ready(qtbot, page)
    np.testing.assert_allclose(page.current_scene.vertices, vertices)
    page.redo()
    ready(qtbot, page)
    assert page.current_scene.metadata["settings"]["thickness_ratio"] == .5


def test_meso_preview_without_fit_uses_two_manual_directions(page, qtbot):
    page.controls.meso_preview_button.click()
    ready(qtbot, page)

    assert page.settings["period_source"] == "manual"
    assert page.settings["manual_second_orientation"] is True
    assert page.settings["thickness_ratio"] == pytest.approx(.70)
    assert page.settings["out_of_plane_deg"] == pytest.approx(0.)
    assert page.current_scene.metadata["status"] == "manual"
    assert set(page.current_scene.branch_ids.tolist()) == {0, 1}
    assert set(page.current_scene.stack_ids.tolist()) == set(range(64))
    assert [item["angle_deg"] for item in page.current_scene.metadata["populations"]] == [30., 120.]
    assert any("not fitted populations" in item for item in page.current_scene.metadata["assumptions"])


def test_fit_driven_preview_retains_observed_branches_and_period(page, qtbot):
    page.set_source(source(.2))
    ready(qtbot, page)
    page.controls.meso_preview_button.click()
    ready(qtbot, page)

    assert page.settings["period_source"] == "radial"
    assert page.settings["manual_second_orientation"] is False
    assert page.current_scene.metadata["populations"][0]["period"] == pytest.approx(2. * np.pi / .2)
    assert set(page.current_scene.branch_ids.tolist()) == {0}
    assert set(page.current_scene.stack_ids.tolist()) == set(range(0, 64, 2))


def test_fresh_ellipse_fit_selects_ellipse_periods_by_default(page, qtbot):
    page.set_source(ellipse_source())
    ready(qtbot, page)

    assert page.settings["period_source"] == "ellipse"
    assert page.current_scene.metadata["period_source"] == "ellipse"
    assert page.current_scene.metadata["available"]


def test_unusable_ellipse_fit_falls_back_to_radial_periods(page, qtbot):
    data = source()
    data["ellipse_fit"] = {"status": "failed", "parameters": {}}
    page.set_source(data)
    ready(qtbot, page)

    assert page.settings["period_source"] == "radial"
    assert page.current_scene.metadata["period_source"] == "radial"
    assert page.current_scene.metadata["available"]


@pytest.mark.parametrize("period_source", ["radial", "manual"])
def test_explicit_period_source_survives_new_fit_source(page, qtbot, period_source):
    page.set_source(ellipse_source())
    ready(qtbot, page)
    page.update_settings({**page.settings, "period_source": period_source})
    ready(qtbot, page)
    assert page._period_source_explicit

    page.set_source(ellipse_source(.16, frame=1))
    ready(qtbot, page)

    assert page.settings["period_source"] == period_source
    assert page.current_scene.metadata["period_source"] == period_source


def test_reset_period_source_returns_to_fit_driven_default(page, qtbot):
    page.set_source(ellipse_source())
    ready(qtbot, page)
    page.update_settings({**page.settings, "period_source": "radial"})
    ready(qtbot, page)

    page.reset_settings()
    ready(qtbot, page)

    assert page.settings["period_source"] == "ellipse"
    assert not page._period_source_explicit


def test_sequence_promotes_an_explicit_small_reference_to_a_shared_safe_scale():
    both = source(.2, 0, image=False)
    both["lobe_radial_peaks"].extend(
        [
            {"q_star": .1, "angle": 5. * np.pi / 6., "valid": True, "q_unit": "nm^-1", "branch_id": 1},
            {"q_star": .1, "angle": 11. * np.pi / 6., "valid": True, "q_unit": "nm^-1", "branch_id": 1},
        ]
    )
    only_a = source(.2, 1, image=False)
    settings = {
        "mode": "multi",
        "period_source": "radial",
        "layer_count": 1,
        "stack_count": 12,
        "reference_period": 1.0,
        "position_jitter_pct": 70.0,
        "seed": 19,
    }

    scenes = page_module._build_scenes([both, only_a], settings, threading.Event())

    common = 2. * np.pi / .1
    assert [item.metadata["reference_period"] for item in scenes] == pytest.approx([common, common])
    by_id = [
        {int(stack): center for stack, center, branch in zip(scene.stack_ids, scene.centers, scene.branch_ids)
         if int(branch) == 0 and int(stack) % 2 == 0}
        for scene in scenes
    ]
    for stack in by_id[1]:
        np.testing.assert_allclose(by_id[0][stack], by_id[1][stack])


@pytest.mark.parametrize("explicit_reference", [None, 100.0])
def test_sequence_does_not_share_reference_period_across_physical_and_unknown_q_units(explicit_reference):
    physical = source(.2, 0, image=False)
    unknown = source(.2, 1, image=False)
    unknown["q_unit"] = "unknown"
    for peak in unknown["lobe_radial_peaks"]:
        peak["q_unit"] = "unknown"
    settings = {
        "mode": "multi",
        "period_source": "radial",
        "layer_count": 1,
        "stack_count": 12,
        "reference_period": explicit_reference,
        "seed": 19,
    }

    scenes = page_module._build_scenes([physical, unknown], settings, threading.Event())

    assert [scene.length_unit for scene in scenes] == ["nm", "relative"]
    assert scenes[0].metadata["populations"][0]["period"] == pytest.approx(2. * np.pi / .2)
    assert scenes[1].metadata["populations"][0]["period"] == pytest.approx(1.)
    assert scenes[1].metadata["reference_period"] == pytest.approx(1.)


def test_same_unit_unknown_q_sequence_keeps_explicit_shared_reference_period():
    frames = []
    for frame, q in enumerate((.2, .1)):
        item = source(q, frame, image=False)
        item["q_unit"] = "unknown"
        for peak in item["lobe_radial_peaks"]:
            peak["q_unit"] = "unknown"
        frames.append(item)

    scenes = page_module._build_scenes(frames, {
        "mode": "multi",
        "period_source": "radial",
        "layer_count": 1,
        "stack_count": 12,
        "reference_period": 7.0,
        "seed": 19,
    }, threading.Event())

    assert [scene.length_unit for scene in scenes] == ["relative", "relative"]
    assert [scene.metadata["populations"][0]["period"] for scene in scenes] == pytest.approx([7.0, 7.0])
    assert [scene.metadata["reference_period"] for scene in scenes] == pytest.approx([7.0, 7.0])


def test_mixed_population_units_are_shown_per_branch(page, qtbot):
    item = source(.2, 0)
    item["q_unit"] = "unknown"
    item["lobe_radial_peaks"][0]["q_unit"] = "nm^-1"
    item["lobe_radial_peaks"][1]["q_unit"] = "nm^-1"
    item["lobe_radial_peaks"].extend([
        {"q_star": .1, "angle": 5. * np.pi / 6., "valid": True, "q_unit": "unknown", "branch_id": 1},
        {"q_star": .1, "angle": 11. * np.pi / 6., "valid": True, "q_unit": "unknown", "branch_id": 1},
    ])

    page.set_source(item)
    ready(qtbot, page)

    assert not page.current_scene.metadata["available"]
    assert "31.42 nm" in page.period_readout.text()
    assert "1 relative" in page.period_readout.text()


def test_control_clamps_spacing_jitter_and_preserves_hidden_reference_period(page, qtbot):
    values = dict(page.settings)
    values.update(thickness_ratio=.25, spacing_jitter_pct=30., reference_period=7.)
    page.controls.set_settings(values)

    with qtbot.waitSignal(page.controls.settingsChanged, timeout=5000):
        page.controls.controls["thickness_ratio"].setValue(.9)

    assert page.settings["spacing_jitter_pct"] == pytest.approx(10.)
    assert page.settings["reference_period"] == pytest.approx(7.)


def test_invalidation_cannot_be_bypassed_by_cosmetic_change(page, qtbot):
    page.set_source(source())
    ready(qtbot, page)
    page.invalidate()
    page.update_settings({**page.settings, "width_ratio": 5.})
    qtbot.wait(250)
    assert page.current_scene is None
    assert not page.export_button.isEnabled()
    with pytest.raises(ValueError, match="current scene"):
        page.export_to("never-written")
    page.update_settings({**page.settings, "period_source": "manual"})
    ready(qtbot, page)
    assert page.current_scene.metadata["status"] == "manual"


def test_palette_undo_preserves_geometry_and_source(page, qtbot):
    data = source()
    page.set_source(data)
    ready(qtbot, page)
    geometry = page.current_scene.vertices.copy()
    colors = page.current_scene.colors.copy()
    page.set_palette("grayscale")
    np.testing.assert_array_equal(page.current_scene.vertices, geometry)
    assert not np.array_equal(colors, page.current_scene.colors)
    assert page.document()["presentation"]["palette"] == "grayscale"
    page.undo()
    np.testing.assert_array_equal(page.current_scene.colors, colors)
    assert page.sources[0] is data


def test_projected_fft_action_receives_current_scene_and_language(page, qtbot, monkeypatch):
    page.set_source(source())
    ready(qtbot, page)
    assert page.scattering_button.isEnabled()
    captured = {}

    class Dialog:
        def __init__(self, scene, *, parent, language):
            captured.update(scene=scene, parent=parent, language=language)

        def exec(self):
            captured["executed"] = True

    monkeypatch.setitem(
        sys.modules,
        "butterfly_saxs.ui.lamellar_scattering_dialog",
        SimpleNamespace(LamellarScatteringDialog=Dialog),
    )
    page.open_projected_fft()

    assert captured == {
        "scene": page.current_scene,
        "parent": page,
        "language": "en",
        "executed": True,
    }
    page.invalidate()
    assert not page.scattering_button.isEnabled()


def test_failed_frame_is_blank_and_does_not_reuse_image(page, qtbot):
    bad = source(.2, 1, status="failed", image=False)
    bad["lobe_radial_peaks"] = []
    page.set_series([source(.2, 0), bad, source(.4, 2)])
    ready(qtbot, page)
    reference = page._bounds.copy()
    camera = page.view3d.camera_state()
    assert page.q_view.observed is not None
    page.select_frame(1)
    ready(qtbot, page)
    assert not page.current_scene.metadata["available"]
    assert not len(page.current_scene.vertices)
    assert page.q_view.observed is None
    assert "frame 1" in page.frame_text.text()
    page.select_frame(2)
    ready(qtbot, page)
    assert page.current_scene.metadata["available"]
    np.testing.assert_allclose(reference, page._bounds)
    assert page.view3d.camera_state() == camera
    assert np.all(page.q_view.observed == 3)


def test_document_roundtrip_and_context_mismatch(page, qtbot):
    page.set_source(source(), context_signature="original input hashes")
    ready(qtbot, page)
    page.view3d.set_camera_state({"yaw": 70.})
    saved = page.document()
    vertices = page.current_scene.vertices.copy()
    assert saved["period_source_explicit"] is False
    assert "observed" not in saved["sources"][0]
    page.restore_document(saved, context_signature="original input hashes")
    ready(qtbot, page)
    np.testing.assert_array_equal(vertices, page.current_scene.vertices)
    assert page.view3d.camera_state()["yaw"] == 70.
    page.restore_document(saved, context_signature="changed input hashes")
    qtbot.wait(250)
    assert page.current_scene is None
    assert not page.export_button.isEnabled()


def test_document_roundtrip_preserves_explicit_period_source(page, qtbot):
    page.set_source(ellipse_source())
    ready(qtbot, page)
    page.update_settings({**page.settings, "period_source": "radial"})
    ready(qtbot, page)
    saved = page.document()

    page.restore_document(saved)
    ready(qtbot, page)
    page.set_source(ellipse_source(.16, frame=1))
    ready(qtbot, page)

    assert saved["period_source_explicit"] is True
    assert page.settings["period_source"] == "radial"
    assert page._period_source_explicit


def test_legacy_document_with_current_context_is_stale_but_manual_remains_available(page, qtbot):
    historical = source(.15, frame=8, image=False)
    document = {
        "version": 1,
        "settings": {"mode": "single", "period_source": "radial", "layer_count": 1, "stack_count": 2},
        "fresh": True,
        "sources": [deepcopy(historical)],
    }
    observed = np.full((16, 16), 17.)

    page.restore_document(
        document,
        context_signature="current input signature",
        observed=observed,
        qx=np.zeros((16, 16)),
        qy=np.zeros((16, 16)),
    )
    qtbot.wait(250)

    assert not page._fresh
    assert page.current_scene is None
    assert page.sources[0]["source_identity"]["frame"] == 8
    assert "observed" not in page.sources[0]
    assert not page.export_button.isEnabled()

    page.update_settings({**page.settings, "period_source": "manual"})
    ready(qtbot, page)
    assert page.current_scene.status == "manual"
    assert not page.current_scene.metadata["source_identity"]


def test_legacy_document_remains_viewable_without_current_input_context(page, qtbot):
    historical = source(.15, frame=8, image=False)
    document = {
        "version": 1,
        "settings": {"mode": "single", "period_source": "radial", "layer_count": 1, "stack_count": 2},
        "fresh": True,
        "sources": [deepcopy(historical)],
    }

    page.restore_document(document)
    ready(qtbot, page)

    assert page._fresh
    assert page.current_scene.metadata["source_identity"]["frame"] == 8


def test_legacy_document_period_source_is_preserved_as_explicit(page, qtbot):
    document = {
        "version": 1,
        "settings": {"mode": "single", "period_source": "radial", "layer_count": 1, "stack_count": 2},
        "fresh": True,
        "sources": [ellipse_source(image=False)],
    }
    page.restore_document(document)
    ready(qtbot, page)

    assert page.settings["period_source"] == "radial"
    assert page._period_source_explicit


def test_unit_display_spelling_is_not_a_new_physical_context():
    import json

    first = {"parameters": {"a": {"value": .3, "unit": "nm⁻¹"}}, "input": {"frame": 7}}
    second = deepcopy(first)
    second["parameters"]["a"]["unit"] = "nm^-1"
    assert page_module.LamellarPage._context_hash(json.dumps(first)) == page_module.LamellarPage._context_hash(json.dumps(second))
    second["parameters"]["a"]["unit"] = "Å^-1"
    assert page_module.LamellarPage._context_hash(json.dumps(first)) != page_module.LamellarPage._context_hash(json.dumps(second))


def test_old_background_result_cannot_replace_new_frame(page, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = page_module._build_scenes

    def delayed(sources, values, cancel):
        if sources[0]["source_identity"]["frame"] == 1:
            entered.set()
            release.wait(5)
            # Ignore cooperative cancellation here to exercise the generation guard.
            return original(sources, values, threading.Event())
        return original(sources, values, cancel)

    monkeypatch.setattr(page_module, "_build_scenes", delayed)
    page.set_source(source(.2, 1))
    qtbot.waitUntil(entered.is_set, timeout=5000)
    page.set_source(source(.4, 2))
    qtbot.waitUntil(lambda: page.current_scene is not None, timeout=10000)
    release.set()
    ready(qtbot, page)
    assert page.current_scene.metadata["source_identity"]["frame"] == 2
    assert page.sources[0]["lobe_radial_peaks"][0]["q_star"] == .4


def test_user_cancel_reports_cancelled_not_permanent_updating(page, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = page_module._build_scenes

    def delayed(sources, values, cancel):
        entered.set()
        release.wait(5)
        return original(sources, values, cancel)

    monkeypatch.setattr(page_module, "_build_scenes", delayed)
    page.set_source(source())
    qtbot.waitUntil(entered.is_set, timeout=5000)
    page.cancel_jobs()
    release.set()
    qtbot.waitUntil(lambda: not page.jobs_running(), timeout=10000)
    assert "cancelled" in page.status.text().lower()


def test_main_window_registers_page_and_carries_current_source(qtbot, monkeypatch):
    monkeypatch.setattr(page_module, "Lamellar3DView", Stub3D)
    from butterfly_saxs.ui.main_window import RefinementMainWindow

    window = RefinementMainWindow(engine={}, auto_preview=False, language="en")
    qtbot.addWidget(window)
    result = source()
    window._source_path = "in-memory"
    window._frame = 7
    window._apply_result(result)
    ready(qtbot, window.lamellar_page)
    assert window.lamellar_page.current_scene.metadata["source_identity"]["frame"] == 7
    assert window.pages.indexOf(window.lamellar_page) >= 0
    assert "lamellar_view" in window.project_to_dict()
    assert window.set_parameter("q_major", .42)
    assert window.lamellar_page.current_scene is None
    window._invalidate_pending_work(clear_fit=True)
    assert window.lamellar_page.current_scene is None
    window.lamellar_page.shutdown()
    qtbot.waitUntil(lambda: not window.lamellar_page.jobs_running(), timeout=15000)
