"""Independent 2D tools follow the active image, calibration and exclusion mask."""
import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from butterfly_saxs.ui.main_window import MainWindow


def test_image_tools_receive_combined_mask_and_new_frame(qtbot, monkeypatch):
    window = MainWindow(engine=object(), auto_preview=False)
    qtbot.addWidget(window)
    received = {}
    for name in ("local_measurement_page", "azimuthal_page", "density2d_page"):
        def capture(data, *, _name=name, **kwargs):
            received[_name] = (np.asarray(data).copy(), kwargs)
        monkeypatch.setattr(getattr(window, name), "set_data", capture)

    y, x = np.indices((20, 24), dtype=float)
    image = x + y
    image[4, 5] = np.nan
    valid = np.ones(image.shape, dtype=bool)
    valid[1, 2] = False
    qmap = {"qx": x / 10, "qy": y / 10, "valid_mask": valid, "q_unit": "nm^-1"}
    window.set_observed_data(image, qmap=qmap)
    assert window.set_exclusion_roi((7, 8, 10, 11))
    for _, context in received.values():
        assert context["q_unit"] == "nm^-1"
        assert not context["valid_mask"][1, 2]
        assert not context["valid_mask"][4, 5]
        assert not context["valid_mask"][9, 8]
        assert context["valid_mask"][15, 16]
    assert window.clear_exclusion_roi()
    for _, context in received.values():
        assert context["valid_mask"][9, 8]
        assert not context["valid_mask"][1, 2]

    window.set_observed_data(np.ones((9, 11)), qmap={
        "qx": np.zeros((9, 11)), "qy": np.ones((9, 11)), "q_unit": "pixel-q",
    })
    for data, context in received.values():
        assert data.shape == (9, 11)
        assert context["q_unit"] == "pixel-q"
        assert context["valid_mask"].all()


def test_new_pages_hide_unrelated_fit_controls_and_translate(qtbot):
    window = MainWindow(engine=object(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.show()
    for page, label in (
        (window.local_measurement_page, "Local measurements"),
        (window.azimuthal_page, "Azimuthal peaks"),
        (window.density2d_page, "Low-q analysis"),
    ):
        window.pages.setCurrentWidget(page)
        assert window.parameters_dock.isHidden()
        assert window.pages.tabText(window.pages.indexOf(page)) == label
    window.pages.setCurrentWidget(window.refinement_page)
    assert not window.parameters_dock.isHidden()


def test_manual_selections_and_angular_settings_survive_project_reload(qtbot, tmp_path):
    from butterfly_saxs.service import ButterflyAnalysisService

    window = MainWindow(engine=ButterflyAnalysisService(), auto_preview=False)
    qtbot.addWidget(window)
    path = tmp_path / "frame.npy"
    np.save(path, np.arange(400, dtype=float).reshape(20, 20))
    assert window.open_image(path)
    window.local_measurement_page._add_point(8, 12)
    window.azimuthal_page.bins_spin.setValue(180)
    project = tmp_path / "session.json"
    assert window.save_project(project)
    window.local_measurement_page.clear_records()
    window.azimuthal_page.bins_spin.setValue(360)
    assert window.load_project(project)
    records = window.local_measurement_page.measurements
    assert len(records) == 1
    assert records[0].intensity == 172
    assert window.azimuthal_page.bins_spin.value() == 180


def test_legacy_scalar_calibration_survives_native_project_roundtrip(qtbot, tmp_path):
    from butterfly_saxs.legacy_saxs import inspect_legacy_saxs, load_legacy_saxs_image
    from butterfly_saxs.service import ButterflyAnalysisService

    image = tmp_path / "frame.npy"
    np.save(image, np.ones((16, 18)))
    session = tmp_path / "fusion.json"
    session.write_text(json.dumps({
        "meta": {"version": 2}, "data_path": "frame.npy",
        "geometry": {"px_mm": 0.172, "dist_mm": 1600, "wl_A": 1.0, "cx": 8.5, "cy": 7.5},
    }), encoding="utf-8")
    window = MainWindow(engine=ButterflyAnalysisService(), auto_preview=False)
    qtbot.addWidget(window)
    legacy = load_legacy_saxs_image(inspect_legacy_saxs(session))
    window._on_legacy_saxs_image_loaded(legacy)
    qx_before = window._qx.copy()
    assert window._active_q_unit() == "nm^-1"
    assert window._poni_path is None
    assert window._source_path == str(image)
    project = tmp_path / "wing.json"
    assert window.save_project(project)
    assert window.open_image(image)
    assert window._legacy_geometry is None
    assert window.load_project(project)
    np.testing.assert_array_equal(window._qx, qx_before)
    assert window._active_q_unit() == "nm^-1"
    assert window._legacy_geometry["session_sha256"] == legacy.inspection.source_sha256


def test_legacy_import_clears_previous_multiframe_mask_selectors(qtbot, tmp_path):
    import json

    from butterfly_saxs.legacy_saxs import inspect_legacy_saxs, load_legacy_saxs_image
    from butterfly_saxs.service import ButterflyAnalysisService

    previous_image = tmp_path / "previous.npy"
    np.save(previous_image, np.ones((16, 18), dtype=np.float32))
    masks = np.zeros((2, 16, 18), dtype=np.uint8)
    masks[1, 3, 4] = 1
    mask_path = tmp_path / "mask_series.npz"
    np.savez(mask_path, mask_series=masks)

    window = MainWindow(engine=ButterflyAnalysisService(), auto_preview=False)
    qtbot.addWidget(window)
    assert window.open_image(previous_image)
    assert window.select_mask(mask_path, mask_frame=1, mask_dataset="mask_series")
    assert window._mask_frame == 1
    assert window._mask_dataset == "mask_series"
    assert window._external_mask[3, 4]

    legacy_image = tmp_path / "legacy.npy"
    np.save(legacy_image, np.arange(16 * 18, dtype=np.float32).reshape(16, 18))
    legacy_session = tmp_path / "fusion.json"
    legacy_session.write_text(json.dumps({
        "meta": {"version": 2}, "data_path": legacy_image.name,
        "geometry": {"px_mm": 0.172, "dist_mm": 1600.0, "wl_A": 1.0,
                     "cx": 8.5, "cy": 7.5},
    }), encoding="utf-8")

    legacy = load_legacy_saxs_image(inspect_legacy_saxs(legacy_session))
    window._on_legacy_saxs_image_loaded(legacy)

    assert window._mask_path is None
    assert window._file_mask is None
    assert window._external_mask is None
    assert window._mask_frame is None
    assert window._mask_dataset is None
    saved = window.project_to_dict()
    assert saved["mask"] is None
    assert saved["mask_frame"] is None
    assert saved["mask_dataset"] is None


def test_old_project_without_poni_clears_active_calibration_and_rollback_restores_it(
    qtbot, tmp_path
):
    import json

    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    from butterfly_saxs.service import ButterflyAnalysisService

    image = tmp_path / "frame.npy"
    replacement = tmp_path / "replacement.npy"
    np.save(image, np.ones((20, 22), dtype=np.float32))
    np.save(replacement, np.full((20, 22), 2.0, dtype=np.float32))
    poni = tmp_path / "geometry.poni"
    AzimuthalIntegrator(
        dist=0.1,
        poni1=0.001,
        poni2=0.0011,
        pixel1=0.0001,
        pixel2=0.0001,
        wavelength=1.0e-10,
    ).save(str(poni))

    window = MainWindow(engine=ButterflyAnalysisService(), auto_preview=False)
    qtbot.addWidget(window)
    assert window.open_image(image)
    assert window.set_poni(poni)
    physical_q = window._qx.copy()
    assert window._active_q_unit() == "nm^-1"

    old_project = tmp_path / "old-project.json"
    old_project.write_text(
        json.dumps({"schema_version": 1, "input": str(image)}),
        encoding="utf-8",
    )
    assert window.load_project(old_project)
    assert window._poni_path is None
    assert window.engine.poni_path is None
    assert window.engine._poni is None
    assert window.project_to_dict()["poni"] is None
    assert window._active_q_unit() == "pixel-q"
    assert not np.allclose(window._qx, physical_q)

    assert window.set_poni(poni)
    blank_poni_project = tmp_path / "old-project-blank-poni.json"
    blank_poni_project.write_text(json.dumps({
        "schema_version": 1, "input": str(image), "poni": "",
    }), encoding="utf-8")
    assert window.load_project(blank_poni_project)
    assert window._poni_path is None
    assert window.engine.poni_path is None
    assert window._active_q_unit() == "pixel-q"

    # A late page-document failure must roll back both UI and service geometry.
    assert window.set_poni(poni)
    window.azimuthal_page.bins_spin.setValue(180)
    expected_source = window._source_path
    expected_qx = window._qx.copy()
    expected_image = window._observed.copy()
    broken_project = tmp_path / "broken-project.json"
    broken_project.write_text(json.dumps({
        "schema_version": 2,
        "input": str(replacement),
        "azimuthal_view": {"schema": "unsupported"},
    }), encoding="utf-8")

    assert not window.load_project(broken_project)
    assert window._source_path == expected_source
    assert window._poni_path == str(poni.resolve())
    assert window.engine.poni_path == str(poni.resolve())
    assert window.engine._poni is not None
    np.testing.assert_array_equal(window._observed, expected_image)
    np.testing.assert_array_equal(window._qx, expected_qx)
    assert window._active_q_unit() == "nm^-1"
    assert window.azimuthal_page.bins_spin.value() == 180
