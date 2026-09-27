from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.legacy_saxs_dialog import LegacySaxsDialog


def _write_session(tmp_path):
    np.savetxt(tmp_path / "detector.csv", np.arange(20).reshape(4, 5), delimiter=",")
    source = {
        "meta": {"version": 2},
        "data_path": "detector.csv",
        "geometry": {
            "px_mm": 0.1,
            "dist_mm": 100,
            "wl_A": 1.54,
            "cx": 2,
            "cy": 1,
        },
        "legacy_results": {"Ln_nm": 42.5},
    }
    path = tmp_path / "sample.saxs_fusion_session.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def test_dialog_inspects_loads_only_supported_image_and_exports_receipt(tmp_path, qtbot):
    session_path = _write_session(tmp_path)
    dialog = LegacySaxsDialog(language="en")
    qtbot.addWidget(dialog)
    emitted = []
    dialog.imageLoaded.connect(emitted.append)

    inspection = dialog.set_source(session_path)
    assert inspection.session is not None
    assert dialog.can_load_image
    assert dialog.load_image_button.isEnabled()
    assert "not converted into WingSAXS observables" in dialog.intro_label.text()
    assert "Ln_nm" in dialog.preview.toPlainText()

    dialog.load_image_button.click()
    assert len(emitted) == 1
    assert emitted[0].image.shape == (4, 5)
    assert emitted[0].qmap.q_unit == "nm^-1"
    assert "No old measurement was converted" in dialog.status_label.text()

    target = tmp_path.parent / f"{tmp_path.name}-receipt.json"
    written = dialog.export_receipt(target)
    assert written == target.resolve()
    receipt = json.loads(target.read_text(encoding="utf-8"))
    assert receipt["original_metadata"]["legacy_results"]["Ln_nm"] == 42.5
    assert "No legacy value is recalculated" in receipt["measurement_interpretation"]

    dialog.set_language("zh_CN")
    assert "兼容性查看器" in dialog.windowTitle()
    assert "归档的二维会话元数据" in dialog.intro_label.text()


def test_dialog_disables_image_load_for_evidence_only_or_missing_data(tmp_path, qtbot):
    session_path = _write_session(tmp_path)
    (tmp_path / "detector.csv").unlink()
    dialog = LegacySaxsDialog(language="zh_CN")
    qtbot.addWidget(dialog)
    dialog.set_source(session_path)
    assert not dialog.can_load_image
    assert not dialog.load_image_button.isEnabled()
    assert "Referenced 2D image is missing" in dialog.preview.toPlainText()

    evidence = tmp_path / "pattern_evidence.json"
    evidence.write_text('{"Ln_nm": 12.0}', encoding="utf-8")
    dialog.set_source(evidence)
    assert not dialog.can_load_image
    assert not dialog.load_image_button.isEnabled()
    assert dialog.export_button.isEnabled()
