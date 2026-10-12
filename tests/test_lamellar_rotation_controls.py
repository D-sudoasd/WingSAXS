from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.lamellar_controls import LamellarControls  # noqa: E402


def test_stack_rotation_is_opt_in_and_clears_conflicting_lateral_shift(qtbot) -> None:
    controls = LamellarControls(language="zh_CN")
    qtbot.addWidget(controls)
    controls.set_settings({"lateral_shift_ratio": 0.4})

    enabled = controls.controls["stack_rotation_enabled"]
    rotation = controls.controls["stack_rotation_deg"]
    lateral = controls.controls["lateral_shift_ratio"]

    assert not enabled.isChecked()
    assert not rotation.isEnabled()
    assert controls._read_controls()["stack_rotation_deg"] is None
    assert lateral.isEnabled()
    assert controls.labels["stack_rotation_enabled"][0].buddy() is enabled
    assert "用户设置" in rotation.toolTip()
    assert "不由椭圆拟合推断" in rotation.toolTip()

    controls.set_language("en")
    tooltip = rotation.toolTip().lower()
    assert "user-set" in tooltip
    assert "not inferred" in tooltip
    assert "ellipse" in tooltip

    with qtbot.waitSignal(controls.settingsChanged, timeout=3000) as emitted:
        enabled.setChecked(True)
    enabled_values = emitted.args[0]
    assert enabled_values["stack_rotation_deg"] == pytest.approx(0.0)
    assert enabled_values["lateral_shift_ratio"] == pytest.approx(0.0)
    assert "stack_rotation_enabled" not in enabled_values
    assert not lateral.isEnabled()

    with qtbot.waitSignal(controls.settingsChanged, timeout=3000) as emitted:
        rotation.setValue(-10.0)
    saved_values = emitted.args[0]
    assert saved_values["stack_rotation_deg"] == pytest.approx(-10.0)
    assert saved_values["lateral_shift_ratio"] == pytest.approx(0.0)

    restored = LamellarControls(language="en")
    qtbot.addWidget(restored)
    restored.set_settings(saved_values)
    assert restored.controls["stack_rotation_enabled"].isChecked()
    assert restored.controls["stack_rotation_deg"].value() == pytest.approx(-10.0)
    assert restored._read_controls()["stack_rotation_deg"] == pytest.approx(-10.0)
    assert not restored.controls["lateral_shift_ratio"].isEnabled()

    with qtbot.waitSignal(controls.settingsChanged, timeout=3000) as emitted:
        enabled.setChecked(False)
    assert emitted.args[0]["stack_rotation_deg"] is None
    assert lateral.isEnabled()


def test_loaded_rotation_clears_conflicting_lateral_shift(qtbot) -> None:
    controls = LamellarControls(language="en")
    qtbot.addWidget(controls)

    controls.set_settings(
        {"stack_rotation_deg": -10.0, "lateral_shift_ratio": 0.4}
    )

    assert controls.controls["stack_rotation_enabled"].isChecked()
    assert controls.controls["stack_rotation_deg"].value() == pytest.approx(-10.0)
    assert controls.controls["lateral_shift_ratio"].value() == pytest.approx(0.0)
    assert not controls.controls["lateral_shift_ratio"].isEnabled()
    assert controls._read_controls()["stack_rotation_deg"] == pytest.approx(-10.0)
