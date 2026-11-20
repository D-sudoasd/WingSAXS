from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from butterfly_saxs import campaign
from butterfly_saxs.analysis_config import validate_analysis_settings
from butterfly_saxs.batch import build_frame_refs
from butterfly_saxs.campaign import (
    CAMPAIGN_CONFIG_SCHEMA,
    CAMPAIGN_RUN_SCHEMA,
    WORKFLOW_PRESETS,
    canonical_workflow_preset,
    discover_manifest,
    init_campaign_config,
    load_campaign_config,
    run_campaign,
    workflow_preset_analysis,
    write_manifest_csv,
)
from butterfly_saxs.project import ProjectConfig, ProjectConfigError, apply_workflow_preset


def test_workflow_presets_are_valid_analysis_settings() -> None:
    assert set(WORKFLOW_PRESETS) == {"annular", "curvature", "sector", "butterfly_pub"}
    for name in WORKFLOW_PRESETS:
        recipe = workflow_preset_analysis(name)
        validated = validate_analysis_settings(recipe)
        assert validated["ridge_method"] == "butterfly_curvature"
        assert validated["butterfly"]["stage"] == "evaluate"
        assert canonical_workflow_preset(name.replace("_", "-").upper()) == name

    with pytest.raises(ValueError, match="unknown workflow preset"):
        workflow_preset_analysis("does_not_exist")


def test_apply_workflow_preset_preserves_none_and_explicit_overrides() -> None:
    base = ProjectConfig(
        inputs=["a.npy"],
        poni="geom.poni",
        analysis={"q_min": 0.12, "butterfly": {"resamples": 16}},
    )
    assert apply_workflow_preset(base, None) is base
    assert apply_workflow_preset(None, None) is None

    applied = apply_workflow_preset(base, "annular")
    assert applied is not None
    assert applied.analysis["ridge_method"] == "butterfly_curvature"
    assert applied.analysis["q_min"] == pytest.approx(0.12)
    assert applied.analysis["butterfly"]["trace_method"] == "annular_peak"
    # Explicit setting wins over preset default.
    assert applied.analysis["butterfly"]["resamples"] == 16
    assert applied.poni_path == "geom.poni"


def test_discover_manifest_natural_sort_metadata_sha256_and_batch_roundtrip(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw_frames"
    raw_dir.mkdir()
    names = [
        "NO50S_350C_run11_frame10.npy",
        "NO50S_350C_run11_frame2.npy",
        "NO50S_350C_run11_frame1.npy",
    ]
    for idx, name in enumerate(names):
        np.save(raw_dir / name, np.full((8, 8), float(idx + 1), dtype=float))
    # Non-image sidecars and default manifest names must be ignored.
    (raw_dir / "notes.json").write_text("{}", encoding="utf-8")
    (raw_dir / "manifest.csv").write_text("path\nignored.npy\n", encoding="utf-8")

    rows = discover_manifest(raw_dir, compute_sha256=True)
    assert [row["frame_id"] for row in rows] == [
        "NO50S_350C_run11_frame1",
        "NO50S_350C_run11_frame2",
        "NO50S_350C_run11_frame10",
    ]
    assert [row["order"] for row in rows] == [0, 1, 2]
    assert [row["parsed_frame_index"] for row in rows] == [1, 2, 10]
    for row in rows:
        assert row["sample"] == "NO50S"
        assert row["temperature_c"] == pytest.approx(350.0)
        assert row["run_id"] == "run11"
        # Missing timestamps are NEVER fabricated.
        assert row["time"] is None
        assert row["frame"] is None
        assert isinstance(row["sha256"], str) and len(row["sha256"]) == 64

    manifest_csv = tmp_path / "out" / "manifest.csv"
    written = write_manifest_csv(rows, manifest_csv)
    assert written == manifest_csv
    with pytest.raises(FileExistsError):
        write_manifest_csv(rows, manifest_csv, force=False)
    write_manifest_csv(rows, manifest_csv, force=True)

    # Round-trip directly into batch.build_frame_refs without selector or time errors.
    refs = build_frame_refs([], manifest=manifest_csv)
    assert len(refs) == 3
    assert [ref.frame_id for ref in refs] == [
        "NO50S_350C_run11_frame1",
        "NO50S_350C_run11_frame2",
        "NO50S_350C_run11_frame10",
    ]
    assert all(ref.time is None for ref in refs)
    assert all(ref.frame_selector is None for ref in refs)


def test_discover_manifest_h5_stack_and_ambiguous_dataset_guard(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")

    stack_path = tmp_path / "NO50S_375C_run02.h5"
    with h5py.File(stack_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=np.ones((4, 12, 12), dtype=np.float32))

    rows = discover_manifest(stack_path, compute_sha256=True)
    assert len(rows) == 4
    assert [row["frame"] for row in rows] == [0, 1, 2, 3]
    assert all(row["dataset"] == "entry/data/data" for row in rows)
    assert all(row["time"] is None for row in rows)
    assert all(row["temperature_c"] == pytest.approx(375.0) for row in rows)
    assert len({row["sha256"] for row in rows}) == 1

    manifest_csv = tmp_path / "h5_manifest.csv"
    write_manifest_csv(rows, manifest_csv)
    refs = build_frame_refs([], manifest=manifest_csv)
    assert [ref.frame_selector for ref in refs] == [0, 1, 2, 3]
    assert all(ref.dataset_id == "entry/data/data" for ref in refs)

    # Ambiguous HDF5 with two non-standard 3D datasets must fail closed unless --dataset is given.
    ambiguous_path = tmp_path / "ambiguous.h5"
    with h5py.File(ambiguous_path, "w") as handle:
        handle.create_dataset("alpha", data=np.ones((2, 8, 8), dtype=np.float32))
        handle.create_dataset("beta", data=np.ones((2, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="ambiguous HDF5 datasets"):
        discover_manifest(ambiguous_path)
    selected = discover_manifest(ambiguous_path, dataset="beta")
    assert len(selected) == 2
    assert all(row["dataset"] == "beta" for row in selected)

    # SPring-8 BL19B2 abs2d_v4 schema (/entry/data/I_abs_2d + uncertainty/statistical).
    bl19b2_path = tmp_path / "NO50S_T350C_run11_0001.h5"
    with h5py.File(bl19b2_path, "w") as handle:
        handle.create_dataset("entry/data/I_abs_2d", data=np.ones((8, 8), dtype=np.float32))
        handle.create_dataset("entry/data/uncertainty/statistical", data=np.ones((8, 8), dtype=np.float64))
    bl_rows = discover_manifest(bl19b2_path)
    assert len(bl_rows) == 1
    assert bl_rows[0]["dataset"] == "entry/data/I_abs_2d"
    assert bl_rows[0]["frame"] is None
    assert bl_rows[0]["time"] is None
    assert bl_rows[0]["temperature_c"] == pytest.approx(350.0)
    assert bl_rows[0]["sample"] == "NO50S"


def test_write_manifest_csv_rejects_empty_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        write_manifest_csv([], tmp_path / "empty.csv")


def test_init_and_load_campaign_config_roundtrip_and_guards(tmp_path: Path) -> None:
    cfg_path = tmp_path / "campaign.toml"
    written = init_campaign_config(
        [
            {"name": "sample_350C", "input_dir": "data/s350"},
            {"name": "sample_375C", "manifest": "data/s375/manifest.csv", "preset": "sector"},
        ],
        cfg_path,
        preset="annular",
        poni="cal/geometry.poni",
        mask="cal/mask.npy",
        q_window=[0.12, 1.35],
    )
    assert written == cfg_path
    with pytest.raises(FileExistsError):
        init_campaign_config([{"name": "a", "input_dir": "x"}], cfg_path, force=False)

    loaded = load_campaign_config(cfg_path)
    assert loaded["schema_version"] == CAMPAIGN_CONFIG_SCHEMA
    assert loaded["preset"] == "annular"
    assert len(loaded["samples"]) == 2
    s0, s1 = loaded["samples"]
    assert s0["name"] == "sample_350C"
    assert s0["preset"] == "annular"
    assert s0["analysis"]["butterfly"]["trace_method"] == "annular_peak"
    assert s0["analysis"]["q_window"] == [0.12, 1.35]
    assert Path(s0["poni"]).is_absolute()
    assert Path(s0["poni"]).name == "geometry.poni"
    assert s1["name"] == "sample_375C"
    assert s1["preset"] == "sector"
    assert s1["analysis"]["butterfly"]["trace_method"] == "radial_sector"

    with pytest.raises(ValueError, match="duplicate campaign sample name"):
        init_campaign_config(
            [
                {"name": "dup", "input_dir": "a"},
                {"name": "dup", "input_dir": "b"},
            ],
            tmp_path / "dup.toml",
        )


def test_run_campaign_isolates_sample_failures_and_supports_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "campaign.toml"
    init_campaign_config(
        [
            {"name": "bad_sample", "input_dir": "bad"},
            {"name": "good_sample", "input_dir": "good"},
            {"name": "warn_sample", "input_dir": "warn"},
        ],
        cfg_path,
        preset="annular",
    )

    def fake_sample_runner(sample: dict[str, Any], sample_dir: Path, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        sample_dir.mkdir(parents=True, exist_ok=True)
        if sample["name"] == "bad_sample":
            raise RuntimeError("corrupt frame header")
        if sample["name"] == "warn_sample":
            return {
                "status": "warning",
                "n_frames": 2,
                "n_success": 1,
                "n_warning": 1,
                "n_failed": 0,
                "warning_reason": "axis_ratio_at_bound",
                "outputs": {"summary": str(sample_dir / "summary.json")},
            }
        return {
            "status": "ok",
            "n_frames": 3,
            "n_success": 3,
            "n_warning": 0,
            "n_failed": 0,
            "outputs": {"summary": str(sample_dir / "summary.json")},
        }

    monkeypatch.setattr(campaign, "_run_sample_batch", fake_sample_runner)

    out_continue = tmp_path / "out_continue"
    summary = run_campaign(cfg_path, out_continue, continue_on_error=True)
    assert summary["schema_version"] == CAMPAIGN_RUN_SCHEMA
    assert summary["status"] == "partial_failure"
    assert summary["exit_code"] == 1
    assert summary["n_samples"] == 3
    assert summary["n_failed"] == 1
    assert summary["n_success"] == 1
    assert summary["n_warning"] == 1
    assert summary["n_skipped"] == 0
    assert summary["scientific_acceptance"] is False
    assert summary["samples"][0]["error"]["type"] == "RuntimeError"
    assert "corrupt frame header" in summary["samples"][0]["error"]["message"]
    assert (out_continue / "campaign_summary.json").is_file()
    json.dumps(summary, allow_nan=False, ensure_ascii=True)

    # Existing non-empty output root refuses overwrite without force=True.
    with pytest.raises(FileExistsError):
        run_campaign(cfg_path, out_continue, force=False)

    # Fail-fast marks remaining samples as skipped.
    out_fail_fast = tmp_path / "out_fail_fast"
    halted = run_campaign(cfg_path, out_fail_fast, continue_on_error=False)
    assert halted["n_failed"] == 1
    assert halted["n_skipped"] == 2
    assert [s["status"] for s in halted["samples"]] == ["failed", "skipped", "skipped"]


def test_run_campaign_end_to_end_with_real_synthetic_frames(tmp_path: Path) -> None:
    from butterfly_saxs.pipeline import synthetic_butterfly

    for sample_name in ("sample_a", "sample_b"):
        sample_dir = tmp_path / "inputs" / sample_name
        sample_dir.mkdir(parents=True)
        for frame_idx in range(2):
            arr, qmap = synthetic_butterfly((32, 32), return_qmap=True, seed=frame_idx + 1)
            np.savez_compressed(
                sample_dir / f"{sample_name}_frame{frame_idx:02d}.npz",
                data=arr,
                qx=qmap["qx"],
                qy=qmap["qy"],
                q=qmap["q"],
                q_unit=np.asarray("pixel-q"),
            )

    cfg_path = tmp_path / "real_campaign.toml"
    init_campaign_config(
        [
            {"name": "sample_a", "input_dir": "inputs/sample_a"},
            {"name": "sample_b", "input_dir": "inputs/sample_b"},
        ],
        cfg_path,
        preset="curvature",
    )
    out_root = tmp_path / "campaign_out"
    summary = run_campaign(cfg_path, out_root)
    assert summary["n_samples"] == 2
    assert summary["n_failed"] == 0
    assert (out_root / "campaign_summary.json").is_file()
    for sample_rec in summary["samples"]:
        assert sample_rec["n_frames"] == 2
        assert Path(sample_rec["output_dir"]).is_dir()
    json.dumps(summary, allow_nan=False, ensure_ascii=True)


def test_butterfly_summary_renders_remediation_bilingually_and_lazily(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PySide6")
    from butterfly_saxs import butterfly_quality
    from butterfly_saxs.ui.butterfly_summary import (
        ButterflyQualitySummary,
        build_butterfly_quality_summary,
    )

    payload_with_remediation = {
        "q_unit": "nm^-1",
        "settings": {"stage": "evaluate"},
        "quality": {
            "status": "WARN",
            "engineering_status": "WARN",
            "flags": ["annular_outer_window_truncated"],
            "remediation": [
                {
                    "flag": "annular_outer_window_truncated",
                    "severity": "warn",
                    "summary_en": "Increase q_max to capture petal tips",
                    "summary_zh": "增大 q_max 以覆盖花瓣外端",
                    "recommended_action": "expand_q_window",
                    "suggested_parameters": {"q_max": 1.6},
                }
            ],
        },
        "candidate_fit": {"axis_ratio": 0.35, "q_star_from_arcs": 0.25},
    }

    card = ButterflyQualitySummary(language="zh_CN")
    qtbot.addWidget(card)
    card.set_state(payload_with_remediation, stage="evaluate")
    assert len(card.state.remediation) == 1
    assert card.remediation_label.isVisible() or card.remediation_label.text()
    assert "建议调整：" in card.remediation_label.text()
    assert "增大 q_max" in card.remediation_label.text()
    assert "q_max=1.6" in card.remediation_label.text()

    card.set_language("en")
    assert "Suggestions: " in card.remediation_label.text()
    assert "Increase q_max" in card.remediation_label.text()
    assert "q_max=1.6" in card.remediation_label.text()

    # Stale/cancelled state clears remediation cleanly.
    card.clear(data_ready=False)
    assert card.state.remediation == ()
    assert card.remediation_label.text() == ""

    # Lazy fallback to butterfly_quality.build_quality_remediation when payload lacks remediation.
    monkeypatch.setattr(
        butterfly_quality,
        "build_quality_remediation",
        lambda flags, metrics=None: [
            {
                "flag": "axis_ratio_at_bound",
                "severity": "warn",
                "summary_en": "Relax axis_ratio_min bound",
                "summary_zh": "放宽轴比下限",
                "recommended_action": "relax_bounds",
                "suggested_parameters": {"axis_ratio_min": 0.01},
            }
        ],
        raising=False,
    )
    state = build_butterfly_quality_summary(
        {
            "q_unit": "nm^-1",
            "settings": {"stage": "evaluate"},
            "quality": {"status": "WARN", "flags": ["axis_ratio_at_bound"]},
            "candidate_fit": {"axis_ratio": 0.005},
        },
        stage="evaluate",
    )
    assert len(state.remediation) == 1
    assert "Relax axis_ratio_min bound" in state.remediation_text(english=True)
    assert "放宽轴比下限" in state.remediation_text(english=False)
    card.close()


def test_figure_export_dialog_journal_style_selector_and_i18n(tmp_path: Path, qtbot) -> None:
    pytest.importorskip("PySide6")
    from PySide6 import QtCore
    from butterfly_saxs.ui.figure_export_dialog import FigureExportDialog, _JOURNAL_STYLES

    dialog = FigureExportDialog(language="en", default_parent=tmp_path)
    qtbot.addWidget(dialog)

    # All existing widget names and defaults preserved.
    assert dialog.width_preset is dialog.width_combo
    assert dialog.width_combo.currentData() == 183.0
    assert dialog.dpi_combo.currentData() == 600
    assert [
        dialog.journal_style_combo.itemData(i)
        for i in range(dialog.journal_style_combo.count())
    ] == list(_JOURNAL_STYLES)
    assert dialog.journal_style_combo.currentData() == "default"
    assert dialog.journal_style_combo.itemText(1) == "Nature (89/183 mm)"
    assert dialog.settings()["journal_style"] == "default"

    # Switch to Nature and verify emitted payload.
    emitted: list[dict[str, Any]] = []
    dialog.startRequested.connect(emitted.append)
    nature_idx = dialog.journal_style_combo.findData("nature")
    dialog.journal_style_combo.setCurrentIndex(nature_idx)
    assert dialog.settings()["journal_style"] == "nature"

    dialog.start_button.click()
    qtbot.wait(10)
    assert len(emitted) == 1
    assert emitted[0]["journal_style"] == "nature"
    assert emitted[0]["width_mm"] == 183.0
    assert emitted[0]["dpi"] == 600
    assert not dialog.journal_style_combo.isEnabled()

    dialog.set_export_running(False)
    assert dialog.journal_style_combo.isEnabled()

    # Bilingual retranslation + tooltip coverage.
    en_tip = dialog.journal_style_combo.toolTip()
    en_item_tip = dialog.journal_style_combo.itemData(
        nature_idx, QtCore.Qt.ItemDataRole.ToolTipRole
    )
    dialog.set_language("zh_CN")
    assert dialog._journal_style_label.text() == "期刊风格"
    assert dialog.journal_style_combo.itemText(0) == "默认"
    assert dialog.journal_style_combo.toolTip() != en_tip
    assert (
        dialog.journal_style_combo.itemData(
            nature_idx, QtCore.Qt.ItemDataRole.ToolTipRole
        )
        != en_item_tip
    )
    dialog.close()
