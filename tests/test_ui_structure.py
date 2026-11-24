"""Structural, modularization, and Qt-boundary verification tests for ``butterfly_saxs.ui``."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs.ui._batch_records import (
    _batch_record_geometry,
    _batch_records_have_failure,
    _batch_warning_has_explicit_failure,
    _classify_evolution_row,
    _evolution_numeric_keys,
    _flatten_evolution_record,
    _result_has_failure,
)
from butterfly_saxs.ui._butterfly_edits import parse_numeric_edit_inputs
from butterfly_saxs.ui._common import (
    _read,
    confirm_save_path,
    display_q_unit,
    is_qt_valid,
    json_safe,
    normalize_save_path,
    safe_single_shot,
    same_q_unit,
    unique_export_dir,
)
from butterfly_saxs.ui._project_io import restore_engine_document, snapshot_engine_document
from butterfly_saxs.ui.butterfly_figure_export import new_figure_export_target
from butterfly_saxs.ui.i18n import CATALOGS
from butterfly_saxs.ui.qt_compat import QT_AVAILABLE, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "src" / "butterfly_saxs" / "ui"


def test_batch_records_module_is_pure_python_and_headless_safe() -> None:
    """``ui._batch_records`` and ``ui`` package top-level must not import Qt."""

    source = (UI_DIR / "_batch_records.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        token in name.lower()
        for name in imported
        for token in ("pyside", "pyqt", "qt_compat", "matplotlib")
    )

    probe = (
        "import sys; sys.modules['PySide6'] = None; "
        "import butterfly_saxs.ui as ui; "
        "import butterfly_saxs.ui._batch_records as br; "
        "import butterfly_saxs.ui.project_document as pd; "
        "assert ui.QT_AVAILABLE is False; "
        "assert callable(br._batch_record_geometry); "
        "assert callable(pd.ProjectDocumentController)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    lazy_probe = (
        "import sys; "
        "import butterfly_saxs.ui as ui; "
        "import butterfly_saxs.ui._common as cm; "
        "import butterfly_saxs.ui._batch_records as br; "
        "import butterfly_saxs.ui.project_document as pd; "
        "loaded_qt = [m for m in sorted(sys.modules) if m.startswith(('PySide6', 'shiboken6', 'pyqtgraph'))]; "
        "assert not loaded_qt, f'unexpected eager Qt imports: {loaded_qt}'; "
        "assert callable(cm._read) and callable(br._batch_record_geometry) and callable(pd.ProjectDocumentController)"
    )
    proc_lazy = subprocess.run(
        [sys.executable, "-c", lazy_probe],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_lazy.returncode == 0, proc_lazy.stderr


def test_common_helpers_read_units_json_and_export_dirs(tmp_path: Path) -> None:
    mapping = {"zero": 0, "empty": "", "flag": False, "nested": "ok"}
    ns = SimpleNamespace(alpha=42, flag=False)

    assert _read(mapping, "zero", 99) == 0
    assert _read(mapping, ("missing", "empty"), "fallback") == ""
    assert _read(mapping, ("missing", "flag"), True) is False
    assert _read(mapping, ("missing",), "default") == "default"
    assert _read(ns, "alpha") == 42
    assert _read(ns, ("missing", "flag"), True) is False
    assert _read(ns, ("missing",), None) is None

    # No duplicate local def _read(...) across consolidated UI modules.
    for consolidated in (
        "models.py",
        "views.py",
        "qspace.py",
        "lamellar_3d.py",
        "butterfly_summary.py",
        "butterfly_workbench.py",
        "main_window.py",
        "publication_qt_geometry.py",
    ):
        txt = (UI_DIR / consolidated).read_text(encoding="utf-8")
        assert "def _read(" not in txt, consolidated
        assert "def _read_value(" not in txt, consolidated

    for no_dup_json in ("butterfly_export.py", "main_window.py"):
        txt = (UI_DIR / no_dup_json).read_text(encoding="utf-8")
        assert "def _jsonable(" not in txt, no_dup_json

    assert display_q_unit("1/nm") == "nm^-1"
    assert display_q_unit("nm⁻¹") == "nm^-1"
    assert display_q_unit("Å⁻¹") == "A^-1"
    assert display_q_unit("pixel") == "pixel-q"
    assert same_q_unit("1/nm", "nm^-1")
    assert same_q_unit("1/A", "Å⁻¹")
    assert not same_q_unit("nm^-1", "pixel-q")

    payload = json_safe(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "path": tmp_path / "x.json",
            "arr": np.asarray([1.0, float("nan")]),
        }
    )
    assert payload == {
        "nan": None,
        "inf": None,
        "path": str(tmp_path / "x.json"),
        "arr": [1.0, None],
    }

    first = unique_export_dir(tmp_path, "bundle")
    assert first == tmp_path.resolve() / "bundle"
    assert not first.exists()
    first.mkdir()
    second = unique_export_dir(tmp_path, "bundle")
    assert second == tmp_path.resolve() / "bundle-2"
    assert new_figure_export_target(tmp_path) == tmp_path.resolve() / "butterfly-figure"

    assert normalize_save_path(tmp_path / "out", ".csv") == tmp_path / "out.csv"
    assert normalize_save_path(tmp_path / "out.CSV", "csv") == tmp_path / "out.CSV"
    assert callable(is_qt_valid) and callable(safe_single_shot)


@pytest.mark.skipif(not QT_AVAILABLE, reason="PySide6 is unavailable")
def test_confirm_save_path_handles_new_and_existing_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fresh = confirm_save_path(None, tmp_path / "new_file", suffix=".csv")
    assert fresh == (tmp_path / "new_file.csv", False)

    existing = tmp_path / "data.csv"
    companion = tmp_path / "data_peaks.csv"
    companion.write_text("peaks", encoding="utf-8")

    prompts: list[str] = []

    def decline(_parent, _title, text, *_args):
        prompts.append(str(text))
        return QtWidgets.QMessageBox.StandardButton.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", decline)
    assert (
        confirm_save_path(
            None,
            tmp_path / "data",
            suffix=".csv",
            extra_targets=lambda p: (p.with_name(f"{p.stem}_peaks.csv"),),
            prompt="Overwrite bundle?",
        )
        is None
    )
    assert prompts == ["Overwrite bundle?"]

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_a, **_kw: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    accepted = confirm_save_path(
        None,
        existing,
        suffix=".csv",
        extra_targets=lambda p: (p.with_name(f"{p.stem}_peaks.csv"),),
    )
    assert accepted == (existing, True)


def test_batch_records_pure_logic_and_evolution_classification() -> None:
    rec = {
        "status": "ok",
        "quality_status": "PASS",
        "geometry_parameters": {"a": 0.25, "b": 0.15, "theta_deg": 12.0},
        "quantitative_parameters": {
            "Ln_from_minor_axis_nm": {
                "value": 41.8,
                "status": "candidate",
                "confidence": "limited",
                "publication_status": "not_assessed",
            }
        },
        "metrics": {"rmse": 0.012},
    }
    geom = _batch_record_geometry(rec)
    assert geom["a"] == pytest.approx(0.25)
    assert geom["b"] == pytest.approx(0.15)
    assert geom["axis_ratio"] == pytest.approx(0.6)
    assert geom["ln"] is None
    assert geom["ln_candidate"] == pytest.approx(41.8)

    assert not _result_has_failure(rec)
    assert _result_has_failure({"status": "failed"})
    assert _batch_warning_has_explicit_failure({"status": "warning", "optimizer_error": "singular"})
    assert not _batch_records_have_failure([{"status": "warning", "quality_status": "WARN"}])
    assert _batch_records_have_failure([{"status": "warning", "solver_status": "failed"}])

    flat = _flatten_evolution_record(rec, 0)
    keys = list(flat.keys())
    num_keys = _evolution_numeric_keys([flat], keys)
    assert "a" in num_keys and "rmse" in num_keys
    assert _classify_evolution_row(flat, "a") == "measured"
    assert _classify_evolution_row(flat, "Ln_from_minor_axis_nm") == "limited"
    assert _classify_evolution_row({"status": "failed", "a": 0.2}, "a") == "failed"


def test_numeric_edit_parser_validates_finite_coordinates_and_shapes() -> None:
    seed, err = parse_numeric_edit_inputs("seed", "0.15", "-0.25", "", english=True)
    assert err is None and seed == {"type": "seed", "qx": 0.15, "qy": -0.25}

    for bad in ("nan", "inf", "-inf", "abc", ""):
        parsed, err_msg = parse_numeric_edit_inputs("seed", bad, "0.1", "", english=False)
        assert parsed is None and err_msg

    rect, err = parse_numeric_edit_inputs(
        "exclude_rectangle", "", "", "0.0, 0.1; 0.2, 0.3", english=True
    )
    assert err is None
    assert rect == {
        "type": "exclude_polygon",
        "points": [[0.0, 0.1], [0.2, 0.1], [0.2, 0.3], [0.0, 0.3]],
    }

    bad_poly, err = parse_numeric_edit_inputs(
        "include_polygon", "", "", "0.0, 0.1; 0.2, 0.3", english=True
    )
    assert bad_poly is None and err is not None


def test_i18n_catalog_parity_and_no_dead_keys() -> None:
    en_keys = set(CATALOGS["en"])
    zh_keys = set(CATALOGS["zh_CN"])
    assert en_keys == zh_keys

    for key in en_keys:
        assert set(re.findall(r"{([^{}]+)}", CATALOGS["en"][key])) == set(
            re.findall(r"{([^{}]+)}", CATALOGS["zh_CN"][key])
        ), key

    corpus_parts: list[str] = []
    for root in (ROOT / "src", ROOT / "tests"):
        for py_file in root.rglob("*.py"):
            if py_file.name in {"i18n.py", "test_ui_structure.py"}:
                continue
            corpus_parts.append(py_file.read_text(encoding="utf-8-sig"))
    corpus = "\n".join(corpus_parts)
    i18n_source = (UI_DIR / "i18n.py").read_text(encoding="utf-8-sig")

    dynamic_prefixes = (
        "tooltip.parameter.",
        "journal_style.",
        "tooltip.journal_style.",
        "profile.series.",
        "manual.",
        "job.",
    )
    unreferenced = [
        key
        for key in sorted(en_keys)
        if key not in corpus
        and not key.startswith(dynamic_prefixes)
        and f'"{key}"' not in i18n_source.split("CATALOGS =", 1)[1]
    ]
    assert unreferenced == []


def test_engine_snapshot_uses_service_contract_without_private_scraping() -> None:
    from butterfly_saxs.service import ButterflyAnalysisService

    svc = ButterflyAnalysisService()
    snap = snapshot_engine_document(svc)
    assert "state" in snap
    assert "_poni" not in snap and "_geometry_cache" not in snap
    svc.set_analysis_settings({"q_min": 0.123})
    assert svc._analysis_settings["q_min"] == pytest.approx(0.123)
    restore_engine_document(svc, snap)
    assert svc._analysis_settings["q_min"] is None

    # Minimal mock engine without snapshot_state remains safe.
    mock_engine = object()
    assert snapshot_engine_document(mock_engine) == {}
    restore_engine_document(mock_engine, {})


@pytest.mark.skipif(not QT_AVAILABLE, reason="PySide6 is unavailable")
def test_decomposed_ui_facades_and_measurement_base_inheritance(qtbot) -> None:
    from butterfly_saxs.ui import MainWindow
    from butterfly_saxs.ui._measurement_base import _MeasurementCanvasPage
    from butterfly_saxs.ui.azimuthal_page import AzimuthalPage
    from butterfly_saxs.ui.butterfly_workbench import ButterflyWorkbench, _ProfilePanel
    from butterfly_saxs.ui.density2d_page import Density2DPage
    import butterfly_saxs.ui.main_window as mw

    assert issubclass(AzimuthalPage, _MeasurementCanvasPage)
    assert issubclass(Density2DPage, _MeasurementCanvasPage)
    assert callable(mw.first_order_ring_overlay)
    assert callable(mw._batch_record_geometry)
    assert callable(mw._batch_records_have_failure)
    assert _ProfilePanel is not None

    az = AzimuthalPage(language="en")
    d2d = Density2DPage(language="en")
    qtbot.addWidget(az)
    qtbot.addWidget(d2d)
    assert az.objectName() == "azimuthalPage"
    assert d2d.objectName() == "density2dPage"
    assert az.canvas.objectName() == "azimuthalCanvas"
    assert d2d.canvas.objectName() == "density2dCanvas"

    window = MainWindow(engine=object(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    assert isinstance(window.butterfly_workbench, ButterflyWorkbench)
    for attr in (
        "measurements_page",
        "profile_tabs",
        "lobe_table",
        "ridge_table",
        "ellipse_table",
        "radial_table",
        "local_measurement_page",
        "azimuthal_page",
        "density2d_page",
    ):
        assert hasattr(window, attr), attr
    window.close()
