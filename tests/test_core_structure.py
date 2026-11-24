"""Structural, import-cycle, facade-parity, and leaf-primitive tests for Phase 3."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import butterfly_saxs
from butterfly_saxs import _primitives


PACKAGE_ROOT = Path(butterfly_saxs.__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. _primitives behavior & zero-heavy-import contract
# ---------------------------------------------------------------------------


def test_primitives_and_butterfly_settings_load_zero_heavy_dependencies() -> None:
    code = (
        "import sys, butterfly_saxs._primitives, butterfly_saxs.butterfly_settings; "
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "{'numpy', 'scipy', 'matplotlib', 'PySide6', 'PyQt6', 'pyFAI', 'h5py'}]; "
        "assert not heavy, f'Unexpected heavy imports: {heavy}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_primitives_finite_float_and_strict_int() -> None:
    assert _primitives.finite_float(True) is None
    assert _primitives.finite_float(False, -1.0) == -1.0
    assert _primitives.finite_float(np.bool_(True)) is None
    assert _primitives.finite_float(np.asarray(True)) is None
    assert _primitives.finite_float(b"1.5") is None
    assert _primitives.finite_float(float("nan")) is None
    assert _primitives.finite_float(float("inf")) is None
    assert _primitives.finite_float(np.float64(2.5)) == pytest.approx(2.5)
    assert _primitives.finite_float(np.asarray(3.25)) == pytest.approx(3.25)
    assert _primitives.finite_float("1.25") == pytest.approx(1.25)

    assert _primitives.strict_int(4, "n", minimum=1, maximum=10) == 4
    assert _primitives.strict_int(4.0, "n") == 4
    assert _primitives.strict_int(np.int64(7), "n") == 7
    assert _primitives.strict_int("+12", "n") == 12
    for bad in (True, False, np.bool_(True), 1.5, "1.5", float("nan"), float("inf"), None):
        with pytest.raises(ValueError):
            _primitives.strict_int(bad, "n")
    with pytest.raises(ValueError, match=">="):
        _primitives.strict_int(0, "n", minimum=1)
    with pytest.raises(ValueError, match="<="):
        _primitives.strict_int(11, "n", maximum=10)


def test_primitives_q_unit_normalization_and_scale() -> None:
    for raw in ("nm^-1", "nm⁻1", "nm⁻¹", "1/nm", b"nm^-1", np.bytes_("nm^-1"), np.asarray(b"nm^-1")):
        assert _primitives.q_scale_to_nm_inv(raw) == pytest.approx(1.0)
    assert _primitives.normalize_q_unit("nm⁻1") == "nm^-1"
    assert _primitives.normalize_q_unit("nm⁻¹") == "nm^-1"
    assert _primitives.normalize_q_unit("Å⁻¹") == "a^-1"

    for ang in ("Å⁻¹", "Å^-1", "A^-1", "1/A", "angstrom^-1", b"A^-1"):
        assert _primitives.q_scale_to_nm_inv(ang) == pytest.approx(10.0)

    for uncal in ("pixel", "pixel-q", "px", "unknown", "", None, "arb"):
        assert _primitives.q_scale_to_nm_inv(uncal) is None


def test_primitives_parse_q_window_gradient_and_q_step() -> None:
    assert _primitives.parse_q_window((0.1, 0.5)) == (0.1, 0.5)
    assert _primitives.parse_q_window({"q_min": 0.2, "q_max": 0.8}) == (0.2, 0.8)
    assert _primitives.parse_q_window(None, allow_none=True) is None

    q_arr = np.array([[0.15, 0.25], [0.35, float("nan")]])
    assert _primitives.parse_q_window(None, q_arr) == (0.15, 0.35)
    empty_q = np.full((2, 2), float("nan"))
    assert _primitives.parse_q_window(
        None, empty_q, empty_fallback=(float("nan"), float("nan"))
    ) == pytest.approx((float("nan"), float("nan")), nan_ok=True)

    for invalid in ((0.5, 0.1), (0.2, 0.2), (True, 0.5), "0.1,0.5"):
        with pytest.raises(ValueError):
            _primitives.parse_q_window(invalid)
    with pytest.raises(ValueError, match="non-negative"):
        _primitives.parse_q_window((-0.1, 0.5), non_negative=True)

    # 1-pixel axis safe gradient
    row_vec = np.array([[1.0, 3.0, 6.0]])
    assert np.all(_primitives.safe_gradient(row_vec, 0) == 0.0)
    assert _primitives.safe_gradient(row_vec, 1).shape == (1, 3)

    # 90-degree rotated detector q-step
    coords = np.linspace(-0.4, 0.4, 9)  # step = 0.1
    qy, qx = np.meshgrid(coords, coords)
    assert _primitives.representative_q_step({"qx": qx, "qy": qy}) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 2. Zero circular imports across Core / Observables / Physics / Tracing / Lamellar
# ---------------------------------------------------------------------------


def _module_name_for_path(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["butterfly_saxs", *parts])


def _resolve_relative_module(current_mod: str, is_pkg: bool, level: int, module: str | None) -> str:
    anchor = current_mod.split(".") if is_pkg else current_mod.split(".")[:-1]
    if level > 1:
        anchor = anchor[: -(level - 1)]
    if module:
        return ".".join([*anchor, module])
    return ".".join(anchor)


def test_zero_import_cycles_across_refactored_families() -> None:
    """Verify both top-level and function-level imports form a cycle-free graph across owned families."""
    tracked_files: list[Path] = []
    top_names = {
        "_primitives.py",
        "_observables_types.py",
        "_observables_angular.py",
        "_observables_ridges.py",
        "_observables_ellipse.py",
        "observables.py",
        "saxs_physics.py",
        "butterfly_ridge.py",
        "butterfly.py",
        "butterfly_quality.py",
        "butterfly_diagnostics.py",
        "butterfly_uncertainty.py",
        "butterfly_settings.py",
        "ridge_inputs.py",
        "ridge_profiles.py",
        "annular_trace.py",
        "annular_measurements.py",
        "sector_peaks.py",
        "sector_trace.py",
        "peak_landmarks.py",
        "public_ellipse.py",
        "intensity.py",
        "arc_geometry.py",
        "arc_support.py",
        "parameters.py",
        "lamellar_utils.py",
        "lamellar_models.py",
        "lamellar_adapter.py",
        "lamellar_sources.py",
        "lamellar_analysis.py",
        "lamellar_scattering.py",
        "lamellar_render.py",
        "lamellar_export.py",
        "lamellar_report_figures.py",
    }
    for name in sorted(top_names):
        candidate = PACKAGE_ROOT / name
        assert candidate.is_file(), f"Expected module {name} to exist"
        tracked_files.append(candidate)
    for subpkg in ("physics", "tracing", "lamellar"):
        subdir = PACKAGE_ROOT / subpkg
        assert subdir.is_dir(), f"Expected subpackage {subpkg}/ to exist"
        tracked_files.extend(sorted(subdir.glob("*.py")))

    # Ensure monolithic lamellar.py was removed in favor of lamellar/ package
    assert not (PACKAGE_ROOT / "lamellar.py").exists()

    known_modules = {_module_name_for_path(p): p for p in tracked_files}
    graph: dict[str, set[str]] = {mod: set() for mod in known_modules}

    for mod_name, path in known_modules.items():
        is_pkg = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    base_target = _resolve_relative_module(mod_name, is_pkg, node.level, node.module)
                elif node.module and node.module.startswith("butterfly_saxs"):
                    base_target = node.module
                else:
                    continue
                for alias in node.names:
                    sub_candidate = f"{base_target}.{alias.name}"
                    if sub_candidate in known_modules and sub_candidate != mod_name:
                        graph[mod_name].add(sub_candidate)
                    elif base_target in known_modules and base_target != mod_name:
                        graph[mod_name].add(base_target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in known_modules and alias.name != mod_name:
                        graph[mod_name].add(alias.name)

    # DFS cycle detection over ALL imports (including lazy function-level imports!)
    state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=done
    stack: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            st = state.get(nxt, 0)
            if st == 1:
                idx = stack.index(nxt)
                cycles.append([*stack[idx:], nxt])
            elif st == 0:
                dfs(nxt)
        stack.pop()
        state[node] = 2

    for mod in sorted(graph):
        if state.get(mod, 0) == 0:
            dfs(mod)

    assert not cycles, f"Detected circular imports: {cycles}"

    # Verify butterfly_settings no longer imports settings or analysis_config
    assert "butterfly_saxs.settings" not in graph["butterfly_saxs.butterfly_settings"]
    assert "butterfly_saxs.analysis_config" not in graph["butterfly_saxs.butterfly_settings"]
    assert graph["butterfly_saxs.butterfly_settings"] == {"butterfly_saxs._primitives"}

    # Verify clean runtime import of every tracked module
    for mod in sorted(known_modules):
        imported = importlib.import_module(mod)
        assert imported is not None


def test_cli_lazy_imports_and_service_qt_free_boundaries() -> None:
    cli_check = (
        "import sys, butterfly_saxs.cli; "
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "{'scipy', 'matplotlib', 'PySide6', 'PyQt6', 'pyFAI', 'h5py'}]; "
        "assert not heavy, f'CLI loaded heavy modules at import time: {heavy}'"
    )
    proc_cli = subprocess.run([sys.executable, "-c", cli_check], capture_output=True, text=True, check=False)
    assert proc_cli.returncode == 0, proc_cli.stderr or proc_cli.stdout

    svc_check = (
        "import sys, butterfly_saxs.service; "
        "qt = [m for m in sys.modules if m.split('.')[0] in {'PySide6', 'PyQt6', 'shiboken6'}]; "
        "assert not qt, f'service.py loaded Qt modules: {qt}'"
    )
    proc_svc = subprocess.run([sys.executable, "-c", svc_check], capture_output=True, text=True, check=False)
    assert proc_svc.returncode == 0, proc_svc.stderr or proc_svc.stdout


# ---------------------------------------------------------------------------
# 3. Facade symbol parity, backward-compatible shims, and dead-code removal
# ---------------------------------------------------------------------------


def test_observables_physics_tracing_and_lamellar_facade_parity() -> None:
    from butterfly_saxs import (
        arc_geometry,
        arc_support,
        butterfly_ridge,
        butterfly_uncertainty,
        lamellar,
        lamellar_adapter,
        lamellar_analysis,
        lamellar_export,
        lamellar_models,
        lamellar_render,
        lamellar_report_figures,
        lamellar_scattering,
        lamellar_sources,
        lamellar_utils,
        observables,
        physics,
        saxs_physics,
        tracing,
    )

    # 1. observables facade parity
    required_obs_privates = (
        "_azimuthal_peak_ridges",
        "_symmetry_diagnostics",
        "_radial_peak",
        "_radial_peak_candidates",
        "_extract_maps",
        "_q_limits",
        "_q_unit",
        "_q_to_nm_inverse_scale",
        "_normalized_q_unit",
        "_SPACING_CENTER_REL_TOL",
        "_linear_fwhm",
        "_periodic_fwhm",
        "_annotate_ridge_continuity",
        "_write_fit_branch_ids",
        "_measure_lobe_radial_observables",
        "_quadratic_peak",
        "_reject_ridge_point",
    )
    for sym in (*observables.__all__, *required_obs_privates):
        assert hasattr(observables, sym), f"observables missing {sym}"

    # 2. saxs_physics <-> physics parity
    assert set(saxs_physics.__all__) == set(physics.__all__)
    for sym in (*physics.__all__, "_fit_1d_transverse_gaussian", "_radial_metrics_from_1d_profile"):
        assert getattr(saxs_physics, sym) is getattr(physics, sym)

    # 3. butterfly_ridge <-> tracing parity
    assert hasattr(tracing, "curvature") and hasattr(tracing, "topology") and hasattr(tracing, "radial_hint")
    for sym in ("_scaled_surface_field", "_raw_candidates", "_nms", "_annotate_scale_stability"):
        assert getattr(butterfly_ridge, sym) is getattr(tracing.curvature, sym)
    for sym in ("_apply_edits", "_apply_seeds", "_arc_topology", "_graph_arcs", "_ordered_component", "_parse_q_window"):
        assert getattr(butterfly_ridge, sym) is getattr(tracing.topology, sym)
    for sym in ("_RadialHintGeometryCache", "_first_order_q_hint", "_fill_sparse_first_order_ring", "_refine_radial_peak_q"):
        assert getattr(butterfly_ridge, sym) is getattr(tracing.radial_hint, sym)

    # 4. lamellar package & shim parity
    assert hasattr(lamellar, "__path__")
    for sym in ("LamellarScene", "LamellarSettings", "build_lamellar_scene", "load_lamellar_sources"):
        assert hasattr(lamellar, sym)
    assert lamellar_models.LamellarScene is lamellar.LamellarScene
    assert lamellar_models.LamellarSettings is lamellar.LamellarSettings
    assert lamellar_sources.load_lamellar_sources is lamellar.load_lamellar_sources
    for sym in ("_scene_arrays", "_scene_payload", "_jsonable", "_parameter_rows", "export_lamellar_scene"):
        assert hasattr(lamellar_export, sym)
    for sym in ("_camera_orientation", "_convex_hull_2d", "_font", "render_lamellar_2d"):
        assert hasattr(lamellar_render, sym)
    assert hasattr(lamellar_adapter, "observed_direction_support")
    assert hasattr(lamellar_analysis, "analyze_lamellar_morphology")
    assert hasattr(lamellar_scattering, "simulate_projected_density_fft")
    assert hasattr(lamellar_report_figures, "render_lamellar_report_frame")
    assert hasattr(lamellar_utils, "_normalise_q_unit")

    # 5. Verified dead symbols removed
    assert not hasattr(observables, "_legacy_fit_symmetric_double_ellipse")
    assert not hasattr(saxs_physics, "_validate_2d_inputs")
    assert not hasattr(arc_geometry, "_arc_interval")
    assert not hasattr(arc_support, "_same_id")
    assert not hasattr(butterfly_uncertainty, "_json_number")
    assert not hasattr(lamellar_render, "_centres")
    assert not hasattr(lamellar.render, "_centres")


def test_peak_landmarks_bridges_coarse_lattice_holes_without_masked_cliff_artifacts() -> None:
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.peak_landmarks import compute_peak_landmarks
    from butterfly_saxs.service import ButterflyAnalysisService

    # 1. Coarse 48x48 unmasked frame with 2-deg bins (Farey lattice empty bins at 45/90 deg)
    case = generate_arc_case("ellipse_ratio_400", seed=231, shape=(48, 48))
    service = ButterflyAnalysisService(
        analysis_settings={
            "ridge_method": "butterfly_curvature",
            "q_window": [0.05, 1.1],
            "ellipse_multistart": 2,
            "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
        }
    )
    state = service.set_observed(case["image"], qmap=case["qmap"])
    landmarks = service.measure_geometry(payload=state)["butterfly"]["peak_landmarks"]
    assert landmarks["status"] == "ok"
    assert len(landmarks["peaks"]) >= 2

    # 2. Coarse 48x48 isotropic ring with a masked wedge: no edge cliff peaks, unsupported stay None
    qx = np.asarray(case["qmap"].qx, dtype=float)
    qy = np.asarray(case["qmap"].qy, dtype=float)
    q = np.hypot(qx, qy)
    chi = np.degrees(np.arctan2(qy, qx)) % 360.0
    ring = 5.0 + 80.0 * np.exp(-0.5 * ((q - 0.40) / 0.04) ** 2)
    valid = ~((chi >= 60.0) & (chi <= 110.0))
    masked_res = compute_peak_landmarks(
        ring, qx, qy, valid_mask=valid, q_window=(0.25, 0.55), q_unit="nm^-1"
    )
    for val, sup in zip(
        masked_res["profiles"]["angular"]["intensity_smoothed"],
        masked_res["profiles"]["angular"]["supported"],
        strict=True,
    ):
        if not sup:
            assert val is None
    assert masked_res["peaks"] == []
