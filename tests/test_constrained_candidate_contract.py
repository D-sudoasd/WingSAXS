from __future__ import annotations

import pytest

from butterfly_saxs.butterfly_quality import ellipse_candidate_is_limited
from butterfly_saxs.candidate_geometry import _candidate_curves
from butterfly_saxs.service import ButterflyAnalysisService, _batch_record_flags


@pytest.mark.parametrize("limitation", [
    {"flags": ["axis_ratio_at_bound"]},
    {"flags": "axis_ratio_at_bound"},
    {"bound_status": {"a": True}},
    {"flags": ["minor_axis_not_resolved_against_localization"]},
])
def test_limited_candidate_cannot_inherit_stale_quantitative_acceptance(monkeypatch, limitation):
    candidate = {
        "success": True, "a": 1.0, "b": 0.3, "axis_ratio": 0.3, "theta_deg": 12.0,
        "Ln_from_minor_axis_nm": 20.0, "Lz_from_draw_axis_nm": 25.0,
        "L_from_major_axis_nm": 6.0, **limitation,
    }
    service = ButterflyAnalysisService()
    monkeypatch.setattr(service, "preview", lambda **kwargs: {
        "ellipse_fit": {"parameters": dict(candidate), "success": True},
        "butterfly": {
            "candidate_fit": candidate,
            "quality": {"status": "WARN", "scientific_status": "NOT_ACCEPTED"},
            "quantitative_parameters": {"b": {"status": "available", "value": 0.3}},
        },
        "parameters": {},
    })
    result = service.measure_geometry()
    geometry = result["geometry_parameters"]
    assert all(geometry[name] is None for name in ("a", "b", "axis_ratio", "theta_deg"))
    for key in ("Ln_from_minor_axis_nm", "Lz_from_draw_axis_nm", "L_from_major_axis_nm"):
        assert key not in geometry
        assert result["candidate_geometry_parameters"][key] == candidate[key]
    assert result["candidate_geometry_parameters"]["a"] == 1.0
    assert result["butterfly"]["candidate_fit"] == candidate
    assert result["butterfly"]["quality"]["scientific_status"] == "NOT_ACCEPTED"
    assert ellipse_candidate_is_limited(_batch_record_flags(result, result.get("metrics")))


def test_limited_overlay_metadata_does_not_claim_scientific_acceptance():
    from butterfly_saxs.fit_overlays import fit_geometry_layers

    result = {
        "candidate_fit": {
            "success": True, "a": 1.0, "b": 0.3, "axis_ratio": 0.3,
            "theta_deg": 12.0, "center_qx": 0.0, "center_qy": 0.0,
            "reference_axis_deg": 0.0,
            "flags": ["axis_ratio_at_bound"],
        },
        "quality": {"status": "WARN", "scientific_status": "PASS"},
    }
    state, curves, metadata = _candidate_curves(result)
    assert state == "candidate" and curves
    assert metadata["bound_or_support_limited"] is True
    assert metadata["scientific_status"] == "NOT_ACCEPTED"
    assessment = fit_geometry_layers(result)["assessment"]
    assert assessment["geometry_status"] == "bound_limited"
    assert assessment["scientific_status"] == "NOT_ACCEPTED"
    assert assessment["scientific_acceptance"] is False


def test_minor_axis_quality_limit_survives_compact_batch_flags():
    flag = "minor_axis_not_resolved_against_localization"
    assert flag in _batch_record_flags({
        "butterfly": {"quality": {"flags": [flag]}},
    }, {})


@pytest.mark.parametrize("failure", ["solver_flag", "solver_status", "quality"])
def test_failed_candidate_retains_values_only_on_diagnostic_surface(monkeypatch, failure):
    candidate = {"success": True, "a": 1.0, "b": 0.3, "axis_ratio": 0.3,
                 "theta_deg": 12.0, "Ln_from_minor_axis_nm": 20.0}
    quality = {"status": "WARN", "scientific_status": "NOT_ACCEPTED"}
    if failure == "solver_flag":
        candidate["success"] = False
    elif failure == "solver_status":
        candidate["status"] = "unconverged"
    else:
        quality["status"] = "FAIL"
    service = ButterflyAnalysisService()
    monkeypatch.setattr(service, "preview", lambda **kwargs: {
        "ellipse_fit": {"parameters": dict(candidate)},
        "butterfly": {"candidate_fit": candidate, "quality": quality},
        "parameters": {},
    })
    result = service.measure_geometry()
    assert result["candidate_geometry_parameters"]["a"] == 1.0
    assert result["butterfly"]["candidate_fit"] == candidate
    assert all(result["geometry_parameters"][name] is None
               for name in ("a", "b", "axis_ratio", "theta_deg"))
    assert "Ln_from_minor_axis_nm" not in result["geometry_parameters"]
