from __future__ import annotations

import math

import pytest

from butterfly_saxs.parameters import ParameterSet, ParameterSpec, default_ellipse_parameters


def test_expression_resolution_and_default_physical_parameterization() -> None:
    parameters = default_ellipse_parameters(a=2.0, axis_ratio=0.4, theta=math.radians(18.0))
    values = parameters.resolve()

    assert values["b"] == pytest.approx(0.8)
    assert values["theta_deg"] == pytest.approx(18.0)
    assert parameters["b"].is_tied
    assert "b" not in parameters.free_names


def test_parameter_expression_unknown_name_and_cycle_fail() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ParameterSet({"x": ParameterSpec(1.0), "y": ParameterSpec(expr="missing")})
    with pytest.raises(ValueError, match="cyclic"):
        ParameterSet({"x": ParameterSpec(expr="y"), "y": ParameterSpec(expr="x")})


def test_fixed_and_bound_values_are_checked() -> None:
    parameters = ParameterSet(
        {
            "x": ParameterSpec(2.0, min=0.0, max=3.0, vary=False),
            "y": {"value": 3.0, "expr": "x+1"},
        }
    )
    assert parameters.fixed_names == ("x",)
    assert parameters.resolve()["y"] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        parameters.update_values({"y": 1.0})


def test_expression_cache_topological_cache_and_fast_copy() -> None:
    from butterfly_saxs.parameters import _expression_names, _expression_tree

    with pytest.raises(ValueError, match="non-empty string"):
        _expression_tree(["not", "hashable"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        _expression_tree("")

    names_a = _expression_names("a * axis_ratio")
    names_a.clear()
    assert _expression_names("a * axis_ratio") == {"a", "axis_ratio"}

    params = default_ellipse_parameters(a=2.5, axis_ratio=0.35, theta=math.radians(22.0))
    topo_1 = params.topological_order()
    resolved_1 = params.resolve()
    assert resolved_1["b"] == pytest.approx(0.875)
    params.set_free_vector([0.1, -0.2, 3.0, 0.4, math.radians(25.0)])
    assert params.topological_order() is topo_1
    resolved_2 = params.resolve()
    assert resolved_2["b"] == pytest.approx(1.2)
    assert resolved_2["theta_deg"] == pytest.approx(25.0)

    clone = params._fast_copy()
    assert clone.resolve() == resolved_2
    assert clone.free_names == params.free_names
    assert clone.fixed_names == params.fixed_names
    assert clone.tied_names == params.tied_names
    clone["a"].set_value(4.0)
    assert params["a"].value == pytest.approx(3.0)
    assert clone.resolve()["b"] == pytest.approx(1.6)

    clone["extra"] = {"expr": "b * 2.0"}
    topo_updated = clone.topological_order()
    assert "extra" in topo_updated and topo_updated.index("b") < topo_updated.index("extra")
    assert clone.resolve()["extra"] == pytest.approx(3.2)
