"""Scalar-list snapshots retain generic conversion and mutable-result semantics."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs import batch, producer
from scripts.benchmark_checkpoint_writes import _strict_equal


@pytest.mark.parametrize("values", [
    [None, True, False, 0, 2**100, "实测", 0.0, -0.0, 1.0000000000000002,
     float.fromhex("0x0.0000000000001p-1022"), float("nan"), float("inf"), -float("inf")],
    [np.float64(-0.0), np.float32(.5), np.int64(3), Path("data/frame.tif"),
     np.array([0.0, -0.0, 1.5]), (1, -0.0), {"nested": [1, None, -0.0]}],
])
def test_scalar_list_copy_matches_generic_sequence(values):
    # Tuples take the original recursive sequence branch, including all
    # scalar/subclass/array fallbacks. Compare types and every float bit.
    expected = batch._checkpoint_safe(tuple(values))
    actual = batch._checkpoint_safe(values)
    _strict_equal(expected, actual)
    assert actual is not values


def test_nested_snapshot_copies_do_not_alias_inputs():
    raw = {"profiles": [[-0.0, 1.0]], "parameters": {"a": 1.0}}
    first = batch._checkpoint_safe(raw)
    raw["profiles"][0][0] = True
    raw["parameters"]["a"] = 2
    second = batch._checkpoint_safe(raw)
    _strict_equal(first, {"profiles": [[-0.0, 1.0]], "parameters": {"a": 1.0}})
    _strict_equal(second, {"profiles": [[True, 1.0]], "parameters": {"a": 2}})


def test_container_subclass_array_and_dataclass_fallbacks_are_retained():
    class ArrayList(list):
        shape = (3,)
        dtype = np.dtype("float64")

    @dataclass
    class Result:
        profiles: list

        @property
        def parameters(self):
            return {"a": -0.0}

    value = ArrayList([1.0, 2.0, 3.0])
    descriptor = batch._checkpoint_safe(value)
    assert descriptor["array_omitted"] is True
    assert descriptor["shape"] == [3]
    _strict_equal(batch._checkpoint_safe(Result([None, -0.0])),
                  {"profiles": [None, -0.0], "parameters": {"a": -0.0}})


def test_scalar_profiles_do_not_dispatch_recursively_per_value(monkeypatch):
    original = batch._checkpoint_safe
    calls = []

    def counted(value):
        calls.append(type(value))
        return original(value)

    monkeypatch.setattr(batch, "_checkpoint_safe", counted)
    _strict_equal(counted({"profile": [float(i) for i in range(2048)]}),
                  {"profile": [float(i) for i in range(2048)]})
    assert calls == [dict, list]


@pytest.mark.parametrize("retain_results", [False, True])
def test_later_sink_mutation_is_included_in_every_new_snapshot(tmp_path, retain_results):
    paths = [tmp_path / f"frame{i}.dat" for i in range(2)]
    for path in paths:
        path.write_bytes(b"input")
    published = []

    def analyze(frame):
        if published:
            # An identity-only memo would lose all three changes below.
            previous = published[0]
            previous.result["profile"][0] = -0.0
            previous.result["profile"][1] = True
            previous.result["parameters"]["a"] = 2
        return {"profile": [0.0, 1], "parameters": {"a": 1.0}}

    checkpoint = tmp_path / "checkpoint.json"
    batch.run_batch(paths, analyze, checkpoint=checkpoint, retain_results=retain_results,
                    result_sink=published.append)
    result = batch.read_checkpoint(checkpoint)["frames"][0]["result"]
    _strict_equal(result, {"profile": [-0.0, True], "parameters": {"a": 2}})


@pytest.mark.parametrize("old_pair", sorted(producer._PREVIOUS_COMPACT_CHECKPOINT_SOURCES))
def test_previous_compact_producer_is_accepted_only_with_exact_other_evidence(old_pair):
    current = {**producer.producer_identity(), "analyzer": "fixture v1"}
    previous = copy.deepcopy(current)
    previous["sources"].update(zip(("batch.py", "producer.py"), old_pair, strict=True))
    before = copy.deepcopy(previous)
    assert producer.calculation_producers_compatible(previous, current)
    assert previous == before  # The original producer must not be rewritten.
    assert not producer.calculation_producers_compatible(current, previous)
    for section, key in [("sources", "pipeline.py"), ("versions", "numpy")]:
        invalid = copy.deepcopy(previous)
        invalid[section][key] = "changed"
        assert not producer.calculation_producers_compatible(invalid, current)
    invalid = copy.deepcopy(previous)
    invalid["analyzer"] = "different"
    assert not producer.calculation_producers_compatible(invalid, current)
