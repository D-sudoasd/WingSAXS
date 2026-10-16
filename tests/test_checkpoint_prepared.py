"""Snapshot conversion stays lossless without a second scientific-tree walk."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs import batch
from scripts.benchmark_checkpoint_writes import _strict_equal


@pytest.mark.parametrize('retain', [True, False])
def test_prepared_snapshot_matches_public_writer_and_observes_mutations(tmp_path, monkeypatch, retain):
    paths = [tmp_path / f'{i}.dat' for i in range(3)]
    for path in paths:
        path.write_text('input')
    items = []
    snapshots = []
    original = batch.write_checkpoint

    def sink(item):
        items.append(item)

    def analyze(frame):
        if items:
            # Sinks can retain the public item. Do not cache stale history.
            items[0].result['profile'][0] = -0.0
        return {'parameters': {'a': 0.5}, 'profile': [1.0, -0.0, np.nextafter(1., 2.),
                float('inf'), None, True, 7], 'image': np.arange(6).reshape(2, 3)}

    def write(path, payload):
        public = tmp_path / 'public.json'
        original(public, dict(payload))
        original(path, payload)
        _strict_equal(batch.read_checkpoint(public), batch.read_checkpoint(path))
        snapshots.append(batch.read_checkpoint(path))
        return Path(path)

    monkeypatch.setattr(batch, 'write_checkpoint', write)
    batch.run_batch(paths, analyze, checkpoint=tmp_path / 'checkpoint.json',
                    retain_results=retain, result_sink=sink)
    assert len(snapshots) == 4
    assert snapshots[0]['frames'][0]['result']['profile'][0] == 1.0
    assert snapshots[1]['frames'][0]['result']['profile'][0].hex() == (-0.0).hex()


def test_safe_conversion_handles_nested_and_non_builtin_leaves():
    @dataclass
    class Result:
        values: object

    value = [Result([np.float32(1.25), Path('input'), (1., float('nan'))]),
             {'x': np.array([1., float('inf')])}, {1, 2}, frozenset({3})]
    result = batch._checkpoint_safe(value)
    _strict_equal(batch._json_safe(result), result)
    assert result[0]['values'][1:] == ['input', [1., None]]
    assert result[0]['values'][0]['min'] == 1.25


def test_array_summary_is_strict_even_if_float_conversion_overflows():
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip('no wider float available')
    result = batch._checkpoint_safe(np.array([np.finfo(np.longdouble).max]))
    assert result['min'] is None and result['max'] is None


def test_prepared_writer_does_not_reconvert_scientific_records(tmp_path, monkeypatch):
    ref = batch.FrameRef(path=tmp_path / 'input.dat')
    item = batch.FrameFitResult(frame=ref, result={'parameters': {'a': .5}, 'profile': [-0., 1.]})
    run = batch.BatchRunResult(frame_results=[item])
    payload = batch._checkpoint_payload(run, [ref], config={})
    monkeypatch.setattr(batch, '_json_safe', lambda _: pytest.fail('duplicate conversion'))
    batch.write_checkpoint(tmp_path / 'checkpoint.json', payload)
    assert batch.read_checkpoint(tmp_path / 'checkpoint.json')['frames'][0]['result']['profile'][0].hex() == (-0.).hex()


@pytest.mark.parametrize('boundary', ['fsync', 'replace'])
def test_prepared_snapshot_failure_retains_previous_generation(tmp_path, monkeypatch, boundary):
    path = tmp_path / 'checkpoint.json'
    prior = {'version': 1, 'frames': [{'result': {'a': -0.0}}]}
    batch.write_checkpoint(path, prior)
    before = path.read_bytes()
    payload = batch._CheckpointPayload({'version': 1, 'frames': []})

    def fail(*args, **kwargs):
        raise OSError('injected commit failure')

    monkeypatch.setattr(batch.os, boundary, fail)
    with pytest.raises(OSError, match='injected'):
        batch.write_checkpoint(path, payload)
    assert path.read_bytes() == before
    assert list(tmp_path.glob('*.tmp')) == []
