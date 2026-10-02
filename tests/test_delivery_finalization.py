"""Last-mile delivery must be repeatable, recoverable, and content-bound."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import pytest

from butterfly_saxs import delivery


def _sample(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        'frame_summary.csv': 'frame_id,status,quality_status,diagnostic\nold,warning,FAIL,conditional candidate\n',
        'parameters_long.csv': 'parameter,value,stderr,unit\nratio,0.42,0.01,1\n',
        'manifest.json': '{"scientific_acceptance":false,"manual_status":"unreviewed"}\n',
        'results.npz': 'existing arrays', 'evolution.png': 'existing plot',
    }.items():
        (root / name).write_text(payload)


def _generated(root: Path) -> dict[str, tuple[bytes, int]]:
    return {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns)
            for name in ('delivery_index.html', 'delivery_summary.json', 'delivery.zip')}


def _no_rearchive(monkeypatch) -> None:
    monkeypatch.setattr(delivery, '_write_source', lambda *a, **k: pytest.fail('unchanged ZIP was rebuilt'))


def test_no_op_resume_keeps_bound_receipt_bytes_and_mtimes(tmp_path: Path, monkeypatch):
    _sample(tmp_path)
    first = delivery.package_batch(tmp_path)
    before = _generated(tmp_path)
    _no_rearchive(monkeypatch)
    for _ in range(3):
        result = delivery.package_batch(tmp_path, resume=True)
        assert result['archive']['status'] == 'reused'
        assert result['archive']['sha256'] == first['archive']['sha256']
        assert result['scientific_acceptance'] is False
        assert result['counts']['quality_warning_frames'] == 1
        assert result['exit_code'] == 1
        assert _generated(tmp_path) == before
    persisted = json.loads((tmp_path / 'delivery_summary.json').read_text())
    assert persisted['archive']['status'] == 'created'
    assert persisted['index_sha256'] == hashlib.sha256(before['delivery_index.html'][0]).hexdigest()


def test_interruption_after_zip_promotion_reuses_completed_zip(tmp_path: Path, monkeypatch):
    _sample(tmp_path)
    delivery.package_batch(tmp_path)
    (tmp_path / 'evolution.png').write_bytes(b'new existing plot')
    original_write = delivery._atomic_text

    def fail_final_receipt(path, text):
        if path.name == 'delivery_summary.json' and json.loads(text)['operation_status'] == 'completed':
            raise OSError('receipt interrupted')
        original_write(path, text)

    monkeypatch.setattr(delivery, '_atomic_text', fail_final_receipt)
    with pytest.raises(OSError, match='receipt interrupted'):
        delivery.package_batch(tmp_path, resume=True)
    pending = json.loads((tmp_path / 'delivery_summary.json').read_text())
    archive = (tmp_path / 'delivery.zip').read_bytes()
    assert pending['archive']['status'] == 'pending'
    assert pending['archive']['sha256'] == hashlib.sha256(archive).hexdigest()
    monkeypatch.setattr(delivery, '_atomic_text', original_write)
    _no_rearchive(monkeypatch)
    result = delivery.package_batch(tmp_path, resume=True)
    assert result['archive']['status'] == 'reused'
    assert (tmp_path / 'delivery.zip').read_bytes() == archive
    assert json.loads((tmp_path / 'delivery_summary.json').read_text())['operation_status'] == 'completed'


def test_interruption_before_zip_promotion_preserves_old_archive(tmp_path: Path, monkeypatch):
    _sample(tmp_path)
    delivery.package_batch(tmp_path)
    before = (tmp_path / 'delivery.zip').read_bytes()
    (tmp_path / 'evolution.png').write_bytes(b'new existing plot')
    original_replace = delivery.os.replace

    def fail_promotion(source, destination):
        if Path(destination).name == 'delivery.zip':
            raise OSError('promotion interrupted')
        original_replace(source, destination)

    monkeypatch.setattr(delivery.os, 'replace', fail_promotion)
    with pytest.raises(OSError, match='promotion interrupted'):
        delivery.package_batch(tmp_path, resume=True)
    assert (tmp_path / 'delivery.zip').read_bytes() == before
    pending = json.loads((tmp_path / 'delivery_summary.json').read_text())
    assert pending['archive']['sha256'] != hashlib.sha256(before).hexdigest()
    assert not list(tmp_path.glob('.delivery-*.zip'))
    monkeypatch.setattr(delivery.os, 'replace', original_replace)
    assert delivery.package_batch(tmp_path, resume=True)['archive']['status'] == 'created'
    with zipfile.ZipFile(tmp_path / 'delivery.zip') as bundle:
        assert bundle.read('evolution.png') == b'new existing plot'


def test_index_only_runs_keep_reusable_archive_binding(tmp_path: Path, monkeypatch):
    _sample(tmp_path)
    first = delivery.package_batch(tmp_path)
    before = (tmp_path / 'delivery.zip').stat().st_mtime_ns
    _no_rearchive(monkeypatch)
    for _ in range(2):
        skipped = delivery.package_batch(tmp_path, resume=True, archive=False)
        assert skipped['archive']['status'] == 'skipped'
        assert 'archive' not in skipped['outputs']
        assert skipped['archive']['previous']['source_signature'] == first['source_signature']
    resumed = delivery.package_batch(tmp_path, resume=True)
    assert resumed['archive']['status'] == 'reused'
    assert resumed['archive']['sha256'] == first['archive']['sha256']
    assert (tmp_path / 'delivery.zip').stat().st_mtime_ns == before


@pytest.mark.parametrize('change', ['source', 'archive'])
def test_retained_archive_is_never_mistaken_for_current_output(tmp_path: Path, change):
    _sample(tmp_path)
    first = delivery.package_batch(tmp_path)
    if change == 'source':
        (tmp_path / 'evolution.png').write_bytes(b'new plot')
    else:
        (tmp_path / 'delivery.zip').write_bytes(b'damaged ZIP')
    skipped = delivery.package_batch(tmp_path, resume=True, archive=False)
    assert skipped['archive']['previous']['source_signature'] == first['source_signature']
    if change == 'source':
        assert skipped['source_signature'] != first['source_signature']
    resumed = delivery.package_batch(tmp_path, resume=True)
    assert resumed['archive']['status'] == 'created'
    with zipfile.ZipFile(tmp_path / 'delivery.zip') as bundle:
        assert bundle.testzip() is None
        assert bundle.read('evolution.png') == (tmp_path / 'evolution.png').read_bytes()


def test_archive_is_content_deterministic_across_time_permissions_and_location(tmp_path: Path):
    first_root, relocated = tmp_path / 'first', tmp_path / 'relocated'
    for root in (first_root, relocated):
        _sample(root)
    first = delivery.package_batch(first_root)
    original = (first_root / 'delivery.zip').read_bytes()
    for path in first_root.iterdir():
        os.utime(path, (1_700_000_000, 1_700_000_000))
        path.chmod(0o600)
    rebuilt = delivery.package_batch(first_root, force=True)
    assert rebuilt['archive']['sha256'] == first['archive']['sha256']
    assert (first_root / 'delivery.zip').read_bytes() == original
    # A root-level sample intentionally uses the root name in the navigation;
    # retain that name when relocating an already-exported dataset.
    same_name = relocated / 'first'
    _sample(same_name)
    other = delivery.package_batch(same_name)
    assert other['archive']['sha256'] == first['archive']['sha256']


def test_csv_changed_before_initial_hash_is_parsed_from_the_same_generation(tmp_path: Path):
    _sample(tmp_path)

    def replace_csv(message):
        if message == 'Indexing frame_summary.csv':
            (tmp_path / 'frame_summary.csv').write_text('frame_id,status\nnew,failed\n')

    result = delivery.package_batch(tmp_path, progress=replace_csv)
    assert result['samples'][0]['frames'][0]['frame_id'] == 'new'
    assert result['counts']['frame_statuses'] == {'failed': 1}
    assert result['exit_code'] == 1
    with zipfile.ZipFile(tmp_path / 'delivery.zip') as bundle:
        assert bundle.read('frame_summary.csv') == b'frame_id,status\nnew,failed\n'


def test_mutation_of_already_copied_file_rejects_promotion(tmp_path: Path):
    _sample(tmp_path)
    delivery.package_batch(tmp_path)
    before = (tmp_path / 'delivery.zip').read_bytes()

    def mutate_previous(message):
        if message.startswith('Packaging') and message.endswith('results.npz'):
            (tmp_path / 'parameters_long.csv').write_text('parameter,value\nratio,0.8\n')

    with pytest.raises(ValueError, match='Output changed while packaging: parameters_long.csv'):
        delivery.package_batch(tmp_path, force=True, progress=mutate_previous)
    assert (tmp_path / 'delivery.zip').read_bytes() == before


@pytest.mark.parametrize('archive', [False, True])
def test_source_changes_during_finalization_do_not_publish_success(tmp_path: Path, monkeypatch, archive):
    _sample(tmp_path)
    delivery.package_batch(tmp_path)
    before = (tmp_path / 'delivery_summary.json').read_bytes()
    render = delivery._render

    def mutate_after_render(report):
        result = render(report)
        (tmp_path / 'parameters_long.csv').write_text('parameter,value\nratio,0.8\n')
        return result

    monkeypatch.setattr(delivery, '_render', mutate_after_render)
    with pytest.raises(ValueError, match='Output changed while packaging: parameters_long.csv'):
        delivery.package_batch(tmp_path, resume=True, archive=archive)
    assert (tmp_path / 'delivery_summary.json').read_bytes() == before


def test_streamed_archive_bytes_are_hashed_instead_of_reopening_source(tmp_path: Path, monkeypatch):
    source = tmp_path / 'results.npz'
    source.write_bytes(b'original')
    record = {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}
    original_open = Path.open

    def changed_read(path, *args, **kwargs):
        if path == source:
            return io.BytesIO(b'modified')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', changed_read)
    with zipfile.ZipFile(tmp_path / 'test.zip', 'w') as bundle:
        with pytest.raises(ValueError, match='Output changed while packaging: results.npz'):
            delivery._write_source(bundle, source, 'results.npz', record)


def test_contents_metadata_changes_invalidate_archive_identity(tmp_path: Path, monkeypatch):
    _sample(tmp_path)
    first = delivery.package_batch(tmp_path)
    collect = delivery.collect_html_dependencies

    def extra_metadata(*args, **kwargs):
        return {**collect(*args, **kwargs), 'collection_revision': 2}

    monkeypatch.setattr(delivery, 'collect_html_dependencies', extra_metadata)
    second = delivery.package_batch(tmp_path, resume=True)
    assert second['source_signature'] != first['source_signature']
    assert second['archive']['status'] == 'created'
    with zipfile.ZipFile(tmp_path / 'delivery.zip') as bundle:
        assert json.loads(bundle.read('delivery_contents.json'))['dependencies']['collection_revision'] == 2


def test_legacy_v1_receipt_reuses_zip_without_recompression(tmp_path: Path, monkeypatch):
    _sample(tmp_path)
    report = delivery.package_batch(tmp_path)
    legacy_signature = hashlib.sha256(delivery._json({
        'files': report['files'], 'index_sha256': report['index_sha256'],
    }).encode()).hexdigest()
    archive_path = tmp_path / 'delivery.zip'
    with zipfile.ZipFile(archive_path) as bundle:
        payloads = {name: bundle.read(name) for name in bundle.namelist()}
    contents = json.loads(payloads['delivery_contents.json'])
    contents['source_signature'] = legacy_signature
    payloads['delivery_contents.json'] = delivery._json(contents).encode()
    with zipfile.ZipFile(archive_path, 'w') as bundle:
        for name, payload in payloads.items():
            bundle.writestr(name, payload)
    report['source_signature'] = legacy_signature
    report.pop('index_sha256')
    report['archive'] = {'status': 'created', 'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                         'bytes': archive_path.stat().st_size}
    (tmp_path / 'delivery_summary.json').write_text(delivery._json(report))
    before = archive_path.read_bytes()
    old_receipt = (tmp_path / 'delivery_summary.json').read_bytes()
    old_mtime = (tmp_path / 'delivery_summary.json').stat().st_mtime_ns
    _no_rearchive(monkeypatch)
    upgraded = delivery.package_batch(tmp_path, resume=True)
    assert upgraded['archive']['status'] == 'reused'
    assert upgraded['source_signature'] == legacy_signature
    assert archive_path.read_bytes() == before
    assert (tmp_path / 'delivery_summary.json').read_bytes() == old_receipt
    assert (tmp_path / 'delivery_summary.json').stat().st_mtime_ns == old_mtime
    final = _generated(tmp_path)
    delivery.package_batch(tmp_path, resume=True)
    assert _generated(tmp_path) == final


@pytest.mark.parametrize('change', ['sample', 'figure', 'artifact'])
@pytest.mark.parametrize('mode', ['archive', 'reuse', 'index'])
def test_new_output_inventory_is_detected_before_finalization(tmp_path: Path, monkeypatch, change, mode):
    _sample(tmp_path)
    delivery.package_batch(tmp_path)
    before = (tmp_path / 'delivery.zip').read_bytes()
    render = delivery._render

    def add_output(report):
        result = render(report)
        if change == 'sample':
            _sample(tmp_path / 'new_sample')
        elif change == 'figure':
            (tmp_path / 'figures').mkdir()
            (tmp_path / 'figures' / 'late.png').write_bytes(b'late plot')
        else:
            (tmp_path / 'ellipse_fit.json').write_text('{"status":"candidate"}')
        return result

    monkeypatch.setattr(delivery, '_render', add_output)
    with pytest.raises(ValueError, match='Output inventory changed while packaging'):
        delivery.package_batch(tmp_path, resume=True, archive=mode != 'index', force=mode == 'archive')
    assert (tmp_path / 'delivery.zip').read_bytes() == before


@pytest.mark.parametrize('archive', [False, True])
def test_interruption_after_changed_index_keeps_honest_partial_receipt(tmp_path: Path, monkeypatch, archive):
    _sample(tmp_path)
    first = delivery.package_batch(tmp_path)
    old_zip = (tmp_path / 'delivery.zip').read_bytes()
    (tmp_path / 'frame_summary.csv').write_text('frame_id,status\nnew,failed\n')
    original_write = delivery._atomic_text

    def interrupt_after_index(path, text):
        original_write(path, text)
        if path.name == 'delivery_index.html':
            raise OSError('interrupted after index')

    monkeypatch.setattr(delivery, '_atomic_text', interrupt_after_index)
    with pytest.raises(OSError, match='interrupted after index'):
        delivery.package_batch(tmp_path, resume=True, archive=archive)
    pending = json.loads((tmp_path / 'delivery_summary.json').read_text())
    assert pending['operation_status'] == 'indexing'
    assert pending['source_signature'] != first['source_signature']
    assert pending['samples'][0]['frames'][0]['frame_id'] == 'new'
    assert (tmp_path / 'delivery.zip').read_bytes() == old_zip
    monkeypatch.setattr(delivery, '_atomic_text', original_write)
    resumed = delivery.package_batch(tmp_path, resume=True, archive=archive)
    assert resumed['operation_status'] == 'completed'
    assert resumed['samples'][0]['frames'][0]['frame_id'] == 'new'
