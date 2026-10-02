"""Synthetic integration against source-pinned, permitted production snapshots.

Never calls production main/apply/plan or touches an original production path.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

ADAPTER = Path(__file__).resolve().parents[1] / 'examples/paired_delivery_adapter.patch'
EXPECTED = {
    'code/repository/src/butterfly_saxs/enrichment_bundle.py': 'dd2398dd5dc74c3c046c825d739fbd3b294490d9b7d7fb5c003fa00a377bdb5d',
    'code/production_scripts/finalize_minimal_delivery.py': '88edfc9ce0f65c75a9855829e841278686f97c6f9dff2338390e65ecf4ee4f75',
    'code/production_scripts/revise_run03_current_delivery.py': 'b92e7ae061d1296ff1e7dc536fb4128926eea83434efcc8fb7d1fa621850a76f',
}


def load_module(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def actual_modules(tmp_path, monkeypatch):
    evidence_path = os.environ.get('WINGSAXS_DELIVERY_EVIDENCE')
    if not evidence_path:
        pytest.skip('permitted production source evidence is not mounted')
    evidence = Path(evidence_path)
    for relative, expected in EXPECTED.items():
        assert hashlib.sha256((evidence / relative).read_bytes()).hexdigest() == expected
    # Importing the actual script prepends its own project/src. Keep that local
    # to this fixture; no actual main/apply/plan entry point is invoked.
    monkeypatch.setattr(sys, 'path', list(sys.path))
    staging = tmp_path / 'staging'
    (staging / 'scripts').mkdir(parents=True)
    for name in ['finalize_minimal_delivery.py', 'revise_run03_current_delivery.py']:
        shutil.copyfile(evidence / 'code/production_scripts' / name, staging / 'scripts' / name)
    subprocess.run(['git', 'apply', '--check', str(ADAPTER)], cwd=staging, check=True)
    subprocess.run(['git', 'apply', str(ADAPTER)], cwd=staging, check=True)
    load_module('butterfly_saxs.enrichment_bundle', evidence / 'code/repository/src/butterfly_saxs/enrichment_bundle.py', monkeypatch)
    original = load_module('original_minimal_delivery', evidence / 'code/production_scripts/finalize_minimal_delivery.py', monkeypatch)
    adapted = load_module('finalize_minimal_delivery', staging / 'scripts/finalize_minimal_delivery.py', monkeypatch)
    revision = load_module('adapted_revision', staging / 'scripts/revise_run03_current_delivery.py', monkeypatch)
    return original, adapted, revision


def fixture(root):
    sample = root / 'sample'
    payloads = {
        '00_guide/index.html': '<a href="../figures/current.html">Current</a>',
        'figures/current.html': '<img src="current.png">' + ''.join(
            f'<a href="../03_profiles/{name}">{name}</a>' for name in
            ['figure_source_map.csv', 'lobe_radial_peaks.csv', 'ellipse_candidates.csv']),
        'figures/current.png': 'existing image bytes',
        '03_profiles/figure_source_map.csv': 'frame,value\n0,0.42\n',
        '03_profiles/lobe_radial_peaks.csv': 'frame,value\n0,0.33\n',
        '03_profiles/ellipse_candidates.csv': 'frame,status\n0,NOT_ACCEPTED\n',
        '03_profiles/figures/normal_profile_diagnostics/index.html': '<a href="../../../figures/current.html">Figures</a>',
        '91_reproducibility/history/navigation_before/index.html.before': '<a href="missing-old.png">Historic</a>',
        '91_reproducibility/history/failed.json': '{"exit_code":2,"scientific_acceptance":false}',
    }
    records = []
    for name, payload in payloads.items():
        p = sample / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload)
        records.append({'path': name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'size_bytes': p.stat().st_size})
    manifest = {'sample_id': sample.name, 'scientific_acceptance': False,
                'figure_qa_status': 'not_reviewed', 'artifacts': records,
                'figure_artifacts': [{'path': 'figures/current.html'}, {'path': 'figures/current.png'}],
                'figure_metadata': {'normal_profile_diagnostics_gallery': {
                    'entrypoint': '03_profiles/figures/normal_profile_diagnostics/index.html'}}}
    (sample / 'bundle_manifest.json').write_text(json.dumps(manifest))
    data = root / 'data.zip'
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr('sample/00_guide/index.html', '<a href="old-only.png">Old superseded navigation</a>')
        z.writestr('sample/data.csv', 'DATA payload stays byte-identical')
    return sample, manifest, data


def test_real_preflight_before_and_after_adapter(tmp_path, actual_modules, monkeypatch):
    original, adapted, _ = actual_modules
    sample, manifest, data = fixture(tmp_path)
    sources = {p: p.read_bytes() for p in sample.rglob('*') if p.is_file()}
    old_data = data.read_bytes()
    initial = original._image_archive_members(sample, manifest)
    with pytest.raises(original.DeliveryError, match='relative-link preflight failed'):
        original._check_sample_relative_links(data, sample, initial)
    read = zipfile.ZipFile.read

    def forbid_data_payload(z, name, *args, **kwargs):
        if getattr(name, 'filename', name) == 'sample/data.csv':
            pytest.fail('DATA payload read during link preflight')
        return read(z, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'read', forbid_data_payload)
    members = adapted._prepare_image_archive_members(data, sample, manifest)
    assert {sample / '03_profiles' / n for n in ['figure_source_map.csv', 'lobe_radial_peaks.csv', 'ellipse_candidates.csv']} <= set(members)
    assert sample / 'data.csv' not in members
    assert data.read_bytes() == old_data
    assert all(p.read_bytes() == value for p, value in sources.items())


def test_real_writer_retry_reuses_without_recompression(tmp_path, actual_modules, monkeypatch):
    original, adapted, _ = actual_modules
    sample, manifest, data = fixture(tmp_path)
    members = adapted._prepare_image_archive_members(data, sample, manifest)
    image = tmp_path / 'images.zip'
    first = adapted._write_image_zip(sample, manifest, image, member_paths=members)
    before = image.read_bytes()
    opened = zipfile.ZipFile.open

    def reject_write(z, name, mode='r', *args, **kwargs):
        if mode == 'w':
            raise AssertionError('unnecessary recompression')
        return opened(z, name, mode, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'open', reject_write)
    with pytest.raises(AssertionError, match='unnecessary recompression'):
        original._write_image_zip(sample, manifest, image, member_paths=members)
    reused = adapted._write_image_zip(sample, manifest, image, member_paths=members)
    assert reused['status'] == 'reused'
    assert reused['sha256'] == first['sha256']
    assert reused['member_hash_checks'] == len(members)
    assert reused['content_hashes_verified'] is True
    assert image.read_bytes() == before
    adapted._check_image_archive_links(data, image, sample.name)


@pytest.mark.parametrize('unregistered', ['figures/current.png', '03_profiles/ellipse_candidates.csv'])
def test_real_adapter_retains_unknown_source_blocker(tmp_path, actual_modules, unregistered):
    _, adapted, _ = actual_modules
    sample, manifest, data = fixture(tmp_path)
    manifest['artifacts'] = [r for r in manifest['artifacts'] if r['path'] != unregistered]
    before = data.read_bytes()
    with pytest.raises(adapted.DeliveryError, match='registered IMAGE member plan failed'):
        adapted._prepare_image_archive_members(data, sample, manifest)
    assert data.read_bytes() == before
    assert not (tmp_path / 'images.zip').exists()


def test_real_revision_preserves_history_and_current_navigation_on_retry(tmp_path, actual_modules, monkeypatch):
    _, adapted, revision = actual_modules
    sample, manifest, data = fixture(tmp_path)
    monkeypatch.setattr(revision, 'SAMPLE', sample)
    members = adapted._image_archive_members(sample, manifest)
    history = [sample / r['path'] for r in manifest['artifacts'] if '/history/' in r['path']]
    members += history
    ready = adapted._prepare_image_archive_members(data, sample, manifest, member_paths=members)
    count = revision.effective_pair_links(data, overrides=ready)
    image = tmp_path / 'images.zip'
    adapted._write_image_zip(sample, manifest, image, member_paths=ready)
    assert revision.effective_pair_links(data, image) == count
    with zipfile.ZipFile(image) as z:
        for path in history:
            assert z.read('sample/' + path.relative_to(sample).as_posix()) == path.read_bytes()
        assert json.loads(z.read('sample/bundle_manifest.json'))['scientific_acceptance'] is False
    # Removing unchanged-on-this-attempt current navigation exposes old DATA HTML.
    without_current = [p for p in ready if p != sample / '00_guide/index.html']
    with pytest.raises(adapted.DeliveryError, match='effective paired archive links failed'):
        revision.effective_pair_links(data, overrides=without_current)
