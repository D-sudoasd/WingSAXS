from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from butterfly_saxs.cli import main
from butterfly_saxs.delivery import package_batch


def _sample(root: Path, *, status: str = "ok", prefix: str = "") -> dict[str, bytes]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{prefix}frame_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "frame_id", "path", "status", "quality_status", "diagnostic", "parameters_json"])
        writer.writeheader()
        writer.writerow({"frame_index": 0, "frame_id": '<sample "A">', "path": "raw/frame.tif", "status": status,
                         "quality_status": "FAIL" if status == "warning" else "", "diagnostic": "line 1\nline 2",
                         "parameters_json": json.dumps({"original": "x" * 150_000, "stderr": 0.00421234567890123})})
    payloads = {"parameters_long.csv": b"parameter,value,stderr,unit\nratio,0.42,0.01,1\n",
                "manifest.json": b'{"schema_version":"a.future.compatible.version","custom":{"unchanged":true}}',
                "results.npz": b"native archive bytes, copied without reinterpretation", "evolution.png": b"existing plot"}
    for name, value in payloads.items():
        (root / f"{prefix}{name}").write_bytes(value)
    return {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}


def test_package_links_multiple_samples_and_preserves_exact_data(tmp_path: Path):
    originals = _sample(tmp_path / 'sample A #1')
    _sample(tmp_path / 'sample B', status="warning")
    figures = tmp_path / 'sample A #1' / 'figures'
    figures.mkdir()
    (figures / 'profile.pdf').write_bytes(b"existing pdf")
    (tmp_path / 'unrelated-secret.txt').write_text("not an export")
    result = package_batch(tmp_path)
    assert result["operation_status"] == "completed"
    assert result["exit_code"] == 1
    assert result["counts"] == {"samples": 2, "frames": 2, "frame_statuses": {"ok": 1, "warning": 1},
                                "quality_warning_frames": 1, "missing_artifacts": 0}
    assert result["scientific_acceptance"] is False
    index = (tmp_path / 'delivery_index.html').read_text()
    assert '&lt;sample &quot;A&quot;&gt;' in index
    assert 'sample%20A%20%231/parameters_long.csv' in index
    assert 'sample%20A%20%231/figures/profile.pdf' in index
    assert 'line 1\nline 2' in index
    with zipfile.ZipFile(tmp_path / 'delivery.zip') as archive:
        assert archive.testzip() is None
        assert 'unrelated-secret.txt' not in archive.namelist()
        assert 'sample A #1/figures/profile.pdf' in archive.namelist()
        for name, value in originals.items():
            assert archive.read(f'sample A #1/{name}') == value
            assert (tmp_path / 'sample A #1' / name).read_bytes() == value
    assert result["archive"]["sha256"] == hashlib.sha256((tmp_path / 'delivery.zip').read_bytes()).hexdigest()


def test_resume_reuses_archive_and_restores_missing_index_without_redrawing(tmp_path: Path, monkeypatch):
    _sample(tmp_path)
    first = package_batch(tmp_path)
    before = (tmp_path / 'delivery.zip').stat().st_mtime_ns
    (tmp_path / 'delivery_index.html').unlink()
    monkeypatch.setattr("butterfly_saxs.delivery._write_source", lambda *a, **k: pytest.fail("unchanged archive recompressed"))
    second = package_batch(tmp_path, resume=True)
    assert second["archive"]["status"] == "reused"
    assert second["archive"]["sha256"] == first["archive"]["sha256"]
    assert (tmp_path / 'delivery.zip').stat().st_mtime_ns == before
    assert (tmp_path / 'delivery_index.html').is_file()


def test_resume_repairs_damaged_archive_and_includes_new_figures(tmp_path: Path):
    _sample(tmp_path)
    package_batch(tmp_path)
    (tmp_path / 'delivery.zip').write_bytes(b"interrupted ZIP")
    repaired = package_batch(tmp_path, resume=True)
    assert repaired["archive"]["status"] == "created"
    figures = tmp_path / 'figures'
    figures.mkdir()
    (figures / 'finished.svg').write_text('<svg/>')
    completed = package_batch(tmp_path, resume=True)
    assert completed["source_signature"] != repaired["source_signature"]
    assert completed["samples"][0]["figures"] == ["figures/finished.svg"]
    with zipfile.ZipFile(tmp_path / 'delivery.zip') as archive:
        assert archive.read('figures/finished.svg') == b'<svg/>'


def test_missing_outputs_are_actionable_without_blocking_good_data(tmp_path: Path):
    _sample(tmp_path)
    (tmp_path / 'evolution.png').unlink()
    result = package_batch(tmp_path)
    assert result["exit_code"] == 1
    assert result["archive"]["status"] == "created"
    assert result["samples"][0]["missing"] == ["evolution"]
    assert 'Missing outputs: evolution' in (tmp_path / 'delivery_index.html').read_text()
    assert 'package --resume' in result["next_action"]


def test_no_archive_is_explicit_and_cli_stdout_stays_json(tmp_path: Path, capsys):
    _sample(tmp_path, prefix="series_")
    assert main(["package", str(tmp_path), "--no-archives"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["archive"]["status"] == "skipped"
    assert "archive" not in report["outputs"]
    assert not (tmp_path / 'delivery.zip').exists()
    assert "Indexing" in captured.err
    assert main(["package", str(tmp_path), "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["archive"]["status"] == "created"


def test_interrupt_can_resume_and_existing_archive_is_preserved(tmp_path: Path, monkeypatch):
    _sample(tmp_path)
    package_batch(tmp_path)
    original_archive = (tmp_path / 'delivery.zip').read_bytes()
    (tmp_path / 'evolution.png').write_bytes(b"new plot")
    from butterfly_saxs import delivery
    original_write = delivery._write_source
    monkeypatch.setattr(delivery, "_write_source", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        package_batch(tmp_path, resume=True)
    assert (tmp_path / 'delivery.zip').read_bytes() == original_archive
    report = json.loads((tmp_path / 'delivery_summary.json').read_text())
    assert report["archive"]["status"] == "pending"
    assert report["operation_status"] == "packaging"
    assert not list(tmp_path.glob('.delivery-*.zip'))
    monkeypatch.setattr(delivery, "_write_source", original_write)
    assert package_batch(tmp_path, resume=True)["archive"]["status"] == "created"


def test_source_change_during_packaging_keeps_previous_archive(tmp_path: Path):
    _sample(tmp_path)
    package_batch(tmp_path)
    before = (tmp_path / 'delivery.zip').read_bytes()

    def change_file(message):
        if message.startswith('Packaging') and message.endswith('results.npz'):
            (tmp_path / 'results.npz').write_bytes(b'changed after indexing')

    with pytest.raises(ValueError, match='Output changed while packaging'):
        package_batch(tmp_path, force=True, progress=change_file)
    assert (tmp_path / 'delivery.zip').read_bytes() == before


def test_explicit_overwrite_and_invalid_root_are_actionable(tmp_path: Path, capsys):
    assert main(["package", str(tmp_path)]) == 2
    assert 'No frame_summary.csv' in json.loads(capsys.readouterr().out)["error"]["message"]
    _sample(tmp_path)
    (tmp_path / 'delivery_index.html').write_text('unrelated file')
    with pytest.raises(FileExistsError):
        package_batch(tmp_path)
    with pytest.raises(ValueError, match='not recognizable'):
        package_batch(tmp_path, resume=True)
    assert (tmp_path / 'delivery_index.html').read_text() == 'unrelated file'
    assert package_batch(tmp_path, force=True)["exit_code"] == 0


def test_symlink_escape_is_not_added(tmp_path: Path):
    root = tmp_path / 'outputs'
    _sample(root)
    outside = tmp_path / 'outside.npz'
    outside.write_bytes(b'private unrelated bytes')
    (root / 'results.npz').unlink()
    (root / 'results.npz').symlink_to(outside)
    with pytest.raises(ValueError, match='regular file inside'):
        package_batch(root)
    assert not (root / 'delivery.zip').exists()


def test_batch_finishes_delivery_and_failed_zip_can_resume_without_refit(tmp_path: Path, monkeypatch, capsys):
    import numpy as np
    from butterfly_saxs import cli, delivery

    source = tmp_path / 'input.npy'
    np.save(source, np.ones((8, 8)))
    output = tmp_path / 'output'
    calls = []

    def analyze(path, **kwargs):
        calls.append(str(path))
        return {"parameters": {"ratio": {"value": 0.42, "unit": "dimensionless"}}, "image": np.ones((8, 8))}

    original_package = delivery.package_batch
    monkeypatch.setattr(cli, 'analyze_frame', analyze)
    monkeypatch.setattr(delivery, 'package_batch', lambda *a, **k: (_ for _ in ()).throw(OSError('temporary disk fault')))
    assert main(['batch', str(source), '-o', str(output), '--package']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['n_frames'] == 1
    assert report['n_failed'] == 0
    assert report['delivery']['operation_status'] == 'incomplete'
    assert report['delivery']['next_command'] == ['bsaxs', 'package', str(output), '--resume']
    assert (output / 'results.npz').is_file()
    assert calls == [str(source)]
    monkeypatch.setattr(delivery, 'package_batch', original_package)
    monkeypatch.setattr(cli, 'analyze_frame', lambda *a, **k: pytest.fail('delivery refitted data'))
    assert main(report['delivery']['next_command'][1:]) == 0
    package_report = json.loads(capsys.readouterr().out)
    assert package_report['archive']['status'] == 'created'
    assert (output / 'delivery_index.html').is_file()


def test_batch_can_automatically_finish_delivery(tmp_path: Path, monkeypatch, capsys):
    import numpy as np
    from butterfly_saxs import cli

    source = tmp_path / 'input.npy'
    np.save(source, np.ones((8, 8)))
    monkeypatch.setattr(cli, 'analyze_frame', lambda *a, **k: {"parameters": {"ratio": 0.42}})
    assert main(['batch', str(source), '-o', str(tmp_path / 'out'), '--package']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['delivery']['operation_status'] == 'completed'
    assert report['delivery']['archive']['status'] == 'created'
    assert Path(report['delivery']['outputs']['archive']).is_file()


def test_native_evolution_keeps_mixed_units_stderr_and_candidates(tmp_path: Path, monkeypatch):
    from butterfly_saxs import visualization
    from butterfly_saxs.batch import FrameFitResult, FrameRef
    from butterfly_saxs.export import _write_evolution
    import matplotlib.pyplot as plt

    captured = {}
    existing_figures = plt.get_fignums()
    original = visualization.plot_parameter_evolution

    def spy(rows, **kwargs):
        captured.update({'rows': rows, **kwargs})
        return original(rows, **kwargs)

    monkeypatch.setattr(visualization, 'plot_parameter_evolution', spy)
    _write_evolution(tmp_path / 'evolution.png', [
        FrameFitResult(FrameRef(tmp_path / 'a.tif', time=10), status='warning',
                       result={'parameters': {'axis_ratio': {'value': .42, 'stderr': .02, 'unit': 'dimensionless'}}}),
        FrameFitResult(FrameRef(tmp_path / 'b.tif'), status='failed', result=None),
        FrameFitResult(FrameRef(tmp_path / 'c.tif', time=20),
                       result={'parameters': {'axis_ratio': {'value': .43, 'unit': 'dimensionless'}}}),
    ])
    assert [row.get('quantity_0') for row in captured['rows']] == [.42, None, .43]
    assert captured['rows'][0]['quantity_0_stderr'] == .02
    assert captured['rows'][0]['quantity_0_status'] == 'warning; candidate'
    assert captured['rows'][2]['quantity_0_status'] == 'candidate'
    assert captured['parameter_units'] == {'quantity_0': 'dimensionless'}
    assert captured['parameter_labels'] == {'quantity_0': 'Axis ratio b/a'}
    assert (tmp_path / 'evolution.png').stat().st_size > 1000
    assert plt.get_fignums() == existing_figures


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'], ids=['lf', 'crlf'])
def test_legacy_csv_columns_and_empty_samples_remain_browsable(tmp_path: Path, newline: bytes):
    legacy_csv = newline.join([b'frame_id,status,custom', b'original-01,warning,retained', b''])
    empty_csv = b'frame_id,status' + newline
    _sample(tmp_path / 'legacy')
    (tmp_path / 'legacy' / 'frame_summary.csv').write_bytes(legacy_csv)
    _sample(tmp_path / 'empty')
    (tmp_path / 'empty' / 'frame_summary.csv').write_bytes(empty_csv)
    report = package_batch(tmp_path)
    assert report['exit_code'] == 1
    legacy = next(sample for sample in report['samples'] if sample['sample'] == 'legacy')
    assert legacy['frames'][0]['csv_record'] == 0
    assert legacy['frames'][0]['frame_index'] == ''
    with zipfile.ZipFile(tmp_path / 'delivery.zip') as archive:
        assert archive.read('legacy/frame_summary.csv') == legacy_csv
        assert archive.read('empty/frame_summary.csv') == empty_csv
    assert (tmp_path / 'legacy' / 'frame_summary.csv').read_bytes() == legacy_csv
    assert (tmp_path / 'empty' / 'frame_summary.csv').read_bytes() == empty_csv


@pytest.mark.parametrize('fail_name', ['delivery_summary.json', 'delivery_index.html'])
def test_initial_receipt_or_index_write_failure_can_resume(tmp_path: Path, monkeypatch, fail_name):
    from butterfly_saxs import delivery
    _sample(tmp_path)
    original_write = delivery._atomic_text

    def fail_initial(path, text):
        if path.name == fail_name:
            raise OSError('initial write interrupted')
        return original_write(path, text)

    monkeypatch.setattr(delivery, '_atomic_text', fail_initial)
    with pytest.raises(OSError, match='initial write interrupted'):
        package_batch(tmp_path)
    monkeypatch.setattr(delivery, '_atomic_text', original_write)
    assert package_batch(tmp_path, resume=True)['operation_status'] == 'completed'


def test_partial_and_hidden_figure_files_are_not_delivered(tmp_path: Path):
    _sample(tmp_path)
    figures = tmp_path / 'figures'
    (figures / '.staging-interrupted').mkdir(parents=True)
    (figures / '.staging-interrupted' / 'unfinished.png').write_bytes(b'partial')
    (figures / '.private-token').write_text('unrelated')
    (figures / 'finished.png').write_bytes(b'finished')
    report = package_batch(tmp_path)
    assert report['samples'][0]['figures'] == ['figures/finished.png']
    with zipfile.ZipFile(tmp_path / 'delivery.zip') as bundle:
        assert not any('.staging' in name or '.private' in name for name in bundle.namelist())


def test_package_projects_large_csv_metadata_without_loading_whole_batch(tmp_path: Path):
    import tracemalloc
    _sample(tmp_path)
    with (tmp_path / 'frame_summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame_index', 'status', 'parameters_json'])
        for index in range(64):
            writer.writerow([index, 'ok', 'x' * (256 * 1024)])
    tracemalloc.start()
    try:
        report = package_batch(tmp_path, archive=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert report['counts']['frames'] == 64
    assert peak < 8 * 1024 * 1024  # Input is 16 MiB; no O(total-metadata) list.
    assert 'delivery.zip' not in report['next_action']
    assert not (tmp_path / 'delivery.zip').exists()
