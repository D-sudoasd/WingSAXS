from __future__ import annotations

import hashlib
import json
from pathlib import Path

from butterfly_saxs.cli import main


def test_verify_selected_closeout_cli_is_read_only_and_strict_json(tmp_path: Path, capsys):
    root = tmp_path / 'output'
    root.mkdir()
    index = root / 'index.html'
    index.write_text('original navigation')
    receipt = tmp_path / 'closeout.json'
    receipt.write_text(json.dumps({'status': 'complete', 'scientific_acceptance': False,
        'bindings': [{'path': 'index.html', 'sha256': hashlib.sha256(index.read_bytes()).hexdigest()}],
        'historical': {'status': 'failed', 'exit_code': 2, 'sha256': '0' * 64}}))
    original = receipt.read_bytes()
    assert main(['verify-delivery', str(receipt), '--root', str(root)]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current['status'] == 'current'
    assert current['scientific_acceptance_assessed'] is False
    index.write_text('later navigation change')
    assert main(['verify-delivery', str(receipt), '--root', str(root)]) == 1
    stale = json.loads(capsys.readouterr().out)
    assert stale['status'] == 'stale'
    assert receipt.read_bytes() == original
    assert sorted(path.name for path in root.iterdir()) == ['index.html']


def test_verify_delivery_cli_reports_invalid_receipt(tmp_path: Path, capsys):
    receipt = tmp_path / 'bad.json'
    receipt.write_text('{"status":"PASS"}')
    assert main(['verify-delivery', str(receipt)]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error['error']['message']
