"""Selected current bindings must stay distinct from retained historical proof."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from butterfly_saxs import delivery_bindings
from butterfly_saxs.delivery_bindings import verify_delivery_bindings


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(root: Path, names: list[str], name: str = "closeout.json", **extra) -> Path:
    path = root / name
    path.write_text(json.dumps({"schema": "wingsaxs.delivery-closeout.v1",
                               "verdict": "PASS", "scientific_acceptance": False,
                               "bindings": [{"path": item, "sha256": _digest(root / item)}
                                            for item in names], **extra}), encoding="utf-8")
    return path


def _tree(root: Path) -> dict[str, tuple[bytes, int]]:
    return {path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_fourteen_sample_root_detects_changed_leaf_without_rewriting_anything(tmp_path):
    names = []
    for number in range(14):
        folder = tmp_path / f"sample_{number:02}"
        folder.mkdir()
        (folder / "bundle_manifest.json").write_text('{"scientific_acceptance":false}')
        names.append(f"{folder.name}/bundle_manifest.json")
    receipt = _receipt(tmp_path, names)
    current = verify_delivery_bindings(receipt)
    assert current["status"] == "current" and current["exit_code"] == 0
    assert current["counts"] == {"checked": 14, "current": 14, "stale": 0}
    (tmp_path / names[3]).write_text('{"scientific_acceptance":false,"figures":"revised"}')
    before = _tree(tmp_path)
    stale = verify_delivery_bindings(receipt)
    assert stale["exit_code"] == 1
    assert stale["counts"] == {"checked": 14, "current": 13, "stale": 1}
    assert [row["path"] for row in stale["checks"] if row["status"] != "current"] == [names[3]]
    assert stale["scientific_acceptance_assessed"] is False
    assert _tree(tmp_path) == before


def test_later_navigation_edit_has_exactly_two_stale_bindings_and_explicit_new_closeout(tmp_path):
    names = ["README.md", "index.html", "bundle.json", "data.zip", "image.zip"]
    for name in names:
        (tmp_path / name).write_text("original " + name)
    failed = tmp_path / "attempt_exit2.json"
    failed.write_text('{"exit_code":2,"scientific_acceptance":false}')
    old = _receipt(tmp_path, names, "old_closeout.json")
    old_document = json.loads(old.read_text())
    for name in names[:2]:
        (tmp_path / name).write_text("navigation revision " + name)
    # A prior PASS is neither rewritten nor accepted as current after an
    # interruption between a navigation update and the next receipt publish.
    stale = verify_delivery_bindings(old)
    assert stale["exit_code"] == 1 and old_document["verdict"] == "PASS"
    assert {row["path"] for row in stale["checks"] if row["status"] == "sha256_mismatch"} == set(names[:2])
    new = _receipt(tmp_path, names, "new_closeout.json",
                   historical_verification=old_document,
                   previous_attempt={"exit_code": 2, "bindings": [{"path": "absent", "sha256": "0" * 64}]})
    before = _tree(tmp_path)
    assert verify_delivery_bindings(new)["status"] == "current"
    assert verify_delivery_bindings(old)["status"] == "stale"
    assert _tree(tmp_path) == before


def test_only_selected_files_are_read_and_explicit_root_can_relocate_receipt(tmp_path, monkeypatch):
    root = tmp_path / "delivery"
    root.mkdir()
    (root / "named.zip").write_bytes(b"archive contents" * 100)
    (root / "unlisted.zip").write_bytes(b"must not be opened")
    receipt = _receipt(root, ["named.zip"])
    relocated = tmp_path / receipt.name
    receipt.rename(relocated)
    original_open = Path.open
    opened = []

    def open_checked(path, *args, **kwargs):
        opened.append(path.name)
        assert path.name != "unlisted.zip"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_checked)
    result = verify_delivery_bindings(relocated, root=root)
    assert result["exit_code"] == 0
    assert set(opened) == {"closeout.json", "named.zip"}


def test_missing_named_file_is_stale_not_a_success_from_original_verdict(tmp_path):
    (tmp_path / "evidence.zip").write_bytes(b"saved archive")
    receipt = _receipt(tmp_path, ["evidence.zip"])
    (tmp_path / "evidence.zip").unlink()
    result = verify_delivery_bindings(receipt)
    assert result["exit_code"] == 1 and result["status"] == "stale"
    assert result["checks"][0]["status"] == "missing_file"
    assert result["checks"][0]["action"]


@pytest.mark.parametrize("document", [None, [], {}, {"bindings": {}}, {"bindings": []},
                                     {"bindings": [1]}, {"bindings": [{}]},
                                     {"bindings": [{"path": "a", "sha256": "invalid"}]},
                                     {"bindings": [{"path": "a", "sha256": 5}]}])
def test_malformed_receipt_raises_actionable_value_error(tmp_path, document):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="receipt|binding|SHA-256"):
        verify_delivery_bindings(receipt)


@pytest.mark.parametrize("payload", ['{', '{"bindings": [], "bindings": []}', '{"bindings": NaN}', '\udcff'])
def test_invalid_or_ambiguous_json_is_rejected(tmp_path, payload):
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(payload.encode("utf-8", errors="surrogateescape"))
    with pytest.raises(ValueError):
        verify_delivery_bindings(receipt)


@pytest.mark.parametrize("name", ["", "/etc/passwd", "../outside", "a/../outside", ".hidden",
                                 "a/.hidden", "a//b", "a/", "a\\b", "C:/secret", "C:secret",
                                 "//host/share", "a:stream", "a\x00b"])
def test_unsafe_relative_paths_are_rejected_before_hashing(tmp_path, monkeypatch, name):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"bindings": [{"path": name, "sha256": "0" * 64}]}))
    monkeypatch.setattr(delivery_bindings, "_hash_file", lambda *args: pytest.fail("unsafe path opened"))
    with pytest.raises(ValueError, match="Unsafe binding path"):
        verify_delivery_bindings(receipt)


def test_symlink_and_symlink_ancestor_are_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("protected")
    root = tmp_path / "root"
    root.mkdir()
    for target, name in [(outside / "file", "linked_file"), (outside, "linked_dir")]:
        (root / name).symlink_to(target, target_is_directory=target.is_dir())
    for name in ["linked_file", "linked_dir/file"]:
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps({"bindings": [{"path": name, "sha256": "0" * 64}]}))
        with pytest.raises(ValueError, match="symlink_path"):
            verify_delivery_bindings(receipt)
    link = root / "receipt_link.json"
    link.symlink_to(receipt)
    with pytest.raises(ValueError, match="symlink"):
        verify_delivery_bindings(link)


def test_self_binding_and_hardlink_to_receipt_are_rejected(tmp_path):
    receipt = tmp_path / "receipt.json"
    for name in [receipt.name, "receipt_alias.json"]:
        receipt.write_text(json.dumps({"bindings": [{"path": name, "sha256": "0" * 64}]}))
        if name != receipt.name:
            os.link(receipt, tmp_path / name)
        with pytest.raises(ValueError, match="cannot bind itself"):
            verify_delivery_bindings(receipt)


def test_conflicting_duplicates_rejected_identical_duplicates_checked_once(tmp_path):
    (tmp_path / "data").write_text("data")
    receipt = _receipt(tmp_path, ["data", "data"])
    assert verify_delivery_bindings(receipt)["counts"]["checked"] == 1
    document = json.loads(receipt.read_text())
    document["bindings"][1]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        verify_delivery_bindings(receipt)


def test_directory_instead_of_named_file_is_stale(tmp_path):
    (tmp_path / "file").write_text("data")
    receipt = _receipt(tmp_path, ["file"])
    (tmp_path / "file").unlink()
    (tmp_path / "file").mkdir()
    result = verify_delivery_bindings(receipt)
    assert result["exit_code"] == 1
    assert result["checks"][0]["status"] == "not_regular_file"


def test_streaming_read_detects_mutation_even_when_read_bytes_match_expected(tmp_path, monkeypatch):
    path = tmp_path / "data.zip"
    path.write_bytes(b"original" * 200_000)
    receipt = _receipt(tmp_path, [path.name])
    expected_digest = _digest(path)
    original_sha256 = hashlib.sha256
    changed = False
    digests = []

    class MutatingHasher:
        def __init__(self):
            self.digest = original_sha256()
            digests.append(self.digest)

        def update(self, block):
            nonlocal changed
            self.digest.update(block)
            if not changed:
                # Windows does not permit replacing an open file. Modify only
                # the already-hashed prefix, preserving the size and unread
                # bytes so the digest can still equal the original binding.
                with path.open("r+b") as writer:
                    writer.write(b"revised!")
                stamp = path.stat()
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
                changed = True

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(delivery_bindings.hashlib, "sha256", MutatingHasher)
    result = verify_delivery_bindings(receipt)
    assert result["exit_code"] == 1
    assert result["checks"][0]["status"] == "changed_during_read"
    assert changed
    assert digests[0].hexdigest() == expected_digest


def _stat_with(value, **overrides):
    fields = {name: getattr(value, name) for name in (
        "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")}
    return SimpleNamespace(**(fields | overrides))


@pytest.mark.parametrize("stale", [False, True])
def test_path_and_handle_ctime_can_differ_without_a_concurrent_change(tmp_path, monkeypatch, stale):
    path = tmp_path / "file"
    path.write_bytes(b"original")
    receipt = _receipt(tmp_path, [path.name])
    if stale:
        path.write_bytes(b"modified")
    before = _tree(tmp_path)
    original_fstat = os.fstat

    def different_ctime(fd):
        value = original_fstat(fd)
        # CPython on Windows can expose creation time via stat/lstat and
        # metadata-change time via fstat for the very same stable file.
        return _stat_with(value, st_ctime_ns=value.st_ctime_ns + 1_000_000_000)

    monkeypatch.setattr(delivery_bindings.os, "fstat", different_ctime)
    result = verify_delivery_bindings(receipt)
    assert result["exit_code"] == int(stale)
    assert result["checks"][0]["status"] == ("sha256_mismatch" if stale else "current")
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("target", ["receipt", "binding"])
def test_handle_ctime_change_during_read_is_still_detected(tmp_path, monkeypatch, target):
    path = tmp_path / "file"
    path.write_bytes(b"original")
    receipt = _receipt(tmp_path, [path.name])
    inode = (receipt if target == "receipt" else path).stat().st_ino
    original_fstat = os.fstat
    reads = 0

    def changed_ctime(fd):
        nonlocal reads
        value = original_fstat(fd)
        if value.st_ino == inode:
            reads += 1
            return _stat_with(value, st_ctime_ns=value.st_ctime_ns + reads * 1_000_000_000)
        return value

    monkeypatch.setattr(delivery_bindings.os, "fstat", changed_ctime)
    if target == "receipt":
        with pytest.raises(ValueError, match="Receipt changed during reading"):
            verify_delivery_bindings(receipt)
    else:
        result = verify_delivery_bindings(receipt)
        assert result["checks"][0]["status"] == "changed_during_read"
    assert reads == 2


@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns"])
@pytest.mark.parametrize("target", ["receipt", "binding"])
def test_path_and_open_handle_must_match_identity_and_content_metadata(tmp_path, monkeypatch, target, field):
    path = tmp_path / "file"
    path.write_bytes(b"original")
    receipt = _receipt(tmp_path, [path.name])
    inode = (receipt if target == "receipt" else path).stat().st_ino
    original_fstat = os.fstat

    def mismatched_handle(fd):
        value = original_fstat(fd)
        if value.st_ino == inode:
            mismatched = value.st_mode ^ 0o200 if field == "st_mode" else getattr(value, field) + 1
            return _stat_with(value, **{field: mismatched})
        return value

    monkeypatch.setattr(delivery_bindings.os, "fstat", mismatched_handle)
    if target == "receipt":
        with pytest.raises(ValueError, match="Receipt changed before reading"):
            verify_delivery_bindings(receipt)
    else:
        result = verify_delivery_bindings(receipt)
        assert result["checks"][0]["status"] == "changed_during_read"


@pytest.mark.parametrize("platform", ["posix", "nt"])
@pytest.mark.parametrize("target", ["receipt", "binding"])
def test_only_windows_cross_api_execute_bits_are_normalized(tmp_path, monkeypatch, platform, target):
    path = tmp_path / "reproduce.cmd"
    path.write_bytes(b"echo reproduce")
    receipt = _receipt(tmp_path, [path.name], name="closeout.cmd")
    selected = (receipt if target == "receipt" else path).stat()
    original_fstat = os.fstat

    def synthetic_execute_bits(fd):
        value = original_fstat(fd)
        if value.st_ino == selected.st_ino:
            return _stat_with(value, st_mode=selected.st_mode ^ 0o111)
        # Neutralize the other file's native Windows execute-bit adjustment
        # when simulating POSIX, so this regression isolates the target file.
        other = (path if target == "receipt" else receipt).stat()
        return _stat_with(value, st_mode=other.st_mode)

    monkeypatch.setattr(delivery_bindings, "os", SimpleNamespace(name=platform, fstat=synthetic_execute_bits))
    if platform == "posix" and target == "receipt":
        with pytest.raises(ValueError, match="Receipt changed before reading"):
            verify_delivery_bindings(receipt)
    else:
        result = verify_delivery_bindings(receipt)
        assert result["checks"][0]["status"] == ("current" if platform == "nt" else "changed_during_read")


@pytest.mark.parametrize("target", ["receipt", "binding"])
def test_windows_handle_execute_bits_changed_during_read_are_not_ignored(tmp_path, monkeypatch, target):
    path = tmp_path / "reproduce.cmd"
    path.write_bytes(b"echo reproduce")
    receipt = _receipt(tmp_path, [path.name])
    inode = (receipt if target == "receipt" else path).stat().st_ino
    original_fstat = os.fstat
    reads = 0

    def changed_execute_bits(fd):
        nonlocal reads
        value = original_fstat(fd)
        if value.st_ino == inode:
            reads += 1
            return _stat_with(value, st_mode=value.st_mode ^ (0o111 if reads == 2 else 0))
        return value

    monkeypatch.setattr(delivery_bindings, "os", SimpleNamespace(name="nt", fstat=changed_execute_bits))
    if target == "receipt":
        with pytest.raises(ValueError, match="Receipt changed during reading"):
            verify_delivery_bindings(receipt)
    else:
        result = verify_delivery_bindings(receipt)
        assert result["checks"][0]["status"] == "changed_during_read"
    assert reads == 2


def test_later_read_cannot_leave_earlier_mutated_binding_current(tmp_path, monkeypatch):
    for name in ["first", "second"]:
        (tmp_path / name).write_text(name)
    receipt = _receipt(tmp_path, ["first", "second"])
    original_hash = delivery_bindings._hash_file

    def mutate_earlier(root, path):
        result = original_hash(root, path)
        if path.name == "second":
            (root / "first").write_text("changed")
        return result

    monkeypatch.setattr(delivery_bindings, "_hash_file", mutate_earlier)
    result = verify_delivery_bindings(receipt)
    assert result["exit_code"] == 1
    assert result["checks"][0]["status"] == "changed_during_read"
    assert result["checks"][1]["status"] == "current"


def test_receipt_changed_during_verification_cannot_report_current(tmp_path, monkeypatch):
    (tmp_path / "file").write_text("data")
    receipt = _receipt(tmp_path, ["file"])
    original_hash = delivery_bindings._hash_file

    def mutate_receipt(root, path):
        result = original_hash(root, path)
        receipt.write_text('{"bindings": []}')
        return result

    monkeypatch.setattr(delivery_bindings, "_hash_file", mutate_receipt)
    with pytest.raises(ValueError, match="Receipt changed"):
        verify_delivery_bindings(receipt)
