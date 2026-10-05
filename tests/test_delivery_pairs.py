"""Synthetic regressions for the recorded paired-delivery failures.

Fixtures reproduce structural causes, not private scientific data or production
paths. The three support CSV basenames and navigation-before shape come from
minimal_delivery_apply_001 and run03_current_revision_apply001 respectively.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import stat
import zipfile

import pytest

from butterfly_saxs.delivery_pairs import (
    check_archive_pair,
    check_planned_pair,
    plan_image_members,
)


def _write(root: Path, name: str, payload: str | bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode() if isinstance(payload, str) else payload)
    return path


def _inventory(root: Path, *names: str) -> list[dict]:
    return [{"path": name, "size_bytes": (root / name).stat().st_size,
             "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
            for name in names]


def _zip(path: Path, files: dict[str, str | bytes], *, sample: str = "sample") -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(f"{sample}/{name}", payload)
    return path


def _reasons(report: dict) -> set[str]:
    return {item["reason"] for item in report["issues"]}


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    root = tmp_path / "sample"
    root.mkdir()
    return root


def test_three_registered_csv_dependencies_repair_real_failure_without_mutation(sample: Path):
    directory = "90_evidence/sample/figures"
    names = [f"{directory}/{name}.csv" for name in
             ("figure_source_map", "lobe_radial_peaks", "ellipse_candidates")]
    gallery = f"{directory}/index.html"
    _write(sample, gallery, "\n".join(f'<a href="{Path(name).name}">data</a>' for name in names))
    for name in names:
        _write(sample, name, "estimate,quality,scientific_acceptance\n0.420000000000001,FAIL,false\n")
    _write(sample, "unrelated.csv", "not referenced and not registered")
    source_records = _inventory(sample, gallery, *names)
    data = _zip(sample.parent / "DATA.zip", {"results.npz": b"opaque DATA"})
    original_data = (data.read_bytes(), data.stat().st_mtime_ns)
    originals = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sample.rglob("*") if p.is_file()}

    before = check_planned_pair(data, sample, [gallery])
    assert before["status"] == "blocked"
    assert {row["reference"] for row in before["issues"]} == {Path(name).name for name in names}
    plan = plan_image_members(sample, [gallery], source_records, data_zip=data)
    assert plan["status"] == "ready"
    assert plan["included"] == sorted(names)
    assert plan["content_hashes_verified"] is False
    assert plan["members"] == sorted(source_records, key=lambda row: row["path"])
    selected = [row["path"] for row in plan["members"]]
    assert check_planned_pair(data, sample, selected)["status"] == "ready"
    image = _zip(sample.parent / "IMAGE.zip", {name: (sample / name).read_bytes() for name in selected})
    assert check_archive_pair(data, image, sample_id="sample")["status"] == "ready"
    # Actual extraction in the documented order must preserve exact CSV bytes.
    extracted = sample.parent / "extracted"
    for path in (data, image):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extracted)
    for name in names:
        assert (extracted / "sample" / name).read_bytes() == (sample / name).read_bytes()
    assert not (extracted / "sample/unrelated.csv").exists()
    assert (data.read_bytes(), data.stat().st_mtime_ns) == original_data
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in originals} == originals


def test_unregistered_existing_file_is_not_promoted_or_hashed(sample: Path, monkeypatch):
    gallery = "gallery.html"
    missing = "frame_HR0000_fit_quality.png"
    _write(sample, gallery, f'<img src="{missing}">')
    _write(sample, missing, b"existing but unregistered source")
    records = _inventory(sample, gallery)
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **k: pytest.fail("preflight must not invent provenance"))
    report = plan_image_members(sample, [gallery], records)
    assert report["status"] == "blocked"
    assert report["issues"] == [{"source": gallery, "reference": missing, "reason": "unregistered_artifact"}]
    assert report["included"] == []
    assert [item["path"] for item in report["members"]] == [gallery]


def test_initial_unregistered_member_and_promotion_receipt_remain_failures(sample: Path):
    receipt = "91_reproducibility/figure_candidate_promotion_v11.json"
    _write(sample, receipt, '{"source_binding":"unknown"}')
    report = plan_image_members(sample, [receipt], [])
    assert _reasons(report) == {"unregistered_artifact"}
    assert report["members"] == []


def test_data_first_image_override_checks_only_effective_html(sample: Path):
    data = _zip(sample.parent / "DATA.zip", {
        "index.html": b"\xff old invalid UTF-8 and stale missing links",
        "table.csv": "already in DATA",
        "data-only.html": '<a href="index.html">Current gallery</a>',
    })
    _write(sample, "index.html", '<a href="table.csv">Data</a><img src="new.png">')
    _write(sample, "new.png", b"already rendered")
    records = _inventory(sample, "index.html", "new.png")
    plan = plan_image_members(sample, ["index.html"], records, data_zip=data)
    assert plan["status"] == "ready"
    assert plan["included"] == ["new.png"]
    assert plan["overridden_html"] == ["index.html"]
    assert plan["html_checked"] == ["data-only.html", "index.html"]
    members = [row["path"] for row in plan["members"]]
    assert check_planned_pair(data, sample, members)["status"] == "ready"
    image = _zip(sample.parent / "IMAGE.zip", {name: (sample / name).read_bytes() for name in members})
    assert check_archive_pair(data, image, sample_id="sample")["status"] == "ready"


@pytest.mark.parametrize("page", ["index.html", "data-only.html"])
def test_genuine_current_missing_link_never_hidden_by_other_archive(sample: Path, page: str):
    data = _zip(sample.parent / "DATA.zip", {
        "index.html": "valid old gallery", "data-only.html": '<img src="missing.png">' if page == "data-only.html" else "valid",
    })
    text = '<img src="missing.png">' if page == "index.html" else "valid new gallery"
    _write(sample, "index.html", text)
    image = _zip(sample.parent / "IMAGE.zip", {"index.html": text})
    for report in [check_planned_pair(data, sample, ["index.html"]),
                   check_archive_pair(data, image, sample_id="sample")]:
        assert report["status"] == "blocked"
        assert report["issues"] == [{"source": page, "reference": "missing.png", "reason": "missing_file"}]


def test_recursive_encoded_links_attributes_and_cycles_are_explicit(sample: Path):
    files = {
        "gallery.html": '<a href="reports/next%20%231.htm?view=1#data">More</a>',
        "reports/next #1.htm": '''<a href="../gallery.html">Back</a>
            <object data="../table%20%231.csv?download=1&amp;x=2#row"></object>
            <img srcset="../preview%20one.png 1x, ../preview%20two.png 2x">
            <video poster="../poster.png"></video><a href="#fragment">same</a>
            <a href="?view=2">same</a><a href="https://example.com/data.csv">external</a>''',
        "table #1.csv": "opaque data", "preview one.png": "1", "preview two.png": "2", "poster.png": "3",
    }
    for name, payload in files.items():
        _write(sample, name, payload)
    report = plan_image_members(sample, ["gallery.html"], _inventory(sample, *files))
    assert report["status"] == "ready"
    assert report["included"] == sorted(files.keys() - {"gallery.html"})
    assert len(report["external"]) == 1
    assert report["html_checked"] == ["gallery.html", "reports/next #1.htm"]
    data = _zip(sample.parent / "DATA.zip", {})
    image = _zip(sample.parent / "IMAGE.zip", files)
    assert check_planned_pair(data, sample, files)["status"] == "ready"
    assert check_archive_pair(data, image, sample_id="sample")["status"] == "ready"


@pytest.mark.parametrize(("reference", "reason"), [
    ("../outside.csv", "outside_sample"),
    ("%2e%2e%2foutside.csv", "outside_sample"),
    ("..%5coutside.csv", "outside_sample"),
    ("/absolute.csv", "absolute_path"),
    ("C:/absolute.csv", "absolute_path"),
    ("file:///private.csv", "absolute_path"),
    (".private/data.csv", "hidden_or_staging_path"),
    ("%00.csv", "invalid_url"),
    ("%ff.csv", "invalid_url"),
])
def test_unsafe_links_fail_in_planner_and_both_pair_checks(sample: Path, reference: str, reason: str):
    text = f'<a href="{reference}">data</a>'
    _write(sample, "index.html", text)
    data = _zip(sample.parent / "DATA.zip", {})
    image = _zip(sample.parent / "IMAGE.zip", {"index.html": text})
    for report in [plan_image_members(sample, ["index.html"], _inventory(sample, "index.html")),
                   check_planned_pair(data, sample, ["index.html"]),
                   check_archive_pair(data, image, sample_id="sample")]:
        assert report["status"] == "blocked"
        assert _reasons(report) == {reason}


@pytest.mark.parametrize("directory", [False, True])
def test_symlink_member_or_ancestor_is_never_opened(sample: Path, directory: bool, monkeypatch):
    outside = sample.parent / "private"
    outside.mkdir()
    _write(outside, "table.csv", b"private")
    if directory:
        (sample / "linked").symlink_to(outside, target_is_directory=True)
        target = "linked/table.csv"
    else:
        (sample / "table.csv").symlink_to(outside / "table.csv")
        target = "table.csv"
    _write(sample, "index.html", f'<a href="{target}">table</a>')
    records = _inventory(sample, "index.html") + [{"path": target, "size_bytes": 7, "sha256": "0" * 64}]
    data = _zip(sample.parent / "DATA.zip", {})
    original = Path.read_bytes

    def guarded(path):
        assert path == sample / "index.html", "a symlink target must never be opened"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert _reasons(plan_image_members(sample, ["index.html"], records)) == {"symlink_path"}
    checked = check_planned_pair(data, sample, ["index.html", target])
    assert "symlink_path" in _reasons(checked)


def test_exact_historical_exclusion_preserves_snapshots_without_hiding_current_failure(sample: Path):
    history = "91_reproducibility/current_figure_revision/revision/navigation_before/00_guide/index.html"
    files = {history: '<a href="../01_ellipse/index.html">old relative navigation</a>',
             "00_guide/index.html": '<a href="../plot.png">Current plot</a>', "plot.png": b"plot"}
    for name, text in files.items():
        _write(sample, name, text)
    data = _zip(sample.parent / "DATA.zip", {history: files[history]})
    image = _zip(sample.parent / "IMAGE.zip", {name: text for name, text in files.items() if name != history})
    assert check_archive_pair(data, image, sample_id="sample")["status"] == "blocked"
    for report in [plan_image_members(sample, files, _inventory(sample, *files), historical_html=[history]),
                   check_planned_pair(data, sample, files, historical_html=[history]),
                   check_archive_pair(data, image, sample_id="sample", historical_html=[history])]:
        assert report["status"] == "ready"
        assert report["historical_html"] == [history]
        assert history not in report["html_checked"]
    _write(sample, "00_guide/index.html", '<a href="missing.svg">stale current navigation</a>')
    failed = check_planned_pair(data, sample, files, historical_html=[history])
    assert failed["status"] == "blocked"
    assert failed["issues"][0]["source"] == "00_guide/index.html"
    bad_exclusion = check_planned_pair(data, sample, files, historical_html=["navigation_before/index.html"])
    assert "historical_html_not_found" in _reasons(bad_exclusion)


def test_only_live_html_is_read_never_data_payload_or_expected_hashes(sample: Path, monkeypatch):
    history = "navigation_before/index.html"
    data = _zip(sample.parent / "DATA.zip", {
        "results.npz": b"do not inflate", "table.csv": "do not read/hash",
        "index.html": b"do not even decode overwritten HTML", history: b"do not read excluded snapshot",
        "retained.html": '<a href="table.csv">DATA table</a>',
    })
    _write(sample, "index.html", '<a href="table.csv">retained DATA</a>')
    records = _inventory(sample, "index.html")
    image = _zip(sample.parent / "IMAGE.zip", {"index.html": (sample / "index.html").read_bytes()})
    reads: list[tuple[str, str]] = []
    original = zipfile.ZipFile.open

    def guarded(archive, name, *args, **kwargs):
        path = name.filename if isinstance(name, zipfile.ZipInfo) else name
        reads.append((str(archive.filename), path))
        assert path.endswith(".html")
        if Path(archive.filename) == data:
            assert path == "sample/retained.html"
        return original(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **k: pytest.fail("no content rehash during preflight"))
    for report in [plan_image_members(sample, ["index.html"], records, data_zip=data, historical_html=[history]),
                   check_planned_pair(data, sample, ["index.html"], historical_html=[history]),
                   check_archive_pair(data, image, sample_id="sample", historical_html=[history])]:
        assert report["status"] == "ready"
        assert report["content_hashes_verified"] is False
    assert reads


@pytest.mark.parametrize("change", ["size", "sha", "missing", "conflict", "boolean_size"])
def test_inventory_metadata_and_local_existence_fail_early(sample: Path, change: str):
    path = _write(sample, "plot.png", b"already rendered")
    rows = _inventory(sample, "plot.png")
    if change == "size":
        rows[0]["size_bytes"] += 1
        reason = "size_mismatch"
    elif change == "sha":
        rows[0]["sha256"] = "missing"
        reason = "invalid_artifact_record"
    elif change == "missing":
        path.unlink()
        reason = "missing_file"
    elif change == "conflict":
        rows.append({**rows[0], "sha256": "0" * 64})
        reason = "conflicting_artifact_record"
    else:
        rows[0]["size_bytes"] = True
        reason = "invalid_artifact_record"
    assert reason in _reasons(plan_image_members(sample, ["plot.png"], rows))


def test_expected_hash_is_not_falsely_reported_as_verified(sample: Path):
    _write(sample, "plot.png", b"same-size-content")
    rows = [{"path": "plot.png", "size_bytes": 17, "sha256": "0" * 64}]
    report = plan_image_members(sample, ["plot.png"], rows)
    assert report["status"] == "ready"
    assert report["members"] == rows
    assert report["content_hashes_verified"] is False


@pytest.mark.parametrize("kind", ["traversal", "backslash", "symlink", "duplicate", "wrong_sample", "case_collision"])
def test_unsafe_or_ambiguous_zip_members_fail_without_extraction(sample: Path, kind: str):
    data = _zip(sample.parent / "DATA.zip", {})
    image = sample.parent / "IMAGE.zip"
    with zipfile.ZipFile(image, "w") as archive:
        if kind == "symlink":
            info = zipfile.ZipInfo("sample/link.csv")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../../private.csv")
        elif kind == "duplicate":
            archive.writestr("sample/index.html", "old")
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("sample/index.html", "new")
        elif kind == "case_collision":
            archive.writestr("sample/plot.png", "one")
            archive.writestr("sample/Plot.png", "two")
        elif kind == "backslash":
            info = zipfile.ZipInfo("sample/index.html")
            # ZipInfo normalizes backslashes on Windows during construction.
            # Set the stored name afterward to create the intended unsafe ZIP.
            info.filename = info.orig_filename = "sample\\index.html"
            archive.writestr(info, "unsafe")
        else:
            name = {"traversal": "sample/../outside.csv",
                    "wrong_sample": "another_sample/index.html"}[kind]
            archive.writestr(name, "unsafe")
    if kind == "backslash":
        with zipfile.ZipFile(image) as archive:
            assert [info.orig_filename for info in archive.infolist()] == ["sample\\index.html"]
    report = check_archive_pair(data, image, sample_id="sample")
    assert report["status"] == "blocked"
    assert _reasons(report) & {"unsafe_archive_member", "nonregular_archive_member", "duplicate_archive_member",
                               "unexpected_sample_member", "case_colliding_member"}
    if kind == "backslash":
        assert _reasons(report) == {"unsafe_archive_member"}


def test_overlays_reject_file_directory_conflicts(sample: Path):
    data = _zip(sample.parent / "DATA.zip", {"figures": "a file cannot become a directory"})
    _write(sample, "figures/index.html", "gallery")
    assert _reasons(check_planned_pair(data, sample, ["figures/index.html"])) == {"file_directory_conflict"}


@pytest.mark.parametrize("text,reason", [
    ('<base href="other/"><a href="data.csv">data</a>', "html_base_url"),
    ('<img srcset="data:image/png;base64,abc 1x, plot.png 2x">', "complex_srcset"),
])
def test_parser_limits_are_reported_not_silently_accepted(sample: Path, text: str, reason: str):
    _write(sample, "index.html", text)
    report = plan_image_members(sample, ["index.html"], _inventory(sample, "index.html"))
    assert report["status"] == "blocked"
    assert _reasons(report) == {reason}


def test_reuse_verifies_sources_and_archive_without_any_write_or_compression(sample: Path, monkeypatch):
    from butterfly_saxs.delivery_pairs import reuse_image_archive

    payloads = {"index.html": '<a href="table.csv">data</a>', "table.csv": "q,value,quality\n0.1,0.42,FAIL\n"}
    for name, text in payloads.items():
        _write(sample, name, text)
    rows = _inventory(sample, *payloads)
    image = _zip(sample.parent / "IMAGE.zip", payloads)
    before = image.read_bytes(), image.stat().st_mtime_ns
    originals = {name: ((sample / name).read_bytes(), (sample / name).stat().st_mtime_ns) for name in payloads}
    original_open = Path.open
    original_zip_init = zipfile.ZipFile.__init__

    def read_only(path, mode="r", *args, **kwargs):
        assert mode in {"r", "rb"}, "reuse must never write"
        return original_open(path, mode, *args, **kwargs)

    def zip_read_only(archive, file, mode="r", *args, **kwargs):
        assert mode == "r", "reuse must never rebuild an archive"
        return original_zip_init(archive, file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", read_only)
    monkeypatch.setattr(zipfile.ZipFile, "__init__", zip_read_only)
    monkeypatch.setattr(zipfile, "_get_compressor", lambda *a, **k: pytest.fail("reuse recompressed content"))
    result = reuse_image_archive(sample, image, rows)
    assert result["status"] == "reused"
    assert result["content_hashes_verified"] is True
    assert result["member_hash_checks"] == result["source_hash_checks"] == result["member_count"] == 2
    assert result["sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert result["size_bytes"] == len(before[0])
    assert result["path"] == str(image)
    assert result["issues"] == []
    assert (image.read_bytes(), image.stat().st_mtime_ns) == before
    assert {name: ((sample / name).read_bytes(), (sample / name).stat().st_mtime_ns) for name in payloads} == originals


@pytest.mark.parametrize(("change", "reason"), [
    ("source_same_size", "source_hash_mismatch"), ("source_size", "size_mismatch"),
    ("archive_same_size", "archive_hash_mismatch"), ("archive_size", "archive_size_mismatch"),
    ("archive_extra", "archive_members_mismatch"), ("archive_missing", "archive_members_mismatch"),
])
def test_reuse_rejects_immutable_disagreements_without_repair(sample: Path, change: str, reason: str):
    from butterfly_saxs.delivery_pairs import reuse_image_archive

    _write(sample, "plot.png", b"original")
    rows = _inventory(sample, "plot.png")
    archived = {"plot.png": b"modified" if change == "archive_same_size" else b"original"}
    if change == "source_same_size":
        _write(sample, "plot.png", b"modified")
    elif change == "source_size":
        _write(sample, "plot.png", b"longer modified source")
    elif change == "archive_size":
        archived["plot.png"] = b"longer modified archive"
    elif change == "archive_extra":
        archived["extra.png"] = b"unknown"
    elif change == "archive_missing":
        archived = {}
    image = _zip(sample.parent / "IMAGE.zip", archived)
    before = image.read_bytes(), image.stat().st_mtime_ns
    result = reuse_image_archive(sample, image, rows)
    assert result["status"] == "blocked"
    assert reason in _reasons(result)
    assert result["content_hashes_verified"] is False
    assert (image.read_bytes(), image.stat().st_mtime_ns) == before


def test_reuse_rejects_duplicate_archive_and_planned_members(sample: Path):
    from butterfly_saxs.delivery_pairs import reuse_image_archive

    _write(sample, "plot.png", b"original")
    rows = _inventory(sample, "plot.png")
    image = _zip(sample.parent / "IMAGE.zip", {"plot.png": b"original"})
    with zipfile.ZipFile(image, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("sample/plot.png", b"original")
    result = reuse_image_archive(sample, image, rows)
    assert _reasons(result) == {"duplicate_archive_member"}
    assert "duplicate_or_invalid_planned_member" in _reasons(reuse_image_archive(sample, image, [*rows, *rows]))


def test_reuse_streams_large_payloads_in_bounded_chunks(sample: Path, monkeypatch):
    from butterfly_saxs.delivery_pairs import reuse_image_archive

    payload = b"0123456789abcdef" * (1024 * 1024 // 16 * 5 + 1)
    _write(sample, "render.png", payload)
    rows = _inventory(sample, "render.png")
    image = _zip(sample.parent / "IMAGE.zip", {"render.png": payload})
    real_path_open = Path.open
    real_zip_open = zipfile.ZipFile.open
    reads: list[int] = []

    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024, "unbounded payload read"
            reads.append(size)
            return self.handle.read(size)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

    monkeypatch.setattr(Path, "open", lambda path, *a, **k: BoundedReader(real_path_open(path, *a, **k)))
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda archive, *a, **k: BoundedReader(real_zip_open(archive, *a, **k)))
    result = reuse_image_archive(sample, image, rows)
    assert result["status"] == "reused"
    assert len(reads) >= 14  # Multiple bounded source and member reads, plus archive hash.


def test_reuse_crc_damage_is_blocked_without_recompression(sample: Path):
    from butterfly_saxs.delivery_pairs import reuse_image_archive

    _write(sample, "plot.png", b"original")
    rows = _inventory(sample, "plot.png")
    image = sample.parent / "IMAGE.zip"
    with zipfile.ZipFile(image, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("sample/plot.png", b"original")
    damaged = image.read_bytes().replace(b"original", b"modified", 1)
    image.write_bytes(damaged)
    result = reuse_image_archive(sample, image, rows)
    assert result["status"] == "blocked"
    assert _reasons(result) == {"unreadable_archive_member"}
    assert image.read_bytes() == damaged


@pytest.mark.parametrize("coarse_metadata", [False, True])
def test_reuse_detects_source_change_during_archive_verification(sample: Path, monkeypatch, coarse_metadata: bool):
    import os
    import butterfly_saxs.delivery_pairs as pairs
    from butterfly_saxs.delivery_pairs import reuse_image_archive

    _write(sample, "plot.png", b"original")
    rows = _inventory(sample, "plot.png")
    image = _zip(sample.parent / "IMAGE.zip", {"plot.png": b"original"})
    original = zipfile.ZipFile.open
    before = (sample / "plot.png").stat()
    if coarse_metadata:
        signature = (*pairs._file_signature(sample / "plot.png")[:-1], None)
        real_signature = pairs._file_signature
        monkeypatch.setattr(pairs, "_file_signature", lambda path: (
            signature if path == sample / "plot.png" else real_signature(path)
        ))

    def change_source(archive, *args, **kwargs):
        _write(sample, "plot.png", b"modified")
        os.utime(sample / "plot.png", ns=(before.st_atime_ns, before.st_mtime_ns))
        return original(archive, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", change_source)
    result = reuse_image_archive(sample, image, rows)
    assert result["status"] == "blocked"
    if coarse_metadata:
        assert _reasons(result) == {"source_hash_mismatch"}
    else:
        assert _reasons(result) <= {"source_changed_during_verification", "source_hash_mismatch"}
        assert result["issues"]
    assert result["content_hashes_verified"] is False


def test_reuse_never_creates_missing_archive(sample: Path):
    from butterfly_saxs.delivery_pairs import reuse_image_archive

    _write(sample, "plot.png", b"original")
    missing = sample.parent / "missing" / "IMAGE.zip"
    result = reuse_image_archive(sample, missing, _inventory(sample, "plot.png"))
    assert result["status"] == "blocked"
    assert _reasons(result) == {"unreadable_archive"}
    assert not missing.parent.exists()


def test_explicit_archive_directory_cannot_be_overwritten_with_file(sample: Path):
    data = _zip(sample.parent / "DATA.zip", {"directory/": b""})
    image = _zip(sample.parent / "IMAGE.zip", {"directory": "file"})
    report = check_archive_pair(data, image, sample_id="sample")
    assert _reasons(report) == {"file_directory_conflict"}


def test_effective_count_is_unique_file_count_after_overlay(sample: Path):
    data = _zip(sample.parent / "DATA.zip", {"index.html": "old", "table.csv": "data", "directory/": b""})
    _write(sample, "index.html", "current")
    _write(sample, "plot.png", b"plot")
    records = _inventory(sample, "index.html", "plot.png")
    image = _zip(sample.parent / "IMAGE.zip", {name: (sample / name).read_bytes() for name in ["index.html", "plot.png"]})
    for report in [plan_image_members(sample, ["index.html", "plot.png"], records, data_zip=data),
                   check_planned_pair(data, sample, ["index.html", "plot.png"]),
                   check_archive_pair(data, image, sample_id="sample")]:
        assert report["status"] == "ready"
        assert report["effective_member_count"] == 3


@pytest.mark.parametrize(("data_names", "image_name"), [
    ({"A": "file"}, "a/child.txt"),
    ({"a/child.txt": "file"}, "A"),
    ({"a/": b""}, "A"),
    ({"a/nested/": b""}, "A"),
])
def test_casefold_file_directory_conflicts_block_planned_and_archived_pairs(
    sample: Path, data_names: dict[str, str | bytes], image_name: str,
):
    data = _zip(sample.parent / "DATA.zip", data_names)
    _write(sample, image_name, "current IMAGE file")
    image = _zip(sample.parent / "IMAGE.zip", {image_name: "current IMAGE file"})
    for report in [plan_image_members(sample, [image_name], _inventory(sample, image_name), data_zip=data),
                   check_planned_pair(data, sample, [image_name]),
                   check_archive_pair(data, image, sample_id="sample")]:
        assert report["status"] == "blocked"
        assert _reasons(report) == {"file_directory_conflict"}


@pytest.mark.parametrize("directory", ["a/", "a/nested/"])
def test_casefold_image_explicit_directory_cannot_replace_data_file(sample: Path, directory: str):
    data = _zip(sample.parent / "DATA.zip", {"A": "regular file"})
    image = _zip(sample.parent / "IMAGE.zip", {directory: b""})
    report = check_archive_pair(data, image, sample_id="sample")
    assert report["status"] == "blocked"
    assert _reasons(report) == {"file_directory_conflict"}
