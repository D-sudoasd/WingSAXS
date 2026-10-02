from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import socket
from urllib.parse import unquote, urlsplit
import urllib.request
import zipfile

import pytest

from butterfly_saxs.delivery import package_batch


def _native_sample(root: Path) -> None:
    """Minimal native outputs; packaging must not interpret or regenerate them."""
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "frame_summary.csv": b"frame_index,frame_id,status,quality_status\n0,frame-0,ok,\n",
        "parameters_long.csv": b"parameter,value,stderr,unit\nratio,0.420000000000001,0.003,1\n",
        "manifest.json": b'{"schema_version":"future.native.version","original":true}\n',
        "results.npz": b"opaque native arrays; not a recomputation input",
        "evolution.png": b"opaque native evolution figure",
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _assert_unchanged(root: Path, originals: dict[str, tuple[bytes, int]]) -> None:
    for relative, (payload, modified) in originals.items():
        assert (root / relative).read_bytes() == payload
        assert (root / relative).stat().st_mtime_ns == modified


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.references.extend(
            value for key, value in attrs
            if key in {"href", "src", "data", "poster"} and value
        )


def _assert_extracted_links_resolve(extracted: Path) -> None:
    """Follow actual delivered HTML links, rather than just checking ZIP names."""
    for path in extracted.rglob("*"):
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        parser = _References()
        parser.feed(path.read_text(encoding="utf-8"))
        for reference in parser.references:
            url = urlsplit(reference)
            if url.scheme or url.netloc or not url.path:
                continue
            destination = (path.parent / unquote(url.path)).resolve()
            assert destination.is_relative_to(extracted.resolve()), reference
            assert destination.is_file(), f"{path.relative_to(extracted)} -> {reference}"


def test_gallery_csv_dependency_survives_extraction_without_changing_native_outputs(tmp_path: Path):
    root = tmp_path / "batch"
    sample = root / "sample A #1"
    _native_sample(sample)
    _write(sample, "figures/gallery.html", '<a href="../profile.csv">Profile data</a>')
    csv = _write(sample, "profile.csv", "q,intensity,stderr\n0.1,123.456789012345,0.002\n")
    _write(root, "unrelated-private.csv", "must not be swept into the bundle\n")
    originals = _snapshot(root)

    report = package_batch(root)

    assert report["status"] == "ready"
    assert report["exit_code"] == 0
    assert report["dependencies"]["issues"] == []
    assert "sample A #1/profile.csv" in report["dependencies"]["included"]
    with zipfile.ZipFile(root / "delivery.zip") as bundle:
        assert bundle.testzip() is None
        assert bundle.read("sample A #1/profile.csv") == csv.read_bytes()
        assert "unrelated-private.csv" not in bundle.namelist()
        for relative, (payload, _) in originals.items():
            if relative.startswith("sample A #1/"):
                assert bundle.read(relative) == payload
        bundle.extractall(tmp_path / "extracted")
    _assert_extracted_links_resolve(tmp_path / "extracted")
    _assert_unchanged(root, originals)


def test_recursive_html_cycles_encoded_paths_and_all_link_attributes(tmp_path: Path):
    root = tmp_path / "batch"
    sample = root / "sample"
    _native_sample(sample)
    _write(sample, "figures/gallery.html", '<a href="../reports/step%20one.htm?view=1#data">More</a>')
    _write(sample, "reports/step one.htm", """
        <a href="../../shared/data%20%231.csv?download=1#row-1">Table</a>
        <a href="cycle.html">Next</a>
        <img src="../../shared/preview%20%231.png?size=full#image">
        <object data="../../shared/profile%20%231.pdf#page=1"></object>
        <video poster="../../shared/poster%20%231.png"></video>
        <a href="#data">Same page</a>
    """)
    _write(sample, "reports/cycle.html", """
        <a href="step%20one.htm#data">Back</a>
        <a href="../figures/gallery.html">Gallery</a>
        <a href="../../shared/data%20%231.csv?download=2">Same table</a>
    """)
    dependencies = {
        "sample/reports/step one.htm",
        "sample/reports/cycle.html",
        "shared/data #1.csv",
        "shared/preview #1.png",
        "shared/profile #1.pdf",
        "shared/poster #1.png",
    }
    for name in sorted(dependencies):
        if not (root / name).exists():
            _write(root, name, f"original bytes for {name}\n")
    originals = _snapshot(root)

    report = package_batch(root)

    included = report["dependencies"]["included"]
    assert dependencies <= set(included)
    assert len(included) == len(set(included))
    assert report["dependencies"]["issues"] == []
    assert report["exit_code"] == 0
    with zipfile.ZipFile(root / "delivery.zip") as bundle:
        assert len(bundle.namelist()) == len(set(bundle.namelist()))
        for relative, (payload, _) in originals.items():
            assert bundle.read(relative) == payload
        bundle.extractall(tmp_path / "extracted")
    _assert_extracted_links_resolve(tmp_path / "extracted")
    _assert_unchanged(root, originals)


@pytest.mark.parametrize("kind", [
    "missing", "outside_root", "encoded_outside_root", "absolute_path",
    "hidden_directory", "hidden_file", "symlink_file", "symlink_directory",
])
def test_missing_or_unsafe_dependency_leaves_actionable_usable_partial_bundle(tmp_path: Path, kind: str):
    root = tmp_path / "batch"
    _native_sample(root)
    secret = b"private data that must never enter the delivery ZIP"
    outside = tmp_path / "outside.csv"
    outside.write_bytes(secret)
    if kind == "missing":
        reference = "../not-yet-exported.csv"
    elif kind == "outside_root":
        reference = "../../outside.csv"
    elif kind == "encoded_outside_root":
        reference = "%2E%2E/%2E%2E/outside.csv"
    elif kind == "absolute_path":
        reference = str(outside)
    elif kind == "hidden_directory":
        _write(root, ".private/secret.csv", secret.decode())
        reference = "../.private/secret.csv"
    elif kind == "hidden_file":
        _write(root, ".private.csv", secret.decode())
        reference = "../.private.csv"
    else:
        try:
            if kind == "symlink_file":
                (root / "linked.csv").symlink_to(outside)
                reference = "../linked.csv"
            else:
                # Even an in-root target must not bypass the no-symlink rule.
                private = _write(root, "private/secret.csv", secret.decode()).parent
                (root / "linked").symlink_to(private, target_is_directory=True)
                reference = "../linked/secret.csv"
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    _write(root, "figures/gallery.html", f'<a href="{reference}">Linked data</a>')

    report = package_batch(root)

    assert report["status"] == "incomplete"
    assert report["exit_code"] == 1
    assert report["archive"]["status"] == "created"
    issues = report["dependencies"]["issues"]
    matching = [issue for issue in issues if issue["reference"] == reference]
    assert len(matching) == 1
    issue = matching[0]
    assert issue["source"] == "figures/gallery.html"
    assert isinstance(issue["reason"], str) and issue["reason"].strip()
    assert isinstance(issue["action"], str) and issue["action"].strip()
    assert report["next_action"]
    with zipfile.ZipFile(root / "delivery.zip") as bundle:
        assert bundle.testzip() is None
        assert bundle.read("parameters_long.csv") == (root / "parameters_long.csv").read_bytes()
        assert bundle.read("results.npz") == (root / "results.npz").read_bytes()
        assert "figures/gallery.html" in bundle.namelist()
        assert all(not name.startswith("/") and ".." not in Path(name).parts for name in bundle.namelist())
        assert all(secret not in bundle.read(name) for name in bundle.namelist())
        assert "linked.csv" not in bundle.namelist()
        assert "linked/secret.csv" not in bundle.namelist()
        assert ".private/secret.csv" not in bundle.namelist()
        assert ".private.csv" not in bundle.namelist()


def test_external_links_are_reported_without_fetching_or_marking_local_bundle_incomplete(tmp_path: Path, monkeypatch):
    root = tmp_path / "batch"
    _native_sample(root)
    references = ["https://example.invalid/never-fetch.csv?download=1", "//example.invalid/never-fetch.png"]
    _write(root, "figures/gallery.html", f'<a href="{references[0]}">Remote</a><img src="{references[1]}">')

    def no_network(*args, **kwargs):
        pytest.fail("packaging attempted to fetch an external HTML dependency")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)

    report = package_batch(root)

    assert report["status"] == "ready"
    assert report["exit_code"] == 0
    assert report["dependencies"]["issues"] == []
    external = report["dependencies"]["external"]
    for reference in references:
        assert any(reference in str(item) for item in external)
    with zipfile.ZipFile(root / "delivery.zip") as bundle:
        assert not any("never-fetch" in name for name in bundle.namelist())
        assert bundle.read("figures/gallery.html") == (root / "figures/gallery.html").read_bytes()


def test_resume_includes_newly_available_html_linked_csv_and_clears_incomplete_status(tmp_path: Path):
    root = tmp_path / "batch"
    _native_sample(root)
    _write(root, "figures/gallery.html", '<a href="../exported%20profile.csv?download=1#data">CSV</a>')
    originals = _snapshot(root)

    first = package_batch(root)

    assert first["status"] == "incomplete"
    assert first["exit_code"] == 1
    assert first["dependencies"]["issues"]
    csv = _write(root, "exported profile.csv", "q,intensity\n0.1,2.300000000000004\n")
    completed = package_batch(root, resume=True)

    assert completed["status"] == "ready"
    assert completed["exit_code"] == 0
    assert completed["operation_status"] == "completed"
    assert completed["archive"]["status"] == "created"
    assert completed["dependencies"]["issues"] == []
    assert "exported profile.csv" in completed["dependencies"]["included"]
    assert completed["source_signature"] != first["source_signature"]
    with zipfile.ZipFile(root / "delivery.zip") as bundle:
        assert bundle.read("exported profile.csv") == csv.read_bytes()
        bundle.extractall(tmp_path / "extracted")
    _assert_extracted_links_resolve(tmp_path / "extracted")
    _assert_unchanged(root, originals)


def test_gallery_can_link_to_generated_index_without_copying_stale_generation(tmp_path: Path):
    root = tmp_path / 'batch'
    _native_sample(root)
    _write(root, 'figures/gallery.html', '<a href="../delivery_index.html#sample-0">Home</a>')
    first = package_batch(root)
    assert first['status'] == 'ready'
    second = package_batch(root, resume=True)
    assert second['status'] == 'ready'
    assert second['archive']['status'] == 'reused'
    with zipfile.ZipFile(root / 'delivery.zip') as bundle:
        assert bundle.namelist().count('delivery_index.html') == 1
        bundle.extractall(tmp_path / 'extracted')
    _assert_extracted_links_resolve(tmp_path / 'extracted')


@pytest.mark.parametrize('reference', ['../delivery_summary.json', '../delivery.zip', '../delivery_contents.json',
                                      '../' + 'a' * 270 + '.csv', '../bad%00name.csv'])
def test_unreadable_or_circular_generated_reference_is_actionable_on_first_run_and_resume(tmp_path: Path, reference: str):
    root = tmp_path / 'batch'
    _native_sample(root)
    _write(root, 'figures/gallery.html', f'<a href="{reference}">Reference</a>')
    for kwargs in ({}, {'resume': True}):
        report = package_batch(root, **kwargs)
        assert report['status'] == 'incomplete'
        assert report['exit_code'] == 1
        assert report['dependencies']['issues'][0]['reference'] == reference
        assert report['dependencies']['issues'][0]['action']
    with zipfile.ZipFile(root / 'delivery.zip') as bundle:
        assert bundle.namelist().count('delivery_index.html') == 1
        assert 'delivery.zip' not in bundle.namelist()
        assert 'delivery_summary.json' not in bundle.namelist()


def test_linked_scientific_files_do_not_require_a_filename_extension_allowlist(tmp_path: Path):
    root = tmp_path / 'batch'
    _native_sample(root)
    names = ['profile.dat', 'calibration.poni', 'arrays.h5', 'recipe.toml', 'extensionless']
    _write(root, 'figures/gallery.html', ''.join(f'<a href="../data/{name}">{name}</a>' for name in names))
    for name in names:
        _write(root, f'data/{name}', f'original unmodified {name} bytes')
    _write(root, 'data/unrelated-private.dat', 'unreferenced bytes')
    report = package_batch(root)
    assert report['status'] == 'ready'
    with zipfile.ZipFile(root / 'delivery.zip') as bundle:
        for name in names:
            assert bundle.read(f'data/{name}') == (root / 'data' / name).read_bytes()
        assert 'data/unrelated-private.dat' not in bundle.namelist()


def test_generated_index_case_must_match_the_portable_zip_name(tmp_path: Path):
    root = tmp_path / 'batch'
    _native_sample(root)
    _write(root, 'figures/gallery.html', '<a href="../DELIVERY_INDEX.HTML">Home</a>')
    report = package_batch(root)
    assert report['status'] == 'incomplete'
    assert report['dependencies']['issues'][0]['reason'] == 'generated_delivery_reference'
