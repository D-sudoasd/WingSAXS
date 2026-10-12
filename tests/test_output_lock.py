from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading

import pytest

from butterfly_saxs.output_lock import output_lock


def _child_script(body: str) -> str:
    return (
        "import sys\n"
        "import os\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from butterfly_saxs.output_lock import output_lock\n"
        f"{body}\n"
    )


def _start_child(root: Path, body: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _child_script(body), str(Path(__file__).resolve().parents[1] / "src"), str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_missing_root_is_not_created_unless_requested(tmp_path: Path):
    root = tmp_path / "new-output"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        with output_lock(root):
            pytest.fail("missing output directory was created")
    assert not root.exists()

    with output_lock(root, create=True):
        assert root.is_dir()
        assert (root / ".wingsaxs.lock").is_file()
    assert (root / ".wingsaxs.lock").is_file()


def test_same_thread_can_reenter_normalized_root_and_lock_file_remains(tmp_path: Path):
    root = tmp_path / "outputs"
    root.mkdir()
    with output_lock(root):
        with output_lock(root / "."):
            assert (root / ".wingsaxs.lock").is_file()
    assert (root / ".wingsaxs.lock").is_file()


def test_different_thread_is_rejected_but_another_root_is_independent(tmp_path: Path):
    root = tmp_path / "outputs"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with output_lock(root):
            entered.set()
            assert release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert entered.wait(10)
        with pytest.raises(OSError, match="locked.*wait.*retry"):
            with output_lock(root):
                pass

        acquired_other: list[bool] = []

        def acquire_other() -> None:
            with output_lock(other):
                acquired_other.append(True)

        independent = threading.Thread(target=acquire_other)
        independent.start()
        independent.join(10)
        assert not independent.is_alive()
        assert acquired_other == [True]
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()


def test_subprocess_cannot_acquire_while_parent_holds_lock(tmp_path: Path):
    root = tmp_path / "outputs"
    root.mkdir()
    with output_lock(root):
        result = subprocess.run(
            [sys.executable, "-c", _child_script(
                "try:\n"
                "    with output_lock(sys.argv[2]):\n"
                "        print('acquired')\n"
                "except OSError:\n"
                "    print('blocked')"
            ), str(Path(__file__).resolve().parents[1] / "src"), str(root)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.stdout.strip() == "blocked"


def test_subprocess_release_and_abnormal_exit_release_os_lock(tmp_path: Path):
    root = tmp_path / "outputs"
    root.mkdir()

    child = _start_child(
        root,
        "with output_lock(sys.argv[2]):\n"
        "    print('ready', flush=True)\n"
        "    sys.stdin.readline()",
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(OSError, match="locked"):
            with output_lock(root):
                pass
        assert child.stdin is not None
        child.stdin.write("release\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    with output_lock(root):
        pass

    crashed = _start_child(
        root,
        "with output_lock(sys.argv[2]):\n"
        "    print('ready', flush=True)\n"
        "    os._exit(23)",
    )
    try:
        assert crashed.stdout is not None
        assert crashed.stdout.readline().strip() == "ready"
        assert crashed.wait(timeout=10) == 23
    finally:
        if crashed.poll() is None:
            crashed.kill()
            crashed.wait(timeout=10)
    with output_lock(root):
        pass


def test_package_conflict_does_not_change_any_existing_output(tmp_path: Path):
    from butterfly_saxs.delivery import package_batch

    root = tmp_path / "outputs"
    root.mkdir()
    for name, payload in {
        "frame_summary.csv": b"frame_index,status\n0,ok\n",
        "delivery.zip": b"previous archive",
        "delivery_summary.json": b"previous summary",
        "delivery_index.html": b"previous index",
    }.items():
        (root / name).write_bytes(payload)

    entered = threading.Event()
    release = threading.Event()
    failure: list[BaseException] = []

    def holder() -> None:
        try:
            with output_lock(root):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("test lock was not released")
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert entered.wait(10)
        before = {path.relative_to(root).as_posix(): path.read_bytes()
                  for path in root.rglob("*") if path.is_file() and path.name != ".wingsaxs.lock"}
        lock_path = root / ".wingsaxs.lock"
        lock_stat = lock_path.stat()
        with pytest.raises(OSError, match="locked"):
            package_batch(root, resume=True)
        after = {path.relative_to(root).as_posix(): path.read_bytes()
                 for path in root.rglob("*") if path.is_file() and path.name != ".wingsaxs.lock"}
        assert after == before
        assert lock_path.is_file()
        assert lock_path.stat().st_mtime_ns == lock_stat.st_mtime_ns
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert not failure
