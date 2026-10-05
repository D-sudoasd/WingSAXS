from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.pipeline import PipelineError, _read_frame_bundle, inspect_frame
from butterfly_saxs.io import DatasetSelectionError, load_image


def _stacked_npz(path: Path, *, qmap_frames: int = 2) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    shape = (12, 14)
    data = np.stack(
        [np.ones(shape, dtype=np.float32), np.full(shape, 2.0, dtype=np.float32)]
    )
    yy, xx = np.indices(shape, dtype=float)
    qx = np.stack([xx, xx + 100.0])[:qmap_frames]
    qy = np.stack([yy - 5.0, yy - 5.0])[:qmap_frames]
    q = np.hypot(qx, qy)
    mask = np.zeros((2, *shape), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 2] = True
    valid_mask = np.ones((2, *shape), dtype=bool)
    valid_mask[0, 0, 1] = False
    valid_mask[1, 3, 4] = False
    np.savez(
        path,
        frames=data,
        qx=qx,
        qy=qy,
        q=q,
        mask=mask,
        valid_mask=valid_mask,
        q_unit=np.asarray("pixel-q"),
    )
    return shape, qx, qy


def test_npz_frame_selector_slices_embedded_coordinates_and_masks(tmp_path: Path) -> None:
    source = tmp_path / "frames.npz"
    shape, qx, qy = _stacked_npz(source)

    bundle = _read_frame_bundle(source, frame=1, dataset="frames")
    report = inspect_frame(source, frame=1, dataset="frames")

    assert report["shape"] == list(shape)
    assert report["qx_range"] == pytest.approx([float(qx[1].min()), float(qx[1].max())])
    assert report["qy_range"] == pytest.approx([float(qy[1].min()), float(qy[1].max())])
    assert bundle.qmap is not None
    assert bundle.qmap["mask"][1, 2]  # frame 1's external mask (True means excluded)
    assert not bundle.qmap["valid_mask"][3, 4]  # False means excluded
    assert not bundle.qmap["mask"][0, 0]  # frame 0's external mask was not carried over
    assert bundle.qmap["valid_mask"][0, 1]  # frame 0's validity was not carried over
    assert report["analysis_domain"]["detector_valid_count"] == shape[0] * shape[1] - 2


def test_npz_shared_two_dimensional_qmap_is_preserved_for_selected_image_frame(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shared-map.npz"
    shape = (12, 14)
    data = np.stack([np.ones(shape), np.full(shape, 2.0)])
    yy, xx = np.indices(shape, dtype=float)
    qx, qy = xx - 6.5, yy - 5.5
    np.savez(source, frames=data, qx=qx, qy=qy, q_unit=np.asarray("pixel-q"))

    report = inspect_frame(source, frame=1, dataset="frames")

    assert report["qx_range"] == pytest.approx([float(qx.min()), float(qx.max())])
    assert report["qy_range"] == pytest.approx([float(qy.min()), float(qy.max())])


def test_npz_per_frame_qmap_requires_an_explicit_frame_selector(tmp_path: Path) -> None:
    source = tmp_path / "map-stack.npz"
    shape = (12, 14)
    yy, xx = np.indices(shape, dtype=float)
    np.savez(
        source,
        image=np.ones(shape),
        qx=np.stack([xx, xx + 1.0]),
        qy=np.stack([yy, yy + 1.0]),
    )

    with pytest.raises(PipelineError, match="embedded qmap field .*explicit frame"):
        inspect_frame(source)


def test_npz_per_frame_qmap_rejects_a_selector_outside_its_frame_count(
    tmp_path: Path,
) -> None:
    source = tmp_path / "short-map-stack.npz"
    _stacked_npz(source, qmap_frames=1)

    with pytest.raises(PipelineError, match="outside NPZ embedded qmap field .*1 frames"):
        inspect_frame(source, frame=1, dataset="frames")


@pytest.mark.parametrize("explicit_dataset", [False, True])
def test_npz_pipeline_opens_once_and_parses_only_selected_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_dataset: bool,
) -> None:
    from numpy.lib.npyio import NpzFile

    source = tmp_path / "frames.npz"
    shape, qx, _ = _stacked_npz(source)
    original_load = np.load
    original_getitem = NpzFile.__getitem__
    opens = []
    parsed = []

    def counted_load(*args, **kwargs):
        opens.append(args[0])
        return original_load(*args, **kwargs)

    def counted_getitem(archive, key):
        parsed.append(key)
        return original_getitem(archive, key)

    monkeypatch.setattr(np, "load", counted_load)
    monkeypatch.setattr(NpzFile, "__getitem__", counted_getitem)
    bundle = _read_frame_bundle(source, frame=1, dataset="frames" if explicit_dataset else None)
    assert opens == [source]
    assert len(parsed) == len(set(parsed)) == 7
    np.testing.assert_array_equal(bundle.image, np.full(shape, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(bundle.qmap["qx"], qx[1])
    assert bundle.frame.dataset == "frames"
    assert bundle.frame.frame == 1
    source.unlink()  # The archive is closed before returning the bundle.


def test_npz_qmap_mode_keeps_ambiguity_and_skips_unselected_intensity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numpy.lib.npyio import NpzFile

    source = tmp_path / "ambiguous.npz"
    data = np.ones((3, 4), dtype=np.float32)
    np.savez(source, first=data, second=data * 2, qx=data, qy=data, q_unit="pixel-q")
    with pytest.raises(DatasetSelectionError):
        load_image(source)
    with pytest.raises(DatasetSelectionError):
        load_image(source, include_qmap=True)

    original_getitem = NpzFile.__getitem__
    parsed = []

    def counted_getitem(archive, key):
        parsed.append(key)
        return original_getitem(archive, key)

    monkeypatch.setattr(NpzFile, "__getitem__", counted_getitem)
    loaded = load_image(source, dataset="second", include_qmap=True)
    assert "first" not in parsed
    assert parsed.count("second") == 1
    np.testing.assert_array_equal(loaded.data, data * 2)
    assert loaded.qmap["q_unit"] == "pixel-q"


def test_npz_qmap_errors_close_archive_and_reject_nonscalar_unit(tmp_path: Path) -> None:
    source = tmp_path / "bad-unit.npz"
    data = np.ones((3, 4))
    np.savez(source, image=data, qx=data, qy=data, q_unit=["nm^-1", "pixel-q"])
    with pytest.raises(PipelineError, match="q_unit.*标量"):
        _read_frame_bundle(source)
    source.unlink()
