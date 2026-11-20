from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.io import (
    DEFAULT_H5_UNCERTAINTY_PATHS,
    DataShapeError,
    DatasetSelectionError,
    FrameSelectionError,
    load_h5_uncertainty,
    load_image,
)


def test_hdf5_explicit_dataset_skips_visititems_tree_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "bl19b2_frame.h5"
    expected = np.arange(48, dtype=np.float32).reshape(6, 8) * 1.5
    sigma = np.full((6, 8), 0.25, dtype=np.float64)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data/I_abs_2d", data=expected)
        handle.create_dataset("entry/data/uncertainty/statistical", data=sigma)
        handle.create_dataset("entry/instrument/extra_tree/node", data=np.zeros((2, 2)))

    visit_calls = 0
    original_visititems = h5py.Group.visititems

    def forbidden_visititems(self, *args, **kwargs):
        nonlocal visit_calls
        visit_calls += 1
        return original_visititems(self, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "visititems", forbidden_visititems)

    loaded = load_image(source, dataset="/entry/data/I_abs_2d")
    assert visit_calls == 0
    assert loaded.dataset == "entry/data/I_abs_2d"
    assert loaded.metadata["dataset_discovery"] == "direct"
    assert loaded.uncertainty is None
    assert loaded.uncertainty_source is None
    assert loaded.data.dtype == expected.dtype
    np.testing.assert_array_equal(loaded.data, expected)

    # Explicit dataset + auto uncertainty also avoids full-tree visititems.
    loaded_with_unc = load_image(
        source,
        dataset="/entry/data/I_abs_2d",
        uncertainty_dataset="auto",
    )
    assert visit_calls == 0
    assert loaded_with_unc.uncertainty is not None
    assert loaded_with_unc.uncertainty.dtype == np.float64
    assert loaded_with_unc.uncertainty_source == "entry/data/uncertainty/statistical"
    assert loaded_with_unc.metadata["uncertainty_source"] == "entry/data/uncertainty/statistical"
    np.testing.assert_allclose(loaded_with_unc.uncertainty, sigma)


def test_hdf5_missing_explicit_dataset_falls_back_to_discovery_for_error(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "missing_ds.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data/I_abs_2d", data=np.ones((4, 4), dtype=np.float32))

    with pytest.raises(DatasetSelectionError, match="entry/data/I_abs_2d"):
        load_image(source, dataset="/entry/data/nonexistent")


def test_hdf5_multiframe_explicit_dataset_slices_frame_without_visititems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "stack.h5"
    stack = np.arange(3 * 6 * 7, dtype=np.float32).reshape(3, 6, 7)
    sigma_stack = np.full((3, 6, 7), 0.5, dtype=np.float64)
    sigma_stack[2] = 1.25
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data/data", data=stack)
        handle.create_dataset("entry/data/uncertainty/statistical", data=sigma_stack)

    visit_calls = 0
    original_visititems = h5py.Group.visititems

    def counting_visititems(self, *args, **kwargs):
        nonlocal visit_calls
        visit_calls += 1
        return original_visititems(self, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "visititems", counting_visititems)

    with pytest.raises(FrameSelectionError, match="explicit frame"):
        load_image(source, dataset="entry/data/data")
    with pytest.raises(FrameSelectionError, match="outside HDF5"):
        load_image(source, dataset="entry/data/data", frame=5)

    loaded = load_image(
        source,
        dataset="/entry/data/data",
        frame=2,
        uncertainty_dataset="auto",
    )
    assert visit_calls == 0
    assert loaded.frame == 2
    np.testing.assert_array_equal(loaded.data, stack[2])
    assert loaded.uncertainty is not None
    np.testing.assert_allclose(loaded.uncertainty, sigma_stack[2])
    assert loaded.uncertainty_source == "entry/data/uncertainty/statistical"


def test_hdf5_uncertainty_loader_guards_and_fallback_paths(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    assert "entry/data/uncertainty/statistical" in DEFAULT_H5_UNCERTAINTY_PATHS

    # File with no companion uncertainty returns (None, None) in auto mode.
    plain_h5 = tmp_path / "plain.h5"
    with h5py.File(plain_h5, "w") as handle:
        handle.create_dataset("entry/data/I_abs_2d", data=np.ones((5, 5), dtype=np.float32))

    loaded_plain = load_image(
        plain_h5, dataset="/entry/data/I_abs_2d", uncertainty_dataset="auto"
    )
    assert loaded_plain.uncertainty is None
    assert loaded_plain.uncertainty_source is None
    assert load_h5_uncertainty(plain_h5, uncertainty_dataset="auto") == (None, None)

    # Explicit missing uncertainty path fails closed.
    with pytest.raises(DatasetSelectionError, match="uncertainty dataset"):
        load_image(
            plain_h5,
            dataset="/entry/data/I_abs_2d",
            uncertainty_dataset="/entry/data/uncertainty/statistical",
        )
    with pytest.raises(DatasetSelectionError, match="uncertainty dataset"):
        load_h5_uncertainty(plain_h5, uncertainty_dataset="/entry/data/missing_sigma")

    # Explicit shape-mismatched uncertainty fails closed with DataShapeError.
    bad_shape_h5 = tmp_path / "bad_shape.h5"
    with h5py.File(bad_shape_h5, "w") as handle:
        handle.create_dataset("entry/data/I_abs_2d", data=np.ones((5, 5), dtype=np.float32))
        handle.create_dataset(
            "entry/data/uncertainty/statistical", data=np.ones((3, 3), dtype=np.float64)
        )
    with pytest.raises(DataShapeError, match="does not match image shape"):
        load_image(
            bad_shape_h5,
            dataset="/entry/data/I_abs_2d",
            uncertainty_dataset="/entry/data/uncertainty/statistical",
        )
    # In auto mode, shape-mismatched companion datasets are safely skipped.
    loaded_skip = load_image(
        bad_shape_h5,
        dataset="/entry/data/I_abs_2d",
        uncertainty_dataset="auto",
    )
    assert loaded_skip.uncertainty is None
    assert loaded_skip.uncertainty_source is None

    # Standalone load_h5_uncertainty returns exact array on valid companion path.
    good_h5 = tmp_path / "good.h5"
    unc = np.linspace(0.1, 1.0, 20, dtype=np.float64).reshape(4, 5)
    with h5py.File(good_h5, "w") as handle:
        handle.create_dataset("entry/data/I_abs_2d", data=np.ones((4, 5), dtype=np.float32))
        handle.create_dataset("entry/data/uncertainty/statistical", data=unc)
    loaded_unc, unc_path = load_h5_uncertainty(
        good_h5,
        uncertainty_dataset="auto",
        image_dataset="/entry/data/I_abs_2d",
        expected_shape=(4, 5),
    )
    assert unc_path == "entry/data/uncertainty/statistical"
    assert loaded_unc is not None
    np.testing.assert_allclose(loaded_unc, unc)

    # Non-HDF5 formats degrade gracefully for "auto" and fail closed for explicit paths.
    npy_path = tmp_path / "frame.npy"
    np.save(npy_path, np.ones((4, 4), dtype=np.float32))
    loaded_npy = load_image(npy_path, uncertainty_dataset="auto")
    assert loaded_npy.uncertainty is None
    assert loaded_npy.uncertainty_source is None
    with pytest.raises(DatasetSelectionError, match="HDF5/NPZ"):
        load_image(npy_path, uncertainty_dataset="sigma")
