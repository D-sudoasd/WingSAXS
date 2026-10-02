from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.io import DatasetSelectionError, FrameSelectionError, load_image


def _write_two_series(path):
    tifffile = pytest.importorskip("tifffile")
    first = np.full((8, 9), 1, dtype=np.uint16)
    second = np.full((6, 7), 2, dtype=np.uint16)
    with tifffile.TiffWriter(path) as writer:
        writer.write(first, metadata={"axes": "YX"}, description="first-series")
        writer.write(second, metadata={"axes": "YX"}, description="second-series")
    return first, second


def test_multi_series_tiff_requires_actionable_series_selector(tmp_path):
    source = tmp_path / "multi-series.tif"
    _write_two_series(source)

    with pytest.raises(DatasetSelectionError) as caught:
        load_image(source)

    message = str(caught.value)
    assert "dataset='series:<index>'" in message
    assert "series:0" in message and "shape=(8, 9)" in message
    assert "series:1" in message and "shape=(6, 7)" in message
    assert "axes='YX'" in message


def test_selected_tiff_series_supplies_data_tags_and_loaded_selector(tmp_path):
    source = tmp_path / "multi-series.tif"
    _, expected = _write_two_series(source)

    loaded = load_image(source, dataset="series:1")

    np.testing.assert_array_equal(loaded.data, expected)
    assert loaded.dataset == "series:1"
    assert loaded.metadata["dataset"] == "series:1"
    assert loaded.metadata["selected_series"] == {
        "index": 1,
        "shape": [6, 7],
        "axes": "YX",
    }
    # TIFF tag names are exposed as their numeric codes; these dimensions must
    # come from series 1's first page rather than the file's first page.
    assert loaded.metadata["tags"]["256"] == 7
    assert loaded.metadata["tags"]["257"] == 6


@pytest.mark.parametrize("selector", ["dataset", "series:", "series:-1", "series:1x"])
def test_tiff_rejects_invalid_series_selector(tmp_path, selector):
    source = tmp_path / "multi-series.tif"
    _write_two_series(source)

    with pytest.raises(DatasetSelectionError, match="series:<zero-based index>"):
        load_image(source, dataset=selector)


def test_tiff_rejects_series_index_outside_file(tmp_path):
    source = tmp_path / "multi-series.tif"
    _write_two_series(source)

    with pytest.raises(DatasetSelectionError, match="available indexes: 0 through 1"):
        load_image(source, dataset="series:2")


def test_single_series_tiff_keeps_default_selection_without_dataset(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    source = tmp_path / "single-series.tif"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    tifffile.imwrite(source, expected, metadata={"axes": "YX"})

    loaded = load_image(source)

    np.testing.assert_array_equal(loaded.data, expected)
    assert loaded.dataset is None
    assert loaded.metadata["selected_series"]["index"] == 0


def test_single_tiff_series_frame_selector_still_selects_page(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    source = tmp_path / "stack.tif"
    expected = np.stack(
        [np.full((4, 5), 3, dtype=np.uint8), np.full((4, 5), 8, dtype=np.uint8)]
    )
    tifffile.imwrite(source, expected, metadata={"axes": "TYX"})

    loaded = load_image(source, frame=1)

    np.testing.assert_array_equal(loaded.data, expected[1])
    assert loaded.frame == 1
    assert loaded.dataset is None


def test_tiff_frame_selector_validates_within_selected_series(tmp_path):
    source = tmp_path / "multi-series.tif"
    _write_two_series(source)

    with pytest.raises(FrameSelectionError):
        load_image(source, dataset="series:1", frame=1)
