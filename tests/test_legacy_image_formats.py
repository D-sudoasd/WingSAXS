import numpy as np
import pytest

from butterfly_saxs.io import DataShapeError, load_image


def test_grayscale_png_retains_16bit_intensity(tmp_path):
    from PIL import Image

    image = np.array([[0, 8192], [65535, 23]], dtype=np.uint16)
    path = tmp_path / "detector.png"
    Image.fromarray(image).save(path)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.data, image)
    assert loaded.preserves_absolute_intensity


def test_colour_png_does_not_silently_take_red_channel(tmp_path):
    from PIL import Image

    path = tmp_path / "colour.png"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(path)
    with pytest.raises(DataShapeError, match="scalar intensity"):
        load_image(path)


def test_dat_is_a_2d_matrix(tmp_path):
    path = tmp_path / "frame.dat"
    path.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    np.testing.assert_array_equal(load_image(path).data, [[1, 2, 3], [4, 5, 6]])


def test_dtrek_roundtrip(tmp_path):
    from fabio.dtrekimage import DtrekImage

    data = np.arange(80, dtype=np.uint16).reshape(8, 10)
    path = tmp_path / "frame.img"
    DtrekImage(data=data).write(str(path))
    np.testing.assert_array_equal(load_image(path).data, data)
