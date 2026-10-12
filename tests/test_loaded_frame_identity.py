import numpy as np

from butterfly_saxs.io import LoadedImage, load_image
from butterfly_saxs.pipeline import analyze_frame, export_result, synthetic_butterfly


def test_loaded_frames_preserve_source_identity_and_distinct_export_names(tmp_path):
    image, qmap = synthetic_butterfly((40, 40), return_qmap=True, seed=23)
    first, second = tmp_path / "first.npy", tmp_path / "second.npy"
    np.save(first, image)
    np.save(second, image)
    loaded = [load_image(first), LoadedImage(image, source=second)]
    output_paths = []
    for frame, path in zip(loaded, (first, second), strict=True):
        result = analyze_frame(frame, qmap=qmap)
        assert result.metadata["path"] == str(path)
        output_paths.extend(export_result(result, tmp_path / "results"))
    names = {path.name for path in output_paths}
    assert {"first.json", "first.npz", "second.json", "second.npz"} <= names


def test_loaded_frames_keep_selectors_for_shared_source(tmp_path):
    image, qmap = synthetic_butterfly((40, 40), return_qmap=True, seed=23)
    source = tmp_path / "series.npz"
    paths = []
    for index in (0, 1):
        loaded = LoadedImage(image, source=source, frame=index, dataset="images")
        result = analyze_frame(loaded, qmap=qmap)
        assert result.metadata["path"] == str(source)
        assert result.metadata["frame"] == index
        assert result.metadata["dataset"] == "images"
        paths.extend(export_result(result, tmp_path / "results"))
    assert {path.name for path in paths} == {
        f"series__frame_{index:04d}__dataset_images.{extension}"
        for index in (0, 1) for extension in ("json", "npz")
    }
