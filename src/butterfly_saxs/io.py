"""Input and mask handling for two-dimensional SAXS detector data.

The loader in this module deliberately has a small, strict contract:

* detector values are returned without normalisation or an implicit cast;
* a data source with more than one frame must be selected explicitly;
* HDF5/NPZ datasets are selected explicitly whenever the file is ambiguous;
* ``valid_mask`` uses ``True`` for a usable pixel, while an ``external_mask``
  uses ``True`` for a rejected pixel.

The latter distinction is important because both conventions are common in
beamline software.  The public result stores only the positive ``valid_mask``
and records the convention in its metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class DataIOError(ValueError):
    """Base error for a data source that cannot satisfy the loader contract."""


class UnsupportedFormatError(DataIOError):
    """The input suffix is not supported by the installed readers."""


class FrameSelectionError(DataIOError):
    """A frame was required, missing, malformed, or out of range."""


class DatasetSelectionError(DataIOError):
    """A dataset/key was required, missing, or ambiguous."""


class DataShapeError(DataIOError):
    """The selected object is not exactly one two-dimensional image."""


@dataclass
class LoadedImage:
    """A selected two-dimensional image and its source provenance.

    ``data`` is kept in the source dtype.  In particular, absolute intensity
    values are not divided by a monitor, exposure, or maximum value here.
    The caller can therefore apply a beamline-specific correction explicitly
    and retain an auditable record of that operation.
    """

    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None
    frame: int | None = None
    dataset: str | None = None
    valid_mask: np.ndarray | None = None
    qmap: dict[str, Any] | None = None
    uncertainty: np.ndarray | None = None
    uncertainty_source: str | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise DataShapeError(
                f"selected image must be 2-D, got shape {self.data.shape!r}"
            )
        if self.valid_mask is not None:
            self.valid_mask = _coerce_mask_array(
                self.valid_mask, self.data.shape, name="valid_mask"
            )
        if self.uncertainty is not None:
            unc = np.asarray(self.uncertainty)
            if unc.dtype.kind == "O":
                raise DataShapeError("uncertainty cannot be object-valued")
            if unc.shape != self.data.shape:
                raise DataShapeError(
                    f"uncertainty shape {unc.shape!r} does not exactly match "
                    f"image shape {self.data.shape!r}"
                )
            self.uncertainty = np.asarray(unc, dtype=np.float64)
        # Keep the source object useful even when callers construct it directly.
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("shape", list(self.data.shape))
        self.metadata.setdefault("dtype", str(self.data.dtype))
        self.metadata.setdefault("absolute_intensity_preserved", True)
        self.metadata.setdefault(
            "intensity_semantics",
            "source values unchanged; no implicit normalisation",
        )
        self.metadata.setdefault(
            "mask_semantics",
            {
                "valid_mask_true": "valid pixel",
                "external_mask_true": "invalid pixel",
            },
        )
        if self.uncertainty_source is not None:
            self.metadata.setdefault("uncertainty_source", self.uncertainty_source)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def mask(self) -> np.ndarray | None:
        """Compatibility alias using the detector convention (``True`` bad).

        New code should use :attr:`valid_mask` or :func:`combine_masks` to make
        the polarity explicit.  The domain ``ImageFrame`` uses this negative
        mask convention as well.
        """

        if self.valid_mask is None:
            return None
        return ~self.valid_mask

    @property
    def valid(self) -> np.ndarray | None:
        """Short alias for the positive valid mask."""

        return self.valid_mask

    @property
    def path(self) -> str | None:
        """Domain-model compatible source-path alias."""

        return None if self.source is None else str(self.source)

    @property
    def frame_id(self) -> int | None:
        return self.frame

    @property
    def preserves_absolute_intensity(self) -> bool:
        return bool(self.metadata.get("absolute_intensity_preserved", True))


# Names used by early clients and by the domain models are kept as aliases.
ImageFrame = LoadedImage
DataFrame = LoadedImage


_IMAGE_SUFFIXES = {
    ".cbf": "fabio",
    ".edf": "fabio",
    ".mccd": "fabio",
    ".img": "fabio",
    ".sfrm": "fabio",
    ".mar3450": "fabio",
    ".mar2300": "fabio",
    ".msk": "fabio",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".npy": "npy",
    ".npz": "npz",
    ".csv": "csv",
    ".txt": "csv",
    ".dat": "csv",
    ".png": "raster",
    ".bmp": "raster",
    ".jpg": "raster",
    ".jpeg": "raster",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".hdf": "hdf5",
}


DEFAULT_H5_UNCERTAINTY_PATHS: tuple[str, ...] = (
    "entry/data/uncertainty/statistical",
    "entry/data/uncertainty/sigma",
    "entry/data/uncertainty",
    "entry/data/errors",
    "entry/data/sigma",
    "entry/data/I_abs_2d_err",
    "entry/data/I_abs_2d_sigma",
    "uncertainty/statistical",
    "uncertainty/sigma",
    "uncertainty",
    "errors",
    "sigma",
)

_NPZ_AUTO_UNCERTAINTY_KEYS: tuple[str, ...] = (
    "uncertainty",
    "sigma",
    "errors",
    "statistical_uncertainty",
    "statistical",
)


def load_image(
    path: str | Path,
    *,
    frame: int | None = None,
    dataset: str | None = None,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
    mask_frame: int | None = None,
    mask_dataset: str | None = None,
    include_qmap: bool = False,
    uncertainty_dataset: str | None = None,
) -> LoadedImage:
    """Read one two-dimensional detector frame.

    Parameters
    ----------
    path:
        CBF/EDF/TIF/TIFF/NPY/NPZ/CSV/TXT/HDF5 input path.
    frame:
        Zero-based frame index.  It is mandatory for a selected array whose
        first dimension contains more than one image.  ``frame=0`` is accepted
        for a single 2-D image as a convenient explicit form.
    dataset:
        HDF5 dataset path or NPZ key.  Ambiguous files fail closed when this is
        omitted.
    valid_mask:
        Optional boolean-like array/path where ``True`` means valid.
    external_mask (or ``mask``):
        Optional boolean-like array/path where ``True`` means masked/invalid.
        The two mask arguments cannot be supplied together.
    mask_frame:
        Zero-based frame selector for a path-valued mask.  It is independent
        from the image ``frame`` selector and is required for a multi-frame
        mask.
    mask_dataset:
        Dataset/key selector for a path-valued mask.  It is independent from
        the image ``dataset`` selector and is required for an ambiguous mask
        source.  When omitted, a mask never inherits the image selector.
    include_qmap:
        Read reserved NPZ q-map fields beside the selected intensity in the
        same archive access. Only this mode excludes q-map keys when choosing
        a sole intensity dataset; multiple intensity datasets remain ambiguous.
    uncertainty_dataset:
        Optional companion statistical uncertainty dataset path/key, or
        ``"auto"`` to probe standard HDF5/NPZ companion paths (including
        ``/entry/data/uncertainty/statistical``).  ``None`` (default) skips
        uncertainty loading.
    """

    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"image source does not exist: {source}")
    if mask is not None:
        if external_mask is not None:
            raise DataIOError("provide only one of mask and external_mask")
        external_mask = mask
    _validate_frame_argument(frame)
    _validate_frame_argument(mask_frame)
    if dataset is not None and not isinstance(dataset, str):
        raise DatasetSelectionError("dataset must be a string path/key")
    if mask_dataset is not None and not isinstance(mask_dataset, str):
        raise DatasetSelectionError("mask_dataset must be a string path/key")
    if uncertainty_dataset is not None and not isinstance(uncertainty_dataset, str):
        raise DatasetSelectionError("uncertainty_dataset must be a string path/key or 'auto'")

    suffix = source.suffix.lower()
    kind = _IMAGE_SUFFIXES.get(suffix)
    if kind is None:
        raise UnsupportedFormatError(
            f"unsupported image format {suffix or '<no suffix>'!r}; "
            f"supported: {', '.join(sorted(_IMAGE_SUFFIXES))}"
        )

    qmap = None
    uncertainty: np.ndarray | None = None
    uncertainty_source: str | None = None
    is_auto_uncertainty = (
        uncertainty_dataset is not None
        and uncertainty_dataset.strip().casefold() == "auto"
    )
    if uncertainty_dataset is not None and not uncertainty_dataset.strip():
        raise DatasetSelectionError("uncertainty_dataset path/key cannot be empty")
    if uncertainty_dataset is not None and not is_auto_uncertainty and kind not in {"hdf5", "npz"}:
        raise DatasetSelectionError(
            f"explicit uncertainty_dataset {uncertainty_dataset!r} is only supported for HDF5/NPZ sources"
        )

    if kind == "fabio":
        array, metadata, selected_frame = _read_fabio(source, frame=frame)
        selected_dataset = None
    elif kind == "tiff":
        array, metadata, selected_frame, selected_dataset = _read_tiff(
            source, frame=frame, dataset=dataset
        )
    elif kind == "raster":
        if dataset is not None:
            raise DatasetSelectionError("dataset is not applicable to a raster image")
        if frame is not None and int(frame) != 0:
            raise FrameSelectionError(
                f"raster source {source} contains one image; frame must be 0"
            )
        from PIL import Image

        with Image.open(source) as image:
            if image.mode not in {"1", "L", "I", "I;16", "I;16L", "I;16B", "I;16N", "F"}:
                raise DataShapeError(
                    "colour/palette images are not scalar intensity data; "
                    "export the original 2-D intensity matrix or a grayscale image"
                )
            array = np.array(image)
        metadata = {"format": suffix.lstrip("."), "raster_intensity": True}
        if suffix in {".jpg", ".jpeg"}:
            metadata["warning"] = "lossy JPEG intensity; use original detector values for quantitative measurements"
        selected_frame = 0 if frame is not None else None
        selected_dataset = None
    elif kind == "npy":
        if dataset is not None:
            raise DatasetSelectionError("dataset is not applicable to NPY")
        array = _safe_numpy_load(source)
        metadata = {"format": "npy"}
        array, selected_frame = _select_frame(
            array, frame=frame, source=source, source_kind="NPY"
        )
        selected_dataset = None
    elif kind == "npz":
        array, metadata, selected_frame, selected_dataset, qmap, uncertainty, uncertainty_source = _read_npz(
            source,
            frame=frame,
            dataset=dataset,
            include_qmap=include_qmap,
            uncertainty_dataset=uncertainty_dataset,
        )
    elif kind == "csv":
        if dataset is not None:
            raise DatasetSelectionError("dataset is not applicable to CSV/TXT")
        try:
            array = np.loadtxt(
                source,
                delimiter="," if suffix == ".csv" else None,
                encoding="utf-8-sig",
            )
        except (OSError, ValueError) as exc:
            raise DataIOError(f"could not read tabular image {source}: {exc}") from exc
        metadata = {"format": "csv" if suffix == ".csv" else "txt"}
        array, selected_frame = _select_frame(
            array, frame=frame, source=source, source_kind="CSV/TXT"
        )
        selected_dataset = None
    else:
        array, metadata, selected_frame, selected_dataset, uncertainty, uncertainty_source = _read_hdf5(
            source,
            frame=frame,
            dataset=dataset,
            uncertainty_dataset=uncertainty_dataset,
        )

    data = np.asarray(array)
    if data.ndim != 2:
        raise DataShapeError(
            f"selected image from {source} must be exactly 2-D, got {data.shape!r}"
        )
    if data.dtype.kind == "O":
        raise DataShapeError("object-valued image arrays are not supported")

    # HDF5 views must be detached before the file closes; copying every selected
    # array also makes the result independent of FabIO/TIFF file handles.
    data = np.array(data, copy=True)
    if uncertainty is not None:
        uncertainty = np.array(uncertainty, dtype=np.float64, copy=True)
    metadata = dict(metadata)
    metadata.update(
        {
            "source": str(source),
            "format": metadata.get("format", kind),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "frame": selected_frame,
            "dataset": selected_dataset,
            "absolute_intensity_preserved": True,
            "intensity_semantics": "source values unchanged; no implicit normalisation",
            "mask_semantics": {
                "valid_mask_true": "valid pixel",
                "external_mask_true": "invalid pixel",
            },
        }
    )
    if uncertainty_source is not None:
        metadata["uncertainty_source"] = uncertainty_source

    combined_mask = combine_masks(
        data.shape,
        valid_mask=_mask_value(
            valid_mask, mask_frame=mask_frame, mask_dataset=mask_dataset
        ),
        external_mask=_mask_value(
            external_mask, mask_frame=mask_frame, mask_dataset=mask_dataset
        ),
    )
    if combined_mask is not None:
        metadata["valid_pixel_count"] = int(np.count_nonzero(combined_mask))
        metadata["invalid_pixel_count"] = int(combined_mask.size - np.count_nonzero(combined_mask))

    return LoadedImage(
        data=data,
        metadata=metadata,
        source=source,
        frame=selected_frame,
        dataset=selected_dataset,
        valid_mask=combined_mask,
        qmap=qmap,
        uncertainty=uncertainty,
        uncertainty_source=uncertainty_source,
    )


def read_image(*args: Any, **kwargs: Any) -> LoadedImage:
    """Alias for :func:`load_image`."""

    return load_image(*args, **kwargs)


def load_frame(*args: Any, **kwargs: Any) -> LoadedImage:
    """Alias for :func:`load_image`."""

    return load_image(*args, **kwargs)


def read_data(*args: Any, **kwargs: Any) -> LoadedImage:
    """Alias for :func:`load_image`."""

    return load_image(*args, **kwargs)


def combine_masks(
    shape: Sequence[int],
    *,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
) -> np.ndarray | None:
    """Combine positive and external mask conventions into ``valid_mask``.

    ``valid_mask=True`` means a pixel is retained; ``external_mask=True``
    means a pixel is rejected.  The returned array is always boolean and has
    exactly ``shape``; broadcasting is intentionally not allowed.
    """

    if mask is not None:
        if external_mask is not None:
            raise DataIOError("provide only one of mask and external_mask")
        external_mask = mask
    shape_tuple = _validate_shape(shape)
    if valid_mask is None and external_mask is None:
        return None
    result = np.ones(shape_tuple, dtype=bool)
    if valid_mask is not None:
        result &= _load_mask_value(valid_mask, shape_tuple, name="valid_mask")
    if external_mask is not None:
        result &= ~_load_mask_value(
            external_mask, shape_tuple, name="external_mask"
        )
    return result


def resolve_valid_mask(
    shape: Sequence[int],
    *,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
) -> np.ndarray | None:
    """Explicitly named alias for :func:`combine_masks`."""

    return combine_masks(
        shape,
        valid_mask=valid_mask,
        external_mask=external_mask,
        mask=mask,
    )


def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
    if isinstance(shape, (str, bytes)):
        raise DataShapeError(f"image shape must contain two integers, got {shape!r}")
    try:
        values = tuple(shape)
    except TypeError as exc:
        raise DataShapeError(f"image shape must contain two integers, got {shape!r}") from exc
    if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values):
        raise DataShapeError(f"image shape must contain two integers, got {shape!r}")
    out = (int(values[0]), int(values[1]))
    if out[0] <= 0 or out[1] <= 0:
        raise DataShapeError(f"image shape must be positive, got {out!r}")
    return out


def _validate_frame_argument(frame: int | None) -> None:
    if frame is None:
        return
    if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
        raise FrameSelectionError("frame must be a non-negative integer or None")
    if int(frame) < 0:
        raise FrameSelectionError("frame must be a non-negative integer")


def _as_detector_plane(array: np.ndarray, *, source: Path, source_kind: str) -> np.ndarray:
    """Reject colour/channel-last volumes; squeeze a trailing singleton."""

    if array.ndim != 3:
        return array
    height, width, channels = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    if channels not in {1, 3, 4} or height < 4 or width < 4 or channels >= min(height, width):
        return array
    if channels == 1:
        return np.squeeze(array, axis=-1)
    raise DataShapeError(
        f"{source_kind} source {source} looks like a colour/channel-last array "
        f"{tuple(int(v) for v in array.shape)}; convert it to a 2-D intensity image"
    )


def _select_frame(
    array: Any,
    *,
    frame: int | None,
    source: Path,
    source_kind: str,
) -> tuple[np.ndarray, int | None]:
    ary = _as_detector_plane(np.asarray(array), source=source, source_kind=source_kind)
    if ary.ndim < 2:
        raise DataShapeError(
            f"selected object from {source} must be 2-D, got {ary.shape!r}"
        )
    if ary.ndim == 2:
        if frame is not None and int(frame) != 0:
            raise FrameSelectionError(
                f"{source_kind} source {source} contains one image; frame must be 0"
            )
        return ary, 0 if frame is not None else None
    if frame is None:
        raise FrameSelectionError(
            f"{source_kind} source {source} contains multiple dimensions/frames; "
            "select an explicit frame=..."
        )
    index = int(frame)
    if index >= ary.shape[0]:
        raise FrameSelectionError(
            f"frame {index} is outside {source_kind} source {source} "
            f"with {ary.shape[0]} frames"
        )
    selected = ary[index]
    if np.asarray(selected).ndim != 2:
        raise DataShapeError(
            f"frame {index} from {source} is not a 2-D image: {np.asarray(selected).shape!r}"
        )
    return np.asarray(selected), index


def _safe_numpy_load(source: Path) -> np.ndarray:
    try:
        value = np.load(source, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise DataIOError(f"could not read NPY source {source}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise DataShapeError(f"NPY source {source} did not contain an ndarray")
    return value


def _read_npz(
    source: Path,
    *,
    frame: int | None,
    dataset: str | None,
    include_qmap: bool,
    uncertainty_dataset: str | None = None,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    int | None,
    str,
    dict[str, Any] | None,
    np.ndarray | None,
    str | None,
]:
    if dataset is not None and not dataset:
        raise DatasetSelectionError("NPZ dataset/key cannot be empty")
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise DataIOError(f"could not read NPZ source {source}: {exc}") from exc
    try:
        keys = list(archive.files)
        qmap_keys = {"qx", "qy", "q", "theta", "chi", "mask", "valid_mask", "q_unit"}
        if dataset is None:
            candidates = (
                [key for key in keys if key not in qmap_keys]
                if include_qmap and len(keys) > 1 else keys
            )
            if len(candidates) != 1:
                raise DatasetSelectionError(
                    f"NPZ source {source} has datasets {keys!r}; select dataset='...'"
                )
            selected_dataset = candidates[0]
        else:
            selected_dataset = dataset
            if selected_dataset not in keys:
                raise DatasetSelectionError(
                    f"dataset {selected_dataset!r} not found in NPZ source {source}; "
                    f"available: {keys!r}"
                )
        array = archive[selected_dataset]
        array, selected_frame = _select_frame(
            array, frame=frame, source=source, source_kind="NPZ"
        )
        metadata = {"format": "npz", "datasets": keys}
        qmap: dict[str, Any] = {}
        if include_qmap:
            for key in qmap_keys - {"q_unit"}:
                if key in keys:
                    qmap[key] = _select_npz_qmap_field(
                        archive[key], key=key, frame=selected_frame, image_shape=array.shape
                    )
            if "q_unit" in keys:
                raw_unit = np.asarray(archive["q_unit"])
                if raw_unit.ndim != 0:
                    raise DataShapeError("NPZ 中的 q_unit 必须是单个标量字符串")
                raw_scalar = raw_unit.item()
                if isinstance(raw_scalar, (bytes, bytearray, np.bytes_)):
                    qmap["q_unit"] = bytes(raw_scalar).decode("utf-8")
                else:
                    qmap["q_unit"] = str(raw_scalar)
        uncertainty: np.ndarray | None = None
        uncertainty_source: str | None = None
        if uncertainty_dataset is not None:
            token = uncertainty_dataset.strip()
            if token.casefold() == "auto":
                for cand_key in _NPZ_AUTO_UNCERTAINTY_KEYS:
                    if cand_key not in keys or cand_key == selected_dataset:
                        continue
                    try:
                        raw_unc = archive[cand_key]
                        unc_arr, _ = _select_frame(
                            raw_unc, frame=selected_frame, source=source, source_kind="NPZ"
                        )
                        unc_arr = np.asarray(unc_arr, dtype=np.float64)
                    except (DataIOError, TypeError, ValueError):
                        continue
                    if unc_arr.shape == array.shape and np.any(np.isfinite(unc_arr)):
                        uncertainty = unc_arr
                        uncertainty_source = cand_key
                        break
            else:
                if token not in keys:
                    raise DatasetSelectionError(
                        f"uncertainty dataset {uncertainty_dataset!r} not found in NPZ source "
                        f"{source}; available: {keys!r}"
                    )
                unc_arr, _ = _select_frame(
                    archive[token], frame=selected_frame, source=source, source_kind="NPZ"
                )
                unc_arr = np.asarray(unc_arr)
                if unc_arr.dtype.kind == "O":
                    raise DataShapeError("object-valued uncertainty arrays are not supported")
                if unc_arr.shape != array.shape:
                    raise DataShapeError(
                        f"uncertainty dataset {token!r} shape {unc_arr.shape!r} does not match "
                        f"image shape {array.shape!r}"
                    )
                uncertainty = np.asarray(unc_arr, dtype=np.float64)
                uncertainty_source = token
        return (
            array,
            metadata,
            selected_frame,
            selected_dataset,
            qmap or None,
            uncertainty,
            uncertainty_source,
        )
    finally:
        archive.close()


def _select_npz_qmap_field(
    value: Any,
    *,
    key: str,
    frame: int | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Select a frame from an NPZ map while retaining shared 2-D maps."""

    array = np.asarray(value)
    if array.ndim != 3:
        # The q-map validator handles malformed non-frame fields.
        return array
    if frame is None:
        raise FrameSelectionError(
            f"NPZ embedded qmap field {key!r} contains {array.shape[0]} frames; "
            "select an explicit frame=..."
        )
    if frame >= array.shape[0]:
        raise FrameSelectionError(
            f"frame {frame} is outside NPZ embedded qmap field {key!r} "
            f"with {array.shape[0]} frames"
        )
    selected = np.asarray(array[frame])
    if selected.shape != image_shape:
        raise DataShapeError(
            f"selected NPZ qmap field {key!r} must have image shape "
            f"{image_shape}; frame {frame} has shape {selected.shape}"
        )
    return selected


def _read_tiff(
    source: Path, *, frame: int | None, dataset: str | None
) -> tuple[np.ndarray, dict[str, Any], int | None, str | None]:
    try:
        import tifffile

        with tifffile.TiffFile(source) as tif:
            pages = len(tif.pages)
            series_axes = [getattr(s, "axes", None) for s in tif.series]
            imagej_metadata = tif.imagej_metadata
            ome_metadata = tif.ome_metadata
            series_count = len(tif.series)
            if not series_count:
                raise DataShapeError(f"TIFF source {source} contains no image series")
            if dataset is None:
                if series_count > 1:
                    available = ", ".join(
                        f"series:{index} (shape={tuple(series.shape)!r}, axes={series.axes!r})"
                        for index, series in enumerate(tif.series)
                    )
                    raise DatasetSelectionError(
                        f"TIFF source {source} has multiple image series; select "
                        f"dataset='series:<index>'; available: {available}"
                    )
                series_index = 0
                selected_dataset = None
            else:
                selector = dataset.strip()
                prefix, separator, index_text = selector.partition(":")
                if (
                    not separator
                    or prefix.casefold() != "series"
                    or not index_text.isascii()
                    or not index_text.isdigit()
                ):
                    raise DatasetSelectionError(
                        "TIFF dataset must use the form 'series:<zero-based index>'"
                    )
                series_index = int(index_text)
                if series_index >= series_count:
                    raise DatasetSelectionError(
                        f"TIFF series index {series_index} is outside {source}; "
                        f"available indexes: 0 through {series_count - 1}"
                    )
                selected_dataset = f"series:{series_index}"
            selected_series = tif.series[series_index]
            tags: dict[str, Any] = {}
            if selected_series.pages:
                for key, tag in selected_series.pages[0].tags.items():
                    try:
                        tags[str(key)] = tag.value
                    except Exception:
                        # A malformed vendor tag must not prevent intensity data
                        # from being read; the tag itself is simply omitted.
                        continue
            series_shape = tuple(int(dim) for dim in selected_series.shape)
            selected: np.ndarray | None = None
            selected_frame: int | None = None
            if (
                frame is not None
                and len(series_shape) == 3
                and not _looks_like_channel_last_shape(series_shape)
                and len(selected_series.pages) == series_shape[0]
            ):
                frame_index = int(frame)
                if frame_index >= series_shape[0]:
                    raise FrameSelectionError(
                        f"frame {frame_index} is outside TIFF source {source} "
                        f"with {series_shape[0]} frames"
                    )
                page = selected_series.pages[frame_index]
                if page is not None:
                    page_array = np.asarray(page.asarray())
                    if page_array.shape == (series_shape[1], series_shape[2]):
                        selected = page_array
                        selected_frame = frame_index
            if selected is None:
                array = selected_series.asarray()
            selected_series_metadata = {
                "index": series_index,
                "shape": list(selected_series.shape),
                "axes": getattr(selected_series, "axes", None),
            }
    except (DatasetSelectionError, FrameSelectionError, DataShapeError):
        raise
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataIOError("TIFF support requires tifffile") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise DataIOError(f"could not read TIFF source {source}: {exc}") from exc
    metadata: dict[str, Any] = {
        "format": "tiff",
        "page_count": pages,
        "series_axes": series_axes,
        "selected_series": selected_series_metadata,
        "imagej_metadata": imagej_metadata,
        "ome_metadata": ome_metadata,
        "tags": tags,
    }
    if selected is None:
        selected, selected_frame = _select_frame(
            array, frame=frame, source=source, source_kind="TIFF"
        )
    return selected, metadata, selected_frame, selected_dataset


def _close_fabio(handle: Any) -> None:
    closer = getattr(handle, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:
        return


def _read_fabio(
    source: Path, *, frame: int | None
) -> tuple[np.ndarray, dict[str, Any], int | None]:
    first = None
    image = None
    try:
        import fabio

        first = fabio.open(str(source))
        nframes = int(getattr(first, "nframes", 1) or 1)
        if frame is None:
            if nframes > 1:
                raise FrameSelectionError(
                    f"FabIO source {source} contains {nframes} frames; "
                    "select an explicit frame=..."
                )
            image = first
            selected_frame = None
        else:
            if int(frame) >= nframes:
                raise FrameSelectionError(
                    f"frame {int(frame)} is outside FabIO source {source} "
                    f"with {nframes} frames"
                )
            image = fabio.open(str(source), frame=int(frame))
            selected_frame = int(frame)
        data = np.array(image.data, copy=True)
        header = dict(getattr(image, "header", {}) or {})
        metadata = {
            "format": source.suffix.lower().lstrip("."),
            "fabio_class": type(image).__name__,
            "frame_count": nframes,
            "header": header,
        }
        return data, metadata, selected_frame
    except FrameSelectionError:
        raise
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataIOError("CBF/EDF support requires fabio") from exc
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        raise DataIOError(f"could not read FabIO source {source}: {exc}") from exc
    finally:
        if image is not None and image is not first:
            _close_fabio(image)
        _close_fabio(first)


def _looks_like_channel_last_shape(shape: Sequence[int]) -> bool:
    if len(shape) != 3:
        return False
    height, width, channels = int(shape[0]), int(shape[1]), int(shape[2])
    return channels in {1, 3, 4} and height >= 4 and width >= 4 and channels < min(height, width)


def _read_hdf5_node_frame(
    node: Any,
    *,
    frame: int | None,
    source: Path,
) -> tuple[np.ndarray, int | None]:
    """Read only the requested frame from an HDF5 dataset when possible."""

    shape = tuple(int(dim) for dim in getattr(node, "shape", ()))
    if len(shape) == 3 and not _looks_like_channel_last_shape(shape):
        if frame is None:
            raise FrameSelectionError(
                f"HDF5 source {source} contains multiple dimensions/frames; "
                "select an explicit frame=..."
            )
        index = int(frame)
        if index >= shape[0]:
            raise FrameSelectionError(
                f"frame {index} is outside HDF5 source {source} with {shape[0]} frames"
            )
        selected = np.asarray(node[index])
        if selected.ndim != 2:
            raise DataShapeError(
                f"frame {index} from {source} is not a 2-D image: {selected.shape!r}"
            )
        return selected, index
    array = np.asarray(node[...])
    return _select_frame(array, frame=frame, source=source, source_kind="HDF5")


def _candidate_h5_uncertainty_paths(image_dataset: str | None) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        cleaned = path.lstrip("/")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)

    if image_dataset:
        cleaned_img = image_dataset.lstrip("/")
        parent, sep, leaf = cleaned_img.rpartition("/")
        if sep and parent:
            add(f"{parent}/uncertainty/statistical")
            add(f"{parent}/uncertainty/sigma")
            add(f"{parent}/uncertainty")
            add(f"{parent}/errors")
            add(f"{parent}/sigma")
            if leaf:
                add(f"{parent}/{leaf}_err")
                add(f"{parent}/{leaf}_errors")
                add(f"{parent}/{leaf}_sigma")
        elif leaf:
            add(f"{leaf}_err")
            add(f"{leaf}_sigma")
    for default_path in DEFAULT_H5_UNCERTAINTY_PATHS:
        add(default_path)
    return candidates


def _read_hdf5_uncertainty_from_handle(
    handle: Any,
    *,
    h5py_module: Any,
    source: Path,
    uncertainty_dataset: str,
    frame: int | None,
    image_dataset: str | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    token = uncertainty_dataset.strip()
    if not token:
        raise DatasetSelectionError("HDF5 uncertainty_dataset path cannot be empty")

    if token.casefold() == "auto":
        primary_clean = image_dataset.lstrip("/") if image_dataset else None
        for cand_path in _candidate_h5_uncertainty_paths(image_dataset):
            if cand_path == primary_clean:
                continue
            node = handle.get(cand_path)
            if node is None or not isinstance(node, h5py_module.Dataset):
                continue
            if getattr(node.dtype, "kind", "O") not in {"f", "i", "u"}:
                continue
            try:
                raw_arr, _ = _read_hdf5_node_frame(node, frame=frame, source=source)
            except (FrameSelectionError, DataShapeError):
                continue
            if raw_arr.ndim != 2:
                continue
            if expected_shape is not None and tuple(raw_arr.shape) != expected_shape:
                continue
            unc = np.asarray(raw_arr, dtype=np.float64)
            if not np.any(np.isfinite(unc)):
                continue
            return np.array(unc, dtype=np.float64, copy=True), cand_path
        return None, None

    selected = token.lstrip("/")
    if not selected:
        raise DatasetSelectionError("HDF5 uncertainty_dataset path cannot be empty")
    node = handle.get(selected)
    if node is None or not isinstance(node, h5py_module.Dataset):
        raise DatasetSelectionError(
            f"uncertainty dataset {uncertainty_dataset!r} not found in HDF5 source {source}"
        )
    if getattr(node.dtype, "kind", "O") == "O":
        raise DataShapeError("object-valued uncertainty arrays are not supported")
    raw_arr, _ = _read_hdf5_node_frame(node, frame=frame, source=source)
    if raw_arr.ndim != 2:
        raise DataShapeError(
            f"uncertainty dataset {uncertainty_dataset!r} from {source} must be 2-D, "
            f"got {raw_arr.shape!r}"
        )
    if expected_shape is not None and tuple(raw_arr.shape) != expected_shape:
        raise DataShapeError(
            f"uncertainty dataset {uncertainty_dataset!r} shape {raw_arr.shape!r} "
            f"does not match image shape {expected_shape!r}"
        )
    unc = np.asarray(raw_arr, dtype=np.float64)
    return np.array(unc, dtype=np.float64, copy=True), selected


def load_h5_uncertainty(
    path: str | Path,
    *,
    uncertainty_dataset: str = "auto",
    frame: int | None = None,
    image_dataset: str | None = None,
    expected_shape: Sequence[int] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Load a companion statistical uncertainty array from an HDF5 file.

    When ``uncertainty_dataset="auto"`` (default), probes standard companion
    paths (including ``/entry/data/uncertainty/statistical``) without scanning
    the full HDF5 tree and returns ``(uncertainty_array, dataset_path)`` or
    ``(None, None)`` if no matching companion dataset exists.  An explicit path
    fails closed with :class:`DatasetSelectionError` if absent.
    """

    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"HDF5 source does not exist: {source}")
    _validate_frame_argument(frame)
    if not isinstance(uncertainty_dataset, str):
        raise DatasetSelectionError("uncertainty_dataset must be a string path or 'auto'")
    shape_tuple = _validate_shape(expected_shape) if expected_shape is not None else None
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DataIOError("HDF5 support requires h5py; install butterfly-saxs[hdf5]") from exc
    try:
        with h5py.File(source, "r") as handle:
            return _read_hdf5_uncertainty_from_handle(
                handle,
                h5py_module=h5py,
                source=source,
                uncertainty_dataset=uncertainty_dataset,
                frame=frame,
                image_dataset=image_dataset,
                expected_shape=shape_tuple,
            )
    except (DatasetSelectionError, FrameSelectionError, DataShapeError):
        raise
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        raise DataIOError(f"could not read HDF5 uncertainty from {source}: {exc}") from exc


def _read_hdf5(
    source: Path,
    *,
    frame: int | None,
    dataset: str | None,
    uncertainty_dataset: str | None = None,
) -> tuple[np.ndarray, dict[str, Any], int | None, str, np.ndarray | None, str | None]:
    if dataset is not None and not dataset:
        raise DatasetSelectionError("HDF5 dataset path cannot be empty")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DataIOError("HDF5 support requires h5py; install butterfly-saxs[hdf5]") from exc
    try:
        with h5py.File(source, "r") as handle:
            datasets: list[str]
            discovery_mode: str
            node: Any = None
            selected_dataset: str

            if dataset is not None:
                selected_dataset = dataset.lstrip("/")
                if not selected_dataset:
                    raise DatasetSelectionError("HDF5 dataset path cannot be empty")
                # Fast path: open the explicit dataset directly without a full
                # HDF5 tree traversal via handle.visititems().
                candidate = handle.get(selected_dataset)
                if candidate is not None and isinstance(candidate, h5py.Dataset):
                    node = candidate
                    datasets = [selected_dataset]
                    discovery_mode = "direct"
                else:
                    datasets = []

                    def visit_missing(name: str, obj: Any) -> None:
                        if isinstance(obj, h5py.Dataset):
                            datasets.append(name)

                    handle.visititems(visit_missing)
                    raise DatasetSelectionError(
                        f"dataset {dataset!r} not found in HDF5 source {source}; "
                        f"available: {datasets!r}"
                    )
            else:
                datasets = []

                def visit(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset):
                        datasets.append(name)

                handle.visititems(visit)
                if len(datasets) != 1:
                    raise DatasetSelectionError(
                        f"HDF5 source {source} has datasets {datasets!r}; "
                        "select dataset='...' explicitly"
                    )
                selected_dataset = datasets[0]
                node = handle[selected_dataset]
                discovery_mode = "tree_scan"

            array, selected_frame = _read_hdf5_node_frame(
                node, frame=frame, source=source
            )
            uncertainty: np.ndarray | None = None
            uncertainty_source: str | None = None
            if uncertainty_dataset is not None:
                uncertainty, uncertainty_source = _read_hdf5_uncertainty_from_handle(
                    handle,
                    h5py_module=h5py,
                    source=source,
                    uncertainty_dataset=uncertainty_dataset,
                    frame=selected_frame,
                    image_dataset=selected_dataset,
                    expected_shape=(int(array.shape[0]), int(array.shape[1])),
                )
            metadata: dict[str, Any] = {
                "format": "hdf5",
                "datasets": datasets,
                "dataset_discovery": discovery_mode,
                "file_attrs": _attribute_dict(handle.attrs),
                "dataset_attrs": _attribute_dict(node.attrs),
            }
            return array, metadata, selected_frame, selected_dataset, uncertainty, uncertainty_source
    except (DatasetSelectionError, FrameSelectionError, DataShapeError):
        raise
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        raise DataIOError(f"could not read HDF5 source {source}: {exc}") from exc


def _attribute_dict(attrs: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in attrs:
        try:
            value = attrs[key]
            if isinstance(value, np.generic):
                value = value.item()
            result[str(key)] = value
        except Exception:
            continue
    return result


def _mask_value(
    value: Any | None,
    *,
    mask_frame: int | None,
    mask_dataset: str | None,
) -> Any | None:
    """Load a path-valued mask using only its own explicit selectors.

    The image's frame/dataset must not leak into an independent mask source.
    Leaving these selectors unset intentionally delegates ambiguity handling to
    :func:`load_image`, which fails closed for multi-frame or multi-dataset
    masks.
    """

    if value is None or not isinstance(value, (str, Path)):
        return value
    loaded = load_image(value, frame=mask_frame, dataset=mask_dataset)
    return loaded.data


def _load_mask_value(value: Any, shape: tuple[int, int], *, name: str) -> np.ndarray:
    if isinstance(value, (str, Path)):
        value = load_image(value).data
    return _coerce_mask_array(value, shape, name=name)


def _coerce_mask_array(value: Any, shape: Sequence[int], *, name: str) -> np.ndarray:
    expected = _validate_shape(shape)
    array = np.asarray(value)
    if array.shape != expected:
        raise DataShapeError(
            f"{name} shape {array.shape!r} does not exactly match image shape {expected!r}"
        )
    if array.dtype.kind == "O":
        raise DataShapeError(f"{name} cannot be object-valued")
    if array.dtype.kind == "f":
        if name == "valid_mask":
            return np.asarray(np.isfinite(array) & (array != 0), dtype=bool)
        return np.asarray((~np.isfinite(array)) | (array != 0), dtype=bool)
    return np.asarray(array != 0, dtype=bool)


__all__ = [
    "DEFAULT_H5_UNCERTAINTY_PATHS",
    "DataFrame",
    "DataIOError",
    "DataShapeError",
    "DatasetSelectionError",
    "FrameSelectionError",
    "ImageFrame",
    "LoadedImage",
    "UnsupportedFormatError",
    "combine_masks",
    "load_frame",
    "load_h5_uncertainty",
    "load_image",
    "read_data",
    "read_image",
    "resolve_valid_mask",
]
