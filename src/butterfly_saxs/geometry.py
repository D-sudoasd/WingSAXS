"""Detector geometry and physical coordinate maps.

All PONI values are interpreted by pyFAI.  This module intentionally does not
reimplement detector rotations or pixel coordinates: ``pyFAI.load`` creates the
official :class:`AzimuthalIntegrator`, and its ``qArray`` plus
``center_array('chi_rad')`` methods are used for every pixel.  The in-plane
components are then the unambiguous polar decomposition
``qx = q*cos(chi)``, ``qy = q*sin(chi)`` (0 degrees is +qx, 90 degrees is
+qy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class GeometryError(ValueError):
    """A PONI file or geometry map cannot satisfy the strict contract."""


class GeometryShapeError(GeometryError):
    """The requested map shape is invalid or does not match a mask."""


@dataclass
class GeometryMaps:
    """Physical coordinates and the mask associated with one detector shape."""

    q_nm_inv: np.ndarray
    chi_rad: np.ndarray
    qx_nm_inv: np.ndarray
    qy_nm_inv: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    chi_deg: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {
            "q_nm_inv": self.q_nm_inv,
            "chi_rad": self.chi_rad,
            "qx_nm_inv": self.qx_nm_inv,
            "qy_nm_inv": self.qy_nm_inv,
        }
        converted: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value, dtype=np.float64)
            converted[name] = array
        shapes = {name: value.shape for name, value in converted.items()}
        if len(set(shapes.values())) != 1:
            raise GeometryShapeError(f"geometry maps have inconsistent shapes: {shapes!r}")
        if len(next(iter(shapes.values()))) != 2:
            raise GeometryShapeError(
                f"geometry maps must be 2-D, got {next(iter(shapes.values()))!r}"
            )
        mask = np.asarray(self.valid_mask, dtype=bool)
        if mask.shape != next(iter(shapes.values())):
            raise GeometryShapeError(
                f"valid_mask shape {mask.shape!r} does not match map shape "
                f"{next(iter(shapes.values()))!r}"
            )
        for name, value in converted.items():
            setattr(self, name, value)
        self.valid_mask = mask
        if self.chi_deg is None:
            self.chi_deg = np.degrees(self.chi_rad) % 360.0
        else:
            chi_deg_arr = np.asarray(self.chi_deg, dtype=np.float64)
            if chi_deg_arr.shape != mask.shape:
                raise GeometryShapeError("chi_deg shape does not match geometry maps")
            self.chi_deg = np.mod(chi_deg_arr, 360.0)
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("shape", list(mask.shape))
        self.metadata.setdefault("q_unit", "nm^-1")
        self.metadata.setdefault("chi_unit", "rad")
        self.metadata.setdefault(
            "chi_convention",
            {
                "zero_degrees": "+qx",
                "ninety_degrees": "+qy",
                "positive_direction": "counter-clockwise in the qx/qy plane",
            },
        )
        self.metadata.setdefault(
            "mask_semantics",
            {
                "valid_mask_true": "valid pixel",
                "external_mask_true": "invalid pixel",
            },
        )

    @property
    def q_unit(self) -> str:
        """PONI maps are physical q; the array names are already nm^-1."""

        return str(self.metadata.get("q_unit", self.metadata.get("unit", "nm^-1")) or "nm^-1")

    @property
    def q(self) -> np.ndarray:
        return self.q_nm_inv

    @property
    def chi(self) -> np.ndarray:
        return self.chi_rad

    @property
    def qx(self) -> np.ndarray:
        return self.qx_nm_inv

    @property
    def qy(self) -> np.ndarray:
        return self.qy_nm_inv

    @property
    def mask(self) -> np.ndarray:
        """Detector-style negative mask (``True`` means invalid).

        The canonical positive representation remains :attr:`valid_mask`;
        this alias lets the result pass through the domain ``QMap`` seam,
        whose historical ``mask`` field follows FabIO/pyFAI convention.
        """

        return ~self.valid_mask

    @property
    def valid(self) -> np.ndarray:
        return self.valid_mask

    @property
    def geometry_fingerprint(self) -> str:
        return self.fingerprint

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.q_nm_inv.shape)  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        """Return metadata and fingerprint without duplicating large arrays."""

        return {
            "fingerprint": self.fingerprint,
            "metadata": dict(self.metadata),
            "shape": list(self.shape),
        }


# QMap is the vocabulary used by the domain models and an intuitive public
# alias for callers that do not need to distinguish map construction details.
QMap = GeometryMaps
GeometryMap = GeometryMaps


def _validate_beam_center_xy(beam_center_xy: Any) -> tuple[float, float]:
    if isinstance(beam_center_xy, (str, bytes)):
        raise ValueError("beam_center_xy must be a finite (cx, cy) pair")
    try:
        center = tuple(beam_center_xy)
    except TypeError as exc:
        raise ValueError("beam_center_xy must be a finite (cx, cy) pair") from exc
    if len(center) != 2 or any(isinstance(v, (bool, np.bool_)) for v in center):
        raise ValueError(f"beam_center_xy must be a finite (cx, cy) pair, got {beam_center_xy!r}")
    try:
        cx, cy = float(center[0]), float(center[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"beam_center_xy must contain finite numbers, got {beam_center_xy!r}") from exc
    if not (math.isfinite(cx) and math.isfinite(cy)):
        raise ValueError(f"beam_center_xy must contain finite numbers, got {beam_center_xy!r}")
    return cx, cy


def apply_beamstop_mask(
    valid_mask: np.ndarray | None,
    shape: tuple[int, int] | Sequence[int],
    beam_center_xy: tuple[float, float] | Sequence[float],
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a circular beamstop exclusion mask via 1D broadcasting and combine with ``valid_mask``.

    Returns ``(combined_valid_mask, beamstop_exclusion_mask)`` where
    ``beamstop_exclusion_mask == True`` marks pixels inside ``radius_px`` of
    ``beam_center_xy = (cx, cy)`` as masked out, and ``combined_valid_mask == True``
    marks retained valid pixels.
    """
    height, width = _validate_shape(shape)
    cx, cy = _validate_beam_center_xy(beam_center_xy)

    if isinstance(radius_px, (bool, np.bool_)):
        raise ValueError("radius_px must be a finite non-negative number")
    try:
        radius = float(radius_px)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"radius_px must be a finite non-negative number, got {radius_px!r}") from exc
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(f"radius_px must be a finite non-negative number, got {radius_px!r}")

    exclusion = np.zeros((height, width), dtype=bool)
    if radius > 0.0:
        y0 = max(0, int(math.floor(cy - radius)))
        y1 = min(height, int(math.ceil(cy + radius)) + 1)
        x0 = max(0, int(math.floor(cx - radius)))
        x1 = min(width, int(math.ceil(cx + radius)) + 1)
        if y0 < y1 and x0 < x1:
            dy = np.arange(y0, y1, dtype=np.float64) - cy
            dx = np.arange(x0, x1, dtype=np.float64) - cx
            exclusion[y0:y1, x0:x1] = (dy[:, None] ** 2 + dx[None, :] ** 2) <= (radius * radius)

    if valid_mask is None:
        combined_valid = ~exclusion if radius > 0.0 else np.ones((height, width), dtype=bool)
    else:
        base = np.asarray(valid_mask, dtype=bool)
        if base.shape != (height, width):
            raise GeometryShapeError(
                f"valid_mask shape {base.shape!r} does not match shape {(height, width)!r}"
            )
        combined_valid = base & ~exclusion if radius > 0.0 else base.copy()

    return combined_valid, exclusion


def _infer_center_and_dq_fast(
    qx: np.ndarray,
    qy: np.ndarray,
    q_mag: np.ndarray,
    *,
    beam_center_xy: Sequence[float] | None = None,
    pixel_dq: float | None = None,
) -> tuple[tuple[float, float], float]:
    """Infer subpixel beam center ``(cx, cy)`` and reciprocal pixel step ``pixel_dq`` in <1 ms."""
    height, width = q_mag.shape
    cx_out: float | None = None
    cy_out: float | None = None
    dq_out: float | None = None

    if beam_center_xy is not None:
        cx_out, cy_out = _validate_beam_center_xy(beam_center_xy)
    if pixel_dq is not None:
        if isinstance(pixel_dq, (bool, np.bool_)):
            raise ValueError("pixel_dq must be a finite positive number")
        dq_val = float(pixel_dq)
        if not math.isfinite(dq_val) or dq_val <= 0.0:
            raise ValueError(f"pixel_dq must be a finite positive number, got {pixel_dq!r}")
        dq_out = dq_val

    if cx_out is None or dq_out is None:
        stride_y = max(1, height // 48)
        stride_x = max(1, width // 48)
        qx_sub = qx[::stride_y, ::stride_x]
        qy_sub = qy[::stride_y, ::stride_x]
        q_sub = q_mag[::stride_y, ::stride_x]
        sub_finite = np.isfinite(qx_sub) & np.isfinite(qy_sub) & np.isfinite(q_sub)
        if int(np.count_nonzero(sub_finite)) >= 12:
            ys = np.arange(0, height, stride_y, dtype=np.float64)
            xs = np.arange(0, width, stride_x, dtype=np.float64)
            gx, gy = np.meshgrid(xs, ys)
            xv = gx[sub_finite]
            yv = gy[sub_finite]
            qxv = qx_sub[sub_finite]
            qyv = qy_sub[sub_finite]
            design = np.column_stack((np.ones_like(xv), xv, yv))
            try:
                coef_x, *_ = np.linalg.lstsq(design, qxv, rcond=None)
                coef_y, *_ = np.linalg.lstsq(design, qyv, rcond=None)
                a0, a1, a2 = float(coef_x[0]), float(coef_x[1]), float(coef_x[2])
                b0, b1, b2 = float(coef_y[0]), float(coef_y[1]), float(coef_y[2])
                det = a1 * b2 - a2 * b1
                max_q = float(np.max(q_sub[sub_finite]))
                pred_x = design @ coef_x
                pred_y = design @ coef_y
                rms = float(np.sqrt(np.mean((pred_x - qxv) ** 2 + (pred_y - qyv) ** 2)))
                if abs(det) > 1.0e-18 and max_q > 0.0 and rms <= 0.15 * max_q:
                    cx_fit = (-b2 * a0 + a2 * b0) / det
                    cy_fit = (b1 * a0 - a1 * b0) / det
                    dq_fit = math.sqrt(0.5 * (a1 * a1 + a2 * a2 + b1 * b1 + b2 * b2))
                    if cx_out is None and math.isfinite(cx_fit) and math.isfinite(cy_fit):
                        cx_out, cy_out = cx_fit, cy_fit
                    if dq_out is None and math.isfinite(dq_fit) and dq_fit > 0.0:
                        dq_out = dq_fit
            except np.linalg.LinAlgError:
                pass

    if cx_out is None or cy_out is None:
        finite_q = np.isfinite(q_mag)
        if np.any(finite_q):
            q_fill = np.where(finite_q, q_mag, np.inf)
            min_y, min_x = np.unravel_index(int(np.argmin(q_fill)), (height, width))
            cx_out, cy_out = float(min_x), float(min_y)
        else:
            cx_out = 0.5 * (width - 1)
            cy_out = 0.5 * (height - 1)

    if dq_out is None:
        stride_y = max(1, height // 64)
        stride_x = max(1, width // 64)
        dx_med = (
            float(np.nanmedian(np.abs(np.diff(qx[::stride_y, :], axis=1))))
            if width > 1
            else 0.0
        )
        dy_med = (
            float(np.nanmedian(np.abs(np.diff(qy[:, ::stride_x], axis=0))))
            if height > 1
            else 0.0
        )
        pos_steps = [v for v in (dx_med, dy_med) if math.isfinite(v) and v > 0.0]
        dq_out = min(pos_steps) if pos_steps else 1.0

    return (float(cx_out), float(cy_out)), float(dq_out)


@dataclass
class SAXSGeometryCache:
    """Reusable detector geometry, beamstop mask, and active reciprocal-space ROI cache."""

    geometry: Any
    shape: tuple[int, int]
    beam_center_xy: tuple[float, float]
    pixel_dq: float
    base_valid_mask: np.ndarray
    beamstop_exclusion_mask: np.ndarray
    valid_mask: np.ndarray
    roi_slices: tuple[slice, slice]
    q_mag_roi: np.ndarray
    qx_roi: np.ndarray
    qy_roi: np.ndarray
    chi_deg_roi: np.ndarray
    valid_roi: np.ndarray
    q_coordinates_validated: bool = True
    beamstop_radius_px: float | None = None
    q_roi_max: float | None = None
    q_unit: str = "nm^-1"
    geometry_roi_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.geometry_roi_mask is None:
            self.geometry_roi_mask = np.isfinite(self.q_mag_roi)

    @property
    def roi_shape(self) -> tuple[int, int]:
        return tuple(self.q_mag_roi.shape)  # type: ignore[return-value]

    @property
    def is_full_frame_roi(self) -> bool:
        return self.roi_shape == self.shape

    def slice_frame(self, arr: Any) -> np.ndarray:
        """Fast view-slice of a full-frame 2D array into the cached active ROI."""
        array = np.asarray(arr)
        if array.ndim != 2:
            raise GeometryShapeError(f"expected a 2-D array for slice_frame, got ndim={array.ndim}")
        if array.shape != self.shape:
            raise GeometryShapeError(
                f"array shape {array.shape!r} does not match cached geometry shape {self.shape!r}"
            )
        return array[self.roi_slices]


def build_geometry_cache(
    geometry: GeometryMaps | Mapping[str, Any],
    valid_mask: Any | None = None,
    *,
    beamstop_radius_px: float | None = None,
    q_roi_max: float | None = None,
    beam_center_xy: Sequence[float] | None = None,
    pixel_dq: float | None = None,
) -> SAXSGeometryCache:
    """Build a reusable :class:`SAXSGeometryCache` with beamstop masking and optional q-ROI slicing."""
    if isinstance(geometry, GeometryMaps):
        q_mag = geometry.q_nm_inv
        qx = geometry.qx_nm_inv
        qy = geometry.qy_nm_inv
        chi_deg = (
            geometry.chi_deg
            if geometry.chi_deg is not None
            else (np.degrees(geometry.chi_rad) % 360.0)
        )
        geom_valid = np.asarray(geometry.valid_mask, dtype=bool)
        q_unit = geometry.q_unit
    elif isinstance(geometry, Mapping):
        if "qx" not in geometry or "qy" not in geometry:
            raise GeometryError("geometry mapping must contain 'qx' and 'qy' arrays")
        qx = np.asarray(geometry["qx"], dtype=np.float64)
        qy = np.asarray(geometry["qy"], dtype=np.float64)
        if qx.ndim != 2 or qy.shape != qx.shape or qx.size == 0:
            raise GeometryShapeError("geometry mapping qx/qy must be matching non-empty 2-D arrays")
        q_mag = (
            np.asarray(geometry["q"], dtype=np.float64)
            if "q" in geometry and geometry["q"] is not None
            else (
                np.asarray(geometry["q_nm_inv"], dtype=np.float64)
                if "q_nm_inv" in geometry and geometry["q_nm_inv"] is not None
                else np.hypot(qx, qy)
            )
        )
        if q_mag.shape != qx.shape:
            raise GeometryShapeError("geometry mapping q shape must match qx/qy shape")
        if "chi_deg" in geometry and geometry["chi_deg"] is not None:
            chi_deg = np.mod(np.asarray(geometry["chi_deg"], dtype=np.float64), 360.0)
        elif "chi_rad" in geometry and geometry["chi_rad"] is not None:
            chi_deg = np.mod(np.degrees(np.asarray(geometry["chi_rad"], dtype=np.float64)), 360.0)
        else:
            chi_deg = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
        if chi_deg.shape != qx.shape:
            raise GeometryShapeError("geometry mapping chi shape must match qx/qy shape")
        if "valid_mask" in geometry and geometry["valid_mask"] is not None:
            geom_valid = np.asarray(geometry["valid_mask"], dtype=bool)
        elif "mask" in geometry and geometry["mask"] is not None:
            geom_valid = ~np.asarray(geometry["mask"], dtype=bool)
        else:
            geom_valid = np.isfinite(q_mag) & np.isfinite(qx) & np.isfinite(qy)
        if geom_valid.shape != qx.shape:
            raise GeometryShapeError("geometry mapping valid_mask shape must match qx/qy shape")
        q_unit = str(geometry.get("q_unit", "nm^-1") or "nm^-1")
    else:
        raise GeometryError("geometry must be a GeometryMaps instance or q-map mapping")

    shape = (int(q_mag.shape[0]), int(q_mag.shape[1]))
    finite_geom = np.isfinite(q_mag) & np.isfinite(qx) & np.isfinite(qy)
    if not np.any(finite_geom):
        raise GeometryError("geometry contains no finite reciprocal-space coordinates")

    if valid_mask is None:
        base_valid = (geom_valid & finite_geom).copy()
    else:
        user_valid = np.asarray(valid_mask, dtype=bool)
        if user_valid.shape != shape:
            raise GeometryShapeError(
                f"valid_mask shape {user_valid.shape!r} does not match geometry shape {shape!r}"
            )
        base_valid = geom_valid & finite_geom & user_valid

    center_xy, dq = _infer_center_and_dq_fast(
        qx,
        qy,
        q_mag,
        beam_center_xy=beam_center_xy,
        pixel_dq=pixel_dq,
    )

    bs_radius_val: float | None = None
    if beamstop_radius_px is not None:
        combined_valid, bs_exclusion = apply_beamstop_mask(
            base_valid,
            shape,
            center_xy,
            beamstop_radius_px,
        )
        bs_radius_val = float(beamstop_radius_px)
    else:
        bs_exclusion = np.zeros(shape, dtype=bool)
        combined_valid = base_valid

    roi_q_val: float | None = None
    if q_roi_max is None:
        roi_slices = (slice(0, shape[0]), slice(0, shape[1]))
        q_mag_roi = q_mag
        qx_roi = qx
        qy_roi = qy
        chi_deg_roi = chi_deg
        valid_roi = combined_valid
        geom_roi_mask = finite_geom
    else:
        if isinstance(q_roi_max, (bool, np.bool_)):
            raise GeometryError("q_roi_max must be a finite positive number")
        roi_q_val = float(q_roi_max)
        if not math.isfinite(roi_q_val) or roi_q_val <= 0.0:
            raise GeometryError(f"q_roi_max must be finite and positive, got {q_roi_max!r}")
        roi_sel = finite_geom & (q_mag <= roi_q_val)
        row_any = np.any(roi_sel, axis=1)
        if not np.any(row_any):
            min_q = float(np.min(q_mag[finite_geom]))
            raise GeometryError(
                f"q_roi_max={roi_q_val:g} excludes all detector pixels (minimum finite q is {min_q:g})"
            )
        col_any = np.any(roi_sel, axis=0)
        y0 = int(np.argmax(row_any))
        y1 = int(shape[0] - np.argmax(row_any[::-1]))
        x0 = int(np.argmax(col_any))
        x1 = int(shape[1] - np.argmax(col_any[::-1]))
        roi_slices = (slice(y0, y1), slice(x0, x1))
        q_mag_roi = np.ascontiguousarray(q_mag[roi_slices])
        qx_roi = np.ascontiguousarray(qx[roi_slices])
        qy_roi = np.ascontiguousarray(qy[roi_slices])
        chi_deg_roi = np.ascontiguousarray(chi_deg[roi_slices])
        valid_roi = np.ascontiguousarray(combined_valid[roi_slices])
        geom_roi_mask = np.ascontiguousarray(finite_geom[roi_slices])

    return SAXSGeometryCache(
        geometry=geometry,
        shape=shape,
        beam_center_xy=center_xy,
        pixel_dq=dq,
        base_valid_mask=base_valid,
        beamstop_exclusion_mask=bs_exclusion,
        valid_mask=combined_valid,
        roi_slices=roi_slices,
        q_mag_roi=q_mag_roi,
        qx_roi=qx_roi,
        qy_roi=qy_roi,
        chi_deg_roi=chi_deg_roi,
        valid_roi=valid_roi,
        q_coordinates_validated=True,
        beamstop_radius_px=bs_radius_val,
        q_roi_max=roi_q_val,
        q_unit=q_unit,
        geometry_roi_mask=geom_roi_mask,
    )


def load_poni(poni: str | Path | Any) -> Any:
    """Load a PONI through pyFAI's supported loader.

    An already-created AzimuthalIntegrator is accepted so a UI can reuse a
    live, user-refined geometry without serialising it first.
    """

    if _looks_like_azimuthal_integrator(poni):
        return poni
    source = Path(poni).expanduser()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"PONI file does not exist: {source}")
    try:
        import pyFAI

        # pyFAI.load parses both legacy and current PONI representations and
        # preserves all detector rotations.  Do not parse rot1/rot2/rot3 here.
        return pyFAI.load(str(source))
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise GeometryError("PONI geometry requires pyFAI") from exc
    except Exception as exc:
        raise GeometryError(f"could not load PONI file {source}: {exc}") from exc


def build_geometry(
    shape: Sequence[int] | str | Path,
    poni: str | Path | Any | Sequence[int] | None = None,
    *,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
) -> GeometryMaps:
    """Build q/chi/qx/qy maps for exactly ``shape`` pixels.

    The canonical call is ``build_geometry((rows, cols), poni_path)``.  For
    compatibility with callers that naturally put the PONI first,
    ``build_geometry(poni_path, (rows, cols))`` is also accepted.
    """

    if _looks_like_shape(poni) and not _looks_like_shape(shape):
        shape, poni = poni, shape
    if poni is None:
        raise GeometryError("a PONI path or AzimuthalIntegrator is required")
    shape_tuple = _validate_shape(shape)
    ai = load_poni(poni)

    try:
        q = np.asarray(ai.qArray(shape_tuple), dtype=np.float64)
        # center_array is the current pyFAI precision path.  chiArray is kept
        # as a compatibility fallback for old pyFAI installations only.
        try:
            chi = np.asarray(
                ai.center_array(shape_tuple, unit="chi_rad", scale=True),
                dtype=np.float64,
            )
        except (AttributeError, TypeError):  # pragma: no cover - old pyFAI
            chi = np.asarray(ai.chiArray(shape_tuple), dtype=np.float64)
    except Exception as exc:
        raise GeometryError(
            f"pyFAI could not calculate q/chi maps for shape {shape_tuple!r}: {exc}"
        ) from exc

    if q.shape != shape_tuple or chi.shape != shape_tuple:
        raise GeometryShapeError(
            f"pyFAI returned q shape {q.shape!r} and chi shape {chi.shape!r}; "
            f"requested {shape_tuple!r}"
        )
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(chi)):
        raise GeometryError("PONI produced non-finite q/chi values")

    # pyFAI's chi is an azimuth in radians.  Explicitly normalise its phase so
    # the public convention remains 0 deg=+qx and 90 deg=+qy for every PONI.
    chi = np.mod(chi, 2.0 * np.pi)
    qx = q * np.cos(chi)
    qy = q * np.sin(chi)

    finite_mask = np.isfinite(q) & np.isfinite(chi) & np.isfinite(qx) & np.isfinite(qy)
    if valid_mask is None and external_mask is None and mask is None:
        combined_mask = finite_mask
    else:
        if mask is not None:
            if external_mask is not None:
                raise GeometryError("provide only one of mask and external_mask")
            external_mask = mask
        from .io import combine_masks  # local import avoids an import cycle

        user_mask = combine_masks(
            shape_tuple, valid_mask=valid_mask, external_mask=external_mask
        )
        combined_mask = finite_mask if user_mask is None else finite_mask & user_mask

    metadata = _geometry_metadata(ai, poni, shape_tuple, combined_mask)
    fingerprint = _fingerprint(ai, poni, shape_tuple, metadata)
    metadata["fingerprint"] = fingerprint
    return GeometryMaps(
        q_nm_inv=q,
        chi_rad=chi,
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=combined_mask,
        metadata=metadata,
        fingerprint=fingerprint,
    )


def geometry_from_poni(
    poni: str | Path | Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> GeometryMaps:
    """PONI-first spelling of :func:`build_geometry`."""

    return build_geometry(shape, poni, **kwargs)


def build_q_map(
    poni: str | Path | Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> GeometryMaps:
    """Alias used by q-map consumers."""

    return geometry_from_poni(poni, shape, **kwargs)


def q_map_from_poni(
    poni: str | Path | Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> GeometryMaps:
    """Alias for :func:`build_q_map`."""

    return build_q_map(poni, shape, **kwargs)


def qmap_from_poni(
    poni: str | Path | Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> GeometryMaps:
    """Underscore-free alias used by the pipeline adapter discovery."""

    return build_q_map(poni, shape, **kwargs)


def build_qmap(
    image: Any | None = None,
    *,
    poni: str | Path | Any | None = None,
    shape: Sequence[int] | None = None,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
    **_: Any,
) -> GeometryMaps:
    """Pipeline-facing q-map adapter.

    ``pipeline.build_qmap`` supplies an image and a PONI keyword.  This
    adapter derives only the image shape; every physical coordinate still
    comes from :func:`build_geometry` and pyFAI.
    """

    if shape is None:
        if image is None or not hasattr(image, "shape"):
            raise GeometryShapeError("build_qmap requires shape or a 2-D image")
        shape = tuple(np.asarray(image).shape)
    if poni is None:
        raise GeometryError("build_qmap requires a PONI path or integrator")
    return build_geometry(
        shape,
        poni,
        valid_mask=valid_mask,
        external_mask=external_mask,
        mask=mask,
    )


compute_qmap = build_qmap
make_qmap = build_qmap
pixel_qmap = build_qmap


def _looks_like_azimuthal_integrator(value: Any) -> bool:
    return value is not None and all(
        callable(getattr(value, name, None)) for name in ("qArray", "center_array")
    )


def _looks_like_shape(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, Path)):
        return False
    try:
        items = tuple(value)
    except TypeError:
        return False
    return len(items) == 2 and all(
        not isinstance(item, bool) and isinstance(item, (int, np.integer))
        for item in items
    )


def _validate_shape(shape: Any) -> tuple[int, int]:
    if not _looks_like_shape(shape):
        raise GeometryShapeError(f"shape must be two integer dimensions, got {shape!r}")
    result = (int(shape[0]), int(shape[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise GeometryShapeError(f"shape must be positive, got {result!r}")
    return result


def _geometry_metadata(
    ai: Any,
    poni: Any,
    shape: tuple[int, int],
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "shape": list(shape),
        "q_unit": "nm^-1",
        "chi_unit": "rad",
        "chi_zero_deg": "+qx",
        "chi_ninety_deg": "+qy",
        "chi_convention": {
            "zero_degrees": "+qx",
            "ninety_degrees": "+qy",
            "positive_direction": "counter-clockwise in the qx/qy plane",
        },
        "mask_semantics": {
            "valid_mask_true": "valid pixel",
            "external_mask_true": "invalid pixel",
        },
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "invalid_pixel_count": int(valid_mask.size - np.count_nonzero(valid_mask)),
    }
    try:
        import pyFAI

        metadata["pyfai_version"] = str(getattr(pyFAI, "__version__", getattr(pyFAI, "version", "unknown")))
    except Exception:  # pragma: no cover - pyFAI import was already successful
        metadata["pyfai_version"] = "unknown"
    for name in (
        "dist",
        "poni1",
        "poni2",
        "rot1",
        "rot2",
        "rot3",
        "wavelength",
        "orientation",
    ):
        try:
            value = getattr(ai, name)
        except Exception:
            continue
        metadata[name] = _json_value(value)
    detector = getattr(ai, "detector", None)
    if detector is not None:
        detector_meta: dict[str, Any] = {"class": type(detector).__name__}
        for name in ("name", "pixel1", "pixel2", "max_shape", "shape", "orientation"):
            try:
                detector_meta[name] = _json_value(getattr(detector, name))
            except Exception:
                continue
        metadata["detector"] = detector_meta
    if isinstance(poni, (str, Path)):
        source = Path(poni).expanduser()
        metadata["poni_source"] = str(source)
        try:
            metadata["poni_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError:
            pass
    else:
        metadata["poni_source"] = "in-memory AzimuthalIntegrator"
    return metadata


def _fingerprint(ai: Any, poni: Any, shape: tuple[int, int], metadata: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {
        "shape": list(shape),
        "metadata": _json_value(dict(metadata)),
    }
    # The bytes digest captures user edits to a PONI that do not happen to be
    # exposed as an AI attribute.  For in-memory geometry, metadata is the
    # complete serialisable parameter record available to us.
    if isinstance(poni, (str, Path)):
        try:
            payload["poni_bytes_sha256"] = hashlib.sha256(
                Path(poni).expanduser().read_bytes()
            ).hexdigest()
        except OSError:
            pass
    payload["ai_class"] = type(ai).__name__
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert numpy/scalar metadata to deterministic JSON-compatible values."""

    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "GeometryError",
    "GeometryMap",
    "GeometryMaps",
    "GeometryShapeError",
    "QMap",
    "SAXSGeometryCache",
    "apply_beamstop_mask",
    "build_geometry",
    "build_geometry_cache",
    "build_qmap",
    "build_q_map",
    "compute_qmap",
    "geometry_from_poni",
    "load_poni",
    "make_qmap",
    "pixel_qmap",
    "qmap_from_poni",
    "q_map_from_poni",
]
