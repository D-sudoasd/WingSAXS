"""Shared validation, reciprocal-unit, periodic-angle, and 2D binning helpers for physics."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from .._primitives import finite_float as _finite
from ..settings import canonical_q_unit

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_GAS_CONSTANT_KJ_MOL_K = 8.31446261815324e-3  # CODATA 2018 exact R in kJ/(mol*K)
_KJ_MOL_PER_EV = 96.48533212331002


def _finite_list(values: np.ndarray) -> list[float | None]:
    arr = np.asarray(values, dtype=float)
    return [float(v) if math.isfinite(float(v)) else None for v in arr]


def _q_unit_info(q_unit: Any) -> tuple[str, float | None, str | None]:
    """Return ``(canonical_label, scale_to_nm_inv, caveat_or_none)``.

    ``scale_to_nm_inv`` converts a ``q`` value in ``q_unit`` into ``nm^-1``:
    ``1.0`` for ``nm^-1``, ``10.0`` for ``A^-1``, and ``None`` for uncalibrated
    ``pixel-q`` or ``unknown`` units.
    """
    canonical = canonical_q_unit(q_unit)
    if canonical == "nm⁻¹":
        return "nm⁻¹", 1.0, None
    if canonical == "Å⁻¹":
        return "Å⁻¹", 10.0, None
    if canonical == "pixel-q":
        return (
            "pixel-q",
            None,
            "Uncalibrated pixel-q coordinates cannot be converted to physical nm lengths.",
        )
    label = str(q_unit or "unknown")
    return (
        label,
        None,
        f"Undeclared or non-physical q_unit ({label!r}) cannot be converted to physical nm lengths.",
    )


def _spacing_nm_from_q(q_value: float | None, q_scale_nm_inv: float | None) -> float | None:
    if q_value is None or q_scale_nm_inv is None:
        return None
    if not math.isfinite(q_value) or q_value <= 0.0:
        return None
    q_nm_inv = q_value * q_scale_nm_inv
    return _finite(2.0 * math.pi / q_nm_inv) if q_nm_inv > 0.0 else None


def _periodic_dist_deg(angle_deg: np.ndarray | float, center_deg: float, period_deg: float = 360.0) -> np.ndarray:
    half = 0.5 * period_deg
    return (np.asarray(angle_deg, dtype=float) - float(center_deg) + half) % period_deg - half


def _unwrap_near_deg(angle_deg: float | None, reference_deg: float, period_deg: float) -> float | None:
    if angle_deg is None or not math.isfinite(angle_deg):
        return None
    half = 0.5 * period_deg
    delta = (float(angle_deg) - float(reference_deg) + half) % period_deg - half
    return float(reference_deg + delta)


def unwrap_periodic_angles_deg(
    angles_deg: Sequence[float | None],
    *,
    period_deg: float = 180.0,
) -> list[float | None]:
    """Unwrap a sequence of periodic angles while preserving ``None``/``NaN`` gaps."""
    if not math.isfinite(period_deg) or period_deg <= 0.0:
        raise ValueError("period_deg must be finite and positive")
    unwrapped: list[float | None] = []
    last: float | None = None
    for raw in angles_deg:
        value = _finite(raw)
        if value is None:
            unwrapped.append(None)
            continue
        if last is None:
            current = float(value % period_deg)
        else:
            current = _unwrap_near_deg(value, last, period_deg)
            assert current is not None
        unwrapped.append(current)
        last = current
    return unwrapped


def _normalise_intensity_mode(intensity_mode: str) -> str:
    mode = str(intensity_mode or "calibrated").strip().lower()
    if mode not in {"calibrated", "raw"}:
        raise ValueError(f"intensity_mode must be 'calibrated' or 'raw', got {intensity_mode!r}")
    return mode




def _resolve_2d_context(
    image: Any,
    qx: Any,
    qy: Any,
    valid_mask: Any = None,
    *,
    geometry_cache: Any = None,
    sigma: Any = None,
    required_q_max: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
    float | None,
]:
    """Validate and resolve 2D inputs, reusing ``SAXSGeometryCache`` ROI when valid.

    Returns ``(img, qx, qy, q_mag_or_None, chi_deg_or_None, geom_mask_or_None, inclusion, sigma_or_None, pixel_dq_hint)``.
    When ``geometry_cache`` is ``None`` and ``required_q_max`` is given, ``q_mag``,
    ``chi_deg``, and ``geom_mask`` are left ``None`` so callers can pre-filter by
    the ``|qx|, |qy| <= required_q_max`` bounding box before computing ``hypot``.
    """
    try:
        img = np.asarray(image, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("image must be a numeric two-dimensional array") from exc
    if img.ndim != 2 or img.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")

    sig_arr: np.ndarray | None = None
    if sigma is not None:
        try:
            sig_arr = np.asarray(sigma, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sigma must be a numeric two-dimensional array") from exc
        if sig_arr.ndim != 2 or sig_arr.shape != img.shape:
            raise ValueError(f"sigma shape {sig_arr.shape!r} must match image shape {img.shape!r}")

    if geometry_cache is not None:
        cache_shape = tuple(geometry_cache.shape)
        roi_shape = tuple(geometry_cache.q_mag_roi.shape)
        cache_q_roi_max = getattr(geometry_cache, "q_roi_max", None)
        img_is_roi = (img.shape == roi_shape) and (roi_shape != cache_shape)
        can_use_roi = (
            img_is_roi
            or cache_q_roi_max is None
            or (required_q_max is not None and float(required_q_max) <= float(cache_q_roi_max) + 1.0e-12)
        )
        if can_use_roi:
            if img.shape == cache_shape and roi_shape != cache_shape:
                img = geometry_cache.slice_frame(img)
                if sig_arr is not None:
                    sig_arr = geometry_cache.slice_frame(sig_arr)
            elif img.shape != roi_shape:
                raise ValueError(
                    f"image shape {img.shape!r} does not match geometry_cache shape {cache_shape!r} "
                    f"or ROI shape {roi_shape!r}"
                )
            qx_arr = geometry_cache.qx_roi
            qy_arr = geometry_cache.qy_roi
            q_mag = geometry_cache.q_mag_roi
            chi_deg = geometry_cache.chi_deg_roi
            geom_mask = (
                geometry_cache.geometry_roi_mask
                if getattr(geometry_cache, "geometry_roi_mask", None) is not None
                else np.isfinite(q_mag)
            )
            cache_valid = geometry_cache.valid_roi
            if valid_mask is None:
                inclusion = cache_valid
            else:
                raw_vm = np.asarray(valid_mask, dtype=bool)
                if raw_vm.shape == cache_shape and roi_shape != cache_shape:
                    raw_vm = geometry_cache.slice_frame(raw_vm)
                elif raw_vm.shape != roi_shape:
                    raise ValueError(
                        f"valid_mask shape {raw_vm.shape!r} does not match geometry_cache shape"
                    )
                inclusion = cache_valid & raw_vm
            return (
                img,
                qx_arr,
                qy_arr,
                q_mag,
                chi_deg,
                geom_mask,
                inclusion,
                sig_arr,
                getattr(geometry_cache, "pixel_dq", None),
            )

        # Fallback to full geometry inside cache when requested q_window exceeds cached q_roi_max
        if img.shape != cache_shape:
            raise ValueError(
                f"image shape {img.shape!r} does not match geometry_cache full shape {cache_shape!r}"
            )
        geom_obj = geometry_cache.geometry
        if hasattr(geom_obj, "qx_nm_inv"):
            qx_arr = geom_obj.qx_nm_inv
            qy_arr = geom_obj.qy_nm_inv
            q_mag = geom_obj.q_nm_inv
            chi_deg = (
                geom_obj.chi_deg
                if geom_obj.chi_deg is not None
                else np.mod(np.degrees(geom_obj.chi_rad), 360.0)
            )
        else:
            qx_arr = np.asarray(geom_obj["qx"], dtype=float)
            qy_arr = np.asarray(geom_obj["qy"], dtype=float)
            q_mag = (
                np.asarray(geom_obj["q"], dtype=float)
                if "q" in geom_obj and geom_obj["q"] is not None
                else np.hypot(qx_arr, qy_arr)
            )
            chi_deg = (
                np.asarray(geom_obj["chi_deg"], dtype=float)
                if "chi_deg" in geom_obj and geom_obj["chi_deg"] is not None
                else None
            )
        geom_mask = np.isfinite(q_mag)
        cache_valid = geometry_cache.valid_mask
        if valid_mask is None:
            inclusion = cache_valid
        else:
            raw_vm = np.asarray(valid_mask, dtype=bool)
            if raw_vm.shape != cache_shape:
                raise ValueError(
                    f"valid_mask shape {raw_vm.shape!r} does not match image shape {cache_shape!r}"
                )
            inclusion = cache_valid & raw_vm
        return (
            img,
            qx_arr,
            qy_arr,
            q_mag,
            chi_deg,
            geom_mask,
            inclusion,
            sig_arr,
            getattr(geometry_cache, "pixel_dq", None),
        )

    if qx is None or qy is None:
        raise ValueError("qx and qy must be supplied when image is provided without geometry_cache")
    shape = img.shape
    try:
        qx_arr = np.asarray(qx, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("qx must be a numeric two-dimensional array") from exc
    if qx_arr.ndim != 2 or qx_arr.shape != shape:
        raise ValueError(f"qx shape {qx_arr.shape!r} must match image shape {shape!r}")
    try:
        qy_arr = np.asarray(qy, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("qy must be a numeric two-dimensional array") from exc
    if qy_arr.ndim != 2 or qy_arr.shape != shape:
        raise ValueError(f"qy shape {qy_arr.shape!r} must match image shape {shape!r}")

    if valid_mask is None:
        inclusion = np.ones(shape, dtype=bool)
    else:
        raw_mask = np.asarray(valid_mask)
        if raw_mask.shape != shape:
            raise ValueError(f"valid_mask shape {raw_mask.shape!r} must match image shape {shape!r}")
        inclusion = raw_mask.astype(bool, copy=False)

    if required_q_max is not None:
        return img, qx_arr, qy_arr, None, None, None, inclusion, sig_arr, None

    q_mag = np.hypot(qx_arr, qy_arr)
    geom_mask = np.isfinite(q_mag)
    if not np.any(geom_mask):
        raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
    return img, qx_arr, qy_arr, q_mag, None, geom_mask, inclusion, sig_arr, None


def _validate_q_window(q_window: Any) -> tuple[float, float]:
    try:
        lo, hi = q_window
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError) as exc:
        raise ValueError("q_window must be a (q_min, q_max) pair of finite numbers") from exc
    if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
        raise ValueError("q_window bounds must be finite")
    if lo_f < 0.0:
        raise ValueError("q_window lower bound must be non-negative")
    if lo_f >= hi_f:
        raise ValueError("q_window must satisfy q_min < q_max")
    return lo_f, hi_f


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    valid = np.asarray(mask, dtype=bool)
    if valid.size == 0 or not np.any(valid):
        return []
    starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
    ends = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def _bin_azimuthal_from_2d(
    img: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    geometry: np.ndarray | None,
    usable: np.ndarray | None,
    *,
    q_window: tuple[float, float] | None,
    n_bins: int,
    q_mag: np.ndarray | None = None,
    chi_deg: np.ndarray | None = None,
    inclusion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | None], tuple[float | None, float | None]]:
    """Bin observed 2D pixels into ``n_bins`` uniform bins over ``[0, 360)`` with q-band pre-filtering."""
    step = 360.0 / float(n_bins)
    inv_step = float(n_bins) / 360.0
    centers = (np.arange(n_bins, dtype=float) + 0.5) * step
    empty_quads: dict[str, float | None] = {"QI": None, "QII": None, "QIII": None, "QIV": None}

    if q_window is not None:
        q_min, q_max = _validate_q_window(q_window)
        q_bounds: tuple[float | None, float | None] = (q_min, q_max)
        if q_mag is None:
            # Fast bounding-box pre-filter skips hypot & arctan2 on ~99% of detector pixels
            in_box = (np.abs(qx) <= q_max) & (np.abs(qy) <= q_max)
            if not np.any(in_box):
                if not np.any(np.isfinite(qx) & np.isfinite(qy)):
                    raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
                return (
                    centers,
                    np.full(n_bins, np.nan, dtype=float),
                    np.zeros(n_bins, dtype=np.int64),
                    np.zeros(n_bins, dtype=np.int64),
                    empty_quads,
                    q_bounds,
                )
            bx = qx[in_box]
            by = qy[in_box]
            bq = np.hypot(bx, by)
            in_band = np.isfinite(bq) & (bq >= q_min) & (bq <= q_max) & (bq > 0.0)
            if not np.any(in_band):
                return (
                    centers,
                    np.full(n_bins, np.nan, dtype=float),
                    np.zeros(n_bins, dtype=np.int64),
                    np.zeros(n_bins, dtype=np.int64),
                    empty_quads,
                    q_bounds,
                )
            cand_x = bx[in_band]
            cand_y = by[in_band]
            cand_img = img[in_box][in_band]
            if usable is not None:
                obs_sel = usable[in_box][in_band]
            elif inclusion is not None:
                obs_sel = inclusion[in_box][in_band] & np.isfinite(cand_img)
            else:
                obs_sel = np.isfinite(cand_img)
            cand_angles = np.mod(np.degrees(np.arctan2(cand_y, cand_x)), 360.0)
        else:
            geom_m = geometry if geometry is not None else np.isfinite(q_mag)
            cand_mask = geom_m & (q_mag >= q_min) & (q_mag <= q_max) & (q_mag > 0.0)
            if not np.any(cand_mask):
                return (
                    centers,
                    np.full(n_bins, np.nan, dtype=float),
                    np.zeros(n_bins, dtype=np.int64),
                    np.zeros(n_bins, dtype=np.int64),
                    empty_quads,
                    q_bounds,
                )
            cand_x = qx[cand_mask]
            cand_y = qy[cand_mask]
            cand_img = img[cand_mask]
            if usable is not None:
                obs_sel = usable[cand_mask]
            elif inclusion is not None:
                obs_sel = inclusion[cand_mask] & np.isfinite(cand_img)
            else:
                obs_sel = np.isfinite(cand_img)
            if chi_deg is not None:
                cand_angles = np.mod(chi_deg[cand_mask], 360.0)
            else:
                cand_angles = np.mod(np.degrees(np.arctan2(cand_y, cand_x)), 360.0)
    else:
        if q_mag is None:
            q_mag = np.hypot(qx, qy)
        geom_m = geometry if geometry is not None else np.isfinite(q_mag)
        if not np.any(geom_m):
            raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
        cand_mask = geom_m & (q_mag > 0.0)
        if not np.any(cand_mask):
            return (
                centers,
                np.full(n_bins, np.nan, dtype=float),
                np.zeros(n_bins, dtype=np.int64),
                np.zeros(n_bins, dtype=np.int64),
                empty_quads,
                (None, None),
            )
        cand_q = q_mag[cand_mask]
        cand_x = qx[cand_mask]
        cand_y = qy[cand_mask]
        cand_img = img[cand_mask]
        if usable is not None:
            obs_sel = usable[cand_mask]
        elif inclusion is not None:
            obs_sel = inclusion[cand_mask] & np.isfinite(cand_img)
        else:
            obs_sel = np.isfinite(cand_img)
        obs_q = cand_q[obs_sel]
        q_bounds = (_finite(np.min(obs_q)), _finite(np.max(obs_q))) if obs_q.size else (None, None)
        if chi_deg is not None:
            cand_angles = np.mod(chi_deg[cand_mask], 360.0)
        else:
            cand_angles = np.mod(np.degrees(np.arctan2(cand_y, cand_x)), 360.0)

    cand_idx = np.floor(cand_angles * inv_step).astype(np.int64)
    np.clip(cand_idx, 0, n_bins - 1, out=cand_idx)
    obs_idx = cand_idx[obs_sel]
    obs_vals = cand_img[obs_sel]

    geom_counts = np.bincount(cand_idx, minlength=n_bins).astype(np.int64)
    counts = np.bincount(obs_idx, minlength=n_bins).astype(np.int64)
    sums = np.bincount(obs_idx, weights=obs_vals, minlength=n_bins).astype(float)
    intensity = np.full(n_bins, np.nan, dtype=float)
    np.divide(sums, counts, out=intensity, where=counts > 0)

    # Quadrant observed means directly from observed pixels in the q band (via O(N) bincount)
    q_obs_x = cand_x[obs_sel]
    q_obs_y = cand_y[obs_sel]
    quad_code = (q_obs_x < 0.0).astype(np.int8) | ((q_obs_y < 0.0).astype(np.int8) << 1)
    q_cnt = np.bincount(quad_code, minlength=4)
    q_sum = np.bincount(quad_code, weights=obs_vals, minlength=4) if obs_vals.size else np.zeros(4, dtype=float)
    quadrant_means: dict[str, float | None] = {}
    for label, code in (("QI", 0), ("QII", 1), ("QIII", 3), ("QIV", 2)):
        quadrant_means[label] = _finite(q_sum[code] / q_cnt[code]) if q_cnt[code] > 0 else None

    return centers, intensity, counts, geom_counts, quadrant_means, q_bounds
