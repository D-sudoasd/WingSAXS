"""Quantitative 2D SAXS butterfly physics observables, models, and kinetics helpers.

This module is strictly Qt-free and operates exclusively on observed, valid
(``valid_mask=True``) detector pixels. Masked detector gaps, beamstop shadows,
and non-finite pixels are never filled, interpolated, or mirrored across
quadrants.

Scientific invariants enforced here:
- Ring/sector radial peak positions ``q*`` and ``L* = 2*pi/q*`` are kept
  strictly separate from ellipse parameters (``a``, ``b/a``, ``theta_deg``,
  conditional ``Ln``/``Lz``).
- Physical lengths in ``nm`` require a declared calibrated reciprocal-space
  unit (``nm^-1`` or ``A^-1``, converted transparently to ``nm^-1``).
  Uncalibrated ``pixel-q`` or ``unknown`` units always yield ``None`` for nm
  lengths with an explicit caveat.
- Window-bounded scattering invariant ``Q_win = integral(q^2 I(q) dq)`` never
  fabricates scattering-length-density contrast ``delta_rho`` or volume
  fraction.
- Apparent lateral coherence length ``W = 2*pi/dq_perp(0)`` is explicitly
  tagged as a model-dependent Ruland/Grubb/Williamson-Hall streak-width
  estimate, never a unique 3D reconstruction.
- All returned summary dictionaries are strict-JSON safe (no NaN or Inf;
  unavailable values are ``None``).
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import t as student_t

from .serialization import json_safe
from .settings import canonical_q_unit

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_GAS_CONSTANT_KJ_MOL_K = 8.31446261815324e-3  # CODATA 2018 exact R in kJ/(mol*K)
_KJ_MOL_PER_EV = 96.48533212331002


# ---------------------------------------------------------------------------
# Shared validation, unit, angle, and strict-JSON helpers
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


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


def _validate_2d_inputs(
    image: Any,
    qx: Any,
    qy: Any,
    valid_mask: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate 2D arrays and return ``(img, qx, qy, geometry_mask, usable_mask)``."""
    try:
        img = np.asarray(image, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("image must be a numeric two-dimensional array") from exc
    if img.ndim != 2 or img.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    shape = img.shape

    def _check_2d(val: Any, name: str) -> np.ndarray:
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a numeric two-dimensional array") from exc
        if arr.ndim != 2 or arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape!r} must match image shape {shape!r}")
        return arr

    qx_arr = _check_2d(qx, "qx")
    qy_arr = _check_2d(qy, "qy")
    if valid_mask is None:
        inclusion = np.ones(shape, dtype=bool)
    else:
        raw_mask = np.asarray(valid_mask)
        if raw_mask.shape != shape:
            raise ValueError(f"valid_mask shape {raw_mask.shape!r} must match image shape {shape!r}")
        inclusion = raw_mask.astype(bool, copy=False)

    q_mag = np.hypot(qx_arr, qy_arr)
    geometry = np.isfinite(q_mag)
    if not np.any(geometry):
        raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
    usable = geometry & inclusion & np.isfinite(img)
    return img, qx_arr, qy_arr, geometry, usable


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


# ---------------------------------------------------------------------------
# 1. Azimuthal Harmonics & Order Parameters
# ---------------------------------------------------------------------------


def _hermans_order(
    angle_deg: np.ndarray,
    intensity: np.ndarray,
    axis_deg: float,
) -> tuple[float | None, float | None]:
    """Compute Hermans f2 (P2) and 4-fold Legendre f4 (P4) relative to ``axis_deg``."""
    w_int = np.maximum(intensity, 0.0)
    beta = np.deg2rad(angle_deg - float(axis_deg))
    weight = w_int * np.abs(np.sin(beta))
    denom = float(np.sum(weight))
    if not math.isfinite(denom) or denom <= 0.0:
        return None, None
    cos2 = np.cos(beta) ** 2
    mean_cos2 = float(np.sum(weight * cos2) / denom)
    mean_cos4 = float(np.sum(weight * (cos2 * cos2)) / denom)
    f2 = 0.5 * (3.0 * mean_cos2 - 1.0)
    f4 = 0.125 * (35.0 * mean_cos4 - 30.0 * mean_cos2 + 3.0)
    return _finite(f2), _finite(f4)


def _friedel_quadrant_ratios(
    angle_deg: np.ndarray,
    intensity: np.ndarray,
    observed: np.ndarray,
) -> tuple[float | None, float | None, int, int]:
    """Compute opposite-quadrant Friedel symmetry ratios from matched observed pairs."""
    n = angle_deg.size
    if n < 4 or not np.any(observed):
        return None, None, 0, 0

    angles = np.mod(angle_deg, 360.0)
    order = np.argsort(angles)
    angles = angles[order]
    intensity = intensity[order]
    observed = observed[order]
    # Detect uniform even-bin grid for exact O(n) opposite-bin pairing
    diffs = np.diff(angles)
    is_uniform_even = (
        n % 2 == 0
        and n >= 4
        and np.all(np.isfinite(diffs))
        and np.allclose(diffs, 360.0 / n, rtol=1e-3, atol=1e-3)
    )
    if is_uniform_even:
        half = n // 2
        opp = (np.arange(n) + half) % n
        both = observed & observed[opp]
        q1_sel = both & (angles >= 0.0) & (angles < 90.0)
        q2_sel = both & (angles >= 90.0) & (angles < 180.0)
        n13 = int(np.count_nonzero(q1_sel))
        n24 = int(np.count_nonzero(q2_sel))
        r13 = None
        if n13 > 0:
            s1 = float(np.sum(intensity[q1_sel]))
            s3 = float(np.sum(intensity[opp[q1_sel]]))
            if s3 > 0.0 and s1 >= 0.0:
                r13 = _finite(s1 / s3)
        r24 = None
        if n24 > 0:
            s2 = float(np.sum(intensity[q2_sel]))
            s4 = float(np.sum(intensity[opp[q2_sel]]))
            if s4 > 0.0 and s2 >= 0.0:
                r24 = _finite(s2 / s4)
        return r13, r24, n13, n24

    # General fallback for irregular or odd-sized angular grids: match within half a bin step
    obs_idx = np.flatnonzero(observed)
    obs_ang = angles[obs_idx]
    obs_val = intensity[obs_idx]
    sorted_ang = np.sort(obs_ang)
    tol = 0.55 * float(np.median(np.diff(sorted_ang))) if sorted_ang.size > 1 else 2.5
    tol = min(max(tol, 0.5), 10.0)

    def _pair_ratio(lo_deg: float, hi_deg: float) -> tuple[float | None, int]:
        sel_a = np.flatnonzero((obs_ang >= lo_deg) & (obs_ang < hi_deg))
        sel_b = np.flatnonzero((obs_ang >= lo_deg + 180.0) & (obs_ang < hi_deg + 180.0))
        if sel_a.size == 0 or sel_b.size == 0:
            return None, 0
        used_b: set[int] = set()
        sum_a = 0.0
        sum_b = 0.0
        matched = 0
        for ia in sel_a:
            target = (obs_ang[ia] + 180.0) % 360.0
            dists = np.abs(_periodic_dist_deg(obs_ang[sel_b], target, 360.0))
            best_local = int(np.argmin(dists))
            ib = int(sel_b[best_local])
            if dists[best_local] <= tol and ib not in used_b:
                used_b.add(ib)
                sum_a += float(obs_val[ia])
                sum_b += float(obs_val[ib])
                matched += 1
        if matched == 0 or sum_b <= 0.0 or sum_a < 0.0:
            return None, matched
        return _finite(sum_a / sum_b), matched

    r13, n13 = _pair_ratio(0.0, 90.0)
    r24, n24 = _pair_ratio(90.0, 180.0)
    return r13, r24, n13, n24


def _detect_folded_lobe_opening(
    angles_obs: np.ndarray,
    intensity_obs: np.ndarray,
    axis_deg: float,
) -> tuple[float | None, float]:
    """Fold observed intensity around ``axis_deg`` after detrending ``c0 + c2*cos(2*d)``.

    Returns ``(alpha_est_deg_or_None, interior_contrast)`` where ``alpha_est_deg``
    lies in ``(6, 84)`` if a genuine interior 4-lobe peak exists.
    """
    if angles_obs.size < 8:
        return None, 0.0
    delta_rad = np.deg2rad(angles_obs - float(axis_deg))
    X = np.column_stack((np.ones_like(delta_rad), np.cos(2.0 * delta_rad)))
    if np.linalg.cond(X) > 1e6:
        return None, 0.0
    beta, *_ = np.linalg.lstsq(X, intensity_obs, rcond=None)
    resid = intensity_obs - X @ beta

    folded = np.abs(_periodic_dist_deg(angles_obs, axis_deg, 180.0))  # [0, 90]
    n_fold = 30  # 3-degree folded bins
    edges = np.linspace(0.0, 90.0, n_fold + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.searchsorted(edges, folded, side="right") - 1, 0, n_fold - 1)
    counts = np.bincount(idx, minlength=n_fold)
    sums = np.bincount(idx, weights=resid, minlength=n_fold)
    prof = np.full(n_fold, np.nan, dtype=float)
    np.divide(sums, counts, out=prof, where=counts > 0)

    valid = np.isfinite(prof)
    if np.count_nonzero(valid) < 5:
        return None, 0.0

    # Find strongest observed bin in interior (6..84 deg)
    interior = valid & (centers >= 6.0) & (centers <= 84.0)
    if not np.any(interior):
        return None, 0.0

    int_indices = np.flatnonzero(interior)
    peak_i = int(int_indices[np.argmax(prof[int_indices])])
    peak_val = float(prof[peak_i])
    val_range = float(np.nanmax(prof[valid]) - np.nanmin(prof[valid]))
    if val_range <= 1e-12:
        return None, 0.0

    # Check that it is an interior peak rather than a boundary slope
    edge_vals = [float(prof[j]) for j in (0, 1, n_fold - 2, n_fold - 1) if valid[j]]
    max_edge = max(edge_vals) if edge_vals else -np.inf
    if peak_val <= max_edge + 0.05 * val_range:
        return None, 0.0

    # Parabolic refinement when immediate neighbors are observed
    alpha_est = float(centers[peak_i])
    if 0 < peak_i < n_fold - 1 and valid[peak_i - 1] and valid[peak_i + 1]:
        y_m, y_0, y_p = float(prof[peak_i - 1]), float(prof[peak_i]), float(prof[peak_i + 1])
        denom = y_m - 2.0 * y_0 + y_p
        if denom < -1e-15:
            shift = 0.5 * (y_m - y_p) / denom
            if abs(shift) <= 1.0:
                alpha_est += shift * (90.0 / n_fold)
    return float(np.clip(alpha_est, 3.0, 87.0)), float(peak_val / max(val_range, 1e-15))


def _fit_even_harmonics_ols(
    angle_deg_obs: np.ndarray,
    intensity_obs: np.ndarray,
) -> tuple[dict[str, float | None], list[int], list[str]]:
    """Estimate even Fourier harmonics (m = 0, 2, 4, 6) strictly on observed bins via OLS."""
    flags: list[str] = []
    n_obs = angle_deg_obs.size
    if n_obs == 0:
        return {}, [], ["no_observed_angular_bins"]

    rad = np.deg2rad(angle_deg_obs)
    chosen_orders: list[int] = []
    beta: np.ndarray | None = None

    for candidate in ((2, 4, 6), (2, 4), (2,), ()):
        n_params = 1 + 2 * len(candidate)
        if n_obs < n_params:
            continue
        cols = [np.ones(n_obs, dtype=float)]
        for m in candidate:
            cols.append(np.cos(m * rad))
            cols.append(np.sin(m * rad))
        X = np.column_stack(cols)
        cond = float(np.linalg.cond(X))
        if math.isfinite(cond) and cond < 1.0e4:
            beta, *_ = np.linalg.lstsq(X, intensity_obs, rcond=None)
            chosen_orders = list(candidate)
            if len(candidate) < 3:
                flags.append("reduced_harmonic_order_due_to_mask_coverage")
            break
    if beta is None:
        flags.append("harmonic_design_ill_conditioned")
        return {"A0": _finite(np.mean(intensity_obs))}, [], flags

    out: dict[str, float | None] = {"A0": _finite(beta[0])}
    for idx, m in enumerate(chosen_orders):
        cm = float(beta[1 + 2 * idx])
        sm = float(beta[2 + 2 * idx])
        out[f"c{m}"] = _finite(cm)
        out[f"s{m}"] = _finite(sm)
        out[f"A{m}"] = _finite(math.hypot(cm, sm))
    return out, chosen_orders, flags


def compute_azimuthal_harmonics_and_order(
    image: Any = None,
    qx: Any = None,
    qy: Any = None,
    *,
    valid_mask: Any = None,
    q_window: tuple[float, float] | None = None,
    q_unit: str = "unknown",
    n_bins: int = 360,
    nominal_draw_axis_deg: float = 90.0,
    bright_axis_convention: str = "draw",
    expected_bright_axis_deg: float | None = None,
    angle_deg: Any = None,
    intensity: Any = None,
    counts: Any = None,
    reference_phase_deg: float | None = None,
    geometry_cache: Any = None,
    intensity_mode: str = "calibrated",
    sigma: Any = None,
) -> dict[str, Any]:
    """Compute even Fourier harmonics, Friedel symmetry bisector, and Hermans order parameters.

    Accepts either 2D arrays ``(image, qx, qy)`` (with optional ``valid_mask``,
    ``q_window``, and ``geometry_cache``) OR a pre-binned 1D angular profile
    ``(angle_deg, intensity)``. All quantities are derived strictly from
    observed pixels/bins; masked gaps are never filled or mirrored.

    ``bright_axis_convention`` controls the expected axis for ``orthogonal_rotation_detected``
    and ``mounting_tilt_deg`` when ``expected_bright_axis_deg`` is omitted:
    - ``"draw"`` (default): expects the bright C2 bridge / acute lobe bisector along
      ``nominal_draw_axis_deg``.
    - ``"reference"``: expects the bright C2 bridge / acute lobe bisector along
      ``(nominal_draw_axis_deg - 90) % 180`` (WingSAXS ``reference_axis_deg``, matching
      equatorial-bridged butterflies such as NO50S with vertical wire ``draw_axis_deg=90``).
    """
    _normalise_intensity_mode(intensity_mode)
    q_unit_label, _, _ = _q_unit_info(q_unit)
    nominal_axis = float(nominal_draw_axis_deg) % 180.0
    nominal_reference_axis = float((nominal_axis - 90.0) % 180.0)
    convention = str(bright_axis_convention or "draw").strip().lower()
    if convention not in {"draw", "reference"}:
        raise ValueError("bright_axis_convention must be 'draw' or 'reference'")
    if expected_bright_axis_deg is not None:
        exp_val = float(expected_bright_axis_deg)
        if not math.isfinite(exp_val):
            raise ValueError("expected_bright_axis_deg must be finite when provided")
        expected_axis = float(exp_val % 180.0)
    else:
        expected_axis = nominal_axis if convention == "draw" else nominal_reference_axis
    flags: list[str] = []
    quadrant_means: dict[str, float | None] = {"QI": None, "QII": None, "QIII": None, "QIV": None}
    q_bounds: tuple[float | None, float | None] = (None, None)

    if image is not None:
        if geometry_cache is None and (qx is None or qy is None):
            raise ValueError("qx and qy must be supplied when image is provided")
        if isinstance(n_bins, (bool, np.bool_)) or int(n_bins) < 8:
            raise ValueError("n_bins must be an integer >= 8")
        req_q_max = _validate_q_window(q_window)[1] if q_window is not None else None
        img, qx_arr, qy_arr, q_mag, chi_deg, geom_mask, inclusion, _, _ = _resolve_2d_context(
            image,
            qx,
            qy,
            valid_mask,
            geometry_cache=geometry_cache,
            sigma=sigma,
            required_q_max=req_q_max,
        )
        ang, prof, cnts, geom_cnts, quadrant_means, q_bounds = _bin_azimuthal_from_2d(
            img,
            qx_arr,
            qy_arr,
            geom_mask,
            None,
            q_window=q_window,
            n_bins=int(n_bins),
            q_mag=q_mag,
            chi_deg=chi_deg,
            inclusion=inclusion,
        )
        observed = np.isfinite(ang) & np.isfinite(prof) & (cnts > 0)
        geom_total = int(np.count_nonzero(geom_cnts > 0))
        obs_total = int(np.count_nonzero(observed))
        angular_coverage = float(obs_total / geom_total) if geom_total > 0 else 0.0
    elif angle_deg is not None and intensity is not None:
        ang = np.asarray(angle_deg, dtype=float)
        prof = np.asarray(intensity, dtype=float)
        if ang.ndim != 1 or prof.ndim != 1 or ang.shape != prof.shape or ang.size == 0:
            raise ValueError("angle_deg and intensity must be non-empty matching 1D arrays")
        observed = np.isfinite(ang) & np.isfinite(prof)
        if counts is not None:
            cnts = np.asarray(counts)
            if cnts.shape != ang.shape:
                raise ValueError("counts must match angle_deg shape")
            observed &= cnts > 0
        else:
            cnts = observed.astype(np.int64)
        if valid_mask is not None:
            vm = np.asarray(valid_mask, dtype=bool)
            if vm.shape != ang.shape:
                raise ValueError("valid_mask must match angle_deg shape")
            observed &= vm
        angular_coverage = float(np.count_nonzero(observed) / ang.size)
        ang_mod = np.mod(ang, 360.0)
        for label, lo, hi in (("QI", 0.0, 90.0), ("QII", 90.0, 180.0), ("QIII", 180.0, 270.0), ("QIV", 270.0, 360.0)):
            sel = observed & (ang_mod >= lo) & (ang_mod < hi)
            quadrant_means[label] = _finite(np.mean(prof[sel])) if np.any(sel) else None
    else:
        raise ValueError("supply either (image, qx, qy) or (angle_deg, intensity)")

    if angular_coverage < 1.0:
        flags.append("incomplete_azimuthal_coverage")

    n_obs = int(np.count_nonzero(observed))
    if n_obs == 0:
        flags.append("no_observed_angular_bins")
        return json_safe({
            "status": "unavailable",
            "q_unit": q_unit_label,
            "q_window": list(q_bounds) if q_bounds[0] is not None else None,
            "n_observed_bins": 0,
            "n_total_bins": int(ang.size),
            "angular_coverage": 0.0,
            "nominal_draw_axis_deg": nominal_axis,
            "nominal_reference_axis_deg": nominal_reference_axis,
            "bright_axis_convention": convention,
            "expected_bright_axis_deg": expected_axis,
            "A0": None,
            "A2": None,
            "A4": None,
            "A6": None,
            "a2": None,
            "a4": None,
            "a6": None,
            "a4_over_a2": None,
            "phi2_deg": None,
            "phi4_deg": None,
            "phi6_deg": None,
            "phi2_unwrapped_draw_deg": None,
            "phi4_unwrapped_draw_deg": None,
            "friedel_bisector_deg": None,
            "orthogonal_axis_deg": None,
            "bright_bridge_axis_deg": None,
            "dark_notch_axis_deg": None,
            "angle_to_draw_axis_deg": None,
            "angle_to_reference_axis_deg": None,
            "mounting_tilt_deg": None,
            "orthogonal_rotation_detected": False,
            "wing_half_opening_estimate_deg": None,
            "f2": None,
            "f4": None,
            "f2_nominal": None,
            "f4_nominal": None,
            "q1_over_q3": None,
            "q2_over_q4": None,
            "friedel_matched_bins_q1_q3": 0,
            "friedel_matched_bins_q2_q4": 0,
            "quadrant_means": quadrant_means,
            "flags": list(dict.fromkeys(flags)),
        })

    ang_obs = np.mod(ang[observed], 360.0)
    y_obs = prof[observed]

    harm, orders, harm_flags = _fit_even_harmonics_ols(ang_obs, y_obs)
    flags.extend(harm_flags)

    A0 = harm.get("A0")
    A2 = harm.get("A2")
    A4 = harm.get("A4")
    A6 = harm.get("A6")
    c2, s2 = harm.get("c2"), harm.get("s2")
    c4, s4 = harm.get("c4"), harm.get("s4")
    c6, s6 = harm.get("c6"), harm.get("s6")

    scale = max(abs(A0 or 0.0), float(np.std(y_obs)), 1.0e-15)
    amp_tol = 1.0e-10 * scale

    a2 = _finite(A2 / A0) if (A0 is not None and A0 > 0.0 and A2 is not None) else None
    a4 = _finite(A4 / A0) if (A0 is not None and A0 > 0.0 and A4 is not None) else None
    a6 = _finite(A6 / A0) if (A0 is not None and A0 > 0.0 and A6 is not None) else None
    a4_over_a2 = _finite(A4 / A2) if (A2 is not None and A4 is not None and A2 > amp_tol) else None

    phi2_raw = (
        float((0.5 * math.degrees(math.atan2(s2, c2))) % 180.0)
        if (c2 is not None and s2 is not None and A2 is not None and A2 > amp_tol)
        else None
    )
    phi4_raw = (
        float((0.25 * math.degrees(math.atan2(s4, c4))) % 90.0)
        if (c4 is not None and s4 is not None and A4 is not None and A4 > amp_tol)
        else None
    )
    phi6_raw = (
        float(((1.0 / 6.0) * math.degrees(math.atan2(s6, c6))) % 60.0)
        if (c6 is not None and s6 is not None and A6 is not None and A6 > amp_tol)
        else None
    )

    if reference_phase_deg is not None and math.isfinite(float(reference_phase_deg)):
        ref_deg = float(reference_phase_deg)
        phi2_deg = _unwrap_near_deg(phi2_raw, ref_deg, 180.0)
        phi4_deg = _unwrap_near_deg(phi4_raw, ref_deg, 90.0)
        phi6_deg = _unwrap_near_deg(phi6_raw, ref_deg, 60.0)
    else:
        phi2_deg = phi2_raw
        phi4_deg = phi4_raw
        phi6_deg = phi6_raw

    phi2_unwrapped_draw = _unwrap_near_deg(phi2_raw, nominal_axis, 180.0)
    phi4_unwrapped_draw = _unwrap_near_deg(phi4_raw, nominal_axis, 90.0)

    # Mirror-symmetry bisector from even Fourier harmonics
    aniso_amp = math.sqrt(sum((harm.get(f"A{m}") or 0.0) ** 2 for m in orders))
    friedel_bisector_deg: float | None = None
    orthogonal_axis_deg: float | None = None
    wing_half_opening_est: float | None = None

    if orders and aniso_amp > 1.0e-8 * scale:
        harm_terms = [
            (m, float(harm[f"c{m}"]), float(harm[f"s{m}"]))
            for m in orders
            if harm.get(f"c{m}") is not None and harm.get(f"s{m}") is not None
        ]

        def _mirror_loss(psi_rad: float) -> float:
            total = 0.0
            for m, cm, sm in harm_terms:
                val = -cm * math.sin(m * psi_rad) + sm * math.cos(m * psi_rad)
                total += val * val
            return total

        grid_rad = np.linspace(0.0, 0.5 * math.pi, 360, endpoint=False)
        losses = np.zeros_like(grid_rad)
        for m, cm, sm in harm_terms:
            val = -cm * np.sin(m * grid_rad) + sm * np.cos(m * grid_rad)
            losses += val * val
        best_idx = int(np.argmin(losses))
        step_rad = 0.5 * math.pi / 360.0
        lo_rad = float(grid_rad[best_idx]) - 1.5 * step_rad
        hi_rad = float(grid_rad[best_idx]) + 1.5 * step_rad
        opt = minimize_scalar(_mirror_loss, bounds=(lo_rad, hi_rad), method="bounded")
        psi_a_deg = float(math.degrees(float(opt.x)) % 180.0)
        psi_b_deg = float((psi_a_deg + 90.0) % 180.0)

        # Disambiguate between the two orthogonal mirror axes (psi_a vs psi_b)
        alpha_a, contrast_a = _detect_folded_lobe_opening(ang_obs, y_obs, psi_a_deg)
        if alpha_a is not None and contrast_a > 0.12 and (A4 or 0.0) > 0.02 * aniso_amp:
            if alpha_a < 44.0:
                friedel_bisector_deg = psi_a_deg
                wing_half_opening_est = alpha_a
            elif alpha_a > 46.0:
                friedel_bisector_deg = psi_b_deg
                wing_half_opening_est = 90.0 - alpha_a
            else:
                # Near-square 45 deg butterfly: tie-break by proximity to expected bright axis
                da = abs(float(_periodic_dist_deg(psi_a_deg, expected_axis, 180.0)))
                db = abs(float(_periodic_dist_deg(psi_b_deg, expected_axis, 180.0)))
                friedel_bisector_deg = psi_a_deg if da <= db else psi_b_deg
                wing_half_opening_est = alpha_a if da <= db else (90.0 - alpha_a)
        elif c2 is not None and s2 is not None and A2 is not None and A2 > amp_tol:
            # 2-lobe / nematic pattern: pick the mirror axis where C2 is maximal
            proj_a = c2 * math.cos(2.0 * math.radians(psi_a_deg)) + s2 * math.sin(2.0 * math.radians(psi_a_deg))
            friedel_bisector_deg = psi_a_deg if proj_a >= 0.0 else psi_b_deg
        else:
            da = abs(float(_periodic_dist_deg(psi_a_deg, expected_axis, 180.0)))
            db = abs(float(_periodic_dist_deg(psi_b_deg, expected_axis, 180.0)))
            friedel_bisector_deg = psi_a_deg if da <= db else psi_b_deg

        orthogonal_axis_deg = float((friedel_bisector_deg + 90.0) % 180.0)
    else:
        flags.append("isotropic_symmetry_axis_undefined")

    angle_to_draw_axis_deg: float | None = None
    angle_to_reference_axis_deg: float | None = None
    mounting_tilt_deg: float | None = None
    bright_bridge_axis_deg: float | None = None
    dark_notch_axis_deg: float | None = None
    orthogonal_rotation_detected = False
    if friedel_bisector_deg is not None:
        angle_to_draw_axis_deg = float(_periodic_dist_deg(friedel_bisector_deg, nominal_axis, 180.0))
        angle_to_reference_axis_deg = float(_periodic_dist_deg(friedel_bisector_deg, nominal_reference_axis, 180.0))
        mounting_tilt_deg = float(_periodic_dist_deg(friedel_bisector_deg, expected_axis, 180.0))
        if abs(mounting_tilt_deg) >= 65.0:
            orthogonal_rotation_detected = True
            flags.append("orthogonal_symmetry_axis_rotation")
        bright_bridge_axis_deg = phi2_raw if phi2_raw is not None else friedel_bisector_deg
        dark_notch_axis_deg = float((bright_bridge_axis_deg + 90.0) % 180.0)

    f2_nominal, f4_nominal = _hermans_order(ang_obs, y_obs, nominal_axis)
    if friedel_bisector_deg is not None:
        f2_sym, f4_sym = _hermans_order(ang_obs, y_obs, friedel_bisector_deg)
    else:
        # For an isotropic distribution P2 and P4 are axis-independent (~0)
        f2_sym, f4_sym = f2_nominal, f4_nominal

    q1_over_q3, q2_over_q4, n13, n24 = _friedel_quadrant_ratios(ang, prof, observed)
    if n13 == 0 or n24 == 0:
        flags.append("incomplete_friedel_opposite_quadrant_pairs")

    return json_safe({
        "status": "estimated",
        "q_unit": q_unit_label,
        "q_window": list(q_bounds) if q_bounds[0] is not None else None,
        "n_observed_bins": n_obs,
        "n_total_bins": int(ang.size),
        "angular_coverage": _finite(angular_coverage),
        "nominal_draw_axis_deg": nominal_axis,
        "nominal_reference_axis_deg": nominal_reference_axis,
        "bright_axis_convention": convention,
        "expected_bright_axis_deg": expected_axis,
        "A0": A0,
        "A2": A2,
        "A4": A4,
        "A6": A6,
        "a2": a2,
        "a4": a4,
        "a6": a6,
        "a4_over_a2": a4_over_a2,
        "phi2_deg": _finite(phi2_deg),
        "phi4_deg": _finite(phi4_deg),
        "phi6_deg": _finite(phi6_deg),
        "phi2_unwrapped_draw_deg": _finite(phi2_unwrapped_draw),
        "phi4_unwrapped_draw_deg": _finite(phi4_unwrapped_draw),
        "friedel_bisector_deg": _finite(friedel_bisector_deg),
        "orthogonal_axis_deg": _finite(orthogonal_axis_deg),
        "bright_bridge_axis_deg": _finite(bright_bridge_axis_deg),
        "dark_notch_axis_deg": _finite(dark_notch_axis_deg),
        "angle_to_draw_axis_deg": _finite(angle_to_draw_axis_deg),
        "angle_to_reference_axis_deg": _finite(angle_to_reference_axis_deg),
        "mounting_tilt_deg": _finite(mounting_tilt_deg),
        "orthogonal_rotation_detected": orthogonal_rotation_detected,
        "wing_half_opening_estimate_deg": _finite(wing_half_opening_est),
        "f2": f2_sym,
        "f4": f4_sym,
        "f2_nominal": f2_nominal,
        "f4_nominal": f4_nominal,
        "q1_over_q3": q1_over_q3,
        "q2_over_q4": q2_over_q4,
        "friedel_matched_bins_q1_q3": n13,
        "friedel_matched_bins_q2_q4": n24,
        "quadrant_means": quadrant_means,
        "flags": list(dict.fromkeys(flags)),
        "definitions": {
            "harmonics": "Truncated even Fourier OLS on observed angular bins only: A0 + sum_{m=2,4,6} A_m cos(m*(chi - phi_m))",
            "hermans_f2_f4": "Uniaxial fiber-symmetry Legendre moments <P2(cos(beta))> and <P4(cos(beta))> weighted by max(I, 0)*|sin(beta)| over observed bins",
            "friedel_ratios": "Sum(I_obs(Q1))/Sum(I_obs(Q3)) and Sum(I_obs(Q2))/Sum(I_obs(Q4)) over matched opposite observed angular bins only",
        },
    })


# ---------------------------------------------------------------------------
# 2. C2-Bridge + 4-Lobe Azimuthal Model
# ---------------------------------------------------------------------------


def _butterfly_lobe_centers_deg(chi_sym_deg: float, alpha_deg: float) -> tuple[float, float, float, float]:
    return (
        float((chi_sym_deg - alpha_deg) % 360.0),
        float((chi_sym_deg + alpha_deg) % 360.0),
        float((chi_sym_deg + 180.0 - alpha_deg) % 360.0),
        float((chi_sym_deg + 180.0 + alpha_deg) % 360.0),
    )


def evaluate_c2_bridge_four_lobe(
    angle_deg: Any,
    *,
    c0: float,
    c2: float,
    chi_sym_deg: float,
    alpha_deg: float,
    lobe_amplitudes: Sequence[float],
    lobe_sigmas_deg: Sequence[float] | float,
) -> np.ndarray:
    """Evaluate the C2-bridge + 4-lobe periodic butterfly azimuthal model."""
    angles = np.asarray(angle_deg, dtype=float)
    amps = [float(a) for a in lobe_amplitudes]
    if len(amps) != 4:
        raise ValueError("lobe_amplitudes must contain exactly 4 values")
    if isinstance(lobe_sigmas_deg, (int, float, np.floating)):
        sigmas = [float(lobe_sigmas_deg)] * 4
    else:
        sigmas = [float(s) for s in lobe_sigmas_deg]
        if len(sigmas) != 4:
            raise ValueError("lobe_sigmas_deg must be a scalar or 4-element sequence")

    centers = _butterfly_lobe_centers_deg(chi_sym_deg, alpha_deg)
    result = float(c0) + float(c2) * np.cos(2.0 * np.deg2rad(angles - float(chi_sym_deg)))
    for amp, mu, sigma in zip(amps, centers, sigmas):
        sig = max(sigma, 1.0e-6)
        dist = _periodic_dist_deg(angles, mu, 360.0)
        result = result + amp * np.exp(-0.5 * (dist / sig) ** 2)
    return result


def _canonicalize_butterfly_params(
    chi_sym_deg: float,
    alpha_deg: float,
    c2: float,
    amps: list[float],
    sigmas: list[float],
    *,
    prefer_acute_opening: bool,
    nominal_draw_axis_deg: float,
) -> tuple[float, float, float, list[float], list[float]]:
    """Map butterfly parameters to a unique canonical representation without altering I(chi)."""
    old_centers = _butterfly_lobe_centers_deg(chi_sym_deg, alpha_deg)
    new_chi = float(chi_sym_deg % 180.0)
    new_alpha = float(alpha_deg)
    new_c2 = float(c2)

    if prefer_acute_opening and new_alpha > 45.0:
        new_chi = float((new_chi + 90.0) % 180.0)
        new_alpha = float(90.0 - new_alpha)
        new_c2 = -new_c2
    elif prefer_acute_opening and abs(new_alpha - 45.0) <= 0.5:
        alt_chi = float((new_chi + 90.0) % 180.0)
        d_cur = abs(float(_periodic_dist_deg(new_chi, nominal_draw_axis_deg, 180.0)))
        d_alt = abs(float(_periodic_dist_deg(alt_chi, nominal_draw_axis_deg, 180.0)))
        if d_alt < d_cur:
            new_chi = alt_chi
            new_alpha = float(90.0 - new_alpha)
            new_c2 = -new_c2

    new_centers = _butterfly_lobe_centers_deg(new_chi, new_alpha)
    # Match each new geometric lobe center to its exact original center
    new_amps: list[float] = []
    new_sigmas: list[float] = []
    for nc in new_centers:
        dists = [abs(float(_periodic_dist_deg(nc, oc, 360.0))) for oc in old_centers]
        best = int(np.argmin(dists))
        new_amps.append(float(amps[best]))
        new_sigmas.append(float(sigmas[best]))
    return new_chi, new_alpha, new_c2, new_amps, new_sigmas


def _aic_bic(sse: float, n_obs: int, n_params: int) -> tuple[float | None, float | None]:
    if n_obs <= n_params or not math.isfinite(sse):
        return None, None
    mse = max(sse / float(n_obs), 1.0e-30)
    aic = n_obs * math.log(mse) + 2.0 * n_params
    bic = n_obs * math.log(mse) + n_params * math.log(float(n_obs))
    return _finite(aic), _finite(bic)


def fit_c2_bridge_four_lobe(
    angle_deg: Any,
    intensity: Any,
    *,
    valid_mask: Any = None,
    counts: Any = None,
    nominal_draw_axis_deg: float = 90.0,
    initial_chi_sym_deg: float | None = None,
    initial_alpha_deg: float | None = None,
    initial_fwhm_deg: float = 30.0,
    fwhm_bounds_deg: tuple[float, float] = (2.0, 120.0),
    alpha_bounds_deg: tuple[float, float] = (3.0, 87.0),
    shared_width: bool = True,
    prefer_acute_opening: bool = True,
    loss: str = "linear",
    max_nfev: int = 5000,
    include_flat_comparison: bool = True,
) -> dict[str, Any]:
    """Fit a Friedel-symmetric C2-bridge + 4-lobe periodic model to observed angular bins.

    Model:
        ``I(chi) = c0 + c2 * cos(2*(chi - chi_sym)) + sum_{k=1..4} A_k * exp(-d(chi, mu_k)^2 / (2*sigma_k^2))``
    with ``mu = (chi_sym - alpha, chi_sym + alpha, chi_sym + 180 - alpha, chi_sym + 180 + alpha)``.

    Also fits the nested flat-baseline (``c2 = 0``) 4-lobe model on the identical
    observed bins for direct RMSE, R^2, AIC/BIC, and FWHM-bias comparison.
    """
    x = np.asarray(angle_deg, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 8:
        raise ValueError("angle_deg and intensity must be matching 1D arrays with at least 8 bins")

    fwhm_lo, fwhm_hi = float(fwhm_bounds_deg[0]), float(fwhm_bounds_deg[1])
    if not (math.isfinite(fwhm_lo) and math.isfinite(fwhm_hi) and 0.0 < fwhm_lo < fwhm_hi <= 360.0):
        raise ValueError("fwhm_bounds_deg must satisfy 0 < low < high <= 360")
    alpha_lo, alpha_hi = float(alpha_bounds_deg[0]), float(alpha_bounds_deg[1])
    if not (math.isfinite(alpha_lo) and math.isfinite(alpha_hi) and 0.0 < alpha_lo < alpha_hi < 90.0):
        raise ValueError("alpha_bounds_deg must satisfy 0 < low < high < 90")
    if loss not in {"linear", "soft_l1", "huber", "cauchy"}:
        raise ValueError("loss must be one of 'linear', 'soft_l1', 'huber', 'cauchy'")

    observed = np.isfinite(x) & np.isfinite(y)
    if counts is not None:
        cnt = np.asarray(counts)
        if cnt.shape != x.shape:
            raise ValueError("counts must match angle_deg shape")
        observed &= cnt > 0
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != x.shape:
            raise ValueError("valid_mask must match angle_deg shape")
        observed &= vm

    n_obs = int(np.count_nonzero(observed))
    n_params_c2 = 9 if shared_width else 12
    flags: list[str] = []
    if n_obs < x.size:
        flags.append("partial_angular_masking")

    if n_obs < n_params_c2 + 2:
        flags.append("insufficient_observed_bins_for_c2_four_lobe_fit")
        return json_safe({
            "success": False,
            "status": "unavailable",
            "message": f"Need at least {n_params_c2 + 2} observed angular bins; found {n_obs}.",
            "n_observed_bins": n_obs,
            "n_total_bins": int(x.size),
            "angular_coverage": _finite(n_obs / float(x.size)),
            "c0": None,
            "c2": None,
            "chi_sym_deg": None,
            "alpha_deg": None,
            "lobe_centers_deg": None,
            "lobe_amplitudes": None,
            "lobe_sigmas_deg": None,
            "lobe_fwhms_deg": None,
            "lobe_fwhm_deg": None,
            "lobe_supported": [False, False, False, False],
            "rmse": None,
            "r_squared": None,
            "aic": None,
            "bic": None,
            "angle_deg": _finite_list(x),
            "observed": _finite_list(np.where(observed, y, np.nan)),
            "model": [None] * int(x.size),
            "bridge_component": [None] * int(x.size),
            "lobes_component": [None] * int(x.size),
            "residual": [None] * int(x.size),
            "flat_baseline_comparison": None,
            "flags": list(dict.fromkeys(flags)),
        })

    xo = np.mod(x[observed], 360.0)
    yo = y[observed]
    y_range = max(float(np.max(yo) - np.min(yo)), 1.0e-12)
    y_abs_max = max(float(np.max(np.abs(yo))), 1.0)
    sigma_lo = fwhm_lo / _FWHM_PER_SIGMA
    sigma_hi = fwhm_hi / _FWHM_PER_SIGMA
    sigma_init = float(np.clip(float(initial_fwhm_deg) / _FWHM_PER_SIGMA, sigma_lo * 1.01, sigma_hi * 0.99))
    amp_hi = max(10.0 * y_range, 10.0 * y_abs_max, 1.0)
    c2_bound = max(5.0 * y_range, 5.0 * y_abs_max, 1.0)

    # Seed candidate symmetry axes: skip full harmonics recompute when caller already seeded chi_sym
    seed_axes: list[float] = []
    if initial_chi_sym_deg is not None and math.isfinite(float(initial_chi_sym_deg)):
        ax0 = float(initial_chi_sym_deg) % 180.0
        seed_axes.append(ax0)
        seed_axes.append((ax0 + 90.0) % 180.0)
    else:
        harm_ols, harm_orders, _ = _fit_even_harmonics_ols(xo, yo)
        bisector_seed: float | None = None
        orth_seed: float | None = None
        if harm_orders:
            harm_terms = [
                (m, float(harm_ols[f"c{m}"]), float(harm_ols[f"s{m}"]))
                for m in harm_orders
                if harm_ols.get(f"c{m}") is not None and harm_ols.get(f"s{m}") is not None
            ]
            if harm_terms:
                grid_rad = np.linspace(0.0, 0.5 * math.pi, 180, endpoint=False)
                losses = np.zeros_like(grid_rad)
                for m, cm, sm in harm_terms:
                    val = -cm * np.sin(m * grid_rad) + sm * np.cos(m * grid_rad)
                    losses += val * val
                best_deg = float(math.degrees(float(grid_rad[int(np.argmin(losses))])) % 180.0)
                bisector_seed = best_deg
                orth_seed = (best_deg + 90.0) % 180.0
        for candidate_axis in (
            bisector_seed,
            orth_seed,
            nominal_draw_axis_deg % 180.0,
            (nominal_draw_axis_deg + 90.0) % 180.0,
        ):
            if candidate_axis is not None and math.isfinite(float(candidate_axis)):
                ax_mod = float(candidate_axis) % 180.0
                if not any(abs(float(_periodic_dist_deg(ax_mod, s, 180.0))) < 5.0 for s in seed_axes):
                    seed_axes.append(ax_mod)

    _lobe_signs = np.array([-1.0, 1.0, -1.0, 1.0], dtype=float)
    _lobe_offsets = np.array([0.0, 0.0, 180.0, 180.0], dtype=float)
    _two_rad = 2.0 * (math.pi / 180.0)

    def _unpack(params: np.ndarray, include_c2: bool) -> tuple[float, float, float, float, list[float], list[float]]:
        if include_c2:
            c0_v, c2_v, chi_v, alpha_v = float(params[0]), float(params[1]), float(params[2]), float(params[3])
            amps_v = [float(v) for v in params[4:8]]
            sigs_v = [float(params[8])] * 4 if shared_width else [float(v) for v in params[8:12]]
        else:
            c0_v, c2_v, chi_v, alpha_v = float(params[0]), 0.0, float(params[1]), float(params[2])
            amps_v = [float(v) for v in params[3:7]]
            sigs_v = [float(params[7])] * 4 if shared_width else [float(v) for v in params[7:11]]
        return c0_v, c2_v, chi_v, alpha_v, amps_v, sigs_v

    def _fit_one_seed(axis_seed: float, alpha_seed: float, include_c2: bool):
        # Project out c0 + c2*cos(2*(chi - axis_seed)) on observed bins for clean initial baseline
        rad = np.deg2rad(xo - axis_seed)
        if include_c2:
            X_bg = np.column_stack((np.ones_like(rad), np.cos(2.0 * rad)))
            if np.linalg.cond(X_bg) < 1.0e5:
                bg_beta, *_ = np.linalg.lstsq(X_bg, yo, rcond=None)
                c0_0 = float(bg_beta[0])
                c2_0 = float(np.clip(bg_beta[1], -0.95 * c2_bound, 0.95 * c2_bound))
            else:
                c0_0 = float(np.percentile(yo, 20.0))
                c2_0 = 0.0
            resid0 = yo - (c0_0 + c2_0 * np.cos(2.0 * rad))
        else:
            c0_0 = float(np.percentile(yo, 20.0))
            c2_0 = 0.0
            resid0 = yo - c0_0

        centers0 = _butterfly_lobe_centers_deg(axis_seed, alpha_seed)
        amps0: list[float] = []
        win_deg = max(12.0, 0.8 * initial_fwhm_deg)
        for mu in centers0:
            near = np.abs(_periodic_dist_deg(xo, mu, 360.0)) <= win_deg
            est = float(np.max(resid0[near])) if np.any(near) else 0.25 * y_range
            amps0.append(float(np.clip(max(est, 0.05 * y_range), 1.0e-6, 0.95 * amp_hi)))

        sig_vec = [sigma_init] if shared_width else [sigma_init] * 4
        chi_lo, chi_hi = axis_seed - 45.0, axis_seed + 45.0

        if include_c2:
            p0 = np.array([c0_0, c2_0, axis_seed, alpha_seed, *amps0, *sig_vec], dtype=float)
            lo = np.array([-np.inf, -c2_bound, chi_lo, alpha_lo, 0.0, 0.0, 0.0, 0.0, *([sigma_lo] * len(sig_vec))], dtype=float)
            hi = np.array([np.inf, c2_bound, chi_hi, alpha_hi, amp_hi, amp_hi, amp_hi, amp_hi, *([sigma_hi] * len(sig_vec))], dtype=float)
        else:
            p0 = np.array([c0_0, axis_seed, alpha_seed, *amps0, *sig_vec], dtype=float)
            lo = np.array([-np.inf, chi_lo, alpha_lo, 0.0, 0.0, 0.0, 0.0, *([sigma_lo] * len(sig_vec))], dtype=float)
            hi = np.array([np.inf, chi_hi, alpha_hi, amp_hi, amp_hi, amp_hi, amp_hi, *([sigma_hi] * len(sig_vec))], dtype=float)

        eps_lo = np.where(np.isfinite(lo), 1.0e-9, 0.0)
        eps_hi = np.where(np.isfinite(hi), 1.0e-9, 0.0)
        p0 = np.minimum(np.maximum(p0, lo + eps_lo), hi - eps_hi)

        if include_c2:
            def _res(params: np.ndarray) -> np.ndarray:
                c0_v = params[0]
                c2_v = params[1]
                chi_v = params[2]
                alpha_v = params[3]
                amps_arr = params[4:8]
                mu = chi_v + _lobe_offsets + _lobe_signs * alpha_v
                d = ((xo[None, :] - mu[:, None] + 180.0) % 360.0) - 180.0
                if shared_width:
                    inv_s = 1.0 / max(float(params[8]), 1.0e-6)
                    u = d * inv_s
                else:
                    inv_s = 1.0 / np.maximum(params[8:12], 1.0e-6)
                    u = d * inv_s[:, None]
                g = np.exp(-0.5 * (u * u))
                return c0_v + c2_v * np.cos(_two_rad * (xo - chi_v)) + (amps_arr @ g) - yo

            def _jac(params: np.ndarray) -> np.ndarray:
                c2_v = params[1]
                chi_v = params[2]
                alpha_v = params[3]
                amps_arr = params[4:8]
                mu = chi_v + _lobe_offsets + _lobe_signs * alpha_v
                d = ((xo[None, :] - mu[:, None] + 180.0) % 360.0) - 180.0
                if shared_width:
                    inv_s = 1.0 / max(float(params[8]), 1.0e-6)
                    u = d * inv_s
                    g = np.exp(-0.5 * (u * u))
                    L = amps_arr[:, None] * g
                    dL_dmu = L * (u * inv_s)
                    dL_dsig = L * ((u * u) * inv_s)
                else:
                    inv_s = 1.0 / np.maximum(params[8:12], 1.0e-6)
                    u = d * inv_s[:, None]
                    g = np.exp(-0.5 * (u * u))
                    L = amps_arr[:, None] * g
                    dL_dmu = L * (u * inv_s[:, None])
                    dL_dsig = L * ((u * u) * inv_s[:, None])
                phase = _two_rad * (xo - chi_v)
                J = np.empty((n_obs, params.size), dtype=float)
                J[:, 0] = 1.0
                J[:, 1] = np.cos(phase)
                J[:, 2] = _two_rad * c2_v * np.sin(phase) + np.sum(dL_dmu, axis=0)
                J[:, 3] = _lobe_signs @ dL_dmu
                J[:, 4:8] = g.T
                if shared_width:
                    J[:, 8] = np.sum(dL_dsig, axis=0)
                else:
                    J[:, 8:12] = dL_dsig.T
                return J
        else:
            def _res(params: np.ndarray) -> np.ndarray:
                c0_v = params[0]
                chi_v = params[1]
                alpha_v = params[2]
                amps_arr = params[3:7]
                mu = chi_v + _lobe_offsets + _lobe_signs * alpha_v
                d = ((xo[None, :] - mu[:, None] + 180.0) % 360.0) - 180.0
                if shared_width:
                    inv_s = 1.0 / max(float(params[7]), 1.0e-6)
                    u = d * inv_s
                else:
                    inv_s = 1.0 / np.maximum(params[7:11], 1.0e-6)
                    u = d * inv_s[:, None]
                g = np.exp(-0.5 * (u * u))
                return c0_v + (amps_arr @ g) - yo

            def _jac(params: np.ndarray) -> np.ndarray:
                chi_v = params[1]
                alpha_v = params[2]
                amps_arr = params[3:7]
                mu = chi_v + _lobe_offsets + _lobe_signs * alpha_v
                d = ((xo[None, :] - mu[:, None] + 180.0) % 360.0) - 180.0
                if shared_width:
                    inv_s = 1.0 / max(float(params[7]), 1.0e-6)
                    u = d * inv_s
                    g = np.exp(-0.5 * (u * u))
                    L = amps_arr[:, None] * g
                    dL_dmu = L * (u * inv_s)
                    dL_dsig = L * ((u * u) * inv_s)
                else:
                    inv_s = 1.0 / np.maximum(params[7:11], 1.0e-6)
                    u = d * inv_s[:, None]
                    g = np.exp(-0.5 * (u * u))
                    L = amps_arr[:, None] * g
                    dL_dmu = L * (u * inv_s[:, None])
                    dL_dsig = L * ((u * u) * inv_s[:, None])
                J = np.empty((n_obs, params.size), dtype=float)
                J[:, 0] = 1.0
                J[:, 1] = np.sum(dL_dmu, axis=0)
                J[:, 2] = _lobe_signs @ dL_dmu
                J[:, 3:7] = g.T
                if shared_width:
                    J[:, 7] = np.sum(dL_dsig, axis=0)
                else:
                    J[:, 7:11] = dL_dsig.T
                return J

        return least_squares(
            _res,
            p0,
            jac=_jac,
            bounds=(lo, hi),
            loss=loss,
            f_scale=max(0.05 * y_range, 1.0e-12),
            max_nfev=max_nfev,
        )

    best_c2_fit = None
    best_c2_sse = float("inf")
    initial_centers_used: tuple[float, ...] = ()

    for ax_idx, ax_seed in enumerate(seed_axes[:4]):
        alpha_candidates: list[float] = []
        if initial_alpha_deg is not None and math.isfinite(float(initial_alpha_deg)):
            a_init = float(initial_alpha_deg) if ax_idx == 0 else (90.0 - float(initial_alpha_deg))
            alpha_candidates.append(float(np.clip(a_init, alpha_lo + 1.0, alpha_hi - 1.0)))
        alpha_est, _ = _detect_folded_lobe_opening(xo, yo, ax_seed)
        if alpha_est is not None:
            a_clipped = float(np.clip(alpha_est, alpha_lo + 1.0, alpha_hi - 1.0))
            if not any(abs(a_clipped - existing) < 6.0 for existing in alpha_candidates):
                alpha_candidates.append(a_clipped)
        if not alpha_candidates:
            alpha_candidates = [float(np.clip(35.0, alpha_lo + 1.0, alpha_hi - 1.0))]
        for a_seed in alpha_candidates[:2]:
            try:
                res = _fit_one_seed(ax_seed, a_seed, include_c2=True)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            sse = float(np.sum(res.fun ** 2))
            if math.isfinite(sse) and sse < best_c2_sse:
                best_c2_sse = sse
                best_c2_fit = res
                initial_centers_used = _butterfly_lobe_centers_deg(ax_seed, a_seed)
        # Early exit when an effectively exact noiseless fit is already found
        if best_c2_sse <= 1.0e-10 * (y_range ** 2) * n_obs:
            break

    if best_c2_fit is None:
        flags.append("c2_four_lobe_fit_failed")
        return json_safe({
            "success": False,
            "status": "failed",
            "message": "C2-bridge + 4-lobe optimizer failed on all initial seeds.",
            "n_observed_bins": n_obs,
            "n_total_bins": int(x.size),
            "angular_coverage": _finite(n_obs / float(x.size)),
            "c0": None,
            "c2": None,
            "chi_sym_deg": None,
            "alpha_deg": None,
            "lobe_centers_deg": None,
            "lobe_amplitudes": None,
            "lobe_sigmas_deg": None,
            "lobe_fwhms_deg": None,
            "lobe_fwhm_deg": None,
            "lobe_supported": [False, False, False, False],
            "rmse": None,
            "r_squared": None,
            "aic": None,
            "bic": None,
            "angle_deg": _finite_list(x),
            "observed": _finite_list(np.where(observed, y, np.nan)),
            "model": [None] * int(x.size),
            "bridge_component": [None] * int(x.size),
            "lobes_component": [None] * int(x.size),
            "residual": [None] * int(x.size),
            "flat_baseline_comparison": None,
            "flags": list(dict.fromkeys(flags)),
        })

    c0_raw, c2_raw, chi_raw, alpha_raw, amps_raw, sigs_raw = _unpack(best_c2_fit.x, include_c2=True)
    chi_sym, alpha_deg, c2_val, amps, sigmas = _canonicalize_butterfly_params(
        chi_raw,
        alpha_raw,
        c2_raw,
        amps_raw,
        sigs_raw,
        prefer_acute_opening=prefer_acute_opening and (alpha_lo <= 45.0 <= alpha_hi),
        nominal_draw_axis_deg=nominal_draw_axis_deg,
    )
    centers = _butterfly_lobe_centers_deg(chi_sym, alpha_deg)
    fwhms = [float(s * _FWHM_PER_SIGMA) for s in sigmas]

    # Check observational support around each lobe
    lobe_supported: list[bool] = []
    for idx_k, (mu, sig) in enumerate(zip(centers, sigmas), start=1):
        near_bins = int(np.count_nonzero(np.abs(_periodic_dist_deg(xo, mu, 360.0)) <= max(12.0, 1.5 * sig)))
        supported_k = near_bins >= 3
        lobe_supported.append(supported_k)
        if not supported_k:
            flags.append(f"lobe_{idx_k}_weak_or_masked_support")

    supported_fwhms = [fw for fw, sup in zip(fwhms, lobe_supported) if sup]
    lobe_fwhm_summary = float(np.median(supported_fwhms)) if supported_fwhms else float(np.median(fwhms))

    # Bound diagnostics
    bound_tol = 1.0e-4
    if abs(alpha_deg - alpha_lo) <= bound_tol or abs(alpha_deg - alpha_hi) <= bound_tol:
        flags.append("alpha_at_bound")
    if any(abs(fw - fwhm_lo) <= bound_tol * max(fwhm_lo, 1.0) or abs(fw - fwhm_hi) <= bound_tol * max(fwhm_hi, 1.0) for fw in fwhms):
        flags.append("lobe_width_at_bound")

    # Evaluate components and residuals strictly on observed bins
    model_all = evaluate_c2_bridge_four_lobe(
        x,
        c0=c0_raw,
        c2=c2_val,
        chi_sym_deg=chi_sym,
        alpha_deg=alpha_deg,
        lobe_amplitudes=amps,
        lobe_sigmas_deg=sigmas,
    )
    bridge_all = c0_raw + c2_val * np.cos(2.0 * np.deg2rad(x - chi_sym))
    lobes_all = model_all - bridge_all
    residual_all = np.full(x.shape, np.nan, dtype=float)
    residual_all[observed] = y[observed] - model_all[observed]

    err_obs = residual_all[observed]
    sse_c2 = float(np.sum(err_obs ** 2))
    rmse_c2 = float(math.sqrt(sse_c2 / float(n_obs)))
    sst = float(np.sum((yo - float(np.mean(yo))) ** 2))
    r2_c2 = float(1.0 - sse_c2 / sst) if sst > 1.0e-30 else None
    aic_c2, bic_c2 = _aic_bic(sse_c2, n_obs, n_params_c2)

    success = bool(best_c2_fit.success and np.all(np.isfinite(best_c2_fit.x)) and np.all(np.isfinite(model_all)))
    if not success:
        flags.append("optimizer_did_not_converge")

    # Nested flat-baseline (c2 = 0) 4-lobe fit for direct scientific comparison
    n_params_flat = n_params_c2 - 1
    best_flat_fit = None
    best_flat_sse = float("inf")
    if include_flat_comparison:
        flat_seeds = [(chi_sym, alpha_deg), ((chi_sym + 90.0) % 180.0, max(alpha_lo + 1.0, min(alpha_hi - 1.0, 90.0 - alpha_deg)))]
        for ax_s, a_s in flat_seeds:
            try:
                fres = _fit_one_seed(ax_s, float(np.clip(a_s, alpha_lo + 0.5, alpha_hi - 0.5)), include_c2=False)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            fsse = float(np.sum(fres.fun ** 2))
            if math.isfinite(fsse) and fsse < best_flat_sse:
                best_flat_sse = fsse
                best_flat_fit = fres

    flat_comparison: dict[str, Any] | None = None
    if best_flat_fit is not None:
        fc0, _, fchi, falpha, famps, fsigs = _unpack(best_flat_fit.x, include_c2=False)
        fchi, falpha, _, famps, fsigs = _canonicalize_butterfly_params(
            fchi,
            falpha,
            0.0,
            famps,
            fsigs,
            prefer_acute_opening=prefer_acute_opening and (alpha_lo <= 45.0 <= alpha_hi),
            nominal_draw_axis_deg=nominal_draw_axis_deg,
        )
        ffwhms = [float(s * _FWHM_PER_SIGMA) for s in fsigs]
        flat_fwhm_med = float(np.median(ffwhms))
        fmodel = evaluate_c2_bridge_four_lobe(
            xo,
            c0=fc0,
            c2=0.0,
            chi_sym_deg=fchi,
            alpha_deg=falpha,
            lobe_amplitudes=famps,
            lobe_sigmas_deg=fsigs,
        )
        ferr = yo - fmodel
        sse_flat = float(np.sum(ferr ** 2))
        rmse_flat = float(math.sqrt(sse_flat / float(n_obs)))
        r2_flat = float(1.0 - sse_flat / sst) if sst > 1.0e-30 else None
        aic_flat, bic_flat = _aic_bic(sse_flat, n_obs, n_params_flat)

        dof_c2 = n_obs - n_params_c2
        f_stat: float | None = None
        if dof_c2 > 0 and sse_c2 > 1.0e-30 and sse_flat >= sse_c2:
            f_stat = _finite(((sse_flat - sse_c2) / 1.0) / (sse_c2 / float(dof_c2)))
        elif dof_c2 > 0 and sse_flat > 1.0e-20 and sse_c2 <= 1.0e-30:
            f_stat = None  # exact noiseless fit with zero residual

        delta_bic = _finite(bic_flat - bic_c2) if (bic_flat is not None and bic_c2 is not None) else None
        delta_aic = _finite(aic_flat - aic_c2) if (aic_flat is not None and aic_c2 is not None) else None
        bridge_improved = bool(
            (delta_bic is not None and delta_bic > 2.0)
            and (rmse_flat - rmse_c2 > 1.0e-6 * y_range)
        )
        flat_comparison = {
            "success": bool(best_flat_fit.success),
            "c0": _finite(fc0),
            "chi_sym_deg": _finite(fchi),
            "alpha_deg": _finite(falpha),
            "lobe_amplitudes": [_finite(v) for v in famps],
            "lobe_fwhms_deg": [_finite(v) for v in ffwhms],
            "lobe_fwhm_deg": _finite(flat_fwhm_med),
            "rmse": _finite(rmse_flat),
            "r_squared": _finite(r2_flat),
            "aic": aic_flat,
            "bic": bic_flat,
            "delta_rmse": _finite(rmse_flat - rmse_c2),
            "delta_r_squared": _finite(r2_c2 - r2_flat) if (r2_c2 is not None and r2_flat is not None) else None,
            "delta_aic": delta_aic,
            "delta_bic": delta_bic,
            "bridge_f_stat": f_stat,
            "bridge_improved": bridge_improved,
            "lobe_fwhm_bias_deg": _finite(flat_fwhm_med - lobe_fwhm_summary),
        }

    return json_safe({
        "success": success,
        "status": "estimated" if success else "warning",
        "message": str(best_c2_fit.message),
        "n_observed_bins": n_obs,
        "n_total_bins": int(x.size),
        "angular_coverage": _finite(n_obs / float(x.size)),
        "c0": _finite(c0_raw),
        "c2": _finite(c2_val),
        "chi_sym_deg": _finite(chi_sym),
        "alpha_deg": _finite(alpha_deg),
        "lobe_centers_deg": [_finite(v) for v in centers],
        "lobe_amplitudes": [_finite(v) for v in amps],
        "lobe_sigmas_deg": [_finite(v) for v in sigmas],
        "lobe_fwhms_deg": [_finite(v) for v in fwhms],
        "lobe_fwhm_deg": _finite(lobe_fwhm_summary),
        "lobe_supported": lobe_supported,
        "initial_centres_deg": [_finite(v) for v in initial_centers_used],
        "rmse": _finite(rmse_c2),
        "r_squared": _finite(r2_c2),
        "aic": aic_c2,
        "bic": bic_c2,
        "angle_deg": _finite_list(x),
        "observed": _finite_list(np.where(observed, y, np.nan)),
        "model": _finite_list(model_all),
        "bridge_component": _finite_list(bridge_all),
        "lobes_component": _finite_list(lobes_all),
        "residual": _finite_list(residual_all),
        "flat_baseline_comparison": flat_comparison,
        "flags": list(dict.fromkeys(flags)),
    })


# ---------------------------------------------------------------------------
# 3. Dual-Channel Radial Coarsening & Invariant Observables
# ---------------------------------------------------------------------------


def _interpolate_crossing(x0: float, x1: float, y0: float, y1: float, target: float) -> float:
    if y1 == y0:
        return float(x0)
    frac = min(1.0, max(0.0, (target - y0) / (y1 - y0)))
    return float(x0 + frac * (x1 - x0))


def _integrate_observed_segments_weights(
    x: np.ndarray,
    valid: np.ndarray,
    dx_bins: np.ndarray,
) -> np.ndarray:
    """Return exact per-bin linear weights matching :func:`_integrate_observed_segments`."""
    weights = np.zeros(x.shape, dtype=float)
    for a, b in _contiguous_segments(valid):
        n_seg = b - a
        if n_seg == 1:
            weights[a] = float(dx_bins[a])
        else:
            dx_int = np.diff(x[a:b])
            weights[a] = 0.5 * float(dx_bins[a]) + 0.5 * float(dx_int[0])
            weights[b - 1] = 0.5 * float(dx_bins[b - 1]) + 0.5 * float(dx_int[-1])
            if n_seg > 2:
                weights[a + 1 : b - 1] = 0.5 * (dx_int[:-1] + dx_int[1:])
    return weights


def _integrate_observed_segments(x: np.ndarray, y: np.ndarray, dx_bins: np.ndarray) -> float | None:
    """Integrate ``y(x)`` strictly over contiguous observed segments without interpolating gaps."""
    valid = np.isfinite(x) & np.isfinite(y)
    segments = _contiguous_segments(valid)
    if not segments:
        return None
    total = 0.0
    for a, b in segments:
        if b - a >= 2:
            # Trapezoid over interior plus half-bin end caps so uniform full-coverage
            # grids match midpoint quadrature over [q_min, q_max]
            seg_int = float(np.trapezoid(y[a:b], x[a:b]))
            seg_int += 0.5 * float(y[a]) * float(dx_bins[a]) + 0.5 * float(y[b - 1]) * float(dx_bins[b - 1])
            total += seg_int
        else:
            total += float(y[a]) * float(dx_bins[a])
    return _finite(total)


def _estimate_local_noise_sigma(values: np.ndarray, valid: np.ndarray) -> float:
    """Estimate local bin-to-bin noise via 3rd differences (annihilates quadratic curvature)."""
    if int(np.count_nonzero(valid)) < 16:
        return 0.0
    diffs: list[np.ndarray] = []
    for a, b in _contiguous_segments(valid):
        if b - a >= 4:
            ys = values[a:b]
            d3 = ys[3:] - 3.0 * ys[2:-1] + 3.0 * ys[1:-2] - ys[:-3]
            finite_d3 = d3[np.isfinite(d3)]
            if finite_d3.size:
                diffs.append(np.abs(finite_d3))
    if not diffs:
        return 0.0
    all_d3 = np.concatenate(diffs)
    if all_d3.size < 4:
        return 0.0
    return float(1.4826 * float(np.median(all_d3)) / math.sqrt(20.0))


def _estimate_radial_background_and_active_mask(
    q_centers: np.ndarray,
    I_q: np.ndarray,
    counts: np.ndarray,
    *,
    zero_pixel_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, int]:
    """Exclude outer zero-padded dead shells and estimate the active high-q diffuse floor.

    Returns ``(active_mask, I_bg_estimate, tail_noise_mad, n_dead_outer_bins)``.
    """
    obs = np.isfinite(q_centers) & np.isfinite(I_q) & (counts > 0)
    if not np.any(obs):
        return obs, 0.0, 0.0, 0

    obs_idx = np.flatnonzero(obs)
    max_val = float(np.max(I_q[obs_idx]))
    dead_outer = np.zeros(q_centers.shape, dtype=bool)

    if max_val > 1.0e-9 and obs_idx.size >= 4:
        if zero_pixel_fraction is not None:
            pos = obs_idx.size - 1
            while pos >= 0 and float(zero_pixel_fraction[obs_idx[pos]]) > 0.90:
                dead_outer[obs_idx[pos]] = True
                pos -= 1
            if np.any(dead_outer) and pos >= 0 and float(zero_pixel_fraction[obs_idx[pos]]) > 0.50:
                dead_outer[obs_idx[pos]] = True
        else:
            # 1D-profile fallback: detect trailing exact-zero shells preceded by a sharp cliff
            pos = obs_idx.size - 1
            while pos >= 0 and abs(float(I_q[obs_idx[pos]])) <= 1.0e-12:
                pos -= 1
            n_trailing_zero = (obs_idx.size - 1) - pos
            if n_trailing_zero >= 1 and pos >= 0:
                prev_val = float(I_q[obs_idx[pos]])
                if prev_val > max(1.0e-4 * max_val, 1.0e-6):
                    for k in range(pos + 1, obs_idx.size):
                        dead_outer[obs_idx[k]] = True

    active = obs & ~dead_outer
    if not np.any(active):
        active = obs
        dead_outer[:] = False
    n_dead = int(np.count_nonzero(dead_outer))

    y_act = I_q[active]
    if y_act.size >= 5:
        n_tail = max(5, int(round(0.15 * y_act.size)))
        tail_vals = y_act[-n_tail:]
        tail_med = float(np.median(tail_vals))
        tail_mad = float(1.4826 * float(np.median(np.abs(tail_vals - tail_med))))
        # Cap at 25th percentile of active profile so a peak near q_max is never subtracted as floor
        p25_global = float(np.percentile(y_act, 25.0))
        I_bg = max(0.0, min(tail_med, p25_global))
    else:
        I_bg = max(0.0, float(np.min(y_act)))
        tail_mad = 0.0
    return active, I_bg, tail_mad, n_dead


def _extract_radial_peak_from_profile(
    q_centers: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
    *,
    q_scale_nm_inv: float | None,
    q_unit_label: str,
    channel_name: str,
    active_mask: np.ndarray | None = None,
    floor_value: float | None = None,
    noise_sigma_per_bin: np.ndarray | None = None,
    min_rel_prominence: float = 0.02,
    min_rel_height: float = 0.05,
    min_snr: float = 5.0,
) -> dict[str, Any]:
    """Extract peak q*, L*_nm, FWHM, prominence, and edge-hit status from a 1D observed profile."""
    valid = np.isfinite(q_centers) & np.isfinite(values) & (counts > 0)
    if active_mask is not None:
        valid = valid & np.asarray(active_mask, dtype=bool)
    n_obs = int(np.count_nonzero(valid))
    caveats: list[str] = []
    if q_scale_nm_inv is None:
        caveats.append(f"L_star_nm unavailable for {channel_name}: q_unit={q_unit_label!r} is not a physical reciprocal length.")

    if n_obs < 3:
        return {
            "status": "unavailable",
            "reason": "insufficient_observed_radial_bins",
            "q_star": None,
            "L_star_nm": None,
            "peak_value": None,
            "fwhm_q": None,
            "prominence": None,
            "edge_hit": False,
            "edge_hit_side": None,
            "n_observed_bins": n_obs,
            "caveats": caveats,
        }

    obs_indices = np.flatnonzero(valid)
    first_obs = int(obs_indices[0])
    last_obs = int(obs_indices[-1])
    y_obs = values[obs_indices]
    global_max_pos = int(obs_indices[int(np.argmax(y_obs))])
    global_max_val = float(values[global_max_pos])
    floor = float(floor_value) if (floor_value is not None and math.isfinite(float(floor_value))) else float(np.min(y_obs))
    y_range = max(global_max_val - floor, 0.0)
    local_d3_sigma = _estimate_local_noise_sigma(values, valid)

    # Search for genuine interior peaks inside each contiguous observed segment
    candidates: list[tuple[float, float, int, int, int]] = []  # (prominence, height, global_idx, seg_a, seg_b)
    segments = _contiguous_segments(valid)
    for a, b in segments:
        if b - a < 3:
            continue
        ys = values[a:b]
        pks, _ = find_peaks(ys)
        if pks.size == 0:
            continue
        proms, _, _ = peak_prominences(ys, pks)
        for pk_local, prom in zip(pks, proms):
            prom_f = float(prom)
            if not math.isfinite(prom_f) or prom_f <= 0.0:
                continue
            pk_global_i = a + int(pk_local)
            height_f = float(ys[pk_local])
            bin_noise = (
                float(noise_sigma_per_bin[pk_global_i])
                if (noise_sigma_per_bin is not None and math.isfinite(float(noise_sigma_per_bin[pk_global_i])))
                else 0.0
            )
            eff_noise = max(local_d3_sigma, bin_noise)
            prom_thresh = max(min_snr * eff_noise, min_rel_prominence * y_range)
            height_thresh = max(min_snr * eff_noise, min_rel_height * y_range)
            if prom_f >= prom_thresh and (height_f - floor) >= height_thresh:
                candidates.append((prom_f, height_f, pk_global_i, a, b))

    if not candidates or y_range <= 1.0e-15:
        edge_side = None
        edge_hit = False
        if global_max_pos <= first_obs + 1:
            edge_hit = True
            edge_side = "low_q"
        elif global_max_pos >= last_obs - 1:
            edge_hit = True
            edge_side = "high_q"
        return {
            "status": "boundary_monotonic" if edge_hit else "no_interior_peak",
            "reason": f"monotonic_maximum_at_{edge_side}_boundary" if edge_hit else "flat_or_no_interior_maximum",
            "q_star": None,
            "L_star_nm": None,
            "peak_value": _finite(global_max_val),
            "boundary_q": _finite(q_centers[global_max_pos]) if edge_hit else None,
            "fwhm_q": None,
            "prominence": None,
            "edge_hit": edge_hit,
            "edge_hit_side": edge_side,
            "n_observed_bins": n_obs,
            "caveats": caveats,
        }

    # Prefer the peak with the largest prominence (tie-break by peak height)
    prom_best, peak_val, pk_global, seg_a, seg_b = max(candidates, key=lambda item: (item[0], item[1]))

    # Sub-bin parabolic refinement using immediate observed neighbors
    q_star = float(q_centers[pk_global])
    if seg_a < pk_global < seg_b - 1:
        y_m = float(values[pk_global - 1])
        y_0 = float(values[pk_global])
        y_p = float(values[pk_global + 1])
        denom = y_m - 2.0 * y_0 + y_p
        if denom < -1.0e-15:
            shift = 0.5 * (y_m - y_p) / denom
            if abs(shift) <= 0.5:
                dq_step = 0.5 * float(q_centers[pk_global + 1] - q_centers[pk_global - 1])
                q_star = float(q_centers[pk_global] + shift * dq_step)

    # Half-height width on the contiguous observed segment (never crosses gaps)
    xs = q_centers[seg_a:seg_b]
    ys = values[seg_a:seg_b]
    pk_loc = pk_global - seg_a
    # Local baseline from the prominence base level (peak_val - prom_best)
    baseline = peak_val - prom_best
    half_height = baseline + 0.5 * prom_best
    left_cross = None
    for i in range(pk_loc - 1, -1, -1):
        if ys[i] <= half_height <= ys[i + 1]:
            left_cross = _interpolate_crossing(float(xs[i]), float(xs[i + 1]), float(ys[i]), float(ys[i + 1]), half_height)
            break
    right_cross = None
    for i in range(pk_loc, ys.size - 1):
        if ys[i] >= half_height >= ys[i + 1]:
            right_cross = _interpolate_crossing(float(xs[i]), float(xs[i + 1]), float(ys[i]), float(ys[i + 1]), half_height)
            break
    fwhm_q = _finite(right_cross - left_cross) if (left_cross is not None and right_cross is not None and right_cross > left_cross) else None

    # Edge-hit classification
    edge_hit = False
    edge_side = None
    if pk_global <= first_obs + 1 or (global_max_pos == first_obs and global_max_val > peak_val and left_cross is None):
        edge_hit = True
        edge_side = "low_q"
    elif pk_global >= last_obs - 1 or (global_max_pos == last_obs and global_max_val > peak_val and right_cross is None):
        edge_hit = True
        edge_side = "high_q"

    L_star_nm = _spacing_nm_from_q(q_star, q_scale_nm_inv)
    status = "edge_truncated_peak" if edge_hit else "estimated"
    return {
        "status": status,
        "reason": f"peak_near_{edge_side}_window_boundary" if edge_hit else None,
        "q_star": _finite(q_star),
        "L_star_nm": L_star_nm,
        "peak_value": _finite(peak_val),
        "fwhm_q": fwhm_q,
        "prominence": _finite(prom_best),
        "edge_hit": edge_hit,
        "edge_hit_side": edge_side,
        "n_observed_bins": n_obs,
        "caveats": caveats,
    }


def _summarize_radial_channel(
    q_edges: np.ndarray,
    I_q: np.ndarray,
    counts: np.ndarray,
    candidate_counts: np.ndarray,
    *,
    q_scale_nm_inv: float | None,
    q_unit_label: str,
    channel_name: str,
    zero_pixel_fraction: np.ndarray | None = None,
    background_level: float | None = None,
    intensity_mode: str = "calibrated",
    sigma_radial: np.ndarray | None = None,
) -> dict[str, Any]:
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    dq_bins = q_edges[1:] - q_edges[:-1]
    obs_mask = np.isfinite(I_q) & (counts > 0)
    active_mask, I_bg_auto, tail_mad, n_dead_outer = _estimate_radial_background_and_active_mask(
        q_centers, I_q, counts, zero_pixel_fraction=zero_pixel_fraction
    )
    if background_level is not None:
        bg_val = float(background_level)
        if not math.isfinite(bg_val) or bg_val < 0.0:
            raise ValueError("background_level must be finite and non-negative when provided")
        I_bg = bg_val
    else:
        I_bg = I_bg_auto
    kratky_raw = np.where(obs_mask, (q_centers ** 2) * I_q, np.nan)
    I_excess = np.where(active_mask, np.maximum(I_q - I_bg, 0.0), np.nan)
    kratky_sub = np.where(active_mask, (q_centers ** 2) * I_excess, np.nan)
    coverage = np.divide(
        counts,
        candidate_counts,
        out=np.full(q_centers.shape, np.nan, dtype=float),
        where=candidate_counts > 0,
    )
    direct_noise = np.full(q_centers.shape, tail_mad, dtype=float)
    kratky_noise = (q_centers ** 2) * tail_mad
    direct_peak = _extract_radial_peak_from_profile(
        q_centers,
        I_q,
        counts,
        q_scale_nm_inv=q_scale_nm_inv,
        q_unit_label=q_unit_label,
        channel_name=f"{channel_name}_direct",
        active_mask=active_mask,
        floor_value=I_bg,
        noise_sigma_per_bin=direct_noise,
    )
    kratky_peak = _extract_radial_peak_from_profile(
        q_centers,
        kratky_sub,
        counts,
        q_scale_nm_inv=q_scale_nm_inv,
        q_unit_label=q_unit_label,
        channel_name=f"{channel_name}_kratky",
        active_mask=active_mask,
        floor_value=0.0,
        noise_sigma_per_bin=kratky_noise,
    )
    kratky_peak["kratky_background_subtracted"] = True
    kratky_peak["I_bg_estimate"] = _finite(I_bg)

    if intensity_mode == "raw":
        q_win = None
        q_win_bg_sub = None
        q_win_stat_stderr = None
    else:
        q_win = _integrate_observed_segments(q_centers, kratky_raw, dq_bins)
        q_win_bg_sub = _integrate_observed_segments(q_centers, kratky_sub, dq_bins)
        q_win_stat_stderr = None
        if sigma_radial is not None:
            sig_ok = obs_mask & np.isfinite(sigma_radial) & (sigma_radial >= 0.0)
            if np.any(sig_ok):
                seg_w = _integrate_observed_segments_weights(q_centers, obs_mask, dq_bins)
                qwin_var = float(np.sum(((q_centers[sig_ok] ** 2) * seg_w[sig_ok] * sigma_radial[sig_ok]) ** 2))
                if math.isfinite(qwin_var) and qwin_var >= 0.0:
                    q_win_stat_stderr = _finite(math.sqrt(qwin_var))

    obs_bins = int(np.count_nonzero(obs_mask))
    cand_bins = int(np.count_nonzero(candidate_counts > 0))
    radial_cov = float(obs_bins / cand_bins) if cand_bins > 0 else 0.0

    return {
        "status": "estimated" if obs_bins > 0 else "unavailable",
        "q_centers": _finite_list(q_centers),
        "q_edges": _finite_list(q_edges),
        "I_q": _finite_list(I_q),
        "sigma_radial": _finite_list(sigma_radial) if sigma_radial is not None else None,
        "q2_I_q": _finite_list(kratky_raw),
        "q2_I_sub_q": _finite_list(kratky_sub),
        "I_bg_estimate": _finite(I_bg),
        "I_bg_noise_mad": _finite(tail_mad),
        "kratky_background_subtracted": True,
        "n_outer_zero_padded_bins_excluded": n_dead_outer,
        "counts": [int(c) for c in counts],
        "candidate_counts": [int(c) for c in candidate_counts],
        "coverage": _finite_list(coverage),
        "radial_bin_coverage": _finite(radial_cov),
        "direct_peak": direct_peak,
        "kratky_peak": kratky_peak,
        "q_win_invariant": q_win,
        "q_win_bg_subtracted": q_win_bg_sub,
        "q_win_stat_stderr": q_win_stat_stderr,
    }


def _compute_cosine_correlation_1d(
    q_centers: np.ndarray,
    kratky: np.ndarray,
    dq_bins: np.ndarray,
    *,
    q_scale_nm_inv: float | None,
    q_unit_label: str,
    r_max_nm: float | None,
    n_r: int,
) -> dict[str, Any]:
    """Compute window-bounded 1D cosine correlation function K(r) for calibrated physical q only."""
    if q_scale_nm_inv is None:
        return {
            "status": "unavailable",
            "reason": "uncalibrated_q_unit",
            "caveat": (
                f"1D cosine correlation function K(r) in nm requires calibrated physical q "
                f"(nm^-1 or A^-1); unavailable for q_unit={q_unit_label!r}."
            ),
            "r_nm": None,
            "K_r": None,
            "first_maximum_r_nm": None,
            "truncation_warnings": ["uncalibrated_q_unit_no_physical_correlation"],
        }

    valid = np.isfinite(q_centers) & np.isfinite(kratky)
    segments = _contiguous_segments(valid)
    if not segments:
        return {
            "status": "unavailable",
            "reason": "no_observed_radial_bins",
            "caveat": "No observed radial bins available for K(r).",
            "r_nm": None,
            "K_r": None,
            "first_maximum_r_nm": None,
            "truncation_warnings": ["no_observed_radial_bins"],
        }

    q_nm = q_centers * q_scale_nm_inv
    dq_nm = dq_bins * q_scale_nm_inv
    # Scale q^2 I(q) consistently to nm^-1 axis: q_nm^2 * I = (q_scale^2) * kratky
    k_nm = kratky * (q_scale_nm_inv ** 2)
    quad_w = _integrate_observed_segments_weights(q_nm, valid, dq_nm)
    q_win_nm = _finite(float(np.dot(k_nm[valid], quad_w[valid])))
    if q_win_nm is None or q_win_nm <= 0.0:
        return {
            "status": "unavailable",
            "reason": "nonpositive_window_invariant",
            "caveat": "Window-bounded invariant integral is non-positive; K(r) cannot be normalized.",
            "r_nm": None,
            "K_r": None,
            "first_maximum_r_nm": None,
            "truncation_warnings": ["nonpositive_window_invariant"],
        }

    obs_q_nm = q_nm[valid]
    q_lo_nm = float(np.min(obs_q_nm))
    q_hi_nm = float(np.max(obs_q_nm))
    if r_max_nm is None:
        r_max = min(150.0, 2.0 * math.pi / max(q_lo_nm, 0.04))
    else:
        r_max = float(r_max_nm)
        if not math.isfinite(r_max) or r_max <= 0.0:
            raise ValueError("r_max_nm must be finite and positive")

    n_pts = max(int(n_r), 16)
    r_grid = np.linspace(0.0, r_max, n_pts, dtype=float)
    weighted_k = k_nm[valid] * quad_w[valid]
    k_vals = (np.cos(np.outer(r_grid, obs_q_nm)) @ weighted_k) / q_win_nm

    # First positive interior maximum (classic 1D correlation peak; ignores negative Gibbs ripples in the trough)
    first_max_r: float | None = None
    pks, _ = find_peaks(k_vals)
    pos_pks = [int(p) for p in pks if math.isfinite(float(k_vals[p])) and float(k_vals[p]) > 0.0]
    if pos_pks:
        first_max_r = _finite(r_grid[pos_pks[0]])

    trunc_warnings = [
        "window_bounded_cosine_transform_only_no_guinier_or_porod_extrapolation",
    ]
    if len(segments) > 1:
        trunc_warnings.append("missing_radial_bins_excluded_not_interpolated")

    return {
        "status": "estimated",
        "reason": None,
        "r_nm": _finite_list(r_grid),
        "K_r": _finite_list(k_vals),
        "first_maximum_r_nm": first_max_r,
        "r_min_resolution_nm": _finite(math.pi / q_hi_nm) if q_hi_nm > 0.0 else None,
        "r_max_window_nm": _finite(math.pi / q_lo_nm) if q_lo_nm > 0.0 else None,
        "q_range_nm_inv": [_finite(q_lo_nm), _finite(q_hi_nm)],
        "truncation_warnings": trunc_warnings,
        "caveat": (
            "Window-bounded 1D cosine correlation function K(r) computed strictly over observed "
            "[q_min, q_max] without Guinier or Porod extrapolation; finite-window truncation "
            "ripples are expected."
        ),
    }


def _detect_wing_angles_from_1d_pixels(
    angles_obs_deg: np.ndarray,
    img_obs: np.ndarray,
    *,
    nominal_draw_axis_deg: float,
    n_bins: int = 180,
) -> tuple[list[float], str]:
    """Detect butterfly wing ray angles directly from already-filtered 1D observed pixels."""
    if angles_obs_deg.size < 16:
        return [], "insufficient_azimuthal_support"
    step = 360.0 / float(n_bins)
    ang = (np.arange(n_bins, dtype=float) + 0.5) * step
    idx = np.floor(angles_obs_deg * (float(n_bins) / 360.0)).astype(np.int64)
    np.clip(idx, 0, n_bins - 1, out=idx)
    cnts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    sums = np.bincount(idx, weights=img_obs, minlength=n_bins).astype(float)
    prof = np.full(n_bins, np.nan, dtype=float)
    np.divide(sums, cnts, out=prof, where=cnts > 0)
    obs = np.isfinite(prof) & (cnts > 0)
    if int(np.count_nonzero(obs)) < 16:
        return [], "insufficient_azimuthal_support"
    harm = compute_azimuthal_harmonics_and_order(
        angle_deg=ang, intensity=prof, counts=cnts, nominal_draw_axis_deg=nominal_draw_axis_deg
    )
    a2 = harm.get("a2") or 0.0
    a4 = harm.get("a4") or 0.0
    bisector = harm.get("friedel_bisector_deg")
    if bisector is None or (a2 < 0.03 and a4 < 0.03):
        return [], "isotropic_no_distinct_wings"

    # If 4-lobe structure is present, refine with C2-bridge + 4-lobe fit
    if a4 >= 0.03:
        fit = fit_c2_bridge_four_lobe(
            ang,
            prof,
            counts=cnts,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            initial_chi_sym_deg=bisector,
            initial_alpha_deg=harm.get("wing_half_opening_estimate_deg"),
            max_nfev=1500,
            include_flat_comparison=False,
        )
        if fit.get("success") and fit.get("lobe_centers_deg"):
            centers = [float(c) for c in fit["lobe_centers_deg"] if c is not None]
            if len(centers) == 4:
                return centers, "c2_bridge_four_lobe"
        alpha_est = harm.get("wing_half_opening_estimate_deg")
        if alpha_est is not None:
            return list(_butterfly_lobe_centers_deg(bisector, float(alpha_est))), "harmonic_folded_wings"

    # 2-lobe fallback
    return [float(bisector % 360.0), float((bisector + 180.0) % 360.0)], "two_lobe_axis"


def _detect_wing_angles_for_sectors(
    img: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    geometry: np.ndarray | None,
    usable: np.ndarray | None,
    *,
    q_window: tuple[float, float],
    nominal_draw_axis_deg: float,
    q_mag: np.ndarray | None = None,
    chi_deg: np.ndarray | None = None,
    inclusion: np.ndarray | None = None,
) -> tuple[list[float], str]:
    """Detect butterfly wing ray angles strictly from observed pixels in ``q_window``."""
    ang, prof, cnts, _, _, _ = _bin_azimuthal_from_2d(
        img,
        qx,
        qy,
        geometry,
        usable,
        q_window=q_window,
        n_bins=180,
        q_mag=q_mag,
        chi_deg=chi_deg,
        inclusion=inclusion,
    )
    obs = np.isfinite(prof) & (cnts > 0)
    if int(np.count_nonzero(obs)) < 16:
        return [], "insufficient_azimuthal_support"
    harm = compute_azimuthal_harmonics_and_order(
        angle_deg=ang, intensity=prof, counts=cnts, nominal_draw_axis_deg=nominal_draw_axis_deg
    )
    a2 = harm.get("a2") or 0.0
    a4 = harm.get("a4") or 0.0
    bisector = harm.get("friedel_bisector_deg")
    if bisector is None or (a2 < 0.03 and a4 < 0.03):
        return [], "isotropic_no_distinct_wings"

    if a4 >= 0.03:
        fit = fit_c2_bridge_four_lobe(
            ang,
            prof,
            counts=cnts,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            initial_chi_sym_deg=bisector,
            initial_alpha_deg=harm.get("wing_half_opening_estimate_deg"),
            max_nfev=1500,
            include_flat_comparison=False,
        )
        if fit.get("success") and fit.get("lobe_centers_deg"):
            centers = [float(c) for c in fit["lobe_centers_deg"] if c is not None]
            if len(centers) == 4:
                return centers, "c2_bridge_four_lobe"
        alpha_est = harm.get("wing_half_opening_estimate_deg")
        if alpha_est is not None:
            return list(_butterfly_lobe_centers_deg(bisector, float(alpha_est))), "harmonic_folded_wings"

    return [float(bisector % 360.0), float((bisector + 180.0) % 360.0)], "two_lobe_axis"


def measure_radial_coarsening_observables(
    image: Any,
    qx: Any = None,
    qy: Any = None,
    *,
    valid_mask: Any = None,
    q_window: tuple[float, float] | None = None,
    q_unit: str = "unknown",
    n_radial_bins: int = 128,
    lobe_angles_deg: Sequence[float] | None = None,
    sector_half_width_deg: float = 15.0,
    nominal_draw_axis_deg: float = 90.0,
    background_level: float | None = None,
    compute_correlation: bool = False,
    r_max_nm: float | None = None,
    n_r: int = 128,
    geometry_cache: Any = None,
    intensity_mode: str = "calibrated",
    sigma: Any = None,
    systematic_k_uncertainty: float | str | None = None,
) -> dict[str, Any]:
    """Compute dual-channel direct I(q) & Kratky q^2 I(q) radial observables and Q_win.

    Extracts BOTH full-azimuth and butterfly lobe-sectored radial profiles strictly
    from observed pixels, returning direct ``I(q)`` peaks, Kratky ``q^2 I(q)``
    peaks, window-bounded scattering invariant proxy ``Q_win`` (strictly ``None``
    when ``intensity_mode='raw'``), optional propagated statistical uncertainty
    (``sigma_radial`` and ``q_win_stat_stderr`` when ``sigma`` is provided), and
    optional calibrated 1D cosine correlation function ``K(r)``.
    """
    mode = _normalise_intensity_mode(intensity_mode)
    if isinstance(n_radial_bins, (bool, np.bool_)) or int(n_radial_bins) < 4:
        raise ValueError("n_radial_bins must be an integer >= 4")
    n_bins = int(n_radial_bins)
    half_w = float(sector_half_width_deg)
    if not math.isfinite(half_w) or not (0.0 < half_w <= 90.0):
        raise ValueError("sector_half_width_deg must be finite and in (0, 90]")

    req_q_max: float | None = None
    if q_window is not None:
        q_min, q_max = _validate_q_window(q_window)
        req_q_max = q_max

    img, qx_arr, qy_arr, q_mag, chi_deg, geom_mask, inclusion, sig_arr, _ = _resolve_2d_context(
        image,
        qx,
        qy,
        valid_mask,
        geometry_cache=geometry_cache,
        sigma=sigma,
        required_q_max=req_q_max,
    )
    q_unit_label, q_scale_nm_inv, unit_caveat = _q_unit_info(q_unit)

    # Pre-filter strictly to pixels inside [q_min, q_max] BEFORE arctan2, binning, and sector checks
    if q_window is not None and q_mag is None:
        in_box = (np.abs(qx_arr) <= q_max) & (np.abs(qy_arr) <= q_max)
        if not np.any(in_box):
            if not np.any(np.isfinite(qx_arr) & np.isfinite(qy_arr)):
                raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
            q_c = np.empty(0, dtype=float)
            qx_c = np.empty(0, dtype=float)
            qy_c = np.empty(0, dtype=float)
            img_c = np.empty(0, dtype=float)
            inc_c = np.empty(0, dtype=bool)
            sig_c = np.empty(0, dtype=float) if sig_arr is not None else None
            chi_c = None
        else:
            bx = qx_arr[in_box]
            by = qy_arr[in_box]
            bq = np.hypot(bx, by)
            in_q = np.isfinite(bq) & (bq >= q_min) & (bq <= q_max)
            q_c = bq[in_q]
            qx_c = bx[in_q]
            qy_c = by[in_q]
            img_c = img[in_box][in_q]
            inc_c = inclusion[in_box][in_q]
            sig_c = sig_arr[in_box][in_q] if sig_arr is not None else None
            chi_c = None
    else:
        assert q_mag is not None and geom_mask is not None
        if q_window is None:
            usable_full = geom_mask & inclusion & np.isfinite(img)
            obs_q = q_mag[usable_full]
            if obs_q.size == 0:
                obs_q = q_mag[geom_mask]
            q_min = float(np.min(obs_q))
            q_max = float(np.max(obs_q))
            if q_min == q_max:
                half = max(abs(q_min), 1.0) * 1.0e-6
                q_min = max(0.0, q_min - half)
                q_max = q_max + half
        in_q = geom_mask & (q_mag >= q_min) & (q_mag <= q_max)
        q_c = q_mag[in_q]
        qx_c = qx_arr[in_q]
        qy_c = qy_arr[in_q]
        img_c = img[in_q]
        inc_c = inclusion[in_q]
        sig_c = sig_arr[in_q] if sig_arr is not None else None
        chi_c = np.mod(chi_deg[in_q], 360.0) if chi_deg is not None else None

    obs_c = inc_c & np.isfinite(img_c)
    if sig_c is not None and np.any(obs_c & np.isfinite(sig_c) & (sig_c < 0.0)):
        raise ValueError("sigma must be non-negative on valid observed pixels")

    q_edges = np.linspace(q_min, q_max, n_bins + 1, dtype=float)
    inv_dq = float(n_bins) / max(q_max - q_min, 1.0e-30)
    q_idx_c = np.floor((q_c - q_min) * inv_dq).astype(np.int64)
    np.clip(q_idx_c, 0, n_bins - 1, out=q_idx_c)

    def _bin_radial_1d(
        cand_sel: np.ndarray | None,
        obs_sel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        c_idx = q_idx_c if cand_sel is None else q_idx_c[cand_sel]
        o_idx = q_idx_c[obs_sel]
        o_img = img_c[obs_sel]
        c_cnt = np.bincount(c_idx, minlength=n_bins).astype(np.int64)
        o_cnt = np.bincount(o_idx, minlength=n_bins).astype(np.int64)
        o_sum = np.bincount(o_idx, weights=o_img, minlength=n_bins).astype(float)
        o_mean = np.full(n_bins, np.nan, dtype=float)
        np.divide(o_sum, o_cnt, out=o_mean, where=o_cnt > 0)
        z_mask = np.abs(o_img) <= 1.0e-12
        z_cnt = np.bincount(o_idx[z_mask], minlength=n_bins).astype(np.int64) if np.any(z_mask) else np.zeros(n_bins, dtype=np.int64)
        z_frac = np.divide(z_cnt, o_cnt, out=np.zeros(n_bins, dtype=float), where=o_cnt > 0)

        se_rad: np.ndarray | None = None
        if sig_c is not None:
            o_sig = sig_c[obs_sel]
            sig_ok = np.isfinite(o_sig) & (o_sig >= 0.0)
            if np.any(sig_ok):
                s_idx = o_idx[sig_ok]
                s_val = o_sig[sig_ok]
                s_cnt = np.bincount(s_idx, minlength=n_bins).astype(float)
                s_var = np.bincount(s_idx, weights=s_val * s_val, minlength=n_bins).astype(float)
                se_rad = np.full(n_bins, np.nan, dtype=float)
                good_s = s_cnt > 0
                se_rad[good_s] = np.sqrt(s_var[good_s]) / s_cnt[good_s]
        return o_mean, o_cnt, c_cnt, z_frac, se_rad

    # Full-azimuth channel
    full_I, full_cnt, full_cand, full_zfrac, full_se = _bin_radial_1d(None, obs_c)
    full_summary = _summarize_radial_channel(
        q_edges,
        full_I,
        full_cnt,
        full_cand,
        q_scale_nm_inv=q_scale_nm_inv,
        q_unit_label=q_unit_label,
        channel_name="full_azimuth",
        zero_pixel_fraction=full_zfrac,
        background_level=background_level,
        intensity_mode=mode,
        sigma_radial=full_se,
    )

    # Azimuthal angles computed strictly on the pre-filtered 1D in-window pixels
    angles_c = chi_c if chi_c is not None else np.mod(np.degrees(np.arctan2(qy_c, qx_c)), 360.0)
    pos_q = q_c > 0.0

    # Lobe-sectored channel
    if lobe_angles_deg is not None:
        sector_angles = [float(a) % 360.0 for a in lobe_angles_deg if math.isfinite(float(a))]
        sector_source = "user_supplied"
    else:
        pos_obs = obs_c & pos_q
        sector_angles, sector_source = _detect_wing_angles_from_1d_pixels(
            angles_c[pos_obs],
            img_c[pos_obs],
            nominal_draw_axis_deg=nominal_draw_axis_deg,
        )

    if sector_angles and angles_c.size > 0:
        in_sector = np.zeros(angles_c.shape, dtype=bool)
        for ray_deg in sector_angles:
            in_sector |= np.abs(((angles_c - ray_deg + 180.0) % 360.0) - 180.0) <= half_w
        in_sector &= pos_q
        lobe_I, lobe_cnt, lobe_cand, lobe_zfrac, lobe_se = _bin_radial_1d(in_sector, obs_c & in_sector)
        lobe_summary = _summarize_radial_channel(
            q_edges,
            lobe_I,
            lobe_cnt,
            lobe_cand,
            q_scale_nm_inv=q_scale_nm_inv,
            q_unit_label=q_unit_label,
            channel_name="lobe_sectored",
            zero_pixel_fraction=lobe_zfrac,
            background_level=background_level,
            intensity_mode=mode,
            sigma_radial=lobe_se,
        )
        lobe_summary["lobe_angles_deg"] = sector_angles
        lobe_summary["sector_half_width_deg"] = half_w
        lobe_summary["lobe_angle_source"] = sector_source
    else:
        lobe_summary = {
            "status": "unavailable",
            "reason": sector_source,
            "lobe_angles_deg": [],
            "sector_half_width_deg": half_w,
            "lobe_angle_source": sector_source,
            "direct_peak": None,
            "kratky_peak": None,
            "q_win_invariant": None,
            "q_win_bg_subtracted": None,
            "q_win_stat_stderr": None,
            "sigma_radial": None,
        }

    # Primary top-level peaks: prefer lobe-sectored when detected and populated, else full-azimuth
    use_lobe = (
        lobe_summary.get("status") == "estimated"
        and isinstance(lobe_summary.get("direct_peak"), dict)
        and isinstance(lobe_summary.get("kratky_peak"), dict)
    )
    primary = lobe_summary if use_lobe else full_summary
    primary_source = "lobe_sectored" if use_lobe else "full_azimuth"

    direct_pk = primary["direct_peak"]
    kratky_pk = primary["kratky_peak"]

    # Optional 1D cosine correlation function K(r) from full-azimuth observed Kratky
    correlation = None
    if compute_correlation:
        q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
        dq_bins = q_edges[1:] - q_edges[:-1]
        kratky_full = np.where(np.isfinite(full_I) & (full_cnt > 0), (q_centers ** 2) * full_I, np.nan)
        correlation = _compute_cosine_correlation_1d(
            q_centers,
            kratky_full,
            dq_bins,
            q_scale_nm_inv=q_scale_nm_inv,
            q_unit_label=q_unit_label,
            r_max_nm=r_max_nm,
            n_r=n_r,
        )

    caveats = [
        "Q_win is a window-bounded scattering invariant proxy over observed [q_min, q_max] only; "
        "it does not estimate scattering-length-density contrast delta_rho or volume fraction.",
        "Ring/sector q* and L* = 2*pi/q* are distinct from ellipse geometry (a, b/a, theta_deg, conditional Ln/Lz).",
        "Kratky peak q_star_kratky is extracted from high-q diffuse-floor-subtracted q^2 * max(I(q) - I_bg, 0); "
        "raw q^2 * I(q) is retained in full_azimuth/lobe_sectored['q2_I_q'] and Q_win.",
    ]
    if unit_caveat is not None:
        caveats.append(unit_caveat)
    if mode == "raw":
        caveats.append(
            "intensity_mode='raw': q_win_invariant and q_win_bg_subtracted are withheld (None) "
            "because uncalibrated raw detector counts cannot produce an absolute scattering invariant."
        )

    flags: list[str] = []
    if mode == "raw":
        flags.append("raw_intensity_mode_q_win_withheld")
    if q_scale_nm_inv is None:
        flags.append("uncalibrated_pixel_q" if q_unit_label == "pixel-q" else "spacing_unavailable_unknown_q_unit")
    if direct_pk["edge_hit"]:
        flags.append(f"direct_peak_edge_hit_{direct_pk['edge_hit_side']}")
    if kratky_pk["edge_hit"]:
        flags.append(f"kratky_peak_edge_hit_{kratky_pk['edge_hit_side']}")
    if (full_summary.get("radial_bin_coverage") or 0.0) < 1.0:
        flags.append("incomplete_radial_coverage")
    if (full_summary.get("n_outer_zero_padded_bins_excluded") or 0) > 0:
        flags.append("outer_zero_padded_shells_excluded")

    direct_low_q_edge_hit = bool(direct_pk["edge_hit"] and direct_pk["edge_hit_side"] == "low_q")
    kratky_low_q_edge_hit = bool(kratky_pk["edge_hit"] and kratky_pk["edge_hit_side"] == "low_q")
    has_sigma = full_se is not None
    sys_k = "unavailable" if systematic_k_uncertainty is None else systematic_k_uncertainty

    return json_safe({
        "status": "estimated" if full_summary["status"] == "estimated" else "unavailable",
        "q_unit": q_unit_label,
        "intensity_mode": mode,
        "q_window": [q_min, q_max],
        "peak_channel_source": primary_source,
        "I_bg_estimate": primary.get("I_bg_estimate"),
        "I_bg_noise_mad": primary.get("I_bg_noise_mad"),
        "kratky_background_subtracted": True,
        "q_star_direct": direct_pk["q_star"],
        "L_star_direct_nm": direct_pk["L_star_nm"],
        "fwhm_direct_q": direct_pk["fwhm_q"],
        "prominence_direct": direct_pk["prominence"],
        "edge_hit_direct": direct_pk["edge_hit"],
        "edge_hit_direct_side": direct_pk["edge_hit_side"],
        "direct_low_q_edge_hit": direct_low_q_edge_hit,
        "q_star_kratky": kratky_pk["q_star"],
        "L_star_kratky_nm": kratky_pk["L_star_nm"],
        "fwhm_kratky_q": kratky_pk["fwhm_q"],
        "prominence_kratky": kratky_pk["prominence"],
        "edge_hit_kratky": kratky_pk["edge_hit"],
        "edge_hit_kratky_side": kratky_pk["edge_hit_side"],
        "kratky_low_q_edge_hit": kratky_low_q_edge_hit,
        "q_win_invariant": full_summary["q_win_invariant"],
        "q_win_bg_subtracted": full_summary["q_win_bg_subtracted"],
        "q_win_stat_stderr": full_summary["q_win_stat_stderr"],
        "sigma_radial": full_summary["sigma_radial"],
        "sigma_source": "supplied_statistical_sigma" if has_sigma else "unavailable",
        "systematic_k_uncertainty": sys_k,
        "q_win_units": f"intensity * ({q_unit_label})^3" if mode == "calibrated" else None,
        "full_azimuth": full_summary,
        "lobe_sectored": lobe_summary,
        "correlation_function": correlation,
        "flags": list(dict.fromkeys(flags)),
        "caveats": caveats,
    })


def _radial_metrics_from_1d_profile(
    q_edges: np.ndarray,
    I_q: np.ndarray,
    counts: np.ndarray,
    candidate_counts: np.ndarray,
    *,
    q_unit: str,
) -> dict[str, Any]:
    """Lightweight helper used by ``report_measurements.measure_frame`` on pre-binned profiles."""
    q_unit_label, q_scale_nm_inv, _ = _q_unit_info(q_unit)
    channel = _summarize_radial_channel(
        q_edges,
        I_q,
        counts,
        candidate_counts,
        q_scale_nm_inv=q_scale_nm_inv,
        q_unit_label=q_unit_label,
        channel_name="frame_radial",
    )
    return {
        "q_win_invariant": channel["q_win_invariant"],
        "q_star_direct": channel["direct_peak"]["q_star"],
        "L_star_direct_nm": channel["direct_peak"]["L_star_nm"],
        "edge_hit_direct": channel["direct_peak"]["edge_hit"],
        "q_star_kratky": channel["kratky_peak"]["q_star"],
        "L_star_kratky_nm": channel["kratky_peak"]["L_star_nm"],
        "edge_hit_kratky": channel["kratky_peak"]["edge_hit"],
    }


# ---------------------------------------------------------------------------
# 4. Perpendicular Butterfly Streak Width / Grubb-Williamson-Hall Analysis
# ---------------------------------------------------------------------------


def _fit_1d_transverse_gaussian(
    q_perp_centers: np.ndarray,
    profile: np.ndarray,
    counts: np.ndarray,
    *,
    q_perp_max: float,
    min_r2: float,
) -> dict[str, Any]:
    """Fit ``b + A * exp(-(q_perp - mu)^2 / (2 * sigma^2))`` strictly on observed bins."""
    obs = np.isfinite(q_perp_centers) & np.isfinite(profile) & (counts > 0)
    n_obs = int(np.count_nonzero(obs))
    support_frac = float(n_obs / float(q_perp_centers.size)) if q_perp_centers.size > 0 else 0.0
    if n_obs < 7:
        return {"valid": False, "reason": "insufficient_observed_transverse_bins", "support_fraction": support_frac, "n_observed_bins": n_obs}

    xo = q_perp_centers[obs]
    yo = profile[obs]
    y_min, y_max = float(np.min(yo)), float(np.max(yo))
    y_range = y_max - y_min
    if not math.isfinite(y_range) or y_range <= 1.0e-12:
        return {"valid": False, "reason": "flat_transverse_profile", "support_fraction": support_frac, "n_observed_bins": n_obs}

    # Check peak is near the ray center and observed on both sides
    i_max = int(np.argmax(yo))
    mu_0 = float(xo[i_max])
    if abs(mu_0) > 0.45 * q_perp_max:
        return {"valid": False, "reason": "peak_off_ray_center", "support_fraction": support_frac, "n_observed_bins": n_obs}
    left_obs = int(np.count_nonzero(xo < mu_0))
    right_obs = int(np.count_nonzero(xo > mu_0))
    if left_obs < 2 or right_obs < 2:
        return {"valid": False, "reason": "one_sided_transverse_support", "support_fraction": support_frac, "n_observed_bins": n_obs}

    dq_step = float(np.median(np.diff(q_perp_centers))) if q_perp_centers.size > 1 else q_perp_max / 10.0
    sig_lo = max(0.35 * dq_step, 1.0e-6)
    sig_hi = max(0.85 * q_perp_max, sig_lo * 2.0)

    b_0 = float(np.percentile(yo, 10.0))
    a_0 = max(y_max - b_0, 0.1 * y_range)
    # Initial sigma from half-height bin count
    above_half = xo[yo >= b_0 + 0.5 * a_0]
    if above_half.size >= 2:
        sig_0 = float((np.max(above_half) - np.min(above_half)) / _FWHM_PER_SIGMA)
    else:
        sig_0 = 0.2 * q_perp_max
    sig_0 = float(np.clip(sig_0, sig_lo * 1.1, sig_hi * 0.9))

    p0 = np.array([b_0, a_0, float(np.clip(mu_0, -0.4 * q_perp_max, 0.4 * q_perp_max)), sig_0], dtype=float)
    lo = np.array([-np.inf, 0.0, -0.45 * q_perp_max, sig_lo], dtype=float)
    hi = np.array([np.inf, max(10.0 * y_range, 10.0 * abs(y_max), 1.0), 0.45 * q_perp_max, sig_hi], dtype=float)

    def _model(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        u = (x - p[2]) / p[3]
        return p[0] + p[1] * np.exp(-0.5 * (u * u))

    def _res_1d(p: np.ndarray) -> np.ndarray:
        u = (xo - p[2]) / p[3]
        return p[0] + p[1] * np.exp(-0.5 * (u * u)) - yo

    def _jac_1d(p: np.ndarray) -> np.ndarray:
        inv_s = 1.0 / p[3]
        u = (xo - p[2]) * inv_s
        g = np.exp(-0.5 * (u * u))
        ag = p[1] * g
        J = np.empty((n_obs, 4), dtype=float)
        J[:, 0] = 1.0
        J[:, 1] = g
        J[:, 2] = ag * (u * inv_s)
        J[:, 3] = ag * ((u * u) * inv_s)
        return J

    try:
        fit = least_squares(_res_1d, p0, jac=_jac_1d, bounds=(lo, hi), loss="linear", max_nfev=1500)
    except (ValueError, RuntimeError, FloatingPointError):
        return {"valid": False, "reason": "transverse_optimizer_failed", "support_fraction": support_frac, "n_observed_bins": n_obs}

    b_f, a_f, mu_f, sig_f = (float(v) for v in fit.x)
    pred = _model(fit.x, xo)
    resid = yo - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((yo - float(np.mean(yo))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1.0e-30 else 0.0
    rmse = float(math.sqrt(sse / float(n_obs)))
    fwhm_f = sig_f * _FWHM_PER_SIGMA

    # Validate fit quality and non-degeneracy
    at_sig_bound = abs(sig_f - sig_lo) <= 1.0e-4 * max(sig_lo, 1.0) or abs(sig_f - sig_hi) <= 1.0e-4 * max(sig_hi, 1.0)
    at_mu_bound = abs(abs(mu_f) - 0.45 * q_perp_max) <= 1.0e-4 * max(q_perp_max, 1.0)
    reason = None
    valid = True
    if not fit.success:
        valid, reason = False, "optimizer_did_not_converge"
    elif a_f <= 0.0:
        valid, reason = False, "nonpositive_amplitude"
    elif r2 < min_r2:
        valid, reason = False, "low_r_squared"
    elif at_sig_bound:
        valid, reason = False, "width_at_bound"
    elif at_mu_bound:
        valid, reason = False, "center_at_bound"

    # Estimate sigma stderr from Jacobian
    fwhm_stderr = None
    if valid and n_obs > 4 and fit.jac is not None:
        try:
            jtj = fit.jac.T @ fit.jac
            s2 = sse / float(n_obs - 4)
            cov = np.linalg.pinv(jtj) * s2
            var_sig = float(cov[3, 3])
            if math.isfinite(var_sig) and var_sig >= 0.0:
                fwhm_stderr = _finite(math.sqrt(var_sig) * _FWHM_PER_SIGMA)
        except np.linalg.LinAlgError:
            pass

    return {
        "valid": valid,
        "reason": reason,
        "support_fraction": _finite(support_frac),
        "n_observed_bins": n_obs,
        "baseline": _finite(b_f),
        "amplitude": _finite(a_f),
        "center_q_perp": _finite(mu_f),
        "sigma_q_perp": _finite(sig_f),
        "fwhm_q_perp": _finite(fwhm_f),
        "fwhm_q_perp_stderr": fwhm_stderr,
        "r_squared": _finite(r2),
        "rmse": _finite(rmse),
    }


def _ols_line_with_stderr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Exact ordinary least squares fit ``y = intercept + slope * x`` with stderr and R^2."""
    n = int(x.size)
    X = np.column_stack((np.ones(n, dtype=float), x))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    pred = X @ beta
    resid = y - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1.0e-30 else (1.0 if sse <= 1.0e-30 else 0.0)
    rmse = float(math.sqrt(sse / float(n)))

    intercept_se = None
    slope_se = None
    if n > 2:
        s2 = sse / float(n - 2)
        cov = np.linalg.pinv(X.T @ X) * s2
        if cov[0, 0] >= 0.0:
            intercept_se = _finite(math.sqrt(float(cov[0, 0])))
        if cov[1, 1] >= 0.0:
            slope_se = _finite(math.sqrt(float(cov[1, 1])))
    return {
        "intercept": intercept,
        "slope": slope,
        "intercept_stderr": intercept_se,
        "slope_stderr": slope_se,
        "r_squared": _finite(r2),
        "rmse": _finite(rmse),
    }


def _fit_streak_width_models(
    q_vals: np.ndarray,
    fwhm_vals: np.ndarray,
    *,
    q_scale_nm_inv: float | None,
    q_unit_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit linear and quadrature Grubb/Williamson-Hall models to ``fwhm(q)``."""
    n = int(q_vals.size)
    if n < 3 or float(np.ptp(q_vals)) <= 1.0e-12:
        unavail = {
            "status": "unavailable",
            "valid": False,
            "dq_perp_0": None,
            "dq_perp_0_stderr": None,
            "mosaic_spread_rad": None,
            "mosaic_spread_deg": None,
            "mosaic_spread_stderr_deg": None,
            "W_nm": None,
            "W_stderr_nm": None,
            "r_squared": None,
            "rmse": None,
            "flags": ["insufficient_valid_slices"],
        }
        return dict(unavail), dict(unavail)

    # 1. Linear model: dq_perp(q) = dq_perp_0 + delta_alpha_rad * q
    lin = _ols_line_with_stderr(q_vals, fwhm_vals)
    lin_flags: list[str] = []
    dq0_lin = lin["intercept"]
    slope_lin = lin["slope"]
    dq0_lin_se = lin["intercept_stderr"]
    slope_lin_se = lin["slope_stderr"]

    mosaic_deg_lin: float | None = None
    mosaic_se_deg_lin: float | None = None
    if slope_lin >= 0.0:
        mosaic_deg_lin = _finite(math.degrees(slope_lin))
        mosaic_se_deg_lin = _finite(math.degrees(slope_lin_se)) if slope_lin_se is not None else None
    else:
        lin_flags.append("negative_mosaic_slope")

    w_lin_nm: float | None = None
    w_lin_se_nm: float | None = None
    if dq0_lin <= 0.0:
        lin_flags.append("nonpositive_intercept_W_unbounded")
    elif q_scale_nm_inv is None:
        lin_flags.append("uncalibrated_pixel_q" if q_unit_label == "pixel-q" else "spacing_unavailable_unknown_q_unit")
    else:
        dq0_nm_inv = dq0_lin * q_scale_nm_inv
        w_lin_nm = _finite(2.0 * math.pi / dq0_nm_inv)
        if w_lin_nm is not None and dq0_lin_se is not None:
            w_lin_se_nm = _finite(w_lin_nm * (dq0_lin_se / dq0_lin))

    lin_valid = bool(dq0_lin > 0.0 and slope_lin >= 0.0)
    linear_result = {
        "status": "estimated" if lin_valid else "warning",
        "valid": lin_valid,
        "dq_perp_0": _finite(dq0_lin),
        "dq_perp_0_stderr": dq0_lin_se,
        "mosaic_spread_rad": _finite(slope_lin),
        "mosaic_spread_deg": mosaic_deg_lin,
        "mosaic_spread_stderr_deg": mosaic_se_deg_lin,
        "W_nm": w_lin_nm,
        "W_stderr_nm": w_lin_se_nm,
        "r_squared": lin["r_squared"],
        "rmse": lin["rmse"],
        "flags": lin_flags,
    }

    # 2. Quadrature model: dq_perp(q)^2 = dq_perp_0^2 + (delta_alpha_rad * q)^2
    quad_ols = _ols_line_with_stderr(q_vals ** 2, fwhm_vals ** 2)
    quad_flags: list[str] = []
    c0_q = quad_ols["intercept"]
    c1_q = quad_ols["slope"]
    c0_se = quad_ols["intercept_stderr"]
    c1_se = quad_ols["slope_stderr"]

    dq0_quad: float | None = None
    dq0_quad_se: float | None = None
    if c0_q > 0.0:
        dq0_quad = math.sqrt(c0_q)
        if c0_se is not None:
            dq0_quad_se = _finite(c0_se / (2.0 * dq0_quad))
    else:
        quad_flags.append("nonpositive_quadrature_intercept_W_unbounded")

    mosaic_rad_quad: float | None = None
    mosaic_deg_quad: float | None = None
    mosaic_se_deg_quad: float | None = None
    if c1_q >= 0.0:
        mosaic_rad_quad = math.sqrt(c1_q)
        mosaic_deg_quad = _finite(math.degrees(mosaic_rad_quad))
        if c1_se is not None and mosaic_rad_quad > 1.0e-12:
            mosaic_se_deg_quad = _finite(math.degrees(c1_se / (2.0 * mosaic_rad_quad)))
    else:
        quad_flags.append("negative_quadrature_mosaic_slope")

    w_quad_nm: float | None = None
    w_quad_se_nm: float | None = None
    if dq0_quad is not None and dq0_quad > 0.0:
        if q_scale_nm_inv is None:
            quad_flags.append("uncalibrated_pixel_q" if q_unit_label == "pixel-q" else "spacing_unavailable_unknown_q_unit")
        else:
            dq0_nm_inv = dq0_quad * q_scale_nm_inv
            w_quad_nm = _finite(2.0 * math.pi / dq0_nm_inv)
            if w_quad_nm is not None and dq0_quad_se is not None:
                w_quad_se_nm = _finite(w_quad_nm * (dq0_quad_se / dq0_quad))

    # Compute R^2 and RMSE in FWHM space as well as squared space
    quad_r2 = quad_ols["r_squared"]
    quad_rmse = quad_ols["rmse"]
    if dq0_quad is not None and mosaic_rad_quad is not None:
        pred_fwhm = np.sqrt(dq0_quad ** 2 + (mosaic_rad_quad * q_vals) ** 2)
        resid_fwhm = fwhm_vals - pred_fwhm
        sse_f = float(np.sum(resid_fwhm ** 2))
        sst_f = float(np.sum((fwhm_vals - float(np.mean(fwhm_vals))) ** 2))
        quad_r2 = _finite(1.0 - sse_f / sst_f) if sst_f > 1.0e-30 else (1.0 if sse_f <= 1.0e-30 else 0.0)
        quad_rmse = _finite(math.sqrt(sse_f / float(n)))

    quad_valid = bool(dq0_quad is not None and dq0_quad > 0.0 and mosaic_rad_quad is not None)
    quadrature_result = {
        "status": "estimated" if quad_valid else "warning",
        "valid": quad_valid,
        "dq_perp_0": _finite(dq0_quad),
        "dq_perp_0_stderr": dq0_quad_se,
        "dq_perp_0_squared": _finite(c0_q),
        "mosaic_spread_rad": _finite(mosaic_rad_quad),
        "mosaic_spread_deg": mosaic_deg_quad,
        "mosaic_spread_stderr_deg": mosaic_se_deg_quad,
        "W_nm": w_quad_nm,
        "W_stderr_nm": w_quad_se_nm,
        "r_squared": quad_r2,
        "r_squared_in_q2": quad_ols["r_squared"],
        "rmse": quad_rmse,
        "flags": quad_flags,
    }
    return linear_result, quadrature_result


def measure_perpendicular_streak_width(
    image: Any,
    qx: Any = None,
    qy: Any = None,
    *,
    valid_mask: Any = None,
    q_window: tuple[float, float],
    q_unit: str = "unknown",
    wing_angles_deg: Sequence[float] | None = None,
    n_q_slices: int = 12,
    transverse_half_angle_deg: float = 20.0,
    min_q_perp_half_width: float | None = None,
    n_perp_bins: int | None = None,
    min_slice_pixels: int = 12,
    min_slice_r2: float = 0.5,
    nominal_draw_axis_deg: float = 90.0,
    geometry_cache: Any = None,
    intensity_mode: str = "calibrated",
    sigma: Any = None,
) -> dict[str, Any]:
    """Measure transverse butterfly streak width ``dq_perp(q)`` along detected wing rays.

    Separates intrinsic angular mosaic spread ``delta_alpha`` (deg) from apparent
    lateral coherence length ``W = 2*pi / dq_perp(0)`` (nm, calibrated physical q
    only) using both linear (Williamson-Hall) and quadrature (Gaussian-convolution)
    fits on strictly observed transverse slices.
    """
    _normalise_intensity_mode(intensity_mode)
    q_min, q_max = _validate_q_window(q_window)
    if isinstance(n_q_slices, (bool, np.bool_)) or int(n_q_slices) < 3:
        raise ValueError("n_q_slices must be an integer >= 3")
    n_slices = int(n_q_slices)
    half_ang = float(transverse_half_angle_deg)
    if not math.isfinite(half_ang) or not (1.0 <= half_ang <= 60.0):
        raise ValueError("transverse_half_angle_deg must be finite and in [1, 60]")

    # Conservative radial upper bound covering all transverse slices before resolving ROI
    tan_half_max = math.tan(math.radians(half_ang))
    q_perp_prelim = q_max * tan_half_max
    if min_q_perp_half_width is not None and math.isfinite(float(min_q_perp_half_width)):
        q_perp_prelim = max(q_perp_prelim, float(min_q_perp_half_width))
    q_roi_prelim = math.hypot(q_max, max(q_perp_prelim, 0.25 * q_max))

    img, qx_arr, qy_arr, q_mag, chi_deg, geom_mask, inclusion, _, pixel_dq_hint = _resolve_2d_context(
        image,
        qx,
        qy,
        valid_mask,
        geometry_cache=geometry_cache,
        sigma=sigma,
        required_q_max=q_roi_prelim,
    )
    q_unit_label, q_scale_nm_inv, unit_caveat = _q_unit_info(q_unit)

    if wing_angles_deg is not None:
        rays = [float(a) % 360.0 for a in wing_angles_deg if math.isfinite(float(a))]
        ray_source = "user_supplied"
    else:
        rays, ray_source = _detect_wing_angles_for_sectors(
            img,
            qx_arr,
            qy_arr,
            geom_mask,
            None,
            q_window=(q_min, q_max),
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            q_mag=q_mag,
            chi_deg=chi_deg,
            inclusion=inclusion,
        )

    caveats = [
        "Apparent lateral coherence length W = 2*pi/dq_perp(0) is a model-dependent "
        "Grubb/Ruland/Williamson-Hall streak-width estimate affected by instrumental "
        "resolution, q-window range, and profile shape; it is not a unique 3D reconstruction."
    ]
    if unit_caveat is not None:
        caveats.append(unit_caveat)

    if not rays:
        return json_safe({
            "status": "unavailable",
            "reason": ray_source,
            "q_unit": q_unit_label,
            "q_window": [q_min, q_max],
            "wing_angle_source": ray_source,
            "wings": [],
            "summary": {
                "n_wings_analyzed": 0,
                "n_valid_wings_linear": 0,
                "n_valid_wings_quadrature": 0,
                "total_valid_slices": 0,
                "linear": {"dq_perp_0": None, "mosaic_spread_deg": None, "W_nm": None},
                "quadrature": {"dq_perp_0": None, "mosaic_spread_deg": None, "W_nm": None},
            },
            "flags": ["no_butterfly_wings_detected"],
            "caveats": caveats,
        })

    # Fast pixel reciprocal spacing estimate (reuse cache or strided subsample; avoids 2.48M-px nanmedian)
    if pixel_dq_hint is not None and math.isfinite(float(pixel_dq_hint)) and float(pixel_dq_hint) > 0.0:
        pixel_dq = float(pixel_dq_hint)
    else:
        stride_y = max(1, qx_arr.shape[0] // 64)
        stride_x = max(1, qx_arr.shape[1] // 64)
        dx_med = (
            float(np.nanmedian(np.abs(np.diff(qx_arr[::stride_y, :], axis=1))))
            if qx_arr.shape[1] > 1
            else 0.0
        )
        dy_med = (
            float(np.nanmedian(np.abs(np.diff(qy_arr[:, ::stride_x], axis=0))))
            if qy_arr.shape[0] > 1
            else 0.0
        )
        pixel_dq = max(
            min(v for v in (dx_med, dy_med) if v > 0.0)
            if (dx_med > 0.0 or dy_med > 0.0)
            else (q_max - q_min) / 100.0,
            1.0e-9,
        )

    slice_edges = np.linspace(q_min, q_max, n_slices + 1, dtype=float)
    slice_centers = 0.5 * (slice_edges[:-1] + slice_edges[1:])
    slice_half_w = 0.5 * (slice_edges[1:] - slice_edges[:-1])

    # Exact radial ROI upper bound covering every slice of every wing
    q_perp_upper = max(
        q_max * tan_half_max,
        float(min_q_perp_half_width)
        if (min_q_perp_half_width is not None and math.isfinite(float(min_q_perp_half_width)))
        else 6.0 * pixel_dq,
    )
    q_roi_upper = math.hypot(q_max, q_perp_upper)

    if q_mag is None:
        in_box = (np.abs(qx_arr) <= q_roi_upper) & (np.abs(qy_arr) <= q_roi_upper)
        if not np.any(in_box):
            if not np.any(np.isfinite(qx_arr) & np.isfinite(qy_arr)):
                raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
            obs_qx = np.empty(0, dtype=float)
            obs_qy = np.empty(0, dtype=float)
            obs_img = np.empty(0, dtype=float)
        else:
            bx = qx_arr[in_box]
            by = qy_arr[in_box]
            bimg = img[in_box]
            binc = inclusion[in_box]
            bq = np.hypot(bx, by)
            in_ring = np.isfinite(bq) & (bq >= q_min) & (bq <= q_roi_upper) & binc & np.isfinite(bimg)
            obs_qx = bx[in_ring]
            obs_qy = by[in_ring]
            obs_img = bimg[in_ring]
    else:
        assert geom_mask is not None
        in_ring = geom_mask & inclusion & np.isfinite(img) & (q_mag >= q_min) & (q_mag <= q_roi_upper)
        obs_qx = qx_arr[in_ring]
        obs_qy = qy_arr[in_ring]
        obs_img = img[in_ring]

    wings_out: list[dict[str, Any]] = []
    all_flags: list[str] = []
    if q_scale_nm_inv is None:
        all_flags.append("uncalibrated_pixel_q" if q_unit_label == "pixel-q" else "spacing_unavailable_unknown_q_unit")

    for wing_idx, theta_deg in enumerate(rays):
        theta_rad = math.radians(theta_deg)
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
        q_par = obs_qx * cos_t + obs_qy * sin_t
        q_perp = -obs_qx * sin_t + obs_qy * cos_t

        # Determine nearest neighboring wing to avoid cross-wing contamination
        other_seps = [
            abs(float(_periodic_dist_deg(other, theta_deg, 360.0)))
            for j, other in enumerate(rays)
            if j != wing_idx
        ]
        min_neighbor_sep = min(other_seps) if other_seps else 180.0
        safe_half_ang = min(half_ang, max(5.0, 0.42 * min_neighbor_sep))
        tan_half = math.tan(math.radians(safe_half_ang))
        tan_neighbor_cap = math.tan(math.radians(max(6.0, 0.46 * min_neighbor_sep)))

        # Precompute per-slice q_perp_max and filter & sort once per wing (replaces 12x full scans)
        slice_q_perp_max = np.empty(n_slices, dtype=float)
        for s_idx in range(n_slices):
            qc = float(slice_centers[s_idx])
            qpm = qc * tan_half
            if min_q_perp_half_width is not None and math.isfinite(float(min_q_perp_half_width)):
                qpm = max(qpm, float(min_q_perp_half_width))
            else:
                qpm = max(qpm, 6.0 * pixel_dq)
            if min_neighbor_sep < 160.0:
                qpm = min(qpm, max(qc * tan_neighbor_cap, 4.0 * pixel_dq))
            slice_q_perp_max[s_idx] = qpm

        wing_q_perp_cap = float(np.max(slice_q_perp_max)) if n_slices > 0 else q_perp_upper
        in_wing = (q_par >= q_min) & (q_par <= q_max) & (np.abs(q_perp) <= wing_q_perp_cap)
        if np.any(in_wing):
            qp_w = q_par[in_wing]
            qt_w = q_perp[in_wing]
            im_w = obs_img[in_wing]
            order_w = np.argsort(qp_w)
            qp_s = qp_w[order_w]
            qt_s = qt_w[order_w]
            im_s = im_w[order_w]
        else:
            qp_s = np.empty(0, dtype=float)
            qt_s = np.empty(0, dtype=float)
            im_s = np.empty(0, dtype=float)

        slices_out: list[dict[str, Any]] = []
        valid_q_list: list[float] = []
        valid_fwhm_list: list[float] = []

        for s_idx in range(n_slices):
            qc = float(slice_centers[s_idx])
            hw_par = float(slice_half_w[s_idx])
            q_perp_max = float(slice_q_perp_max[s_idx])

            lo_i = int(np.searchsorted(qp_s, qc - hw_par, side="left"))
            hi_i = int(np.searchsorted(qp_s, qc + hw_par, side="right"))
            if hi_i > lo_i:
                qt_seg = qt_s[lo_i:hi_i]
                im_seg = im_s[lo_i:hi_i]
                sel_perp = np.abs(qt_seg) <= q_perp_max
                p_vals = qt_seg[sel_perp]
                i_vals = im_seg[sel_perp]
                n_px = int(p_vals.size)
            else:
                p_vals = np.empty(0, dtype=float)
                i_vals = np.empty(0, dtype=float)
                n_px = 0

            if n_px < int(min_slice_pixels):
                slices_out.append({
                    "q_center": _finite(qc),
                    "q_half_width": _finite(hw_par),
                    "q_perp_max": _finite(q_perp_max),
                    "n_pixels": n_px,
                    "valid": False,
                    "reason": "insufficient_pixels",
                    "fwhm_q_perp": None,
                    "sigma_q_perp": None,
                    "r_squared": None,
                })
                continue

            if n_perp_bins is not None:
                nb = max(int(n_perp_bins), 11)
            else:
                nb = int(np.clip(round((2.0 * q_perp_max) / max(1.1 * pixel_dq, 1.0e-9)), 15, 61))
                if nb % 2 == 0:
                    nb += 1

            p_edges = np.linspace(-q_perp_max, q_perp_max, nb + 1, dtype=float)
            p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
            p_idx = np.floor((p_vals + q_perp_max) * (float(nb) / max(2.0 * q_perp_max, 1.0e-30))).astype(np.int64)
            np.clip(p_idx, 0, nb - 1, out=p_idx)
            p_cnt = np.bincount(p_idx, minlength=nb).astype(np.int64)
            p_sum = np.bincount(p_idx, weights=i_vals, minlength=nb).astype(float)
            p_mean = np.full(nb, np.nan, dtype=float)
            np.divide(p_sum, p_cnt, out=p_mean, where=p_cnt > 0)

            sfit = _fit_1d_transverse_gaussian(
                p_centers, p_mean, p_cnt, q_perp_max=q_perp_max, min_r2=float(min_slice_r2)
            )
            slice_record = {
                "q_center": _finite(qc),
                "q_half_width": _finite(hw_par),
                "q_perp_max": _finite(q_perp_max),
                "n_pixels": n_px,
                **sfit,
            }
            slices_out.append(slice_record)
            if sfit["valid"] and sfit.get("fwhm_q_perp") is not None:
                valid_q_list.append(qc)
                valid_fwhm_list.append(float(sfit["fwhm_q_perp"]))

        q_arr = np.asarray(valid_q_list, dtype=float)
        fwhm_arr = np.asarray(valid_fwhm_list, dtype=float)
        linear_fit, quad_fit = _fit_streak_width_models(
            q_arr, fwhm_arr, q_scale_nm_inv=q_scale_nm_inv, q_unit_label=q_unit_label
        )
        wing_flags = list(dict.fromkeys([*linear_fit["flags"], *quad_fit["flags"]]))
        if len(valid_q_list) < n_slices:
            wing_flags.append("partial_slice_validity")
        all_flags.extend(wing_flags)

        wings_out.append({
            "wing_index": wing_idx,
            "wing_angle_deg": _finite(theta_deg),
            "n_total_slices": n_slices,
            "n_valid_slices": len(valid_q_list),
            "valid_q": [_finite(v) for v in valid_q_list],
            "valid_fwhm_q_perp": [_finite(v) for v in valid_fwhm_list],
            "linear_fit": linear_fit,
            "quadrature_fit": quad_fit,
            "slices": slices_out,
            "flags": wing_flags,
        })

    # Aggregate median estimates across valid wings
    def _agg(model_key: str) -> dict[str, Any]:
        dq0s = [w[model_key]["dq_perp_0"] for w in wings_out if w[model_key]["valid"] and w[model_key]["dq_perp_0"] is not None]
        mosaics = [w[model_key]["mosaic_spread_deg"] for w in wings_out if w[model_key]["valid"] and w[model_key]["mosaic_spread_deg"] is not None]
        ws = [w[model_key]["W_nm"] for w in wings_out if w[model_key]["valid"] and w[model_key]["W_nm"] is not None]
        r2s = [w[model_key]["r_squared"] for w in wings_out if w[model_key]["valid"] and w[model_key]["r_squared"] is not None]
        return {
            "n_valid_wings": len(dq0s),
            "dq_perp_0": _finite(float(np.median(dq0s))) if dq0s else None,
            "mosaic_spread_deg": _finite(float(np.median(mosaics))) if mosaics else None,
            "W_nm": _finite(float(np.median(ws))) if ws else None,
            "median_r_squared": _finite(float(np.median(r2s))) if r2s else None,
        }

    lin_agg = _agg("linear_fit")
    quad_agg = _agg("quadrature_fit")
    total_valid = sum(w["n_valid_slices"] for w in wings_out)
    overall_status = (
        "estimated"
        if (lin_agg["n_valid_wings"] > 0 or quad_agg["n_valid_wings"] > 0)
        else ("warning" if total_valid > 0 else "unavailable")
    )

    return json_safe({
        "status": overall_status,
        "q_unit": q_unit_label,
        "q_window": [q_min, q_max],
        "wing_angle_source": ray_source,
        "wings": wings_out,
        "summary": {
            "n_wings_analyzed": len(wings_out),
            "n_valid_wings_linear": lin_agg["n_valid_wings"],
            "n_valid_wings_quadrature": quad_agg["n_valid_wings"],
            "total_valid_slices": total_valid,
            "linear": lin_agg,
            "quadrature": quad_agg,
        },
        "flags": list(dict.fromkeys(all_flags)),
        "caveats": caveats,
    })


# ---------------------------------------------------------------------------
# 5. Kinetics & Thermal Scaling Helpers
# ---------------------------------------------------------------------------


def _resolve_1d_weights(
    n: int,
    sigma: Any,
    weights: Any,
) -> tuple[np.ndarray, bool]:
    """Return ``(sqrt_weights, has_absolute_sigma)`` after validating inputs."""
    if sigma is not None and weights is not None:
        raise ValueError("provide either sigma or weights, not both")
    if sigma is not None:
        s = np.asarray(sigma, dtype=float)
        if s.shape != (n,):
            raise ValueError("sigma must match data shape")
        if np.any(~np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError("sigma values must be finite and strictly positive")
        return 1.0 / s, True
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must match data shape")
        if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError("weights must be finite and strictly positive")
        return np.sqrt(w), False
    return np.ones(n, dtype=float), False


def fit_power_law_coarsening(
    t: Any,
    L: Any,
    *,
    sigma_L: Any = None,
    weights: Any = None,
    model: str = "power_law",
    t_min: float | None = None,
    t_max: float | None = None,
    confidence_level: float = 0.95,
    fix_L0: float | None = None,
) -> dict[str, Any]:
    """Fit power-law coarsening kinetics ``L(t) = A * t^n`` or ``L(t) = L0 + k * t^n``.

    Supports weighted (``sigma_L`` or ``weights``) and unweighted regression,
    strictly omitting non-finite samples and returning parameter standard errors
    and Student-t confidence intervals.
    """
    t_arr = np.asarray(t, dtype=float)
    L_arr = np.asarray(L, dtype=float)
    if t_arr.ndim != 1 or L_arr.ndim != 1 or t_arr.shape != L_arr.shape or t_arr.size == 0:
        raise ValueError("t and L must be non-empty matching 1D arrays")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must lie in (0, 1)")

    norm_model = str(model or "power_law").strip().lower().replace("-", "_")
    if norm_model in {"power_law", "pure", "at_n", "a_t_n"}:
        model_kind = "power_law"
    elif norm_model in {"offset_power_law", "offset", "l0_kt_n", "l0_k_t_n", "lsw"}:
        model_kind = "offset_power_law"
    else:
        raise ValueError("model must be 'power_law' (A*t^n) or 'offset_power_law' (L0 + k*t^n)")

    mask = np.isfinite(t_arr) & np.isfinite(L_arr)
    if t_min is not None:
        mask &= t_arr >= float(t_min)
    if t_max is not None:
        mask &= t_arr <= float(t_max)
    if model_kind == "power_law":
        mask &= (t_arr > 0.0) & (L_arr > 0.0)
    else:
        mask &= t_arr >= 0.0

    # Validate weights/sigma on selected finite entries
    idx_sel = np.flatnonzero(mask)
    if sigma_L is not None and weights is not None:
        raise ValueError("provide either sigma_L or weights, not both")
    sig_sel = None
    w_sel = None
    if sigma_L is not None:
        s_full = np.asarray(sigma_L, dtype=float)
        if s_full.shape != t_arr.shape:
            raise ValueError("sigma_L must match t and L shape")
        sig_sel = s_full[idx_sel]
    if weights is not None:
        w_full = np.asarray(weights, dtype=float)
        if w_full.shape != t_arr.shape:
            raise ValueError("weights must match t and L shape")
        w_sel = w_full[idx_sel]

    t_use = t_arr[idx_sel]
    L_use = L_arr[idx_sel]
    n_pts = int(t_use.size)
    n_params = 2 if (model_kind == "power_law" or fix_L0 is not None) else 3
    if n_pts < n_params:
        return json_safe({
            "success": False,
            "status": "unavailable",
            "model": model_kind,
            "message": f"Need at least {n_params} valid points for {model_kind}; found {n_pts}.",
            "n_points": n_pts,
            "dof": 0,
            "parameters": None,
            "stderr": None,
            "ci": None,
            "r_squared": None,
            "rmse": None,
            "weighted_rmse": None,
            "reduced_chi_square": None,
            "flags": ["insufficient_points_for_coarsening_fit"],
        })

    sw, abs_sigma = _resolve_1d_weights(n_pts, sig_sel, w_sel)
    flags: list[str] = []

    if model_kind == "power_law":
        param_names = ("A", "n")
        # Weighted log-log linear seed
        lx = np.log(t_use)
        ly = np.log(L_use)
        X_ll = np.column_stack((np.ones(n_pts, dtype=float), lx))
        w_ll = sw * L_use  # delta-method weight for log(L)
        b_ll, *_ = np.linalg.lstsq(X_ll * w_ll[:, None], ly * w_ll, rcond=None)
        A0 = max(math.exp(float(np.clip(b_ll[0], -50.0, 50.0))), 1.0e-12)
        n0 = float(np.clip(b_ll[1], -5.0, 5.0))
        p0 = np.array([A0, n0], dtype=float)
        lo = np.array([1.0e-15, -10.0], dtype=float)
        hi = np.array([np.inf, 10.0], dtype=float)

        def _eval(p: np.ndarray, tv: np.ndarray) -> np.ndarray:
            return p[0] * np.power(tv, p[1])

        fit = least_squares(lambda p: sw * (_eval(p, t_use) - L_use), p0, bounds=(lo, hi), max_nfev=4000)
    else:
        # Offset power-law: L(t) = L0 + k * t^n
        fixed_l0 = float(fix_L0) if fix_L0 is not None else None
        if fixed_l0 is not None and not math.isfinite(fixed_l0):
            raise ValueError("fix_L0 must be finite when provided")

        best_fit = None
        best_cost = float("inf")
        n_grid = (0.15, 0.25, 1.0 / 3.0, 0.5, 0.75, 1.0)

        if fixed_l0 is not None:
            param_names = ("k", "n")

            def _eval(p: np.ndarray, tv: np.ndarray) -> np.ndarray:
                return fixed_l0 + p[0] * np.power(tv, p[1])

            for n_seed in n_grid:
                tn = np.power(t_use, n_seed)
                denom = float(np.sum((sw * tn) ** 2))
                k_seed = float(np.sum((sw ** 2) * tn * (L_use - fixed_l0)) / denom) if denom > 0.0 else 1.0
                k_seed = max(k_seed, 1.0e-9)
                p0 = np.array([k_seed, n_seed], dtype=float)
                res = least_squares(
                    lambda p: sw * (_eval(p, t_use) - L_use),
                    p0,
                    bounds=([0.0, 1.0e-3], [np.inf, 5.0]),
                    max_nfev=4000,
                )
                if math.isfinite(res.cost) and res.cost < best_cost:
                    best_cost = float(res.cost)
                    best_fit = res
        else:
            param_names = ("L0", "k", "n")

            def _eval(p: np.ndarray, tv: np.ndarray) -> np.ndarray:
                return p[0] + p[1] * np.power(tv, p[2])

            for n_seed in n_grid:
                tn = np.power(t_use, n_seed)
                X_seed = np.column_stack((np.ones(n_pts, dtype=float), tn))
                beta_s, *_ = np.linalg.lstsq(X_seed * sw[:, None], L_use * sw, rcond=None)
                l0_s = max(float(beta_s[0]), 0.0)
                k_s = max(float(beta_s[1]), 1.0e-9)
                p0 = np.array([l0_s, k_s, n_seed], dtype=float)
                res = least_squares(
                    lambda p: sw * (_eval(p, t_use) - L_use),
                    p0,
                    bounds=([0.0, 0.0, 1.0e-3], [np.inf, np.inf, 5.0]),
                    max_nfev=4000,
                )
                if math.isfinite(res.cost) and res.cost < best_cost:
                    best_cost = float(res.cost)
                    best_fit = res
        assert best_fit is not None
        fit = best_fit

    pred = _eval(fit.x, t_use)
    raw_resid = L_use - pred
    w_resid = sw * raw_resid
    sse_raw = float(np.sum(raw_resid ** 2))
    sse_w = float(np.sum(w_resid ** 2))
    rmse = float(math.sqrt(sse_raw / float(n_pts)))
    w_rmse = float(math.sqrt(sse_w / float(n_pts)))
    sst_raw = float(np.sum((L_use - float(np.mean(L_use))) ** 2))
    r2 = float(1.0 - sse_raw / sst_raw) if sst_raw > 1.0e-30 else (1.0 if sse_raw <= 1.0e-30 else None)

    dof = n_pts - n_params
    red_chi2 = _finite(sse_w / float(dof)) if dof > 0 else None

    params_dict: dict[str, float | None] = {name: _finite(val) for name, val in zip(param_names, fit.x)}
    if model_kind == "offset_power_law" and fix_L0 is not None:
        params_dict["L0"] = _finite(fix_L0)

    stderr_dict: dict[str, float | None] = {name: None for name in param_names}
    ci_dict: dict[str, list[float | None] | None] = {name: None for name in param_names}

    if dof > 0 and fit.jac is not None:
        try:
            jtj = fit.jac.T @ fit.jac
            scale_var = 1.0 if (abs_sigma and (red_chi2 is None or red_chi2 <= 1.0)) else (sse_w / float(dof))
            cov = np.linalg.pinv(jtj) * scale_var
            t_crit = float(student_t.ppf(0.5 * (1.0 + float(confidence_level)), df=dof))
            for i, name in enumerate(param_names):
                var_i = float(cov[i, i])
                if math.isfinite(var_i) and var_i >= 0.0:
                    se_i = math.sqrt(var_i)
                    stderr_dict[name] = _finite(se_i)
                    val_i = float(fit.x[i])
                    ci_dict[name] = [_finite(val_i - t_crit * se_i), _finite(val_i + t_crit * se_i)]
        except np.linalg.LinAlgError:
            flags.append("covariance_estimation_failed")
    else:
        flags.append("zero_dof_no_uncertainty_interval")

    if not fit.success:
        flags.append("optimizer_did_not_converge")

    return json_safe({
        "success": bool(fit.success),
        "status": "estimated" if fit.success else "warning",
        "model": model_kind,
        "message": str(fit.message),
        "n_points": n_pts,
        "dof": dof,
        "confidence_level": float(confidence_level),
        "parameters": params_dict,
        "stderr": stderr_dict,
        "ci": ci_dict,
        "r_squared": _finite(r2),
        "rmse": _finite(rmse),
        "weighted_rmse": _finite(w_rmse),
        "reduced_chi_square": red_chi2,
        "t_used": _finite_list(t_use),
        "L_used": _finite_list(L_use),
        "fitted_values": _finite_list(pred),
        "residuals": _finite_list(raw_resid),
        "flags": list(dict.fromkeys(flags)),
    })


def fit_arrhenius_rate(
    temperatures: Any,
    rates: Any,
    *,
    temperature_unit: str = "K",
    sigma_rates: Any = None,
    weights: Any = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Fit Arrhenius relation ``ln(k) = ln(A) - Ea / (R * T)`` returning ``Ea`` in kJ/mol."""
    T_raw = np.asarray(temperatures, dtype=float)
    k_raw = np.asarray(rates, dtype=float)
    if T_raw.ndim != 1 or k_raw.ndim != 1 or T_raw.shape != k_raw.shape or T_raw.size == 0:
        raise ValueError("temperatures and rates must be non-empty matching 1D arrays")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must lie in (0, 1)")

    unit_norm = str(temperature_unit or "K").strip().lower().replace("°", "")
    if unit_norm in {"k", "kelvin"}:
        T_kelvin = T_raw.copy()
    elif unit_norm in {"c", "degc", "celsius"}:
        T_kelvin = T_raw + 273.15
    else:
        raise ValueError("temperature_unit must be 'K' or 'C'")

    mask = np.isfinite(T_kelvin) & (T_kelvin > 0.0) & np.isfinite(k_raw) & (k_raw > 0.0)
    idx_sel = np.flatnonzero(mask)
    if sigma_rates is not None and weights is not None:
        raise ValueError("provide either sigma_rates or weights, not both")

    T_use = T_kelvin[idx_sel]
    k_use = k_raw[idx_sel]
    n_pts = int(T_use.size)
    if n_pts < 2 or float(np.ptp(T_use)) <= 1.0e-12:
        return json_safe({
            "success": False,
            "status": "unavailable",
            "message": f"Need at least 2 distinct positive temperatures with positive rates; found {n_pts}.",
            "n_points": n_pts,
            "dof": 0,
            "Ea_kJ_mol": None,
            "Ea_stderr_kJ_mol": None,
            "Ea_ci_kJ_mol": None,
            "Ea_eV": None,
            "ln_A": None,
            "ln_A_stderr": None,
            "A_prefactor": None,
            "r_squared": None,
            "rmse_ln": None,
            "flags": ["insufficient_points_for_arrhenius_fit"],
        })

    sig_ln = None
    w_ln = None
    if sigma_rates is not None:
        s_full = np.asarray(sigma_rates, dtype=float)
        if s_full.shape != T_raw.shape:
            raise ValueError("sigma_rates must match temperatures shape")
        s_sel = s_full[idx_sel]
        if np.any(~np.isfinite(s_sel)) or np.any(s_sel <= 0.0):
            raise ValueError("sigma_rates must be finite and strictly positive")
        sig_ln = s_sel / k_use  # first-order error propagation for ln(k)
    if weights is not None:
        w_full = np.asarray(weights, dtype=float)
        if w_full.shape != T_raw.shape:
            raise ValueError("weights must match temperatures shape")
        w_ln = w_full[idx_sel]

    sw, abs_sigma = _resolve_1d_weights(n_pts, sig_ln, w_ln)

    # x = 1 / (R * T) in mol/kJ, y = ln(k) => y = ln_A - Ea * x
    x = 1.0 / (_GAS_CONSTANT_KJ_MOL_K * T_use)
    y = np.log(k_use)
    X = np.column_stack((np.ones(n_pts, dtype=float), -x))
    Xw = X * sw[:, None]
    yw = y * sw

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    ln_A = float(beta[0])
    Ea_kJ = float(beta[1])
    pred_ln = X @ beta
    resid_ln = y - pred_ln
    sse_ln = float(np.sum(resid_ln ** 2))
    sse_w = float(np.sum((sw * resid_ln) ** 2))
    sst_ln = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse_ln / sst_ln) if sst_ln > 1.0e-30 else (1.0 if sse_ln <= 1.0e-30 else None)
    rmse_ln = float(math.sqrt(sse_ln / float(n_pts)))

    dof = n_pts - 2
    flags: list[str] = []
    Ea_se: float | None = None
    ln_A_se: float | None = None
    Ea_ci: list[float | None] | None = None
    ln_A_ci: list[float | None] | None = None

    if dof > 0:
        red_chi2 = sse_w / float(dof)
        scale_var = 1.0 if (abs_sigma and red_chi2 <= 1.0) else red_chi2
        cov = np.linalg.pinv(Xw.T @ Xw) * scale_var
        t_crit = float(student_t.ppf(0.5 * (1.0 + float(confidence_level)), df=dof))
        if cov[0, 0] >= 0.0:
            ln_A_se = _finite(math.sqrt(float(cov[0, 0])))
            if ln_A_se is not None:
                ln_A_ci = [_finite(ln_A - t_crit * ln_A_se), _finite(ln_A + t_crit * ln_A_se)]
        if cov[1, 1] >= 0.0:
            Ea_se = _finite(math.sqrt(float(cov[1, 1])))
            if Ea_se is not None:
                Ea_ci = [_finite(Ea_kJ - t_crit * Ea_se), _finite(Ea_kJ + t_crit * Ea_se)]
    else:
        flags.append("zero_dof_exact_two_point_fit")

    if Ea_kJ < 0.0:
        flags.append("negative_apparent_activation_energy")

    A_prefactor = _finite(math.exp(ln_A)) if abs(ln_A) < 700.0 else None

    return json_safe({
        "success": True,
        "status": "estimated",
        "n_points": n_pts,
        "dof": dof,
        "confidence_level": float(confidence_level),
        "Ea_kJ_mol": _finite(Ea_kJ),
        "Ea_stderr_kJ_mol": Ea_se,
        "Ea_ci_kJ_mol": Ea_ci,
        "Ea_eV": _finite(Ea_kJ / _KJ_MOL_PER_EV),
        "ln_A": _finite(ln_A),
        "ln_A_stderr": ln_A_se,
        "ln_A_ci": ln_A_ci,
        "A_prefactor": A_prefactor,
        "r_squared": _finite(r2),
        "rmse_ln": _finite(rmse_ln),
        "temperatures_K": _finite_list(T_use),
        "inv_T_K_inv": _finite_list(1.0 / T_use),
        "inv_RT_mol_per_kJ": _finite_list(x),
        "ln_rates": _finite_list(y),
        "fitted_ln_rates": _finite_list(pred_ln),
        "residuals_ln": _finite_list(resid_ln),
        "flags": flags,
        "caveat": (
            "Apparent activation energy Ea from Arrhenius regression of ln(rate) vs 1/(R*T) "
            "represents an effective kinetic barrier over the sampled temperature window."
        ),
    })


# ---------------------------------------------------------------------------
# 6. Native Multi-Window & Azimuthal-Model Sensitivity Audit
# ---------------------------------------------------------------------------


def run_window_sensitivity_audit(
    intensity: Any,
    qx: Any = None,
    qy: Any = None,
    valid_mask: Any = None,
    *,
    q_unit: str = "nm^-1",
    q_windows: Sequence[tuple[float, float]] = (
        (0.25, 1.35),
        (0.11, 1.35),
        (0.12, 1.20),
        (0.15, 1.00),
    ),
    nominal_draw_axis_deg: float = 90.0,
    bright_axis_convention: str = "reference",
    intensity_mode: str = "calibrated",
    sigma: Any = None,
    geometry_cache: Any = None,
    n_radial_bins: int = 64,
    n_azimuthal_bins: int = 180,
    sector_half_width_deg: float = 20.0,
    azimuthal_q_max: float = 0.45,
    systematic_k_uncertainty: float | str | None = None,
) -> dict[str, Any]:
    """Compare direct/Kratky peaks, ``Q_win``, and flat vs C2-bridge models across q-windows.

    Performs a single ROI pre-filter / geometry pass and evaluates every requested
    ``q_window`` strictly on observed valid pixels, returning a strict-JSON
    (``allow_nan=False``) audit summary.
    """
    mode = _normalise_intensity_mode(intensity_mode)
    convention = str(bright_axis_convention or "reference").strip().lower()
    if convention not in {"draw", "reference"}:
        raise ValueError("bright_axis_convention must be 'draw' or 'reference'")
    validated_windows = [_validate_q_window(w) for w in q_windows]
    if not validated_windows:
        raise ValueError("q_windows must contain at least one (q_min, q_max) pair")
    if isinstance(n_radial_bins, (bool, np.bool_)) or int(n_radial_bins) < 4:
        raise ValueError("n_radial_bins must be an integer >= 4")
    if isinstance(n_azimuthal_bins, (bool, np.bool_)) or int(n_azimuthal_bins) < 12:
        raise ValueError("n_azimuthal_bins must be an integer >= 12")

    q_unit_label, _, unit_caveat = _q_unit_info(q_unit)
    global_q_max = max(w[1] for w in validated_windows)
    ref_span = max(w[1] - w[0] for w in validated_windows)

    img, qx_arr, qy_arr, q_mag, chi_deg, geom_mask, inclusion, sig_arr, pixel_dq_hint = _resolve_2d_context(
        intensity,
        qx,
        qy,
        valid_mask,
        geometry_cache=geometry_cache,
        sigma=sigma,
        required_q_max=global_q_max,
    )

    # Build a single compact in-memory ROI cache for the whole multi-window pass
    if q_mag is None:
        in_box = (np.abs(qx_arr) <= global_q_max) & (np.abs(qy_arr) <= global_q_max)
        row_any = np.any(in_box, axis=1)
        if not np.any(row_any):
            if not np.any(np.isfinite(qx_arr) & np.isfinite(qy_arr)):
                raise ValueError("qx and qy contain no finite reciprocal-space coordinates")
            sl = (slice(0, img.shape[0]), slice(0, img.shape[1]))
        else:
            col_any = np.any(in_box, axis=0)
            y0 = int(np.argmax(row_any))
            y1 = int(img.shape[0] - np.argmax(row_any[::-1]))
            x0 = int(np.argmax(col_any))
            x1 = int(img.shape[1] - np.argmax(col_any[::-1]))
            sl = (slice(y0, y1), slice(x0, x1))
        img_roi = np.ascontiguousarray(img[sl])
        qx_roi = np.ascontiguousarray(qx_arr[sl])
        qy_roi = np.ascontiguousarray(qy_arr[sl])
        q_roi = np.hypot(qx_roi, qy_roi)
        chi_roi = np.mod(np.degrees(np.arctan2(qy_roi, qx_roi)), 360.0)
        geom_roi = np.isfinite(q_roi)
        valid_roi = np.ascontiguousarray(inclusion[sl]) & geom_roi
        sig_roi = np.ascontiguousarray(sig_arr[sl]) if sig_arr is not None else None
    else:
        img_roi = img
        qx_roi = qx_arr
        qy_roi = qy_arr
        q_roi = q_mag
        chi_roi = (
            np.mod(chi_deg, 360.0)
            if chi_deg is not None
            else np.mod(np.degrees(np.arctan2(qy_roi, qx_roi)), 360.0)
        )
        geom_roi = geom_mask if geom_mask is not None else np.isfinite(q_roi)
        valid_roi = inclusion & geom_roi
        sig_roi = sig_arr

    from .geometry import SAXSGeometryCache

    pass_cache = SAXSGeometryCache(
        geometry=geometry_cache.geometry if geometry_cache is not None else {"qx": qx_roi, "qy": qy_roi, "q": q_roi},
        shape=img_roi.shape,
        beam_center_xy=getattr(geometry_cache, "beam_center_xy", (0.0, 0.0)),
        pixel_dq=float(pixel_dq_hint) if pixel_dq_hint is not None else 1.0,
        base_valid_mask=valid_roi,
        beamstop_exclusion_mask=np.zeros(img_roi.shape, dtype=bool),
        valid_mask=valid_roi,
        roi_slices=(slice(0, img_roi.shape[0]), slice(0, img_roi.shape[1])),
        q_mag_roi=q_roi,
        qx_roi=qx_roi,
        qy_roi=qy_roi,
        chi_deg_roi=chi_roi,
        valid_roi=valid_roi,
        q_coordinates_validated=True,
        q_roi_max=global_q_max,
        q_unit=q_unit_label,
        geometry_roi_mask=geom_roi,
    )

    # Seed reference lobe centers once from the widest window's azimuthal band
    widest_lo, widest_hi = max(validated_windows, key=lambda w: w[1] - w[0])
    seed_az_hi = min(float(azimuthal_q_max), widest_hi)
    seed_az_win = (widest_lo, seed_az_hi) if seed_az_hi > widest_lo else (widest_lo, widest_hi)
    seed_ang, seed_prof, seed_cnt, _, _, _ = _bin_azimuthal_from_2d(
        img_roi,
        qx_roi,
        qy_roi,
        geom_roi,
        None,
        q_window=seed_az_win,
        n_bins=int(n_azimuthal_bins),
        q_mag=q_roi,
        chi_deg=chi_roi,
        inclusion=valid_roi,
    )
    seed_harm = compute_azimuthal_harmonics_and_order(
        angle_deg=seed_ang,
        intensity=seed_prof,
        counts=seed_cnt,
        q_unit=q_unit,
        nominal_draw_axis_deg=nominal_draw_axis_deg,
        bright_axis_convention=convention,
    )
    shared_lobe_centers: list[float] | None = None
    if (seed_harm.get("a4") or 0.0) >= 0.02 and int(seed_harm.get("n_observed_bins") or 0) >= 12:
        seed_fit = fit_c2_bridge_four_lobe(
            seed_ang,
            seed_prof,
            counts=seed_cnt,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            initial_chi_sym_deg=seed_harm.get("friedel_bisector_deg"),
            initial_alpha_deg=seed_harm.get("wing_half_opening_estimate_deg"),
            max_nfev=1500,
        )
        if seed_fit.get("success") and seed_fit.get("lobe_centers_deg"):
            centers = [float(c) for c in seed_fit["lobe_centers_deg"] if c is not None]
            if len(centers) == 4:
                shared_lobe_centers = centers

    window_rows: list[dict[str, Any]] = []
    all_flags: list[str] = []
    any_sigma = False

    for idx_w, (w_lo, w_hi) in enumerate(validated_windows):
        n_bins_w = max(16, int(round(float(n_radial_bins) * (w_hi - w_lo) / max(ref_span, 1.0e-12))))
        rad = measure_radial_coarsening_observables(
            img_roi,
            q_window=(w_lo, w_hi),
            q_unit=q_unit,
            n_radial_bins=n_bins_w,
            lobe_angles_deg=shared_lobe_centers,
            sector_half_width_deg=sector_half_width_deg,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            geometry_cache=pass_cache,
            intensity_mode=mode,
            sigma=sig_roi,
            systematic_k_uncertainty=systematic_k_uncertainty,
        )
        if rad.get("sigma_source") == "supplied_statistical_sigma":
            any_sigma = True

        az_hi = min(float(azimuthal_q_max), w_hi)
        az_win = (w_lo, az_hi) if az_hi > w_lo else (w_lo, w_hi)
        w_ang, w_prof, w_cnt, _, _, _ = _bin_azimuthal_from_2d(
            img_roi,
            qx_roi,
            qy_roi,
            geom_roi,
            None,
            q_window=az_win,
            n_bins=int(n_azimuthal_bins),
            q_mag=q_roi,
            chi_deg=chi_roi,
            inclusion=valid_roi,
        )
        harm = compute_azimuthal_harmonics_and_order(
            angle_deg=w_ang,
            intensity=w_prof,
            counts=w_cnt,
            q_unit=q_unit,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            bright_axis_convention=convention,
        )
        c2_fit = fit_c2_bridge_four_lobe(
            w_ang,
            w_prof,
            counts=w_cnt,
            nominal_draw_axis_deg=nominal_draw_axis_deg,
            initial_chi_sym_deg=harm.get("friedel_bisector_deg"),
            initial_alpha_deg=harm.get("wing_half_opening_estimate_deg"),
            max_nfev=1500,
        )
        fcmp = c2_fit.get("flat_baseline_comparison") or {}
        all_flags.extend(rad.get("flags") or ())
        all_flags.extend(harm.get("flags") or ())

        window_rows.append({
            "window_index": idx_w,
            "window_label": f"{w_lo:g}_{w_hi:g}",
            "q_window": [w_lo, w_hi],
            "azimuthal_q_window": [az_win[0], az_win[1]],
            "n_radial_bins": n_bins_w,
            "peak_channel_source": rad.get("peak_channel_source"),
            "q_star_direct": rad.get("q_star_direct"),
            "L_star_direct_nm": rad.get("L_star_direct_nm"),
            "edge_hit_direct": rad.get("edge_hit_direct"),
            "direct_low_q_edge_hit": rad.get("direct_low_q_edge_hit"),
            "q_star_kratky": rad.get("q_star_kratky"),
            "L_star_kratky_nm": rad.get("L_star_kratky_nm"),
            "edge_hit_kratky": rad.get("edge_hit_kratky"),
            "kratky_low_q_edge_hit": rad.get("kratky_low_q_edge_hit"),
            "I_bg_estimate": rad.get("I_bg_estimate"),
            "q_win_invariant": rad.get("q_win_invariant"),
            "q_win_bg_subtracted": rad.get("q_win_bg_subtracted"),
            "q_win_stat_stderr": rad.get("q_win_stat_stderr"),
            "a2": harm.get("a2"),
            "a4": harm.get("a4"),
            "a4_over_a2": harm.get("a4_over_a2"),
            "friedel_bisector_deg": harm.get("friedel_bisector_deg"),
            "orthogonal_rotation_detected": bool(harm.get("orthogonal_rotation_detected", False)),
            "c2_bridge_model": {
                "success": bool(c2_fit.get("success", False)),
                "c0": c2_fit.get("c0"),
                "c2": c2_fit.get("c2"),
                "chi_sym_deg": c2_fit.get("chi_sym_deg"),
                "alpha_deg": c2_fit.get("alpha_deg"),
                "lobe_fwhm_deg": c2_fit.get("lobe_fwhm_deg"),
                "rmse": c2_fit.get("rmse"),
                "r_squared": c2_fit.get("r_squared"),
                "aic": c2_fit.get("aic"),
                "bic": c2_fit.get("bic"),
            },
            "flat_baseline_model": {
                "success": bool(fcmp.get("success", False)),
                "c0": fcmp.get("c0"),
                "chi_sym_deg": fcmp.get("chi_sym_deg"),
                "alpha_deg": fcmp.get("alpha_deg"),
                "lobe_fwhm_deg": fcmp.get("lobe_fwhm_deg"),
                "rmse": fcmp.get("rmse"),
                "r_squared": fcmp.get("r_squared"),
                "aic": fcmp.get("aic"),
                "bic": fcmp.get("bic"),
            },
            "model_comparison": {
                "delta_rmse": fcmp.get("delta_rmse"),
                "delta_r_squared": fcmp.get("delta_r_squared"),
                "delta_aic": fcmp.get("delta_aic"),
                "delta_bic": fcmp.get("delta_bic"),
                "bridge_f_stat": fcmp.get("bridge_f_stat"),
                "bridge_improved": bool(fcmp.get("bridge_improved", False)),
                "lobe_fwhm_bias_deg": fcmp.get("lobe_fwhm_bias_deg"),
            },
        })

    def _span(values: list[float | None]) -> list[float] | None:
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        return [min(finite), max(finite)] if finite else None

    delta_bics = [
        float(r["model_comparison"]["delta_bic"])
        for r in window_rows
        if r["model_comparison"]["delta_bic"] is not None
    ]
    caveats = [
        "Window sensitivity audit evaluates strictly observed valid pixels across each declared q_window "
        "without fabricating or extrapolating outside the measured reciprocal-space domain.",
    ]
    if unit_caveat is not None:
        caveats.append(unit_caveat)
    if mode == "raw":
        caveats.append(
            "intensity_mode='raw': q_win_invariant and q_win_bg_subtracted are withheld (None) across all windows."
        )

    sys_k = "unavailable" if systematic_k_uncertainty is None else systematic_k_uncertainty
    return json_safe({
        "schema_version": "wingsaxs.window_sensitivity_audit.v1",
        "status": "estimated" if window_rows else "unavailable",
        "q_unit": q_unit_label,
        "intensity_mode": mode,
        "nominal_draw_axis_deg": float(nominal_draw_axis_deg) % 180.0,
        "bright_axis_convention": convention,
        "sigma_source": "supplied_statistical_sigma" if any_sigma else "unavailable",
        "systematic_k_uncertainty": sys_k,
        "windows": window_rows,
        "summary": {
            "n_windows": len(window_rows),
            "n_windows_direct_peak": sum(1 for r in window_rows if r["q_star_direct"] is not None),
            "n_windows_kratky_peak": sum(1 for r in window_rows if r["q_star_kratky"] is not None),
            "n_windows_bridge_improved": sum(1 for r in window_rows if r["model_comparison"]["bridge_improved"]),
            "q_star_direct_range": _span([r["q_star_direct"] for r in window_rows]),
            "L_star_direct_nm_range": _span([r["L_star_direct_nm"] for r in window_rows]),
            "q_star_kratky_range": _span([r["q_star_kratky"] for r in window_rows]),
            "L_star_kratky_nm_range": _span([r["L_star_kratky_nm"] for r in window_rows]),
            "q_win_bg_subtracted_range": _span([r["q_win_bg_subtracted"] for r in window_rows]),
            "median_delta_bic_c2_vs_flat": _finite(float(np.median(delta_bics))) if delta_bics else None,
        },
        "flags": list(dict.fromkeys(all_flags)),
        "caveats": caveats,
    })


__all__ = [
    "compute_azimuthal_harmonics_and_order",
    "evaluate_c2_bridge_four_lobe",
    "fit_arrhenius_rate",
    "fit_c2_bridge_four_lobe",
    "fit_power_law_coarsening",
    "measure_perpendicular_streak_width",
    "measure_radial_coarsening_observables",
    "run_window_sensitivity_audit",
    "unwrap_periodic_angles_deg",
]
