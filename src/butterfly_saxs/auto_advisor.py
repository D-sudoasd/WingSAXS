"""Smart, observed-pixel-only SAXS frame parameter and morphology advisor.

This module is strictly Qt-free and operates solely on observed valid pixels
(``valid_mask=True`` and finite intensity/q/azimuth). It never fabricates,
mirrors, or interpolates masked pixels or detector gaps, and never modifies
input files, masks, or PONI calibrations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .butterfly_quality import build_quality_remediation
from .geometry import apply_beamstop_mask
from .settings import canonical_q_unit

SCHEMA_VERSION = "wingsaxs.auto_advisor.v1"

MORPHOLOGY_CLASSES = (
    "isotropic_ring",
    "two_lobe",
    "four_lobe_butterfly",
    "orthogonal_flipped_butterfly",
    "streak_contaminated",
    "low_signal",
)

_ALLOWED_ANGLE_BINS = (16, 24, 36, 48, 60, 72, 90, 120, 144, 180, 240, 360)


def _sanitize_json(value: Any) -> Any:
    """Convert nested structures to strict JSON-safe native Python values."""
    if isinstance(value, np.generic):
        return _sanitize_json(value.item())
    if isinstance(value, np.ndarray):
        return [_sanitize_json(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _signed_angle_diff_mod180(angle_deg: float, ref_deg: float) -> float:
    """Return signed minimal angular deviation in (-90, 90] modulo 180 deg."""
    diff = ((float(angle_deg) - float(ref_deg) + 90.0) % 180.0) - 90.0
    if diff <= -90.0:
        diff += 180.0
    return float(diff)


def _circular_angle_diff_360(a_deg: float, b_deg: float) -> float:
    """Return unsigned minimal distance in [0, 180] modulo 360 deg."""
    return float(abs(((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0))


def _normalise_q_unit(q_unit: Any) -> str:
    canonical = canonical_q_unit(q_unit)
    if canonical == "nm⁻¹":
        return "nm^-1"
    if canonical == "Å⁻¹":
        return "Å^-1"
    return str(canonical or "unknown")


def build_beamstop_exclusion_mask(
    shape: tuple[int, int] | Sequence[int],
    beam_center_xy: tuple[float, float] | Sequence[float],
    radius_px: float,
) -> np.ndarray:
    """Build a circular beamstop exclusion mask (``True`` = excluded pixel).

    Follows ``butterfly_saxs.masking`` detector-exclusion polarity: ``True``
    marks pixels inside ``radius_px`` of ``beam_center_xy = (cx, cy)`` to be
    excluded from analysis, and ``False`` marks pixels retained.
    """
    if isinstance(shape, (str, bytes)):
        raise ValueError("shape must be a (height, width) pair of positive integers")
    try:
        dims = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must be a (height, width) pair of positive integers") from exc
    if len(dims) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in dims):
        raise ValueError(f"shape must be two positive integers, got {shape!r}")
    height, width = int(dims[0]), int(dims[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"shape dimensions must be positive, got {(height, width)!r}")

    if isinstance(beam_center_xy, (str, bytes)):
        raise ValueError("beam_center_xy must be a finite (cx, cy) pair")
    try:
        center = tuple(beam_center_xy)
    except TypeError as exc:
        raise ValueError("beam_center_xy must be a finite (cx, cy) pair") from exc
    if len(center) != 2 or any(isinstance(v, bool) for v in center):
        raise ValueError(f"beam_center_xy must be a finite (cx, cy) pair, got {beam_center_xy!r}")
    cx, cy = _finite_float(center[0]), _finite_float(center[1])
    if cx is None or cy is None:
        raise ValueError(f"beam_center_xy must contain finite numbers, got {beam_center_xy!r}")

    if isinstance(radius_px, bool):
        raise ValueError("radius_px must be a finite non-negative number")
    radius = _finite_float(radius_px)
    if radius is None or radius < 0.0:
        raise ValueError(f"radius_px must be a finite non-negative number, got {radius_px!r}")

    if radius == 0.0:
        return np.zeros((height, width), dtype=bool)
    _, exclusion = apply_beamstop_mask(None, (height, width), (cx, cy), radius)
    return exclusion


def _infer_beam_geometry(
    q_map: np.ndarray,
    chi_deg: np.ndarray,
    *,
    beam_center_xy: Sequence[float] | None = None,
    pixel_dq: float | None = None,
) -> tuple[tuple[float, float], float, str]:
    """Infer subpixel beam center (cx, cy) and pixel q-step from coordinate maps."""
    height, width = q_map.shape
    inferred_cx: float | None = None
    inferred_cy: float | None = None
    inferred_dq: float | None = None
    source = "explicit"

    finite_q = np.isfinite(q_map)
    finite_both = finite_q & np.isfinite(chi_deg)

    if beam_center_xy is not None:
        if isinstance(beam_center_xy, (str, bytes)):
            raise ValueError("beam_center_xy must be a (cx, cy) pair")
        pair = tuple(beam_center_xy)
        if len(pair) != 2:
            raise ValueError("beam_center_xy must be a (cx, cy) pair")
        cx_val, cy_val = _finite_float(pair[0]), _finite_float(pair[1])
        if cx_val is None or cy_val is None:
            raise ValueError("beam_center_xy must contain finite coordinates")
        inferred_cx, inferred_cy = cx_val, cy_val

    if pixel_dq is not None:
        if isinstance(pixel_dq, bool):
            raise ValueError("pixel_dq must be a positive finite number")
        dq_val = _finite_float(pixel_dq)
        if dq_val is None or dq_val <= 0.0:
            raise ValueError(f"pixel_dq must be a positive finite number, got {pixel_dq!r}")
        inferred_dq = dq_val

    if inferred_cx is None or inferred_dq is None:
        # Affine fit (x, y) -> (qx, qy) recovers subpixel beam center and dq
        # even when the beam center itself is masked or outside the detector.
        stride_y = max(1, height // 48)
        stride_x = max(1, width // 48)
        sub_mask = finite_both[::stride_y, ::stride_x]
        if np.count_nonzero(sub_mask) >= 12:
            ys = np.arange(0, height, stride_y, dtype=np.float64)
            xs = np.arange(0, width, stride_x, dtype=np.float64)
            gx, gy = np.meshgrid(xs, ys)
            xv = gx[sub_mask]
            yv = gy[sub_mask]
            qv = q_map[::stride_y, ::stride_x][sub_mask]
            chiv = np.deg2rad(chi_deg[::stride_y, ::stride_x][sub_mask])
            qxv = qv * np.cos(chiv)
            qyv = qv * np.sin(chiv)
            design = np.column_stack((np.ones_like(xv), xv, yv))
            try:
                coef_x, *_ = np.linalg.lstsq(design, qxv, rcond=None)
                coef_y, *_ = np.linalg.lstsq(design, qyv, rcond=None)
                a0, a1, a2 = float(coef_x[0]), float(coef_x[1]), float(coef_x[2])
                b0, b1, b2 = float(coef_y[0]), float(coef_y[1]), float(coef_y[2])
                det = a1 * b2 - a2 * b1
                pred_x = design @ coef_x
                pred_y = design @ coef_y
                max_q = float(np.max(qv)) if qv.size else 0.0
                rms = float(np.sqrt(np.mean((pred_x - qxv) ** 2 + (pred_y - qyv) ** 2)))
                if abs(det) > 1e-18 and max_q > 0.0 and rms <= 0.15 * max_q:
                    cx_fit = (-b2 * a0 + a2 * b0) / det
                    cy_fit = (b1 * a0 - a1 * b0) / det
                    dq_fit = math.sqrt(0.5 * (a1 * a1 + a2 * a2 + b1 * b1 + b2 * b2))
                    if math.isfinite(cx_fit) and math.isfinite(cy_fit) and inferred_cx is None:
                        inferred_cx, inferred_cy = cx_fit, cy_fit
                        source = "q_chi_affine"
                    if math.isfinite(dq_fit) and dq_fit > 0.0 and inferred_dq is None:
                        inferred_dq = dq_fit
            except np.linalg.LinAlgError:
                pass

    if inferred_cx is None or inferred_cy is None:
        if np.any(finite_q):
            q_fill = np.where(finite_q, q_map, np.inf)
            min_idx = np.unravel_index(int(np.argmin(q_fill)), q_map.shape)
            inferred_cy, inferred_cx = float(min_idx[0]), float(min_idx[1])
            source = "q_minimum"
        else:
            inferred_cx = 0.5 * (width - 1)
            inferred_cy = 0.5 * (height - 1)
            source = "image_center_fallback"

    if inferred_dq is None:
        if height >= 2 and width >= 2:
            dq_dy, dq_dx = np.gradient(np.where(finite_q, q_map, np.nan))
            grad = np.hypot(dq_dx, dq_dy)
            valid_grad = grad[np.isfinite(grad) & (grad > 0.0)]
            if valid_grad.size:
                inferred_dq = float(np.median(valid_grad))
        if inferred_dq is None or not math.isfinite(inferred_dq) or inferred_dq <= 0.0:
            inferred_dq = 1.0

    return (float(inferred_cx), float(inferred_cy)), float(inferred_dq), source


def _detect_active_outer_q_cutoff(
    intensity: np.ndarray,
    q_map: np.ndarray,
    obs_valid: np.ndarray,
    pixel_dq: float,
    *,
    q_obs: np.ndarray | None = None,
    i_obs: np.ndarray | None = None,
) -> tuple[float, bool]:
    """Detect and exclude unmasked outer zero-padded dead high-q shells.

    Some calibrated 2D containers (such as circularly integrated H5 images)
    pad detector corners beyond an outer cutoff with exact ``0.0`` without
    setting the detector mask. Excluding contiguous outer shells with
    ``frac(|I| <= 1e-12) > 0.90`` prevents zero-fill cliffs from collapsing
    the diffuse background floor to zero.
    """
    if q_obs is not None and i_obs is not None:
        q_use = q_obs
        i_use = i_obs
    elif intensity.shape[0] >= 512 and intensity.shape[1] >= 512:
        # Uniform 2x2 spatial grid (~620k px on Pilatus 2M) avoids 1D stride aliasing in corners
        ov_sub = obs_valid[::2, ::2]
        q_use = q_map[::2, ::2][ov_sub]
        i_use = intensity[::2, ::2][ov_sub]
    else:
        q_use = q_map[obs_valid]
        i_use = intensity[obs_valid]
    if q_use.size < 64:
        return (float(np.max(q_use)) if q_use.size else 0.0), False
    q_min_obs = float(np.min(q_use))
    q_max_obs = float(np.max(q_use))
    if q_max_obs <= q_min_obs:
        return q_max_obs, False

    span_px = (q_max_obs - q_min_obs) / max(pixel_dq, 1e-12)
    n_shells = int(np.clip(round(span_px / 3.0), 24, 120))
    edges = np.linspace(q_min_obs, q_max_obs, n_shells + 1, dtype=np.float64)
    inv_dq = float(n_shells) / (q_max_obs - q_min_obs)
    # q_use >= q_min_obs guarantees non-negative scaled coordinates -> astype(intp) == floor
    idx = ((q_use - q_min_obs) * inv_dq).astype(np.intp)
    idx[idx >= n_shells] = n_shells - 1
    counts = np.bincount(idx, minlength=n_shells)
    zero_sel = (i_use >= -1e-12) & (i_use <= 1e-12)
    zero_counts = np.bincount(idx[zero_sel], minlength=n_shells) if np.any(zero_sel) else np.zeros(n_shells, dtype=np.int64)

    min_shell_cnt = max(5, int(round(0.00015 * q_use.size))) if q_use.size > 100_000 else 5
    populated = np.flatnonzero(counts >= min_shell_cnt)
    if populated.size < 6:
        populated = np.flatnonzero(counts >= 5)
    if populated.size < 6:
        return q_max_obs, False

    zero_frac = np.zeros(n_shells, dtype=np.float64)
    zero_frac[populated] = zero_counts[populated] / counts[populated]

    # Walk inward from outermost populated shell while >90% pixels are exact zero
    last_active_pos = populated.size - 1
    dead_shells = 0
    dead_pixels = 0
    while last_active_pos >= 0:
        b = int(populated[last_active_pos])
        if zero_frac[b] > 0.90:
            dead_shells += 1
            dead_pixels += int(counts[b])
            last_active_pos -= 1
        else:
            break

    if dead_shells < 2 and dead_pixels < int(0.05 * q_use.size):
        return q_max_obs, False
    if last_active_pos < 3:
        return q_max_obs, False

    # Verify genuine positive active scattering exists inside the dead outer zone
    inner_bins = populated[: last_active_pos + 1]
    if not np.any(zero_frac[inner_bins] < 0.50):
        return q_max_obs, False

    # Step past any partial-zero transition shell immediately adjacent to the cliff
    while last_active_pos > 3 and zero_frac[int(populated[last_active_pos])] > 0.20:
        last_active_pos -= 1

    active_bin = int(populated[last_active_pos])
    q_active_max = float(edges[active_bin + 1])
    # Reuse already-binned shell counts & zero counts (exact for q <= edges[active_bin + 1])
    n_active = int(np.sum(counts[: active_bin + 1]))
    if n_active < 16:
        return q_max_obs, False
    n_zero_active = int(np.sum(zero_counts[: active_bin + 1]))
    if n_zero_active >= int(math.ceil(0.90 * (n_active - 1))) + 1:
        return q_max_obs, False
    return q_active_max, True


def _compute_beamstop_shells(
    intensity: np.ndarray,
    obs_valid: np.ndarray,
    cx: float,
    cy: float,
    r_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-integer-radius total/valid counts, medians, and zero-fractions inside ``r <= r_limit``."""
    height, width = intensity.shape
    n_shells = r_limit + 1
    total_cnt = np.zeros(n_shells, dtype=np.int64)
    valid_cnt = np.zeros(n_shells, dtype=np.int64)
    shell_med = np.full(n_shells, np.nan, dtype=np.float64)
    shell_zero_frac = np.full(n_shells, np.nan, dtype=np.float64)

    r_box = r_limit + 1.0
    y0 = max(0, int(math.floor(cy - r_box)))
    y1 = min(height, int(math.ceil(cy + r_box)) + 1)
    x0 = max(0, int(math.floor(cx - r_box)))
    x1 = min(width, int(math.ceil(cx + r_box)) + 1)
    if y0 < y1 and x0 < x1:
        dy2 = (np.arange(y0, y1, dtype=np.float32) - np.float32(cy)) ** 2
        dx2 = (np.arange(x0, x1, dtype=np.float32) - np.float32(cx)) ** 2
        r2_box = dy2[:, None] + dx2[None, :]
        r_lim2 = np.float32((r_limit + 0.5) ** 2)
        in_search = r2_box <= r_lim2
        if np.any(in_search):
            r_in = (np.sqrt(r2_box[in_search]) + np.float32(0.5)).astype(np.intp)
            r_in[r_in > r_limit] = r_limit
            total_cnt = np.bincount(r_in, minlength=n_shells).astype(np.int64)
            v_in = obs_valid[y0:y1, x0:x1][in_search]
            if np.any(v_in):
                r_valid = r_in[v_in]
                i_valid = intensity[y0:y1, x0:x1][in_search][v_in]
                valid_cnt = np.bincount(r_valid, minlength=n_shells).astype(np.int64)
                z_sel = i_valid <= 1e-12
                zero_cnt = (
                    np.bincount(r_valid[z_sel], minlength=n_shells)
                    if np.any(z_sel)
                    else np.zeros(n_shells, dtype=np.int64)
                )
                has_v = valid_cnt > 0
                shell_zero_frac[has_v] = zero_cnt[has_v] / valid_cnt[has_v]
                # Single-pass O(N) uint16 radix-sort grouping replaces 516x O(N) mask scans
                order = np.argsort(r_valid.astype(np.uint16), kind="stable")
                i_sorted = i_valid[order]
                ends = np.cumsum(valid_cnt)
                starts = ends - valid_cnt
                for radius in np.flatnonzero(has_v):
                    shell_med[radius] = float(np.median(i_sorted[starts[radius] : ends[radius]]))
    return total_cnt, valid_cnt, shell_med, shell_zero_frac


def _detect_beamstop(
    intensity: np.ndarray,
    q_map: np.ndarray,
    obs_valid: np.ndarray,
    beam_center_xy: tuple[float, float],
    pixel_dq: float,
    *,
    q_active_max: float | None = None,
    q_obs: np.ndarray | None = None,
    i_obs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Detect masked or unmasked central beamstop shadow around beam center."""
    height, width = intensity.shape
    cx, cy = beam_center_xy

    # Exact maximum radius across the 4 image corners (zero 2D index allocation)
    max_dx = max(abs(cx), abs((width - 1) - cx))
    max_dy = max(abs(cy), abs((height - 1) - cy))
    max_r_in_image = int(math.ceil(math.hypot(max_dx, max_dy)))
    r_full_max = min(max_r_in_image, max(24, int(round(0.35 * min(height, width)))))

    # Outer detector floor computed strictly on active (non-zero-padded) high-q pixels
    if q_obs is not None and i_obs is not None:
        if q_active_max is not None:
            act_sel = q_obs <= q_active_max
            if np.any(act_sel):
                valid_q = q_obs[act_sel]
                valid_i = i_obs[act_sel]
            else:
                valid_q = q_obs
                valid_i = i_obs
        else:
            valid_q = q_obs
            valid_i = i_obs
    else:
        active_valid = obs_valid if q_active_max is None else (obs_valid & (q_map <= q_active_max))
        if not np.any(active_valid):
            active_valid = obs_valid
        valid_q = q_map[active_valid]
        valid_i = intensity[active_valid]
    if valid_q.size > 120_000:
        step = max(1, valid_q.size // 100_000)
        vq_stat = valid_q[::step]
        vi_stat = valid_i[::step]
    else:
        vq_stat = valid_q
        vi_stat = valid_i
    q_p80 = float(np.percentile(vq_stat, 80.0))
    outer_vals = vi_stat[vq_stat >= q_p80]
    if outer_vals.size == 0:
        outer_vals = vi_stat
    outer_bg = float(np.median(outer_vals))
    outer_mad = float(1.4826 * np.median(np.abs(outer_vals - outer_bg)))
    outer_pos_frac = float(np.mean(outer_vals > 1e-12))

    # Fast inner probe (r <= 128 px = 257x257 box) handles >99.9% of beamstops in ~3 ms;
    # automatically expands to r_full_max if core remains masked/unrecovered at r=128.
    r_probe = min(r_full_max, 128) if r_full_max > 140 else r_full_max
    total_cnt, valid_cnt, shell_med, shell_zero_frac = _compute_beamstop_shells(
        intensity, obs_valid, cx, cy, r_probe
    )
    valid_frac_probe = np.divide(
        valid_cnt, total_cnt, out=np.full(total_cnt.shape, np.nan, dtype=np.float64), where=total_cnt > 0
    )
    need_expand = r_probe < r_full_max and (
        not np.any(np.isfinite(valid_frac_probe) & (valid_frac_probe >= 0.5))
        or int(np.argmax(np.isfinite(valid_frac_probe) & (valid_frac_probe >= 0.5))) > r_probe - 12
        or (np.isfinite(shell_zero_frac[-1]) and shell_zero_frac[-1] >= 0.5)
    )
    if need_expand:
        r_search_max = r_full_max
        total_cnt, valid_cnt, shell_med, shell_zero_frac = _compute_beamstop_shells(
            intensity, obs_valid, cx, cy, r_search_max
        )
    else:
        r_search_max = r_probe

    n_shells = r_search_max + 1
    valid_frac = np.full(n_shells, np.nan, dtype=np.float64)
    has_total = total_cnt > 0
    valid_frac[has_total] = valid_cnt[has_total] / total_cnt[has_total]

    # 1. Check already-masked core (valid_frac < 0.35 from r=0 outward)
    r_masked = 0
    if np.any(np.isfinite(valid_frac) & (valid_frac >= 0.5)):
        first_valid_shell = int(np.argmax(np.isfinite(valid_frac) & (valid_frac >= 0.5)))
        if first_valid_shell > 0:
            inner_present = has_total[:first_valid_shell]
            if np.any(inner_present) and np.all(valid_frac[:first_valid_shell][inner_present] < 0.35):
                r_masked = first_valid_shell

    # 2. Check unmasked beamstop shadow starting at r_masked
    unmasked_shadow = False
    r_shadow_end = -1
    detection_mode = "none"

    valid_shells = np.flatnonzero(np.isfinite(shell_med) & (np.arange(n_shells) >= r_masked))
    if valid_shells.size >= 5:
        med_vals = shell_med[valid_shells]
        inner_peak_val = float(np.max(med_vals))
        inner_min_val = float(np.min(med_vals))
        dynamic_range = inner_peak_val - inner_min_val
        r_cap = min(r_search_max - 2, max(16, int(round(0.30 * min(height, width)))))

        # Signature A: contiguous zero/non-positive core right from r_masked
        if valid_shells[0] <= r_masked + 1 and shell_zero_frac[valid_shells[0]] >= 0.5:
            run_end = int(valid_shells[0])
            for k in valid_shells[1:]:
                if int(k) > r_cap:
                    break
                if shell_zero_frac[int(k)] >= 0.5:
                    run_end = int(k)
                else:
                    break
            post_slice = np.arange(run_end + 1, min(n_shells, run_end + 7))
            post_valid = post_slice[np.isfinite(shell_med[post_slice])]
            if post_valid.size >= 2:
                post_peak = float(np.max(shell_med[post_valid]))
                post_zero = float(np.median(shell_zero_frac[post_valid[: min(3, post_valid.size)]]))
                jump_rise_1 = float(shell_med[post_valid[0]]) - float(shell_med[run_end])
                jump_rise_2 = (
                    float(shell_med[post_valid[1]]) - float(shell_med[run_end])
                    if post_valid.size >= 2
                    else jump_rise_1
                )
                sharp_edge = max(jump_rise_1, jump_rise_2) >= 0.35 * max(
                    post_peak - float(shell_med[run_end]), 1e-12
                )
                if (
                    post_peak > max(outer_bg + 3.0 * outer_mad, 1e-6)
                    and post_zero < 0.35
                    and (outer_pos_frac >= 0.45 or sharp_edge)
                ):
                    unmasked_shadow = True
                    r_shadow_end = run_end
                    detection_mode = "unmasked_zero_core"

        # Signature B: suppressed low-count core with sharp 1-px or 2-px rising rim
        if not unmasked_shadow and dynamic_range > max(5.0 * outer_mad, 1e-9):
            k0 = int(valid_shells[0])
            i_core0 = float(shell_med[k0])
            for k in range(k0, min(r_cap, n_shells - 3)):
                if not (np.isfinite(shell_med[k]) and np.isfinite(shell_med[k + 1])):
                    continue
                post_idx = np.arange(k + 1, min(n_shells, k + 6))
                post_idx = post_idx[np.isfinite(shell_med[post_idx])]
                if post_idx.size < 2:
                    continue
                i_post_peak = float(np.max(shell_med[post_idx]))
                jump = i_post_peak - i_core0
                if jump <= max(5.0 * outer_mad, 0.20 * dynamic_range, 1e-9):
                    continue
                core_slice = np.arange(k0, k + 1)
                core_slice = core_slice[np.isfinite(shell_med[core_slice])]
                if core_slice.size == 0:
                    continue
                core_max = float(np.max(shell_med[core_slice]))
                core_med = float(np.median(shell_med[core_slice]))
                # Core must stay flat and dark (never brighter at r=0 than at edge)
                if core_max > i_core0 + 0.15 * jump + 2.0 * outer_mad:
                    continue
                if core_med > 0.25 * i_post_peak:
                    continue
                # Check sharp rise across 1-px or 2-px beamstop rim
                step_1px = float(shell_med[post_idx[0]]) - float(shell_med[k])
                post_2px_max = float(np.max(shell_med[post_idx[: min(2, post_idx.size)]]))
                step_2px = post_2px_max - core_med
                if step_2px < 0.50 * jump:
                    continue
                step_2px_edge = post_2px_max - float(shell_med[k])
                ratio_2px = post_2px_max / max(core_med, outer_mad, 1e-6)
                darker_than_outer = (outer_bg > 0.0) and (core_med < 0.50 * outer_bg)
                very_sharp_step = (step_1px >= 0.40 * jump) or (
                    step_2px_edge >= 0.65 * jump and step_1px >= 0.20 * jump
                )
                huge_ratio_rim = (ratio_2px >= 10.0) and (step_2px_edge >= 0.65 * jump)
                # Post-edge scattering stays elevated after the 1-2 px rim
                post_stays_elevated = (
                    post_idx.size >= 3
                    and float(shell_med[post_idx[1]]) >= i_core0 + 0.35 * jump
                    and float(shell_med[post_idx[2]]) >= i_core0 + 0.35 * jump
                )
                if darker_than_outer or ((very_sharp_step or huge_ratio_rim) and post_stays_elevated):
                    unmasked_shadow = True
                    r_shadow_end = k
                    detection_mode = "unmasked_low_count_core"
                    break

        # Account for a 1-px partially shadowed rim immediately outside r_shadow_end
        if unmasked_shadow and r_shadow_end + 2 < n_shells:
            s1 = shell_med[r_shadow_end + 1]
            s2 = shell_med[r_shadow_end + 2]
            if np.isfinite(s1) and np.isfinite(s2) and s2 > 0 and s1 < 0.75 * s2:
                r_shadow_end += 1

    detected_radius_px = 0.0
    if unmasked_shadow:
        detected_radius_px = float(r_shadow_end)
    elif r_masked > 0:
        detected_radius_px = float(r_masked - 1)
        detection_mode = "pre_masked_core"

    shadow_detected = bool(unmasked_shadow or r_masked > 0)
    if shadow_detected:
        safety_margin_px = max(2.0, float(math.ceil(0.15 * max(detected_radius_px, 1.0))))
        recommended_mask_radius_px = float(detected_radius_px + safety_margin_px)
    else:
        safety_margin_px = 0.0
        recommended_mask_radius_px = 0.0

    # Compute corresponding q values directly from observed geometry
    q_shadow_edge = float(detected_radius_px * pixel_dq) if shadow_detected else 0.0
    q_safe_min = float(recommended_mask_radius_px * pixel_dq) if shadow_detected else 0.0

    return {
        "detected": shadow_detected,
        "unmasked_beamstop": bool(unmasked_shadow),
        "detection_mode": detection_mode,
        "beam_center_xy": [float(cx), float(cy)],
        "pixel_dq": float(pixel_dq),
        "pre_masked_radius_px": float(max(0, r_masked - 1)) if r_masked > 0 else 0.0,
        "shadow_radius_px": float(detected_radius_px),
        "safety_margin_px": float(safety_margin_px),
        "recommended_mask_radius_px": float(recommended_mask_radius_px),
        "shadow_edge_q": float(q_shadow_edge),
        "safe_q_min": float(q_safe_min),
    }


def _radial_profile_and_window(
    intensity: np.ndarray,
    q_map: np.ndarray,
    obs_valid: np.ndarray,
    beamstop: Mapping[str, Any],
    q_unit: str,
    *,
    q_active_max: float | None = None,
    outer_zero_padded: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute observed radial stats outside beamstop and choose a clean q_window."""
    pixel_dq = float(beamstop["pixel_dq"])
    cx, cy = float(beamstop["beam_center_xy"][0]), float(beamstop["beam_center_xy"][1])
    rec_radius_px = float(beamstop["recommended_mask_radius_px"])

    if rec_radius_px > 0.0:
        bs_excl = build_beamstop_exclusion_mask(intensity.shape, (cx, cy), rec_radius_px)
        clean_valid = obs_valid & (~bs_excl)
        if not np.any(clean_valid):
            clean_valid = obs_valid
    else:
        clean_valid = obs_valid

    # On large 2D frames (>=512x512), coarse 160-bin radial stats use a uniform 2x2
    # spatial grid (~620k px -> ~3.8k px/bin) with zero mixed-weight boundary artifacts,
    # while fine peak-finding inside [q_floor, q_max] runs at full 1x1 pixel resolution.
    use_coarse_2d_stride = intensity.shape[0] >= 512 and intensity.shape[1] >= 512
    coarse_stride = 2 if use_coarse_2d_stride else 1

    if use_coarse_2d_stride:
        q_sub_2d = q_map[::2, ::2]
        i_sub_2d = intensity[::2, ::2]
        cv_sub = clean_valid[::2, ::2]
        q_vals_all = q_sub_2d[cv_sub]
        if q_vals_all.size < 64:
            use_coarse_2d_stride = False
            coarse_stride = 1
            q_vals_all = q_map[clean_valid]
            i_vals_all = intensity[clean_valid]
        else:
            i_vals_all = i_sub_2d[cv_sub]
    else:
        q_vals_all = q_map[clean_valid]
        i_vals_all = intensity[clean_valid]

    q_valid_min = float(np.min(q_vals_all))
    q_valid_max = float(np.max(q_vals_all))

    # Exclude outer zero-padded dead shells from radial profile & diffuse floor estimation
    active_max = q_valid_max if q_active_max is None else min(q_valid_max, float(q_active_max))
    in_act_1d = q_vals_all <= active_max
    if int(np.count_nonzero(in_act_1d)) < 16:
        in_act_1d = np.ones(q_vals_all.shape, dtype=bool)
        active_max = q_valid_max
        outer_zero_padded = False
        active_clean = clean_valid
    else:
        active_clean = clean_valid if q_active_max is None else (clean_valid & (q_map <= active_max))

    q_vals = q_vals_all[in_act_1d]
    i_vals = i_vals_all[in_act_1d]

    # Avoid exact q == 0 bin when no beamstop is present
    q_floor = max(q_valid_min, float(beamstop["safe_q_min"]))
    if rec_radius_px <= 0.0:
        q_floor = max(q_floor, 2.0 * pixel_dq)
        if q_floor >= active_max:
            q_floor = q_valid_min

    span_px = (active_max - q_floor) / max(pixel_dq, 1e-12)
    n_rbins = int(np.clip(round(span_px / 2.0), 16, 160))
    if active_max <= q_floor:
        n_rbins = 1
        edges = np.array([q_valid_min, active_max + max(pixel_dq, 1e-9)], dtype=np.float64)
    else:
        edges = np.linspace(q_floor, active_max, n_rbins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Evaluate candidate and observed radial profiles on the same uniform spatial grid
    e_lo = float(edges[0])
    e_hi = float(edges[-1])
    inv_dq_e = float(n_rbins) / max(e_hi - e_lo, 1e-30)

    q_cand_src = q_map[::coarse_stride, ::coarse_stride].ravel()
    in_cand = (q_cand_src >= e_lo) & (q_cand_src <= e_hi)
    cand_idx = ((q_cand_src[in_cand] - e_lo) * inv_dq_e).astype(np.intp)
    cand_idx[cand_idx >= n_rbins] = n_rbins - 1
    cand_counts_raw = np.bincount(cand_idx, minlength=n_rbins)

    in_obs = (q_vals >= e_lo) & (q_vals <= e_hi)
    obs_q = q_vals[in_obs]
    obs_i = i_vals[in_obs]
    obs_idx = ((obs_q - e_lo) * inv_dq_e).astype(np.intp)
    obs_idx[obs_idx >= n_rbins] = n_rbins - 1
    obs_counts_raw = np.bincount(obs_idx, minlength=n_rbins)

    radial_mean = np.full(n_rbins, np.nan, dtype=np.float64)
    radial_med = np.full(n_rbins, np.nan, dtype=np.float64)
    radial_p90 = np.full(n_rbins, np.nan, dtype=np.float64)
    radial_p25 = np.full(n_rbins, np.nan, dtype=np.float64)
    radial_cov = np.zeros(n_rbins, dtype=np.float64)
    has_cand = cand_counts_raw > 0
    radial_cov[has_cand] = obs_counts_raw[has_cand] / cand_counts_raw[has_cand]

    has_obs_bin = obs_counts_raw > 0
    if np.any(has_obs_bin):
        obs_sums = np.bincount(obs_idx, weights=obs_i, minlength=n_rbins)
        radial_mean[has_obs_bin] = obs_sums[has_obs_bin] / obs_counts_raw[has_obs_bin]
        # Single-pass O(N) uint8 radix-sort grouping replaces 160x O(N) scans
        obs_order = np.argsort(obs_idx.astype(np.uint8), kind="stable")
        i_grouped = obs_i[obs_order]
        obs_ends = np.cumsum(obs_counts_raw)
        obs_starts = obs_ends - obs_counts_raw
        for b in np.flatnonzero(has_obs_bin):
            seg = i_grouped[obs_starts[b] : obs_ends[b]]
            p25_v, med_v, p90_v = np.percentile(seg, (25.0, 50.0, 90.0))
            radial_p25[b] = float(p25_v)
            radial_med[b] = float(med_v)
            radial_p90[b] = float(p90_v)
    else:
        i_grouped = np.empty(0, dtype=np.float64)
        obs_starts = np.zeros(n_rbins, dtype=np.int64)
        obs_ends = np.zeros(n_rbins, dtype=np.int64)

    # Estimate active outer diffuse background floor & noise from outermost 20% of active bins.
    # Use raw strided bin counts (obs_counts_raw) for sample-size thresholds and standard errors,
    # since radial_mean/p90 average obs_counts_raw pixels rather than obs_counts_raw * coarse_scale.
    valid_bins = np.flatnonzero(np.isfinite(radial_med) & (obs_counts_raw >= 3))
    if valid_bins.size == 0:
        valid_bins = np.flatnonzero(np.isfinite(radial_med))

    tail_count = max(2, int(math.ceil(0.20 * valid_bins.size)))
    tail_bins = valid_bins[-tail_count:] if valid_bins.size else np.array([], dtype=np.int64)
    if tail_bins.size:
        if tail_bins.size == 1 or np.all(np.diff(tail_bins) == 1):
            tail_pixels = i_grouped[obs_starts[tail_bins[0]] : obs_ends[tail_bins[-1]]]
        else:
            tail_pixels = np.concatenate([i_grouped[obs_starts[b] : obs_ends[b]] for b in tail_bins])
        bg_level = float(np.median(tail_pixels))
        bg_mad = float(1.4826 * np.median(np.abs(tail_pixels - bg_level)))
        bg_p90 = float(np.median(radial_p90[tail_bins]))
        bg_aniso = float(np.median(radial_p90[tail_bins] - radial_p25[tail_bins]))
    else:
        bg_level = float(np.median(i_vals))
        bg_mad = float(1.4826 * np.median(np.abs(i_vals - bg_level)))
        p25_g, p90_g = np.percentile(i_vals, (25.0, 90.0))
        bg_p90 = float(p90_g)
        bg_aniso = float(p90_g - p25_g)

    noise_scale = max(bg_mad, 1e-12 * max(abs(float(np.max(obs_i))) if obs_i.size else 1.0, 1.0))

    # Signal profile combines radial excess and baseline-adjusted anisotropic p90 wing excess
    excess_mean = radial_mean - bg_level
    excess_p90 = radial_p90 - bg_p90
    aniso_excess = (radial_p90 - radial_p25) - bg_aniso
    signal_curve = np.fmax(excess_mean, np.fmax(excess_p90, aniso_excess))

    finite_sig = np.isfinite(signal_curve)
    peak_signal = float(np.nanmax(signal_curve)) if np.any(finite_sig) else 0.0
    peak_aniso = float(np.nanmax(aniso_excess)) if np.any(np.isfinite(aniso_excess)) else 0.0
    sqrt_n = np.sqrt(np.maximum(obs_counts_raw, 1))
    se_mean = np.where(obs_counts_raw > 0, noise_scale / sqrt_n, np.inf)
    se_p90 = np.where(obs_counts_raw > 0, 1.71 * noise_scale / sqrt_n, np.inf)
    se_aniso = np.where(obs_counts_raw > 0, 2.15 * noise_scale / sqrt_n, np.inf)
    snr_mean = np.where(np.isfinite(excess_mean), excess_mean / np.maximum(se_mean, 1e-12), 0.0)
    snr_p90 = np.where(np.isfinite(excess_p90), excess_p90 / np.maximum(se_p90, 1e-12), 0.0)
    snr_aniso = np.where(np.isfinite(aniso_excess), aniso_excess / np.maximum(se_aniso, 1e-12), 0.0)
    snr_curve = np.fmax(snr_mean, np.fmax(snr_p90, snr_aniso))
    pixel_snr_curve = np.where(finite_sig, signal_curve / noise_scale, 0.0)
    peak_snr = float(np.max(pixel_snr_curve)) if pixel_snr_curve.size else 0.0
    peak_bin_snr = float(np.max(snr_curve)) if snr_curve.size else 0.0

    # Determine q_max where SAXS excess/anisotropy merges into the active high-q diffuse floor
    cov_bins = np.flatnonzero((radial_cov >= 0.25) & np.isfinite(radial_med))
    q_cov_max = float(edges[int(cov_bins[-1]) + 1]) if cov_bins.size else active_max

    mean_active = np.isfinite(excess_mean) & (excess_mean >= max(2.0 * noise_scale, 0.002 * peak_signal))
    p90_active = np.isfinite(excess_p90) & (excess_p90 >= max(2.5 * noise_scale, 0.003 * peak_signal))
    aniso_active = np.isfinite(aniso_excess) & (
        aniso_excess >= max(2.0 * noise_scale, 0.005 * max(peak_aniso, 0.0))
    )
    pixel_sig_mask = (mean_active | p90_active | aniso_active) & (radial_cov >= 0.15)
    if np.any(pixel_sig_mask):
        active_sig_bins = np.flatnonzero(pixel_sig_mask)
        # Walk outward from strongest signal bin; stop after >=3 consecutive diffuse-floor bins
        # unless a genuine strong outer ring re-emerges.
        peak_bin_idx = int(active_sig_bins[int(np.argmax(signal_curve[active_sig_bins]))])
        strong_reemerge = (
            (np.isfinite(excess_mean) & (excess_mean >= max(3.5 * noise_scale, 0.04 * peak_signal)))
            | (np.isfinite(aniso_excess) & (aniso_excess >= max(3.5 * noise_scale, 0.06 * max(peak_aniso, 0.0))))
        ) & (radial_cov >= 0.15)
        last_sig_idx = peak_bin_idx
        quiet_run = 0
        for b in range(peak_bin_idx + 1, n_rbins):
            if pixel_sig_mask[b]:
                if quiet_run >= 3 and not strong_reemerge[b]:
                    break
                last_sig_idx = b
                quiet_run = 0
            elif np.isfinite(radial_med[b]):
                quiet_run += 1
        outer_sig_bins = active_sig_bins[active_sig_bins <= last_sig_idx]
    else:
        # Very low-amplitude regime: fall back to bin-averaged SNR
        outer_sig_bins = np.flatnonzero(finite_sig & (snr_curve >= 5.0) & (radial_cov >= 0.15))

    outer_signal_q: float | None = None
    if outer_sig_bins.size:
        last_sig_idx = int(outer_sig_bins[-1])
        outer_signal_q = float(edges[last_sig_idx + 1])
        q_max = min(q_cov_max, max(outer_signal_q * 1.15, float(edges[min(n_rbins, last_sig_idx + 3)])))
    else:
        q_max = q_cov_max

    # Detect direct I(q) peak and baseline-subtracted Kratky q^2*(I - I_bg) peak
    # on a fine grid strictly within [q_floor, q_max] (never on the flat diffuse floor!).
    q_star_direct: float | None = None
    q_star_kratky: float | None = None
    low_q_peak_near_beamstop = False

    def _find_best_peak(
        curve: np.ndarray,
        bin_centers: np.ndarray,
        bin_mask: np.ndarray,
        *,
        min_abs_prom: float = 0.0,
    ) -> tuple[float | None, bool]:
        idx_ok = np.flatnonzero(bin_mask & np.isfinite(curve))
        if idx_ok.size < 3:
            return None, False
        vals = curve[idx_ok]
        smooth = gaussian_filter1d(vals, sigma=1.0, mode="nearest")
        vmax = float(np.max(smooth))
        vmin = float(np.min(smooth))
        if vmax <= max(vmin + 1e-12, 0.0):
            return None, False
        prom_floor = max(0.05 * (vmax - vmin), min_abs_prom)
        local_peaks: list[tuple[float, int]] = []
        for j in range(1, smooth.size - 1):
            if smooth[j] >= smooth[j - 1] and smooth[j] >= smooth[j + 1]:
                if smooth[j] > smooth[j - 1] or smooth[j] > smooth[j + 1]:
                    left_min = float(np.min(smooth[: j + 1]))
                    right_min = float(np.min(smooth[j:]))
                    prom = smooth[j] - max(left_min, right_min)
                    if prom >= prom_floor:
                        local_peaks.append((float(smooth[j]), int(idx_ok[j])))
        if local_peaks:
            local_peaks.sort(key=lambda item: item[0], reverse=True)
            best_idx = local_peaks[0][1]
            near_inner = best_idx <= idx_ok[0] + 2
            return float(bin_centers[best_idx]), near_inner
        if int(np.argmax(smooth)) == 0:
            return None, True
        return None, False

    if valid_bins.size >= 3 and peak_signal > 0.0 and q_max > q_floor:
        if use_coarse_2d_stride:
            in_fine_2d = active_clean & (q_map >= q_floor) & (q_map <= q_max)
            fine_q = q_map[in_fine_2d]
            fine_i = intensity[in_fine_2d]
        else:
            in_fine = (q_vals >= q_floor) & (q_vals <= q_max)
            fine_q = q_vals[in_fine]
            fine_i = i_vals[in_fine]
        n_fine = int(np.clip(round((q_max - q_floor) / max(1.5 * pixel_dq, 1e-12)), 24, 96))
        fine_edges = np.linspace(q_floor, q_max, n_fine + 1, dtype=np.float64)
        fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
        fine_idx = ((fine_q - q_floor) * (float(n_fine) / max(q_max - q_floor, 1e-30))).astype(np.intp)
        fine_idx[fine_idx >= n_fine] = n_fine - 1
        fine_cnt = np.bincount(fine_idx, minlength=n_fine)
        fine_mean = np.full(n_fine, np.nan, dtype=np.float64)
        fine_p90 = np.full(n_fine, np.nan, dtype=np.float64)
        has_fine = fine_cnt > 0
        if np.any(has_fine):
            fine_sums = np.bincount(fine_idx, weights=fine_i, minlength=n_fine)
            fine_mean[has_fine] = fine_sums[has_fine] / fine_cnt[has_fine]
            fine_order = np.argsort(fine_idx.astype(np.uint8), kind="stable")
            fine_grouped = fine_i[fine_order]
            fine_ends = np.cumsum(fine_cnt)
            fine_starts = fine_ends - fine_cnt
            for fb in np.flatnonzero(has_fine):
                fine_p90[fb] = float(np.percentile(fine_grouped[fine_starts[fb] : fine_ends[fb]], 90.0))
        in_peak_win = fine_cnt >= 3
        if np.count_nonzero(in_peak_win) < 3:
            in_peak_win = fine_cnt > 0
        direct_curve = np.where(np.isfinite(fine_p90), fine_p90 - bg_level, np.nan)
        kratky_excess = np.where(
            np.isfinite(fine_mean) & ((fine_mean - bg_level) > 0.5 * noise_scale),
            fine_mean - bg_level,
            0.0,
        )
        k_curve = np.where(np.isfinite(fine_mean), (fine_centers ** 2) * kratky_excess, np.nan)
        q_star_direct, near_inner_direct = _find_best_peak(
            direct_curve, fine_centers, in_peak_win, min_abs_prom=2.0 * noise_scale
        )
        q_star_kratky, near_inner_kratky = _find_best_peak(k_curve, fine_centers, in_peak_win)
        low_q_peak_near_beamstop = bool(near_inner_direct or near_inner_kratky)

    # Choose clean q_min: right above safe beamstop radius, avoiding low-q peak clipping
    q_min = float(q_floor)
    if beamstop["detected"]:
        q_min = max(q_min, float(beamstop["safe_q_min"]))
        if q_star_direct is not None and q_min > 0.85 * q_star_direct:
            q_min = max(float(beamstop["shadow_edge_q"]) + pixel_dq, min(q_min, 0.80 * q_star_direct))
    else:
        sig_thresh_inner = max(3.0 * noise_scale, 0.05 * peak_signal)
        strong_bins = np.flatnonzero(finite_sig & (signal_curve >= sig_thresh_inner) & (obs_counts_raw >= 3))
        if strong_bins.size:
            first_sig_idx = int(strong_bins[0])
            inner_idx = max(0, first_sig_idx - max(2, int(round(0.15 * first_sig_idx))))
            q_min = float(edges[inner_idx])
        if q_star_direct is not None:
            q_min = min(q_min, 0.65 * q_star_direct)
        q_min = max(q_min, q_valid_min)

    if q_star_direct is not None:
        q_max = max(q_max, min(active_max, q_star_direct * 1.8))
    q_max = min(q_max, active_max)
    if q_max <= q_min:
        q_min = q_valid_min
        q_max = active_max

    # Focus q-band for azimuthal symmetry: avoid outer noise diluting harmonics
    if outer_sig_bins.size:
        strong_azim_bins = np.flatnonzero(
            finite_sig
            & (signal_curve >= max(2.0 * noise_scale, 0.10 * peak_signal))
            & (centers >= q_min)
            & (centers <= q_max)
        )
        if strong_azim_bins.size:
            azim_q_min = max(q_min, float(edges[int(strong_azim_bins[0])]))
            azim_q_max = min(q_max, float(edges[int(strong_azim_bins[-1]) + 1]))
            if azim_q_max <= azim_q_min:
                azim_q_min, azim_q_max = q_min, q_max
        else:
            azim_q_min, azim_q_max = q_min, q_max
    else:
        azim_q_min, azim_q_max = q_min, q_max

    rationale_parts: list[str] = []
    if beamstop["detected"]:
        rationale_parts.append(
            f"q_min={q_min:.4g} clears beamstop exclusion radius {rec_radius_px:.1f} px"
        )
    else:
        rationale_parts.append(f"q_min={q_min:.4g} preserves low-q signal without beamstop obstruction")
    if outer_zero_padded:
        rationale_parts.append(f"excluded outer zero-padded dead shells above q_active_max={active_max:.4g}")
    if outer_signal_q is not None:
        rationale_parts.append(
            f"q_max={q_max:.4g} covers observed signal above diffuse floor ({bg_level:.4g}) up to {outer_signal_q:.4g}"
        )
    else:
        rationale_parts.append(f"q_max={q_max:.4g} spans available active detector coverage")

    q_window_info = {
        "q_min": float(q_min),
        "q_max": float(q_max),
        "window": [float(q_min), float(q_max)],
        "q_unit": q_unit,
        "q_beamstop_safe": float(beamstop["safe_q_min"]),
        "q_valid_min": float(q_valid_min),
        "q_valid_max": float(q_valid_max),
        "q_active_max": float(active_max),
        "outer_zero_padded": bool(outer_zero_padded),
        "diffuse_floor_estimate": float(bg_level),
        "diffuse_noise_mad": float(bg_mad),
        "q_star_direct_estimate": q_star_direct,
        "q_star_kratky_estimate": q_star_kratky,
        "outer_signal_q": outer_signal_q,
        "low_q_peak_near_beamstop": low_q_peak_near_beamstop,
        "rationale": "; ".join(rationale_parts),
    }
    stats = {
        "clean_valid": active_clean,
        "bg_level": bg_level,
        "bg_mad": bg_mad,
        "noise_scale": noise_scale,
        "peak_signal": peak_signal,
        "peak_snr": peak_snr,
        "peak_bin_snr": peak_bin_snr,
        "azim_q_band": (float(azim_q_min), float(azim_q_max)),
        "radial_cov_median": float(np.median(radial_cov[valid_bins])) if valid_bins.size else 0.0,
    }
    return q_window_info, stats


def _normalized_circular_smooth(values: np.ndarray, valid_mask: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Smooth a periodic 1D profile strictly over observed bins (never zero-filling gaps)."""
    n = values.size
    weights = valid_mask.astype(np.float64)
    filled = np.where(valid_mask, values, 0.0)
    num = gaussian_filter1d(filled, sigma=sigma_bins, mode="wrap")
    den = gaussian_filter1d(weights, sigma=sigma_bins, mode="wrap")
    out = np.full(n, np.nan, dtype=np.float64)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def _fit_harmonics_observed(chi_centers_deg: np.ndarray, profile: np.ndarray) -> dict[str, Any]:
    """Least-squares m=0,2,4,6 Fourier harmonic fit on observed angular bins only."""
    ok = np.isfinite(profile)
    n_obs = int(np.count_nonzero(ok))
    if n_obs < 4:
        return {
            "n_valid_bins": n_obs,
            "A0": None,
            "A2": None,
            "A4": None,
            "A6": None,
            "a2": 0.0,
            "a4": 0.0,
            "a6": 0.0,
            "phi2_deg": None,
            "phi4_deg": None,
        }
    chi_rad = np.deg2rad(chi_centers_deg[ok])
    y = profile[ok]
    cols = [np.ones_like(chi_rad)]
    orders = [2, 4, 6] if n_obs >= 12 else ([2, 4] if n_obs >= 8 else [2])
    for m in orders:
        cols.append(np.cos(m * chi_rad))
        cols.append(np.sin(m * chi_rad))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    c0 = float(coef[0])
    amps: dict[int, float] = {2: 0.0, 4: 0.0, 6: 0.0}
    phases: dict[int, float | None] = {2: None, 4: None, 6: None}
    for idx, m in enumerate(orders):
        cm = float(coef[1 + 2 * idx])
        sm = float(coef[2 + 2 * idx])
        amps[m] = float(math.hypot(cm, sm))
        if amps[m] > 1e-15:
            phases[m] = float((math.degrees(math.atan2(sm, cm)) / m) % (360.0 / m))

    shift = min(0.0, float(np.min(y)))
    denom = max(c0 - shift, 1e-12)
    return {
        "n_valid_bins": n_obs,
        "A0": c0,
        "A2": amps[2],
        "A4": amps[4],
        "A6": amps[6],
        "a2": float(amps[2] / denom),
        "a4": float(amps[4] / denom),
        "a6": float(amps[6] / denom),
        "phi2_deg": phases[2],
        "phi4_deg": phases[4],
    }


def _find_circular_peaks(
    chi_deg: np.ndarray,
    profile: np.ndarray,
    *,
    min_prominence_frac: float = 0.12,
    min_sep_deg: float = 20.0,
) -> list[dict[str, float]]:
    """Find peaks and their FWHM on a periodic 360-deg observed profile."""
    n = profile.size
    step_deg = 360.0 / n
    valid = np.isfinite(profile)
    if np.count_nonzero(valid) < 6:
        return []
    vmax = float(np.nanmax(profile))
    vmin = float(np.nanmin(profile))
    dyn = vmax - vmin
    if dyn <= 1e-12:
        return []
    prom_thresh = min_prominence_frac * dyn

    candidates: list[tuple[float, int, float]] = []
    for i in range(n):
        if not valid[i]:
            continue
        prev_v = profile[(i - 1) % n]
        next_v = profile[(i + 1) % n]
        if not (np.isfinite(prev_v) and np.isfinite(next_v)):
            continue
        if profile[i] >= prev_v and profile[i] >= next_v and (profile[i] > prev_v or profile[i] > next_v):
            # Local prominence within +/- 45 deg
            half_win = max(2, int(round(45.0 / step_deg)))
            left_vals = [profile[(i - k) % n] for k in range(1, half_win + 1) if valid[(i - k) % n]]
            right_vals = [profile[(i + k) % n] for k in range(1, half_win + 1) if valid[(i + k) % n]]
            if not left_vals or not right_vals:
                continue
            local_base = max(min(left_vals), min(right_vals))
            prom = float(profile[i] - local_base)
            if prom >= prom_thresh:
                # Parabolic sub-bin refinement
                denom = prev_v - 2.0 * profile[i] + next_v
                delta = 0.5 * (prev_v - next_v) / denom if abs(denom) > 1e-15 else 0.0
                delta = float(np.clip(delta, -0.5, 0.5))
                angle = float((chi_deg[i] + delta * step_deg) % 360.0)
                candidates.append((float(profile[i]), i, angle))

    # Greedy non-maximum suppression by height
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, int, float]] = []
    for height_val, idx, angle in candidates:
        if all(_circular_angle_diff_360(angle, existing[2]) >= min_sep_deg for existing in selected):
            selected.append((height_val, idx, angle))

    # Measure FWHM for each selected peak; never fabricate a crossing at a masked gap.
    med_base = float(np.nanpercentile(profile, 20.0))
    peaks: list[dict[str, float]] = []
    for height_val, idx, angle in selected:
        half_level = med_base + 0.5 * (height_val - med_base)
        left_steps = 0.0
        left_crossed = False
        for step in range(1, n // 2):
            val = profile[(idx - step) % n]
            prev_val = profile[(idx - step + 1) % n]
            if not np.isfinite(val):
                left_crossed = False
                break
            if val <= half_level:
                frac = (prev_val - half_level) / max(prev_val - val, 1e-12)
                left_steps = float(step - 1 + np.clip(frac, 0.0, 1.0))
                left_crossed = True
                break
            left_steps = float(step)
        right_steps = 0.0
        right_crossed = False
        for step in range(1, n // 2):
            val = profile[(idx + step) % n]
            prev_val = profile[(idx + step - 1) % n]
            if not np.isfinite(val):
                right_crossed = False
                break
            if val <= half_level:
                frac = (prev_val - half_level) / max(prev_val - val, 1e-12)
                right_steps = float(step - 1 + np.clip(frac, 0.0, 1.0))
                right_crossed = True
                break
            right_steps = float(step)
        fwhm_deg = float((left_steps + right_steps) * step_deg) if (left_crossed and right_crossed) else float("nan")
        peaks.append(
            {
                "angle_deg": angle,
                "intensity": height_val,
                "fwhm_deg": fwhm_deg,
                "bin_index": float(idx),
            }
        )
    peaks.sort(key=lambda item: item["angle_deg"])
    return peaks


def _refine_reflection_bisector(
    chi_deg: np.ndarray,
    profile: np.ndarray,
    seed_deg: float,
    search_half_width_deg: float = 12.0,
) -> float:
    """Refine a reflection symmetry bisector strictly using observed opposite bin pairs."""
    n = profile.size
    valid = np.isfinite(profile)
    if np.count_nonzero(valid) < 8:
        return float(seed_deg % 180.0)
    # Search candidate bisectors every 0.5 deg around seed using linear
    # interpolation strictly between observed adjacent bins (never across gaps).
    offsets = np.linspace(-search_half_width_deg, search_half_width_deg, 49)
    step = 360.0 / n
    scored: list[tuple[float, float, float]] = []
    for off in offsets:
        cand = (seed_deg + float(off)) % 180.0
        # Reflect each valid bin across cand: reflected = (2*cand - chi) % 360
        refl_chi = (2.0 * cand - chi_deg) % 360.0
        pos = ((refl_chi - 0.5 * step) / step) % n
        idx0 = np.floor(pos).astype(np.int64) % n
        idx1 = (idx0 + 1) % n
        frac = pos - np.floor(pos)
        both = valid & valid[idx0] & valid[idx1]
        if np.count_nonzero(both) < 6:
            continue
        refl_val = (1.0 - frac[both]) * profile[idx0[both]] + frac[both] * profile[idx1[both]]
        diffs = profile[both] - refl_val
        err = float(np.mean(diffs * diffs))
        if math.isfinite(err):
            scored.append((err, float(off), cand))
    if not scored:
        return float(seed_deg % 180.0)
    min_err = min(item[0] for item in scored)
    tie_tol = 1e-12 * max(min_err, 1.0)
    tied = [item for item in scored if item[0] <= min_err + tie_tol]
    # Center tied minima symmetrically instead of biasing toward the leftmost offset.
    mean_off = float(np.mean([item[1] for item in tied]))
    return float((seed_deg + mean_off) % 180.0)


def _assess_streak_and_ray_geometry(
    intensity: np.ndarray,
    q_map: np.ndarray,
    chi_deg: np.ndarray,
    clean_valid: np.ndarray,
    q_window: tuple[float, float],
    noise_scale: float,
    bg_level: float,
    lobe_angles_deg: list[float],
) -> dict[str, Any]:
    """Check radial streak persistence and straight-ray vs curved-arc geometry."""
    q_lo, q_hi = q_window
    in_win = clean_valid & (q_map >= q_lo) & (q_map <= q_hi)
    n_q = 10
    n_chi = 72
    q_edges = np.linspace(q_lo, q_hi, n_q + 1, dtype=np.float64)
    chi_step = 360.0 / n_chi
    chi_centers = (np.arange(n_chi, dtype=np.float64) + 0.5) * chi_step

    polar = np.full((n_q, n_chi), np.nan, dtype=np.float64)
    if np.any(in_win):
        qv = q_map[in_win]
        chiv = np.mod(chi_deg[in_win], 360.0)
        iv = intensity[in_win]
        qi = np.clip(np.searchsorted(q_edges, qv, side="right") - 1, 0, n_q - 1)
        ci = np.clip((chiv / chi_step).astype(np.int64), 0, n_chi - 1)
        flat = qi * n_chi + ci
        sums = np.bincount(flat, weights=iv, minlength=n_q * n_chi)
        cnts = np.bincount(flat, minlength=n_q * n_chi)
        valid_cell = cnts > 0
        polar_flat = np.full(n_q * n_chi, np.nan, dtype=np.float64)
        polar_flat[valid_cell] = sums[valid_cell] / cnts[valid_cell]
        polar = polar_flat.reshape(n_q, n_chi)

    # Check sharp streak persistence across radial shells
    sharp_shell_hits = 0
    valid_shells = 0
    sharp_angles: list[float] = []
    for r in range(n_q):
        row = polar[r]
        ok = np.isfinite(row)
        if np.count_nonzero(ok) < 12:
            continue
        valid_shells += 1
        row_med = float(np.median(row[ok]))
        row_mad = float(1.4826 * np.median(np.abs(row[ok] - row_med)))
        row_max = float(np.max(row[ok]))
        denom = max(row_mad, noise_scale, 1e-12)
        if (row_max - row_med) / denom >= 6.0:
            # Count fraction of valid bins above half-height
            half = row_med + 0.5 * (row_max - row_med)
            frac_above = float(np.mean(row[ok] >= half))
            if frac_above <= 0.11:  # <= ~40 deg total across full 360 deg (e.g. <= 20 deg per side)
                sharp_shell_hits += 1
                sharp_angles.append(float(chi_centers[int(np.nanargmax(row))] % 180.0))

    streak_persistence = float(sharp_shell_hits / valid_shells) if valid_shells > 0 else 0.0
    streak_angle_aligned = False
    if len(sharp_angles) >= 3:
        ang_rad = np.deg2rad(np.asarray(sharp_angles) * 2.0)
        r_vec = float(math.hypot(float(np.mean(np.cos(ang_rad))), float(np.mean(np.sin(ang_rad)))))
        streak_angle_aligned = r_vec >= 0.90

    # Measure straight-ray vs curved-arc geometry along detected lobes
    ray_stds: list[float] = []
    ray_drifts: list[float] = []
    outer_active_lobes = 0
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    for lobe_deg in lobe_angles_deg:
        shell_q: list[float] = []
        shell_chi: list[float] = []
        for r in range(n_q):
            row = polar[r]
            ok = np.isfinite(row)
            if np.count_nonzero(ok) < 8:
                continue
            # Restrict to +/- 22 deg sector around the lobe
            d_ang = ((chi_centers - lobe_deg + 180.0) % 360.0) - 180.0
            in_sec = ok & (np.abs(d_ang) <= 22.0)
            if np.count_nonzero(in_sec) < 3:
                continue
            sec_vals = row[in_sec]
            sec_d = d_ang[in_sec]
            row_base = float(np.percentile(row[ok], 25.0))
            pk_local_idx = int(np.argmax(sec_vals))
            pk_val = float(sec_vals[pk_local_idx])
            if pk_val - row_base < max(2.5 * noise_scale, 0.08 * max(pk_val - bg_level, 0.0)):
                continue
            # Parabolic / weighted centroid around local peak in sector
            top_mask = sec_vals >= (row_base + 0.70 * (pk_val - row_base))
            weights = np.maximum(sec_vals[top_mask] - row_base, 0.0)
            if float(np.sum(weights)) <= 0.0:
                continue
            mean_off = float(np.sum(weights * sec_d[top_mask]) / np.sum(weights))
            shell_q.append(float(q_centers[r]))
            shell_chi.append(mean_off)
        if len(shell_chi) >= 4:
            q_arr = np.asarray(shell_q, dtype=np.float64)
            c_arr = np.asarray(shell_chi, dtype=np.float64)
            ray_stds.append(float(np.std(c_arr)))
            # Linear drift across observed radial span
            if float(np.ptp(q_arr)) > 0.0:
                slope = float(np.polyfit(q_arr, c_arr, 1)[0])
                ray_drifts.append(abs(slope * float(np.ptp(q_arr))))
            else:
                ray_drifts.append(0.0)
            if q_arr[-1] >= q_lo + 0.85 * (q_hi - q_lo):
                outer_active_lobes += 1

    med_std = float(np.median(ray_stds)) if ray_stds else None
    med_drift = float(np.median(ray_drifts)) if ray_drifts else None
    wing_reaches_outer_edge = bool(lobe_angles_deg and outer_active_lobes >= max(1, len(lobe_angles_deg) // 2))

    straight_ray_score: float | None = None
    if med_std is not None and med_drift is not None:
        # 1.0 = perfectly straight radial ray; 0.0 = strongly curved arc
        std_term = max(0.0, 1.0 - med_std / 8.0)
        drift_term = max(0.0, 1.0 - med_drift / 12.0)
        straight_ray_score = float(0.5 * (std_term + drift_term))

    return {
        "streak_persistence": streak_persistence,
        "streak_angle_aligned": streak_angle_aligned,
        "radial_ray_angle_std_deg": med_std,
        "radial_ray_angular_drift_deg": med_drift,
        "straight_ray_score": straight_ray_score,
        "wing_reaches_outer_edge": wing_reaches_outer_edge,
    }


def _analyze_symmetry_and_morphology(
    intensity: np.ndarray,
    q_map: np.ndarray,
    chi_deg: np.ndarray,
    radial_stats: Mapping[str, Any],
    q_window_info: Mapping[str, Any],
    nominal_draw_axis_deg: float,
    *,
    bright_axis_convention: str = "draw",
    expected_bright_axis_deg: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze azimuthal harmonics, Friedel bisector, mounting tilt, and morphology."""
    reference_axis_deg = float((nominal_draw_axis_deg - 90.0) % 180.0)
    if expected_bright_axis_deg is not None:
        expected_bright_axis = float(expected_bright_axis_deg % 180.0)
    elif bright_axis_convention == "reference":
        expected_bright_axis = reference_axis_deg
    else:
        expected_bright_axis = float(nominal_draw_axis_deg % 180.0)
    clean_valid = radial_stats["clean_valid"]
    bg_level = float(radial_stats["bg_level"])
    noise_scale = float(radial_stats["noise_scale"])
    peak_snr = float(radial_stats["peak_snr"])
    peak_bin_snr = float(radial_stats["peak_bin_snr"])
    azim_q_lo, azim_q_hi = radial_stats["azim_q_band"]
    q_lo, q_hi = float(q_window_info["q_min"]), float(q_window_info["q_max"])

    in_band = clean_valid & (q_map >= azim_q_lo) & (q_map <= azim_q_hi)
    if np.count_nonzero(in_band) < 16:
        in_band = clean_valid & (q_map >= q_lo) & (q_map <= q_hi)
    if np.count_nonzero(in_band) < 8:
        in_band = clean_valid

    n_chi = 72
    chi_step = 360.0 / n_chi
    chi_centers = (np.arange(n_chi, dtype=np.float64) + 0.5) * chi_step

    chiv = np.mod(chi_deg[in_band], 360.0)
    iv = intensity[in_band]
    c_idx = np.clip((chiv / chi_step).astype(np.int64), 0, n_chi - 1)
    counts = np.bincount(c_idx, minlength=n_chi)
    sums = np.bincount(c_idx, weights=iv, minlength=n_chi)
    raw_profile = np.full(n_chi, np.nan, dtype=np.float64)
    has_obs = counts > 0
    raw_profile[has_obs] = sums[has_obs] / counts[has_obs]

    # Harmonics from raw observed angular bins (strictly observed, zero gap-filling)
    harmonics = _fit_harmonics_observed(chi_centers, raw_profile)
    a2 = float(harmonics["a2"])
    a4 = float(harmonics["a4"])
    a6 = float(harmonics["a6"])
    phi2_deg = harmonics["phi2_deg"]
    phi4_deg = harmonics["phi4_deg"]

    # Smoothed profile for stable lobe peak finding; raw profile for razor-thin streak width
    smooth_profile = _normalized_circular_smooth(raw_profile, has_obs, sigma_bins=1.0)
    smooth_peaks = _find_circular_peaks(chi_centers, smooth_profile, min_prominence_frac=0.12, min_sep_deg=20.0)
    raw_peaks = _find_circular_peaks(chi_centers, raw_profile, min_prominence_frac=0.15, min_sep_deg=15.0)

    # Azimuthal modulation contrast
    obs_vals = raw_profile[has_obs]
    if obs_vals.size >= 4:
        p95 = float(np.percentile(obs_vals, 95.0))
        p05 = float(np.percentile(obs_vals, 5.0))
        shift = min(0.0, float(np.min(obs_vals)))
        azim_contrast = float((p95 - p05) / max((p95 - shift) + (p05 - shift), 1e-12))
    else:
        azim_contrast = 0.0

    # Pair peaks by Friedel symmetry (+180 deg)
    friedel_pairs: list[tuple[dict[str, float], dict[str, float]]] = []
    used: set[int] = set()
    for i, pk_i in enumerate(smooth_peaks):
        if i in used:
            continue
        best_j = None
        best_diff = 18.0
        for j in range(i + 1, len(smooth_peaks)):
            if j in used:
                continue
            target = (pk_i["angle_deg"] + 180.0) % 360.0
            diff = _circular_angle_diff_360(smooth_peaks[j]["angle_deg"], target)
            if diff <= best_diff:
                best_diff = diff
                best_j = j
        if best_j is not None:
            used.add(i)
            used.add(best_j)
            friedel_pairs.append((pk_i, smooth_peaks[best_j]))

    lobe_angles = [pk["angle_deg"] for pk in smooth_peaks[:6]]
    ray_info = _assess_streak_and_ray_geometry(
        intensity,
        q_map,
        chi_deg,
        clean_valid,
        (q_lo, q_hi),
        noise_scale,
        bg_level,
        lobe_angles,
    )

    # Determine narrowest peak FWHM (both raw and smoothed), skipping gap-interrupted NaNs
    raw_min_fwhm = min(
        (pk["fwhm_deg"] for pk in raw_peaks if math.isfinite(pk["fwhm_deg"])),
        default=None,
    )
    smooth_min_fwhm = min(
        (pk["fwhm_deg"] for pk in smooth_peaks if math.isfinite(pk["fwhm_deg"])),
        default=None,
    )
    max_raw_peak = max((pk["intensity"] for pk in raw_peaks), default=0.0)
    raw_med = float(np.nanmedian(raw_profile)) if np.any(has_obs) else 0.0
    raw_mad = float(1.4826 * np.nanmedian(np.abs(raw_profile[has_obs] - raw_med))) if np.any(has_obs) else 0.0
    raw_peak_z = (max_raw_peak - raw_med) / max(raw_mad, noise_scale, 1e-12)

    # Check if there is a razor-thin parasitic streak (1 or 2 very sharp spikes persisting across q)
    has_unpaired_dominant_spike = False
    if raw_peaks:
        top_raw = max(raw_peaks, key=lambda p: p["intensity"])
        opp_angle = (top_raw["angle_deg"] + 180.0) % 360.0
        opp_bin = int(round((opp_angle - 0.5 * chi_step) / chi_step)) % n_chi
        if has_obs[opp_bin]:
            opp_val = float(raw_profile[opp_bin])
            if (
                math.isfinite(top_raw["fwhm_deg"])
                and top_raw["fwhm_deg"] <= 14.0
                and (top_raw["intensity"] - raw_med) >= 3.0 * max(opp_val - raw_med, noise_scale)
                and raw_peak_z >= 8.0
            ):
                has_unpaired_dominant_spike = True

    is_narrow_persistent_streak = bool(
        (raw_min_fwhm is not None and raw_min_fwhm <= 12.0)
        and raw_peak_z >= 7.0
        and ray_info["streak_persistence"] >= 0.60
        and ray_info["streak_angle_aligned"]
        and (
            len(friedel_pairs) <= 1
            or (a6 >= 0.35 * max(a2, 1e-9) and len(friedel_pairs) < 2)
            or (
                # Razor-thin spike superimposed on a weaker butterfly/ring
                len(raw_peaks) >= 1
                and max_raw_peak - raw_med >= 2.5 * max(float(np.percentile(obs_vals, 90.0)) - raw_med, noise_scale)
            )
        )
    )

    # Check low signal (handles both pure noise and flat/zero images)
    is_low_signal = bool(
        float(radial_stats["peak_signal"]) <= 1e-12
        or (peak_snr < 3.0 and peak_bin_snr < 6.0)
    )

    # Classify morphology and compute principal Friedel symmetry bisector
    friedel_bisector_deg: float | None = None
    mounting_tilt_deg: float | None = None
    angle_to_draw_axis_deg: float | None = None
    angle_to_reference_axis_deg: float | None = None
    orthogonal_residual_tilt_deg: float | None = None
    orthogonal_flip_detected = False
    wing_half_opening_deg: float | None = None
    bright_bridge_axis_deg: float | None = None
    dark_notch_axis_deg: float | None = None

    if is_low_signal:
        morphology_class = "low_signal"
    elif has_unpaired_dominant_spike or is_narrow_persistent_streak:
        morphology_class = "streak_contaminated"
        if phi2_deg is not None:
            friedel_bisector_deg = _refine_reflection_bisector(chi_centers, smooth_profile, phi2_deg)
    elif (
        a2 < 0.08
        and a4 < 0.08
        and azim_contrast < 0.14
        and len(friedel_pairs) == 0
    ):
        morphology_class = "isotropic_ring"
    elif (len(friedel_pairs) >= 2 and (a4 >= 0.06 or azim_contrast >= 0.14)) or (
        a4 >= 0.12 and a4 >= 0.45 * a2 and len(smooth_peaks) >= 3
    ):
        # Four-lobe butterfly: determine acute bisector from the top two Friedel pairs
        if len(friedel_pairs) >= 2:
            ranked_pairs = sorted(
                friedel_pairs,
                key=lambda pr: pr[0]["intensity"] + pr[1]["intensity"],
                reverse=True,
            )[:2]
            pair_axes = sorted(
                0.5 * (pr[0]["angle_deg"] % 180.0 + pr[1]["angle_deg"] % 180.0)
                if abs((pr[0]["angle_deg"] % 180.0) - (pr[1]["angle_deg"] % 180.0)) <= 90.0
                else (
                    0.5 * ((pr[0]["angle_deg"] % 180.0) + (pr[1]["angle_deg"] % 180.0) + 180.0)
                ) % 180.0
                for pr in ranked_pairs
            )
            a_lo, a_hi = pair_axes[0], pair_axes[1]
            gap_inner = a_hi - a_lo
            gap_outer = 180.0 - gap_inner
            if gap_inner <= gap_outer:
                b_acute = 0.5 * (a_lo + a_hi)
                half_open = 0.5 * gap_inner
            else:
                b_acute = (0.5 * (a_hi + a_lo + 180.0)) % 180.0
                half_open = 0.5 * gap_outer
            b_obtuse = (b_acute + 90.0) % 180.0

            # When lobes sit near +/-45 deg (37..53 deg), align to the brighter m=2 C2 bridge
            # so small angular drifts across 45 deg never induce spurious 90-deg axis jumps.
            if abs(half_open - 45.0) <= 8.0 and phi2_deg is not None and a2 >= 0.05:
                d_ac = abs(_signed_angle_diff_mod180(b_acute, phi2_deg))
                d_ob = abs(_signed_angle_diff_mod180(b_obtuse, phi2_deg))
                seed_bisector = b_acute if d_ac <= d_ob else b_obtuse
                wing_half_opening_deg = half_open if d_ac <= d_ob else (90.0 - half_open)
            else:
                seed_bisector = b_acute
                wing_half_opening_deg = half_open
            friedel_bisector_deg = _refine_reflection_bisector(
                chi_centers, smooth_profile, seed_bisector, search_half_width_deg=10.0
            )
        else:
            # Fallback when one lobe is masked by a detector gap: use phi4 / phi2
            if phi2_deg is not None and a2 >= 0.05:
                seed_bisector = phi2_deg
            elif phi4_deg is not None:
                cands = [(phi4_deg + 45.0 * k) % 180.0 for k in range(4)]
                seed_bisector = min(cands, key=lambda c: abs(_signed_angle_diff_mod180(c, expected_bright_axis)))
            else:
                seed_bisector = expected_bright_axis
            friedel_bisector_deg = _refine_reflection_bisector(
                chi_centers, smooth_profile, seed_bisector, search_half_width_deg=12.0
            )

        angle_to_draw_axis_deg = _signed_angle_diff_mod180(friedel_bisector_deg, nominal_draw_axis_deg)
        angle_to_reference_axis_deg = _signed_angle_diff_mod180(friedel_bisector_deg, reference_axis_deg)
        mounting_tilt_deg = _signed_angle_diff_mod180(friedel_bisector_deg, expected_bright_axis)
        orthogonal_residual_tilt_deg = _signed_angle_diff_mod180(
            friedel_bisector_deg, expected_bright_axis + 90.0
        )
        # Orthogonal flip: principal symmetry axis rotated ~90 deg from expected bright axis
        orthogonal_flip_detected = abs(abs(mounting_tilt_deg) - 90.0) <= 25.0
        morphology_class = (
            "orthogonal_flipped_butterfly" if orthogonal_flip_detected else "four_lobe_butterfly"
        )
    elif a2 >= 0.08 or len(friedel_pairs) == 1 or azim_contrast >= 0.14:
        morphology_class = "two_lobe"
        seed = phi2_deg if phi2_deg is not None else (
            friedel_pairs[0][0]["angle_deg"] % 180.0 if friedel_pairs else expected_bright_axis
        )
        friedel_bisector_deg = _refine_reflection_bisector(
            chi_centers, smooth_profile, seed, search_half_width_deg=12.0
        )
        angle_to_draw_axis_deg = _signed_angle_diff_mod180(friedel_bisector_deg, nominal_draw_axis_deg)
        angle_to_reference_axis_deg = _signed_angle_diff_mod180(friedel_bisector_deg, reference_axis_deg)
        mounting_tilt_deg = _signed_angle_diff_mod180(friedel_bisector_deg, expected_bright_axis)
        orthogonal_residual_tilt_deg = _signed_angle_diff_mod180(
            friedel_bisector_deg, expected_bright_axis + 90.0
        )
        orthogonal_flip_detected = abs(abs(mounting_tilt_deg) - 90.0) <= 25.0
    else:
        morphology_class = "isotropic_ring"

    if morphology_class == "streak_contaminated" and friedel_bisector_deg is not None:
        angle_to_draw_axis_deg = _signed_angle_diff_mod180(friedel_bisector_deg, nominal_draw_axis_deg)
        angle_to_reference_axis_deg = _signed_angle_diff_mod180(friedel_bisector_deg, reference_axis_deg)
        mounting_tilt_deg = _signed_angle_diff_mod180(friedel_bisector_deg, expected_bright_axis)
        orthogonal_residual_tilt_deg = _signed_angle_diff_mod180(
            friedel_bisector_deg, expected_bright_axis + 90.0
        )
        orthogonal_flip_detected = abs(abs(mounting_tilt_deg) - 90.0) <= 25.0

    if friedel_bisector_deg is not None and np.any(has_obs):
        def _axis_saddle(axis_deg: float) -> float | None:
            d0 = ((chi_centers - axis_deg + 180.0) % 360.0) - 180.0
            d1 = ((chi_centers - (axis_deg + 180.0) + 180.0) % 360.0) - 180.0
            near = has_obs & ((np.abs(d0) <= 10.0) | (np.abs(d1) <= 10.0))
            return float(np.mean(raw_profile[near])) if np.any(near) else None

        orth_axis = (friedel_bisector_deg + 90.0) % 180.0
        s_main = _axis_saddle(friedel_bisector_deg)
        s_orth = _axis_saddle(orth_axis)
        if s_main is not None and s_orth is not None:
            if s_main >= s_orth:
                bright_bridge_axis_deg = float(friedel_bisector_deg)
                dark_notch_axis_deg = float(orth_axis)
            else:
                bright_bridge_axis_deg = float(orth_axis)
                dark_notch_axis_deg = float(friedel_bisector_deg)

    # Opposite-quadrant Friedel symmetry ratio from observed pixels only
    quad_means: list[float | None] = []
    for q_idx in range(4):
        q_start = q_idx * 90.0
        q_end = (q_idx + 1) * 90.0
        in_q = has_obs & (chi_centers >= q_start) & (chi_centers < q_end)
        quad_means.append(float(np.mean(raw_profile[in_q])) if np.any(in_q) else None)
    q1_over_q3 = (
        float(quad_means[0] / quad_means[2])
        if (quad_means[0] is not None and quad_means[2] not in (None, 0.0))
        else None
    )
    q2_over_q4 = (
        float(quad_means[1] / quad_means[3])
        if (quad_means[1] is not None and quad_means[3] not in (None, 0.0))
        else None
    )

    symmetry = {
        "morphology_class": morphology_class,
        "bright_axis_convention": bright_axis_convention,
        "nominal_draw_axis_deg": float(nominal_draw_axis_deg),
        "nominal_reference_axis_deg": reference_axis_deg,
        "expected_bright_axis_deg": expected_bright_axis,
        "friedel_bisector_deg": friedel_bisector_deg,
        "angle_to_draw_axis_deg": angle_to_draw_axis_deg,
        "angle_to_reference_axis_deg": angle_to_reference_axis_deg,
        "mounting_tilt_deg": mounting_tilt_deg,
        "orthogonal_residual_tilt_deg": orthogonal_residual_tilt_deg,
        "orthogonal_flip_detected": bool(orthogonal_flip_detected),
        "wing_half_opening_deg": wing_half_opening_deg,
        "bright_bridge_axis_deg": bright_bridge_axis_deg,
        "dark_notch_axis_deg": dark_notch_axis_deg,
        "harmonics": {
            "A0": harmonics["A0"],
            "A2": harmonics["A2"],
            "A4": harmonics["A4"],
            "A6": harmonics["A6"],
            "a2": a2,
            "a4": a4,
            "a6": a6,
            "a4_over_a2": float(a4 / a2) if a2 > 1e-9 else None,
            "phi2_deg": phi2_deg,
            "phi4_deg": phi4_deg,
        },
        "azimuthal_contrast": azim_contrast,
        "n_observed_angular_bins": int(harmonics["n_valid_bins"]),
        "detected_lobe_angles_deg": [float(pk["angle_deg"]) for pk in smooth_peaks],
        "min_lobe_fwhm_deg": smooth_min_fwhm,
        "friedel_q1_q3_ratio": q1_over_q3,
        "friedel_q2_q4_ratio": q2_over_q4,
    }
    return symmetry, ray_info


def _recommend_parameters(
    beamstop: Mapping[str, Any],
    q_window_info: Mapping[str, Any],
    symmetry: Mapping[str, Any],
    ray_info: Mapping[str, Any],
    radial_stats: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Recommend Nyquist-safe bins, workflow, draw_axis, and identifiability risk."""
    pixel_dq = float(beamstop["pixel_dq"])
    q_min = float(q_window_info["q_min"])
    q_max = float(q_window_info["q_max"])
    morphology = str(symmetry["morphology_class"])

    # Nyquist-safe annular_radial_bins: >= 2 pixels per radial bin, within [4, 192]
    dr_px = max((q_max - q_min) / max(pixel_dq, 1e-12), 1.0)
    nyquist_max_radial = max(4, int(math.floor(dr_px / 2.0)))
    target_radial = int(round(dr_px / 3.0))
    annular_radial_bins = int(np.clip(target_radial, 4, min(96, nyquist_max_radial)))
    annular_radial_bins = max(4, min(192, annular_radial_bins))

    # Nyquist-safe annular_angle_bins: >= 2 arc pixels per bin at mid-window radius, within [16, 720]
    r_mid_px = max(0.5 * (q_min + q_max) / max(pixel_dq, 1e-12), 4.0)
    nyquist_max_angle = max(16, int(math.floor(math.pi * r_mid_px)))
    # Choose the largest clean divisor <= min(180, nyquist_max_angle)
    angle_cap = min(180, nyquist_max_angle)
    valid_angle_choices = [b for b in _ALLOWED_ANGLE_BINS if b <= angle_cap]
    annular_angle_bins = int(valid_angle_choices[-1]) if valid_angle_choices else 16

    # Assess ellipse identifiability risk (especially straight radial X-streak b/a collapse)
    straight_score = ray_info.get("straight_ray_score")
    ray_std = ray_info.get("radial_ray_angle_std_deg")
    ray_drift = ray_info.get("radial_ray_angular_drift_deg")
    wing_outer = bool(ray_info.get("wing_reaches_outer_edge", False))

    if morphology in {"isotropic_ring", "low_signal", "streak_contaminated"}:
        risk = "high"
        if morphology == "isotropic_ring":
            risk_reason = "Near-circular ring lacks an identifiable ellipse major-axis orientation."
        elif morphology == "low_signal":
            risk_reason = "Insufficient signal-to-noise ratio to constrain ellipse geometry."
        else:
            risk_reason = "Parasitic radial streak contaminates azimuthal geometry."
    elif morphology == "two_lobe":
        risk = "moderate"
        risk_reason = "Two-lobe pattern provides only two quadrants of support; four-side butterfly ellipse is under-constrained."
    else:
        # Four-lobe or orthogonal-flipped butterfly
        if (
            (straight_score is not None and straight_score >= 0.65)
            or (
                ray_std is not None
                and ray_drift is not None
                and ray_std <= 3.5
                and ray_drift <= 6.0
            )
        ):
            risk = "high"
            risk_reason = (
                "Observed butterfly wings follow near-straight radial rays; origin-centered "
                "very_flat_ellipse fitting risks b/a lower-bound collapse and unphysical Ln explosion."
            )
        elif wing_outer or (straight_score is not None and straight_score >= 0.40):
            risk = "moderate"
            risk_reason = (
                "Wings show low curvature or reach the outer q_window edge; verify major-axis tip "
                "and check bound_flags."
            )
        else:
            risk = "low"
            risk_reason = "Curved four-lobe butterfly arcs stay within observed q_window support."

    # Recommend workflow: annular, curvature, or sector
    if morphology in {"isotropic_ring", "two_lobe", "streak_contaminated", "low_signal"}:
        recommended_workflow = "sector"
        recommended_trace_method = "radial_sector"
    elif risk == "low" and float(radial_stats["peak_snr"]) >= 8.0:
        recommended_workflow = "curvature"
        recommended_trace_method = "curvature"
    else:
        recommended_workflow = "annular"
        recommended_trace_method = "annular_peak"

    if risk == "high" and morphology in {"four_lobe_butterfly", "orthogonal_flipped_butterfly"}:
        recommended_ellipse_preset = "flat_ellipse"
    elif morphology in {"four_lobe_butterfly", "orthogonal_flipped_butterfly"}:
        recommended_ellipse_preset = "standard"
    else:
        recommended_ellipse_preset = "standard"

    if symmetry["friedel_bisector_deg"] is not None:
        if symmetry.get("bright_axis_convention") == "reference":
            recommended_draw_axis_deg = float((float(symmetry["friedel_bisector_deg"]) + 90.0) % 180.0)
        else:
            recommended_draw_axis_deg = float(symmetry["friedel_bisector_deg"])
    else:
        recommended_draw_axis_deg = float(symmetry["nominal_draw_axis_deg"])

    # Only flag outer window truncation when wing signal genuinely reaches the active detector limit
    q_active_max = _finite_float(q_window_info.get("q_active_max", q_window_info.get("q_valid_max")))
    outer_sig_q = _finite_float(q_window_info.get("outer_signal_q"))
    detector_outer_truncated = bool(
        wing_outer
        and q_active_max is not None
        and q_active_max > 0.0
        and outer_sig_q is not None
        and outer_sig_q >= 0.90 * q_active_max
    )

    # Assemble active warnings
    warnings: list[str] = []
    if beamstop["unmasked_beamstop"]:
        warnings.append("beamstop_unmasked_low_q")
    if q_window_info["low_q_peak_near_beamstop"]:
        warnings.append("low_q_edge_hit")
    if symmetry["orthogonal_flip_detected"]:
        warnings.append("orthogonal_axis_flip")
    if morphology == "isotropic_ring":
        warnings.append("near_circular_axis_angle_unconstrained")
    elif morphology == "streak_contaminated":
        warnings.append("streak_contaminated")
    elif morphology == "low_signal":
        warnings.append("low_signal")
    if (
        morphology in {"four_lobe_butterfly", "orthogonal_flipped_butterfly"}
        and risk == "high"
    ):
        warnings.append("axis_ratio_at_bound")
    if detector_outer_truncated and morphology in {"four_lobe_butterfly", "orthogonal_flipped_butterfly"}:
        warnings.append("annular_outer_window_truncated")
    if _normalise_q_unit(q_window_info["q_unit"]) not in {"nm^-1", "Å^-1"}:
        warnings.append("uncalibrated_pixel_q")

    parameters = {
        "q_window": [float(q_min), float(q_max)],
        "draw_axis_deg": float(recommended_draw_axis_deg),
        "annular_radial_bins": annular_radial_bins,
        "annular_angle_bins": annular_angle_bins,
        "nyquist_max_radial_bins": nyquist_max_radial,
        "nyquist_max_angle_bins": nyquist_max_angle,
        "recommended_workflow": recommended_workflow,
        "recommended_trace_method": recommended_trace_method,
        "recommended_ellipse_preset": recommended_ellipse_preset,
        "ellipse_identifiability_risk": risk,
        "ellipse_identifiability": {
            "risk": risk,
            "reason": risk_reason,
            "straight_ray_score": straight_score,
            "radial_ray_angle_std_deg": ray_std,
            "radial_ray_angular_drift_deg": ray_drift,
            "wing_reaches_outer_edge": wing_outer,
            "detector_outer_truncated": detector_outer_truncated,
        },
    }
    return parameters, list(dict.fromkeys(warnings))


def _normalise_bright_axis_convention(value: str) -> str:
    text = str(value or "draw").strip().lower()
    if text not in {"draw", "reference"}:
        raise ValueError(
            f"bright_axis_convention must be 'draw' or 'reference', got {value!r}"
        )
    return text


def advise_from_arrays(
    intensity: Any,
    q_map: Any,
    chi_deg: Any,
    valid_mask: Any,
    *,
    q_unit: str,
    beam_center_xy: Sequence[float] | None = None,
    pixel_dq: float | None = None,
    nominal_draw_axis_deg: float = 90.0,
    bright_axis_convention: str = "draw",
    expected_bright_axis_deg: float | None = None,
    beamstop_radius_px: float | None = None,
    geometry_cache: Any = None,
) -> dict[str, Any]:
    """Recommend beamstop mask, q_window, symmetry axis, and workflow from 2D arrays.

    Strictly uses observed valid pixels (``valid_mask=True`` and finite values).
    Returns a strict-JSON-serializable dictionary (``allow_nan=False``).
    """
    if isinstance(nominal_draw_axis_deg, bool):
        raise ValueError("nominal_draw_axis_deg must be a finite number")
    nominal_axis = _finite_float(nominal_draw_axis_deg)
    if nominal_axis is None:
        raise ValueError(f"nominal_draw_axis_deg must be finite, got {nominal_draw_axis_deg!r}")
    convention = _normalise_bright_axis_convention(bright_axis_convention)
    expected_axis_val: float | None = None
    if expected_bright_axis_deg is not None:
        if isinstance(expected_bright_axis_deg, bool):
            raise ValueError("expected_bright_axis_deg must be a finite number or None")
        expected_axis_val = _finite_float(expected_bright_axis_deg)
        if expected_axis_val is None:
            raise ValueError(
                f"expected_bright_axis_deg must be finite, got {expected_bright_axis_deg!r}"
            )

    i_arr = np.asarray(intensity, dtype=np.float64)
    q_arr = np.asarray(q_map, dtype=np.float64)
    c_arr = np.asarray(chi_deg, dtype=np.float64)
    if i_arr.ndim != 2 or i_arr.size == 0:
        raise ValueError(f"intensity must be a non-empty 2-D array, got shape {i_arr.shape!r}")
    if q_arr.shape != i_arr.shape:
        raise ValueError(f"q_map shape {q_arr.shape!r} does not match intensity shape {i_arr.shape!r}")
    if c_arr.shape != i_arr.shape:
        raise ValueError(f"chi_deg shape {c_arr.shape!r} does not match intensity shape {i_arr.shape!r}")

    if valid_mask is None:
        v_arr = np.ones(i_arr.shape, dtype=bool)
    else:
        v_arr = np.asarray(valid_mask, dtype=bool)
        if v_arr.shape != i_arr.shape:
            raise ValueError(
                f"valid_mask shape {v_arr.shape!r} does not match intensity shape {i_arr.shape!r}"
            )

    if geometry_cache is not None:
        if beam_center_xy is None and hasattr(geometry_cache, "beam_center_xy"):
            beam_center_xy = geometry_cache.beam_center_xy
        if pixel_dq is None and hasattr(geometry_cache, "pixel_dq"):
            pixel_dq = geometry_cache.pixel_dq

    center_xy, dq, center_source = _infer_beam_geometry(
        q_arr,
        c_arr,
        beam_center_xy=beam_center_xy,
        pixel_dq=pixel_dq,
    )
    if beamstop_radius_px is not None:
        v_arr, _ = apply_beamstop_mask(v_arr, i_arr.shape, center_xy, beamstop_radius_px)

    obs_valid = v_arr & np.isfinite(i_arr) & np.isfinite(q_arr) & np.isfinite(c_arr)
    if not np.any(obs_valid):
        raise ValueError("analysis domain is empty: no finite observed pixels in valid_mask")

    unit = _normalise_q_unit(q_unit)
    # On large 2D frames (>=512x512), extract uniform 2x2 sub-grid once (~1 ms)
    # for outer-cutoff & high-q background floor; keep full 1x1 resolution for beamstop & symmetry.
    if i_arr.shape[0] >= 512 and i_arr.shape[1] >= 512:
        ov_2d = obs_valid[::2, ::2]
        q_obs_stat = q_arr[::2, ::2][ov_2d]
        i_obs_stat = i_arr[::2, ::2][ov_2d]
        if q_obs_stat.size < 64:
            q_obs_stat = q_arr[obs_valid]
            i_obs_stat = i_arr[obs_valid]
    else:
        q_obs_stat = q_arr[obs_valid]
        i_obs_stat = i_arr[obs_valid]
    q_active_max, outer_zero_padded = _detect_active_outer_q_cutoff(
        i_arr, q_arr, obs_valid, dq, q_obs=q_obs_stat, i_obs=i_obs_stat
    )
    beamstop = _detect_beamstop(
        i_arr, q_arr, obs_valid, center_xy, dq, q_active_max=q_active_max, q_obs=q_obs_stat, i_obs=i_obs_stat
    )
    beamstop["beam_center_source"] = center_source

    q_window_info, radial_stats = _radial_profile_and_window(
        i_arr,
        q_arr,
        obs_valid,
        beamstop,
        unit,
        q_active_max=q_active_max,
        outer_zero_padded=outer_zero_padded,
    )
    symmetry, ray_info = _analyze_symmetry_and_morphology(
        i_arr,
        q_arr,
        c_arr,
        radial_stats,
        q_window_info,
        nominal_axis,
        bright_axis_convention=convention,
        expected_bright_axis_deg=expected_axis_val,
    )
    parameters, warnings = _recommend_parameters(
        beamstop, q_window_info, symmetry, ray_info, radial_stats
    )

    remediation_metrics = {
        "q_min": q_window_info["q_min"],
        "q_max": q_window_info["q_max"],
        "q_active_max": q_window_info["q_active_max"],
        "q_valid_max": q_window_info["q_valid_max"],
        "recommended_q_min": q_window_info["q_min"],
        "recommended_mask_radius_px": beamstop["recommended_mask_radius_px"],
        "beam_center_xy": beamstop["beam_center_xy"],
        "friedel_bisector_deg": symmetry["friedel_bisector_deg"],
        "recommended_draw_axis_deg": parameters["draw_axis_deg"],
        "observed_q_extent": q_window_info["outer_signal_q"],
    }
    remediation = build_quality_remediation(warnings, metrics=remediation_metrics)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "beamstop": beamstop,
        "q_window": q_window_info,
        "symmetry": symmetry,
        "parameters": parameters,
        "warnings": warnings,
        "remediation": remediation,
    }
    return _sanitize_json(payload)


def advise_frame(
    image_path: str | Path,
    *,
    poni_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    image_dataset: str | None = None,
    image_frame_index: int | None = None,
    mask_dataset: str | None = None,
    mask_frame_index: int | None = None,
    nominal_draw_axis_deg: float = 90.0,
    bright_axis_convention: str = "draw",
    expected_bright_axis_deg: float | None = None,
    beamstop_radius_px: float | None = None,
    geometry_cache: Any = None,
) -> dict[str, Any]:
    """Load one detector frame from disk and compute smart analysis recommendations."""
    from .io import load_image

    source = Path(image_path)
    include_qmap = source.suffix.lower() == ".npz"
    loaded = load_image(
        source,
        frame=image_frame_index,
        dataset=image_dataset,
        external_mask=mask_path,
        mask_frame=mask_frame_index,
        mask_dataset=mask_dataset,
        include_qmap=include_qmap,
    )
    image = np.asarray(loaded.data, dtype=np.float64)
    shape = image.shape
    valid_mask = (
        np.asarray(loaded.valid_mask, dtype=bool)
        if loaded.valid_mask is not None
        else np.ones(shape, dtype=bool)
    )

    beam_center_xy: tuple[float, float] | None = None
    pixel_dq: float | None = None
    if geometry_cache is not None:
        if geometry_cache.shape != shape:
            raise ValueError(
                f"geometry_cache shape {geometry_cache.shape!r} does not match image shape {shape!r}"
            )
        geom_obj = geometry_cache.geometry
        if hasattr(geom_obj, "q_nm_inv"):
            q_map = geom_obj.q_nm_inv
            chi_deg = (
                geom_obj.chi_deg
                if geom_obj.chi_deg is not None
                else np.degrees(geom_obj.chi_rad) % 360.0
            )
            q_unit = geom_obj.q_unit
        elif isinstance(geom_obj, Mapping) and "qx" in geom_obj and "qy" in geom_obj:
            qx = np.asarray(geom_obj["qx"], dtype=np.float64)
            qy = np.asarray(geom_obj["qy"], dtype=np.float64)
            q_map = (
                np.asarray(geom_obj["q"], dtype=np.float64)
                if "q" in geom_obj
                else np.hypot(qx, qy)
            )
            chi_deg = (
                np.asarray(geom_obj["chi_deg"], dtype=np.float64)
                if "chi_deg" in geom_obj
                else np.degrees(np.arctan2(qy, qx)) % 360.0
            )
            q_unit = str(getattr(geometry_cache, "q_unit", "nm^-1") or "nm^-1")
        else:
            if not getattr(geometry_cache, "is_full_frame_roi", False) or geometry_cache.q_mag_roi.shape != shape:
                raise ValueError(
                    "geometry_cache with cropped ROI must retain a full-frame geometry object for advise_frame"
                )
            q_map = geometry_cache.q_mag_roi
            chi_deg = geometry_cache.chi_deg_roi
            q_unit = str(getattr(geometry_cache, "q_unit", "unknown") or "unknown")
        valid_mask = valid_mask & geometry_cache.valid_mask
        beam_center_xy = geometry_cache.beam_center_xy
        pixel_dq = geometry_cache.pixel_dq
    elif poni_path is not None:
        from .geometry import build_geometry, load_poni

        # Load PONI strictly ONCE and reuse the AzimuthalIntegrator for both geometry and Fit2D center
        ai = load_poni(poni_path)
        geom = build_geometry(shape, ai, valid_mask=valid_mask)
        q_map = geom.q_nm_inv
        chi_deg = geom.chi_deg if geom.chi_deg is not None else np.degrees(geom.chi_rad) % 360.0
        valid_mask = valid_mask & geom.valid_mask
        q_unit = geom.q_unit
        try:
            fit2d = ai.getFit2D()
            cx = _finite_float(fit2d.get("centerX"))
            cy = _finite_float(fit2d.get("centerY"))
            if cx is not None and cy is not None:
                beam_center_xy = (cx, cy)
        except Exception:
            beam_center_xy = None
    elif isinstance(loaded.qmap, Mapping) and "qx" in loaded.qmap and "qy" in loaded.qmap:
        qx = np.asarray(loaded.qmap["qx"], dtype=np.float64)
        qy = np.asarray(loaded.qmap["qy"], dtype=np.float64)
        q_map = (
            np.asarray(loaded.qmap["q"], dtype=np.float64)
            if "q" in loaded.qmap
            else np.hypot(qx, qy)
        )
        chi_deg = np.degrees(np.arctan2(qy, qx)) % 360.0
        q_unit = str(loaded.qmap.get("q_unit", "pixel-q") or "pixel-q")
    else:
        height, width = shape
        cx, cy = 0.5 * (width - 1), 0.5 * (height - 1)
        dy = np.arange(height, dtype=np.float64) - cy
        dx = np.arange(width, dtype=np.float64) - cx
        q_map = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
        chi_deg = np.degrees(np.arctan2(dy[:, None], dx[None, :])) % 360.0
        q_unit = "pixel-q"
        beam_center_xy = (cx, cy)
        pixel_dq = 1.0

    result = advise_from_arrays(
        image,
        q_map,
        chi_deg,
        valid_mask,
        q_unit=q_unit,
        beam_center_xy=beam_center_xy,
        pixel_dq=pixel_dq,
        nominal_draw_axis_deg=nominal_draw_axis_deg,
        bright_axis_convention=bright_axis_convention,
        expected_bright_axis_deg=expected_bright_axis_deg,
        beamstop_radius_px=beamstop_radius_px,
        geometry_cache=geometry_cache,
    )
    result["provenance"] = _sanitize_json(
        {
            "image_path": str(source),
            "poni_path": None if poni_path is None else str(Path(poni_path)),
            "mask_path": None if mask_path is None else str(Path(mask_path)),
            "image_dataset": loaded.dataset,
            "image_frame_index": loaded.frame,
            "shape": list(shape),
            "q_unit": _normalise_q_unit(q_unit),
            "beamstop_radius_px": _finite_float(beamstop_radius_px) if beamstop_radius_px is not None else None,
        }
    )
    return result


__all__ = [
    "MORPHOLOGY_CLASSES",
    "SCHEMA_VERSION",
    "advise_frame",
    "advise_from_arrays",
    "build_beamstop_exclusion_mask",
]
