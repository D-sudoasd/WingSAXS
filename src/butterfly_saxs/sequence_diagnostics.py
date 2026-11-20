"""Sequence-level quality and anomaly diagnostics for SAXS trajectories.

This module is strictly diagnostic: it preserves source-frame identity and
sequence positions, and never interpolates, replaces, or overwrites failed or
missing frames.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .butterfly_quality import build_quality_remediation
from .csv_utils import safe_csv_cell
from .settings import canonical_q_unit

SCHEMA_VERSION = "wingsaxs.sequence_diagnostics.v1"

AXIS_RATIO_COLLAPSE_THRESHOLD = 0.002

_PER_FRAME_COLUMNS = (
    "index",
    "frame_id",
    "status",
    "timestamp_s",
    "dt_s",
    "content_hash",
    "is_duplicate",
    "duplicate_group_size",
    "intensity_unit",
    "q_unit",
    "axis_ratio",
    "axis_ratio_collapsed",
    "q_star",
    "low_q_edge_hit",
    "high_q_edge_hit",
    "orientation_deg",
    "orthogonal_flip",
    "orientation_jump_deg",
    "coverage",
    "coverage_drop",
    "anomaly_flags",
)


def _sanitize_json(value: Any) -> Any:
    """Recursively convert values into strict JSON-safe Python types."""
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
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _signed_angle_diff_mod180(angle_deg: float, ref_deg: float) -> float:
    diff = ((float(angle_deg) - float(ref_deg) + 90.0) % 180.0) - 90.0
    if diff <= -90.0:
        diff += 180.0
    return float(diff)


def _array_sha256(array: np.ndarray) -> str:
    """Compute a deterministic SHA-256 digest over shape, dtype, and pixel bytes."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    header = f"ndarray|{arr.shape}|{arr.dtype.str}|".encode("ascii")
    digest.update(header)
    digest.update(arr.tobytes())
    return digest.hexdigest()


def compute_frame_content_hashes(
    frames_or_paths: Sequence[Any] | str | Path,
    *,
    dataset: str | None = None,
    frame_indices: Sequence[int] | None = None,
) -> list[str]:
    """Compute SHA-256 pixel-content hashes for arrays, LoadedImage objects, or paths.

    Hashing decoded 2D pixel arrays detects frozen/duplicate detector readouts
    even when container timestamps or filenames differ.
    """
    from .io import load_image

    # Single multi-frame file path expanded across explicit frame_indices
    if isinstance(frames_or_paths, (str, Path)):
        source = Path(frames_or_paths)
        if frame_indices is None:
            loaded = load_image(source, dataset=dataset)
            return [_array_sha256(np.asarray(loaded.data))]
        hashes: list[str] = []
        for idx in frame_indices:
            loaded = load_image(source, frame=int(idx), dataset=dataset)
            hashes.append(_array_sha256(np.asarray(loaded.data)))
        return hashes

    items = list(frames_or_paths)
    if frame_indices is not None and len(frame_indices) != len(items):
        raise ValueError(
            f"frame_indices length ({len(frame_indices)}) must match frames_or_paths length ({len(items)})"
        )

    hashes = []
    for pos, item in enumerate(items):
        explicit_idx = None if frame_indices is None else int(frame_indices[pos])
        if isinstance(item, np.ndarray):
            hashes.append(_array_sha256(item))
        elif isinstance(item, (bytes, bytearray, memoryview)):
            hashes.append(hashlib.sha256(bytes(item)).hexdigest())
        elif isinstance(item, Mapping):
            if "data" in item or "image" in item:
                arr = np.asarray(item.get("data", item.get("image")))
                hashes.append(_array_sha256(arr))
            elif "path" in item:
                frame_sel = item.get("frame", item.get("frame_index", explicit_idx))
                ds_sel = item.get("dataset", dataset)
                loaded = load_image(
                    item["path"],
                    frame=None if frame_sel is None else int(frame_sel),
                    dataset=ds_sel,
                )
                hashes.append(_array_sha256(np.asarray(loaded.data)))
            else:
                raise ValueError(f"unsupported mapping item at index {pos}: missing 'data'/'image'/'path'")
        elif hasattr(item, "data") and not isinstance(item, (str, Path)):
            hashes.append(_array_sha256(np.asarray(item.data)))
        elif hasattr(item, "image") and not isinstance(item, (str, Path)):
            hashes.append(_array_sha256(np.asarray(item.image)))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], (str, Path)):
            loaded = load_image(item[0], frame=int(item[1]), dataset=dataset)
            hashes.append(_array_sha256(np.asarray(loaded.data)))
        elif isinstance(item, (str, Path)):
            loaded = load_image(item, frame=explicit_idx, dataset=dataset)
            hashes.append(_array_sha256(np.asarray(loaded.data)))
        else:
            hashes.append(_array_sha256(np.asarray(item)))
    return hashes


def _read_nested(record: Any, keys: Sequence[str], nested_blocks: Sequence[str] = ()) -> Any:
    """Read the first present non-None field from a record or its nested sub-mappings."""
    blocks: list[Any] = [record]
    if isinstance(record, Mapping):
        for block_name in nested_blocks:
            sub = record.get(block_name)
            if isinstance(sub, Mapping):
                blocks.append(sub)
    else:
        for block_name in nested_blocks:
            sub = getattr(record, block_name, None)
            if isinstance(sub, Mapping) or sub is not None:
                blocks.append(sub)

    for block in blocks:
        for key in keys:
            if isinstance(block, Mapping):
                if key in block and block[key] is not None:
                    return block[key]
            else:
                val = getattr(block, key, None)
                if val is not None:
                    return val
    return None


def _extract_flags(record: Any) -> set[str]:
    found: set[str] = set()
    sources = [record]
    if isinstance(record, Mapping):
        for name in ("quality", "ellipse_fit", "candidate_fit", "advisor"):
            sub = record.get(name)
            if isinstance(sub, Mapping):
                sources.append(sub)
    for src in sources:
        if not isinstance(src, Mapping):
            continue
        for key in ("flags", "quality_flags", "warnings"):
            raw = src.get(key)
            if isinstance(raw, str):
                found.update(part.strip() for part in raw.split(",") if part.strip())
            elif isinstance(raw, Mapping):
                found.update(str(k) for k, v in raw.items() if v is True)
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                found.update(str(item) for item in raw if item)
    return found


def _normalise_intensity_unit(record: Any) -> str | None:
    raw = _read_nested(
        record,
        ("intensity_unit", "calibration_unit", "flux_unit", "units"),
        ("metadata", "calibration", "provenance"),
    )
    if raw is not None and str(raw).strip():
        text = str(raw).strip()
        low = text.lower().replace(" ", "")
        if low in {"cm^-1", "cm-1", "1/cm", "cm⁻¹", "abs", "absolute", "calibrated"}:
            return "cm^-1"
        if low in {"raw", "counts", "raw_counts", "adu", "uncalibrated", "arb", "a.u."}:
            return "raw_counts"
        return text
    cal_flag = _read_nested(
        record,
        ("abs_calibrated", "intensity_calibrated", "calibrated_intensity"),
        ("metadata", "calibration", "provenance"),
    )
    if isinstance(cal_flag, bool):
        return "cm^-1" if cal_flag else "raw_counts"
    return None


def _normalise_q_unit_field(record: Any) -> str | None:
    raw = _read_nested(
        record,
        ("q_unit",),
        ("metadata", "ellipse_fit", "candidate_fit", "observables"),
    )
    if raw is None or not str(raw).strip():
        return None
    canonical = canonical_q_unit(raw)
    if canonical == "nm⁻¹":
        return "nm^-1"
    if canonical == "Å⁻¹":
        return "Å^-1"
    return str(canonical)


def _record_is_failed(record: Any) -> bool:
    if record is None:
        return True
    success = _read_nested(record, ("success",), ("ellipse_fit", "candidate_fit"))
    if success is False:
        return True
    for key in ("status", "quality_status", "measurement_status", "fit_status"):
        val = _read_nested(record, (key,), ("quality", "ellipse_fit"))
        if isinstance(val, str) and val.strip().lower() in {"fail", "failed", "error", "invalid", "missing"}:
            return True
    err = _read_nested(record, ("error", "traceback"), ())
    if err:
        return True
    return False


def diagnose_sequence(
    records: Sequence[Any],
    *,
    frame_hashes: Sequence[str | None] | None = None,
    timestamps_s: Sequence[float | None] | None = None,
    q_window: Sequence[float] | None = None,
    nominal_draw_axis_deg: float = 90.0,
) -> dict[str, Any]:
    """Run comprehensive sequence-level quality and anomaly diagnostics.

    Purely diagnostic: retains failed/missing frames at their original indices
    without interpolation or overwriting.
    """
    if isinstance(nominal_draw_axis_deg, bool) or _finite_float(nominal_draw_axis_deg) is None:
        raise ValueError(f"nominal_draw_axis_deg must be finite, got {nominal_draw_axis_deg!r}")
    nominal_axis = float(nominal_draw_axis_deg)

    rec_list = list(records)
    n_frames = len(rec_list)

    if frame_hashes is not None and len(frame_hashes) != n_frames:
        raise ValueError(
            f"frame_hashes length ({len(frame_hashes)}) must match records length ({n_frames})"
        )
    if timestamps_s is not None and len(timestamps_s) != n_frames:
        raise ValueError(
            f"timestamps_s length ({len(timestamps_s)}) must match records length ({n_frames})"
        )

    resolved_q_window: tuple[float, float] | None = None
    if q_window is not None:
        pair = tuple(q_window)
        if len(pair) != 2:
            raise ValueError("q_window must be a (q_min, q_max) pair")
        q_lo, q_hi = _finite_float(pair[0]), _finite_float(pair[1])
        if q_lo is None or q_hi is None or q_hi <= q_lo:
            raise ValueError(f"q_window must satisfy finite q_min < q_max, got {q_window!r}")
        resolved_q_window = (q_lo, q_hi)

    # Extract per-frame observable arrays without mutating input records
    frame_ids: list[str] = []
    statuses: list[str] = []
    failed_indices: list[int] = []
    hashes: list[str | None] = []
    times: list[float | None] = []
    intensity_units: list[str | None] = []
    q_units: list[str | None] = []
    axis_ratios: list[float | None] = []
    q_stars: list[float | None] = []
    orientations: list[float | None] = []
    coverages: list[float | None] = []
    rec_flags_list: list[set[str]] = []

    for idx, rec in enumerate(rec_list):
        fid = _read_nested(rec, ("frame_id", "id", "name", "stem"), ("metadata",))
        frame_ids.append(str(fid) if fid is not None else f"frame_{idx:04d}")

        failed = _record_is_failed(rec)
        if failed:
            failed_indices.append(idx)
            statuses.append("failed")
        else:
            raw_status = _read_nested(rec, ("status", "quality_status"), ("quality",))
            statuses.append(str(raw_status) if raw_status is not None else "ok")

        if frame_hashes is not None:
            h_val = frame_hashes[idx]
        else:
            h_val = _read_nested(
                rec,
                ("content_hash", "frame_hash", "pixel_sha256", "sha256"),
                ("metadata", "provenance"),
            )
        hashes.append(str(h_val) if h_val not in (None, "") else None)

        if timestamps_s is not None:
            t_val = _finite_float(timestamps_s[idx])
        else:
            t_val = _finite_float(
                _read_nested(
                    rec,
                    ("timestamp_s", "time_s", "elapsed_s", "timestamp", "time"),
                    ("metadata",),
                )
            )
        times.append(t_val)

        intensity_units.append(_normalise_intensity_unit(rec))
        q_units.append(_normalise_q_unit_field(rec))

        ar = _finite_float(
            _read_nested(
                rec,
                ("axis_ratio_b_over_a", "axis_ratio"),
                ("ellipse_fit", "candidate_fit", "parameters"),
            )
        )
        axis_ratios.append(ar)

        qs = _finite_float(
            _read_nested(
                rec,
                (
                    "q_star",
                    "q_star_direct",
                    "q_star_from_arcs",
                    "q_star_sector_median",
                    "q_star_kratky",
                ),
                ("observables", "candidate_fit", "ellipse_fit", "measurement_summary"),
            )
        )
        q_stars.append(qs)

        orient = _finite_float(
            _read_nested(
                rec,
                (
                    "friedel_bisector_deg",
                    "bisector_deg",
                    "orientation_deg",
                    "orientation2_deg",
                    "theta_deg",
                ),
                ("symmetry", "observables", "ellipse_fit", "candidate_fit", "parameters"),
            )
        )
        orientations.append(orient)

        cov = _finite_float(
            _read_nested(
                rec,
                ("coverage", "valid_fraction", "azimuthal_coverage", "radial_coverage"),
                ("quality", "metrics", "analysis_domain"),
            )
        )
        coverages.append(cov)
        rec_flags_list.append(_extract_flags(rec))

        if resolved_q_window is None:
            rec_qw = _read_nested(rec, ("q_window",), ("analysis", "metadata"))
            if isinstance(rec_qw, (list, tuple)) and len(rec_qw) == 2:
                lo, hi = _finite_float(rec_qw[0]), _finite_float(rec_qw[1])
                if lo is not None and hi is not None and hi > lo:
                    resolved_q_window = (lo, hi)

    per_frame_anomalies: list[list[str]] = [[] for _ in range(n_frames)]

    # (a) Duplicate / frozen pixel frames
    hash_to_indices: dict[str, list[int]] = {}
    for idx, h in enumerate(hashes):
        if h is not None:
            hash_to_indices.setdefault(h, []).append(idx)
    n_hashed = sum(1 for h in hashes if h is not None)
    n_unique = len(hash_to_indices)
    unique_ratio = float(n_unique / n_hashed) if n_hashed > 0 else None
    duplicate_clusters = [
        {
            "hash": h,
            "count": len(idxs),
            "first_index": idxs[0],
            "indices": idxs,
        }
        for h, idxs in hash_to_indices.items()
        if len(idxs) > 1
    ]
    duplicate_clusters.sort(key=lambda item: item["first_index"])
    duplicate_indices: set[int] = set()
    for cluster in duplicate_clusters:
        # Subsequent copies after the first occurrence are duplicates; mark all
        # members of frozen groups for visibility in group_size.
        for dup_idx in cluster["indices"][1:]:
            duplicate_indices.add(dup_idx)
            per_frame_anomalies[dup_idx].append("duplicate_frame")

    # Consecutive frozen runs
    frozen_runs: list[dict[str, Any]] = []
    run_start: int | None = None
    for idx in range(n_frames):
        if hashes[idx] is not None and idx > 0 and hashes[idx] == hashes[idx - 1]:
            if run_start is None:
                run_start = idx - 1
        else:
            if run_start is not None:
                frozen_runs.append(
                    {
                        "hash": hashes[run_start],
                        "start_index": run_start,
                        "end_index": idx - 1,
                        "length": idx - run_start,
                    }
                )
                run_start = None
    if run_start is not None:
        frozen_runs.append(
            {
                "hash": hashes[run_start],
                "start_index": run_start,
                "end_index": n_frames - 1,
                "length": n_frames - run_start,
            }
        )
    duplicate_frame_count = int(n_hashed - n_unique) if n_hashed > 0 else 0

    # (b) Timestamp gaps / chronology anomalies
    dts: list[float | None] = [None] * n_frames
    non_monotonic_indices: list[int] = []
    stalled_indices: list[int] = []
    missing_time_indices: list[int] = [i for i, t in enumerate(times) if t is None]
    positive_dts: list[float] = []
    last_valid_idx: int | None = None
    for idx, t in enumerate(times):
        if t is None:
            continue
        if last_valid_idx is not None:
            dt = float(t - times[last_valid_idx])  # type: ignore[operator]
            dts[idx] = dt
            if dt < 0.0:
                non_monotonic_indices.append(idx)
                per_frame_anomalies[idx].append("non_monotonic_timestamp")
            elif dt == 0.0:
                stalled_indices.append(idx)
                per_frame_anomalies[idx].append("stalled_timestamp")
            else:
                positive_dts.append(dt)
        last_valid_idx = idx

    median_dt = float(np.median(positive_dts)) if positive_dts else None
    mad_dt = (
        float(1.4826 * np.median(np.abs(np.asarray(positive_dts) - median_dt)))
        if (positive_dts and median_dt is not None)
        else None
    )
    gaps: list[dict[str, Any]] = []
    if median_dt is not None and median_dt > 0.0:
        # Flag any interval exceeding 3x the median positive cadence (bounded
        # MAD tolerance prevents a small-N outlier from masking itself).
        bounded_mad = min(mad_dt or 0.0, 0.5 * median_dt)
        gap_thresh = max(3.0 * median_dt, median_dt + 3.0 * bounded_mad)
        prev_idx: int | None = None
        for idx, t in enumerate(times):
            if t is None:
                continue
            if prev_idx is not None and dts[idx] is not None and dts[idx] > gap_thresh:
                gaps.append(
                    {
                        "index": idx,
                        "prev_index": prev_idx,
                        "dt_s": float(dts[idx]),
                        "ratio_to_median": float(dts[idx] / median_dt),
                    }
                )
                per_frame_anomalies[idx].append("timestamp_gap")
            prev_idx = idx

    valid_times = [t for t in times if t is not None]
    span_s = float(max(valid_times) - min(valid_times)) if len(valid_times) >= 2 else (0.0 if len(valid_times) == 1 else None)
    chronology_ok = not (gaps or non_monotonic_indices or stalled_indices)

    # (c) Mixed intensity unit / calibration provenance flags
    intensity_unit_counts: dict[str, int] = {}
    q_unit_counts: dict[str, int] = {}
    raw_count_indices: list[int] = []
    uncalibrated_q_indices: list[int] = []
    for idx in range(n_frames):
        iu = intensity_units[idx]
        if iu is not None:
            intensity_unit_counts[iu] = intensity_unit_counts.get(iu, 0) + 1
            if iu == "raw_counts":
                raw_count_indices.append(idx)
        qu = q_units[idx]
        if qu is not None:
            q_unit_counts[qu] = q_unit_counts.get(qu, 0) + 1
            if qu not in {"nm^-1", "Å^-1"}:
                uncalibrated_q_indices.append(idx)
    mixed_intensity_units = len(intensity_unit_counts) > 1
    mixed_q_units = len(q_unit_counts) > 1
    if mixed_intensity_units:
        for idx in raw_count_indices:
            per_frame_anomalies[idx].append("raw_count_in_mixed_sequence")

    # (d) axis_ratio_b_over_a bound-collapse rate (<= 0.002)
    collapsed_flags: list[bool] = [False] * n_frames
    collapsed_indices: list[int] = []
    evaluated_ar_count = 0
    for idx, ar in enumerate(axis_ratios):
        has_flag = bool({"axis_ratio_at_bound", "axis_ratio_collapsed_to_line"} & rec_flags_list[idx])
        if ar is not None:
            evaluated_ar_count += 1
            if ar <= AXIS_RATIO_COLLAPSE_THRESHOLD or has_flag:
                collapsed_flags[idx] = True
                collapsed_indices.append(idx)
                per_frame_anomalies[idx].append("axis_ratio_collapsed")
        elif has_flag:
            evaluated_ar_count += 1
            collapsed_flags[idx] = True
            collapsed_indices.append(idx)
            per_frame_anomalies[idx].append("axis_ratio_collapsed")
    bound_collapse_rate = (
        float(len(collapsed_indices) / evaluated_ar_count) if evaluated_ar_count > 0 else None
    )

    # (e) Low-q / high-q window edge pile-ups
    low_q_hits: list[bool] = [False] * n_frames
    high_q_hits: list[bool] = [False] * n_frames
    low_q_indices: list[int] = []
    high_q_indices: list[int] = []
    q_star_evaluated = 0
    edge_margin: float | None = None
    if resolved_q_window is not None:
        edge_margin = max(0.03 * (resolved_q_window[1] - resolved_q_window[0]), 1e-6)

    for idx, qs in enumerate(q_stars):
        fset = rec_flags_list[idx]
        low_flagged = "low_q_edge_hit" in fset
        high_flagged = bool({"high_q_edge_hit", "annular_outer_window_truncated"} & fset)
        if qs is not None:
            q_star_evaluated += 1
            if resolved_q_window is not None and edge_margin is not None:
                if qs <= resolved_q_window[0] + edge_margin:
                    low_flagged = True
                if qs >= resolved_q_window[1] - edge_margin:
                    high_flagged = True
        if low_flagged:
            low_q_hits[idx] = True
            low_q_indices.append(idx)
            per_frame_anomalies[idx].append("low_q_edge_hit")
        if high_flagged:
            high_q_hits[idx] = True
            high_q_indices.append(idx)
            per_frame_anomalies[idx].append("high_q_edge_hit")

    denom_q = max(q_star_evaluated, len(set(low_q_indices) | set(high_q_indices)))
    low_q_edge_rate = float(len(low_q_indices) / denom_q) if denom_q > 0 else 0.0
    high_q_edge_rate = float(len(high_q_indices) / denom_q) if denom_q > 0 else 0.0

    # (f) Abrupt orientation flips or regime transitions
    orthogonal_flags: list[bool] = [False] * n_frames
    orthogonal_indices: list[int] = []
    orientation_jumps: list[dict[str, Any]] = []
    orientation_jump_deg_per_frame: list[float | None] = [None] * n_frames
    orient_evaluated = 0
    prev_orient_idx: int | None = None
    for idx, orient in enumerate(orientations):
        if orient is None:
            continue
        orient_evaluated += 1
        tilt = _signed_angle_diff_mod180(orient, nominal_axis)
        if abs(abs(tilt) - 90.0) <= 25.0:
            orthogonal_flags[idx] = True
            orthogonal_indices.append(idx)
            per_frame_anomalies[idx].append("orthogonal_axis_flip")
        if prev_orient_idx is not None:
            prev_val = orientations[prev_orient_idx]
            assert prev_val is not None
            jump = abs(_signed_angle_diff_mod180(orient, prev_val))
            orientation_jump_deg_per_frame[idx] = float(jump)
            if jump >= 25.0:
                orientation_jumps.append(
                    {
                        "index": idx,
                        "prev_index": prev_orient_idx,
                        "prev_deg": float(prev_val),
                        "curr_deg": float(orient),
                        "delta_deg": float(jump),
                    }
                )
                per_frame_anomalies[idx].append("abrupt_orientation_jump")
        prev_orient_idx = idx
    orthogonal_fraction = float(len(orthogonal_indices) / orient_evaluated) if orient_evaluated > 0 else 0.0

    # Regime shifts in q_star (abrupt jumps >= 25% relative change and >= 4 * MAD of step sizes)
    q_regime_shifts: list[dict[str, Any]] = []
    valid_q_pairs: list[tuple[int, int, float, float, float]] = []
    prev_q_idx: int | None = None
    for idx, qs in enumerate(q_stars):
        if qs is None or qs <= 0.0:
            continue
        if prev_q_idx is not None:
            pq = q_stars[prev_q_idx]
            assert pq is not None and pq > 0.0
            step = abs(qs - pq)
            valid_q_pairs.append((prev_q_idx, idx, pq, qs, step))
        prev_q_idx = idx
    if valid_q_pairs:
        steps = np.asarray([item[4] for item in valid_q_pairs], dtype=np.float64)
        med_step = float(np.median(steps))
        mad_step = float(1.4826 * np.median(np.abs(steps - med_step)))
        step_thresh = max(med_step + 4.0 * mad_step, 1e-6)
        for prev_i, curr_i, pq, cq, step in valid_q_pairs:
            rel = abs(cq - pq) / max(pq, 1e-12)
            if rel >= 0.25 and step >= step_thresh:
                q_regime_shifts.append(
                    {
                        "index": curr_i,
                        "prev_index": prev_i,
                        "prev_q_star": float(pq),
                        "curr_q_star": float(cq),
                        "relative_change": float(rel),
                    }
                )
                per_frame_anomalies[curr_i].append("q_star_regime_shift")

    # (g) Coverage drops
    valid_covs = [c for c in coverages if c is not None]
    median_cov = float(np.median(valid_covs)) if valid_covs else None
    min_cov = float(np.min(valid_covs)) if valid_covs else None
    mad_cov = (
        float(1.4826 * np.median(np.abs(np.asarray(valid_covs) - median_cov)))
        if (valid_covs and median_cov is not None)
        else None
    )
    coverage_drop_flags: list[bool] = [False] * n_frames
    coverage_drop_indices: list[int] = []
    for idx, cov in enumerate(coverages):
        if cov is None:
            continue
        drop_abs = cov < 0.50
        drop_rel = (
            median_cov is not None
            and median_cov > 0.20
            and cov < median_cov - max(0.20, 3.0 * (mad_cov or 0.0))
        )
        if drop_abs or drop_rel:
            coverage_drop_flags[idx] = True
            coverage_drop_indices.append(idx)
            per_frame_anomalies[idx].append("coverage_drop")

    # Aggregate sequence-level warnings & actionable remediation
    warnings: list[str] = []
    if duplicate_frame_count > 0:
        warnings.append("duplicate_frames_detected")
    if not chronology_ok:
        warnings.append("timestamp_anomaly")
    if mixed_intensity_units:
        warnings.append("mixed_intensity_units")
    if uncalibrated_q_indices:
        warnings.append("uncalibrated_pixel_q")
    if collapsed_indices:
        warnings.append("axis_ratio_at_bound")
    if low_q_indices:
        warnings.append("low_q_edge_hit")
    if high_q_indices:
        warnings.append("high_q_edge_hit")
    if orthogonal_indices and (orthogonal_fraction >= 0.30 or len(orthogonal_indices) >= 2 or orient_evaluated == 1):
        warnings.append("orthogonal_axis_flip")
    if orientation_jumps:
        warnings.append("abrupt_orientation_transition")
    if coverage_drop_indices:
        warnings.append("coverage_drop")
    warnings = list(dict.fromkeys(warnings))

    remediation_metrics: dict[str, Any] = {
        "q_window": list(resolved_q_window) if resolved_q_window is not None else None,
        "bound_collapse_rate": bound_collapse_rate,
        "duplicate_frame_count": duplicate_frame_count,
        "unique_ratio": unique_ratio,
    }
    if orthogonal_indices:
        orth_vals = [orientations[i] for i in orthogonal_indices if orientations[i] is not None]
        if orth_vals:
            remediation_metrics["friedel_bisector_deg"] = float(np.median(orth_vals))
    remediation = build_quality_remediation(warnings, metrics=remediation_metrics)

    # Build lossless per-frame table preserving every index & failed frame
    per_frame: list[dict[str, Any]] = []
    for idx in range(n_frames):
        h = hashes[idx]
        group_size = len(hash_to_indices[h]) if h is not None else 0
        per_frame.append(
            {
                "index": idx,
                "frame_id": frame_ids[idx],
                "status": statuses[idx],
                "timestamp_s": times[idx],
                "dt_s": dts[idx],
                "content_hash": h,
                "is_duplicate": idx in duplicate_indices,
                "duplicate_group_size": group_size,
                "intensity_unit": intensity_units[idx],
                "q_unit": q_units[idx],
                "axis_ratio": axis_ratios[idx],
                "axis_ratio_collapsed": collapsed_flags[idx],
                "q_star": q_stars[idx],
                "low_q_edge_hit": low_q_hits[idx],
                "high_q_edge_hit": high_q_hits[idx],
                "orientation_deg": orientations[idx],
                "orthogonal_flip": orthogonal_flags[idx],
                "orientation_jump_deg": orientation_jump_deg_per_frame[idx],
                "coverage": coverages[idx],
                "coverage_drop": coverage_drop_flags[idx],
                "anomaly_flags": list(dict.fromkeys(per_frame_anomalies[idx])),
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "n_frames": n_frames,
        "n_failed": len(failed_indices),
        "n_valid": n_frames - len(failed_indices),
        "failed_indices": failed_indices,
        "nominal_draw_axis_deg": nominal_axis,
        "q_window": list(resolved_q_window) if resolved_q_window is not None else None,
        "duplicates": {
            "detected": duplicate_frame_count > 0,
            "n_hashed": n_hashed,
            "n_unique": n_unique,
            "unique_ratio": unique_ratio,
            "duplicate_frame_count": duplicate_frame_count,
            "duplicate_clusters": duplicate_clusters,
            "frozen_runs": frozen_runs,
        },
        "timestamps": {
            "chronology_ok": chronology_ok,
            "n_with_timestamp": len(valid_times),
            "missing_indices": missing_time_indices,
            "span_s": span_s,
            "median_dt_s": median_dt,
            "min_dt_s": float(min(positive_dts)) if positive_dts else None,
            "max_dt_s": float(max(positive_dts)) if positive_dts else None,
            "gaps": gaps,
            "non_monotonic_indices": non_monotonic_indices,
            "stalled_indices": stalled_indices,
        },
        "calibration": {
            "mixed_intensity_units": mixed_intensity_units,
            "mixed_q_units": mixed_q_units,
            "intensity_unit_counts": intensity_unit_counts,
            "q_unit_counts": q_unit_counts,
            "raw_count_frame_indices": raw_count_indices,
            "uncalibrated_q_frame_indices": uncalibrated_q_indices,
        },
        "axis_ratio_collapse": {
            "threshold": AXIS_RATIO_COLLAPSE_THRESHOLD,
            "evaluated_count": evaluated_ar_count,
            "bound_collapse_count": len(collapsed_indices),
            "bound_collapse_rate": bound_collapse_rate,
            "collapsed_indices": collapsed_indices,
        },
        "window_edge_pileup": {
            "q_window": list(resolved_q_window) if resolved_q_window is not None else None,
            "edge_margin": edge_margin,
            "evaluated_count": q_star_evaluated,
            "low_q_edge_hits": len(low_q_indices),
            "low_q_edge_rate": low_q_edge_rate,
            "low_q_edge_indices": low_q_indices,
            "low_q_onset_index": low_q_indices[0] if low_q_indices else None,
            "high_q_edge_hits": len(high_q_indices),
            "high_q_edge_rate": high_q_edge_rate,
            "high_q_edge_indices": high_q_indices,
            "high_q_onset_index": high_q_indices[0] if high_q_indices else None,
        },
        "orientation_and_regimes": {
            "nominal_draw_axis_deg": nominal_axis,
            "evaluated_count": orient_evaluated,
            "orthogonal_flip_detected": bool(
                orthogonal_indices and (orthogonal_fraction >= 0.30 or len(orthogonal_indices) >= 2 or orient_evaluated == 1)
            ),
            "orthogonal_flip_count": len(orthogonal_indices),
            "orthogonal_flip_fraction": orthogonal_fraction,
            "orthogonal_flip_indices": orthogonal_indices,
            "orientation_jumps": orientation_jumps,
            "q_star_regime_shifts": q_regime_shifts,
        },
        "coverage": {
            "evaluated_count": len(valid_covs),
            "median_coverage": median_cov,
            "min_coverage": min_cov,
            "coverage_drop_count": len(coverage_drop_indices),
            "coverage_drop_indices": coverage_drop_indices,
        },
        "warnings": warnings,
        "remediation": remediation,
        "per_frame": per_frame,
    }
    return _sanitize_json(payload)


def write_sequence_diagnostics(
    diag: Mapping[str, Any],
    output_dir: str | Path,
    *,
    force: bool = True,
) -> dict[str, Path]:
    """Write ``sequence_diagnostics.json`` and ``sequence_diagnostics.csv`` to ``output_dir``."""
    target_dir = Path(output_dir)
    json_path = target_dir / "sequence_diagnostics.json"
    csv_path = target_dir / "sequence_diagnostics.csv"

    if not force:
        existing = [p for p in (json_path, csv_path) if p.exists()]
        if existing:
            raise FileExistsError(
                f"sequence diagnostics outputs already exist: {', '.join(str(p) for p in existing)}"
            )

    target_dir.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_json(dict(diag))
    json_path.write_text(
        json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = sanitized.get("per_frame", [])
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes, Mapping)):
        rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_PER_FRAME_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            formatted: dict[str, Any] = {}
            for col in _PER_FRAME_COLUMNS:
                val = row.get(col)
                if col == "anomaly_flags" and isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
                    val = ";".join(str(item) for item in val)
                formatted[col] = safe_csv_cell(val)
            writer.writerow(formatted)

    return {"json": json_path, "csv": csv_path}


__all__ = [
    "AXIS_RATIO_COLLAPSE_THRESHOLD",
    "SCHEMA_VERSION",
    "compute_frame_content_hashes",
    "diagnose_sequence",
    "write_sequence_diagnostics",
]
