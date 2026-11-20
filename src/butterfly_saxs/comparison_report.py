"""Multi-sample comparative analytics, tables, and Nature-styled overview figures.

Strictly preserves WingSAXS scientific boundaries:
- Calibrated absolute-intensity runs (e.g., ``cm^-1``) are never mixed with
  uncalibrated raw-count runs on the same absolute-intensity scale.
- ``estimate``, ``candidate``, and ``failed``/``missing`` states remain explicit
  in all exported tables and plots.
- Real-space correlation periods ``L* = 2*pi/q*`` in nm are computed only when
  physical calibrated ``q`` units are recorded (never from ``pixel-q``).
- Generated HTML links stay strictly relative and portable for ``delivery``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import html
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import numpy as np

from .composite_figures import (
    CompositeFigureBuilder,
    export_publication_figure,
    plot_evolution_panel,
)
from .csv_utils import iter_csv_rows, safe_csv_cell
from .journal_style import (
    JournalPreset,
    format_axis_label,
    get_journal_preset,
    get_okabe_ito_palette,
    get_saxs_colormap,
    get_thermal_palette,
    is_physical_q_unit,
    journal_rc_context,
    q_to_period_nm,
    style_1d_axes,
    style_2d_axes,
)
from .serialization import json_safe


COMPARISON_SCHEMA = "wingsaxs.comparison_report.v1"

_TEMP_REGEX = re.compile(r"(?:^|[_\-\sT])(\d{2,4}(?:\.\d+)?)\s*(?:°\s*)?C(?=$|[_\-\s])", re.IGNORECASE)
_BEAMTIME_REGEX = re.compile(r"(20\d{2}[A-Z]\d{3,5}|20\d{2})")


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        val = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return val if math.isfinite(val) else None


def _read_csv_if_exists(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(iter_csv_rows(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: safe_csv_cell(row.get(col)) for col in columns})


def _infer_temperature_c(entry: Mapping[str, Any], sample_name: str) -> float | None:
    for key in ("temperature_C", "temperature_c", "temperature", "temp_C", "temp_c"):
        val = _num(entry.get(key))
        if val is not None:
            return val
    for text in (entry.get("condition"), sample_name):
        if not text:
            continue
        match = _TEMP_REGEX.search(str(text))
        if match:
            return _num(match.group(1))
    return None


def _infer_beamtime(entry: Mapping[str, Any], sample_name: str) -> str:
    if entry.get("beamtime"):
        return str(entry["beamtime"]).strip()
    match = _BEAMTIME_REGEX.search(sample_name)
    return match.group(1) if match else ""


def _normalize_calibration_mode(raw: Any, intensity_unit: str | None = None) -> str:
    if isinstance(raw, bool):
        return "calibrated" if raw else "raw_counts"
    if raw is not None and str(raw).strip():
        key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
        if key in {"calibrated", "abs", "absolute", "abs2d", "cm_1", "cm^-1", "true"}:
            return "calibrated"
        if key in {"raw", "raw_counts", "counts", "uncalibrated", "adu", "false"}:
            return "raw_counts"
        if key in {"mixed", "partial"}:
            return "mixed"
    if intensity_unit:
        u = str(intensity_unit).strip().lower().replace(" ", "")
        if u in {"cm^-1", "cm-1", "1/cm", "cm⁻¹", "cm−1", "mm^-1", "m^-1"}:
            return "calibrated"
        if any(tok in u for tok in ("raw", "count", "adu", "uncal")):
            return "raw_counts"
    return "raw_counts"


def _canonical_q_unit(unit: Any) -> str:
    text = str(unit or "").strip()
    low = text.lower().replace(" ", "")
    if low in {"nm^-1", "nm-1", "1/nm", "nm⁻¹", "nm−1"}:
        return "nm^-1"
    if low in {"a^-1", "a-1", "1/a", "å^-1", "å-1", "1/å", "å⁻¹", "å−1", "angstrom^-1"}:
        return "Å^-1"
    if low in {"pixel", "pixel-q", "pixel_q", "pixelq"}:
        return "pixel-q"
    return text or "unknown"


def _extract_nested_scalar(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            val = _num(row.get(key))
            if val is not None:
                return val
    # Handle JSON-encoded nested dicts from frame_measurements.csv (e.g., intensity, in_plane_intensity_tensor).
    for container_key in ("intensity", "in_plane_intensity_tensor", "saxs_physics"):
        raw = row.get(container_key)
        container: Mapping[str, Any] | None = None
        if isinstance(raw, Mapping):
            container = raw
        elif isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, Mapping):
                    container = parsed
            except json.JSONDecodeError:
                container = None
        if container is None:
            continue
        for key in keys:
            short = key.removeprefix(f"{container_key}_")
            for candidate_key in (key, short):
                if candidate_key in container:
                    val = _num(container.get(candidate_key))
                    if val is not None:
                        return val
    return None


def _time_in_hours(row: Mapping[str, Any]) -> tuple[float | None, str]:
    t_h = _num(row.get("time_h"))
    if t_h is not None:
        return t_h, "h"
    t_s = _num(row.get("time_s"))
    if t_s is not None:
        return t_s / 3600.0, "h"
    t = _num(row.get("time"))
    if t is None:
        return None, ""
    unit = str(row.get("time_unit") or "").strip().lower()
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return t, "h"
    if unit in {"min", "mins", "minute", "minutes"}:
        return t / 60.0, "h"
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return t / 3600.0, "h"
    if unit in {"ms", "millisecond", "milliseconds"}:
        return t / 3_600_000.0, "h"
    return t, unit or "unspecified"


def _classify_evidence(
    value: float | None,
    candidate_value: float | None,
    raw_status: Any,
) -> tuple[float | None, str]:
    status_str = str(raw_status or "").strip().lower()
    if status_str in {"failed", "failure", "error", "not_processed", "unavailable", "invalid", "missing"}:
        return None, "failed"
    if value is not None:
        if "candidate" in status_str or status_str in {"provisional"}:
            return value, "candidate"
        return value, "estimate"
    if candidate_value is not None:
        return candidate_value, "candidate"
    return None, "missing"


def _load_sample_data(entry: Mapping[str, Any], index: int) -> dict[str, Any]:
    sample_name = str(
        entry.get("sample") or entry.get("sample_id") or entry.get("name") or f"sample_{index + 1:02d}"
    )
    measurements: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []

    if entry.get("measurements") is not None:
        measurements = [dict(r) for r in entry["measurements"] if isinstance(r, Mapping)]
    elif entry.get("measurements_path") is not None:
        measurements = _read_csv_if_exists(Path(entry["measurements_path"]))

    if entry.get("parameters") is not None:
        parameters = [dict(r) for r in entry["parameters"] if isinstance(r, Mapping)]
    elif entry.get("parameters_path") is not None:
        parameters = _read_csv_if_exists(Path(entry["parameters_path"]))

    if entry.get("frames") is not None:
        frames = [dict(r) for r in entry["frames"] if isinstance(r, Mapping)]
    elif entry.get("frames_path") is not None:
        frames = _read_csv_if_exists(Path(entry["frames_path"]))

    report_dir = Path(entry["report_dir"]) if entry.get("report_dir") else None
    batch_dir = Path(entry["batch_dir"]) if entry.get("batch_dir") else None
    prefix = str(entry.get("export_prefix") or entry.get("prefix") or "")

    if not measurements and report_dir is not None:
        measurements = _read_csv_if_exists(report_dir / "frame_measurements.csv")
    if not parameters:
        if batch_dir is not None:
            parameters = _read_csv_if_exists(batch_dir / f"{prefix}parameters_long.csv")
        if not parameters and report_dir is not None:
            parameters = _read_csv_if_exists(report_dir / "parameters_long.csv")
    if not frames and batch_dir is not None:
        frames = _read_csv_if_exists(batch_dir / f"{prefix}frame_summary.csv")

    # Build unified per-frame records keyed by frame_index.
    frame_map: dict[int, dict[str, Any]] = {}

    def _slot(row: Mapping[str, Any], fallback_idx: int) -> int:
        raw_idx = _num(row.get("frame_index"))
        return int(raw_idx) if raw_idx is not None else fallback_idx

    for idx, row in enumerate(frames):
        slot = _slot(row, idx)
        frame_map.setdefault(slot, {"frame_index": slot}).update(row)
    for idx, row in enumerate(measurements):
        slot = _slot(row, idx)
        rec = frame_map.setdefault(slot, {"frame_index": slot})
        for key, val in row.items():
            if val not in (None, "") or key not in rec:
                rec[key] = val

    # Index parameters_long by (frame_index, parameter).
    param_by_frame: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for idx, prow in enumerate(parameters):
        slot = _slot(prow, idx)
        pname = str(prow.get("parameter") or prow.get("name") or "").strip()
        if pname:
            param_by_frame[slot][pname] = dict(prow)
            frame_map.setdefault(slot, {"frame_index": slot})

    sorted_indices = sorted(frame_map)
    intensity_unit = str(entry.get("intensity_unit") or "").strip()
    if not intensity_unit:
        for idx in sorted_indices:
            iu = str(frame_map[idx].get("intensity_unit") or "").strip()
            if iu:
                intensity_unit = iu
                break

    q_unit = _canonical_q_unit(entry.get("q_unit"))
    if q_unit == "unknown":
        for idx in sorted_indices:
            qu = _canonical_q_unit(frame_map[idx].get("q_unit"))
            if qu != "unknown":
                q_unit = qu
                break

    # Detect mixed per-frame calibration if frame rows declare calibration_mode / intensity_unit.
    frame_cal_modes: set[str] = set()
    for idx in sorted_indices:
        frow = frame_map[idx]
        if frow.get("calibration_mode") is not None or frow.get("calibrated") is not None or frow.get("intensity_unit"):
            frame_cal_modes.add(
                _normalize_calibration_mode(
                    frow.get("calibration_mode", frow.get("calibrated")),
                    str(frow.get("intensity_unit") or ""),
                )
            )
    if entry.get("calibration_mode") is not None or entry.get("calibrated") is not None:
        cal_mode = _normalize_calibration_mode(
            entry.get("calibration_mode", entry.get("calibrated")),
            intensity_unit,
        )
    elif len(frame_cal_modes) > 1:
        cal_mode = "mixed"
    elif len(frame_cal_modes) == 1:
        cal_mode = next(iter(frame_cal_modes))
    else:
        cal_mode = _normalize_calibration_mode(None, intensity_unit)

    temp_c = _infer_temperature_c(entry, sample_name)
    beamtime = _infer_beamtime(entry, sample_name)
    condition = str(entry.get("condition") or (f"{temp_c:g} C" if temp_c is not None else sample_name))
    quarantined = bool(entry.get("quarantined", False))

    # Extract per-frame normalized observables.
    per_frame: list[dict[str, Any]] = []
    for slot in sorted_indices:
        frow = frame_map[slot]
        pdict = param_by_frame.get(slot, {})
        frame_q_unit = _canonical_q_unit(frow.get("q_unit") or q_unit)
        frame_status = str(frow.get("status") or "ok").strip().lower()
        t_val, t_unit = _time_in_hours(frow)

        def _param_or_frame(
            frame_keys: Sequence[str],
            param_names: Sequence[str],
            *,
            convert_q_to_L_nm: bool = False,
        ) -> tuple[float | None, float | None, str]:
            # Priority 1: explicit parameter_long row with value / candidate_value / status.
            for pname in param_names:
                prow = pdict.get(pname)
                if prow is None:
                    continue
                v = _num(prow.get("value"))
                cv = _num(prow.get("candidate_value"))
                st = prow.get("publication_status") or prow.get("evidence_status") or prow.get("status") or frame_status
                if convert_q_to_L_nm:
                    p_unit = prow.get("unit") or frame_q_unit
                    v = q_to_period_nm(v, p_unit) if v is not None else None
                    cv = q_to_period_nm(cv, p_unit) if cv is not None else None
                val, ev = _classify_evidence(v, cv, st)
                if ev in {"estimate", "candidate"}:
                    return (val if ev == "estimate" else None), (val if ev == "candidate" else cv), ev
                if ev == "failed":
                    return None, None, "failed"

            # Priority 2: measurement / frame summary scalar.
            v_frame = _extract_nested_scalar(frow, *frame_keys)
            c_frame = _extract_nested_scalar(frow, *(f"{k}_candidate" for k in frame_keys))
            if convert_q_to_L_nm:
                v_frame = q_to_period_nm(v_frame, frame_q_unit) if v_frame is not None else None
                c_frame = q_to_period_nm(c_frame, frame_q_unit) if c_frame is not None else None
            st = frow.get("evidence_status") or frow.get("publication_status") or frame_status
            val, ev = _classify_evidence(v_frame, c_frame, st)
            return (val if ev == "estimate" else None), (val if ev == "candidate" else c_frame), ev

        # L* (nm): ONLY from physical calibrated q!
        if is_physical_q_unit(frame_q_unit):
            l_val, l_cand, l_status = _param_or_frame(
                ("L_star_direct_nm", "L_star_nm", "L_from_observed_radius_nm", "Ln_nm"),
                ("L_star_direct_nm", "L_star_nm", "L_from_observed_radius_nm", "Ln_nm"),
            )
            if l_status == "missing":
                l_val, l_cand, l_status = _param_or_frame(
                    ("q_star_direct", "q_star", "q_star_from_arcs"),
                    ("q_star_direct", "q_star", "q_star_from_arcs"),
                    convert_q_to_L_nm=True,
                )
        else:
            l_val, l_cand, l_status = None, None, ("failed" if frame_status in {"failed", "error"} else "missing")

        q_star_val, q_star_cand, q_star_status = _param_or_frame(
            ("q_star_direct", "q_star", "q_star_from_arcs"),
            ("q_star_direct", "q_star", "q_star_from_arcs"),
        )
        q_win_val, q_win_cand, q_win_status = _param_or_frame(
            ("q_win_invariant", "Q_win", "scattering_invariant", "porod_invariant_window"),
            ("q_win_invariant", "Q_win", "scattering_invariant"),
        )
        i_sum_val, i_sum_cand, i_sum_status = _param_or_frame(
            ("intensity_sum", "intensity_total", "sum"),
            ("intensity_sum",),
        )
        a4_val, a4_cand, a4_status = _param_or_frame(
            ("a4", "A4", "four_fold_amplitude"),
            ("a4", "A4", "four_fold_amplitude"),
        )
        aniso_val, aniso_cand, aniso_status = _param_or_frame(
            ("anisotropy", "tensor_anisotropy"),
            ("anisotropy", "tensor_anisotropy"),
        )
        alpha_val, alpha_cand, alpha_status = _param_or_frame(
            ("wing_half_opening_deg", "half_opening_deg", "alpha_deg"),
            ("wing_half_opening_deg", "half_opening_deg", "alpha_deg"),
        )
        fwhm_val, fwhm_cand, fwhm_status = _param_or_frame(
            ("lobe_fwhm_deg", "azimuthal_fwhm_deg"),
            ("lobe_fwhm_deg", "azimuthal_fwhm_deg"),
        )
        bisector_val, bisector_cand, bisector_status = _param_or_frame(
            ("friedel_bisector_deg", "bisector_deg"),
            ("friedel_bisector_deg", "bisector_deg"),
        )

        per_frame.append({
            "frame_index": slot,
            "frame_id": str(frow.get("frame_id") or ""),
            "time_h": t_val,
            "time_unit": t_unit,
            "frame_status": frame_status,
            "q_unit": frame_q_unit,
            "L_star_nm": (l_val, l_cand, l_status),
            "q_star": (q_star_val, q_star_cand, q_star_status),
            "Q_win": (q_win_val, q_win_cand, q_win_status),
            "intensity_sum": (i_sum_val, i_sum_cand, i_sum_status),
            "A4": (a4_val, a4_cand, a4_status),
            "anisotropy": (aniso_val, aniso_cand, aniso_status),
            "half_opening_deg": (alpha_val, alpha_cand, alpha_status),
            "lobe_fwhm_deg": (fwhm_val, fwhm_cand, fwhm_status),
            "bisector_deg": (bisector_val, bisector_cand, bisector_status),
        })

    return {
        "sample": sample_name,
        "condition": condition,
        "temperature_C": temp_c,
        "beamtime": beamtime,
        "calibration_mode": cal_mode,
        "intensity_unit": intensity_unit or ("cm^-1" if cal_mode == "calibrated" else "raw_counts"),
        "q_unit": q_unit,
        "quarantined": quarantined,
        "frames": per_frame,
    }


def _series_x(frames: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, str]:
    times = [f["time_h"] for f in frames]
    units = [f["time_unit"] for f in frames]
    if frames and all(t is not None for t in times) and len(set(units)) == 1:
        u = units[0]
        label = "Time (h)" if u == "h" else (f"Time ({u})" if u else "Time")
        return np.asarray(times, dtype=np.float64), label
    return np.asarray([float(f["frame_index"]) for f in frames], dtype=np.float64), "Frame index"


def _extract_observable_arrays(
    frames: Sequence[Mapping[str, Any]], key: str
) -> tuple[np.ndarray, list[str]]:
    ys: list[float] = []
    statuses: list[str] = []
    for f in frames:
        val, cand, ev = f[key]
        if ev == "estimate" and val is not None:
            ys.append(val)
            statuses.append("estimate")
        elif ev == "candidate" and cand is not None:
            ys.append(cand)
            statuses.append("candidate")
        else:
            ys.append(float("nan"))
            statuses.append("failed" if ev == "failed" else "missing")
    return np.asarray(ys, dtype=np.float64), statuses


def _endpoint_stats(frames: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    est_vals: list[float] = []
    cand_vals: list[float] = []
    for f in frames:
        val, cand, ev = f[key]
        if ev == "estimate" and val is not None:
            est_vals.append(val)
        elif ev == "candidate" and cand is not None:
            cand_vals.append(cand)
    return {
        "early_estimate": est_vals[0] if est_vals else None,
        "late_estimate": est_vals[-1] if est_vals else None,
        "late_candidate": cand_vals[-1] if cand_vals else None,
        "n_estimate": len(est_vals),
        "n_candidate": len(cand_vals),
    }


def _build_evolution_figure(
    samples: Sequence[Mapping[str, Any]],
    output_stem: Path,
    *,
    preset: JournalPreset,
    formats: Sequence[str],
    dpi: int,
) -> dict[str, Any]:
    # Sort samples by temperature (ascending) then name for clean thermal palette assignment.
    ordered = sorted(
        samples,
        key=lambda s: (
            s["temperature_C"] is None,
            s["temperature_C"] if s["temperature_C"] is not None else 0.0,
            s["sample"],
        ),
    )
    palette = get_thermal_palette(max(1, len(ordered)))

    # Decide whether Q_win or intensity_sum is populated across calibrated samples.
    cal_samples = [s for s in ordered if s["calibration_mode"] == "calibrated"]
    raw_samples = [s for s in ordered if s["calibration_mode"] != "calibrated"]
    has_q_win = any(
        any(f["Q_win"][2] in {"estimate", "candidate"} for f in s["frames"])
        for s in cal_samples
    )
    qwin_key = "Q_win" if has_q_win else "intensity_sum"
    qwin_ylabel = r"$Q_{\mathrm{win}}$ (calibrated)" if has_q_win else "Intensity sum (calibrated)"

    has_a4 = any(
        any(f["A4"][2] in {"estimate", "candidate"} for f in s["frames"])
        for s in ordered
    )
    order_key = "A4" if has_a4 else "anisotropy"
    order_ylabel = r"4-fold order $A_4$" if has_a4 else "Tensor anisotropy"

    # Determine common x-axis label across samples.
    x_labels = {_series_x(s["frames"])[1] for s in ordered if s["frames"]}
    common_xlabel = next(iter(x_labels)) if len(x_labels) == 1 else "Time (h) / frame index"

    builder = CompositeFigureBuilder(
        preset=preset,
        width="double",
        height_mm=138.0,
        nrows=2,
        ncols=2,
        dpi=dpi,
    )
    ax_a = builder.add_panel("a", 0, 0)
    ax_b = builder.add_panel("b", 0, 1)
    ax_c = builder.add_panel("c", 1, 0)
    ax_d = builder.add_panel("d", 1, 1)

    source_l_rows: list[dict[str, Any]] = []
    source_qwin_rows: list[dict[str, Any]] = []
    source_order_rows: list[dict[str, Any]] = []
    source_land_rows: list[dict[str, Any]] = []

    l_series: list[dict[str, Any]] = []
    qwin_series: list[dict[str, Any]] = []
    order_series: list[dict[str, Any]] = []

    markers = ("o", "s", "^", "D", "v", "P", "X", "h")
    for idx, sample in enumerate(ordered):
        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]
        is_cal = sample["calibration_mode"] == "calibrated"
        tag = "" if is_cal else " [raw]"
        label = f"{sample['sample']}{tag}"
        x_arr, _ = _series_x(sample["frames"])

        # Panel a: L*(t) (physical calibrated q only).
        if is_physical_q_unit(sample["q_unit"]):
            y_l, st_l = _extract_observable_arrays(sample["frames"], "L_star_nm")
            if np.any(np.isfinite(y_l)):
                l_series.append({
                    "label": label,
                    "x": x_arr,
                    "y": y_l,
                    "status": st_l,
                    "color": color,
                    "marker": marker,
                    "open_symbol": not is_cal,
                })
                for xv, yv, sv in zip(x_arr, y_l, st_l):
                    source_l_rows.append({
                        "sample": sample["sample"],
                        "calibration_mode": sample["calibration_mode"],
                        "x": float(xv),
                        "L_star_nm": None if not np.isfinite(yv) else float(yv),
                        "evidence_status": sv,
                    })

        # Panel b: Q_win(t) STRICTLY calibrated runs only!
        if is_cal:
            y_q, st_q = _extract_observable_arrays(sample["frames"], qwin_key)
            if np.any(np.isfinite(y_q)):
                qwin_series.append({
                    "label": sample["sample"],
                    "x": x_arr,
                    "y": y_q,
                    "status": st_q,
                    "color": color,
                    "marker": marker,
                })
                for xv, yv, sv in zip(x_arr, y_q, st_q):
                    source_qwin_rows.append({
                        "sample": sample["sample"],
                        "calibration_mode": sample["calibration_mode"],
                        "x": float(xv),
                        qwin_key: None if not np.isfinite(yv) else float(yv),
                        "evidence_status": sv,
                    })

        # Panel c: A4(t) (dimensionless — calibrated + clearly tagged raw runs).
        y_o, st_o = _extract_observable_arrays(sample["frames"], order_key)
        if np.any(np.isfinite(y_o)):
            order_series.append({
                "label": label,
                "x": x_arr,
                "y": y_o,
                "status": st_o,
                "color": color,
                "marker": marker,
                "open_symbol": not is_cal,
            })
            for xv, yv, sv in zip(x_arr, y_o, st_o):
                source_order_rows.append({
                    "sample": sample["sample"],
                    "calibration_mode": sample["calibration_mode"],
                    "x": float(xv),
                    order_key: None if not np.isfinite(yv) else float(yv),
                    "evidence_status": sv,
                })

    if l_series:
        plot_evolution_panel(
            ax_a,
            l_series,
            xlabel=common_xlabel,
            ylabel=r"Correlation spacing $L^*$",
            y_unit="nm",
            title=r"Spacing evolution $L^*(t) = 2\pi/q^*$",
            preset=preset,
        )
    else:
        with journal_rc_context(preset):
            ax_a.text(
                0.5,
                0.5,
                "No calibrated physical-q L* estimates",
                transform=ax_a.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )
            ax_a.set_title(r"Spacing evolution $L^*(t)$", fontsize=preset.axis_title_size)
            style_1d_axes(ax_a, preset=preset)

    if qwin_series:
        plot_evolution_panel(
            ax_b,
            qwin_series,
            xlabel=common_xlabel,
            ylabel=qwin_ylabel,
            title=r"Scattering invariant $Q_{\mathrm{win}}(t)$ (calibrated only)" if has_q_win else "Intensity evolution (calibrated only)",
            preset=preset,
        )
        if raw_samples:
            ax_b.text(
                0.03,
                0.04,
                f"{len(raw_samples)} raw-count run(s) excluded from absolute scale",
                transform=ax_b.transAxes,
                ha="left",
                va="bottom",
                fontsize=max(preset.tick_label_size - 0.6, 4.8),
                color="#555555",
                bbox={"facecolor": "#FFFFFF", "edgecolor": "#DDDDDD", "alpha": 0.9, "pad": 1.4},
            )
    else:
        with journal_rc_context(preset):
            msg = (
                "Raw-count runs only; absolute $Q_{\\mathrm{win}}$\nnot comparable across uncalibrated runs"
                if raw_samples
                else "No calibrated $Q_{\\mathrm{win}}$ measurements"
            )
            ax_b.text(0.5, 0.5, msg, transform=ax_b.transAxes, ha="center", va="center", color="#666666")
            ax_b.set_title(r"Scattering invariant $Q_{\mathrm{win}}(t)$", fontsize=preset.axis_title_size)
            style_1d_axes(ax_b, preset=preset)

    if order_series:
        plot_evolution_panel(
            ax_c,
            order_series,
            xlabel=common_xlabel,
            ylabel=order_ylabel,
            title=r"Butterfly 4-fold order $A_4(t)$" if has_a4 else "In-plane anisotropy evolution",
            preset=preset,
        )
    else:
        with journal_rc_context(preset):
            ax_c.text(0.5, 0.5, "No azimuthal order data", transform=ax_c.transAxes, ha="center", va="center", color="#666666")
            ax_c.set_title(order_ylabel, fontsize=preset.axis_title_size)
            style_1d_axes(ax_c, preset=preset)

    # Panel d: Temperature / condition endpoint landscape.
    with journal_rc_context(preset):
        temps = [s["temperature_C"] for s in ordered]
        use_temp_axis = bool(ordered) and all(t is not None for t in temps)
        x_pos = np.asarray(temps if use_temp_axis else list(range(len(ordered))), dtype=np.float64)
        oi = get_okabe_ito_palette()
        c_l, c_a = oi[0], oi[1]

        ax_d_right = ax_d.twinx()
        l_drawn = False
        a_drawn = False
        for idx, sample in enumerate(ordered):
            l_ep = _endpoint_stats(sample["frames"], "L_star_nm")
            a_ep = _endpoint_stats(sample["frames"], order_key)
            is_cal = sample["calibration_mode"] == "calibrated"
            xv = x_pos[idx]
            l_val = l_ep["late_estimate"] if l_ep["late_estimate"] is not None else l_ep["late_candidate"]
            l_is_cand = l_ep["late_estimate"] is None and l_ep["late_candidate"] is not None
            a_val = a_ep["late_estimate"] if a_ep["late_estimate"] is not None else a_ep["late_candidate"]
            a_is_cand = a_ep["late_estimate"] is None and a_ep["late_candidate"] is not None

            if l_val is not None:
                ax_d.scatter(
                    [xv],
                    [l_val],
                    s=30,
                    marker=("o" if is_cal else "^"),
                    facecolors=("#FFFFFF" if (l_is_cand or not is_cal) else c_l),
                    edgecolors=c_l,
                    linewidths=0.95,
                    zorder=4,
                    label=(r"Endpoint $L^*$ (est.)" if (not l_drawn and not l_is_cand) else None),
                )
                l_drawn = True
            if a_val is not None:
                ax_d_right.scatter(
                    [xv],
                    [a_val],
                    s=28,
                    marker=("s" if is_cal else "D"),
                    facecolors=("#FFFFFF" if (a_is_cand or not is_cal) else c_a),
                    edgecolors=c_a,
                    linewidths=0.95,
                    zorder=4,
                    label=(f"Endpoint {order_key}" if not a_drawn else None),
                )
                a_drawn = True

            source_land_rows.append({
                "sample": sample["sample"],
                "temperature_C": sample["temperature_C"],
                "calibration_mode": sample["calibration_mode"],
                "L_star_endpoint_nm": l_val,
                "L_star_evidence": "candidate" if l_is_cand else ("estimate" if l_val is not None else "missing"),
                f"{order_key}_endpoint": a_val,
                f"{order_key}_evidence": "candidate" if a_is_cand else ("estimate" if a_val is not None else "missing"),
            })

        ax_d.set_xlabel(
            format_axis_label("Temperature", "degC") if use_temp_axis else "Sample condition",
            fontsize=preset.axis_label_size,
        )
        if not use_temp_axis and ordered:
            ax_d.set_xticks(x_pos)
            ax_d.set_xticklabels([s["sample"] for s in ordered], rotation=30, ha="right", fontsize=max(preset.tick_label_size - 0.5, 5.0))
        ax_d.set_ylabel(r"Endpoint $L^*$ ($\mathrm{nm}$)", color=c_l, fontsize=preset.axis_label_size)
        ax_d_right.set_ylabel(f"Endpoint {order_ylabel}", color=c_a, fontsize=preset.axis_label_size)
        ax_d.set_title("Endpoint temperature / condition landscape", fontsize=preset.axis_title_size, pad=3.0)
        style_1d_axes(ax_d, preset=preset, grid=True)
        ax_d_right.spines["top"].set_visible(False)
        ax_d_right.spines["right"].set_visible(True)
        ax_d_right.spines["right"].set_linewidth(preset.spine_linewidth)
        ax_d_right.tick_params(
            axis="y",
            direction=preset.tick_direction,
            length=preset.tick_length,
            width=preset.tick_width,
            labelsize=preset.tick_label_size,
            colors=c_a,
        )
        ax_d.tick_params(axis="y", colors=c_l)
        if not l_drawn and not a_drawn:
            ax_d.text(0.5, 0.5, "No finite endpoint observables", transform=ax_d.transAxes, ha="center", va="center", color="#666666")

    builder.register_source_data("panel_a_L_star_evolution", source_l_rows)
    builder.register_source_data("panel_b_Q_win_calibrated", source_qwin_rows)
    builder.register_source_data("panel_c_order_evolution", source_order_rows)
    builder.register_source_data("panel_d_endpoint_landscape", source_land_rows)

    manifest = builder.export(
        output_stem,
        formats=formats,
        dpi=dpi,
        metadata={
            "title": "Cross-sample SAXS evolution overview",
            "description": "Multi-sample L*(t), calibrated Q_win(t), azimuthal order, and endpoint landscape.",
            "n_samples": len(ordered),
            "n_calibrated": len(cal_samples),
            "n_raw_or_mixed": len(raw_samples),
        },
    )
    builder.figure.clear()
    return manifest


def _build_qc_heatmap_figure(
    endpoint_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
    *,
    preset: JournalPreset,
    formats: Sequence[str],
    dpi: int,
) -> dict[str, Any]:
    metrics = [
        ("reported_fraction", "Reported\nframes"),
        ("is_calibrated_flag", "Calibrated\nabs. scale"),
        ("L_star_estimate_fraction", r"$L^*$ est." "\nrate"),
        ("L_star_candidate_fraction", r"$L^*$ cand." "\nonly"),
        ("order_coverage_fraction", r"$A_4$/aniso." "\ncoverage"),
        ("failed_fraction", "Failed /\nmissing"),
    ]
    n_samples = max(1, len(endpoint_rows))
    height_mm = min(preset.max_height_mm, max(62.0, 28.0 + 7.5 * n_samples))
    builder = CompositeFigureBuilder(
        preset=preset,
        width="double",
        height_mm=height_mm,
        nrows=1,
        ncols=1,
        dpi=dpi,
    )
    ax = builder.add_panel("a", 0, 0)

    matrix = np.full((len(endpoint_rows), len(metrics)), np.nan, dtype=np.float64)
    sample_labels: list[str] = []
    source_rows: list[dict[str, Any]] = []
    for r_idx, row in enumerate(endpoint_rows):
        cal_badge = "cal" if row["calibration_mode"] == "calibrated" else row["calibration_mode"]
        sample_labels.append(f"{row['sample']} ({cal_badge})")
        src_row: dict[str, Any] = {
            "sample": row["sample"],
            "calibration_mode": row["calibration_mode"],
            "q_unit": row["q_unit"],
        }
        for c_idx, (mkey, _mlabel) in enumerate(metrics):
            val = _num(row.get(mkey))
            if val is not None:
                matrix[r_idx, c_idx] = val
            src_row[mkey] = val
        source_rows.append(src_row)

    with journal_rc_context(preset):
        cmap = get_saxs_colormap("cividis", preset=preset)
        if endpoint_rows:
            im = ax.imshow(
                np.ma.masked_invalid(matrix),
                aspect="auto",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_xticks(np.arange(len(metrics)))
            ax.set_xticklabels([mlabel for _, mlabel in metrics], fontsize=preset.tick_label_size)
            ax.set_yticks(np.arange(len(endpoint_rows)))
            ax.set_yticklabels(sample_labels, fontsize=preset.tick_label_size)
            for r_idx in range(len(endpoint_rows)):
                for c_idx in range(len(metrics)):
                    val = matrix[r_idx, c_idx]
                    if np.isfinite(val):
                        txt_color = "#111111" if val > 0.58 else "#FFFFFF"
                        ax.text(
                            c_idx,
                            r_idx,
                            f"{val * 100.0:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=max(preset.tick_label_size - 0.5, 5.0),
                            color=txt_color,
                            fontweight="medium",
                        )
                    else:
                        ax.text(c_idx, r_idx, "—", ha="center", va="center", fontsize=preset.tick_label_size, color="#444444")
            cbar = builder.figure.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
            cbar.set_label("Fraction (0–1)", fontsize=preset.tick_label_size)
            cbar.ax.tick_params(labelsize=max(preset.tick_label_size - 0.5, 5.0))
        else:
            ax.text(0.5, 0.5, "No samples provided", transform=ax.transAxes, ha="center", va="center", color="#666666")
        ax.set_title("Cross-sample QC completeness & calibration matrix", fontsize=preset.axis_title_size, pad=4.0)
        style_2d_axes(ax, preset=preset, equal_aspect=False)

    builder.register_source_data("panel_a_qc_completeness", source_rows)
    manifest = builder.export(
        output_stem,
        formats=formats,
        dpi=dpi,
        metadata={
            "title": "Cross-sample QC completeness heatmap",
            "description": "Frame completeness, calibration separation, and estimate vs candidate rates per sample.",
        },
    )
    builder.figure.clear()
    return manifest


def _render_markdown(
    summary: Mapping[str, Any],
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# WingSAXS Cross-Sample Comparison Report",
        "",
        f"- **Journal style**: `{summary['journal_style']}`",
        f"- **Samples**: {summary['sample_count']} "
        f"(calibrated: {len(summary['calibrated_samples'])}, "
        f"raw/mixed: {len(summary['raw_count_samples'])})",
        "",
        "## Scientific Boundaries",
        "- Calibrated absolute-intensity runs (`cm^-1`) and raw-count runs are strictly separated on absolute intensity scales (`Q_win`).",
        "- Finite `estimate` values, `candidate`-only values, and `failed`/`missing` frames are reported separately and never conflated or interpolated across.",
        "- Real-space spacings `L* = 2*pi/q*` in nm are reported only for samples with calibrated physical `q` units.",
        "",
        "## Cross-Sample Endpoint Summary",
        "",
        "| Sample | Condition | Calibration | q unit | Frames | Reported % | L* late est. (nm) | L* late cand. (nm) | A4 late est. |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in endpoint_rows:
        def _fmt(val: Any) -> str:
            n = _num(val)
            return "—" if n is None else f"{n:.3g}"

        rep_pct = _num(row.get("reported_fraction"))
        rep_str = "—" if rep_pct is None else f"{rep_pct * 100:.1f}%"
        lines.append(
            f"| {row['sample']} | {row['condition']} | {row['calibration_mode']} | {row['q_unit']} "
            f"| {row['n_frames']} | {rep_str} | {_fmt(row['L_star_late_estimate_nm'])} "
            f"| {_fmt(row['L_star_late_candidate_nm'])} | {_fmt(row['A4_late_estimate'])} |"
        )
    lines.extend([
        "",
        "## Tables & Figures",
        f"- Endpoint summary CSV: [`{summary['tables']['endpoint_summary']}`]({summary['tables']['endpoint_summary']})",
        f"- Long evolution CSV: [`{summary['tables']['evolution_long']}`]({summary['tables']['evolution_long']})",
    ])
    for fig_name, rel_files in summary["figures"].items():
        links = ", ".join(f"[{Path(p).suffix.lstrip('.').upper()}]({p})" for p in rel_files)
        lines.append(f"- **{fig_name}**: {links}")
    lines.append("")
    return "\n".join(lines)


def _render_html(
    summary: Mapping[str, Any],
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> str:
    def _a(rel_path: str, label: str) -> str:
        return f'<a href="{quote(rel_path, safe="/")}">{html.escape(label)}</a>'

    def _fmt(val: Any) -> str:
        n = _num(val)
        return "—" if n is None else f"{n:.4g}"

    table_rows_html = []
    for row in endpoint_rows:
        rep = _num(row.get("reported_fraction"))
        rep_txt = "—" if rep is None else f"{rep * 100:.1f}%"
        table_rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['sample']))}</td>"
            f"<td>{html.escape(str(row['condition']))}</td>"
            f"<td>{html.escape(str(row['calibration_mode']))}</td>"
            f"<td>{html.escape(str(row['q_unit']))}</td>"
            f"<td>{int(row['n_frames'])}</td>"
            f"<td>{rep_txt}</td>"
            f"<td>{_fmt(row['L_star_late_estimate_nm'])}</td>"
            f"<td>{_fmt(row['L_star_late_candidate_nm'])}</td>"
            f"<td>{_fmt(row['Q_win_late_estimate'])}</td>"
            f"<td>{_fmt(row['A4_late_estimate'])}</td>"
            "</tr>"
        )

    fig_blocks = []
    for fig_name, rel_files in summary["figures"].items():
        links = " · ".join(_a(p, Path(p).name) for p in rel_files)
        png = next((p for p in rel_files if p.endswith(".png")), None)
        img_tag = f'<p><img src="{quote(png, safe="/")}" alt="{html.escape(fig_name)}"></p>' if png else ""
        fig_blocks.append(f"<section><h2>{html.escape(fig_name.replace('_', ' ').title())}</h2><p>{links}</p>{img_tag}</section>")

    table_links = " · ".join(
        _a(rel, name.replace("_", " "))
        for name, rel in summary["tables"].items()
    )
    md_link = _a("comparison_summary.md", "Markdown summary")
    json_link = _a("comparison_summary.json", "JSON summary")

    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>WingSAXS Cross-Sample Comparison Report</title>"
        "<style>"
        "body{font:15px/1.5 system-ui,-apple-system,Arial,sans-serif;color:#1f2937;background:#f8fafc;max-width:1240px;margin:32px auto;padding:0 24px}"
        "a{color:#0072B2;text-decoration:none}a:hover{text-decoration:underline}"
        "section{background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px 20px;margin:16px 0}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{text-align:left;padding:7px 9px;border-bottom:1px solid #e5e7eb}"
        "th{background:#f1f5f9;font-weight:600}"
        "img{max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:4px}"
        ".note{color:#475569;font-size:13px}"
        "</style>"
        "<h1>WingSAXS Cross-Sample Comparison Report</h1>"
        f'<p class="note">Preset: <strong>{html.escape(str(summary["journal_style"]))}</strong> · '
        f'Samples: {summary["sample_count"]} (calibrated: {len(summary["calibrated_samples"])}, '
        f'raw/mixed: {len(summary["raw_count_samples"])})</p>'
        f"<section><h2>Exports</h2><p>{table_links} · {md_link} · {json_link}</p>"
        '<p class="note">Calibrated absolute-intensity runs and raw-count runs are strictly separated on '
        "intensity-scale axes. Estimates, candidate-only values, and failed/missing frames remain distinct.</p></section>"
        "<section><h2>Endpoint Summary</h2><table><thead><tr>"
        "<th>Sample</th><th>Condition</th><th>Calibration</th><th>q unit</th><th>Frames</th>"
        "<th>Reported</th><th>L* late est. (nm)</th><th>L* late cand. (nm)</th>"
        "<th>Q_win late est.</th><th>A4 late est.</th>"
        f"</tr></thead><tbody>{''.join(table_rows_html)}</tbody></table></section>"
        + "".join(fig_blocks)
        + "</html>"
    )


def build_comparison_report(
    sample_entries: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    journal_style: str | JournalPreset = "nature",
    formats: Sequence[str] = ("pdf", "svg", "png"),
    dpi: int | None = None,
) -> dict[str, Any]:
    """Generate cross-sample comparative CSVs, Nature-styled figures, and HTML/Markdown summary.

    Parameters
    ----------
    sample_entries:
        Sequence of sample descriptor mappings (accepting in-memory ``measurements`` /
        ``parameters`` / ``frames`` lists and/or ``measurements_path`` / ``parameters_path`` /
        ``report_dir`` / ``batch_dir`` paths).
    output_dir:
        Destination directory for comparative tables, figures, manifests, and HTML/MD indices.
    journal_style:
        Journal preset name (``"nature"``, ``"nature_comm"``, ``"science"``, ``"macromolecules"``)
        or ``JournalPreset`` instance.
    formats:
        Figure formats to export via ``export_publication_figure``.
    dpi:
        Raster DPI (defaults to the preset DPI, e.g., 600 for Nature).
    """
    preset = get_journal_preset(journal_style)
    export_dpi = preset.dpi if dpi is None else int(dpi)
    if isinstance(export_dpi, bool) or export_dpi <= 0:
        raise ValueError(f"dpi must be a positive integer, got {dpi!r}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    loaded_samples = [_load_sample_data(entry, idx) for idx, entry in enumerate(sample_entries)]

    # Build long evolution table & endpoint summary table.
    long_columns = (
        "sample",
        "condition",
        "temperature_C",
        "beamtime",
        "calibration_mode",
        "q_unit",
        "intensity_unit",
        "frame_index",
        "frame_id",
        "time_h",
        "time_unit",
        "frame_status",
        "observable",
        "value",
        "candidate_value",
        "evidence_status",
        "unit",
    )
    observable_specs = (
        ("L_star_nm", "nm"),
        ("q_star", None),
        ("Q_win", None),
        ("intensity_sum", None),
        ("A4", "dimensionless"),
        ("anisotropy", "dimensionless"),
        ("half_opening_deg", "deg"),
        ("lobe_fwhm_deg", "deg"),
        ("bisector_deg", "deg"),
    )
    long_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []

    for sample in loaded_samples:
        frames = sample["frames"]
        n_frames = len(frames)
        n_failed = sum(1 for f in frames if f["frame_status"] in {"failed", "failure", "error", "not_processed"})
        n_reported = n_frames - n_failed

        l_ep = _endpoint_stats(frames, "L_star_nm")
        qwin_ep = _endpoint_stats(frames, "Q_win")
        isum_ep = _endpoint_stats(frames, "intensity_sum")
        a4_ep = _endpoint_stats(frames, "A4")
        aniso_ep = _endpoint_stats(frames, "anisotropy")
        alpha_ep = _endpoint_stats(frames, "half_opening_deg")
        fwhm_ep = _endpoint_stats(frames, "lobe_fwhm_deg")

        order_covered = max(
            a4_ep["n_estimate"] + a4_ep["n_candidate"],
            aniso_ep["n_estimate"] + aniso_ep["n_candidate"],
        )
        endpoint_rows.append({
            "sample": sample["sample"],
            "condition": sample["condition"],
            "temperature_C": sample["temperature_C"],
            "beamtime": sample["beamtime"],
            "calibration_mode": sample["calibration_mode"],
            "q_unit": sample["q_unit"],
            "intensity_unit": sample["intensity_unit"],
            "quarantined": sample["quarantined"],
            "n_frames": n_frames,
            "n_reported": n_reported,
            "n_failed": n_failed,
            "reported_fraction": (n_reported / n_frames) if n_frames else 0.0,
            "failed_fraction": (n_failed / n_frames) if n_frames else 0.0,
            "is_calibrated_flag": 1.0 if sample["calibration_mode"] == "calibrated" else (0.5 if sample["calibration_mode"] == "mixed" else 0.0),
            "L_star_estimate_count": l_ep["n_estimate"],
            "L_star_candidate_count": l_ep["n_candidate"],
            "L_star_estimate_fraction": (l_ep["n_estimate"] / n_frames) if n_frames else 0.0,
            "L_star_candidate_fraction": (l_ep["n_candidate"] / n_frames) if n_frames else 0.0,
            "L_star_early_estimate_nm": l_ep["early_estimate"],
            "L_star_late_estimate_nm": l_ep["late_estimate"],
            "L_star_late_candidate_nm": l_ep["late_candidate"],
            "Q_win_late_estimate": qwin_ep["late_estimate"] if sample["calibration_mode"] == "calibrated" else None,
            "Q_win_late_candidate": qwin_ep["late_candidate"] if sample["calibration_mode"] == "calibrated" else None,
            "intensity_sum_late_estimate": isum_ep["late_estimate"],
            "A4_early_estimate": a4_ep["early_estimate"],
            "A4_late_estimate": a4_ep["late_estimate"],
            "A4_late_candidate": a4_ep["late_candidate"],
            "anisotropy_late_estimate": aniso_ep["late_estimate"],
            "order_coverage_fraction": (order_covered / n_frames) if n_frames else 0.0,
            "half_opening_late_deg": alpha_ep["late_estimate"] if alpha_ep["late_estimate"] is not None else alpha_ep["late_candidate"],
            "lobe_fwhm_late_deg": fwhm_ep["late_estimate"] if fwhm_ep["late_estimate"] is not None else fwhm_ep["late_candidate"],
        })

        for f in frames:
            for obs_key, fixed_unit in observable_specs:
                val, cand, ev = f[obs_key]
                if fixed_unit is not None:
                    obs_unit = fixed_unit
                elif obs_key == "q_star":
                    obs_unit = f["q_unit"]
                else:
                    obs_unit = sample["intensity_unit"]
                # Preserve calibrated vs raw-count separation on absolute Q_win rows.
                long_rows.append({
                    "sample": sample["sample"],
                    "condition": sample["condition"],
                    "temperature_C": sample["temperature_C"],
                    "beamtime": sample["beamtime"],
                    "calibration_mode": sample["calibration_mode"],
                    "q_unit": f["q_unit"],
                    "intensity_unit": sample["intensity_unit"],
                    "frame_index": f["frame_index"],
                    "frame_id": f["frame_id"],
                    "time_h": f["time_h"],
                    "time_unit": f["time_unit"],
                    "frame_status": f["frame_status"],
                    "observable": obs_key,
                    "value": val,
                    "candidate_value": cand,
                    "evidence_status": ev,
                    "unit": obs_unit,
                })

    endpoint_columns = (
        tuple(endpoint_rows[0].keys())
        if endpoint_rows
        else (
            "sample",
            "condition",
            "temperature_C",
            "beamtime",
            "calibration_mode",
            "q_unit",
            "intensity_unit",
            "quarantined",
            "n_frames",
            "n_reported",
            "n_failed",
            "reported_fraction",
            "failed_fraction",
            "is_calibrated_flag",
            "L_star_estimate_count",
            "L_star_candidate_count",
            "L_star_estimate_fraction",
            "L_star_candidate_fraction",
            "L_star_early_estimate_nm",
            "L_star_late_estimate_nm",
            "L_star_late_candidate_nm",
            "Q_win_late_estimate",
            "Q_win_late_candidate",
            "intensity_sum_late_estimate",
            "A4_early_estimate",
            "A4_late_estimate",
            "A4_late_candidate",
            "anisotropy_late_estimate",
            "order_coverage_fraction",
            "half_opening_late_deg",
            "lobe_fwhm_late_deg",
        )
    )
    endpoint_csv = out_root / "cross_sample_endpoint_summary.csv"
    evolution_csv = out_root / "cross_sample_evolution_long.csv"
    _write_csv(endpoint_csv, endpoint_columns, endpoint_rows)
    _write_csv(evolution_csv, long_columns, long_rows)

    evo_manifest = _build_evolution_figure(
        loaded_samples,
        fig_dir / "comparison_evolution_overview",
        preset=preset,
        formats=formats,
        dpi=export_dpi,
    )
    qc_manifest = _build_qc_heatmap_figure(
        endpoint_rows,
        fig_dir / "comparison_qc_completeness",
        preset=preset,
        formats=formats,
        dpi=export_dpi,
    )

    def _rel_figure_files(manifest: Mapping[str, Any]) -> list[str]:
        rels: list[str] = []
        for fmt, abs_path in manifest["figure_paths"].items():
            rels.append(Path(abs_path).relative_to(out_root).as_posix())
        rels.append(Path(manifest["manifest_path"]).relative_to(out_root).as_posix())
        return rels

    calibrated_names = [s["sample"] for s in loaded_samples if s["calibration_mode"] == "calibrated"]
    raw_names = [s["sample"] for s in loaded_samples if s["calibration_mode"] != "calibrated"]

    summary: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA,
        "journal_style": preset.name,
        "sample_count": len(loaded_samples),
        "calibrated_samples": calibrated_names,
        "raw_count_samples": raw_names,
        "tables": {
            "endpoint_summary": endpoint_csv.name,
            "evolution_long": evolution_csv.name,
        },
        "figures": {
            "comparison_evolution_overview": _rel_figure_files(evo_manifest),
            "comparison_qc_completeness": _rel_figure_files(qc_manifest),
        },
        "manifests": {
            "comparison_evolution_overview": Path(evo_manifest["manifest_path"]).relative_to(out_root).as_posix(),
            "comparison_qc_completeness": Path(qc_manifest["manifest_path"]).relative_to(out_root).as_posix(),
        },
    }

    md_path = out_root / "comparison_summary.md"
    html_path = out_root / "comparison_index.html"
    json_path = out_root / "comparison_summary.json"

    md_path.write_text(_render_markdown(summary, endpoint_rows), encoding="utf-8")
    html_path.write_text(_render_html(summary, endpoint_rows), encoding="utf-8")

    summary["outputs"] = {
        "html": str(html_path),
        "markdown": str(md_path),
        "summary_json": str(json_path),
        "endpoint_summary_csv": str(endpoint_csv),
        "evolution_long_csv": str(evolution_csv),
    }
    json_path.write_text(
        json.dumps(json_safe(summary), ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "COMPARISON_SCHEMA",
    "build_comparison_report",
]
