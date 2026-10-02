"""Measure and present existing 2-D batch results without repeating their fits.

Native NPZ arrays remain the source of the detector data. This module reads
one frame at a time and writes tabular measurements, reusable figures and a
local HTML entry point. Candidate estimates and missing frames stay visible.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
import csv
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import quote

import numpy as np

from .csv_utils import iter_csv_rows, safe_csv_cell
from .serialization import json_safe

SCHEMA = "wingsaxs.analysis_report.v1"
_VERSION = 1


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(iter_csv_rows(handle))


def _plot_parameter_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    names = {"frame_index", "frame_id", "time", "time_s", "time_unit", "status", "parameter", "value", "stderr", "unit",
             "confidence", "publication_status", "identifiability_status", "parameter_source", "candidate_value"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: value for key, value in row.items() if key in names} for row in iter_csv_rows(handle)]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _same_contents(first: Path, second: Path) -> bool:
    """Compare generated collection tables without loading them into memory."""
    if not second.is_file() or first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while chunk := left.read(1024 * 1024):
            if chunk != right.read(len(chunk)):
                return False
    return True


class _Table:
    """Append tables as frames finish; do not retain detector-sized data."""

    def __init__(self, path: Path, columns: Sequence[str]):
        self.path = path
        self.columns = tuple(columns)
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.columns)
        self.writer.writeheader()
        self.count = 0

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            values = {}
            for key in self.columns:
                value = row.get(key)
                if isinstance(value, (Mapping, list, tuple)):
                    value = json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)
                values[key] = safe_csv_cell(value)
            self.writer.writerow(values)
            self.count += 1

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> _Table:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _detail_records(folder: Path, prefix: str) -> Iterator[dict[str, Any]]:
    for name in ("frame_details.jsonl", "ellipse_fit.jsonl"):
        path = folder / f"{prefix}{name}"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError(f"Expected a frame object in {path.name}")
                        yield record
            return
    path = folder / f"{prefix}ellipse_fit.json"
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        yield from document.get("frames", ())


def _array_groups(names: Sequence[str]) -> dict[int, dict[str, str]]:
    groups: dict[int, dict[str, str]] = defaultdict(dict)
    for name in names:
        match = re.match(r"frame_(\d+)__(.+)", name)
        if match:
            groups[int(match[1])][match[2]] = name
    return groups


def _get_array(archive: Any, keys: Mapping[str, str], *names: str) -> np.ndarray | None:
    for name in names:
        if name in keys:
            return np.asarray(archive[keys[name]])
    return None


def _coordinates(archive: Any, keys: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    qx = _get_array(archive, keys, "qmap__qx", "qmap__qx_nm_inv", "qx", "qx_nm_inv")
    qy = _get_array(archive, keys, "qmap__qy", "qmap__qy_nm_inv", "qy", "qy_nm_inv")
    if qx is None or qy is None:
        raise ValueError("Stored qx/qy maps are missing; export this frame with its calibration before reporting it.")
    if qx.ndim != 2 or qy.shape != qx.shape:
        raise ValueError("Stored qx/qy maps must be matching two-dimensional arrays.")
    mask = _get_array(archive, keys, "analysis_domain__fit_valid_mask", "valid_mask", "fit_valid_mask", "qmap__valid_mask")
    if mask is not None and mask.shape != qx.shape:
        raise ValueError("Stored usable-pixel mask does not match the q maps.")
    return qx, qy, None if mask is None else mask.astype(bool)


def _result_detail(record: Mapping[str, Any]) -> Mapping[str, Any]:
    result = record.get("result")
    return result if isinstance(result, Mapping) else record


def _restore_profile_arrays(value: Any, archive: Any, keys: Mapping[str, str], path: str = "") -> Any:
    """Restore small profile vectors for drawing, leaving detector maps linked."""
    if isinstance(value, Mapping):
        shape = value.get("shape")
        if isinstance(shape, list) and len(shape) == 1 and path in keys:
            return np.asarray(archive[keys[path]])
        return {key: _restore_profile_arrays(item, archive, keys,
                    path + ("__" if path else "") + re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key)).strip("_"))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_profile_arrays(item, archive, keys, f"{path}__{index}") for index, item in enumerate(value)]
    return value


def _q_unit(record: Mapping[str, Any]) -> str:
    from .settings import canonical_q_unit

    result = _result_detail(record)
    qmap = result.get("qmap", {})
    if isinstance(qmap, Mapping) and qmap.get("q_unit"):
        return canonical_q_unit(qmap["q_unit"])
    ellipse = result.get("ellipse_fit", {})
    if isinstance(ellipse, Mapping) and ellipse.get("q_unit"):
        return canonical_q_unit(ellipse["q_unit"])
    return canonical_q_unit(record.get("q_unit") or "unknown")


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Expose scalar diagnostics and supplied profile samples as searchable rows."""
    if isinstance(value, Mapping):
        # Detector arrays are linked to NPZ, never expanded into scalar rows.
        if "shape" in value and ("dtype" in value or "array_omitted" in value):
            return
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def _diagnostic_rows(base: Mapping[str, Any], record: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    result = _result_detail(record)
    for name in ("ellipse_fit", "butterfly", "full2d", "analysis", "flags", "analysis_domain", "fit_audit"):
        if name in result:
            for path, value in _leaves(result[name], name):
                yield {**base, "field": path, "value": value, "value_type": type(value).__name__}
    observables = result.get("observables", {})
    if isinstance(observables, Mapping) and "butterfly" in observables and "butterfly" not in result:
        for path, value in _leaves(observables["butterfly"], "observables.butterfly"):
            yield {**base, "field": path, "value": value, "value_type": type(value).__name__}


def _candidate_rows(base: Mapping[str, Any], record: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = _result_detail(record)
    ellipse = result.get("ellipse_fit", {})
    if not isinstance(ellipse, Mapping):
        return []
    candidates = ellipse.get("candidate_solutions", [])
    if isinstance(candidates, Mapping):
        candidates = list(candidates.values())
    rows = []
    for index, candidate in enumerate(candidates or ()):
        if not isinstance(candidate, Mapping):
            continue
        for field, value in _leaves(candidate):
            rows.append({**base, "candidate_index": index, "selected_start_index": ellipse.get("selected_start_index"),
                         "field": field, "value": value, "value_type": type(value).__name__})
    return rows


def _normal_profile_rows(base: Mapping[str, Any], result: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    butterfly = result.get("butterfly")
    if not isinstance(butterfly, Mapping):
        butterfly = result.get("observables", {}).get("butterfly", {})
    profiles = butterfly.get("profiles", []) if isinstance(butterfly, Mapping) else []
    if isinstance(profiles, Mapping):
        profiles = list(profiles.values())
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            continue
        vectors = {}
        for name in ("offset_q", "raw_intensity", "fit_intensity"):
            value = profile.get(name)
            vectors[name] = value if isinstance(value, (list, np.ndarray)) else []
        count = max((len(value) for value in vectors.values()), default=0)
        for sample in range(count):
            values = {name: _number(value[sample]) if sample < len(value) else None for name, value in vectors.items()}
            observed, fitted = values["raw_intensity"], values["fit_intensity"]
            yield {**base, "profile_index": index, "sample_index": sample, **values,
                   "residual": observed - fitted if observed is not None and fitted is not None else None,
                   **{name: profile.get(name) for name in ("point_id", "valid", "model", "reason", "snr", "normal_fwhm_q", "localization_sigma_q", "support_fraction", "uncertainty_source")}}


def _profile_rows(base: Mapping[str, Any], archive: Any, keys: Mapping[str, str], array_table: _Table) -> Iterator[dict[str, Any]]:
    for role, key in keys.items():
        # Every native array is catalogued; only one-dimensional measured
        # profile/fit vectors are expanded for ordinary spreadsheet use.
        with archive.zip.open(key + ".npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                # Unicode structured dtypes use NPY v3. Let NumPy's public
                # reader handle that uncommon representation.
                handle.seek(0)
                value = np.load(handle, allow_pickle=False)
                shape, dtype = value.shape, value.dtype
        array_table.write([{**base, "role": role, "array_key": key, "shape": json.dumps(shape), "dtype": str(dtype)}])
        if len(shape) != 1 or not any(word in role for word in ("profile", "angular", "residual", "assignment", "covariance")):
            continue
        array = np.asarray(archive[key])
        for index, value in enumerate(array):
            yield {**base, "profile": role, "sample_index": index, "value": value.item(), "array_key": key}


def _parameter_trends(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("parameter", row.get("name", ""))), str(row.get("unit") or "unknown")].append(row)
    trends = []
    for (parameter, unit), group in groups.items():
        for field in ("value", "candidate_value"):
            finite = [(row, _number(row.get(field))) for row in group]
            finite = [(row, value) for row, value in finite if value is not None]
            if not finite:
                continue
            values = np.asarray([value for row, value in finite])
            first, last = finite[0], finite[-1]
            trends.append({"parameter": parameter, "unit": unit, "value_field": field,
                           "n_records": len(group), "n_finite": len(finite),
                           "first_frame_index": first[0].get("frame_index"), "last_frame_index": last[0].get("frame_index"),
                           "first_value": first[1], "last_value": last[1], "change": last[1] - first[1],
                           "mean": float(values.mean()), "std_across_frames": float(values.std(ddof=1)) if len(values) > 1 else None,
                           "min": float(values.min()), "max": float(values.max())})
    return trends


def _compact_profile(rows: Sequence[Mapping[str, Any]], *, radial: bool) -> dict[str, np.ndarray]:
    coordinates = ("q_min", "q_max", "q_center") if radial else ("angle_min_deg", "angle_max_deg", "angle_center_deg")
    return {name: np.asarray([row.get(name) for row in rows], dtype=float)
            for name in (*coordinates, "mean", "count", "coverage")}


def _link(path: str, label: str) -> str:
    return f'<a href="{quote(path, safe="/")}">{html.escape(label)}</a>'


def _page(title: str, body: str) -> str:
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>body{{font:16px system-ui,sans-serif;color:#243345;background:#f6f8fb;max-width:1280px;margin:36px auto;padding:0 24px}}'
            'a{color:#0865a1}section{background:white;border:1px solid #dfe5ec;border-radius:8px;padding:20px;margin:18px 0}'
            'table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e2e8ef;vertical-align:top}'
            'img{width:100%;max-width:1100px;height:auto}.scroll{overflow:auto}.note{color:#536477}code{overflow-wrap:anywhere}</style>'
            f'<h1>{html.escape(title)}</h1>{body}</html>')


def _render_index(report: Mapping[str, Any]) -> str:
    source_links = " · ".join(_link(path, name) for name, path in report["sources"].items())
    tables = " · ".join(_link(path, name.replace("_", " ")) for name, path in report["tables"].items())
    sequence = "".join(f'<p>{_link(path, Path(path).name)}</p>' +
                       (f'<img src="{quote(path, safe="/")}" loading="lazy" alt="Sequence measurements">' if path.endswith(".png") else "")
                       for path in report["sequence_figures"].values())
    frames = []
    for frame in report["frames"]:
        figures = " · ".join(_link(path, Path(path).name) for path in frame.get("figures", {}).values())
        data = " · ".join(_link(path, label.replace("_", " ")) for label, path in frame.get("data", {}).items())
        image = next((path for path in frame.get("figures", {}).values() if path.endswith(".png")), None)
        note = html.escape(str(frame.get("error") or frame.get("diagnostic") or ""))
        frames.append(f'<details><summary>Frame {frame["frame_index"]}: {html.escape(str(frame.get("frame_id", "")))} — '
                      f'{html.escape(str(frame.get("status", "")))} / {html.escape(str(frame.get("report_status", "")))}</summary>'
                      f'<p>{note}</p><p>{data}</p><p>{figures}</p>' +
                      (f'<img src="{quote(image, safe="/")}" loading="lazy" alt="Observed frame measurements">' if image else "") + '</details>')
    return _page("WingSAXS analysis report", '<p>Observed 2-D measurements and existing fit estimates. '
                 'Open the native data to inspect full detector arrays and fit records.</p>'
                 f'<section><h2>Data</h2><p>{source_links}</p><p>{tables}</p>'
                 '<p class="note">Empty bins and missing frames remain missing. Profile SEM describes within-bin pixel sampling; '
                 'it is not a calibrated measurement uncertainty. Intensity and in-plane moments describe the supplied mask and q window.</p></section>'
                 f'<section><h2>Sequence</h2>{sequence}</section><section><h2>Frames ({len(frames)})</h2>{"".join(frames)}</section>')


def _source_state(folder: Path, prefix: str) -> dict[str, list[int]]:
    return {name: [path.stat().st_size, path.stat().st_mtime_ns] for name in (
        "frame_summary.csv", "parameters_long.csv", "results.npz", "frame_details.jsonl", "ellipse_fit.jsonl", "ellipse_fit.json", "ridge_points.csv", "provenance.json", "manifest.json")
        if (path := folder / f"{prefix}{name}").is_file()}


def _manifest_time_units(folder: Path, prefix: str) -> list[str]:
    path = folder / f"{prefix}manifest.json"
    if not path.is_file():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    common = manifest.get("time_unit") or (metadata.get("time_unit") if isinstance(metadata, Mapping) else None)
    units = []
    for frame in manifest.get("frames", []):
        metadata = frame.get("metadata", {})
        unit = frame.get("time_unit") or (metadata.get("time_unit") if isinstance(metadata, Mapping) else None) or common
        units.append(str(unit or ""))
    return units


def _build_sample(summary_path: Path, *, force: bool, resume: bool, settings: Mapping[str, Any], progress: Callable[[str], Any] | None) -> dict[str, Any]:
    from .report_figures import render_frame_report, render_sequence_report
    from .report_measurements import measure_frame

    folder = summary_path.parent
    prefix = summary_path.name.removesuffix("frame_summary.csv")
    archive_path = folder / f"{prefix}results.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Native arrays are missing: {archive_path}; finish the batch export first.")
    target = folder / "figures" / f"{prefix}analysis_report"
    state = _source_state(folder, prefix)
    receipt = target / "report_summary.json"
    if target.exists():
        if not (resume or force):
            raise FileExistsError(f"Report already exists: {target}; use --resume or --force.")
        if resume and not force and receipt.is_file():
            old = json.loads(receipt.read_text(encoding="utf-8"))
            if old.get("source_state") == state and old.get("settings") == settings and old.get("method_version") == _VERSION:
                expected = old.get("artifacts", [])
                if expected and all((target / path).is_file() for path in expected):
                    return {**old, "operation_status": "reused", "outputs": {"index": str(target / "index.html"), "summary": str(receipt)}}
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".analysis-report-", dir=target.parent))
    tables: dict[str, _Table] = {}
    frames = _read_rows(summary_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        native_metadata = json.loads(str(archive["__metadata__"].item())) if "__metadata__" in archive.files else {}
    missing_indices = [int(index) for index in native_metadata.get("missing_frames", [])]
    expected_count = max([len(frames), *[index + 1 for index in missing_indices]])
    for position in range(len(frames), expected_count):
        frames.append({"frame_index": position, "frame_id": "", "status": "not_processed",
                       "error": "This frame position has no processed record in the native export; resume the batch to measure it."})
    parameters = _plot_parameter_rows(folder / f"{prefix}parameters_long.csv")
    time_units = _manifest_time_units(folder, prefix)
    base_columns = ("frame_index", "frame_id", "time", "time_unit", "status", "q_unit")
    profile_columns = ("mean", "sum", "std", "sem", "count", "candidate_count", "coverage", "sem_kind")
    try:
        for name, columns in {
            "radial_profiles": (*base_columns, "bin_index", "q_min", "q_max", "q_center", *profile_columns),
            "angular_profiles": (*base_columns, "bin_index", "angle_min_deg", "angle_max_deg", "angle_center_deg", *profile_columns),
            "stored_profile_samples": (*base_columns, "profile", "sample_index", "value", "array_key"),
            "array_catalog": (*base_columns, "role", "array_key", "shape", "dtype"),
            "fit_diagnostics_long": (*base_columns, "field", "value", "value_type"),
            "ellipse_candidates": (*base_columns, "candidate_index", "selected_start_index", "field", "value", "value_type"),
            "normal_profiles": (*base_columns, "profile_index", "point_id", "sample_index", "offset_q", "raw_intensity", "fit_intensity", "residual",
                                "valid", "model", "reason", "snr", "normal_fwhm_q", "localization_sigma_q", "support_fraction", "uncertainty_source"),
        }.items():
            tables[name] = _Table(stage / f"{name}.csv", columns)
        records = _detail_records(folder, prefix)
        radial: list[Any] = []
        angular: list[Any] = []
        summaries = []
        frame_reports = []
        # Units can change between input sources. Each group receives its own
        # q-bin grid and sequence plots, with no conversion inferred here.
        bounds: dict[str, list[float]] = {}
        with np.load(archive_path, allow_pickle=False) as archive:
            groups = _array_groups(archive.files)
            unit_records = _detail_records(folder, prefix)
            for position, source in enumerate(frames):
                record = next(unit_records, {})
                unit = _q_unit(record)
                keys = groups.get(position, {})
                try:
                    qx, qy, mask = _coordinates(archive, keys)
                    q = np.hypot(qx, qy)
                    usable = np.isfinite(q) if mask is None else np.isfinite(q) & mask
                    if np.any(usable):
                        low, high = float(q[usable].min()), float(q[usable].max())
                        previous = bounds.get(unit, [low, high])
                        bounds[unit] = [min(low, previous[0]), max(high, previous[1])]
                except ValueError:
                    continue
            grids = {unit: np.linspace(low, high if high > low else low + max(abs(low), 1.) * 1e-6, settings["radial_bins"] + 1)
                     for unit, (low, high) in bounds.items()}
            for position, source in enumerate(frames):
                if progress:
                    progress(f"Reporting {folder.name}: frame {position + 1}/{len(frames)}")
                record = next(records, {})
                if record and int(record.get("frame_index", position)) != position:
                    raise ValueError("Stored frame records do not match the batch CSV order; regenerate its export.")
                unit = _q_unit(record)
                metadata = record.get("frame_metadata", {})
                time_unit = source.get("time_unit") or (metadata.get("time_unit") if isinstance(metadata, Mapping) else None)
                if not time_unit and position < len(time_units):
                    time_unit = time_units[position]
                base = {"frame_index": position, "frame_id": source.get("frame_id", ""), "time": _number(source.get("time")),
                        "time_unit": str(time_unit or ""), "status": source.get("status", "unknown"), "q_unit": unit}
                frame_report = {**base, "quality_status": source.get("quality_status"), "diagnostic": source.get("diagnostic"), "error": source.get("error"), "figures": {}}
                tables["fit_diagnostics_long"].write(_diagnostic_rows(base, record))
                tables["ellipse_candidates"].write(_candidate_rows(base, record))
                keys = groups.get(position, {})
                tables["stored_profile_samples"].write(_profile_rows(base, archive, keys, tables["array_catalog"]))
                frame_folder = stage / "frames" / f"frame_{position:04d}"
                frame_folder.mkdir(parents=True)
                _write_json(frame_folder / "fit_details.json", record)
                frame_report["data"] = {"fit_details": (frame_folder / "fit_details.json").relative_to(stage).as_posix()}
                try:
                    image = _get_array(archive, keys, "image", "data")
                    if image is None:
                        raise ValueError("Stored detector image is missing for this frame; other frames remain available.")
                    qx, qy, mask = _coordinates(archive, keys)
                    measurement = measure_frame(image, qx, qy, valid_mask=mask, q_unit=unit,
                                                q_edges=grids.get(unit), radial_bins=settings["radial_bins"], angular_bins=settings["angular_bins"])
                    plot_result = _restore_profile_arrays(_result_detail(record), archive, keys)
                    plot_record = {**record, "result": plot_result}
                    tables["normal_profiles"].write(_normal_profile_rows(base, plot_result))
                    ridges = plot_result.get("ridges", [])
                    observed_points = [point for point in ridges if isinstance(point, Mapping)
                                       and point.get("valid", True) and point.get("accepted", True)
                                       and _number(point.get("qx")) is not None and _number(point.get("qy")) is not None]
                    if observed_points:
                        measurement["measured_ridge_qx"] = np.asarray([point["qx"] for point in observed_points])
                        measurement["measured_ridge_qy"] = np.asarray([point["qy"] for point in observed_points])
                    summary = {**base, **measurement["summary"]}
                    measurement["summary"] = summary
                    summaries.append(summary)
                    tables["radial_profiles"].write([{**base, **row} for row in measurement["radial_rows"]])
                    tables["angular_profiles"].write([{**base, **row} for row in measurement["angular_rows"]])
                    radial.append(_compact_profile(measurement["radial_rows"], radial=True))
                    angular.append(_compact_profile(measurement["angular_rows"], radial=False))
                    np.savez_compressed(frame_folder / "polar_measurements.npz", **measurement["polar"])
                    _write_json(frame_folder / "measurements.json", summary)
                    frame_report["data"].update({"measurements": (frame_folder / "measurements.json").relative_to(stage).as_posix(),
                                                  "polar_arrays": (frame_folder / "polar_measurements.npz").relative_to(stage).as_posix()})
                    model = _get_array(archive, keys, "full2d__model_image", "full2d__model")
                    residual = _get_array(archive, keys, "full2d__residual_image", "full2d__residual")
                    figure_paths = render_frame_report(frame_folder, image=image, qx=qx, qy=qy, valid_mask=mask, q_unit=unit,
                                                       measurement=measurement, fit_record=plot_record, model=model, residual=residual,
                                                       title=f"Frame {position}: {base['frame_id']}", formats=settings["formats"], dpi=settings["dpi"])
                    frame_report["figures"] = {key: path.relative_to(stage).as_posix() for key, path in figure_paths.items()}
                    frame_report["report_status"] = "completed"
                except (ValueError, TypeError, FloatingPointError) as exc:
                    if len(summaries) == position:
                        summaries.append(base)
                        radial.append([])
                        angular.append([])
                    frame_report["report_status"] = "incomplete"
                    frame_report["error"] = str(exc)
                frame_reports.append(frame_report)
        for table in tables.values():
            table.close()
        sequence_folder = stage / "sequence"
        sequence_folder.mkdir()
        figures = render_sequence_report(sequence_folder, summaries=summaries, radial_profiles=radial,
                                         angular_profiles=angular, parameter_rows=parameters, formats=settings["formats"], dpi=settings["dpi"])
        summary_columns = list(dict.fromkeys(key for row in summaries for key in row)) or list(base_columns)
        with _Table(stage / "frame_measurements.csv", summary_columns) as table:
            table.write(summaries)
        trends = _parameter_trends(parameters)
        with _Table(stage / "parameter_changes.csv", list(trends[0]) if trends else ("parameter", "unit", "n_records", "n_finite")) as table:
            table.write(trends)
        report = {"schema_version": SCHEMA, "method_version": _VERSION, "operation_status": "completed", "settings": dict(settings),
                  "source_state": state, "frames": frame_reports,
                  "native_array_status": {key: value for key, value in native_metadata.items() if key != "arrays"},
                  "counts": {"frames": len(frames), "reported": sum(row["report_status"] == "completed" for row in frame_reports),
                             "incomplete": sum(row["report_status"] != "completed" for row in frame_reports),
                             "warning_frames": sum(row.get("status") not in {"ok", "success", "recovered"} or str(row.get("quality_status", "")).upper() in {"WARN", "WARNING", "FAIL", "FAILED"} for row in frame_reports)},
                  "sources": {name: f"../../{prefix}{name}" for name in state},
                  "tables": {name: table.path.name for name, table in tables.items()} | {"frame_measurements": "frame_measurements.csv", "parameter_changes": "parameter_changes.csv"},
                  "sequence_figures": {key: path.relative_to(stage).as_posix() for key, path in figures.items()}}
        report["exit_code"] = 1 if report["counts"]["incomplete"] or report["counts"]["warning_frames"] or not frames else 0
        report["status"] = "ready_with_warnings" if report["exit_code"] else "ready"
        (stage / "index.html").write_text(_render_index(report), encoding="utf-8")
        _write_json(stage / "README.json", {"profile_sem": "Within-bin sample standard deviation / sqrt(pixel count); detector correlations and calibration uncertainty are not included.",
                                           "mask": "Stored analysis usable-pixel mask; q-window, excluded pixels and ROI are retained.",
                                           "moments": "Descriptive nonnegative-intensity-weighted in-plane moments, conditional on mask and q window.",
                                           "native_arrays": "array_catalog.csv maps frame identity to unchanged results.npz keys.",
                                           "fit_details": "Existing fit records and candidates; reporting does not refit or run full2d."})
        report["artifacts"] = sorted(path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()) + ["report_summary.json"]
        _write_json(stage / "report_summary.json", report)
        if _source_state(folder, prefix) != state:
            raise ValueError("Batch outputs changed during reporting; finish the export then retry report --resume.")
        backup = target.with_name(f".{target.name}-previous")
        if backup.exists():
            raise FileExistsError(f"Previous report backup exists: {backup}; recover it before replacing the report.")
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(stage, target)
        except OSError:
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return {**report, "outputs": {"index": str(target / "index.html"), "summary": str(target / "report_summary.json")}}
    finally:
        for table in tables.values():
            table.close()
        if stage.exists():
            shutil.rmtree(stage)


def build_analysis_report(output_dir: str | os.PathLike[str], *, prefix: str = "", force: bool = False, resume: bool = False,
                          radial_bins: int = 128, angular_bins: int = 72, formats: Sequence[str] = ("png", "svg", "pdf"),
                          dpi: int = 180, progress: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Report one batch or all sample batches under a parent output directory."""
    for name, value in (("radial_bins", radial_bins), ("angular_bins", angular_bins)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    if radial_bins * angular_bins > 1_000_000:
        raise ValueError("radial_bins * angular_bins must be <= 1000000; choose fewer report bins.")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    formats = list(dict.fromkeys(str(value).lower() for value in formats))
    if not formats or any(value not in {"png", "svg", "pdf", "tiff"} for value in formats):
        raise ValueError("formats must contain png, svg, pdf or tiff")
    if prefix and (Path(prefix).name != prefix or "/" in prefix or "\\" in prefix):
        raise ValueError("prefix must be a file-name prefix without directory separators")
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Batch output directory does not exist: {root}")
    summaries = sorted(path for path in root.rglob(f"{prefix}*frame_summary.csv")
                       if not any(part.startswith(".") or part == "figures" for part in path.relative_to(root).parts))
    if not summaries:
        raise FileNotFoundError(f"No frame_summary.csv batch exports found under {root}")
    settings = {"radial_bins": radial_bins, "angular_bins": angular_bins, "formats": formats, "dpi": dpi}
    reports = [_build_sample(path, force=force, resume=resume, settings=settings, progress=progress) for path in summaries]
    if len(reports) == 1:
        return reports[0]
    collection = root / "analysis_reports.html"
    collection_tables = {}
    for table_name, source_name in (("collection_parameters", "parameters_long.csv"), ("collection_measurements", "frame_measurements.csv")):
        sources = []
        columns = ["sample", "export_prefix"]
        for path, report in zip(summaries, reports, strict=True):
            sample = str(path.parent.relative_to(root))
            sample_prefix = path.name.removesuffix("frame_summary.csv")
            source = path.parent / f"{sample_prefix}{source_name}" if table_name == "collection_parameters" else Path(report["outputs"]["summary"]).parent / source_name
            if source.is_file():
                sources.append((sample, sample_prefix, source))
                with source.open(encoding="utf-8-sig", newline="") as handle:
                    columns.extend(name for name in csv.DictReader(handle).fieldnames or () if name not in columns)
        destination = root / f"{table_name}.csv"
        # Reuse byte-identical tables without changing their timestamps.
        with tempfile.TemporaryDirectory(prefix=".collection-report-", dir=root) as temporary:
            candidate = Path(temporary) / destination.name
            with _Table(candidate, columns) as table:
                for sample, sample_prefix, source in sources:
                    with source.open(encoding="utf-8-sig", newline="") as handle:
                        table.write({"sample": sample, "export_prefix": sample_prefix, **row} for row in iter_csv_rows(handle))
            if not _same_contents(candidate, destination):
                os.replace(candidate, destination)
        collection_tables[table_name] = destination.name
    body = "".join(f'<section><h2>{html.escape(str(path.parent.relative_to(root)))}</h2><p>'
                   f'{_link(os.path.relpath(report["outputs"]["index"], root).replace(os.sep, "/"), "Open data and figures")}</p>'
                   f'<p>{report["counts"]["reported"]}/{report["counts"]["frames"]} frames reported</p></section>'
                   for path, report in zip(summaries, reports, strict=True))
    table_links = " · ".join(_link(path, label.replace("_", " ")) for label, path in collection_tables.items())
    document = _page("WingSAXS sample reports", f'<section><h2>All samples</h2><p>{table_links}</p></section>' + body)
    if not collection.exists() or collection.read_text(encoding="utf-8") != document:
        collection.write_text(document, encoding="utf-8")
    combined = {"schema_version": SCHEMA, "status": "ready_with_warnings" if any(report["exit_code"] for report in reports) else "ready",
            "operation_status": "completed", "exit_code": int(any(report["exit_code"] for report in reports)),
            "counts": {"samples": len(reports), **{key: sum(report["counts"][key] for report in reports) for key in ("frames", "reported", "incomplete", "warning_frames")}},
            "samples": [{"sample": str(path.parent.relative_to(root)), "counts": report["counts"], "outputs": report["outputs"]} for path, report in zip(summaries, reports, strict=True)],
            "outputs": {"index": str(collection), "summary": str(root / "analysis_report_summary.json"),
                        **{key: str(root / value) for key, value in collection_tables.items()}}}
    summary = root / "analysis_report_summary.json"
    text = json.dumps(combined, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if not summary.exists() or summary.read_text(encoding="utf-8") != text:
        summary.write_text(text, encoding="utf-8")
    return combined


__all__ = ["build_analysis_report"]
