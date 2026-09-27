"""Compare independent and directional warm starts on a synthetic SAXS series.

This benchmark isolates optimization initialization and frame order.  Its
images come from the finite real-space stack generator in
``benchmark_sequence``; they are not experimental detector data and do not
establish measurement accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from . import batch, benchmark_arcs, benchmark_sequence
from .pipeline import analyze_frame


METRICS = ("q_star_from_arcs", "a", "b", "axis_ratio", "theta_deg", "rmse")
BASE_OUTPUTS = (
    "campaign_manifest.json",
    "summary.json",
    "per_frame_comparison.csv",
    "warm_start_calls.csv",
    "report.md",
    "inputs/geometry.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _segment(
    *,
    shape: tuple[int, int],
    seed: int,
    spacing_start_nm: float,
    spacing_end_nm: float,
    tilt_start_deg: float,
    tilt_end_deg: float,
    rotation_start_deg: float,
    rotation_end_deg: float,
    include_noise_control: bool = False,
) -> tuple[dict[str, Any], ...]:
    settings = benchmark_sequence.ObliqueStackSettings(
        n_signal_frames=3,
        shape=shape,
        pixel_size_nm=1.0,
        spacing_start_nm=spacing_start_nm,
        spacing_end_nm=spacing_end_nm,
        tilt_start_deg=tilt_start_deg,
        tilt_end_deg=tilt_end_deg,
        stack_rotation_start_deg=rotation_start_deg,
        stack_rotation_end_deg=rotation_end_deg,
        target_first_order_snr=15.0,
        seed=seed,
        local_missing_lobe_frame=1,
        include_clean_control=False,
        include_noise_control=include_noise_control,
    )
    return benchmark_sequence.generate_oblique_stack_sequence(settings)


def build_initialization_sequence(
    *, shape: tuple[int, int] = (192, 192), seed: int = 20260927
) -> tuple[dict[str, Any], ...]:
    """Build six signal frames plus an all-masked failure and a noise control."""

    if len(shape) != 2 or min(shape) < 64:
        raise ValueError("shape must contain two dimensions, each at least 64")
    before_jump = _segment(
        shape=shape,
        seed=seed,
        spacing_start_nm=11.4,
        spacing_end_nm=11.9,
        tilt_start_deg=20.0,
        tilt_end_deg=22.0,
        rotation_start_deg=22.0,
        rotation_end_deg=20.0,
    )
    after_jump = _segment(
        shape=shape,
        seed=seed + 100,
        spacing_start_nm=15.5,
        spacing_end_nm=15.9,
        tilt_start_deg=20.0,
        tilt_end_deg=22.0,
        rotation_start_deg=22.0,
        rotation_end_deg=20.0,
        include_noise_control=True,
    )

    identified: list[dict[str, Any]] = []
    for index, source in enumerate(before_jump[:3]):
        frame = dict(source)
        frame.update(
            frame_index=index,
            frame_id=f"pre_{index:02d}",
            sequence_role="signal_before_jump",
        )
        identified.append(frame)

    jump = dict(after_jump[0])
    jump.update(
        frame_index=3,
        frame_id="jump_03",
        sequence_role="abrupt_q_jump",
    )
    identified.append(jump)

    # This frame has finite pixel values but no valid detector support.  It
    # exercises the production batch failure path rather than a fake result.
    failure = dict(jump)
    failure.update(
        frame_index=4,
        frame_id="all_masked_failure",
        sequence_role="all_pixels_masked_failure",
        intensity_noisy=np.zeros(shape, dtype=float),
        intensity=np.zeros(shape, dtype=float),
        noise=np.zeros(shape, dtype=float),
        mask=np.ones(shape, dtype=bool),
        valid_mask=np.zeros(shape, dtype=bool),
        detector_beamstop_mask=np.ones(shape, dtype=bool),
        missing_lobe_mask=np.zeros(shape, dtype=bool),
        structural_q0_nm_inv=None,
        structure_truth=None,
        mask_diagnostics={
            "beamstop_pixels": int(np.prod(shape)),
            "missing_lobe_pixels": 0,
            "masked_pixels": int(np.prod(shape)),
        },
    )
    identified.append(failure)

    for index, source in enumerate(after_jump[1:3], start=5):
        frame = dict(source)
        frame.update(
            frame_index=index,
            frame_id=f"post_{index:02d}",
            sequence_role="signal_after_jump",
        )
        identified.append(frame)

    noise_control = dict(after_jump[-1])
    noise_control.update(
        frame_index=7,
        frame_id="noise_only",
        sequence_role="noise_only_control",
    )
    identified.append(noise_control)

    return tuple(identified)


def build_arc_control_sequence(
    *, shape: tuple[int, int] = (64, 64), seed: int = 20260927
) -> tuple[dict[str, Any], ...]:
    """Build known-arc images with a ratio jump and failure between signal frames."""

    if len(shape) != 2 or min(shape) < 32:
        raise ValueError("shape must contain two dimensions, each at least 32")
    cases = []
    for index, ratio in enumerate((0.40, 0.38, 0.20, 0.22)):
        case = benchmark_arcs.generate_arc_case(
            {"case_id": "ellipse_ratio_400", "axis_ratio": ratio},
            seed=seed + index,
            shape=shape,
        )
        cases.append(case)

    identified: list[dict[str, Any]] = []
    for index, case in enumerate(cases[:3]):
        identified.append(
            {
                "frame_index": index,
                "frame_id": f"arc_{index:02d}",
                "sequence_role": "known_arc_signal_before_jump" if index < 2 else "known_arc_ratio_jump",
                "intensity_noisy": np.asarray(case["image"], dtype=float),
                "mask": np.asarray(case["mask"], dtype=bool),
                "qmap": case["qmap"],
                "truth": case["truth"],
            }
        )

    failure_source = cases[2]
    identified.append(
        {
            "frame_index": 3,
            "frame_id": "arc_all_masked_failure",
            "sequence_role": "all_pixels_masked_failure",
            "intensity_noisy": np.zeros(shape, dtype=float),
            "mask": np.ones(shape, dtype=bool),
            "qmap": failure_source["qmap"],
            "truth": None,
        }
    )
    final_case = cases[3]
    identified.append(
        {
            "frame_index": 4,
            "frame_id": "arc_04",
            "sequence_role": "known_arc_signal_after_jump",
            "intensity_noisy": np.asarray(final_case["image"], dtype=float),
            "mask": np.asarray(final_case["mask"], dtype=bool),
            "qmap": final_case["qmap"],
            "truth": final_case["truth"],
        }
    )
    noise_scale = 0.05 * float(np.max(final_case["image"]))
    rng = np.random.default_rng(seed + 50_000)
    noise = np.clip(rng.normal(0.0, noise_scale, size=shape), 0.0, None)
    identified.append(
        {
            "frame_index": 5,
            "frame_id": "arc_noise_only",
            "sequence_role": "noise_only_control",
            "intensity_noisy": noise,
            "mask": np.zeros(shape, dtype=bool),
            "qmap": final_case["qmap"],
            "truth": None,
            "noise_sigma": noise_scale,
        }
    )
    return tuple(identified)


def _metric_record(result: Any) -> dict[str, Any]:
    butterfly = getattr(result, "butterfly", None)
    butterfly = butterfly if isinstance(butterfly, Mapping) else {}
    fit = butterfly.get("candidate_fit", {})
    fit = fit if isinstance(fit, Mapping) else {}
    candidates = fit.get("candidate_solutions", ())
    if not candidates:
        ellipse_fit = getattr(result, "ellipse_fit", None)
        ellipse_fit = ellipse_fit if isinstance(ellipse_fit, Mapping) else {}
        candidates = ellipse_fit.get("candidate_solutions", ())
    effective_starts = [
        _json_safe(candidate["start_values"])
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and isinstance(candidate.get("start_values"), Mapping)
    ] if isinstance(candidates, (list, tuple)) else []
    points = getattr(result, "ridges", None) or []
    accepted = [
        point
        for point in points
        if isinstance(point, Mapping)
        and point.get("accepted") is True
        and point.get("valid", True) is not False
    ]
    radii = [
        float(np.hypot(point["qx"], point["qy"]))
        for point in accepted
        if _finite(point.get("qx")) is not None and _finite(point.get("qy")) is not None
    ]
    quality = butterfly.get("quality", {})
    return {
        "measurement_status": butterfly.get("measurement_status"),
        "quality_status": quality.get("status") if isinstance(quality, Mapping) else None,
        "quality_flags": quality.get("flags", []) if isinstance(quality, Mapping) else [],
        "warm_start_eligible": butterfly.get("warm_start_eligible"),
        "accepted_ridge_point_count": len(accepted),
        "accepted_ridge_q_median_nm_inv": _finite(np.median(radii)) if radii else None,
        "effective_optimizer_starts": effective_starts or None,
        **{name: _finite(fit.get(name)) for name in METRICS},
    }


def _normalise_metric_difference(name: str, left: float, right: float) -> float:
    difference = abs(left - right)
    if name == "theta_deg":
        difference = abs((left - right + 90.0) % 180.0 - 90.0)
    return float(difference)


def _difference_summary(
    left_rows: Mapping[str, Mapping[str, Any]],
    right_rows: Mapping[str, Mapping[str, Any]],
    frame_roles: Mapping[str, str],
    *,
    role_filter: set[str] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in METRICS:
        differences: list[float] = []
        relative: list[float] = []
        used: list[str] = []
        for frame_id, role in frame_roles.items():
            if role_filter is not None and role not in role_filter:
                continue
            left = _finite(left_rows.get(frame_id, {}).get(name))
            right = _finite(right_rows.get(frame_id, {}).get(name))
            if left is None or right is None:
                continue
            delta = _normalise_metric_difference(name, left, right)
            differences.append(delta)
            if name != "theta_deg":
                relative.append(100.0 * delta / max(abs(left), abs(right), 1e-12))
            used.append(frame_id)
        summary[name] = {
            "n_paired": len(differences),
            "frame_ids": used,
            "median_absolute_difference": float(np.median(differences)) if differences else None,
            "maximum_absolute_difference": max(differences) if differences else None,
            "maximum_relative_difference_percent": max(relative) if relative else None,
        }
    return summary


def _seed_usage(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeded = [row for row in calls if int(row.get("initial_parameter_count", 0)) > 0]
    return {
        "calls": len(calls),
        "calls_with_initial_parameters": len(seeded),
        "seeded_frame_ids": [str(row["frame_id"]) for row in seeded],
        "status": "exercised" if seeded else "not_exercised_no_eligible_seed",
    }


def _component_hashes() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[2]
    paths = (
        source_root / "src" / "butterfly_saxs" / "benchmark_initialization.py",
        source_root / "src" / "butterfly_saxs" / "benchmark_arcs.py",
        source_root / "src" / "butterfly_saxs" / "benchmark_sequence.py",
        source_root / "src" / "butterfly_saxs" / "batch.py",
        source_root / "src" / "butterfly_saxs" / "pipeline.py",
        source_root / "scripts" / "benchmark_initialization.py",
    )
    return {
        path.relative_to(source_root).as_posix(): _sha256(path)
        for path in paths
    }


def _run_order(
    frames: Sequence[Mapping[str, Any]],
    input_paths: Mapping[str, Path],
    mask_paths: Mapping[str, Path],
    qmap: Mapping[str, Any],
    analysis_config: Mapping[str, Any],
    *,
    mode: str,
    direction: str | None = None,
    campaign: str = "stack_fft",
    frame_order: Sequence[int],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame_ids = [str(frame["frame_id"]) for frame in frames]
    refs_manifest = [
        {
            "path": str(input_paths[frame_ids[index]]),
            "frame_id": frame_ids[index],
            "time": index,
            "order": order_index,
            "metadata": {"campaign_frame_id": frame_ids[index]},
        }
        for order_index, index in enumerate(frame_order)
    ]
    masks = {
        frame_id: np.load(mask_paths[frame_id], allow_pickle=False)
        for frame_id in frame_ids
    }
    call_rows: list[dict[str, Any]] = []

    def analyze(ref: batch.FrameRef, initial_parameters: Any = None) -> Any:
        frame_id = str(ref.frame_id)
        initial = _json_safe(initial_parameters)
        call_rows.append(
            {
                "call_index": len(call_rows),
                "mode": mode,
                "direction": direction or mode,
                "campaign": campaign,
                "frame_id": frame_id,
                "initial_parameters": initial,
                "initial_parameter_count": (
                    len(initial) if isinstance(initial, Mapping) else 0
                ),
            }
        )
        image = np.load(ref.path, allow_pickle=False)
        return analyze_frame(
            image,
            qmap=qmap,
            mask=masks[frame_id],
            config=analysis_config,
            full2d=False,
            initial_parameters=initial_parameters,
        )

    batch_config = {
        "analysis": dict(analysis_config.get("analysis", {})),
        "qmap": dict(qmap),
        "mask_sha256_by_frame": {
            frame_id: _sha256(mask_paths[frame_id]) for frame_id in frame_ids
        },
    }
    result = batch.run_batch(
        (),
        analyze,
        mode=mode,  # type: ignore[arg-type]
        config=batch_config,
        manifest=refs_manifest,
        allow_mixed_series=True,
    )
    key_to_id = {batch.FrameRef(path, frame_id=frame_id).key: frame_id
                 for frame_id, path in input_paths.items()}
    rows: dict[str, dict[str, Any]] = {}
    for item in result.frame_results:
        frame_id = str(item.frame.frame_id)
        row: dict[str, Any] = {
            "frame_id": frame_id,
            "status": item.status,
            "error": item.error,
            "diagnostic": item.diagnostic,
            "elapsed_s": _finite(item.elapsed_s),
            "warm_start_from": key_to_id.get(item.warm_start_from)
            if item.warm_start_from is not None
            else None,
        }
        if item.result is not None:
            row.update(_metric_record(item.result))
        rows[frame_id] = row
    run_record = {
        "mode": mode,
        "input_hash": result.input_hash,
        "config_hash": result.config_hash,
        "elapsed_s": _finite(result.elapsed_s),
        "processed_count": result.processed_count,
        "total_count": result.total_count,
        "cancelled": result.cancelled,
        "processing_order": [frame_ids[index] for index in frame_order],
        "status_counts": {
            status: sum(row["status"] == status for row in rows.values())
            for status in ("ok", "warning", "failed", "skipped")
        },
    }
    return rows, call_rows, run_record


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def run_initialization_campaign(
    output: str | Path,
    *,
    shape: tuple[int, int] = (192, 192),
    seed: int = 20260927,
    force: bool = False,
) -> dict[str, Any]:
    """Run independent, forward warm-start, and reverse warm-start analyses."""

    destination = Path(output).expanduser().resolve()
    frame_ids_expected = [
        "pre_00", "pre_01", "pre_02", "jump_03", "all_masked_failure",
        "post_05", "post_06", "noise_only",
    ]
    expected = [destination / name for name in BASE_OUTPUTS]
    expected.extend(destination / "inputs" / f"{frame_id}{suffix}"
                    for frame_id in frame_ids_expected for suffix in (".npy", "_mask.npy"))
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "campaign outputs already exist; choose a new directory or pass --force"
        )
    destination.mkdir(parents=True, exist_ok=True)
    input_dir = destination / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    frames = build_initialization_sequence(shape=shape, seed=seed)
    frame_ids = [str(frame["frame_id"]) for frame in frames]
    if frame_ids != frame_ids_expected:
        raise AssertionError(f"unexpected campaign frame order: {frame_ids!r}")

    input_paths: dict[str, Path] = {}
    mask_paths: dict[str, Path] = {}
    frame_manifest: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = str(frame["frame_id"])
        image_path = input_dir / f"{frame_id}.npy"
        mask_path = input_dir / f"{frame_id}_mask.npy"
        np.save(image_path, np.asarray(frame["intensity_noisy"], dtype=np.float64), allow_pickle=False)
        np.save(mask_path, np.asarray(frame["mask"], dtype=bool), allow_pickle=False)
        input_paths[frame_id] = image_path
        mask_paths[frame_id] = mask_path
        truth = frame.get("structure_truth")
        frame_manifest.append(
            {
                "frame_index": int(frame["frame_index"]),
                "frame_id": frame_id,
                "sequence_role": frame["sequence_role"],
                "input_path": image_path.relative_to(destination).as_posix(),
                "input_sha256": _sha256(image_path),
                "mask_path": mask_path.relative_to(destination).as_posix(),
                "mask_sha256": _sha256(mask_path),
                "q0_nm_inv": _finite(frame.get("structural_q0_nm_inv")),
                "layer_spacing_nm": _finite(
                    truth.get("layer_spacing_nm") if isinstance(truth, Mapping) else None
                ),
                "noise_sigma": _finite(frame.get("noise_sigma")),
                "missing_lobe_pixels": int(
                    frame.get("mask_diagnostics", {}).get("missing_lobe_pixels", 0)
                ),
            }
        )

    reference = frames[0]
    qmap = {
        "qx": np.asarray(reference["qx"], dtype=float),
        "qy": np.asarray(reference["qy"], dtype=float),
        "q": np.asarray(reference["q"], dtype=float),
        "q_unit": str(reference["q_unit"]),
    }
    geometry_path = input_dir / "geometry.npz"
    np.savez_compressed(geometry_path, **{key: value for key, value in qmap.items() if key != "q_unit"})

    analysis_config = benchmark_sequence.pipeline_analysis_config()
    analysis_config["analysis"]["ellipse"]["multistart"] = 3
    analysis_config["analysis"]["max_nfev"] = 400
    settings_record = {
        "generator": "finite Gaussian slab stacks, incoherent FFT power sum",
        "shape": list(shape),
        "pixel_size_nm": 1.0,
        "q_unit": qmap["q_unit"],
        "seed": seed,
        "target_first_order_snr": 15.0,
        "pre_jump_spacing_nm": [11.4, 11.9],
        "post_jump_spacing_nm": [15.5, 15.9],
        "jump_ratio_q0_post_to_pre": (
            (2.0 * np.pi / 15.5) / (2.0 * np.pi / 11.9)
        ),
        "analysis_config": analysis_config,
        "note": "Synthetic initialization/order benchmark; no experimental SAXS data.",
    }
    manifest = {
        "schema_version": 1,
        "campaign": "JAC initialization and sequence-order comparison",
        "settings": settings_record,
        "frames": frame_manifest,
        "geometry_file": {
            "path": geometry_path.relative_to(destination).as_posix(),
            "sha256": _sha256(geometry_path),
            "q_unit": qmap["q_unit"],
        },
        "source_sha256_before": _component_hashes(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    file_hashes_before = {
        path.relative_to(destination).as_posix(): _sha256(path)
        for path in [*input_paths.values(), *mask_paths.values(), geometry_path]
    }

    natural_order = tuple(range(len(frames)))
    reverse_order = tuple(reversed(natural_order))
    independent, independent_calls, independent_run = _run_order(
        frames, input_paths, mask_paths, qmap, analysis_config,
        mode="independent", direction="independent", frame_order=natural_order,
    )
    forward, forward_calls, forward_run = _run_order(
        frames, input_paths, mask_paths, qmap, analysis_config,
        mode="warm_start", direction="forward", frame_order=natural_order,
    )
    reverse, reverse_calls, reverse_run = _run_order(
        frames, input_paths, mask_paths, qmap, analysis_config,
        mode="warm_start", direction="reverse", frame_order=reverse_order,
    )


    # Recompute every saved input and source identity after all three analyses.
    # The arrays on disk are the shared evidence, so any unexpected mutation
    # makes this campaign unsuitable as reproducible evidence.
    file_hashes_after = {
        path.relative_to(destination).as_posix(): _sha256(path)
        for path in [*input_paths.values(), *mask_paths.values(), geometry_path]
    }
    source_hashes_after = _component_hashes()
    file_integrity = {
        path: {
            "sha256_before": file_hashes_before[path],
            "sha256_after": file_hashes_after[path],
            "unchanged": file_hashes_before[path] == file_hashes_after[path],
        }
        for path in file_hashes_before
    }
    source_integrity = {
        path: {
            "sha256_before": manifest["source_sha256_before"][path],
            "sha256_after": source_hashes_after[path],
            "unchanged": manifest["source_sha256_before"][path] == source_hashes_after[path],
        }
        for path in manifest["source_sha256_before"]
    }
    integrity = {
        "saved_inputs_unchanged": all(item["unchanged"] for item in file_integrity.values()),
        "analysis_sources_unchanged": all(item["unchanged"] for item in source_integrity.values()),
        "files": file_integrity,
        "sources": source_integrity,
    }

    by_id = {str(row["frame_id"]): row for row in frame_manifest}
    frame_roles = {frame_id: str(by_id[frame_id]["sequence_role"]) for frame_id in frame_ids}
    signal_roles = {"signal_before_jump", "abrupt_q_jump", "signal_after_jump"}
    comparisons: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = str(frame["frame_id"])
        record: dict[str, Any] = {
            "frame_index": int(frame["frame_index"]),
            "frame_id": frame_id,
            "sequence_role": frame["sequence_role"],
            "q0_nm_inv": _finite(frame.get("structural_q0_nm_inv")),
            "independent_status": independent[frame_id]["status"],
            "forward_status": forward[frame_id]["status"],
            "reverse_status": reverse[frame_id]["status"],
            "forward_warm_start_from": forward[frame_id]["warm_start_from"],
            "reverse_warm_start_from": reverse[frame_id]["warm_start_from"],
            "independent_error": independent[frame_id]["error"],
            "forward_error": forward[frame_id]["error"],
            "reverse_error": reverse[frame_id]["error"],
            "independent_effective_optimizer_starts": independent[frame_id].get(
                "effective_optimizer_starts"
            ),
            "forward_effective_optimizer_starts": forward[frame_id].get(
                "effective_optimizer_starts"
            ),
            "reverse_effective_optimizer_starts": reverse[frame_id].get(
                "effective_optimizer_starts"
            ),
        }
        for name in METRICS:
            base = _finite(independent[frame_id].get(name))
            forward_value = _finite(forward[frame_id].get(name))
            reverse_value = _finite(reverse[frame_id].get(name))
            record[f"independent_{name}"] = base
            record[f"forward_{name}"] = forward_value
            record[f"reverse_{name}"] = reverse_value
            record[f"forward_abs_delta_{name}"] = (
                _normalise_metric_difference(name, base, forward_value)
                if base is not None and forward_value is not None else None
            )
            record[f"reverse_abs_delta_{name}"] = (
                _normalise_metric_difference(name, base, reverse_value)
                if base is not None and reverse_value is not None else None
            )
        for mode, row in (("independent", independent[frame_id]),
                          ("forward", forward[frame_id]), ("reverse", reverse[frame_id])):
            record[f"{mode}_accepted_ridge_points"] = row.get("accepted_ridge_point_count")
            record[f"{mode}_observed_ridge_q_median_nm_inv"] = row.get(
                "accepted_ridge_q_median_nm_inv"
            )
            record[f"{mode}_effective_optimizer_starts"] = row.get(
                "effective_optimizer_starts"
            )
        comparisons.append(record)

    frame_role_filter = {frame_id: frame_roles[frame_id] for frame_id in frame_ids}
    signal_filter = {frame_id: frame_roles[frame_id] for frame_id in frame_ids
                     if frame_roles[frame_id] in signal_roles}
    summary = {
        "campaign": "JAC initialization and sequence-order comparison",
        "synthetic_only": True,
        "frame_count": len(frames),
        "signal_frame_count": sum(role in signal_roles for role in frame_roles.values()),
        "runs": {
            "independent": independent_run,
            "forward_warm_start": forward_run,
            "reverse_warm_start": reverse_run,
        },
        "integrity": integrity,
        "status_by_role": {
            frame_id: {
                "role": frame_roles[frame_id],
                "independent": independent[frame_id]["status"],
                "forward": forward[frame_id]["status"],
                "reverse": reverse[frame_id]["status"],
                "independent_effective_optimizer_starts": independent[frame_id].get(
                    "effective_optimizer_starts"
                ),
                "forward_effective_optimizer_starts": forward[frame_id].get(
                    "effective_optimizer_starts"
                ),
                "reverse_effective_optimizer_starts": reverse[frame_id].get(
                    "effective_optimizer_starts"
                ),
            }
            for frame_id in frame_ids
        },
        "lineage": {
            "forward": {frame_id: forward[frame_id]["warm_start_from"] for frame_id in frame_ids},
            "reverse": {frame_id: reverse[frame_id]["warm_start_from"] for frame_id in frame_ids},
            "expected_failure_frame": "all_masked_failure",
            "expected_noise_control": "noise_only",
        },
        "warm_start_seed_usage": {
            "forward": _seed_usage(forward_calls),
            "reverse": _seed_usage(reverse_calls),
        },
        "disagreement_on_signal_frames": {
            "independent_vs_forward": _difference_summary(
                independent, forward, signal_filter, role_filter=signal_roles
            ),
            "independent_vs_reverse": _difference_summary(
                independent, reverse, signal_filter, role_filter=signal_roles
            ),
            "forward_vs_reverse": _difference_summary(
                forward, reverse, signal_filter, role_filter=signal_roles
            ),
        },
        "all_frame_disagreement_including_controls": {
            "independent_vs_forward": _difference_summary(independent, forward, frame_role_filter),
            "independent_vs_reverse": _difference_summary(independent, reverse, frame_role_filter),
            "forward_vs_reverse": _difference_summary(forward, reverse, frame_role_filter),
        },
        "interpretation": (
            "The batch runner records previous-frame geometry passed to pipeline.analyze_frame and the candidate "
            "optimizer start_values returned by the fitter. In butterfly_curvature arc fits, a data-derived algebraic "
            "seed can replace free parameter starts when enough current-frame arcs are available; this comparison "
            "therefore does not establish sensitivity to previous-frame optimizer starts unless the recorded starts "
            "differ. It does not measure accuracy against experimental truth or validate a structural mechanism."
        ),
    }

    calls = [*independent_calls, *forward_calls, *reverse_calls]
    for row in calls:
        selected = forward if row["direction"] == "forward" else reverse
        row["warm_start_from"] = (
            selected[row["frame_id"]]["warm_start_from"]
            if row["direction"] in {"forward", "reverse"}
            else None
        )
    manifest["runs"] = summary["runs"]
    manifest["source_integrity"] = source_integrity
    manifest["saved_input_integrity"] = file_integrity
    report = _format_report(summary, comparisons, settings_record)

    comparison_fields = list(comparisons[0]) if comparisons else []
    call_fields = ["direction", "call_index", "frame_id", "warm_start_from",
                   "initial_parameter_count", "initial_parameters"]
    _write_json(destination / "campaign_manifest.json", manifest)
    _write_json(destination / "summary.json", summary)
    _write_csv(destination / "per_frame_comparison.csv", comparisons, comparison_fields)
    _write_csv(destination / "warm_start_calls.csv", calls, call_fields)
    (destination / "report.md").write_text(report, encoding="utf-8", newline="\n")
    summary["output_directory"] = str(destination)
    return summary


def run_known_arc_campaign(
    output: str | Path,
    *,
    seed: int = 20260927,
    force: bool = False,
) -> dict[str, Any]:
    """Run only the small known-arc control, without regenerating the FFT run."""

    destination = Path(output).expanduser().resolve()
    frame_ids = [
        "arc_00", "arc_01", "arc_02", "arc_all_masked_failure", "arc_04", "arc_noise_only",
    ]
    output_names = (
        "campaign_manifest.json", "summary.json", "per_frame_comparison.csv",
        "warm_start_calls.csv", "report.md", "inputs/arc_geometry.npz",
    )
    expected = [destination / name for name in output_names]
    expected.extend(
        destination / "inputs" / f"{frame_id}{suffix}"
        for frame_id in frame_ids for suffix in (".npy", "_mask.npy")
    )
    if any(path.exists() for path in expected) and not force:
        raise FileExistsError(
            "known-arc outputs already exist; choose a new directory or pass --force"
        )
    destination.mkdir(parents=True, exist_ok=True)
    input_dir = destination / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    frames = build_arc_control_sequence(shape=(64, 64), seed=seed)
    if [str(frame["frame_id"]) for frame in frames] != frame_ids:
        raise AssertionError("unexpected known-arc frame order")
    input_paths: dict[str, Path] = {}
    mask_paths: dict[str, Path] = {}
    manifest_frames: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = str(frame["frame_id"])
        image_path = input_dir / f"{frame_id}.npy"
        mask_path = input_dir / f"{frame_id}_mask.npy"
        np.save(image_path, np.asarray(frame["intensity_noisy"], dtype=np.float64), allow_pickle=False)
        np.save(mask_path, np.asarray(frame["mask"], dtype=bool), allow_pickle=False)
        input_paths[frame_id] = image_path
        mask_paths[frame_id] = mask_path
        truth = frame.get("truth")
        manifest_frames.append(
            {
                "frame_index": int(frame["frame_index"]),
                "frame_id": frame_id,
                "sequence_role": frame["sequence_role"],
                "input_path": image_path.relative_to(destination).as_posix(),
                "input_sha256": _sha256(image_path),
                "mask_path": mask_path.relative_to(destination).as_posix(),
                "mask_sha256": _sha256(mask_path),
                "generator_case": truth.get("case_id") if isinstance(truth, Mapping) else None,
                "axis_ratio_truth": _finite(truth.get("axis_ratio") if isinstance(truth, Mapping) else None),
                "noise_sigma": _finite(frame.get("noise_sigma")),
            }
        )

    qmap_object = frames[0]["qmap"]
    qmap = {
        "qx": np.asarray(qmap_object["qx"], dtype=float),
        "qy": np.asarray(qmap_object["qy"], dtype=float),
        "q": np.asarray(qmap_object["q"], dtype=float),
        "q_unit": str(qmap_object["q_unit"]),
    }
    geometry_path = input_dir / "arc_geometry.npz"
    np.savez_compressed(
        geometry_path,
        **{key: value for key, value in qmap.items() if key != "q_unit"},
    )
    analysis_config = benchmark_sequence.pipeline_analysis_config(q_window=(0.05, 1.1))
    analysis_config["analysis"]["draw_axis_deg"] = 90.0
    analysis_config["analysis"]["max_nfev"] = 300
    analysis_config["analysis"]["ellipse"]["multistart"] = 1
    source_before = _component_hashes()
    file_paths = [*input_paths.values(), *mask_paths.values(), geometry_path]
    file_before = {
        path.relative_to(destination).as_posix(): _sha256(path) for path in file_paths
    }
    manifest = {
        "schema_version": 1,
        "campaign": "known-arc active warm-start and sequence-order control",
        "synthetic_only": True,
        "generator": benchmark_arcs.GENERATOR_VERSION,
        "generator_sha256": _sha256(Path(benchmark_arcs.__file__).resolve()),
        "seed": seed,
        "shape": [64, 64],
        "q_unit": qmap["q_unit"],
        "axis_ratio_sequence": [0.40, 0.38, 0.20, 0.22],
        "analysis_config": analysis_config,
        "frames": manifest_frames,
        "geometry_file": {
            "path": geometry_path.relative_to(destination).as_posix(),
            "sha256": _sha256(geometry_path),
        },
        "source_sha256_before": source_before,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    forward_order = tuple(range(len(frames)))
    reverse_order = tuple(reversed(forward_order))
    independent, independent_calls, independent_run = _run_order(
        frames, input_paths, mask_paths, qmap, analysis_config,
        mode="independent", direction="independent", campaign="known_arc",
        frame_order=forward_order,
    )
    forward, forward_calls, forward_run = _run_order(
        frames, input_paths, mask_paths, qmap, analysis_config,
        mode="warm_start", direction="forward", campaign="known_arc",
        frame_order=forward_order,
    )
    reverse, reverse_calls, reverse_run = _run_order(
        frames, input_paths, mask_paths, qmap, analysis_config,
        mode="warm_start", direction="reverse", campaign="known_arc",
        frame_order=reverse_order,
    )

    file_after = {
        path.relative_to(destination).as_posix(): _sha256(path) for path in file_paths
    }
    source_after = _component_hashes()
    file_integrity = {
        key: {
            "sha256_before": file_before[key],
            "sha256_after": file_after[key],
            "unchanged": file_before[key] == file_after[key],
        }
        for key in file_before
    }
    source_integrity = {
        key: {
            "sha256_before": source_before[key],
            "sha256_after": source_after[key],
            "unchanged": source_before[key] == source_after[key],
        }
        for key in source_before
    }
    integrity = {
        "saved_inputs_unchanged": all(row["unchanged"] for row in file_integrity.values()),
        "analysis_sources_unchanged": all(row["unchanged"] for row in source_integrity.values()),
        "files": file_integrity,
        "sources": source_integrity,
    }

    signal_roles = {
        "known_arc_signal_before_jump", "known_arc_ratio_jump", "known_arc_signal_after_jump"
    }
    frame_roles = {str(frame["frame_id"]): str(frame["sequence_role"]) for frame in frames}
    comparisons: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = str(frame["frame_id"])
        truth = frame.get("truth")
        row: dict[str, Any] = {
            "frame_index": int(frame["frame_index"]),
            "frame_id": frame_id,
            "sequence_role": frame["sequence_role"],
            "axis_ratio_truth": _finite(
                truth.get("axis_ratio") if isinstance(truth, Mapping) else None
            ),
            "independent_status": independent[frame_id]["status"],
            "forward_status": forward[frame_id]["status"],
            "reverse_status": reverse[frame_id]["status"],
            "independent_warm_start_eligible": independent[frame_id].get("warm_start_eligible"),
            "forward_warm_start_eligible": forward[frame_id].get("warm_start_eligible"),
            "reverse_warm_start_eligible": reverse[frame_id].get("warm_start_eligible"),
            "forward_warm_start_from": forward[frame_id]["warm_start_from"],
            "reverse_warm_start_from": reverse[frame_id]["warm_start_from"],
            "independent_error": independent[frame_id]["error"],
            "forward_error": forward[frame_id]["error"],
            "reverse_error": reverse[frame_id]["error"],
            "independent_effective_optimizer_starts": independent[frame_id].get(
                "effective_optimizer_starts"
            ),
            "forward_effective_optimizer_starts": forward[frame_id].get(
                "effective_optimizer_starts"
            ),
            "reverse_effective_optimizer_starts": reverse[frame_id].get(
                "effective_optimizer_starts"
            ),
        }
        for name in METRICS:
            base = _finite(independent[frame_id].get(name))
            fwd = _finite(forward[frame_id].get(name))
            rev = _finite(reverse[frame_id].get(name))
            row[f"independent_{name}"] = base
            row[f"forward_{name}"] = fwd
            row[f"reverse_{name}"] = rev
            row[f"forward_abs_delta_{name}"] = (
                _normalise_metric_difference(name, base, fwd)
                if base is not None and fwd is not None else None
            )
            row[f"reverse_abs_delta_{name}"] = (
                _normalise_metric_difference(name, base, rev)
                if base is not None and rev is not None else None
            )
        comparisons.append(row)

    signal_filter = {
        frame_id: role for frame_id, role in frame_roles.items() if role in signal_roles
    }
    directional_difference = {
        "independent_vs_forward": _difference_summary(
            independent, forward, signal_filter, role_filter=signal_roles
        ),
        "independent_vs_reverse": _difference_summary(
            independent, reverse, signal_filter, role_filter=signal_roles
        ),
        "forward_vs_reverse": _difference_summary(
            forward, reverse, signal_filter, role_filter=signal_roles
        ),
    }
    stack_lineage = {
        "forward": {frame_id: forward[frame_id]["warm_start_from"] for frame_id in frame_ids},
        "reverse": {frame_id: reverse[frame_id]["warm_start_from"] for frame_id in frame_ids},
    }
    summary = {
        "campaign": manifest["campaign"],
        "synthetic_only": True,
        "frame_count": len(frames),
        "signal_frame_count": sum(role in signal_roles for role in frame_roles.values()),
        "runs": {
            "independent": independent_run,
            "forward_warm_start": forward_run,
            "reverse_warm_start": reverse_run,
        },
        "status_by_frame": {
            frame_id: {
                "role": frame_roles[frame_id],
                "independent": independent[frame_id]["status"],
                "forward": forward[frame_id]["status"],
                "reverse": reverse[frame_id]["status"],
                "warm_start_eligible": independent[frame_id].get("warm_start_eligible"),
                "independent_effective_optimizer_starts": independent[frame_id].get(
                    "effective_optimizer_starts"
                ),
                "forward_effective_optimizer_starts": forward[frame_id].get(
                    "effective_optimizer_starts"
                ),
                "reverse_effective_optimizer_starts": reverse[frame_id].get(
                    "effective_optimizer_starts"
                ),
            }
            for frame_id in frame_ids
        },
        "lineage": stack_lineage,
        "warm_start_seed_usage": {
            "forward": _seed_usage(forward_calls),
            "reverse": _seed_usage(reverse_calls),
        },
        "disagreement_on_signal_frames": directional_difference,
        "integrity": integrity,
        "interpretation": (
            "This known-geometry arc control confirms batch-to-pipeline propagation of eligible fitted geometry and "
            "records effective optimizer start_values. The curvature arc fitter may replace free starts with a "
            "current-frame algebraic seed; previous-frame initialization sensitivity is established only when the "
            "recorded optimizer starts differ. Disagreement is for this synthetic image family only."
        ),
    }
    calls = [*independent_calls, *forward_calls, *reverse_calls]
    for row in calls:
        direction = str(row["direction"])
        chosen = forward if direction == "forward" else reverse if direction == "reverse" else {}
        row["warm_start_from"] = chosen.get(row["frame_id"], {}).get("warm_start_from")

    manifest["runs"] = summary["runs"]
    manifest["warm_start_seed_usage"] = summary["warm_start_seed_usage"]
    manifest["source_integrity"] = source_integrity
    manifest["saved_input_integrity"] = file_integrity
    report_lines = [
        "# Known-arc initialization propagation control",
        "",
        "This small supplementary campaign uses the independent known-observable-arc image generator and the production `run_batch` plus `pipeline.analyze_frame` path. It is synthetic geometry evidence, not an experimental SAXS validation.",
        "",
        "The 64×64 sequence uses axis ratios 0.40, 0.38, then an abrupt change to 0.20 and 0.22. An all-masked frame follows the jump; a noise-only control is last. The same images, q map, masks, and analysis settings are used in independent, forward warm-start, and reverse warm-start runs.",
        "",
        f"Saved inputs unchanged: `{integrity['saved_inputs_unchanged']}`. Analysis source hashes unchanged: `{integrity['analysis_sources_unchanged']}`.",
        "",
        "## Seed use and lineage",
        "",
        f"- Forward: {summary['warm_start_seed_usage']['forward']['calls_with_initial_parameters']} of {summary['warm_start_seed_usage']['forward']['calls']} calls received initial parameters.",
        f"- Reverse: {summary['warm_start_seed_usage']['reverse']['calls_with_initial_parameters']} of {summary['warm_start_seed_usage']['reverse']['calls']} calls received initial parameters.",
        "Call rows record parameters received by the analysis callback; the per-frame table records optimizer `start_values`. In the curvature arc path, current-frame algebraic initialization can replace free starts, so parameter receipt alone does not establish effective warm-start dependence.",
        "- Forward lineage: " + "; ".join(f"`{key} ← {value or 'no seed'}`" for key, value in stack_lineage['forward'].items()),
        "- Reverse lineage: " + "; ".join(f"`{key} ← {value or 'no seed'}`" for key, value in stack_lineage['reverse'].items()),
        "",
        "## Parameter disagreement on known-arc signal frames",
        "",
        "| Comparison | Parameter | Paired frames | Median absolute difference | Maximum absolute difference | Maximum relative difference (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for comparison, label in (
        ("independent_vs_forward", "Independent vs forward"),
        ("independent_vs_reverse", "Independent vs reverse"),
        ("forward_vs_reverse", "Forward vs reverse"),
    ):
        for name, detail in directional_difference[comparison].items():
            def show(value: Any) -> str:
                parsed = _finite(value)
                return "NA" if parsed is None else f"{parsed:.6g}"

            report_lines.append(
                f"| {label} | `{name}` | {detail['n_paired']} | "
                f"{show(detail['median_absolute_difference'])} | "
                f"{show(detail['maximum_absolute_difference'])} | "
                f"{show(detail['maximum_relative_difference_percent'])} |"
            )
    report_lines.extend([
        "",
        "## Frame status",
        "",
        "| Frame | Role | Independent | Forward | Reverse | Candidate eligible |",
        "|---|---|---|---|---|---|",
    ])
    for frame_id in frame_ids:
        row = summary["status_by_frame"][frame_id]
        report_lines.append(
            f"| `{frame_id}` | `{row['role']}` | {row['independent']} | {row['forward']} | "
            f"{row['reverse']} | {row['warm_start_eligible']} |"
        )
    report_lines.extend([
        "",
        "The all-masked frame remains in the sequence and must not seed the following valid frame. The noise control is reported separately; any candidate it produces is not evidence of a butterfly pattern.",
        "",
        "This campaign tests batch-to-pipeline parameter propagation and records optimizer starts selected by the production path. It does not establish sensitivity to previous-frame optimizer starts when current-frame algebraic initialization replaces them, nor accuracy on instrument data, uncertainty coverage, or a physical SAXS mechanism.",
        "",
    ])
    summary["output_directory"] = str(destination)
    _write_json(destination / "campaign_manifest.json", manifest)
    _write_json(destination / "summary.json", summary)
    _write_csv(
        destination / "per_frame_comparison.csv",
        comparisons,
        list(comparisons[0]) if comparisons else [],
    )
    _write_csv(
        destination / "warm_start_calls.csv",
        calls,
        ["campaign", "direction", "call_index", "frame_id", "warm_start_from",
         "initial_parameter_count", "initial_parameters"],
    )
    (destination / "report.md").write_text("\n".join(report_lines), encoding="utf-8", newline="\n")
    return summary


def _format_report(
    summary: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> str:
    lines = [
        "# Initialization sensitivity and sequence-order benchmark",
        "",
        "This report compares independent frame fits with forward and reverse warm-start runs.",
        "The frames are synthetic: finite lamellar slabs were generated in real space and transformed by a two-dimensional FFT.",
        "No experimental SAXS data were available for this campaign.",
        "",
        "## Protocol",
        "",
        f"- Detector-array shape: `{settings['shape'][0]} × {settings['shape'][1]}`.",
        f"- Declared reciprocal-space unit: `{settings['q_unit']}`.",
        f"- First-order target SNR: `{settings['target_first_order_snr']}`.",
        f"- Nominal spacing changes from {settings['pre_jump_spacing_nm'][1]:.3g} nm to {settings['post_jump_spacing_nm'][0]:.3g} nm at `jump_03`.",
        "- `all_masked_failure` has no valid detector pixels and is placed between the jump and a subsequent signal frame.",
        "- `noise_only` is a separate negative control. Each run uses the same saved intensity arrays, masks, q maps, and analysis settings.",
        "- Reverse processing changes the order of `run_batch`; results are joined to the original sequence by frame ID.",
        "",
        "## Run completion",
        "",
        f"Input arrays and analysis sources unchanged during fitting: `{summary['integrity']['saved_inputs_unchanged']}` and `{summary['integrity']['analysis_sources_unchanged']}`.",
        "",
        "| Run | OK | Warning | Failed | Skipped |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("Independent", "independent"),
                       ("Forward warm start", "forward_warm_start"),
                       ("Reverse warm start", "reverse_warm_start")):
        counts = summary["runs"][key]["status_counts"]
        lines.append(
            f"| {label} | {counts['ok']} | {counts['warning']} | {counts['failed']} | {counts['skipped']} |"
        )
    lines.extend(["", "## Warm-start lineage", ""])
    for direction in ("forward", "reverse"):
        seed_status = summary["warm_start_seed_usage"][direction]
        lines.append(
            f"- **{direction.title()} seeds used**: {seed_status['calls_with_initial_parameters']} of {seed_status['calls']} frame calls (`{seed_status['status']}`)."
        )
    for direction in ("forward", "reverse"):
        lineage = summary["lineage"][direction]
        lines.append(f"- **{direction.title()}**: " + "; ".join(
            f"`{frame_id} ← {source or 'no seed'}`" for frame_id, source in lineage.items()
        ))
    lines.extend([
        "",
        "A failed frame must not become a seed. The lineage records the last usable result passed by the batch API; the `all_masked_failure` row is retained in its original position.",
        "",
        "## Parameter disagreement on signal frames",
        "",
        "Values below summarize paired finite estimates only. `n_paired` can be lower than the number of signal frames when a fit does not return an estimate.",
        "",
        "| Comparison | Parameter | Paired frames | Median absolute difference | Maximum absolute difference | Maximum relative difference (%) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for comparison, label in (("independent_vs_forward", "Independent vs forward"),
                              ("independent_vs_reverse", "Independent vs reverse"),
                              ("forward_vs_reverse", "Forward vs reverse")):
        values = summary["disagreement_on_signal_frames"][comparison]
        for name, detail in values.items():
            def show(number: Any) -> str:
                parsed = _finite(number)
                return "NA" if parsed is None else f"{parsed:.6g}"
            lines.append(
                f"| {label} | `{name}` | {detail['n_paired']} | "
                f"{show(detail['median_absolute_difference'])} | "
                f"{show(detail['maximum_absolute_difference'])} | "
                f"{show(detail['maximum_relative_difference_percent'])} |"
            )
    lines.extend([
        "",
        "## Per-frame status",
        "",
        "| Frame | Role | Independent | Forward | Reverse | Forward seed source | Reverse seed source |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in comparisons:
        status = summary["status_by_role"][str(row["frame_id"])]
        lineage = summary["lineage"]
        lines.append(
            f"| `{row['frame_id']}` | `{row['sequence_role']}` | {status['independent']} | "
            f"{status['forward']} | {status['reverse']} | "
            f"{lineage['forward'][str(row['frame_id'])] or '—'} | "
            f"{lineage['reverse'][str(row['frame_id'])] or '—'} |"
        )
    lines.extend([
        "",
        "## Scope",
        "",
        "This campaign measures numerical dependence on initial parameters and processing direction for this generated sequence. It does not test trajectory-method accuracy, uncertainty coverage, detector calibration error, sample-to-sample transfer, or performance on real experimental images. The finite-stack FFT model is a simplified two-dimensional generator; its latent spacing is not a unique inverse target for all fitted ellipse parameters.",
        "",
        "Saved arrays, masks, geometry, hashes, per-frame estimates, and warm-start inputs are in this directory. Re-run `py -3.13 scripts/benchmark_initialization.py --output results/validation/jac_initialization --force` to regenerate this campaign.",
        "",
    ])
    return "\n".join(lines)


def _reported_issues(result: Mapping[str, Any]) -> bool:
    """Return whether the completed campaign reports warnings or partial work."""

    runs = result.get("runs")
    if not isinstance(runs, Mapping) or not runs:
        return True
    for run in runs.values():
        if not isinstance(run, Mapping) or bool(run.get("cancelled", False)):
            return True
        counts = run.get("status_counts")
        if not isinstance(counts, Mapping):
            return True
        for status in ("warning", "failed", "skipped"):
            try:
                if int(counts.get(status, 0) or 0) > 0:
                    return True
            except (TypeError, ValueError, OverflowError):
                return True

    integrity = result.get("integrity")
    if isinstance(integrity, Mapping):
        for key in ("saved_inputs_unchanged", "analysis_sources_unchanged"):
            if key in integrity and integrity[key] is not True:
                return True
        for key in ("files", "sources"):
            records = integrity.get(key)
            if isinstance(records, Mapping) and any(
                not isinstance(record, Mapping) or record.get("unchanged") is not True
                for record in records.values()
            ):
                return True
    return False


class _JsonArgumentParser(argparse.ArgumentParser):
    """Emit machine-readable input errors on stdout and explanations on stderr."""

    def error(self, message: str) -> None:
        print(
            json.dumps(
                {"status": "error", "error": message},
                ensure_ascii=True,
                allow_nan=False,
            ),
            file=sys.stdout,
        )
        self.exit(2, f"{self.prog}: error: {message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _JsonArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/validation/jac_initialization"),
        help="campaign output directory (default: results/validation/jac_initialization)",
    )
    parser.add_argument("--size", type=int, default=192, help="square detector side in pixels")
    parser.add_argument("--seed", type=int, default=20260927, help="deterministic generator seed")
    parser.add_argument("--force", action="store_true", help="replace this campaign's named outputs")
    parser.add_argument(
        "--known-arcs-only",
        action="store_true",
        help="run only the 64×64 known-arc active-seed control",
    )
    args = parser.parse_args(argv)
    try:
        if args.known_arcs_only:
            result = run_known_arc_campaign(
                args.output,
                seed=args.seed,
                force=args.force,
            )
        else:
            result = run_initialization_campaign(
                args.output,
                shape=(args.size, args.size),
                seed=args.seed,
                force=args.force,
            )
    except (FileExistsError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": str(exc)},
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            _json_safe(result),
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 1 if _reported_issues(result) else 0


if __name__ == "__main__":
    raise SystemExit(main())
