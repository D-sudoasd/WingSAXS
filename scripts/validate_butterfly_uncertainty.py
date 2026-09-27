"""Repeated-trial validation of butterfly intervals on known synthetic images.

This development benchmark measures empirical central-95% interval coverage
for the production curvature pipeline and reports deterministic q-map
perturbation sensitivity. It does not establish calibrated confidence levels
or replace validation with experimental SAXS data.

Example::

    py -3.13 scripts/validate_butterfly_uncertainty.py \
        --output results/validation/uncertainty_run --trials 6 --resamples 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from butterfly_saxs.benchmark_arcs import GENERATOR_HASH, generate_arc_case  # noqa: E402
from butterfly_saxs.butterfly_settings import METHOD_VERSION  # noqa: E402
from butterfly_saxs.butterfly_uncertainty import (  # noqa: E402
    _copy_qmap_with_perturbation,
    resample_butterfly,
)
from butterfly_saxs.pipeline import analyze_frame  # noqa: E402
from butterfly_saxs.uncertainty_validation import (  # noqa: E402
    DEFAULT_PARAMETERS,
    summarize_calibration_sensitivity,
    summarize_uncertainty_trials,
)


Q_WINDOW = (0.05, 1.1)
VALIDATION_DEPENDENCIES = (
    "analysis_config.py",
    "arc_geometry.py",
    "arc_support.py",
    "benchmark_arcs.py",
    "butterfly.py",
    "butterfly_quality.py",
    "butterfly_ridge.py",
    "butterfly_settings.py",
    "butterfly_uncertainty.py",
    "cancellation.py",
    "ellipse.py",
    "geometry.py",
    "io.py",
    "masking.py",
    "models.py",
    "observables.py",
    "parameters.py",
    "pipeline.py",
    "public_ellipse.py",
    "ridge_inputs.py",
    "ridge_profiles.py",
    "serialization.py",
    "settings.py",
    "uncertainty_validation.py",
    "validation.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hashes() -> dict[str, Any]:
    source_root = REPO / "src" / "butterfly_saxs"
    source_files = [source_root / name for name in VALIDATION_DEPENDENCIES]
    missing_files = [str(path) for path in source_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"missing validation dependency files: {missing_files}")
    tree_digest = hashlib.sha256()
    for path in source_files:
        relative = str(path.relative_to(REPO)).replace("\\", "/")
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(path.read_bytes())
        tree_digest.update(b"\0")
    return {
        "validation_dependency_sha256": tree_digest.hexdigest(),
        "scope": "listed production fitting, ridge, q-map, IO and scoring dependencies",
        "validation_script_sha256": _sha256(Path(__file__).resolve()),
        "generator_sha256": GENERATOR_HASH,
        "source_file_count": len(source_files),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _analysis(seed: int, *, max_nfev: int) -> dict[str, Any]:
    return {
        "ridge_method": "butterfly_curvature",
        "q_window": list(Q_WINDOW),
        "draw_axis_deg": 90.0,
        "ellipse_multistart": 1,
        "max_nfev": int(max_nfev),
        "butterfly": {
            "stage": "evaluate",
            "resamples": 0,
            "sensitivity": False,
            "seed": int(seed),
        },
    }


def _fit_summary(image: np.ndarray, qmap: Any, mask: np.ndarray, *, seed: int, max_nfev: int) -> dict[str, Any]:
    result = analyze_frame(
        image,
        qmap=qmap,
        mask=mask,
        config={"analysis": _analysis(seed, max_nfev=max_nfev)},
        full2d=False,
    ).butterfly
    if not isinstance(result, dict):
        raise RuntimeError("production pipeline returned no butterfly result")
    candidate_fit = result.get("candidate_fit")
    candidate_fit = candidate_fit if isinstance(candidate_fit, dict) else {}
    quantitative = result.get("quantitative_parameters")
    quantitative = quantitative if isinstance(quantitative, dict) else {}
    groups = {
        (point.get("branch_id"), point.get("side"))
        for point in result.get("points", [])
        if isinstance(point, dict)
        and point.get("accepted")
        and point.get("side") in ("upper", "lower")
        and point.get("branch_id") in (0, 1)
    }
    missing_sides = sorted({(branch, side) for branch in (0, 1) for side in ("upper", "lower")} - groups)
    return {
        "candidate": {key: candidate_fit.get(key) for key in (*DEFAULT_PARAMETERS, "theta_deg")},
        "solver_success": bool(candidate_fit.get("success", False)),
        "measurement_status": result.get("measurement_status"),
        "parameter_status": {
            key: value.get("status")
            for key, value in quantitative.items()
            if isinstance(value, dict)
        },
        "confidence": {
            key: value.get("confidence")
            for key, value in quantitative.items()
            if isinstance(value, dict)
        },
        "condition": candidate_fit.get("condition"),
        "fit_message": candidate_fit.get("message"),
        "topology_failed": bool(missing_sides),
        "missing_observed_sides": [[branch, side] for branch, side in missing_sides],
    }


def _truth_summary(truth: dict[str, Any]) -> dict[str, Any]:
    return {
        "a": truth.get("a"),
        "b": truth.get("b"),
        "axis_ratio": truth.get("axis_ratio", truth.get("b_over_a")),
    }


def _run_trial(case_id: str, seed: int, shape: int, resamples: int, max_nfev: int) -> dict[str, Any]:
    case = generate_arc_case(case_id, seed=seed, shape=(shape, shape))
    image, qmap, mask = case["image"], case["qmap"], case["mask"]
    started = time.perf_counter()
    truth = _truth_summary(case["truth"])
    try:
        baseline = _fit_summary(image, qmap, mask, seed=seed, max_nfev=max_nfev)
    except Exception as exc:
        return {
            "trial_id": f"{case_id}_{seed}",
            "case_id": case_id,
            "seed": int(seed),
            "shape": [int(shape), int(shape)],
            "execution_status": "error",
            "error": str(exc),
            "solver_success": False,
            "truth": truth,
            "candidate": {},
            "parameter_status": {},
            "confidence": {},
            "condition": None,
            "intervals": {},
            "resampling": {"status": "not_run_baseline_fit_error", "attempted": 0, "successes": 0, "failed": 0},
            "elapsed_s": float(time.perf_counter() - started),
            "generator_sha256": case["truth"]["generator_sha256"],
        }

    def refit(resampled_image: np.ndarray, options: dict[str, Any]) -> dict[str, Any]:
        result = _fit_summary(
            resampled_image,
            options.get("qmap", qmap),
            np.asarray(options.get("mask", mask), dtype=bool),
            seed=int(options.get("seed", seed)),
            max_nfev=max_nfev,
        )
        callback_result = {
            "candidate_fit": {
                **result["candidate"],
                "success": result["solver_success"],
            },
            "measurement_status": result["measurement_status"],
        }
        if result["topology_failed"]:
            callback_result["topology_failed"] = True
            callback_result["message"] = "resampling lost observed sides: " + repr(result["missing_observed_sides"])
        return callback_result

    if baseline["solver_success"]:
        try:
            interval_result = resample_butterfly(
                image,
                qmap,
                mask=mask,
                q_window=Q_WINDOW,
                refit=refit,
                options={
                    "resamples": int(resamples),
                    "seed": int(seed + 97_003),
                    "block_shape": (8, 8),
                    "smoothing_sigma": 1.0,
                },
            )
            intervals = interval_result["intervals"]
            interval_kind = interval_result["interval_kind"]
            interval_calibrated = interval_result["coverage_calibrated"]
            resampling = {
                "status": "complete",
                "attempted": interval_result["n_draws"],
                "successes": interval_result["successes"],
                "failed": interval_result["failed"],
                "failed_topology": interval_result["failed_topology"],
                "failed_refits": interval_result["failed_refits"],
                "success_fraction": interval_result["success_fraction"],
            }
        except Exception as exc:
            # The baseline fit is still a measured candidate if optional
            # uncertainty evaluation fails; do not erase it or its truth.
            intervals = {}
            interval_kind = "central95_empirical_interval"
            interval_calibrated = False
            resampling = {
                "status": "error",
                "attempted": int(resamples),
                "successes": 0,
                "failed": int(resamples),
                "error": str(exc),
            }
    else:
        intervals = {}
        interval_kind = "central95_empirical_interval"
        interval_calibrated = False
        resampling = {
            "status": "not_run_baseline_solver_failed",
            "attempted": 0,
            "successes": 0,
            "failed": 0,
            "success_fraction": 0.0,
        }
    return {
        "trial_id": f"{case_id}_{seed}",
        "case_id": case_id,
        "seed": int(seed),
        "shape": [int(shape), int(shape)],
        "execution_status": "complete",
        "solver_success": baseline["solver_success"],
        "measurement_status": baseline["measurement_status"],
        "truth": truth,
        "candidate": baseline["candidate"],
        "parameter_status": baseline["parameter_status"],
        "confidence": baseline["confidence"],
        "condition": baseline["condition"],
        "intervals": intervals,
        "interval_kind": interval_kind,
        "interval_calibrated": interval_calibrated,
        "resampling": resampling,
        "elapsed_s": float(time.perf_counter() - started),
        "generator_sha256": case["truth"]["generator_sha256"],
    }


def _run_identifiability_case(case_id: str, seed: int, shape: int, resamples: int, max_nfev: int) -> dict[str, Any]:
    case = generate_arc_case(case_id, seed=seed, shape=(shape, shape))
    image, qmap, mask = case["image"], case["qmap"], case["mask"]
    baseline = _fit_summary(image, qmap, mask, seed=seed, max_nfev=max_nfev)

    def refit(resampled_image: np.ndarray, options: dict[str, Any]) -> dict[str, Any]:
        result = _fit_summary(
            resampled_image,
            options.get("qmap", qmap),
            np.asarray(options.get("mask", mask), dtype=bool),
            seed=int(options.get("seed", seed)),
            max_nfev=max_nfev,
        )
        callback_result = {"candidate_fit": {**result["candidate"], "success": result["solver_success"]}}
        if result["topology_failed"]:
            callback_result["topology_failed"] = True
            callback_result["message"] = "resampling lost observed sides: " + repr(result["missing_observed_sides"])
        return callback_result

    intervals: dict[str, Any] = {}
    resampling: dict[str, Any] = {
        "status": "not_run_baseline_solver_failed",
        "attempted": 0,
        "successes": 0,
        "failed": 0,
        "success_fraction": 0.0,
    }
    if baseline["solver_success"]:
        try:
            interval_result = resample_butterfly(
                image,
                qmap,
                mask=mask,
                q_window=Q_WINDOW,
                refit=refit,
                options={"resamples": int(resamples), "seed": int(seed + 197_003), "block_shape": (8, 8)},
            )
            intervals = interval_result["intervals"]
            resampling = {
                "status": "complete",
                "attempted": interval_result["n_draws"],
                "successes": interval_result["successes"],
                "failed": interval_result["failed"],
                "success_fraction": interval_result["success_fraction"],
            }
        except Exception as exc:
            resampling = {
                "status": "error",
                "attempted": int(resamples),
                "successes": 0,
                "failed": int(resamples),
                "error": str(exc),
            }
    return {
        "case_id": case_id,
        "seed": int(seed),
        "shape": [int(shape), int(shape)],
        "execution_status": "complete",
        "solver_success": baseline["solver_success"],
        "measurement_status": baseline["measurement_status"],
        "truth": _truth_summary(case["truth"]),
        "candidate": baseline["candidate"],
        "parameter_status": baseline["parameter_status"],
        "confidence": baseline["confidence"],
        "condition": baseline["condition"],
        "intervals": intervals,
        "resampling": resampling,
    }


def _run_calibration_sensitivity(case_id: str, seed: int, shape: int, max_nfev: int) -> dict[str, Any]:
    case = generate_arc_case(case_id, seed=seed, shape=(shape, shape))
    image, qmap, mask = case["image"], case["qmap"], case["mask"]
    baseline = _fit_summary(image, qmap, mask, seed=seed, max_nfev=max_nfev)
    variants: list[dict[str, Any]] = []
    perturbations = (
        ("qcal_scale_minus_1pct", (0.0, 0.0), 0.99),
        ("qcal_scale_plus_1pct", (0.0, 0.0), 1.01),
        ("center_row_minus_0p5px", (-0.5, 0.0), 1.0),
        ("center_row_plus_0p5px", (0.5, 0.0), 1.0),
        ("center_col_minus_0p5px", (0.0, -0.5), 1.0),
        ("center_col_plus_0p5px", (0.0, 0.5), 1.0),
    )
    for label, center_delta, qcal_scale in perturbations:
        perturbed_qmap = _copy_qmap_with_perturbation(qmap, image.shape, center_delta, qcal_scale)
        try:
            fit = _fit_summary(image, perturbed_qmap, mask, seed=seed, max_nfev=max_nfev)
            variants.append({
                "label": label,
                "execution_status": "complete",
                "solver_success": fit["solver_success"],
                "candidate": fit["candidate"],
                "perturbation": {"center_delta_px": list(center_delta), "qcal_scale": qcal_scale},
            })
        except Exception as exc:
            variants.append({
                "label": label,
                "execution_status": "error",
                "solver_success": False,
                "candidate": {},
                "error": str(exc),
                "perturbation": {"center_delta_px": list(center_delta), "qcal_scale": qcal_scale},
            })
    return {
        "case_id": case_id,
        "seed": int(seed),
        "truth": _truth_summary(case["truth"]),
        "baseline_solver_success": baseline["solver_success"],
        "baseline_measurement_status": baseline["measurement_status"],
        "baseline_condition": baseline["condition"],
        "assumed_perturbation_basis": {
            "relative_q_calibration": 0.01,
            "beam_center": 0.5,
            "meaning": "deterministic synthetic sensitivity grid; values are not instrument uncertainty estimates",
        },
        **summarize_calibration_sensitivity(baseline["candidate"], variants),
    }


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="ellipse_ratio_400")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--resamples", type=int, default=12)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--identifiability-case", default="partial_arcs")
    parser.add_argument("--identifiability-resamples", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.trials < 1 or args.resamples < 1 or args.shape < 24:
        raise ValueError("trials/resamples must be >= 1 and shape must be >= 24")
    if args.identifiability_resamples < 1 or args.max_nfev < 1:
        raise ValueError("identifiability-resamples and max-nfev must be >= 1")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"use a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(f"use a new output directory: {report_path}")

    hashes_before = _code_hashes()
    started = time.perf_counter()
    trials: list[dict[str, Any]] = []
    for index in range(args.trials):
        trial_seed = int(args.seed + index)
        print(f"trial {index + 1}/{args.trials}: {args.case} seed={trial_seed}", file=sys.stderr, flush=True)
        try:
            trial = _run_trial(args.case, trial_seed, args.shape, args.resamples, args.max_nfev)
        except Exception as exc:
            trial = {
                "trial_id": f"{args.case}_{trial_seed}",
                "case_id": args.case,
                "seed": trial_seed,
                "shape": [args.shape, args.shape],
                "execution_status": "error",
                "error": str(exc),
                "truth": {},
                "candidate": {},
                "intervals": {},
                "resampling": {"attempted": 0, "successes": 0, "failed": 0},
            }
        trials.append(trial)
    print(f"identifiability diagnostic: {args.identifiability_case}", file=sys.stderr, flush=True)
    try:
        identifiability = _run_identifiability_case(
            args.identifiability_case,
            int(args.seed + 997_003),
            args.shape,
            args.identifiability_resamples,
            args.max_nfev,
        )
        identifiability["scored_summary"] = summarize_uncertainty_trials([identifiability])
    except Exception as exc:
        identifiability = {
            "case_id": args.identifiability_case,
            "execution_status": "error",
            "error": str(exc),
            "scored_summary": summarize_uncertainty_trials([]),
        }
    print("calibration sensitivity: deterministic q-map perturbations", file=sys.stderr, flush=True)
    try:
        calibration = _run_calibration_sensitivity(
            args.case, int(args.seed + 1_997_003), args.shape, args.max_nfev
        )
    except Exception as exc:
        calibration = {"execution_status": "error", "error": str(exc)}

    hashes_after = _code_hashes()
    report = {
        "schema_version": "wingsaxs.uncertainty_validation.v1",
        "context": {
            "method_version": METHOD_VERSION,
            "generator_sha256": GENERATOR_HASH,
            "code_sha256_before": hashes_before,
            "code_sha256_after": hashes_after,
            "code_unchanged": hashes_before == hashes_after,
            "python": sys.version,
            "versions": {name: _package_version(name) for name in ("numpy", "scipy", "pyFAI")},
            "case": args.case,
            "trial_count_requested": args.trials,
            "resamples_per_trial": args.resamples,
            "shape": [args.shape, args.shape],
            "seed_start": args.seed,
            "fit": {"ridge_method": "butterfly_curvature", "q_window": list(Q_WINDOW), "max_nfev": args.max_nfev},
            "interval_method": "empirical_image_residual_spatial_block",
            "interval_kind": "central95_empirical_interval",
            "interval_calibrated": False,
            "interval_scope": "image residual resampling only; calibration excluded",
            "parameters_scored": list(DEFAULT_PARAMETERS),
            "parameters_not_scored": {
                "theta_deg": "not scored because the axial angle requires periodic error and interval handling",
                "q_star": "not included in this fit-parameter coverage campaign",
            },
            "elapsed_s": float(time.perf_counter() - started),
        },
        "coverage": summarize_uncertainty_trials(trials),
        "trials": trials,
        "identifiability_diagnostic": identifiability,
        "calibration_sensitivity": calibration,
        "limits": [
            "synthetic known-truth benchmark only; no real experimental dataset was supplied",
            "the independent arc generator is geometric and is not a physical SAXS intensity forward model",
            "coverage is conditional on this generator, case, image size, fitting recipe and spatial-block residual model",
            "the partial-arc identifiability check is a single seed and does not scan all axis ratios, angles, arc lengths or q windows",
            "the deterministic q-map sensitivity grid is assumed for diagnosis and is not a measured calibration error distribution",
            "coverage metrics score a, b and b/a only; theta_deg and q_star are not coverage-scored",
            "repeated-trial counts are a modest development check and do not support a general 95% coverage claim",
        ],
    }
    report_path.write_text(
        json.dumps(_json_finite(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"report": str(report_path), "trial_count": len(trials), "coverage": report["coverage"]},
            ensure_ascii=True,
            allow_nan=False,
        ),
        flush=True,
    )
    has_errors = any(row.get("execution_status") != "complete" for row in trials)
    has_errors |= any(row.get("resampling", {}).get("status") == "error" for row in trials)
    has_errors |= identifiability.get("execution_status") == "error"
    has_errors |= identifiability.get("resampling", {}).get("status") == "error"
    has_errors |= calibration.get("execution_status") == "error"
    has_errors |= any(
        variant.get("execution_status") != "complete"
        for variant in calibration.get("variants", {}).values()
    )
    return int(has_errors or not report["context"]["code_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
