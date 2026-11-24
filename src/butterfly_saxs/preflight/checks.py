"""Check accumulation for the preflight orchestration boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .context import (
    PreflightContext,
    first_value as _default_first_value,
    nested_mapping as _default_nested_mapping,
)
from .hash import display_path as _default_display_path, json_safe as _default_json_safe
from .io import PreflightReadState
from .uncertainty import (
    correction_check as _default_correction_check,
    resolve_correction_value as _default_resolve_correction_value,
    state_value as _default_state_value,
    uncertainty_check as _default_uncertainty_check,
    uncertainty_provenance as _default_uncertainty_provenance,
)


@dataclass
class CheckBuilder:
    """Build ordered checks while retaining legacy warning/error lists."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, check_id: str, status: str, reason: str, evidence: Any) -> None:
        check = {
            "id": check_id,
            "status": status,
            "reason": str(reason),
            "evidence": evidence,
        }
        self.checks.append(check)
        if status == "warn":
            self.warnings.append(str(reason))
        elif status == "fail":
            self.errors.append(str(reason))

    @staticmethod
    def append(
        checks: list[dict[str, Any]],
        warnings: list[str],
        errors: list[str],
        check_id: str,
        status: str,
        reason: str,
        evidence: Any,
    ) -> None:
        """Compatibility adapter for callers/tests using the old helper."""

        builder = CheckBuilder(checks=checks, warnings=warnings, errors=errors)
        builder.add(check_id, status, reason, evidence)


def header_geometry_fields(metadata: Mapping[str, Any]) -> dict[str, float]:
    header = metadata.get("header")
    if not isinstance(header, Mapping):
        return {}
    aliases = {
        "dist": "dist",
        "distance": "dist",
        "poni1": "poni1",
        "poni2": "poni2",
        "rot1": "rot1",
        "rot2": "rot2",
        "rot3": "rot3",
        "wavelength": "wavelength",
    }
    result: dict[str, float] = {}
    for key, value in header.items():
        normal = aliases.get(str(key).casefold().replace(" ", ""))
        if normal is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result[normal] = numeric
    return result


def header_geometry_check(
    image_metadata: Sequence[Mapping[str, Any]],
    qmap_metadata: Mapping[str, Any],
    has_poni: bool,
) -> tuple[str, str, dict[str, Any]]:
    if not has_poni:
        return (
            "pass",
            "header geometry comparison is not applicable to pixel-q",
            {"applicable": False},
        )
    poni_fields: dict[str, float] = {}
    for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"):
        try:
            number = float(qmap_metadata[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(number):
            poni_fields[key] = number
    headers = [header_geometry_fields(metadata) for metadata in image_metadata]
    comparable = (
        sorted(set(poni_fields).intersection(*(set(item) for item in headers)))
        if headers
        else []
    )
    differences: dict[str, Any] = {}
    for key in comparable:
        values = [item[key] for item in headers]
        if not all(
            math.isclose(item, poni_fields[key], rel_tol=1e-6, abs_tol=1e-9)
            for item in values
        ):
            differences[key] = {"poni": poni_fields[key], "headers": values}
    if not comparable:
        return (
            "warn",
            "header geometry fields are unavailable for comparison with PONI",
            {
                "poni_fields": sorted(poni_fields),
                "comparable_fields": [],
                "differences": {},
            },
        )
    if differences:
        return (
            "warn",
            "header geometry differs from PONI fields",
            {
                "poni_fields": sorted(poni_fields),
                "comparable_fields": comparable,
                "differences": differences,
            },
        )
    return (
        "pass",
        "available header geometry agrees with PONI",
        {
            "poni_fields": sorted(poni_fields),
            "comparable_fields": comparable,
            "differences": {},
        },
    )


def status_from_checks(checks: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    values = {str(item.get("status", "pass")) for item in checks}
    if "fail" in values:
        return "red", "FAIL"
    if "warn" in values:
        return "yellow", "WARN"
    return "green", "PASS"


def _default_add_check(
    checks: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    check_id: str,
    status: str,
    reason: str,
    evidence: Any,
) -> None:
    CheckBuilder.append(
        checks,
        warnings,
        errors,
        check_id,
        status,
        reason,
        _default_json_safe(evidence),
    )


@dataclass(frozen=True)
class PreflightCheckState:
    """Ordered scientific and structural checks plus their derived status."""

    checks: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    convention_for_report: str | None
    finite_q_for_report: np.ndarray
    correction_value: Any
    correction_check_value: Any
    uncertainty_value: Any
    uncertainty_label: str
    uncertainty_provenance: dict[str, Any]
    status_color: str
    scientific_status: str
    exit_code: int


def build_preflight_checks(
    *,
    context: PreflightContext,
    read_state: PreflightReadState,
    hash_groups: Mapping[str, list[dict[str, Any]]],
    configured_external_roots: Sequence[Any] | None = None,
    correction_state: Any = None,
    uncertainty_state: Any = None,
    image_metadata: Sequence[Mapping[str, Any]] = (),
    output: Any = None,
    add_check: Callable[..., None] = _default_add_check,
    display_path: Callable[..., str] = _default_display_path,
    nested_mapping: Callable[..., Mapping[str, Any]] = _default_nested_mapping,
    first_value: Callable[..., Any] = _default_first_value,
    resolve_correction_value: Callable[
        ..., tuple[Any, Any]
    ] = _default_resolve_correction_value,
    correction_check: Callable[
        ..., tuple[str, str, dict[str, Any]]
    ] = _default_correction_check,
    state_value: Callable[..., Any] = _default_state_value,
    uncertainty_check: Callable[
        ..., tuple[str, str, dict[str, Any]]
    ] = _default_uncertainty_check,
    uncertainty_provenance: Callable[
        ..., dict[str, Any]
    ] = _default_uncertainty_provenance,
    header_geometry_check: Callable[
        ..., tuple[str, str, dict[str, Any]]
    ] = header_geometry_check,
    status_from_checks: Callable[..., tuple[str, str]] = status_from_checks,
) -> PreflightCheckState:
    """Build checks in the historical order without reading detector arrays."""

    package_root = context.package_root
    refs = context.refs
    selected_manifest = context.selected_manifest
    manifest_source = context.manifest_source
    manifest_quality = context.manifest_quality
    selected_convention = context.selected_convention
    context_value = context.context_value
    frames = read_state.frames
    first_shape = read_state.first_shape
    qmap = read_state.qmap
    mask_array = read_state.mask_array
    mask_source = read_state.mask_source
    external_mask = read_state.external_mask
    valid_mask = read_state.valid_mask
    reference_domain = read_state.reference_domain
    domain_records = read_state.domain_records
    convention_for_report = selected_convention if mask_array is not None else None
    finite_q_for_report = np.asarray(qmap["q"])[np.isfinite(qmap["q"])]

    check_builder = CheckBuilder()
    checks = check_builder.checks
    warnings = check_builder.warnings
    errors = check_builder.errors
    add_check(
        checks,
        warnings,
        errors,
        "package",
        "pass",
        "package directory is readable",
        {"path": package_root.as_posix()},
    )
    manifest_status = "pass"
    if manifest_quality is not None:
        manifest_evidence = dict(manifest_quality)
        manifest_reasons: list[str] = []
        if manifest_quality["duplicate_orders"]:
            manifest_status = "fail"
            manifest_reasons.append("manifest order values must be unique")
        if not manifest_quality["selector_unique"]:
            manifest_status = "fail"
            manifest_reasons.append(
                "manifest path/dataset/frame selector identities must be unique"
            )
        if not manifest_quality["time_finite"]:
            manifest_status = "fail"
            manifest_reasons.append("manifest time values must be finite numbers")
        if not manifest_quality["time_monotonic"]:
            manifest_status = "fail"
            manifest_reasons.append("manifest finite time values must be monotonic")
        if manifest_quality["missing_time_indices"] and manifest_status == "pass":
            manifest_status = "warn"
            manifest_reasons.append("manifest time is missing for one or more frames")
        manifest_reason = "; ".join(manifest_reasons) or "manifest structure is valid"
    else:
        manifest_evidence = {"provided": False, "frame_count": len(refs)}
        manifest_reason = "frames use natural filename order"
    add_check(
        checks,
        warnings,
        errors,
        "manifest",
        manifest_status,
        manifest_reason,
        {
            **manifest_evidence,
            "provided": selected_manifest is not None,
            "path": display_path(manifest_source, package_root)
            if manifest_source
            else None,
            "frame_count": len(refs),
        },
    )
    add_check(
        checks,
        warnings,
        errors,
        "frames",
        "pass",
        "all selected frames are readable and shape-consistent",
        {"count": len(frames), "shape": list(first_shape)},
    )
    intensity_unit = nested_mapping(context_value, "source_intensity").get("unit")
    if intensity_unit is None:
        add_check(
            checks,
            warnings,
            errors,
            "intensity_unit",
            "warn",
            "source intensity unit is not declared",
            {"unit": None},
        )
    else:
        add_check(
            checks,
            warnings,
            errors,
            "intensity_unit",
            "pass",
            "source intensity unit is declared",
            {"unit": str(intensity_unit)},
        )
    if qmap["source"] == "pixel-q":
        add_check(
            checks,
            warnings,
            errors,
            "q_unit",
            "warn",
            "pixel-q is uncalibrated and has no physical reciprocal-space unit",
            {"q_unit": "pixel-q"},
        )
        geometry_reason = "PONI not supplied; deterministic pixel-q fallback used"
        geometry_status = "warn"
    else:
        geometry_reason = "q map built from supplied geometry"
        geometry_status = "pass"
        if str(qmap.get("q_unit", "unknown")).casefold() in {"unknown", "", "none"}:
            geometry_status = "warn"
            geometry_reason = "q map unit is unknown"
    add_check(
        checks,
        warnings,
        errors,
        "geometry",
        geometry_status,
        geometry_reason,
        {"source": qmap["source"], "q_unit": qmap["q_unit"], "shape": list(first_shape)},
    )
    if mask_array is None:
        add_check(
            checks,
            warnings,
            errors,
            "mask",
            "pass",
            "no external mask supplied",
            {"provided": False},
        )
    else:
        invalid_count = (
            int(np.count_nonzero(external_mask))
            if external_mask is not None
            else int(np.count_nonzero(~valid_mask))
        )
        add_check(
            checks,
            warnings,
            errors,
            "mask",
            "pass",
            "mask shape and convention are valid",
            {
                "provided": True,
                "path": display_path(mask_source, package_root)
                if mask_source
                else "in-memory",
                "convention": convention_for_report,
                "invalid_count": invalid_count,
                "invalid_fraction": invalid_count / mask_array.size,
            },
        )
    add_check(
        checks,
        warnings,
        errors,
        "analysis_domain",
        "pass",
        "all frames share the unified fit domain contract",
        {"counts": reference_domain.counts, "per_frame": domain_records},
    )

    negative_count = sum(int(item["summary"].get("negative_count", 0)) for item in frames)
    if negative_count:
        add_check(
            checks,
            warnings,
            errors,
            "negative_intensity",
            "warn",
            "negative finite intensity values were observed; no values were rewritten",
            {"count": negative_count},
        )
    else:
        add_check(
            checks,
            warnings,
            errors,
            "negative_intensity",
            "pass",
            "no negative finite intensity values observed",
            {"count": 0},
        )

    correction_value, correction_check_value = resolve_correction_value(
        correction_state,
        context_value,
    )
    correction_status, correction_reason, correction_evidence = correction_check(
        correction_check_value
    )
    add_check(
        checks,
        warnings,
        errors,
        "correction_state",
        correction_status,
        correction_reason,
        correction_evidence,
    )
    uncertainty_value = state_value(uncertainty_state, context_value, "uncertainty_state")
    uncertainty_status, uncertainty_reason, uncertainty_evidence = uncertainty_check(
        uncertainty_value
    )
    provenance_info = uncertainty_provenance(
        image_metadata,
        package_root,
        hash_groups["uncertainty"],
        configured_external_roots,
    )
    uncertainty_evidence = {**uncertainty_evidence, **provenance_info}
    if isinstance(uncertainty_value, Mapping):
        uncertainty_label = str(
            first_value(
                uncertainty_value.get("status"),
                uncertainty_value.get("state"),
                uncertainty_value.get("uncertainty_status"),
                "unknown",
            )
        ).casefold()
    else:
        uncertainty_label = str(uncertainty_value or "unknown").casefold()
    complete_uncertainty = uncertainty_label in {"complete", "completed"}
    complete_inventory = (
        provenance_info["declared_file_count"] > 0
        and provenance_info["resolved_file_count"]
        == provenance_info["declared_file_count"]
        and provenance_info["dataset_inventory_status"] == "reference_schema_read"
        and bool(provenance_info["datasets"])
        and bool(provenance_info["units"])
    )
    if complete_uncertainty and not complete_inventory:
        uncertainty_status = "fail"
        uncertainty_reason = (
            "complete uncertainty was declared without a complete file/dataset/unit inventory"
        )
    add_check(
        checks,
        warnings,
        errors,
        "uncertainty_state",
        uncertainty_status,
        uncertainty_reason,
        uncertainty_evidence,
    )

    frame_structure = nested_mapping(context_value, "frame_structure")
    time_policy = str(first_value(frame_structure.get("time_zero_policy"), "")).casefold()
    provisional = "provisional" in time_policy or any(
        "provisional"
        in str(
            item.get("manifest_frame", {}).get("metadata", {}).get("time_basis", "")
        ).casefold()
        for item in frames
    )
    if provisional:
        add_check(
            checks,
            warnings,
            errors,
            "time_basis",
            "warn",
            "time basis is provisional",
            {"time_zero_policy": time_policy},
        )
    else:
        add_check(
            checks,
            warnings,
            errors,
            "time_basis",
            "pass",
            "no provisional time basis was declared",
            {"time_zero_policy": time_policy or None},
        )

    header_status, header_reason, header_evidence = header_geometry_check(
        image_metadata,
        qmap.get("metadata", {}),
        qmap["source"] == "poni",
    )
    add_check(
        checks,
        warnings,
        errors,
        "header_geometry_comparison",
        header_status,
        header_reason,
        header_evidence,
    )

    if output is None:
        add_check(
            checks,
            warnings,
            errors,
            "evidence_persistence",
            "warn",
            "output was not requested; JSON/NPZ evidence was not persisted",
            {"output": None, "arrays_persisted": False},
        )
    else:
        add_check(
            checks,
            warnings,
            errors,
            "evidence_persistence",
            "pass",
            "strict JSON and NPZ evidence targets were selected",
            {"output": str(output), "arrays_persisted": True},
        )

    status_color, scientific_status = status_from_checks(checks)
    structural_failure_ids = {
        "package",
        "manifest",
        "frames",
        "q_unit",
        "geometry",
        "mask",
        "analysis_domain",
    }
    structural_failure = any(
        item.get("status") == "fail" and item.get("id") in structural_failure_ids
        for item in checks
    )
    exit_code = 0 if status_color == "green" else (2 if structural_failure else 1)
    return PreflightCheckState(
        checks=checks,
        warnings=warnings,
        errors=errors,
        convention_for_report=convention_for_report,
        finite_q_for_report=finite_q_for_report,
        correction_value=correction_value,
        correction_check_value=correction_check_value,
        uncertainty_value=uncertainty_value,
        uncertainty_label=uncertainty_label,
        uncertainty_provenance=provenance_info,
        status_color=status_color,
        scientific_status=scientific_status,
        exit_code=exit_code,
    )


__all__ = [
    "CheckBuilder",
    "PreflightCheckState",
    "build_preflight_checks",
    "header_geometry_check",
    "header_geometry_fields",
    "status_from_checks",
]
