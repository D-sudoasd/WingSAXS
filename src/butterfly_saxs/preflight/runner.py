"""Read-only package preflight orchestration for two-dimensional SAXS data.

The preflight layer deliberately stops before ridge extraction or any fitting.
It resolves the same frame and selector contracts used by :mod:`batch` and
:mod:`io`, then asks :func:`validation.build_analysis_domain` for the pixel
population that a later analysis would be allowed to use.
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..validation import (
    RESULT_SCHEMA_VERSION,
    AnalysisDomainError,
    build_analysis_domain,
)
from .artifacts import PreflightArtifactWriter
from .checks import build_preflight_checks
from .context import resolve_inputs
from .geometry import build_qmap, load_mask
from .hash import PreflightError
from .io import frame_record, read_one_frame, read_preflight_inputs
from .report import PreflightReportHelpers, build_preflight_report_from_stages

PREFLIGHT_SCHEMA_VERSION = "lamellarsaxs2d.preflight.v1"


def run_preflight(
    package: str | os.PathLike[str] | Path,
    *,
    manifest: Any = None,
    poni: Any = None,
    mask: Any = None,
    context: Any = None,
    image_glob: str | os.PathLike[str] | None = None,
    frame: int | None = None,
    dataset: str | None = None,
    mask_frame: int | None = None,
    mask_dataset: str | None = None,
    q_window: Any = None,
    mask_convention: str | None = None,
    correction_state: Any = None,
    uncertainty_state: Any = None,
    output: str | os.PathLike[str] | Path | None = None,
    force: bool = False,
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> dict[str, Any]:
    """Run a read-only preflight and optionally write its three artifacts."""

    from .. import preflight as preflight_pkg

    active_loader = preflight_pkg.load_image
    artifact_writer = PreflightArtifactWriter()
    resolved_inputs = resolve_inputs(
        package,
        manifest=manifest,
        poni=poni,
        mask=mask,
        context=context,
        image_glob=image_glob,
        frame=frame,
        dataset=dataset,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
        q_window=q_window,
        mask_convention=mask_convention,
        output=output,
        external_roots=external_roots,
        error_type=PreflightError,
    )
    preflight_context = resolved_inputs.context
    hash_groups = resolved_inputs.hash_groups

    read_state = read_preflight_inputs(
        preflight_context,
        frame=frame,
        dataset=dataset,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
        hash_groups=hash_groups,
        read_one_frame=lambda *args, **kwargs: read_one_frame(
            *args, load_image=active_loader, **kwargs
        ),
        frame_record=frame_record,
        load_mask=lambda *args, **kwargs: load_mask(
            *args, load_image=active_loader, **kwargs
        ),
        build_qmap=build_qmap,
        build_analysis_domain=build_analysis_domain,
        preflight_error=PreflightError,
        analysis_domain_error=AnalysisDomainError,
    )

    check_state = build_preflight_checks(
        context=preflight_context,
        read_state=read_state,
        hash_groups=hash_groups,
        configured_external_roots=preflight_context.external_roots,
        correction_state=correction_state,
        uncertainty_state=uncertainty_state,
        image_metadata=read_state.image_metadata,
        output=output,
    )
    output_paths, overwritten_paths = artifact_writer.prepare(output, force)
    package_src_root = Path(__file__).resolve().parent.parent
    report_result = build_preflight_report_from_stages(
        context=preflight_context,
        read_state=read_state,
        check_state=check_state,
        hash_groups=hash_groups,
        frame=frame,
        dataset=dataset,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
        correction_state=correction_state,
        uncertainty_state=uncertainty_state,
        output_paths=output_paths,
        overwritten_paths=overwritten_paths,
        force=force,
        schema_source=package_src_root / "validation.py",
        code_source=Path(__file__).resolve(),
        result_schema_version=RESULT_SCHEMA_VERSION,
        preflight_schema_version=PREFLIGHT_SCHEMA_VERSION,
        helpers=PreflightReportHelpers(),
    )
    safe_report = report_result.report
    q_keys = report_result.q_keys

    if output_paths is not None:
        q_arrays = {
            q_keys["qx"]: np.asarray(read_state.qmap["qx"], dtype=float),
            q_keys["qy"]: np.asarray(read_state.qmap["qy"], dtype=float),
            q_keys["q"]: np.asarray(read_state.qmap["q"], dtype=float),
            q_keys["chi"]: np.arctan2(
                np.asarray(read_state.qmap["qy"], dtype=float),
                np.asarray(read_state.qmap["qx"], dtype=float),
            ),
        }
        artifact_writer.write(
            output_paths,
            safe_report,
            {
                **q_arrays,
                "finite_mask": np.asarray(
                    read_state.reference_domain.finite_mask, dtype=bool
                ),
                "detector_valid_mask": np.asarray(
                    read_state.reference_domain.detector_valid_mask, dtype=bool
                ),
                "external_valid_mask": np.asarray(
                    read_state.reference_domain.external_valid_mask, dtype=bool
                ),
                "q_window_mask": np.asarray(
                    read_state.reference_domain.q_window_mask, dtype=bool
                ),
                "roi_exclusion_mask": np.asarray(
                    read_state.reference_domain.roi_exclusion_mask, dtype=bool
                ),
                "weight_valid_mask": np.asarray(
                    read_state.reference_domain.weight_valid_mask, dtype=bool
                ),
                "fit_valid_mask": np.asarray(
                    read_state.reference_domain.fit_valid_mask, dtype=bool
                ),
                "sampled_valid_mask": np.asarray(
                    read_state.reference_domain.sampled_valid_mask, dtype=bool
                ),
                "frame_id": np.asarray(preflight_context.refs[0].id),
            },
        )
    return safe_report


__all__ = ["PREFLIGHT_SCHEMA_VERSION", "PreflightError", "run_preflight"]
