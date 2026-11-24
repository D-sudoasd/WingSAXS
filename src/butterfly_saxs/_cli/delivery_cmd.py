"""Preflight, delivery packaging, and delivery binding verification handlers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

from .common import print_json


def handle_preflight(args: argparse.Namespace) -> int:
    from ..service import ButterflyAnalysisService

    report = ButterflyAnalysisService().preflight(
        args.package,
        manifest=args.manifest,
        poni=args.poni,
        mask=args.mask,
        context=args.context,
        image_glob=args.image_glob,
        frame=args.frame,
        dataset=args.dataset,
        mask_frame=args.mask_frame,
        mask_dataset=args.mask_dataset,
        q_window=args.q_window,
        mask_convention=args.mask_convention,
        correction_state=args.correction_state,
        uncertainty_state=args.uncertainty_state,
        output=args.output,
        force=args.force,
    )
    print_json(report)
    status = report.get("status")
    if isinstance(status, Mapping):
        return int(status["exit_code"])
    return 0 if status == "green" else 1


def handle_package(args: argparse.Namespace) -> int:
    from ..delivery import package_batch

    report = package_batch(
        args.output_dir,
        archive=not args.no_archive,
        resume=args.resume,
        force=args.force,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print_json(report)
    return int(report["exit_code"])


def handle_verify_delivery(args: argparse.Namespace) -> int:
    from ..delivery.bindings import verify_delivery_bindings

    report = verify_delivery_bindings(
        Path(args.receipt),
        root=Path(args.root) if args.root else None,
    )
    print_json(report)
    return int(report["exit_code"])
