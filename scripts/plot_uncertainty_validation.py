"""Plot saved empirical intervals without rerunning or changing the analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destinations = [args.output.with_suffix(suffix) for suffix in (".png", ".svg")]
    if args.report.resolve() in [path.resolve() for path in destinations]:
        parser.error("plot output must not overwrite the input report")
    if not args.force and any(path.exists() for path in destinations):
        parser.error("plot output exists; choose a new output or pass --force")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["trials"]
    if not rows:
        parser.error("report has no trial records")
    figure, axes = plt.subplots(1, 3, figsize=(10, 4.8), sharey=True)
    for axis, parameter, label in zip(axes, ("a", "b", "axis_ratio"),
                                      ("Semi-major axis a", "Semi-minor axis b", "Axis ratio b/a"),
                                      strict=True):
        for index, row in enumerate(rows, start=1):
            truth = row.get("truth", {}).get(parameter)
            candidate = row.get("candidate", {}).get(parameter)
            interval = row.get("intervals", {}).get(parameter)
            if truth is None or not np.isfinite(truth) or truth == 0:
                axis.text(0, index, "reference unavailable", fontsize=7, ha="center")
                continue
            if interval is not None and len(interval) == 2 and all(
                value is not None and np.isfinite(value) for value in interval
            ):
                low, high = (100 * (value / truth - 1) for value in interval)
                covered = interval[0] <= truth <= interval[1]
                axis.plot([low, high], [index, index], color="#19777C" if covered else "#B84332",
                          linewidth=2, solid_capstyle="round")
            else:
                axis.text(0, index + .22, "interval unavailable", fontsize=6.5, ha="center")
            if candidate is not None and np.isfinite(candidate):
                axis.plot(100 * (candidate / truth - 1), index, "o", markersize=3.5,
                          color="#182F45", zorder=3)
        axis.axvline(0, color="0.4", linestyle="--", linewidth=.9)
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("Difference from reference (%)", fontsize=9)
        axis.grid(axis="x", color="0.9", linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("Independent noisy-image trial")
    axes[0].set_yticks(range(1, len(rows) + 1))
    axes[0].set_ylim(len(rows) + .6, .4)
    resamples = report["context"]["resamples_per_trial"]
    figure.suptitle(f"Synthetic interval coverage: {len(rows)} images, {resamples} resamples per image",
                   fontsize=12, y=.98)
    figure.text(.5, .025, "Dots: fitted estimates. Lines: central empirical 95% intervals. "
                "Dashed line: generator reference.\n"
                "This development run does not establish calibrated 95% confidence intervals.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .1, 1, .94))
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"outputs": [str(path) for path in destinations]}, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
