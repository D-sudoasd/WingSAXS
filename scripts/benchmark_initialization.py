"""Run the synthetic independent/forward/reverse initialization campaign."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from butterfly_saxs.benchmark_initialization import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
