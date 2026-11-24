"""Environment diagnostics for first-run and support workflows.

The module intentionally depends only on the Python standard library so it can
explain a broken scientific/UI environment instead of failing while importing
the package's optional stacks.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from . import __version__

SUPPORTED_MIN = (3, 11)
SUPPORTED_MAX_EXCLUSIVE = (3, 14)

_CORE_MODULES: tuple[tuple[str, str, str], ...] = (
    ("NumPy", "numpy", "numpy"),
    ("SciPy", "scipy", "scipy"),
    ("Matplotlib", "matplotlib", "matplotlib"),
    ("FabIO", "fabio", "fabio"),
    ("pyFAI", "pyFAI", "pyFAI"),
    ("tifffile", "tifffile", "tifffile"),
    ("PyYAML", "yaml", "PyYAML"),
)
_UI_MODULES: tuple[tuple[str, str, str], ...] = (
    ("PySide6", "PySide6.QtWidgets", "PySide6"),
    ("pyqtgraph", "pyqtgraph", "pyqtgraph"),
)
_OPTIONAL_MODULES: tuple[tuple[str, str, str], ...] = (
    ("h5py (HDF5)", "h5py", "h5py"),
)


@dataclass(frozen=True)
class DiagnosticCheck:
    """One machine-readable environment check."""

    name: str
    category: str
    required: bool
    ok: bool
    detail: str
    remediation: str = ""


def _version_text(
    distribution: str,
    *,
    version_getter: Callable[[str], str],
) -> str:
    try:
        return str(version_getter(distribution))
    except importlib.metadata.PackageNotFoundError:
        return "installed (version metadata unavailable)"
    except Exception as exc:  # pragma: no cover - unusual metadata backends
        return f"installed (version lookup failed: {type(exc).__name__})"


def _is_this_project(root: Path) -> bool:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        with pyproject.open("rb") as handle:
            parsed = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    project = parsed.get("project", {}) if isinstance(parsed, dict) else {}
    return isinstance(project, dict) and project.get("name") == "butterfly-saxs"


def _project_root(cwd: Path) -> Path:
    candidate = cwd.expanduser().resolve(strict=False)
    if candidate.is_file():
        candidate = candidate.parent
    for root in (candidate, *candidate.parents):
        if _is_this_project(root):
            return root
    module_file = Path(__file__).resolve()
    for root in (module_file.parent, *module_file.parents):
        if _is_this_project(root):
            return root
    return candidate


def _declared_specifiers(root: Path, *, require_ui: bool) -> dict[str, str]:
    """Read the project's real dependency declarations without importing it."""

    pyproject = root / "pyproject.toml"
    document: dict[str, Any] = {}
    if pyproject.is_file():
        try:
            with pyproject.open("rb") as handle:
                parsed = tomllib.load(handle)
            if isinstance(parsed, dict):
                document = parsed
        except (OSError, tomllib.TOMLDecodeError):
            document = {}
    project = document.get("project", {}) if isinstance(document, dict) else {}
    dependencies: list[Any] = []
    if isinstance(project, dict):
        raw = project.get("dependencies", [])
        if isinstance(raw, list):
            dependencies.extend(raw)
        optional = project.get("optional-dependencies", {})
        if require_ui and isinstance(optional, dict) and isinstance(optional.get("ui"), list):
            dependencies.extend(optional["ui"])
    specs: dict[str, str] = {}
    requirement_pattern = re.compile(
        r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*([^;]*)"
    )
    for raw in dependencies:
        if not isinstance(raw, str):
            continue
        match = requirement_pattern.match(raw)
        if match is None:
            continue
        name, specifier = match.groups()
        specs[name.casefold().replace("_", "-")] = specifier.strip()
    if not specs:
        try:
            installed_requirements = importlib.metadata.requires("butterfly-saxs") or []
        except importlib.metadata.PackageNotFoundError:
            installed_requirements = []
        extra_pattern = re.compile(r"""\bextra\s*==\s*['"]([^'"]+)['"]""")
        for raw in installed_requirements:
            if not isinstance(raw, str):
                continue
            base, _sep, marker = raw.partition(";")
            extra_match = extra_pattern.search(marker) if marker else None
            if extra_match is not None:
                # Ignore optional extras unless the caller explicitly requested
                # that extra (currently only ``ui``).
                if not (require_ui and extra_match.group(1).casefold() == "ui"):
                    continue
            match = requirement_pattern.match(base)
            if match is None:
                continue
            name, specifier = match.groups()
            key = name.casefold().replace("_", "-")
            if extra_match is None or key not in specs:
                specs[key] = specifier.strip()
    return specs


_PEP440_RE = re.compile(
    r"^\s*(?:(?P<epoch>\d+)!)?"
    r"(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?P<pre_l>a|alpha|b|beta|rc|c|pre|preview)[-_.]?(?P<pre_n>\d*))?"
    r"(?:[-_.]?(?:post|rev|r)[-_.]?(?P<post_n>\d*))?"
    r"(?:[-_.]?dev[-_.]?(?P<dev_n>\d*))?"
    r"(?:\+[^\s]+)?\s*$",
    re.IGNORECASE,
)


def _parse_fallback_version(
    value: str,
) -> tuple[int, tuple[int, ...], int, int, int, int] | None:
    """Parse a PEP 440 version into a comparable tuple with pre/post/dev ordering."""

    match = _PEP440_RE.match(str(value))
    if match is None:
        return None
    epoch = int(match.group("epoch") or 0)
    release = tuple(int(part) for part in match.group("release").split("."))
    pre_label = (match.group("pre_l") or "").lower()
    post_raw = match.group("post_n")
    dev_raw = match.group("dev_n")
    has_post = re.search(r"(?:post|rev|r)\d*", str(value), re.IGNORECASE) is not None and post_raw is not None
    has_dev = re.search(r"dev\d*", str(value), re.IGNORECASE) is not None and dev_raw is not None
    pre_num = int(match.group("pre_n") or 0) if pre_label else 0
    post_num = int(post_raw or 0) if has_post else -1
    dev_num = int(dev_raw or 0) if has_dev else -1
    if pre_label in {"a", "alpha"}:
        stage = -3
    elif pre_label in {"b", "beta"}:
        stage = -2
    elif pre_label in {"rc", "c", "pre", "preview"}:
        stage = -1
    elif has_dev and not has_post:
        stage = -4
    elif has_post:
        stage = 1
    else:
        stage = 0
    return epoch, release, stage, pre_num, post_num, dev_num


def _compare_fallback_versions(
    left: tuple[int, tuple[int, ...], int, int, int, int],
    right: tuple[int, tuple[int, ...], int, int, int, int],
) -> int:
    if left[0] != right[0]:
        return -1 if left[0] < right[0] else 1
    width = max(len(left[1]), len(right[1]))
    l_rel = left[1] + (0,) * (width - len(left[1]))
    r_rel = right[1] + (0,) * (width - len(right[1]))
    if l_rel != r_rel:
        return -1 if l_rel < r_rel else 1
    l_tail = left[2:]
    r_tail = right[2:]
    if l_tail != r_tail:
        return -1 if l_tail < r_tail else 1
    return 0


def _version_key(value: str) -> tuple[int, ...] | None:
    parsed = _parse_fallback_version(value)
    if parsed is not None:
        return parsed[1]
    parts = re.findall(r"\d+", str(value))
    if not parts:
        return None
    return tuple(int(part) for part in parts[:4])


def _version_satisfies(version: str, specifier: str) -> bool:
    if not specifier:
        return True
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        SpecifierSet = None  # type: ignore[assignment]
        Version = None  # type: ignore[assignment]
        InvalidSpecifier = ValueError  # type: ignore[assignment]
        InvalidVersion = ValueError  # type: ignore[assignment]

    if SpecifierSet is not None and Version is not None:
        try:
            parsed_version = Version(str(version))
            spec_set = SpecifierSet(str(specifier))
        except (InvalidVersion, InvalidSpecifier):
            pass
        else:
            return bool(spec_set.contains(parsed_version, prereleases=None))

    actual = _parse_fallback_version(version)
    if actual is None:
        return False
    for raw_clause in specifier.split(","):
        clause = raw_clause.strip()
        if not clause:
            continue
        match = re.match(r"^(~=|==|!=|>=|<=|>|<)\s*([0-9][^,\s]*)$", clause)
        if match is None:
            return False
        operator, target_text = match.groups()
        target = _parse_fallback_version(target_text)
        if target is None:
            return False
        cmp = _compare_fallback_versions(actual, target)
        if operator == "==" and cmp != 0:
            return False
        if operator == "!=" and cmp == 0:
            return False
        if operator == ">=" and cmp < 0:
            return False
        if operator == "<=" and cmp > 0:
            return False
        if operator == ">" and cmp <= 0:
            return False
        if operator == "<" and cmp >= 0:
            return False
        if operator == "~=":
            prefix_len = max(1, len(target[1]) - 1)
            padded_actual = actual[1] + (0,) * max(0, prefix_len - len(actual[1]))
            if cmp < 0 or actual[0] != target[0] or padded_actual[:prefix_len] != target[1][:prefix_len]:
                return False
    return True


def _dependency_check(
    display_name: str,
    module_name: str,
    distribution: str,
    *,
    category: str,
    required: bool,
    importer: Callable[[str], Any],
    version_getter: Callable[[str], str],
    specifier: str = "",
    cwd: Path | None = None,
) -> DiagnosticCheck:
    try:
        importer(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostics must report import/DLL errors
        return DiagnosticCheck(
            name=display_name,
            category=category,
            required=required,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            remediation=_repair_command(cwd),
        )
    try:
        version = str(version_getter(distribution))
    except Exception as exc:  # noqa: BLE001 - metadata failure is not readiness
        return DiagnosticCheck(
            name=display_name,
            category=category,
            required=required,
            ok=False,
            detail=f"version metadata unavailable: {type(exc).__name__}",
            remediation=_repair_command(cwd),
        )
    if specifier and not _version_satisfies(version, specifier):
        return DiagnosticCheck(
            name=display_name,
            category=category,
            required=required,
            ok=False,
            detail=f"{version} does not satisfy {specifier}",
            remediation=_repair_command(cwd),
        )
    return DiagnosticCheck(
        name=display_name,
        category=category,
        required=required,
        ok=True,
        detail=f"{version}{(' ' + specifier) if specifier else ''}",
    )


def _repair_command(cwd: Path | None = None) -> str:
    root = _project_root(Path.cwd() if cwd is None else Path(cwd))
    constraint = root / "constraints" / "validation-py311-313.txt"
    if (root / "pyproject.toml").exists():
        if constraint.exists():
            return (
                f'"{sys.executable}" -m pip install '
                '-c constraints/validation-py311-313.txt -e ".[all]"'
            )
        return f'"{sys.executable}" -m pip install -e ".[all]"'
    return (
        "Run from the WingSAXS project directory, then install "
        'with: python -m pip install -e ".[all]"'
    )


def collect_diagnostics(
    *,
    require_ui: bool = False,
    importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    version_info: Sequence[int] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Collect deterministic environment checks without starting Qt."""

    root = _project_root(Path.cwd() if cwd is None else Path(cwd))
    declared = _declared_specifiers(root, require_ui=bool(require_ui))
    core_declared = {
        distribution.casefold().replace("_", "-")
        for _display, _module, distribution in _CORE_MODULES
    }
    ui_declared = {
        distribution.casefold().replace("_", "-")
        for _display, _module, distribution in _UI_MODULES
    }
    declarations_complete = core_declared <= set(declared)
    version = tuple(version_info or sys.version_info[:3])
    python_ok = SUPPORTED_MIN <= version[:2] < SUPPORTED_MAX_EXCLUSIVE
    checks: list[DiagnosticCheck] = [
        DiagnosticCheck(
            name="Python",
            category="runtime",
            required=True,
            ok=python_ok,
            detail=(
                f"{version[0]}.{version[1]}.{version[2]} "
                f"({sys.executable})"
            ),
            remediation=(
                ""
                if python_ok
                else "Use Python 3.11, 3.12, or 3.13; Python 3.14+ is unsupported."
            ),
        )
    ]
    for display_name, module_name, distribution in _CORE_MODULES:
        checks.append(
            _dependency_check(
                display_name,
                module_name,
                distribution,
                category="core",
                required=True,
                importer=importer,
                version_getter=version_getter,
                specifier=(
                    declared.get(distribution.casefold().replace("_", "-"), "")
                    if declarations_complete
                    else ""
                ),
                cwd=root,
            )
        )
    for display_name, module_name, distribution in _UI_MODULES:
        checks.append(
            _dependency_check(
                display_name,
                module_name,
                distribution,
                category="ui",
                required=bool(require_ui),
                importer=importer,
                version_getter=version_getter,
                specifier=(
                    declared.get(distribution.casefold().replace("_", "-"), "")
                    if declarations_complete and ui_declared <= set(declared)
                    else ""
                ),
                cwd=root,
            )
        )
    for display_name, module_name, distribution in _OPTIONAL_MODULES:
        checks.append(
            _dependency_check(
                display_name,
                module_name,
                distribution,
                category="optional",
                required=False,
                importer=importer,
                version_getter=version_getter,
                specifier=declared.get(distribution.casefold().replace("_", "-"), ""),
                cwd=root,
            )
        )

    required_failures = [check for check in checks if check.required and not check.ok]
    return {
        "schema_version": 1,
        "ready": not required_failures,
        "require_ui": bool(require_ui),
        "application": "WingSAXS",
        "distribution": "butterfly-saxs",
        "version": str(__version__),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": python_ok,
            "supported_range": ">=3.11,<3.14",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "working_directory": str(root.resolve()),
        "declared_requirements": declared,
        "requirements_declaration_complete": bool(
            declarations_complete and (not require_ui or ui_declared <= set(declared))
        ),
        "checks": [asdict(check) for check in checks],
        "required_failures": [check.name for check in required_failures],
        "repair_command": _repair_command(root),
    }


def _format_text(report: dict[str, Any]) -> str:
    status = "READY" if report["ready"] else "NOT READY"
    lines = [
        "WingSAXS environment diagnostics / 环境诊断",
        f"Status / 状态: {status}",
        f"Python support: {report['python']['supported_range']}",
        "",
    ]
    for check in report["checks"]:
        if check["ok"]:
            marker = "OK"
        elif check["required"]:
            marker = "FAIL"
        else:
            marker = "OPTIONAL-MISSING"
        requirement = "required" if check["required"] else "optional"
        lines.append(
            f"[{marker}] {check['name']} ({check['category']}, {requirement}): "
            f"{check['detail']}"
        )
    if not report["ready"]:
        lines.extend(
            [
                "",
                "Repair / 修复建议:",
                report["repair_command"],
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsaxs-doctor",
        description=(
            "Check the supported Python version and WingSAXS dependencies "
            "without opening the GUI."
        ),
    )
    parser.add_argument(
        "--require-ui",
        action="store_true",
        help="Treat PySide6 and pyqtgraph as required.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a strict machine-readable JSON report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print nothing; communicate readiness through the exit code.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(list(argv) if argv is not None else None)
    report = collect_diagnostics(require_ui=options.require_ui)
    if not options.quiet:
        if options.json:
            print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        else:
            print(_format_text(report))
    return 0 if report["ready"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through console script
    raise SystemExit(main())
