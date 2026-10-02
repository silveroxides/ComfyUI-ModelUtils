"""Select repository tests from changed paths or named subsystem groups."""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPOSITORY_ROOT.parent.parent
MANIFEST_PATH = Path(__file__).with_name("test_groups.toml")
TEMP_ROOT = REPOSITORY_ROOT / ".pytest-tmp"


@dataclass(frozen=True)
class TestGroup:
    name: str
    paths: tuple[str, ...]
    python_tests: tuple[str, ...]


@dataclass
class Selection:
    groups: set[str] = field(default_factory=set)
    python_tests: set[str] = field(default_factory=set)
    reasons: dict[str, set[str]] = field(default_factory=dict)
    unmapped: set[str] = field(default_factory=set)


def _parse_groups(text: str) -> dict[str, TestGroup]:
    data = tomllib.loads(text)
    return {
        name: TestGroup(
            name=name,
            paths=tuple(values.get("paths", ())),
            python_tests=tuple(values.get("python_tests", ())),
        )
        for name, values in data.get("groups", {}).items()
    }


def load_groups(path: Path = MANIFEST_PATH) -> dict[str, TestGroup]:
    return _parse_groups(path.read_text(encoding="utf-8"))


def load_groups_from_revision(revision: str) -> dict[str, TestGroup]:
    manifest = MANIFEST_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{revision}:{manifest}"], cwd=REPOSITORY_ROOT,
        check=False, capture_output=True, text=True,
    )
    return _parse_groups(result.stdout) if result.returncode == 0 else {}


def git_lines(*args: str) -> set[str]:
    result = subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True,
    )
    return {
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines() if line.strip()
    }


def changed_paths(base: str | None = None) -> set[str]:
    paths = git_lines("diff", "--name-only", "--relative", "HEAD")
    if base:
        paths.update(git_lines("diff", "--name-only", "--relative", f"{base}...HEAD"))
    return paths


def is_production_source(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized == "__init__.py" or (
        normalized.startswith("nodes/") and normalized.endswith(".py")
    )


def _add_group(selection: Selection, group: TestGroup, reason: str) -> None:
    selection.groups.add(group.name)
    selection.python_tests.update(group.python_tests)
    selection.reasons.setdefault(group.name, set()).add(reason)


def _path_matches_group(path: str, group: TestGroup) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in group.paths)


def _existing_group(group: TestGroup) -> TestGroup:
    return TestGroup(
        name=group.name, paths=group.paths,
        python_tests=tuple(
            path for path in group.python_tests if (REPOSITORY_ROOT / path).is_file()
        ),
    )


def select_tests(
    paths: set[str], groups: dict[str, TestGroup],
    explicit_groups: tuple[str, ...] = (),
    historical_groups: tuple[dict[str, TestGroup], ...] = (),
) -> Selection:
    selection = Selection()
    for name in explicit_groups:
        if name not in groups:
            raise ValueError(f"Unknown test group: {name}")
        _add_group(selection, groups[name], "explicit selection")

    for raw_path in sorted(paths):
        path = raw_path.replace("\\", "/")
        if path.startswith("tests/test_") and path.endswith(".py"):
            if (REPOSITORY_ROOT / path).is_file():
                selection.python_tests.add(path)
                selection.reasons.setdefault("direct test", set()).add(path)
            continue

        matched = False
        for group in groups.values():
            if _path_matches_group(path, group):
                _add_group(selection, group, path)
                matched = True
        if not matched and not (REPOSITORY_ROOT / path).exists():
            for historical in historical_groups:
                for historical_group in historical.values():
                    if _path_matches_group(path, historical_group):
                        active = groups.get(historical_group.name)
                        _add_group(
                            selection, active or _existing_group(historical_group),
                            f"{path} (deleted)",
                        )
                        matched = True
        if not matched and is_production_source(path) and (REPOSITORY_ROOT / path).exists():
            selection.unmapped.add(path)
    return selection


def tracked_final_tests() -> set[str]:
    tracked = git_lines("ls-files", "tests/test_*.py")
    return {path for path in tracked if (REPOSITORY_ROOT / path).is_file()}


def print_selection(selection: Selection) -> None:
    for group in sorted(selection.groups):
        reasons = ", ".join(sorted(selection.reasons.get(group, ())))
        print(f"group {group}: {reasons}")
    for reason in sorted(selection.reasons.get("direct test", ())):
        print(f"direct test: {reason}")
    print("python tests:")
    for path in sorted(selection.python_tests):
        print(f"  {path}")


def run_selection(selection: Selection) -> int:
    TEMP_ROOT.mkdir(exist_ok=True)
    basetemp = (TEMP_ROOT / f"run-{uuid.uuid4().hex}").resolve()
    if TEMP_ROOT.resolve() not in basetemp.parents:
        raise RuntimeError(f"Unsafe pytest temporary path: {basetemp}")
    try:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")))
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--import-mode=importlib",
                "-o",
                "addopts=",
                "--capture=sys",
                "--basetemp",
                str(basetemp),
                *[
                    str((REPOSITORY_ROOT / path).resolve())
                    for path in sorted(selection.python_tests)
                ],
            ],
            cwd=COMFYUI_ROOT,
            env=environment,
            check=False,
        )
        return result.returncode
    finally:
        if basetemp.exists():
            shutil.rmtree(basetemp)
        if TEMP_ROOT.exists() and not any(TEMP_ROOT.iterdir()):
            TEMP_ROOT.rmdir()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true", help="run every tracked test")
    mode.add_argument("--list-groups", action="store_true", help="list configured groups")
    result.add_argument("--changed", action="store_true", help="select changed paths (default)")
    result.add_argument("--base", help="include committed changes since BASE")
    result.add_argument("--group", action="append", default=[], help="add a named group")
    result.add_argument("--dry-run", action="store_true", help="explain without executing")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    groups = load_groups()
    if args.list_groups:
        for name in sorted(groups):
            print(name)
        return 0
    if args.final:
        selection = Selection(
            groups={"final"}, python_tests=tracked_final_tests(),
            reasons={"final": {"every tracked test"}},
        )
    else:
        paths = changed_paths(args.base) if args.changed or args.base or not args.group else set()
        revisions = ["HEAD", *([args.base] if args.base else [])]
        historical = tuple(load_groups_from_revision(rev) for rev in revisions) if paths else ()
        selection = select_tests(paths, groups, tuple(args.group), historical)

    if selection.unmapped:
        print("Unmapped production source files:", file=sys.stderr)
        for path in sorted(selection.unmapped):
            print(f"  {path}", file=sys.stderr)
        print("Update tests/test_groups.toml or use --final.", file=sys.stderr)
        return 2
    if args.dry_run:
        print_selection(selection)
        return 0
    if not selection.python_tests:
        print("No tests selected.")
        return 0
    return run_selection(selection)


if __name__ == "__main__":
    raise SystemExit(main())
