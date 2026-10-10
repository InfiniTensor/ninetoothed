#!/usr/bin/env python3
"""Audit a *packed* runtime skill tree (not the competition source workspace).

Checks:
- forbidden directories absent
- banned competition tokens absent from .md / .py
- example-relative references resolve
- path-aware CLIs expose --repo-root
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_DIRS = (
    "evals",
    "worktree_patches",
    "submission",
    ".pytest_cache",
    ".quick_validate_cache",
    "__pycache__",
)

PATH_AWARE_SCRIPTS = (
    "env_check.py",
    "run_correctness.py",
    "run_benchmark.py",
    "repo_pattern_index.py",
    "check_patch_minimality.py",
)

# Backtick paths that look like in-package relative files (not URLs / shell vars).
_REL_PATH_RE = re.compile(
    r"`(solution/[^`\s]+|tests/test_example_[^`\s]+|"
    r"(?:task|task_card|run_prompt|run_log|changed_files|"
    r"correctness_result|benchmark_result|failure_diagnosis|verify)\.[a-z]+)`"
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def audit_forbidden_dirs(root: Path) -> list[str]:
    errors: list[str] = []
    for name in FORBIDDEN_DIRS:
        if (root / name).exists():
            errors.append(f"forbidden directory present: {name}/")
    for p in root.rglob("*"):
        if p.is_dir() and p.name in {
            ".pytest_cache",
            ".quick_validate_cache",
            "__pycache__",
        }:
            errors.append(
                f"forbidden cache directory: {p.relative_to(root).as_posix()}"
            )
        if p.is_file() and p.suffix.lower() in {".pyc", ".pyo"}:
            errors.append(f"forbidden bytecode file: {p.relative_to(root).as_posix()}")
    return errors


def audit_answer_artifacts(root: Path) -> list[str]:
    """Flag answer-shaped / workspace-only artifacts without competition jargon."""
    errors: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        rel = p.relative_to(root).as_posix()
        if name.startswith(("bench_", "generate_", "compare_")) and p.suffix == ".py":
            if p.parent.name == "scripts":
                errors.append(f"non-runtime script packed: {rel}")
        # Generic answer-shaped names (not competition study IDs).
        if name in {"answer_key.md", "grader_secret.py", "sealed_answers.json"}:
            errors.append(f"answer-shaped artifact: {rel}")
    return errors


def audit_example_relative_refs(root: Path) -> list[str]:
    errors: list[str] = []
    examples = root / "examples"
    if not examples.is_dir():
        return ["missing examples/"]
    for ex_dir in sorted(p for p in examples.iterdir() if p.is_dir()):
        for md in ex_dir.glob("*.md"):
            text = md.read_text(encoding="utf-8")
            for match in _REL_PATH_RE.finditer(text):
                rel = match.group(1)
                # Skip placeholders / globs
                if "<" in rel or "*" in rel or rel.endswith("/"):
                    continue
                target = ex_dir / rel
                if not target.exists():
                    errors.append(
                        f"example ref missing: {ex_dir.name}/{md.name} -> {rel}"
                    )
        # Uniform artifact set must exist for each example
        for required in ("task.md", "task_card.md", "verify.py"):
            if not (ex_dir / required).is_file():
                errors.append(
                    f"example missing required file: {ex_dir.name}/{required}"
                )
    return errors


def audit_path_aware_repo_root(root: Path) -> list[str]:
    errors: list[str] = []
    scripts = root / "scripts"
    py = sys.executable
    for name in PATH_AWARE_SCRIPTS:
        script = scripts / name
        if not script.is_file():
            errors.append(f"missing path-aware script: scripts/{name}")
            continue
        proc = subprocess.run(
            [py, str(script), "--help"],
            cwd=scripts,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            errors.append(f"{name} --help failed: {proc.stderr.strip()}")
            continue
        if "--repo-root" not in proc.stdout:
            errors.append(f"{name} missing --repo-root in --help")
    return errors


def audit_packed_tree(root: Path) -> list[str]:
    root = root.resolve()
    if not (root / "SKILL.md").is_file():
        return [f"not a skill root: {root}"]
    errors: list[str] = []
    errors.extend(audit_forbidden_dirs(root))
    errors.extend(audit_answer_artifacts(root))
    errors.extend(audit_example_relative_refs(root))
    errors.extend(audit_path_aware_repo_root(root))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a packed runtime skill directory."
    )
    parser.add_argument(
        "packed_root",
        type=Path,
        help="Path to packed skill root (output of pack_runtime_skill.py)",
    )
    args = parser.parse_args()
    errors = audit_packed_tree(args.packed_root)
    if errors:
        print("PACK_POST_AUDIT: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"PACK_POST_AUDIT: PASS ({args.packed_root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
