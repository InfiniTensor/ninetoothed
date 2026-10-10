#!/usr/bin/env python3
"""Fast offline validation for the runtime skill package (no self-test task IDs)."""

from __future__ import annotations

import argparse
import sys

from _paths import (
    add_skill_root_arg,
    resolve_skill_root,
)
from _paths import (
    skill_root as detect_skill_root,
)

REQUIRED_REFERENCES = [
    "00_repo_map.md",
    "01_ninetoothed_concepts.md",
    "02_arrangement_application_patterns.md",
    "03_elementwise_broadcast_patterns.md",
    "04_reduction_block_patterns.md",
    "05_layout_stride_offset_patterns.md",
    "06_correctness_testing_patterns.md",
    "07_benchmark_patterns.md",
    "08_generated_source_aot_debugging.md",
    "09_failure_diagnosis_playbook.md",
    "10_patch_minimality_checklist.md",
    "11_unsupported_cases.md",
]

# Formal runtime CLIs + private helpers (_paths, _task_spec).
REQUIRED_SCRIPTS = [
    "env_check.py",
    "make_task_card.py",
    "run_correctness.py",
    "run_benchmark.py",
    "repo_pattern_index.py",
    "check_patch_minimality.py",
    "score_task.py",
    "gate_eval.py",
    "summarize_run.py",
    "quick_validate.py",
    "audit_packed_skill.py",
    "_paths.py",
    "_task_spec.py",
]

REQUIRED_CAPABILITY_REFS = {
    "elementwise_broadcast": "03_elementwise_broadcast_patterns.md",
    "reduction_block": "04_reduction_block_patterns.md",
    "layout_sensitive": "05_layout_stride_offset_patterns.md",
    "performance_diagnosis": "07_benchmark_patterns.md",
}


def _fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ninetoothed-op-dev-skill package structure (runtime)."
    )
    add_skill_root_arg(parser)
    args = parser.parse_args()

    try:
        root = resolve_skill_root(args) if args.skill_root else detect_skill_root()
    except FileNotFoundError as exc:
        return _fail(str(exc))

    skill_md = root / "SKILL.md"
    if not skill_md.is_file():
        return _fail("missing SKILL.md")

    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return _fail("SKILL.md missing YAML frontmatter")
    if "name: ninetoothed-op-dev-skill" not in text:
        return _fail("SKILL.md missing skill name in frontmatter")

    # Packed installs omit the packer; competition-only trees must be absent there.
    is_packed = not (root / "scripts" / "pack_runtime_skill.py").is_file()

    if "## Execution decision tree (MANDATORY)" not in text:
        return _fail("SKILL.md missing mandatory Execution decision tree")
    for token in (
        "### D1 — Emit task card first",
        "### D3 — `rg` before code",
        "### D5 — Layout branch",
        "### D7 — Failure loop",
        "### D8 — Performance branch",
        "assert not inp.is_contiguous()",
    ):
        if token not in text:
            return _fail(f"SKILL.md missing decision-tree constraint: {token}")

    ref = root / "references"
    for name in REQUIRED_REFERENCES:
        if not (ref / name).is_file():
            return _fail(f"missing reference: {name}")

    for family, name in REQUIRED_CAPABILITY_REFS.items():
        if not (ref / name).is_file():
            return _fail(f"missing capability reference for {family}: {name}")

    scripts = root / "scripts"
    for name in REQUIRED_SCRIPTS:
        if not (scripts / name).is_file():
            return _fail(f"missing script: {name}")

    forbidden_runtime = [
        root / ".quick_validate_cache",
        root / "submission",
        root / "worktree_patches",
        root / "evals",
    ]
    if is_packed:
        # Packed trees must not contain competition-only corpora.
        pass
    for path in forbidden_runtime:
        if path.exists():
            # Source workspace may still have evals/; packed installs must not.
            if path.name == "evals" and not is_packed:
                continue
            return _fail(
                f"runtime skill must not contain `{path.relative_to(root)}` "
                "(exclude workspace-only / answer artifacts from the package)"
            )

    # Generic cache / bytecode artifacts must not ship in runtime trees.
    for p in root.rglob("*"):
        if p.is_dir() and p.name in {
            "__pycache__",
            ".pytest_cache",
            ".quick_validate_cache",
        }:
            return _fail(f"forbidden cache directory: {p.relative_to(root)}")
        if p.is_file() and p.suffix.lower() in {".pyc", ".pyo"}:
            return _fail(f"forbidden bytecode file: {p.relative_to(root)}")

    print("Skill is valid!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
