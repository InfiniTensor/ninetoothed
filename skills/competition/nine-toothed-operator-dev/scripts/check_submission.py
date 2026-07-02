#!/usr/bin/env python3
"""Check the competition skill package for required files."""

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_FILES = [
    "SKILL.md",
    "README.md",
    "HONOR_CODE.md",
    "REFERENCE.md",
    "references/operator-patterns.md",
    "references/nine-toothed-api-notes.md",
    "references/testing.md",
    "references/performance.md",
    "references/failure-diagnosis.md",
    "references/final-submission.md",
    "references/repository-pattern-index-ninetoothed.md",
    "references/repository-pattern-index-examples.md",
    "scripts/scan_repo.py",
    "scripts/build_pattern_index.py",
    "scripts/make_selftest_task.py",
    "scripts/check_submission.py",
    "tests/selftest_manifest.md",
    "reports/final_report_template.md",
]

SELFTEST_DIRS = [
    "examples/01-elementwise-add",
    "examples/02-softmax-reduction",
    "examples/03-layout-stride-offset",
    "examples/04-performance-diagnosis",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-dir", default=".")
    args = parser.parse_args()

    root = Path(args.skill_dir).resolve()
    if not root.exists():
        raise SystemExit(f"skill dir does not exist: {root}")

    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            errors.append(f"missing file: {rel}")

    for rel in SELFTEST_DIRS:
        task = root / rel / "task.md"
        if not task.is_file():
            errors.append(f"missing self-test task: {rel}/task.md")
            continue
        text = task.read_text(encoding="utf-8", errors="ignore")
        for heading in ("Input Task Statement", "Correctness", "Benchmark"):
            if heading not in text:
                errors.append(f"{rel}/task.md missing heading: {heading}")

    skill = root / "SKILL.md"
    if skill.is_file():
        text = skill.read_text(encoding="utf-8", errors="ignore")
        if not text.startswith("---"):
            errors.append("SKILL.md missing YAML frontmatter")
        for phrase in ("Required Workflow", "Hard Rules", "Reference Routing"):
            if phrase not in text:
                errors.append(f"SKILL.md missing section: {phrase}")

    if errors:
        print("Submission check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Submission check passed: {root}")
    print(
        "Reminder: replace TODO result fields with real correctness and benchmark logs before final submission."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
