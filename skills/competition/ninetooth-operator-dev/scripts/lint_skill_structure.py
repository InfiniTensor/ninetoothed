#!/usr/bin/env python3
"""Validate the NineToothed operator skill framework."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SCRIPTS = [
    "scripts/lint_skill_structure.py",
    "scripts/collect_repo_map.py",
    "scripts/scaffold_selftest.py",
    "scripts/scaffold_subagent_session.py",
]

REQUIRED_FILES = [
    "SKILL.md",
    "agents/openai.yaml",
    "references/repo-map.md",
    "references/operator-task-contract.md",
    "references/dsl-pattern-index.md",
    "references/verification-matrix.md",
    "references/performance-diagnostics.md",
    "references/failure-playbook.md",
    "references/subagent-orchestration.md",
    "references/entropy-gc.md",
    "references/script-index.md",
    *REQUIRED_SCRIPTS,
    "tests/eval_cases.yaml",
    "tests/test_structure.py",
    "subagent-sessions/_template/request.md",
    "subagent-sessions/_template/summary.md",
    "subagent-sessions/_template/actions.md",
    "subagent-sessions/_template/artifacts.md",
    "subagent-sessions/_template/errors.md",
]

REQUIRED_EXAMPLES = [
    "examples/elementwise-broadcast/TASK.md",
    "examples/reduction-block/TASK.md",
    "examples/layout-sensitive/TASK.md",
    "examples/performance-diagnostics/TASK.md",
]

REQUIRED_TASK_HEADINGS = [
    "## Input Task",
    "## Agent Execution Summary",
    "## Repository Files Inspected",
    "## Patch Summary",
    "## Correctness Command",
    "## Correctness Result",
    "## Benchmark Command",
    "## Benchmark Result",
    "## Performance Conclusion",
    "## Failure Diagnosis",
    "## Risks and Unsupported Scope",
]

SUBAGENT_TEMPLATE_HEADINGS = {
    "subagent-sessions/_template/request.md": [
        "## Objective",
        "## Parent Context",
        "## Allowed Write Scope",
        "## Forbidden Scope",
        "## Expected Result",
        "## Validation Command",
        "## Return Summary Level",
    ],
    "subagent-sessions/_template/summary.md": [
        "## L0 Parent Return",
        "## L1 Session Summary",
        "## L2 Action Log Pointer",
        "## L3 Raw Evidence Pointer",
    ],
    "subagent-sessions/_template/actions.md": [
        "## Files Read",
        "## Files Changed",
        "## Commands Run",
        "## Decisions",
        "## Failed Attempts Worth Keeping",
    ],
    "subagent-sessions/_template/artifacts.md": [
        "## Patch Summary",
        "## Test Output Summary",
        "## Benchmark Summary",
        "## Generated Source or AOT Summary",
    ],
    "subagent-sessions/_template/errors.md": [
        "## Environment Errors",
        "## Tool Errors",
        "## Test Failures",
        "## Resolution or Workaround",
    ],
}

SUBAGENT_REFERENCE_SECTIONS = [
    "Mandatory Subagent Scenarios",
    "Parent-Agent Contract",
    "Subagent Tool Boundary",
    "Session Folder",
    "Progressive Summary Levels",
    "Return Format",
    "dry-run validation",
]

FORBIDDEN_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"secret\s*[:=]", re.IGNORECASE),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
    re.compile(r"hidden[_ -]?eval[_ -]?answer\s*[:=]", re.IGNORECASE),
    re.compile(r"delete tests to pass", re.IGNORECASE),
]

UNOWNED_TODO = re.compile(
    r"\b(?:TODO(?!\([^)]+\):)|FIXME|TBD)\b(?:\s*[:\-]\s*|$)",
    re.IGNORECASE,
)

PERFORMANCE_CLAIM_PATTERNS = [
    re.compile(
        r"\b\d+(?:\.\d+)?\s*x\s+(?:faster|slower|speedup|slowdown)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:candidate|kernel|implementation|operator)\b[^.\n]{0,100}"
        r"\b(?:faster|slower|outperforms|underperforms|speedup|slowdown|parity)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:speedup|slowdown|regression|parity)\b[^.\n]{0,80}\b\d", re.IGNORECASE
    ),
]

BENCHMARK_EVIDENCE_MARKERS = [
    "candidate_ms",
    "baseline_ms",
    "blocked:",
    "blocker:",
    "hardware/runtime blocker",
    "generated-source fallback evidence",
    "generated-source/AOT fallback evidence",
]


def fail(message: str) -> int:
    print(f"[skill-structure] ERROR: {message}")
    print(
        "[skill-structure] Fix: update the skill framework or its map, then rerun this script."
    )
    return 1


def iter_lintable_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {
            ".pyc",
            ".png",
            ".jpg",
            ".jpeg",
        }:
            continue
        if path.name == "lint_skill_structure.py":
            continue
        yield path


def markdown_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s|\Z)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def strip_inline_code(text: str) -> str:
    return re.sub(r"`[^`]*`", "", text)


def check_required_files(root: Path) -> int:
    missing = [
        path
        for path in REQUIRED_FILES + REQUIRED_EXAMPLES
        if not (root / path).exists()
    ]
    if missing:
        return fail("missing required files: " + ", ".join(missing))
    return 0


def check_script_executability(root: Path) -> int:
    for relative in REQUIRED_SCRIPTS:
        if not (root / relative).stat().st_mode & 0o111:
            return fail(f"script is not executable: {relative}")
    return 0


def check_skill_size(root: Path) -> int:
    lines = (root / "SKILL.md").read_text(encoding="utf-8").splitlines()
    if len(lines) > 180:
        return fail(
            "SKILL.md is too large; keep it as a map and move details into references/"
        )
    return 0


def check_examples(root: Path) -> int:
    for relative in REQUIRED_EXAMPLES:
        text = (root / relative).read_text(encoding="utf-8")
        missing = [heading for heading in REQUIRED_TASK_HEADINGS if heading not in text]
        if missing:
            return fail(f"{relative} is missing headings: {', '.join(missing)}")
    return 0


def check_unowned_todos(root: Path) -> int:
    for path in iter_lintable_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if UNOWNED_TODO.search(strip_inline_code(line)):
                return fail(
                    "unowned TODO marker found in "
                    f"{path.relative_to(root)}:{line_number}. "
                    "Use TODO(owner): with a clear owner/action, or move the debt "
                    "into references/entropy-gc.md."
                )
    return 0


def check_example_performance_claims(root: Path) -> int:
    for relative in REQUIRED_EXAMPLES:
        text = (root / relative).read_text(encoding="utf-8")
        benchmark_result = markdown_section(text, "## Benchmark Result")
        conclusion = markdown_section(text, "## Performance Conclusion")
        if not conclusion:
            continue

        has_evidence = any(
            marker in benchmark_result for marker in BENCHMARK_EVIDENCE_MARKERS
        )
        pending_or_optional = (
            "not run yet" in benchmark_result.lower()
            or "not required" in benchmark_result.lower()
        )
        has_claim = any(
            pattern.search(conclusion) for pattern in PERFORMANCE_CLAIM_PATTERNS
        )
        if has_claim and (pending_or_optional or not has_evidence):
            return fail(
                "performance claim without benchmark evidence found in "
                f"{relative}. Fix: record the real benchmark output or a concrete "
                "blocker under ## Benchmark Result before claiming speedup, "
                "slowdown, parity, or regression."
            )
    return 0


def check_subagent_summary(root: Path) -> int:
    for relative, headings in SUBAGENT_TEMPLATE_HEADINGS.items():
        text = (root / relative).read_text(encoding="utf-8")
        missing = [heading for heading in headings if heading not in text]
        if missing:
            return fail(f"{relative} is missing headings: {', '.join(missing)}")

    reference = (root / "references/subagent-orchestration.md").read_text(
        encoding="utf-8"
    )
    for phrase in SUBAGENT_REFERENCE_SECTIONS:
        if phrase not in reference:
            return fail(
                f"subagent orchestration reference is missing section: {phrase}"
            )
    return 0


def check_forbidden_artifacts(root: Path) -> int:
    pdfs = sorted(path.relative_to(root) for path in root.rglob("*.pdf"))
    if pdfs:
        return fail(
            "PDF files must not be packaged in the skill: " + ", ".join(map(str, pdfs))
        )

    for path in iter_lintable_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                return fail(
                    "forbidden artifact pattern "
                    f"{pattern.pattern!r} found in {path.relative_to(root)}"
                )
    return 0


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
    checks = [
        check_required_files,
        check_script_executability,
        check_skill_size,
        check_examples,
        check_unowned_todos,
        check_example_performance_claims,
        check_subagent_summary,
        check_forbidden_artifacts,
    ]
    for check in checks:
        result = check(root)
        if result:
            return result
    print(f"[skill-structure] OK: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
