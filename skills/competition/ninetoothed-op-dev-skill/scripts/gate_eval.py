"""Evidence-based gate evaluation and scorecard scoring (no pytest→10/10 auto map)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from _task_spec import (
    TaskSpec,
    find_tbd_fields,
    load_task_spec,
    operator_contract_violations,
    validate_card_text,
)

RUBRIC_MAX = {
    "task_completion": 4,
    "tests_verification": 2,
    "performance_awareness": 1,
    "patch_minimality": 1,
    "repo_style": 1,
    "process_compliance": 1,
}


@dataclass
class GateEvidence:
    task_id: str
    task_card: Path | None = None
    task_spec_path: Path | None = None
    correctness_log: Path | None = None
    benchmark_log: Path | None = None
    test_source: Path | None = None
    patch_dir: Path | None = None
    pytest_verdict: str = "UNKNOWN"
    skill_used: str = "yes"


@dataclass
class GateResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    total: int = 0
    max_total: int = 10

    def add_failure(self, reason: str) -> None:
        self.failures.append(reason)
        self.ok = False


def _read(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_correctness_verdict(log_text: str) -> str:
    m = re.search(r"Verdict:\s*\*\*(PASS|FAIL|SKIP)\*\*", log_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.search(r"\bSKIPPED\b", log_text, re.IGNORECASE):
        return "SKIP"
    if "Exit code: `0`" in log_text:
        return "PASS"
    if "Exit code:" in log_text:
        return "FAIL"
    return "UNKNOWN"


def _benchmark_log_ok(log_text: str) -> bool:
    if not log_text.strip():
        return False
    if re.search(r"\bSKIP\b", log_text, re.IGNORECASE):
        return False
    return bool(re.search(r"benchmark|ms/iter|ratio", log_text, re.IGNORECASE))


def evaluate_gate(evidence: GateEvidence) -> GateResult:
    result = GateResult(ok=True, scores={k: 0 for k in RUBRIC_MAX})

    spec: TaskSpec | None = None
    if evidence.task_spec_path and evidence.task_spec_path.is_file():
        spec = load_task_spec(evidence.task_spec_path)

    card_text = _read(evidence.task_card)
    if not card_text:
        result.add_failure("missing task_card.md")
    else:
        tbd = validate_card_text(card_text)
        if tbd:
            for item in tbd:
                result.add_failure(item)
        if spec:
            for item in find_tbd_fields(spec):
                result.add_failure(f"task spec field incomplete: {item}")

    corr_text = _read(evidence.correctness_log)
    corr_verdict = _parse_correctness_verdict(corr_text)
    if not corr_text:
        result.add_failure("missing correctness log")
    elif corr_verdict == "SKIP":
        result.add_failure("correctness log indicates SKIP")
    elif corr_verdict == "FAIL":
        result.add_failure("correctness log verdict FAIL")
    elif corr_verdict != "PASS":
        result.add_failure(f"correctness log verdict not PASS ({corr_verdict})")

    if evidence.pytest_verdict.upper() == "SKIP":
        result.add_failure("pytest verdict SKIP")
    elif evidence.pytest_verdict.upper() == "FAIL":
        result.add_failure("pytest verdict FAIL")

    test_text = _read(evidence.test_source)
    if not test_text:
        result.add_failure("missing test source for operator audit")
    elif spec or evidence.task_id:
        for v in operator_contract_violations(evidence.task_id, test_text, spec):
            result.add_failure(v)

    bench_required = spec.benchmark_required if spec else False
    bench_text = _read(evidence.benchmark_log)
    if bench_required:
        if not bench_text:
            result.add_failure("benchmark required but log missing")
        elif not _benchmark_log_ok(bench_text):
            result.add_failure("benchmark log empty or SKIP")

    # Itemized scores — derived from evidence, never pytest alone
    if not any(
        "operator" in f or "forbidden" in f or "GELU" in f for f in result.failures
    ):
        if test_text and corr_verdict == "PASS":
            result.scores["task_completion"] = RUBRIC_MAX["task_completion"]

    if corr_text and corr_verdict == "PASS" and evidence.correctness_log:
        result.scores["tests_verification"] = RUBRIC_MAX["tests_verification"]

    if bench_required:
        if bench_text and _benchmark_log_ok(bench_text):
            result.scores["performance_awareness"] = RUBRIC_MAX["performance_awareness"]
    elif bench_text and _benchmark_log_ok(bench_text):
        result.scores["performance_awareness"] = RUBRIC_MAX["performance_awareness"]
    elif bench_text and re.search(r"\bN/A\b", bench_text, re.IGNORECASE):
        result.scores["performance_awareness"] = RUBRIC_MAX["performance_awareness"]
    elif corr_text and re.search(
        r"benchmark.*\bN/A\b|not required", corr_text, re.IGNORECASE
    ):
        result.scores["performance_awareness"] = RUBRIC_MAX["performance_awareness"]

    if evidence.patch_dir and evidence.patch_dir.is_dir():
        files = list(evidence.patch_dir.rglob("*"))
        nontrivial = [p for p in files if p.is_file() and p.name != "README.md"]
        if len(nontrivial) <= 6:
            result.scores["patch_minimality"] = RUBRIC_MAX["patch_minimality"]

    if test_text:
        has_arr = "def arrangement" in test_text
        has_app = "def application" in test_text
        has_make = "ninetoothed.make" in test_text or "ntl." in test_text
        if has_arr and has_app and has_make:
            result.scores["repo_style"] = RUBRIC_MAX["repo_style"]

    card_ok = card_text and not validate_card_text(card_text)
    has_session = bool(
        evidence.task_card
        and evidence.task_card.parent.joinpath("session_evidence.json").is_file()
    )
    has_patch_gate = bool(evidence.patch_dir and evidence.patch_dir.is_dir())
    if (
        card_ok
        and corr_text
        and corr_verdict == "PASS"
        and (has_session or has_patch_gate)
    ):
        result.scores["process_compliance"] = RUBRIC_MAX["process_compliance"]

    result.total = sum(result.scores.values())
    result.max_total = sum(RUBRIC_MAX.values())

    if result.failures:
        result.ok = False
    elif result.total < result.max_total:
        result.ok = False
        result.failures.append(
            f"incomplete rubric score {result.total}/{result.max_total}"
        )

    return result


def format_scorecard_markdown(evidence: GateEvidence, result: GateResult) -> str:
    lines = [
        f"# Scorecard — {evidence.task_id}",
        "",
        f"- Skill used: **{evidence.skill_used}**",
        f"- Gate: **{'PASS' if result.ok else 'FAIL'}**",
        f"- Pytest (input only, not auto-scored): `{evidence.pytest_verdict}`",
        "- Rubric: official 0–10 operator rubric (see SKILL.md)",
        "",
        "| Sub-item | Max | Score | Evidence / notes |",
        "|----------|-----|-------|------------------|",
        f"| Task completion | 4 | {result.scores['task_completion']} | test source + operator contract |",
        f"| Tests & verification | 2 | {result.scores['tests_verification']} | correctness log |",
        f"| Performance awareness | 1 | {result.scores['performance_awareness']} | benchmark log if required |",
        f"| Patch minimality | 1 | {result.scores['patch_minimality']} | patch dir size |",
        f"| Repo style consistency | 1 | {result.scores['repo_style']} | ninetoothed patterns in test |",
        f"| Process & compliance | 1 | {result.scores['process_compliance']} | task_card + logs |",
        f"| **Total** | **10** | **{result.total}** | evidence-based |",
        "",
        "## Gate failures",
        "",
    ]
    if result.failures:
        lines.extend(f"- {f}" for f in result.failures)
    else:
        lines.append("- (none)")
    lines.extend(["", "## Evidence paths", ""])
    for label, path in (
        ("task_card", evidence.task_card),
        ("correctness", evidence.correctness_log),
        ("benchmark", evidence.benchmark_log),
        ("test_source", evidence.test_source),
    ):
        lines.append(f"- {label}: `{path}`" if path else f"- {label}: (missing)")
    return "\n".join(lines) + "\n"
