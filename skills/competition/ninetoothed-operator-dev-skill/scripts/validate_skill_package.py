#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

REQUIRED_FILES = {
    "SKILL.md",
    "README.md",
    "README.zh-CN.md",
    "HONOR_CODE.md",
    "REFERENCE.md",
    "PR_DESCRIPTION.md",
    "SUBMISSION_CHECKLIST.md",
    "SUBMISSION_COMMANDS.md",
    "agents/openai.yaml",
    "references/index.md",
    "reports/final_report.md",
    "reports/123123_九齿skill创新挑战_T3-1-1_赛题报告.pdf",
}

REQUIRED_DIRS = {"agents", "references", "scripts", "examples", "tests", "reports"}

REQUIRED_REFERENCES = {
    "repo_reading_routes.md",
    "elementwise_broadcast_guide.md",
    "reduction_blocking_guide.md",
    "layout_stride_offset_guide.md",
    "testing_correctness_guide.md",
    "performance_benchmark_guide.md",
    "generated_source_aot_integration_guide.md",
    "failure_diagnosis_playbook.md",
    "patch_applicability_guide.md",
    "evidence_and_compliance_policy.md",
}

SELFTESTS = {
    "SELFTEST-EW-001",
    "SELFTEST-RED-001",
    "SELFTEST-LAYOUT-001",
    "SELFTEST-PERF-AOT-001",
    "SELFTEST-IMPL-001",
}

FORBIDDEN_NAMES = {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules"}


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("SKILL.md frontmatter is not closed")
    block = text[4:end]
    keys = re.findall(r"^([A-Za-z][A-Za-z0-9_-]*):", block, re.MULTILINE)
    metadata: dict[str, str] = {key: "" for key in keys}
    name_match = re.search(r"^name:\s*([^\n]+)$", block, re.MULTILINE)
    if name_match:
        metadata["name"] = name_match.group(1).strip().strip("\"'")
    description_match = re.search(
        r"^description:\s*(?:>-?|\|-?)?\s*\n(?P<body>(?:^[ \t]+.*\n?)*)",
        block,
        re.MULTILINE,
    )
    if description_match:
        metadata["description"] = " ".join(
            line.strip() for line in description_match.group("body").splitlines()
        )
    return metadata, text[end + 5 :]


def validate_benchmarks(root: Path, failures: list[str]) -> int:
    count = 0
    required = {
        "operator",
        "shape",
        "dtype",
        "warmup",
        "repeat",
        "baseline_median_ms",
        "baseline_mean_ms",
        "baseline_min_ms",
        "baseline_samples_ms",
        "candidate_median_ms",
        "candidate_mean_ms",
        "candidate_min_ms",
        "candidate_samples_ms",
        "speedup_median",
        "correctness_status",
    }
    for path in sorted((root / "examples" / "selftests").glob("*/benchmark.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            failures.append(f"empty benchmark: {path.relative_to(root)}")
            continue
        row = rows[0]
        missing = sorted(required - set(row))
        if missing:
            failures.append(
                f"benchmark missing fields {missing}: {path.relative_to(root)}"
            )
            continue
        if row["correctness_status"] != "PASS":
            failures.append(
                f"benchmark correctness is not PASS: {path.relative_to(root)}"
            )
            continue
        try:
            baseline_samples = [
                float(value) for value in row["baseline_samples_ms"].split(";")
            ]
            candidate_samples = [
                float(value) for value in row["candidate_samples_ms"].split(";")
            ]
            warmup = int(row["warmup"])
            repeat = int(row["repeat"])
            baseline_mean = float(row["baseline_mean_ms"])
            baseline_median = float(row["baseline_median_ms"])
            baseline_min = float(row["baseline_min_ms"])
            candidate_mean = float(row["candidate_mean_ms"])
            candidate_median = float(row["candidate_median_ms"])
            candidate_min = float(row["candidate_min_ms"])
            speedup = float(row["speedup_median"])
        except ValueError:
            failures.append(
                f"benchmark has nonnumeric timing metadata: {path.relative_to(root)}"
            )
            continue
        if warmup < 0 or repeat <= 0:
            failures.append(
                f"benchmark has invalid warmup/repeat: {path.relative_to(root)}"
            )
            continue
        if len(baseline_samples) != repeat or len(candidate_samples) != repeat:
            failures.append(
                f"benchmark sample count does not equal repeat: {path.relative_to(root)}"
            )
            continue
        expected = (
            (baseline_mean, statistics.mean(baseline_samples), "baseline mean"),
            (baseline_median, statistics.median(baseline_samples), "baseline median"),
            (baseline_min, min(baseline_samples), "baseline min"),
            (candidate_mean, statistics.mean(candidate_samples), "candidate mean"),
            (
                candidate_median,
                statistics.median(candidate_samples),
                "candidate median",
            ),
            (candidate_min, min(candidate_samples), "candidate min"),
            (speedup, baseline_median / candidate_median, "speedup"),
        )
        inconsistent = [
            label
            for actual, calculated, label in expected
            if not math.isclose(actual, calculated, rel_tol=1e-5, abs_tol=2e-6)
        ]
        if inconsistent:
            failures.append(
                f"benchmark summary disagrees with samples ({', '.join(inconsistent)}): {path.relative_to(root)}"
            )
            continue
        if row.get("baseline") != "pytorch" or row.get("candidate") != "ntops":
            failures.append(
                f"benchmark baseline/candidate labels are invalid: {path.relative_to(root)}"
            )
            continue
        if not row.get("timestamp") or row.get("device") in {"", "NA"}:
            failures.append(
                f"benchmark is missing timestamp or device: {path.relative_to(root)}"
            )
            continue
        count += 1
    if count < 2:
        failures.append(f"expected at least two valid benchmark records, found {count}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the self-contained skill package."
    )
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--strict-identity", action="store_true")
    args = parser.parse_args()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    failures: list[str] = []
    warnings: list[str] = []

    for rel in sorted(REQUIRED_FILES):
        if not (root / rel).is_file():
            failures.append(f"missing file: {rel}")
    for rel in sorted(REQUIRED_DIRS):
        if not (root / rel).is_dir():
            failures.append(f"missing directory: {rel}")

    skill = root / "SKILL.md"
    if skill.is_file():
        text = skill.read_text(encoding="utf-8")
        try:
            metadata, body = parse_frontmatter(text)
        except ValueError as exc:
            failures.append(str(exc))
        else:
            if set(metadata) != {"name", "description"}:
                failures.append("frontmatter must contain only name and description")
            if metadata.get("name") != root.name:
                failures.append("frontmatter name must match the skill directory")
            if len(metadata.get("description", "")) < 120:
                failures.append(
                    "frontmatter description is too short for reliable triggering"
                )
            if "NineToothed" not in metadata.get("description", ""):
                failures.append("frontmatter description must identify NineToothed")
            if not body.strip():
                failures.append("SKILL.md body is empty")
        line_count = len(text.splitlines())
        if not 150 <= line_count <= 230:
            failures.append(f"SKILL.md must contain 150-230 lines, found {line_count}")

    refs = root / "references"
    for name in sorted(REQUIRED_REFERENCES):
        path = refs / name
        if not path.is_file() or path.stat().st_size < 500:
            failures.append(f"missing or undersized reference: references/{name}")

    selftests = root / "examples" / "selftests"
    for name in sorted(SELFTESTS):
        readme = selftests / name / "README.md"
        if not readme.is_file() or readme.stat().st_size < 700:
            failures.append(f"missing or undersized self-test README: {name}")

    trigger_path = root / "tests" / "fixtures" / "trigger_cases.json"
    if trigger_path.is_file():
        data = json.loads(trigger_path.read_text(encoding="utf-8"))
        if len(data.get("positive", [])) < 10 or len(data.get("negative", [])) < 10:
            failures.append(
                "trigger fixture requires at least 10 positive and 10 negative cases"
            )
        if not all(
            isinstance(item, str) and item.strip()
            for key in ("positive", "negative")
            for item in data.get(key, [])
        ):
            failures.append("trigger fixture entries must be nonempty strings")
    else:
        failures.append("missing tests/fixtures/trigger_cases.json")

    benchmark_count = validate_benchmarks(root, failures)

    forbidden = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in {".pyc", ".pyo"}
    )
    if forbidden:
        failures.append("forbidden generated files: " + ", ".join(forbidden))

    honor = root / "HONOR_CODE.md"
    identity_ready = (
        honor.is_file()
        and "REQUIRED BEFORE SUBMISSION" not in honor.read_text(encoding="utf-8")
    )
    if not identity_ready:
        warnings.append("participant identity and signature are still required")
        if args.strict_identity:
            failures.append("strict identity validation failed")

    if failures:
        print("FAIL validate_skill_package.py")
        for item in failures:
            print(f"- {item}")
        return 1

    print("PASS validate_skill_package.py")
    print(f"root={root}")
    print(f"benchmark_records={benchmark_count}")
    print(f"identity_ready={str(identity_ready).lower()}")
    for item in warnings:
        print(f"WARNING: {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
