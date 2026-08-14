#!/usr/bin/env python3
"""Summarize baseline/treatment runs for initial-round report."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Row:
    mode: str
    task: str
    preflight_pass: bool
    pytest_pass: Optional[bool]
    steps: int
    intervention_count: int
    elapsed_seconds: int


def parse_optional_bool(value: str) -> Optional[bool]:
    v = (value or "").strip().lower()
    if v in {"", "na", "n/a", "not_run", "not run", "skip", "skipped", "unknown"}:
        return None
    return v in {"1", "true", "yes", "y", "pass", "passed"}


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "mode",
            "task",
            "preflight_pass",
            "pytest_pass",
            "steps",
            "intervention_count",
            "elapsed_seconds",
        }
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"CSV missing columns: {', '.join(missing)}")
        for r in reader:
            rows.append(
                Row(
                    mode=r["mode"].strip(),
                    task=r["task"].strip(),
                    preflight_pass=parse_optional_bool(r["preflight_pass"]) or False,
                    pytest_pass=parse_optional_bool(r["pytest_pass"]),
                    steps=int(r["steps"]),
                    intervention_count=int(r["intervention_count"]),
                    elapsed_seconds=int(r["elapsed_seconds"]),
                )
            )
    return rows


def safe_rate(numer: float, denom: float) -> Optional[float]:
    if denom <= 0:
        return None
    return numer / denom


def summarize(rows: list[Row], mode: str) -> dict[str, Optional[float]]:
    items = [r for r in rows if r.mode == mode]
    if not items:
        return {
            "count": 0,
            "preflight_rate": None,
            "pytest_rate": None,
            "avg_steps": 0.0,
            "avg_intervention": 0.0,
            "avg_elapsed_seconds": 0.0,
        }
    total = len(items)
    pytest_items = [r for r in items if r.pytest_pass is not None]
    pytest_total = len(pytest_items)
    return {
        "count": total,
        "preflight_rate": sum(r.preflight_pass for r in items) / total,
        "pytest_rate": safe_rate(
            sum(bool(r.pytest_pass) for r in pytest_items), pytest_total
        ),
        "avg_steps": sum(r.steps for r in items) / total,
        "avg_intervention": sum(r.intervention_count for r in items) / total,
        "avg_elapsed_seconds": sum(r.elapsed_seconds for r in items) / total,
    }


def pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate A/B run logs")
    p.add_argument("--input", required=True, type=Path, help="CSV input path")
    p.add_argument("--output", required=True, type=Path, help="Markdown report path")
    args = p.parse_args()

    rows = load_rows(args.input)
    baseline = summarize(rows, "baseline")
    treatment = summarize(rows, "treatment")
    if treatment["count"] == 0:
        print("WARN: no treatment rows; run `forge gate` then re-run eval_ab")

    def delta_rate(a: Optional[float], b: Optional[float]) -> str:
        if a is None or b is None:
            return "N/A"
        return pct(b - a)

    def delta_num(a: float, b: float) -> str:
        return f"{b - a:+.2f}"

    t_pytest = treatment["pytest_rate"]
    if t_pytest is not None and t_pytest >= 1.0:
        quality_line = (
            f"- Quality: Treatment 在 GPU 环境 pytest 通过率 {pct(t_pytest)}；"
            f"Baseline 无 skill 时 preflight {pct(baseline['preflight_rate'])}、pytest 未跑通。"
        )
    elif t_pytest is None:
        quality_line = (
            "- Quality: Treatment pytest 数据待补；请在 GPU 机重跑 `forge gate` 后 "
            "`record_run.py --pytest-pass`。"
        )
    else:
        quality_line = f"- Quality: Treatment pytest 通过率 {pct(t_pytest)}。"

    report = f"""# A/B Evaluation Report

## Data Summary

- Total baseline runs: {int(baseline["count"])}
- Total treatment runs: {int(treatment["count"])}

## Metrics

| Metric | Baseline | Treatment | Delta (Treatment - Baseline) |
|---|---:|---:|---:|
| preflight pass rate | {pct(baseline["preflight_rate"])} | {pct(treatment["preflight_rate"])} | {delta_rate(baseline["preflight_rate"], treatment["preflight_rate"])} |
| pytest pass rate | {pct(baseline["pytest_rate"])} | {pct(treatment["pytest_rate"])} | {delta_rate(baseline["pytest_rate"], treatment["pytest_rate"])} |
| avg steps | {baseline["avg_steps"]:.2f} | {treatment["avg_steps"]:.2f} | {delta_num(baseline["avg_steps"], treatment["avg_steps"])} |
| avg interventions | {baseline["avg_intervention"]:.2f} | {treatment["avg_intervention"]:.2f} | {delta_num(baseline["avg_intervention"], treatment["avg_intervention"])} |
| avg elapsed seconds | {baseline["avg_elapsed_seconds"]:.2f} | {treatment["avg_elapsed_seconds"]:.2f} | {delta_num(baseline["avg_elapsed_seconds"], treatment["avg_elapsed_seconds"])} |

## Conclusion

{quality_line}
- Efficiency: Treatment 平均步骤 {treatment["avg_steps"]:.1f} vs Baseline {baseline["avg_steps"]:.1f}（{delta_num(baseline["avg_steps"], treatment["avg_steps"])}）。
- Human effort: Treatment 人工介入 {treatment["avg_intervention"]:.1f} 次 vs Baseline {baseline["avg_intervention"]:.1f} 次（{delta_num(baseline["avg_intervention"], treatment["avg_intervention"])}）。
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
