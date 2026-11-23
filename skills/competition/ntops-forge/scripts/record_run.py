#!/usr/bin/env python3
"""Append one A/B evaluation row to docs/ab_runs.csv."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "docs" / "ab_runs.csv"
FIELDS = [
    "mode",
    "task",
    "preflight_pass",
    "pytest_pass",
    "steps",
    "intervention_count",
    "elapsed_seconds",
    "recorded_at",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Record one skill run for A/B report")
    p.add_argument("--mode", choices=("baseline", "treatment"), default="treatment")
    p.add_argument("--task", required=True)
    p.add_argument("--preflight-pass", action="store_true")
    p.add_argument("--pytest-pass", action="store_true")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--interventions", type=int, default=0)
    p.add_argument("--elapsed", type=int, default=0)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = p.parse_args()

    row = {
        "mode": args.mode,
        "task": args.task,
        "preflight_pass": str(args.preflight_pass).lower(),
        "pytest_pass": "true" if args.pytest_pass else "not_run",
        "steps": str(args.steps),
        "intervention_count": str(args.interventions),
        "elapsed_seconds": str(args.elapsed),
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    write_header = not args.csv.is_file() or args.csv.stat().st_size == 0
    with args.csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"OK: recorded {args.mode}/{args.task} -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
