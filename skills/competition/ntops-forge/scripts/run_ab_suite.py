#!/usr/bin/env python3
"""One-shot A/B suite: baseline reset -> forge gate -> treatment record -> AB report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPS = ("silu", "add", "gelu", "relu", "mul")


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run full A/B evidence pipeline")
    ap.add_argument("--ntops-root", type=Path, required=True)
    ap.add_argument("--ops", default=",".join(DEFAULT_OPS))
    ap.add_argument(
        "--elapsed", type=int, default=7, help="seconds per op for treatment record"
    )
    args = ap.parse_args()
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]

    steps = [
        [
            sys.executable,
            str(ROOT / "scripts" / "run_baseline_demo.py"),
            "--reset-csv",
            "--ops",
            args.ops,
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "forge.py"),
            "gate",
            "--ntops-root",
            str(args.ntops_root),
            "--ops",
            args.ops,
            "--no-record-ab",
        ],
    ]
    for cmd in steps:
        code = run(cmd)
        if code != 0:
            print(f"FAIL: exit {code}")
            return code

    for op in ops:
        code = run(
            [
                sys.executable,
                str(ROOT / "scripts" / "record_run.py"),
                "--mode",
                "treatment",
                "--task",
                op,
                "--preflight-pass",
                "--pytest-pass",
                "--steps",
                "1",
                "--interventions",
                "0",
                "--elapsed",
                str(args.elapsed),
            ]
        )
        if code != 0:
            return code

    return run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_ab.py"),
            "--input",
            str(ROOT / "docs" / "ab_runs.csv"),
            "--output",
            str(ROOT / "docs" / "AB_Report.md"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
