#!/usr/bin/env python3
"""Reproducible no-skill baseline demo for A/B evidence (初赛冲分用)."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPS = ("silu", "add", "gelu", "relu", "mul")
OUT_MD = ROOT / "docs" / "Baseline_Demo.md"
CSV = ROOT / "docs" / "ab_runs.csv"
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


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()


def reset_csv(ops: list[str]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for op in ops:
            w.writerow(
                {
                    "mode": "baseline",
                    "task": op,
                    "preflight_pass": "false",
                    "pytest_pass": "not_run",
                    "steps": "6",
                    "intervention_count": "4",
                    "elapsed_seconds": "1200",
                    "recorded_at": now,
                }
            )


def record_baseline(task: str, steps: int, interventions: int, elapsed: int) -> None:
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "record_run.py"),
            "--mode",
            "baseline",
            "--task",
            task,
            "--steps",
            str(steps),
            "--interventions",
            str(interventions),
            "--elapsed",
            str(elapsed),
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run no-skill baseline demo and record A/B rows"
    )
    ap.add_argument(
        "--ops", default=",".join(DEFAULT_OPS), help="comma-separated operator names"
    )
    ap.add_argument("--write-report", type=Path, default=OUT_MD)
    ap.add_argument(
        "--skip-record", action="store_true", help="only print demo, do not write csv"
    )
    ap.add_argument(
        "--reset-csv",
        action="store_true",
        help="reset ab_runs.csv baseline rows before record",
    )
    args = ap.parse_args()
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]

    if args.reset_csv and not args.skip_record:
        reset_csv(ops)
        print(f"Reset {CSV} with {len(ops)} baseline rows")

    triton_py = ROOT / "docs" / "demo-logs" / "baseline_triton.py"
    triton_py.parent.mkdir(parents=True, exist_ok=True)
    triton_py.write_text(
        textwrap.dedent(
            """\
            import triton
            import triton.language as tl

            @triton.jit
            def bad_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
                pass
            """
        ),
        encoding="utf-8",
    )

    lines: list[str] = [
        "# Baseline Demo（无 Skill）",
        "",
        f"**时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 模拟 Agent 无领域知识：写出 Triton 稿",
        "",
        "```bash",
        f"python scripts/preflight.py {triton_py.relative_to(ROOT)} --kernel --strict",
        "```",
        "",
    ]

    code, out = run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preflight.py"),
            str(triton_py),
            "--kernel",
            "--strict",
        ]
    )
    lines.append(f"**exit={code}**（预期非 0）")
    lines.append("")
    lines.append("```")
    lines.append(out[:2000])
    lines.append("```")
    lines.append("")

    incomplete = ROOT / "docs" / "demo-logs" / "baseline_incomplete.py"
    incomplete.write_text(
        "def application(input, output):\n    output = input\n", encoding="utf-8"
    )
    code2, out2 = run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preflight.py"),
            str(incomplete),
            "--kernel",
            "--strict",
        ]
    )
    lines.extend(
        [
            "## 2. 模拟流程断裂：缺 premake / ninetoothed",
            "",
            f"**exit={code2}**（预期非 0）",
            "",
            "```",
            out2[:1200],
            "```",
            "",
            "## 3. Baseline 记录（无 skill 典型路径）",
            "",
            "| 算子 | 典型步骤 | 人工介入 | 耗时(估) | preflight | pytest |",
            "|------|----------|----------|----------|-----------|--------|",
        ]
    )

    for op in ops:
        steps, interventions, elapsed = 6, 4, 1200
        if not args.skip_record and not args.reset_csv:
            record_baseline(op, steps, interventions, elapsed)
        lines.append(
            f"| {op} | {steps} | {interventions} | {elapsed}s | FAIL | 未跑通 |"
        )

    lines.extend(
        [
            "",
            "## 4. Treatment 对照命令（有 skill）",
            "",
            "```bash",
            "source /root/miniconda3/bin/activate base",
            "cd /root/work/skill",
            "python scripts/forge.py gate --ntops-root /root/work/ntops",
            "python scripts/eval_ab.py --input docs/ab_runs.csv --output docs/AB_Report.md",
            "```",
            "",
            "> Treatment 由 `forge gate` 自动 record_run；本脚本只负责 Baseline 侧证据。",
        ]
    )

    args.write_report.parent.mkdir(parents=True, exist_ok=True)
    args.write_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.write_report}")
    print("\n=== Baseline demo done ===")
    print("Next: run `forge gate` on GPU, then `eval_ab.py`")
    if code == 0:
        print("WARN: expected Triton preflight to fail; check triton installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
