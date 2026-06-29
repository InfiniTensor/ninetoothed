#!/usr/bin/env python3
"""Pack proposal/initial-round materials into one submission zip."""

from __future__ import annotations

import argparse
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE = datetime.now().strftime("%Y%m%d")

COMMON_INCLUDE = [
    "README.md",
    "HONOR_CODE.md",
    "REFERENCE.md",
    "skills/ntops-copilot",
    "skills/ntops-forge",
    "scripts",
]


def iter_stage_include(stage: str) -> list[str]:
    if stage == "proposal":
        return [*COMMON_INCLUDE, "docs/Proposal.md"]
    return [
        *COMMON_INCLUDE,
        "docs/Proposal.md",
        "docs/MidTermReport.md",
        "docs/于鸿伟_九齿skill创新挑战_中期报告.pdf",
        "docs/SelfTestPlan.md",
        "docs/SubmissionGuide.md",
        "docs/InitialRound.md",
        "docs/GPU_Test_Report.md",
        "docs/ab_runs.csv",
        "docs/AB_Report.md",
        "docs/Forge_Design.md",
        "docs/DemoShowcase.md",
        "docs/CompetitiveAnalysis.md",
        "docs/OptimizationSummary.md",
        "docs/FinalsRoadmap.md",
        "docs/BenchmarkPlan.md",
        "docs/CloudRun.md",
        "docs/ScoringAlignment.md",
        "docs/DualSkillGuide.md",
        "docs/selftests",
        "docs/st2_st3_gpu_results.md",
        "docs/Baseline_Demo.md",
        "docs/PR_TEMPLATE.md",
        "docs/forge_runs.jsonl",
        "docs/bench_silu.json",
        "docs/screenshots",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Pack contest submission zip")
    p.add_argument(
        "--stage",
        choices=("proposal", "initial"),
        default="initial",
        help="submission stage (default: initial)",
    )
    args = p.parse_args()
    out = ROOT / f"submission-{args.stage}-{DATE}.zip"
    include = iter_stage_include(args.stage)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            path = ROOT / rel
            if path.is_file():
                zf.write(path, rel)
            elif path.is_dir():
                for f in path.rglob("*"):
                    if f.is_file() and "__pycache__" not in f.parts:
                        zf.write(f, f.relative_to(ROOT).as_posix())
            else:
                print(f"skip missing: {rel}")
    print(f"Created {out}")


if __name__ == "__main__":
    main()
