#!/usr/bin/env python3
"""Summarize artifacts from a single agent run directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS = [
    "task.md",
    "task_card.md",
    "run_prompt.md",
    "run_log.md",
    "changed_files.md",
    "correctness_result.md",
    "benchmark_result.md",
    "failure_diagnosis.md",
    "scorecard.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a task run directory.")
    parser.add_argument("run_dir", type=Path, help="Directory containing run artifacts")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    out = args.output or run_dir / "RUN_SUMMARY.md"
    present = [name for name in ARTIFACTS if (run_dir / name).is_file()]
    missing = [name for name in ARTIFACTS if name not in present]

    body = f"""# Run summary

- Directory: `{run_dir}`
- At (UTC): {datetime.now(timezone.utc).isoformat()}

## Present artifacts

{chr(10).join("- " + p for p in present) or "- (none)"}

## Missing (optional unless task requires)

{chr(10).join("- " + m for m in missing)}

## Quick links

"""
    for name in present:
        body += f"- [{name}](./{name})\n"

    out.write_text(body, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
