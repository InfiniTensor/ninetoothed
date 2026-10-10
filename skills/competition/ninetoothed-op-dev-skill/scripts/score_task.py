#!/usr/bin/env python3
"""Scorecard template generation and evidence-based gate evaluation."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from _paths import add_skill_root_arg
from gate_eval import GateEvidence, evaluate_gate, format_scorecard_markdown


def _cmd_template(args: argparse.Namespace) -> int:
    if args.output is None:
        # Default: stdout only — never create evals/scorecards inside the skill tree.
        out_path: Path | None = None
    else:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
    body = f"""# Scorecard — {args.task_id}

- Run: `{args.run_id}`
- Skill used: **{args.skill_used}**
- At (UTC): {datetime.now(timezone.utc).isoformat()}
- Rubric: official 0–10 operator rubric (see SKILL.md)

| Sub-item | Max | Score | Evidence / notes |
|----------|-----|-------|------------------|
| Task completion | 4 | | fill from evidence |
| Tests & verification | 2 | | fill from evidence |
| Performance awareness | 1 | | fill from evidence |
| Patch minimality | 1 | | fill from evidence |
| Repo style consistency | 1 | | fill from evidence |
| Process & compliance | 1 | | fill from evidence |
| **Total** | **10** | | do not copy pytest PASS → 10 |

## Commands

```
(record exact commands from the run)
```

## Log paths

- correctness:
- benchmark:
- patch check:

## Verdict

- [ ] PASS for self-test gate (requires `score_task.py evaluate`)
- [ ] Needs iteration
"""
    if out_path is None:
        print(body, end="")
    else:
        out_path.write_text(body, encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    evidence = GateEvidence(
        task_id=args.task_id,
        task_card=args.task_card,
        task_spec_path=args.task_spec,
        correctness_log=args.correctness_log,
        benchmark_log=args.benchmark_log,
        test_source=args.test_source,
        patch_dir=args.patch_dir,
        pytest_verdict=args.pytest_verdict,
        skill_used=args.skill_used,
    )
    result = evaluate_gate(evidence)
    body = format_scorecard_markdown(evidence, result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(body, end="")

    if not result.ok:
        for failure in result.failures:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"GATE PASS: {result.total}/10", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Scorecard template or evidence gate.")
    add_skill_root_arg(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p_tpl = sub.add_parser("template", help="Emit empty scorecard template")
    p_tpl.add_argument("--task-id", required=True)
    p_tpl.add_argument("--run-id", default="run1")
    p_tpl.add_argument("--skill-used", choices=("yes", "no"), default="yes")
    p_tpl.add_argument("--output", type=Path, default=None)
    p_tpl.set_defaults(func=_cmd_template)

    p_ev = sub.add_parser(
        "evaluate", help="Score from evidence; fail gate on missing/TBD/SKIP"
    )
    p_ev.add_argument("--task-id", required=True)
    p_ev.add_argument("--task-card", type=Path, required=True)
    p_ev.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help="Task specification markdown/YAML (required for operator-contract checks)",
    )
    p_ev.add_argument("--correctness-log", type=Path, default=None)
    p_ev.add_argument("--benchmark-log", type=Path, default=None)
    p_ev.add_argument("--test-source", type=Path, default=None)
    p_ev.add_argument("--patch-dir", type=Path, default=None)
    p_ev.add_argument("--pytest-verdict", default="UNKNOWN")
    p_ev.add_argument("--skill-used", choices=("yes", "no"), default="yes")
    p_ev.add_argument("--output", type=Path, default=None)
    p_ev.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
