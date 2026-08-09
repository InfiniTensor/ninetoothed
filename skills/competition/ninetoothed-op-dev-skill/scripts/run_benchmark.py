#!/usr/bin/env python3
"""Run a benchmark command and save raw output plus summary markdown."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _paths import add_repo_root_args, resolve_logs_dir, resolve_repo_root


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark command with logging.")
    add_repo_root_args(parser)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--baseline", default="torch reference (document in report)")
    parser.add_argument("--input-sizes", default="document in report")
    parser.add_argument("--conclusion", default="fill after reviewing raw output")
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command required")

    try:
        root = resolve_repo_root(args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    cwd = args.cwd or root
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_md = (
        args.output
        or resolve_logs_dir(args) / "benchmark" / f"{_slug(args.task_id)}_{ts}.md"
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_md.with_suffix(".raw.txt")

    proc = subprocess.run(args.command, cwd=cwd, capture_output=True, text=True)
    raw = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    raw_path.write_text(raw, encoding="utf-8")

    try:
        raw_rel = str(raw_path.relative_to(root))
    except ValueError:
        raw_rel = str(raw_path)

    report = f"""# Benchmark run

- Task: `{args.task_id}`
- At (UTC): {datetime.now(timezone.utc).isoformat()}
- Working directory: `{cwd}`
- Command: `{" ".join(shlex.quote(c) for c in args.command)}`
- Exit code: `{proc.returncode}`
- Baseline: {args.baseline}
- Input sizes: {args.input_sizes}

## Raw output

`{raw_rel}`

```
{raw[-12000:] if len(raw) > 12000 else raw}
```

## Conclusion

{args.conclusion}
"""
    out_md.write_text(report, encoding="utf-8")
    print(f"benchmark log: {out_md} (exit {proc.returncode})")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
