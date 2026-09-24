#!/usr/bin/env python3
"""Run a correctness command (typically pytest) and save logs under logs/correctness/."""

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


def write_report(
    out_md: Path,
    *,
    command: list[str],
    cwd: Path,
    returncode: int,
    stdout: str,
    stderr: str,
    task_id: str,
    repo: Path,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    log_base = out_md.with_suffix("")
    stdout_path = Path(str(log_base) + ".stdout.txt")
    stderr_path = Path(str(log_base) + ".stderr.txt")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo))
        except ValueError:
            return str(p)

    verdict = "PASS" if returncode == 0 else "FAIL"
    body = f"""# Correctness run

- Task: `{task_id}`
- At (UTC): {datetime.now(timezone.utc).isoformat()}
- Working directory: `{cwd}`
- Command: `{" ".join(shlex.quote(c) for c in command)}`
- Exit code: `{returncode}`
- Verdict: **{verdict}**

## Logs

- stdout: `{_rel(stdout_path)}`
- stderr: `{_rel(stderr_path)}`

## stdout (tail)

```
{stdout[-8000:] if len(stdout) > 8000 else stdout}
```

## stderr (tail)

```
{stderr[-4000:] if len(stderr) > 4000 else stderr}
```
"""
    out_md.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run correctness command with logging."
    )
    add_repo_root_args(parser)
    parser.add_argument(
        "--task-id", required=True, help="Task identifier for log naming"
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Working directory (default: --repo-root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown report path (default: <repo>/logs/correctness/<task>_<ts>.md)",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command after --, e.g. pytest ..."
    )
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command required after optional --")

    try:
        root = resolve_repo_root(args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    cwd = args.cwd or root
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (
        args.output
        or resolve_logs_dir(args) / "correctness" / f"{_slug(args.task_id)}_{ts}.md"
    )

    proc = subprocess.run(
        args.command,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    write_report(
        out,
        command=args.command,
        cwd=cwd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        task_id=args.task_id,
        repo=root,
    )
    print(f"correctness log: {out} (exit {proc.returncode})")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
