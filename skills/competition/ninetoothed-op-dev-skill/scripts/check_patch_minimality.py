#!/usr/bin/env python3
"""Check git diff for patch minimality and competition compliance warnings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _paths import add_repo_root_args, resolve_logs_dir, resolve_repo_root


def run_git(args: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch minimality and compliance warnings."
    )
    add_repo_root_args(parser)
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Git repo to inspect (default: --repo-root)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        root = resolve_repo_root(args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    repo = args.repo or root
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or resolve_logs_dir(args) / "patch_minimality" / f"check_{ts}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    notes: list[str] = []

    code, inside = run_git(["rev-parse", "--is-inside-work-tree"], repo)
    if code != 0 or inside != "true":
        body = "# Patch minimality check\n\n**SKIP**: not a git repository.\n"
        out.write_text(body, encoding="utf-8")
        print(f"wrote {out} (skip)")
        return 0

    _, stat = run_git(["diff", "--stat"], repo)
    _, status = run_git(["status", "--short"], repo)
    _, names = run_git(["diff", "--name-only"], repo)

    deleted_tests = [n for n in names.splitlines() if "test" in n.lower() and n.strip()]
    src_core = [
        n
        for n in names.splitlines()
        if n.startswith("src/ninetoothed/")
        and any(x in n for x in ("generation.py", "cudaifier.py", "build.py"))
    ]
    skill_only = [n for n in names.splitlines() if "ninetoothed-op-dev-skill" in n]

    if deleted_tests:
        # Heuristic only — warn if test files appear in diff names with D status
        for line in status.splitlines():
            if line.startswith("D") and "test" in line.lower():
                warnings.append(f"Deleted test-related path: `{line}`")

    if src_core:
        warnings.append(
            "Diff touches compiler-core-ish files under src/ninetoothed/: "
            + ", ".join(f"`{n}`" for n in src_core[:8])
        )
    if skill_only and not names.strip():
        notes.append("Only skill package paths changed.")

    body = f"""# Patch minimality check

- At (UTC): {datetime.now(timezone.utc).isoformat()}
- Repo: `{repo}`

## git status --short

```
{status or "(clean)"}
```

## git diff --stat

```
{stat or "(no unstaged diff)"}
```

## Warnings

{chr(10).join(f"- {w}" for w in warnings) if warnings else "- (none)"}

## Notes

{chr(10).join(f"- {n}" for n in notes) if notes else "- Prefer minimal operator/test patches; avoid unrelated refactors."}
"""
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out}")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
