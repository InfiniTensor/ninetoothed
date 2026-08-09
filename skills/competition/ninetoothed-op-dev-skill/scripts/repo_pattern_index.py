#!/usr/bin/env python3
"""Scan a NineToothed repo (and optional examples) and emit a keyword pattern index."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _paths import (
    add_repo_root_args,
    resolve_examples_root,
    resolve_repo_root,
    resolve_skill_root,
)
from _paths import (
    skill_root as detect_skill_root,
)

KEYWORDS = [
    "arrangement",
    "application",
    "ninetoothed.make",
    "ninetoothed.jit",
    "benchmark",
    "pytest",
    "generated",
    "aot",
    "stride",
    "contiguous",
    "softmax",
    "max_pool",
    "data_ptr",
    "ntl.where",
    "atomic_add",
]


def scan_repo(root: Path) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = defaultdict(list)
    if not root.is_dir():
        return hits
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        for kw in KEYWORDS:
            if kw in text:
                hits[kw].append(rel)
    return {k: v[:40] for k, v in hits.items()}


def render(nt_hits: dict, ex_hits: dict, nt_label: str, ex_label: str) -> str:
    lines = [
        "# Generated repo pattern index",
        "",
        f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
        "- **Auto-generated** by `scripts/repo_pattern_index.py` — re-run after upstream updates.",
        f"- ninetoothed root label: `{nt_label}`",
        f"- examples root label: `{ex_label}`",
        "",
        "## <repo-root>",
        "",
    ]
    for kw in KEYWORDS:
        files = nt_hits.get(kw, [])
        lines.append(f"### `{kw}` ({len(files)} files)")
        for f in files[:15]:
            lines.append(f"- `{nt_label}/{f}`")
        if len(files) > 15:
            lines.append(f"- … and {len(files) - 15} more")
        lines.append("")

    lines.extend(["## examples-root (optional)", ""])
    if not ex_hits:
        lines.append("_No `--examples-root` / examples tree found._")
        lines.append("")
    for kw in KEYWORDS:
        files = ex_hits.get(kw, [])
        lines.append(f"### `{kw}` ({len(files)} files)")
        for f in files[:15]:
            lines.append(f"- `{ex_label}/{f}`")
        if len(files) > 15:
            lines.append(f"- … and {len(files) - 15} more")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build keyword index of NineToothed repos."
    )
    add_repo_root_args(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown (default: <skill>/references/generated_repo_pattern_index.md)",
    )
    args = parser.parse_args()

    try:
        nt_root = resolve_repo_root(args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    examples = resolve_examples_root(args)
    try:
        sk = resolve_skill_root(args) if args.skill_root else detect_skill_root()
    except FileNotFoundError:
        sk = Path(__file__).resolve().parents[1]

    out = args.output or (sk / "references" / "generated_repo_pattern_index.md")
    nt = scan_repo(nt_root)
    ex = scan_repo(examples) if examples else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render(
            nt, ex, nt_label=".", ex_label=str(examples) if examples else "(missing)"
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
