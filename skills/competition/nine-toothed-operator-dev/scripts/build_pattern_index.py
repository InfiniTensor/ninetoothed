#!/usr/bin/env python3
"""Build a compact index of NineToothed operator, test, and benchmark patterns."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CATEGORY_PATTERNS = {
    "operator_kernel": re.compile(
        r"def arrangement|def application|ninetoothed\.make|@ninetoothed\.jit"
    ),
    "test_correctness": re.compile(
        r"torch\.allclose|pytest\.mark\.parametrize|get_available_devices"
    ),
    "benchmark": re.compile(
        r"pytest\.mark\.benchmark|triton\.testing|do_bench|benchmark\("
    ),
    "debug_aot_generated": re.compile(
        r"simulate_arrangement|ninetoothed\.aot|generated|cache_source|CodeGenerator|build"
    ),
    "layout_sensitive": re.compile(
        r"stride|strides|contiguous|permute|as_strided|storage_offset|offset"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-category", type=int, default=40)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()

    if not repo.exists():
        raise SystemExit(f"repo does not exist: {repo}")

    files = [p for p in repo.rglob("*.py") if ".git" not in p.parts]
    sections: dict[str, list[str]] = {name: [] for name in CATEGORY_PATTERNS}

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel = path.relative_to(repo).as_posix()

        for category, pattern in CATEGORY_PATTERNS.items():
            if pattern.search(text):
                sections[category].append(rel)

    lines = [
        "# Repository Pattern Index",
        "",
        f"Repository: `{repo}`",
        "",
        "Use this index to choose nearby examples before editing. Regenerate it with `scripts/build_pattern_index.py` when the target repository changes.",
        "",
    ]

    for category, paths in sections.items():
        lines.append(f"## {category}")
        lines.append("")

        if not paths:
            lines.append("- No matches found.")
        else:
            for rel in sorted(paths)[: args.max_per_category]:
                lines.append(f"- `{rel}`")

            if len(paths) > args.max_per_category:
                lines.append(f"- ... {len(paths) - args.max_per_category} more")

        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
