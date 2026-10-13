#!/usr/bin/env python3
"""Summarize useful NineToothed files for an operator task."""

from __future__ import annotations

import argparse
from pathlib import Path

PATTERNS = {
    "operators": (
        "def arrangement",
        "def application",
        "ninetoothed.make",
        "@ninetoothed.jit",
    ),
    "tests": ("pytest", "torch.allclose", "get_available_devices"),
    "benchmarks": ("benchmark", "perf_report", "do_bench", "pytest.mark.benchmark"),
    "diagnostics": ("generated", "generate", "aot", "debug", "build"),
    "layout": ("stride", "offset", "contiguous", "permute", "as_strided"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", required=True, help="Path to a NineToothed repository checkout"
    )
    parser.add_argument("--limit", type=int, default=20, help="Max files per category")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()

    if not repo.exists():
        raise SystemExit(f"repo does not exist: {repo}")

    py_files = [p for p in repo.rglob("*.py") if ".git" not in p.parts]
    print(f"Repository: {repo}")
    print(f"Python files scanned: {len(py_files)}")

    for category, needles in PATTERNS.items():
        matches: list[Path] = []

        for path in py_files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if any(needle in text for needle in needles):
                matches.append(path)

        print(f"\n[{category}] {len(matches)} candidate files")

        for path in matches[: args.limit]:
            print(path.relative_to(repo).as_posix())

    print("\nSuggested next searches:")
    print('rg -n "def arrangement|def application|ninetoothed.make|@ninetoothed.jit" .')
    print(
        'rg -n "softmax|add|relu|gelu|max_pool|stride|offset|benchmark|generated|aot" .'
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
