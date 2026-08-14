#!/usr/bin/env python3
"""Bootstrap a forge spec YAML from a short natural-language line."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# op keyword → (family, pattern, formula)
KNOWN: dict[str, tuple[str, str, str]] = {
    "silu": (
        "elementwise_unary",
        "unary",
        "output = input / (1 + ntl.exp(-ntl.cast(input, ntl.float32)))",
    ),
    "relu": ("elementwise_unary", "unary", "output = max(0.0, input)"),
    "gelu": (
        "elementwise_unary",
        "unary",
        "output = input * 0.5 * (1 + ntl.erf(input / ntl.sqrt(2.0)))",
    ),
    "sigmoid": (
        "elementwise_unary",
        "unary",
        "output = 1 / (1 + ntl.exp(-ntl.cast(input, ntl.float32)))",
    ),
    "add": ("elementwise_binary", "binary", "output = input + alpha * other"),
    "sub": ("elementwise_binary", "binary", "output = input - alpha * other"),
    "mul": ("elementwise_binary", "binary", "output = input * other"),
    "div": ("elementwise_binary", "binary", "output = input / other"),
}


def parse_nl(line: str) -> dict:
    low = line.lower().strip()
    op = None
    for name in KNOWN:
        if re.search(rf"\b{re.escape(name)}\b", low):
            op = name
            break
    if not op:
        raise ValueError(f"cannot detect operator from: {line!r}")

    family, pattern, formula = KNOWN[op]
    if "binary" in low or pattern == "binary":
        family, pattern = KNOWN[op][0], KNOWN[op][1]

    return {
        "id": f"forge-{op}-custom",
        "op": op,
        "family": family,
        "pattern": pattern,
        "spec": f"Auto-generated from: {line}",
        "formula": formula,
        "reference": f"src/ntops/kernels/{op}.py",
        "pytest": f"tests/test_{op}.py",
        "contest_id": "T1-1-X",
        "github_id": "hongwei-2026",
        "acceptance": {
            "preflight_strict": True,
            "compare_ref": True,
            "pytest": True,
        },
    }


def to_yaml(data: dict) -> str:
    lines = [
        f"id: {data['id']}",
        f"op: {data['op']}",
        f"family: {data['family']}",
        f"pattern: {data['pattern']}",
        "spec: |",
        f"  {data['spec']}",
        f'formula: "{data["formula"]}"',
        f"reference: {data['reference']}",
        f"pytest: {data['pytest']}",
        f"contest_id: {data['contest_id']}",
        f"github_id: {data['github_id']}",
        "acceptance:",
        "  preflight_strict: true",
        "  compare_ref: true",
        "  pytest: true",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="NL → forge spec YAML")
    p.add_argument("line", help='e.g. "silu unary" or "add binary x plus y"')
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    try:
        data = parse_nl(args.line)
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_yaml(data), encoding="utf-8")
    print(f"Wrote {args.out} [{data['op']}/{data['family']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
