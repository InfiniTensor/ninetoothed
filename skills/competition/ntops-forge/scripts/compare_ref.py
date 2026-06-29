#!/usr/bin/env python3
"""Compare application() in candidate kernel vs reference ntops kernel."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REF_FN_CANDIDATES = ("application", "default_application")


def _function_source(text: str, node: ast.FunctionDef) -> str:
    if hasattr(ast, "get_source_segment"):
        seg = ast.get_source_segment(text, node)
        if seg:
            return seg.strip()
    lines = text.splitlines()
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "\n".join(lines[start:end]).strip()


def _function_names(tree: ast.AST) -> list[str]:
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def extract_application_source(path: Path, *, reference: bool = False) -> str | None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    names = _function_names(tree)

    lookup: list[str] = []
    if reference:
        lookup.extend(REF_FN_CANDIDATES)
        lookup.extend(sorted(n for n in names if n.endswith("_application")))
    else:
        lookup.append("application")

    seen: set[str] = set()
    for target in lookup:
        if target in seen or target not in names:
            continue
        seen.add(target)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == target:
                return _function_source(text, node)
    return None


def normalize(body: str) -> str:
    lines = []
    for ln in body.splitlines():
        ln = ln.strip()
        if ln.startswith("def "):
            continue
        if "#" in ln:
            ln = ln.split("#", 1)[0].strip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Diff application() vs reference kernel")
    p.add_argument("candidate", type=Path, help="candidate kernel .py")
    p.add_argument("--ref", type=Path, required=True, help="reference kernel .py")
    args = p.parse_args()

    cand = extract_application_source(args.candidate, reference=False)
    ref = extract_application_source(args.ref, reference=True)
    if cand is None:
        print("FAIL: candidate missing application()", file=sys.stderr)
        return 1
    if ref is None:
        print("FAIL: reference missing application()", file=sys.stderr)
        return 1

    nc, nr = normalize(cand), normalize(ref)
    if nc == nr:
        print("OK: application() matches reference")
        return 0

    print("WARN: application() differs from reference")
    print("--- candidate")
    print(nc)
    print("--- reference")
    print(nr)
    print("--- hint: align formula using skills/ntops-copilot/formulas.md")
    return 2


if __name__ == "__main__":
    sys.exit(main())
