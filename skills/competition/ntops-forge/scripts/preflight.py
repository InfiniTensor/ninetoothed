#!/usr/bin/env python3
"""Preflight checks for ntops-copilot skill layout or kernel .py files."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

SKILL_REQUIRED = ("SKILL.md", "reference.md", "formulas.md")
KERNEL_REQUIRED_CALLS = ("application",)


def check_skill_dir(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_dir():
        return [f"not a directory: {path}"]
    for name in SKILL_REQUIRED:
        if not (path / name).is_file():
            errors.append(f"missing {name}")
    skill_md = path / "SKILL.md"
    if skill_md.is_file():
        text = skill_md.read_text(encoding="utf-8")
        if "name:" not in text[:400]:
            errors.append("SKILL.md missing YAML frontmatter name:")
        if "premake" not in text and "ninetoothed.make" not in text:
            errors.append(
                "SKILL.md should mention premake or ninetoothed.make workflow"
            )
    return errors


def _has_triton_jit(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Attribute) and dec.attr == "jit":
                    mod = dec.value
                    if isinstance(mod, ast.Name) and mod.id == "triton":
                        return True
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr == "jit":
                        return True
    src = ast.unparse(tree) if hasattr(ast, "unparse") else ""
    return "@triton.jit" in src


def check_kernel(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"not a file: {path}"]
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    if "@triton.jit" in text or _has_triton_jit(tree):
        errors.append(
            "found Triton @triton.jit — use NineToothed premake/application instead"
        )

    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in KERNEL_REQUIRED_CALLS:
        if fn not in funcs:
            errors.append(f"missing function: {fn}")

    has_premake = "premake" in funcs
    has_arrangement = "arrangement" in funcs
    has_make = "ninetoothed.make" in text or (
        any(
            isinstance(n, ast.Call)
            and isinstance(getattr(n.func, "attr", None), str)
            and n.func.attr == "make"
            for n in ast.walk(tree)
        )
    )

    if has_premake:
        if "element_wise" not in text and "arrangement" not in text:
            errors.append(
                "ntops premake kernel should import ntops.kernels.element_wise.arrangement"
            )
    elif has_arrangement and has_make:
        pass  # examples style
    else:
        errors.append("kernel must use premake (ntops) or arrangement+make (examples)")

    if "ninetoothed" not in text:
        errors.append("missing ninetoothed import/usage")

    return errors


def check_kernel_strict(path: Path) -> list[str]:
    errors = check_kernel(path)
    text = path.read_text(encoding="utf-8")
    if "TODO" in text:
        errors.append("application() still contains TODO — finish formula before PR")
    if "output = x" in text or "output = input + alpha * other  # TODO" in text:
        errors.append("application() still uses scaffold placeholder")
    if "# noqa: F841" not in text:
        errors.append("missing # noqa: F841 on output assignment (ntops convention)")
    return errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="skill directory or kernel .py")
    p.add_argument("--kernel", action="store_true", help="treat path as kernel file")
    p.add_argument(
        "--strict", action="store_true", help="reject TODO/placeholder in kernel"
    )
    args = p.parse_args()

    if args.kernel:
        errors = (
            check_kernel_strict(args.path) if args.strict else check_kernel(args.path)
        )
    else:
        errors = check_skill_dir(args.path)
    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("OK: preflight passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
