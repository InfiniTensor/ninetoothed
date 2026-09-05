#!/usr/bin/env python3
"""Check submission completeness for the NineToothed .skill competition.

Validates that all required files exist, self-test records are present,
and the skill structure meets competition requirements.

Usage:
    python scripts/check_submission.py --skill-dir .
    python scripts/check_submission.py --skill-dir /path/to/nt-devskill --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_ROOT_FILES = [
    "SKILL.md",
    "README.md",
    "HONOR_CODE.md",
    "REFERENCE.md",
]

REQUIRED_DIRS = [
    "references",
    "scripts",
    "examples",
    "specs",
    "tests",
    "agents",
]

REQUIRED_REFERENCES = [
    "API_REFERENCE.md",
    "OPTIMIZATION_GUIDE.md",
    "FIX_CARDS.md",
    "TAXONOMY.md",
    "PATTERNS.md",
    "CODE_TEMPLATES.md",
    "LAYOUT.md",
]

REQUIRED_SCRIPTS = [
    "validate.py",
    "benchmark.py",
    "doctor.py",
    "generate_op.py",
]

EXPECTED_OPERATORS = [
    "add", "softmax", "matmul", "fused_rms_norm", "silu",
    "bmm", "addmm", "scaled_dot_product_attention",
    "swiglu", "conv2d", "rotary_position_embedding", "max_pool2d",
]


def check_file(path: Path, label: str, verbose: bool) -> bool:
    exists = path.exists()
    if verbose or not exists:
        status = "OK" if exists else "MISSING"
        size = f" ({path.stat().st_size} bytes)" if exists and path.is_file() else ""
        print(f"  [{status:>7s}] {label}: {path.relative_to(path.parent.parent)}{size}")
    return exists


def check_dir(path: Path, label: str, verbose: bool) -> bool:
    exists = path.is_dir()
    if verbose or not exists:
        status = "OK" if exists else "MISSING"
        count = f" ({len(list(path.iterdir()))} files)" if exists else ""
        print(f"  [{status:>7s}] {label}: {path.name}/{count}")
    return exists


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skill-dir", type=Path, default=Path("."), help="Skill root directory")
    p.add_argument("--verbose", "-v", action="store_true", help="Show all checks, not just failures")
    args = p.parse_args(argv)

    root = args.skill_dir.resolve()
    if not (root / "SKILL.md").exists():
        print(f"ERROR: {root} does not contain SKILL.md")
        return 1

    print(f"=== Submission Check: {root.name} ===\n")

    total = 0
    passed = 0

    # Root files
    print("Root files:")
    for f in REQUIRED_ROOT_FILES:
        total += 1
        if check_file(root / f, f, args.verbose):
            passed += 1

    # Directories
    print("\nDirectories:")
    for d in REQUIRED_DIRS:
        total += 1
        if check_dir(root / d, d, args.verbose):
            passed += 1

    # Reference files
    print("\nReference files:")
    ref_dir = root / "references"
    for f in REQUIRED_REFERENCES:
        total += 1
        if ref_dir.exists():
            if check_file(ref_dir / f, f, args.verbose):
                passed += 1
        elif args.verbose:
            print(f"  ❌ {f}: references/ directory missing")

    # Script files
    print("\nScript files:")
    scripts_dir = root / "scripts"
    for f in REQUIRED_SCRIPTS:
        total += 1
        if scripts_dir.exists():
            if check_file(scripts_dir / f, f, args.verbose):
                passed += 1

    # Example operators
    print("\nExample operators:")
    examples_dir = root / "examples"
    for op in EXPECTED_OPERATORS:
        total += 1
        op_dir = examples_dir / op if examples_dir.exists() else None
        if op_dir and op_dir.is_dir():
            kernel = op_dir / "kernel.py"
            torch_impl = op_dir / "torch_impl.py"
            has_kernel = kernel.exists()
            has_wrapper = torch_impl.exists()
            if has_kernel and has_wrapper:
                passed += 1
                if args.verbose:
                    print(f"  [     OK] {op}: kernel.py + torch_impl.py")
            else:
                missing = []
                if not has_kernel:
                    missing.append("kernel.py")
                if not has_wrapper:
                    missing.append("torch_impl.py")
                print(f"  [MISSING] {op}: missing {', '.join(missing)}")
        else:
            print(f"  [MISSING] {op}: directory not found")

    # Spec files
    print("\nSpec files:")
    specs_dir = root / "specs"
    for op in EXPECTED_OPERATORS:
        total += 1
        spec = specs_dir / f"{op}.yaml" if specs_dir.exists() else None
        if spec and spec.exists():
            passed += 1
            if args.verbose:
                print(f"  [     OK] {op}.yaml")
        else:
            print(f"  [MISSING] {op}.yaml: not found")

    # Agent config
    print("\nAgent config:")
    total += 1
    agent_yaml = root / "agents" / "openai.yaml"
    if check_file(agent_yaml, "openai.yaml", args.verbose):
        passed += 1

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Result: {passed}/{total} checks passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("[PASS] Submission is COMPLETE")
        return 0
    else:
        missing = total - passed
        print(f"[WARN] Submission is INCOMPLETE ({missing} items missing)")
        print("       Run with --verbose for full details")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
