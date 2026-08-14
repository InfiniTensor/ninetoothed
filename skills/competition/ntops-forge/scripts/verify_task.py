#!/usr/bin/env python3
"""End-to-end verify: preflight -> registration -> optional pytest -> optional ref diff."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def is_registered(ntops_root: Path, name: str) -> bool:
    kernels_init = ntops_root / "src" / "ntops" / "kernels" / "__init__.py"
    torch_init = ntops_root / "src" / "ntops" / "torch" / "__init__.py"
    k = kernels_init.read_text(encoding="utf-8")
    t = torch_init.read_text(encoding="utf-8")
    return bool(re.search(rf"^\s*{re.escape(name)}\s*,?\s*$", k, re.MULTILINE)) and (
        f"from ntops.torch.{name} import {name}" in t
    )


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def main() -> int:
    p = argparse.ArgumentParser(description="Verify ntops operator task completion")
    p.add_argument("--name", required=True, help="operator name")
    p.add_argument("--ntops-root", type=Path, required=True)
    p.add_argument(
        "--kernel", type=Path, help="kernel path (default: ntops kernels/<name>.py)"
    )
    p.add_argument(
        "--pytest", action="store_true", help="run scoped pytest for this op"
    )
    p.add_argument(
        "--pytest-file",
        help="explicit test file, e.g. tests/test_add.py (avoids -k substring matches)",
    )
    p.add_argument(
        "--compare-ref", type=Path, help="reference kernel for application() diff"
    )
    p.add_argument("--strict-preflight", action="store_true")
    args = p.parse_args()

    kernel = args.kernel or (
        args.ntops_root / "src" / "ntops" / "kernels" / f"{args.name}.py"
    )
    if not kernel.is_file():
        print(f"FAIL: kernel not found: {kernel}", file=sys.stderr)
        return 1

    py = sys.executable
    preflight_cmd = [py, str(SCRIPTS / "preflight.py"), str(kernel), "--kernel"]
    if args.strict_preflight:
        preflight_cmd.append("--strict")
    if run(preflight_cmd) != 0:
        return 1

    if not is_registered(args.ntops_root, args.name):
        print(
            f"FAIL: '{args.name}' not registered in kernels/torch __init__.py",
            file=sys.stderr,
        )
        print(
            f"FIX: python scripts/register_op.py --name {args.name} --ntops-root {args.ntops_root}"
        )
        return 1
    print(f"OK: '{args.name}' registered")

    if args.compare_ref:
        code = run(
            [
                py,
                str(SCRIPTS / "compare_ref.py"),
                str(kernel),
                "--ref",
                str(args.compare_ref),
            ]
        )
        if code == 1:
            return 1

    if args.pytest:
        pytest_args = [py, "-m", "pytest", "-q"]
        if args.pytest_file:
            pytest_args.append(args.pytest_file)
        else:
            default_test = args.ntops_root / "tests" / f"test_{args.name}.py"
            if default_test.is_file():
                pytest_args.append(f"tests/test_{args.name}.py")
            else:
                pytest_args.extend(["tests", "-k", f"test_{args.name}"])
        code = run(pytest_args, cwd=args.ntops_root)
        if code != 0:
            return code
        print("OK: pytest passed")

    print("OK: verify_task finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
