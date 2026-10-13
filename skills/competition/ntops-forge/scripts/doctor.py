#!/usr/bin/env python3
"""Check environment readiness for ntops-copilot workflow."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


def ok(msg: str) -> None:
    print(f"OK  {msg}")


def warn(msg: str) -> None:
    print(f"WARN {msg}")


def fail(msg: str) -> None:
    print(f"FAIL {msg}")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    errors = 0

    print("ntops-copilot doctor")
    print("-" * 40)

    if shutil.which("python") or shutil.which("python3"):
        ok("python executable found")
    else:
        fail("python not found")
        errors += 1

    for mod, label in [
        ("ninetoothed", "ninetoothed"),
        ("torch", "torch"),
        ("pytest", "pytest"),
    ]:
        if has_module(mod):
            ok(f"{label} importable")
        elif mod == "pytest":
            fail(f"{label} not installed (pip install pytest)")
            errors += 1
        else:
            warn(f"{label} not installed")

    if has_module("torch"):
        import torch

        if torch.cuda.is_available():
            ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warn("CUDA not available (ntops pytest will skip cuda tests)")

    if has_module("ntops"):
        ok("ntops installed")
    else:
        warn("ntops not installed (run: pip install -e <ntops-root>)")

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
            ok("nvidia-smi: " + out.strip().split("\n")[0])
        except Exception:
            warn("nvidia-smi exists but failed")
    else:
        warn("nvidia-smi not found")

    if shutil.which("git"):
        ok("git available")
    else:
        warn("git not found")

    print("-" * 40)
    if errors:
        print(f"doctor finished with {errors} error(s)")
        return 1
    print("doctor finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
