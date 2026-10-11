#!/usr/bin/env python3
"""
Environment Check Script.

Verifies that all required dependencies for the ninetoothed-operator-skill
are installed and accessible. Run before executing any self-test tasks.

Usage:
    python scripts/env_check.py
"""

import sys


def check_package(name, import_name=None):
    """Check if a Python package is importable."""
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_cuda():
    """Check if CUDA is available via PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            return (
                True,
                f"CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}",
            )
        else:
            return False, "PyTorch installed but CUDA not available"
    except ImportError as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("NineToothed Operator Skill — Environment Check")
    print("=" * 60)

    all_ok = True

    # Core dependencies
    checks = [
        ("torch", "torch"),
        ("triton", "triton"),
        ("ninetoothed", "ninetoothed"),
        ("pytest", "pytest"),
    ]

    for name, import_name in checks:
        ok, detail = check_package(name, import_name)
        status = "✓" if ok else "✗"
        info = f"({detail})" if ok else f"— MISSING: {detail}"
        print(f"  [{status}] {name:<20} {info}")
        if not ok:
            all_ok = False

    # CUDA check
    cuda_ok, cuda_detail = check_cuda()
    status = "✓" if cuda_ok else "✗"
    print(f"  [{status}] {'cuda':<20} {cuda_detail}")
    if not cuda_ok:
        all_ok = False

    # Version info
    print("\n--- Version Info ---")
    try:
        import torch

        print(f"  PyTorch:  {torch.__version__}")
    except ImportError:
        pass
    try:
        import ninetoothed

        print(f"  NineToothed: {getattr(ninetoothed, '__version__', 'unknown')}")
    except ImportError:
        pass
    try:
        import triton

        print(f"  Triton:   {getattr(triton, '__version__', 'unknown')}")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed — environment ready.")
    else:
        print("Some checks failed — install missing packages before running tasks.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
