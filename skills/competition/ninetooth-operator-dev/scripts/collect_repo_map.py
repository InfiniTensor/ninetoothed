#!/usr/bin/env python3
"""Print a compact map for a NineToothed checkout."""

from __future__ import annotations

import sys
from pathlib import Path

KEY_NAMES = {
    "README.md",
    "pyproject.toml",
    "tests/test_add.py",
    "tests/test_softmax.py",
    "tests/test_max_pool2d.py",
    "tests/test_aot.py",
    "tests/test_debugging.py",
    "src/ninetoothed/make.py",
    "src/ninetoothed/generation.py",
    "src/ninetoothed/debugging.py",
    "src/ninetoothed/auto_tuner.py",
}


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    if not (root / "src" / "ninetoothed").exists():
        print(f"ERROR: {root} does not look like a NineToothed checkout")
        return 1

    print(f"# Repository map for {root}")
    print("\n## Anchors")
    for name in sorted(KEY_NAMES):
        path = root / name
        status = "present" if path.exists() else "missing"
        print(f"- {name}: {status}")

    print("\n## Operator-like test files")
    for path in sorted((root / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if (
            "def arrangement" in text
            or "def application" in text
            or "ninetoothed.make" in text
        ):
            print(f"- {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
