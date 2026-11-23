#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HIGH_RISK = {
    "hidden task passed",
    "first prize guaranteed",
    "full non-contiguous support is verified",
    "aot verified",
    "generated source verified",
    "infinicore dispatch verified",
    "broad speedup",
}
SAFE_CONTEXT = {
    "not ",
    "no ",
    "never",
    "do not",
    "instead of",
    "cannot",
    "unverified",
    "[blocked]",
    "[todo-gpu]",
    "不声称",
    "不扩展",
    "未验证",
    "不得",
    "不能",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reject unsupported high-risk success claims."
    )
    parser.add_argument("root", nargs="?", type=Path)
    args = parser.parse_args()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    hits: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".md",
            ".txt",
            ".json",
            ".csv",
        }:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            lower = line.lower()
            if any(phrase in lower for phrase in HIGH_RISK) and not any(
                marker in lower for marker in SAFE_CONTEXT
            ):
                hits.append(
                    f"{path.relative_to(root).as_posix()}:{number}: {line.strip()[:180]}"
                )

    if hits:
        print("FAIL check_false_verified_claims.py")
        for item in hits:
            print(f"- {item}")
        return 1
    print("PASS check_false_verified_claims.py")
    print(f"root={root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
