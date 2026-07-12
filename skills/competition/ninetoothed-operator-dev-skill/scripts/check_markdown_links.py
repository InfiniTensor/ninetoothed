#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate local Markdown links in a skill package."
    )
    parser.add_argument("root", nargs="?", type=Path)
    args = parser.parse_args()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    failures: list[str] = []

    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for raw in LINK_RE.findall(text):
            target = raw.strip().split()[0].strip("<>")
            if target.lower().startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                failures.append(
                    f"link escapes package: {path.relative_to(root)} -> {raw}"
                )
                continue
            if not resolved.exists():
                failures.append(f"missing link: {path.relative_to(root)} -> {raw}")

    if failures:
        print("FAIL check_markdown_links.py")
        for item in failures:
            print(f"- {item}")
        return 1
    print("PASS check_markdown_links.py")
    print(f"root={root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
