#!/usr/bin/env python3
"""Create a subagent session record folder from templates."""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "subagent-sessions" / "_template"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "task"


def main() -> int:
    title = " ".join(sys.argv[1:]).strip() or "task"
    target = (
        ROOT / "subagent-sessions" / f"{datetime.now():%Y%m%d-%H%M}-{slugify(title)}"
    )
    if target.exists():
        print(f"ERROR: session already exists: {target}")
        return 1
    shutil.copytree(TEMPLATE, target)
    request = target / "request.md"
    text = request.read_text(encoding="utf-8")
    text = text.replace("## Objective\n", f"## Objective\n\n{title}\n")
    request.write_text(text, encoding="utf-8")
    print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
