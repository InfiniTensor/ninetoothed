#!/usr/bin/env python3
"""List available ntops-copilot task cards."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "skills" / "ntops-copilot" / "tasks"


def load_task(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    data: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


def main() -> int:
    p = argparse.ArgumentParser(description="List ntops-copilot tasks")
    p.add_argument("--skill-root", type=Path, default=ROOT)
    args = p.parse_args()

    tasks_dir = args.skill_root / "skills" / "ntops-copilot" / "tasks"
    files = sorted(tasks_dir.glob("task_*.yaml"))
    if not files:
        print(f"No tasks under {tasks_dir}")
        return 1

    print("id\tpattern\top\tpytest_file")
    for f in files:
        t = load_task(f)
        op = t.get("op_name", f.stem.replace("task_", ""))
        print(
            f"{t.get('id', f.stem)}\t{t.get('pattern', '?')}\t{op}\t"
            f"{t.get('pytest_file', f'tests/test_{op}.py')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
