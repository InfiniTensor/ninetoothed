#!/usr/bin/env python3
"""Collect test or benchmark logs for the competition skill report."""

from __future__ import annotations

import argparse
import datetime as _datetime
import pathlib
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a test or benchmark log into the skill reports/logs directory."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=pathlib.Path,
        help="Path to a text log produced by pytest, Ruff, or a benchmark command.",
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Short label used in the copied log filename.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input

    if not source.exists() or not source.is_file():
        raise SystemExit(f"Input log does not exist or is not a file: {source}")

    skill_dir = pathlib.Path(__file__).resolve().parents[1]
    logs_dir = skill_dir / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = (
        "".join(
            char if char.isalnum() or char in ("-", "_") else "_" for char in args.label
        ).strip("_")
        or "run"
    )
    destination = logs_dir / f"{timestamp}_{safe_label}_{source.name}"
    shutil.copy2(source, destination)

    print(destination)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
