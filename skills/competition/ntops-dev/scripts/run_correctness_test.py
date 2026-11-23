#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a narrow ntops correctness test.")
    parser.add_argument("--repo", default="ntops", help="Path to the ntops repository.")
    parser.add_argument("--test", default="tests/test_gelu.py", help="Pytest target.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        raise SystemExit(f"ntops repository not found: {repo}")

    command = [sys.executable, "-m", "pytest", args.test, "-q"]
    print(f"Running in {repo}: {' '.join(command)}")
    return subprocess.call(command, cwd=repo)


if __name__ == "__main__":
    raise SystemExit(main())
