#!/usr/bin/env python3
"""
Run correctness tests for NineToothed operators.
Usage:
    python scripts/run_correctness.py                     # non-benchmark tests
    python scripts/run_correctness.py --op relu           # single op
    python scripts/run_correctness.py --op relu --dtype fp16
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default=None, help="Operator name (e.g. relu, softmax)")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default=None,
                        help="dtype filter for tests that expose fp16/fp32 ids")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-m", "not benchmark"]
    filters = []
    if args.op:
        filters.append(args.op)
    if args.dtype:
        filters.append(args.dtype)
    if filters:
        cmd += ["-k", " and ".join(filters)]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
