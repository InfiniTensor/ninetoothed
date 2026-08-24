#!/usr/bin/env python3
"""
Run benchmark tests for NineToothed operators.
Usage:
    python scripts/run_benchmark.py                       # all benchmarks
    python scripts/run_benchmark.py --op softmax          # single op benchmark
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default=None, help="Operator benchmark to run (e.g. relu, softmax)")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "benchmark", "-v", "--tb=short"]
    if args.op:
        cmd += ["-k", args.op]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
