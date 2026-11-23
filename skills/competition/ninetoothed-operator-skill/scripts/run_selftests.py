#!/usr/bin/env python3
"""
Self-Test Runner Script.

Discovers and runs all correctness tests across the four self-test tasks.
Run this script from the NinToothed repository root after installing the skill.

Usage:
    python skills/competition/ninetoothed-operator-skill/scripts/run_selftests.py

Or to run a specific task:
    python skills/competition/ninetoothed-operator-skill/scripts/run_selftests.py --task t1
"""

import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = SKILL_DIR / "examples"

TASK_TESTS = {
    "t1": EXAMPLES_DIR / "task1_elementwise_broadcast" / "test_correctness.py",
    "t2": EXAMPLES_DIR / "task2_reduction_block" / "test_correctness.py",
    "t3": [
        EXAMPLES_DIR / "task3_layout_sensitive" / "test_correctness.py",
        EXAMPLES_DIR / "task3_layout_sensitive" / "test_layout_compare.py",
    ],
    "t4": None,  # T4 is a meta-task; use diagnosis_record.md
}


def run_pytest(test_path, verbose=True):
    """Run pytest on a single test file."""
    if not Path(test_path).exists():
        print(f"  ✗ Test file not found: {test_path}")
        return False

    cmd = ["pytest", str(test_path)]
    if verbose:
        cmd.append("-v")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    args = sys.argv[1:]

    # Filter to specific task if requested
    if "--task" in args:
        idx = args.index("--task")
        task_key = args[idx + 1]
        tasks_to_run = {task_key: TASK_TESTS.get(task_key)}
    else:
        tasks_to_run = TASK_TESTS

    print("=" * 60)
    print("NineToothed Operator Skill — Self-Test Runner")
    print("=" * 60)

    results = {}
    for task_name, test_path in tasks_to_run.items():
        print(f"\n--- {task_name.upper()} ---")
        if test_path is None:
            print(f"  (no automated test for {task_name}; see diagnosis_record.md)")
            results[task_name] = "SKIPPED"
            continue

        paths = test_path if isinstance(test_path, list) else [test_path]
        all_passed = True
        for p in paths:
            if not run_pytest(p):
                all_passed = False
        results[task_name] = "PASSED" if all_passed else "FAILED"

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    for task_name, status in results.items():
        symbol = "✓" if status == "PASSED" else ("✗" if status == "FAILED" else "○")
        print(f"  [{symbol}] {task_name}: {status}")
    print("=" * 60)

    # Only count actual test runs — SKIPPED tasks are not failures
    actual_results = {k: v for k, v in results.items() if v != "SKIPPED"}
    all_passed = (
        all(s == "PASSED" for s in actual_results.values()) if actual_results else True
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
