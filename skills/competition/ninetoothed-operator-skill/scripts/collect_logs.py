"""Collect self-test logs for the NineToothed operator skill."""

import subprocess
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = SKILL_ROOT / "examples"
LOG_DIR = SKILL_ROOT / "reports" / "logs"


TASK_TESTS = {
    "task-01": EXAMPLES_DIR / "task-01" / "test_add.py",
    "task-02": EXAMPLES_DIR / "task-02" / "test_softmax.py",
    "task-03": EXAMPLES_DIR / "task-03" / "test_transpose_add.py",
}


def run_task(task_id, test_path):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{task_id}.log"

    if not test_path.exists():
        message = f"SKIP: {test_path} does not exist\n"
        log_path.write_text(message, encoding="utf-8")
        return task_id, "missing", log_path

    command = ["python", "-m", "pytest", str(test_path), "-ra"]
    result = subprocess.run(
        command,
        cwd=SKILL_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

    if result.returncode == 0:
        status = "passed_or_skipped"
    else:
        status = "failed"

    return task_id, status, log_path


def main():
    results = []

    for task_id, test_path in TASK_TESTS.items():
        results.append(run_task(task_id, test_path))

    summary_path = LOG_DIR / "summary.md"
    lines = [
        "# Self-test Log Summary",
        "",
        "Note: Tests without CUDA or ninetoothed will skip.",
        "",
        "| Task | Status | Log |",
        "|------|--------|-----|",
    ]

    for task_id, status, log_path in results:
        lines.append(f"| {task_id} | {status} | `{log_path}` |")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
