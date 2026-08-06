#!/usr/bin/env python3
"""Run the skill framework structure check and focused lint regressions."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = ROOT / "scripts" / "lint_skill_structure.py"


def run_lint(root: Path) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(root)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode, result.stdout


def copy_skill_tree(target: Path) -> Path:
    clone = target / "skill"
    shutil.copytree(
        ROOT,
        clone,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return clone


class SkillStructureTests(unittest.TestCase):
    def test_current_skill_structure_passes(self) -> None:
        code, output = run_lint(ROOT)
        print(output, end="")
        self.assertEqual(code, 0, output)

    def test_unowned_todo_is_rejected_with_fix_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clone = copy_skill_tree(Path(directory))
            task = clone / "examples" / "elementwise-broadcast" / "TASK.md"
            marker = "TO" + "DO: investigate stale benchmark claim"
            task.write_text(
                task.read_text(encoding="utf-8") + f"\n{marker}\n", encoding="utf-8"
            )

            code, output = run_lint(clone)

        self.assertNotEqual(code, 0, output)
        self.assertIn("unowned " + "TO" + "DO marker", output)
        self.assertIn("TO" + "DO(owner):", output)

    def test_performance_claim_without_benchmark_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clone = copy_skill_tree(Path(directory))
            task = clone / "examples" / "elementwise-broadcast" / "TASK.md"
            text = task.read_text(encoding="utf-8")
            start = text.index("## Performance Conclusion")
            end = text.index("## Failure Diagnosis")
            replacement = (
                "## Performance Conclusion\n\n"
                "The candidate kernel is 2x faster than PyTorch.\n\n"
            )
            task.write_text(text[:start] + replacement + text[end:], encoding="utf-8")

            code, output = run_lint(clone)

        self.assertNotEqual(code, 0, output)
        self.assertIn("performance claim without benchmark evidence", output)
        self.assertIn("record the real benchmark output", output)

    def test_missing_failure_diagnosis_section_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clone = copy_skill_tree(Path(directory))
            task = clone / "examples" / "elementwise-broadcast" / "TASK.md"
            text = task.read_text(encoding="utf-8")
            start = text.index("## Failure Diagnosis")
            end = text.index("## Risks and Unsupported Scope")
            task.write_text(text[:start] + text[end:], encoding="utf-8")

            code, output = run_lint(clone)

        self.assertNotEqual(code, 0, output)
        self.assertIn("missing headings", output)
        self.assertIn("## Failure Diagnosis", output)

    def test_non_executable_script_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clone = copy_skill_tree(Path(directory))
            script = clone / "scripts" / "collect_repo_map.py"
            script.chmod(script.stat().st_mode & ~0o111)

            code, output = run_lint(clone)

        self.assertNotEqual(code, 0, output)
        self.assertIn("script is not executable", output)
        self.assertIn("scripts/collect_repo_map.py", output)

    def test_common_github_token_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clone = copy_skill_tree(Path(directory))
            task = clone / "examples" / "elementwise-broadcast" / "TASK.md"
            token = "ghp_" + "A" * 36
            task.write_text(
                task.read_text(encoding="utf-8") + f"\n{token}\n",
                encoding="utf-8",
            )

            code, output = run_lint(clone)

        self.assertNotEqual(code, 0, output)
        self.assertIn("forbidden artifact pattern", output)

    def test_private_key_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clone = copy_skill_tree(Path(directory))
            task = clone / "examples" / "elementwise-broadcast" / "TASK.md"
            marker = "-----BEGIN " + "PRIVATE KEY-----"
            task.write_text(
                task.read_text(encoding="utf-8") + f"\n{marker}\n",
                encoding="utf-8",
            )

            code, output = run_lint(clone)

        self.assertNotEqual(code, 0, output)
        self.assertIn("forbidden artifact pattern", output)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SkillStructureTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
