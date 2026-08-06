#!/usr/bin/env python3
"""Validate the competition submission package layout and metadata."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTICIPANT_NAME = "赖泉laiquan"
REQUIRED_FILES = {
    "FINAL_REPORT.md",
    "HONOR_CODE.md",
    "README.md",
    "README_zh.md",
    "REFERENCE.md",
    "SKILL.md",
    "proposal.md",
}
OBSOLETE_PROPOSAL_PATHS = {
    "dsl-patterns.md",
    "failure-diagnosis-playbook.md",
    "operator-requirement-checklist.md",
    "repo-reading-map.md",
    "testing-and-benchmarking.md",
}
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)]+)\)")


class SubmissionPackageTests(unittest.TestCase):
    def test_required_submission_files_are_present(self) -> None:
        missing = sorted(name for name in REQUIRED_FILES if not (ROOT / name).is_file())

        self.assertEqual(missing, [])

    def test_standalone_repository_files_are_not_nested(self) -> None:
        unexpected = [
            path.name
            for path in (
                ROOT / ".git",
                ROOT / ".github",
                ROOT / ".gitignore",
                ROOT / "AGENTS.md",
                ROOT / "skills",
            )
            if path.exists()
        ]

        self.assertEqual(unexpected, [])

    def test_participant_identity_is_complete(self) -> None:
        honor_code_path = ROOT / "HONOR_CODE.md"
        proposal_path = ROOT / "proposal.md"

        self.assertTrue(honor_code_path.is_file())
        self.assertTrue(proposal_path.is_file())

        honor_code = honor_code_path.read_text(encoding="utf-8")
        proposal = proposal_path.read_text(encoding="utf-8")

        self.assertIn(f"Participant name: {PARTICIPANT_NAME}", honor_code)
        self.assertIn(f"选手姓名：`{PARTICIPANT_NAME}`", proposal)
        self.assertNotIn("to be filled", honor_code.lower())
        self.assertNotIn("待填写", proposal)

    def test_readme_uses_the_upstream_checkout_layout(self) -> None:
        readme_path = ROOT / "README.md"

        self.assertTrue(readme_path.is_file())

        readme = readme_path.read_text(encoding="utf-8")

        self.assertIn(
            "skills/competition/ninetooth-operator-dev/SKILL.md",
            readme,
        )
        self.assertNotIn(
            "git clone https://github.com/LaiQuan-conquer/"
            "NineToothed-OperatorSkills.git",
            readme,
        )

    def test_chinese_readme_uses_the_upstream_checkout_layout(self) -> None:
        readme_path = ROOT / "README_zh.md"

        self.assertTrue(readme_path.is_file())

        readme = readme_path.read_text(encoding="utf-8")

        self.assertIn(
            "skills/competition/ninetooth-operator-dev/SKILL.md",
            readme,
        )
        self.assertNotIn(
            "git clone https://github.com/LaiQuan-conquer/"
            "NineToothed-OperatorSkills.git",
            readme,
        )

    def test_proposal_describes_the_shipped_reference_files(self) -> None:
        proposal = (ROOT / "proposal.md").read_text(encoding="utf-8")

        for obsolete_path in sorted(OBSOLETE_PROPOSAL_PATHS):
            with self.subTest(path=obsolete_path):
                self.assertNotIn(obsolete_path, proposal)

        for shipped_path in (
            "dsl-pattern-index.md",
            "failure-playbook.md",
            "operator-task-contract.md",
            "repo-map.md",
            "verification-matrix.md",
        ):
            with self.subTest(path=shipped_path):
                self.assertIn(shipped_path, proposal)

    def test_local_markdown_links_resolve(self) -> None:
        broken_links: list[str] = []

        for document in ROOT.rglob("*.md"):
            text = document.read_text(encoding="utf-8")

            for target in MARKDOWN_LINK_PATTERN.findall(text):
                path_text = target.split("#", maxsplit=1)[0]

                if (
                    not path_text
                    or "://" in path_text
                    or path_text.startswith("mailto:")
                ):
                    continue

                if not (document.parent / path_text).resolve().exists():
                    broken_links.append(f"{document.relative_to(ROOT)} -> {target}")

        self.assertEqual(broken_links, [])

    def test_skill_checks_run_from_the_upstream_repository_root(self) -> None:
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

        self.assertIn(
            "python skills/competition/ninetooth-operator-dev/scripts/"
            "lint_skill_structure.py skills/competition/ninetooth-operator-dev",
            skill,
        )
        self.assertIn(
            "python skills/competition/ninetooth-operator-dev/tests/test_structure.py",
            skill,
        )

    def test_final_submission_blocker_is_explicit(self) -> None:
        proposal = (ROOT / "proposal.md").read_text(encoding="utf-8")
        final_report = (ROOT / "FINAL_REPORT.md").read_text(encoding="utf-8")

        for marker in (
            "Final submission status",
            "23 collection errors",
            "not achieved results",
        ):
            with self.subTest(document="proposal.md", marker=marker):
                self.assertIn(marker, proposal)

        for marker in (
            "Validation Environment Blocker",
            "No matching distribution found for triton>=3.0.0",
            "23 errors during collection",
        ):
            with self.subTest(document="FINAL_REPORT.md", marker=marker):
                self.assertIn(marker, final_report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
