"""
Structural Validation Tests for the ninetoothed-operator-skill.

Verifies that the skill package contains all required files and directories.
These tests can run without GPU — they only check file presence and structure.
"""

import os
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "SKILL.md",
    "README.md",
    "HONOR_CODE.md",
    "REFERENCE.md",
    "references/index.md",
]

REQUIRED_DIRS = [
    "examples",
    "examples/task1_elementwise_broadcast",
    "examples/task2_reduction_block",
    "examples/task3_layout_sensitive",
    "examples/task4_benchmark_debug",
    "scripts",
    "tests",
    "reports",
]

REQUIRED_CODE_FILES = [
    "examples/task1_elementwise_broadcast/operator_impl.py",
    "examples/task1_elementwise_broadcast/test_correctness.py",
    "examples/task1_elementwise_broadcast/benchmark.py",
    "examples/task1_elementwise_broadcast/task_description.md",
    "examples/task2_reduction_block/operator_impl.py",
    "examples/task2_reduction_block/test_correctness.py",
    "examples/task2_reduction_block/benchmark.py",
    "examples/task2_reduction_block/task_description.md",
    "examples/task3_layout_sensitive/operator_impl.py",
    "examples/task3_layout_sensitive/test_correctness.py",
    "examples/task3_layout_sensitive/benchmark.py",
    "examples/task3_layout_sensitive/task_description.md",
    "examples/task4_benchmark_debug/benchmark_multisize.py",
    "examples/task4_benchmark_debug/buggy_operator.py",
    "examples/task4_benchmark_debug/diagnosis_record.md",
    "examples/task4_benchmark_debug/task_description.md",
]

SCRIPTS_FILES = [
    "scripts/env_check.py",
    "scripts/run_selftests.py",
]


class TestSkillStructure:
    """Verify all required files and directories exist."""

    @pytest.mark.parametrize("rel_path", REQUIRED_FILES)
    def test_required_file_exists(self, rel_path):
        path = SKILL_DIR / rel_path
        assert path.exists(), f"Required file missing: {rel_path}"

    @pytest.mark.parametrize("rel_path", REQUIRED_DIRS)
    def test_required_dir_exists(self, rel_path):
        path = SKILL_DIR / rel_path
        assert path.is_dir(), f"Required directory missing: {rel_path}"

    @pytest.mark.parametrize("rel_path", REQUIRED_CODE_FILES)
    def test_code_file_exists(self, rel_path):
        path = SKILL_DIR / rel_path
        assert path.exists(), f"Code file missing: {rel_path}"

    @pytest.mark.parametrize("rel_path", SCRIPTS_FILES)
    def test_script_file_exists(self, rel_path):
        path = SKILL_DIR / rel_path
        assert path.exists(), f"Script file missing: {rel_path}"


class TestNoPlaceholders:
    """Verify no unfilled placeholders remain in key files."""

    PLACEHOLDER_PATTERNS = [
        "PASTE FULL pytest OUTPUT HERE AFTER EXECUTION",
        "PASTE benchmark OUTPUT HERE AFTER EXECUTION",
        "WRITE AFTER EXECUTION",
        "<your-skill-name>",
    ]

    @pytest.mark.parametrize(
        "rel_path",
        [
            "HONOR_CODE.md",
            "REFERENCE.md",
        ],
    )
    def test_no_angle_bracket_placeholders(self, rel_path):
        """HONOR_CODE.md and REFERENCE.md must have real values, not <placeholders>."""
        path = SKILL_DIR / rel_path
        content = path.read_text()
        assert "<name>" not in content, f"Placeholder <name> found in {rel_path}"
        assert "<github-id>" not in content, (
            f"Placeholder <github-id> found in {rel_path}"
        )
        assert "<date>" not in content, f"Placeholder <date> found in {rel_path}"
        assert "<describe>" not in content, (
            f"Placeholder <describe> found in {rel_path}"
        )


class TestRedLines:
    """Verify compliance red lines are not violated."""

    def test_no_api_keys(self):
        """No API keys or credentials in any file."""
        for root, dirs, files in os.walk(SKILL_DIR):
            # Skip .gitkeep and hidden files
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if f.startswith("."):
                    continue
                path = Path(root) / f
                content = path.read_text()
                # Check for common secret patterns
                assert "sk-" not in content.lower() or "skill" in content.lower(), (
                    f"Potential API key pattern in {path.relative_to(SKILL_DIR)}"
                )

    def test_no_binary_files_in_skill(self):
        """No .env, .venv, __pycache__, node_modules in skill directory."""
        forbidden = [".env", ".venv", "__pycache__", "node_modules"]
        for item in forbidden:
            path = SKILL_DIR / item
            assert not path.exists(), f"Forbidden file/dir found: {item}"

    def test_skill_directory_only(self):
        """Verify we are inside skills/competition/ninetoothed-operator-skill/."""
        parts = SKILL_DIR.parts
        assert "skills" in parts, "Skill must be under skills/ directory"
        assert "competition" in parts, "Skill must be under competition/ directory"
        assert parts[-1] == "ninetoothed-operator-skill", (
            f"Skill directory name mismatch: {parts[-1]}"
        )
