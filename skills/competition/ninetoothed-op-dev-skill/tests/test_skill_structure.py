"""Skill package structure tests (no GPU, no fixed competition task IDs)."""

from __future__ import annotations

from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_REFERENCES = [
    "00_repo_map.md",
    "01_ninetoothed_concepts.md",
    "02_arrangement_application_patterns.md",
    "03_elementwise_broadcast_patterns.md",
    "04_reduction_block_patterns.md",
    "05_layout_stride_offset_patterns.md",
    "06_correctness_testing_patterns.md",
    "07_benchmark_patterns.md",
    "08_generated_source_aot_debugging.md",
    "09_failure_diagnosis_playbook.md",
    "10_patch_minimality_checklist.md",
    "11_unsupported_cases.md",
]

CAPABILITY_REFERENCES = {
    "elementwise_broadcast": "03_elementwise_broadcast_patterns.md",
    "reduction_block": "04_reduction_block_patterns.md",
    "layout_sensitive": "05_layout_stride_offset_patterns.md",
    "performance_diagnosis": "07_benchmark_patterns.md",
}

REQUIRED_SCRIPTS = [
    "env_check.py",
    "make_task_card.py",
    "run_correctness.py",
    "run_benchmark.py",
    "repo_pattern_index.py",
    "check_patch_minimality.py",
    "score_task.py",
    "gate_eval.py",
    "summarize_run.py",
    "quick_validate.py",
    "audit_packed_skill.py",
    "_paths.py",
    "_task_spec.py",
]

FORBIDDEN_RUNTIME_PATHS = [
    ".quick_validate_cache",
    "worktree_patches",
]


def test_skill_md_frontmatter():
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: ninetoothed-op-dev-skill" in text
    assert "description:" in text


def test_skill_md_no_answer_artifact_paths():
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for token in (
        "worktree_patches",
        "evals/",
        "submission/",
    ):
        assert token not in text, f"SKILL.md must not hardcode {token}"


def test_skill_md_has_mandatory_decision_tree():
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "## Execution decision tree (MANDATORY)" in text
    required = [
        "### D1 — Emit task card first",
        "### D2 — Route by family",
        "### D3 — `rg` before code",
        "### D5 — Layout branch",
        "assert not inp.is_contiguous()",
        ".contiguous()",
        "### D7 — Failure loop",
        "### D8 — Performance branch",
        "Correctness first",
        "Warmup",
    ]
    for token in required:
        assert token in text, f"missing decision-tree constraint: {token}"


def test_references_complete():
    ref = SKILL_ROOT / "references"
    for name in REQUIRED_REFERENCES:
        assert (ref / name).is_file(), name


def test_capability_categories_documented():
    ref = SKILL_ROOT / "references"
    for family, name in CAPABILITY_REFERENCES.items():
        assert (ref / name).is_file(), family


def test_scripts_exist():
    scripts = SKILL_ROOT / "scripts"
    for name in REQUIRED_SCRIPTS:
        assert (scripts / name).is_file(), name


def test_runtime_excludes_answer_artifacts():
    for rel in FORBIDDEN_RUNTIME_PATHS:
        assert not (SKILL_ROOT / rel).exists(), f"runtime skill must not contain {rel}"


def test_paths_helper_detects_src_layout(tmp_path: Path):
    import sys

    scripts = SKILL_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    import _paths  # noqa: E402

    fake = tmp_path / "ninetoothed"
    (fake / "src" / "ninetoothed").mkdir(parents=True)
    (fake / "tests").mkdir()
    assert _paths.is_ninetoothed_repo(fake)
    assert _paths.ninetoothed_repo_root(fake) == fake.resolve()


def test_reference_md_is_runtime_only():
    text = (SKILL_ROOT / "REFERENCE.md").read_text(encoding="utf-8")
    for token in (
        "submission/",
        "logs/run_all_gates",
        "worktrees/",
    ):
        assert token not in text, (
            f"REFERENCE.md must not keep workspace evidence: {token}"
        )
    assert "Apache-2.0" in text
    assert "external evidence bundle" in text.lower() or "PR description" in text
