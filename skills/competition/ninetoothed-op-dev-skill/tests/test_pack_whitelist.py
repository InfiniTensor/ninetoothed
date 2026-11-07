"""Whitelist pack tests (no GPU)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
PACKER = SCRIPTS / "pack_runtime_skill.py"

RUNTIME_SCRIPTS = {
    "_paths.py",
    "_task_spec.py",
    "env_check.py",
    "quick_validate.py",
    "make_task_card.py",
    "gate_eval.py",
    "score_task.py",
    "run_correctness.py",
    "run_benchmark.py",
    "repo_pattern_index.py",
    "check_patch_minimality.py",
    "summarize_run.py",
    "audit_packed_skill.py",
}


@pytest.mark.skipif(not PACKER.is_file(), reason="packer absent in packed install")
def test_pack_runtime_whitelist_only(tmp_path: Path):
    dst = tmp_path / "runtime_skill"
    proc = subprocess.run(
        [
            sys.executable,
            str(PACKER),
            "--src",
            str(SKILL_ROOT),
            "--dst",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (dst / "SKILL.md").is_file()
    assert (dst / "pyproject.toml").is_file()
    assert (dst / "references" / "00_repo_map.md").is_file()
    assert (dst / "references" / "11_unsupported_cases.md").is_file()
    assert not (dst / "references" / "generated_repo_pattern_index.md").exists()
    assert not (dst / "evals").exists()
    assert not (dst / "submission").exists()
    assert not (dst / "worktree_patches").exists()
    assert not (dst / "tests" / "test_eval_gate.py").exists()
    script_names = {p.name for p in (dst / "scripts").iterdir() if p.is_file()}
    assert script_names == RUNTIME_SCRIPTS
    assert "pack_runtime_skill.py" not in script_names
    assert "task_spec.py" not in script_names
    for bad_prefix in ("bench_", "generate_", "compare_"):
        assert not any(n.startswith(bad_prefix) for n in script_names)
    example_dirs = {p.name for p in (dst / "examples").iterdir() if p.is_dir()}
    assert example_dirs == {
        "01_elementwise_broadcast_add",
        "03_rowwise_softmax_reduce",
        "05_non_contiguous_transpose_add",
        "09_performance_regression_fix",
    }
    for name in example_dirs:
        ex = dst / "examples" / name
        assert (ex / "task.md").is_file()
        assert (ex / "task_card.md").is_file()
        assert (ex / "verify.py").is_file()
        assert not (ex / "scorecard.md").exists()
        assert not any(ex.rglob("*worktree*"))
    assert (
        dst
        / "examples"
        / "01_elementwise_broadcast_add"
        / "solution"
        / "broadcast_add.py"
    ).is_file()
    assert (
        dst
        / "examples"
        / "05_non_contiguous_transpose_add"
        / "tests"
        / "test_example_strided_add.py"
    ).is_file()
    assert (
        dst
        / "examples"
        / "09_performance_regression_fix"
        / "solution"
        / "add_tunable.py"
    ).is_file()
    ref_text = (dst / "REFERENCE.md").read_text(encoding="utf-8")
    assert "Apache-2.0" in ref_text
    assert "submission/" not in ref_text
