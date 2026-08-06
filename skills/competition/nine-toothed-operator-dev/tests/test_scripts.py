#!/usr/bin/env python3
"""Tests for the competition helper scripts.

These exercise the four command-line helpers as real subprocesses so the
behavior matches how the skill and report invoke them. They depend only on
the standard library and pytest, not on torch or ninetoothed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = SKILL_DIR / "scripts"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPTS_DIR / name), *args]

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **_base_env()},
    )


def _base_env() -> dict[str, str]:
    import os

    return dict(os.environ)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake_ninetoothed"
    repo.mkdir()
    (repo / "ops.py").write_text(
        "import ninetoothed\n\n\n@ninetoothed.jit\ndef arrangement():\n    pass\n\n\n"
        "def application():\n    pass\n\n\nkernel = ninetoothed.make(arrangement, application)\n",
        encoding="utf-8",
    )
    (repo / "test_ops.py").write_text(
        "import torch\n\n\ndef test_add():\n    assert torch.allclose(torch.zeros(2), torch.zeros(2))\n\n\n"
        "def devices():\n    return get_available_devices()\n",
        encoding="utf-8",
    )
    (repo / "bench.py").write_text(
        "import triton.testing\n\n\ndef timing():\n    return do_bench(lambda: None)\n",
        encoding="utf-8",
    )
    (repo / "layout.py").write_text(
        "def view(x):\n    return x.as_strided((2, 2), (1, 2)).permute(1, 0)\n",
        encoding="utf-8",
    )
    git_dir = repo / ".git"
    git_dir.mkdir()
    (git_dir / "hook.py").write_text("def arrangement():\n    pass\n", encoding="utf-8")

    return repo


def test_scan_repo_reports_categories(fake_repo: Path) -> None:
    result = run_script("scan_repo.py", "--repo", str(fake_repo))

    assert result.returncode == 0, result.stderr
    assert "Python files scanned: 4" in result.stdout
    assert "[operators]" in result.stdout
    assert "[layout]" in result.stdout
    assert "ops.py" in result.stdout


def test_scan_repo_missing_repo_errors(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    result = run_script("scan_repo.py", "--repo", str(missing))

    assert result.returncode != 0
    assert "repo does not exist" in result.stderr


def test_scan_repo_handles_unicode_and_space_path(tmp_path: Path) -> None:
    repo = tmp_path / "有 空格 仓库"
    repo.mkdir()
    (repo / "ops.py").write_text(
        "@ninetoothed.jit\ndef arrangement():\n    pass\n", encoding="utf-8"
    )
    result = run_script("scan_repo.py", "--repo", str(repo))

    assert result.returncode == 0, result.stderr
    assert "Python files scanned: 1" in result.stdout


def test_build_pattern_index_content(fake_repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "index.md"
    result = run_script(
        "build_pattern_index.py", "--repo", str(fake_repo), "--out", str(out)
    )

    assert result.returncode == 0, result.stderr
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "# Repository Pattern Index" in text
    assert "## operator_kernel" in text
    assert "## layout_sensitive" in text
    assert "`ops.py`" in text
    assert "`layout.py`" in text
    assert ".git" not in text


def test_build_pattern_index_missing_repo_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    out = tmp_path / "index.md"
    result = run_script(
        "build_pattern_index.py", "--repo", str(missing), "--out", str(out)
    )

    assert result.returncode != 0
    assert "repo does not exist" in result.stderr
    assert not out.exists()


def test_make_selftest_task_creates_template(tmp_path: Path) -> None:
    out_dir = tmp_path / "08-new-task"
    result = run_script(
        "make_selftest_task.py",
        "--name",
        "rms-norm",
        "--kind",
        "Reduction/block",
        "--out",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    task = out_dir / "task.md"
    assert task.is_file()
    text = task.read_text(encoding="utf-8")
    assert "# Self-Test: rms-norm" in text

    for heading in ("Input Task Statement", "Correctness Command", "Unsupported Cases"):
        assert heading in text


def test_make_selftest_task_refuses_overwrite(tmp_path: Path) -> None:
    out_dir = tmp_path / "08-new-task"
    first = run_script(
        "make_selftest_task.py",
        "--name",
        "a",
        "--kind",
        "b",
        "--out",
        str(out_dir),
    )

    assert first.returncode == 0, first.stderr

    second = run_script(
        "make_selftest_task.py",
        "--name",
        "a",
        "--kind",
        "b",
        "--out",
        str(out_dir),
    )

    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


def test_check_submission_passes_on_real_package() -> None:
    result = run_script("check_submission.py", "--skill-dir", str(SKILL_DIR))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Structure check passed" in result.stdout


def test_check_submission_detects_missing_file(tmp_path: Path) -> None:
    clone = tmp_path / "skill"
    shutil.copytree(
        SKILL_DIR, clone, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (clone / "references" / "testing.md").unlink()
    result = run_script("check_submission.py", "--skill-dir", str(clone))

    assert result.returncode == 1
    assert "Structure check failed" in result.stdout
    assert "references/testing.md" in result.stdout
