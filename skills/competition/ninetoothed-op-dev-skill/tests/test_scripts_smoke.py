"""Smoke tests for helper scripts (--help and no-GPU paths)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

REPO_ACCESS_SCRIPTS = {
    "env_check.py",
    "run_correctness.py",
    "run_benchmark.py",
    "repo_pattern_index.py",
    "check_patch_minimality.py",
}

TEXT_ONLY_SCRIPTS = {
    "make_task_card.py",
    "score_task.py",
    "summarize_run.py",
    "quick_validate.py",
}

MIXED_LAYOUT_TASK = """# Task — mixed_layout_probe

## 1. Background

Implement **masked where** with mixed layouts.

## 2. Operator contract

- input `x` (M, N) float16 non-contiguous
- input `y` (M, N) float16 contiguous
- input `mask` (M, N) bool contiguous
- output `out` (M, N) float16

## 3. Layout / broadcast / dtype

Per-tensor layouts as listed in inputs. Broadcast: none.

## 4. Constraints

M,N ≥ 1; mask must be bool.

## 5. Correctness requirements

pytest vs `torch.where(mask, x, y)`.

## 6. Benchmark requirements

Benchmark not required.
"""

MISSING_FIELDS_TASK = """# Task — missing_fields_probe

## 1. Background

Implement **elementwise add**.

## 2. Operator contract

Two tensors; details omitted on purpose.
"""

EXPLICIT_BENCH_NA_TASK = """# Task — bench_na_probe

## 1. Background

Implement **elementwise add**.

## 2. Operator contract

| Tensor | Shape | dtype | Layout |
|--------|-------|-------|--------|
| a | (N,) | float32 | contiguous |
| b | (N,) | float32 | contiguous |

| Tensor | Shape | dtype |
|--------|-------|-------|
| output | (N,) | float32 |

semantics = `torch.add(a, b)`.

## 3. Layout / broadcast / dtype

Contiguous. Broadcast: none.

## 4. Constraints

N ≥ 1.

## 5. Correctness requirements

pytest vs `torch.add`.

## 6. Benchmark requirements

Benchmark not required.
"""

OPERATOR_NOT_INPUT_TASK = """# Task — op_name_not_input

## 1. Background

Implement **elementwise mul**.

## 2. Operator contract

semantics = `mul(x, y)`; `x` (N,) float32 contiguous; `y` (N,) float32 contiguous; output (N,) float32.

## 3. Layout / broadcast / dtype

Contiguous. Broadcast: none.

## 4. Constraints

N ≥ 1.

## 5. Correctness requirements

pytest vs `torch.mul`.

## 6. Benchmark requirements

Benchmark not required.
"""

INLINE_MIXED_BOUNDARY_TASK = """# Task — inline_mixed_boundary

## 1. Background

Implement **masked where** with mixed layouts.

## 2. Operator contract

Inputs: `x` (M,1) float16 non-contiguous; `y` (1,N) float16 contiguous; `mask` (M,N) bool contiguous. Output `out` (M,N) float16.

## 3. Layout / broadcast / dtype

Per-tensor layouts as listed. Broadcast: none.

## 4. Boundary cases

Odd M,N and all-false mask.

## 5. Correctness requirements

pytest vs `torch.where(mask, x, y)`.

## 6. Benchmark requirements

Benchmark not required.
"""


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=SCRIPTS,
        capture_output=True,
        text=True,
    )


def test_scripts_help():
    for name in sorted(REPO_ACCESS_SCRIPTS | TEXT_ONLY_SCRIPTS):
        proc = _run([name, "--help"])
        assert proc.returncode == 0, f"{name}: {proc.stderr}"
        if name in REPO_ACCESS_SCRIPTS:
            assert "--repo-root" in proc.stdout, name
        if name in TEXT_ONLY_SCRIPTS:
            assert "--repo-root" not in proc.stdout, name


def test_score_task_writes(tmp_path: Path):
    out = tmp_path / "sc.md"
    proc = _run(
        [
            "score_task.py",
            "template",
            "--task-id",
            "demo_elementwise_add",
            "--output",
            str(out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    assert "do not copy pytest PASS" in out.read_text(encoding="utf-8")


def test_make_task_card_mixed_layout_per_tensor(tmp_path: Path):
    task = tmp_path / "task.md"
    task.write_text(MIXED_LAYOUT_TASK, encoding="utf-8")
    out = tmp_path / "task_card.md"
    proc = _run(
        [
            "make_task_card.py",
            "--task-file",
            str(task),
            "--output",
            str(out),
            "--strict",
        ]
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    text = out.read_text(encoding="utf-8")
    assert "| x | (M,N) | float16 | non-contiguous |" in text.replace(" ", "") or (
        "| x |" in text and "float16" in text and "non-contiguous" in text
    )
    # Per-tensor layouts must not be collapsed to one global value.
    assert "non-contiguous" in text
    assert "| y |" in text and "contiguous" in text
    assert "| mask |" in text and "bool" in text
    # y and mask contiguous; x non-contiguous — all present
    lines = [
        ln
        for ln in text.splitlines()
        if ln.startswith("| x ") or ln.startswith("| y ") or ln.startswith("| mask ")
    ]
    by_name = {}
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) >= 4:
            by_name[cells[0]] = cells
    assert by_name["x"][2] == "float16" and by_name["x"][3] == "non-contiguous"
    assert by_name["y"][2] == "float16" and by_name["y"][3] == "contiguous"
    assert by_name["mask"][2] == "bool" and by_name["mask"][3] == "contiguous"
    assert "Not required" in text


def test_make_task_card_missing_fields_strict_fails(tmp_path: Path):
    task = tmp_path / "task.md"
    task.write_text(MISSING_FIELDS_TASK, encoding="utf-8")
    out = tmp_path / "task_card.md"
    proc = _run(
        [
            "make_task_card.py",
            "--task-file",
            str(task),
            "--output",
            str(out),
            "--strict",
        ]
    )
    assert proc.returncode == 1, proc.stdout
    assert (
        "TBD" in proc.stderr
        or "TODO" in proc.stderr
        or "invalid" in proc.stderr.lower()
    )


def test_make_task_card_explicit_benchmark_not_required(tmp_path: Path):
    task = tmp_path / "task.md"
    task.write_text(EXPLICIT_BENCH_NA_TASK, encoding="utf-8")
    out = tmp_path / "task_card.md"
    proc = _run(
        [
            "make_task_card.py",
            "--task-file",
            str(task),
            "--output",
            str(out),
            "--strict",
        ]
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Not required" in out.read_text(encoding="utf-8")


def test_make_task_card_operator_name_not_input_tensor(tmp_path: Path):
    task = tmp_path / "task.md"
    task.write_text(OPERATOR_NOT_INPUT_TASK, encoding="utf-8")
    out = tmp_path / "task_card.md"
    proc = _run(
        [
            "make_task_card.py",
            "--task-file",
            str(task),
            "--output",
            str(out),
            "--strict",
        ]
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    text = out.read_text(encoding="utf-8")
    assert "| mul |" not in text.lower().replace(" ", "") or "| mul |" not in text
    in_inputs = False
    for ln in text.splitlines():
        if ln.strip().startswith("## Inputs"):
            in_inputs = True
            continue
        if in_inputs and ln.strip().startswith("## "):
            break
        if in_inputs and ln.strip().startswith("|"):
            assert not ln.strip().lower().startswith("| mul |"), ln
    assert "| x |" in text and "| y |" in text


def _io_rows_by_name(card_text: str) -> dict[str, list[str]]:
    by_name: dict[str, list[str]] = {}
    for ln in card_text.splitlines():
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0].lower() in {"tensor", "--------", ""}:
            continue
        by_name[cells[0]] = cells
    return by_name


def test_make_task_card_inline_mixed_layout_and_boundary(tmp_path: Path):
    task = tmp_path / "task.md"
    task.write_text(INLINE_MIXED_BOUNDARY_TASK, encoding="utf-8")
    out = tmp_path / "task_card.md"
    proc = _run(
        [
            "make_task_card.py",
            "--task-file",
            str(task),
            "--output",
            str(out),
            "--strict",
        ]
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    text = out.read_text(encoding="utf-8")
    by_name = _io_rows_by_name(text)
    assert by_name["x"][2] == "float16" and by_name["x"][3] == "non-contiguous"
    assert by_name["y"][2] == "float16" and by_name["y"][3] == "contiguous"
    assert by_name["mask"][2] == "bool" and by_name["mask"][3] == "contiguous"
    assert by_name["out"][2] == "float16"
    assert "Odd M,N and all-false mask" in text
    assert "Not required" in text


def test_make_task_card_three_formats_same_per_tensor_attrs(tmp_path: Path):
    """Bullet / inline / markdown table must agree on per-tensor dtype/layout."""
    common_tail = """
## 3. Layout / broadcast / dtype

Per-tensor layouts as listed. Broadcast: none.

## 4. Boundary cases

Odd M,N and all-false mask.

## 5. Correctness requirements

pytest vs reference.

## 6. Benchmark requirements

Benchmark not required.
"""
    bullet = (
        """# Task — fmt_bullet

## 1. Background

Implement **masked where**.

## 2. Operator contract

- input `x` (M,1) float16 non-contiguous
- input `y` (1,N) float16 contiguous
- input `mask` (M,N) bool contiguous
- output `out` (M,N) float16
"""
        + common_tail
    )
    inline = (
        """# Task — fmt_inline

## 1. Background

Implement **masked where**.

## 2. Operator contract

Inputs: `x` (M,1) float16 non-contiguous; `y` (1,N) float16 contiguous; `mask` (M,N) bool contiguous. Output `out` (M,N) float16.
"""
        + common_tail
    )
    table = (
        """# Task — fmt_table

## 1. Background

Implement **masked where**.

## 2. Operator contract

| Tensor | Shape | dtype | Layout |
|--------|-------|-------|--------|
| x | (M,1) | float16 | non-contiguous |
| y | (1,N) | float16 | contiguous |
| mask | (M,N) | bool | contiguous |

| Tensor | Shape | dtype |
|--------|-------|-------|
| out | (M,N) | float16 |
"""
        + common_tail
    )

    results = []
    for label, body in (
        ("bullet", bullet),
        ("inline", inline),
        ("table", table),
    ):
        task = tmp_path / f"{label}.md"
        out = tmp_path / f"{label}_card.md"
        task.write_text(body, encoding="utf-8")
        proc = _run(
            [
                "make_task_card.py",
                "--task-file",
                str(task),
                "--output",
                str(out),
                "--strict",
            ]
        )
        assert proc.returncode == 0, f"{label}: {proc.stderr}"
        by_name = _io_rows_by_name(out.read_text(encoding="utf-8"))
        key = {
            "x": (by_name["x"][2], by_name["x"][3]),
            "y": (by_name["y"][2], by_name["y"][3]),
            "mask": (by_name["mask"][2], by_name["mask"][3]),
            "out": (by_name["out"][2],),
        }
        results.append(key)
    assert results[0] == results[1] == results[2]
    assert results[0]["x"] == ("float16", "non-contiguous")
    assert results[0]["y"] == ("float16", "contiguous")
    assert results[0]["mask"] == ("bool", "contiguous")
    assert results[0]["out"] == ("float16",)
