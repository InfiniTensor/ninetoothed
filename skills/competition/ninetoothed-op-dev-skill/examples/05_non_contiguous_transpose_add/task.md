# Example 05 — layout / stride (non-contiguous transpose add)

## Task description

Add two **non-contiguous** CUDA tensors with NineToothed without illegally calling `.contiguous()`:

\[
\mathrm{out} = a + b
\]

Inputs are strided views (e.g. transpose-derived). Tests must assert `not a.is_contiguous()`.

## Workflow (must follow SKILL.md D1–D9)

1. **Task card** → see `task_card.md` (layout ≠ contiguous-only)
2. **Route** → layout / stride (`references/05_*`)
3. **`rg` in repo** (from `--repo-root`):

```bash
rg -n "is_contiguous|stride|as_strided|transpose" tests/test_clone.py tests/test_data_ptr.py
rg -n "def arrangement|ninetoothed.make" tests/test_add.py
```

4. **Minimal implementation** → `solution/strided_add_kernel.py` + `tests/test_example_strided_add.py`
5. **pytest / verify** → see `correctness_result.md`
6. **Benchmark** → N/A (layout contract demo; see `benchmark_result.md`)

## Deliverables in this directory

`task_card.md`, `run_prompt.md`, `run_log.md`, `changed_files.md`,
`correctness_result.md`, `benchmark_result.md`, `solution/`, `tests/`, `verify.py`.
