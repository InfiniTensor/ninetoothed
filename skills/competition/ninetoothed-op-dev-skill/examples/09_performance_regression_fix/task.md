# Example 09 — performance / diagnosis (block-size regression)

## Task description

Compare the **same** elementwise-add kernel under two fixed `BLOCK_SIZE` values
(e.g. 32 vs 1024) with a fair micro-benchmark **after** correctness (SKILL.md **D8**).

## Workflow (must follow SKILL.md D1–D9)

1. **Task card** → see `task_card.md`
2. **Route** → perf / diagnosis (`references/07_*`)
3. **`rg` in repo** (from `--repo-root`):

```bash
rg -n "BLOCK_SIZE|constexpr|meta=True" tests/test_generation.py tests/test_add.py
rg -n "Symbol\\(\"BLOCK" tests/ -g "*.py" | head
```

4. **Minimal implementation** → `solution/add_tunable.py` + `tests/test_example_block_size.py`
5. **Correctness first** at **each** `BLOCK_SIZE` to be timed
6. **Benchmark** → `verify.py` (see `benchmark_result.md`); diagnose absurd ratios in `failure_diagnosis.md`

## Deliverables in this directory

`task_card.md`, `run_prompt.md`, `run_log.md`, `changed_files.md`,
`correctness_result.md`, `benchmark_result.md`, `failure_diagnosis.md`,
`solution/`, `tests/`, `verify.py`.
