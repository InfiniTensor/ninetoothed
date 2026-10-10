# Example 01 — elementwise / broadcast add

## Task description

Implement broadcast add with NineToothed arrange-and-apply:

\[
\mathrm{out}[i,j] = a[i,0] + b[0,j]
\]

Shapes: `a` `(M,1)`, `b` `(1,N)`, `out` `(M,N)`.

**Correctness dtypes:** `float32` and `float16` (fp16 uses explicit atol/rtol in pytest).  
**Benchmark dtype:** `float32` only — do not extend performance claims to float16.

## Workflow (must follow SKILL.md D1–D9)

1. **Task card** → see `task_card.md`
2. **Route** → elementwise / broadcast (`references/03_*`)
3. **`rg` in repo** (from `--repo-root`):

```bash
rg -n "arrangement|ninetoothed.make|broadcast|expand" tests/test_add.py tests/test_expand.py
rg -n "def add|broadcast" tests/ -g "*.py" | head
```

4. **Minimal implementation** → `solution/broadcast_add.py` + `tests/test_example_broadcast_add.py`
5. **pytest / verify** → see `correctness_result.md`
6. **Benchmark** → optional micro-bench in `verify.py` (float32 only; see `benchmark_result.md`)

## Deliverables in this directory

`task_card.md`, `run_prompt.md`, `run_log.md`, `changed_files.md`,
`correctness_result.md`, `benchmark_result.md`, `solution/`, `tests/`, `verify.py`.
