# Example 03 — reduction / block (row-wise softmax)

## Task description

Implement or validate row-wise softmax with NineToothed reduction/block patterns:

\[
\mathrm{out}[i,:] = \mathrm{softmax}(\mathrm{in}[i,:])
\]

Stable form: subtract row max, then `exp` / row-sum. Shapes `(M,N)` → `(M,N)`, dtype `float32`.

## Workflow (must follow SKILL.md D1–D9)

1. **Task card** → see `task_card.md`
2. **Route** → reduction / block (`references/04_*`)
3. **`rg` in repo** (from `--repo-root`):

```bash
rg -n "softmax|ntl.max|ntl.sum|ntl.exp" tests/test_softmax.py
rg -n "def arrangement|tile\(\(1," tests/ -g "*.py" | head
```

4. **Prefer upstream first** → if `tests/test_softmax.py` already covers the op, run it before writing a new kernel
5. **Minimal demo** (optional) → `solution/softmax_kernel.py` + `verify.py`
6. **pytest** → see `correctness_result.md`
7. **Benchmark** → N/A for default correctness demo (see `benchmark_result.md`)

## Deliverables in this directory

`task_card.md`, `run_prompt.md`, `run_log.md`, `changed_files.md`,
`correctness_result.md`, `benchmark_result.md`, `solution/`, `verify.py`.
