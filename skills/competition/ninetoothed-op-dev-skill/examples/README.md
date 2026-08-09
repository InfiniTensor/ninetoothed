# Examples — four capability families (fork-runnable)

Each example is a **complete agent trajectory** for use inside a NineToothed
fork/clone (`--repo-root` = directory containing `src/ninetoothed/`).

| Dir | Family | Runnable entry |
|-----|--------|----------------|
| `01_elementwise_broadcast_add/` | Elementwise / broadcast | `python …/verify.py` |
| `03_rowwise_softmax_reduce/` | Reduction / block | upstream `pytest tests/test_softmax.py` (+ optional `verify.py`) |
| `05_non_contiguous_transpose_add/` | Layout / stride | `python …/verify.py` |
| `09_performance_regression_fix/` | Perf / diagnosis | `python …/verify.py` |

## Uniform artifact set

| File | Role |
|------|------|
| `task.md` | Task description |
| `task_card.md` | D1 card (math / shape / dtype / broadcast / layout / boundaries / reference) |
| `run_prompt.md` | First agent message |
| `run_log.md` | D2–D6/D8 execution notes (`rg` → patch → pytest) |
| `changed_files.md` | Minimal file list |
| `correctness_result.md` | Exact pytest / verify commands |
| `benchmark_result.md` | Numbers protocol **or** N/A + reason |
| `solution/` + `tests/` | Minimal implementation (01/05/09; 03 optional) |
| `verify.py` | One-shot correctness (+ bench when applicable) |

## How to run

Requires editable NineToothed + CUDA for kernel demos:

```bash
cd /path/to/ninetoothed
pip install -e .
source ~/venvs/ninetoothed-skill/bin/activate   # or your venv

EX=/path/to/ninetoothed-op-dev-skill/examples

# 01 elementwise / broadcast
python "$EX/01_elementwise_broadcast_add/verify.py"
pytest "$EX/01_elementwise_broadcast_add/tests" -v --tb=short

# 03 reduction / block (prefer upstream)
pytest tests/test_softmax.py -v --tb=short
python "$EX/03_rowwise_softmax_reduce/verify.py"   # optional local demo

# 05 layout / stride
python "$EX/05_non_contiguous_transpose_add/verify.py"

# 09 perf / diagnosis (correctness then D8)
python "$EX/09_performance_regression_fix/verify.py"
```

These four families are **generic demos** for the D1–D9 workflow. They are not
competition answer keys; always re-run verify/pytest on the target fork.
