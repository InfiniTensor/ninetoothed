# Run log — example 03

## D1 Task card

See `task_card.md`.

## D2–D3 Route + rg

Family: **reduction / block**.

```bash
cd "$REPO"   # NineToothed root (has src/ninetoothed/)
rg -n "softmax|ntl.max|ntl.sum" tests/test_softmax.py
rg -n "tile\(\(1," tests/ -g "*softmax*"
```

Nearest pattern: `tests/test_softmax.py` (row tile `(1, BLOCK_SIZE)`, max/exp/sum).

## D4 Minimal patch

- Prefer **no** fork `src/` change — run upstream test.
- Optional local demo: `solution/softmax_kernel.py` + `verify.py` (same arrange-and-apply idea).

## D6 Correctness

```bash
pytest tests/test_softmax.py -v --tb=short
# optional local demo:
python .../examples/03_rowwise_softmax_reduce/verify.py
```

## D8 Benchmark

**N/A** for the default correctness trajectory (see `benchmark_result.md`).
