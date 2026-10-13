# Self-test T2 — Reduction / block lane

## 1. Task description
- **Operator type:** row-wise softmax over the last dimension.
- **Inputs:** one 2-D tensor `(m, n)`.
- **Output:** one 2-D tensor `(m, n)`, softmax applied per row.
- **Shape constraints:** output matches input; reduction over the last axis.
- **Dtype constraints:** float32 (tested).
- **Layout constraints:** contiguous; each row tiled as `(1, BLOCK_SIZE)`.
- **Boundary cases:** `BLOCK_SIZE` (row width) may exceed the tile; out-of-bounds
  lanes filled with `-inf` via `Tensor(2, other=float("-inf"))` so they do not
  corrupt the max/sum reduction. This is the key reduction-lane rule.

## 2. Agent execution summary (skill workflow)
- **Step 0 classify:** softmax / reduction → **L2**.
- **Step 1 read repo:** `tests/test_softmax.py` (per `task-to-example-index.md`).
- **Step 2/3 semantics/layout:** reduction over last dim; needs `-inf` fill.
- **Step 4 implement:** row tiled `(1, BLOCK_SIZE)`, `BLOCK_SIZE=constexpr` set to
  row width at call time; body uses `ntl.max`, `ntl.exp`, `ntl.sum` for a
  numerically stable softmax (subtract row max first).
- **Step 5 verify:** PyTorch reference `torch.softmax(input, dim=-1)`.

## 3. Files
- `test_softmax_op.py` — operator + test.

## 4. Correctness test
Command:
```bash
pytest skills/competition/ninetoothed-operator-skill/examples/t2-reduction-block/test_softmax_op.py -v -p no:cacheprovider
```
Result:
```text
test_softmax_op.py::test[1823-781-dtype0-cuda] PASSED
1 passed
```
Environment: WSL2 Ubuntu, RTX 5060, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1.

## 5. Benchmark
See `../t4-performance-diagnosis/` — T4 benchmarks the elementwise kernel. The
reduction benchmark can reuse `scripts/bench.py` with `torch.softmax` as baseline.

## 6. Failure diagnosis
None for this task — passed on first run. The `-inf` boundary fill is the pitfall
the skill pre-empts (see `references/failure-recovery.md` row #2).
