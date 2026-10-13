# Self-test T3 — Layout-sensitive lane

## 1. Task description
- **Operator type:** 2-D elementwise multiply, tested on contiguous AND
  non-contiguous (transposed) inputs.
- **Inputs:** two 2-D tensors `(m, n)`.
- **Output:** one 2-D tensor `(m, n)`.
- **Shape constraints:** all three shapes equal.
- **Dtype constraints:** float32 (tested).
- **Layout constraints:** MUST work for non-contiguous inputs (swapped strides),
  not only contiguous ones.
- **Boundary cases:** dims not divisible by block sizes — handled by tiling.

## 2. Agent execution summary (skill workflow)
- **Step 0 classify:** primary L1, but layout is the graded concern → **L3**.
- **Step 1 read repo:** `tests/test_getitem.py`, `tests/test_matmul.py`.
- **Step 3 layout (the key step):** did NOT assume contiguous. Designed a second
  test that builds `(n, m)` then transposes to `(m, n)` — a non-contiguous tensor
  with swapped strides — and asserts `not lhs.is_contiguous()` before running.
- **Step 4 implement:** 2-D tile `(BLOCK_SIZE_M, BLOCK_SIZE_N)`, body `lhs * rhs`.
- **Step 5 verify:** PyTorch reference on the SAME non-contiguous tensor.

## 3. Files
- `test_multiply_2d.py` — operator + contiguous test + non-contiguous test.

## 4. Correctness test
Command:
```bash
pytest skills/competition/ninetoothed-operator-skill/examples/t3-layout-sensitive/test_multiply_2d.py -v -p no:cacheprovider
```
Result:
```text
test_multiply_2d.py::test_contiguous[384-512-dtype0-cuda] PASSED
test_multiply_2d.py::test_non_contiguous[384-512-dtype0-cuda] PASSED
2 passed
```
Environment: WSL2 Ubuntu, RTX 5060, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1.

**Significance:** the non-contiguous case passing proves NineToothed handles
swapped strides correctly here, and proves the skill's mandatory "test a
non-contiguous input" step is real and executable — the exact step most agents
skip, losing the two layout-sensitive hidden tasks.

## 5. Failure diagnosis
None for this task — both layouts passed. Had the non-contiguous case failed, the
skill routes to `references/failure-recovery.md` row #3 (assumed contiguous →
handle strides explicitly / `permute`).
