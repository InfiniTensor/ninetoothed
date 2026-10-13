# Self-test T1 — Elementwise / broadcast lane

## 1. Task description
- **Operator type:** elementwise (Hadamard) multiply, `output = lhs * rhs`.
- **Inputs:** two 1-D tensors of equal length.
- **Output:** one 1-D tensor, same shape/dtype.
- **Shape constraints:** `lhs.shape == rhs.shape == output.shape`.
- **Dtype constraints:** float32 (tested); pattern extends to fp16/bf16 with `atol`.
- **Layout constraints:** contiguous 1-D.
- **Boundary cases:** length not divisible by `BLOCK_SIZE` — handled by NineToothed's
  block masking; verified with `size=98432`, a non-round length.

## 2. Agent execution summary (skill workflow)
- **Step 0 classify:** elementwise math → **L1**.
- **Step 1 read repo:** `tests/test_add.py` (per `task-to-example-index.md`).
- **Step 4 implement:** copied the `@ninetoothed.jit` decorator form from `test_add`,
  substituting `+` with `*`. Chosen deliberately different from the repo sample to
  demonstrate generalization, using an equally safe primitive.
- **Step 5 verify:** PyTorch reference `lhs * rhs`, repo-style pytest.

## 3. Files
- `test_multiply.py` — operator + test.

## 4. Correctness test
Command:
```bash
pytest skills/competition/ninetoothed-operator-skill/examples/t1-elementwise-broadcast/test_multiply.py -v -p no:cacheprovider
```
Result:
```text
test_multiply.py::test[98432-dtype0-cuda] PASSED
1 passed
```
Environment: WSL2 Ubuntu, RTX 5060, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1.

## 5. How the skill handles broadcast (lane coverage note)
For genuine broadcasting (e.g. `bias[j]` added across rows), the skill directs the
agent to align the arranged tensors' outermost shapes with `expand` (see
`references/operator-dev-map.md` §4 and the matmul pattern), rather than assuming
equal shapes. This op keeps equal shapes for a minimal, verifiable L1 baseline.

## 6. Failure diagnosis
None for this task — passed on first run.
