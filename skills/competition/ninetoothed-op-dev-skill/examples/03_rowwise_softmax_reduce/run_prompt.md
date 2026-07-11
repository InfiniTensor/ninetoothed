Use skill `ninetoothed-op-dev-skill` (SKILL.md decision tree D1–D9).

## Task

Row-wise softmax on `(M,N)` float32 tensors (reduction / block family).

## Required process

1. Write a task card (math, shape, dtype, layout, boundaries, reference).
2. From the NineToothed repo root, `rg` `tests/test_softmax.py` and reduction patterns **before** coding.
3. Prefer validating the upstream test; only add a minimal kernel under this example if needed for a local demo.
4. Run correctness (`pytest tests/test_softmax.py` and/or `verify.py`).
5. Do not invent APIs; do not edit compiler core.
