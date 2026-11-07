Use skill `ninetoothed-op-dev-skill` (SKILL.md decision tree D1–D9, especially **D5**).

## Task

Elementwise add on non-contiguous (e.g. transposed) float32 CUDA tensors.

## Required process

1. Write a task card with layout = non-contiguous / stride.
2. From the NineToothed repo root, `rg` `tests/test_clone.py` / stride patterns **before** coding.
3. Add a minimal `ninetoothed.make` kernel + pytest under this example; tests must assert non-contiguous inputs.
4. Do **not** call `.contiguous()` unless the card explicitly allows materialization.
5. Run correctness; skip benchmark unless the task asks for D8.
