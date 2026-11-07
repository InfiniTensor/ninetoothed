Use skill `ninetoothed-op-dev-skill` (SKILL.md decision tree D1–D9).

## Task

Broadcast add: `a` shape `(M,1)`, `b` shape `(1,N)`, `out = a + b` → `(M,N)`, float32.

## Required process

1. Write a task card (math, shape, dtype, broadcast, layout, boundaries, reference).
2. From the NineToothed repo root, `rg` nearest patterns in `tests/test_add.py` / `tests/test_expand.py` **before** coding.
3. Add a **minimal** `ninetoothed.make` kernel + pytest under this example directory.
4. Run correctness; optionally micro-bench vs `torch.add` with warmup.
5. Do not invent APIs; do not edit compiler core.
