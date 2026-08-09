# Task-to-Example Index

Every NineToothed operator task maps to an existing, tested operator in the repo.
**Copy its structure first, then adapt.** This is faster and far more reliable
than writing from memory.

## Lane → reference file → what to copy

### L1 — Elementwise / broadcast
- **Read:** `tests/test_add.py`
- **Copy:** the `@ninetoothed.jit` decorator form; each tensor annotated
  `Tensor(1).tile((BLOCK_SIZE,))`; body `output = lhs + rhs`.
- **Adapt for broadcast:** give inputs different arranged shapes and align the
  outermost shapes with `expand`.
- **Adapt for masking/boundary:** construct inputs with `Tensor(..., other=0)` (or
  `other=float("-inf")` for max reductions).
- **Test target:** `expected = input <op> other`.

### L2 — Reduction / block
- **Read:** `tests/test_softmax.py` (row reduction), `tests/test_max_pool2d.py`
  (windowed reduction with `ravel`/`flatten`).
- **Copy (softmax):** `tile((1, BLOCK_SIZE))`, `BLOCK_SIZE = Symbol(..., constexpr=True)`
  set to the row width, body uses `ntl.max`, `ntl.exp`, `ntl.sum`.
- **Copy (pooling):** `tile((1,1,WIN_H,WIN_W))` → `.ravel()` → `.flatten(...)` →
  `.tile((BLOCK_SIZE,-1))`, body `ntl.max(input, axis=1)`.
- **Critical:** out-of-bounds fill via `Tensor(..., other=float("-inf"))` for max,
  `other=0` for sum. Forgetting this corrupts boundary blocks.
- **Test target:** `torch.softmax(x, dim=-1)`, `F.max_pool2d(...)`, etc.

### L3 — Layout-sensitive
- **Read:** `tests/test_getitem.py` (slicing/stride semantics),
  `tests/test_matmul.py` (`permute`/`expand`/`squeeze` composition).
- **Copy:** use `permute` for transposed inputs; use `tile(..., strides=...)` for
  strided access.
- **Critical:** do NOT assume contiguous. Add a test that feeds a non-contiguous
  input (`x.T`, `x[::2]`, `x.narrow(...)`) and compare to the contiguous result.
- **Test target:** the same PyTorch op applied to the non-contiguous tensor.

### L4 — Performance / diagnosis / integration
- **Read:** `tests/test_matmul.py` (accumulator + `ntl.dot` loop),
  `docs/source/build.rst` (AOT build), `src/ninetoothed/debugging.py`
  (`simulate_arrangement`).
- **Benchmark:** `scripts/bench.py` wraps `triton.testing.do_bench`.
- **Generated source:** pass `kernel_name=` and `output_dir=` to `ninetoothed.make`.
- **AOT:** `ninetoothed.build(premake, configs, meta_parameters=..., lazy=True)`;
  needs `output_dir` to exist and an `if __name__ == "__main__":` guard.
- **Failing test:** reproduce → read the first real error → consult
  `failure-recovery.md` → minimal fix → re-run.

## Quick decision hints

- "broadcast", "mask", "elementwise", named math op → **L1**
- "softmax", "sum", "max", "mean", "norm", "pool", "reduce" → **L2**
- "transpose", ".T", "stride", "offset", "slice", "non-contiguous", "view" → **L3**
- "benchmark", "generated source", "AOT", "build", "regression", "failing test" → **L4**
