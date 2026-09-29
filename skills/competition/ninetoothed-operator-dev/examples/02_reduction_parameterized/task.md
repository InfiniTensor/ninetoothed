# Task 2 — parameterized row reduction

**Family:** reduction / blocking.

**Statement.** Reduce the last dim of `x:(B,N)` with `reduction in
{'none','mean','sum'}`. fp16 and fp32; non-power-of-two `N`. Accumulate in fp32.

**Approach.** `reduction='none'` is a pure copy. `sum`/`mean` use a row-tiled
kernel (`tile((1, BLOCK_SIZE))`, `BLOCK_SIZE = N`) that sums the loaded row in
fp32; `mean` divides by `N` in the wrapper. Input declared `other=0.0` so the
tail block's masked lanes contribute 0.

**Confidence note.** The scalar→(1,1) output assignment is the one thing to
validate first on CUDA; if it fails, switch to the `application_axis` form noted
in `reduction.py`. This is exactly the kind of arrangement uncertainty the
skill's verify-arrangement step (`simulate_arrangement`) is meant to catch.

**Verify.** `python ../../scripts/run_correctness_matrix.py test_reduction.py`
then `python bench_reduction.py`.

**Not supported.** Reduction over a non-last axis (transpose first);
`reduction='none'` does not fuse a following op.
