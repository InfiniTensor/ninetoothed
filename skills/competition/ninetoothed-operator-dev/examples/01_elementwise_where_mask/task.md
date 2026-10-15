# Task 1 — masked elementwise add (broadcast)

**Family:** elementwise / broadcast.

**Statement.** Implement `out = where(mask, a + b, 0)` where `b` broadcasts over
the rows of `a` (`a:(B,N)`, `b:(1,N)`) and `mask` is a 0/1 tensor shaped like
`a`. Support fp16 and fp32, and a non-power-of-two `N`.

**Approach.** Broadcast is resolved in the wrapper (`expand_as().contiguous()`),
then a flat 1-D kernel applies `ntl.where(mask != 0, a + b, 0.0)`. NineToothed
auto-masks the tail block, so `N` need not divide `block_size`.

**Verify.** `python ../../scripts/run_correctness_matrix.py test_masked_add.py`

**Files:** `masked_add.py` (kernel + wrapper + reference), `test_masked_add.py`.

**Not supported.** Broadcasting that is not expressible by `expand_as`
(e.g. ragged); integer mask dtypes other than 0/1.
