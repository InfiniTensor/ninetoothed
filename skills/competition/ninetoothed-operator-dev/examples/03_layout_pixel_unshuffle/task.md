# Task 3 — layout-sensitive (non-contiguous materialization)

**Family:** layout-sensitive (non-contiguous / stride / offset).

**Statement.** Correctly handle a non-contiguous input in a NineToothed kernel.
The runnable core materializes `x.t()` (a non-contiguous view) into a contiguous
tensor *through* a NineToothed copy — so NineToothed performs the strided read,
not torch. Cover non-square and non-power-of-two shapes, fp16 and fp32.

**Why this is the layout primitive.** Space-to-depth ops (pixel_unshuffle),
`flip`, `narrow`, and any strided/offset input all reduce to "read a
non-contiguous logical layout and write contiguous." Getting the transpose copy
right is the prerequisite; the extension below builds on it.

**Extension (documented, oracle provided).** `pixel_unshuffle(x, r)` =
`x.view(B,C,H//r,r,W//r,r).permute(0,1,3,5,2,4)` materialized contiguous. Same
non-contiguous-read primitive on a 6-D view; `reference_pixel_unshuffle` is the
oracle. Validate the transpose copy first, then extend the arrangement.

**Verify.** `python ../../scripts/run_correctness_matrix.py test_layout_transpose.py`

**Confidence.** MEDIUM-HIGH: the copy is trivial; the validated behavior is that
NineToothed honors the source tensor's runtime strides on a non-contiguous read.
Use `simulate_arrangement` (perf-diag.md) to confirm the tiling first.

**Not supported (v0).** In-arrangement reverse-stride `flip`; the 6-D
pixel_unshuffle arrangement (oracle provided, kernel is the next increment).
