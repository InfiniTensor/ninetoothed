# Task 4 — softmax performance-regression diagnosis

**Family:** performance / diagnosis / integration.

**Statement.** A row-softmax kernel runs slower than it should. Diagnose the
regression with evidence and apply a minimal fix; both versions must stay
numerically correct vs `torch.softmax`.

**Setup.** `softmax_slow` uses an oversized `BLOCK_SIZE` (8192) so each row runs
exp/max/sum over mostly-masked `-inf` lanes; `softmax_fast` uses `BLOCK_SIZE = N`.

**Diagnosis workflow.**
1. Benchmark: `python bench_softmax.py` → `bench.csv` (fast vs slow vs torch).
2. Inspect generated source: `python ../../scripts/inspect_generated_source.py`
   for each variant; compare the tile size / masked-lane count.
3. Conclude: the slow variant wastes **vector compute/occupancy** on masked
   lanes (masked loads fetch no DRAM, so it is not extra bandwidth). The cost is
   exposed when `SLOW_BLOCK >> N` (~1.6-1.9x at N<=1024) and shrinks toward
   parity as N approaches the block size and the kernel becomes bandwidth-bound.
4. Minimal fix: set `BLOCK_SIZE = x.shape[-1]` (the fast wrapper). No kernel
   rewrite needed.

**Verify.** `python ../../scripts/run_correctness_matrix.py test_softmax_variants.py`
(both variants must pass), then `bench.csv` shows fast < slow latency.

**Not supported.** Rows longer than the largest supported tile; softmax over a
non-last axis (transpose first).
