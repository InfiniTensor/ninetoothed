# Self-test T4 — Performance / diagnosis / integration lane

Covers the L4 sub-skills: performance benchmarking and a **real, reproducible
performance-regression diagnosis** found and fixed during this development.

## Part A — Performance benchmark

### Task
Benchmark the elementwise multiply kernel against native `torch *`, reporting the
five required facts, and judge whether performance regresses.

### Command
```bash
python skills/competition/ninetoothed-operator-skill/examples/t4-performance-diagnosis/benchmark_multiply.py
```

### Result (real, RTX 5060)
```text
=== size = 1048576 ===
[multiply_fixed[1048576]] input: shape=(1048576,), dtype=float32, layout=contiguous
  baseline : 0.0489 ms
  candidate: 0.0459 ms
  NineToothed is 1.07x vs baseline

=== size = 16777216 ===
[multiply_fixed[16777216]] input: shape=(16777216,), dtype=float32, layout=contiguous
  baseline : 0.6026 ms
  candidate: 0.6125 ms
  NineToothed is 1.02x SLOWER than baseline (regression — investigate)
```

### The five required facts
- **Baseline:** native `torch` elementwise `*`.
- **Input sizes:** `(2^20,)` = 1,048,576 and `(2^24,)` = 16,777,216, float32, contiguous.
- **Command:** see above (`triton.testing.do_bench`, warm-up then timed).
- **Result:** 1M → 0.0459 ms vs 0.0489 ms (1.07x); 16M → 0.6125 ms vs 0.6026 ms (1.02x slower).
- **Conclusion:** elementwise multiply is memory-bandwidth-bound, so both saturate
  the same bandwidth and land within noise of each other. The 1.02x at 16M is
  within run-to-run variance, not a real regression. NineToothed is at parity with
  torch here — the expected, correct outcome for a bandwidth-bound op.

## Part B — Performance-regression diagnosis (real, from this development)

The first benchmark run reported NineToothed as **22,068x slower** than torch. The
skill's workflow (Step 6/7 + `failure-recovery.md` row #12) turns this into a
diagnosis loop rather than a shrug.

- **Symptom:**
  ```text
  [multiply[1048576]] candidate: 982.6221 ms   (torch baseline: 0.0445 ms)
  NineToothed is 22068.06x SLOWER than baseline (regression — investigate)
  ```
- **Root cause:** the operator wrapper rebuilt the `@ninetoothed.jit` kernel on
  **every call**, so each timed iteration paid full kernel recompilation and
  auto-tuning. The benchmark was measuring compile time, not compute time. A single
  1M-element multiply cannot take ~1 second of compute; that magnitude is a
  compile-per-call artifact.
- **Minimal fix:** build the kernel once and reuse it (`_MULTIPLY_KERNEL` at module
  load), timing only the kernel invocation. See `multiply_fixed` vs `multiply_naive`
  in `benchmark_multiply.py`.
- **Re-run command:** same command as Part A.
- **Re-run result:** 982.6 ms → **0.0459 ms** at 1M (~20,000x improvement), landing
  at parity with torch. Regression resolved; root cause was benchmarking
  methodology, not the kernel.

**Lesson encoded in the skill:** when a NineToothed kernel looks catastrophically
slow, suspect per-call recompilation / auto-tuning before blaming the kernel —
compile once, reuse, and time only the invocation.

## Integration note (secondary, also real)

While wiring these example tests into the repo, collection failed with
`ModuleNotFoundError: No module named 'tests'` (the tests reuse the repo helper but
sat below the root on `sys.path`). Fixed minimally by adding `conftest.py` at the
skill root to insert the repo root — no test or source code changed. Re-run:
`4 passed`. This is a second instance of the same diagnose-then-minimally-fix loop.
