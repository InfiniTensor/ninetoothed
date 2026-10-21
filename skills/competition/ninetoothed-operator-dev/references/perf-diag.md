# Performance / Diagnosis / Integration

Covers: verify arrangement, read generated source, benchmark + Roofline, AOT
build, and locate a regression. These map to the performance-awareness and
diagnostic rubric items.

> Provenance tags — calibrate trust per claim: **[E]** exercised end-to-end on
> GPU in this project's episode runs (raw run logs kept with the companion
> harness, not committed to this package — re-run to reproduce) · **[S]** verified
> against the `ninetoothed==0.25.0` source, not executed here · **[I]**
> inferred from adjacent patterns — re-verify before relying on it.

## A. Verify the arrangement (deterministic, no kernel run) — [S]

**Step 5 of the SKILL.md workflow.** Call this before compiling the kernel.

### Via the skill's debug script (recommended for agents)

```bash
# From the skill root, pass module_dotpath:fn_name
python scripts/debug_arrangement.py examples.02_reduction_parameterized.kernel:arrangement
```

Or import as a library from your own driver script:

```python
from scripts.debug_arrangement import check_no_oob, summarise
from my_kernel import arrangement, _TENSORS

ok = check_no_oob(arrangement, _TENSORS)   # prints summary + returns bool
# Output:
#   tensor[0]  source=(64, 512)  target=(64, 1, 512)  programs=64  tile=(1, 512)
#              oob=0  unique_read=32768/32768
#   tensor[1]  source=(64,)      target=(64, 1)        programs=64  tile=(1,)
#              oob=0  unique_read=64/64
#   status: OK  all_covered=True
```

Interpret the output:

| Field | Meaning | Alarm condition |
|-------|---------|-----------------|
| `oob_count` | tiles that read a -1 sentinel (OOB access) | any value > 0 |
| `unique_read` / `total` | how many distinct source elements are accessed | < total may mean elements are missed |
| `tile_shape` | shape seen inside each GPU program | should match your intended block dims |

### Underlying API (for reference)

```python
from ninetoothed.debugging import simulate_arrangement
src, tgt = simulate_arrangement(arrangement, tensors)   # device defaults to cuda
# src[i]: source index grid; tgt[i]: arranged grid with source indices
```

### Visualization for human review (optional, requires matplotlib)

**Boundary rules — read before calling:**

| API | Headless-safe | Who should use it |
|-----|:-:|---|
| `debug_arrangement.visualize_and_save(arrangement, tensors, save_dir=".")` | ✅ yes | Agent generates PNG → human inspects |
| `ninetoothed.visualization.visualize(tensor, save_path="x.png")` | ✅ yes | Same — headless PNG |
| `ninetoothed.visualization.visualize_arrangement(arrangement, tensors)` | ❌ no | Local development only (tkinter GUI) |

Install optional deps first: `bash setup.sh --with-viz`

```python
from scripts.debug_arrangement import visualize_and_save
paths = visualize_and_save(arrangement, tensors, save_dir="report/")
# gracefully prints a warning and returns [] if matplotlib not installed
```

The agent should call `visualize_and_save` ONLY if the `--with-viz` flag is
confirmed available in the environment; otherwise skip and note "visualization
not generated (matplotlib not installed)" in the trace log.

## B. Read the generated Triton source

The base `--digest` report is **[E: run live by agents in the pd04 diagnosis
episodes, 2026-07-10/11; raw run logs kept with the companion harness, not
committed here]**. The `mask=` counting and the `--contract` check (below) are
newer than those episodes and are **[S]** — passing the offline test suite, not
yet run in a GPU episode.

NineToothed caches generated source at `~/.ninetoothed/<sha256>.py`
(`ninetoothed.generation.CACHE_DIR`). After a kernel has been built once, run:

```
python scripts/inspect_generated_source.py            # newest cached kernel
python scripts/inspect_generated_source.py --digest <sha256>
```

It reports the `tl.*` ops used, any `num_warps` / `num_stages` constants, tile
sizes, load/store counts, and how many loads/stores carry a `mask=` — the
evidence you cite when judging whether a kernel is reasonable or has
regressed. The script is read-only and parses with `ast` (no execution).

### Semantic contract check (`--contract`) — [S]

A green correctness matrix samples inputs; it does not prove the compiled
kernel implements the claimed semantics. The classic escape: a scatter-class
kernel using plain stores passes every collision-free test case and races on
real data; a "reduction" that copies through passes when the test data
happens to make the fold trivial. `--contract` closes this by reading what
was actually compiled:

```
python scripts/inspect_generated_source.py --contract reduction        # needs a fold primitive
python scripts/inspect_generated_source.py --contract stable_softmax   # needs exp AND row-max
python scripts/inspect_generated_source.py --contract atomic           # needs tl.atomic_*
python scripts/inspect_generated_source.py --contract matmul           # needs tl.dot
python scripts/inspect_generated_source.py --contract elementwise      # loads+stores, no dot
```

Exit code 2 on violation — treat that as a correctness failure even if pytest
is green, and capture the output in the artifact checklist (SKILL.md §4).
Note: `--contract atomic` failing on a many-to-one scatter usually means the
task is outside the DSL (`dsl_limit`, SKILL.md §3) — declare the fallback
rather than shipping a race.

## C. Benchmark + Roofline — [S: Roofline classifier self-tested offline on CPU (GEMM 4096³ vs H100 ridge point); the `do_bench`/CUDA-event timing path is source-derived, not run in a GPU episode here]

**Time with `triton.testing.do_bench` — it is what NineToothed itself uses**
(`src/ninetoothed/auto_tuner.py`, `src/ninetoothed/build.py`) and the Triton
standard: quantile-based timing with an L2-cache flush between reps, so numbers
are more stable and repo-consistent than a hand-rolled loop. `scripts/bench_compare.py`
prefers it automatically and falls back to `torch.cuda.Event`, then `perf_counter`:

```python
from scripts.bench_compare import benchmark, roofline
r = benchmark(lambda: my_op(x), warmup=25, iters=100)  # -> do_bench if triton+GPU present
# r['timer'] tells you which path ran; r['mean_ms'] is do_bench's median (p50)
gbps = bytes_moved / (r["mean_ms"] * 1e-3) / 1e9
verdict = roofline(flops=flops, bytes_moved=bytes_moved)  # gpu auto-detected
# verdict -> "compute-bound" | "memory-bound" (+ ridge_point used), or
# "unknown" with ridge_point None when the device can't be resolved to a spec
```

Or call `do_bench` directly if you don't need the roofline/throughput helpers:

```python
import triton
p50, p20, p80 = triton.testing.do_bench(lambda: my_op(x),
                                        warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
```

Note: under `do_bench`, `warmup`/`rep` are **millisecond budgets** (it auto-sizes
the iteration count to fill them), not raw counts.

Benchmark requirements (rubric): a PyTorch (or repo Triton) baseline, ≥3 input
sizes incl. one non-power-of-two, a robust central latency + spread (do_bench
p50 with p20–p80, or mean±std over ≥30 iters), GB/s or TFLOPS, and a
compute/memory-bound conclusion.

**Ridge points** (arithmetic intensity FLOP/byte; below ⇒ memory-bound) are a
**label only**: H100 ≈ 295, A100 ≈ 156. `roofline()` resolves the ridge from the
live device name (override via `$NT_GPU` or the `gpu=` arg); an unlisted card
(e.g. RTX 5090) yields verdict `"unknown"` — a missing label, never a guessed
one. Ridge = peak fp16 FLOPs ÷ peak HBM bandwidth, and peak FLOPs isn't probeable
at runtime, so this stays a tiny two-anchor table on purpose.

The **reward-hacking bandwidth ceiling is not table-driven** — `robust_bench.
measure_peak_bw()` times a tuned streaming copy on the *actual* device and flags
any kernel claiming to beat it by more than `SUSPICIOUS_BW_MARGIN`. That covers
any card with no spec entry and can't drift from the real hardware.

## D. AOT build — [S]

`make` with a non-`torch` caller emits ahead-of-time artifacts instead of a
JIT handle:

```python
ninetoothed.make(arrangement, application, tensors,
                 caller="cuda", kernel_name="my_op", output_dir="build/")
# writes a .py launcher and a .h header into build/
```

Smoke-test with `scripts/aot_build_smoke.sh build/` (checks the expected
`.py` + `.h` exist and are non-empty). If AOT fails where JIT worked, suspect
`num_warps` / `num_stages` defaults vs the target — pass them explicitly to
`make(...)`.

## E. Locate a performance regression — [S]

1. Reproduce: benchmark current vs the known-good revision on identical shapes.
2. Diff generated source: `inspect_generated_source.py` on both digests; compare
   tile sizes, num_warps/num_stages, and load/store counts.
3. Attribute: a regression usually shows up as a changed tile/auto-tune config
   or extra load/store. State the evidence (which field changed) in the report.
4. Minimal fix: pin the better config via `make(..., num_warps=, num_stages=)`
   or `block_size()` bounds — do not refactor unrelated code.

## F. fp16/bf16 tolerance reference (for correctness during perf work) — [S]

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |

matmul/attention in fp16 typically need rtol≈1e-2; elementwise stays tight.
Always compare against PyTorch with `torch.testing.assert_close(got, expected,
atol=, rtol=)`.
