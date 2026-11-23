# Failure Recovery Table

NineToothed-specific symptoms → root cause → minimal fix. Each hidden task runs
**once** and is not retried on skill-induced failure, so self-correct within the
single run: match the symptom, apply the minimal fix, re-run, and record the loop.

**Golden rule:** read the FIRST real error, not the last line of the traceback.
Do not thrash with random edits.

| # | Symptom | Root cause | Minimal fix |
|---|---|---|---|
| 1 | `make()` / compile error about mismatched shapes | Outermost shapes of the arranged parameter tensors differ | Align them with `expand` so all outermost shapes match; drop size-1 dims with `.dtype.squeeze(dim)` |
| 2 | Wrong result only at the edges / last block | Missing out-of-bounds fill | Construct the tensor with `Tensor(..., other=float("-inf"))` (max) or `other=0` (sum) |
| 3 | Wrong result on transposed / sliced input, correct on contiguous | Code assumed contiguous layout | Handle layout explicitly: `permute` for transpose, `tile(..., strides=...)` for strided; add a non-contiguous test |
| 4 | `IndexError` / wrong loop range in `application` | Indexing a 2-D `(1, ...)` or `(..., 1)` block instead of squeezed | `.dtype.squeeze(0/1)` after `expand`, then index `x[k]` and range over `x.shape[0]` |
| 5 | Accumulation overflow / precision loss (fp16) | Accumulating in fp16 | `accumulator = ntl.zeros(shape, dtype=ntl.float32)`, cast out with `.to(ntl.float16)` |
| 6 | All tests show `SKIPPED`, none run | No GPU: `get_available_devices()` returned empty | This is EXPECTED off-GPU. Record the skip honestly; do NOT rewrite as pass |
| 7 | `allclose` fails by a small margin on fp16/bf16 | Tolerance too tight | Add an appropriate `atol`/`rtol` (see `test_matmul.py` using `atol`); do not loosen to hide a real bug |
| 8 | `ruff` / style checker fails | PEP8 / blank-line rules (blank line around `if`/`for`, before `return`) | Run `python scripts/check_contributing_style.py --fix` then `ruff format`; re-check |
| 9 | `commit-msg` / `pre-push` hook rejects | Branch not kebab-case ≤50 chars, or commit not imperative/capitalized/no-punctuation | Rename branch; rewrite commit e.g. `Add ... skill`; never `--no-verify` |
| 10 | AOT `ninetoothed.build` deadlocks or errors on import | Missing `lazy=True`, missing `if __name__=="__main__":`, or `output_dir` absent | Add all three; create `output_dir` first; delete it to force rebuild |
| 11 | `constexpr` symbol error at call time | A fixed-size symbol not supplied | Pass it explicitly at call time, e.g. `kernel(x, out, BLOCK_SIZE=x.shape[-1])` |
| 12 | Kernel silently slow / suspected regression | No auto-tuning or bad block size | Use `Symbol(meta=True)` / `block_size()` for auto-tuning; benchmark with `scripts/bench.py` and report baseline vs result |

## When the fix is not in the table

1. Reproduce with the smallest possible input.
2. Read the first real error line.
3. Compare your operator against the closest reference in `tests/` line by line.
4. If layout-related, use `ninetoothed.debugging.simulate_arrangement(arrangement,
   tensors)` to see how each element maps.
5. Record the diagnosis honestly, including any part you could not resolve, and
   state the limitation rather than faking a pass.
