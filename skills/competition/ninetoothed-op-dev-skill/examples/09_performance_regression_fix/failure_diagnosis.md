# Failure diagnosis — example 09

If timing looks absurd or percentiles collapse:

1. **Symptom** — paste `--benchmark` JSON (`raw_batch_ms`, median/p10/p90, `stability`)
2. **Hypothesis** — allocation inside timed region / `meta=True` / cold compile / GPU contention
3. **Minimal fix** — keep constexpr `BLOCK_SIZE`; preallocate `out=`; warmup ≥30 **per** block; keep all 30 batches (no outlier drops)
4. **Re-verify** — new results directory (never overwrite prior JSON/MD)

## Stability labels

- `spread_ratio = p90 / median`
- `<=2` stable; `<=5` noisy; `>5` highly_noisy
- Keep all batches; quantitative claims only for stable/noisy

## Size dependence / inversion

- Fastest block differing across N → conclude **规模依赖**
- Larger-N median < 50% of previous size median → `size_inversion_warning` (warn only)

See SKILL.md D7–D8 and `benchmark_result.md`.
