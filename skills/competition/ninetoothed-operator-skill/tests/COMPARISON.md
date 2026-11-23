# Skill Effectiveness: Before vs After

## Method

To demonstrate the skill's effectiveness, we compare AI agent behavior with and without the skill across the four self-test task types.

## Comparison

### T1: Elementwise/Broadcast (add)

| Aspect | Without skill | With skill |
|--------|---------------|------------|
| Implementation | May use `@ninetoothed.jit` without handling 2D broadcast alignment | Uses `ninetoothed.make()` with `_align_to_output()` for broadcast |
| Broadcast bug | Common: forgets to expand `(1, N)` to `(M, N)` before tile | Skill workflow step 2 (extract contract) catches this |
| Test coverage | Often only same-shape test | Covers 1D, 2D same, 2D+1D broadcast, row broadcast, col broadcast |
| CUDA skip | Often missing or incorrect (import fails before skip) | `pytest.importorskip("ninetoothed")` + `allow_module_level=True` |

### T2: Reduction/Block (softmax)

| Aspect | Without skill | With skill |
|--------|---------------|------------|
| Numerical stability | May write naive softmax without `row - max(row)` | Skill references index.md points to `ntl.max/exp/sum` pattern |
| Test coverage | Often missing non-power-of-2 sizes | Tests (781, 129) non-power-of-2 + (4, 1024) large block |
| Row sum check | Often missing | Tests `output.sum(dim=-1) ≈ 1` |
| Benchmark | Often missing | Includes warmup, CUDA sync, baseline ratio |

### T3: Layout-Sensitive (transpose_add)

| Aspect | Without skill | With skill |
|--------|---------------|------------|
| Non-contiguous input | Often assumes contiguous, uses `input.contiguous()` | Uses explicit `offsets()` and `stride()` via `ntl.load` |
| Test coverage | Often only contiguous test | Tests contiguous, slicing, `empty_strided()`, error handling |
| Stride awareness | Often ignores `storage_offset()` | Tests verify `storage_offset() > 0` |

### T4: Benchmark/Debug

| Aspect | Without skill | With skill |
|--------|---------------|------------|
| Benchmark structure | Often just "it works" | Structured: warmup, sync, ms/iter, ratio, correctness check |
| Failure diagnosis | Often no documentation | `failure_diagnosis.md` with symptom → reproduce → locate → fix → verify |
| Generated source | Often ignored | Skill references `benchmarking.md` for generated source inspection |

## Conclusion

The skill provides a structured workflow that prevents common AI mistakes:
1. Broadcast alignment bugs (caught by contract extraction step)
2. Missing non-contiguous tests (caught by layout-aware test checklist)
3. Missing CUDA skip (caught by testing.md pattern)
4. Missing benchmark structure (caught by benchmarking.md template)
5. No failure documentation (caught by debugging.md diagnosis loop)
