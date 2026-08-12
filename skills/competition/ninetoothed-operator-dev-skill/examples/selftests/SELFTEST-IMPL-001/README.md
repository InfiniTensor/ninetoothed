# SELFTEST-IMPL-001: Real Production Wrapper Change

## Task Description

Demonstrate an end-to-end repository change that modifies production wrapper
behavior, adds correctness coverage, produces a minimal patch, passes clean
apply-check, and has real device test evidence.

This case packages the strongest implementation artifact from the layout
comparison. It was produced by the no-skill baseline and is identified that way
to avoid overstating the skill-guided result.

## Agent Execution Summary

The implementation normalizes integer `kernel_size` to a two-element tuple in
both `avg_pool2d` and `max_pool2d` before defaulting `stride`. It adds focused
pooling cases for integer kernels, asymmetric stride/padding, dilation,
boundary shapes, source views, output shape, and a shared deterministic input
helper.

## Production And Test Files Changed

Patch: [implementation.patch](implementation.patch).

- Production: `src/ntops/torch/avg_pool2d.py`.
- Production: `src/ntops/torch/max_pool2d.py`.
- Tests: `tests/test_avg_pool2d.py`.
- Tests: `tests/test_max_pool2d.py`.
- Test helper: `tests/utils.py`.

Patch size: 5 files, 189 insertions, no deletions. The production behavior
change itself is six lines; the remaining change is focused coverage.

## Correctness

Commands:

```bash
python -m pytest tests/test_max_pool2d.py
python -m pytest tests/test_avg_pool2d.py
```

Results:

- max pooling: `62 passed`, `54 skipped`, `1 xpassed`.
- average pooling: `26 passed`, `18 skipped`, `1 xpassed`.
- Final independent combined rerun: `88 passed`, `72 skipped`, `2 xpassed`.
- CUDA was available on the recorded server run.
- Skipped and unexpected-passed cases remain separately visible.

The exact compact outputs are retained in the layout self-test:

- [max pooling output](../SELFTEST-LAYOUT-001/max_pool2d_server_excerpt.log)
- [average pooling output](../SELFTEST-LAYOUT-001/avg_pool2d_server_excerpt.log)
- [final independent rerun](../../../reports/independent_audit/patch_recheck_fixed.log)

## Patch Applicability

Target revision: ntops `6bc90d5aba29146a8757fe0b67e7e1966b92bb5f`.

Evidence: [apply_check.log](apply_check.log).

- `git diff --check` on the originating change: no whitespace error recorded.
- `git apply --check`: PASS.
- `git apply --check --whitespace=error-all`: PASS.

## Benchmark

No benchmark was run for this implementation. It is a correctness and API
normalization case, and no performance conclusion is attached.

## Failure And Closure

- Device correctness did not fail for non-skipped pooling cases.
- The main risk is overgeneralizing selected source views into arbitrary layout
  support; the final claim remains limited to the recorded cases.

## No-Skill Versus Skill

- Baseline advantage: produced the real production change and passed server
  correctness.
- Skill-guided advantage in the paired task: narrower patch and better process
  records, but it did not produce the production change and failed server patch
  handoff.
- Lesson incorporated into the final skill: implementation requests now require
  a production change when needed, a before/after correctness test, and clean
  patch validation before handoff.

## Unsupported Or Unverified

- Full non-contiguous support is not claimed.
- No convolution, generated-source, AOT, dispatch, or speedup claim is made.
