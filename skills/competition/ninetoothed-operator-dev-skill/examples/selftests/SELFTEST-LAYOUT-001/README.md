# SELFTEST-LAYOUT-001: Pooling Layout And Patch Portability

## Task Description

Exercise `max_pool2d` and `avg_pool2d` across integer and tuple kernel sizes,
stride, padding, dilation, source views, output shape, and boundary behavior.
Keep convolution diagnostic-only and avoid claiming general non-contiguous
support.

## Agent Execution Summary

The no-skill baseline changed pooling wrappers, two test files, and a shared
test helper. The skill-guided session changed only the two test files and
produced clearer requirement/evidence records. Both patches passed clean local
apply-check before server evaluation. On the isolated server, the baseline
patch applied and executed; the skill-guided copy failed to apply and therefore
did not run there.

Afterward, both preserved patches were normalized to UTF-8 LF and revalidated
against ntops commit `6bc90d5aba29146a8757fe0b67e7e1966b92bb5f`.
The final audit then found that the skill-guided test helper converted dtype
after slicing, which made float16 views contiguous. The helper now converts
before slicing; the repaired patch passed strict apply-check and GPU tests.

## Production And Test Files Changed

Baseline patch: [baseline.patch](baseline.patch).

- `src/ntops/torch/avg_pool2d.py`
- `src/ntops/torch/max_pool2d.py`
- `tests/test_avg_pool2d.py`
- `tests/test_max_pool2d.py`
- `tests/utils.py`

Skill-guided patch: [skill_enabled.patch](skill_enabled.patch).

- `tests/test_avg_pool2d.py`
- `tests/test_max_pool2d.py`

## Correctness

Commands from the isolated ntops workspace:

```bash
python -m pytest tests/test_max_pool2d.py
python -m pytest tests/test_avg_pool2d.py
```

Raw excerpts:

- [max_pool2d_server_excerpt.log](max_pool2d_server_excerpt.log)
- [avg_pool2d_server_excerpt.log](avg_pool2d_server_excerpt.log)

Baseline results:

- max pooling: `62 passed`, `54 skipped`, `1 xpassed`.
- average pooling: `26 passed`, `18 skipped`, `1 xpassed`.
- Skipped and unexpected-passed cases are reported separately and are not
  counted as ordinary pass evidence.
- Final skill-guided audit result: `80 passed`, `72 skipped` on CUDA.
- Independent logs: [initial failure](../../../reports/independent_audit/patch_recheck.log)
  and [fixed rerun](../../../reports/independent_audit/patch_recheck_fixed.log).

## Patch Applicability

- [baseline_apply_check.log](baseline_apply_check.log): strict local check PASS.
- [skill_enabled_apply_check.log](skill_enabled_apply_check.log): strict local
  check PASS for the repaired test helper.
- Historical server outcome: skill-guided raw copy failed at
  `tests/test_avg_pool2d.py:6` and `tests/test_max_pool2d.py:6`.

## Benchmark

No benchmark was run for this case. The server scope was correctness-only and
no timing or speedup is claimed.

## Failure And Closure

- Symptom: skill-guided server `git apply --check` failed.
- Root cause: patch portability/source-context handoff, not runtime correctness.
- Minimal repair: preserve the original, normalize line endings, and validate
  against the recorded clean revision without changing patch semantics.
- Audit symptom: after successful apply, two float16 layout cases failed because
  `.to(dtype)` materialized a contiguous tensor after slicing.
- Audit repair: convert dtype/device before slicing the source view.
- Re-run result: strict apply-check PASS and `80 passed, 72 skipped`. This does
  not retroactively convert the historical raw-copy failure into a pass.

## No-Skill Versus Skill

- Baseline advantage: actual production wrapper normalization plus stronger
  server execution evidence.
- Skill advantage: narrower test-only scope and clearer evidence structure.
- Overall: baseline remains stronger because it changes production code; the
  repaired skill-guided test patch now has an executable server result.

## Unsupported Or Unverified

- No full non-contiguous support claim.
- Convolution remained diagnostic-only and was not rerun for this comparison.
- No benchmark, generated-source, AOT, or InfiniCore dispatch claim.
