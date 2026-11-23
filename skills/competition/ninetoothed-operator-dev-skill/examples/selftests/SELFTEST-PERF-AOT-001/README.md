# SELFTEST-PERF-AOT-001: Performance, AOT, And Dispatch Diagnosis

## Task Description

Map ntops runtime generation, InfiniCore SiLU dispatch, generated-source and
AOT evidence gates, then produce a minimal patch without claiming runtime paths
that were not exercised.

## Agent Execution Summary

The baseline performed a broad repository scan and added a documentation note.
The skill-guided session added a narrow guard around the SiLU ntops fast path.
The baseline patch applied in the isolated server workspace. The skill-guided
raw copy did not apply there. The server could import Torch, Triton,
NineToothed, and ntops, but `infinicore` failed because its native library was
missing, so dispatch and dependent correctness were blocked.

Both preserved patches were later normalized to UTF-8 LF and now pass strict
local apply-check against InfiniCore commit
`d2758a5c3b28c70edb3743ca8cdb5cdbd97d237c`.

## Production And Test Files Changed

Baseline patch: [baseline.patch](baseline.patch).

- Added `docs/perf_aot_integration_notes.md` only.

Skill-guided patch: [skill_enabled.patch](skill_enabled.patch).

- Changed `python/infinicore/nn/functional/silu.py`.
- Added `hasattr` and `AttributeError` fallback around the ntops fast path.
- Added no correctness test in the fresh session; this limits runtime credit.

## Correctness And Dispatch

Server probe excerpt: [server_import_probe_excerpt.log](server_import_probe_excerpt.log).

- CUDA was visible and ntops imported.
- `infinicore` import failed with `ModuleNotFoundError` for `infinicore.lib`.
- `use_ntops`, SiLU route selection, and output correctness were not reached.
- No dispatch success is claimed.
- A final controlled stub test exercised the missing-operator fallback, ntops
  fast path, and `AttributeError` fallback without claiming native dispatch:
  [perf_silu_stub_recheck.log](../../../reports/independent_audit/perf_silu_stub_recheck.log).

## Generated Source And AOT

- A repository scan located relevant generation/build routes.
- No generated artifact was produced and consumed by a correctness run.
- No AOT build was executed because the native import gate was already blocked
  and no safe operator-only build command was established.

## Benchmark

No timing ran. Correctness and dispatch gates were blocked, so no benchmark or
speedup is claimed.

## Patch Applicability

- [baseline_apply_check.log](baseline_apply_check.log): strict local PASS after
  LF normalization; documentation-only.
- [skill_enabled_apply_check.log](skill_enabled_apply_check.log): strict local
  PASS; controlled branch logic passed, while native runtime remains unverified.
- The later local result is remediation evidence, not a rewrite of the server
  failure.

## Failure And Closure

- Symptom 1: skill-guided patch did not apply on the server copy.
- Repair 1: normalize and revalidate against the recorded clean revision.
- Result 1: local strict apply-check passes.
- Symptom 2: `infinicore.lib` missing during import.
- Root cause boundary: native package/build environment incomplete.
- Result 2: dispatch, AOT, correctness, and timing remain `[BLOCKED]` rather
  than being inferred from source scans.

## No-Skill Versus Skill

- Baseline advantage: broader code-path map and lower-risk documentation-only
  change; it also applied on the server.
- Skill advantage: more structured evidence and a narrowly targeted runtime
  fallback idea.
- Limitation: without a test and native import, the runtime patch cannot be
  credited as verified behavior.

## Unsupported Or Unverified

- Generated source, AOT output, InfiniCore dispatch, SiLU correctness, timing,
  and cross-device behavior remain unverified.
