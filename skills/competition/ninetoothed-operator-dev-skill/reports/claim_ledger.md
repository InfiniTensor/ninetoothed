# Claim Ledger

| Claim | Status | Evidence |
| --- | --- | --- |
| Final skill tree is self-contained | `[VERIFIED]` after final validation | Package tests and clean-extraction logs |
| add focused server correctness | `[VERIFIED]` | `SELFTEST-EW-001/server_correctness_excerpt.log` |
| relu focused server correctness | `[VERIFIED]` | `SELFTEST-EW-001/server_correctness_excerpt.log` |
| softmax focused server correctness | `[VERIFIED]` | `SELFTEST-RED-001/server_correctness_excerpt.log` |
| final combined add/relu/softmax correctness rerun | `[VERIFIED]` | `reports/independent_audit/focused_correctness_recheck.log`: 32 passed |
| add short timing record | `[VERIFIED]` | `SELFTEST-EW-001/benchmark.csv` |
| softmax short timing record | `[VERIFIED]` | `SELFTEST-RED-001/benchmark.csv` |
| layout baseline production patch applies at ntops `6bc90d5` | `[VERIFIED]` | `SELFTEST-IMPL-001/apply_check.log` |
| layout skill-guided patch applies locally after LF normalization | `[VERIFIED]` | `SELFTEST-LAYOUT-001/skill_enabled_apply_check.log` |
| performance integration patches apply locally at InfiniCore `d2758a5c` | `[VERIFIED]` | `SELFTEST-PERF-AOT-001/*apply_check.log` |
| repaired layout skill-guided tests on CUDA | `[VERIFIED]` | `reports/independent_audit/patch_recheck_fixed.log`: 80 passed, 72 skipped |
| production pooling implementation rerun on CUDA | `[VERIFIED]` | `reports/independent_audit/patch_recheck_fixed.log`: 88 passed, 72 skipped, 2 xpassed |
| SiLU fast-path and fallback branch logic under controlled stubs | `[VERIFIED]` | `reports/independent_audit/perf_silu_stub_recheck.log` |
| layout skill-guided patch passed historical server apply-check | `[BLOCKED]` | Historical server output records failure |
| SiLU InfiniCore dispatch | `[BLOCKED]` | `infinicore.lib` missing in server import probe |
| Generated source consumed at runtime | `[TODO-GPU]` | No generated runtime artifact evidence |
| AOT build output | `[TODO-GPU]` | Build was not run |
| Full non-contiguous support | `[TODO-GPU]` | Only selected layout classes were exercised |
| Broad speedup | `[BLOCKED]` | Both selected-shape records are regressions and the scope is insufficient |
| Hidden evaluation success or final score | `[BLOCKED]` | No hidden evaluation was available or run |
| Participant identity and signature | `[VERIFIED]` | `HONOR_CODE.md`: 刘李宏 / 123123 / Dreamt-Deer-Waking-Fish, signed 2026-07-12 |
