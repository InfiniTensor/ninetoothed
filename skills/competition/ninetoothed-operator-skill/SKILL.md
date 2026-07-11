---
name: ninetoothed-operator-skill
description: Implement, test, benchmark, debug, and report NineToothed GPU operators. Use when developing NineToothed operators, correctness tests, benchmarks, layout handling, generated source analysis, or AOT builds.
---

# NineToothed Operator Skill

Guide AI agents to develop, test, benchmark, debug, and integrate NineToothed operators.

## Trigger

Activate this skill when the task involves:
- Implementing, testing, or optimizing NineToothed GPU operators
- Writing correctness tests or benchmarks for NineToothed kernels
- Debugging arrangement/application/tensor errors
- Checking generated source or AOT build output
- Preparing PR materials for NineToothed operator tasks

## Workflow

1. **Read task** — Open `task.md`, check `references/index.md` for repo layout, read upstream `tests/` for similar operators.
2. **Extract contract** — Input/output shape, dtype, device, broadcast rules, layout constraints, numerical tolerance, boundary cases.
3. **Choose implementation** — Simple single-kernel: `@ninetoothed.jit`. Multi-step arrangement or cached kernel: `ninetoothed.make()`.
4. **Implement** — Write tensor arrangement, tile/block shape, mask/boundary, stride/offset. Keep patch minimal.
5. **Test** — Write pytest with PyTorch reference. Cover normal, edge, non-contiguous, broadcast cases. See `references/testing.md`.
6. **Benchmark** — If performance-sensitive, run benchmark with warmup, CUDA sync, baseline ratio. See `references/benchmarking.md`.
7. **Check generated source / AOT** — For perf-sensitive tasks, inspect compiled output. See `references/benchmarking.md`.
8. **Diagnose failures** — Classify and fix by environment → import → compile → shape → numerical → layout → performance. See `references/debugging.md`.
9. **Wrap up** — Update `REFERENCE.md`, `HONOR_CODE.md`, document limitations and unsupported cases.

## Constraints

- **No hidden answers** — Do not hardcode task names, expected outputs, or evaluation bypasses.
- **No credentials** — No API keys, account credentials, or private data.
- **No test bypass** — Do not delete or weaken tests. CUDA-unavailable must skip, not fail.
- **No fabricated results** — If GPU unavailable, state clearly. Do not fake pytest or benchmark output.
- **Minimal patch** — No unrelated refactoring, no large-scale formatting changes.
- **Real implementation** — NineToothed kernel must be the actual implementation, not a PyTorch wrapper.
- **Disclose references** — All external code, docs, and AI assistance go in `REFERENCE.md`.

## Out of Scope

| Scenario | Reason |
|----------|--------|
| Dynamic shape | Requires shape guard, beyond fixed tile pattern |
| bfloat16 / float64 | Not fully validated in self-tests |
| Modifying NineToothed compiler core | High risk, out of scope |
| Multi-GPU / distributed | Self-tests are single-GPU only |
| CPU-only deployment | NineToothed targets GPU; use PyTorch reference as fallback |

## Quick Reference

| Need | Read |
|------|------|
| Repo structure, API entry points | `references/index.md` |
| Test patterns, skip logic | `references/testing.md` |
| Benchmark, generated source, AOT | `references/benchmarking.md` |
| Failure diagnosis | `references/debugging.md` |
