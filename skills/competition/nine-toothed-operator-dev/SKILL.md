---
name: nine-toothed-operator-dev
description: Guide AI agents implementing, testing, benchmarking, diagnosing, or integrating NineToothed GPU operators. Use for NineToothed arrangement/application DSL tasks, operator correctness tests, PyTorch reference comparisons, generated source or AOT build checks, benchmark regression analysis, non-contiguous layout handling, and minimal patch repair work.
---

# NineToothed Operator Development

Use this skill to complete NineToothed operator tasks with a minimal verified patch. The priority order is correctness, reproducibility, repository style, then performance.

## Required Workflow

1. Restate the task contract before editing:
   - operator semantics
   - input/output tensors
   - shape rules
   - dtype rules
   - broadcasting rules
   - boundary behavior
   - layout assumptions, including non-contiguous, stride, and offset cases
   - unsupported cases
2. Inspect the repository before implementation:
   - Search for similar operators, tests, examples, generated source helpers, AOT build paths, and benchmark files.
   - Prefer existing NineToothed idioms over new abstractions.
   - Read `references/operator-patterns.md` when writing or modifying operator code.
   - Read `references/nine-toothed-api-notes.md` when unsure about tensor meta-operations, arrangement shape alignment, debugging, or AOT.
3. Design the arrangement before coding the application:
   - Write down the intended outer launch shape for every arranged tensor.
   - Ensure all arranged parameter tensors have the same outermost shape unless the repository has an explicit scalar/constexpr pattern.
   - Ensure the second outermost tensor level is exactly what each program should consume.
   - Use `Tensor(..., other=identity)` for out-of-bound tile values in reductions and boundary-sensitive operators.
4. Implement the smallest repository-style change:
   - Use the arrange-and-apply pattern when the repository uses kernel modules.
   - Use `@ninetoothed.jit` when the surrounding tests use inline kernels.
   - Keep tensor ranks, `Symbol` kinds, `Tensor(..., other=...)`, tiling, flattening, and dtype squeeze/expand behavior explicit.
5. Add or update correctness tests:
   - Compare against PyTorch or an existing trusted implementation.
   - Cover at least one ordinary case and one boundary/layout case when relevant.
   - Read `references/testing.md` before writing tests.
6. Run targeted validation:
   - Run the narrowest relevant `pytest` command first.
   - If CUDA or dependencies are unavailable, record the exact blocker and still run import/static checks where possible.
   - For complex arrangements, validate with `ninetoothed.debugging.simulate_arrangement` or `arranged.eval()` when available.
7. Add performance evidence when the task is performance-sensitive:
   - Read `references/performance.md`.
   - Benchmark against PyTorch, Triton, or the existing implementation when available.
   - Record input sizes, dtype, device, command, timing, and conclusion.
8. Diagnose failures with a closed loop:
   - Read `references/failure-diagnosis.md`.
   - Record symptom, command, suspected root cause, minimal fix, rerun command, and result.
9. Finish with an audit note:
   - changed files
   - correctness commands and results
   - benchmark commands and results, if any
   - unsupported cases
   - residual risk

## Hard Rules

- Do not delete, skip, weaken, or fake tests.
- Do not hard-code hidden task names, hidden answers, environment-specific paths, secrets, or online-only dependencies.
- Do not make broad formatting-only changes or unrelated refactors.
- Do not claim benchmark success without command, input scale, device/dtype, and observed result.
- Do not assume contiguous layout unless the task or existing operator explicitly requires it.
- Do not leave generated-source, AOT, or benchmark failures unexplained.

## Reference Routing

- Operator implementation: `references/operator-patterns.md`
- NineToothed API concepts and gotchas: `references/nine-toothed-api-notes.md`
- Correctness tests and self-test design: `references/testing.md`
- Benchmarks and performance analysis: `references/performance.md`
- Generated source, AOT, failing tests, and recovery: `references/failure-diagnosis.md`
- Final competition packaging: `references/final-submission.md`

## Useful Scripts

- `scripts/scan_repo.py --repo <path>`: summarize candidate files and patterns in a NineToothed checkout.
- `scripts/build_pattern_index.py --repo <path> --out references/repository-pattern-index.md`: generate a compact index of nearby operator/test/benchmark files.
- `scripts/make_selftest_task.py --name <task-name> --kind <kind> --out <dir>`: create a self-test task log template.
- `scripts/check_submission.py --skill-dir <path>`: check required competition files and self-test material.

Scripts are helpers only. If a script fails, inspect and continue manually.
