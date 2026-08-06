# Verification Matrix

Use this file to design self-tests and hidden-task style checks. Keep it as a
verification map: it should say what evidence is required, not teach the DSL.

## Per-Task Score Frame

| Area | Points | Evidence |
| --- | ---: | --- |
| Task completion | 4 | Operator semantics are correct; shape, dtype, broadcast, layout, mask, and boundary behavior match the task contract. |
| Tests and verification | 2 | Correctness command is named, executed, and recorded with pass/fail result; failures include fix and rerun evidence. |
| Performance awareness | 1 | Benchmark, generated source, AOT build, or explicit performance non-requirement is recorded for the task category. |
| Patch minimality | 1 | Diff is limited to the named operator, test, example, benchmark, or diagnostic surface; no unrelated formatting. |
| Repository style | 1 | Naming, file placement, test style, tolerance style, and examples mirror nearby NineToothed patterns. |
| Process and compliance | 1 | Commands and assumptions are traceable; unsupported scope is explicit; no secrets, hidden answers, or test bypasses. |

Score every self-test and hidden-task simulation out of 10. A task may be
partially complete, but missing correctness evidence caps the score at 6, and
fabricated benchmark or test results cap the score at 4.

## Required Self-Test Categories

| Category | Required Evidence | Benchmark Required |
| --- | --- | --- |
| elementwise or broadcast operator | PyTorch or repository oracle, ordinary shape, broadcast variant, dtype/tolerance decision, tail or mask boundary. | Yes, unless the task is explicitly correctness-only. |
| reduction or blocked operator | Reduce axis, block/tile shape, numerical stability choice, tail-block or non-power-of-two case, tolerance rationale. | Yes. |
| stride, offset, or non-contiguous layout case | Sliced, transposed, strided, or offset input case; stride-aware load/store evidence or explicit contiguous-only unsupported scope. | Optional unless the prompt is performance-sensitive. |
| performance, generated-source, AOT build, or failure-diagnosis case | Repro command, baseline, input sizes, generated source or AOT notes when relevant, failure root cause, rerun after fix or blocker. | Yes. |

Use `tests/eval_cases.yaml` as the machine-readable index for these categories.

## Mandatory Agent Steps

Every self-test task must direct the agent to complete these steps before it is
marked complete:

- extract input tensors, output tensors, shape rules, dtype rules, broadcast
  rules, boundary behavior, layout assumptions, and unsupported scope from the
  operator requirement;
- read relevant upstream arrangement, application, tensor meta-operation,
  load/store, generated-source/AOT, benchmark, test, and example code before
  choosing the DSL shape;
- choose a NineToothed DSL expression that mirrors nearby repository style and
  explain why it fits the requirement;
- write a correctness test aligned with PyTorch or an existing repository
  reference implementation;
- inspect generated source, AOT build output, or benchmark results when the
  category or prompt requires performance evidence;
- record optimization notes for performance-sensitive work, including redundant
  load/store, avoidable broadcast computation, contiguous/stride assumptions,
  tile/block choices, and AOT/benchmark follow-up when relevant;
- record failure symptoms, diagnosis path, root cause or blocker, minimal fix
  or workaround, rerun command, and rerun result whenever a command fails;
- explicitly name unsupported dtype, dynamic shape, non-contiguous layout,
  hardware/runtime, benchmark, AOT, or generated-source scope.

## Required Self-Test Sections

Each `examples/*/TASK.md` self-test record must contain:

- input task说明;
- AI agent execution summary;
- repository files inspected;
- operator code, example, test, or fix patch summary;
- correctness command and real result or explicit `Not run yet`;
- benchmark command and result, blocker, or explicit non-requirement;
- performance conclusion that is backed by benchmark evidence, generated-source
  fallback evidence, or an explicit non-requirement/blocker;
- failure diagnosis section with symptoms, diagnosis path, root cause or
  blocker, minimal fix or workaround, rerun command, and result when a failure
  exists;
- risks and unsupported scope.

At least two benchmark-required self-test categories must provide benchmark
commands and require recording baseline, input sizes, run command, result, and
performance conclusion.

## Correctness Checklist

Every completed task should answer each item below. If an item does not apply,
record why in `unsupported scope` instead of silently omitting it.

- reference implementation: PyTorch expression, existing NineToothed behavior,
  generated-source expectation, or hand-computable oracle.
- ordinary shape: one representative dense case that should pass without edge
  handling.
- boundary shape: smallest valid shape, empty or size-one dimension if valid,
  or a shape with a partial tail block.
- non-power-of-two size: a shape that exercises mask or tail logic when the
  operator uses block/tile execution.
- dtype variants: supported input/output dtypes, accumulator dtype when
  relevant, and dtype-specific tolerance.
- broadcast variants: scalar, singleton-dimension, trailing-dimension, or
  batch broadcast case when the operator accepts broadcasting.
- non-contiguous variants: sliced, transposed, strided, or offset tensor case,
  or an explicit contiguous-only unsupported-scope entry.
- tolerance: `rtol`, `atol`, equality policy, and NaN/Inf handling when
  relevant.
- unsupported scope: excluded dtype, shape, layout, device, benchmark, AOT, or
  generated-source claim.

## Benchmark-Required Rule

Benchmark or generated-source/AOT evidence is required when any of these are
true:

- the task asks for performance, optimization, regression diagnosis, generated
  source, or AOT build behavior;
- the operator is a reduction, blocked, or memory-layout-sensitive kernel where
  a slow implementation can be functionally correct but unsuitable;
- the self-test category in `tests/eval_cases.yaml` has
  `benchmark_required: true`.

When benchmark execution is blocked by the environment, record the command,
hardware/runtime blocker, intended baseline, input sizes, and a fallback static
inspection result. Do not turn a blocked benchmark into a performance claim.

## Result Template

- task id:
- command:
- environment:
- reference:
- input sizes:
- baseline:
- result:
- benchmark or generated-source/AOT evidence:
- failures:
- fix:
- rerun:
- unsupported scope:
- score notes:
