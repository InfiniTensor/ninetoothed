---
name: ninetoothed-operator-skill
description: Guide Codex through NineToothed GPU operator development, testing, benchmarking, debugging, generated-source dump checks, AOT workflow triage, and PR integration. Use when a task involves NineToothed DSL operators, arrangement/application patterns, tensor shape or dtype semantics, broadcast or mask behavior, non-contiguous fallback handling, PyTorch reference tests, performance benchmarks, failing tests, or repository integration.
---

# NineToothed Operator Development

Use this skill for NineToothed operator implementation, correctness tests,
benchmarking, failure diagnosis, generated-source dump checks, and integration
work. Do not use it for compiler-internal changes or raw Triton kernels.

## Evidence Boundary

Validated by this package:

- ReLU elementwise correctness for fp32/fp16 and fp16 benchmark timing
- 2D last-dimension softmax correctness and benchmark comparison
- RMSNorm with fixed `eps=1e-6`
- Non-contiguous compatibility through explicit `.contiguous()` fallback
- Generated-source dump trigger with post-dump correctness validation

Documented as workflow guidance, not self-test proof:

- AOT build configuration and success
- Detailed generated-source load/store/mask analysis
- Native arbitrary-stride 2D reductions
- Dynamic scalar parameters such as RMSNorm `eps`

Not supported by this package unless a task adds direct implementation,
correctness, benchmark, and diagnostic evidence:

- Matmul/GEMM implementation claims

Do not claim support for unvalidated capabilities unless the task adds direct
evidence: code, command, output, and tests.

## Read First

Before editing code, read:

1. `README.md` for package scope and validation commands.
2. `references/repo-structure-index.md` for where to find NineToothed patterns.
3. `references/dsl-quick-reference.md` for DSL syntax.
4. `references/operator-patterns.md` only after selecting the operator family.
5. `references/v0.26-known-failures.md` when a compile/runtime failure appears.
6. `references/benchmarking-and-diagnostics.md` for performance, dump, or AOT tasks.

In a real NineToothed repository, also read `CONTRIBUTING.md`, nearby operator
implementations, tests, examples, and benchmark files before writing code.

## Workflow

### Step 0 — Extract The Operator Contract

Write down:

- Inputs and outputs
- Shape/rank constraints
- dtype requirements and accumulation dtype
- broadcast, mask, and boundary behavior
- contiguous vs non-contiguous layout expectations
- PyTorch or repository reference implementation
- performance sensitivity and baseline

Do not start coding until the contract is explicit.

### Step 1 — Select The Arrangement Pattern

Choose the closest existing pattern:

- Elementwise/broadcast: see `references/operator-patterns.md`
- Row-wise reduction: see `references/operator-patterns.md`
- Layout fallback or diagnostics: see `references/benchmarking-and-diagnostics.md`
- Spatial/pooling: pattern is documented; verify against an existing example

Prefer an existing arrangement/application style over inventing a new one.

### Step 2 — Implement Minimally

Use the NineToothed mental model:

```python
def arrangement(...):
    ...


def application(...):
    output = ...  # noqa: F841


tensors = (...)
kernel = ninetoothed.make(arrangement, application, tensors)
```

Rules:

- Do not write raw `tl.program_id`, `tl.load`, `tl.store`, or manual offsets.
- Keep `arrangement` and `application` separate.
- Use `ntl.*` for reductions and math.
- Cast to fp32 before reductions when precision matters.
- Keep the patch small and local to the target operator/test/docs.

### Step 3 — Respect Known v0.26.0 Constraints

Before debugging from scratch, check `references/v0.26-known-failures.md`.
Important constraints:

- Do not use `Tensor(0)` for arithmetic scalar parameters such as `eps`.
- Do not rely on outer-scope `Symbol` closure capture inside `application`.
- Use positive shape indices such as `input.shape[1]`, not `shape[-1]`.
- If only one scalar literal is supported, reject unsupported wrapper arguments.

### Step 4 — Handle Layout Honestly

For non-contiguous inputs, state which path is used:

- Native stride support, if the task proves it with tests.
- Explicit `.contiguous()` fallback, if native stride support is not safe.

Fallback tests must compare against the original non-contiguous PyTorch
reference and should benchmark copy overhead when performance matters.

### Step 5 — Write Correctness Tests

Every operator needs PyTorch or repository reference tests covering:

- At least three shapes: small, exact tile multiple, non-multiple
- Relevant dtypes; do not claim fp16 coverage unless it is tested
- Boundary/mask cases
- Non-contiguous input or documented fallback path
- Tolerances appropriate for dtype and reduction precision

Run the focused test first, then the package-level correctness command.

### Step 6 — Benchmark When Performance Matters

Use `references/benchmarking-and-diagnostics.md`.

Record:

- command
- hardware and dependency versions
- input shapes/dtypes
- PyTorch or repository baseline
- timing method
- result table
- conclusion and limitations

Do not equate "benchmark test passed" with "performance target met" unless a
baseline or threshold supports that claim.

### Step 7 — Generated Source And AOT

Generated-source self-test coverage in this package checks the dump trigger and
post-dump correctness. Source analysis is a separate task: inspect
load/store/mask patterns and tile constants when the task explicitly asks for
it.

AOT build is a workflow item only. If a task asks for AOT, run the actual build
command and record stdout/stderr before claiming success.

### Step 8 — Diagnose Failures

For every failure, record:

- symptom and exact error message
- command that failed
- suspected root cause
- minimal fix
- command used to verify the fix
- final result

Common causes:

- wrong tile shape or tensor rank
- wrong `other` fill value for padded reductions
- missing fp32 accumulation
- unsupported scalar parameter modeling
- unsupported native stride layout
- generated-source or AOT claim without evidence

## Output Checklist

Before finishing a task:

- [ ] Operator contract is explicit.
- [ ] Implementation uses `arrangement` + `application` + `make`.
- [ ] Correctness test compares against PyTorch or repository reference.
- [ ] Non-contiguous behavior is tested and labeled native or fallback.
- [ ] Benchmark or generated-source/AOT evidence is included when relevant.
- [ ] Unsupported scope is stated instead of hidden.
- [ ] No unrelated refactor, mass formatting, API key, or hidden-test logic.
- [ ] Final response summarizes files changed, commands run, and results.
