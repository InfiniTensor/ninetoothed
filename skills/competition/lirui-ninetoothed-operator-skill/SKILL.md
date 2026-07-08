# NineToothed Operator Development Skill

## Skill purpose

This skill guides an AI coding agent through NineToothed operator development tasks. It is a reusable workflow for understanding requirements, implementing kernels, testing correctness, diagnosing failures, checking generated source or AOT builds, benchmarking, and preparing PR-ready changes.

This skill does not provide hidden evaluation answers, hidden task names, hard-coded contest cases, private credentials, or shortcuts around tests.

## When to use this skill

Use this skill when a task involves any of the following:

- NineToothed operator development.
- Correctness test design or repair.
- Benchmark design, execution, or interpretation.
- Generated source inspection.
- AOT build with `ninetoothed.build` or `ninetoothed.make(..., caller="cuda")`.
- Failing test diagnosis.
- PR integration for the NineToothed repository.

## Required first actions

Before editing code, the agent must read the local repository context:

1. Read `README.md`.
2. Read `CONTRIBUTING.md`.
3. Inspect `docs/`, especially `docs/source/basics.rst` and `docs/source/build.rst`.
4. Inspect `tests/` for similar operators and expected test style.
5. Inspect relevant examples or task documents in this skill package.
6. Search for existing `arrangement`, `application`, `Tensor`, `ninetoothed.make`, and `ninetoothed.build` patterns.
7. Confirm whether the requested change is allowed to modify files outside `skills/competition/lirui-ninetoothed-operator-skill/`. For this competition package, default to only changing this directory unless the user explicitly asks otherwise.

Useful searches:

```bash
rg -n "def arrangement|def application|ninetoothed\.make|ninetoothed\.build" tests src docs
rg -n "tile\(|expand\(|squeeze\(|permute\(|flatten\(|ravel\(|pad\(" tests src docs
```

## Operator requirement extraction checklist

For every operator task, extract and record:

- Inputs: names, ranks, semantic meaning, scalar versus tensor arguments.
- Outputs: shape, dtype, in-place or out-of-place behavior.
- Shape: symbolic dimensions, fixed dimensions, dynamic dimensions, batch axes.
- Dtype: supported input dtypes, accumulator dtype, output dtype.
- Broadcast rule: which dimensions broadcast and which must match.
- Mask: out-of-bounds handling, padding behavior, jagged or boundary masks.
- Boundary cases: zero-size-like limits if supported, small tensors, non-power-of-two sizes, odd dimensions, last partial block.
- Tolerance: `rtol` and `atol` for floating point comparison.
- Layout constraints: contiguous, non-contiguous, stride, offset, view, transpose, slice.

Do not implement before this checklist is clear enough to write a PyTorch reference.

## Layout checklist

Every layout-sensitive task must check:

- Contiguous input.
- Non-contiguous input.
- Stride differences.
- Storage offset differences where possible.
- View-created tensors.
- Transposed tensors.
- Sliced tensors.

At least one correctness test must use a non-contiguous input. Prefer explicit cases such as `x.t()`, `x[:, ::2]`, `x.transpose(-1, -2)`, or a narrowed view when the operator semantics allow it.

## Implementation workflow

1. Search for a similar test or implementation in `tests/` and `src/`.
2. Identify the closest local pattern, such as elementwise, matmul-like tiling, reduction block, layout transformation, jagged tensor, or AOT build.
3. Choose `arrangement` first:
   - Use `tile` to define per-program blocks.
   - Use `expand` to align outer launch shapes for broadcasting.
   - Use `squeeze`, `permute`, `flatten`, `ravel`, or `pad` only when they express the operator layout clearly.
4. Choose `application` second:
   - Keep block-local computation simple.
   - Use `ninetoothed.language` functions where existing tests do.
   - Use explicit accumulator dtype for reductions or dot-like operations.
5. Define `Tensor` specs:
   - Use `Tensor(ndim)` for symbolic shapes.
   - Use `Tensor(shape=..., dtype=...)` when AOT, dtype specialization, or fixed dimensions matter.
   - Use zero-dimensional `Tensor(0)` for scalar arguments.
6. Integrate with `ninetoothed.make`.
7. Prefer the smallest testable patch.
8. Do not perform unrelated refactoring.
9. Do not apply large formatting changes outside files touched for the task.
10. Re-run the narrowest relevant test first, then broaden only when needed.

## Correctness workflow

1. Write a PyTorch reference or reuse an existing repository reference.
2. Cover normal input.
3. Cover broadcast input when applicable.
4. Cover boundary input, including partial blocks or non-power-of-two sizes when applicable.
5. Cover at least one non-contiguous input for layout-sensitive operators.
6. Run pytest with a focused command first.
7. Run broader pytest when the change affects shared behavior.
8. Record the command and exact result. If tests cannot run because of CUDA, GPU, dependency, or environment limits, record that honestly.

Result fields must not be invented. Use:

```text
待真实运行后填写
```

until the command has actually been run.

## Benchmark workflow

For benchmark work, record:

- Baseline: PyTorch operator, existing NineToothed implementation, or previous commit.
- Input size: exact shapes.
- Dtype: exact dtype for each input and output.
- Layout: contiguous, transposed, sliced, or other view.
- Command: exact command used to run the benchmark.
- Result: measured timing, throughput, or speedup with units.
- Conclusion: what the result means and whether it is actionable.

Never write only "performance is good". If no benchmark was run, write:

```text
待真实运行后填写
```

## Generated source and AOT workflow

Use this workflow for `ninetoothed.build` or `ninetoothed.make(..., caller="cuda")` tasks:

1. Confirm the target `output_dir` exists before calling `ninetoothed.build`.
2. Define a `premake` function returning `arrangement`, `application`, and `tensors`.
3. Define `configs` as `(args, kwargs, compilation_configs)` tuples.
4. Put performance-only knobs such as block sizes in `meta_parameters` when auto-tuning should choose them.
5. Check generated `.cpp`, `.h`, `.so`, `.csv`, and `.fingerprint` files only in the intended output directory.
6. Record whether cache reuse happened or a rebuild was forced.
7. Do not commit generated build artifacts unless the task explicitly requires them.

## Failure diagnosis workflow

For every failure, record:

- Symptom.
- Error message.
- Suspected root cause.
- Minimal fix.
- Re-run command.
- Re-run result.

Diagnosis should prefer small reproducible cases. Do not hide failures by deleting tests, loosening assertions without justification, or bypassing code paths.

## PR integration workflow

For this competition package:

1. Only change files under `skills/competition/lirui-ninetoothed-operator-skill/`.
2. Before PR, inspect `git diff --stat` and confirm the scope.
3. Run formatting and checks when the environment allows:

```bash
ruff format
ruff check
python scripts/check_contributing_style.py
pytest
```

4. If full pytest cannot run locally, record the reason and any focused tests that did run.
5. The PR description must include a `pytest` output code block.
6. PR title and commit message should start with an uppercase letter, use imperative mood, and not end with punctuation.

## Forbidden actions

The agent must not:

- Include hidden evaluation answers.
- Hard-code hidden task names or hidden test cases.
- Include API keys.
- Include private credentials or account tokens.
- Add online-only dependencies that are required for normal use.
- Delete tests to make results pass.
- Bypass tests or validation logic.
- Falsify pytest results.
- Falsify benchmark results.
- Perform unrelated refactoring.
- Commit unrelated files such as `.env`, `.venv`, `__pycache__`, `node_modules`, large build artifacts, or local logs unless explicitly requested.


