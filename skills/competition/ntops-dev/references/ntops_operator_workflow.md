# ntops Operator Workflow

Use this reference when implementing or auditing an `ntops` operator.

Source: derived from the checked-in `ntops` file layout, existing operators, exports, and tests. The workflow conventions are observations of this repository rather than new framework rules.

## Repository Roles

- `ninetoothed/`: NineToothed DSL and compiler implementation. Read for API behavior and generated-code concepts.
- `ntops/`: operator library built on NineToothed. Most operator tasks should modify this repository only.

## File Pattern

A typical operator has four surfaces:

- `src/ntops/kernels/<op>.py`: NineToothed arrangement/application/premake logic.
- `src/ntops/torch/<op>.py`: PyTorch-like wrapper that allocates output and calls `_cached_make`.
- `src/ntops/kernels/__init__.py`: kernel module import and `__all__` export.
- `src/ntops/torch/__init__.py`: public torch wrapper import and `__all__` export.
- `tests/test_<op>.py`: correctness test against PyTorch or a known reference.

## Pattern Selection

- Unary elementwise: start from `silu`, `tanh`, `relu`, or `gelu`.
- Binary elementwise/broadcast: start from `add`, `sub`, `mul`, or `div`.
- Matrix operators: start from `mm`, `bmm`, `addmm`, and `matmul`.
- Reductions/pooling: start from `softmax`, `max_pool2d`, `avg_pool2d`, or `rms_norm`.

“Start from” means inspect the live kernel, wrapper, and tests for repository conventions. It does not mean copy the nearest operator's arrangement. For a new reduction, compare a wrapper-normalized fixed-rank design with a direct arbitrary-rank arrangement. Prefer the smaller mapping when it satisfies the contract and verify copying/stride consequences explicitly.

Do not use an expanded zero-stride output view as a scatter target for multiple logical outputs. Keep the output descriptor aligned with the physical output and make the program-to-output mapping explicit.

## Test Expectations

- Use CUDA skip guards for GPU-only kernels.
- Cover float16 and float32 unless the operator contract excludes one.
- Include representative random shapes and boundary-sensitive cases.
- Compare against PyTorch with tolerances appropriate for dtype.
- Record the exact command and result in the self-test log.
- Before the parameterized suite, run fresh-process smoke cases for each structurally distinct path, especially full reduction, non-last-dimension reduction, and non-contiguous/offset input. A passing case in a warm process does not prove the final cached kernel works for other shapes or branches.

## Self-Test Record Format

Each self-test should include:

- Input task description.
- Files inspected by the agent.
- Files created or changed.
- Correctness command and result.
- Benchmark command and result, or CUDA-pending status.
- Failure diagnosis if anything fails.
- Known unsupported cases.
