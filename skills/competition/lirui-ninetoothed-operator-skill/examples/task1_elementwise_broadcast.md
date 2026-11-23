# Task 1: Elementwise / Broadcast Self-Test Plan

## Task goal

Validate that the skill can guide an AI agent through a small NineToothed elementwise operator with broadcasting. The suggested operator is broadcast add or masked add, because it exercises `tile`, `expand`, scalar or singleton broadcast handling, PyTorch reference comparison, and partial-block masking.

This task is intentionally simple enough to expose whether the agent follows the workflow instead of jumping straight into code.

## Related repository files

- `README.md`: matrix/vector examples and `ninetoothed.make` workflow.
- `docs/source/basics.rst`: arrange-and-apply paradigm and vector addition explanation.
- `tests/test_add.py`: direct elementwise add pattern using `ninetoothed.jit` and `Tensor(...).tile(...)`.
- `tests/test_addmm.py`: scalar parameters, `ninetoothed.make`, PyTorch reference, and tolerance comparison pattern.
- `tests/test_pow.py`: elementwise operation with scalar-like exponent and block tiling.
- `tests/utils.py`: device parametrization with `get_available_devices()`.
- `src/ninetoothed/generation.py`: imports Triton during NineToothed import and was part of the local collection failure traceback.

## Input / output / shape / dtype / layout

- Input A: tensor with shape `(m, n)` or `(size,)`.
- Input B: tensor with shape `(1, n)`, `(m, 1)`, `(n,)`, or scalar-like `Tensor(0)` depending on the chosen broadcast rule.
- Output: tensor with broadcasted shape matching PyTorch semantics for the selected case.
- Dtype: start with `torch.float32`; optionally extend to `torch.float16` if tolerance is specified.
- Layout: include a contiguous case and, if semantics allow, one non-contiguous case such as `a[:, ::2]` with a matching output shape.

## Expected arrangement / application / Tensor pattern

1. Search for `tests/test_add.py`, `tests/test_addmm.py`, and `tests/test_pow.py` before implementation.
2. Define `arrangement` to tile all tensor arguments into compatible blocks.
3. Use `expand` in the arrangement when singleton dimensions need to align with the output outer shape.
4. Define `application` as the block-local add or masked add expression.
5. Define `tensors` with `Tensor(1)`, `Tensor(2)`, or `Tensor(0)` according to the chosen input ranks.
6. Integrate with `ninetoothed.make` for the self-test implementation.
7. Keep the patch minimal and avoid unrelated formatting.

## PyTorch reference plan

Use PyTorch as reference:

```python
expected = input_a + input_b
```

Required cases:

- Same-shape add.
- Broadcast along leading dimension.
- Broadcast along trailing dimension.
- Non-power-of-two size to exercise the last partial block.
- Optional masked add with a boolean or numeric mask if the selected operator includes masking.
- At least one non-contiguous input if compatible with the selected operator.

## Pytest command

Focused baseline command attempted locally:

```powershell
$env:PYTHONPATH = "$PWD\src"
pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q
```

After a real self-test implementation is added, record its exact real path and command here. Do not keep a fake path.

## Correctness test result

This local run did not reach correctness execution. Pytest collection failed because the local Windows Python environment could import `ninetoothed` from `src`, but `ninetoothed` imports Triton and Triton is not installed in this environment.

This is recorded as a local environment limitation, not as an operator correctness failure.

Key error:

```text
ModuleNotFoundError: No module named 'triton'
```

Involved files from the traceback and command:

- `tests/test_add.py`
- `tests/test_addmm.py`
- `tests/test_pow.py`
- `src/ninetoothed/generation.py`

Log:

```text
skills/competition/lirui-ninetoothed-operator-skill/reports/logs/20260708-103141_pytest_t1_missing_triton_pytest_t1_missing_triton.log
```

TODO: rerun in a supported Linux/WSL/CUDA/Triton environment with repository dependencies installed.

## Benchmark plan

Baseline: PyTorch `input_a + input_b` or `torch.add(input_a, input_b)`.

Benchmark cases to record:

- Shape `(1024, 1024)`, dtype `float32`, contiguous.
- Shape `(4096, 257)`, dtype `float32`, broadcast along one dimension.
- Optional non-contiguous case if supported.

Record exact command, input sizes, dtype, layout, timing unit, and conclusion.

## Benchmark command

```text
待真实运行后填写
```

## Benchmark result

```text
待真实运行后填写
```

## Failure diagnosis plan

Recorded local failure diagnosis:

- Symptom: pytest collection failed.
- Error message: `ModuleNotFoundError: No module named 'triton'`.
- Root cause: local environment lacks Triton dependency required by the `ninetoothed` import path.
- Minimal fix: install a supported Triton environment or rerun inside a Linux/WSL/CUDA environment with repository dependencies installed.
- Re-run command: `pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q`.
- Re-run result: 待真实运行后填写.

This diagnosis must not be treated as a correctness failure for the elementwise/broadcast task.
