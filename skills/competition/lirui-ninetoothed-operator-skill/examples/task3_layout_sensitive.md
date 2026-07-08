# Task 3: Layout-Sensitive Self-Test Plan

## Task goal

Validate that the skill forces layout-aware testing. This task checks transpose, slice, stride, storage offset, and non-contiguous inputs instead of only testing contiguous tensors.

The suggested operator can be elementwise identity/add, matmul-like read, or another simple operator where layout bugs are easy to isolate.

## Related repository files

- `tests/test_matmul.py`: matmul arrangement pattern and PyTorch comparison.
- `tests/test_conv2d.py`: complex layout transformation using `pad`, `tile`, `squeeze`, `ravel`, `flatten`, and `permute`.
- `tests/test_clone.py`: copy/clone-style behavior useful for layout-sensitive identity checks.
- `tests/test_getitem.py`: indexing and slicing behavior.
- `src/ninetoothed/tensor.py`: symbolic tensor shape, stride strings, slicing, `permute`, `flatten`, `ravel`, and `pad`.
- `src/ninetoothed/generation.py`: pointer, offsets, mask, and stride-related code generation.

## Input / output / shape / dtype / layout

- Input: tensor with base shape such as `(m, n)` or `(batch, m, n)`.
- Output: same logical shape as input for identity/add, or expected matmul/logical result shape for matmul-like tasks.
- Dtype: start with `torch.float32`; optionally extend to `torch.float16` with tolerance.
- Layouts required:
  - Contiguous.
  - Transpose.
  - Slice with non-unit stride.
  - View with storage offset.
  - Non-contiguous input verified by `tensor.is_contiguous() == False`.

## PyTorch reference construction for non-contiguous tensors

Use explicit layout constructors in the test body:

```python
base = torch.randn((m, n), dtype=dtype, device=device)
x_contiguous = base.contiguous()

x_transposed = torch.randn((n, m), dtype=dtype, device=device).t()
assert not x_transposed.is_contiguous()

x_sliced = torch.randn((m, n * 2), dtype=dtype, device=device)[:, ::2]
assert not x_sliced.is_contiguous()

x_offset = torch.randn((m + 2, n + 2), dtype=dtype, device=device)[1:-1, 1:-1]
assert x_offset.storage_offset() != 0

x_view = torch.randn((m, 2, n), dtype=dtype, device=device)[:, 0, :]
assert not x_view.is_contiguous() or x_view.storage_offset() != 0
```

Reference output should use PyTorch on the same logical tensor:

```python
expected = torch_reference(x_case)
```

## Expected arrangement / application / Tensor pattern

1. Start with a simple operator so layout is the variable under test.
2. Avoid assuming contiguous memory in test expectations.
3. Confirm the NineToothed-generated code receives shape and stride information through tensor arguments.
4. If the first implementation fails only on non-contiguous inputs, diagnose arrangement offsets and generated stride usage before changing tolerances.
5. Keep the non-contiguous test in place even after the fix.

## PyTorch reference plan

Required cases:

- Contiguous baseline.
- Transposed input.
- Sliced input with step `::2`.
- Offset view using `[1:-1, 1:-1]`.
- Shape with non-power-of-two dimensions, such as `(257, 1025)`.

For each case record:

- `shape`.
- `stride()`.
- `storage_offset()`.
- `is_contiguous()`.
- PyTorch reference comparison result.

## Pytest command

`ash
pytest tests/test_clone.py tests/test_getitem.py tests/test_matmul.py tests/test_conv2d.py -q
` 

After a real layout-sensitive self-test implementation is added, record its exact real path and command here. Do not keep a fake path.

## Correctness test result

```text
待真实运行后填写
```

## Benchmark plan

Benchmark at least two layouts:

- Contiguous baseline: `(1024, 1024)`, `float32`.
- Non-contiguous sliced or transposed input with the same logical shape.

Record whether layout changes performance, but do not claim an improvement without measured numbers.

## Benchmark command

```text
待真实运行后填写
```

## Benchmark result

```text
待真实运行后填写
```

## Failure diagnosis plan

Likely root causes:

- Generated offsets assume contiguous strides.
- Arrangement lost a dimension during `squeeze` or `flatten`.
- Output allocation is contiguous but input is not, and comparison accidentally uses a different logical shape.
- Test reference accidentally calls `.contiguous()` and hides the bug.

Current diagnosis:

```text
待真实运行后填写
```

