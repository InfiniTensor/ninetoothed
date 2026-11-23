# Task 2: Reduction / Block Self-Test Plan

## Task goal

Validate that the skill can guide an AI agent through a reduction-style NineToothed operator. Suggested operators are reduce sum, block max, or a softmax-like reduction. This task checks whether the agent handles block-local reductions, accumulator dtype, masks for partial blocks, and numerical tolerance.

## Related repository files

- `tests/test_softmax.py`: primary reference for reduction. It uses `Tensor(2, other=float("-inf")).tile((1, BLOCK_SIZE))`, `ntl.max`, `ntl.exp`, and `ntl.sum`.
- `tests/test_attention.py`: block computation with reductions in an attention-like pattern.
- `tests/test_generation.py`: basic generated-kernel tests and `ninetoothed.make` examples.
- `docs/source/basics.rst`: arrangement/application execution model.
- `src/ninetoothed/tensor.py`: `other` values and tile behavior for out-of-bounds positions.

## Input / output / shape / dtype / layout

- Input: 2D tensor with shape `(m, n)`.
- Output for reduce sum or block max: shape `(m,)` or `(m, 1)` depending on selected semantics.
- Output for softmax-like reduction: same shape as input.
- Dtype: begin with `torch.float32`; if `float16` is added, define a looser tolerance.
- Layout: start contiguous; add non-contiguous row or column slicing only after the basic case passes.

## Expected arrangement / application / Tensor pattern

1. Read `tests/test_softmax.py` first and mirror its reduction structure where appropriate.
2. Use a block-size symbol such as `BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)` or `ninetoothed.block_size()` depending on whether the task needs runtime specialization or auto-tuning.
3. Use `Tensor(..., other=...)` to define out-of-bounds values when partial blocks need neutral elements.
4. In `application`, use `ntl.sum`, `ntl.max`, or a stable softmax expression.
5. Use an accumulator dtype that matches the numerical requirement.
6. Compare with a PyTorch reference such as `torch.sum(input, dim=-1)`, `torch.max(input, dim=-1).values`, or `torch.softmax(input, dim=-1)`.

## PyTorch reference plan

Required cases:

- Normal shape such as `(128, 256)`.
- Non-power-of-two reduction dimension such as `(781, 1823)`, matching the spirit of `tests/test_softmax.py`.
- Small reduction dimension such as `(4, 7)`.
- Values that test numerical stability if softmax-like reduction is selected.
- Optional non-contiguous input after the contiguous baseline passes.

Reference examples:

```python
expected_sum = torch.sum(input, dim=-1)
expected_max = torch.max(input, dim=-1).values
expected_softmax = torch.softmax(input, dim=-1)
```

## Pytest command

```bash
pytest tests/test_softmax.py tests/test_attention.py -q
```

After a real reduction self-test implementation is added, record its exact real path and command here. Do not keep a fake path.

## Correctness test result

```text
待真实运行后填写
```

## Benchmark plan

Use one selected reduction operator consistently for baseline and candidate measurements. Warm up both implementations, use the same device and synchronization method, and record repeat count and timing units with the raw log.

Planned cases:

- `(1024, 1024)`, `float32`, contiguous.
- `(781, 1823)`, `float32`, contiguous, non-power-of-two reduction dimension.
- Optional `(4096, 257)`, `float32`, sliced or transposed if supported.

Do not infer speedup without numeric evidence.

| Case | Baseline | Shape | Dtype | Layout | Command | Mean time | Variance | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2-B1 contiguous reduction | 待真实运行后填写 | `(1024, 1024)` | `float32` | contiguous | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| T2-B2 partial reduction block | 待真实运行后填写 | `(781, 1823)` | `float32` | contiguous | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| T2-B3 layout variant | 待真实运行后填写 | `(4096, 257)` | `float32` | sliced or transposed | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |

## Benchmark result

```text
待真实运行后填写
```

## Failure diagnosis plan

Likely root causes to check:

- Wrong neutral `other` value for partial blocks.
- Reduction axis mismatch.
- Missing stable max subtraction for softmax.
- Accumulator dtype mismatch.
- Tolerance too strict for dtype.
- Layout assumption broken by non-contiguous input.

For every observed failure, record symptom, error message, suspected root cause, minimal fix, re-run command, and re-run result.

Current diagnosis:

```text
待真实运行后填写
```
