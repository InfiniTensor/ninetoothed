# Self-Test: Softmax Reduction

## Category

Reduction / block operator.

## Input Task Statement

Implement a row-wise softmax operator over the last dimension. Must use numerically stable formula (subtract max before exp).

Required coverage:
- non-power-of-two reduction length
- stable subtract-max formulation
- fp32 accumulation for fp16 input
- benchmark over multiple reduction sizes

## Expected Agent Workflow

1. Search for existing softmax, rms_norm in `examples/`.
2. Use Pattern 5 (2D Row-wise with Full Row) from `references/CODE_TEMPLATES.md`.
3. Set `BLOCK_SIZE = input.shape[-1]` to fit entire row.
4. Use `Tensor(2, other=float("-inf"))` for stable max.
5. Implement online softmax: max → subtract → exp → sum → normalize.
6. Compare against `torch.softmax(input, dim=-1)`.

## Implementation Notes

- **Arrangement**: `tile((1, BLOCK_SIZE))` where BLOCK_SIZE = last dim.
- **Application**: Online softmax with max-subtraction for numerical stability.
- **Precision**: `ntl.cast(x, ntl.float32)` before exp and sum.
- **Boundary**: `Tensor(other=float("-inf"))` handles partial tiles.
- **Performance**: Memory-bound for small N, compute-bound for large N.

## Correctness

| Shape | dtype | Expected |
|-------|-------|----------|
| (128, 1024) | fp32 | PASS |
| (128, 1024) | fp16 | PASS (atol=1e-3) |
| (4096, 4096) | fp32 | PASS |
| (100, 333) | fp32 | PASS (non-power-of-2) |

## Performance

- **Bottleneck**: Memory-bound (reads entire row, writes normalized row)
- **Target**: >100 GB/s on MetaX C500 for large shapes
- **Key factor**: BLOCK_SIZE must equal row length; no sweep needed

## Known Limitations

- Only supports softmax along dim=-1.
- BLOCK_SIZE must equal `input.shape[-1]`; large rows (>8192) may exceed private memory.
