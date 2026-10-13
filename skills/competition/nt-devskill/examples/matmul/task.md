# Self-Test: MatMul (Matrix Multiplication)

## Category

MatMul / compute-bound operator.

## Input Task Statement

Implement a NineToothed matrix multiplication operator. Must use 3-level tiling (BLOCK_M, BLOCK_N, BLOCK_K) with K-loop and fp32 accumulator.

Required coverage:
- fp32 and fp16 correctness
- non-square matrices
- benchmark over multiple sizes
- Roofline classification (should be compute-bound for large shapes)

## Expected Agent Workflow

1. Search for existing matmul in `examples/matmul/`.
2. Use Pattern 6 (3D MatMul with K-loop) from `references/CODE_TEMPLATES.md`.
3. Use `block_size()` for autotuning BLOCK_M/N/K.
4. Implement K-loop with `ntl.dot` and fp32 accumulator.
5. Compare against `torch.mm`.
6. Run `bench_compare.py --op matmul --gpu metax` for Roofline.

## Implementation Notes

- **Arrangement**: 3-level tile: `tile((BLOCK_M, BLOCK_K))` + inner `tile((1, -1))` + expand.
- **Application**: K-loop with `acc += ntl.dot(a[k], b[k])`, fp32 accumulator.
- **libdevice**: Not needed — uses `ntl.dot` for matrix multiply.
- **Precision**: fp32 accumulator is mandatory; `acc.to(output.dtype)` at the end.
- **Composition**: bmm, addmm, conv2d all reuse this arrangement/application.

## Correctness

| Shape | dtype | Expected |
|-------|-------|----------|
| (512, 512) | fp16 | PASS |
| (1024, 1024) | fp16 | PASS |
| (1024, 1024) | fp32 | PASS |
| (256, 1024) x (1024, 512) | fp16 | PASS (non-square) |

## Performance

- **Bottleneck**: Compute-bound for shapes >= 1024x1024
- **Ridge point**: MetaX C500 ~100 GFLOP/s per GB/s
- **Autotuning**: `block_size()` for BLOCK_M ∈ [32,64,128], BLOCK_K ∈ [16,32,64]
- **Target**: >500 GFLOP/s on 4096x4096

## Known Limitations

- Only supports 2D x 2D → 2D matmul (use bmm for batched).
- Small shapes (< 256) are launch-bound, not compute-bound.
- Autotuning first call takes 30-120s.
