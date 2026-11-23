# Self-Test: Elementwise Add

## Category

Elementwise / binary operator.

## Input Task Statement

Implement a NineToothed elementwise add operator. The operator accepts two tensors of the same shape, produces an output matching `torch.add` semantics.

Required coverage:
- fp32 and fp16 correctness
- at least one non-power-of-two size
- non-contiguous input (transposed, strided)
- benchmark vs PyTorch

## Expected Agent Workflow

1. Search for existing add or elementwise kernels in `examples/`.
2. Restate shape, dtype, and layout assumptions.
3. Use Pattern 2 (1D Binary) from `references/CODE_TEMPLATES.md`.
4. Implement kernel with shared `element_wise` arrangement.
5. Compare against `torch.add` over shape x dtype x layout matrix.
6. Run `scripts/diag_overhead.py --op add` for performance diagnosis.

## Implementation Notes

- **Arrangement**: 1D tile with `BLOCK_SIZE` (shared `element_wise` arrangement).
- **Application**: `output = a + b` (trivial elementwise).
- **libdevice**: Not needed — uses native `+` operator.
- **Precision**: No fp32 upcast needed for simple addition.
- **Non-contiguous**: NineToothed handles stride automatically.

## Correctness

Expected: all shapes x dtypes x layouts PASS.

| Shape | dtype | Layout | Expected |
|-------|-------|--------|----------|
| (1024,) | fp32 | contiguous | PASS |
| (1024,) | fp16 | contiguous | PASS (atol=1e-3) |
| (4096, 4096) | fp32 | contiguous | PASS |
| (100, 333) | fp32 | contiguous | PASS (non-power-of-2) |
| (512, 256) | fp32 | transposed | PASS |

## Performance

- **Expected bottleneck**: memory-bound (arithmetic intensity < 1)
- **Target bandwidth**: >200 GB/s on MetaX C500
- **Tile sweep**: BLOCK_SIZE ∈ [256, 512, 1024, 2048, 4096]
- **Expected speedup**: 0.5-1.0x vs PyTorch (elementwise ops have high launch overhead)

## Known Limitations

- Does not support broadcasting (different shapes).
- Does not support complex dtypes.
- Small shapes (<256 elements) dominated by launch overhead.
