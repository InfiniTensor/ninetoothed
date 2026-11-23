# Debugging Guide

## Failure Classification

| Type | Symptom | First Check |
|------|---------|-------------|
| Environment | CUDA not available, import error | `torch.cuda.is_available()`, `pip install -e .` |
| Compilation | Triton/NineToothed compile error | Reduce to minimal shape, fix first error |
| Shape mismatch | Arrangement outermost shape inconsistent | Print shapes, verify tile alignment |
| Numerical | allclose fails | Compare max error, check mask/boundary/reduction dim |
| Layout | Non-contiguous input wrong result | Print `stride()`, `storage_offset()`, verify not assuming contiguous |
| Performance | Slower than baseline | Exclude JIT time (warmup), check generated source for redundant loads |

## Diagnosis Loop

1. Reproduce with minimal command
2. Classify failure type
3. Fix one hypothesis at a time
4. Re-run minimal test
5. Record: symptom, root cause, fix, verification command and result
