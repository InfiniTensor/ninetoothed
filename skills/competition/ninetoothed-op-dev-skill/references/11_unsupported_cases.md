# 11 — Unsupported cases

Declare explicitly in every task report when applicable.

## Environment

| Condition | Statement template |
|-----------|-------------------|
| No CUDA | GPU kernel tests not executed; logic documented only |
| sm_120 GPU + wrong torch | Requires PyTorch nightly `cu128`; stable cu124 insufficient |
| `nvcc` not found | AOT compile skipped; analysis limited to generated source / docs |
| WSL GPU passthrough fail | `nvidia-smi` in WSL must work before GPU tests |

## Operator / dtype

| Condition | Notes |
|-----------|-------|
| CPU-only execution | Most ninetoothed tests use `get_available_devices()` → CUDA only |
| `mlu` | Only if `torch_mlu` installed — rare in competition env |
| Dynamic shape beyond test patterns | State fixed shapes used; no claim of full dynamic support |
| dtypes not in nearest test | Do not invent; extend from closest `parametrize` list |

## Performance

| Condition | Notes |
|-----------|-------|
| Autotuning disabled for time | Benchmark numbers are **not** peak performance |
| Examples not installed | `<examples-root>` benchmarks require `pip install -e .` |

## Skill boundaries

- Does not modify NineToothed compiler core by default  
- Does not provide hidden evaluation answers or task-specific bypasses  
- Does not run network installs automatically — user controls `pip`/`git`

## Example report snippet

```markdown
## Unsupported
- nvcc: not installed — AOT build not executed
- dtypes: int64 elementwise not implemented; float32/float16 only
```

## float16 correctness

- Covered by `examples/01_elementwise_broadcast_add/tests/test_example_broadcast_add.py::test_broadcast_add_float16_matches_torch`
- Uses explicit `atol=1e-2`, `rtol=1e-2`
- Benchmark evidence remains **float32 only**; do not extend performance claims to float16

