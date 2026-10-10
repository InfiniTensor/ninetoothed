# 03 — Elementwise / broadcast patterns

## When to use

add, relu, gelu, mask, where, dtype-specific elementwise ops.

## Reference files

| Op | Path |
|----|------|
| add (`make`) | `<examples-root>/ops/ninetoothed/kernels/add.py` |
| add (`jit`) | `<repo-root>/tests/test_add.py` |
| pow / unary | `<repo-root>/tests/test_pow.py` |
| expand / broadcast | `<repo-root>/tests/test_expand.py` |
| dropout / mask-like | `<repo-root>/tests/test_dropout.py` |

## Contract checklist

- [ ] Same shape or explicit broadcast rules  
- [ ] `dtype` in parametrize matches torch ref  
- [ ] `device` from `get_available_devices()`  
- [ ] `torch.allclose` tolerances (`rtol`/`atol` for fp16)

## Minimal test pattern

```python
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
def test(..., dtype, device):
    ...
    assert torch.allclose(output, expected)
```

## Benchmark (optional)

Examples repo: `pytest -m benchmark` under `--examples-root` (⚠️ requires examples install).

## Common errors

- Using CPU tensor when tests expect CUDA  
- Forgetting `torch.empty_like` output buffer before kernel call  
- Mask dtype not bool / shape not broadcastable
