# 04 — Reduction / block patterns

## When to use

softmax, sum/max along dim, matmul, max_pool2d, conv2d.

## Reference files

| Op | Path |
|----|------|
| softmax | `tests/test_softmax.py`, `examples/.../softmax.py` |
| matmul | `tests/test_matmul.py`, `examples/.../mm.py` |
| max_pool2d | `tests/test_max_pool2d.py`, `examples/.../max_pool2d.py` |
| conv2d | `tests/test_conv2d.py`, `examples/.../conv2d.py` |
| attention | `tests/test_attention.py`, `examples/.../scaled_dot_product_attention.py` |

## Numerical stability (softmax)

Pattern from `test_softmax.py`:

```python
row_minus_max = input_row - ntl.max(input_row)
numerator = ntl.exp(row_minus_max)
denominator = ntl.sum(numerator)
output_row = numerator / denominator
```

## Block / window ops

`max_pool2d` example uses `WINDOW_HEIGHT/WIDTH` symbols + multi-step `tile`/`flatten`.

## Validation

```bash
pytest tests/test_softmax.py -v
pytest tests/test_max_pool2d.py -v
```

## Performance notes

- Autotuning (`Symbol(..., meta=True)`) can take minutes — disable for quick iteration (examples README)  
- Report `num_warps` / block sizes when benchmarking
