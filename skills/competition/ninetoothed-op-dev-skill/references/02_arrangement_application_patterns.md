# 02 — Arrangement / application patterns

## When to use

Implementing any new kernel: define how tiles map to memory (`arrangement`) and what compute runs per tile (`application`).

## Search order

1. `<examples-root>/ops/ninetoothed/kernels/<op>.py` (if `--examples-root` is set)  
2. `<repo-root>/tests/test_<op>.py`  
3. `<repo-root>/README.md` (matmul example)

## Common arrangement moves

| Pattern | Example location |
|---------|------------------|
| `tile((BLOCK,))` 1D | `<examples-root>/.../add.py` |
| `tile((M,N))` 2D | `<repo-root>/tests/test_softmax.py` (row tiles) |
| `expand` broadcast | `<repo-root>/README.md` matmul arrangement |
| `ravel` / `flatten` | `<examples-root>/.../max_pool2d.py` |
| `dtype.squeeze` on arranged tensor | `max_pool2d.py`, matmul README |

## Application patterns

- Elementwise: `output = input + other`  
- Reduce: `ntl.max(input, axis=1)`, `ntl.sum`, softmax stable form in `test_softmax.py`  
- Matmul: accumulator loop with `ntl.dot` in README  

## `make` kwargs (performance)

From `<repo-root>/tests/test_generation.py`:

- `num_warps`, `num_stages` — Triton launch params  
- Passing tuples triggers autotuning paths  

For fast correctness dev: fix block sizes as integers / `constexpr=True`.

## Validation

```bash
cd <repo-root>
pytest tests/test_<nearest_op>.py -v --tb=short
```

## Failure notes

- Wrong `tile`/`expand` → shape mismatch or silent wrong results; compare torch ref first  
- Mixing `jit` and `make` styles in one op → pick one reference and stick to it
