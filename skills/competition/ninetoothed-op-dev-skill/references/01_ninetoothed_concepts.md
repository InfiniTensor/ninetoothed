# 01 — NineToothed concepts

## TOM and arrange-and-apply

NineToothed uses **tensor-oriented meta-programming (TOM)**. User code splits into:

1. **`arrangement`** — how logical tensors are tiled/expanded/viewed for parallel work  
2. **`application`** — serial semantics over arranged tiles (`ntl` ops)  
3. **`tensors`** — `Tensor(rank)` metadata tuple passed to `make`

```python
kernel = ninetoothed.make(arrangement, application, tensors)
kernel(*torch_tensors)
```

Official intro: `<repo-root>/README.md`, `<repo-root>/docs/source/basics.rst`.

## Key types

| Symbol | Role |
|--------|------|
| `Tensor(n)` | n-dimensional meta-tensor in arrangement |
| `Symbol("NAME", meta=True)` | Autotuning dimension |
| `Symbol(..., constexpr=True)` | Fixed at kernel launch |
| `block_size()` | Special symbol for block tiling |
| `ntl` | `import ninetoothed.language as ntl` — ops inside `application` |

## Alternative: `@ninetoothed.jit`

Some tests use decorated functions instead of explicit `make`:

```python
@ninetoothed.jit
def add_kernel(lhs: Tensor(1).tile((BLOCK_SIZE,)), ...):
    output = lhs + rhs
```

See `tests/test_add.py`. Prefer the style of your nearest reference file.

## JIT vs AOT

| API | When |
|-----|------|
| `ninetoothed.make` / `jit` | JIT; first call compiles |
| `ninetoothed.build` | AOT; emits `.so` to disk (`docs/source/build.rst`) |

## Generated source

Compiled kernels expose generated source path via `kernel._source` (see `test_generation.py`). Use for redundancy / load-store diagnosis.

## When to read more

- API details → `docs/source/python_api/*.rst`  
- Build/AOT → `docs/source/build.rst`  
- Repo file index → `00_repo_map.md`
