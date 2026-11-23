# NineToothed Operator Development Map

Condensed, source-verified reference (repo version 0.26.0). Read this when unsure
about the API or a pattern. External resources: operators
<https://github.com/InfiniTensor/ntops>, examples
<https://github.com/InfiniTensor/ninetoothed-examples>, docs <https://ninetoothed.org/>.

## 1. The paradigm: arrange-and-apply

NineToothed is a Triton-based DSL built on **tensor-oriented meta-programming**.
You do not write pointer arithmetic. Instead:

1. **Symbolic tensors** describe data — they store symbolic `shape`/`strides`, not values.
2. **Meta-operations** (`tile`, `expand`, `squeeze`, `permute`, `flatten`, `ravel`,
   `pad`) reshape/retile tensors at compile time.
3. The compiler launches one program per element of the **outermost** arranged
   tensor and maps the **second-outermost** tensor (a block) into each program.

Define `arrangement` (how to tile), `application` (what each block computes), and
`tensors` (ranks), then combine with `ninetoothed.make`.

## 2. Two equivalent forms

**Decorator form** (simple ops, tile inline in annotations):

```python
def add(lhs, rhs):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    @ninetoothed.jit
    def add_kernel(
        lhs: Tensor(1).tile((BLOCK_SIZE,)),
        rhs: Tensor(1).tile((BLOCK_SIZE,)),
        output: Tensor(1).tile((BLOCK_SIZE,)),
    ):
        output = lhs + rhs  # noqa: F841

    output = torch.empty_like(lhs)
    add_kernel(lhs, rhs, output)
    return output
```

**make form** (complex ops, explicit arrangement/application):

```python
def arrangement(lhs, rhs, output):
    ...  # returns (lhs_arranged, rhs_arranged, output_arranged)

def application(lhs, rhs, output):
    ...  # params are BLOCKS, not whole tensors

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))
```

**Critical:** in `application`, parameters are the second-outermost elements
(blocks), not the original tensors.

## 3. API cheat sheet

Exported from `ninetoothed`: `Symbol`, `Tensor`, `block_size`, `make`, `build`,
`jit`, `eval`, `subs`, dtypes (`float16/32/64`, `bfloat16`, `int8/16/32/64`, `uint*`).

```python
Symbol("BLOCK_SIZE", meta=True)        # compiler auto-tunes this
Symbol("BLOCK_SIZE", constexpr=True)   # compile-time constant, pass at call time
Symbol("WIN_H", constexpr=True, upper_bound=16)
ninetoothed.block_size()               # == Symbol(meta=True) block size

Tensor(2)                              # 2-D, symbolic shape
Tensor(shape=(4, 8))                   # concrete shape
Tensor(2, other=float("-inf"))         # out-of-bounds fill value
Tensor(4, dtype=ninetoothed.float16)
```

Meta-operations (`src/ninetoothed/tensor.py`): `tile(tile_shape, strides=None,
dilation=None, floor_mode=False)`, `expand(shape)`, `squeeze(dim)`,
`unsqueeze(dim)`, `permute(dims)`, `flatten(start_dim, end_dim)`, `ravel()`,
`pad(pad)`. `-1` in a tile/expand shape means "take the full extent / keep".

`ninetoothed.language as ntl` primitives seen in use: `ntl.zeros(shape, dtype=)`,
`ntl.dot(a, b)`, `ntl.max(x[, axis=])`, `ntl.sum(x)`, `ntl.exp(x)`, `ntl.float32`,
`.to(ntl.float16)`.

## 4. Per-lane templates

### T1 Elementwise / broadcast (see `test_add.py`)
Tile each tensor `tile((BLOCK_SIZE,))`; body `output = lhs + rhs`. Broadcast by
aligning arranged shapes with `expand`. Handle boundaries with `Tensor(other=...)`.
Verify against `input <op> other`.

### T2 Reduction / block (see `test_softmax.py`, `test_max_pool2d.py`)
Softmax: `tile((1, BLOCK_SIZE))`, `BLOCK_SIZE=constexpr` set to row width; body
`ntl.max/exp/sum`. Pooling: `tile((1,1,WIN_H,WIN_W)).ravel().flatten(...).tile((BLOCK_SIZE,-1))`;
body `ntl.max(input, axis=1)`. **Fill `other=-inf` (max) / `0` (sum)** or boundary
blocks break. Verify against `torch.softmax` / `F.max_pool2d`.

### T3 Layout-sensitive / matmul-style (see `test_matmul.py`, `test_getitem.py`)
Three-level tile: `tile(block).tile((1,-1)).expand((-1, out.shape[1]))`, then
`dtype = dtype.squeeze(0)`. **All arranged outermost shapes must match** — align
with `expand`. In `application`: `acc = ntl.zeros(...); for k in range(input.shape[0]):
acc += ntl.dot(input[k], other[k])`. Never assume contiguous — test `x.T`, `x[::2]`.
Verify against `torch.matmul` (fp16 needs `atol`).

### T4 Performance / diagnosis / integration
- **Generated source:** `make(..., kernel_name="x", output_dir=path)` writes codegen to disk.
- **Simulate layout:** `ninetoothed.debugging.simulate_arrangement(arrangement, tensors)`.
- **AOT build** (`docs/source/build.rst`): `ninetoothed.build(premake, configs,
  meta_parameters=(...), kernel_name=, output_dir=, lazy=True)`. `premake(...)`
  returns `(arrangement, application, tensors)` per variant; `configs` are
  `(args, kwargs, compilation_configs)` triples where the third slot holds
  `num_warps`/`num_stages`; `meta_parameters` are auto-tuned into a CSV at build
  time and chosen at runtime by the generated dispatcher; non-meta params (dtype)
  compile a separate `.so` each. `output_dir` must exist; `lazy=True` +
  `if __name__=="__main__":` guard are required (spawn re-import deadlock).
  Same `output_dir` with existing `.so`+CSV skips rebuild; delete to force.
- **Benchmark:** no built-in script — use `triton.testing.do_bench` (see `scripts/bench.py`).

## 5. Testing conventions (from `tests/`)

- Parametrize `device` from `tests/utils.py: get_available_devices()` — returns
  cuda/mlu, **empty off-GPU so tests SKIP (not fail)**. Record skips honestly.
- `tests/conftest.py` seeds RNG by module/test name for reproducibility.
- Assert `torch.allclose(output, expected)`; give `atol` for fp16.
- Standard test skeleton: build random inputs → call the NineToothed op → build a
  PyTorch reference `expected` → `allclose`.

## 6. Repo compliance (enforced by hooks/CI)

- Env: `pip install -e .[all]` + `pip install -r requirements.txt` +
  `git config core.hooksPath .githooks`.
- Branch: kebab-case, ≤50 chars. Commit/PR title: capitalized, imperative, no
  trailing punctuation.
- PR description must contain pytest output.
- Local CI: `python scripts/check_contributing_style.py --fix` → `ruff format` →
  `ruff check` → `python scripts/check_contributing_style.py` → `pytest`.
- Style: PEP8; comments are full sentences; blank line around `if`/`for` and before
  `return`; no blank line between a docstring-less signature and its body.
- **Never** touch `src/ninetoothed/`; never `--no-verify`.

