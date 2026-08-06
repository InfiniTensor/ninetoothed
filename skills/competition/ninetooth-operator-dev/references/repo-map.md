# NineToothed Repository Map

Source clone used for this framework:

- remote: `git@github.com:InfiniTensor/ninetoothed.git`
- upstream checkout path: set `NINETOOTHED_REPO` to your local clone
- observed commit: `c9ebd4950a185beed8d4c1db9ff4a1fd133934ae`
- observed date: 2026-07-14

This file is a map, not a manual. Prefer searching the live repository before
trusting any stale example in this skill.

## How To Read Upstream

1. Start with `README.md` to refresh the arrange-and-apply paradigm.
2. Search `tests/` for the closest operator shape before opening core library
   files.
3. Open only the core file that explains the concept currently blocking the
   task.
4. Return to tests for the correctness style, dtype choices, device fixtures,
   and tolerated numerical error.

## Package Entry Points

- `README.md` - arrange-and-apply overview and the minimal matrix multiplication
  example.
- `pyproject.toml` - package metadata, `python >=3.10`, dependencies
  (`triton`, `sympy`, `numpy`), optional `torch`, and Ruff settings.
- `src/ninetoothed/__init__.py` - public exports such as `Tensor`, `Symbol`,
  `make`, `jit`, dtype names, and `block_size`.
- `src/ninetoothed/make.py` - connects `arrangement`, `application`, and
  `Tensor` metadata; dispatches to JIT or AOT.
- `src/ninetoothed/jit.py` - generated source path, dynamic import, launch
  handle, and default `num_warps` / `num_stages`.
- `src/ninetoothed/tensor.py` - symbolic tensor metadata and meta-operations:
  `tile`, `expand`, `pad`, `permute`, `flatten`, `ravel`, `squeeze`,
  `unsqueeze`, offsets, strides, and jagged metadata.
- `src/ninetoothed/language.py` - `ninetoothed.language` symbol bridge; tests
  import it as `ntl` for reductions, dot products, exp, where, zeros, and dtype
  casts.
- `src/ninetoothed/generation.py` - code generation, offset/mask generation,
  generated source bookkeeping, and cache paths.
- `src/ninetoothed/aot.py` - AOT source generation, caller ABI, tensor shape and
  stride handling, overflow checks, and contiguity/divisibility variants.
- `src/ninetoothed/build.py` - package/build helper path and benchmark timing
  via `triton.testing.do_bench`.
- `src/ninetoothed/auto_tuner.py` - auto-tuning cache, timing collection, and
  best function selection.
- `src/ninetoothed/debugging.py` - `simulate_arrangement` for inspecting source
  and target tensors without committing to a full operator run.
- `src/ninetoothed/eval.py` - symbolic tensor evaluation used by lower-level
  tests for arrangement behavior.
- `tests/utils.py` - device discovery helper used by operator tests.

## Search Anchors

Run searches from your NineToothed checkout, for example
`cd "$NINETOOTHED_REPO"`, unless a command says otherwise.

```bash
rg -n "def arrangement|def application|ninetoothed\\.make|@ninetoothed\\.jit|Tensor\\(" tests src
rg -n "tile\\(|expand\\(|pad\\(|permute\\(|flatten\\(|ravel\\(|squeeze\\(|unsqueeze\\(" tests src
rg -n "other=|shape_options|constexpr|upper_bound|meta=True|block_size\\(" tests src
rg -n "ntl\\.|language as ntl|dot\\(|sum\\(|max\\(|where\\(|exp\\(|zeros\\(" tests src
rg -n "aot|generated|_source|output_dir|kernel_name|auto_tun|do_bench|simulate_arrangement" tests src
rg -n "stride|offset|contiguous|jagged|floor_mode|dilation|mask|overflow" tests src
```

## Operator Test Map

- Elementwise and scalar-style:
  - `tests/test_add.py` - direct `@ninetoothed.jit` elementwise add over a 1-D
    tiled tensor.
  - `tests/test_pow.py` - elementwise power with one tensor exponent argument.
  - `tests/test_dropout.py` - scalar parameters, random seed, and elementwise
    mask-like behavior.
  - `tests/test_clone.py` - simple copy/application variants.
  - `tests/test_data_ptr.py` - shape constexpr path and output expansion.
- Broadcast, shape, and indexing:
  - `tests/test_expand.py` - `expand` behavior.
  - `tests/test_unsqueeze.py` - `unsqueeze`, `expand`, and tile composition.
  - `tests/test_pad.py` - pad-style arrangement and slicing parameters.
  - `tests/test_getitem.py` - indexing behavior for symbolic tensors.
  - `tests/test_generation.py` - generation edge cases, indexing, scalar
    constexpr arguments, and generated-source assertions.
- Reduction and blocked operators:
  - `tests/test_softmax.py` - row-wise reduction using `ntl.max`, `ntl.exp`,
    and `ntl.sum` with out-of-bounds `other=-inf`.
  - `tests/test_max_pool2d.py` - tiled window reduction, `floor_mode`, and
    `ceil_mode`.
- Matmul-family and layout-heavy operators:
  - `tests/test_matmul.py` - core blocked matmul arrangement with nested tiles,
    expand, dtype squeeze, `ntl.dot`, and float16 output.
  - `tests/test_addmm.py` - matmul plus scalar alpha/beta inputs.
  - `tests/test_conv2d.py` - im2col-like transform using `pad`, `tile`
    `strides`, `squeeze`, `ravel`, `flatten`, `permute`, and the matmul
    application.
  - `tests/test_attention.py` - attention-like blocked arrangement, causal
    masking, upper-bound shape options, and `ntl.where`.
- Jagged and non-regular layouts:
  - `tests/test_jagged.py` - `Tensor(..., jagged_dim=...)`, nested tensor
    conversion, offsets, and padded copy.
  - `tests/test_eval.py` - symbolic shape, stride, tile, dtype, and eval
    behavior for arrangement-level reasoning.
- AOT, generated source, tuning, and debugging:
  - `tests/test_aot.py` - AOT kernels, dtype metadata, generated output
    directory, caller handling, overflow checks, and static non-power-of-two
    sizes.
  - `tests/test_aot_auto_tuning.py` - AOT auto-tuning with block-size choices.
  - `tests/test_auto_tuner.py` - auto-tuner cache and timing behavior.
  - `tests/test_debugging.py` - expected `simulate_arrangement` source/target
    tensors for small shapes.

## Core Concept Anchors

- `arrangement` answers how symbolic tensors are reshaped, tiled, padded,
  expanded, indexed, or mapped into a computation view.
- `application` answers what computation runs over the arranged tensors; most
  operator tests use `ninetoothed.language as ntl` for reductions and math.
- `Tensor` metadata controls ndim, shape, dtype, constexpr scalar behavior,
  `other` out-of-bounds values, shape options, jagged dimensions, offsets, and
  source strides.
- `ninetoothed.make` is the main integration path for separated
  `arrangement`/`application`; direct `@ninetoothed.jit` appears in simpler
  elementwise tests.
- Generated-source and AOT work usually needs `kernel._source`,
  `ninetoothed.generation.CACHE_DIR`, explicit `kernel_name`, `output_dir`,
  `caller`, dtype metadata, and launch configuration.
- Correctness tests generally compare against PyTorch or repository behavior
  with `torch.allclose`, parametrized dtype/shape/device fixtures, and explicit
  tolerance for numerically sensitive operators.

## First File By Task Type

- Elementwise/broadcast task: start with `tests/test_add.py`, then search
  `test_pow.py`, `test_dropout.py`, `test_expand.py`, and `test_generation.py`.
- Reduction task: start with `tests/test_softmax.py`; for windowed reductions
  open `tests/test_max_pool2d.py`.
- Blocked matmul-like task: start with `tests/test_matmul.py`; for composition
  open `tests/test_addmm.py`, `tests/test_conv2d.py`, or `tests/test_attention.py`.
- Layout-sensitive task: start with `src/ninetoothed/tensor.py`, then open
  `tests/test_conv2d.py`, `tests/test_jagged.py`, `tests/test_eval.py`, and
  `tests/test_debugging.py`.
- Generated source or AOT task: start with `tests/test_generation.py` or
  `tests/test_aot.py`, then open `src/ninetoothed/generation.py`,
  `src/ninetoothed/aot.py`, and `src/ninetoothed/jit.py`.
- Benchmark or auto-tuning task: start with `src/ninetoothed/build.py`,
  `src/ninetoothed/auto_tuner.py`, `tests/test_auto_tuner.py`, and
  `tests/test_aot_auto_tuning.py`.
