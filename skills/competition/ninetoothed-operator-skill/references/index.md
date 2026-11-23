# NineToothed Repository Index

Quick reference for navigating the upstream NineToothed repository during operator tasks.

## Top-Level Layout

NineToothed 是基于 Triton 的 DSL，核心采用 arrange-and-apply 模式：arrangement 在编译期描述 tensor meta-operation，application 描述每个 block 的计算逻辑。

| Path | Description |
|------|-------------|
| `src/ninetoothed/` | Core DSL, compiler, and runtime |
| `src/ninetoothed/make.py` | JIT kernel builder via `ninetoothed.make()` |
| `src/ninetoothed/jit.py` | Decorator-based kernel via `@ninetoothed.jit` |
| `src/ninetoothed/tensor.py` | `Tensor` meta-operations: `tile`, `permute`, `expand`, offsets |
| `src/ninetoothed/language.py` | `ntl` language primitives: load, store, max, sum, exp |
| `src/ninetoothed/generation.py` | Generated source and offset computation |
| `tests/` | Operator and framework tests (primary reference for test style) |
| `docs/source/basics.rst` | Arrange-and-apply tutorial |
| `docs/source/build.rst` | AOT kernel building with `ninetoothed.build` |
| `scripts/` | Contributing style checkers (ruff, etc.) |
| `CONTRIBUTING.md` | Fork, branch, commit, PR workflow |
| `README.md` | Project introduction |

## Core Concepts

- **arrangement** — Compile-time tensor meta-operations (`tile`, `expand`, `permute`, `squeeze`) that define how inputs map to programs/blocks.
- **application** — Per-block compute logic written with `ntl` or Python assignments; each program sees the inner block of arranged tensors.
- **tensors** — Symbolic tensor type signatures passed to `ninetoothed.make(arrangement, application, tensors)`.
- **generated source** — Triton/C++ code emitted by the compiler on first kernel invocation; inspect for load/store patterns and indexing bugs.
- **AOT build** — `ninetoothed.build(premake, configs, output_dir=...)` pre-compiles kernels to `.so` on disk; see `docs/source/build.rst`.

## Key Entry Points

| API | File | Notes |
|-----|------|-------|
| `ninetoothed.make` | `src/ninetoothed/make.py` | Returns cached kernel; good for multi-step arrangement |
| `ninetoothed.jit` | `src/ninetoothed/jit.py` | Decorator; good for single-function kernels |
| `ninetoothed.build` | `src/ninetoothed/build.py` | AOT compilation; writes `.so` + CSV to `output_dir` |
| `Tensor` | `src/ninetoothed/tensor.py` | Meta-operations and stride/offset semantics |
| `ntl` | `src/ninetoothed/language.py` | Block-level primitives |

## Recommended Test References

| Operator Type | Test File | What to Learn |
|---------------|-----------|---------------|
| Elementwise | `tests/test_add.py` | parametrize, device, allclose |
| Reduction | `tests/test_softmax.py` | row-wise reduction, normalization check |
| Layout | `tests/test_clone.py` | non-contiguous, stride, storage offset |
| Debugging | `tests/test_debugging.py` | failure reproduction patterns |

## External Resources

| Resource | URL |
|----------|-----|
| NineToothed Repository | https://github.com/InfiniTensor/ninetoothed |
| Documentation (Read the Docs) | https://ninetoothed.readthedocs.io/ |
| Operators (ntops) | https://github.com/InfiniTensor/ntops |
| Examples Repository | https://github.com/InfiniTensor/ninetoothed-examples |
| PyTorch Reference | https://pytorch.org/docs/stable/ |
| Competition Rules v0.8 | 赛道官方 PDF |

## Environment Note

本 skill 在 Windows 11 + RTX 5060 Laptop GPU (8GB) + CUDA 12.8 环境完成全部验证。所有 pytest 和 benchmark 均已在 GPU 上实际执行。
