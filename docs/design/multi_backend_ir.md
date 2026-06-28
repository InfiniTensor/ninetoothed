# NineToothed Multi-Backend IR Design

## Objectives

NineToothed currently exposes a tensor-oriented metaprogramming DSL and lowers it to Triton. The target backend programming languages are now:

- Triton: keep the current production path and use it as the correctness baseline.
- CUDA: add native CUDA source generation while preserving the current AOT CUDA launcher path.
- TileLang: add a TileLang path for tile-level scheduling experiments.
- TVM: add a TensorIR/TVMScript path for a widely used tensor compiler stack.

SGLang is intentionally excluded from this backend list because it is an inference engine, not a kernel programming language target.

## IR Stack

The design should remain multi-level instead of forcing all targets through one flat IR:

- TOM Graph IR: tensor arrangement history, shapes, strides, jagged metadata, dtypes, constexpr values, and symbolic constraints.
- Program IR: loads, stores, pointer/index expressions, masks, arithmetic, dot, reductions, math intrinsics, helper calls, and function boundaries.
- Schedule IR: grid/block decomposition, thread/warp mapping, memory scopes, vectorization, unrolling, pipelining, and target capabilities.
- Target IR: backend-specific syntax, ABI metadata, runtime dispatch metadata, and compiler flags.

## First Implementation Slice

The current implementation slice adds a compact `KernelIR` bridge:

- `TensorTypeIR` records public tensor parameter metadata.
- `LaunchIR` records launch function name, launch args, and grid text.
- `KernelIR` records canonical source, entrypoint, launch metadata, compiler options, and extensible metadata.
- Backend lowerers consume `KernelIR` and return `BackendArtifact`.

This bridge does not replace the Triton generator yet. It creates a stable backend contract while the production Triton path remains untouched.

## Backend Status

| Backend | Current Status | Execution Status |
| --- | --- | --- |
| Triton | Existing production path plus artifact lowerer | Executable |
| CUDA | Emits `.cu` manifest and ABI shell; existing `caller="cuda"` AOT remains the only executable CUDA path | Native source shell is not executable |
| TileLang | Emits TileLang Python module shell and metadata | Not executable |
| TVM | Emits TVMScript module shell and metadata | Not executable |

## API Direction

```python
artifact = ninetoothed.lower(arrangement, application, tensors, backend="triton")
cuda_artifact = ninetoothed.lower(arrangement, application, tensors, backend="cuda")
tilelang_artifact = ninetoothed.lower(arrangement, application, tensors, backend="tilelang")
tvm_artifact = ninetoothed.lower(arrangement, application, tensors, backend="tvm")
```

Existing `make`, `jit`, and `aot` calls should keep working. New backend selection should remain explicit until native CUDA, TileLang, and TVM lowering are feature complete enough to pass the full correctness gates.

## Lowering Roadmap

1. Keep the bridge IR and backend registry in place.
2. Extract TOM Graph IR from `Tensor` arrangement history instead of reflecting only generated source.
3. Split semantic lowering from Triton rendering.
4. Add Program IR statement and expression nodes for loads, stores, masks, arithmetic, dot, reductions, and intrinsics.
5. Re-render Triton from Program IR and prove parity with the existing generator.
6. Implement native CUDA rendering for elementwise, reductions, and matmul-style tiled dot.
7. Implement TileLang rendering for elementwise, reductions, and matmul-style tiled dot.
8. Implement TVM TensorIR rendering for elementwise, reductions, and matmul-style tiled dot.
9. Add target capability descriptions for AscendC, BangC, and similar vendor languages.

## Validation Plan

The acceptance gate is intentionally strict:

- All existing NineToothed tests pass.
- All `ntops` tests pass.
- All runnable `ntops.lab` operators pass.
- Backend artifact audit passes for Triton, CUDA, TileLang, and TVM.
- Executable backend artifacts compile and run.
- Outputs match existing PyTorch/reference checks.

Until CUDA, TileLang, and TVM emit real executable target code, backend compatibility must be reported as incomplete.
