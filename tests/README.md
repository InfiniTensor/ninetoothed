# Running the tests

Run `pytest -ra` to include reasons for skipped tests and expected failures.

Generic numerical tests discover CUDA, MLU, and Ascend NPU devices. Importing
the optional `torch_npu` package registers the NPU runtime. For NPU cases,
`conftest.py` selects the Ascend backend and the platform corresponding to the
device name (including names such as `Ascend910B4-1`). Backend-specific CUDA
and Triton tests retain their original device requirements. Pure generation
and Python auto-tuner tests do not require an accelerator.

On Ascend 910B with triton-ascend 3.2.0 and CANN 9.0:

- FP8 matrix tests and `math.pow` require unsupported capabilities and skip
  with explicit reasons.
- Matrix and convolution examples use 32-element tiles and constexpr shapes
  on NPU to avoid the default 256-element tiles exceeding local memory.
- The matmul and addmm numerical tests currently hit a compiler crash in
  `triton-adapter-opt`. They still execute and have strict expected-failure
  marks restricted to `MLIRCompilationError`. Unexpected exceptions fail;
  a future successful run also fails as XPASS so the mark can be removed.
- Dropout uses PyTorch's dtype-aware comparison tolerances for float16
  division rounding.

The dedicated `test_ascend_runtime.py` and `test_ascend_aot.py` suites exercise
Ascend execution and artifact reload. CUDA-specific AOT, layout, and debugging
tests still need their corresponding hardware and toolchains.
