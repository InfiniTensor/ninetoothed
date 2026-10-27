# Running the tests

Run `pytest -ra` to include reasons for skipped tests and expected failures.

Run the complete test suite from the repository root with
`python -m pytest -q -ra --tb=short`. The CI command also enables
`--doctest-modules`, which imports documentation and visualization modules;
that mode needs the dependencies in `requirements.txt` and Python's Tkinter
module installed. A missing visualization dependency is a collection error,
not an accelerator kernel failure.

Generic numerical tests discover CUDA, MLU, and Ascend NPU devices. Importing
the optional `torch_npu` package registers the NPU runtime. For NPU cases,
`conftest.py` selects the Ascend backend and the platform corresponding to the
device name (including names such as `Ascend910B4-1`). Backend-specific CUDA
and Triton tests retain their original device requirements. Pure generation
and Python auto-tuner tests do not require an accelerator.

NineToothed stores Triton binaries under an environment-specific subdirectory of
`~/.ninetoothed/triton` (or `NINETOOTHED_CACHE_DIR/triton`). The directory identity
includes Python, PyTorch, torch_npu, Triton, and PyTorch's C++ ABI, so changing
environments rebuilds native launchers instead of loading incompatible cached
libraries. An explicit `TRITON_CACHE_DIR` overrides this isolation; use a separate
directory for each environment when setting it manually. Existing unpartitioned
cache files are no longer selected by default and need not be deleted.

On Ascend 910B with triton-ascend 3.2.0 and CANN 9.0:

- FP8 matrix tests and `math.pow` require unsupported capabilities and skip
  with explicit reasons.
- Matrix and convolution examples use 32-element tiles and constexpr shapes
  on NPU to avoid the default 256-element tiles exceeding local memory.
- Dropout uses PyTorch's dtype-aware comparison tolerances for float16
  division rounding.

The dedicated `test_ascend_runtime.py` and `test_ascend_aot.py` suites exercise
Ascend execution and artifact reload. CUDA-specific AOT, layout, and debugging
tests still need their corresponding hardware and toolchains.
