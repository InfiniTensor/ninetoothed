AscendC backend
===============

The AscendC backend compiles typed SSA with the local CANN CCEC compiler and
launches the resulting library on the current ``torch_npu`` stream. Install a
matching CANN and ``torch_npu`` environment and source the toolkit's
``set_env.sh`` before using the backend. Compiler discovery uses
``NINETOOTHED_ASCENDC_COMPILER``, ``PATH`` and the installed toolkit paths.

Select ``backend="ascendc"`` in ``ninetoothed.make``. For the Ascend910B2C,
``platform="ascend-910b2c"`` records the concrete device contract. Execution
uses NPU devices even with the legacy ``generic`` platform description.
The only backend option is ``arch``: ``native`` by default, or ``dav-c220``
and supported Ascend 910B family names. ``NINETOOTHED_ASCENDC_ARCH`` can supply
a concrete architecture for native selection. The resolved architecture and
local compiler identity participate in the compilation cache key.

The baseline uses the shared SSA walker for indexed and masked memory access,
broadcasting, loops, reductions and decomposed linear algebra. A single scalar
executor visits the logical task domain, including tails and in-place writes.
GM access uses byte-counted DMA through a reserved UB scratch slot, avoiding
scalar DCache writeback conflicts between disjoint views on different streams.
Transfers complete before the slot is reused; eligible tile arenas cannot
overlap this scratch region.
Eligible loop and reduction programs reuse the shared bounded tile planner.
Its local storage is allocated from one UB arena, with typed lifetimes and
simultaneous region yields. Pure elementwise values may be recomputed at their
consumers to fit the resource budget. A nonoverlap guard protects the tile
path; programs outside its legality conditions retain scalar access.

FP16 and BF16 operations preserve their declared result boundaries. Scalar
reductions accumulate these types in FP32 and convert the result back to its
declared type. Integer data and indices use native integer arithmetic.
FP16 host scalar arguments use an explicit 16-bit storage ABI. Built artifacts
support AOT construction and ``ninetoothed.compiler.load_built_artifact``
reload, including validation of this scalar ABI.

Atomic updates are rejected: serial execution within a kernel does not provide
atomicity with other streams. Float64 and float8 device arithmetic are also
unsupported. This baseline has no vector instruction selection, overlapped
DMA schedule, Cube matrix primitives or autotuning. Large fused kernels can
therefore be slow; target performance optimizations belong to subsequent
changes.
