BangC backend
=============

The BangC backend compiles typed SSA to Cambricon BANG C and executes it through
the local CNCC/CNRT SDK. Install the matching ``torch_mlu`` package and make
``cncc`` available through ``PATH``, ``NEUWARE_HOME`` or
``NINETOOTHED_BANGC_COMPILER``.

Select ``backend="bangc"`` in ``ninetoothed.make``. The optional
``platform="cambricon-mlu590"`` selects the concrete MLU590 platform contract.
Use this explicit platform when exporting precise device metadata; the legacy
``generic`` profile description is retained, while BangC execution uses MLU.
The ``arch`` backend option defaults to ``native``; explicit values are
``compute_50`` or ``compute_30``. The compiler and resolved architecture are
recorded in the compilation cache identity.

This baseline follows the shared SSA access maps, masks, loops, reductions and
decomposed linear algebra. Bounded temporary tensor values in eligible SSA loops
and programs with multiple reductions are stored locally with typed storage and
region-yield lifetimes; this avoids unbounded producer recomputation in attention
and entropy programs. The storage plan uses an explicit resource budget and a
nonoverlap guard, retaining the original access path when it cannot prove
legality. One task otherwise walks a scalar output chunk. Large logical task
domains are split into bounded CNRT launches on the current MLU stream.
Atomic updates use a single task over the complete domain, preserving updates
without relying on unsupported hardware scalar atomics.

Integer data use paired 32-bit words for exact signed and unsigned 64-bit
arithmetic; address and shape calculations retain native index arithmetic.
Low-precision arithmetic preserves explicit FP16/BF16 rounding. Float8 is not
supported by this toolchain mapping.

Native built artifacts support AOT construction and
``ninetoothed.compiler.load_built_artifact`` reload. The shared native runtime
performs device and argument validation, binds the current MLU queue and pins
the compiled library identity.

The baseline intentionally has no vectorized NRAM dataflow, DMA schedules,
batched row execution or native matrix primitives. Those optimizations can be
added independently; large fused kernels may be slow on the scalar path.
