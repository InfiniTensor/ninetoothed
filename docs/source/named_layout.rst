Named layouts
=============

Named layouts remove the manual broadcast and hierarchy plumbing from common
one- and two-dimensional programs. They opt in through ``Tensor.axes()``;
existing unnamed programs and meta operations keep their previous meaning.

For example, the old GEMM arrangement is::

    output_tiled = output.tile((BM, BN))
    lhs_tiled = lhs.tile((BM, BK)).tile((1, -1)).expand((-1, output_tiled.shape[1]))
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)
    rhs_tiled = rhs.tile((BK, BN)).tile((-1, 1)).expand((output_tiled.shape[0], -1))
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

The named definition is::

    import ninetoothed as nt
    import ninetoothed.language as ntl

    BM, BN, BK = 16, 64, 32

    def arrangement(lhs, rhs, output):
        return (
            lhs.axes("m", "k").tile((BM, BK)),
            rhs.axes("k", "n").tile((BK, BN)),
            output.axes("m", "n").tile((BM, BN)),
        )

    def application(lhs, rhs, output):
        acc = ntl.zeros(output.shape, dtype=ntl.float32)
        for k in range(lhs.shape[0]):
            acc = acc + ntl.dot(lhs[k], rhs[k])
        output = acc.to(output.dtype)

    kernel = nt.make(
        arrangement, application,
        tuple(nt.Tensor(2, dtype="float16") for _ in range(3)),
    )

The application sees a sequence of ``ceil(K / BK)`` blocks for each input.
``lhs[k]`` has shape ``(BM, BK)``, ``rhs[k]`` has shape ``(BK, BN)``, and
``output`` is the current ``(BM, BN)`` block. ``k`` is a block index, not an
element index. Axis names identify matching data; ``dot`` specifies the local
contraction; the ordinary loop specifies the cross-block accumulation order.
There is no implicit reduction or new loop interface.

Contract
--------

Call ``axes`` on a root tensor, followed by at most one ``permute`` and exactly
one ``tile``. Names are unique nonempty strings. ``axes("mk")`` denotes one
axis, not two. A transpose input must explicitly permute its logical layout;
alignment never substitutes a matrix transpose merely because sizes match.

Declared tile sizes belong to the layout, not to a backend preset list. Fixed
integers, bounded compile-time symbols and explicit specialization use the same
layout rules regardless of symbol spelling. Backend candidates provide execution
strategies for that layout; their effective tile metadata is derived from SSA
block shapes. A missing preset is not a target capability failure. Actual target
and resource restrictions still apply. Runtime tuning candidates retain their
own resolved tile values when a layout symbol has multiple legal values.

All positive-rank arguments in a named arrangement must be named. Scalars
need not be named. Same-name axes require equal logical extents and equal
tile sizes. Only an absent axis broadcasts; a same-name size-one axis is not
silently broadcast. For example a column bias uses ``axes("n").tile((BN,))``
and a row scale uses ``axes("m").tile((BM,))``.

There is one final output, identified from its full-tile assignment, not from
its parameter name or return position. Both ``output = value`` and
``output[...] = value`` are accepted. No output may be read before that write.

At most one top-level non-nested ``range`` loop and one loop-carried value are
accepted. Its bound is either paired stream's ``shape[0]``; zero start and
explicit unit step are optional. Both streams use the same unmodified loop
index. Local aliases of extracted blocks are allowed. Offset indexing, early
exit, multiple states and unsupported control flow are rejected.

This first version supports pointwise work, missing-axis broadcasting,
reordering, GEMM, and same-output-domain bias/ReLU/residual postprocessing.
It does not add batch, row reductions, attention, scans, gather, windows,
multiple outputs, nested tiles, indirect pointers, asynchronous memory, or
automatic fusion. These restrictions apply only to named programs.

Runtime and compilation
-----------------------

The frontend lowers named views into the existing TensorLayout, AccessMap
and SSA operations. Emitters and default optimization pipelines do not
interpret axis names. The frozen module and LaunchABI retain the versioned
layout obligations so specialization, caching and reload cannot drop checks.

Logical sizes must be positive; input strides must be positive. Noncontiguous
inputs are allowed. Output writes must be provably non-overlapping, and
output storage must not overlap any input. Runtime checks run before tuning
or kernel work; inability to prove safety is an error.

Checked Torch calls on the Triton JIT runtime are supported first, including
JIT artifact reload. CUDA and TileLang support device-free source generation
for these definitions; named execution on those backends and C/AOT callers
is explicitly rejected in this version. This does not restrict their legacy
unnamed callers. No extra kernel launch or global intermediate tensor is
introduced by layout alignment. Performance equivalence requires comparing
the actual layouts and generated code, not assuming all frontend changes
are free.
