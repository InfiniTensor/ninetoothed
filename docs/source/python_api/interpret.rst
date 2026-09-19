CPU Reference Interpreter
==========================

``ninetoothed.interpret`` executes a lowered ``ssa.Program`` with NumPy instead of
a GPU backend. It reuses the existing lowering chain: the arrangement produces the
layout and index mapping, the Python frontend produces the ``ssa.Program``, and
the interpreter then walks that program once per program instance.

Because no CUDA device is involved, applications can be developed, traced and
diff-tested on any machine. The interpreter is also useful on a GPU host: it is
the reference implementation a kernel is compared against when a numeric
discrepancy has to be attributed to either the application or the compiler.

.. note::

   The interpreter is a reference implementation, not a fast one. It executes one
   program instance at a time in Python, so it is meant for small inputs
   (hundreds to a few thousand elements), not for production runs.

Quick Start
-----------

.. code-block:: python

    import numpy as np

    import ninetoothed.language as ntl
    from ninetoothed import Tensor, block_size
    from ninetoothed.interpret import interpret

    BLOCK_SIZE = block_size()


    def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
        return x.tile((1, BLOCK_SIZE)), out.tile((1, BLOCK_SIZE))


    def application(x, out):
        shifted = x - ntl.max(x, axis=1)[:, None]
        numerator = ntl.exp(shifted)
        out = numerator / ntl.sum(numerator, axis=1)[:, None]


    x = np.random.default_rng(0).random((3, 11), dtype=np.float32)
    out = np.zeros_like(x)

    result = interpret(
        arrangement,
        application,
        tensors=(Tensor(2, other=float("-inf")), Tensor(2)),
        inputs=(x, out),
        symbols={"BLOCK_SIZE": 16},
    )

    print(result.launch_shape)          # (3, 1)
    print(result.output("out"))

Symbols may be bound either by their internal (prefixed) name or by the name of
the arrangement parameter they are bound to, so ``{"BLOCK_SIZE": 16}`` works
even though the frontend renames the symbol.

Execution Model
---------------

The SSA program has a single entry block and no explicit program loop: on a GPU a
program instance is identified by ``program_id``, which never appears in the SSA.
The interpreter therefore executes the entry block once per program instance and
exposes the instance id to the layout expressions as ``outer_index``.

The **launch domain** is the view domain of the primary output, which is the
target of the first ``mem.store``. A tensor that was tiled exposes its tile grid;
a tensor that was only sliced is a single tile covering the whole view domain,
giving a launch shape of ``(1,)``. Every tiled tensor in the arrangement must
imply the same number of instances; a mismatch raises a
:class:`~ninetoothed.interpret.ProgramDomainError` instead of being ignored.

Dtype Levels
------------

A tensor carries a hierarchy of *dtype levels* built by successive ``tile``
calls. ``dtype_shapes`` records the extent of each level; the application sees
level ``0`` and the innermost level is the one the access map resolves. For a
nested tile such as ``x.tile((1, BLOCK)).tile((1, -1))`` the application must
index the outer level before reading elements:

.. code-block:: python

    def application(x, out):
        acc = ntl.zeros(out.shape, dtype=out.dtype)

        for i in range(x.shape[1]):
            acc += ntl.sum(x[0, i], axis=-1)   # index level 0, read level 1

        out = acc

Reading a partially indexed view raises
:class:`~ninetoothed.interpret.UnsupportedAccessError` naming the level and
pointing at ``tensor.extract``.

Masking and Bounds
------------------

The CPU memory model enforces two invariants:

* a masked-out access never touches the backing buffer: it contributes ``other``
  (the value declared on the ``Tensor`` descriptor, ``0`` by default) when
  reading, and is skipped when writing;
* an access that is not masked out but falls outside the buffer is an error, not
  a silent read of the wrong element.

This mirrors what the generated GPU code does, so a discrepancy caused by an
incorrect arrangement surfaces on the CPU instead of as a wrong answer or a
segmentation fault on the device.

Dtypes
------

Integer and boolean semantics are bit-exact: integer casts truncate towards zero
exactly like the C casts the CUDA backend emits, and ``%`` uses C-style
remainder, so the sign follows the dividend. Floating point results are produced
by NumPy in the requested width, so ``float32`` is never silently widened to
``float64`` and comparisons against a NumPy or PyTorch reference stay within the
documented tolerance.

``bfloat16`` and the ``float8`` formats are understood but refused with an
explicit :class:`~ninetoothed.interpret.UnsupportedDTypeError`, because the host
cannot represent them exactly.

Supported Operations
--------------------

Every operation is registered by opcode in a table, so new operations can be
added without touching the interpreter core. The support matrix is available at
runtime:

.. code-block:: python

    from ninetoothed.interpret import format_support_matrix

    print(format_support_matrix())

The families covered are ``arith.*`` (arithmetic, bitwise, unary, ``constant``),
``cmp.*``, ``math.*``, ``select.where``, ``tensor.*`` (``zeros``, ``full``,
``empty``, ``extract``, ``view``, ``cast``, ``stride``), ``shape.dim``,
``index.offset``, ``symbol.attr``, ``tuple.construct``, ``mem.*`` (``load``,
``store``, ``data_ptr``), ``reduce.*``, ``linalg.*`` and ``scf.*``.

Two operations are knowingly unsupported and say so with a specific reason:

.. list-table::
   :header-rows: 1

   * - Operation
     - Reason
   * - ``mem.atomic_add``
     - Accumulation order is not deterministic, so a reference result is not
       well defined.
   * - ``math.rand``
     - Random draws are not reproducible across backends.

Target intrinsics such as ``triton.cdiv`` arrive as ``call.*`` opcodes and have
no CPU reference implementation; the interpreter reports them by name and
suggests replacing them with an equivalent arithmetic expression.

Tracing
-------

A :class:`~ninetoothed.interpret.Tracer` records, for every executed operation,
the program instance, the SSA location, the operand and result values, and the
mask applied to memory accesses:

.. code-block:: python

    from ninetoothed.interpret import Tracer

    result = interpret(
        arrangement,
        application,
        inputs=(x, out),
        symbols={"BLOCK_SIZE": 16},
        trace=Tracer(opcodes={"mem.store"}),
    )

    for event in result.trace_events(opcode="mem.store"):
        print(event.program_id, event.mask)

    print(result.render_trace(limit=20))

Traces are plain data: ``event.to_dict()`` is JSON-friendly, and a tracer accepts
``program_ids``, ``opcodes``, ``watch`` and ``limit`` filters plus a
``breakpoints`` mapping of opcode or SSA location to a callback, which makes
single-stepping possible from a notebook.

Differential Debugging
----------------------

The interpreter is most useful when it compares two runs instead of showing one.
:func:`~ninetoothed.interpret.compare_pipeline` interprets the same program with
and without a pass pipeline. A pipeline has to preserve semantics, so any output
difference is a compiler bug:

.. code-block:: python

    from ninetoothed.interpret import compare_pipeline

    diff = compare_pipeline(
        arrangement,
        application,
        inputs=(x, out),
        symbols={"BLOCK_SIZE": 16},
        pipeline=["ssa.canonicalize", "ssa.analyze_effects"],
        trace=True,
    )

    print(diff.render())

    if not diff.matches:
        print(diff.minimal_reproduction())

:func:`~ninetoothed.interpret.compare_interpretations` compares any two
interpretations, for example a reference run against a GPU run whose outputs were
saved with ``numpy.save``. Both return a
:class:`~ninetoothed.interpret.ProgramDiff` with per-output mismatch counts, the
first mismatching indices, the maximum error, the first trace divergence, and
``to_dict``/``to_json`` serialization for CI logs.

Floating point outputs are compared with ``rtol=1e-3`` and ``atol=1e-3`` by
default, which is the tolerance the project uses for ``float32`` against a NumPy
or PyTorch reference. Integer and boolean outputs are always compared exactly,
whatever tolerances are passed.

Reproducible Inputs
~~~~~~~~~~~~~~~~~~~

A failing comparison is only actionable when the inputs can be rebuilt, so the
seed is first-class: generate the inputs with
:func:`~ninetoothed.interpret.random_inputs`, hand the same ``seed`` to
``interpret``, and the reproduction carries it forward.

.. code-block:: python

    from ninetoothed.interpret import random_inputs

    x, out = random_inputs([((3, 11), "float32"), ((3, 11), "float32")], seed=17)

Locating the Pass That Broke a Program
--------------------------------------

Knowing *that* a pipeline is not semantics preserving is only half the answer.
:func:`~ninetoothed.interpret.compare_passes` applies the pipeline one pass at a
time, cumulatively, and interprets the program at every prefix, so it names the
first pass whose own rewrite changed the result:

.. code-block:: python

    from ninetoothed.interpret import compare_passes

    diff = compare_passes(
        arrangement,
        application,
        inputs=(x, out),
        symbols={"BLOCK_SIZE": 16},
    )

    print(diff.render())
    # pipeline diff: application pipeline
    # passes: ssa.canonicalize, ssa.analyze_effects, ssa.select_schedule, ...
    # result: MATCH
    # stages:
    #   [base ]  0 <frontend>
    #   [match]  1 ssa.canonicalize
    #   [match]  2 ssa.analyze_effects
    #   ...

Passing ``pipeline=None`` bisects the default pipeline of ``backend``; passing a
sequence of names bisects that sequence instead. A stage the interpreter cannot
execute is recorded with its error instead of aborting the scan, which matters
when a pass has not been taught to the interpreter yet.

When a stage does diverge, :meth:`~ninetoothed.interpret.PipelineDiff.localize`
pins the difference on a program instance and on the SSA operations that produce
it:

.. code-block:: python

    if not diff.matches:
        print(diff.localize().render())
        # first semantic difference introduced by `ssa.select_schedule`
        #   output      : out(1,)
        #   expected    : 2.0
        #   actual      : 3.0
        #   program id  : 0
        #   stores      :
        #     entry:2:mem.store

The instance is recovered from the same access map the interpreter executed with,
so the lookup stays exact for nested tiles and masked tails.

Minimal Reproductions
---------------------

:meth:`~ninetoothed.interpret.ProgramDiff.minimal_reproduction` and
:meth:`~ninetoothed.interpret.PipelineDiff.reproduction` render a self-contained
snippet; :func:`~ninetoothed.interpret.build_reproduction` returns the same thing
as structured data:

.. code-block:: python

    from ninetoothed.interpret import build_reproduction

    reproduction = build_reproduction(result, label="softmax")

    print(reproduction.seed)             # the recorded random seed
    print(reproduction.launch_shape)
    print(reproduction.to_json())        # SSA + data + shape + dtype + seed

A :class:`~ninetoothed.interpret.Reproduction` carries the five things a minimal
reproduction needs (the executed SSA, the input data, the shapes, the dtypes and
the random seed), plus the resolved symbols, the pass pipeline and, when the
functions are still importable, their source.

Inspecting Layouts
------------------

:func:`~ninetoothed.interpret.access_offsets` and
:func:`~ninetoothed.interpret.access_mask` expose the resolved access map per
program instance, shaped ``view_shape + tile_shape``. This is the CPU analogue of
the arrangement debugger and agrees exactly with
:func:`ninetoothed.eval._eval`:

.. code-block:: python

    from ninetoothed.interpret import access_mask, access_offsets

    offsets = access_offsets(result, "x")
    mask = access_mask(result, "x")

    # The same mapping the compiler reports, with the mask folded into -1.
    folded = np.where(mask, offsets, -1)

Known Limitations
-----------------

* ``bfloat16`` and ``float8`` dtypes are refused instead of approximated.
* ``mem.atomic_add`` and ``math.rand`` are not implemented.
* ``call.*`` target intrinsics have no CPU implementation.
* Reading a view before every outer dtype level has been indexed is rejected;
  index down with ``tensor.extract`` first.
* The frontend lowers an application by reading its source, so the arrangement and
  the application must be module-level functions; a ``lambda`` cannot be lowered.
* Execution is single-threaded and one program instance at a time, so large
  inputs are slow by construction.

API
---

.. autosummary::
   :toctree: generated
   :nosignatures:

   ninetoothed.interpret.interpret
   ninetoothed.interpret.interpret_program
   ninetoothed.interpret.Interpretation
   ninetoothed.interpret.apply_pipeline
   ninetoothed.interpret.access_map
   ninetoothed.interpret.access_offsets
   ninetoothed.interpret.access_mask
   ninetoothed.interpret.random_inputs
   ninetoothed.interpret.resolve_symbols

Tracing and Diffing
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ninetoothed.interpret.Tracer
   ninetoothed.interpret.TraceEvent
   ninetoothed.interpret.compare_interpretations
   ninetoothed.interpret.compare_pipeline
   ninetoothed.interpret.compare_passes
   ninetoothed.interpret.compare_traces
   ninetoothed.interpret.ProgramDiff
   ninetoothed.interpret.OutputDiff
   ninetoothed.interpret.TraceDiff
   ninetoothed.interpret.PipelineDiff
   ninetoothed.interpret.PassStage
   ninetoothed.interpret.Localization
   ninetoothed.interpret.localize_divergence

Reproductions
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ninetoothed.interpret.build_reproduction
   ninetoothed.interpret.render_reproduction
   ninetoothed.interpret.Reproduction
   ninetoothed.interpret.BufferSpec

Errors
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ninetoothed.interpret.InterpreterError
   ninetoothed.interpret.UnsupportedOperationError
   ninetoothed.interpret.UnsupportedDTypeError
   ninetoothed.interpret.UnsupportedAccessError
   ninetoothed.interpret.UnsupportedControlFlowError
   ninetoothed.interpret.MissingSymbolError
   ninetoothed.interpret.ProgramDomainError
