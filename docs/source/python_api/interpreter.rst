Interpreter
===========

The interpreter is a CPU reference implementation of the NineToothed execution
model. It executes the same ``ssa.Program`` a backend receives, one program
instance at a time, using NumPy as its execution core. It never compiles or
launches a GPU kernel, so it runs on any machine that can import NineToothed,
and it makes the value of every intermediate SSA operation observable.

Design
------

The interpreter reuses the production front end instead of implementing a second
source-to-NumPy translator:

.. code-block:: text

    arrangement ------> layouts and index mappings -----.
                                                       |
    application --> Python front end --> ssa.Program --> SSA pass pipeline
                                                       |
                                                       v
                                            NumPy CPU interpreter
                                                       |
                                            result / trace / differential test

``ninetoothed.interpret`` mirrors ``ninetoothed.make``. It takes the same
arrangement, application, and symbolic tensors, lowers them through the same
front end and the same pass pipeline, and then hands the resulting
``ssa.Program`` to the interpreter.

Three properties follow from that design:

- The interpreted program is the program under test, not a parallel
  reimplementation of it, so a front-end or pass-pipeline change is exercised by
  the interpreter automatically.
- The offsets, strides, and masks come from ``ninetoothed.eval``, the same
  arrangement evaluator ``simulate_arrangement`` uses.
- Nothing in the execution path imports or calls a CUDA runtime, so the
  interpreter works with ``CUDA_VISIBLE_DEVICES=""``.

Execution model
---------------

``Executor`` walks the entry block once per program instance, in ascending
program id order, and each instance gets a fresh ``RuntimeState`` holding the SSA
value environment, the CPU buffers, the resolved layout symbols, the program id,
and the active mask. Tensor arguments are bound by gathering the elements their
arrangement selects.

Memory is modelled as one flat NumPy buffer per tensor argument. Every access is
a gather or a scatter driven by the arrangement offset table and the arrangement
mask, so a lane whose offset is out of bounds but masked out never reads or
writes storage. A masked load fills its lanes with the ``other`` value of the
symbolic tensor, which is how a tiled reduction ignores its tail lanes.

Quick start
-----------

.. code-block:: python

    import numpy as np

    import ninetoothed
    from ninetoothed import Tensor, block_size


    def arrangement(lhs, rhs, output, BLOCK_SIZE=block_size()):
        return (
            lhs.tile((BLOCK_SIZE,)),
            rhs.tile((BLOCK_SIZE,)),
            output.tile((BLOCK_SIZE,)),
        )


    def application(lhs, rhs, output):
        output = lhs + rhs


    kernel = ninetoothed.interpret(
        arrangement,
        application,
        (Tensor(1), Tensor(1), Tensor(1)),
        kernel_name="add",
    )

    lhs = np.arange(1000, dtype=np.float32)
    rhs = np.ones(1000, dtype=np.float32)
    output = np.zeros(1000, dtype=np.float32)

    kernel(lhs, rhs, output)

Arguments may be NumPy arrays or PyTorch CPU tensors. A NumPy array has to be
C-contiguous, because the interpreter models a tensor argument as one flat
buffer. The kernel returns the primary output, that is the buffer the program
stores into.

.. code-block:: python

    import torch

    lhs = torch.arange(1000, dtype=torch.float32)
    rhs = torch.ones(1000, dtype=torch.float32)
    output = torch.zeros(1000, dtype=torch.float32)

    kernel(lhs, rhs, output)

Symbols
-------

``meta`` binds the meta and constexpr symbols of the arrangement, exactly the
way a backend launch binds them:

.. code-block:: python

    kernel = ninetoothed.interpret(
        arrangement,
        application,
        (Tensor(1), Tensor(1), Tensor(1)),
        meta={"BLOCK_SIZE": 256},
    )

A meta symbol declared with ``block_size()`` has a default and may be omitted. A
constexpr symbol declared with ``Symbol(..., constexpr=True)`` has to be
provided, either through ``meta`` or through the keyword arguments of the call,
and ``InvalidArgumentError`` reports the symbols that are missing.

Execution trace
---------------

``trace=True`` records every executed SSA operation:

.. code-block:: python

    kernel = ninetoothed.interpret(
        arrangement, application, tensors, trace=True, meta={"BLOCK_SIZE": 256}
    )
    kernel(lhs, rhs, output)

    print(kernel.last_trace.format())

Each event reports the program id, the SSA location, the opcode, the operands,
the input and output values, and the mask that was active for a memory access:

.. code-block:: text

    [program 2] entry:1:mem.store mem.store (%1, output)
        in : %1=array(shape=(4,), dtype=float32, [16.0, 18.0, 0.0, 0.0]); output=array(shape=(4,), dtype=float32, [0.0, 0.0, 0.0, 0.0])
        mask: array(shape=(4,), dtype=bool, [True, True, False, False])

``TraceOptions`` restricts what is recorded, and ``ExecutionTrace.filter``
restricts what an existing trace shows, by program id and by opcode:

.. code-block:: python

    from ninetoothed.interpreter import TraceOptions

    kernel = ninetoothed.interpret(
        arrangement,
        application,
        tensors,
        trace=True,
        trace_options=TraceOptions(program_ids=(0,), opcodes=("mem.",), keep_values=True),
    )

    trace = kernel.last_trace
    stores = trace.filter(opcodes=("mem.",))
    first_program = trace.filter(program_ids=(0,))

An opcode pattern that ends with a dot matches the opcodes with that prefix, so
``"mem."`` selects every memory access and ``"mem.store"`` selects the stores.
``keep_values=True`` copies the traced values into the event, which is useful
when the buffers are written after the operation ran.

``TraceOptions.max_elements`` bounds how many elements of a traced value are
spelled out; the remainder is elided. It applies to the mask of a memory
access as well, so raising it is what makes a wide mask fully visible.

Interactive debugging
---------------------

``Interpreter.debug`` opens a session over the same kernel. A session stops
*before* an operation runs and reports the program instance, the SSA location,
the operands with the values they hold at that point, the mask that is active,
and the values on the watch list:

.. code-block:: python

    from ninetoothed.interpreter import Breakpoint

    session = kernel.debug(
        lhs,
        rhs,
        output,
        breakpoints=(Breakpoint(program_ids=(1,), opcodes=("mem.",)),),
        watch=("output",),
    )

    while (stop := session.resume()) is not None:
        print(stop.format())

.. code-block:: text

    [program 1] stop 5: entry:2:mem.store (%1, output)
        in   : %1=array(shape=(4,), dtype=float32, [8.0, 10.0, 12.0, 14.0]); output=array(shape=(4,), dtype=float32, [0.0, 0.0, 0.0, 0.0])
        mask : array(shape=(4,), dtype=bool, [True, True, True, True])
        watch: output=array(shape=(4,), dtype=float32, [0.0, 0.0, 0.0, 0.0])

The session reports the value a name holds at the stop, which is *before* the
operation ran, so stepping through a kernel shows how a value is built up. A
name that is not bound yet is reported as ``<unbound>``.

.. list-table::
   :header-rows: 1

   * - Method
     - Meaning
   * - ``step()``
     - Run to the next operation, in program id order.
   * - ``resume()``
     - Run to the next breakpoint.
   * - ``finish()``
     - Run to the end and return the primary output.
   * - ``watch(*names)``
     - Report more SSA names at every stop.
   * - ``value(name)``
     - The value of one SSA name at the last stop, as a copy.
   * - ``reset()``
     - Forget the reported stops, so the next one starts from the first
       operation.
   * - ``format_history()``
     - Render every stop the session reported.

A ``Breakpoint`` matches a program id, an opcode, an SSA location, or a
combination of them, and the opcode and location patterns are matched exactly or
by a prefix that ends with a dot, the same way a trace filter is matched. A
breakpoint that only names a program id stops at every operation of that
instance.

The interpreter is vectorized and runs a whole program instance in one call, so
a session replays the run for every stop instead of keeping a suspended stack.
The tensor arguments are restored to the state they had when the session was
created before every replay, which keeps an application that accumulates into
its output consistent with a plain call. The cost is re-running the operations
that precede the stop.

Exporting a reproduction
------------------------

A failing differential comparison is worth more than the numbers it printed, so
``export_reproduction`` writes everything needed to re-run the failing case:

.. code-block:: python

    from ninetoothed.interpreter import export_reproduction

    if not np.allclose(expected, actual, rtol=rtol, atol=atol):
        export_reproduction(
            "repro",
            name="masked",
            arrangement=masked_arrangement,
            application=masked_double,
            tensors=(Tensor(1), Tensor(1)),
            arguments=(input, output),
            meta={"MASKED_BLOCK_SIZE": 256},
            expected=expected,
            actual=actual,
            tolerance=(rtol, atol),
            seed=20260916,
            reason="the backend result differs from the interpreter result",
        )

.. list-table::
   :header-rows: 1

   * - File
     - Content
   * - ``inputs.npz``
     - One ``argument_<index>`` array per call argument.
   * - ``expected.npy``, ``actual.npy``
     - The two results, when both were given.
   * - ``ssa_frontend.txt``, ``ssa_optimized.txt``
     - The SSA before the pipeline and after it.
   * - ``passes.txt``
     - The passes the pipeline ran.
   * - ``case.json``
     - The case name, the seed, the meta values, the tolerance, the shape, the
       dtype, and the element count of every argument, the largest absolute
       difference, the number of mismatching elements, and the names the script
       could not export.
   * - ``reproduce.py``
     - A self-contained script that re-runs the case on the CPU.

``reproduce.py`` embeds the source of the arrangement and of the application,
because the front end lowers source, and it re-creates the meta and constexpr
symbols that source uses. Identifiers the exporter cannot turn into a literal
are listed in ``case.json`` under ``unresolved_names`` and marked in the script,
so a reproduction never silently drops part of a case.

.. code-block:: text

    $ python repro/reproduce.py
    seed: 20260916
    tolerance: rtol=0.001, atol=0.001
    interpreter matches the recorded result: True
    max |difference|: 0.0

Passing ``--backend`` also runs the production backend on the recorded inputs,
which needs a CUDA device. ``export_reproduction`` returns a ``Reproduction``
holding the directory, the files it wrote, the reason, and the unresolved names.

Supported operations
--------------------

``supported_operations()`` returns the support matrix, and
``unsupported_operations()`` returns the operations the interpreter rejects
together with the reason.

.. list-table::
   :header-rows: 1

   * - Opcode
     - Interpreted as
   * - ``arith.constant``
     - The recorded constant.
   * - ``arith.*``
     - ``numpy`` arithmetic, including the unary operators.
   * - ``cmp.*``
     - ``numpy`` comparisons, always producing ``bool``.
   * - ``math.*``
     - The ``numpy`` counterpart of the math intrinsic.
   * - ``call.*``
     - The ``numpy`` counterpart of the intrinsic call.
   * - ``reduce.*``
     - ``numpy`` reductions, with an optional compile-time ``axis``.
   * - ``select.where``
     - ``numpy.where``.
   * - ``tensor.zeros``, ``tensor.empty``, ``tensor.full``
     - A NumPy allocation. ``tensor.empty`` is zero-filled, because the
       interpreter has no undefined values.
   * - ``tensor.extract``
     - Indexing a value with runtime indices.
   * - ``tensor.cast``
     - An explicit dtype cast.
   * - ``tensor.view``
     - A reshape of a value.
   * - ``shape.dim``, ``tensor.stride``, ``symbol.attr``
     - The symbolic layout expressions of the arrangement.
   * - ``tuple.construct``
     - A Python tuple.
   * - ``mem.store``
     - A masked scatter through the arrangement window.
   * - ``scf.for``, ``scf.if``, ``scf.yield``
     - Structured control flow, including loop-carried values.
   * - ``linalg.dot``, ``linalg.matmul``
     - A matrix product, accumulated in ``float32`` for narrow float inputs.
   * - ``linalg.transpose``
     - A swap of the last two dimensions.

``float32``, ``int32``, and ``bool`` are the dtypes the interpreter is validated
against. ``float16`` is exercised by the matrix product. ``bfloat16`` and the
``float8`` family are rejected with ``UnsupportedDTypeError``, because NumPy has
no native equivalent and the interpreter does not emulate one.

Known limitations
-----------------

- Only the SSA operations listed above are interpreted. Atomics, data pointers,
  random number generation, offset vectors, and transposed matrix
  multiplication are rejected with ``UnsupportedOperationError``, which reports
  the opcode and the SSA location.
- The rejected ``mem.load`` and ``mem.data_ptr`` opcodes are not a gap in
  masked loading. A masked tile read reaches the interpreter as
  ``tensor.extract`` on a tensor argument, and a masked write reaches it as
  ``mem.store``; both go through the arrangement mask, and lanes the mask
  rejects never index the storage. The ``mem.load`` form is the one that reads
  through a raw pointer obtained from ``mem.data_ptr``, which is the indirect
  access this interpreter does not model. The masked read path is
  ``TensorBuffer.gather`` in ``ninetoothed/interpreter/memory.py``, which
  replaces rejected offsets with ``0`` before indexing and fills those lanes
  with the ``other`` value.
- A ``mem.store`` has to write a value that already has the element count of its
  arrangement window. The interpreter reshapes such a value, but it never
  replicates it, because replicating over the window is not what the emitted
  code does. A mismatch is reported with ``UnsupportedAccessError``.
- The interpreter simulates no GPU concurrency. Warps, block scheduling, shared
  memory, and races are outside its scope, and it is not a performance target.
- Reductions and matrix products accumulate with NumPy semantics, so the
  floating-point rounding order may differ from a backend for ``float16``
  operands. Compare ``float32`` results with ``rtol=1e-3`` and ``atol=1e-3``,
  and use a looser tolerance for ``float16``.
- A program whose arrangement produces a value shape the interpreter cannot
  reconcile with the SSA contract raises ``InvalidProgramError`` rather than
  producing a result.

Differential testing
--------------------

The interpreter is meant to be compared against independent references:

- Against NumPy or PyTorch, by running the same inputs through both and
  comparing the outputs.
- Against the SSA pipeline, by interpreting the program twice, once with
  ``run_passes=False`` and once with the default pipeline. ``Interpreter.program``
  is the optimized program, ``Interpreter.raw_program`` is the program the front
  end produced, and ``Interpreter.pass_trace`` lists the passes that ran.
- Against the production backend, by passing the same arrangement and
  application to ``ninetoothed.make`` and comparing the ``cuda`` result with the
  interpreted one.

``export_reproduction`` turns a comparison that failed into a directory that
re-runs the case, and ``Breakpoint`` stops a session on the operation that is
worth looking at, so a difference can be narrowed down without a device.

``compare_passes`` automates the second comparison. It interprets the low-level
SSA program and then every prefix of the backend pipeline, compares each result
with the front end's, and reports the first prefix that changed the semantics:

.. code-block:: python

    from ninetoothed.interpreter import compare_passes

    comparison = compare_passes(
        arrangement, application, tensors, arguments, meta={"BLOCK_SIZE": 256}
    )

    print(comparison.format())
    # front end: matches (max |difference| = 0.000e+00)
    # ssa.canonicalize: matches (max |difference| = 0.000e+00)
    # ssa.analyze_effects: matches (max |difference| = 0.000e+00)
    # ssa.select_schedule: matches (max |difference| = 0.000e+00)
    # ...
    # Every pass preserved the front-end semantics.

``arguments`` is a callable that returns a fresh argument tuple for every run, so
each prefix starts from clean buffers. ``PassComparison.first_mismatch`` returns
the step that diverged, and ``PassStep.interpretation`` carries the lowered
program of that step for further inspection.

API reference
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ninetoothed.interpret
   ninetoothed.interpreter.Interpreter
   ninetoothed.interpreter.Interpretation
   ninetoothed.interpreter.lower
   ninetoothed.interpreter.ExecutionTrace
   ninetoothed.interpreter.TraceEvent
   ninetoothed.interpreter.TraceOptions
   ninetoothed.interpreter.RuntimeState
   ninetoothed.interpreter.Executor
   ninetoothed.interpreter.TensorAccess
   ninetoothed.interpreter.derive_access
   ninetoothed.interpreter.Breakpoint
   ninetoothed.interpreter.Stop
   ninetoothed.interpreter.DebugSession
   ninetoothed.interpreter.export_reproduction
   ninetoothed.interpreter.Reproduction
   ninetoothed.interpreter.compare_passes
   ninetoothed.interpreter.PassComparison
   ninetoothed.interpreter.PassStep
   ninetoothed.interpreter.supported_operations
   ninetoothed.interpreter.unsupported_operations
   ninetoothed.interpreter.register
   ninetoothed.interpreter.InterpretationError
   ninetoothed.interpreter.UnsupportedOperationError
   ninetoothed.interpreter.UnsupportedDTypeError
   ninetoothed.interpreter.UnsupportedAccessError
   ninetoothed.interpreter.InvalidProgramError
   ninetoothed.interpreter.InvalidArgumentError
