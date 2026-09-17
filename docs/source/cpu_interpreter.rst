CPU reference interpreter
=========================

``ninetoothed.interpret`` executes an application on NumPy arrays using the
same arrangement-to-layout and application-to-SSA frontend as the compiler.
It does not compile GPU source, initialize a GPU runtime, or run the original
application as ordinary Python. Its purpose is semantic validation and debugging,
not measuring GPU performance.

Quick start
-----------

Save the following in a Python file: the existing frontend inspects function
source, so a function entered only in an interactive ``python -c`` string may
not have retrievable source.

.. code-block:: python

   import numpy as np
   from ninetoothed import Tensor, interpret

   def arrangement(x, out):
       return x.tile((4,)), out.tile((4,))

   def application(x, out):
       out = x * 2 + 1

   kernel = interpret(
       arrangement, application,
       (Tensor(1, name="x", dtype="float32"),
        Tensor(1, name="out", dtype="float32")),
       trace=True,
   )
   x = np.arange(11, dtype=np.float32)
   out = np.empty_like(x)
   result = kernel(x, out)
   np.testing.assert_array_equal(out, x * 2 + 1)
   print(result.outputs["out"])

Output arrays are modified in place. Inputs are NumPy arrays, CPU Python/NumPy
scalars, or already-loaded PyTorch CPU tensors. A strided CPU Torch tensor is
adapted with ``tensor.numpy()`` without copying, so writes reach the original
storage and ``result.outputs["out"] is out`` remains true for a Torch output.
Non-contiguous CPU tensor views retain their strides. CUDA/non-CPU tensors,
``requires_grad=True``, sparse layouts, conjugate/negative view bits and dtypes
that cannot be represented without copying are rejected explicitly. Detaching
or moving tensors is the caller's decision; the interpreter never does it
implicitly. The interpreter itself never imports PyTorch or Triton for NumPy
execution.
Array dtypes must match declared SSA types. The acceptance-tested arithmetic
types are float32, int32, and bool; index values use int64 internally.
Object, complex, string, structured, and unsupported dtypes are rejected.
Signed integer ``floordiv`` rounds toward negative infinity, and ``mod`` has
the divisor's sign, following the SSA/Python contract. The Triton emitter
corrects its native signed division/remainder when needed. Division by zero
and the signed overflow case ``INT_MIN / -1`` are outside the validated inputs.

For a fresh environment without Torch or Triton, follow
:ref:`cpu-wheel-install`. It installs a real wheel from this checkout and runs
the demo outside the repository, without ``PYTHONPATH`` or an editable install.
The wheel retains the existing GPU dependency metadata; the installation guide
explains the explicit CPU override and its ``pip check`` limitation.

Frontend versus backend SSA
---------------------------

By default, ``kernel.program`` is the original frontend SSA. To execute the SSA
pass pipeline used by a backend, select it explicitly:

.. code-block:: python

   kernel = interpret(arrangement, application, descriptors, backend="triton")
   # These are separate, inspectable programs.
   original = kernel.frontend_program
   transformed = kernel.program

``backend`` calls the existing ``lower_for_target`` pass pipeline, without
source emission or GPU materialization. It is therefore useful on a CPU-only
machine. ``pipeline`` and ``pass_options`` select the same pipeline options as
the compiler. There is no fallback to the original program if an operation in
the transformed program is unsupported.

To evaluate an already available program, use:

.. code-block:: python

   from ninetoothed.interpreter import interpret_program

   result = interpret_program(
       transformed, {"x": x, "out": out}, tensors=kernel.tensors,
       symbols=kernel.meta, trace=True,
   )
   # Or retain an existing handle's layouts and binding order:
   result = kernel.with_program(transformed)(x, out)

The lower-level entry point accepts the actual ``ssa.Program`` object, not a
string containing generated Triton/CUDA code. ``TensorSpec.layout`` provides
the same structured coordinate maps and predicates used by emitters.

Memory and control flow
-----------------------

Arranged blocks are mapped to source coordinates with the existing ``IndexExpr``
and ``AccessMap`` records. They support broadcast program domains, strided source
arrays, nested layout levels and non-divisible tails. Masked-out coordinates
are never used to index the source allocation. ``Tensor(other=...)`` controls
the masked load value; use zero for sums/dots and negative infinity for a
softmax/max input where that is the intended kernel semantics.

The explicit pointer subset has these operand contracts:

* ``mem.data_ptr(tensor)`` returns a checked element pointer.
* ``arith.add(pointer, offsets)`` creates element-offset addresses.
* ``mem.load(pointer, mask?, other?)`` loads only active addresses.
* ``mem.store(value, destination, mask?)`` follows the frontend's value-first
  order. ``attrs["other"]`` supplies a load fill value when no third operand is
  given.

Raw pointer operations currently require C-contiguous storage. Arranged tensor
accesses use source coordinates and support NumPy strides, including negative
strides. An active out-of-bounds lane raises ``InterpretationError`` containing
the operation location and program ID; inactive lanes may have invalid addresses.

``scf.if`` requires a scalar condition and executes only the selected region.
``scf.for`` obeys Python range bounds and steps, including negative steps and
zero iterations. Region arguments and ``scf.yield`` carry values explicitly.

The default arranged launch uses a flat program ID, as current emitters do.
The lower-level API also accepts explicit one-to-three-dimensional grids for
program-ID operations. Trace IDs always contain three coordinates; unused
coordinates are zero. An explicit arranged grid must cover the same number of
programs as its layout domain.

Scalar dot across output tiles
------------------------------

The default Triton and CUDA pipelines can lower a two-dimensional dot into
scalar SSA arithmetic and a K loop. The CPU interpreter executes those actual
operations for every valid output lane, including multiple M/N output tiles.
It does not replace the lowered program with a call to NumPy matmul.

All scalar dot/matmul and transpose decompositions share a storage preflight,
including untiled and single-program execution. The output must be a named
array with independent, element-aligned storage. Potential aliasing with another
input, partial-element byte strides and overlapping output elements (including
overlapping zero-stride views) are rejected before writes. Reading output values
inside the scalar program is also rejected, even if the same ``out`` binding is
used as a matmul operand; shape, stride and offset metadata reads remain allowed.
Normal independent untiled arrays and aligned positive/negative strides remain
supported. This is a contract for scalar lane replay, not a restriction on every
undecomposed vector operation.

The multi-program contract is deliberately bounded:

* The two operands and output are rank-2 input bindings with arranged
  descriptors and one rank-2 value level. Their source dimensions describe
  ``(M, K) @ (K, N) -> (M, N)``.
* Each output program contains the complete K reduction. Masked K padding is
  zero, and operand coordinates must match the output rows, output columns,
  and the complete K coordinate range. Splitting K into independent programs
  without an accumulating loop is rejected.
* Every valid output element is written once. The output domain must cover the
  complete matrix, and masks must match the logical in-bounds lanes.
* The shared scalar storage preflight applies to every output program.
  Independent strided inputs and outputs are supported, including positive and
  negative output strides. The overlap check is conservative and may reject
  disjoint views of a shared allocation.
* There is one top-level decomposed output store. Other stores or atomic
  effects, including those inside nested regions, are rejected before any
  output write. Replaying such effects once per scalar lane would change the
  original program's semantics.

These contract failures identify an SSA operation location. Computed operands,
additional layout levels, split-K accumulation and arbitrary matrix views are
outside this multi-program implementation. Scalar transpose decomposition
continues to require a single logical program.

For example, save and execute the following as a Python file on CPU:

.. code-block:: python

   import numpy as np
   import ninetoothed.language as ntl
   from ninetoothed import Tensor, interpret

   def matrix_tiles(a, b, out):
       return a.tile((4, 4)), b.tile((4, 4)), out.tile((4, 4))

   def matrix_dot(a, b, out):
       out = ntl.dot(a, b)

   tensors = (
       Tensor(2, name="a", dtype="float32", other=0),
       Tensor(2, name="b", dtype="float32", other=0),
       Tensor(2, name="out", dtype="float32"),
   )
   kernel = interpret(matrix_tiles, matrix_dot, tensors, backend="triton")
   a = np.arange(21, dtype=np.float32).reshape(7, 3) / 7
   b = np.arange(18, dtype=np.float32).reshape(3, 6) / 5
   out = np.full((7, 6), -731, dtype=np.float32)
   kernel(a, b, out)
   np.testing.assert_allclose(out, a @ b, rtol=1e-3, atol=1e-3)

This example has four M/N output programs and a padded K tile of size four.
``index.offset`` operations produced by matmul decomposition explicitly use
``coordinate_space="value"``: row and column indices are local to the output
tile. The CPU executor and source emitters then use the shared layout maps to
resolve storage coordinates. Offsets without this attribute keep their
existing source-coordinate meaning.

CPU regression cases cover float32 and int32, non-square dimensions, M/N/K
tails, strided inputs, independent strided outputs, trace consistency and
rejection-before-write checks. The validated 4090 GPU dot case is scalar float32;
it does not cover Tensor Core instructions, arbitrary layouts or split-K.
New source-emission checks and CPU results do not constitute a GPU run.

Tracing, extension and pass debugging
-------------------------------------

``TraceEvent`` records ``program_id``, ``location``, ``opcode``, execution-before
``inputs`` snapshots, ``results`` snapshots, ``mask``, loop ``iteration``, and
``watched`` values. A snapshot stores shape, dtype and values. Memory destination
and pointer snapshots store reference metadata instead of dereferencing arbitrary
addresses. Lazy tensor references in results and watch lists likewise retain
metadata; observing them never introduces an unmasked memory read. Numeric
source snapshots are copied, so later mutations do not alter the recorded trace.

The lower-level API accepts ``program_ids``, ``opcodes`` and ``watch`` filters.
``callback(event)`` receives each selected event synchronously. ``StepDebugger``
offers terminal and deterministic scripted-command interfaces without a
background thread. Pauses occur after an operation completes. Writes performed
before a pause or quit remain in the output buffer.

.. code-block:: python

   from ninetoothed.interpreter.debugger import StepDebugger

   debugger = StepDebugger(
       commands=["watch %0", "print %0", "step", "break mem.store", "continue"],
       output=print,
   )
   kernel = interpret(arrangement, application, descriptors, callback=debugger)
   result = kernel(x, out)
   print([event.location for event in debugger.pauses])

Omit ``commands`` to read commands interactively. Use ``step``/``s`` to stop at
the next completed operation, ``continue``/``c`` to run to the next breakpoint,
``break LOCATION``/``b LOCATION`` to add an exact location or prefix,
``break OPCODE`` to break on an operation type, ``delete``/``d`` to remove it,
``watch NAME``/``w NAME`` to track an SSA value, and ``print NAME``/``p NAME`` to
show its latest observed snapshot. ``quit``/``q`` raises ``DebuggerQuit``.
An empty command steps; an exhausted scripted command stream continues.

``debugger.inspect(name)`` returns the latest observed snapshot. The ``values``
cache resets for each program ID and scalar output lane; it does not claim that a previously observed
region-local name remains in scope. Dynamic watch names are requested from the
execution engine at each event. Avoid trace filters during full debugging if
all intermediate definitions should remain observable.

After installation, run the example from the repository root. It accepts
``--debug`` for scripted stepping and breakpoints, or ``--interactive-debug``
to read commands from the terminal. ``-I`` uses the installed package:

.. code-block:: console

   python -I docs/cpu_interpreter_demo.py --debug

The demo checks the correct ``x * 2 + 1`` result against NumPy, then deliberately
changes the SSA constant ``2`` to ``3``. ``check_passes`` identifies
``injected_bad_constant`` and the first different ``arith.constant`` operation.
This is fault injection for teaching the debugger, not a historical compiler bug.
Eleven inputs with tiles of four also demonstrate three program IDs and a
masked final lane.

Use a new directory to save both programs and replay the same difference in a
separate process, without rerunning the frontend or the injected pass:

.. code-block:: console

   python -I docs/cpu_interpreter_demo.py --debug --export /tmp/nine-demo
   python -I /tmp/nine-demo/replay.py

``check_passes`` automatically creates ``reference/``, ``previous/`` and
``candidate/`` bundles, ``failure.json`` and a generic top-level ``replay.py``.
The demo checks the correct result against NumPy before injecting the fault;
the replay independently compares the saved SSA, without a hard-coded affine
formula. All bundles preserve the original inputs, layouts, dtype and seed.
Exit code zero means the recorded output/operation difference or execution
exception was reproduced. A repaired candidate or a different failure makes
the replay fail, including under ``python -O``. The saved boundary does not
claim to rerun or independently rediscover the Python pass.

An extension can register a handler for an individual operation without changing
the frontend or matching an entire application:

.. code-block:: python

   result = interpret_program(
       program, inputs,
       handlers={"math.my_op": lambda operation, operands: operands[0] + 2},
   )

Unregistered unsupported operations fail with ``UnsupportedOperationError`` and
an operation location. Handlers return one NumPy result, or a tuple for multiple
SSA results, and remain responsible for their operation's semantics.

.. code-block:: python

   from ninetoothed.backends.core import Target
   from ninetoothed.compiler.passes import Context, default_pipeline
   from ninetoothed.interpreter.debugger import check_passes

   context = Context(
       backend=Target.TRITON, compiler_options={}, kernel_metadata={},
       tensors=kernel.tensors,
   )
   pipeline = default_pipeline(Target.TRITON)
   checks = tuple(
       (pass_.name, lambda program, pass_=pass_: pass_.run(program, context))
       for pass_ in pipeline.passes
   )
   report = check_passes(
       kernel.frontend_program, checks, {"x": x, "out": out},
       tensors=kernel.tensors, symbols=kernel.meta,
       failure_dir="/tmp/nine-pass-failure", seed=2026,
   )
   print(report.first_bad_pass, report.localization)
   print(report.reproducer, report.export_error)

``check_passes`` first verifies that the original reference executes, then
compares every intermediate program both to the original and to the preceding
pass output using independent input copies. It stops at the first failing
boundary. The original comparison detects cumulative drift; the adjacent
comparison also catches differences hidden by cancellation against the original.
Integer and
boolean outputs require exact equality. Floating outputs default to
``rtol=1e-3, atol=1e-3``. ``ProgramComparison.first_operation`` identifies the first
different corresponding operation only when SSA structures and the complete
trace event sequence align, including program ID, operation location, opcode,
loop iteration and scalar lane. Input, result and effective memory-mask snapshots
are compared; ``OperationDifference.component`` tells which snapshot differed.

``PassCheck.difference`` retains the original-reference result.
``adjacent_difference`` adds the preceding-pass result, allowing an earlier
correct restructuring without losing later operation localization.
``localization.reference`` explicitly names ``original`` or ``previous``.
``localization.basis`` distinguishes four observation scopes:

* ``full_trace``: corresponding operations across the entire aligned trace.
* ``aligned_prefix``: a difference before the first unmatched execution event,
  for example a changed condition before the branches diverge. Static SSA
  structure must still align; no later events are paired by position.
* ``retained_boundary``: the earliest differing input/mask/result at an operation
  explicitly preserved by one pass. Input/output fingerprints, operation
  equality, one-to-one preserve relations, and the complete filtered event
  sequence must match. This can locate a changed value at an unchanged consumer
  after a split or merge. ``reference_location`` records the pre-pass location;
  ``operation.location`` records the candidate location.
* ``mapped_result``: the earliest differing result with an explicit value
  correspondence declared by the pass. The mapping can compare identical
  result types, a reference tile with a scalar lane, or the supported matmul
  input, product and accumulation checkpoints described below.

These identify observed differences, not necessarily the unique faulty generated
instruction. When no valid observation correspondence exists, ``localization``
is ``None`` and declared source candidates remain available. A split/merge
origin alone never establishes equivalent intermediate values. Unmapped
restructuring and changed event counts therefore do not acquire guessed exact
locations.

Explicit SSA provenance
-----------------------

``ssa.Operation.origins`` is an immutable tuple of original-SSA operation IDs;
its default is empty. It is nonsemantic: it does not affect operation equality
or ``ssa.render``. JSON serialization preserves it separately from operation
attributes. Origins refer to static locations in the seeded SSA, not Python
source lines or a proven unique cause of a numerical error.

The APIs are available from ``ninetoothed.ir.provenance``:

* ``seed_origins(program)`` assigns stable IDs and a source catalog on first
  use. Reapplying it preserves the existing history, including unknown origins.
* ``operation_locations(program)`` returns static ``(location, operation)``
  pairs, including nested regions, using the interpreter's location convention.
* ``ProvenancePass(before, name)`` tracks one explicit transformation.
  ``derive(targets, sources, relation="replace", recursive=False)`` declares
  replacement, split or merge relations. ``recursive=True`` includes generated
  region operations. ``delete(*sources)`` declares removal without successors,
  and ``finish(after)`` records the completed relation set.
* ``record_pass(before, after, name)`` accepts a matching explicit pass record
  or records retained operation objects. It does not infer replacements from
  copied IDs, names, opcodes or structural similarity. Unknown replacements
  and removals remain unknown; a relation with partly unknown inputs does not
  present the known subset as a complete source range.
* ``source_candidates(program, locations=None)`` returns ``SourceCandidate``
  records with ``origin_id``, ``location``, ``opcode`` and ``result_names``.
  Explicit locations select output operations; omitting them selects declared
  changes in the latest pass. These are declared source candidates, not a
  complete causal slice or a unique fault location.

``Pipeline.run`` and ``check_passes`` seed and record pass provenance. The
actual linalg decomposition declares the original dot/transpose and store as
sources of its generated scalar instructions. Temporary allocation reserves
names across the entire program, including nested regions. Unrelated or
unmapped transformations do not acquire an invented source relation.

``ProgramComparison`` exposes ``equal``, ``output_differences``,
``first_operation``, ``traces_aligned`` and ``source_candidates``.
``PassCheck.source_candidates`` also retains a declared transformation scope
when a pass produces an interpretation error. A nonempty candidate tuple must
not be reported as an exact Python line, value alignment or proof of causality.
``first_bad_pass`` identifies the first observed mismatch or unsupported/error
stage in the supplied sequence, which is distinct from proving the underlying
cause of a compiler defect.

To inspect executable opcodes, walk the program blocks and nested regions.
Provenance catalogs intentionally retain names of removed operations, so a
substring search over the entire serialized SSA metadata is not a valid check
that an opcode has been eliminated.

Declared result correspondence
------------------------------

Operation origins alone do not establish result equality. A pass can separately
declare an equality that the debugger should check:

.. code-block:: python

   tracker = ProvenancePass(before, "split_scale")
   generated = tracker.derive(new_operations, (old_operation,), relation="split")
   tracker.map_result(old_operation, generated[-1])
   after = tracker.finish(rewritten_program)

``map_result`` accepts ``source_result`` and ``target_result`` when a producer
has several results. The default ``projection="identity"`` requires matching
SSA types. ``projection="lane"`` compares a reference numeric tile with the
candidate's executed scalar lane. The actual linalg decomposition declares
this mapping for the completed matmul K-loop result and transpose extract;
partial accumulators are not compared with the final matrix result.
The same pass additionally declares ``matmul_lhs``, ``matmul_rhs``,
``matmul_term`` and ``matmul_prefix`` checkpoints inside its scalar K loop.
Their references come from the original matmul's captured input tiles:
``A[row, k]``, ``B[k, column]``, their product, and the prefix accumulated from
zero with the declared scalar casts. They never use a candidate intermediate
as the reference. Every active lane must cover the original K domain exactly
once in order; changed loop coverage yields a diagnostic and retains the
completed-result comparison. These formulas do not assert equivalence for
arbitrary reassociation, split-K or fused floating-point arithmetic.
Mixed-dtype lowering remains available: where the original result and generated
accumulator have different dtypes, this optional equality is not declared.
Omitting an unrepresentable debug contract does not change the generated SSA.

Mappings require recorded producer relations, matching SSA fingerprints and
compatible enclosing control-flow headers. Every occurrence must align by
program ID, iteration and scalar lane before that mapping reports a mismatch.
One reference result may feed several explicitly mapped targets, but a target
result cannot have conflicting mappings. A mapping is the pass author's
semantic contract, checked on the supplied inputs, not a proof for all inputs.
Unknown, stale and unaligned mappings do not acquire guessed correspondences.

``ProgramComparison.mapped_operation`` reports a failing mapped result.
``localization`` selects the earliest candidate execution event among valid
full-trace, prefix, retained-consumer and mapped-result observations.
Its ``basis="mapped_result"`` names a producer where a declared equality
fails; ``projection`` identifies the equality or input-derived checkpoint.
It does not prove that this producer uniquely caused the error; for a
split operation the mapping can name the group's completed result, while an
unmapped internal instruction remains unresolved. Output equality remains
the pass success condition, so dead or cancelled internal changes do not by
themselves turn a successful output comparison into a failed pass.

``dependency_slice`` on both ``ProgramComparison`` and ``PassCheck`` follows
executed SSA producers, loop-carried values, selected yields and control
operands. Each event retains its trace index, program ID, iteration and lane.
The slice is a value-dependency explanation, not a minimal reproducer or a set
of independently faulty operations. Its ``boundaries`` explicitly identifies
unresolved values, incomplete traces, unknown handler effects and memory
ordering that cannot be established. ``memory_dependencies`` links each read
to the last observed same-program write of the intersecting byte ranges.
``mapping_issues`` explains unavailable correspondences.

``TraceEvent.memory`` records actual checked reads and writes with stable
storage names and relative half-open byte intervals. It never contains process
addresses. Snapshot/watch evaluation is excluded from this recording; scalar
tensor extraction reads only its selected logical coordinates. Masks exclude
inactive addresses before coordinate conversion, including large unsigned
offsets. Positive, negative and zero strides are represented by actual bytes.
``TraceEvent.sequence`` detects missing or filtered events. Cross-program
dependencies and overlapping write lanes remain explicit boundaries: serial
CPU execution is not a GPU happens-before or race-freedom proof. No memory
recorder is allocated when tracing and callbacks are both disabled.

Mapping analysis indexes each trace once by producer location. Dependency
analysis stops at the selected observation because later events cannot supply
its inputs. Numeric contents, ordering, alignment checks and tolerances remain
part of the diagnosis; an index never substitutes for these checks.

Automatic failure bundles include these observations and dependencies. Replay
checks them, including the pass-level selected location; legacy bundles without
the additional fields replay with their original diagnostic semantics. New
failure schema-2 bundles require ``diagnostics_version=2`` and their recorded
memory/projection fields; removing those fields is an error. See the CPU tests in
``tests/test_interpreter_value_mapping.py``,
``tests/test_interpreter_memory_dependencies.py``,
``tests/test_interpreter_matmul_checkpoints.py`` and the dated acceptance reports for
verified scope. These checks do not extend the existing GPU race/alias model.

Replay bundles
--------------

``compare_programs(..., failure_dir=PATH, seed=SEED)`` and
``check_passes(..., failure_dir=PATH, seed=SEED)`` automatically export at the
first failure. The directory must not exist; success writes nothing. With
``failure_dir=None`` (the default), neither API writes files. The seed is supplied
by the caller and stored as metadata; replay uses the saved numeric inputs and
does not attempt to infer or regenerate their random seed.

``failure.json`` contains schema, failure type, named pass history, both
comparison reports, localization scope, tolerances and environment versions.
Each program bundle contains readable SSA, structured SSA, numeric input data,
shape, dtype, strides, aliases, grid and symbols. The optional ``previous/``
bundle is the last program accepted before the failing pass. A generic replay
uses an installed version with this API:

.. code-block:: console

   python /tmp/nine-pass-failure/replay.py

The replay requires the same differing outputs, operation observations, trace
alignment or execution exception. It is a reproducer for saved SSA, not an
export of arbitrary executable Python pass code. A transform that raises or
produces invalid SSA instead exports an explicitly non-replayable diagnostic
snapshot. No pickled callables are written or loaded.

``reproducer`` is the completed directory; ``export_error`` separately reports
capture failures such as an existing directory or full disk. A partial export
retains an ``INCOMPLETE`` marker and the replay refuses it. The original mismatch
or exception is preserved; a capture error never converts failure into success.
For direct comparison execution exceptions, these two fields are attached to
the exception when capture is requested. Check ``export_error`` in automation.

``export_reproducer(directory, program, inputs, tensors=..., symbols=..., seed=...)``
exports structured JSON, readable SSA, numeric NPZ inputs, shape/dtype/seed metadata,
array strides and write permissions, and a replay script. Differential copies
and loaded bundles preserve positive, negative and zero strides in independent
storage. ``load_reproducer`` loads JSON and NumPy data with
``allow_pickle=False``. Existing bundle files are never overwritten.
The supplied case is preserved; automatic shape/operation minimization is not
implemented. Multiple input names bound to the same array object retain that
alias relationship in differential copies and loaded bundles. Distinct
overlapping numeric views also preserve their shared storage, individual
strides, dtypes and write permissions. Such cases use program-bundle schema 2:
numeric byte buffers plus checked view offsets. Only bytes represented by
the supplied views are copied; allocation gaps are zero-filled. Loading checks
every accessible byte range before creating a view and rejects object dtypes.
This preserves CPU view semantics without relaxing the scalar decomposition's
independent-output restriction or asserting GPU race freedom.

Replay JSON preserves operation origins and the provenance catalog/history.
Legacy schema-1 bundles without origins or provenance metadata remain readable;
missing origins default to empty rather than inventing a transformation history.
Ordinary independent-input bundles continue to use program-bundle schema 1.

Current boundaries
------------------

The basic subset includes arithmetic, comparison, cast, selection, broadcasting,
masked memory, sum/max/min reductions, structured control flow and program IDs.
Direct ``linalg.dot``/``linalg.matmul`` and the exp/max/sum composition used for
softmax are also implemented. Scalar matmul/transpose decompositions execute the
actual SSA operations and region loops once per valid rank-2 output lane. This
supports one output store in a single logical program and the bounded
multi-program matmul contract described above. Scalar transpose remains limited
to one program. The ``TraceEvent.lane`` field
identifies the output coordinate; normal vector execution has ``lane=None``.
Inactive output lanes are never executed or written. This is a diagnostic CPU
path, not a fast matrix multiplication implementation.

Split-K scalar decomposition, computed matmul operands, extra value levels,
overlapping storage and combinations with other output stores or atomic effects
are not supported by the multi-program matmul path. Direct/undecomposed linalg
operations retain their normal vector semantics. Atomics, indirect pointers,
random-number operations, float8, jagged layouts and arbitrary external calls
are not part of the validated interpreter subset. GPU warp/block scheduling,
shared memory and race behavior are not simulated.

The decomposition pass now derives a fixed or compound reduction extent through
``shape.dim(lhs, -1)`` from the arranged operand, instead of referring to an
undefined fallback ``k`` symbol. The symbolic-K path is preserved. Invalid SSA
from any pipeline still reports the original verification cause, with no
fallback to the frontend program.

CPU tests cannot establish agreement with A100 hardware or measure GPU speed.
That requires the separate real-GPU differential run on matching kernels,
inputs, layouts, dtypes, seeds and pass settings.

Layout expressions represent positional call operands. Keyword arguments,
including ``**`` expansion, are rejected during parsing rather than discarded.
The interpreter's integer-normalized ``next_power_of_2`` call and named padded
shape bindings return zero for nonpositive extents and round positive extents
upward. This agrees with the pinned Triton 3.1.0 integer utility for signed
64-bit inputs; it does not specify physical GPU launch-block sizing.


Latest validation and CPU automation
------------------------------------

The current submission integrates upstream target architecture revision
``22e74c3afe47e12ee29d7d0bcfaf1de8286f4560``. Its selected CPU suite passes
793 tests with 15 actual GPU cases deselected in a NumPy-only environment.
This includes platform profiles, target capabilities, memory/alias semantics,
layout calls, expression semantics, and 23 extraction-geometry regressions.
The default pipeline also checks ``ssa.validate_target_capabilities``.

Untraced scalar matmul execution can reuse the checked address map of an
innermost extraction. Each internal tensor reference holds at most one map
of at most 65536 logical elements. Array contents are read on every extraction;
shape, stride, dtype, typed symbols, extraction coordinates and actual compiled
expression plans participate in the key. Shape proofs use the original symbol
context, before address-coordinate overrides. Unproven integer operations,
custom numeric objects and warning-sensitive paths retain ordinary evaluation.
Tracing, callbacks, watches, custom handlers and event filters disable reuse.

Three fresh-process CPU comparisons of three selected small matmuls give
3.847--3.860 times the previous execution speed, with equal complete trace
fingerprints and all control cases within the predefined five-percent limit.
The warm tracemalloc peak increases by 141--183 KB; this is an explicit memory
tradeoff, not a GPU speedup or a reduction in process RSS. The initial candidate
passed timing but failed two shape-context exception regressions; the corrected
version has separate passing regressions and newly measured timings. Earlier
constant-folding experiments remain rejected under their original protocols.

On the final geometry-cache computation source, a manual RTX 4090 D run passes
all 15 existing actual GPU differential cases and three additional target-shape
matmul checks. The latter compare emitted Triton GPU output, raw/target SSA and
NumPy, preserve output guards and inputs, and confirm address-map reuse. Another
131 geometry/layout/expression regressions pass with NumPy 1.26.4, Torch 2.6.0a0
and Triton 3.1.0; these interpreter tests are not 131 GPU kernels. The run verifies
201 frozen input files before and after execution. See the
`fixed geometry evidence <https://github.com/a962695448-rgb/ninetoothed/tree/ef787a082e98f44a740dd3f06b3fcc894810ca1b/docs/validation/extraction-geometry-20260918>`_
for complete sources, limitations, initial failures and reproducible records.
This establishes correctness for the listed cases, not a full repository GPU
suite or GPU performance. The earlier 0706213 GPU evidence retains its own scope.

The recorded A100 full suite of 835 passed and two multi-GPU skips belongs to
older computation source ``ed332733db28dbf16de06f166b16766760148958``. It was
not rerun after integrating the new compiler/platform changes and must not be
reported as hardware validation of the current combination. The current
acceptance and compatibility reports distinguish these evidence scopes.

The ``CPU interpreter`` workflow runs on ordinary GitHub-hosted Ubuntu runners
for Python 3.10 and 3.12, including fork pull requests. Its checkout regression
entry point rejects environments containing Torch or Triton, disables GPU
visibility and third-party pytest plugin loading, and selects the documented
interpreter/SSA test scope. It then builds the normal wheel and exercises the
installed demo and standalone replay outside the checkout. JUnit results and
dependency versions are retained as workflow artifacts.

In a fresh environment without Torch or Triton, run from the repository root:

.. code-block:: console

   python -m pip install -r requirements-cpu.txt
   python scripts/run_cpu_tests.py --junitxml /tmp/nine-cpu-results.xml

These are CPU checks, not a replacement for the separate hardware differential
suite. The wheel retains its existing Triton dependency metadata; the isolated
installation uses the explicit override documented in :ref:`cpu-wheel-install`.

Historical validation at 6ecce58
--------------------------------

The following records belong to an earlier implementation and are retained for
traceability. They do not describe the current acceptance baseline above.

That historical implementation was frozen in
``6ecce58da28bb9709aa35fc6c25c1f361aff736f``. Its selected NumPy CPU suite passed
307 tests, with 15 GPU cases deselected, in 32.68 seconds (exit 0). That
environment contains neither Torch nor Triton, so this run does not cover the
Torch adapter, real GPU execution or the entire repository suite.

Earlier selected runs remain separate: ``f35fb51`` reported 1 failed, 294 passed
and 15 deselected in 35.42 seconds because a metadata-string assertion matched
an opcode retained in provenance; ``56f091e`` then passed 295 tests with 15
deselected in 34.18 seconds. New single-program storage regressions first
reported 8 failed and 4 passed; the corrected related combination passed 137.
These development and frozen-run counts are not added together.

A separate RTX 4090 run at the same computation commit passed 15 Triton
GPU cases (nine programs, ten categories) and one scalar float32 CUDA dot
probe. The CUDA case compares NumPy, frontend CPU, lowered SSA CPU and
actual GPU output, with unchanged inputs and guards. It is one backend
correctness probe, not a performance benchmark or a new independent program
beyond the Triton matrix. At that historical checkpoint, newer A100, full-repository and multi-device
runs were outside its evidence. Earlier A100 results did not validate those
runtime, pass, emitter or provenance changes; the later matching A100 evidence
is summarized in the current validation section above.

Two earlier Sphinx attempts failed on missing matplotlib and then tkinter.
The closeout defers Tk imports to the interactive visualization entry point;
static documentation images use a headless backend. Build status and original
logs are recorded separately from CPU and GPU validation in the acceptance
notes and the interpreter_optimization_20260906 evidence directory.
