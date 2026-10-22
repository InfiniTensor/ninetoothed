Native backend planning
=======================

CUDA C, BangC and AscendC share typed SSA, effect-aware CSE, reduction reuse,
layout proofs and numeric bounds. Target schedule passes select legal hardware
strategies; source renderers consume those decisions. Triton's existing tuning
and opt-in block-reduction interfaces retain their original behavior.

Pass and analysis boundaries
---------------------------

``ssa.analyze_effects`` owns a snapshot of shared facts. Reduction reuse,
numeric bounds, layout and uniform-store proofs remain independent analysis
functions. Structural or tensor-layout changes require fresh facts. Diagnostic
metadata records analysis and schedule decisions; editing it cannot manufacture
a proof. ``schedule_analysis`` records the input facts used for scheduling.

``ssa.plan_materialization`` retains the actual immutable stage programs and
scratch specifications. Emission carries that plan to materialization; runtime
execution does not repartition SSA. Ordered Cube partials preserve their separate
numeric contract while using the common stage executor. Scratch storage remains
fresh for each call and bound to that call's stream. The logical plan records
its input SSA and current binding. Unknown SSA changes discard an implicit plan;
an explicit stale request raises an error. Shape specialization and the existing
linalg decomposition update the binding without partitioning again.

Pass descriptors declare their reads, outputs and SSA effects. Custom pipelines
keep the supplied order; optional missing facts select conservative lowering.
There is no automatic dependency sorting or second pass manager.

Target scheduling
-----------------

Correctness lowering is independent of optional native optimization. The
BangC and AscendC ``lower_schedule`` passes assign proved row ownership to
serial tasks. Low-precision dot accumulation remains FP32 in
``decompose_linalg``, even when hardware optimization is omitted. Scalar math,
tail ownership, alias guards, SDK binding and library reload are baseline
requirements, not optional performance policies.

The three native ``optimize_schedule`` passes enable their target planners.
Omitting one from a custom pipeline keeps indexed scalar execution without
silently selecting vector, transfer or matrix fast paths. CSE also controls
coordinate hoisting and reduction reuse; the reuse proof remains an independent
analysis function. Bounded unroll and staged materialization are separate
optional passes. Missing or illegal explicit schedule candidates raise errors.

Use ``backend_options={"vector_chunk": 8192}`` to request a BangC or AscendC
packet size. The target proves the route, storage budget and instruction limits;
an illegal or unconsumed request raises an error. There is no separate vector
resource policy pass. AscendC queue slots follow the proved packet lifetime.

Explicit schedule candidates remain available through their target pass:

.. code-block:: python

   pass_options={"ssa.cuda.optimize_schedule": {"candidate": "wmma-32x32"}}

AscendC's ``vector-native-division`` candidate requires the previously validated
local CANN compiler and header fingerprints. Its packed validity checks, operand
lifetimes and sparse IEEE repair remain part of the implementation. CUDA ordered
accumulation, BangC row packing and complete indexed fallbacks retain their
existing numerical and memory-ownership contracts.

Option migration
----------------

BangC geometry belongs to ``arch``, ``task_chunk``, ``vector_chunk`` and
``row_batch_rows`` (``auto``, 8 or 16). AscendC accepts ``arch``, ``task_chunk``
and ``vector_chunk``. CUDA's existing options and numerical accumulation policy
remain available. Explicit geometry must satisfy target resources and be consumed.

The unmerged reduction-reuse, numeric-bounds, vector-resource-policy and AscendC
serial-queue pass wrappers have been removed. Their analyses and storage policy
are consumed directly by the shared analysis and target planning stages. Removed
native-path switches, explicit ``strict_division_strategy`` /
``strict_validity_summary`` aliases and experimental fast-math options are
rejected; legal DMA,
vector and matrix routes are selected automatically with complete fallbacks.

Development tools
-----------------

Native offline measurement and SSH deployment are maintained outside the stable
compiler package. Device servers compile with their local SDK. Compilation
failures propagate directly instead of retrying with a different remote compiler
under the same cache identity. One resolved local compiler identity supplies
the command, cache key and saved manifest. JIT binding uses the immutable cache
library; reloading an exported library snapshots its actual bytes before loading,
so rebuilding the same publication path cannot reuse an older loaded image. The experimental ``Compiler.tune_native`` entry
point and policy-only pass names have been removed before upstream publication.
