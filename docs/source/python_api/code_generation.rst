Code Generation
===============

.. autosummary::
   :toctree: generated
   :nosignatures:

   ninetoothed.build
   ninetoothed.jit
   ninetoothed.make

Frozen frontend modules
-----------------------

The compiler can capture an arrangement and application once, then compile the
result for different targets or specialize its dtypes without inspecting Python
again. Existing ``make``, ``lower``, ``jit`` and ``aot`` entry points remain
available; they use the same frontend boundary internally.

.. code-block:: python

   import ninetoothed as nt
   from ninetoothed.compiler import Compiler, CompileRequest

   def arrangement(x, y, output):
       return tuple(t.tile((32,)) for t in (x, y, output))

   def application(x, y, output):
       output = x + y

   compiler = Compiler()
   module = compiler.frontend(CompileRequest(
       arrangement=arrangement,
       application=application,
       tensors=tuple(nt.Tensor(1) for _ in range(3)),
   ))
   compiled = compiler.compile(CompileRequest(
       frontend=module,
       backend="triton",
       tensor_dtypes={"x": "float32", "y": "float32", "output": "float32"},
   ))
   kernel = compiler.materialize(compiled)

``Compiler.frontend`` produces a verified ``FrontendModule`` containing tensor
specifications, structured layouts, SSA computation, symbols and specialization
domains. It contains no live Tensor objects, Python callbacks or backend source.
Its optional source text is diagnostic only. Supplying ``frontend`` together with
an arrangement, application or Tensor arguments is an error.

Compilation verifies this contract again before specialization and backend
passes. ``tensor_dtypes`` binds public parameter names; ``specialization_values``
binds declared index, extent, stride or meta symbols. Runtime dtype specialization
also operates on this frozen representation, rather than rerunning arrangement
code. Treat arrangements as pure functions: Python side effects are not a
runtime-specialization mechanism.

Frontend verification additionally checks source-view provenance. Subsequent SSA
passes still verify operation signatures, shapes, dtypes, control flow and effects;
new local values need not retain frontend source-view attributes.

Shared or fixed physical shape and stride symbols are retained in the launch
ABI and checked before execution, including after artifact reload. C++ export
preserves direct symbol equality checks and constant or fully specialized
composite checks. Composite physical guards that still require runtime symbol
evaluation must be specialized before C++ export. A source artifact alone is
not a checked executable.

This boundary does not introduce named axes, change the meaning of ``tile`` or
other layout primitives, or add optimization passes. A future frontend must
produce the same verified module contract; backend code generation does not
depend on its surface syntax.
