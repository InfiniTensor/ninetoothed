"""CPU reference interpreter for NineToothed SSA programs.

The interpreter executes exactly the SSA program a backend would receive, one
program instance at a time, using NumPy as its execution core. It reuses the
existing arrangement evaluation for offsets and masks, so masked lanes never
touch CPU memory.

``Interpreter.debug`` opens an interactive session that stops before an
operation, and ``export_reproduction`` writes everything needed to re-run a
differential case that failed.

.. code-block:: python

    import numpy as np
    import ninetoothed
    from ninetoothed import Tensor


    def arrangement(lhs, rhs, output, BLOCK_SIZE=ninetoothed.block_size()):
        return (
            lhs.tile((BLOCK_SIZE,)),
            rhs.tile((BLOCK_SIZE,)),
            output.tile((BLOCK_SIZE,)),
        )


    def application(lhs, rhs, output):
        output = lhs + rhs


    kernel = ninetoothed.interpret(
        arrangement, application, (Tensor(1), Tensor(1), Tensor(1))
    )

    lhs = np.arange(8, dtype=np.float32)
    rhs = np.ones(8, dtype=np.float32)
    output = np.empty_like(lhs)

    kernel(lhs, rhs, output)
"""

from ninetoothed.interpreter.access import TensorAccess, derive_access
from ninetoothed.interpreter.debugger import Breakpoint, DebugSession, Stop
from ninetoothed.interpreter.diagnostics import (
    PassComparison,
    PassStep,
    compare_passes,
)
from ninetoothed.interpreter.errors import (
    InterpretationError,
    InvalidArgumentError,
    InvalidProgramError,
    UnsupportedAccessError,
    UnsupportedDTypeError,
    UnsupportedOperationError,
)
from ninetoothed.interpreter.executor import Executor
from ninetoothed.interpreter.kernel import Interpreter, interpret
from ninetoothed.interpreter.lowering import Interpretation, lower
from ninetoothed.interpreter.ops import (
    OperationSupport,
    register,
    supported_operations,
    unsupported_operations,
)
from ninetoothed.interpreter.reproduction import Reproduction, export_reproduction
from ninetoothed.interpreter.runtime import RuntimeState
from ninetoothed.interpreter.trace import (
    ExecutionTrace,
    TraceEvent,
    TraceOptions,
    summarize_trace,
)

__all__ = [
    "Breakpoint",
    "DebugSession",
    "ExecutionTrace",
    "Executor",
    "PassComparison",
    "PassStep",
    "Reproduction",
    "Interpretation",
    "InterpretationError",
    "Interpreter",
    "InvalidArgumentError",
    "InvalidProgramError",
    "OperationSupport",
    "Stop",
    "RuntimeState",
    "TensorAccess",
    "TraceEvent",
    "TraceOptions",
    "UnsupportedAccessError",
    "UnsupportedDTypeError",
    "UnsupportedOperationError",
    "compare_passes",
    "derive_access",
    "export_reproduction",
    "interpret",
    "lower",
    "register",
    "summarize_trace",
    "supported_operations",
    "unsupported_operations",
]
