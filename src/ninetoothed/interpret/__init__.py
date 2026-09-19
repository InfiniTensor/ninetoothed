"""CPU reference interpreter and differential debugger for NineToothed.

The interpreter reuses the existing lowering chain — the arrangement produces the
layout and index mapping, the Python frontend produces an ``ssa.Program`` — and
executes that program with NumPy.  It never calls a GPU backend, so applications
can be developed, traced and diff-tested on machines without a CUDA device.

Typical use::

    import numpy as np
    from ninetoothed import Tensor
    from ninetoothed.interpret import interpret


    def arrangement(x, out, BLOCK_SIZE=128):
        return x.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))


    def application(x, out):
        out = x * 2.0


    x = np.arange(1000, dtype=np.float32)
    out = np.zeros_like(x)
    result = interpret(
        arrangement,
        application,
        tensors=(Tensor(1), Tensor(1)),
        inputs=(x, out),
        symbols={"BLOCK_SIZE": 256},
    )

    assert np.allclose(result.output("out"), x * 2.0)
"""

from . import operations  # noqa: F401  (registers the built-in operations)
from .api import (
    DEFAULT_CONSTEXPR_VALUE,
    Interpretation,
    access_map,
    access_mask,
    access_offsets,
    apply_pipeline,
    arrangement_symbol_aliases,
    interpret,
    interpret_program,
    random_inputs,
    resolve_program_symbols,
    resolve_symbols,
)
from .diff import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    BufferSpec,
    Localization,
    OutputDiff,
    PassStage,
    PipelineDiff,
    ProgramDiff,
    Reproduction,
    TraceDiff,
    TraceDivergence,
    build_reproduction,
    compare_interpretations,
    compare_passes,
    compare_pipeline,
    compare_traces,
    localize_divergence,
    render_reproduction,
)
from .errors import (
    InterpreterError,
    MissingSymbolError,
    ProgramDomainError,
    TraceStop,
    UnsupportedAccessError,
    UnsupportedControlFlowError,
    UnsupportedDTypeError,
    UnsupportedOperationError,
)
from .expr import Expression, ExpressionError
from .memory import AccessTemplate, CPUMemory, TensorRuntime, View
from .registry import (
    UNSUPPORTED_OPERATIONS,
    OperationSpec,
    format_support_matrix,
    register,
    support_matrix,
    supported_opcodes,
)
from .trace import TraceEvent, Tracer, ValueSnapshot

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_CONSTEXPR_VALUE",
    "DEFAULT_RTOL",
    "UNSUPPORTED_OPERATIONS",
    "AccessTemplate",
    "BufferSpec",
    "CPUMemory",
    "Expression",
    "ExpressionError",
    "Interpretation",
    "InterpreterError",
    "Localization",
    "MissingSymbolError",
    "OperationSpec",
    "OutputDiff",
    "PassStage",
    "PipelineDiff",
    "ProgramDiff",
    "ProgramDomainError",
    "Reproduction",
    "TensorRuntime",
    "TraceDiff",
    "TraceDivergence",
    "TraceEvent",
    "TraceStop",
    "Tracer",
    "UnsupportedAccessError",
    "UnsupportedControlFlowError",
    "UnsupportedDTypeError",
    "UnsupportedOperationError",
    "ValueSnapshot",
    "View",
    "access_map",
    "access_mask",
    "access_offsets",
    "apply_pipeline",
    "arrangement_symbol_aliases",
    "build_reproduction",
    "compare_interpretations",
    "compare_passes",
    "compare_pipeline",
    "compare_traces",
    "format_support_matrix",
    "interpret",
    "interpret_program",
    "localize_divergence",
    "random_inputs",
    "register",
    "render_reproduction",
    "resolve_program_symbols",
    "resolve_symbols",
    "support_matrix",
    "supported_opcodes",
]
