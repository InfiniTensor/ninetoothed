"""Prove common expressions and row reductions obey effects and geometry."""

from ninetoothed.compiler.cse import eliminate_common_subexpressions
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import TensorSpec, ssa


def _program(source):
    return from_source(
        source,
        tuple(
            TensorSpec(ndim=2, shape=(1, 136), dtype="float32", name=name)
            for name in ("x", "out")
        ),
        kind="reuse",
    )


def _operations(program):
    return program.blocks[0].operations


def test_cse_rewrites_repeated_values_and_is_idempotent():
    program = _program("def app(x, out):\n    out = (x + 1.0) * (x + 1.0)\n")
    result, aliases = eliminate_common_subexpressions(program)
    ssa.verify_program(result)
    assert aliases
    additions = tuple(op for op in _operations(result) if op.opcode == "arith.add")
    assert len(additions) == 1
    multiply = next(op for op in _operations(result) if op.opcode == "arith.mul")
    assert multiply.operands == (additions[0].results[0].name,) * 2
    again, aliases = eliminate_common_subexpressions(result)
    assert again == result
    assert not aliases
