"""Structured control-flow operations: ``scf.for``, ``scf.if``, ``scf.yield``."""

import numpy as np

from ..errors import UnsupportedControlFlowError
from ..registry import register
from ..values import materialize
from .common import as_index, bind, build_value, operand


@register(
    "scf.for",
    category="scf",
    summary="Sequential loop with block-argument carried values.",
)
def _handle_for(state, operation):
    lower = as_index(operand(state, operation, 0))
    upper = as_index(operand(state, operation, 1))
    step = as_index(operand(state, operation, 2))

    if step == 0:
        raise UnsupportedControlFlowError(
            "The `scf.for` step must not be zero.",
            opcode=operation.opcode,
            location=state.location,
        )

    if not operation.regions:
        raise UnsupportedControlFlowError(
            "The `scf.for` opcode requires exactly one region.",
            opcode=operation.opcode,
            location=state.location,
        )

    region = operation.regions[0]
    induction = region.args[0]
    carried_args = region.args[1:]
    carried = [state.value(name) for name in operation.operands[3:]]

    for value in range(lower, upper, step):
        with state.scope():
            state.values[induction.name] = build_value(
                induction.type, np.asarray(value)
            )

            for argument, current in zip(carried_args, carried):
                state.values[argument.name] = current

            yields = state.execute_region(region, operation)
            carried = [state.value(name) for name in yields]

    if operation.results:
        bind(state, operation, *carried)


@register(
    "scf.if",
    category="scf",
    summary="Conditional execution with an optional else region.",
)
def _handle_if(state, operation):
    condition = materialize(operand(state, operation, 0), state.context)
    condition = np.asarray(condition)

    if condition.size != 1:
        raise UnsupportedControlFlowError(
            "The CPU interpreter requires a scalar `scf.if` condition; got a "
            f"condition of shape {condition.shape}.",
            opcode=operation.opcode,
            location=state.location,
        )

    taken = 0 if bool(condition.reshape(())) else 1

    if operation.results:
        if taken >= len(operation.regions):
            raise UnsupportedControlFlowError(
                "Result-producing `scf.if` requires both then and else regions.",
                opcode=operation.opcode,
                location=state.location,
            )

        with state.scope():
            yields = state.execute_region(operation.regions[taken], operation)
            bind(state, operation, *(state.value(name) for name in yields))

        return

    if taken < len(operation.regions):
        with state.scope():
            state.execute_region(operation.regions[taken], operation)


@register(
    "scf.yield",
    category="scf",
    summary="Region terminator; consumed by the enclosing region executor.",
)
def _handle_yield(state, operation):
    del state, operation


__all__ = []
