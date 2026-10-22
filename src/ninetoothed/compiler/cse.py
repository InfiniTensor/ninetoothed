"""Block-local expression reuse with explicit memory and region boundaries."""

from collections.abc import Mapping
from dataclasses import replace

from ninetoothed.compiler.effects import pure_expression
from ninetoothed.compiler.reduction_geometry import (
    prove_unit_parallel_reductions as plan_reduction_reuse,
)
from ninetoothed.ir import ssa


def eliminate_common_subexpressions(program):
    """Reuse identical typed expressions without reassociation or hoisting."""
    aliases = {}
    eliminated = []

    def rewrite(value):
        if isinstance(value, str):
            return aliases.get(value, value)

        if isinstance(value, Mapping):
            return {key: rewrite(item) for key, item in value.items()}

        if isinstance(value, (tuple, list)):
            return tuple(rewrite(item) for item in value)
        return value

    def block_rewrite(block):
        available = {}
        operations = []

        for operation in block.operations:
            current = replace(
                operation,
                operands=tuple(rewrite(name) for name in operation.operands),
                attrs=rewrite(operation.attrs),
                regions=tuple(block_rewrite(region) for region in operation.regions),
            )

            if not pure_expression(current) or len(current.results) != 1:
                available.clear()
                operations.append(current)
                continue

            result = current.results[0]
            key = (
                current.opcode,
                current.operands,
                _key(current.attrs),
                _key(result.type),
            )

            if key in available:
                aliases[result.name] = available[key]
                eliminated.append((result.name, available[key]))
                continue

            available[key] = result.name
            operations.append(current)

        return replace(block, operations=tuple(operations))

    blocks = tuple(block_rewrite(block) for block in program.blocks)

    return (
        replace(
            program,
            outputs=tuple(
                replace(value, name=rewrite(value.name)) for value in program.outputs
            ),
            blocks=blocks,
        ),
        tuple(eliminated),
    )


def _key(value):
    if isinstance(value, ssa.Type):
        return (value.kind, value.shape, value.dtype, _key(value.attrs))

    if isinstance(value, Mapping):
        return tuple(sorted((key, _key(item)) for key, item in value.items()))

    if isinstance(value, (tuple, list)):
        return tuple(_key(item) for item in value)

    if isinstance(value, float):
        # Python equality merges +0.0 and -0.0; division and signbit do not.
        return (float, value.hex())
    return (type(value), value)


__all__ = ["eliminate_common_subexpressions", "plan_reduction_reuse"]
