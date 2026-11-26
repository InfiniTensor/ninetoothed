"""Common subexpression elimination for pure SSA operations.

The backend emitter re-materializes a producer's full operand chain when a
value is consumed inside a later reduction loop.  When the same pure
expression is computed more than once -- once before the reduction and once
after it -- the late copy forces the reduction to be emitted a second time,
producing nested O(n^2) loops (for example ``log_softmax``).

This pass removes duplicate pure computations so that every value is
materialized once, at its first definition.  It is hardware-independent and
runs before effect analysis and scheduling, so all backends benefit.

Only operations with no memory effects, no regions, and deterministic results
are candidates.  Memory operations, calls, region-bearing operations, and
random values are deliberately excluded.
"""

from dataclasses import replace

from ninetoothed.ir import ssa

# Opcodes whose results are a pure, deterministic function of their operands.
_CSE_OPCODE_PREFIXES = ("arith.", "cmp.", "math.", "reduce.")
_CSE_OPCODES = {
    "dtype.constant",
    "dtype.of",
    "dtype.select",
    "index.offset",
    "shape.dim",
    "tensor.cast",
    "tensor.extract",
    "tensor.stride",
    "tensor.view",
    "select.where",
    "symbol.attr",
    "tuple.construct",
}

# ``math.rand`` is stateful; two calls with the same operands are not equal.
_EXCLUDED_OPCODES = {"math.rand"}


def _is_cse_candidate(op: ssa.Operation) -> bool:
    if op.regions or not op.results:
        return False

    code = op.opcode

    if code in _EXCLUDED_OPCODES:
        return False

    return code in _CSE_OPCODES or code.startswith(_CSE_OPCODE_PREFIXES)


def _rename_value(name, replacements):
    return replacements.get(name, name) if isinstance(name, str) else name


def _rename_attrs(attrs, replacements):
    """Rename SSA value names stored in attributes, leaving other values alone."""

    def rename(value):
        if isinstance(value, str):
            return replacements.get(value, value)

        if isinstance(value, tuple):
            return tuple(rename(item) for item in value)

        return value

    return {key: rename(value) for key, value in attrs.items()}


def _apply_replacements(op: ssa.Operation, replacements):
    if not replacements:
        return op

    operands = tuple(_rename_value(name, replacements) for name in op.operands)
    attrs = _rename_attrs(op.attrs, replacements)

    if operands == op.operands and attrs == dict(op.attrs):
        return op

    return replace(op, operands=operands, attrs=attrs)


def _cse_key(op: ssa.Operation):
    """Return a structural key for a duplicateable operation, or ``None``."""
    if not _is_cse_candidate(op):
        return None

    return (
        op.opcode,
        op.operands,
        tuple(sorted((key, repr(value)) for key, value in op.attrs.items())),
        tuple(
            (
                result.type.kind,
                result.type.shape,
                result.type.dtype,
                repr(result.type.attrs),
            )
            for result in op.results
        ),
    )


def eliminate_common_subexpressions(program: ssa.Program) -> ssa.Program:
    """Replace duplicate pure SSA operations with their first definition."""
    replacements: dict[str, str] = {}

    def rewrite_block(block: ssa.Block) -> ssa.Block:
        seen: dict[tuple, tuple[str, ...]] = {}
        operations = []

        for current in block.operations:
            current = _apply_replacements(current, replacements)
            key = _cse_key(current)

            if key is not None and key in seen:
                canonical = seen[key]

                for result, target in zip(current.results, canonical):
                    replacements[result.name] = target
                continue

            if key is not None:
                seen[key] = tuple(result.name for result in current.results)

            operations.append(
                replace(
                    current,
                    regions=tuple(rewrite_block(region) for region in current.regions),
                )
            )

        return replace(block, operations=tuple(operations))

    blocks = tuple(rewrite_block(block) for block in program.blocks)
    outputs = tuple(
        replace(value, name=replacements.get(value.name, value.name))
        for value in program.outputs
    )

    return replace(program, blocks=blocks, outputs=outputs)
