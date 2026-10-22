"""Conservative IEEE float32 lower bounds for SSA expressions.

Bounds describe ordered results only: NaN and positive infinity remain
possible. They never imply finite inputs or permit reassociation, approximate
arithmetic, or removal of exception handling for an unproved domain.
"""

import math
import struct


def _float32_constant(operation):
    value = operation.attrs.get("value")

    if type(value) not in (int, float):
        return None

    try:
        rounded = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        return None

    return rounded if math.isfinite(rounded) else None


def analyze_numeric_bounds(program):
    """Prove nonnegative float32 expressions without altering operations.

    Region arguments and results have no inferred bounds. Locally proved
    values can use dominating definitions, but yields do not transfer facts
    across branches or loop iterations. Existing metadata is never trusted.
    """
    proved = {}
    types = {value.name: value.type for value in program.inputs}

    def analyze_block(block, incoming):
        available = dict(incoming)

        for argument in block.args:
            available.pop(argument.name, None)
            types[argument.name] = argument.type

        for operation in block.operations:
            for region in operation.regions:
                analyze_block(region, available)

            types.update((result.name, result.type) for result in operation.results)

            if operation.regions or len(operation.results) != 1:
                continue

            result = operation.results[0]

            if result.type.kind not in {"scalar", "tensor"} or (
                result.type.dtype != "float32"
            ):
                continue

            lower = None
            operands = operation.operands
            float_operands = all(
                name in types and types[name].dtype == "float32" for name in operands
            )

            if operation.opcode == "arith.constant":
                value = _float32_constant(operation)

                if value is not None and value >= 0:
                    lower = value
            elif (
                operation.opcode in {"math.exp", "math.sqrt"}
                and len(operands) == 1
                and float_operands
            ):
                lower = 0.0
            elif operation.opcode == "arith.mul" and len(operands) == 2:
                if operands[0] == operands[1] and float_operands:
                    lower = 0.0
            elif operation.opcode == "arith.add" and len(operands) == 2:
                left = available.get(operands[0])
                right = available.get(operands[1])

                if left is not None and right is not None:
                    # Taking the maximum avoids a rounded sum or overflow.
                    # Addition of two nonnegative float32 operands cannot
                    # produce an ordered result below either operand.
                    lower = max(left, right)

            if lower is not None:
                # Targets can flush subnormal operands or results to zero.
                # Keep a portable lower bound without changing the literal.
                if 0 < lower < float.fromhex("0x1p-126"):
                    lower = 0.0

                available[result.name] = lower
                proved[result.name] = {
                    "lower": lower,
                    "may_nan": True,
                    "may_inf": True,
                }

    for block in program.blocks:
        analyze_block(block, {})

    return proved
