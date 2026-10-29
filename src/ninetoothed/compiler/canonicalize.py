"""Type normalization required by all source backends."""

from dataclasses import replace


def canonicalize_dtype_expressions(program):
    """Erase redundant type operands after the frontend proves a result dtype.

    Unknown dtype expressions remain explicit. Only unused, pure type values
    are removed; tensor arithmetic, memory effects and regions retain order.
    """
    references = ("dtype_ref", "dtype_source", "dtype_operand")
    used = {value.name for value in program.outputs}

    def rewrite(block):
        operations = []

        # SSA definitions dominate their uses, including captures in nested regions.
        for current in reversed(block.operations):
            if (
                current.opcode
                in {"tensor.cast", "tensor.zeros", "tensor.empty", "tensor.full"}
                and len(current.results) == 1
                and current.results[0].type.dtype is not None
            ):
                attrs = {
                    key: value
                    for key, value in current.attrs.items()
                    if key not in references
                }
                attrs["dtype"] = current.results[0].type.dtype
                operands = (
                    current.operands[:1]
                    if current.opcode == "tensor.cast"
                    else current.operands[:-1]
                    if current.operands
                    and current.operands[-1] == current.attrs.get("dtype_operand")
                    else current.operands
                )
                current = replace(current, attrs=attrs, operands=operands)

            if current.opcode in {
                "dtype.constant",
                "dtype.of",
                "dtype.select",
            } and not any(result.name in used for result in current.results):
                continue

            used.update(current.operands)
            used.update(
                current.attrs[key]
                for key in references
                if current.attrs.get(key) is not None
            )
            operations.append(
                replace(
                    current,
                    regions=tuple(rewrite(region) for region in current.regions),
                )
            )

        return replace(block, operations=tuple(reversed(operations)))

    return replace(
        program,
        blocks=tuple(rewrite(block) for block in reversed(program.blocks))[::-1],
    )
