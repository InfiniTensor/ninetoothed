"""Type normalization required by all source backends."""

from dataclasses import replace


def canonicalize_dtype_expressions(program):
    """Erase redundant type operands after the frontend proves a result dtype.

    Unknown dtype expressions remain explicit. Only unused, pure type values
    are removed; tensor arithmetic, memory effects and regions retain order.
    """
    references = ("dtype_ref", "dtype_source", "dtype_operand")

    def walk(operations):
        for operation in operations:
            yield operation

            for region in operation.regions:
                yield from walk(region.operations)

    def rewrite(block):
        nonlocal removed
        operations = []

        for operation in block.operations:
            current = replace(
                operation,
                regions=tuple(rewrite(region) for region in operation.regions),
            )

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

            operations.append(current)

        used = {name for operation in walk(operations) for name in operation.operands}
        used.update(value.name for value in program.outputs)
        used.update(
            operation.attrs[key]
            for operation in walk(operations)
            for key in references
            if operation.attrs.get(key) is not None
        )
        retained = tuple(
            operation
            for operation in operations
            if operation.opcode not in {"dtype.constant", "dtype.of", "dtype.select"}
            or any(result.name in used for result in operation.results)
        )
        removed |= len(retained) != len(operations)

        return replace(block, operations=retained)

    while True:
        removed = False
        program = replace(
            program, blocks=tuple(rewrite(block) for block in program.blocks)
        )

        if not removed:
            return program
