"""Type normalization required by all source backends."""

from dataclasses import replace


def canonicalize_dtype_expressions(program, *, provenance=None):
    """Erase redundant type operands after the frontend proves a result dtype.

    Unknown dtype expressions remain explicit. Only unused, pure type values
    are removed; tensor arithmetic, memory effects and regions retain order.
    """
    references = ("dtype_ref", "dtype_source", "dtype_operand")
    used = {value.name for value in program.outputs}

    def rewrite(block):
        operations = []

        # SSA definitions dominate their uses, including captures in nested regions.
        for original in reversed(block.operations):
            current = original

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

                if attrs != current.attrs or operands != current.operands:
                    current = replace(current, attrs=attrs, operands=operands)

            if current.opcode in {
                "dtype.constant",
                "dtype.of",
                "dtype.select",
            } and not any(result.name in used for result in current.results):
                if provenance is not None:
                    provenance.delete(original)

                continue

            used.update(current.operands)
            used.update(
                current.attrs[key]
                for key in references
                if current.attrs.get(key) is not None
            )
            regions = tuple(rewrite(region) for region in current.regions)

            if any(new is not old for new, old in zip(regions, current.regions)):
                current = replace(current, regions=regions)

            if current is not original and provenance is not None:
                (current,) = provenance.derive((current,), (original,))

            operations.append(current)

        operations = tuple(reversed(operations))

        if len(operations) == len(block.operations) and all(
            new is old for new, old in zip(operations, block.operations)
        ):
            return block
        return replace(block, operations=operations)

    return replace(
        program,
        blocks=tuple(rewrite(block) for block in reversed(program.blocks))[::-1],
    )
