"""Prove complete row producer and store geometry without choosing reuse."""

from ninetoothed.compiler.effects import pure_expression


def prove_unit_parallel_reductions(program):
    """Identify a pure, same-row reduction DAG before one terminal store.

    The plan is intentionally independent of dtype and backend syntax. It
    does not move values across a memory effect or structured control flow.
    All reductions share the last input axis, so preceding reduced values
    are invariant while a later reduction walks that axis.
    """
    if len(program.blocks) != 1:
        return ()

    operations = program.blocks[0].operations

    if (
        not operations
        or operations[-1].opcode != "mem.store"
        or len(operations[-1].operands) != 2
        or set(operations[-1].attrs) - {"target", "target_shape"}
        or any(not pure_expression(operation) for operation in operations[:-1])
    ):
        return ()

    definitions = {
        result.name: operation
        for operation in operations
        for result in operation.results
    }
    types = {value.name: value.type for value in program.inputs}
    types.update(
        (result.name, result.type)
        for operation in operations
        for result in operation.results
    )
    reachable = set()
    pending = [operations[-1].operands[0]]

    while pending:
        name = pending.pop()

        if name in reachable:
            continue

        reachable.add(name)
        operation = definitions.get(name)

        if operation is not None:
            pending.extend(operation.operands)

    for value in program.inputs:
        if value.name not in reachable or value.type.kind != "tensor":
            continue

        attrs = value.type.attrs

        if attrs.get("other") is not None:
            source = attrs.get("source_shape", ())
            tile = attrs.get("application_shape", ())

            if len(source) != 2 or len(tile) != 2 or str(source[-1]) != str(tile[-1]):
                # No padding values may disappear when promoting a scalar
                # fallback to a row schedule. Exact-width source rows are safe.
                return ()

    reductions = tuple(
        operation
        for operation in operations
        if operation.opcode.startswith("reduce.")
        and operation.results
        and operation.results[0].name in reachable
    )

    if not reductions:
        return ()

    shapes = set()

    for operation in reductions:
        if len(operation.operands) != 1 or len(operation.results) != 1:
            return ()

        operand_type = types.get(operation.operands[0])

        if operand_type is None or operand_type.kind != "tensor":
            return ()

        shape = tuple(str(axis) for axis in operand_type.shape)
        axis = operation.attrs.get("axis")

        if len(shape) != 2 or isinstance(axis, bool):
            return ()

        if axis is None:
            if any(size != "1" for size in shape[:-1]):
                return ()
        elif not isinstance(axis, int) or axis not in {-1, len(shape) - 1}:
            return ()

        shapes.add(shape)

    if len(shapes) != 1:
        return ()

    shape = next(iter(shapes))
    store = operations[-1]
    output = types.get(store.operands[1])
    stored = types.get(store.operands[0])

    if output is None or stored is None:
        return ()

    domain = tuple(str(axis) for axis in store.attrs.get("target_shape", output.shape))

    if domain not in {shape, shape[:-1], (*shape[:-1], "1")} or (
        tuple(str(axis) for axis in stored.shape) != domain
        and not all(str(axis) == "1" for axis in (*stored.shape, *domain))
    ):
        return ()

    return tuple(operation.results[0].name for operation in reductions)
