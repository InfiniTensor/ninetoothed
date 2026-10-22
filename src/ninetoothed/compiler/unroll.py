"""Expose small pure static recurrences to the shared expression pipeline."""

from collections.abc import Mapping
from dataclasses import replace

from ninetoothed.ir import ssa

_ELEMENTWISE = {
    "arith.constant",
    "arith.add",
    "arith.sub",
    "arith.mul",
    "arith.div",
    "arith.neg",
    "arith.maximum",
    "arith.minimum",
    "arith.pow",
    "arith.floordiv",
    "arith.mod",
    "arith.lt",
    "arith.le",
    "arith.gt",
    "arith.ge",
    "arith.eq",
    "arith.ne",
    "cmp.lt",
    "cmp.le",
    "cmp.gt",
    "cmp.ge",
    "cmp.eq",
    "cmp.ne",
    "select.where",
    "tensor.cast",
} | {
    "math." + name
    for name in (
        "abs",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log1p",
        "sin",
        "cos",
        "tan",
        "tanh",
        "erf",
        "ceil",
        "floor",
        "sigmoid",
    )
}


def unroll_static_elementwise_loops(program, *, max_iterations=8, max_operations=64):
    """Inline typed pure operations in their original order, without reassociation.

    Bounds must be literal integers and carry/yield types must match exactly.
    Memory, reductions, indexing, dot operations and nested regions are excluded.
    A program-wide operation budget bounds source growth.
    """
    names = {value.name for value in (*program.inputs, *program.outputs)}

    def inventory(block):
        names.update(value.name for value in block.args)

        for operation in block.operations:
            names.update(value.name for value in operation.results)

            for region in operation.regions:
                inventory(region)

    for block in program.blocks:
        inventory(block)

    counter = 0
    expanded = []
    budget = max_operations
    aliases = {}
    types = {value.name: value.type for value in program.inputs}

    def fresh():
        nonlocal counter

        while True:
            name = f"%nt_unroll_{counter}"
            counter += 1

            if name not in names:
                names.add(name)

                return name

    def rewrite(item, bindings):
        if isinstance(item, str):
            return bindings.get(item, item)

        if isinstance(item, Mapping):
            return {key: rewrite(value, bindings) for key, value in item.items()}

        if isinstance(item, (list, tuple)):
            return tuple(rewrite(value, bindings) for value in item)
        return item

    def plan(operation, constants):
        if (
            operation.opcode != "scf.for"
            or len(operation.regions) != 1
            or len(operation.operands) < 3
        ):
            return None

        bounds = [constants.get(name) for name in operation.operands[:3]]

        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in bounds
        ):
            return None

        lower, upper, step = bounds

        if step <= 0:
            return None

        count = max(0, (upper - lower + step - 1) // step)

        if count > max_iterations:
            return None

        region = operation.regions[0]

        if (
            set(operation.attrs) - {"induction", "iter_args", "python_target"}
            or not region.operations
            or region.operations[-1].opcode != "scf.yield"
            or region.operations[-1].results
            or region.operations[-1].attrs
            or region.operations[-1].regions
            or len(region.args) != len(operation.operands) - 2
            or len(operation.results) != len(region.args) - 1
            or len(region.operations[-1].operands) != len(operation.results)
            or operation.attrs.get("induction") != region.args[0].name
        ):
            return None

        iter_args = operation.attrs.get("iter_args", ())

        if len(iter_args) != len(operation.results) or any(
            item.get("block_arg") != argument.name or item.get("initial") != initial
            for item, argument, initial in zip(
                iter_args, region.args[1:], operation.operands[3:]
            )
        ):
            return None

        body = region.operations[:-1]

        if any(
            op.opcode not in _ELEMENTWISE or op.regions or len(op.results) != 1
            for op in body
        ):
            return None

        if count * (len(body) + 1) > budget:
            return None

        available = types | {value.name: value.type for value in region.args}
        available.update(
            (value.name, value.type) for op in body for value in op.results
        )

        for initial, argument, result, yielded in zip(
            operation.operands[3:],
            region.args[1:],
            operation.results,
            region.operations[-1].operands,
        ):
            if not (
                types.get(initial)
                == argument.type
                == result.type
                == available.get(yielded)
            ):
                return None
        return region, range(lower, upper, step), body

    blocks = []

    def rewrite_region(block):
        return replace(
            block,
            operations=tuple(
                replace(
                    operation,
                    operands=rewrite(operation.operands, aliases),
                    attrs=rewrite(operation.attrs, aliases),
                    regions=tuple(
                        rewrite_region(region) for region in operation.regions
                    ),
                )
                for operation in block.operations
            ),
        )

    for block in program.blocks:
        constants = {}
        operations = []

        for original in block.operations:
            current = replace(
                original,
                operands=rewrite(original.operands, aliases),
                attrs=rewrite(original.attrs, aliases),
                regions=tuple(rewrite_region(region) for region in original.regions),
            )
            loop = plan(current, constants)

            if loop is None:
                operations.append(current)
                types.update((value.name, value.type) for value in current.results)

                if current.opcode == "arith.constant" and len(current.results) == 1:
                    constants[current.results[0].name] = current.attrs.get("value")

                continue

            region, iterations, body = loop
            carried = current.operands[3:]

            for iteration in iterations:
                induction = fresh()
                operations.append(
                    ssa.Operation(
                        opcode="arith.constant",
                        results=(ssa.Value(name=induction, type=region.args[0].type),),
                        attrs={"value": iteration},
                    )
                )
                bindings = (
                    aliases
                    | dict(zip((value.name for value in region.args[1:]), carried))
                    | {region.args[0].name: induction}
                )

                for operation in body:
                    result = operation.results[0]
                    clone = replace(
                        operation,
                        operands=rewrite(operation.operands, bindings),
                        attrs=rewrite(operation.attrs, bindings),
                        results=(replace(result, name=fresh()),),
                    )
                    operations.append(clone)
                    bindings[result.name] = clone.results[0].name
                    types[clone.results[0].name] = clone.results[0].type

                carried = tuple(
                    rewrite(name, bindings) for name in region.operations[-1].operands
                )

            aliases.update(
                (value.name, name) for value, name in zip(current.results, carried)
            )
            budget -= len(iterations) * (len(body) + 1)
            expanded.append(
                (original.results[0].name if original.results else "", len(iterations))
            )

        blocks.append(replace(block, operations=tuple(operations)))
    return replace(
        program,
        blocks=tuple(blocks),
        outputs=tuple(
            replace(value, name=rewrite(value.name, aliases))
            for value in program.outputs
        ),
    ), tuple(expanded)
