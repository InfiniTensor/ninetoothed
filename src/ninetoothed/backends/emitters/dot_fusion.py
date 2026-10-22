"""Fuse independent complete output tiles without changing K-stage grouping."""

import math
from dataclasses import replace

from ninetoothed.backends.emitters.analysis import dot_accumulation_plan
from ninetoothed.backends.emitters.dot_layout import (
    dense_tile_attributes,
    prove_dense_dot_access,
)
from ninetoothed.compiler.cse import pure_expression
from ninetoothed.ir import ssa


def fuse_dot_output_tiles(context):
    """Derive a 64x64 output grid from proved independent 32x32 dot loops."""
    if (
        len(context.outputs) != 1
        or len(context.stores) != 1
        or len(context.kernel.ssa.blocks) != 1
        or tuple(map(str, context.axes)) != ("32", "32")
    ):
        return None

    plan = dot_accumulation_plan(context.kernel.ssa)

    if plan is None:
        return None

    loop, dot, store = plan.loop, plan.dot, plan.store
    region = loop.regions[0]

    if (
        len(region.operations) != 5
        or any(op.regions for op in region.operations)
        or [op.opcode for op in region.operations]
        != ["tensor.extract", "tensor.extract", "linalg.dot", "arith.add", "scf.yield"]
        or len(loop.operands) != 4
        or set(store.attrs) - {"target", "target_shape"}
    ):
        return None

    top = context.kernel.ssa.blocks[0].operations

    if any(
        op is not loop
        and op is not store
        and (not pure_expression(op) or op.opcode.startswith("reduce."))
        for op in top
    ):
        return None

    output = context.outputs[0]

    try:
        m, n = map(int, context.tensors[output].attrs["source_shape"])
        tasks = math.prod(map(int, context.outer_axes))
        extracts = [context.operations[name] for name in dot.operands]
        names = [op.operands[0] for op in extracts]
        k = int(extracts[0].results[0].type.attrs["source_shape"][1])
    except (KeyError, TypeError, ValueError):
        return None

    if (
        any(dim <= 0 or dim % 64 for dim in (m, n))
        or k <= 0
        or k % 32
        or tasks != (m // 32) * (n // 32)
        or len(set(names)) != 2
        or any(name not in context.variables for name in names)
    ):
        return None

    stages = k // 32
    bounds = [context.operations.get(name) for name in loop.operands[:3]]

    if (
        any(op is None for op in bounds)
        or bounds[0].opcode != "arith.constant"
        or bounds[0].attrs.get("value") != 0
        or bounds[2].opcode != "arith.constant"
        or bounds[2].attrs.get("value") != 1
    ):
        return None

    high = bounds[1]

    if high.opcode == "arith.constant":
        extent = high.attrs.get("value")
    elif high.opcode == "shape.dim" and not high.attrs.get("source"):
        try:
            extent = int(
                context.value_types[high.operands[0]].shape[int(high.attrs["dim"])]
            )
        except (KeyError, ValueError, TypeError, IndexError):
            return None
    else:
        return None

    if extent != stages or stages > 4096:
        return None

    original_row = f"floor(outer_index / {n // 32}) * 32 + value_0"
    original_column = f"Mod(outer_index, {n // 32}) * 32 + value_1"
    original_maps = (
        (original_row, "extract_0_0 * 32 + value_1"),
        ("extract_0_0 * 32 + value_0", original_column),
        (original_row, original_column),
    )

    for attrs, expected, physical in zip(
        [extract.results[0].type.attrs for extract in extracts]
        + [context.tensors[output].attrs],
        original_maps,
        ((m, k), (k, n), (m, n)),
    ):
        if (
            tuple(map(str, attrs.get("source_shape", ()))) != tuple(map(str, physical))
            or prove_dense_dot_access(
                attrs,
                shape=(32, 32),
                tasks=tasks,
                stages=stages,
                expected_offsets=expected,
            )
            is None
        ):
            return None
    # The epilogue must depend only on the accumulator and uniform literals.

    def pure_epilogue(name):
        if name == loop.results[0].name:
            return True

        op = context.operations.get(name)

        if op is None or len(op.results) != 1 or not pure_expression(op):
            return False

        if op.opcode == "shape.dim":
            return not op.attrs.get("source")

        if (
            op.opcode.startswith("reduce.")
            or op.opcode.startswith("tensor.")
            and op.opcode != "tensor.cast"
        ):
            return False
        return all(pure_epilogue(operand) for operand in op.operands)

    if not pure_epilogue(store.operands[0]):
        return None

    shape_constants = {}

    for op in top:
        if op.opcode == "shape.dim":
            if op.attrs.get("source") or len(op.operands) != 1:
                return None

            try:
                shape_constants[id(op)] = int(
                    context.value_types[op.operands[0]].shape[int(op.attrs["dim"])]
                )
            except (KeyError, TypeError, ValueError, IndexError):
                return None

    row = f"floor(outer_index / {n // 64}) * 64 + value_0"
    column = f"Mod(outer_index, {n // 64}) * 64 + value_1"
    infos = dict(context.tensors)
    replacements = {}

    for extract, name, physical, shape, offsets in zip(
        extracts,
        names,
        ((m, k), (k, n)),
        ((64, 32), (32, 64)),
        ((row, "extract_0_0 * 32 + value_1"), ("extract_0_0 * 32 + value_0", column)),
    ):
        info = infos[name]
        attrs = dense_tile_attributes(
            info, offsets, info.attrs["source_strides"], physical, tile_shape=shape
        )
        infos[name] = replace(info, attrs=attrs)
        value = extract.results[0]
        replacements[value.name] = replace(
            value, type=replace(value.type, shape=tuple(map(str, shape)), attrs=attrs)
        )

    infos[output] = replace(
        infos[output],
        attrs=dense_tile_attributes(
            infos[output],
            (row, column),
            infos[output].attrs["source_strides"],
            (m, n),
            tile_shape=(64, 64),
        ),
    )

    def value_tile(value):
        if value.name in replacements:
            return replacements[value.name]

        if tuple(map(str, value.type.shape)) == ("32", "32"):
            return replace(value, type=replace(value.type, shape=("64", "64")))
        return value

    def block_tile(block):
        return replace(
            block,
            args=tuple(value_tile(value) for value in block.args),
            operations=tuple(
                replace(
                    op,
                    opcode="arith.constant" if id(op) in shape_constants else op.opcode,
                    operands=() if id(op) in shape_constants else op.operands,
                    results=tuple(value_tile(value) for value in op.results),
                    regions=tuple(block_tile(region) for region in op.regions),
                    attrs={"value": shape_constants[id(op)]}
                    if id(op) in shape_constants
                    else {**op.attrs, "target_shape": ("64", "64")}
                    if op is store
                    else op.attrs,
                )
                for op in block.operations
            ),
        )

    inputs = tuple(
        replace(
            value_tile(value),
            type=replace(value_tile(value).type, attrs=infos[value.name].attrs),
        )
        if value.name in infos
        else value_tile(value)
        for value in context.kernel.ssa.inputs
    )
    program = replace(
        context.kernel.ssa,
        inputs=inputs,
        outputs=tuple(
            next(value for value in inputs if value.name == output)
            for _ in context.kernel.ssa.outputs
        ),
        blocks=(block_tile(context.kernel.ssa.blocks[0]),),
        metadata={
            **context.kernel.ssa.metadata,
            "output_tile_fusion": (2, 2),
            "k_stage_grouping": "preserved",
        },
    )
    ssa.verify_program(program)
    definitions = {}
    types = {value.name: value.type for value in program.inputs}

    def inventory(block):
        types.update((value.name, value.type) for value in block.args)

        for op in block.operations:
            definitions.update((value.name, op) for value in op.results)
            types.update((value.name, value.type) for value in op.results)

            for region in op.regions:
                inventory(region)

    inventory(program.blocks[0])
    tiled_store = next(
        op for op in program.blocks[0].operations if op.opcode == "mem.store"
    )

    return replace(
        context,
        kernel=replace(context.kernel, ssa=program),
        tensors=infos,
        operations=definitions,
        value_types=types,
        stores=(tiled_store,),
        axes=("64", "64"),
        outer_axes=(str(m // 64), str(n // 64)),
    )
