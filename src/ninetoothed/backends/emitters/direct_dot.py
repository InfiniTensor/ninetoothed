"""Derive complete dot tiles from proved full-matrix SSA applications."""

from dataclasses import dataclass, replace

from ninetoothed.backends.emitters.dot_layout import (
    dense_tile_attributes,
    prove_full_matrix_access,
)
from ninetoothed.compiler.cse import pure_expression
from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import ssa


@dataclass(frozen=True)
class DirectDotTiles:
    context: object
    right_transposed: bool
    dimensions: tuple[int, int, int]


def tile_direct_dot(context):
    """Return a target-neutral 32-tile schedule without naming an application.

    Original source maps and predicates must prove full dense matrix domains.
    Only a direct dot and a pointwise epilogue can be retiled. Reductions,
    regions, partial tiles, external epilogue tensors and effects stay indexed.
    The output ownership changes with the derived grid, which callers must
    guard consistently with their original fallback grid.
    """
    if (
        len(context.kernel.ssa.blocks) != 1
        or len(context.stores) != 1
        or len(context.outputs) != 1
        or tuple(map(str, context.outer_axes)) != ("1", "1")
    ):
        return None

    operations = context.kernel.ssa.blocks[0].operations
    dots = [op for op in operations if op.opcode in {"linalg.dot", "linalg.matmul"}]

    if len(dots) != 1 or any(op.regions for op in operations):
        return None

    dot = dots[0]
    store = context.stores[0]

    if (
        operations[-1] is not store
        or len(dot.operands) != 2
        or len(dot.results) != 1
        or set(store.attrs) - {"target", "target_shape"}
        or normalize_dtype(dot.results[0].type.dtype) != "float32"
    ):
        return None

    names = []
    transposed = []

    for name in dot.operands:
        transpose = context.operations.get(name)
        swapped = transpose is not None and transpose.opcode == "linalg.transpose"

        if swapped:
            if (
                transpose.attrs
                or len(transpose.operands) != 1
                or len(transpose.results) != 1
            ):
                return None

            name = transpose.operands[0]

        if (
            name not in context.variables
            or normalize_dtype(context.tensors[name].dtype) != "float16"
        ):
            return None

        names.append(name)
        transposed.append(swapped)

    if names[0] == names[1] or transposed[0]:
        return None

    try:
        m, k = map(int, context.value_types[dot.operands[0]].shape)
        rk, n = map(int, context.value_types[dot.operands[1]].shape)
    except (ValueError, TypeError):
        return None

    if (
        k != rk
        or any(dim <= 0 or dim % 32 for dim in (m, k, n))
        or k // 32 > 4096
        or tuple(map(str, context.axes)) != (str(m), str(n))
    ):
        return None

    output = context.outputs[0]
    physical = ((m, k), (n, k) if transposed[1] else (k, n), (m, n))

    if any(
        prove_full_matrix_access(context.tensors[name], shape) is None
        for name, shape in zip((*names, output), physical)
    ):
        return None

    if normalize_dtype(context.tensors[output].dtype) not in {"float16", "float32"}:
        return None

    needed = set()

    def collect(name):
        if name == dot.results[0].name:
            return True

        op = context.operations.get(name)

        if op is None or not pure_expression(op) or len(op.results) != 1:
            return False

        if (
            op.opcode.startswith("reduce.")
            or op.opcode.startswith("tensor.")
            and op.opcode != "tensor.cast"
        ):
            return False

        if op.results[0].type.shape and tuple(map(str, op.results[0].type.shape)) != (
            str(m),
            str(n),
        ):
            return False

        if not all(collect(operand) for operand in op.operands):
            return False

        needed.add(id(op))

        return True

    if not collect(store.operands[0]):
        return None

    allowed = {id(dot), id(store)} | needed
    allowed.update(
        id(op)
        for op in operations
        if op.opcode == "linalg.transpose" and op.results[0].name in dot.operands
    )

    if any(id(op) not in allowed for op in operations):
        return None

    occupied = set(context.value_types)

    def fresh(label):
        name = "%nt_direct_" + label

        while name in occupied:
            name += "_"

        occupied.add(name)

        return name

    low, high, step, zero, induction, carry, left, right, partial, added = [
        fresh(label)
        for label in (
            "low",
            "high",
            "step",
            "zero",
            "index",
            "carry",
            "left",
            "right",
            "part",
            "sum",
        )
    ]
    tile_type = ssa.Type(kind="tensor", shape=("32", "32"), dtype="float32")
    index_type = ssa.Type(kind="index")
    row = f"floor(outer_index / {n // 32}) * 32 + value_0"
    column = f"Mod(outer_index, {n // 32}) * 32 + value_1"
    descriptors = []
    infos = dict(context.tensors)

    for name, shape, offsets, swap in zip(
        names,
        ((m, k), (k, n)),
        ((row, "extract_0_0 * 32 + value_1"), ("extract_0_0 * 32 + value_0", column)),
        transposed,
    ):
        info = infos[name]
        strides = info.attrs["source_strides"]

        if swap:
            strides = tuple(reversed(strides))

        attrs = dense_tile_attributes(info, offsets, strides, shape)
        infos[name] = replace(info, attrs=attrs)
        descriptors.append(
            ssa.Type(kind="tensor", shape=("32", "32"), dtype="float16", attrs=attrs)
        )

    infos[output] = replace(
        infos[output],
        attrs=dense_tile_attributes(
            infos[output], (row, column), infos[output].attrs["source_strides"], (m, n)
        ),
    )
    region = ssa.Block(
        name="dot_tiles",
        args=(
            ssa.Value(name=induction, type=index_type),
            ssa.Value(name=carry, type=tile_type),
        ),
        operations=(
            ssa.Operation(
                opcode="tensor.extract",
                operands=(names[0], induction),
                results=(ssa.Value(name=left, type=descriptors[0]),),
            ),
            ssa.Operation(
                opcode="tensor.extract",
                operands=(names[1], induction),
                results=(ssa.Value(name=right, type=descriptors[1]),),
            ),
            ssa.Operation(
                opcode="linalg.dot",
                operands=(left, right),
                results=(ssa.Value(name=partial, type=tile_type),),
            ),
            ssa.Operation(
                opcode="arith.add",
                operands=(carry, partial),
                results=(ssa.Value(name=added, type=tile_type),),
            ),
            ssa.Operation(opcode="scf.yield", operands=(added,)),
        ),
    )
    prefix = tuple(
        ssa.Operation(
            opcode="arith.constant",
            results=(ssa.Value(name=name, type=index_type),),
            attrs={"value": value},
        )
        for name, value in ((low, 0), (high, k // 32), (step, 1))
    )
    zero_op = ssa.Operation(
        opcode="tensor.zeros", results=(ssa.Value(name=zero, type=tile_type),)
    )
    loop = ssa.Operation(
        opcode="scf.for",
        operands=(low, high, step, zero),
        results=(replace(dot.results[0], type=tile_type),),
        attrs={
            "induction": induction,
            "iter_args": ({"initial": zero, "block_arg": carry},),
        },
        regions=(region,),
    )
    epilogue = tuple(
        replace(
            op,
            results=tuple(
                replace(value, type=replace(value.type, shape=("32", "32")))
                if value.type.shape
                else value
                for value in op.results
            ),
        )
        for op in operations
        if id(op) in needed
    )
    tile_store = replace(store, attrs={**store.attrs, "target_shape": ("32", "32")})
    inputs = tuple(
        replace(value, type=replace(value.type, attrs=infos[value.name].attrs))
        if value.name in infos
        else value
        for value in context.kernel.ssa.inputs
    )
    program = replace(
        context.kernel.ssa,
        inputs=inputs,
        outputs=tuple(
            next(value for value in inputs if value.name == output)
            for _ in context.kernel.ssa.outputs
        ),
        blocks=(ssa.Block(operations=(*prefix, zero_op, loop, *epilogue, tile_store)),),
    )
    ssa.verify_program(program)
    defs = {
        result.name: op for op in program.blocks[0].operations for result in op.results
    }
    defs.update({result.name: op for op in region.operations for result in op.results})
    types = {value.name: value.type for value in program.inputs}
    types.update(
        {
            value.name: value.type
            for op in (*program.blocks[0].operations, *region.operations)
            for value in op.results
        }
    )
    tiled = replace(
        context,
        kernel=replace(context.kernel, ssa=program),
        tensors=infos,
        operations=defs,
        value_types=types,
        stores=(tile_store,),
        axes=("32", "32"),
        outer_axes=(str(m // 32), str(n // 32)),
    )

    return DirectDotTiles(tiled, transposed[1], (m, k, n))
