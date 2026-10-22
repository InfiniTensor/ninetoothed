"""Derive independent row tiles from complete, explicitly broadcast SSA domains."""

from dataclasses import dataclass, replace

from ninetoothed.backends.emitters.dot_layout import (
    dense_tile_attributes,
    prove_full_matrix_access,
)
from ninetoothed.compiler.cse import pure_expression
from ninetoothed.ir import ssa


@dataclass(frozen=True)
class FullRowTiles:
    context: object
    rows: int
    columns: int


def tile_full_rows(context):
    """Preserve reduction axes, rank-drop broadcasting and physical ownership.

    Only complete two-dimensional applications with explicit row broadcasts
    can be split. Shape queries retain the original application values. The
    original kernel remains the caller's fallback under its original grid.
    """
    if (
        len(context.kernel.ssa.blocks) != 1
        or len(context.stores) != 1
        or len(context.outputs) != 1
        or tuple(map(str, context.outer_axes)) != ("1", "1")
        or len(context.axes) != 2
    ):
        return None

    try:
        rows, columns = map(int, context.axes)
    except (TypeError, ValueError):
        return None

    if rows <= 1 or columns <= 1:
        return None

    matrix = (str(rows), str(columns))
    row = (str(rows),)
    expanded = (str(rows), "1")
    allowed_shapes = {(), ("1",), row, expanded, matrix}
    operations = context.kernel.ssa.blocks[0].operations
    store = context.stores[0]

    if (
        not operations
        or operations[-1] is not store
        or set(store.attrs) - {"target", "target_shape"}
    ):
        return None

    reductions = []
    constants = {}

    for operation in operations[:-1]:
        if (
            operation.regions
            or len(operation.results) != 1
            or not pure_expression(operation)
        ):
            return None

        result_shape = tuple(map(str, operation.results[0].type.shape or ()))

        if result_shape not in allowed_shapes:
            return None

        operand_shapes = [
            tuple(map(str, context.value_types[name].shape or ()))
            for name in operation.operands
        ]

        if operation.opcode in {"reduce.sum", "reduce.max", "reduce.min"}:
            if (
                operation.attrs.get("axis") not in {1, -1}
                or operand_shapes != [matrix]
                or result_shape != row
            ):
                return None

            reductions.append(operation.results[0].name)
        elif operation.opcode == "shape.dim":
            if operation.attrs.get("source") or len(operation.operands) != 1:
                return None

            try:
                constants[id(operation)] = int(
                    operand_shapes[0][operation.attrs["dim"]]
                )
            except (KeyError, TypeError, ValueError, IndexError):
                return None
        elif operation.opcode == "tensor.view":
            if (
                operation.attrs.get("source")
                or operation.attrs.get("subscript") != "(:, None)"
                or operand_shapes != [row]
                or result_shape != expanded
            ):
                return None
        elif (
            operation.opcode in {"tensor.cast", "tensor.full", "tensor.zeros"}
            or operation.opcode.startswith(("arith.", "math.", "cmp."))
            or operation.opcode == "select.where"
        ):
            # A rank-drop row vector aligns to columns in matrix expressions;
            # it must first pass through the explicit [:, None] view above.
            if result_shape == matrix and row in operand_shapes:
                return None

            if result_shape in {row, expanded} and matrix in operand_shapes:
                return None

            if result_shape == row and expanded in operand_shapes:
                return None

            if result_shape == expanded and row in operand_shapes:
                return None
        else:
            return None

    if not reductions:
        return None

    value_shape = tuple(map(str, context.value_types[store.operands[0]].shape or ()))

    if value_shape not in {matrix, expanded, (), ("1",)}:
        return None

    live = set()

    def collect_live(name):
        if name in context.variables:
            live.add(name)

            return

        operation = context.operations.get(name)

        if operation is not None and operation.opcode != "shape.dim":
            for operand in operation.operands:
                collect_live(operand)

    collect_live(store.operands[0])
    variables = tuple(name for name in context.variables if name in live)
    infos = dict(context.tensors)

    for name in (*variables, *context.outputs):
        info = infos[name]

        if info.ndim == 0:
            continue

        if info.ndim != 2 or prove_full_matrix_access(info, (rows, columns)) is None:
            return None

        attrs = dense_tile_attributes(
            info,
            ("outer_index + value_0", "value_1"),
            info.attrs["source_strides"],
            (rows, columns),
            tile_shape=(1, columns),
        )
        attrs.update(view_shape=(str(rows), "1"), view_ndim=2)
        template = dict(attrs["access_templates"][0])
        template["mask"] = (
            f"(outer_index >= 0) & (outer_index < {rows}) & (value_0 >= 0) & (value_0 < 1) & (value_1 >= 0) & (value_1 < {columns})"
        )
        attrs["access_templates"] = (template,)
        infos[name] = replace(info, attrs=attrs)

    def tile_shape(shape):
        shape = tuple(map(str, shape or ()))

        return (
            ("1", str(columns))
            if shape == matrix
            else ("1",)
            if shape == row
            else ("1", "1")
            if shape == expanded
            else shape
        )

    def tile_value(value):
        return replace(
            value,
            type=replace(
                value.type,
                shape=tile_shape(value.type.shape),
                attrs=infos[value.name].attrs
                if value.name in infos
                else value.type.attrs,
            ),
        )

    rewritten = []

    for operation in operations:
        if id(operation) in constants:
            rewritten.append(
                replace(
                    operation,
                    opcode="arith.constant",
                    operands=(),
                    attrs={"value": constants[id(operation)]},
                )
            )
            continue

        attrs = dict(operation.attrs)

        if operation is store:
            attrs["target_shape"] = ("1", str(columns))
        elif operation.opcode == "tensor.full" and "shape" in attrs:
            attrs["shape"] = tile_shape(attrs["shape"])

        rewritten.append(
            replace(
                operation,
                results=tuple(map(tile_value, operation.results)),
                attrs=attrs,
            )
        )

    program = replace(
        context.kernel.ssa,
        inputs=tuple(map(tile_value, context.kernel.ssa.inputs)),
        outputs=tuple(map(tile_value, context.kernel.ssa.outputs)),
        blocks=(replace(context.kernel.ssa.blocks[0], operations=tuple(rewritten)),),
        metadata={
            **context.kernel.ssa.metadata,
            "full_row_tiling": {
                "source_shape": matrix,
                "tile_shape": ("1", str(columns)),
                "shape_queries": "frozen-original",
            },
            "reduction_reuse": {
                "enabled": True,
                "results": tuple(reductions),
                "source": "target-derived-full-row-proof",
            },
        },
    )
    from ninetoothed.compiler.passes import AnalyzeEffects, Context

    program = AnalyzeEffects().run(
        program,
        Context(
            backend=context.kernel.compiler_options["target"]["backend"],
            target=context.kernel.ssa.facts.get("target"),
            compiler_options=context.kernel.compiler_options,
            kernel_metadata=context.kernel.metadata,
            tensors=context.kernel.tensors,
        ),
    )
    ssa.verify_program(program)
    types = {value.name: value.type for value in program.inputs}
    definitions = {}

    for operation in program.blocks[0].operations:
        types.update((value.name, value.type) for value in operation.results)
        definitions.update((value.name, operation) for value in operation.results)

    derived = replace(
        context,
        lowering_plans={},
        execution_plans={},
        kernel=replace(context.kernel, ssa=program),
        tensors=infos,
        value_types=types,
        operations=definitions,
        stores=(rewritten[-1],),
        variables=variables,
        axes=("1", str(columns)),
        outer_axes=(str(rows), "1"),
        cooperative_reduction_program=True,
    )

    return FullRowTiles(derived, rows, columns)


__all__ = ["FullRowTiles", "tile_full_rows"]
