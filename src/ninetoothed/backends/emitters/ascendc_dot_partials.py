"""Assign one Cube task to each original K32 dot and output tile."""

import re
from dataclasses import replace

from ninetoothed.backends.emitters.ascendc_dot import CubeDotPlan, render_cube_plan
from ninetoothed.backends.emitters.dot_layout import (
    DenseDotAccess,
    prove_dense_dot_access,
)
from ninetoothed.dtype import normalize_dtype


def _constant(name, definitions):
    operation = definitions.get(name)

    if operation is None or operation.opcode != "arith.constant":
        return None

    value = operation.attrs.get("value")

    return value if isinstance(value, int) and not isinstance(value, bool) else None


def plan_ordered_dot_partials(context):
    """Prove the no-carry partial stage independently of metadata or names.

    The Cartesian product of original K groups and output tiles maps bijectively
    to tasks. A task computes exactly two K16 Mmad operations into a freshly
    initialized FP32 accumulator, then writes its disjoint K32 partial matrix.
    The ordinary second stage owns all original ordered adds and epilogues.
    """
    program = context.kernel.ssa

    if (
        len(program.blocks) != 1
        or len(program.inputs) != 3
        or len(program.outputs) != 1
    ):
        return None

    top = program.blocks[0].operations

    if not top or top[-1].opcode != "scf.for":
        return None

    loop = top[-1]
    definitions = {value.name: op for op in top for value in op.results}

    if (
        len(loop.operands) != 3
        or loop.results
        or len(loop.regions) != 1
        or loop.attrs.get("iter_args")
        or any(op.opcode != "arith.constant" or op.regions for op in top[:-1])
        or {value.name for op in top[:-1] for value in op.results} != set(loop.operands)
    ):
        return None

    lower, groups, step = (_constant(name, definitions) for name in loop.operands)

    if lower != 0 or step != 1 or groups is None or not 0 < groups <= 2047:
        return None

    region = loop.regions[0]

    if (
        len(region.args) != 1
        or region.args[0].name != loop.attrs.get("induction")
        or tuple(op.opcode for op in region.operations)
        not in {
            (
                "tensor.extract",
                "tensor.extract",
                "linalg.dot",
                "mem.store",
                "scf.yield",
            ),
            (
                "tensor.extract",
                "tensor.extract",
                "linalg.matmul",
                "mem.store",
                "scf.yield",
            ),
        }
        or any(op.regions for op in region.operations)
        or any(len(op.results) != 1 for op in region.operations[:3])
    ):
        return None

    left, right, dot, store, yielded = region.operations
    output = program.outputs[0].name

    if (
        dot.operands != (left.results[0].name, right.results[0].name)
        or dot.results[0].type.kind != "tensor"
        or normalize_dtype(dot.results[0].type.dtype) != "float32"
        or tuple(map(str, dot.results[0].type.shape)) != ("32", "32")
        or store.results
        or store.operands != (dot.results[0].name, output)
        or store.attrs
        != {
            "target": output,
            "indices": (region.args[0].name,),
            "target_shape": ("32", "32"),
            "target_dtype_level": 1,
            "base_dtype_level": 0,
        }
        or yielded.operands
        or yielded.results
        or normalize_dtype(program.outputs[0].type.dtype) != "float32"
        or tuple(map(str, program.outputs[0].type.shape)) != (str(groups),)
    ):
        return None

    names = []
    shapes = []

    for extract in (left, right):
        if (
            len(extract.operands) != 2
            or extract.operands[1] != region.args[0].name
            or extract.operands[0] not in context.variables
            or normalize_dtype(extract.results[0].type.dtype) != "float16"
            or extract.results[0].type.kind != "tensor"
            or tuple(map(str, extract.results[0].type.shape)) != ("32", "32")
        ):
            return None

        name = extract.operands[0]
        info = context.tensors[name]

        try:
            shape = tuple(map(int, info.source_shape))
        except (TypeError, ValueError):
            return None

        if len(shape) != 2 or normalize_dtype(info.dtype) != "float16":
            return None

        names.append(name)
        shapes.append(shape)

    if names[0] == names[1] or {value.name for value in program.inputs} != {
        *names,
        output,
    }:
        return None

    (m, k), (right_k, n) = shapes

    try:
        output_info = context.tensors[output]
        output_shape = tuple(map(int, output_info.source_shape))
    except (KeyError, TypeError, ValueError):
        return None

    if (
        m <= 0
        or n <= 0
        or m % 32
        or n % 32
        or k != groups * 32
        or right_k != k
        or output_shape != (groups * m, n)
        or normalize_dtype(output_info.dtype) != "float32"
        or groups * m * n * 4 > 32 * 1024 * 1024
    ):
        return None

    columns = n // 32
    tasks = (m // 32) * columns
    row = f"outer_index // {columns} * 32 + value_0"
    column = f"outer_index % {columns} * 32 + value_1"
    expected = (
        (row, "extract_0_0 * 32 + value_1"),
        ("extract_0_0 * 32 + value_0", column),
        (f"extract_0_0 * {m} + {row}", column),
    )
    attrs = (
        left.results[0].type.attrs,
        right.results[0].type.attrs,
        {**output_info.attrs, "dtype_level": 1},
    )
    accesses = []

    for name, attributes, offsets, shape in zip(
        (*names, output), attrs, expected, (*shapes, output_shape)
    ):
        access = prove_dense_dot_access(
            attributes,
            shape=(32, 32),
            tasks=tasks,
            stages=groups,
            expected_offsets=offsets,
        )

        if (
            access is None
            or access.source_shape != shape
            or access.strides
            != tuple(map(str, context.tensors[name].attrs.get("source_strides", ())))
        ):
            return None
        # Because task = group * tasks + original_tile is bijective over the
        # proved Cartesian domain, the original all-true masks remain true.

        replacements = {
            "outer_index": f"(outer_index % {tasks})",
            "extract_0_0": f"(outer_index // {tasks})",
        }
        rewritten = tuple(
            re.sub(
                r"\b(outer_index|extract_0_0)\b",
                lambda match: replacements[match.group()],
                offset,
            )
            for offset in access.offsets
        )
        accesses.append(DenseDotAccess(rewritten, access.strides, access.source_shape))
    return CubeDotPlan(
        *names,
        output,
        *accesses,
        1,
        tasks * groups,
        "float32",
        origin="explicit-dot-group-partials",
    )


def render_ordered_dot_partials(context, target, *, task_batch=None, output_tile=None):
    """Render the proved first stage; selection and launch stay with its owner."""
    plan = plan_ordered_dot_partials(context)

    if plan is None:
        return None

    if output_tile is None:
        # The measured large Cartesian domain benefits from 2x2 output unions.
        # Keep at least 128 jobs after the union and eight-task setup reuse;
        # small domains and incomplete unions retain the original 32x32 tiles.
        m = plan.left_access.source_shape[0]
        n = plan.right_access.source_shape[1]
        output_tile = (
            64
            if plan.tasks >= 4096
            and m >= 512
            and n >= 512
            and m % 64 == 0
            and n % 64 == 0
            else 32
        )

    if task_batch is None:
        # Before output unions, keep at least 256 launch jobs after setup
        # reuse, well above the platform's 24 Cube cores.
        task_batch = 8 if plan.tasks >= 2048 else 1

    native = render_cube_plan(
        plan, target, task_batch=task_batch, output_tile=output_tile
    )

    if native is None:
        return None

    details = {
        **native.metadata["ascendc_cube_dot"],
        "group_width": 32,
        "task_mapping": "original-group-times-output-tiles-plus-output-tile",
        "accumulation_order": "single-original-k32-dot",
        "requires_reassociation": False,
        "scratch_budget_bytes": 32 * 1024 * 1024,
    }

    if output_tile == 64:
        details["task_mapping"] = "original-k32-group-times-64-output-unions"

    alignment = " && ".join(
        f"((uintptr_t)({name})) % 32 == 0"
        for name in (plan.left, plan.right, plan.output)
    )

    return replace(
        native,
        guard=f"({native.guard}) && ({alignment})",
        metadata={"ascendc_cube_dot": details},
    )


__all__ = ["plan_ordered_dot_partials", "render_ordered_dot_partials"]
