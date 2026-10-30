"""Materialize bounded SSA tile values once per program and loop iteration.

This is a conservative execution strategy for native scalar targets. It uses
the common SSA expression/index walker, preserves typed storage boundaries and
simultaneous region yields, and reuses scratch slots by SSA last use.
"""

from collections import Counter
from dataclasses import dataclass
from functools import cache
from math import prod

from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import walk_ops
from ninetoothed.backends.emitters.context import EmitContext
from ninetoothed.backends.emitters.expressions import c_integer_expr
from ninetoothed.ir import ssa


@dataclass(frozen=True)
class ScalarTilePlan:
    body: str
    tasks: str
    scratch_bytes: int


def plan_scalar_tiles(context, target, *, storage, budget, inline_elementwise=False):
    if any(
        name.startswith("nt_")
        for name in (*context.variables, *context.outputs, *context.shape_params)
    ):
        return None

    program = context.kernel.ssa

    if program is None or len(program.blocks) != 1:
        return None

    operations = program.blocks[0].operations
    all_ops = tuple(walk_ops(operations))

    loop_products = (
        any(op.opcode == "scf.for" for op in all_ops)
        and sum(op.opcode in ("linalg.dot", "linalg.matmul") for op in all_ops) >= 2
    )
    repeated_reductions = sum(op.opcode.startswith("reduce.") for op in all_ops) >= 2

    if not (loop_products or repeated_reductions):
        return None

    if any(
        op.opcode.startswith("mem.")
        and op.opcode not in ("mem.load", "mem.data_ptr", "mem.store")
        for op in all_ops
    ):
        return None

    if any(op.opcode == "mem.store" for op in all_ops if op not in operations):
        return None

    yielded = {
        name for op in all_ops if op.opcode == "scf.yield" for name in op.operands
    }
    pure = {
        "arith.add",
        "arith.sub",
        "arith.mul",
        "arith.div",
        "arith.truediv",
        "arith.floordiv",
        "arith.mod",
        "arith.neg",
        "arith.cast",
        "arith.bitand",
        "arith.bitor",
        "arith.bitxor",
        "arith.lshift",
        "arith.rshift",
        "cmp.eq",
        "cmp.ne",
        "cmp.lt",
        "cmp.le",
        "cmp.gt",
        "cmp.ge",
        "tensor.zeros",
        "tensor.empty",
    }
    inline = {
        op.results[0].name: op
        for op in all_ops
        if inline_elementwise
        and op.opcode in pure
        and not op.regions
        and len(op.results) == 1
        and op.results[0].type.kind == "tensor"
        and op.results[0].type.shape
        and op.results[0].name not in yielded
    }

    @cache
    def dependencies(name):
        op = inline.get(name)

        return (
            tuple(leaf for operand in op.operands for leaf in dependencies(operand))
            if op is not None
            else (name,)
        )

    def operands(op):
        return tuple(leaf for name in op.operands for leaf in dependencies(name))

    initial = {}

    for op in all_ops:
        for result, attr in zip(op.results, op.attrs.get("iter_args", ())):
            initial[result.name] = attr["initial"]
            initial[attr["block_arg"]] = attr["initial"]

    lines = []
    ctx = EmitContext(
        target=target,
        kernel=context.kernel,
        program=program,
        operations=context.operations,
        value_types=context.value_types,
        lines=lines,
        memo={},
        tensor_infos=context.tensors,
        output=context.outputs[0],
        output_axes=context.axes,
        index_expr="nt_tile_index",
        inner_index_expr="nt_tile_index",
        outer_index_expr="nt_outer_index",
        mask_expr=None,
        type_initials=initial,
        temp_counter=[0],
        parameter_names=frozenset(
            (*context.variables, *context.outputs, *context.shape_params)
        ),
    )
    slots = []
    active = {}
    declarations = []

    def shape(name):
        type_ = context.value_types[name]
        axes = tuple(map(str, type_.shape))

        if type_.kind != "tensor" or not axes:
            return ()

        if len(axes) > 2 or any(not x.isdecimal() or int(x) <= 0 for x in axes):
            raise ValueError("Nonstatic scalar tile shape.")

        if prod(map(int, axes)) > 8192:
            raise ValueError("Scalar tile exceeds bounded storage.")
        return axes

    def allocate(name, local):
        axes = shape(name)
        count = prod(map(int, axes)) if axes else 1
        witness = common.resolve_value_dtype(name, local)

        if witness.dtype is None:
            raise ValueError("Unresolved scalar tile storage element type.")

        dtype = (
            target.type_name(witness.dtype)
            if witness.dtype
            else target.value_dtype(witness.sample)
        )
        width = (
            2
            if witness.dtype in ("float16", "bfloat16")
            else 8
            if witness.dtype in ("index", "int64", "uint64", "float64")
            else 4
        )

        for index, slot in enumerate(slots):
            if index not in active.values() and slot[:2] == (dtype, count):
                active[name] = index

                return slot[2], axes

        buffer = f"nt_tile_buffer_{len(slots)}"
        active[name] = len(slots)
        slots.append((dtype, count, buffer, width))

        if not callable(storage):
            declarations.append(f"{storage} {dtype} {buffer}[{count}];")

        if sum(n * w for _, n, _, w in slots) > budget:
            raise ValueError("Scalar tile scratch budget exceeded.")
        return buffer, axes

    def fill(destination, value, axes, local):
        index = common.fresh_temp(local, "nt_tile_index")
        count = prod(map(int, axes)) if axes else 1
        coordinates = common.coords_from_linear(index, axes, target) if axes else ()
        statements = []
        body = local.child(
            lines=statements,
            memo={
                n: v
                for n, v in local.memo.items()
                if not context.value_types.get(n, ssa.Type(kind="scalar")).shape
            },
            coordinate_exprs=coordinates,
            output_axes=axes,
            index_expr=index,
            inner_index_expr=index,
            local_suffix=common.nested_local_suffix(local, index),
        )
        expression = common.emit_element(value, coordinates, body)
        statements.append(f"{destination}[{index}] = {expression};")
        local.lines.append(target.loop_header(index, "0", str(count), "1"))
        local.lines.extend("    " + statement for statement in statements)
        local.lines.append("}")

    def copy_yields(op, region, local):
        values = region.operations[-1].operands

        for result, value in zip(op.results, values):
            buffer, axes = local.tensor_buffers[result.name]
            fill(buffer, value, axes, local)

    def render(ops, local, protected=frozenset()):
        uses = Counter(
            operand
            for op in walk_ops(ops)
            if not (op.results and op.results[0].name in inline)
            for operand in operands(op)
        )

        for op in ops:
            if op.results and op.results[0].name in inline:
                continue

            if op.opcode == "scf.yield":
                continue

            if op.opcode == "scf.for":
                if set(op.regions[0].operations[-1].operands) & {
                    attr["block_arg"] for attr in op.attrs.get("iter_args", ())
                }:
                    raise ValueError(
                        "Direct carried-buffer yields require a parallel copy."
                    )

                for result, attr in zip(op.results, op.attrs.get("iter_args", ())):
                    buf = allocate(result.name, local)
                    fill(buf[0], attr["initial"], buf[1], local)
                    local.tensor_buffers[result.name] = buf

                loop_var = common.fresh_temp(local, "nt_tile_loop")
                bounds = [common.emit_loop_bound(n, local) for n in op.operands[:3]]
                local.lines.append(target.loop_header(loop_var, *bounds))
                body_lines = []
                buffers = dict(local.tensor_buffers)

                for result, attr in zip(op.results, op.attrs.get("iter_args", ())):
                    buffers[attr["block_arg"]] = buffers[result.name]

                body = local.child(
                    lines=body_lines,
                    memo=dict(local.memo),
                    bindings={
                        **(local.bindings or {}),
                        op.attrs["induction"]: loop_var,
                    },
                    tensor_buffers=buffers,
                )
                keep = frozenset(active)
                render(op.regions[0].operations, body, keep)
                copy_yields(op, op.regions[0], body)

                for name in set(active) - keep:
                    active.pop(name, None)

                local.lines.extend("    " + line for line in body_lines)
                local.lines.append("}")
            elif op.opcode == "scf.if":
                for result in op.results:
                    local.tensor_buffers[result.name] = allocate(result.name, local)

                condition = common.emit_value(op.operands[0], local)

                for i, region in enumerate(op.regions):
                    local.lines.append(f"if ({condition}) {{" if i == 0 else "else {")
                    branch_lines = []
                    branch = local.child(
                        lines=branch_lines,
                        memo=dict(local.memo),
                        tensor_buffers=dict(local.tensor_buffers),
                    )
                    keep = frozenset(active)
                    render(region.operations, branch, keep)
                    copy_yields(op, region, branch)

                    for name in set(active) - keep:
                        active.pop(name, None)

                    local.lines.extend("    " + line for line in branch_lines)
                    local.lines.append("}")
            elif op.opcode == "mem.store":
                value = op.operands[0]
                axes = shape(value)
                index = common.fresh_temp(local, "nt_tile_store")
                coordinates = common.coords_from_linear(index, axes, target)
                statements = []
                body = local.child(
                    lines=statements,
                    memo=dict(local.memo),
                    coordinate_exprs=coordinates,
                    output_axes=axes,
                    index_expr=index,
                    inner_index_expr=index,
                )
                common.emit_operation(op, body)
                local.lines.append(
                    target.loop_header(index, "0", str(prod(map(int, axes))), "1")
                )
                local.lines.extend("    " + line for line in statements)
                local.lines.append("}")
            elif op.results and not op.opcode.startswith("dtype."):
                if len(op.results) != 1:
                    raise ValueError("Unsupported multi-result scalar tile operation.")

                name = op.results[0].name

                if shape(name):
                    buffer, axes = allocate(name, local)
                    fill(buffer, name, axes, local)
                    local.tensor_buffers[name] = (buffer, axes)
                else:
                    common.emit_value(name, local)

            for operand in operands(op):
                uses[operand] -= 1

                if uses[operand] == 0 and operand not in protected:
                    active.pop(operand, None)

    try:
        render(operations, ctx)
    except (ValueError, KeyError, IndexError) as error:
        context.lowering_plans["scalar_tile_rejection"] = str(error)

        return None

    if callable(storage):
        declarations = storage(slots)

    prefix = [
        "const int64_t nt_outer_index = (int64_t)(" + target.program_id(0) + ");",
        *declarations,
    ]
    body = c_integer_expr("\n".join((*prefix, *lines)))
    tasks = c_integer_expr(" * ".join(f"({n})" for n in context.outer_axes) or "1")

    return ScalarTilePlan(body, tasks, sum(n * w for _, n, _, w in slots))
