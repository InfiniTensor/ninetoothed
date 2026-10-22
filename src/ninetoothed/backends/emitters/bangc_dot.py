"""Lower proved additive dot regions to BANG-C LT primitives."""

import math
import re
from dataclasses import dataclass
from types import SimpleNamespace

from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import dot_accumulation_plan
from ninetoothed.backends.emitters.context import EmitContext
from ninetoothed.backends.emitters.dot_layout import (
    integer_interval as _range,
)
from ninetoothed.backends.emitters.dot_layout import (
    predicate_always_true as _always_true,
)
from ninetoothed.backends.emitters.dot_layout import (
    prove_dense_dot_access,
)
from ninetoothed.compiler.cse import pure_expression


@dataclass(frozen=True)
class DotInput:
    name: str
    row: str
    column: str
    strides: tuple[str, str]


@dataclass(frozen=True)
class NativeDotBody:
    body: str
    tasks: str
    guard: str


def _input(context, target, operand, stages, tasks, shape):
    import sympy

    extract = context.operations.get(operand)

    if (
        extract is None
        or extract.opcode != "tensor.extract"
        or len(extract.operands) != 2
        or extract.operands[0] not in context.variables
    ):
        return None

    info = context.tensors[extract.operands[0]]
    attrs = extract.results[0].type.attrs

    if (
        common.normalize_dtype(info.dtype) != "float16"
        or common.normalize_dtype(extract.results[0].type.dtype) != "float16"
        or attrs.get("source_ndim") != 2
        or attrs.get("jagged_dim") is not None
    ):
        return None

    templates = attrs.get("access_templates", ())

    if len(templates) != 1:
        return None

    template = templates[0]

    if template.get("level", 0) != attrs.get("dtype_level", 0) or template.get(
        "jagged"
    ):
        return None

    offsets = template.get("offsets", ())
    strides = attrs.get("source_strides", ())

    if len(offsets) != 2 or len(strides) != 2:
        return None

    if tuple(map(str, template.get("shape", ()))) != tuple(map(str, shape)):
        return None

    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(map(str, offsets)))
        if name not in {"floor", "Mod"}
    }

    for name in ("outer_index", "value_0", "value_1", "extract_0_0"):
        symbols.setdefault(name, sympy.Symbol(name, integer=True))

    domains = {
        symbols["outer_index"]: (0, tasks - 1),
        symbols["value_0"]: (0, shape[0] - 1),
        symbols["value_1"]: (0, shape[1] - 1),
        symbols["extract_0_0"]: (0, stages - 1),
    }

    try:
        parsed = tuple(sympy.sympify(str(text), locals=symbols) for text in offsets)

        for axis, expression in enumerate(parsed):
            for value_axis in range(2):
                if sympy.diff(expression, symbols[f"value_{value_axis}"]) != int(
                    axis == value_axis
                ):
                    return None

        mask = sympy.sympify(str(template.get("mask", True)), locals=symbols)

        if not _always_true(mask, domains):
            return None

        source_shape = tuple(map(int, attrs["source_shape"]))

        if len(source_shape) != 2 or any(extent <= 0 for extent in source_shape):
            return None

        for expression, extent in zip(parsed, source_shape):
            bounds = _range(expression, domains)

            if bounds is None or bounds[0] < 0 or bounds[1] >= extent:
                return None
    except (TypeError, ValueError, KeyError, AttributeError, SyntaxError):
        return None

    def render(text):
        for old, new in (
            ("outer_index", "taskIdX"),
            ("extract_0_0", "nt_dot_stage"),
            ("value_0", "nt_dot_row"),
            ("value_1", "0"),
        ):
            text = re.sub(rf"\b{old}\b", new, str(text))
        return common.target_index_expr(target, text)

    return DotInput(
        extract.operands[0], render(offsets[0]), render(offsets[1]), tuple(strides)
    )


class _StagedStoreTarget:
    """Reuse the original typed epilogue while redirecting its proved store."""

    def __init__(self, target, output):
        self._target = target
        self._output = output

    def __getattr__(self, name):
        return getattr(self._target, name)

    def store(self, tensor, index, value, *, mask=None):
        if tensor != self._output:
            return self._target.store(tensor, index, value, mask=mask)

        assignment = f"nt_dot_output[nt_dot_lane] = {value};"

        return assignment if mask is None else f"if ({mask}) {{\n    {assignment}\n}}"


def _output(context, m, n, tasks):
    info = context.tensors[context.outputs[0]]
    dtype = common.normalize_dtype(info.dtype)

    if dtype not in {"float16", "float32"}:
        return None

    attrs = dict(info.attrs or {})

    try:
        rows, cols = map(int, attrs.get("source_shape", info.source_shape))
    except (ValueError, TypeError):
        return None

    if (
        rows <= 0
        or cols <= 0
        or cols * (2 if dtype == "float16" else 4) > 4294967295
        or rows % m
        or cols % n
        or tasks != (rows // m) * (cols // n)
    ):
        return None

    access = prove_dense_dot_access(
        attrs,
        shape=(m, n),
        tasks=tasks,
        stages=1,
        expected_offsets=(
            f"floor(outer_index / {cols // n}) * {m} + value_0",
            f"Mod(outer_index, {cols // n}) * {n} + value_1",
        ),
    )

    if access is None:
        return None
    return access, dtype, cols


def _vector_epilogue(context, loop, output, target, m, n, budget):
    """Treat the proved dot result as the input to the ordinary typed DAG."""
    from ninetoothed.backends.emitters import bangc_vector
    from ninetoothed.backends.emitters.bangc import _bangc_float_literal
    from ninetoothed.backends.emitters.vector_dag import plan_vector_dag

    def classify(operation, operands, dtype):
        if dtype not in {"float32", "float16"}:
            return None

        lowering = bangc_vector.classify(operation, operands, dtype)

        if lowering is None or lowering.mode not in {
            "binary",
            "scalar_binary",
            "unary",
            "cast",
            "square",
            "fill",
            "scalar",
        }:
            return None
        return lowering

    name = loop.results[0].name
    capacity = m * n
    dag = plan_vector_dag(
        SimpleNamespace(
            operations=context.operations,
            stores=context.stores,
            tensors={**context.tensors, name: SimpleNamespace(dtype="float32")},
        ),
        (name,),
        (context.outputs[0],),
        normalize_dtype=common.normalize_dtype,
        classify=classify,
        dtype_bytes={"float32": 4, "float16": 2},
    )

    if (
        dag is None
        or dag.bytes_per_element * capacity > budget
        or len(dag.inputs) != 1
        or dag.inputs[0].value.scalar
        or dag.stores[0].value.scalar
        or dag.stores[0].value.dtype != output[1]
    ):
        return None

    # Reduction and strict transcendental modes are excluded above.
    def unreachable_reduce(_operator):
        raise AssertionError("A dot epilogue cannot contain a reduction.")

    source = f"nt_dag_buf_{dag.inputs[0].buffer}"
    lines = [
        f"const int64_t nt_cnt = {capacity};",
        f"const uint32_t nt_aligned = {capacity};",
        *bangc_vector._buffers(dag, capacity),
        f"__memcpy({source}, nt_dot_acc, {n} * sizeof(float), NRAM2NRAM, {n} * sizeof(float), 64 * sizeof(float), {m - 1});",
    ]
    lines.extend(
        bangc_vector.render_steps(dag, target, _bangc_float_literal, unreachable_reduce)
    )

    return lines, f"nt_dag_buf_{dag.stores[0].buffer}"


def render_native_dot(
    context,
    target,
    *,
    dma=False,
    stage_output=False,
    vector_epilogue=False,
    right_transposed=False,
):
    """Return a primitive schedule; preserve the SSA store and its epilogue."""
    if right_transposed and not dma:
        return None

    if (
        len(context.stores) != 1
        or len(context.outputs) != 1
        or len(context.kernel.ssa.blocks) != 1
    ):
        return None

    plan = dot_accumulation_plan(context.kernel.ssa)

    if plan is None:
        return None

    loop, dot, store = plan.loop, plan.dot, plan.store

    if store.attrs.get("indices") or store.attrs.get("source"):
        return None

    allowed = {"tensor.extract", "arith.add", "scf.yield"}

    if any(
        op.opcode not in allowed | {"linalg.dot", "linalg.matmul"} or op.regions
        for op in loop.regions[0].operations
    ):
        return None

    top = context.kernel.ssa.blocks[0].operations

    for operation in top:
        if operation is loop or operation is store:
            continue

        if not pure_expression(operation) or operation.opcode.startswith("reduce."):
            return None

    if common.normalize_dtype(loop.results[0].type.dtype) != "float32":
        return None

    try:
        m, k = map(
            int, common.value_axes_from_types(dot.operands[0], context.value_types)
        )
        rk, n = map(
            int, common.value_axes_from_types(dot.operands[1], context.value_types)
        )
        axes = tuple(map(int, context.axes))
        tasks = math.prod(map(int, context.outer_axes))
    except (ValueError, TypeError):
        return None
    # These LT sizes and alignment have device proofs; broader shapes fall back.

    if (
        k != rk
        or (m, k, n) not in {(32, 32, 32), (64, 32, 64)}
        or axes != (m, n)
        or tasks <= 0
    ):
        return None

    if (m, k, n) == (64, 32, 64) and tuple(
        context.kernel.ssa.metadata.get("output_tile_fusion", ())
    ) != (2, 2):
        # The fused adapter also proves the changed fallback launch ownership.
        return None

    emit = EmitContext(
        target=target,
        kernel=context.kernel,
        program=context.kernel.ssa,
        operations=context.operations,
        value_types=context.value_types,
        lines=[],
        memo={},
        tensor_infos=context.tensors,
        output=context.outputs[0],
        output_axes=context.axes,
        index_expr="nt_dot_lane",
        inner_index_expr="nt_dot_lane",
        outer_index_expr="taskIdX",
        mask_expr=None,
        coordinate_exprs=(f"(nt_dot_lane / {n})", f"(nt_dot_lane % {n})"),
        bindings={},
        type_values={},
        type_initials={},
        temp_counter=[0],
        materialized={},
        parameter_names=frozenset(
            (*context.variables, *context.outputs, *context.shape_params)
        ),
    )
    lower = common.emit_loop_bound(loop.operands[0], emit)
    step = common.emit_loop_bound(loop.operands[2], emit)
    bound = context.operations.get(loop.operands[1])

    if bound is None:
        return None

    try:
        if bound.opcode == "arith.constant":
            stages = int(bound.attrs["value"])
        elif bound.opcode == "shape.dim" and not bound.attrs.get("source"):
            stages = int(
                common.value_axes_from_types(bound.operands[0], context.value_types)[
                    int(bound.attrs["dim"])
                ]
            )
        else:
            return None
    except (ValueError, KeyError, IndexError):
        return None

    if lower != "0" or step != "1" or not 0 < stages <= 4096:
        return None
    # The region must extract exactly its induction argument.

    induction = str(loop.attrs.get("induction"))

    for operand in dot.operands[:2]:
        producer = context.operations.get(operand)

        if producer is None or producer.operands[1:] != (induction,):
            return None

    left = _input(context, target, dot.operands[0], stages, tasks, (m, k))
    right = _input(context, target, dot.operands[1], stages, tasks, (k, n))

    if left is None or right is None:
        return None

    output = _output(context, m, n, tasks) if stage_output else None

    if stage_output and output is None:
        return None

    emit.lines.clear()
    # The installed LT primitive needs physical K/N alignment of 64 elements.
    # Zero padding belongs to target layout lowering, not the user's load mask.
    padded_k = padded_n = 64
    epilogue = None

    if vector_epilogue and output is not None:
        nram_bytes = (m * padded_k + 2 * padded_k * padded_n) * 2 + 2 * m * padded_n * 4
        epilogue = _vector_epilogue(
            context,
            loop,
            output,
            target,
            m,
            n,
            192 * 1024 - nram_bytes,
        )

    emit.memo[loop.results[0].name] = (
        f"nt_dot_acc[(nt_dot_lane / {n}) * {padded_n} + (nt_dot_lane % {n})]"
    )

    if stage_output:
        emit.target = _StagedStoreTarget(target, context.outputs[0])

    common.emit_operation(store, emit)
    lines = [
        f"__nram__ half nt_dot_left[{m * padded_k}];",
        f"__nram__ half nt_dot_right[{padded_k * padded_n}];",
        f"__nram__ half nt_dot_transpose[{padded_k * padded_n}];",
        f"__wram__ half nt_dot_weight[{padded_k * padded_n}];",
        f"__nram__ float nt_dot_part[{m * padded_n}];",
        f"__nram__ float nt_dot_acc[{m * padded_n}];",
        f"__bang_write_zero(nt_dot_acc, {m * padded_n});",
        f"for (int nt_dot_stage=0; nt_dot_stage<{stages}; ++nt_dot_stage) {{",
    ]

    if stage_output and epilogue is None:
        c_type = "half" if output[1] == "float16" else "float"
        lines.insert(0, f"__nram__ {c_type} nt_dot_output[{m * n}];")

    for operand, buffer, rows, cols, padded_rows, padded_cols in (
        (left, "nt_dot_left", m, k, m, padded_k),
        (right, "nt_dot_right", k, n, padded_k, padded_n),
    ):
        lines.append(f"    __bang_write_zero({buffer}, {padded_rows * padded_cols});")

        if dma:
            row = re.sub(r"\bnt_dot_row\b", "0", operand.row)
            stride = operand.strides[1 if operand is right and right_transposed else 0]
            address = f"{operand.name}+({row})*({operand.strides[0]})+({operand.column})*({operand.strides[1]})"
            transfer = f"    __memcpy({buffer}, {address}, {cols}*sizeof(half), GDRAM2NRAM, {padded_cols}*sizeof(half), ({stride})*sizeof(half), {rows - 1});"

            lines.append(transfer)
        else:
            lines += [
                f"    for (int nt_dot_row=0; nt_dot_row<{rows}; ++nt_dot_row) {{",
                f"        __memcpy({buffer}+nt_dot_row*{padded_cols}, {operand.name}+({operand.row})*({operand.strides[0]})+({operand.column}), {cols}*sizeof(half), GDRAM2NRAM);",
                "    }",
            ]

    weight_source = "nt_dot_right" if right_transposed else "nt_dot_transpose"

    if not right_transposed:
        lines.append(
            f"    __bang_transpose(nt_dot_transpose, nt_dot_right, {padded_k}, {padded_n});"
        )

    lines += [
        f"    __memcpy(nt_dot_weight, {weight_source}, {padded_k * padded_n}*sizeof(half), NRAM2WRAM);",
        f"    __bang_matmul(nt_dot_part, nt_dot_left, nt_dot_weight, {m}, {padded_k}, {padded_n});",
        f"    __bang_add(nt_dot_acc, nt_dot_acc, nt_dot_part, {m * padded_n});",
        "}",
    ]

    if epilogue is not None:
        lines.extend(epilogue[0])
    else:
        lines += [
            f"for (int nt_dot_lane=0; nt_dot_lane<{m * n}; ++nt_dot_lane) {{",
            common.indent_block("\n".join(emit.lines), "    "),
            "}",
        ]

    guard = f"({left.strides[1]}) == 1 && ({right.strides[0 if right_transposed else 1]}) == 1"

    if dma:
        row_strides = (left.strides[0], right.strides[1 if right_transposed else 0])
        guard += " && " + " && ".join(
            f"({stride}) > 0 && ({stride}) <= 1073741823" for stride in row_strides
        )

    if stage_output:
        access, dtype, cols = output
        base_row, base_column = access.base_offsets({"outer_index": "taskIdX"})
        name = context.outputs[0]
        c_type = "half" if dtype == "float16" else "float"
        output_buffer = epilogue[1] if epilogue is not None else "nt_dot_output"
        lines.append(
            f"__memcpy({name} + ({base_row}) * ({access.strides[0]}) + ({base_column}), {output_buffer}, {n} * sizeof({c_type}), NRAM2GDRAM, ({access.strides[0]}) * sizeof({c_type}), {n} * sizeof({c_type}), {m - 1});"
        )
        guard += (
            f" && ({access.strides[0]}) == {cols} && ({access.strides[1]}) == 1"
            f" && (reinterpret_cast<uintptr_t>({name}) % 32) == 0"
        )
    return NativeDotBody("\n".join(lines), str(tasks), guard)
