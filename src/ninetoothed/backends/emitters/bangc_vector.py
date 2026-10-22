"""BANG primitives consuming the shared typed vector def-use plan."""

from dataclasses import dataclass, replace

from ninetoothed.backends.emitters.analysis import shape_dimension_expression
from ninetoothed.backends.emitters.analysis import source_shape as _source_shape
from ninetoothed.backends.emitters.expressions import normalize_dtype
from ninetoothed.backends.emitters.patterns import canonical_row_view
from ninetoothed.backends.emitters.staging import dense_staging_memory_guard
from ninetoothed.backends.emitters.vector_dag import VectorLowering, plan_vector_dag
from ninetoothed.compiler.cse import pure_expression
from ninetoothed.compiler.math_approximations import bounded_polynomial
from ninetoothed.compiler.math_erf import bounded_erf
from ninetoothed.compiler.math_exponential import bounded_exponential
from ninetoothed.compiler.passes import program_facts
from ninetoothed.compiler.vector_resources import Allocation

_BYTES = {"float32": 4, "float16": 2, "int32": 4, "bool": 1}
_TYPES = {"float32": "float", "float16": "half", "int32": "int32_t", "bool": "bool"}
_BINARY = {
    "arith.add": "__bang_add",
    "arith.sub": "__bang_sub",
    "arith.mul": "__bang_mul",
    "arith.div": "__bang_div",
    "arith.maximum": "__bang_nan_maximum",
    "arith.minimum": "__bang_nan_minimum",
}
_SCALAR_BINARY = {
    "arith.add": "__bang_add_scalar",
    "arith.sub": "__bang_sub_scalar",
    "arith.mul": "__bang_mul_scalar",
    "arith.maximum": "__bang_nan_maximum_scalar",
}

_UNARY = {
    "math.abs": "__bang_abs",
    "math.floor": "__bang_floor",
    "math.sqrt": "__bang_sqrt",
    "math.rsqrt": "__bang_rsqrt",
    "math.log2": "__bang_log",
    "math.reciprocal": "__bang_recip",
    "arith.neg": "__bang_neg",
}
_STRICT_MATH = {
    "exp",
    "exp2",
    "sin",
    "cos",
    "tanh",
    "erf",
    "atan",
    "ceil",
    "floor",
    "round",
    "sigmoid",
    "log",
    "log2",
    "sqrt",
    "rsqrt",
    "abs",
    "reciprocal",
    "pow",
    "log1p",
    "expm1",
}
_SIGNS = {
    "arith.add": "+",
    "arith.sub": "-",
    "arith.mul": "*",
    "arith.div": "/",
    "arith.lt": "<",
    "arith.le": "<=",
    "arith.gt": ">",
    "arith.ge": ">=",
    "arith.eq": "==",
    "arith.ne": "!=",
}
_SIGNS.update(
    {
        "cmp." + code: sign
        for code, sign in (
            ("lt", "<"),
            ("le", "<="),
            ("gt", ">"),
            ("ge", ">="),
            ("eq", "=="),
            ("ne", "!="),
        )
    }
)
_COMPARE = {
    code: "__bang_"
    + {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}[sign]
    for code, sign in _SIGNS.items()
    if sign not in {"+", "-", "*", "/"}
}


def _storage_dtype(dtype):
    return "float32" if dtype == "bool" else dtype


def _opcode(operation):
    code = operation.opcode

    return "math." + code[5:] if code.startswith("call.") else code


def _pure_vector_expression(operation):
    if operation.opcode.startswith("call.") and operation.opcode[5:] in _STRICT_MATH:
        operation = replace(operation, opcode=_opcode(operation))
    return pure_expression(operation)


def classify(operation, operands, dtype):
    code = _opcode(operation)

    if (
        code == "tensor.view"
        and len(operands) == 1
        and operands[0].scalar
        and operands[0].dtype == dtype
        and operation.attrs.get("source") is False
        and operation.attrs.get("subscript") == "(:, None)"
        and tuple(map(str, operation.results[0].type.shape)) == ("1", "1")
    ):
        return VectorLowering("identity")

    if (
        dtype == "float32"
        and code != "select.where"
        and any(not value.scalar and value.dtype != "float32" for value in operands)
    ):
        return None

    if code.startswith("reduce.") and dtype == "float32":
        return VectorLowering("reduce", ("float32",))

    if (
        code in {"tensor.full", "tensor.zeros", "tensor.ones"}
        and dtype == "float32"
        and all(value.scalar for value in operands)
    ):
        if (code == "tensor.full" and len(operands) == 1) or (
            code != "tensor.full" and not operands
        ):
            return VectorLowering("fill")

    if code in {"arith.pow", "math.pow"} and dtype == "float32" and len(operands) == 2:
        if operands[1].literal == 2 and not operands[0].scalar:
            return VectorLowering("square")
        return VectorLowering("scalar")

    if (
        code
        in {
            "math.exp",
            "math.exp2",
            "math.sin",
            "math.cos",
            "math.tanh",
            "math.sigmoid",
            "math.erf",
        }
        and dtype == "float32"
        and len(operands) == 1
        and not operands[0].scalar
    ):
        return VectorLowering(
            "strict_" + code[5:], ("float32",) * (3 if code == "math.tanh" else 2)
        )

    if code in _BINARY and dtype == "float32" and len(operands) == 2:
        if any(
            value.dtype != "float32"
            and value.literal is None
            and value.dtype != "index"
            for value in operands
        ):
            return None

        if all(value.scalar for value in operands):
            return VectorLowering("scalar")

        if operands[0].literal == 1 and code == "arith.div" and not operands[1].scalar:
            return VectorLowering("reciprocal")

        if code in _SCALAR_BINARY and (
            operands[1].scalar
            or operands[0].scalar
            and code in {"arith.add", "arith.mul", "arith.maximum"}
        ):
            return VectorLowering("scalar_binary")
        return VectorLowering(
            "binary", ("float32",) if any(value.scalar for value in operands) else ()
        )

    if (
        code in {"math.floor", "math.ceil"}
        and dtype == "float32"
        and len(operands) == 1
        and not operands[0].scalar
    ):
        return VectorLowering("round", ("float32", "int32", "int32"))

    if code in _UNARY and dtype == "float32" and len(operands) == 1:
        return VectorLowering("scalar" if operands[0].scalar else "unary")

    if (
        code == "math.log"
        and dtype == "float32"
        and len(operands) == 1
        and not operands[0].scalar
    ):
        return VectorLowering("log", ("float32",))

    if code == "tensor.cast" and len(operands) == 1:
        if operands[0].dtype == dtype and dtype in _TYPES:
            return VectorLowering("identity")

        if operands[0].dtype == "float32" and dtype in {"float16", "int32"}:
            return VectorLowering("cast", ("float32",) if dtype == "int32" else ())
        return None

    if code in _SIGNS and dtype == "bool" and len(operands) == 2:
        if any(not value.scalar and value.dtype != "float32" for value in operands):
            return None
        return VectorLowering(
            "scalar" if all(value.scalar for value in operands) else "compare",
            ("float32",) if any(value.scalar for value in operands) else (),
        )

    if code == "select.where" and len(operands) == 3 and dtype == "float32":
        if any(not value.scalar and value.dtype != "float32" for value in operands[1:]):
            return None

        if operands[0].dtype != "bool":
            return None

        if operands[0].scalar:
            return VectorLowering("scalar")
        return VectorLowering(
            "where", ("float32", "float32", "int32", "int32", "int32")
        )

    if code.startswith("math.") and code[5:] in _STRICT_MATH and dtype == "float32":
        return VectorLowering("scalar")
    return None


def plan_flat(context, inputs, outputs):
    program = context.kernel.ssa

    if (
        len(program.blocks) != 1
        or any(
            operation.opcode != "mem.store" and not _pure_vector_expression(operation)
            for operation in program.blocks[0].operations
        )
        or any(
            set(store.attrs) - {"target", "target_shape"} for store in context.stores
        )
    ):
        return None

    scalars = {
        name: info.dtype
        for name, info in context.tensors.items()
        if info.ndim == 0 and name in context.variables
    }

    return plan_vector_dag(
        context,
        tuple(name for name, _ in inputs),
        tuple(name for name, _ in outputs),
        normalize_dtype=normalize_dtype,
        classify=classify,
        dtype_bytes=_BYTES,
        scalar_inputs=scalars,
        storage_dtype=_storage_dtype,
    )


@dataclass(frozen=True)
class RowVectorPlan:
    dag: object
    rows: str
    cols: str
    capacity: int
    guard: str
    input_offsets: dict[str, str]
    scalar_expressions: dict[str, str]
    output_widths: dict[str, str]


def _bind_uniform_shape_parameters(context, scalar_inputs, scalar_expressions):
    for name, operation in context.operations.items():
        expression = shape_dimension_expression(context, operation, default_dim=0)

        if expression is not None:
            scalar_inputs[name] = "index"
            scalar_expressions[name] = expression


def plan_row_vectors(context, *, budget=192 * 1024, allow_reductions=False):
    """Stage a pure 1×K domain using proved original broadcast input maps."""
    from ninetoothed.backends.emitters.row_input_layout import prove_row_input

    if (
        len(context.kernel.ssa.blocks) != 1
        or len(context.axes) != 2
        or tuple(map(str, context.axes[:1])) != ("1",)
        or len(context.outer_axes) != 2
        or str(context.outer_axes[1]) != "1"
        or not str(context.outer_axes[0]).isdecimal()
        or not str(context.axes[1]).isdecimal()
        or any(
            operation.opcode != "mem.store" and not _pure_vector_expression(operation)
            for operation in context.kernel.ssa.blocks[0].operations
        )
        or any(
            set(store.attrs) - {"target", "target_shape"} for store in context.stores
        )
    ):
        return None

    rows, cols = int(context.outer_axes[0]), int(context.axes[1])

    if rows <= 0 or cols <= 1:
        return None

    scalar_inputs = {}
    scalar_expressions = {}
    inputs = []
    offsets = {}
    layouts = {}

    for name in context.variables:
        info = context.tensors[name]

        if info.ndim == 0:
            scalar_inputs[name] = info.dtype
            scalar_expressions[name] = name
            continue

        layout = prove_row_input(context, name, rows, cols)

        if layout is None or normalize_dtype(info.dtype) != "float32":
            return None

        layouts[name] = layout
        offset = layout.base_offset("(int64_t)(taskIdX)")
        offsets[name] = offset

        if layout.mode == "scalar":
            scalar_inputs[name] = info.dtype
            scalar_expressions[name] = f"{name}[{offset}]"
        else:
            inputs.append(name)

    _bind_uniform_shape_parameters(context, scalar_inputs, scalar_expressions)

    dag = plan_vector_dag(
        context,
        tuple(inputs),
        context.outputs,
        normalize_dtype=normalize_dtype,
        classify=classify,
        dtype_bytes=_BYTES,
        scalar_inputs=scalar_inputs,
        reduction_axis=1 if allow_reductions else None,
        storage_dtype=_storage_dtype,
    )

    if dag is None or (
        not allow_reductions
        and any(step.node.lowering.mode == "reduce" for step in dag.steps)
    ):
        return None

    widths = {}
    guards = [layout.guard for layout in layouts.values()]

    for store in dag.stores:
        info = context.tensors[store.output]
        shape = _source_shape(info)

        if (
            len(shape) != 2
            or shape[0] != str(rows)
            or shape[1] not in ({"1", str(cols)} if allow_reductions else {str(cols)})
            or not canonical_row_view(info)
            or normalize_dtype(info.dtype) not in {"float32", "float16", "int32"}
            or normalize_dtype(info.dtype) != store.value.dtype
            or (store.value.scalar and not allow_reductions)
            or (store.value.scalar and store.value.dtype != "float32")
            or (shape[1] == "1" and not store.value.scalar)
        ):
            return None

        strides = info.source_strides or (info.attrs or {}).get("source_strides", ())

        if len(strides) != 2:
            return None

        guards.extend((f"({strides[0]}) == {shape[1]}", f"({strides[1]}) == 1"))
        widths[store.output] = shape[1]

    capacity = (cols + 127) // 128 * 128

    if capacity > 8192 or capacity * dag.bytes_per_element > budget:
        return None

    guards.append(dense_staging_memory_guard(context, tuple(layouts), tuple(widths)))

    return RowVectorPlan(
        dag,
        str(rows),
        str(cols),
        capacity,
        " && ".join(guards),
        offsets,
        scalar_expressions,
        widths,
    )


def plan_rows(context, *, budget=192 * 1024):
    """Require shared same-row/effect proof and exact canonical source views."""
    program = context.kernel.ssa
    reuse = program_facts(program, "reduction_reuse")

    if not context.cooperative_reduction_program or not reuse.get("results"):
        return None

    candidates = [
        name
        for name in context.variables
        if context.tensors[name].ndim == 2
        and len(_source_shape(context.tensors[name])) == 2
        and _source_shape(context.tensors[name])[-1] != "1"
    ]

    if not candidates:
        return None

    primary = candidates[0]
    rows, cols = _source_shape(context.tensors[primary])
    scalar_inputs = {}
    scalar_expressions = {}
    inputs = []
    offsets = {}

    for name in context.variables:
        info = context.tensors[name]

        if info.ndim == 0:
            scalar_inputs[name] = info.dtype
            scalar_expressions[name] = name
            continue

        shape = _source_shape(info)

        if info.ndim != 2 or len(shape) != 2:
            continue

        if shape[1] == "1":
            scalar_inputs[name] = info.dtype
            scalar_expressions[name] = f"{name}[(int64_t)(taskIdX)]"
        else:
            inputs.append(name)

        offsets[name] = f"(int64_t)(taskIdX) * ({cols})"

    _bind_uniform_shape_parameters(context, scalar_inputs, scalar_expressions)

    dag = plan_vector_dag(
        context,
        tuple(inputs),
        context.outputs,
        normalize_dtype=normalize_dtype,
        classify=classify,
        dtype_bytes=_BYTES,
        scalar_inputs=scalar_inputs,
        reduction_axis=1,
        storage_dtype=_storage_dtype,
    )

    if dag is None or not any(
        step.node.lowering.mode == "reduce" for step in dag.steps
    ):
        return None

    widths = {}
    guarded_names = []

    for input_ in dag.inputs:
        name = input_.value.name
        info = context.tensors.get(name)

        if info is None or info.ndim == 0:
            continue

        shape = _source_shape(info)

        if (
            normalize_dtype(info.dtype) != "float32"
            or not canonical_row_view(info)
            or shape != (rows, "1" if input_.value.scalar else cols)
        ):
            return None

        guarded_names.append(name)

    for store in dag.stores:
        info = context.tensors[store.output]
        shape = _source_shape(info)

        if (
            normalize_dtype(info.dtype) not in {"float32", "float16", "int32"}
            or normalize_dtype(info.dtype) != store.value.dtype
            or not canonical_row_view(info)
            or len(shape) != 2
            or shape[0] != rows
            or shape[1] not in {"1", cols}
        ):
            return None

        if shape[1] == "1" and not store.value.scalar:
            return None

        if store.value.scalar and store.value.dtype != "float32":
            return None

        widths[store.output] = shape[1]
        guarded_names.append(store.output)

    limit = min(8192, budget // max(dag.bytes_per_element, 1) // 128 * 128)

    if limit < 128:
        return None

    capacity = min(limit, (int(cols) + 127) // 128 * 128) if cols.isdecimal() else limit

    if cols.isdecimal() and int(cols) > capacity:
        return None

    predicates = [f"({cols}) > 0", f"({cols}) <= {capacity}"]

    for name in guarded_names:
        info = context.tensors[name]
        strides = info.source_strides or (info.attrs or {}).get("source_strides", ())
        shape = _source_shape(info)

        if strides:
            if len(strides) != 2:
                return None

            predicates.extend(
                (f"({strides[1]}) == 1", f"({strides[0]}) == ({shape[1]})")
            )

    predicates.append(
        dense_staging_memory_guard(
            context,
            tuple(
                item.value.name
                for item in dag.inputs
                if item.value.name in context.tensors
            ),
            tuple(widths),
        )
    )

    return RowVectorPlan(
        dag,
        rows,
        cols,
        capacity,
        " && ".join(predicates),
        offsets,
        scalar_expressions,
        widths,
    )


def _scalar_name(value, target):
    return "nt_dag_scalar_" + target.symbol(value.name)


def _operand(value, buffer, target, literal, lane="nt_j"):
    if value.literal is not None:
        return literal(float(value.literal))

    if value.scalar:
        return _scalar_name(value, target)
    return f"nt_dag_buf_{buffer}[{lane}]"


def _expression(node, args, target):
    code = _opcode(node.operation)

    if code == "tensor.view" and node.lowering.mode == "identity":
        return args[0]

    if code in _SIGNS:
        return f"(({args[0]}) {_SIGNS[code]} ({args[1]}))"

    if code in {"arith.maximum", "arith.minimum"}:
        return (
            f"nt_{'max' if code.endswith('maximum') else 'min'}({args[0]}, {args[1]})"
        )

    if code == "arith.neg":
        return f"-({args[0]})"

    if code == "select.where":
        return target.where(*args)

    if code == "tensor.cast":
        return target.cast(node.result.dtype, args[0])
    return target.call(code.split(".", 1)[1], args)


def _polynomial(destination, variable, coefficients, literal, *, use_memory_pair=False):
    """Compute50 memory fusion retains separately rounded multiply and add."""
    lines = [
        f"        __bang_write_value({destination}, nt_aligned, {literal(coefficients[0])});"
    ]

    for coefficient in coefficients[1:]:
        if use_memory_pair:
            lines.append(
                f"        ::__bang_fusion(::FUSION_FMA, {destination}, {destination}, {variable}, {literal(coefficient)}, nt_aligned, nt_aligned);"
            )
            continue

        lines.extend(
            (
                f"        __bang_mul(nt_dag_scratch_float32_0, {destination}, {variable}, nt_aligned);",
                f"        __bang_add_scalar({destination}, nt_dag_scratch_float32_0, {literal(coefficient)}, nt_aligned);",
            )
        )
    return lines


def _render_strict_math(step, target, literal, halving_reduce):
    """Polynomial on a proven bounded domain; original helper for outliers.

    The vector result is never trusted outside that domain. A chunk max-abs
    reduction avoids per-element scalar checks for ordinary inputs. Nonfinite
    or larger values trigger correction of just those lanes using the original
    full-domain scalar primitive, with unchanged precision requirements.
    """
    node = step.node
    mode = node.lowering.mode[7:]
    source = f"nt_dag_buf_{step.operand_buffers[0]}"
    destination = f"nt_dag_buf_{step.destination}"
    recipe = bounded_polynomial(f"math.{mode}")
    exponential = bounded_exponential(f"math.{mode}")
    erf = bounded_erf(f"math.{mode}")
    use_memory_pair = getattr(target, "use_polynomial_memory_pair", False)
    bound = (
        recipe.domain.upper
        if recipe is not None
        else exponential.domain.upper
        if exponential is not None
        else erf.domain.upper
        if erf is not None
        else 2.25
        if mode == "tanh"
        else 4.0
    )
    domain_lines = [
        "        float* nt_wrk = nt_dag_scratch_float32_0;",
        f"        __bang_abs(nt_wrk, {source}, nt_aligned);",
        "        for (int64_t nt_j = nt_cnt; nt_j < nt_aligned; ++nt_j) { nt_wrk[nt_j] = -INFINITY; }",
        "        int64_t nt_len = nt_aligned;",
    ]
    domain_lines.extend("    " + line for line in halving_reduce("max"))
    domain_lines.extend(
        (
            "        float nt_domain = -INFINITY;",
            "        for (int64_t nt_j = 0; nt_j < nt_len; ++nt_j) { nt_domain = nt_max(nt_domain, nt_wrk[nt_j]); }",
        )
    )
    lines = ["    {"]

    if getattr(target, "use_bounded_domain_pool", False):
        # The domain is used only by <= bound decisions below. Each predicate
        # is an exact F32 integer 0/1, with a sum <= 16384 < 2**24. The SDK
        # C16 sumpool and the exact sum of its 16 integer counts were verified
        # at 128-aligned counts up to 16384; the count merge is exact as well.
        # Scratch1 is dead until the original scale/square operation below;
        # the source remains a separate DAG buffer for scalar outlier repair.
        lines.extend(
            (
                "        float nt_domain;",
                "        if (nt_aligned > 0 && nt_aligned <= 16384 && nt_aligned % 128 == 0 && nt_cnt >= 0 && nt_cnt <= nt_aligned) {",
            )
        )
        bounds = (bound / 2, bound) if exponential is not None else (bound,)

        for index, decision_bound in enumerate(bounds):
            if index:
                lines.append(
                    f"            if (!(nt_domain <= {literal(bounds[0])})) {{"
                )

            lines.extend(
                (
                    f"            __bang_abs(nt_dag_scratch_float32_0, {source}, nt_aligned);",
                    f"            ::__bang_le_scalar(nt_dag_scratch_float32_0, nt_dag_scratch_float32_0, {literal(decision_bound)}, (uint32_t)nt_aligned);",
                    "            for (int64_t nt_j = nt_cnt; nt_j < nt_aligned; ++nt_j) { nt_dag_scratch_float32_0[nt_j] = 0.0f; }",
                    "            ::__bang_sumpool(nt_dag_scratch_float32_1, nt_dag_scratch_float32_0, 16, 1, (uint32_t)(nt_aligned / 16), 1, (uint32_t)(nt_aligned / 16), (uint32_t)(nt_aligned / 16), 1);",
                    "            float nt_domain_count = 0.0f;",
                    "            for (int nt_c = 0; nt_c < 16; ++nt_c) { nt_domain_count += nt_dag_scratch_float32_1[nt_c]; }",
                    f"            nt_domain = nt_domain_count == (float)nt_cnt ? {literal(decision_bound)} : INFINITY;",
                )
            )

            if index:
                lines.append("            }")

        lines.append("        } else {")
        lines.extend(
            "    " + line.replace("float nt_domain =", "nt_domain =")
            for line in domain_lines
        )
        lines.append("        }")
    else:
        lines.extend(domain_lines)

    if mode in {"sin", "cos"}:
        lines.append(
            f"        __bang_mul(nt_dag_scratch_float32_1, {source}, {source}, nt_aligned);"
        )
        lines.extend(
            _polynomial(
                destination,
                "nt_dag_scratch_float32_1",
                recipe.coefficients,
                literal,
                use_memory_pair=use_memory_pair,
            )
        )

        if recipe.multiply_input:
            lines.extend(
                (
                    f"        __bang_mul(nt_dag_scratch_float32_0, {destination}, {source}, nt_aligned);",
                    f"        __memcpy({destination}, nt_dag_scratch_float32_0, nt_aligned * sizeof(float), NRAM2NRAM);",
                )
            )
    elif mode == "erf":
        lines.extend(
            (
                f"        __bang_mul(nt_dag_scratch_float32_0, {source}, {source}, nt_aligned);",
                f"        __bang_mul_scalar(nt_dag_scratch_float32_1, nt_dag_scratch_float32_0, {literal(erf.variable_scale)}, nt_aligned);",
                f"        __bang_add_scalar(nt_dag_scratch_float32_0, nt_dag_scratch_float32_1, {literal(erf.variable_shift)}, nt_aligned);",
                "        __memcpy(nt_dag_scratch_float32_1, nt_dag_scratch_float32_0, nt_aligned * sizeof(float), NRAM2NRAM);",
            )
        )
        lines.extend(
            _polynomial(
                destination,
                "nt_dag_scratch_float32_1",
                erf.coefficients,
                literal,
                use_memory_pair=use_memory_pair,
            )
        )
        lines.extend(
            (
                f"        __bang_mul(nt_dag_scratch_float32_0, {destination}, {source}, nt_aligned);",
                f"        __memcpy({destination}, nt_dag_scratch_float32_0, nt_aligned * sizeof(float), NRAM2NRAM);",
            )
        )
    elif mode == "tanh":
        lines.append(
            f"        __bang_mul(nt_dag_scratch_float32_1, {source}, {source}, nt_aligned);"
        )
        lines.extend(
            _polynomial(
                destination,
                "nt_dag_scratch_float32_1",
                [1.0, 378.0, 17325.0, 135135.0],
                literal,
                use_memory_pair=use_memory_pair,
            )
        )
        lines.append(
            f"        __bang_mul(nt_dag_scratch_float32_2, {destination}, {source}, nt_aligned);"
        )
        lines.extend(
            _polynomial(
                destination,
                "nt_dag_scratch_float32_1",
                [28.0, 3150.0, 62370.0, 135135.0],
                literal,
                use_memory_pair=use_memory_pair,
            )
        )
        lines.extend(
            (
                f"        __bang_div(nt_dag_scratch_float32_0, nt_dag_scratch_float32_2, {destination}, nt_aligned);",
                f"        __memcpy({destination}, nt_dag_scratch_float32_0, nt_aligned * sizeof(float), NRAM2NRAM);",
            )
        )
    else:
        scale = exponential.input_scale
        lines.extend(
            (
                f"        bool nt_math_narrow = nt_domain <= {literal(exponential.domain.upper / 2)};",
                f"        float nt_math_scale = nt_math_narrow ? {literal(scale * 2)} : {literal(scale)};",
                f"        __bang_mul_scalar(nt_dag_scratch_float32_1, {source}, nt_math_scale, nt_aligned);",
            )
        )
        lines.extend(
            _polynomial(
                destination,
                "nt_dag_scratch_float32_1",
                exponential.coefficients,
                literal,
                use_memory_pair=use_memory_pair,
            )
        )

        for index in range(exponential.squarings):
            if index == exponential.squarings - 1:
                lines.append("        if (!nt_math_narrow) {")

            lines.append(
                f"        __bang_mul({destination}, {destination}, {destination}, nt_aligned);"
            )

            if index == exponential.squarings - 1:
                lines.append("        }")

        if exponential.sigmoid:
            lines.extend(
                (
                    f"        __bang_add_scalar(nt_dag_scratch_float32_0, {destination}, 1.0f, nt_aligned);",
                    "        __bang_write_value(nt_dag_scratch_float32_1, nt_aligned, 1.0f);",
                    f"        __bang_div({destination}, nt_dag_scratch_float32_1, nt_dag_scratch_float32_0, nt_aligned);",
                )
            )

    scalar_expression = _expression(node, [f"{source}[nt_j]"], target)
    limit = literal(bound)
    lines.extend(
        (
            f"        if (!(nt_domain <= {limit})) {{",
            "            for (int64_t nt_j = 0; nt_j < nt_cnt; ++nt_j) {",
            f"                float nt_value = {source}[nt_j];",
            f"                if (!(nt_value >= -{limit} && nt_value <= {limit})) {{ {destination}[nt_j] = {scalar_expression}; }}",
            "            }",
            "        }",
            "    }",
        )
    )

    return lines


def render_int32_cast(source, destination, scratch, halving_reduce):
    """Keep the defined invalid-value result, avoiding scalar scans for safe chunks."""
    lines = [
        f"    __bang_float2int32_tz({destination}, {source}, nt_aligned, 0);",
        "    {",
        f"        float* nt_wrk = {scratch};",
        f"        __bang_abs(nt_wrk, {source}, nt_aligned);",
        "        for (int64_t nt_j = nt_cnt; nt_j < nt_aligned; ++nt_j) { nt_wrk[nt_j] = -INFINITY; }",
        "        int64_t nt_len = nt_aligned;",
    ]
    lines.extend("    " + line for line in halving_reduce("max"))
    lines.extend(
        (
            "        float nt_cast_domain = -INFINITY;",
            "        for (int64_t nt_j = 0; nt_j < nt_len; ++nt_j) { nt_cast_domain = nt_max(nt_cast_domain, nt_wrk[nt_j]); }",
            "        if (!(nt_cast_domain < 2147483648.0f)) {",
            "            for (int64_t nt_j = 0; nt_j < nt_cnt; ++nt_j) {",
            f"                float nt_value = {source}[nt_j];",
            "                if (!(nt_value >= -2147483648.0f && nt_value < 2147483648.0f)) {",
            f"                    {destination}[nt_j] = (-2147483647 - 1);",
            "                }",
            "            }",
            "        }",
            "    }",
        )
    )

    return lines


def render_steps(dag, target, literal, halving_reduce):
    lines = []

    for step in dag.steps:
        node = step.node
        mode = node.lowering.mode
        args = [
            _operand(value, buffer, target, literal)
            for value, buffer in zip(node.operands, step.operand_buffers)
        ]
        dest = f"nt_dag_buf_{step.destination}"
        scalar = _scalar_name(node.result, target)

        if mode.startswith("strict_"):
            lines.extend(_render_strict_math(step, target, literal, halving_reduce))
        elif mode == "reduce":
            operator = node.operation.opcode.split(".")[1]
            neutral = (
                "0.0f"
                if operator == "sum"
                else "-INFINITY"
                if operator == "max"
                else "INFINITY"
            )
            source = f"nt_dag_buf_{step.operand_buffers[0]}"
            lines.extend(
                (
                    f"    float {scalar};",
                    "    {",
                    "        float* nt_wrk = nt_dag_scratch_float32_0;",
                    f"        __memcpy(nt_wrk, {source}, (uint32_t)nt_cnt * sizeof(float), NRAM2NRAM);",
                    f"        for (int64_t nt_j = nt_cnt; nt_j < nt_aligned; ++nt_j) {{ nt_wrk[nt_j] = {neutral}; }}",
                    "        int64_t nt_len = nt_aligned;",
                )
            )

            lines.extend("    " + line for line in halving_reduce(operator))

            lines.append(f"        float nt_result = {neutral};")
            update = (
                "nt_result + nt_wrk[nt_j]"
                if operator == "sum"
                else f"nt_{operator}(nt_result, nt_wrk[nt_j])"
            )
            lines.extend(
                (
                    f"        for (int64_t nt_j = 0; nt_j < nt_len; ++nt_j) {{ nt_result = {update}; }}",
                    f"        {scalar} = nt_result;",
                    "    }",
                )
            )
        elif node.result.scalar:
            lines.append(
                f"    {_TYPES[node.result.dtype]} {scalar} = {_expression(node, args, target)};"
            )
        elif mode == "identity":
            lines.append(
                f"    __memcpy({dest}, nt_dag_buf_{step.operand_buffers[0]}, nt_aligned * sizeof({_TYPES[node.result.dtype]}), NRAM2NRAM);"
            )
        elif mode == "scalar_binary":
            index = 0 if node.operands[1].scalar else 1
            lines.append(
                f"    {_SCALAR_BINARY[_opcode(node.operation)]}({dest}, nt_dag_buf_{step.operand_buffers[index]}, {args[1 - index]}, nt_aligned);"
            )
        elif mode == "reciprocal":
            lines.append(
                f"    __bang_recip({dest}, nt_dag_buf_{step.operand_buffers[1]}, nt_aligned);"
            )
        elif mode == "round":
            source = f"nt_dag_buf_{step.operand_buffers[0]}"

            if _opcode(node.operation) == "math.ceil":
                lines.extend(
                    (
                        f"    __bang_neg(nt_dag_scratch_float32_0, {source}, nt_aligned);",
                        f"    __bang_floor({dest}, nt_dag_scratch_float32_0, nt_aligned);",
                        f"    __bang_neg(nt_dag_scratch_float32_0, {dest}, nt_aligned);",
                        f"    __memcpy({dest}, nt_dag_scratch_float32_0, nt_aligned * sizeof(float), NRAM2NRAM);",
                    )
                )
            else:
                lines.append(f"    __bang_floor({dest}, {source}, nt_aligned);")

            lines.extend(
                (
                    f"    __bang_eq_scalar(nt_dag_scratch_float32_0, {dest}, 0.0f, nt_aligned);",
                    "    __bang_float2int32_tz(nt_dag_scratch_int32_0, nt_dag_scratch_float32_0, nt_aligned, 0);",
                    "    __bang_mul_scalar(nt_dag_scratch_int32_0, nt_dag_scratch_int32_0, (-2147483647 - 1), nt_aligned);",
                    f"    __bang_bxor(nt_dag_scratch_int32_1, (int32_t*){source}, (int32_t*){dest}, nt_aligned);",
                    "    __bang_band(nt_dag_scratch_int32_1, nt_dag_scratch_int32_1, nt_dag_scratch_int32_0, nt_aligned);",
                    f"    __bang_bxor((int32_t*){dest}, (int32_t*){dest}, nt_dag_scratch_int32_1, nt_aligned);",
                )
            )
        elif mode in {"binary", "compare"}:
            buffers = []

            for value, buffer, expr in zip(node.operands, step.operand_buffers, args):
                if value.scalar:
                    lines.append(
                        f"    __bang_write_value(nt_dag_scratch_float32_0, nt_aligned, (float)({expr}));"
                    )
                    buffers.append("nt_dag_scratch_float32_0")
                else:
                    buffers.append(f"nt_dag_buf_{buffer}")

            code = _opcode(node.operation)

            if code == "arith.minimum":
                buffers.reverse()  # Vendor ties pick lhs; scalar nt_min picks rhs.

            function = _BINARY[code] if mode == "binary" else _COMPARE[code]
            lines.append(
                f"    {function}({dest}, {buffers[0]}, {buffers[1]}, nt_aligned);"
            )
        elif mode == "where":
            branches = []

            for index, (value, buffer, expr) in enumerate(
                zip(node.operands[1:], step.operand_buffers[1:], args[1:])
            ):
                if value.scalar:
                    branch = f"nt_dag_scratch_float32_{index}"
                    lines.append(
                        f"    __bang_write_value({branch}, nt_aligned, (float)({expr}));"
                    )
                else:
                    branch = f"nt_dag_buf_{buffer}"

                branches.append(branch)

            mask = f"nt_dag_buf_{step.operand_buffers[0]}"
            lines.extend(
                (
                    f"    __bang_float2int32_tz(nt_dag_scratch_int32_0, {mask}, nt_aligned, 0);",
                    "    __bang_mul_scalar(nt_dag_scratch_int32_1, nt_dag_scratch_int32_0, -1, nt_aligned);",
                    f"    __bang_bxor(nt_dag_scratch_int32_2, (int32_t*){branches[0]}, (int32_t*){branches[1]}, nt_aligned);",
                    "    __bang_band(nt_dag_scratch_int32_0, nt_dag_scratch_int32_2, nt_dag_scratch_int32_1, nt_aligned);",
                    f"    __bang_bxor((int32_t*){dest}, nt_dag_scratch_int32_0, (int32_t*){branches[1]}, nt_aligned);",
                )
            )
        elif mode == "unary":
            lines.append(
                f"    {_UNARY[_opcode(node.operation)]}({dest}, nt_dag_buf_{step.operand_buffers[0]}, nt_aligned);"
            )
        elif mode == "square":
            source = f"nt_dag_buf_{step.operand_buffers[0]}"
            lines.append(f"    __bang_mul({dest}, {source}, {source}, nt_aligned);")
        elif mode == "fill":
            fill = (
                args[0]
                if args
                else "1.0f"
                if node.operation.opcode == "tensor.ones"
                else "0.0f"
            )
            lines.append(
                f"    __bang_write_value({dest}, nt_aligned, (float)({fill}));"
            )
        elif mode == "log":
            lines.extend(
                (
                    f"    __bang_log(nt_dag_scratch_float32_0, nt_dag_buf_{step.operand_buffers[0]}, nt_aligned);",
                    f"    __bang_mul_scalar({dest}, nt_dag_scratch_float32_0, 0.6931471805599453f, nt_aligned);",
                )
            )
        elif mode == "cast":
            source = f"nt_dag_buf_{step.operand_buffers[0]}"

            if node.result.dtype == "float16":
                lines.append(f"    __bang_float2half({dest}, {source}, nt_aligned);")
            else:
                lines.extend(
                    render_int32_cast(
                        source, dest, "nt_dag_scratch_float32_0", halving_reduce
                    )
                )
        else:
            lines.extend(
                (
                    "    for (int64_t nt_j = 0; nt_j < nt_cnt; ++nt_j) {",
                    f"        {dest}[nt_j] = {_expression(node, args, target)};",
                    "    }",
                )
            )
    return lines


def _buffers(dag, capacity):
    lines = [
        f"    __nram__ {_TYPES[dtype]} nt_dag_buf_{index}[{capacity}];"
        for index, dtype in enumerate(dag.buffer_dtypes)
    ]
    counters = {}

    for dtype in dag.scratch_dtypes:
        index = counters.get(dtype, 0)
        counters[dtype] = index + 1
        lines.append(
            f"    __nram__ {_TYPES[dtype]} nt_dag_scratch_{dtype}_{index}[{capacity}];"
        )
    return lines


def nram_allocations(dag):
    """Account for the same individually aligned arrays rendered by _buffers."""
    allocations = [
        Allocation(f"nt_dag_buf_{index}", _BYTES[dtype])
        for index, dtype in enumerate(dag.buffer_dtypes)
    ]
    counters = {}

    for dtype in dag.scratch_dtypes:
        index = counters.get(dtype, 0)
        counters[dtype] = index + 1
        allocations.append(Allocation(f"nt_dag_scratch_{dtype}_{index}", _BYTES[dtype]))
    return tuple(allocations)


def render_flat(dag, extent, chunk, target, literal, halving_reduce):
    lines = [
        f"    const int64_t nt_chunk = {chunk};",
        "    const int64_t nt_base = (int64_t)(taskIdX) * nt_chunk;",
        f"    if (nt_base >= ({extent})) {{ return; }}",
        f"    int64_t nt_cnt = ({extent}) - nt_base;",
        "    if (nt_cnt > nt_chunk) { nt_cnt = nt_chunk; }",
        "    const uint32_t nt_aligned = (uint32_t)((nt_cnt + 127) / 128 * 128);",
        *_buffers(dag, chunk),
    ]

    for input_ in dag.inputs:
        if input_.value.scalar:
            lines.append(
                f"    {_TYPES[input_.value.dtype]} {_scalar_name(input_.value, target)} = {input_.value.name};"
            )
        else:
            lines.append(
                f"    __memcpy(nt_dag_buf_{input_.buffer}, {input_.value.name} + nt_base, (uint32_t)nt_cnt * sizeof({_TYPES[input_.value.dtype]}), GDRAM2NRAM);"
            )

    lines.extend(render_steps(dag, target, literal, halving_reduce))

    for store in dag.stores:
        if store.value.scalar:
            return None

        lines.append(
            f"    __memcpy({store.output} + nt_base, nt_dag_buf_{store.buffer}, (uint32_t)nt_cnt * sizeof({_TYPES[store.value.dtype]}), NRAM2GDRAM);"
        )
    return "\n".join(lines)


def render_rows(plan, target, literal, halving_reduce):
    dag = plan.dag
    lines = [
        f"    const int64_t nt_cnt = ({plan.cols});",
        "    const uint32_t nt_aligned = (uint32_t)((nt_cnt + 127) / 128 * 128);",
        *_buffers(dag, plan.capacity),
    ]

    for input_ in dag.inputs:
        value = input_.value

        if value.scalar:
            type_ = "int64_t" if value.dtype == "index" else _TYPES[value.dtype]
            lines.append(
                f"    {type_} {_scalar_name(value, target)} = {plan.scalar_expressions[value.name]};"
            )
        else:
            lines.append(
                f"    __memcpy(nt_dag_buf_{input_.buffer}, {value.name} + ({plan.input_offsets[value.name]}), (uint32_t)nt_cnt * sizeof(float), GDRAM2NRAM);"
            )

    lines.extend(render_steps(dag, target, literal, halving_reduce))

    for store in dag.stores:
        if store.value.scalar:
            scalar = _scalar_name(store.value, target)

            if plan.output_widths[store.output] == "1":
                lines.append(f"    {store.output}[(int64_t)(taskIdX)] = {scalar};")
            else:
                lines.extend(
                    (
                        f"    __bang_write_value(nt_dag_scratch_float32_0, nt_aligned, {scalar});",
                        f"    __memcpy({store.output} + (int64_t)(taskIdX) * nt_cnt, nt_dag_scratch_float32_0, (uint32_t)nt_cnt * sizeof(float), NRAM2GDRAM);",
                    )
                )
        else:
            lines.append(
                f"    __memcpy({store.output} + (int64_t)(taskIdX) * nt_cnt, nt_dag_buf_{store.buffer}, (uint32_t)nt_cnt * sizeof({_TYPES[store.value.dtype]}), NRAM2GDRAM);"
            )
    return "\n".join(lines)
