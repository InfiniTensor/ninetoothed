"""Ascend UB execution of the shared typed row-expression DAG.

The shared planner owns def-use order, uniform values and buffer lifetimes.
This target adapter proves complete row views, budgets UB storage, and maps
primitive SSA operations onto checked vector intrinsics or scalar UB operations.
"""

import re
from dataclasses import dataclass, field, replace

from ninetoothed.backends.ascendc_resources import (
    ASCENDC_UB_BYTES,
    configure_validity_summary,
)
from ninetoothed.backends.emitters import patterns
from ninetoothed.backends.emitters.analysis import shape_dimension_expression
from ninetoothed.backends.emitters.analysis import source_shape as _source_shape
from ninetoothed.backends.emitters.ascendc_predicates import (
    PREDICATE_OPCODES,
    classify_predicate,
    render_predicate,
    scalar_predicate_expression,
)
from ninetoothed.backends.emitters.ascendc_primitives import (
    numeric_lower_bound,
    render_division,
    render_extrema,
    render_nan_propagation,
    render_refinement,
)
from ninetoothed.backends.emitters.row_input_layout import prove_row_input
from ninetoothed.backends.emitters.ssa import normalize_dtype
from ninetoothed.backends.emitters.staging import dense_staging_memory_guard
from ninetoothed.backends.emitters.vector_dag import VectorLowering, plan_vector_dag
from ninetoothed.compiler.math_approximations import bounded_polynomial
from ninetoothed.compiler.passes import program_facts

_BINARY = {
    "arith.add": ("Add", "+"),
    "arith.sub": ("Sub", "-"),
    "arith.mul": ("Mul", "*"),
    "arith.div": ("Div", "/"),
}
_UNARY = {
    "math.exp": ("Exp", "nt_exp"),
    "math.log": ("Ln", "nt_log"),
    "math.sqrt": ("Sqrt", "sqrt"),
    "math.abs": ("Abs", "nt_fabs"),
    "math.rsqrt": ("RsqrtStrict", "nt_rsqrt"),
    "math.reciprocal": ("ReciprocalStrict", None),
}
_SCALAR_MATH = {
    "math.sin": "nt_sin",
    "math.cos": "nt_cos",
    "math.tanh": "nt_tanh",
    "math.erf": "nt_erf",
    "math.sigmoid": "nt_sigmoid",
    "math.log1p": "nt_log1p",
    "math.expm1": "nt_expm1",
}
_REDUCE = {
    "reduce.sum": "ReduceSum",
    "reduce.max": "ReduceMax",
    "reduce.min": "ReduceMin",
}
_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "int32": 4,
    "bool": 1,
    "index": 8,
    "int64": 8,
    "uint8": 1,
}
_CPP_TYPES = {"float32": "float", "float16": "half", "int32": "int32_t"}


def _opcode(operation):
    opcode = operation.opcode

    return "math." + opcode[5:] if opcode.startswith("call.") else opcode


def _classify(operation, operands, dtype):
    opcode = _opcode(operation)

    if opcode in PREDICATE_OPCODES:
        return classify_predicate(operation, operands, dtype)

    if opcode == "tensor.cast" and len(operands) == 1:
        if operands[0].dtype == dtype and dtype in _CPP_TYPES:
            return VectorLowering("identity")

        if operands[0].dtype == "float32" and dtype in {"float16", "int32"}:
            return VectorLowering(
                "cast", ("uint8", "uint8") if dtype == "int32" else ()
            )
        return None

    if dtype != "float32":
        return None

    if any(not operand.scalar and operand.dtype != "float32" for operand in operands):
        return None

    if opcode in _REDUCE:
        return VectorLowering("reduce", ("float32", "float32", "uint8", "uint8"))

    if opcode in _BINARY or opcode in _UNARY:
        scratch = (
            ("float32", "float32", "uint8", "uint8")
            if opcode in {"arith.div", "math.rsqrt", "math.reciprocal"}
            else ()
        )

        return VectorLowering("vector", scratch)

    if opcode == "arith.neg":
        return VectorLowering("vector")

    if opcode in {"arith.maximum", "arith.minimum"}:
        return VectorLowering("extrema", ("uint8", "uint8", "uint8"))

    if bounded_polynomial(opcode, dtype=dtype) is not None:
        return VectorLowering(
            "bounded_polynomial", ("float32", "uint8", "uint8", "uint8")
        )

    if (
        opcode == "arith.pow"
        and len(operands) == 2
        and operands[1].scalar
        and operands[1].literal == 2
    ):
        return VectorLowering("square")

    if opcode == "math.tanh" and len(operands) == 1 and not operands[0].scalar:
        return VectorLowering("bounded_tanh", ("float32", "float32", "uint8", "uint8"))

    if opcode in _SCALAR_MATH or opcode in {
        "arith.maximum",
        "arith.minimum",
        "arith.pow",
    }:
        return VectorLowering("scalar")

    if opcode == "tensor.view" and all(operand.scalar for operand in operands):
        if all(str(size) == "1" for size in operation.results[0].type.shape):
            return VectorLowering("identity")
    return None


@dataclass(frozen=True)
class AscendRowDAG:
    plan: object
    rows: str
    cols: int
    collapsed: tuple[str, ...]
    scalar_bindings: tuple[tuple[str, str], ...]
    ub_bytes: int = ASCENDC_UB_BYTES
    input_layouts: tuple[tuple[str, object], ...] = ()
    native_division: bool = False
    validity_summary: bool = False
    numeric_bounds: object = field(default_factory=dict, repr=False)


def _scalar_parameter_types(context):
    """Auxiliary ABI values include typed user scalars as well as indices."""
    result = {}

    for name in context.shape_params:
        info = context.tensors.get(name)

        if info is None:
            result[name] = "index"
            continue

        if info.dtype is None:
            return None

        dtype = normalize_dtype(info.dtype)

        if info.ndim != 0 or dtype not in _DTYPE_BYTES:
            return None

        result[name] = dtype

    return result


def match_flat_dag(context, *, ub_bytes=ASCENDC_UB_BYTES):
    """Pack pure canonical pointwise domains into complete UB packets."""
    from ninetoothed.backends.emitters.staging import canonical_flat_view

    if len(context.kernel.ssa.blocks) != 1 or not context.outputs:
        return None

    operations = context.kernel.ssa.blocks[0].operations

    allowed = (
        set(_BINARY)
        | set(_UNARY)
        | set(_SCALAR_MATH)
        | PREDICATE_OPCODES
        | {
            "arith.constant",
            "arith.neg",
            "arith.maximum",
            "arith.minimum",
            "arith.pow",
            "tensor.cast",
            "shape.dim",
            "mem.store",
        }
    )

    if any(
        operation.regions or _opcode(operation) not in allowed
        for operation in operations
    ):
        return None

    inputs = []
    scalar_inputs = _scalar_parameter_types(context)

    if scalar_inputs is None:
        return None

    scalar_bindings = {}
    extent = None

    for name in (*context.variables, *context.outputs):
        info = context.tensors[name]

        dtype = normalize_dtype(info.dtype)

        if (
            name in context.outputs
            and dtype not in _CPP_TYPES
            or name not in context.outputs
            and dtype != "float32"
        ):
            return None

        if info.ndim == 0 and name not in context.outputs:
            scalar_inputs[name] = "float32"
            continue

        if info.ndim != 1 or not canonical_flat_view(info):
            return None

        try:
            current = int(_source_shape(info)[0])
        except (ValueError, IndexError):
            return None

        if current <= 0 or extent is not None and current != extent:
            return None

        extent = current

        if name in context.variables:
            inputs.append(name)

    if extent is None:
        return None

    for operation in operations:
        if operation.opcode == "shape.dim":
            if len(operation.operands) != 1 or len(operation.results) != 1:
                return None

            expression = shape_dimension_expression(context, operation)

            if expression is None:
                return None

            name = operation.results[0].name
            scalar_inputs[name] = "index"
            scalar_bindings[name] = expression

    plan = plan_vector_dag(
        context,
        tuple(inputs),
        context.outputs,
        normalize_dtype=normalize_dtype,
        classify=_classify,
        dtype_bytes=_DTYPE_BYTES,
        scalar_inputs=scalar_inputs,
        storage_dtype=lambda dtype: "float32" if dtype == "bool" else dtype,
    )

    if plan is None or any(store.value.scalar for store in plan.stores):
        return None

    for store in context.stores:
        if (
            set(store.attrs) - {"target", "target_shape"}
            or len(store.operands) != 2
            or store.operands[1] not in context.outputs
            or tuple(map(str, store.attrs.get("target_shape", ())))
            != tuple(map(str, context.axes))
            or tuple(map(str, context.value_types[store.operands[0]].shape))
            != tuple(map(str, context.axes))
        ):
            return None

    chunk = 4096

    while chunk > extent and chunk >= 256:
        chunk //= 2

    while plan.bytes_per_element * chunk > ub_bytes and chunk >= 256:
        chunk //= 2

    if chunk < 128 or extent % chunk:
        return None

    from ninetoothed.backends.ascendc_resources import configure_flat_dag

    chunk = configure_flat_dag(
        context, plan, extent, default_chunk=chunk, budget_bytes=ub_bytes
    )

    return AscendRowDAG(
        plan, str(extent // chunk), chunk, (), tuple(scalar_bindings.items()), ub_bytes
    )


def flat_dag_memory_guard(context, match):
    """Complete packets also require contiguous and aligned actual GM views."""
    terms = []

    for name in (*context.variables, *context.outputs):
        info = context.tensors[name]

        if info.ndim == 0:
            continue

        stride = str(info.source_strides[0]) if info.source_strides else "1"
        terms.extend((f"({stride}) == 1", f"(((uintptr_t){name}_gm) & 63ULL) == 0"))

    terms.append(
        dense_staging_memory_guard(
            context,
            tuple(
                item.value.name
                for item in match.plan.inputs
                if item.value.name in context.tensors
            ),
            tuple(item.output for item in match.plan.stores),
            pointer_suffix="_gm",
        )
    )

    return " && ".join(terms)


def render_flat_dag(context, match, target):
    """Reuse UB execution, with packets independent of application tile size."""
    body, tasks = render_row_dag(context, match, target)

    return body.replace("Shared typed row DAG", "Shared typed flat DAG"), tasks


def match_row_dag(context, *, ub_bytes=ASCENDC_UB_BYTES):
    """Prove one complete float32 row per application before planning values."""
    return _match_row_dag(context, ub_bytes=ub_bytes)


def match_pointwise_rows(context, *, ub_bytes=ASCENDC_UB_BYTES):
    """Consume complete read maps only for pure pointwise full-row stores."""
    from ninetoothed.backends.emitters.base import private_parameter_conflicts
    from ninetoothed.compiler.cse import pure_expression

    if (
        context.cooperative_reduction_program
        or context.block_program
        or private_parameter_conflicts(
            context, ("nt_",), excluded_prefixes=("nt_materialized_", "nt_dot_partials")
        )
    ):
        return None

    if len(context.kernel.ssa.blocks) != 1 or set(context.variables).intersection(
        context.outputs
    ):
        return None

    if any(
        operation.opcode.startswith("reduce.")
        or (operation.opcode != "mem.store" and not pure_expression(operation))
        for operation in context.kernel.ssa.blocks[0].operations
    ):
        return None

    return _match_row_dag(context, ub_bytes=ub_bytes, pointwise=True)


def _match_row_dag(context, *, ub_bytes, pointwise=False):
    if (not pointwise and not context.cooperative_reduction_program) or len(
        context.kernel.ssa.blocks
    ) != 1:
        return None

    if any(operation.regions for operation in context.kernel.ssa.blocks[0].operations):
        return None

    allowed = (
        set(_BINARY)
        | set(_UNARY)
        | set(_SCALAR_MATH)
        | set(_REDUCE)
        | PREDICATE_OPCODES
        | {
            "arith.constant",
            "arith.neg",
            "arith.maximum",
            "arith.minimum",
            "arith.pow",
            "shape.dim",
            "tensor.view",
            "tensor.cast",
            "mem.store",
        }
    )

    if any(
        _opcode(operation) not in allowed
        for operation in context.kernel.ssa.blocks[0].operations
    ):
        return None

    if any(
        operation.opcode.startswith("mem.") and operation.opcode != "mem.store"
        for operation in context.kernel.ssa.blocks[0].operations
    ):
        return None

    primary = (
        context.tensors.get(context.outputs[0])
        if pointwise and context.outputs
        else next(
            (
                context.tensors[name]
                for name in context.variables
                if context.tensors[name].ndim == 2
                and len(_source_shape(context.tensors[name])) == 2
                and _source_shape(context.tensors[name])[1] != "1"
            ),
            None,
        )
    )

    if primary is None or primary.ndim != 2 or len(_source_shape(primary)) != 2:
        return None

    rows, width = _source_shape(primary)

    try:
        cols = int(width)
    except ValueError:
        return None

    if (
        cols <= 0
        or cols % 8
        or cols > 8192
        or tuple(map(str, context.axes)) != ("1", width)
    ):
        return None

    if pointwise:
        try:
            row_count = int(rows)
        except ValueError:
            return None

        if (
            row_count <= 0
            or row_count > (1 << 32) - 1
            or row_count * cols * 4 > (1 << 63) - 1
            or tuple(map(str, context.outer_axes)) != (rows, "1")
        ):
            return None

    inputs = []
    scalar_inputs = {}
    input_layouts = []

    for name in context.variables:
        info = context.tensors[name]

        if info.ndim == 0:
            if pointwise:
                return None

            scalar_inputs[name] = info.dtype
            continue

        if pointwise:
            if normalize_dtype(info.dtype) != "float32":
                return None

            layout = prove_row_input(context, name, row_count, cols)

            if layout is None:
                return None

            input_layouts.append((name, layout))

            if layout.mode == "scalar":
                scalar_inputs[name] = info.dtype
            else:
                inputs.append(name)

            continue

        if (
            info.ndim != 2
            or normalize_dtype(info.dtype) != "float32"
            or not patterns.canonical_row_view(info)
        ):
            return None

        shape = _source_shape(info)

        if shape == (rows, width):
            inputs.append(name)
        elif shape == (rows, "1"):
            scalar_inputs[name] = info.dtype
        else:
            return None

    parameter_types = _scalar_parameter_types(context)

    if parameter_types is None:
        return None

    scalar_inputs.update(parameter_types)
    scalar_bindings = {}

    for operation in context.kernel.ssa.blocks[0].operations:
        if operation.opcode != "shape.dim":
            continue

        if len(operation.operands) != 1 or len(operation.results) != 1:
            return None

        expression = shape_dimension_expression(
            context, operation, require_source_info=True
        )

        if expression is None:
            return None

        name = operation.results[0].name
        scalar_inputs[name] = "index"
        scalar_bindings[name] = expression

    collapsed = []

    for name in context.outputs:
        info = context.tensors[name]

        if (
            info.ndim != 2
            or normalize_dtype(info.dtype) not in _CPP_TYPES
            or not patterns.canonical_row_view(info)
        ):
            return None

        shape = _source_shape(info)

        if (
            normalize_dtype(info.dtype) == "float16"
            and shape != (rows, "1")
            and cols % 16
        ):
            return None

        if shape == (rows, "1"):
            if pointwise:
                return None

            collapsed.append(name)
        elif shape != (rows, width):
            return None

    plan = plan_vector_dag(
        context,
        tuple(inputs),
        context.outputs,
        normalize_dtype=normalize_dtype,
        classify=_classify,
        dtype_bytes=_DTYPE_BYTES,
        scalar_inputs=scalar_inputs,
        reduction_axis=None if pointwise else 1,
        storage_dtype=lambda dtype: "float32" if dtype == "bool" else dtype,
    )

    if plan is None or plan.bytes_per_element * cols + 64 * len(collapsed) > ub_bytes:
        return None

    for store in plan.stores:
        if pointwise and store.value.scalar:
            return None

        shape = tuple(map(str, context.value_types[store.value.name].shape))
        operation = next(
            operation
            for operation in context.stores
            if operation.operands[1] == store.output
        )
        expected = ("1", "1") if store.output in collapsed else ("1", width)

        if (
            tuple(
                map(
                    str,
                    operation.attrs.get(
                        "target_shape", context.value_types[store.output].shape
                    ),
                )
            )
            != expected
        ):
            return None

        if operation.attrs.get("target_dtype_level", 0) or operation.attrs.get(
            "base_dtype_level", 0
        ):
            return None

        if store.output in collapsed and not store.value.scalar:
            return None

        if (
            store.value.scalar
            and store.output not in collapsed
            and store.value.dtype != "float32"
        ):
            return None

        if store.value.scalar:
            if shape and any(size != "1" for size in shape):
                return None
        elif shape != ("1", width):
            return None
    return AscendRowDAG(
        plan,
        rows,
        cols,
        tuple(collapsed),
        tuple(scalar_bindings.items()),
        ub_bytes,
        tuple(input_layouts),
    )


def row_dag_memory_guard(context, match):
    """Require physical row strides and aligned bases independently of IR views."""
    if match.input_layouts:
        return _pointwise_row_memory_guard(context, match)

    terms = []
    names = [item.value.name for item in match.plan.inputs]
    names.extend(item.output for item in match.plan.stores)

    for name in names:
        info = context.tensors.get(name)

        if info is None or info.ndim == 0:
            continue

        shape = _source_shape(info)
        strides = tuple(map(str, info.source_strides)) or (shape[1], "1")
        terms.extend(
            (
                f"({strides[0]}) == ({shape[1]})",
                f"({strides[1]}) == 1",
                f"(((uintptr_t){name}_gm) & 63ULL) == 0",
            )
        )

    terms.append(
        dense_staging_memory_guard(
            context,
            tuple(
                item.value.name
                for item in match.plan.inputs
                if item.value.name in context.tensors
            ),
            tuple(item.output for item in match.plan.stores),
            pointer_suffix="_gm",
        )
    )

    return " && ".join(f"({term})" for term in terms) or "true"


def _pointwise_row_memory_guard(context, match):
    """Dense read proofs do not permit hoisting reads across aliased stores."""
    terms = []
    inputs = []
    outputs = []

    for name, layout in match.input_layouts:
        terms.extend((layout.guard, f"(((uintptr_t){name}_gm) & 63ULL) == 0"))
        inputs.append((name, layout.source_rows * layout.source_columns * 4))

    for store in match.plan.stores:
        name = store.output
        info = context.tensors[name]
        shape = _source_shape(info)
        strides = tuple(map(str, info.source_strides)) or (shape[1], "1")
        size = int(match.rows) * match.cols * _DTYPE_BYTES[normalize_dtype(info.dtype)]
        outputs.append((name, size))
        terms.extend(
            (
                f"({strides[0]}) == ({shape[1]})",
                f"({strides[1]}) == 1",
                f"(((uintptr_t){name}_gm) & 63ULL) == 0",
            )
        )

    # Subtract only the smaller unsigned address; no endpoint addition wraps.
    for index, (left, left_size) in enumerate(outputs):
        for right, right_size in (*inputs, *outputs[index + 1 :]):
            p, q = f"(uintptr_t){left}_gm", f"(uintptr_t){right}_gm"
            terms.append(
                f"(({p} >= {q}) ? (({p} - {q}) >= {right_size}ULL) "
                f": (({q} - {p}) >= {left_size}ULL))"
            )

    return " && ".join(f"({term})" for term in terms)


def _scalar_name(value):
    return "nt_scalar_" + re.sub(r"[^A-Za-z0-9_]", "_", value.name)


def _scalar_operand(value, target):
    if isinstance(value.literal, bool):
        return "1.0f" if value.literal else "0.0f"

    return (
        target.literal(value.literal)
        if value.literal is not None
        else _scalar_name(value)
    )


def _scalar_expression(operation, args, target):
    opcode = _opcode(operation)
    predicate = scalar_predicate_expression(operation, args)

    if predicate is not None:
        return predicate

    if opcode in _BINARY:
        return f"({args[0]} {_BINARY[opcode][1]} {args[1]})"

    if opcode == "arith.neg":
        return f"(-{args[0]})"

    if opcode == "tensor.view":
        return args[0]

    if opcode == "tensor.cast":
        return target.cast(operation.results[0].type.dtype, args[0])

    if opcode in {"arith.maximum", "arith.minimum"}:
        cmp = ">" if opcode == "arith.maximum" else "<"

        return f"({args[0]} != {args[0]} ? {args[0]} : ({args[1]} != {args[1]} ? {args[1]} : ({args[0]} {cmp} {args[1]} ? {args[0]} : {args[1]})))"

    if opcode == "arith.pow":
        return f"nt_pow({args[0]}, {args[1]})"

    if opcode in _UNARY:
        function = _UNARY[opcode][1]

        return f"{function}({args[0]})" if function else f"(1.0f / {args[0]})"
    return f"{_SCALAR_MATH[opcode]}({args[0]})"


def _render_int32_cast(dst, source, count):
    """Truncate finite legal lanes and retain the scalar invalid-value policy."""
    return f"""        {{ // Checked float32-to-int32 conversion.
        Cast({dst}, {source}, RoundMode::CAST_TRUNC, {count});
        PipeBarrier<PIPE_ALL>();
        const int32_t nt_cast_prefix = (({count}) / 128) * 128;
        if (nt_cast_prefix != 0) {{
            CompareScalar(nt_ref_mask0, {source}, -2147483648.0f, CMPMODE::GE, nt_cast_prefix);
            CompareScalar(nt_ref_mask1, {source}, 2147483648.0f, CMPMODE::LT, nt_cast_prefix);
            PipeBarrier<PIPE_ALL>();
            And(nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_cast_prefix / 16);
            PipeBarrier<PIPE_ALL>();
            auto nt_cast_words = nt_ref_mask0.ReinterpretCast<uint32_t>();
            for (int32_t nt_word = 0; nt_word < nt_cast_prefix / 32; nt_word++) {{
                uint32_t nt_pending = ~nt_cast_words.GetValue(nt_word);
                if (nt_pending == 0) {{ continue; }}
                for (int32_t nt_bit = 0; nt_bit < 32; nt_bit++) {{
                    if ((nt_pending & (1U << nt_bit)) != 0) {{
                        {dst}.SetValue(nt_word * 32 + nt_bit, (-2147483647 - 1));
                    }}
                }}
            }}
        }}
        for (int32_t nt_lane = nt_cast_prefix; nt_lane < ({count}); nt_lane++) {{
            {dst}.SetValue(nt_lane, nt_cast_int32({source}.GetValue(nt_lane)));
        }}
        }}"""


def _render_bounded_tanh(dst, source, count):
    """Refine the bounded exponential recipe and retain scalar exceptional lanes."""
    return f"""        {{ // Bounded tanh recipe with full-domain scalar correction.
        Abs(nt_ref0, {source}, {count});
        PipeBarrier<PIPE_V>();
        Muls(nt_ref0, nt_ref0, 2.0f, {count});
        PipeBarrier<PIPE_V>();
        Exp(nt_ref0, nt_ref0, {count});
        PipeBarrier<PIPE_V>();
        Adds(nt_ref0, nt_ref0, 1.0f, {count});
        PipeBarrier<PIPE_V>();
        Reciprocal({dst}, nt_ref0, {count});
        PipeBarrier<PIPE_V>();
        for (int32_t nt_tanh_refine = 0; nt_tanh_refine < 2; nt_tanh_refine++) {{
            Mul(nt_ref1, nt_ref0, {dst}, {count});
            PipeBarrier<PIPE_V>();
            Muls(nt_ref1, nt_ref1, -1.0f, {count});
            PipeBarrier<PIPE_V>();
            Adds(nt_ref1, nt_ref1, 2.0f, {count});
            PipeBarrier<PIPE_V>();
            Mul({dst}, {dst}, nt_ref1, {count});
            PipeBarrier<PIPE_V>();
        }}
        Muls({dst}, {dst}, -2.0f, {count});
        PipeBarrier<PIPE_V>();
        Adds({dst}, {dst}, 1.0f, {count});
        PipeBarrier<PIPE_V>();
        const int32_t nt_tanh_prefix = (({count}) / 128) * 128;
        if (nt_tanh_prefix != 0) {{
            Muls(nt_ref1, {dst}, -1.0f, nt_tanh_prefix);
            PipeBarrier<PIPE_V>();
            CompareScalar(nt_ref_mask0, {source}, 0.0f, CMPMODE::GE, nt_tanh_prefix);
            PipeBarrier<PIPE_ALL>();
            Select({dst}, nt_ref_mask0, {dst}, nt_ref1, SELMODE::VSEL_TENSOR_TENSOR_MODE, nt_tanh_prefix);
            PipeBarrier<PIPE_V>();
            Abs(nt_ref1, {source}, nt_tanh_prefix);
            PipeBarrier<PIPE_V>();
            CompareScalar(nt_ref_mask0, nt_ref1, 0.001f, CMPMODE::GE, nt_tanh_prefix);
            CompareScalar(nt_ref_mask1, nt_ref1, 8.8f, CMPMODE::LE, nt_tanh_prefix);
            PipeBarrier<PIPE_ALL>();
            And(nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_tanh_prefix / 16);
            PipeBarrier<PIPE_ALL>();
            auto nt_tanh_words = nt_ref_mask0.ReinterpretCast<uint32_t>();
            for (int32_t nt_word = 0; nt_word < nt_tanh_prefix / 32; nt_word++) {{
                uint32_t nt_pending = ~nt_tanh_words.GetValue(nt_word);
                if (nt_pending == 0) {{ continue; }}
                for (int32_t nt_bit = 0; nt_bit < 32; nt_bit++) {{
                    if ((nt_pending & (1U << nt_bit)) != 0) {{
                        const int32_t nt_lane = nt_word * 32 + nt_bit;
                        {dst}.SetValue(nt_lane, nt_tanh({source}.GetValue(nt_lane)));
                    }}
                }}
            }}
        }}
        for (int32_t nt_lane = nt_tanh_prefix; nt_lane < ({count}); nt_lane++) {{
            {dst}.SetValue(nt_lane, nt_tanh({source}.GetValue(nt_lane)));
        }}
        }}"""


def _render_bounded_polynomial(opcode, dst, source, count, target):
    """Use the shared separately rounded recipe, repairing exceptional lanes."""
    recipe = bounded_polynomial(opcode)
    coefficients = [target.literal(value) for value in recipe.coefficients]
    lines = [
        "        { // Bounded float32 polynomial; full-domain scalar correction.",
        f"        Mul(nt_ref0, {source}, {source}, {count});",
        "        PipeBarrier<PIPE_V>();",
        f"        Duplicate({dst}, {coefficients[0]}, {count});",
        "        PipeBarrier<PIPE_V>();",
    ]

    for coefficient in coefficients[1:]:
        lines.extend(
            (
                f"        Mul({dst}, {dst}, nt_ref0, {count});",
                "        PipeBarrier<PIPE_V>();",
                f"        Adds({dst}, {dst}, {coefficient}, {count});",
                "        PipeBarrier<PIPE_V>();",
            )
        )

    if recipe.multiply_input:
        lines.extend(
            (
                f"        Mul({dst}, {dst}, {source}, {count});",
                "        PipeBarrier<PIPE_V>();",
            )
        )

    lines.extend(
        (
            f"        const int32_t nt_poly_prefix = (({count}) / 128) * 128;",
            "        if (nt_poly_prefix != 0) {",
            f"            CompareScalar(nt_ref_mask0, {source}, {target.literal(recipe.domain.lower)}, CMPMODE::GE, nt_poly_prefix);",
            f"            CompareScalar(nt_ref_mask1, {source}, {target.literal(recipe.domain.upper)}, CMPMODE::LE, nt_poly_prefix);",
            "            PipeBarrier<PIPE_ALL>();",
            "            And(nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_poly_prefix / 16);",
            "            PipeBarrier<PIPE_ALL>();",
            # Zero and subnormal lanes retain the scalar primitive: device
            # vector arithmetic need not preserve tiny products or zero signs.
            f"            CompareScalar(nt_ref_mask1, {source}, 0x1p-126f, CMPMODE::GE, nt_poly_prefix);",
            f"            CompareScalar(nt_ref_mask2, {source}, -0x1p-126f, CMPMODE::LE, nt_poly_prefix);",
            "            PipeBarrier<PIPE_ALL>();",
            "            Or(nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_ref_mask2.ReinterpretCast<uint16_t>(), nt_poly_prefix / 16);",
            "            PipeBarrier<PIPE_ALL>();",
            "            And(nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_poly_prefix / 16);",
            "            PipeBarrier<PIPE_ALL>();",
            "            auto nt_poly_words = nt_ref_mask0.ReinterpretCast<uint32_t>();",
            "            for (int32_t nt_word = 0; nt_word < nt_poly_prefix / 32; nt_word++) {",
            "                uint32_t nt_pending = ~nt_poly_words.GetValue(nt_word);",
            "                if (nt_pending == 0) { continue; }",
            "                for (int32_t nt_bit = 0; nt_bit < 32; nt_bit++) {",
            "                    if ((nt_pending & (1U << nt_bit)) != 0) {",
            "                        int32_t nt_lane = nt_word * 32 + nt_bit;",
            f"                        {dst}.SetValue(nt_lane, {_SCALAR_MATH[opcode]}({source}.GetValue(nt_lane)));",
            "                    }",
            "                }",
            "            }",
            "        }",
            f"        for (int32_t nt_lane = nt_poly_prefix; nt_lane < ({count}); nt_lane++) {{",
            f"            {dst}.SetValue(nt_lane, {_SCALAR_MATH[opcode]}({source}.GetValue(nt_lane)));",
            "        }",
            "        }",
        )
    )

    return "\n".join(lines)


def plan_row_execution(context, match, *, route="typed-row-dag"):
    """Bind SDK choices to the final DAG and its physical operand lifetimes."""
    from ninetoothed.backends.ascendc_resources import configure_dag_division_strategy

    plan = match.plan
    native_division = configure_dag_division_strategy(context, plan)
    has_strict_repair = any(
        not step.node.result.scalar
        and (
            _opcode(step.node.operation) in {"math.rsqrt", "math.reciprocal"}
            or (
                _opcode(step.node.operation) == "arith.div"
                and not step.node.operands[1].scalar
            )
        )
        for step in plan.steps
    )
    validity_summary = configure_validity_summary(
        context,
        route=route,
        maximum_elements=match.cols,
        strict_scratch=has_strict_repair
        and "float32" in plan.scratch_dtypes
        and plan.scratch_dtypes.count("uint8") >= 2,
    )

    if "ascendc_division_strategy" in context.scheduled_metadata:
        context.scheduled_metadata["ascendc_division_strategy"]["route"] = route
    return replace(
        match,
        native_division=native_division,
        validity_summary=validity_summary,
        numeric_bounds=program_facts(context.kernel.ssa, "numeric_bounds"),
    )


def render_row_dag(context, match, target):
    """Stage rows once, execute every DAG node once, and own complete outputs."""
    plan = match.plan
    native_division = match.native_division
    validity_summary = match.validity_summary
    cols = match.cols
    input_layouts = dict(match.input_layouts)
    bounds = match.numeric_bounds
    bound_consumers = []
    lines = [
        "    // Shared typed row DAG: one staging load and one evaluation per SSA value.",
        f"    // Usable UB budget: {match.ub_bytes} bytes (CANN platform UB size).",
        "    const int64_t nt_row0 = (int64_t)GetBlockIdx() * 16;",
        f"    if (nt_row0 >= ({match.rows})) {{ return; }}",
        f"    const int32_t nt_nr = (int32_t)(((({match.rows}) - nt_row0) < 16) ? (({match.rows}) - nt_row0) : 16);",
        f"    const int32_t nt_cols = {cols};",
        "    TPipe nt_pipe;",
    ]
    tensor_names = [
        item.value.name
        for item in plan.inputs
        if context.tensors.get(item.value.name) is not None
        and context.tensors[item.value.name].ndim != 0
    ]
    tensor_names.extend(item.output for item in plan.stores)

    for name in tensor_names:
        cpp = _CPP_TYPES[normalize_dtype(context.tensors[name].dtype)]
        lines.extend(
            (
                f"    GlobalTensor<{cpp}> nt_gm_{name};",
                f"    nt_gm_{name}.SetGlobalBuffer((__gm__ {cpp}*){name}_gm);",
            )
        )

    for index, dtype in enumerate(plan.buffer_dtypes):
        cpp = _CPP_TYPES[dtype]
        lines.extend(
            (
                f"    TBuf<TPosition::VECCALC> nt_dag_buf{index};",
                f"    nt_pipe.InitBuffer(nt_dag_buf{index}, {cols} * sizeof({cpp}));",
                f"    LocalTensor<{cpp}> nt_dag_v{index} = nt_dag_buf{index}.Get<{cpp}>();",
            )
        )

    scratch_names = []
    counter = {"float32": 0, "uint8": 0}

    for index, dtype in enumerate(plan.scratch_dtypes):
        count = counter[dtype]
        counter[dtype] += 1
        name = f"nt_ref{count}" if dtype == "float32" else f"nt_ref_mask{count}"
        scratch_names.append(name)
        cpp = "float" if dtype == "float32" else "uint8_t"
        size = (
            f"{cols} * sizeof(float)"
            if dtype == "float32"
            else str(((cols + 255) // 256) * 32)
        )
        lines.extend(
            (
                f"    TBuf<TPosition::VECCALC> nt_dag_s{index};",
                f"    nt_pipe.InitBuffer(nt_dag_s{index}, {size});",
                f"    LocalTensor<{cpp}> {name} = nt_dag_s{index}.Get<{cpp}>();",
            )
        )

    for output in match.collapsed:
        cpp = _CPP_TYPES[normalize_dtype(context.tensors[output].dtype)]
        lines.append(f"    {cpp} nt_results_{output}[16];")

    lines.append("    for (int32_t nt_r = 0; nt_r < nt_nr; nt_r++) {")
    lines.append("        int64_t nt_row = nt_row0 + nt_r;")

    for item in plan.inputs:
        name = item.value.name

        if item.value.scalar:
            info = context.tensors.get(name)
            layout = input_layouts.get(name)
            value = (
                f"nt_gm_{name}.GetValue((uint32_t)({layout.base_offset('nt_row')}))"
                if layout is not None
                else dict(match.scalar_bindings).get(name)
                or (
                    f"nt_gm_{name}.GetValue((uint32_t)nt_row)"
                    if info is not None and info.ndim != 0
                    else name
                )
            )
            cpp = "int64_t" if item.value.dtype == "index" else "float"
            conversion = (
                f"(({value}) ? 1.0f : 0.0f)"
                if item.value.dtype == "bool"
                else f"({cpp})({value})"
            )
            lines.append(f"        {cpp} {_scalar_name(item.value)} = {conversion};")
        else:
            layout = input_layouts.get(name)
            offset = (
                layout.base_offset("nt_row")
                if layout is not None
                else "nt_row * nt_cols"
            )
            lines.append(
                f"        DataCopy(nt_dag_v{item.buffer}, nt_gm_{name}[{offset}], (uint32_t)nt_cols);"
            )

    lines.append("        PipeBarrier<PIPE_ALL>();")

    for step in plan.steps:
        node = step.node
        opcode = _opcode(node.operation)
        args = [
            _scalar_operand(value, target) if value.scalar else f"nt_dag_v{buffer}"
            for value, buffer in zip(node.operands, step.operand_buffers)
        ]
        result = _scalar_name(node.result)

        if opcode in _REDUCE:
            source = args[0]

            lines.append(
                f"        {_REDUCE[opcode]}(nt_ref1, {source}, nt_ref0, nt_cols);"
            )
            lines.extend(
                (
                    "        PipeBarrier<PIPE_ALL>();",
                    f"        float {result} = nt_ref1.GetValue(0);",
                )
            )

            if opcode != "reduce.sum":
                lines.append(
                    render_nan_propagation(result, source, "nt_cols", "nt_ref_mask0")
                )
        elif node.result.scalar:
            cpp = (
                "float"
                if node.result.dtype == "bool"
                else _CPP_TYPES[node.result.dtype]
            )
            lines.append(
                f"        {cpp} {result} = {_scalar_expression(node.operation, args, target)};"
            )
        else:
            dst = f"nt_dag_v{step.destination}"

            if node.lowering.mode == "scalar":
                scalar_args = [
                    arg if value.scalar else f"{arg}.GetValue(nt_j)"
                    for value, arg in zip(node.operands, args)
                ]
                lines.extend(
                    (
                        "        PipeBarrier<PIPE_ALL>();",
                        "        for (int32_t nt_j = 0; nt_j < nt_cols; nt_j++) {",
                        f"            {dst}.SetValue(nt_j, {_scalar_expression(node.operation, scalar_args, target)});",
                        "        }",
                    )
                )
            elif node.lowering.mode == "identity":
                lines.append(f"        DataCopy({dst}, {args[0]}, (uint32_t)nt_cols);")
            elif node.lowering.mode == "cast":
                if node.result.dtype == "float16":
                    lines.append(
                        f"        Cast({dst}, {args[0]}, RoundMode::CAST_RINT, nt_cols);"
                    )
                else:
                    lines.append(_render_int32_cast(dst, args[0], "nt_cols"))
            elif node.lowering.mode == "square":
                lines.append(f"        Mul({dst}, {args[0]}, {args[0]}, nt_cols);")
            elif node.lowering.mode == "bounded_polynomial":
                lines.append(
                    _render_bounded_polynomial(opcode, dst, args[0], "nt_cols", target)
                )
            elif node.lowering.mode == "bounded_tanh":
                lines.append(_render_bounded_tanh(dst, args[0], "nt_cols"))
            elif node.lowering.mode in {"predicate_compare", "predicate_select"}:
                predicate = render_predicate(step, dst, args, "nt_cols")
                assert predicate is not None
                lines.append(predicate)
            elif node.lowering.mode == "extrema":
                lines.append(
                    render_extrema(
                        "MaxStrict" if opcode == "arith.maximum" else "MinStrict",
                        dst,
                        args[0],
                        args[1],
                        "nt_cols",
                        scalar_lhs=node.operands[0].scalar,
                        scalar_rhs=node.operands[1].scalar,
                    )
                )
            elif opcode in _UNARY:
                intrinsic = _UNARY[opcode][0]

                if intrinsic.endswith("Strict"):
                    lower = numeric_lower_bound(bounds, node.operands[0].name)

                    if lower is not None and lower >= 2.0**-120:
                        bound_consumers.append(
                            {
                                "operation": opcode,
                                "destination": node.result.name,
                                "ssa_values": (node.operands[0].name,),
                                "lower": lower,
                            }
                        )

                    lines.append(
                        render_refinement(
                            intrinsic,
                            dst,
                            args[0],
                            "nt_cols",
                            known_lower_bound=lower,
                            validity_summary=validity_summary,
                            native_division=native_division,
                            operands_preserved=native_division,
                        )
                    )
                else:
                    lines.append(f"        {intrinsic}({dst}, {args[0]}, nt_cols);")
            elif opcode == "arith.neg":
                lines.append(f"        Muls({dst}, {args[0]}, -1.0f, nt_cols);")
            elif opcode == "arith.div":
                lhs, rhs = node.operands

                if rhs.scalar:
                    lines.extend(
                        (
                            f"        if (nt_fabs((float)({args[1]})) >= 0x1p-120f && nt_fabs((float)({args[1]})) <= 0x1p120f) {{",
                            f"            Muls({dst}, {args[0]}, (float)(1.0f / {args[1]}), nt_cols);",
                            "        } else {",
                            "            PipeBarrier<PIPE_ALL>();",
                            "            for (int32_t nt_j = 0; nt_j < nt_cols; nt_j++) {",
                            f"                {dst}.SetValue(nt_j, {args[0]}.GetValue(nt_j) / (float)({args[1]}));",
                            "            }",
                            "        }",
                        )
                    )
                else:
                    lower = numeric_lower_bound(bounds, rhs.name)

                    if lower is not None and lower >= 2.0**-120:
                        bound_consumers.append(
                            {
                                "operation": opcode,
                                "destination": node.result.name,
                                "ssa_values": (rhs.name,),
                                "lower": lower,
                            }
                        )

                    lines.append(
                        render_division(
                            dst,
                            args[0],
                            args[1],
                            "nt_cols",
                            scalar_lhs=lhs.scalar,
                            known_rhs_lower_bound=lower,
                            validity_summary=validity_summary,
                            native_division=native_division,
                            operands_preserved=native_division,
                        )
                    )
            else:
                lhs, rhs = node.operands
                intrinsic = _BINARY[opcode][0]

                if lhs.scalar or rhs.scalar:
                    uniform, vector = (
                        (args[0], args[1]) if lhs.scalar else (args[1], args[0])
                    )

                    if opcode == "arith.sub":
                        if lhs.scalar:
                            lines.append(
                                f"        Muls({dst}, {vector}, -1.0f, nt_cols);"
                            )
                            lines.append(
                                f"        Adds({dst}, {dst}, (float)({uniform}), nt_cols);"
                            )
                        else:
                            lines.append(
                                f"        Adds({dst}, {vector}, (float)(-{uniform}), nt_cols);"
                            )
                    else:
                        scalar_op = "Adds" if opcode == "arith.add" else "Muls"
                        lines.append(
                            f"        {scalar_op}({dst}, {vector}, (float)({uniform}), nt_cols);"
                        )
                else:
                    lines.append(
                        f"        {intrinsic}({dst}, {args[0]}, {args[1]}, nt_cols);"
                    )

        lines.append("        PipeBarrier<PIPE_ALL>();")

    for store in plan.stores:
        if store.output in match.collapsed:
            lines.append(
                f"        nt_results_{store.output}[nt_r] = {_scalar_operand(store.value, target)};"
            )
        else:
            src = f"nt_dag_v{store.buffer}"

            if store.value.scalar:
                src = "nt_ref0"
                lines.append(
                    f"        Duplicate({src}, (float)({_scalar_operand(store.value, target)}), nt_cols);"
                )
                lines.append("        PipeBarrier<PIPE_ALL>();")

            lines.append(
                f"        DataCopy(nt_gm_{store.output}[nt_row * nt_cols], {src}, (uint32_t)nt_cols);"
            )
            lines.append("        PipeBarrier<PIPE_ALL>();")

    lines.append("    }")

    for output in match.collapsed:
        lines.extend(
            (
                "    for (int32_t nt_r = 0; nt_r < nt_nr; nt_r++) {",
                f"        nt_gm_{output}.SetValue((uint32_t)(nt_row0 + nt_r), nt_results_{output}[nt_r]);",
                "    }",
            )
        )

    if bound_consumers:
        context.scheduled_metadata["ascendc_numeric_bounds"] = {
            "path": "typed-row-dag",
            "consumers": bound_consumers,
            "repair": "upper-bound-and-NaN-Inf-with-original-scalar-tail",
        }
    return "\n".join(lines), f"(({match.rows}) + 15) / 16"


__all__ = [
    "flat_dag_memory_guard",
    "match_flat_dag",
    "match_pointwise_rows",
    "match_row_dag",
    "render_flat_dag",
    "render_row_dag",
    "row_dag_memory_guard",
]
