"""BANG primitives consuming a proved typed vector DAG."""

from ninetoothed.backends.emitters.native_vector import plan_native_vector
from ninetoothed.backends.emitters.vector_ops import COMPARISON_OPERATORS as _COMPARE
from ninetoothed.backends.emitters.vector_ops import classify_pointwise as classify

_TYPES = {"float32": "float", "float16": "half"}
_WIDTHS = {"float32": 4, "float16": 2}
_BINARY = {"arith.add": "add", "arith.sub": "sub", "arith.mul": "mul"}


def plan_vector(context, elements):
    return plan_native_vector(
        context, classify, elements=elements, dtype_bytes=_WIDTHS, budget=384 * 1024
    )


def render_vector(plan, target):
    """Render the already selected domain and allocation; do not replan."""
    dag, chunk = plan.dag, plan.storage.elements
    lines = [
        f"__nram__ {_TYPES[dtype]} nt_vec_b{i}[{chunk}];"
        for i, dtype in enumerate(dag.buffer_dtypes)
    ]
    lines += [
        f"const int64_t nt_vec_start = (int64_t)taskIdX * {chunk};",
        f"int nt_vec_count = (int)((({plan.extent}) - nt_vec_start) < {chunk} ? (({plan.extent}) - nt_vec_start) : {chunk});",
        "int nt_vec_aligned = (nt_vec_count + 127) / 128 * 128;",
    ]

    for value, buffer in dag.inputs:
        dest = f"nt_vec_b{buffer}"
        lines += [
            f"for (int nt_vec_i = 0; nt_vec_i < nt_vec_aligned; ++nt_vec_i) {{ {dest}[nt_vec_i] = 0; }}",
            f"{target.global_helper('__memcpy')}({dest}, {value.name} + nt_vec_start, nt_vec_count * sizeof({_TYPES[value.dtype]}), ::GDRAM2NRAM);",
        ]

    def operand(value, buffer, index=None):
        if not value.scalar:
            return f"nt_vec_b{buffer}" + (f"[{index}]" if index else "")

        literal = (
            target.literal(value.literal) if value.literal is not None else value.name
        )

        return target.cast(value.dtype if value.dtype in _TYPES else "float32", literal)

    for step in dag.steps:
        code = step.operation.opcode
        destination = f"nt_vec_b{step.destination}"
        arguments = [operand(v, b) for v, b in zip(step.operands, step.sources)]

        if step.mode == "binary":
            lines.append(
                f"{target.global_helper('__bang_' + _BINARY[code])}({destination}, {arguments[0]}, {arguments[1]}, nt_vec_aligned);"
            )
        elif step.mode == "scalar_binary":
            if step.operands[0].scalar:
                arguments.reverse()

            lines.append(
                f"{target.global_helper('__bang_' + _BINARY[code] + '_scalar')}({destination}, {arguments[0]}, {arguments[1]}, nt_vec_aligned);"
            )
        elif step.mode == "cast":
            source_dtype = step.operands[0].dtype
            target_dtype = step.result.dtype

            if source_dtype == target_dtype:
                lines.append(
                    f"{target.global_helper('__memcpy')}({destination}, {arguments[0]}, nt_vec_aligned * sizeof({_TYPES[target_dtype]}), ::NRAM2NRAM);"
                )
            else:
                instruction = (
                    "__bang_float2half_rn"
                    if target_dtype == "float16"
                    else "__bang_half2float"
                )
                lines.append(
                    f"{target.global_helper(instruction)}({destination}, {arguments[0]}, nt_vec_aligned);"
                )
        else:
            args = [
                operand(v, b, "nt_vec_i") for v, b in zip(step.operands, step.sources)
            ]

            if code in _COMPARE:
                expression = f"({args[0]} {_COMPARE[code]} {args[1]})"
            elif code == "select.where":
                expression = f"({args[0]} ? {args[1]} : {args[2]})"
            else:
                sign = {"arith.add": "+", "arith.sub": "-", "arith.mul": "*"}[code]
                # Half arithmetic with an FP32 scalar rounds only after the
                # operation, rather than converting the scalar to half first.
                args = [f"(float)({arg})" for arg in args]
                expression = f"({args[0]} {sign} {args[1]})"

            lines.append(
                f"for (int nt_vec_i = 0; nt_vec_i < nt_vec_count; ++nt_vec_i) {{ {destination}[nt_vec_i] = {target.cast(step.result.dtype if step.result.dtype != 'bool' else 'float32', expression)}; }}"
            )

    for output, buffer, value in dag.stores:
        lines.append(
            f"{target.global_helper('__memcpy')}({output} + nt_vec_start, nt_vec_b{buffer}, nt_vec_count * sizeof({_TYPES[value.dtype]}), ::NRAM2GDRAM);"
        )

    return "\n".join(lines)
