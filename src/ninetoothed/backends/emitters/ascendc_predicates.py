"""Typed float32 comparisons and selection through packed AscendC masks."""

from ninetoothed.backends.emitters.vector_dag import VectorLowering

_COMPARE = {
    prefix + suffix: (suffix.upper(), symbol)
    for prefix in ("arith.", "cmp.")
    for suffix, symbol in (
        ("lt", "<"),
        ("le", "<="),
        ("gt", ">"),
        ("ge", ">="),
        ("eq", "=="),
        ("ne", "!="),
    )
}
_REVERSE = {"LT": "GT", "LE": "GE", "GT": "LT", "GE": "LE", "EQ": "EQ", "NE": "NE"}
PREDICATE_OPCODES = frozenset((*_COMPARE, "select.where"))


def _float_operand(value):
    return value.dtype == "float32" or (
        value.scalar
        and value.dtype in {"int32", "int64", "index"}
        and isinstance(value.literal, (int, float))
        and not isinstance(value.literal, bool)
    )


def classify_predicate(operation, operands, dtype):
    """Classify logical bool values using float32 0/1 intermediate storage.

    The caller's storage_dtype maps intermediate bool to float32. Integer
    tensor comparisons and casts, non-bool conditions and wide data reject.
    One packed mask and one float scratch buffer cover all scalar branches.
    """
    if operation.regions:
        return None

    if operation.opcode in _COMPARE:
        if (
            dtype == "bool"
            and len(operands) == 2
            and all(map(_float_operand, operands))
        ):
            return VectorLowering("predicate_compare", ("uint8", "float32"))
    elif operation.opcode == "select.where":
        if (
            dtype == "float32"
            and len(operands) == 3
            and operands[0].dtype == "bool"
            and all(map(_float_operand, operands[1:]))
        ):
            return VectorLowering("predicate_select", ("uint8", "float32"))
    return None


def scalar_predicate_expression(operation, args):
    """Return numeric predicates or the selected operand without bool casts.

    Logical bool intermediates use float32 storage. CCEC does not support a
    bool-to-float conversion, so comparisons explicitly choose 1.0f or 0.0f.
    """
    if operation.opcode in _COMPARE and len(args) == 2:
        comparison = f"(({args[0]}) {_COMPARE[operation.opcode][1]} ({args[1]}))"

        return f"({comparison} ? 1.0f : 0.0f)"

    if operation.opcode == "select.where" and len(args) == 3:
        return f"((({args[0]}) != 0.0f) ? ({args[1]}) : ({args[2]}))"
    return None


def render_predicate(
    step, destination, args, count, *, mask="nt_ref_mask0", scratch="nt_ref0"
):
    """Emit complete 128-lane mask packets and exact original scalar tails.

    ``args`` contains target expressions for scalar values or LocalTensor
    names for vector values. Sources and destination obey the shared DAG's
    non-aliasing lifetime proof. The caller allocates the two declared scratch
    buffers and retains the complete indexed fallback for an invalid layout.
    """
    operation, operands = step.node.operation, step.node.operands

    if (
        len(args) != len(operands)
        or classify_predicate(operation, operands, step.node.result.dtype) is None
    ):
        return None

    scalar_args = [
        f"(float)({argument})"
        if value.scalar
        else f"{argument}.GetValue(nt_predicate_lane)"
        for argument, value in zip(args, operands)
    ]
    expression = scalar_predicate_expression(operation, scalar_args)
    lines = [
        "        { // Typed packed comparison/select; logical bool stored as float32.",
        f"        const int32_t nt_predicate_prefix = (({count}) / 128) * 128;",
        "        if (nt_predicate_prefix != 0) {",
    ]

    if operation.opcode in _COMPARE:
        mode = _COMPARE[operation.opcode][0]

        if all(value.scalar for value in operands):
            scalar_expression = scalar_predicate_expression(operation, scalar_args)
            lines.append(
                f"            Duplicate({destination}, {scalar_expression}, nt_predicate_prefix);"
            )
        else:
            # C220 native NE is ordered and can return false for NaN.
            # Inverting the EQ mask preserves C/Python unordered != truth.
            comparison_mode = "EQ" if mode == "NE" else mode

            if operands[1].scalar:
                compare = f"CompareScalar({mask}, {args[0]}, (float)({args[1]}), CMPMODE::{comparison_mode}, (uint32_t)nt_predicate_prefix);"
            elif operands[0].scalar:
                compare = f"CompareScalar({mask}, {args[1]}, (float)({args[0]}), CMPMODE::{_REVERSE[comparison_mode]}, (uint32_t)nt_predicate_prefix);"
            else:
                compare = f"Compare({mask}, {args[0]}, {args[1]}, CMPMODE::{comparison_mode}, (uint32_t)nt_predicate_prefix);"

            lines.append("            " + compare)

            if mode == "NE":
                lines.extend(
                    (
                        "            PipeBarrier<PIPE_ALL>();",
                        f"            Not({mask}.ReinterpretCast<uint16_t>(), {mask}.ReinterpretCast<uint16_t>(), nt_predicate_prefix / 16);",
                        "            PipeBarrier<PIPE_V>();",
                    )
                )

            lines.extend(
                (
                    f"            Duplicate({scratch}, 1.0f, nt_predicate_prefix);",
                    "            PipeBarrier<PIPE_ALL>();",
                    f"            Select({destination}, {mask}, {scratch}, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, (uint32_t)nt_predicate_prefix);",
                )
            )
    else:
        condition, yes, no = args

        if operands[0].scalar:
            lines.extend(
                (
                    f"            Duplicate({scratch}, (({condition}) != 0.0f) ? 1.0f : 0.0f, nt_predicate_prefix);",
                    "            PipeBarrier<PIPE_V>();",
                    f"            CompareScalar({mask}, {scratch}, 0.0f, CMPMODE::GT, (uint32_t)nt_predicate_prefix);",
                )
            )
        else:
            lines.append(
                f"            CompareScalar({mask}, {condition}, 0.0f, CMPMODE::GT, (uint32_t)nt_predicate_prefix);"
            )

        lines.append("            PipeBarrier<PIPE_ALL>();")

        if operands[1].scalar:
            lines.extend(
                (
                    f"            Duplicate({scratch}, (float)({yes}), nt_predicate_prefix);",
                    "            PipeBarrier<PIPE_V>();",
                )
            )
            yes = scratch

        if operands[2].scalar:
            mode = "VSEL_TENSOR_SCALAR_MODE"
            no = f"(float)({no})"
        else:
            mode = "VSEL_TENSOR_TENSOR_MODE"

        lines.append(
            f"            Select({destination}, {mask}, {yes}, {no}, SELMODE::{mode}, (uint32_t)nt_predicate_prefix);"
        )

    lines.extend(
        (
            "            PipeBarrier<PIPE_ALL>();",
            "        }",
            f"        for (int32_t nt_predicate_lane = nt_predicate_prefix; nt_predicate_lane < ({count}); nt_predicate_lane++) {{",
            f"            {destination}.SetValue(nt_predicate_lane, (float)({expression}));",
            "        }",
            "        }",
        )
    )

    return "\n".join(lines)
