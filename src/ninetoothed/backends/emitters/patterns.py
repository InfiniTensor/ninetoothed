"""Shared pattern matchers for C-style SSA backends.

These matchers identify optimization opportunities (elementwise chains,
row reductions, softmax shapes) from the SSA program without any
backend-specific knowledge.  Backends consume the returned patterns and
emit target-specific code.
"""

from typing import Mapping


def match_reduce_broadcast(context) -> tuple | None:
    """Detect ``reduce(x) [+ scale] broadcast to the row shape`` patterns.

    Covers ``reduce(x) + x*0``, ``reduce(x) / n + x*0`` (mean), and
    ``sqrt(sum(x*x)) + x*0`` (norm2).

    Returns ``(operator, input, output, rows, cols, scale)``.
    """
    stores = [operation for operation in context.stores if len(operation.operands) == 2]

    if len(stores) != 1 or len(context.outputs) != 1:
        return None

    output = context.outputs[0]
    store = stores[0]

    if store.operands[1] != output:
        return None

    producer = context.operations.get(store.operands[0])

    if producer is None or producer.opcode != "arith.add":
        return None

    if len(producer.operands) != 2:
        return None

    reduce_result = None

    for operand in producer.operands:
        chain = _extract_reduce_chain(operand, context)

        if chain is not None:
            reduce_result = chain
            break

    if reduce_result is None:
        return None

    operator, x_name, scale = reduce_result
    input_info = context.tensors.get(x_name)
    output_info = context.tensors.get(output)

    if input_info is None or output_info is None:
        return None

    if input_info.ndim != 2 or output_info.ndim != 2:
        return None

    attrs = input_info.attrs or {}
    shape = attrs.get("source_shape") or input_info.shape

    if not shape or len(shape) != 2:
        return None

    if operator not in {"sum", "max", "min", "sum_sq"}:
        return None

    return (operator, x_name, output, str(shape[0]), str(shape[1]), scale)


def _extract_reduce_chain(value: str, context) -> tuple | None:
    """Walk a value back through wrappers to the reduction + scale.

    Returns ``(operator, input_name, scale_expr)``.
    """
    if value is None:
        return None

    node = context.operations.get(value)

    if node is None:
        return None

    opcode = node.opcode

    if opcode.startswith("reduce."):
        operator = opcode[len("reduce.") :]

        if operator not in {"sum", "max", "min"} or not node.operands:
            return None

        return (operator, node.operands[0], "")

    if opcode == "arith.div" and len(node.operands) == 2:
        inner = _extract_reduce_chain(node.operands[0], context)

        if inner is None:
            return None

        divisor = context.operations.get(node.operands[1])

        if divisor is not None and divisor.opcode == "arith.constant":
            value2 = divisor.attrs.get("value") if divisor.attrs else None

            if isinstance(value2, (int, float)) and value2 != 0:
                text = f"{float(value2):.9g}"

                if "." not in text:
                    text += ".0"

                return (inner[0], inner[1], " / " + text + "f")

        if divisor is not None and divisor.opcode == "shape.dim":
            dim = divisor.attrs.get("dim") if divisor.attrs else None

            if dim == 1:
                input_info = context.tensors.get(inner[1])

                if input_info is not None:
                    attrs = input_info.attrs or {}
                    shape = attrs.get("source_shape")

                    if shape and len(shape) == 2:
                        try:
                            cols = float(shape[1])
                            text = f"{cols:.9g}"

                            if "." not in text:
                                text += ".0"

                            return (inner[0], inner[1], " / " + text + "f")
                        except (ValueError, TypeError):
                            pass

        return None

    if opcode == "math.sqrt" and len(node.operands) == 1:
        sum_op = context.operations.get(node.operands[0])

        if sum_op is not None and sum_op.opcode == "reduce.sum" and sum_op.operands:
            sq_op = context.operations.get(sum_op.operands[0])

            if (
                sq_op is not None
                and sq_op.opcode == "arith.mul"
                and len(sq_op.operands) == 2
                and sq_op.operands[0] == sq_op.operands[1]
            ):
                return ("sum_sq", sq_op.operands[0], "")

        return None

    return None


def match_softmax(context) -> tuple | None:
    """Detect ``out = exp(x - max(x)) / sum(exp(x - max(x)))`` rows.

    Returns ``(input, output, rows, cols)`` or ``None``.
    """
    stores = [operation for operation in context.stores if len(operation.operands) == 2]

    if len(stores) != 1 or len(context.outputs) != 1:
        return None

    output = context.outputs[0]
    store = stores[0]

    if store.operands[1] != output:
        return None

    div = context.operations.get(store.operands[0])

    if div is None or div.opcode != "arith.div" or len(div.operands) != 2:
        return None

    num, den = div.operands
    exp_op = context.operations.get(num)

    if exp_op is None or exp_op.opcode not in {"math.exp", "call.exp"}:
        return None

    sub = context.operations.get(exp_op.operands[0])

    if sub is None or sub.opcode != "arith.sub" or len(sub.operands) != 2:
        return None

    x_name, max_result = sub.operands
    max_op = context.operations.get(max_result)

    if max_op is None or max_op.opcode != "reduce.max":
        return None

    if max_op.operands[0] != x_name:
        return None

    sum_op = context.operations.get(den)

    if sum_op is None or sum_op.opcode != "reduce.sum":
        return None

    if sum_op.operands[0] != num:
        return None

    input_info = context.tensors.get(x_name)
    output_info = context.tensors.get(output)

    if input_info is None or output_info is None:
        return None

    if input_info.ndim != 2 or output_info.ndim != 2:
        return None

    attrs = input_info.attrs or {}
    shape = attrs.get("source_shape") or input_info.shape

    if not shape or len(shape) != 2:
        return None

    return (x_name, output, str(shape[0]), str(shape[1]))


def constant_value(context, value: str) -> float | None:
    """Return the numeric value of an ``arith.constant`` operand."""
    producer = context.operations.get(value)

    if producer is None or producer.opcode != "arith.constant":
        return None

    raw = producer.attrs.get("value")

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None

    return float(raw)


def match_binary_chain(
    context,
    binary_ops: Mapping[str, str],
    *,
    scalar_ops: Mapping[str, str] | None = None,
    max_chain: int = 4,
) -> list[tuple] | None:
    """Detect a chain (or tree) of binary operations feeding a single store.

    ``binary_ops`` maps SSA opcodes to target function names;
    ``scalar_ops`` maps the same opcodes to their scalar-operand
    variants.  Leaf operands are staged tensor parameters or (when
    ``scalar_ops`` is given) numeric constants.

    Returns ``(steps, constants)`` where ``steps`` is a list of
    ``(opcode, intermediate, lhs, rhs, kind)`` tuples in topological
    execution order (``kind`` is ``"tensor"`` or ``"scalar"``) and
    ``constants`` maps constant operand names to their float values, or
    ``None``.
    """
    stores = [operation for operation in context.stores if len(operation.operands) == 2]

    if len(stores) != 1 or len(context.outputs) != 1:
        return None

    output = context.outputs[0]
    store = stores[0]

    if store.operands[1] != output:
        return None

    output_info = context.tensors.get(output)

    if output_info is None:
        return None

    ops: list[tuple] = []
    constants: dict[str, float] = {}
    collected: set[str] = set()
    visiting: set[str] = set()

    def covered(value: str) -> bool:
        producer = context.operations.get(value)

        if (
            producer is None
            or producer.opcode not in binary_ops
            or len(producer.operands) != 2
        ):
            info = context.tensors.get(value)

            if info is not None and info.ndim != 0:
                return True

            if scalar_ops is not None:
                constant = constant_value(context, value)

                if constant is not None:
                    constants[value] = constant

                    return True

            return False

        if value in collected:
            return True

        if value in visiting or len(ops) >= max_chain:
            return False

        visiting.add(value)

        lhs, rhs = producer.operands

        ok = covered(lhs) and covered(rhs)

        visiting.discard(value)

        if not ok:
            return False

        scalar_step = scalar_ops is not None and (lhs in constants or rhs in constants)

        if scalar_step and producer.opcode not in scalar_ops:
            return False

        intermediate = producer.results[0].name if producer.results else value
        safe = intermediate.replace("%", "v").replace("@", "i")
        ops.append(
            (
                producer.opcode,
                safe,
                lhs,
                rhs,
                "scalar" if scalar_step else "tensor",
            )
        )
        collected.add(intermediate)

        return True

    if not covered(store.operands[0]):
        return None

    if len(ops) < 2:
        return None

    return (ops, constants)


def is_pure_elementwise(context) -> bool:
    """Check that every top-level effect is an elementwise store."""
    for operation in context.operations.values():
        if operation.opcode == "mem.atomic_add":
            return False

        if operation.opcode.startswith("reduce."):
            return False

    return True


__all__ = [
    "match_reduce_broadcast",
    "match_softmax",
    "match_binary_chain",
    "is_pure_elementwise",
]
