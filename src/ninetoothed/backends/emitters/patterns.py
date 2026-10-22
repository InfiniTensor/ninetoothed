"""Shared pattern matchers for C-style SSA backends.

These matchers identify optimization opportunities (elementwise chains,
row reductions, softmax shapes) from the SSA program without any
backend-specific knowledge.  Backends consume the returned patterns and
emit target-specific code.
"""

import math
from dataclasses import dataclass
from typing import Mapping

from ninetoothed.backends.emitters.row_geometry import (
    canonical_row_view,
)
from ninetoothed.backends.emitters.row_geometry import (
    tensor_source_shape as _source_shape,
)


@dataclass(frozen=True)
class ReductionPattern:
    """A row reduction with its original broadcast term preserved.

    ``broadcast_zero`` identifies an input multiplied by zero. It is not
    numerically zero for nonfinite inputs, so renderers must retain that
    expression instead of discarding it. Iteration retains the original
    six-field matcher contract.
    """

    operator: str
    input: str
    output: str
    rows: str
    cols: str
    scale: str
    broadcast_zero: str | None = None

    def __iter__(self):
        return iter(
            (self.operator, self.input, self.output, self.rows, self.cols, self.scale)
        )

    def __getitem__(self, index):
        return tuple(self)[index]

    def __len__(self):
        return 6


def _single_plain_store(context):
    """Require one unconditional store with no additional kernel effects."""
    if len(context.stores) != 1 or len(context.outputs) != 1:
        return None

    store = context.stores[0]

    if len(store.operands) != 2 or store.operands[1] != context.outputs[0]:
        return None

    if store.attrs.get("indices") or store.attrs.get("source"):
        return None

    program = getattr(context.kernel, "ssa", None)

    if program is not None:
        for block in program.blocks:
            for operation in block.operations:
                if operation.regions or (
                    operation.opcode.startswith("mem.")
                    and operation.opcode not in {"mem.store", "mem.load"}
                ):
                    return None

    return store


def _row_tensor_pair(context, input_name, output_name):
    input_info = context.tensors.get(input_name)
    output_info = context.tensors.get(output_name)

    if input_info is None or output_info is None:
        return None

    if input_info.ndim != 2 or output_info.ndim != 2:
        return None

    if input_info.dtype != "float32" or output_info.dtype != "float32":
        return None

    input_shape = _source_shape(input_info)
    output_shape = _source_shape(output_info)

    if len(input_shape) != 2 or len(output_shape) != 2:
        return None

    if input_shape[0] != output_shape[0]:
        return None

    if not canonical_row_view(input_info, input_shape) or not canonical_row_view(
        output_info, output_shape
    ):
        return None

    return input_shape, output_shape


def _reduction_store_chain(context, store, *, allow_direct=False):
    """Match a row reduction plus a literal zero or an explicit zero product."""
    if allow_direct:
        direct = _extract_reduce_chain(store.operands[0], context)

        if direct is not None:
            return (*direct, None)

    producer = context.operations.get(store.operands[0])

    if producer is None or producer.opcode != "arith.add":
        return None

    if len(producer.operands) != 2:
        return None

    for index, operand in enumerate(producer.operands):
        chain = _extract_reduce_chain(operand, context)
        other = context.operations.get(producer.operands[1 - index])

        if chain is None or other is None:
            continue

        if other.opcode == "arith.constant" and other.attrs.get("value") == 0:
            return (*chain, None)

        if other.opcode != "arith.mul" or len(other.operands) != 2:
            continue

        for zero_index, child in enumerate(other.operands):
            constant = context.operations.get(child)

            if (
                constant is not None
                and constant.opcode == "arith.constant"
                and constant.attrs.get("value") == 0
                and other.operands[1 - zero_index] == chain[1]
            ):
                return (*chain, chain[1])

    return None


def match_reduce_broadcast(context, *, allow_direct=False) -> ReductionPattern | None:
    """Detect ``reduce(x) [+ scale] broadcast to the row shape`` patterns.

    Covers ``reduce(x) + x*0``, ``reduce(x) / n + x*0`` (mean), and
    ``sqrt(sum(x*x)) + x*0`` (norm2).

    Returns a ``ReductionPattern``, including any multiplied-zero term.
    """
    store = _single_plain_store(context)

    if store is None:
        return None

    chain = _reduction_store_chain(context, store, allow_direct=allow_direct)

    if chain is None:
        return None

    operator, x_name, scale, broadcast_zero = chain
    output = context.outputs[0]
    shapes = _row_tensor_pair(context, x_name, output)

    if shapes is None or shapes[0] != shapes[1]:
        return None

    rows, cols = shapes[0]

    return ReductionPattern(operator, x_name, output, rows, cols, scale, broadcast_zero)


def match_row_reduce(context, *, allow_direct=False) -> ReductionPattern | None:
    """Detect ``reduce(x) [+ scale]`` rows collapsed to a ``(rows, 1)`` output.

    Mirrors ``match_reduce_broadcast`` for row reductions whose store
    collapses each row to one scalar (``output.tile((1, 1))``).

    Returns a ``ReductionPattern``, including any multiplied-zero term.
    """
    store = _single_plain_store(context)

    if store is None:
        return None

    chain = _reduction_store_chain(context, store, allow_direct=allow_direct)

    if chain is None:
        return None

    operator, x_name, scale, broadcast_zero = chain
    output = context.outputs[0]
    shapes = _row_tensor_pair(context, x_name, output)

    if shapes is None or shapes[1][1] != "1":
        return None

    rows, cols = shapes[0]

    return ReductionPattern(operator, x_name, output, rows, cols, scale, broadcast_zero)


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

        if (
            operator not in {"sum", "max", "min"}
            or len(node.operands) != 1
            or node.attrs.get("axis") not in {1, -1}
            or not _scalar_row_result(node)
        ):
            return None

        return (operator, node.operands[0], "")

    if opcode == "arith.div" and len(node.operands) == 2:
        inner = _extract_reduce_chain(node.operands[0], context)

        if inner is None:
            return None

        divisor = context.operations.get(node.operands[1])

        if divisor is not None and divisor.opcode == "arith.constant":
            value2 = divisor.attrs.get("value") if divisor.attrs else None

            if (
                isinstance(value2, (int, float))
                and value2 != 0
                and math.isfinite(float(value2))
            ):
                text = f"{float(value2):.9g}"

                if "." not in text and "e" not in text.lower():
                    text += ".0"

                return (inner[0], inner[1], inner[2] + " / " + text + "f")

        if divisor is not None and divisor.opcode == "shape.dim":
            dim = divisor.attrs.get("dim") if divisor.attrs else None

            if dim == 1 and divisor.operands == (inner[1],):
                input_info = context.tensors.get(inner[1])

                if input_info is not None:
                    shape = _source_shape(input_info)

                    if shape and len(shape) == 2:
                        try:
                            cols = float(shape[1])
                            text = f"{cols:.9g}"

                            if "." not in text:
                                text += ".0"

                            return (inner[0], inner[1], inner[2] + " / " + text + "f")
                        except (ValueError, TypeError):
                            pass

        return None

    if opcode == "math.sqrt" and len(node.operands) == 1:
        sum_op = context.operations.get(node.operands[0])

        if (
            sum_op is not None
            and sum_op.opcode == "reduce.sum"
            and len(sum_op.operands) == 1
            and sum_op.attrs.get("axis") in {1, -1}
            and _scalar_row_result(sum_op)
        ):
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


def _scalar_row_result(operation):
    """Require one value per application row, not a vector of row results."""
    if len(operation.results) != 1:
        return False

    type_ = operation.results[0].type

    return type_.kind == "scalar" or (
        type_.kind == "tensor" and tuple(map(str, type_.shape)) == ("1",)
    )


def match_softmax(context) -> tuple | None:
    """Detect ``out = exp(x - max(x)) / sum(exp(x - max(x)))`` rows.

    Returns ``(input, output, rows, cols)`` or ``None``.
    """
    store = _single_plain_store(context)

    if store is None:
        return None

    output = context.outputs[0]

    div = context.operations.get(store.operands[0])

    if div is None or div.opcode != "arith.div" or len(div.operands) != 2:
        return None

    num, den = div.operands
    exp_op = context.operations.get(num)

    if (
        exp_op is None
        or exp_op.opcode not in {"math.exp", "call.exp"}
        or len(exp_op.operands) != 1
    ):
        return None

    sub = context.operations.get(exp_op.operands[0])

    if sub is None or sub.opcode != "arith.sub" or len(sub.operands) != 2:
        return None

    x_name, max_result = sub.operands
    max_op = context.operations.get(max_result)

    if (
        max_op is None
        or max_op.opcode != "reduce.max"
        or len(max_op.operands) != 1
        or max_op.attrs.get("axis") not in {1, -1}
        or not _scalar_row_result(max_op)
    ):
        return None

    if max_op.operands[0] != x_name:
        return None

    sum_op = context.operations.get(den)

    if (
        sum_op is None
        or sum_op.opcode != "reduce.sum"
        or len(sum_op.operands) != 1
        or sum_op.attrs.get("axis") not in {1, -1}
        or not _scalar_row_result(sum_op)
    ):
        return None

    if sum_op.operands[0] != num:
        return None

    shapes = _row_tensor_pair(context, x_name, output)

    if shapes is None or shapes[0] != shapes[1]:
        return None

    return (x_name, output, *shapes[0])


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
    store = _single_plain_store(context)

    if store is None:
        return None

    output = context.outputs[0]

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


__all__ = [
    "ReductionPattern",
    "canonical_row_view",
    "match_reduce_broadcast",
    "match_row_reduce",
    "match_softmax",
    "constant_value",
    "match_binary_chain",
]
