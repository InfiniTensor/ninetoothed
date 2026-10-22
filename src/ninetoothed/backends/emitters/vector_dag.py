"""Typed def-use and storage planning for proven flat or row vector domains.

The caller proves access maps, masks, effects and the common iteration extent.
This module plans values, target legality and buffer lifetimes; it never chooses
an operator implementation by kernel name or assumes target intrinsics exist.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class VectorValue:
    name: str
    dtype: str
    scalar: bool = False
    literal: int | float | None = None


@dataclass(frozen=True)
class VectorLowering:
    mode: str
    scratch_dtypes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VectorNode:
    operation: Any
    result: VectorValue
    operands: tuple[VectorValue, ...]
    lowering: VectorLowering


@dataclass(frozen=True)
class VectorStep:
    node: VectorNode
    destination: int | None
    operand_buffers: tuple[int | None, ...]


@dataclass(frozen=True)
class VectorInput:
    value: VectorValue
    buffer: int | None


@dataclass(frozen=True)
class VectorStore:
    output: str
    value: VectorValue
    buffer: int | None


@dataclass(frozen=True)
class VectorDAGPlan:
    inputs: tuple[VectorInput, ...]
    stores: tuple[VectorStore, ...]
    steps: tuple[VectorStep, ...]
    buffer_dtypes: tuple[str, ...]
    scratch_dtypes: tuple[str, ...]
    bytes_per_element: int
    reduction_axis: int | None = None


def plan_vector_dag(
    context,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    *,
    normalize_dtype: Callable[[str | None], str | None],
    classify: Callable[[Any, tuple[VectorValue, ...], str], VectorLowering | None],
    dtype_bytes: dict[str, int],
    scalar_inputs: Mapping[str, str] | None = None,
    reduction_axis: int | None = None,
    storage_dtype: Callable[[str], str] | None = None,
) -> VectorDAGPlan | None:
    """Plan all stores together, sharing producers and recycling dead buffers.

    Buffer reuse never aliases an operation's live operands and destination.
    Targets can therefore consume this plan without assuming that an intrinsic
    supports in-place operation. Exact dtype matching governs physical reuse.

    ``reduction_axis`` permits only reductions along that caller-proven row
    domain. Their results and operations consuming only scalar values execute
    once per row. A scalar store can then either broadcast over the row or write
    a collapsed output; the caller proves which store mapping is valid.
    ``scalar_inputs`` declares caller-proven uniform arguments (for example a
    row bias or a shape parameter). No shape or layout is guessed here.
    """
    scalar_inputs = scalar_inputs or {}
    storage_dtype = storage_dtype or (lambda dtype: dtype)

    if set(input_names).intersection(scalar_inputs):
        return None

    if len(context.stores) != len(output_names) or not context.stores:
        return None

    stores = []
    output_set = set(output_names)

    for store in context.stores:
        if (
            store.opcode != "mem.store"
            or len(store.operands) != 2
            or store.attrs.get("indices")
            or store.attrs.get("source")
        ):
            return None

        value, output = store.operands

        if output not in output_set:
            return None

        output_set.remove(output)
        stores.append((value, output))

    if output_set:
        return None

    values: dict[str, VectorValue] = {}
    nodes: list[VectorNode] = []
    active: set[str] = set()
    used_inputs: set[str] = set()

    def collect(name):
        if name in values:
            return values[name]

        if name in active:
            return None

        if name in scalar_inputs:
            dtype = normalize_dtype(scalar_inputs[name])

            if dtype is None:
                return None

            value = VectorValue(name, dtype, scalar=True)
            values[name] = value
            used_inputs.add(name)

            return value

        if name in input_names:
            info = context.tensors.get(name)

            if info is None:
                return None

            dtype = normalize_dtype(info.dtype)

            if dtype not in dtype_bytes:
                return None

            value = VectorValue(name, dtype)
            values[name] = value
            used_inputs.add(name)

            return value

        operation = context.operations.get(name)

        if operation is None or len(operation.results) != 1:
            return None

        if operation.opcode == "arith.constant":
            literal = operation.attrs.get("value")

            if isinstance(literal, bool) or not isinstance(literal, (int, float)):
                return None
            # Constants are scalar operands. A target decides the legal
            # conversion/promotion for each consuming operation.

            dtype = normalize_dtype(operation.results[0].type.dtype)

            if dtype is None:
                dtype = "float32" if isinstance(literal, float) else "int64"

            value = VectorValue(name, dtype, scalar=True, literal=literal)
            values[name] = value

            return value

        active.add(name)
        operands = tuple(collect(operand) for operand in operation.operands)
        active.remove(name)

        if any(operand is None for operand in operands):
            return None

        is_reduction = operation.opcode.startswith("reduce.")

        if is_reduction:
            axis = operation.attrs.get("axis")

            if isinstance(axis, int) and axis < 0 and len(operation.operands) == 1:
                operand_name = operation.operands[0]
                info = context.tensors.get(operand_name)
                producer = context.operations.get(operand_name)
                rank = getattr(info, "ndim", None)

                if rank is None and producer is not None:
                    rank = len(producer.results[0].type.shape) or None

                if rank is not None:
                    axis += rank

            if (
                reduction_axis is None
                or operation.opcode not in {"reduce.sum", "reduce.max", "reduce.min"}
                or len(operands) != 1
                or operands[0].scalar
                or axis != reduction_axis
            ):
                return None

        dtype = normalize_dtype(operation.results[0].type.dtype)

        if dtype is None and operands:
            # A literal does not establish an unknown operand's type. Targets
            # still decide the legality of promotion and casts via classify.
            promotion_operands = (
                operands[1:] if operation.opcode == "select.where" else operands
            )
            known = {
                operand.dtype
                for operand in promotion_operands
                if operand.literal is None
            }

            if operation.opcode in {
                "cmp.lt",
                "cmp.le",
                "cmp.gt",
                "cmp.ge",
                "cmp.eq",
                "cmp.ne",
                "arith.lt",
                "arith.le",
                "arith.gt",
                "arith.ge",
                "arith.eq",
                "arith.ne",
            }:
                dtype = "bool"
            elif (
                operation.opcode == "select.where"
                and len(promotion_operands) == 2
                and all(
                    operand.dtype == "float32" and isinstance(operand.literal, float)
                    for operand in promotion_operands
                )
            ):
                dtype = "float32"
            elif (
                "float32" in known
                and known <= {"float32", "index"}
                and all(
                    value.dtype != "index" or value.scalar
                    for value in promotion_operands
                )
            ):
                dtype = "float32"

        if dtype not in dtype_bytes:
            return None

        lowering = classify(operation, operands, dtype)

        if lowering is None or any(
            scratch not in dtype_bytes for scratch in lowering.scratch_dtypes
        ):
            return None

        scalar = is_reduction or (
            bool(operands)
            and all(operand.scalar for operand in operands)
            and operation.opcode not in {"tensor.full", "tensor.zeros", "tensor.ones"}
        )
        value = VectorValue(name, dtype, scalar=scalar)
        values[name] = value
        nodes.append(VectorNode(operation, value, operands, lowering))

        return value

    try:
        for value, output in stores:
            resolved = collect(value)
            output_dtype = normalize_dtype(context.tensors[output].dtype)

            if (
                resolved is None
                or resolved.dtype != output_dtype
                or (resolved.scalar and reduction_axis is None and not scalar_inputs)
            ):
                return None
    except RecursionError:
        return None

    if any(
        not value.scalar and storage_dtype(value.dtype) not in dtype_bytes
        for value in values.values()
    ):
        return None

    uses = Counter(
        operand.name
        for node in nodes
        for operand in node.operands
        if not operand.scalar
    )
    uses.update(value for value, _ in stores if not values[value].scalar)
    buffer_dtypes: list[str] = []
    free: dict[str, list[int]] = {}
    bindings: dict[str, int] = {}

    def allocate(dtype):
        dtype = storage_dtype(dtype)

        if free.get(dtype):
            return free[dtype].pop()

        buffer = len(buffer_dtypes)
        buffer_dtypes.append(dtype)

        return buffer

    inputs = []

    for name in (*input_names, *scalar_inputs):
        if name not in used_inputs:
            continue

        dtype = values[name].dtype
        buffer = None if values[name].scalar else allocate(dtype)

        if buffer is not None:
            bindings[name] = buffer

        inputs.append(VectorInput(values[name], buffer))

    steps = []
    scratch_dtypes = []

    for node in nodes:
        destination = None if node.result.scalar else allocate(node.result.dtype)
        operand_buffers = tuple(
            None if operand.scalar else bindings[operand.name]
            for operand in node.operands
        )
        steps.append(VectorStep(node, destination, operand_buffers))

        if destination is not None:
            bindings[node.result.name] = destination

        for operand in node.operands:
            if operand.scalar:
                continue

            uses[operand.name] -= 1

            if not uses[operand.name]:
                free.setdefault(storage_dtype(operand.dtype), []).append(
                    bindings[operand.name]
                )
        # A target's node-local scratch is shared across successive nodes.
        # Preserve multiplicity when one node requires two same-type buffers.

        required = Counter(node.lowering.scratch_dtypes)
        existing = Counter(scratch_dtypes)

        for dtype, count in required.items():
            scratch_dtypes.extend([dtype] * max(0, count - existing[dtype]))

    mapped_stores = tuple(
        VectorStore(output, values[value], bindings.get(value))
        for value, output in stores
    )
    physical_types = (*buffer_dtypes, *scratch_dtypes)

    return VectorDAGPlan(
        inputs=tuple(inputs),
        stores=mapped_stores,
        steps=tuple(steps),
        buffer_dtypes=tuple(buffer_dtypes),
        scratch_dtypes=tuple(scratch_dtypes),
        bytes_per_element=sum(dtype_bytes[dtype] for dtype in physical_types),
        reduction_axis=reduction_axis,
    )
