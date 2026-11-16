"""Typed pure def-use plans with target-independent buffer lifetimes."""

from collections import Counter
from dataclasses import dataclass

from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import ssa


@dataclass(frozen=True)
class VectorValue:
    name: str
    dtype: str
    scalar: bool = False
    literal: int | float | None = None


@dataclass(frozen=True)
class VectorStep:
    operation: object
    result: VectorValue
    operands: tuple[VectorValue, ...]
    mode: str
    destination: int | None
    sources: tuple[int | None, ...]


@dataclass(frozen=True)
class VectorDAG:
    inputs: tuple[tuple[VectorValue, int], ...]
    steps: tuple[VectorStep, ...]
    stores: tuple[tuple[str, int, VectorValue], ...]
    buffer_dtypes: tuple[str, ...]


def plan_vector_dag(
    context, classify, *, supported_dtypes, storage_dtype=lambda dtype: dtype
):
    """Plan proved accesses; the caller owns effect and iteration-domain proofs.

    Allocate each destination before releasing its operands, so targets need
    not assume SDK primitives permit in-place execution. Scalar literals and
    parameters do not allocate broadcast buffers. Reuse requires exact dtype.
    """
    values, nodes, active = {}, [], set()
    used = set()

    def collect(name):
        if name in values:
            return values[name]

        if name in active:
            return None

        if name in context.variables:
            info = context.tensors[name]
            dtype = normalize_dtype(info.dtype)

            if dtype not in supported_dtypes:
                return None

            value = VectorValue(name, dtype, scalar=info.ndim == 0)
            values[name] = value
            used.add(name)

            return value

        operation = context.operations.get(name)

        if operation is None or len(operation.results) != 1 or operation.regions:
            return None

        if operation.opcode == "arith.constant":
            literal = operation.attrs.get("value")

            if isinstance(literal, bool) or not isinstance(literal, (int, float)):
                return None

            dtype = normalize_dtype(operation.results[0].type.dtype)
            value = VectorValue(name, dtype or "float32", True, literal)
            values[name] = value

            return value

        if tuple(map(str, operation.results[0].type.shape)) != context.axes:
            return None

        active.add(name)
        operands = tuple(collect(n) for n in operation.operands)
        active.remove(name)

        if not operands or any(value is None for value in operands):
            return None

        dtype = normalize_dtype(operation.results[0].type.dtype)

        if dtype is None:
            known = {value.dtype for value in operands if not value.scalar}

            if operation.opcode.startswith("cmp."):
                dtype = "bool"
            elif operation.opcode == "select.where":
                known = {value.dtype for value in operands[1:] if not value.scalar}
                dtype = next(iter(known)) if len(known) == 1 else None
            elif len(known) == 1:
                dtype = next(iter(known))

        mode = classify(operation, operands, dtype)

        if mode is None or all(value.scalar for value in operands):
            return None

        value = VectorValue(name, dtype)
        values[name] = value
        nodes.append((operation, value, operands, mode))

        return value

    stores = []

    try:
        for store in context.stores:
            value, output = store.operands
            resolved = collect(value)

            if resolved is None or resolved.scalar:
                return None

            output_dtype = normalize_dtype(context.tensors[output].dtype)

            if resolved.dtype != output_dtype:
                # A store is itself a typed boundary, even when the source
                # graph has no explicit cast operation before mem.store.
                cast_name = f"%vector_store_{len(stores)}"

                while cast_name in context.operations or cast_name in values:
                    cast_name += "_"

                cast = ssa.Operation(
                    opcode="tensor.cast",
                    operands=(resolved.name,),
                    results=(
                        ssa.Value(
                            name=cast_name,
                            type=ssa.Type(
                                kind="tensor", shape=context.axes, dtype=output_dtype
                            ),
                        ),
                    ),
                )
                mode = classify(cast, (resolved,), output_dtype)

                if mode is None:
                    return None

                result = VectorValue(cast_name, output_dtype)
                values[cast_name] = result
                nodes.append((cast, result, (resolved,), mode))
                resolved = result

            stores.append((output, resolved))
    except RecursionError:
        return None

    if not nodes:
        return None

    uses = Counter(value.name for _, _, operands, _ in nodes for value in operands)
    uses.update(value.name for _, value in stores)
    buffers, free, bindings, inputs, steps = [], {}, {}, [], []

    def allocate(dtype):
        dtype = storage_dtype(dtype)

        if free.get(dtype):
            return free[dtype].pop()

        result = len(buffers)
        buffers.append(dtype)

        return result

    for name in context.variables:
        if name not in used or values[name].scalar:
            continue

        value = values[name]
        index = allocate(value.dtype)
        bindings[name] = index
        inputs.append((value, index))

    for operation, result, operands, mode in nodes:
        destination = allocate(result.dtype)
        sources = tuple(None if v.scalar else bindings[v.name] for v in operands)
        steps.append(
            VectorStep(operation, result, operands, mode, destination, sources)
        )
        bindings[result.name] = destination

        for value in operands:
            uses[value.name] -= 1

            if not value.scalar and uses[value.name] == 0:
                free.setdefault(storage_dtype(value.dtype), []).append(
                    bindings[value.name]
                )

    return VectorDAG(
        tuple(inputs),
        tuple(steps),
        tuple((output, bindings[value.name], value) for output, value in stores),
        tuple(buffers),
    )
