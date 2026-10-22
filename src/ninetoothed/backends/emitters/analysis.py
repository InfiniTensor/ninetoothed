"""SSA traversal and value-analysis helpers for backend emitters."""

from collections.abc import Iterator, Mapping

from ninetoothed.backends.emitters.context import CooperativeDotPlan
from ninetoothed.ir import Kernel, ssa


def schedule_int(kernel: Kernel, name: str, default: int) -> int:
    """Return one positive integer schedule value from the lowered program."""
    schedule = dict(kernel.ssa.metadata.get("schedule", {})) if kernel.ssa else {}
    value = schedule.get(name, default)

    if isinstance(value, bool):
        raise ValueError(
            f"Schedule `{name}` must be a positive integer, got {value!r}."
        )

    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Schedule `{name}` must be a positive integer, got {value!r}."
        ) from exc

    if normalized < 1:
        raise ValueError(f"Schedule `{name}` must be positive, got {normalized}.")
    return normalized


def value_depends_on(
    value: str,
    dependency: str,
    operations: Mapping[str, ssa.Operation],
    seen: set[str] | None = None,
) -> bool:
    if value == dependency:
        return True

    if not value.startswith("%"):
        return False

    seen = set() if seen is None else seen

    if value in seen:
        return False

    seen.add(value)
    producer = operations.get(value)

    return bool(
        producer is not None
        and any(
            value_depends_on(operand, dependency, operations, seen)
            for operand in producer.operands
        )
    )


def walk_ops(operations: tuple[ssa.Operation, ...]) -> Iterator[ssa.Operation]:
    for operation in operations:
        yield operation

        for region in operation.regions:
            yield from walk_ops(region.operations)


def dot_accumulation_plan(program: ssa.Program) -> CooperativeDotPlan | None:
    """Match a whole, single-output dot accumulation without assuming tile sizes.

    The replacement emits the loop and its output epilogue. Reject all other
    effects and collectives, including ones outside that output's dependency
    chain, so matching one loop cannot discard another part of the program.
    """
    if len(program.blocks) != 1:
        return None

    root = program.blocks[0].operations
    walked = tuple(walk_ops(root))
    loops = [op for op in walked if op.opcode == "scf.for"]
    dots = [op for op in walked if op.opcode in {"linalg.dot", "linalg.matmul"}]
    stores = [op for op in walked if op.opcode == "mem.store"]

    if len(loops) != 1 or len(dots) != 1 or len(stores) != 1:
        return None

    loop, dot, store = loops[0], dots[0], stores[0]

    if (
        loop not in root
        or store not in root
        or len(loop.results) != 1
        or len(loop.regions) != 1
        or len(store.operands) != 2
        or dot not in loop.regions[0].operations
        or len(dot.results) != 1
    ):
        return None

    for op in walked:
        if op is loop or op is store:
            continue

        if op.regions or not (
            op.opcode.startswith(("arith.", "cmp.", "math.", "dtype.", "shape."))
            or op.opcode
            in {
                "index.offset",
                "tensor.zeros",
                "tensor.full",
                "tensor.cast",
                "tensor.extract",
                "tensor.view",
                "select.where",
                "linalg.transpose",
                "linalg.dot",
                "linalg.matmul",
                "scf.yield",
            }
        ):
            return None

    operations = {value.name: op for op in walked for value in op.results}
    args = tuple(loop.attrs.get("iter_args", ()))

    if len(args) != 1 or len(loop.operands) < 4:
        return None

    initial = operations.get(str(args[0].get("initial", "")))

    if initial is None or not (
        initial.opcode == "tensor.zeros"
        or initial.opcode in {"tensor.full", "arith.constant"}
        and initial.attrs.get("value") == 0
    ):
        return None

    carry = str(args[0].get("block_arg", ""))
    yielded = loop.regions[0].operations[-1]
    add = operations.get(yielded.operands[0]) if yielded.operands else None

    if (
        yielded.opcode != "scf.yield"
        or len(yielded.operands) != 1
        or add is None
        or add.opcode != "arith.add"
        or len(add.operands) != 2
        or set(add.operands) != {carry, dot.results[0].name}
        or any(value_depends_on(name, carry, operations) for name in dot.operands)
        or not value_depends_on(store.operands[0], loop.results[0].name, operations)
    ):
        return None
    # Intermediate loop values cannot be re-evaluated outside their region.

    body_values = {
        value.name for op in loop.regions[0].operations for value in op.results
    }
    pending = [store.operands[0]]
    seen = {loop.results[0].name}

    while pending:
        name = pending.pop()

        if name in seen:
            continue

        seen.add(name)

        if name in body_values:
            return None

        producer = operations.get(name)

        if producer is not None:
            if producer.opcode in {
                "linalg.transpose",
                "tensor.view",
                "tensor.extract",
            } and value_depends_on(name, loop.results[0].name, operations):
                return None

            pending.extend(producer.operands)
    return CooperativeDotPlan(loop=loop, dot=dot, store=store)


def atomic_output_tensors(
    operations: tuple[ssa.Operation, ...],
    op_by_result: Mapping[str, ssa.Operation],
) -> tuple[str, ...]:
    outputs: list[str] = []

    for operation in operations:
        if operation.opcode != "mem.atomic_add" or not operation.operands:
            continue

        pointer = op_by_result.get(operation.operands[0])

        if pointer is None or pointer.opcode != "mem.data_ptr" or not pointer.operands:
            continue

        tensor = pointer.operands[0]

        if tensor not in outputs:
            outputs.append(tensor)
    return tuple(outputs)


def program_value_types(program: ssa.Program) -> dict[str, ssa.Type]:
    value_types = {
        value.name: value.type for value in (*program.inputs, *program.outputs)
    }

    for block in program.blocks:
        collect_value_types(block, value_types)
    return value_types


def collect_value_types(
    block: ssa.Block,
    value_types: dict[str, ssa.Type],
) -> None:
    value_types.update({argument.name: argument.type for argument in block.args})

    for operation in block.operations:
        value_types.update({result.name: result.type for result in operation.results})

        for region in operation.regions:
            collect_value_types(region, value_types)


__all__ = [
    "atomic_output_tensors",
    "collect_value_types",
    "dot_accumulation_plan",
    "program_value_types",
    "schedule_int",
    "value_depends_on",
    "walk_ops",
]
