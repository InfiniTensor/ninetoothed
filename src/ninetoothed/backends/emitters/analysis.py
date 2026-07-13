"""SSA traversal and value-analysis helpers for backend emitters."""

from collections.abc import Iterator, Mapping

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
    "program_value_types",
    "schedule_int",
    "value_depends_on",
    "walk_ops",
]
