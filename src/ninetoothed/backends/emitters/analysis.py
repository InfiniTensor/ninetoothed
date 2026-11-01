"""SSA traversal and value-analysis helpers for backend emitters."""

from collections import defaultdict, deque
from collections.abc import Iterator, Mapping

from ninetoothed.backends.emitters.context import CooperativeDotPlan
from ninetoothed.compiler.effects import broadcast_view_parts, is_pointwise_expression
from ninetoothed.ir import Kernel, ssa


def operation_value_references(operation):
    """Include SSA values used by indexed stores as well as ordinary operands."""
    indices = (
        operation.attrs.get("indices", ()) if operation.opcode == "mem.store" else ()
    )

    return operation.operands + (
        (indices,) if isinstance(indices, str) else tuple(indices)
    )


class ScalarCarryProjectionAnalysis:
    """Resolve scalarized carry uses by lexical SSA identity, once per domain."""

    def __init__(self, program, is_effectful, *, live_operations=None):
        self.is_effectful = is_effectful
        self.uses = defaultdict(list)
        self.results = {}
        self.operands = {}
        self.definitions = {}
        self.arguments = {}
        self.yield_owners = {}
        self.types = {}
        self.cache = {}
        self.result_refs = {}
        self.value_projections = {}

        def visit(block, inherited, owner=None):
            visible = dict(inherited)
            arguments = tuple((id(block), value.name) for value in block.args)
            self.arguments[id(block)] = arguments

            for value, ref in zip(block.args, arguments):
                visible[value.name] = ref
                self.types[ref] = value.type

            for operation in block.operations:
                references = tuple(
                    visible.get(name, (0, name))
                    for name in operation_value_references(operation)
                )
                outputs = tuple((id(block), value.name) for value in operation.results)
                self.results[id(operation)] = outputs
                self.operands[id(operation)] = references[: len(operation.operands)]

                for slot, ref in enumerate(outputs):
                    self.definitions[ref] = (operation, slot)

                if live_operations is None or id(operation) in live_operations:
                    for index, ref in enumerate(references):
                        self.uses[ref].append((operation, index))

                if operation.opcode == "scf.yield":
                    self.yield_owners[id(operation)] = (owner, block)

                for region in operation.regions:
                    visit(region, visible, operation)

                for value, ref in zip(operation.results, outputs):
                    visible[value.name] = ref
                    self.types[ref] = value.type
                    self.result_refs[value.name] = ref

        visible = {}

        for value in program.inputs:
            ref = (id(program), value.name)
            visible[value.name] = ref
            self.types[ref] = value.type

        for block in program.blocks:
            visit(block, visible)

    def for_loop(self, loop, output_axes):
        key = (id(loop), output_axes)

        if key in self.cache:
            return self.cache[key]

        result = {}

        for slot, value in enumerate(loop.results):
            axes = tuple(map(str, value.type.shape))

            if not axes or len(axes) >= len(output_axes):
                continue

            seeds = (
                self.results[id(loop)][slot],
                self.arguments[id(loop.regions[0])][slot + 1],
                *self._initial_effect_values(self.operands[id(loop)][slot + 3], axes),
            )
            pending = deque(seeds)
            seen = set()
            projections = set()

            def forward(refs):
                for ref in refs:
                    if tuple(map(str, self.types[ref].shape)) != axes:
                        raise ValueError(
                            f"Scalarized carry {value.name} changes its logical domain."
                        )

                    pending.append(ref)

            while pending:
                ref = pending.popleft()

                if ref in seen:
                    continue

                seen.add(ref)

                for operation, operand_index in self.uses[ref]:
                    code = operation.opcode
                    outputs = self.results[id(operation)]

                    if code in {"dtype.of", "shape.dim", "tensor.stride"}:
                        continue

                    if code == "scf.yield":
                        owner, region = self.yield_owners[id(operation)]
                        forward((self.results[id(owner)][operand_index],))

                        if owner.opcode == "scf.for":
                            forward((self.arguments[id(region)][operand_index + 1],))

                        continue

                    if code == "scf.for" and operand_index >= 3:
                        index = operand_index - 3
                        forward(
                            (
                                outputs[index],
                                self.arguments[id(operation.regions[0])][index + 1],
                            )
                        )
                        continue

                    if code == "scf.if" and outputs:
                        forward(outputs)
                        continue

                    if code == "mem.store" and operand_index == 0:
                        projections.add(_logical_projection(axes, output_axes))
                        continue

                    if len(outputs) == 1:
                        consumer_axes = tuple(map(str, self.types[outputs[0]].shape))

                        if is_pointwise_expression(operation):
                            if consumer_axes == axes:
                                forward(outputs)
                                continue

                            projected = _logical_projection(consumer_axes, output_axes)
                            offset = len(consumer_axes) - len(axes)
                            projections.add(
                                tuple(
                                    None if axis == "1" else projected[index + offset]
                                    for index, axis in enumerate(axes)
                                )
                            )
                            continue

                        if code == "tensor.view":
                            parts = broadcast_view_parts(operation)

                            if parts is not None:
                                if consumer_axes == axes:
                                    forward(outputs)
                                else:
                                    projected = _logical_projection(
                                        consumer_axes, output_axes
                                    )
                                    projections.add(
                                        tuple(
                                            axis
                                            for part, axis in zip(parts, projected)
                                            if part != "None"
                                        )
                                    )

                                continue

                    raise ValueError(
                        f"Scalarized carry {value.name} has an unproven coordinate consumer {code}."
                    )

            if len(projections) > 1:
                raise ValueError(
                    f"Scalarized carry {value.name} requires multiple coordinate projections."
                )

            projection = (
                next(iter(projections))
                if projections
                else _logical_projection(axes, output_axes)
            )
            result[value.name] = projection
            result[loop.regions[0].args[slot + 1].name] = projection

            for ref in seen:
                previous = self.value_projections.setdefault(
                    (output_axes, ref), projection
                )

                if previous != projection:
                    raise ValueError(
                        "A materialized SSA value requires multiple coordinate projections."
                    )

        self.cache[key] = result

        return result

    def _initial_effect_values(self, initial, axes):
        """Find effectful initializers through coordinate-preserving value edges."""
        pending = [initial]
        seen = set()
        result = set()

        while pending:
            ref = pending.pop()

            if ref in seen or ref not in self.definitions:
                continue

            seen.add(ref)

            if tuple(map(str, self.types[ref].shape)) != axes:
                continue

            op, slot = self.definitions[ref]

            if op.opcode in {"scf.if", "scf.for"}:
                if self.is_effectful(op):
                    result.add(ref)

                pending.extend(
                    self.operands[id(region.operations[-1])][slot]
                    for region in op.regions
                )

                if op.opcode == "scf.for":
                    pending.append(self.operands[id(op)][slot + 3])
                else:
                    pending.append(self.operands[id(op)][0])
            elif is_pointwise_expression(op) or broadcast_view_parts(op) == (
                ":",
            ) * len(axes):
                pending.extend(self.operands[id(op)])

        return result

    def for_value(self, name, output_axes):
        return self.value_projections.get((output_axes, self.result_refs.get(name)))


def _logical_projection(axes, output_axes):
    """Project a broadcast domain onto logical output axes, before source emission."""
    offsets = tuple(dict.fromkeys((len(output_axes) - len(axes), 0)))

    for offset in offsets:
        if (
            offset >= 0
            and len(axes) + offset <= len(output_axes)
            and all(
                axis == "1" or axis == output_axes[index + offset]
                for index, axis in enumerate(axes)
            )
        ):
            return tuple(
                None if axis == "1" else index + offset
                for index, axis in enumerate(axes)
            )

    raise ValueError(f"Cannot prove projection of {axes} onto {output_axes}.")


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
