"""Backend-neutral reduction-domain analysis for SSA scheduling."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ninetoothed.ir import ssa

_ELEMENTWISE_USERS = frozenset(
    {
        "select.where",
        "tensor.cast",
        "tensor.view",
    }
)
_SUPPORTED_REDUCTIONS = frozenset({"reduce.sum", "reduce.max", "reduce.min"})


@dataclass(frozen=True, kw_only=True)
class ReductionDomain:
    """The iteration domain of one fine-grained SSA reduction."""

    result: str
    operand: str
    operator: str
    axis: int
    input_shape: tuple[str, ...]
    result_shape: tuple[str, ...]
    scope: tuple[int, ...]
    store_shapes: tuple[tuple[str, ...], ...] = ()
    program_shapes: tuple[tuple[str, ...], ...] = ()
    program_constraints: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()
    program_compatible: bool = True

    @property
    def extent(self) -> str:
        return self.input_shape[self.axis]

    @property
    def parallel_axes(self) -> tuple[int, ...]:
        return tuple(axis for axis in range(len(self.input_shape)) if axis != self.axis)

    @property
    def parallel_shape(self) -> tuple[str, ...]:
        return tuple(self.input_shape[axis] for axis in self.parallel_axes)

    def as_metadata(self) -> Mapping[str, Any]:
        return {
            "result": self.result,
            "operand": self.operand,
            "operator": self.operator,
            "axis": self.axis,
            "input_shape": self.input_shape,
            "result_shape": self.result_shape,
            "scope": self.scope,
            "store_shapes": self.store_shapes,
            "program_shapes": self.program_shapes,
            "program_constraints": self.program_constraints,
            "program_compatible": self.program_compatible,
            "extent": self.extent,
            "parallel_axes": self.parallel_axes,
            "parallel_shape": self.parallel_shape,
        }


def analyze_reductions(program: ssa.Program, tensors=()) -> Mapping[str, Any]:
    """Return reduction domains and a conservative shared schedule contract."""
    value_types = _value_types(program)
    scoped_operations = tuple(_walk_program(program))
    users = _users(scoped_operations)
    definitions = _definitions(scoped_operations)
    forwarded_values = _forwarded_values(program)
    tensor_specs = {tensor.name: tensor for tensor in tensors}
    domains = []
    rejections = []

    for operation, scope in scoped_operations:
        if not operation.opcode.startswith("reduce."):
            continue

        if operation.opcode not in _SUPPORTED_REDUCTIONS:
            raise ValueError(f"Unsupported SSA reduction `{operation.opcode}`.")

        if operation.operands:
            operand_type = value_types.get(operation.operands[0])

            if operand_type is not None and operand_type.kind == "scalar":
                _validate_scalar_fallback(operation)
                continue

        domain, rejection = _domain(
            operation,
            scope,
            value_types,
            users,
            definitions,
            forwarded_values,
            tensor_specs,
        )
        domains.append(domain)

        if rejection is not None:
            rejections.append({"result": domain.result, "reason": rejection})

    schedule = _schedule_contract(tuple(domains), tuple(rejections))

    return {
        "reduction_domains": tuple(domain.as_metadata() for domain in domains),
        "reduction_rejections": tuple(rejections),
        "reduction_schedule": schedule,
    }


def _validate_scalar_fallback(operation):
    if len(operation.operands) != 1 or len(operation.results) != 1:
        raise ValueError(
            f"SSA reduction `{operation.opcode}` requires one operand and one result."
        )

    axis = operation.attrs.get("axis")

    if isinstance(axis, bool) or axis not in {None, -1, 0}:
        raise ValueError("Scalar fallback reduction axis must be omitted, -1, or 0.")

    result = operation.results[0]

    if result.type.kind != "scalar" or result.type.shape:
        raise ValueError(
            f"Scalar fallback reduction result `{result.name}` must be scalar."
        )


def _domain(
    operation,
    scope,
    value_types,
    users,
    definitions,
    forwarded_values,
    tensor_specs,
):
    if len(operation.operands) != 1 or len(operation.results) != 1:
        raise ValueError(
            f"SSA reduction `{operation.opcode}` requires one operand and one result."
        )

    operand = operation.operands[0]
    result = operation.results[0]
    operand_type = value_types.get(operand)

    if operand_type is None or operand_type.kind != "tensor" or not operand_type.shape:
        raise ValueError(
            f"SSA reduction `{operation.opcode}` requires a ranked tensor operand."
        )

    input_shape = tuple(str(dim) for dim in operand_type.shape)
    raw_axis = operation.attrs.get("axis")
    rejection = None

    if raw_axis is None:
        non_unit_axes = tuple(
            index for index, extent in enumerate(input_shape) if extent != "1"
        )

        if len(non_unit_axes) != 1:
            axis = 0
            rejection = "full reduction does not have exactly one non-unit axis"
        else:
            axis = non_unit_axes[0]

        expected_shape = ()
    else:
        if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
            raise ValueError("Reduction axis must be a compile-time integer.")

        axis = raw_axis + len(input_shape) if raw_axis < 0 else raw_axis

        if axis < 0 or axis >= len(input_shape):
            raise ValueError(
                f"Reduction axis {raw_axis} is outside tensor rank {len(input_shape)}."
            )

        expected_shape = input_shape[:axis] + input_shape[axis + 1 :]

    result_shape = tuple(str(dim) for dim in result.type.shape)
    expected_kind = "tensor" if expected_shape else "scalar"

    if result.type.kind != expected_kind or result_shape != expected_shape:
        raise ValueError(
            f"Reduction result `{result.name}` has type {result.type.kind}"
            f"{result_shape}, expected {expected_kind}{expected_shape}."
        )

    compatible, consumer_reason = _consumer_contract(
        result.name,
        users,
        axis=axis,
        input_shape=input_shape,
        result_shape=result_shape,
        value_types=value_types,
    )
    stores = _reachable_stores(
        result.name,
        users,
        value_types,
        definitions,
        forwarded_values,
    )
    store_shapes = tuple(sorted({shape for _target, shape in stores}))
    (
        program_shapes,
        program_constraints,
        program_compatible,
        program_reason,
    ) = _program_contract(
        operand,
        definitions,
        stores,
        tensor_specs,
    )

    if not compatible and rejection is None:
        rejection = consumer_reason

    if program_reason is not None and rejection is None:
        rejection = program_reason

    if len(program_shapes) > 1 and rejection is None:
        rejection = "reduction stores do not share one outer program domain"

    if scope and rejection is None:
        rejection = "nested-region reductions require scalar fallback"

    return (
        ReductionDomain(
            result=result.name,
            operand=operand,
            operator=operation.opcode.removeprefix("reduce."),
            axis=axis,
            input_shape=input_shape,
            result_shape=result_shape,
            scope=scope,
            store_shapes=store_shapes,
            program_shapes=program_shapes,
            program_constraints=program_constraints,
            program_compatible=program_compatible,
        ),
        rejection,
    )


def _consumer_contract(
    result,
    users,
    *,
    axis,
    input_shape,
    result_shape,
    value_types,
):
    broadcast_shape = input_shape[:axis] + ("1",) + input_shape[axis + 1 :]
    allowed_shapes = {(), result_shape, broadcast_shape, input_shape}
    pending = [(result, _is_right_aligned(axis, input_shape, result_shape))]
    seen = set()

    while pending:
        value, broadcast_ready = pending.pop()
        state = (value, broadcast_ready)

        if state in seen:
            continue

        seen.add(state)

        for operation in users.get(value, ()):
            consumer_broadcast_ready = broadcast_ready

            if operation.opcode == "mem.store":
                stored_type = value_types.get(value)
                target_type = value_types.get(operation.operands[1])

                if (
                    stored_type is not None
                    and target_type is not None
                    and stored_type.shape != target_type.shape
                ):
                    return False, (
                        "store target does not match the reduction value domain"
                    )

                continue

            if operation.opcode.startswith("reduce."):
                continue

            if not _supported_consumer(operation.opcode) or not operation.results:
                return False, (
                    f"consumer `{operation.opcode}` is not row-vector compatible"
                )

            if operation.opcode == "tensor.view":
                source_type = value_types.get(value)
                source_shape = (
                    ()
                    if source_type is None
                    else tuple(str(dim) for dim in source_type.shape)
                )

                view_contract = _broadcast_view_contract(operation, axis, source_shape)

                if view_contract is None:
                    return False, ("tensor view does not preserve the reduction domain")

                consumer_broadcast_ready |= view_contract

            for consumer_result in operation.results:
                shape = tuple(str(dim) for dim in consumer_result.type.shape)

                if shape not in allowed_shapes:
                    return False, "consumer shape is not in the reduction domain"

                if shape == input_shape and not consumer_broadcast_ready:
                    return False, ("consumer broadcast does not preserve parallel axes")

                pending.append(
                    (
                        consumer_result.name,
                        consumer_broadcast_ready or shape in {(), input_shape},
                    )
                )

    return True, None


def _reachable_stores(
    result,
    users,
    value_types,
    definitions,
    forwarded_values,
):
    pending = [result]
    seen = set()
    stores = set()

    while pending:
        value = pending.pop()

        if value in seen:
            continue

        seen.add(value)
        pending.extend(forwarded_values.get(value, ()))

        for operation in users.get(value, ()):
            if operation.opcode in {"mem.store", "mem.atomic_add"}:
                target, target_type = _effect_target(
                    operation,
                    value_types,
                    definitions,
                )

                if target is not None and target_type is not None:
                    stores.add((target, tuple(str(dim) for dim in target_type.shape)))

                continue

            pending.extend(item.name for item in operation.results)

            for block in operation.regions:
                stores.update(_region_stores(block, value_types, definitions))

    return tuple(sorted(stores))


def _region_stores(block, value_types, definitions):
    stores = set()

    for operation in block.operations:
        if operation.opcode in {"mem.store", "mem.atomic_add"}:
            target, target_type = _effect_target(operation, value_types, definitions)

            if target is not None and target_type is not None:
                stores.add((target, tuple(str(dim) for dim in target_type.shape)))

        for region in operation.regions:
            stores.update(_region_stores(region, value_types, definitions))

    return stores


def _effect_target(operation, value_types, definitions):
    target_index = 1 if operation.opcode == "mem.store" else 0
    target = operation.operands[target_index]
    target_type = value_types.get(target)
    definition = definitions.get(target)

    if definition is not None and definition.opcode == "mem.data_ptr":
        target = definition.operands[0]
        target_type = value_types.get(target)

    return target, target_type


def _program_contract(
    operand,
    definitions,
    stores,
    tensor_specs,
):
    producer_targets = _producer_targets(operand, definitions, tensor_specs)
    producer_specs = tuple(tensor_specs[target] for target in producer_targets)
    store_specs = tuple(tensor_specs.get(target) for target, _shape in stores)
    legacy_global_domain = (
        producer_specs
        and all(tensor.layout is None for tensor in producer_specs)
        and store_specs
        and all(tensor is not None and tensor.layout is None for tensor in store_specs)
    )

    if legacy_global_domain:
        return ((),), (((), ()),), True, None

    if not producer_specs or any(tensor.layout is None for tensor in producer_specs):
        return (
            (),
            (),
            True,
            "reduction operand has no structured Layout IR program domain",
        )

    producer_shapes = tuple(
        sorted(
            {
                tuple(expression.render() for expression in tensor.layout.view_shape)
                for tensor in producer_specs
            }
        )
    )
    shapes = set()
    constraints = []
    compatible = True
    reason = None

    for target, _store_shape in stores:
        tensor = tensor_specs.get(target)

        if tensor is None or tensor.layout is None:
            return (
                (),
                (),
                True,
                "store target has no structured Layout IR program domain",
            )

        store_program_shape = tuple(
            expression.render() for expression in tensor.layout.view_shape
        )

        shapes.add(store_program_shape)

        for producer_shape in producer_shapes:
            constraints.append((store_program_shape, producer_shape))
            actual_static = _static_product(store_program_shape)
            expected_static = _static_product(producer_shape)

            if (
                actual_static is not None
                and expected_static is not None
                and actual_static != expected_static
            ):
                compatible = False
                reason = "reduction operand and store program domains do not match"

    if len(shapes) > 1:
        compatible = False
        reason = "reduction stores do not share one outer program domain"

    return (
        tuple(sorted(shapes)),
        tuple(sorted(constraints)),
        compatible,
        reason,
    )


def _producer_targets(value, definitions, tensor_specs):
    pending = [value]
    seen = set()
    targets = set()

    while pending:
        current = pending.pop()

        if current in seen:
            continue

        seen.add(current)
        tensor = tensor_specs.get(current)

        if tensor is not None:
            if tensor.ndim > 0 and not tensor.constexpr:
                targets.add(current)

            continue

        operation = definitions.get(current)

        if operation is not None:
            pending.extend(operation.operands)

    return tuple(sorted(targets))


def _static_product(shape):
    values = tuple(_static_integer_expression(expression) for expression in shape)

    if any(value is None for value in values):
        return None

    result = 1

    for value in values:
        result *= value
    return result


def _static_integer_expression(expression):
    try:
        import sympy

        value = sympy.sympify(expression)

        return None if value.free_symbols else int(value)
    except (TypeError, ValueError):
        return None


def _is_right_aligned(axis, input_shape, result_shape):
    if not result_shape:
        return True

    mapped_axes = tuple(range(len(input_shape) - len(result_shape), len(input_shape)))
    parallel_axes = tuple(index for index in range(len(input_shape)) if index != axis)

    return mapped_axes == parallel_axes


def _broadcast_view_contract(operation, axis, source_shape):
    subscript = str(operation.attrs.get("subscript", "")).strip().strip("()")
    parts = tuple(part.strip() for part in subscript.split(",") if part.strip())

    if not parts or any(part not in {":", "None"} for part in parts):
        return None

    new_axes = tuple(index for index, part in enumerate(parts) if part == "None")
    result_shape = tuple(str(dim) for dim in operation.results[0].type.shape)

    if sum(part != "None" for part in parts) != len(source_shape):
        return None

    if not new_axes:
        return False if result_shape == source_shape else None

    compatible = new_axes == (axis,) and result_shape == (
        source_shape[:axis] + ("1",) + source_shape[axis:]
    )

    return True if compatible else None


def _supported_consumer(opcode):
    return (
        opcode in _ELEMENTWISE_USERS
        or opcode.startswith("arith.")
        or opcode.startswith("cmp.")
        or opcode.startswith("math.")
    )


def _schedule_contract(domains, rejections):
    if not domains:
        return {"mode": "none"}

    live_domains = tuple(domain for domain in domains if domain.store_shapes)

    if not live_domains:
        return {"mode": "none"}

    live_results = {domain.result for domain in live_domains}
    live_rejections = tuple(
        rejection for rejection in rejections if rejection["result"] in live_results
    )
    store_domains = {
        store_shape for domain in live_domains for store_shape in domain.store_shapes
    }
    program_domains = {
        program_shape
        for domain in live_domains
        for program_shape in domain.program_shapes
    }
    emittable = (
        len(store_domains) == 1
        and len(program_domains) <= 1
        and all(domain.program_compatible for domain in live_domains)
    )

    if live_rejections:
        return {
            "mode": "scalar-fallback",
            "reason": live_rejections[0]["reason"],
            "emittable": emittable,
        }

    keys = {
        (
            domain.scope,
            domain.input_shape,
            domain.axis,
            domain.result_shape,
            domain.program_shapes,
            domain.program_constraints,
        )
        for domain in live_domains
    }

    if len(keys) != 1:
        return {
            "mode": "scalar-fallback",
            "reason": "reductions do not share one iteration domain",
            "emittable": emittable,
        }

    domain = live_domains[0]

    return {
        "mode": "row-vector",
        "axis": domain.axis,
        "value_shape": domain.input_shape,
        "result_shape": domain.result_shape,
        "extent": domain.extent,
        "parallel_axes": domain.parallel_axes,
        "parallel_shape": domain.parallel_shape,
        "program_shape": domain.program_shapes[0],
        "program_constraints": domain.program_constraints,
        "reductions": tuple(item.result for item in live_domains),
    }


def _value_types(program):
    result = {value.name: value.type for value in (*program.inputs, *program.outputs)}

    def collect_block(block):
        result.update({value.name: value.type for value in block.args})

        for operation in block.operations:
            result.update({value.name: value.type for value in operation.results})

            for region in operation.regions:
                collect_block(region)

    for block in program.blocks:
        collect_block(block)
    return result


def _users(scoped_operations):
    result = defaultdict(list)

    for operation, _ in scoped_operations:
        for operand in operation.operands:
            result[operand].append(operation)
    return result


def _definitions(scoped_operations):
    return {
        value.name: operation
        for operation, _ in scoped_operations
        for value in operation.results
    }


def _forwarded_values(program):
    result = defaultdict(set)

    def collect_block(block):
        for operation in block.operations:
            for region in operation.regions:
                if region.operations and region.operations[-1].opcode == "scf.yield":
                    for source, target in zip(
                        region.operations[-1].operands,
                        operation.results,
                    ):
                        result[source].add(target.name)

                    if operation.opcode == "scf.for":
                        for source, target in zip(
                            region.operations[-1].operands,
                            region.args[1:],
                        ):
                            result[source].add(target.name)

                collect_block(region)

    for block in program.blocks:
        collect_block(block)
    return result


def _walk_program(program):
    for block in program.blocks:
        yield from _walk_block(block, ())


def _walk_block(block, scope):
    for operation_index, operation in enumerate(block.operations):
        yield operation, scope

        for region_index, region in enumerate(operation.regions):
            yield from _walk_block(
                region,
                (*scope, operation_index, region_index),
            )


__all__ = ["ReductionDomain", "analyze_reductions"]
