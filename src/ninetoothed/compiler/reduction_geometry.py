"""Prove compatible source-bound domains for arranged reductions."""

from functools import cache

from ninetoothed.compiler.effects import (
    broadcast_view_parts,
    is_pointwise_expression,
)


def reduction_inputs_share_domain(operations, value_types, tensors):
    """Prove matching source-bound domains for regular, single-level tiles."""
    inputs = {tensor.name: tensor for tensor in tensors}
    producers = {value.name: op for op in operations for value in op.results}

    @cache
    def domains(name):
        if name in inputs:
            tensor = inputs[name]
            attrs = {**value_types[name].attrs, **tensor.attrs}

            if attrs.get("source_ndim", tensor.ndim) == 0:
                return frozenset()

            if attrs.get("other") is not None and tuple(
                map(str, attrs.get("source_shape", ()))
            ) != tuple(map(str, attrs.get("application_shape", tensor.shape))):
                return None

            layout = tensor.layout

            if layout is None and not attrs:
                return frozenset({(tuple(map(str, tensor.shape)), "True")})

            if (
                layout is None
                or len(layout.levels) != 1
                or len(layout.value_accesses) != 1
                or tensor.jagged_dim is not None
                or attrs.get("custom_tiling", True)
                or attrs.get("layout_operations") is None
                or set(attrs["layout_operations"])
                - {"tile", "expand", "unsqueeze", "squeeze"}
            ):
                return None

            return frozenset(
                {
                    (
                        tuple(dim.render() for dim in layout.application_shape),
                        layout.value_accesses[0].predicate.render(),
                    )
                }
            )

        op = producers.get(name)

        if op is None:
            type_ = value_types.get(name)

            if type_ is not None and type_.kind == "tensor":
                return (
                    None
                    if type_.attrs
                    else frozenset({(tuple(map(str, type_.shape)), "True")})
                )
            return None if name.startswith("%") else frozenset()

        if op.opcode.startswith(("dtype.", "shape.", "reduce.")):
            return frozenset()

        dependencies = tuple(domains(operand) for operand in op.operands)
        parts = broadcast_view_parts(op)

        if op.opcode in {"tensor.zeros", "tensor.full"} and all(
            dependency == frozenset() for dependency in dependencies
        ):
            return frozenset()

        if (
            parts is not None
            and dependencies
            and all(dependency == frozenset() for dependency in dependencies)
        ):
            return frozenset()

        if not is_pointwise_expression(op) and not (
            parts is not None and all(part == ":" for part in parts)
        ):
            return None

        result = frozenset()

        for operand, dependency in zip(op.operands, dependencies):
            if dependency is None or (
                dependency and value_types[operand].shape != value_types[name].shape
            ):
                return None

            result |= dependency

        return result if len(result) <= 1 else None

    return all(
        domains(op.operands[0]) is not None
        for op in operations
        if op.opcode.startswith("reduce.")
    )
