"""Instantiate value dtypes from frozen SSA using the common inference rules."""

import ast
from dataclasses import replace

from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import ssa
from ninetoothed.ir.operations import type_condition_value
from ninetoothed.ir.type_inference import (
    _binary_type,
    _bool_type,
    _load_type,
    _math_result_type,
    _matmul_type,
    _reduce_type,
    _select_type,
)

_BINARY = {
    "add": ast.Add,
    "sub": ast.Sub,
    "mul": ast.Mult,
    "div": ast.Div,
    "floordiv": ast.FloorDiv,
    "mod": ast.Mod,
    "pow": ast.Pow,
    "bitwise_and": ast.BitAnd,
    "bitwise_or": ast.BitOr,
    "bitwise_xor": ast.BitXor,
    "bitwise_left_shift": ast.LShift,
    "bitwise_right_shift": ast.RShift,
}


def instantiate_program(program, tensors):
    """Rebind parameter types and infer dependent dtypes without Python source."""
    by_name = {spec.name: spec for spec in tensors}
    inputs = tuple(
        replace(
            value,
            type=replace(
                value.type,
                dtype=by_name[value.name].dtype,
                attrs=dict(value.type.attrs) | dict(by_name[value.name].attrs),
            ),
        )
        for value in program.inputs
    )
    values = {
        name: ssa.Value(name=name, type=ssa.Type(kind="index", dtype="index"))
        for name in program.metadata.get("symbols", ())
    } | {value.name: value for value in inputs}
    aliases = {}
    conditions = {}

    def reference(name):
        while name in aliases:
            name = aliases[name]

        return name

    def typed(original, dtype):
        return replace(
            original,
            shape=tuple(reference(dim) for dim in original.shape),
            dtype=dtype,
        )

    def attributes(original):
        attrs = dict(original)

        for name in ("dtype_ref", "dtype_operand", "preserve_dtype"):
            if name in attrs:
                attrs[name] = reference(attrs[name])

        for name in ("indices", "shape_values"):
            if name in attrs:
                attrs[name] = (
                    reference(attrs[name])
                    if isinstance(attrs[name], str)
                    else tuple(reference(item) for item in attrs[name])
                )

        if "iter_args" in attrs:
            attrs["iter_args"] = tuple(
                dict(item) | {"initial": reference(item["initial"])}
                for item in attrs["iter_args"]
            )

        return attrs

    def block(region, visible):
        operations = []
        current = dict(visible)
        current.update((arg.name, arg) for arg in region.args)

        for operation in region.operations:
            operation = replace(
                operation,
                operands=tuple(reference(name) for name in operation.operands),
                attrs=attributes(operation.attrs),
            )
            operands = tuple(current.get(name) for name in operation.operands)
            code = operation.opcode
            regions = operation.regions

            condition = type_condition_value(operation, operands, conditions)

            if condition is not None and operation.results:
                conditions[operation.results[0].name] = condition

            if code == "scf.if" and operation.operands[0] in conditions:
                branch = 0 if conditions[operation.operands[0]] else 1

                if branch >= len(regions):
                    continue

                selected = block(regions[branch], current)
                selected_operations = (
                    selected.operations[:-1]
                    if operation.results
                    else selected.operations
                )
                operations.extend(selected_operations)
                current.update(
                    (value.name, value)
                    for nested in selected_operations
                    for value in nested.results
                )

                if operation.results:
                    for result, yielded in zip(
                        operation.results, selected.operations[-1].operands
                    ):
                        aliases[result.name] = reference(yielded)

                continue

            if code == "scf.for":
                body = regions[0]
                args = (
                    body.args[0],
                    *(
                        replace(arg, type=typed(arg.type, value.type.dtype))
                        for arg, value in zip(body.args[1:], operands[3:])
                    ),
                )
                regions = (block(replace(body, args=args), current),)
                results = tuple(
                    replace(result, type=typed(result.type, value.type.dtype))
                    for result, value in zip(operation.results, operands[3:])
                )
            elif code == "scf.if":
                regions = tuple(block(body, current) for body in regions)
                yielded = (
                    regions[0].operations[-1].operands if operation.results else ()
                )
                defined = {
                    value.name: value
                    for nested in regions[0].operations
                    for value in nested.results
                } | current
                results = tuple(
                    replace(
                        result,
                        type=typed(result.type, defined[name].type.dtype),
                    )
                    for result, name in zip(operation.results, yielded)
                )
            else:
                results = tuple(
                    replace(
                        result,
                        type=typed(
                            result.type,
                            _dtype(operation, operands, current, result, conditions),
                        ),
                    )
                    for result in operation.results
                )

            operation = replace(operation, results=results, regions=regions)
            operations.append(operation)
            current.update((value.name, value) for value in results)

        values.update(current)

        return replace(region, operations=tuple(operations))

    blocks = tuple(block(region, values) for region in program.blocks)

    return ssa.verify_program(
        replace(
            program,
            inputs=inputs,
            blocks=blocks,
            outputs=tuple(values[reference(value.name)] for value in program.outputs),
        )
    )


def _dtype(operation, operands, visible, result, conditions):
    code = operation.opcode
    name = code.split(".", 1)[1]

    if code == "dtype.constant":
        return normalize_dtype(operation.attrs["value"])

    if code == "dtype.of":
        return operands[0].type.dtype

    if code == "dtype.select":
        condition = conditions.get(operation.operands[0])

        if condition is not None:
            return operands[1 if condition else 2].type.dtype
        return (
            operands[1].type.dtype
            if operands[1].type.dtype == operands[2].type.dtype
            else None
        )

    if operation.attrs.get("preserve_dtype"):
        return visible[operation.attrs["preserve_dtype"]].type.dtype

    if code == "tensor.cast":
        ref = operation.attrs.get("dtype_ref")

        if ref is not None:
            return visible[ref].type.dtype

        if len(operands) == 2:
            return operands[1].type.dtype
        return normalize_dtype(operation.attrs.get("dtype"))

    if code in {"tensor.zeros", "tensor.empty", "tensor.full"}:
        dtype_operand = operation.attrs.get("dtype_operand")

        if dtype_operand is not None:
            return visible[dtype_operand].type.dtype

        ref = operation.attrs.get("dtype_ref")

        return visible[ref].type.dtype if ref is not None else result.type.dtype

    if code in {
        "linalg.dot",
        "linalg.matmul",
        "linalg.transpose",
    } and operation.attrs.get("dtype_ref"):
        return visible[operation.attrs["dtype_ref"]].type.dtype

    if code in {
        "tensor.view",
        "tensor.extract",
        "linalg.transpose",
    }:
        return operands[0].type.dtype

    if code.startswith("reduce."):
        return _reduce_type(
            operands[0].type, operation.attrs.get("axis"), operator=name, strict=True
        ).dtype

    if code.startswith("cmp.") or code in {"arith.and", "arith.or"}:
        return _bool_type(operands[0], operands[1]).dtype

    if code.startswith("arith.") and name in _BINARY:
        return _binary_type(_BINARY[name](), *(value.type for value in operands)).dtype

    if code == "arith.not":
        return "bool"

    if code.startswith("arith.") and name in {"neg", "pos", "invert"}:
        return operands[0].type.dtype

    if code.startswith("math.") or code in {"arith.maximum", "arith.minimum"}:
        return _math_result_type(name, operands).dtype

    if code == "select.where":
        return _select_type(*(value.type for value in operands)).dtype

    if code in {"linalg.dot", "linalg.matmul"}:
        return _matmul_type(
            *(value.type for value in operands), strict=code == "linalg.matmul"
        ).dtype

    if code == "mem.load":
        return _load_type(operands[0].type).dtype

    if code == "mem.atomic_add":
        return operands[1].type.dtype

    if code == "mem.data_ptr" or code.startswith("call."):
        return operands[0].type.dtype if operands else result.type.dtype
    return result.type.dtype


__all__ = ["instantiate_program"]
