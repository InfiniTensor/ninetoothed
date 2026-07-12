"""Backend-neutral symbolic specialization for runtime-known Launch IR values."""

import ast
import re
from dataclasses import replace
from typing import Any, Mapping

from ninetoothed.ir import TensorSpec, ssa


def specialize_tensor_specs(
    specs: tuple[TensorSpec, ...], values: Mapping[str, Any]
) -> tuple[TensorSpec, ...]:
    return tuple(
        replace(
            spec,
            shape=tuple(_specialize_value(axis, values) for axis in spec.shape),
            attrs=_specialize_value(spec.attrs, values),
        )
        for spec in specs
    )


def specialize_program(program: ssa.Program, values: Mapping[str, Any]) -> ssa.Program:
    return replace(
        program,
        inputs=tuple(_specialize_ssa_value(value, values) for value in program.inputs),
        outputs=tuple(
            _specialize_ssa_value(value, values) for value in program.outputs
        ),
        blocks=tuple(_specialize_block(block, values) for block in program.blocks),
        metadata=_specialize_value(program.metadata, values),
    )


def _specialize_block(block: ssa.Block, values: Mapping[str, Any]) -> ssa.Block:
    return replace(
        block,
        args=tuple(_specialize_ssa_value(value, values) for value in block.args),
        operations=tuple(
            _specialize_operation(operation, values) for operation in block.operations
        ),
    )


def _specialize_operation(
    operation: ssa.Operation, values: Mapping[str, Any]
) -> ssa.Operation:
    return replace(
        operation,
        operands=tuple(_specialize_value(item, values) for item in operation.operands),
        results=tuple(
            _specialize_ssa_value(value, values) for value in operation.results
        ),
        attrs=_specialize_value(operation.attrs, values),
        regions=tuple(
            _specialize_block(region, values) for region in operation.regions
        ),
    )


def _specialize_ssa_value(value: ssa.Value, values: Mapping[str, Any]) -> ssa.Value:
    return replace(value, type=_specialize_type(value.type, values))


def _specialize_type(type_: ssa.Type, values: Mapping[str, Any]) -> ssa.Type:
    return replace(
        type_,
        shape=tuple(_specialize_value(axis, values) for axis in type_.shape),
        attrs=_specialize_value(type_.attrs, values),
    )


def _specialize_value(value: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _specialize_text(value, values)
    if isinstance(value, tuple):
        return tuple(_specialize_value(item, values) for item in value)
    if isinstance(value, list):
        return [_specialize_value(item, values) for item in value]
    if isinstance(value, Mapping):
        return {key: _specialize_value(item, values) for key, item in value.items()}
    return value


def _specialize_text(text: str, values: Mapping[str, Any]) -> str:
    result = text
    for symbol in sorted(values, key=len, reverse=True):
        result = re.sub(
            rf"(?<!\w){re.escape(symbol)}(?!\w)",
            repr(values[symbol]),
            result,
        )
    if result == text:
        return text

    return _evaluate_constant_expression(result)


def _evaluate_constant_expression(text: str) -> str:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return text
    if any(
        isinstance(node, (ast.Name, ast.Call, ast.Attribute)) for node in ast.walk(tree)
    ):
        return text
    try:
        value = eval(
            compile(tree, "<ssa-specialization>", "eval"), {"__builtins__": {}}
        )
    except (ArithmeticError, TypeError, ValueError):
        return text
    return repr(value) if isinstance(value, (bool, int, float)) else text


__all__ = ["specialize_program", "specialize_tensor_specs"]
