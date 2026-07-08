"""SSA IR nodes and text rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class Type:
    """A compact SSA value type."""

    kind: str
    shape: tuple[str, ...] = ()
    dtype: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Value:
    """A named SSA value such as ``%0`` or a public tensor argument."""

    name: str
    type: Type


@dataclass(frozen=True, kw_only=True)
class Operation:
    """A single SSA operation."""

    opcode: str
    operands: tuple[str, ...] = ()
    results: tuple[Value, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    regions: tuple["Block", ...] = ()


@dataclass(frozen=True, kw_only=True)
class Block:
    """A straight-line SSA block."""

    name: str = "entry"
    args: tuple[Value, ...] = ()
    operations: tuple[Operation, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Program:
    """Canonical SSA-like IR for backend generation."""

    kind: str
    inputs: tuple[Value, ...] = ()
    outputs: tuple[Value, ...] = ()
    blocks: tuple[Block, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def render(program: Program | None) -> str:
    """Render SSA IR as a readable textual form, not JSON."""
    if program is None:
        return "<not-available>"

    lines = [f"ssa @{program.kind} {{"]

    if program.inputs:
        lines.append("  inputs:")

        for value in program.inputs:
            lines.append(f"    {value.name} : {_format_type(value.type)}")

    if program.outputs:
        lines.append("  outputs:")

        for value in program.outputs:
            lines.append(f"    {value.name} : {_format_type(value.type)}")

    for block in program.blocks:
        _render_block(block, lines, indent=2)

    lines.append("}")

    return "\n".join(lines)


def _render_block(block: Block, lines: list[str], *, indent: int) -> None:
    prefix = " " * indent
    args = ""

    if block.args:
        args = (
            "("
            + ", ".join(f"{arg.name}: {_format_type(arg.type)}" for arg in block.args)
            + ")"
        )

    lines.append(f"{prefix}^{block.name}{args}:")

    for operation in block.operations:
        _render_operation(operation, lines, indent=indent + 2)


def _render_operation(operation: Operation, lines: list[str], *, indent: int) -> None:
    prefix = " " * indent
    results = ", ".join(result.name for result in operation.results)
    operands = ", ".join(operation.operands)
    lhs = f"{results} = " if results else ""
    attrs = _format_attrs(operation.attrs)
    suffix = f" {attrs}" if attrs else ""
    operand_text = f" {operands}" if operands else ""
    lines.append(f"{prefix}{lhs}{operation.opcode}{operand_text}{suffix}".rstrip())

    for region in operation.regions:
        _render_block(region, lines, indent=indent + 2)


def _format_type(type_: Type) -> str:
    shape = ""

    if type_.shape:
        shape = "<" + "x".join(type_.shape) + ">"

    dtype = f"x{type_.dtype}" if type_.dtype else ""

    return f"{type_.kind}{shape}{dtype}"


def _format_attrs(attrs: Mapping[str, Any]) -> str:
    cleaned = {
        key: value for key, value in attrs.items() if value is not None and value != ()
    }

    if not cleaned:
        return ""
    return (
        "{"
        + ", ".join(f"{key}={_format_attr(value)}" for key, value in cleaned.items())
        + "}"
    )


def _format_attr(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)

    if isinstance(value, tuple):
        return "(" + ", ".join(_format_attr(item) for item in value) + ")"

    if isinstance(value, list):
        return "[" + ", ".join(_format_attr(item) for item in value) + "]"

    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{key}: {_format_attr(item)}" for key, item in value.items())
            + "}"
        )
    return repr(value)


__all__ = [
    "Block",
    "Operation",
    "Program",
    "Type",
    "Value",
    "render",
]
