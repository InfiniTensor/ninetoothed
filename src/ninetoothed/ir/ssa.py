"""SSA IR nodes, verification, and text rendering."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ninetoothed.ir.frozen import freeze


class VerificationError(ValueError):
    """Raised when a program violates the structured SSA contract."""


@dataclass(frozen=True, kw_only=True)
class Type:
    """A compact SSA value type."""

    kind: str
    shape: tuple[str, ...] = ()
    dtype: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "attrs", freeze(self.attrs))


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

    def __post_init__(self):
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "attrs", freeze(self.attrs))
        object.__setattr__(self, "regions", tuple(self.regions))


@dataclass(frozen=True, kw_only=True)
class Block:
    """A straight-line SSA block."""

    name: str = "entry"
    args: tuple[Value, ...] = ()
    operations: tuple[Operation, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "operations", tuple(self.operations))


@dataclass(frozen=True, kw_only=True)
class Program:
    """Canonical SSA-like IR for backend generation."""

    kind: str
    inputs: tuple[Value, ...] = ()
    outputs: tuple[Value, ...] = ()
    blocks: tuple[Block, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "metadata", freeze(self.metadata))


def verify_program(program: Program) -> Program:
    """Verify the structured SSA invariants consumed by all backend passes."""
    if len(program.blocks) != 1:
        raise VerificationError(
            f"SSA program `{program.kind}` must contain exactly one entry block; "
            f"got {len(program.blocks)}."
        )

    definitions: set[str] = set()

    for value in program.inputs:
        if value.name in definitions:
            raise VerificationError(f"Duplicate SSA input `{value.name}`.")

        definitions.add(value.name)

    symbols = set(str(name) for name in program.metadata.get("symbols", ()))

    for value in (*program.inputs, *program.outputs):
        for dimension in value.type.shape:
            symbols.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(dimension)))

    _verify_block(
        program.blocks[0], definitions | symbols, set(definitions), path="entry"
    )
    top_level_definitions = definitions | {
        result.name
        for operation in program.blocks[0].operations
        for result in operation.results
    }

    missing_outputs = tuple(
        value.name
        for value in program.outputs
        if value.name not in top_level_definitions
    )

    if missing_outputs:
        raise VerificationError(f"Undefined SSA outputs: {', '.join(missing_outputs)}.")
    return program


def _verify_block(
    block: Block,
    visible: set[str],
    all_definitions: set[str],
    *,
    path: str,
) -> None:
    local_visible = set(visible)
    block_arguments: list[str] = []

    for argument in block.args:
        if argument.name in all_definitions:
            raise VerificationError(
                f"Duplicate SSA definition `{argument.name}` in block `{path}`."
            )

        local_visible.add(argument.name)
        all_definitions.add(argument.name)
        block_arguments.append(argument.name)

    for index, operation in enumerate(block.operations):
        location = f"{path}:{index}:{operation.opcode}"
        missing = tuple(
            operand for operand in operation.operands if operand not in local_visible
        )

        if missing:
            raise VerificationError(
                f"Operation `{location}` uses undefined values: {', '.join(missing)}."
            )

        for result in operation.results:
            if result.name in all_definitions:
                raise VerificationError(
                    f"Duplicate SSA definition `{result.name}` at `{location}`."
                )

            local_visible.add(result.name)
            all_definitions.add(result.name)

        for region_index, region in enumerate(operation.regions):
            _verify_block(
                region,
                local_visible,
                all_definitions,
                path=f"{location}/region{region_index}",
            )

        _verify_region_contract(operation, location)

    all_definitions.difference_update(block_arguments)


def _verify_region_contract(operation: Operation, location: str) -> None:
    if operation.opcode == "scf.for":
        if len(operation.regions) != 1:
            raise VerificationError(
                f"Operation `{location}` requires exactly one region."
            )

        expected = len(operation.results)
        _verify_yield(operation.regions[0], expected, location)

        if len(operation.regions[0].args) != expected + 1:
            raise VerificationError(
                f"Operation `{location}` requires one induction argument and {expected} "
                "loop-carried arguments."
            )
    elif operation.opcode == "scf.if":
        if operation.results and len(operation.regions) != 2:
            raise VerificationError(
                f"Result-producing `{location}` requires then and else regions."
            )

        for region in operation.regions:
            if operation.results:
                _verify_yield(region, len(operation.results), location)
    elif operation.opcode == "scf.yield" and operation.regions:
        raise VerificationError(
            f"Operation `{location}` cannot contain nested regions."
        )


def _verify_yield(block: Block, expected: int, location: str) -> None:
    if not block.operations or block.operations[-1].opcode != "scf.yield":
        raise VerificationError(f"Region of `{location}` must end with `scf.yield`.")

    actual = len(block.operations[-1].operands)

    if actual != expected:
        raise VerificationError(
            f"Region of `{location}` yields {actual} values; expected {expected}."
        )


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
    "VerificationError",
    "render",
    "verify_program",
]
