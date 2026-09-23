"""SSA IR nodes, verification, and text rendering."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ninetoothed.dtype import normalize_dtype
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

    definitions: dict[str, Type] = {}

    for value in program.inputs:
        if value.name in definitions:
            raise VerificationError(f"Duplicate SSA input `{value.name}`.")

        definitions[value.name] = value.type

    symbols = set(str(name) for name in program.metadata.get("symbols", ()))

    for value in (*program.inputs, *program.outputs):
        for dimension in value.type.shape:
            symbols.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(dimension)))

    visible = {name: Type(kind="index") for name in symbols} | definitions
    top_level = _verify_block(
        program.blocks[0], visible, set(definitions), path="entry"
    )
    output_names = set()

    for value in program.outputs:
        if value.name in output_names:
            raise VerificationError(f"Duplicate SSA output `{value.name}`.")

        output_names.add(value.name)

        if value.name not in top_level or (
            value.name in symbols and value.name not in definitions
        ):
            raise VerificationError(f"Undefined SSA output `{value.name}`.")

        _verify_type(value.type, top_level[value.name], f"output `{value.name}`")

    return program


def _verify_block(
    block: Block,
    visible: dict[str, Type],
    all_definitions: set[str],
    *,
    path: str,
    owner: Operation | None = None,
) -> dict[str, Type]:
    local_visible = dict(visible)
    block_arguments: list[str] = []

    for argument in block.args:
        if argument.name in all_definitions or argument.name in local_visible:
            raise VerificationError(
                f"Duplicate SSA definition `{argument.name}` in block `{path}`."
            )

        local_visible[argument.name] = argument.type
        all_definitions.add(argument.name)
        block_arguments.append(argument.name)

    for index, operation in enumerate(block.operations):
        location = f"{path}:{index}:{operation.opcode}"
        references = operation.operands

        if operation.opcode == "mem.store":
            indices = operation.attrs.get("indices", ())
            references += (indices,) if isinstance(indices, str) else tuple(indices)

        missing = tuple(
            operand for operand in references if operand not in local_visible
        )

        if missing:
            raise VerificationError(
                f"Operation `{location}` uses undefined values: {', '.join(missing)}."
            )

        for result in operation.results:
            if result.name in all_definitions or result.name in local_visible:
                raise VerificationError(
                    f"Duplicate SSA definition `{result.name}` at `{location}`."
                )

            all_definitions.add(result.name)

        _verify_region_contract(operation, location, local_visible)
        _verify_memory_contract(operation, location, local_visible)

        if operation.opcode == "scf.yield":
            if owner is None or index != len(block.operations) - 1:
                raise VerificationError(
                    f"Operation `{location}` must terminate an scf region."
                )

            if operation.results or operation.regions:
                raise VerificationError(
                    f"Operation `{location}` cannot define results or regions."
                )

            if len(operation.operands) != len(owner.results):
                raise VerificationError(
                    f"Operation `{location}` yields {len(operation.operands)} values; "
                    f"expected {len(owner.results)}."
                )

            for slot, (operand, result) in enumerate(
                zip(operation.operands, owner.results)
            ):
                types = (local_visible[operand], result.type)

                if owner.opcode == "scf.for":
                    types += (
                        local_visible[owner.operands[slot + 3]],
                        owner.regions[0].args[slot + 1].type,
                    )

                _verify_types(types, location)

        for region_index, region in enumerate(operation.regions):
            _verify_block(
                region,
                local_visible,
                all_definitions,
                path=f"{location}/region{region_index}",
                owner=operation,
            )

        # Results become visible only after their defining regions finish.
        local_visible.update((result.name, result.type) for result in operation.results)

    all_definitions.difference_update(block_arguments)

    return local_visible


def _verify_types(types: tuple[Type, ...], location: str) -> None:
    # Unknown dtypes must not hide conflicts between known types in the same slot.
    expected = next(
        (type_ for type_ in types if _verification_signature(type_)[2] is not None),
        types[0],
    )

    for actual in types:
        _verify_type(actual, expected, location)


def _verify_type(actual: Type, expected: Type, location: str) -> None:
    # Provenance and layout attributes are not part of the SSA value signature.
    actual_kind, actual_shape, actual_dtype = _verification_signature(actual)
    expected_kind, expected_shape, expected_dtype = _verification_signature(expected)
    dtype_mismatch = (
        actual_dtype is not None
        and expected_dtype is not None
        and actual_dtype != expected_dtype
    )

    if (actual_kind, actual_shape) != (expected_kind, expected_shape) or dtype_mismatch:
        raise VerificationError(
            f"Type mismatch at {location}: got {_format_type(actual)}; "
            f"expected {_format_type(expected)}."
        )


def _verification_signature(type_: Type) -> tuple[str, tuple[str, ...], str | None]:
    dtype = normalize_dtype(type_.dtype)

    # Shape dimensions and induction values use index, while their integer
    # arithmetic uses scalar/int64. Compare these existing representations
    # without rewriting the IR or admitting other integer widths or signedness.
    if not type_.shape and (
        (type_.kind == "index" and dtype in {None, "index", "int64"})
        or (type_.kind == "scalar" and dtype == "index")
    ):
        return "scalar", (), "int64"

    return type_.kind, type_.shape, dtype


def _verify_region_contract(
    operation: Operation, location: str, visible: dict[str, Type]
) -> None:
    if operation.opcode == "scf.for":
        if len(operation.regions) != 1:
            raise VerificationError(
                f"Operation `{location}` requires exactly one region."
            )

        expected = len(operation.results)

        if len(operation.operands) != expected + 3:
            raise VerificationError(
                f"Operation `{location}` requires three bounds and {expected} "
                "loop-carried operands."
            )

        for operand in operation.operands[:3]:
            type_ = visible[operand]

            if type_.kind != "index" and not (
                type_.kind == "scalar"
                and not type_.shape
                and normalize_dtype(type_.dtype)
                in {
                    None,
                    "index",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                }
            ):
                raise VerificationError(
                    f"Operation `{location}` requires integer scalar bounds."
                )

        _verify_yield(operation.regions[0], expected, location)

        if len(operation.regions[0].args) != expected + 1:
            raise VerificationError(
                f"Operation `{location}` requires one induction argument and {expected} "
                "loop-carried arguments."
            )

        if operation.regions[0].args[0].type.kind != "index":
            raise VerificationError(
                f"Operation `{location}` requires an index induction."
            )

        if operation.attrs.get("induction", "%iv") != operation.regions[0].args[0].name:
            raise VerificationError(
                f"Operation `{location}` has inconsistent induction."
            )

        bindings = operation.attrs.get("iter_args", ())
        pairs = tuple(zip(operation.operands[3:], operation.regions[0].args[1:]))

        if len(bindings) != len(pairs) or any(
            binding.get("initial") != initial
            or binding.get("block_arg") != argument.name
            for binding, (initial, argument) in zip(bindings, pairs)
        ):
            raise VerificationError(
                f"Operation `{location}` has inconsistent loop-carried bindings."
            )

    elif operation.opcode == "scf.if":
        if len(operation.operands) != 1 or len(operation.regions) not in (1, 2):
            raise VerificationError(
                f"Operation `{location}` requires one condition and one or two regions."
            )

        if operation.results and len(operation.regions) != 2:
            raise VerificationError(
                f"Result-producing `{location}` requires then and else regions."
            )

        for region in operation.regions:
            if region.args:
                raise VerificationError(
                    f"Region of `{location}` cannot declare block arguments."
                )

            if operation.results:
                _verify_yield(region, len(operation.results), location)
    elif operation.opcode != "scf.yield" and (
        operation.regions or operation.opcode.startswith("scf.")
    ):
        raise VerificationError(
            f"Operation `{location}` has no supported region contract."
        )


def _verify_memory_contract(
    operation: Operation, location: str, visible: dict[str, Type]
) -> None:
    if not operation.opcode.startswith("mem."):
        return

    signatures = {
        "mem.store": (2, 0),
        "mem.data_ptr": (1, 1),
        "mem.load": (1, 1),
        "mem.atomic_add": (2, 1),
    }

    if operation.opcode not in signatures:
        raise VerificationError(f"Unknown memory effect at `{location}`.")

    operands, results = signatures[operation.opcode]

    if len(operation.operands) != operands or len(operation.results) != results:
        raise VerificationError(
            f"Operation `{location}` requires {operands} operands and {results} results."
        )

    target = operation.operands[1 if operation.opcode == "mem.store" else 0]
    expected = (
        {"pointer"}
        if operation.opcode in {"mem.load", "mem.atomic_add"}
        else {"tensor", "scalar"}
    )

    if visible[target].kind not in expected:
        raise VerificationError(f"Invalid memory target `{target}` at `{location}`.")

    if (
        operation.opcode == "mem.data_ptr"
        and operation.results[0].type.kind != "pointer"
    ):
        raise VerificationError(f"Operation `{location}` must produce a pointer.")


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
