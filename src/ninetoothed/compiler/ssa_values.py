"""Shared reference substitution and bit-exact structural keys for SSA passes."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from struct import pack


def rewrite_references(value, bindings):
    """Substitute references, including those carried by SSA value types."""
    from ninetoothed.ir import ssa

    if isinstance(value, str):
        while value in bindings:
            value = bindings[value]
        return value

    if isinstance(value, ssa.Type):
        return replace(
            value,
            shape=rewrite_references(value.shape, bindings),
            attrs=rewrite_references(value.attrs, bindings),
        )

    if isinstance(value, Mapping):
        return {
            key: item if key == "value" else rewrite_references(item, bindings)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return tuple(rewrite_references(item, bindings) for item in value)
    return value


def rewrite_value(value, bindings, *, rename=False):
    """Keep definitions' identities while updating their complete type."""
    return replace(
        value,
        name=rewrite_references(value.name, bindings) if rename else value.name,
        type=rewrite_references(value.type, bindings),
    )


def structural_key(value):
    """Compare actual float bit patterns instead of Python's signed-zero equality."""
    if isinstance(value, float):
        return (float, pack("!d", value))

    if isinstance(value, Mapping):
        return tuple(sorted((key, structural_key(item)) for key, item in value.items()))

    if isinstance(value, (tuple, list)):
        return tuple(structural_key(item) for item in value)

    if is_dataclass(value):
        return (
            type(value),
            tuple(
                (field.name, structural_key(getattr(value, field.name)))
                for field in fields(value)
                if field.compare
            ),
        )
    return (type(value), value)


def program_structure(program):
    return structural_key((program.inputs, program.outputs, program.blocks))
