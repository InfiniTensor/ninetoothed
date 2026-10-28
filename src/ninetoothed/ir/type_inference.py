"""Shared shape and dtype rules for frontend lowering and IR instantiation."""

import ast
import re
from dataclasses import replace
from typing import Any

from ninetoothed.dtype import normalize_dtype
from ninetoothed.errors import LoweringError
from ninetoothed.ir import ssa

# These fields are consumed by shared inference or emitters.
# Other inherited Tensor attributes are legacy provenance; memory padding and
# physical contracts are authoritative on TensorSpec, not intermediate values.
TYPE_LAYOUT_ATTRS = frozenset(
    {
        "dtype_level",
        "partial_indices",
        "offset_subscript",
        "offset_unsupported",
        "offset_value_dims",
        "dtype_shapes",
        "dtype_target_dims",
        "access_templates",
        "source_shape",
        "source_strides",
        "source_name",
        "source_ndim",
        "ndim",
    }
)


def _shape_dim_from_type(
    type_: ssa.Type, dim: Any, *, source: bool = False
) -> str | None:
    if source:
        shape = tuple(str(item) for item in type_.attrs.get("source_shape", ()))
    else:
        shape = tuple(str(item) for item in type_.shape)

    if not shape:
        return None

    index = int(dim or 0)

    if index < 0:
        index += len(shape)

    if index < 0 or index >= len(shape):
        return None
    return shape[index]


def _index_broadcast_shape(*shapes):
    """Require a provable index domain, without tightening legacy arithmetic."""
    result = ()

    for shape in shapes:
        shape = tuple(str(dim) for dim in shape)

        if any(
            lhs != rhs and lhs != "1" and rhs != "1"
            for lhs, rhs in zip(reversed(result), reversed(shape))
        ):
            raise LoweringError(
                "Tensor index shapes must have proven equal or singleton dimensions."
            )

        result = _broadcast_shapes(result, shape)
    return result


def _normalize_subscript(type_, node, *, source=False):
    """Canonicalize only slices proven to preserve their entire dimension."""
    elements = node.elts if isinstance(node, ast.Tuple) else (node,)
    shape = type_.attrs.get("source_shape", type_.shape) if source else type_.shape
    consumed = sum(
        not (isinstance(item, ast.Constant) and item.value in (None, Ellipsis))
        for item in elements
    )
    normalized = []
    dim = 0

    for item in elements:
        if isinstance(item, ast.Constant) and item.value is None:
            normalized.append(item)

            continue

        if isinstance(item, ast.Constant) and item.value is Ellipsis:
            dim += len(shape) - consumed
        else:
            if isinstance(item, ast.Slice):
                full = (
                    (item.lower is None or ast.unparse(item.lower) == "0")
                    and (item.step is None or ast.unparse(item.step) == "1")
                    and (
                        item.upper is None
                        or dim < len(shape)
                        and ast.dump(item.upper)
                        == ast.dump(ast.parse(str(shape[dim]), mode="eval").body)
                    )
                )

                if full:
                    item = ast.Slice()

            dim += 1

        normalized.append(item)

    return (
        ast.Tuple(elts=normalized, ctx=ast.Load())
        if isinstance(node, ast.Tuple)
        else normalized[0]
    )


def _subscript_type(
    type_: ssa.Type,
    slice_node: ast.AST,
    *,
    source: bool = False,
    index_types: tuple[ssa.Type, ...] = (),
) -> ssa.Type:
    slice_node = _normalize_subscript(type_, slice_node, source=source)

    if type_.kind != "tensor":
        if index_types or not (
            isinstance(slice_node, ast.Constant) and slice_node.value is Ellipsis
        ):
            raise LoweringError("Integer indexing requires a tensor domain.")
        return type_

    elements = (
        tuple(slice_node.elts) if isinstance(slice_node, ast.Tuple) else (slice_node,)
    )
    shape = tuple(
        str(dim)
        for dim in (
            type_.attrs.get("source_shape", type_.shape) if source else type_.shape
        )
    )

    if (
        sum(
            isinstance(item, ast.Constant) and item.value is Ellipsis
            for item in elements
        )
        > 1
    ):
        raise LoweringError("Only one ellipsis is allowed in tensor indexing.")

    if sum(
        not (isinstance(item, ast.Constant) and item.value in (None, Ellipsis))
        for item in elements
    ) > len(shape):
        raise LoweringError("Too many indices for the tensor domain.")

    for index_type in index_types:
        if index_type.kind not in {"scalar", "index", "tensor"} or normalize_dtype(
            index_type.dtype
        ) not in {
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
        }:
            raise LoweringError("Tensor indices must have an integer element type.")

    if any(
        isinstance(item, ast.Constant) and item.value is Ellipsis for item in elements
    ):
        missing = len(shape) - sum(
            not (isinstance(item, ast.Constant) and item.value in (None, Ellipsis))
            for item in elements
        )
        elements = tuple(
            expanded
            for item in elements
            for expanded in (
                (ast.Slice(),) * max(0, missing)
                if isinstance(item, ast.Constant) and item.value is Ellipsis
                else (item,)
            )
        )

    result_shape: list[str] = []
    position = 0
    consumed = 0
    indexed_positions = []
    index_insertion = None
    offset_unsupported = bool(type_.attrs.get("offset_subscript"))
    index_shape = _index_broadcast_shape(*(item.shape for item in index_types))

    for element_position, element in enumerate(elements):
        if isinstance(element, ast.Constant) and element.value is None:
            offset_unsupported = True
            result_shape.append("1")
            continue

        if isinstance(element, ast.Slice):
            offset_unsupported |= any(
                item is not None
                for item in (element.lower, element.upper, element.step)
            )

            if position < len(shape):
                result_shape.append(shape[position])
                position += 1

            continue

        if position < len(shape):
            if index_insertion is None:
                index_insertion = len(result_shape)

            indexed_positions.append(element_position)
            offset_unsupported |= bool(result_shape) or (
                isinstance(element, ast.Constant) and element.value is Ellipsis
            )
            position += 1
            consumed += 1

    result_shape.extend(shape[position:])
    attrs = dict(type_.attrs)
    attrs["offset_subscript"] = True
    attrs["offset_unsupported"] = offset_unsupported or not consumed

    if not result_shape:
        next_shape = None if source else _next_dtype_shape(type_)

        if next_shape is not None:
            if any(dim != "1" for dim in index_shape):
                raise LoweringError(
                    "Tensor indices across dtype levels are not supported."
                )

            # A statically one-element index is a scalar hierarchy selector.
            # Preserve the legacy nested view without introducing a vector
            # index domain into the next dtype level.

            level = int(attrs.get("dtype_level", 0)) + 1
            attrs["dtype_level"] = level

            return ssa.Type(
                kind="tensor",
                shape=next_shape,
                dtype=type_.dtype,
                attrs=attrs,
            )

        if index_shape:
            return ssa.Type(
                kind="tensor", shape=index_shape, dtype=type_.dtype, attrs=attrs
            )
        return ssa.Type(kind="scalar", dtype=type_.dtype, attrs=attrs)

    if consumed:
        attrs["partial_indices"] = int(attrs.get("partial_indices", 0)) + consumed

    if index_shape:
        if not indexed_positions:
            raise LoweringError("Tensor indices require an indexed tensor dimension.")
        # Adjacent advanced indices retain their position; separated indices
        # move their common broadcast domain to the front, as in array indexing.

        adjacent = indexed_positions == list(
            range(indexed_positions[0], indexed_positions[-1] + 1)
        )
        insertion = index_insertion if adjacent else 0
        result_shape[insertion:insertion] = index_shape
    return ssa.Type(
        kind="tensor",
        shape=tuple(result_shape),
        dtype=type_.dtype,
        attrs=attrs,
    )


def _next_dtype_shape(type_: ssa.Type) -> tuple[str, ...] | None:
    shapes = tuple(
        tuple(str(dim) for dim in shape)
        for shape in type_.attrs.get("dtype_shapes", ())
    )
    level = int(type_.attrs.get("dtype_level", 0))

    if level + 1 >= len(shapes):
        return None
    return shapes[level + 1]


def _reduce_type(
    type_: ssa.Type, axis: Any, *, operator: str, strict: bool
) -> ssa.Type:
    if operator == "sum":
        promoted_dtype = {
            "bool": "uint32",
            "int8": "int32",
            "int16": "int32",
            "uint8": "uint32",
            "uint16": "uint32",
        }.get(normalize_dtype(type_.dtype))

        if promoted_dtype is not None:
            type_ = replace(type_, dtype=promoted_dtype)

    if type_.kind != "tensor":
        return type_

    shape = tuple(str(dim) for dim in type_.shape)

    if axis is None:
        return ssa.Type(kind="scalar", dtype=type_.dtype, attrs=dict(type_.attrs))

    try:
        index = int(axis)
    except (TypeError, ValueError) as exc:
        raise LoweringError(
            f"Reduction axis must be a compile-time integer: {axis!r}."
        ) from exc

    if index < 0:
        index += len(shape)

    if index < 0 or index >= len(shape):
        if strict:
            raise LoweringError(
                f"Reduction axis {axis} is outside tensor rank {len(shape)}."
            )
        return type_

    result_shape = shape[:index] + shape[index + 1 :]

    if not result_shape:
        return ssa.Type(kind="scalar", dtype=type_.dtype, attrs=dict(type_.attrs))
    return ssa.Type(
        kind="tensor", shape=result_shape, dtype=type_.dtype, attrs=dict(type_.attrs)
    )


def _offset_type(type_: ssa.Type, dim: Any) -> ssa.Type:
    if type_.attrs.get("offset_unsupported"):
        raise LoweringError(
            "Offsets() after indexing supports only a single subscript with "
            "leading scalar indices followed by full slices; None, non-prefix "
            "indices, sliced views, and chained subscripts are unsupported."
        )

    if type_.kind != "tensor":
        return ssa.Type(kind="scalar", dtype="index")

    shape = tuple(str(item) for item in type_.shape)
    level = int(type_.attrs.get("dtype_level", 0))

    for template in type_.attrs.get("access_templates", ()):
        if int(template.get("level", -1)) != level:
            continue

        offsets = tuple(template.get("offsets", ()))
        source_dim = int(dim or 0)

        if source_dim < 0:
            source_dim += len(offsets)

        if not 0 <= source_dim < len(offsets):
            break

        partial_indices = int(type_.attrs.get("partial_indices", 0))
        dimensions = tuple(
            sorted(
                {
                    int(index)
                    for index in re.findall(
                        r"\bvalue_(\d+)\b", str(offsets[source_dim])
                    )
                    if int(index) >= partial_indices
                }
            )
        )
        template_shape = tuple(str(axis) for axis in template.get("shape", ()))
        result_shape = tuple(template_shape[index] for index in dimensions)

        return ssa.Type(
            kind="tensor" if result_shape else "scalar",
            shape=result_shape,
            dtype="index",
            attrs={"offset_value_dims": dimensions},
        )

    dtype_target_dims = tuple(
        tuple(None if item is None else str(item) for item in dims)
        for dims in type_.attrs.get("dtype_target_dims", ())
    )
    level = int(type_.attrs.get("dtype_level", 0))
    target_dims = dtype_target_dims[level] if level < len(dtype_target_dims) else ()

    if not target_dims:
        return ssa.Type(kind="tensor", shape=shape, dtype="index")

    source_ndim = int(type_.attrs.get("source_ndim", len(target_dims)))
    source_dim = int(dim or 0)

    if source_dim < 0:
        source_dim += source_ndim

    kept = tuple(
        axis
        for axis, target_dim in zip(shape, target_dims)
        if target_dim is not None and int(target_dim) == source_dim
    )

    if not kept:
        return ssa.Type(kind="scalar", dtype="index")
    return ssa.Type(kind="tensor", shape=kept, dtype="index")


def _matmul_type(lhs: ssa.Type, rhs: ssa.Type, *, strict: bool = True) -> ssa.Type:
    lhs_shape = tuple(str(dim) for dim in lhs.shape)
    rhs_shape = tuple(str(dim) for dim in rhs.shape)
    dtype = _dot_accumulator_dtype(lhs.dtype or rhs.dtype)
    attrs = dict(lhs.attrs)

    if not lhs_shape or not rhs_shape:
        raise LoweringError("Matrix multiplication requires rank-1 or higher operands.")

    lhs_k = lhs_shape[-1]
    rhs_k = rhs_shape[-2] if len(rhs_shape) >= 2 else rhs_shape[0]

    if lhs_k != rhs_k:
        if strict:
            raise LoweringError(
                f"Matrix multiplication contracting dimensions differ: {lhs_k} vs {rhs_k}."
            )

        if len(lhs_shape) >= 2 and len(rhs_shape) >= 2:
            return ssa.Type(
                kind="tensor",
                shape=(lhs_shape[-2], rhs_shape[-1]),
                dtype=dtype,
                attrs=attrs,
            )
        return _broadcast_type(lhs, rhs)

    lhs_batch = lhs_shape[:-2] if len(lhs_shape) >= 2 else ()
    rhs_batch = rhs_shape[:-2] if len(rhs_shape) >= 2 else ()
    batch = _broadcast_shapes(lhs_batch, rhs_batch)

    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        return ssa.Type(kind="scalar", dtype=dtype, attrs=attrs)

    if len(lhs_shape) == 1:
        result_shape = (*batch, rhs_shape[-1])
    elif len(rhs_shape) == 1:
        result_shape = (*batch, lhs_shape[-2])
    else:
        result_shape = (*batch, lhs_shape[-2], rhs_shape[-1])
    return ssa.Type(kind="tensor", shape=result_shape, dtype=dtype, attrs=attrs)


def _dot_accumulator_dtype(dtype: str | None) -> str | None:
    normalized = (dtype or "").lower()

    if normalized in {
        "float16",
        "fp16",
        "bfloat16",
        "bf16",
        "float8_e4m3fn",
        "float8_e5m2",
    }:
        return "float32"
    return dtype


def _binary_type(operator: ast.operator, lhs: ssa.Type, rhs: ssa.Type) -> ssa.Type:
    if isinstance(operator, ast.MatMult):
        return _matmul_type(lhs, rhs)

    if isinstance(operator, ast.Add):
        if lhs.kind == "pointer" and rhs.kind != "pointer":
            return _offset_pointer_type(lhs, rhs)

        if rhs.kind == "pointer" and lhs.kind != "pointer":
            return _offset_pointer_type(rhs, lhs)

    if isinstance(operator, ast.Sub) and lhs.kind == "pointer":
        if rhs.kind == "pointer":
            shape = _broadcast_shape(lhs, rhs)
            kind = "tensor" if shape else "index"

            return ssa.Type(kind=kind, shape=shape, dtype="index")
        return _offset_pointer_type(lhs, rhs)

    result = _broadcast_type(lhs, rhs)

    if (
        isinstance(operator, ast.Div)
        and result.dtype is not None
        and normalize_dtype(result.dtype) not in {"float32", "float64"}
    ):
        return replace(result, dtype="float32")

    return result


def _offset_pointer_type(pointer: ssa.Type, offset: ssa.Type) -> ssa.Type:
    return ssa.Type(
        kind="pointer",
        shape=_broadcast_shape(pointer, offset),
        dtype=pointer.dtype,
        attrs=dict(pointer.attrs),
    )


def _load_type(pointer: ssa.Type) -> ssa.Type:
    shape = tuple(str(dim) for dim in pointer.shape)

    return ssa.Type(
        kind="tensor" if shape else "scalar",
        shape=shape,
        dtype=pointer.dtype,
        attrs=dict(pointer.attrs),
    )


def _broadcast_shape(lhs: ssa.Type, rhs: ssa.Type) -> tuple[str, ...]:
    return _broadcast_shapes(
        tuple(str(dim) for dim in lhs.shape), tuple(str(dim) for dim in rhs.shape)
    )


def _broadcast_shapes(
    lhs_shape: tuple[str, ...], rhs_shape: tuple[str, ...]
) -> tuple[str, ...]:
    result: list[str] = []

    for lhs_dim, rhs_dim in zip(reversed(lhs_shape), reversed(rhs_shape)):
        if lhs_dim == rhs_dim or rhs_dim == "1":
            result.append(lhs_dim)
        elif lhs_dim == "1":
            result.append(rhs_dim)
        else:
            if lhs_dim.lstrip("-").isdigit() and rhs_dim.lstrip("-").isdigit():
                raise LoweringError(
                    f"Cannot broadcast dimensions `{lhs_dim}` and `{rhs_dim}`."
                )

            result.append(lhs_dim)

    longer = lhs_shape if len(lhs_shape) > len(rhs_shape) else rhs_shape
    prefix = longer[: abs(len(lhs_shape) - len(rhs_shape))]

    return tuple(prefix) + tuple(reversed(result))


def _select_type(condition: ssa.Type, lhs: ssa.Type, rhs: ssa.Type) -> ssa.Type:
    """Broadcast all three shapes without promoting values with the condition."""
    values = _broadcast_type(lhs, rhs)
    shape = _broadcast_shapes(condition.shape, values.shape)

    return ssa.Type(
        kind="tensor" if shape else values.kind,
        shape=shape,
        dtype=values.dtype,
        attrs=values.attrs,
    )


def _broadcast_type(lhs: ssa.Type, rhs: ssa.Type) -> ssa.Type:
    if lhs.kind != "tensor" and rhs.kind != "tensor":
        return ssa.Type(
            kind="scalar",
            shape=_broadcast_shape(lhs, rhs),
            dtype=_promote_dtype(lhs.dtype, rhs.dtype),
            attrs=dict(lhs.attrs),
        )

    if lhs.kind == "tensor" and rhs.kind != "tensor":
        return ssa.Type(
            kind="tensor",
            shape=lhs.shape,
            dtype=_promote_dtype(lhs.dtype, rhs.dtype),
            attrs=dict(lhs.attrs),
        )

    if rhs.kind == "tensor" and lhs.kind != "tensor":
        return ssa.Type(
            kind="tensor",
            shape=rhs.shape,
            dtype=_promote_dtype(lhs.dtype, rhs.dtype),
            attrs=dict(rhs.attrs),
        )

    shape = _broadcast_shape(lhs, rhs)
    dtype = _promote_dtype(lhs.dtype, rhs.dtype)
    attrs = dict(lhs.attrs if lhs.kind == "tensor" else rhs.attrs)

    return ssa.Type(kind="tensor", shape=shape, dtype=dtype, attrs=attrs)


def _math_result_type(name: str, operands: tuple[ssa.Value, ...]) -> ssa.Type:
    if not operands:
        return ssa.Type(kind="scalar", dtype="float32")

    result = operands[0].type

    for operand in operands[1:]:
        result = _broadcast_type(result, operand.type)

    if name == "rand":
        shape = tuple(str(dim) for dim in result.shape)

        return ssa.Type(
            kind="tensor" if shape else "scalar",
            shape=shape,
            dtype="float32",
            attrs=dict(result.attrs),
        )
    return result


def _promote_dtype(lhs: str | None, rhs: str | None) -> str | None:
    lhs = normalize_dtype(lhs)
    rhs = normalize_dtype(rhs)

    if lhs is None or rhs is None:
        return None

    if lhs == rhs:
        return lhs

    if {lhs, rhs} == {"float16", "bfloat16"}:
        return "float32"

    ranks = {
        "bool": 0,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 3,
        "uint32": 3,
        "index": 4,
        "int64": 4,
        "uint64": 4,
        "float8_e4m3fn": 5,
        "float8_e5m2": 5,
        "float16": 6,
        "bfloat16": 6,
        "float32": 7,
        "float64": 8,
    }

    if lhs not in ranks or rhs not in ranks:
        raise LoweringError(f"Cannot promote unsupported dtypes `{lhs}` and `{rhs}`.")
    return lhs if ranks[lhs] >= ranks[rhs] else rhs


def _cast_type(type_: ssa.Type, dtype: str | None) -> ssa.Type:
    normalized = normalize_dtype(dtype)

    if normalized == "dtype":
        normalized = normalize_dtype(type_.dtype)

    return ssa.Type(
        kind=type_.kind,
        shape=type_.shape,
        dtype=normalized,
        attrs=dict(type_.attrs),
    )


def _bool_type(lhs: ssa.Value, rhs: ssa.Value | None = None) -> ssa.Type:
    if rhs is not None and rhs.type.kind == "tensor":
        shape = _broadcast_type(lhs.type, rhs.type).shape

        return ssa.Type(kind="tensor", shape=shape, dtype="bool")

    if lhs.type.kind == "tensor":
        return ssa.Type(kind="tensor", shape=lhs.type.shape, dtype="bool")
    return ssa.Type(kind="scalar", dtype="bool")


def _transpose_type(type_: ssa.Type) -> ssa.Type:
    if type_.kind != "tensor" or len(type_.shape) < 2:
        return type_
    return ssa.Type(
        kind=type_.kind,
        shape=tuple(reversed(type_.shape)),
        dtype=type_.dtype,
        attrs=dict(type_.attrs),
    )


__all__ = [
    "_shape_dim_from_type",
    "_subscript_type",
    "_next_dtype_shape",
    "_reduce_type",
    "_offset_type",
    "_matmul_type",
    "_dot_accumulator_dtype",
    "_binary_type",
    "_offset_pointer_type",
    "_load_type",
    "_broadcast_shape",
    "_broadcast_shapes",
    "_broadcast_type",
    "_math_result_type",
    "_promote_dtype",
    "_cast_type",
    "_bool_type",
    "_transpose_type",
]
