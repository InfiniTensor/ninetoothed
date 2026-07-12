"""Shape and dtype inference for the Python SSA frontend."""

import ast
from typing import Any

from ninetoothed.frontend.errors import LoweringError
from ninetoothed.ir import ssa


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


def _subscript_type(
    type_: ssa.Type, slice_node: ast.AST, *, source: bool = False
) -> ssa.Type:
    if type_.kind != "tensor":
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
    result_shape: list[str] = []
    position = 0
    consumed = 0

    for element in elements:
        if isinstance(element, ast.Constant) and element.value is None:
            result_shape.append("1")
            continue

        if isinstance(element, ast.Slice):
            if position < len(shape):
                result_shape.append(shape[position])
                position += 1

            continue

        if position < len(shape):
            position += 1
            consumed += 1

    result_shape.extend(shape[position:])
    attrs = dict(type_.attrs)

    if not result_shape:
        next_shape = _next_dtype_shape(type_)

        if next_shape is not None:
            level = int(attrs.get("dtype_level", 0)) + 1
            attrs["dtype_level"] = level

            return ssa.Type(
                kind="tensor", shape=next_shape, dtype=type_.dtype, attrs=attrs
            )
        return ssa.Type(kind="scalar", dtype=type_.dtype, attrs=attrs)

    if consumed:
        attrs["partial_indices"] = int(attrs.get("partial_indices", 0)) + consumed
    return ssa.Type(
        kind="tensor", shape=tuple(result_shape), dtype=type_.dtype, attrs=attrs
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


def _reduce_type(type_: ssa.Type, axis: Any, *, strict: bool) -> ssa.Type:
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
    if type_.kind != "tensor":
        return ssa.Type(kind="scalar", dtype="index")

    shape = tuple(str(item) for item in type_.shape)
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

    return _broadcast_type(lhs, rhs)


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
    normalized = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
        "float": "float32",
    }
    lhs = None if lhs is None else lhs.rsplit(".", 1)[-1]
    rhs = None if rhs is None else rhs.rsplit(".", 1)[-1]
    lhs = normalized.get(lhs or "", lhs)
    rhs = normalized.get(rhs or "", rhs)

    if lhs is None:
        return rhs

    if rhs is None or lhs == rhs:
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


def _cast_type(type_: ssa.Type, dtype: str) -> ssa.Type:
    normalized = dtype.rsplit(".", 1)[-1]

    if normalized == "dtype":
        normalized = type_.dtype

    normalized = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }.get(normalized, normalized)

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
