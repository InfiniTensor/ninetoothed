"""Backend-neutral lowering of arranged Tensor views.

This module owns the index and mask semantics of ``Tensor`` meta operations.
It deliberately has no dependency on a backend source generator.
"""

import ast
import copy
from typing import Any

import ninetoothed.naming as naming
from ninetoothed.frontend.python import LoweringError
from ninetoothed.ir import (
    AccessMap,
    IndexExpr,
    LayoutLevel,
    TensorLayout,
    TensorSpec,
)
from ninetoothed.language import call
from ninetoothed.symbol import Symbol


def tensor_specs(params, tensors) -> tuple[TensorSpec, ...]:
    return tuple(
        tensor_spec(str(name), tensor) for name, tensor in zip(params, tensors)
    )


def tensor_spec(name: str, tensor) -> TensorSpec:
    source = getattr(tensor, "source", tensor)
    dtype = getattr(source, "dtype", getattr(tensor, "dtype", None))
    shape = getattr(tensor, "shape", getattr(source, "shape", ()))
    application_shape = _application_shape(tensor)

    attrs = _source_attrs(tensor) | {
        "application_shape": application_shape,
        "application_ndim": len(application_shape),
        "dtype_shapes": _dtype_shapes(tensor),
        "dtype_target_dims": _dtype_target_dims(tensor),
        "access_templates": _access_templates(tensor),
    }

    return TensorSpec(
        ndim=int(getattr(tensor, "ndim", getattr(source, "ndim", len(shape)))),
        shape=_launch_shape(tensor),
        dtype=None if dtype is None else str(dtype),
        jagged_dim=getattr(source, "jagged_dim", getattr(tensor, "jagged_dim", None)),
        constexpr=bool(
            getattr(source, "constexpr", getattr(tensor, "constexpr", False))
        ),
        name=name,
        layout=_tensor_layout(attrs),
        attrs=attrs,
    )


def overall_offsets_and_mask(tensor, indices):
    offsets, mask = offsets_and_mask(tensor, indices)
    overall = sum(
        offsets[dim] * Symbol(tensor.source.stride_string(dim))
        for dim in range(tensor.source.ndim)
    )

    if tensor.source.jagged_dim is not None:
        overall += Symbol(f"{tensor.source.name}_seq_start") * Symbol(
            tensor.source.stride_string(tensor.source.jagged_dim)
        )
    return overall, mask


def offsets_and_mask(tensor, indices):
    """Evaluate a Tensor view's recorded meta operations symbolically."""
    offsets = [Symbol(0) for _ in range(tensor.source.ndim)]
    tensor.source._mask = Symbol(True)
    current = tensor
    start = 0

    while _tensor_like(current):
        stop = start + current.ndim
        current._inputs = [list(indices[start:stop])]
        start = stop
        current = current.dtype

    for level in reversed(tensor._levels):
        for value in level:
            value.offsets()

    for dim, offset in enumerate(tensor.source._outputs[0]):
        offsets[dim] += offset

    current = tensor

    while _tensor_like(current):
        current._inputs.clear()
        current = current.dtype
    return offsets, tensor.source._mask


def innermost_indices(tensor, *, use_power_of_2_sizes: bool = True):
    class NextPowerOfTwo(ast.NodeTransformer):
        def visit_Constant(self, node):
            value = node.value

            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return ast.copy_location(
                    ast.Constant(value=1 << (value - 1).bit_length()), node
                )
            return self.generic_visit(node)

        def visit_Name(self, node):
            if not naming.is_meta(node.id):
                return Symbol(naming.make_next_power_of_2(node.id)).node
            return self.generic_visit(node)

    result = []
    innermost = tensor.innermost()

    for size, target_dim in zip(innermost.shape, innermost.target_dims):
        if use_power_of_2_sizes:
            size = NextPowerOfTwo().visit(Symbol(copy.deepcopy(size)).node)

        slices = tuple(
            slice(None) if candidate == target_dim else None
            for candidate in innermost.target_dims
        )
        result.append(call("arange", 0, size)[slices])
    return tuple(result)


def _source_attrs(tensor) -> dict[str, Any]:
    source = getattr(tensor, "source", tensor)
    source_shape = getattr(source, "shape", ())
    source_ndim = int(getattr(source, "ndim", len(source_shape)))
    dtype = getattr(source, "dtype", getattr(tensor, "dtype", None))
    attrs = {
        "source_name": str(getattr(source, "name", getattr(tensor, "name", "tensor"))),
        "source_ndim": source_ndim,
        "source_shape": tuple(_text(size) for size in source_shape),
        "source_dtype": None if dtype is None else str(dtype),
        "other": getattr(source, "other", None),
        "source_strides": tuple(
            str(source.stride_string(dim))
            for dim in range(source_ndim)
            if hasattr(source, "stride_string")
        ),
        "target_dims": tuple(
            None if dim is None else str(dim)
            for dim in getattr(tensor, "target_dims", ())
        ),
        "view_ndim": int(getattr(tensor, "ndim", source_ndim)),
        "view_shape": tuple(_text(size) for size in getattr(tensor, "shape", ())),
    }

    if getattr(source, "jagged_dim", None) is not None:
        attrs |= {
            "jagged_values_param": str(source.values_string()),
            "jagged_values_numel_param": str(source.values_numel_string()),
            "jagged_offsets_param": str(source.offsets_string()),
            "jagged_offsets_numel_param": str(source.offsets_numel_string()),
            "jagged_max_seq_len_param": str(source.max_seq_len_string()),
            "jagged_seq_len_param": str(source.seq_len_string()),
        }
    return attrs | _view_index_attrs(tensor)


def _launch_shape(tensor) -> tuple[str, ...]:
    shape = tuple(_text(size) for size in getattr(tensor, "shape", ()))
    source = getattr(tensor, "source", tensor)

    if getattr(source, "jagged_dim", None) is None:
        return shape

    seq_len = str(source.seq_len_string())
    max_seq_len = str(source.max_seq_len_string())

    return tuple(size.replace(seq_len, max_seq_len) for size in shape)


def _application_shape(tensor) -> tuple[str, ...]:
    dtype = getattr(tensor, "dtype", None)
    value = dtype if _tensor_like(dtype) else tensor

    return tuple(_text(size) for size in getattr(value, "shape", ()))


def _dtype_shapes(tensor) -> tuple[tuple[str, ...], ...]:
    result = []
    current = getattr(tensor, "dtype", None)
    seen = set()

    while _tensor_like(current) and id(current) not in seen:
        seen.add(id(current))
        result.append(tuple(_text(size) for size in current.shape))
        current = current.dtype
    return tuple(result)


def _dtype_target_dims(tensor) -> tuple[tuple[str | None, ...], ...]:
    result = []
    current = getattr(tensor, "dtype", None)
    seen = set()

    while _tensor_like(current) and id(current) not in seen:
        seen.add(id(current))
        result.append(
            tuple(None if dim is None else str(dim) for dim in current.target_dims)
        )
        current = current.dtype
    return tuple(result)


def _access_templates(tensor) -> tuple[dict[str, Any], ...]:
    shapes = _dtype_shapes(tensor)

    if not shapes:
        return ()

    view = copy.deepcopy(tensor)
    _replace_jagged_view_extent(view)
    outer_indices = tuple(type(view)._unravel_index(Symbol("outer_index"), view.shape))
    source_shape = getattr(view.source, "shape", ())
    source_strides = tuple(
        Symbol(view.source.stride_string(dim)) for dim in range(len(source_shape))
    )
    level = len(shapes) - 1
    shape = shapes[level]
    indices = list(outer_indices)

    for prior_level, prior_shape in enumerate(shapes[:level]):
        indices.extend(
            Symbol(f"extract_{prior_level}_{dim}") for dim in range(len(prior_shape))
        )

    indices.extend(Symbol(f"value_{dim}") for dim in range(len(shape)))

    try:
        offsets, mask = offsets_and_mask(view, tuple(indices))
    except Exception as exc:
        raise LoweringError(
            f"Cannot derive value access map for tensor `{view.source.name}`: {exc}."
        ) from exc

    linear = Symbol(0)

    for offset, stride in zip(offsets, source_strides):
        linear += Symbol(offset) * Symbol(stride)

    template = {
        "level": level,
        "shape": tuple(shape),
        "linear_offset": _text(linear),
        "offsets": tuple(_text(offset) for offset in offsets),
        "mask": _text(mask),
    }
    source = view.source
    jagged_dim = getattr(source, "jagged_dim", None)

    if jagged_dim is not None:
        template["jagged"] = {
            "offsets_param": str(source.offsets_string()),
            "seq_start": f"{source.name}_seq_start",
            "seq_len": str(source.seq_len_string()),
            "batch_offset": _text(offsets[0]),
            "stride": _text(source_strides[jagged_dim]),
        }
    return (template,)


def _view_index_attrs(tensor) -> dict[str, str]:
    if int(getattr(tensor, "ndim", 0)) == 0:
        return {}

    view = copy.deepcopy(tensor)
    _replace_jagged_view_extent(view)
    source_shape = getattr(view.source, "shape", ())
    view_indices = tuple(type(view)._unravel_index(Symbol("index"), view.shape))

    try:
        offsets, mask = offsets_and_mask(view, view_indices)
    except Exception as exc:
        if _dtype_shapes(view):
            return {}

        raise LoweringError(
            f"Cannot derive view access map for tensor `{view.source.name}`: {exc}."
        ) from exc

    source_strides = tuple(
        Symbol(view.source.stride_string(dim)) for dim in range(len(source_shape))
    )
    linear = Symbol(0)

    for offset, stride in zip(offsets, source_strides):
        linear += Symbol(offset) * Symbol(stride)
    return {
        "view_linear_offset": _text(linear),
        "view_mask": _text(mask),
        "view_offsets": tuple(_text(offset) for offset in offsets),
    }


def _tensor_layout(attrs: dict[str, Any]) -> TensorLayout:
    levels = tuple(
        LayoutLevel(
            shape=tuple(IndexExpr.parse(dim) for dim in shape),
            target_dims=tuple(
                None if dim is None else IndexExpr.parse(dim) for dim in target_dims
            ),
        )
        for shape, target_dims in zip(
            attrs.get("dtype_shapes", ()), attrs.get("dtype_target_dims", ())
        )
    )
    value_accesses = tuple(
        AccessMap(
            source_indices=tuple(
                IndexExpr.parse(value) for value in template.get("offsets", ())
            ),
            linear_index=IndexExpr.parse(template.get("linear_offset", 0)),
            predicate=IndexExpr.parse(template.get("mask", True)),
        )
        for template in attrs.get("access_templates", ())
    )
    view_access = None

    if attrs.get("view_linear_offset") is not None:
        view_access = AccessMap(
            source_indices=tuple(
                IndexExpr.parse(value) for value in attrs.get("view_offsets", ())
            ),
            linear_index=IndexExpr.parse(attrs["view_linear_offset"]),
            predicate=IndexExpr.parse(attrs.get("view_mask", True)),
        )
    return TensorLayout(
        source_shape=tuple(
            IndexExpr.parse(dim) for dim in attrs.get("source_shape", ())
        ),
        source_strides=tuple(
            IndexExpr.parse(dim) for dim in attrs.get("source_strides", ())
        ),
        view_shape=tuple(IndexExpr.parse(dim) for dim in attrs.get("view_shape", ())),
        application_shape=tuple(
            IndexExpr.parse(dim) for dim in attrs.get("application_shape", ())
        ),
        levels=levels,
        view_access=view_access,
        value_accesses=value_accesses,
    )


def _replace_jagged_view_extent(tensor) -> None:
    source = getattr(tensor, "source", tensor)

    if getattr(source, "jagged_dim", None) is None:
        return

    seq_len = Symbol(source.seq_len_string())
    max_seq_len = Symbol(source.max_seq_len_string())

    for size in tensor.shape:
        if hasattr(size, "find_and_replace"):
            size.find_and_replace(seq_len, max_seq_len)


def _tensor_like(value: Any) -> bool:
    return value is not None and hasattr(value, "shape") and hasattr(value, "ndim")


def _text(value: Any) -> str:
    return str(value)


__all__ = [
    "innermost_indices",
    "offsets_and_mask",
    "overall_offsets_and_mask",
    "tensor_spec",
    "tensor_specs",
]
