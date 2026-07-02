"""High-level SSA lowering entrypoint.

The public artifact API mirrors the original Triton ``generation.py`` route in
spirit: read the NineToothed Python application, turn the computation into an
intermediate representation, then ask a backend emitter to spell that IR in a
target language.  This module intentionally does not infer whole-kernel
operator templates such as matmul, reduction, or attention.
"""

from __future__ import annotations

import copy
import inspect
import textwrap
from pathlib import Path
from typing import Any

import sympy

from ninetoothed.backends import emit as emit_kernel
from ninetoothed.backends.core import Artifact, normalize_options
from ninetoothed.frontend.python import LoweringError, from_application
from ninetoothed.ir import Kernel, TensorSpec


def lower(
    arrangement,
    application,
    tensors,
    *,
    backend: str | None = None,
    caller: str = "torch",
    kernel_name: str | None = None,
    output_dir: str | Path | None = None,
    num_warps: int | tuple[int, ...] | None = None,
    num_stages: int | tuple[int, ...] | None = None,
    max_num_configs: int | None = None,
    write: bool = False,
    **backend_options: Any,
) -> Artifact:
    """Lower a NineToothed kernel to a backend artifact through SSA only."""
    from ninetoothed.utils import calculate_default_configs

    params = inspect.signature(application).parameters
    arranged = arrangement(*tensors)
    arranged = arranged if isinstance(arranged, tuple) else (arranged,)
    application.__annotations__ = {
        param: arranged_type for param, arranged_type in zip(params, arranged)
    }

    default_num_warps, default_num_stages = calculate_default_configs()

    if num_warps is None:
        num_warps = default_num_warps

    if num_stages is None:
        num_stages = default_num_stages

    if kernel_name is None:
        kernel_name = application.__name__

    options = normalize_options(
        backend, caller=caller, emit_only=True, **backend_options
    )
    tensor_irs = _application_tensor_irs(params, arranged)

    try:
        ssa_program = from_application(
            application,
            tensor_irs,
            kind=kernel_name or application.__name__,
        )
    except LoweringError as exc:
        raise LoweringError(
            f"Cannot lower `{application.__name__}` through the SSA backend path: {exc}."
        ) from exc

    if ssa_program is None:
        raise LoweringError(
            f"Cannot lower `{application.__name__}` through the SSA backend path: "
            "source inspection did not produce ssa.Program."
        )

    kernel_ir = Kernel(
        kernel_name=kernel_name,
        source=_application_source(application),
        source_language="ninetoothed-python",
        entrypoint=kernel_name,
        tensors=tensor_irs,
        compiler_options={
            "num_warps": num_warps,
            "num_stages": num_stages,
            "max_num_configs": max_num_configs,
        },
        metadata={
            "caller": caller,
            "ssa_ir_source": "application_ast",
            "ssa_tensor_ir_source": "arrangement_views",
            "generation_py_fallback": False,
        },
        ssa=ssa_program,
    )

    artifact = emit_kernel(kernel_ir, options=options)

    if write:
        if output_dir is None:
            raise ValueError("Output directory is required when `write=True`.")

        artifact.write_to(output_dir)

    return artifact


def _application_source(application) -> str:
    try:
        return textwrap.dedent(inspect.getsource(application))
    except OSError:
        return f"def {application.__name__}(...):\n    pass\n"


def _public_tensor_irs(params, tensors) -> tuple[TensorSpec, ...]:
    """Return TensorSpec objects for public source tensors.

    This helper is kept for source-audit scripts.  The public ``lower`` API uses
    ``_application_tensor_irs`` so the SSA sees the arranged application views.
    """
    return tuple(
        _public_tensor_ir(name, tensor) for name, tensor in zip(params, tensors)
    )


def _application_tensor_irs(params, tensors) -> tuple[TensorSpec, ...]:
    return tuple(
        _application_tensor_ir(name, tensor) for name, tensor in zip(params, tensors)
    )


def _public_tensor_ir(name: str, tensor) -> TensorSpec:
    source = getattr(tensor, "source", tensor)
    dtype = getattr(source, "dtype", getattr(tensor, "dtype", None))
    shape = getattr(source, "shape", getattr(tensor, "shape", ()))

    return TensorSpec(
        name=str(name),
        ndim=int(getattr(source, "ndim", getattr(tensor, "ndim", len(shape)))),
        dtype=None if dtype is None else str(dtype),
        shape=tuple(_shape_text(size) for size in shape),
        constexpr=bool(
            getattr(source, "constexpr", getattr(tensor, "constexpr", False))
        ),
        jagged_dim=getattr(source, "jagged_dim", getattr(tensor, "jagged_dim", None)),
        attrs=_tensor_source_attrs(tensor),
    )


def _application_tensor_ir(name: str, tensor) -> TensorSpec:
    source = getattr(tensor, "source", tensor)
    dtype = getattr(source, "dtype", getattr(tensor, "dtype", None))
    shape = getattr(tensor, "shape", getattr(source, "shape", ()))
    application_shape = _tensor_application_shape(tensor)

    return TensorSpec(
        name=str(name),
        ndim=int(getattr(tensor, "ndim", getattr(source, "ndim", len(shape)))),
        dtype=None if dtype is None else str(dtype),
        shape=tuple(_shape_text(size) for size in shape),
        constexpr=bool(
            getattr(source, "constexpr", getattr(tensor, "constexpr", False))
        ),
        jagged_dim=getattr(source, "jagged_dim", getattr(tensor, "jagged_dim", None)),
        attrs=_tensor_source_attrs(tensor)
        | {
            "application_shape": application_shape,
            "application_ndim": len(application_shape),
            "dtype_shapes": _tensor_dtype_shapes(tensor),
            "dtype_target_dims": _tensor_dtype_target_dims(tensor),
            "access_templates": _tensor_access_templates(tensor),
        },
    )


def _tensor_source_attrs(tensor) -> dict[str, Any]:
    source = getattr(tensor, "source", tensor)
    source_shape = getattr(source, "shape", ())
    source_ndim = int(getattr(source, "ndim", len(source_shape)))
    dtype = getattr(source, "dtype", getattr(tensor, "dtype", None))

    return {
        "source_name": str(getattr(source, "name", getattr(tensor, "name", "tensor"))),
        "source_ndim": source_ndim,
        "source_shape": tuple(_shape_text(size) for size in source_shape),
        "source_dtype": None if dtype is None else str(dtype),
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
        "view_shape": tuple(_shape_text(size) for size in getattr(tensor, "shape", ())),
    } | _view_index_attrs(tensor)


def _tensor_application_shape(tensor) -> tuple[str, ...]:
    dtype = getattr(tensor, "dtype", None)

    if _tensor_like(dtype):
        return tuple(_shape_text(size) for size in getattr(dtype, "shape", ()))

    shape = getattr(tensor, "shape", ())

    return tuple(_shape_text(size) for size in shape)


def _tensor_dtype_shapes(tensor) -> tuple[tuple[str, ...], ...]:
    shapes: list[tuple[str, ...]] = []
    current = getattr(tensor, "dtype", None)
    seen: set[int] = set()

    while _tensor_like(current) and id(current) not in seen:
        seen.add(id(current))
        shapes.append(
            tuple(_shape_text(size) for size in getattr(current, "shape", ()))
        )
        current = getattr(current, "dtype", None)
    return tuple(shapes)


def _tensor_dtype_target_dims(tensor) -> tuple[tuple[str | None, ...], ...]:
    target_dims: list[tuple[str | None, ...]] = []
    current = getattr(tensor, "dtype", None)
    seen: set[int] = set()

    while _tensor_like(current) and id(current) not in seen:
        seen.add(id(current))
        target_dims.append(
            tuple(
                None if dim is None else str(dim)
                for dim in getattr(current, "target_dims", ())
            )
        )
        current = getattr(current, "dtype", None)
    return tuple(target_dims)


def _tensor_access_templates(tensor) -> tuple[dict[str, Any], ...]:
    dtype_shapes = _tensor_dtype_shapes(tensor)

    if not dtype_shapes:
        return ()

    try:
        from ninetoothed.generation import CodeGenerator
        from ninetoothed.symbol import Symbol

        view = copy.deepcopy(tensor)
        outer_indices = tuple(
            type(view)._unravel_index(Symbol("outer_index"), view.shape)
        )
        source_shape = getattr(view.source, "shape", ())
        source_strides = tuple(type(view)._calculate_default_strides(source_shape))
    except Exception:
        return ()

    templates: list[dict[str, Any]] = []
    last_level = len(dtype_shapes) - 1

    for level, shape in enumerate(dtype_shapes):
        if level != last_level:
            continue

        indices = list(outer_indices)

        for prior_level, prior_shape in enumerate(dtype_shapes[:level]):
            indices.extend(
                Symbol(f"extract_{prior_level}_{dim}")
                for dim in range(len(prior_shape))
            )

        indices.extend(Symbol(f"value_{dim}") for dim in range(len(shape)))

        try:
            offsets, mask = CodeGenerator._generate_offsets_and_mask(
                view, tuple(indices)
            )
            linear_offset = Symbol(0)

            for offset, stride in zip(offsets, source_strides):
                linear_offset += Symbol(offset) * Symbol(stride)
        except Exception:
            continue

        templates.append(
            {
                "level": level,
                "shape": tuple(shape),
                "linear_offset": _shape_text(linear_offset),
                "offsets": tuple(_shape_text(offset) for offset in offsets),
                "mask": _shape_text(mask),
            }
        )
    return tuple(templates)


def _tensor_like(value: Any) -> bool:
    return value is not None and hasattr(value, "shape") and hasattr(value, "ndim")


def _shape_text(value) -> str:
    try:
        return str(sympy.simplify(str(value)))
    except Exception:
        return str(value)


def _view_index_attrs(tensor) -> dict[str, str]:
    if int(getattr(tensor, "ndim", 0)) == 0:
        return {}

    try:
        from ninetoothed.generation import CodeGenerator
        from ninetoothed.symbol import Symbol

        view = copy.deepcopy(tensor)
        source_shape = getattr(view.source, "shape", ())
        view_indices = tuple(type(view)._unravel_index(Symbol("index"), view.shape))
        offsets, mask = CodeGenerator._generate_offsets_and_mask(view, view_indices)
        source_strides = tuple(type(view)._calculate_default_strides(source_shape))
        linear_offset = Symbol(0)

        for offset, stride in zip(offsets, source_strides):
            linear_offset += Symbol(offset) * Symbol(stride)
    except Exception:
        return {}
    return {
        "view_linear_offset": _shape_text(linear_offset),
        "view_mask": _shape_text(mask),
    }
