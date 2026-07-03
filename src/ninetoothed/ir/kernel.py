"""Kernel-level IR records."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import sympy

from ninetoothed.ir.ssa import Program


@dataclass(frozen=True)
class TensorSpec:
    """A backend-neutral view of an application tensor parameter."""

    name: str
    ndim: int
    dtype: str | None = None
    shape: tuple[str, ...] = ()
    constexpr: bool = False
    jagged_dim: int | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tensor(cls, tensor: Any) -> "TensorSpec":
        source = getattr(tensor, "source", tensor)
        dtype = getattr(source, "dtype", getattr(tensor, "dtype", None))
        shape = getattr(tensor, "shape", getattr(source, "shape", ()))
        source_shape = getattr(source, "shape", ())
        source_ndim = int(
            getattr(source, "ndim", getattr(tensor, "ndim", len(source_shape)))
        )

        return cls(
            name=str(getattr(source, "name", getattr(tensor, "name", "tensor"))),
            ndim=int(getattr(tensor, "ndim", getattr(source, "ndim", len(shape)))),
            dtype=None if dtype is None else str(dtype),
            shape=tuple(_shape_text(size) for size in shape),
            constexpr=bool(
                getattr(source, "constexpr", getattr(tensor, "constexpr", False))
            ),
            jagged_dim=getattr(
                source, "jagged_dim", getattr(tensor, "jagged_dim", None)
            ),
            attrs={
                "source_name": str(
                    getattr(source, "name", getattr(tensor, "name", "tensor"))
                ),
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
            },
        )


@dataclass(frozen=True)
class Launch:
    """Runtime launch metadata shared by generated backends."""

    name: str
    args: tuple[str, ...] = ()
    grid: str | None = None


@dataclass(frozen=True)
class Kernel:
    """A kernel-level IR record consumed by backend emitters."""

    kernel_name: str
    source: str
    source_path: str | None = None
    source_language: str = "triton"
    entrypoint: str | None = None
    launch: Launch | None = None
    tensors: tuple[TensorSpec, ...] = ()
    compiler_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ssa: Program | None = None

    @classmethod
    def from_codegen(
        cls,
        code_generator: Any,
        source_file: str | Path,
        *,
        kernel_name: str,
        source_language: str = "triton",
        compiler_options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Kernel":
        path = Path(source_file)
        source = path.read_text(encoding="utf-8")

        launch_func = getattr(code_generator, "launch_func", None)
        launch_args: Sequence[str] = ()

        if launch_func is not None:
            launch_args = tuple(arg.arg for arg in launch_func.args.args)

        raw_grid = getattr(code_generator, "raw_grid", None)
        grid = None if raw_grid is None else _safe_unparse(raw_grid)

        tensors = tuple(
            TensorSpec.from_tensor(tensor)
            for tensor in getattr(code_generator, "tensors", ())
        )

        return cls(
            kernel_name=kernel_name,
            source=source,
            source_path=str(path),
            source_language=source_language,
            entrypoint=kernel_name,
            launch=Launch(
                name=getattr(
                    code_generator, "launch_func_name", f"launch_{kernel_name}"
                ),
                args=tuple(launch_args),
                grid=grid,
            ),
            tensors=tensors,
            compiler_options=dict(compiler_options or {}),
            metadata=dict(metadata or {}),
        )

    def with_metadata(self, **metadata: Any) -> "Kernel":
        return type(self)(
            kernel_name=self.kernel_name,
            source=self.source,
            source_path=self.source_path,
            source_language=self.source_language,
            entrypoint=self.entrypoint,
            launch=self.launch,
            tensors=self.tensors,
            compiler_options=self.compiler_options,
            metadata=dict(self.metadata) | metadata,
            ssa=self.ssa,
        )


def ir_to_dict(value: Any) -> Any:
    """Return a JSON-serializable representation of IR dataclasses."""
    if is_dataclass(value):
        return {
            field.name: ir_to_dict(getattr(value, field.name))
            for field in fields(value)
        }

    if isinstance(value, tuple):
        return [ir_to_dict(item) for item in value]

    if isinstance(value, list):
        return [ir_to_dict(item) for item in value]

    if isinstance(value, MappingABC):
        return {str(key): ir_to_dict(item) for key, item in value.items()}
    return value


def _shape_text(value: Any) -> str:
    try:
        return str(sympy.simplify(str(value)))
    except Exception:
        return str(value)


def _safe_unparse(node: Any) -> str:
    try:
        import ast

        return ast.unparse(node)
    except Exception:
        return repr(node)


__all__ = [
    "Kernel",
    "Launch",
    "TensorSpec",
    "ir_to_dict",
]
