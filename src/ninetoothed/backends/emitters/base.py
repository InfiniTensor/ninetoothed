"""Contracts shared by SSA backend source emitters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from ninetoothed.backends.core import Target
from ninetoothed.ir import ssa


@dataclass(frozen=True, kw_only=True)
class ModuleRenderContext:
    """Backend-neutral facts collected before module-level source rendering."""

    kernel: Any
    variables: tuple[str, ...]
    outputs: tuple[str, ...]
    shape_params: tuple[str, ...]
    total: str
    body: str
    tensors: Mapping[str, Any]
    value_types: Mapping[str, ssa.Type]
    operations: Mapping[str, ssa.Operation]
    stores: tuple[ssa.Operation, ...]
    outer_axes: tuple[str, ...]
    grid_total: str
    axes: tuple[str, ...]
    vector_program: bool
    block_program: bool
    scalar_program: bool
    backend_options: Mapping[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class EmitterTarget(ABC):
    """Backend syntax hooks consumed by the target-independent SSA walker."""

    backend: Target
    language: str
    suffix: str
    source_route: str
    buffer_suffix: str = ""
    index_name: str = "index"
    block_size: int = 256
    entrypoint_prefix: str = "launch_"
    is_cuda: bool = False
    is_triton: bool = False
    is_tilelang: bool = False
    is_tvm: bool = False
    is_tir: bool = False

    def symbol(self, name: str) -> str:
        return f"v{name[1:]}" if name.startswith("%") else name

    def tensor_ref(self, tensor: str) -> str:
        return f"{tensor}{self.buffer_suffix}"

    def entrypoint(self, kernel_name: str) -> str:
        return f"{self.entrypoint_prefix}{kernel_name}"

    def type_name(self, dtype: str | None, kind: str | None = None) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a C-style type spelling."
        )

    def block_coords(self, axes: tuple[str, ...]) -> tuple[str, ...]:
        raise NotImplementedError(f"{type(self).__name__} has no block value domain.")

    def block_shape(self, axes: tuple[str, ...]) -> str:
        raise NotImplementedError(f"{type(self).__name__} has no block value domain.")

    def render_view(self, operation, context) -> str:
        raise NotImplementedError(f"{type(self).__name__} has no block view syntax.")

    def needs_block_init(self, name: str, value: ssa.Value, context) -> bool:
        del name, value, context
        return False

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        del context
        return operation.results[0].type

    def coerce_binary_args(self, operation, args, context):
        del operation, context
        return args

    def emit_dot_operand(self, name, coords, context):
        del name, coords, context
        return None

    def emit_block_dot(self, operation, context, coords=None):
        del operation, context, coords
        return None

    def emit_reduction_loop(self, local, operation, context):
        del local, operation, context
        return None

    @abstractmethod
    def literal(self, value: Any) -> str: ...

    @abstractmethod
    def load(
        self,
        tensor: str,
        index: str,
        *,
        mask: str | None = None,
        other: Any = 0.0,
    ) -> str: ...

    @abstractmethod
    def store(
        self,
        tensor: str,
        index: str,
        value: str,
        *,
        mask: str | None = None,
    ) -> str: ...

    @abstractmethod
    def cast(self, dtype: str, value: str) -> str: ...

    @abstractmethod
    def where(self, cond: str, yes: str, no: str) -> str: ...

    @abstractmethod
    def call(self, name: str, args: tuple[str, ...]) -> str: ...

    @abstractmethod
    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str: ...

    @abstractmethod
    def loop_header(self, var: str, lower: str, upper: str, step: str) -> str: ...

    @abstractmethod
    def reduce_update(self, operator: str, acc: str, term: str) -> str: ...

    @abstractmethod
    def render_module(self, context: ModuleRenderContext) -> str: ...


__all__ = ["EmitterTarget", "ModuleRenderContext"]
