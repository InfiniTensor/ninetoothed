"""Contracts shared by SSA backend source emitters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    cooperative_reduction_program: bool = False
    private_meta_parameters: tuple[tuple[str, int], ...] = ()
    scheduled_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class EmitterTarget(ABC):
    """Backend syntax hooks consumed by the target-independent SSA walker."""

    backend: Target
    language: str
    suffix: str
    source_route: str
    buffer_suffix: str = ""
    index_name: str = "index"
    entrypoint_prefix: str = "launch_"
    c_style_syntax: bool = False
    vector_value_semantics: bool = False
    tir_value_semantics: bool = False
    native_block_matmul: bool = False
    max_vector_numel: int | None = None

    def symbol(self, name: str) -> str:
        return f"v{name[1:]}" if name.startswith("%") else name

    def tensor_ref(self, tensor: str) -> str:
        return f"{tensor}{self.buffer_suffix}"

    def entrypoint(self, kernel_name: str) -> str:
        return f"{self.entrypoint_prefix}{kernel_name}"

    def type_name(self, dtype: str | None, kind: str | None = None) -> str:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} does not expose a C-style type spelling."
        )

    def block_coords(self, axes: tuple[str, ...]) -> tuple[str, ...]:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} has no block value domain."
        )

    def block_shape(self, axes: tuple[str, ...]) -> str:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} has no block value domain."
        )

    def render_view(self, operation, context) -> str:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} has no block view syntax."
        )

    def needs_block_init(self, name: str, value: ssa.Value, context) -> bool:
        del name, value, context

        return False

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        del context

        return operation.results[0].type

    def coerce_binary_args(self, operation, args, context):
        del operation, context

        return args

    def coerce_dot_args(self, operation, args, context):
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

    def supports_cooperative_reduction(self, schedule: Mapping[str, Any]) -> bool:
        del schedule

        return False

    def thread_id(self) -> str:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} has no thread-id expression."
        )

    def thread_count(self) -> str:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} has no thread-count expression."
        )

    def emit_cooperative_reduction(self, local, operation, context):
        del local, operation, context

        return None

    def program_id(self, axis: int = 0) -> str:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} has no program-id expression."
        )

    def vector_reduce(self, operator: str, operand: str, axis: int) -> str:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} has no vector reduction syntax."
        )

    def vector_splat(self, shape: str, value: str, dtype: str) -> str:
        raise NotImplementedError(
            f"Emitter {type(self).__name__} has no vector splat syntax."
        )

    def atomic_add(self, operands: tuple[str, ...], dtype: str) -> str:
        del dtype

        return self.call("atomic_add", operands)

    def index_cast(self, value: str) -> str:
        return value

    def uses_mutable_scalar_slots(self) -> bool:
        return False

    def mutable_scalar_decl(self, type_: ssa.Type, name: str, init: str) -> list[str]:
        return [self.local_decl(type_, name, init)]

    def mutable_scalar_read(self, name: str) -> str:
        return name

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value}"

    def schedule_context(self, context: ModuleRenderContext) -> ModuleRenderContext:
        return context

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
