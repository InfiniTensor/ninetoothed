"""Backend-neutral state shared while traversing SSA regions."""

from dataclasses import dataclass
from typing import Any, Mapping

from ninetoothed.backends.emitters.base import EmitterTarget
from ninetoothed.ir import Kernel, ssa


@dataclass(frozen=True, kw_only=True)
class TensorInfo:
    """Structured tensor and layout facts consumed during source emission."""

    name: str
    ndim: int = 1
    shape: tuple[str, ...] = ()
    dtype: str = "float32"
    source_name: str | None = None
    source_shape: tuple[str, ...] = ()
    source_strides: tuple[str, ...] = ()
    view_linear_offset: str | None = None
    view_mask: str | None = None
    attrs: Mapping[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class CooperativeDotPlan:
    """SSA operations participating in one cooperative dot schedule."""

    loop: ssa.Operation
    dot: ssa.Operation
    store: ssa.Operation


@dataclass(kw_only=True)
class EmitContext:
    """Mutable traversal state passed to backend rendering hooks."""

    target: EmitterTarget
    kernel: Kernel
    program: ssa.Program
    operations: Mapping[str, ssa.Operation]
    value_types: Mapping[str, ssa.Type]
    lines: list[str]
    memo: dict[str, str]
    tensor_infos: Mapping[str, TensorInfo]
    output: str
    output_axes: tuple[str, ...]
    index_expr: str
    outer_index_expr: str
    inner_index_expr: str
    mask_expr: str | None
    row_expr: str | None = None
    col_expr: str | None = None
    coordinate_exprs: tuple[str, ...] = ()
    reduce_axis: int | None = None
    reduce_index: str | None = None
    reduce_flattened: bool = False
    bindings: Mapping[str, str] | None = None
    temp_counter: list[int] | None = None
    materialized: dict[tuple[str, str], str] | None = None
    indent: str = ""
    local_suffix: str = ""
    block_program: bool = False
    native_block_program: bool = False
    layout_contiguous: bool = False
    vector_program: bool = False
    reduction_lane: str | None = None
    scheduled_reductions: frozenset[str] = frozenset()

    def child(
        self,
        *,
        lines: list[str] | None = None,
        memo: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> "EmitContext":
        """Return a nested context that shares analysis and temporary state."""
        data = {
            "target": self.target,
            "kernel": self.kernel,
            "program": self.program,
            "operations": self.operations,
            "value_types": self.value_types,
            "lines": self.lines if lines is None else lines,
            "memo": self.memo if memo is None else memo,
            "tensor_infos": self.tensor_infos,
            "output": self.output,
            "output_axes": self.output_axes,
            "index_expr": self.index_expr,
            "outer_index_expr": self.outer_index_expr,
            "inner_index_expr": self.inner_index_expr,
            "mask_expr": self.mask_expr,
            "row_expr": self.row_expr,
            "col_expr": self.col_expr,
            "coordinate_exprs": self.coordinate_exprs,
            "reduce_axis": self.reduce_axis,
            "reduce_index": self.reduce_index,
            "reduce_flattened": self.reduce_flattened,
            "bindings": self.bindings,
            "temp_counter": self.temp_counter,
            "materialized": self.materialized if lines is None else {},
            "indent": self.indent,
            "local_suffix": self.local_suffix,
            "block_program": self.block_program,
            "native_block_program": self.native_block_program,
            "layout_contiguous": self.layout_contiguous,
            "vector_program": self.vector_program,
            "reduction_lane": self.reduction_lane,
            "scheduled_reductions": self.scheduled_reductions,
        }
        data.update(kwargs)

        return EmitContext(**data)


__all__ = ["CooperativeDotPlan", "EmitContext", "TensorInfo"]
