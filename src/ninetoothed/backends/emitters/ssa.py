"""Unified SSA-to-source emitters for backend code generation.

This module is intentionally organized around SSA operations, values, blocks,
and regions.  It does not classify whole kernels into matmul/reduction/etc.
plans before lowering.  Backend-specific code is limited to spelling scalar
expressions, loops, buffers, and launch wrappers.
"""

import json
import re
from dataclasses import replace
from typing import Any, Mapping

from ninetoothed.backends.core import Artifact
from ninetoothed.backends.emitters.analysis import (
    atomic_output_tensors as _atomic_output_tensors,
)
from ninetoothed.backends.emitters.analysis import (
    program_value_types as _program_value_types,
)
from ninetoothed.backends.emitters.analysis import schedule_int as _schedule_int
from ninetoothed.backends.emitters.analysis import (
    value_depends_on as _value_depends_on,
)
from ninetoothed.backends.emitters.analysis import (
    walk_ops as _walk_ops,
)
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.context import (
    CooperativeDotPlan as _CooperativeDotPlan,
)
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.backends.emitters.context import TensorInfo as _TensorInfo
from ninetoothed.backends.emitters.expressions import (
    default_strides as _default_strides,
)
from ninetoothed.backends.emitters.expressions import (
    integer_expr as _integer_expr,
)
from ninetoothed.backends.emitters.expressions import (
    is_one_expr as _is_one_expr,
)
from ninetoothed.backends.emitters.expressions import (
    is_zero_expr as _is_zero_expr,
)
from ninetoothed.backends.emitters.expressions import (
    linearized_index as _linearized_index,
)
from ninetoothed.backends.emitters.expressions import (
    normalize_dtype as _normalize_dtype,
)
from ninetoothed.backends.emitters.expressions import (
    product as _product,
)
from ninetoothed.backends.emitters.expressions import (
    replace_index_symbol as _replace_index_symbol,
)
from ninetoothed.backends.emitters.expressions import (
    replace_symbols as _replace_symbols,
)
from ninetoothed.backends.emitters.expressions import (
    rewrite_index_math as _rewrite_index_math,
)
from ninetoothed.backends.emitters.expressions import (
    shape_dim as _shape_dim,
)
from ninetoothed.backends.emitters.expressions import (
    stride_dim as _stride_dim,
)
from ninetoothed.backends.emitters.expressions import (
    symbols_in_text as _symbols_in_text,
)
from ninetoothed.backends.emitters.expressions import (
    valid_symbol as _valid_symbol,
)
from ninetoothed.ir import Kernel, TensorSpec, ir_to_dict, ssa

_BINARY = {
    "add": "+",
    "and": "&",
    "sub": "-",
    "subtract": "-",
    "mul": "*",
    "multiply": "*",
    "div": "/",
    "truediv": "/",
    "mod": "%",
    "bitwise_and": "&",
    "bitwise_or": "|",
    "bitwise_xor": "^",
    "bitwise_left_shift": "<<",
    "bitwise_right_shift": ">>",
    "or": "|",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}

_UNARY = {
    "neg": "-",
    "pos": "+",
    "not": "!",
    "invert": "~",
}


_Target = EmitterTarget


def emit(kernel: Kernel, target: EmitterTarget) -> Artifact:
    if kernel.ssa is None:
        raise ValueError("Backend emission requires ssa.Program.")

    backend = target.backend
    block = kernel.ssa.blocks[0] if kernel.ssa.blocks else ssa.Block()
    walked_ops = tuple(_walk_ops(block.operations))
    operations = {result.name: op for op in walked_ops for result in op.results}
    stores = tuple(
        op for op in walked_ops if op.opcode == "mem.store" and len(op.operands) == 2
    )
    outputs = tuple(
        dict.fromkeys(
            (
                *[store.operands[1] for store in stores],
                *_atomic_output_tensors(walked_ops, operations),
            )
        )
    ) or tuple(value.name for value in kernel.ssa.outputs)
    public_variables = tuple(
        tensor.name
        for tensor in kernel.tensors
        if tensor.name not in outputs and not tensor.constexpr
    )
    auxiliary_bindings = _auxiliary_pointer_bindings(kernel.tensors)
    variables = public_variables + tuple(
        binding["name"] for binding in auxiliary_bindings
    )
    shape_params = _shape_params(kernel.tensors, block.operations)
    source, render_context = _render_source(
        kernel,
        target,
        shape_params=shape_params,
    )
    metadata = {
        "backend": backend.value,
        "kernel_name": kernel.kernel_name,
        "lowering_ir": "ssa.Program",
        "source_route": target.source_route,
        "shape_params": shape_params,
        "variables": variables,
        "outputs": outputs,
        "auxiliary_bindings": auxiliary_bindings,
        "ssa": ir_to_dict(kernel.ssa),
        "kernel_metadata": dict(kernel.metadata),
        "ssa_metadata": dict(kernel.ssa.metadata),
        "ssa_pass_trace": tuple(kernel.ssa.metadata.get("pass_trace", ())),
        "ssa_schedule": dict(kernel.ssa.metadata.get("schedule", {})),
        "ssa_pipeline_selection": dict(
            kernel.ssa.metadata.get("pipeline_selection", {})
        ),
        "ssa_optimization": dict(kernel.ssa.metadata.get("optimization", {})),
        "tensors": [ir_to_dict(tensor) for tensor in kernel.tensors],
        "launch_grid": (render_context.grid_total,),
        "launch_block": _launch_block(kernel),
        "program_mode": {
            "block": render_context.block_program,
            "scalar": render_context.scalar_program,
            "vector": render_context.vector_program,
        },
        "vector_numel_limit": target.max_vector_numel,
    }

    return Artifact(
        backend=backend,
        kernel_name=kernel.kernel_name,
        language=target.language,
        sources={
            f"{kernel.kernel_name}.{target.suffix}": source,
            f"{kernel.kernel_name}.{backend.value}.json": json.dumps(
                ir_to_dict(metadata), indent=2
            ),
        },
        entrypoint=target.entrypoint(kernel.kernel_name),
        metadata=metadata,
    )


def _row_reduction_schedule(
    program: ssa.Program,
    target: _Target,
) -> Mapping[str, Any] | None:
    schedule = program.metadata.get("schedule", {})

    if not isinstance(schedule, Mapping):
        return None

    reduction = schedule.get("reduction", {})

    if not isinstance(reduction, Mapping):
        return None

    if reduction.get("mode") == "scalar-fallback" and not reduction.get(
        "emittable", True
    ):
        raise ValueError(
            "Reduction outputs with different domains require separate kernels."
        )

    if reduction.get("mode") != "row-vector":
        return None

    extent = _static_integer(reduction.get("extent"))
    limit = target.max_vector_numel

    if limit is not None and extent is not None and extent > limit:
        block = 1 << (extent - 1).bit_length()
        raise ValueError(
            f"Row-vector reduction for {target.backend.value} extent {extent} "
            f"requires BLOCK={block}, exceeding the backend tensor numel "
            f"limit {limit}; hierarchical reduction is not implemented."
        )
    return reduction


def _static_integer(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _render_source(
    kernel: Kernel,
    target: _Target,
    *,
    shape_params: tuple[str, ...],
) -> tuple[str, ModuleRenderContext]:
    program = kernel.ssa
    assert program is not None
    reduction_schedule = _row_reduction_schedule(program, target)
    block = program.blocks[0] if program.blocks else ssa.Block()
    tensor_infos = {tensor.name: _tensor_info(tensor) for tensor in kernel.tensors}
    auxiliary_bindings = _auxiliary_pointer_bindings(kernel.tensors)
    tensor_infos.update(
        {
            binding["name"]: _TensorInfo(
                ndim=1,
                shape=(str(binding.get("storage_extent", "n")),),
                dtype=str(binding["dtype"]),
                name=str(binding["name"]),
            )
            for binding in auxiliary_bindings
        }
    )
    walked_ops = tuple(_walk_ops(block.operations))
    operations = {result.name: op for op in walked_ops for result in op.results}
    value_types = _program_value_types(program)
    stores = tuple(
        op for op in walked_ops if op.opcode == "mem.store" and len(op.operands) == 2
    )
    atomic_outputs = _atomic_output_tensors(walked_ops, operations)
    outputs = tuple(
        dict.fromkeys((*[store.operands[1] for store in stores], *atomic_outputs))
    ) or tuple(value.name for value in program.outputs)
    public_variables = tuple(
        tensor.name
        for tensor in kernel.tensors
        if tensor.name not in outputs and not tensor.constexpr
    )
    variables = public_variables + tuple(
        binding["name"] for binding in auxiliary_bindings
    )

    if "index" in {*variables, *outputs, *shape_params}:
        target = replace(target, index_name="__nt_index")

    output = (
        outputs[0]
        if outputs
        else (kernel.tensors[-1].name if kernel.tensors else "out")
    )
    output_info = tensor_infos.get(output)
    outer_axes = _tensor_axes(output_info, fallback=("n",))
    primary_store = next(
        (store for store in stores if store.operands[1] == output), None
    )
    store_value_axes = _store_value_axes(primary_store, value_types)
    primary_atomic = None

    if store_value_axes is None:
        primary_atomic = next(
            (
                operation
                for operation in walked_ops
                if operation.opcode == "mem.atomic_add" and operation.operands
            ),
            None,
        )

        if primary_atomic is not None:
            store_value_axes = _value_type_axes(
                value_types.get(primary_atomic.operands[-1])
            )

    if store_value_axes is None:
        value_axes = (
            _operation_domain_axes(walked_ops, value_types)
            or _value_type_axes(value_types.get(output))
            or tuple(
                str(dim)
                for dim in (output_info.attrs or {}).get("application_shape", ())
            )
        )
    else:
        value_axes = store_value_axes

    if target.vector_value_semantics and reduction_schedule is not None:
        value_axes = tuple(
            str(axis) for axis in reduction_schedule.get("value_shape", ())
        )

    output_attrs = output_info.attrs or {} if output_info is not None else {}
    split_outer_inner = bool(
        value_axes
        and output_info is not None
        and output_attrs.get("application_shape")
        and output_attrs.get("dtype_shapes")
    )

    has_dot = any(op.opcode in {"linalg.dot", "linalg.matmul"} for op in walked_ops)
    vector_block_program = bool(
        target.vector_value_semantics
        and split_outer_inner
        and len(value_axes) == 2
        and has_dot
    )
    vector_scalar_program = bool(
        target.vector_value_semantics and primary_atomic is not None and not value_axes
    )
    vector_reduction_program = bool(
        target.vector_value_semantics
        and reduction_schedule is not None
        and not vector_block_program
    )
    native_block_program = bool(
        target.native_block_matmul
        and split_outer_inner
        and len(value_axes) == 2
        and has_dot
        and program.metadata.get("optimization", {}).get("preserve_linalg")
    )

    if vector_scalar_program:
        axes = outer_axes
        total = _target_index_expr(target, _product(outer_axes))
        grid_total = total
        outer_index_expr = target.program_id(0)
        inner_index_expr = target.program_id(0)
    elif vector_block_program:
        axes = value_axes
        total = _target_index_expr(target, _product(value_axes))
        grid_total = _target_index_expr(target, _product(outer_axes))
        outer_index_expr = target.program_id(0)
        inner_index_expr = "0"
    elif vector_reduction_program:
        axes = value_axes
        reduction_axis = int(reduction_schedule["axis"])
        parallel_shape = tuple(
            str(axis) for axis in reduction_schedule.get("parallel_shape", ())
        )
        parallel_total = _product(parallel_shape)
        program_index = target.program_id(0)
        parallel_index = (
            "0"
            if _is_one_expr(parallel_total)
            else _target_index_expr(target, f"({program_index}) % ({parallel_total})")
        )
        coordinate_exprs = []
        parallel_dim = 0

        for dim in range(len(value_axes)):
            if dim == reduction_axis:
                coordinate_exprs.append("offsets")
                continue

            coordinate_exprs.append(
                _axis_offset_expr(
                    parallel_shape,
                    parallel_dim,
                    parallel_index,
                    target,
                )
            )
            parallel_dim += 1

        coordinate_exprs = tuple(coordinate_exprs)
        total = _target_index_expr(target, str(reduction_schedule["extent"]))
        outer_program_shape = tuple(
            str(axis) for axis in reduction_schedule.get("program_shape", ())
        )
        grid_total = _target_index_expr(
            target,
            _product((*outer_program_shape, *parallel_shape)),
        )
        outer_index_expr = (
            "0"
            if not outer_program_shape
            else program_index
            if _is_one_expr(parallel_total)
            else _target_index_expr(
                target, f"floor(({program_index})/({parallel_total}))"
            )
        )
        inner_index_expr = _linearized_index(coordinate_exprs, value_axes)
    elif native_block_program:
        axes = value_axes
        total = _target_index_expr(target, _product(value_axes))
        tile_rows = f"(({value_axes[0]}) + 15) / 16"
        tile_cols = f"(({value_axes[1]}) + 15) / 16"
        grid_total = _target_index_expr(
            target, f"({_product(outer_axes)}) * ({tile_rows}) * ({tile_cols})"
        )
        outer_index_expr = "nt_outer_index"
        inner_index_expr = f"(nt_matrix_row) * ({value_axes[1]}) + nt_matrix_col"
    elif split_outer_inner:
        axes = value_axes
        inner_total = _product(value_axes)
        total = _target_index_expr(
            target, f"({_product(outer_axes)}) * ({inner_total})"
        )
        grid_total = total
        outer_index_expr = _target_index_expr(
            target, f"floor(({target.index_name})/({inner_total}))"
        )
        inner_index_expr = _target_index_expr(
            target, f"({target.index_name} % ({inner_total}))"
        )
    else:
        axes = outer_axes
        total = _target_index_expr(target, _product(axes))
        grid_total = total
        outer_index_expr = target.index_name
        inner_index_expr = target.index_name

    body = _render_body(
        kernel,
        target,
        block.operations,
        operations,
        value_types,
        tensor_infos,
        outputs,
        axes,
        total,
        outer_index_expr,
        inner_index_expr,
        block_program=vector_block_program,
        native_block_program=native_block_program,
        vector_program=vector_reduction_program,
        coordinate_exprs=(coordinate_exprs if vector_reduction_program else ()),
        reduction_schedule=(reduction_schedule if vector_reduction_program else None),
    )
    body = _with_contiguous_1d_fast_path(
        kernel,
        target,
        body,
        operations,
        value_types,
        tensor_infos,
        outputs,
        axes,
        total,
        outer_index_expr,
        inner_index_expr,
        block_program=vector_block_program,
        native_block_program=native_block_program,
        vector_program=vector_reduction_program,
    )

    context = ModuleRenderContext(
        kernel=kernel,
        variables=variables,
        outputs=outputs,
        shape_params=shape_params,
        total=total,
        body=body,
        tensors=tensor_infos,
        value_types=value_types,
        operations=operations,
        stores=stores,
        outer_axes=outer_axes,
        grid_total=grid_total,
        axes=axes,
        vector_program=vector_reduction_program,
        block_program=(
            vector_block_program
            if target.vector_value_semantics
            else native_block_program
        ),
        scalar_program=vector_scalar_program,
    )

    return target.render_module(context), context


def _launch_block(kernel: Kernel) -> tuple[str, ...]:
    if kernel.ssa is None:
        return ()

    target_backend = kernel.ssa.metadata.get("target_backend")
    warps = kernel.compiler_options.get("num_warps")
    fallback = (
        32 * warps if target_backend == "triton" and isinstance(warps, int) else 256
    )
    threads = _schedule_int(kernel, "threads", fallback)

    return (str(threads),)


def _with_contiguous_1d_fast_path(
    kernel: Kernel,
    target: _Target,
    generic_body: str,
    operations: Mapping[str, ssa.Operation],
    value_types: Mapping[str, ssa.Type],
    tensor_infos: Mapping[str, _TensorInfo],
    outputs: tuple[str, ...],
    axes: tuple[str, ...],
    total: str,
    outer_index_expr: str,
    inner_index_expr: str,
    *,
    block_program: bool,
    native_block_program: bool,
    vector_program: bool,
) -> str:
    logical_infos = tuple(tensor_infos[tensor.name] for tensor in kernel.tensors)

    if vector_program:
        return generic_body

    if not logical_infos or any(
        info.ndim != 1
        or info.attrs is None
        or info.attrs.get("jagged_offsets_param")
        or len(info.source_strides) != 1
        for info in logical_infos
    ):
        return generic_body

    stride_params = tuple(
        info.source_strides[0]
        for info in logical_infos
        if not _is_one_expr(info.source_strides[0])
    )

    if not stride_params:
        return generic_body

    contiguous_infos = dict(tensor_infos)

    for info in logical_infos:
        attrs = dict(info.attrs or {})
        replacements = {stride: "1" for stride in info.source_strides}
        attrs["source_strides"] = ("1",)

        if attrs.get("view_linear_offset"):
            attrs["view_linear_offset"] = _simplify_unit_stride_expr(
                _replace_symbols(str(attrs["view_linear_offset"]), replacements)
            )

        attrs["access_templates"] = tuple(
            dict(template)
            | {
                "linear_offset": _replace_symbols(
                    str(template.get("linear_offset", "")), replacements
                )
            }
            for template in attrs.get("access_templates", ())
        )
        contiguous_infos[info.name] = replace(
            info,
            source_strides=("1",),
            view_linear_offset=(
                _simplify_unit_stride_expr(
                    _replace_symbols(info.view_linear_offset, replacements)
                )
                if info.view_linear_offset
                else None
            ),
            attrs=attrs,
        )

    contiguous_body = _render_body(
        kernel,
        target,
        kernel.ssa.blocks[0].operations if kernel.ssa is not None else (),
        operations,
        value_types,
        contiguous_infos,
        outputs,
        axes,
        total,
        outer_index_expr,
        inner_index_expr,
        block_program=block_program,
        native_block_program=native_block_program,
        layout_contiguous=True,
        vector_program=vector_program,
    )
    predicate = (
        " && ".join(f"({stride} == 1)" for stride in stride_params)
        if target.c_style_syntax
        else " and ".join(f"({stride} == 1)" for stride in stride_params)
    )

    if target.c_style_syntax:
        return (
            f"if ({predicate}) {{\n"
            f"{_indent_block(contiguous_body, '    ')}\n"
            "} else {\n"
            f"{_indent_block(generic_body, '    ')}\n"
            "}"
        )
    return (
        f"if {predicate}:\n"
        f"{_indent_block(contiguous_body, '    ')}\n"
        "else:\n"
        f"{_indent_block(generic_body, '    ')}"
    )


def _simplify_unit_stride_expr(expr: str) -> str:
    return re.sub(
        r"\(?\s*index\s*\)?\s*\*\s*\(?\s*1\s*\)?",
        "index",
        expr,
    )


def _render_body(
    kernel: Kernel,
    target: _Target,
    operations: tuple[ssa.Operation, ...],
    op_by_result: Mapping[str, ssa.Operation],
    value_types: Mapping[str, ssa.Type],
    tensor_infos: Mapping[str, _TensorInfo],
    outputs: tuple[str, ...],
    axes: tuple[str, ...],
    total: str,
    outer_index_expr: str,
    inner_index_expr: str,
    *,
    block_program: bool = False,
    native_block_program: bool = False,
    layout_contiguous: bool = False,
    vector_program: bool = False,
    coordinate_exprs: tuple[str, ...] = (),
    reduction_schedule: Mapping[str, Any] | None = None,
) -> str:
    output = outputs[0] if outputs else "out"
    lines: list[str] = []
    enable_index_cse = (
        not target.vector_value_semantics and outer_index_expr != inner_index_expr
    )

    if block_program:
        coordinate_exprs = target.block_coords(axes)
        inner_index_expr = _linearized_index(coordinate_exprs, axes)
    elif enable_index_cse:
        index_type = ssa.Type(kind="index", dtype="index")

        if outer_index_expr not in {target.index_name, "nt_outer_index"}:
            lines.append(
                target.local_decl(index_type, "nt_outer_index", outer_index_expr)
            )
            outer_index_expr = "nt_outer_index"

        if inner_index_expr != target.index_name:
            lines.append(
                target.local_decl(index_type, "nt_inner_index", inner_index_expr)
            )
            inner_index_expr = "nt_inner_index"

        coord_names: list[str] = []

        for dim in range(len(axes)):
            name = f"nt_i{dim}"
            lines.append(
                target.local_decl(
                    index_type,
                    name,
                    _axis_offset_expr(axes, dim, inner_index_expr, target),
                )
            )
            coord_names.append(name)

        coordinate_exprs = tuple(coord_names)

    ctx = _EmitContext(
        target=target,
        kernel=kernel,
        program=kernel.ssa,  # type: ignore[arg-type]
        operations=op_by_result,
        value_types=value_types,
        lines=lines,
        memo={},
        tensor_infos=tensor_infos,
        output=output,
        output_axes=axes,
        index_expr=inner_index_expr,
        outer_index_expr=outer_index_expr,
        inner_index_expr=inner_index_expr,
        mask_expr=(
            "nt_matrix_active"
            if native_block_program
            else "mask"
            if target.vector_value_semantics and not block_program
            else None
        ),
        row_expr=coordinate_exprs[0]
        if len(coordinate_exprs) >= 1
        else _axis_offset_expr(axes, 0, inner_index_expr, target)
        if len(axes) >= 2
        else None,
        col_expr=coordinate_exprs[1]
        if len(coordinate_exprs) >= 2
        else _axis_offset_expr(axes, 1, inner_index_expr, target)
        if len(axes) >= 2
        else None,
        coordinate_exprs=coordinate_exprs,
        bindings={},
        temp_counter=[0],
        materialized={},
        indent="",
        block_program=block_program,
        native_block_program=native_block_program,
        layout_contiguous=layout_contiguous,
        vector_program=vector_program,
        reduction_lane="offsets" if reduction_schedule is not None else None,
        scheduled_reductions=frozenset(
            str(result) for result in (reduction_schedule or {}).get("reductions", ())
        ),
    )

    for op in operations:
        if _is_top_level_effect(op):
            _emit_operation(op, ctx)

    if not ctx.lines:
        ctx.lines.append("pass" if not target.c_style_syntax else "/* no-op */")
    return "\n".join(ctx.lines)


def _is_top_level_effect(op: ssa.Operation) -> bool:
    """Return whether an operation must be emitted without a value user.

    Pure SSA producers are demand-driven: the backend emits them when a store,
    loop, if, or other effect recursively asks for the value.  This keeps the
    source generator operator-agnostic while avoiding dead top-level temporaries.
    """
    if op.opcode in {"mem.store", "mem.atomic_add"}:
        return True

    if op.opcode in {"scf.for", "scf.if"} and not op.results:
        return True
    return False


def _local_symbol(name: str, ctx: _EmitContext) -> str:
    base = ctx.target.symbol(name)

    if not name.startswith("%") or not ctx.local_suffix:
        return base
    return f"{base}{ctx.local_suffix}"


def _nested_local_suffix(ctx: _EmitContext, label: str) -> str:
    clean = re.sub(r"\W+", "_", label).strip("_") or "region"
    suffix = f"_{clean}_body"

    return f"{ctx.local_suffix}{suffix}" if ctx.local_suffix else suffix


def _coords_use_reduction_lane(coords: tuple[str, ...], ctx: _EmitContext) -> bool:
    return (
        ctx.vector_program
        and ctx.reduction_lane is not None
        and any(ctx.reduction_lane in _expression_symbols(coord) for coord in coords)
    )


def _expression_symbols(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))


def _mask_for_coords(coords: tuple[str, ...], ctx: _EmitContext) -> str | None:
    if not ctx.vector_program:
        return ctx.mask_expr
    return ctx.mask_expr if _coords_use_reduction_lane(coords, ctx) else None


def _mask_for_axes(axes: tuple[str, ...], ctx: _EmitContext) -> str | None:
    return _mask_for_coords(_current_coords(axes, ctx), ctx)


def _emit_operation(op: ssa.Operation, ctx: _EmitContext) -> None:
    if op.opcode == "scf.yield":
        return

    if op.opcode == "mem.store":
        value_name = op.operands[0]
        value = _emit_value(value_name, ctx)
        tensor = op.operands[1]
        view_index = _store_index(op, ctx)
        info = ctx.tensor_infos.get(tensor)

        if op.attrs.get("source"):
            rendered = tuple(
                _emit_index_value(str(index), ctx)
                for index in op.attrs.get("indices", ())
            )
            source_axes = _source_axes(info, fallback=())
            remaining = max(0, len(source_axes) - len(rendered))

            if remaining:
                value_axes = _value_axes(value_name, ctx)
                implicit = _current_coords(value_axes, ctx)

                if len(implicit) < remaining:
                    implicit = _coords_from_linear(
                        ctx.inner_index_expr, source_axes[-remaining:], ctx.target
                    )

                rendered = (*rendered, *implicit[-remaining:])

            store_index = _target_index_expr(
                ctx.target, _source_linear_index(info, rendered)
            )
        else:
            target_level = int(
                op.attrs.get("target_dtype_level", _dtype_level(tensor, ctx))
            )
            base_level = int(
                op.attrs.get("base_dtype_level", _dtype_level(tensor, ctx))
            )
            extract_indices: tuple[str, ...] = ()

            if target_level > base_level:
                target_axes = tuple(
                    str(dim) for dim in op.attrs.get("target_shape", ())
                )
                rendered = tuple(
                    _emit_store_index_value(str(index), target_axes, ctx)
                    for index in op.attrs.get("indices", ())
                )
                extract_indices = rendered
                view_index = ctx.inner_index_expr

            store_index = _target_index_expr(
                ctx.target,
                _source_index_for_value(
                    info,
                    view_index,
                    ctx,
                    level=target_level,
                    extract_indices=extract_indices,
                ),
            )

        store_index = _materialize_index_expr(store_index, ctx)

        if op.attrs.get("source"):
            mask = _source_bounds_mask(
                info,
                rendered,
                base_mask=_mask_for_coords(rendered, ctx),
            )
        else:
            mask = _store_mask(
                ctx.target,
                _mask_for_axes(_value_axes(tensor, ctx), ctx),
                info,
                view_index,
                ctx=ctx,
                level=target_level,
                extract_indices=extract_indices,
            )

        mask = _materialize_bool_expr(mask, ctx)
        ctx.lines.append(ctx.target.store(tensor, store_index, value, mask=mask))

        return

    if op.opcode == "mem.atomic_add":
        expression = _operation_expr(op, ctx)
        ctx.lines.append(expression + (";" if ctx.target.c_style_syntax else ""))

        return

    if op.opcode == "scf.for" and not op.results:
        _emit_scf_for("loop", op, ctx)

        return

    if op.opcode == "scf.if" and not op.results:
        _emit_scf_if_statement(op, ctx)

        return

    for result in op.results:
        _emit_value(result.name, ctx)


def _emit_value(name: str, ctx: _EmitContext) -> str:
    if ctx.bindings and name in ctx.bindings:
        return ctx.bindings[name]

    if name in ctx.memo:
        return ctx.memo[name]

    if not name.startswith("%"):
        if name not in ctx.tensor_infos:
            if _is_bool_scalar_value(name, ctx) and ctx.target.tir_value_semantics:
                return f"({name} != 0)"
            return name
        return _tensor_value(name, ctx)

    op = ctx.operations[name]
    local = _local_symbol(name, ctx)

    if op.opcode.startswith("reduce."):
        operand_type = ctx.value_types.get(op.operands[0]) if op.operands else None

        if operand_type is not None and operand_type.kind == "scalar":
            operand = _emit_value(op.operands[0], ctx)
            ctx.memo[name] = operand

            return operand

        if ctx.vector_program and name in ctx.scheduled_reductions:
            operator = op.opcode[len("reduce.") :]
            operand_axes = _value_axes(op.operands[0], ctx)
            operand = _emit_element(
                op.operands[0], _current_coords(operand_axes, ctx), ctx
            )
            result_type = ssa.Type(
                kind="scalar",
                dtype=op.results[0].type.dtype if op.results else "float32",
            )
            identity = _reduction_identity(operator, result_type, ctx.target)

            if ctx.mask_expr is not None:
                operand = ctx.target.where(ctx.mask_expr, operand, identity)

            expr = ctx.target.vector_reduce(operator, operand, 0)
            ctx.lines.append(ctx.target.local_decl(result_type, local, expr))
            ctx.memo[name] = local

            return local

        if ctx.block_program:
            operator = op.opcode[len("reduce.") :]
            operand = _emit_value(op.operands[0], ctx)
            axis = int(op.attrs.get("axis", 0) or 0)
            expr = ctx.target.vector_reduce(operator, operand, axis)
            ctx.lines.append(ctx.target.local_decl(op.results[0].type, local, expr))
            ctx.memo[name] = local

            return local

        if op.results and op.results[0].type.kind == "tensor":
            expr = _emit_reduce_element(
                op, _current_coords(_value_axes(name, ctx), ctx), ctx, local=local
            )
        else:
            expr = _emit_reduce(local, op, ctx)

        ctx.memo[name] = expr

        return expr

    if op.opcode == "scf.for":
        expr = _emit_scf_for(local, op, ctx)
        ctx.memo[name] = expr or local

        return ctx.memo[name]

    if op.opcode == "scf.if":
        if len(op.results) > 1:
            _emit_scf_if_results(op, ctx)

            return ctx.memo[name]

        expr = _scf_if_expr(op, ctx)
    elif (
        not ctx.target.vector_value_semantics
        and op.results
        and op.results[0].type.kind == "tensor"
        and op.opcode.startswith("arith.")
        and ctx.reduce_axis is not None
    ) or _should_emit_tensor_value_as_element(op):
        expr = _emit_element(name, _reduction_value_coords(name, ctx), ctx)
    else:
        expr = _operation_expr(op, ctx)

    result_type = ctx.target.arithmetic_result_type(op, ctx)
    ctx.lines.append(ctx.target.local_decl(result_type, local, expr))
    ctx.memo[name] = local

    return local


def _reduction_value_coords(name: str, ctx: _EmitContext) -> tuple[str, ...]:
    axes = _value_axes(name, ctx)
    coords = list(_current_coords(axes, ctx))

    if ctx.reduce_index is None:
        return tuple(coords)

    if ctx.reduce_flattened:
        return _coords_from_linear(ctx.reduce_index, axes, ctx.target)

    if ctx.reduce_axis is None:
        return tuple(coords)

    axis = ctx.reduce_axis if ctx.reduce_axis >= 0 else ctx.reduce_axis + len(axes)

    if 0 <= axis < len(coords):
        coords[axis] = ctx.reduce_index
    return tuple(coords)


def _is_bool_scalar_value(name: str, ctx: _EmitContext) -> bool:
    type_ = ctx.value_types.get(name)

    return bool(
        type_ is not None
        and type_.kind == "scalar"
        and _normalize_dtype(type_.dtype) == "bool"
    )


def _operation_expr(op: ssa.Operation, ctx: _EmitContext) -> str:
    target = ctx.target
    opcode = op.opcode

    if opcode == "arith.constant":
        return target.literal(op.attrs.get("value"))

    if opcode == "index.offset":
        axes = op.results[0].type.shape if op.results else ()

        return _emit_offset_element(
            op,
            _current_coords(tuple(str(axis) for axis in axes), ctx),
            ctx,
        )

    if opcode == "shape.dim":
        tensor = op.operands[0]
        info = ctx.tensor_infos.get(tensor)
        axes = (
            _source_axes(info, fallback=ctx.output_axes)
            if op.attrs.get("source")
            else _value_axes(tensor, ctx)
        )

        return _target_index_expr(ctx.target, _shape_dim(axes, op.attrs.get("dim", 0)))

    if opcode == "tensor.stride":
        tensor = op.operands[0]
        info = ctx.tensor_infos.get(tensor)

        if op.attrs.get("source"):
            strides = _source_strides(info)
            dim = int(op.attrs.get("dim", 0) or 0)

            return strides[dim] if dim < len(strides) else "1"

        axes = _tensor_axes(ctx.tensor_infos.get(tensor), fallback=ctx.output_axes)

        return _stride_dim(axes, op.attrs.get("dim", 0))

    if opcode == "mem.data_ptr":
        return ctx.target.tensor_ref(op.operands[0])

    if opcode == "mem.load":
        return _emit_pointer_load(
            op.operands[0],
            _current_coords(_value_axes(op.results[0].name, ctx), ctx),
            ctx,
        )

    if opcode == "mem.atomic_add":
        operands = tuple(_emit_value(operand, ctx) for operand in op.operands)

        value_type = ctx.value_types.get(op.operands[-1])
        dtype = _normalize_dtype(None if value_type is None else value_type.dtype)

        return target.atomic_add(operands, dtype)

    if opcode == "tensor.view":
        if ctx.block_program:
            return ctx.target.render_view(op, ctx)
        return _emit_value(op.operands[0], ctx)

    if opcode in {"tensor.zeros", "tensor.empty"}:
        return "0.0"

    if opcode == "tensor.full":
        if op.operands:
            return _emit_value(op.operands[0], ctx)
        return target.literal(op.attrs.get("value", 0.0))

    if opcode == "tensor.extract":
        tensor = op.operands[0]
        indices = tuple(_emit_index_value(operand, ctx) for operand in op.operands[1:])

        if op.attrs.get("source"):
            return _load_source_tensor(tensor, indices, ctx)

        index = _linearized_index(indices, _value_axes(tensor, ctx))

        return _load_tensor(tensor, index, ctx)

    if opcode == "tensor.cast":
        return _cast_value(op, _emit_value(op.operands[0], ctx), ctx)

    if opcode == "select.where":
        args = tuple(_emit_value(operand, ctx) for operand in op.operands)
        args = (_materialize_bool_expr(args[0], ctx) or args[0], args[1], args[2])

        return target.where(args[0], args[1], args[2])

    if opcode.startswith("cmp."):
        return _binary_expr(opcode[len("cmp.") :], op, ctx)

    if opcode.startswith("arith."):
        operator = opcode[len("arith.") :]
        args = tuple(_emit_value(operand, ctx) for operand in op.operands)

        if operator in _UNARY:
            return f"({_UNARY[operator]}{args[0]})"

        if operator == "floordiv":
            return (
                f"(({args[0]}) // ({args[1]}))"
                if not target.c_style_syntax
                else f"(({args[0]}) / ({args[1]}))"
            )

        if operator == "pow":
            return target.call("pow", args)

        if operator in {"maximum", "max"}:
            return target.call("maximum", args)

        if operator in {"minimum", "min"}:
            return target.call("minimum", args)
        return _binary_expr(operator, op, ctx)

    if opcode.startswith("math."):
        name = opcode[len("math.") :]
        callee = str(op.attrs.get("callee", ""))

        if target.vector_value_semantics and "libdevice." in callee:
            name = f"libdevice.{name}"
        return target.call(
            name,
            tuple(_emit_value(operand, ctx) for operand in op.operands),
        )

    if opcode.startswith("call."):
        return target.call(
            opcode[len("call.") :],
            tuple(_emit_value(operand, ctx) for operand in op.operands),
        )

    if opcode == "symbol.attr":
        return _target_index_expr(target, str(op.attrs.get("expr", "0")))

    if opcode == "tuple.construct":
        return (
            "(" + ", ".join(_emit_value(operand, ctx) for operand in op.operands) + ")"
        )

    if opcode in {"linalg.matmul", "linalg.dot"}:
        return _emit_linalg_dot(op, ctx)

    if opcode == "linalg.transpose":
        return _emit_value(op.operands[0], ctx)

    raise ValueError(f"Unsupported SSA opcode `{opcode}` for unified backend emitter.")


def _binary_expr(operator: str, op: ssa.Operation, ctx: _EmitContext) -> str:
    if (
        ctx.target.c_style_syntax
        and operator in {"mul", "multiply"}
        and op.results
        and op.results[0].type.kind == "tensor"
        and all(operand in ctx.tensor_infos for operand in op.operands)
    ):
        return _element_binary(
            operator,
            op,
            _current_coords(_value_axes(op.results[0].name, ctx), ctx),
            ctx,
        )

    args = tuple(_emit_value(operand, ctx) for operand in op.operands)
    args = ctx.target.coerce_binary_args(op, args, ctx)
    symbol = _BINARY[operator]

    return f"({args[0]} {symbol} {args[1]})"


def _emit_linalg_dot(
    op: ssa.Operation, ctx: _EmitContext, coords: tuple[str, ...] | None = None
) -> str:
    if len(op.operands) < 2 or not op.results:
        return ctx.target.call(
            "dot", tuple(_emit_value(operand, ctx) for operand in op.operands)
        )

    lhs, rhs = op.operands[:2]
    lhs_axes = _value_axes(lhs, ctx)
    rhs_axes = _value_axes(rhs, ctx)
    result_axes = tuple(str(dim) for dim in op.results[0].type.shape)

    if ctx.native_block_program and len(lhs_axes) == 2 and len(rhs_axes) == 2:
        result = ctx.target.emit_block_dot(op, ctx, coords=coords)

        if result is not None:
            return result

    if ctx.block_program and len(lhs_axes) == 2 and len(rhs_axes) == 2:
        lhs_value = _emit_element(lhs, ctx.target.block_coords(lhs_axes), ctx)
        rhs_value = _emit_element(rhs, ctx.target.block_coords(rhs_axes), ctx)

        return ctx.target.call("block_dot", (lhs_value, rhs_value))

    if not lhs_axes or not rhs_axes:
        return ctx.target.call(
            "dot", tuple(_emit_value(operand, ctx) for operand in op.operands)
        )

    local = f"{_local_symbol(op.results[0].name, ctx)}_dot"
    accumulator_dtype = _dot_accumulator_dtype(op, ctx)
    acc_type = ssa.Type(kind="scalar", dtype=accumulator_dtype)
    init = "0.0"

    if ctx.target.vector_value_semantics and ctx.mask_expr is not None:
        dtype = _normalize_dtype(acc_type.dtype or "float32")
        init = ctx.target.vector_splat("(BLOCK,)", init, dtype)

    mutable = _uses_mutable_scalar_slots(ctx.target)

    if mutable:
        ctx.lines.extend(_mutable_scalar_decl_lines(ctx.target, acc_type, local, init))
        acc_expr = _mutable_scalar_read(ctx.target, local)
    else:
        ctx.lines.append(ctx.target.local_decl(acc_type, local, init))
        acc_expr = local

    k_extent = lhs_axes[-1]
    loop_var = f"{local}_k"
    ctx.lines.append(ctx.target.loop_header(loop_var, "0", k_extent, "1"))
    body_lines: list[str] = []
    body = ctx.child(
        lines=body_lines,
        memo=dict(ctx.memo),
        local_suffix=_nested_local_suffix(ctx, local),
    )
    result_coords = (
        tuple(coords) if coords is not None else _current_coords(result_axes, ctx)
    )
    lhs_coords, rhs_coords = _dot_operand_coords(
        lhs_axes, rhs_axes, result_coords, loop_var
    )
    lhs_value, lhs_mask = _emit_dot_operand(lhs, lhs_coords, body)
    rhs_value, rhs_mask = _emit_dot_operand(rhs, rhs_coords, body)
    lhs_value = _cast_dot_operand(lhs, lhs_value, accumulator_dtype, ctx)
    rhs_value = _cast_dot_operand(rhs, rhs_value, accumulator_dtype, ctx)
    product = f"({lhs_value} * {rhs_value})"
    product_masks = tuple(mask for mask in (lhs_mask, rhs_mask) if mask)

    if product_masks:
        product_mask = " && ".join(f"({mask})" for mask in product_masks)
        product = f"(({product_mask}) ? ({product}) : 0.0)"

    body_lines.append(
        _assign_scalar(
            ctx.target,
            local,
            f"{acc_expr} + {product}",
            mutable=mutable,
        )
    )
    ctx.lines.extend(_indent_lines(body_lines, ctx.target))

    if ctx.target.c_style_syntax:
        ctx.lines.append("}")
    return acc_expr


def _emit_dot_operand(
    name: str, coords: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, str | None]:
    specialized = ctx.target.emit_dot_operand(name, coords, ctx)

    if specialized is not None:
        return specialized
    return _emit_element(name, coords, ctx), None


def _dot_accumulator_dtype(op: ssa.Operation, ctx: _EmitContext) -> str:
    operand_dtypes = {
        _normalize_dtype(type_.dtype)
        for operand in op.operands[:2]
        if (type_ := ctx.value_types.get(operand)) is not None
    }

    if operand_dtypes & {
        "float8_e4m3fn",
        "float8_e5m2",
        "float16",
        "bfloat16",
    }:
        return "float32"
    return _normalize_dtype(op.results[0].type.dtype or "float32")


def _cast_dot_operand(
    operand: str, value: str, accumulator_dtype: str, ctx: _EmitContext
) -> str:
    type_ = ctx.value_types.get(operand)
    operand_dtype = _normalize_dtype(type_.dtype if type_ is not None else None)
    producer = ctx.operations.get(operand)

    if producer is not None and producer.opcode == "tensor.cast":
        operand_dtype = _normalize_dtype(_resolved_cast_dtype(producer, ctx))

    if operand_dtype == accumulator_dtype:
        return value
    return ctx.target.cast(accumulator_dtype, value)


def _emit_element(name: str, coords: tuple[str, ...], ctx: _EmitContext) -> str:
    if ctx.bindings and name in ctx.bindings and not coords:
        return ctx.bindings[name]

    if name in ctx.memo and coords == _current_coords(_value_axes(name, ctx), ctx):
        return ctx.memo[name]

    if not name.startswith("%"):
        if name not in ctx.tensor_infos:
            return name
        return _load_tensor_at(name, coords, ctx)

    op = ctx.operations.get(name)

    if op is None:
        return _emit_value(name, ctx)

    if op.opcode == "arith.constant":
        return ctx.target.literal(op.attrs.get("value"))

    if op.opcode in {"tensor.zeros", "tensor.empty"}:
        return "0.0"

    if op.opcode == "tensor.full":
        if op.operands:
            return _emit_element(op.operands[0], (), ctx)
        return ctx.target.literal(op.attrs.get("value", 0.0))

    if op.opcode == "tensor.extract":
        base = op.operands[0]
        extract_indices = tuple(
            _emit_index_value(operand, ctx) for operand in op.operands[1:]
        )

        if base in ctx.tensor_infos:
            if op.attrs.get("source"):
                return _load_source_tensor(base, (*extract_indices, *coords), ctx)

            level = int(
                op.results[0].type.attrs.get("dtype_level", _dtype_level(base, ctx))
            )

            return _load_tensor_at(
                base, coords, ctx, level=level, extract_indices=extract_indices
            )
        return _emit_element(base, (*extract_indices, *coords), ctx)

    if op.opcode == "tensor.view":
        if ctx.block_program:
            return ctx.target.render_view(op, ctx)
        return _emit_element(op.operands[0], _view_base_coords(op, coords, ctx), ctx)

    if op.opcode == "linalg.transpose":
        return _emit_element(op.operands[0], tuple(reversed(coords)), ctx)

    if op.opcode == "tensor.cast":
        return _cast_value(op, _emit_element(op.operands[0], coords, ctx), ctx)

    if op.opcode == "index.offset":
        return _emit_offset_element(op, coords, ctx)

    if op.opcode == "mem.data_ptr":
        return ctx.target.tensor_ref(op.operands[0])

    if op.opcode == "mem.load":
        return _emit_pointer_load(op.operands[0], coords, ctx)

    if op.opcode == "select.where":
        result_axes = (
            tuple(str(dim) for dim in op.results[0].type.shape)
            if op.results
            else ctx.output_axes
        )
        args = []

        for operand in op.operands:
            operand_axes = _value_axes(operand, ctx)
            operand_coords = _broadcast_coords(coords, result_axes, operand_axes)
            args.append(_emit_element(operand, operand_coords, ctx))

        args[0] = _materialize_bool_expr(args[0], ctx) or args[0]

        return ctx.target.where(args[0], args[1], args[2])

    if op.opcode.startswith("cmp."):
        return _element_binary(op.opcode[len("cmp.") :], op, coords, ctx)

    if op.opcode.startswith("arith."):
        operator = op.opcode[len("arith.") :]

        if operator in _UNARY:
            return f"({_UNARY[operator]}{_emit_element(op.operands[0], coords, ctx)})"

        if operator in {"maximum", "max"}:
            return ctx.target.call("maximum", _element_args(op, coords, ctx))

        if operator in {"minimum", "min"}:
            return ctx.target.call("minimum", _element_args(op, coords, ctx))

        if operator == "pow":
            return ctx.target.call("pow", _element_args(op, coords, ctx))
        return _element_binary(operator, op, coords, ctx)

    if op.opcode.startswith("math."):
        name = op.opcode[len("math.") :]
        callee = str(op.attrs.get("callee", ""))

        if ctx.target.vector_value_semantics and "libdevice." in callee:
            name = f"libdevice.{name}"
        return ctx.target.call(
            name,
            tuple(
                _emit_element_arg(op, operand, coords, ctx) for operand in op.operands
            ),
        )

    if op.opcode.startswith("reduce."):
        if ctx.block_program or ctx.vector_program:
            return _emit_value(name, ctx)
        return _emit_reduce_element(op, coords, ctx)

    if op.opcode in {"linalg.dot", "linalg.matmul"}:
        return _emit_linalg_dot(op, ctx, coords=coords)

    if op.opcode == "scf.if":
        return _scf_if_element(op, coords, ctx)

    if op.opcode == "scf.for":
        return _emit_value(name, ctx)
    return _emit_value(name, ctx)


def _should_emit_tensor_value_as_element(op: ssa.Operation) -> bool:
    if not op.results or op.results[0].type.kind != "tensor":
        return False
    return op.opcode in {"index.offset", "tensor.extract", "tensor.view"}


def _element_args(
    op: ssa.Operation, coords: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, ...]:
    return tuple(_emit_element_arg(op, operand, coords, ctx) for operand in op.operands)


def _emit_element_arg(
    op: ssa.Operation, operand: str, coords: tuple[str, ...], ctx: _EmitContext
) -> str:
    result_axes = (
        tuple(str(dim) for dim in op.results[0].type.shape)
        if op.results
        else ctx.output_axes
    )
    operand_axes = _value_axes(operand, ctx)

    return _emit_element(
        operand, _broadcast_coords(coords, result_axes, operand_axes), ctx
    )


def _element_binary(
    operator: str, op: ssa.Operation, coords: tuple[str, ...], ctx: _EmitContext
) -> str:
    masks: tuple[str, ...] = ()

    if ctx.target.c_style_syntax and operator in {"mul", "multiply"}:
        result_axes = (
            tuple(str(dim) for dim in op.results[0].type.shape)
            if op.results
            else ctx.output_axes
        )
        values_and_masks = tuple(
            _emit_dot_operand(
                operand,
                _broadcast_coords(coords, result_axes, _value_axes(operand, ctx)),
                ctx,
            )
            for operand in op.operands
        )
        args = tuple(value for value, _ in values_and_masks)
        masks = tuple(mask for _, mask in values_and_masks if mask)
    else:
        args = _element_args(op, coords, ctx)

    args = ctx.target.coerce_binary_args(op, args, ctx)

    if operator == "floordiv":
        return (
            f"(({args[0]}) // ({args[1]}))"
            if not ctx.target.c_style_syntax
            else f"(({args[0]}) / ({args[1]}))"
        )

    symbol = _BINARY[operator]
    result = f"({args[0]} {symbol} {args[1]})"

    if masks:
        result_type = ctx.target.arithmetic_result_type(op, ctx)
        local = _local_symbol(op.results[0].name, ctx)
        ctx.lines.append(
            f"/* guarded core: {ctx.target.type_name(result_type.dtype)} "
            f"{local} = {result}; */"
        )
        mask = " && ".join(f"({item})" for item in masks)

        return f"(({mask}) ? ({result}) : 0.0)"
    return result


def _emit_pointer_load(pointer: str, coords: tuple[str, ...], ctx: _EmitContext) -> str:
    address = _pointer_address(pointer, coords, ctx)

    if address is None:
        if ctx.vector_program:
            raise ValueError(
                "Row-vector reductions require a decomposable pointer address."
            )

        return ctx.target.call("load", (_emit_element(pointer, coords, ctx),))

    base, offset = address

    mask = _mask_for_coords(coords, ctx) if ctx.vector_program else None

    return ctx.target.load(base, offset, mask=mask)


def _pointer_address(
    value: str, coords: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, str] | None:
    op = ctx.operations.get(value)

    if op is None:
        return (value, "0") if value in ctx.tensor_infos else None

    if op.opcode == "mem.data_ptr" and op.operands:
        return op.operands[0], "0"

    if op.opcode == "tensor.view" and op.operands:
        return _pointer_address(op.operands[0], coords, ctx)

    if op.opcode not in {"arith.add", "arith.sub"} or len(op.operands) != 2:
        return None

    lhs, rhs = op.operands
    lhs_type = ctx.value_types.get(lhs)
    pointer_operand = (
        lhs if lhs_type is not None and lhs_type.kind == "pointer" else rhs
    )
    offset_operand = rhs if pointer_operand == lhs else lhs
    address = _pointer_address(pointer_operand, coords, ctx)

    if address is None:
        return None

    base, current_offset = address
    offset = _emit_value(offset_operand, ctx)
    operator = "-" if op.opcode == "arith.sub" and pointer_operand == lhs else "+"

    if _is_zero_expr(current_offset):
        return base, f"-({offset})" if operator == "-" else offset
    return base, f"({current_offset}) {operator} ({offset})"


def _reduction_identity(operator: str, type_: ssa.Type, target: _Target) -> str:
    dtype = _normalize_dtype(type_.dtype or "float32")

    if operator == "sum":
        return target.literal(0.0 if "float" in dtype else 0)

    if dtype == "bool":
        return target.literal(operator == "min")

    if dtype in {"float8_e5m2", "float16", "bfloat16", "float32", "float64"}:
        return target.literal(float("-inf") if operator == "max" else float("inf"))

    limits: Mapping[str, tuple[int | float, int | float]] = {
        "float8_e4m3fn": (-448.0, 448.0),
        "int8": (-128, 127),
        "uint8": (0, 255),
        "int16": (-32768, 32767),
        "uint16": (0, 65535),
        "int32": (-2147483648, 2147483647),
        "uint32": (0, 4294967295),
        "int64": (-9223372036854775808, 9223372036854775807),
        "uint64": (0, 18446744073709551615),
    }

    if dtype not in limits:
        raise ValueError(f"Unsupported reduction identity dtype: {dtype!r}.")

    minimum, maximum = limits[dtype]

    return target.literal(minimum if operator == "max" else maximum)


def _emit_reduce_element(
    op: ssa.Operation,
    coords: tuple[str, ...],
    ctx: _EmitContext,
    *,
    local: str | None = None,
) -> str:
    operator = op.opcode[len("reduce.") :]
    operand = op.operands[0]
    operand_axes = _value_axes(operand, ctx)
    axis_attr = op.attrs.get("axis")

    if axis_attr is None:
        axis = None
        upper = _product(operand_axes)
    else:
        axis = int(axis_attr)

        if axis < 0:
            axis += len(operand_axes)

        upper = (
            operand_axes[axis]
            if 0 <= axis < len(operand_axes)
            else _axis_extent(ctx, axis)
        )

    if local is None:
        local = f"{_local_symbol(op.results[0].name, ctx)}_elem"

    result_type = ssa.Type(
        kind="scalar", dtype=op.results[0].type.dtype if op.results else "float32"
    )
    init = _reduction_identity(operator, result_type, ctx.target)

    if ctx.target.vector_value_semantics and ctx.mask_expr is not None:
        dtype = _normalize_dtype(result_type.dtype or "float32")
        init = ctx.target.vector_splat("(BLOCK,)", init, dtype)

    mutable = _uses_mutable_scalar_slots(ctx.target)

    if mutable:
        ctx.lines.extend(
            _mutable_scalar_decl_lines(ctx.target, result_type, local, init)
        )
        acc_expr = _mutable_scalar_read(ctx.target, local)
    else:
        ctx.lines.append(ctx.target.local_decl(result_type, local, init))
        acc_expr = local

    loop_var = f"{local}_i"
    ctx.lines.append(ctx.target.loop_header(loop_var, "0", upper, "1"))
    body_lines: list[str] = []
    body = ctx.child(
        lines=body_lines,
        memo=dict(ctx.memo),
        local_suffix=_nested_local_suffix(ctx, local),
    )

    if axis is None:
        operand_coords = tuple(
            _axis_offset_expr(operand_axes, dim, loop_var, ctx.target)
            for dim in range(len(operand_axes))
        )
    else:
        operand_coords = coords[:axis] + (loop_var,) + coords[axis:]

    term = _emit_element(operand, operand_coords, body)
    body_lines.append(
        _assign_scalar(
            ctx.target,
            local,
            ctx.target.reduce_update(operator, acc_expr, term),
            mutable=mutable,
        )
    )
    ctx.lines.extend(_indent_lines(body_lines, ctx.target))

    if ctx.target.c_style_syntax:
        ctx.lines.append("}")
    return acc_expr


def _emit_offset_element(
    op: ssa.Operation, coords: tuple[str, ...], ctx: _EmitContext
) -> str:
    operand = op.operands[0]
    dim = int(op.attrs.get("dim", 0) or 0)

    if operand in ctx.tensor_infos:
        return _offset_from_template(
            ctx.tensor_infos.get(operand),
            coords,
            ctx,
            level=_dtype_level(operand, ctx),
            dim=dim,
        )

    producer = ctx.operations.get(operand)

    if producer is not None and producer.opcode == "tensor.extract":
        base = producer.operands[0]
        extract_indices = tuple(
            _emit_index_value(item, ctx) for item in producer.operands[1:]
        )
        level = int(
            producer.results[0].type.attrs.get("dtype_level", _dtype_level(base, ctx))
        )

        return _offset_from_template(
            ctx.tensor_infos.get(base),
            coords,
            ctx,
            level=level,
            dim=dim,
            extract_indices=extract_indices,
        )
    return _emit_value(op.results[0].name, ctx)


def _load_tensor_at(
    name: str,
    coords: tuple[str, ...],
    ctx: _EmitContext,
    *,
    level: int | None = None,
    extract_indices: tuple[str, ...] = (),
) -> str:
    if name.startswith("%") and name in ctx.operations:
        producer = ctx.operations[name]

        if producer.opcode in {"tensor.view", "mem.data_ptr"} and producer.operands:
            name = producer.operands[0]

    info = ctx.tensor_infos.get(name)
    dtype_level = _dtype_level(name, ctx) if level is None else level
    axes = _access_axes(info, ctx, dtype_level, fallback=_value_axes(name, ctx))
    view_index = _linearized_index(coords, axes) if coords else "0"
    source_index = _target_index_expr(
        ctx.target,
        _source_index_for_value(
            info,
            view_index,
            ctx,
            level=dtype_level,
            extract_indices=extract_indices,
            value_coords=coords,
        ),
    )
    source_index = _materialize_index_expr(source_index, ctx)
    base_mask = _load_base_mask(source_index, ctx)

    if ctx.vector_program:
        base_mask = _mask_for_coords(coords, ctx)
    elif extract_indices:
        base_mask = None

    mask = _combined_mask(
        ctx.target,
        base_mask,
        info,
        view_index,
        ctx=ctx,
        level=dtype_level,
        extract_indices=extract_indices,
        value_coords=coords,
    )

    return _masked_load(name, source_index, mask, info, ctx)


def _masked_load(
    name: str,
    source_index: str,
    mask: str | None,
    info: _TensorInfo | None,
    ctx: _EmitContext,
) -> str:
    other = _load_other(info)
    load = ctx.target.load(name, source_index, mask=mask, other=other)

    if ctx.target.c_style_syntax and mask is not None:
        predicate = _materialize_bool_expr(mask, ctx) or mask
        other_value = ctx.target.literal(other)

        if info is not None:
            other_value = ctx.target.cast(info.dtype, other_value)
        return f"(({predicate}) ? ({load}) : ({other_value}))"

    if ctx.target.tir_value_semantics and mask is not None:
        predicate = _materialize_bool_expr(mask, ctx) or mask
        other_value = ctx.target.literal(other)

        if info is not None:
            other_value = ctx.target.cast(info.dtype, other_value)
        return ctx.target.where(predicate, load, other_value)
    return load


def _access_axes(
    info: _TensorInfo | None,
    ctx: _EmitContext,
    level: int,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    template = _access_template(info, level)

    if template is not None:
        shape = tuple(str(dim) for dim in template.get("shape", ()) if str(dim))

        if shape:
            return shape
    return fallback


def _offset_from_template(
    info: _TensorInfo | None,
    coords: tuple[str, ...],
    ctx: _EmitContext,
    *,
    level: int,
    dim: int,
    extract_indices: tuple[str, ...] = (),
) -> str:
    template = _access_template(info, level)

    if template is None:
        axes = _tensor_axes(info, fallback=ctx.output_axes)

        if not axes:
            return "0"
        return _axis_offset_expr(axes, dim, ctx.inner_index_expr, ctx.target)

    offsets = tuple(str(offset) for offset in template.get("offsets", ()))
    source_ndim = len(offsets)

    if dim < 0:
        dim += source_ndim

    if dim < 0 or dim >= source_ndim:
        return "0"

    shape = tuple(str(axis) for axis in template.get("shape", ())) or ctx.output_axes

    if len(coords) == len(shape):
        value_coords = coords
    else:
        value_coords = _offset_value_coords(
            info, shape, coords, level=level, dim=dim, ctx=ctx
        )

    replacements = {"outer_index": ctx.outer_index_expr}
    replacements.update(
        {f"value_{index}": coord for index, coord in enumerate(value_coords)}
    )

    for index, value in enumerate(extract_indices):
        replacements[f"extract_0_{index}"] = value
    return _target_index_expr(ctx.target, _replace_symbols(offsets[dim], replacements))


def _offset_value_coords(
    info: _TensorInfo | None,
    shape: tuple[str, ...],
    coords: tuple[str, ...],
    *,
    level: int,
    dim: int,
    ctx: _EmitContext,
) -> tuple[str, ...]:
    attrs = {} if info is None or info.attrs is None else info.attrs
    levels = tuple(attrs.get("dtype_target_dims", ()))
    target_dims = tuple(levels[level]) if level < len(levels) else ()
    source_ndim = len(info.source_shape) if info is not None else len(target_dims)
    source_dim = dim + source_ndim if dim < 0 else dim

    if target_dims and coords:
        result: list[str] = []
        coord_index = 0

        for target_dim in target_dims:
            if target_dim is not None and int(target_dim) == source_dim:
                result.append(coords[min(coord_index, len(coords) - 1)])
                coord_index += 1
            else:
                result.append("0")

        if len(result) == len(shape):
            return tuple(result)
    return _coords_from_linear(ctx.inner_index_expr, shape, ctx.target)


def _current_coords(axes: tuple[str, ...], ctx: _EmitContext) -> tuple[str, ...]:
    if not axes:
        return ()

    if ctx.block_program:
        return ctx.target.block_coords(axes)

    output_axes = tuple(str(axis) for axis in ctx.output_axes)

    if output_axes:
        output_coords = ctx.coordinate_exprs or tuple(
            _axis_offset_expr(output_axes, dim, ctx.inner_index_expr, ctx.target)
            for dim in range(len(output_axes))
        )
        vector_reduction_axis = (
            output_coords.index(ctx.reduction_lane)
            if ctx.vector_program and ctx.reduction_lane in output_coords
            else None
        )
        coords: tuple[str, ...] | None = None

        if len(axes) == len(output_axes):
            coords = tuple(
                "0" if axis == "1" else output_coords[index]
                for index, axis in enumerate(axes)
            )
        elif vector_reduction_axis is not None and len(axes) == len(output_axes) - 1:
            parallel_dims = tuple(
                index
                for index in range(len(output_axes))
                if index != vector_reduction_axis
            )

            if axes == tuple(output_axes[index] for index in parallel_dims):
                coords = tuple(
                    "0" if axis == "1" else output_coords[parallel_dim]
                    for axis, parallel_dim in zip(axes, parallel_dims)
                )
        elif len(axes) < len(output_axes):
            if _axes_compatible_prefix(axes, output_axes):
                coords = tuple(
                    "0" if axis == "1" else output_coords[index]
                    for index, axis in enumerate(axes)
                )
            else:
                offset = len(output_axes) - len(axes)

                if _axes_compatible_suffix(axes, output_axes):
                    coords = tuple(
                        "0" if axis == "1" else output_coords[index + offset]
                        for index, axis in enumerate(axes)
                    )

        if coords is not None:
            if (
                ctx.reduce_axis is not None
                and ctx.reduce_index is not None
                and 0 <= ctx.reduce_axis < len(coords)
            ):
                coords = tuple(
                    ctx.reduce_index if index == ctx.reduce_axis else coord
                    for index, coord in enumerate(coords)
                )
            return coords

    if len(axes) == 1:
        return (
            ctx.reduce_index
            if ctx.reduce_axis == 0 and ctx.reduce_index
            else _axis_offset_expr(axes, 0, ctx.inner_index_expr, ctx.target),
        )

    coords = [
        _axis_offset_expr(axes, dim, ctx.inner_index_expr, ctx.target)
        for dim in range(len(axes))
    ]

    if (
        ctx.reduce_axis is not None
        and ctx.reduce_index is not None
        and 0 <= ctx.reduce_axis < len(coords)
    ):
        coords[ctx.reduce_axis] = ctx.reduce_index
    return tuple(coords)


def _axes_compatible_prefix(
    axes: tuple[str, ...], output_axes: tuple[str, ...]
) -> bool:
    return len(axes) <= len(output_axes) and all(
        axis == "1" or _same_axis_dim(axis, output_axes[index])
        for index, axis in enumerate(axes)
    )


def _axes_compatible_suffix(
    axes: tuple[str, ...], output_axes: tuple[str, ...]
) -> bool:
    if len(axes) > len(output_axes):
        return False

    offset = len(output_axes) - len(axes)

    return all(
        axis == "1" or _same_axis_dim(axis, output_axes[index + offset])
        for index, axis in enumerate(axes)
    )


def _same_axis_dim(lhs: str, rhs: str) -> bool:
    return lhs == rhs or _axis_dim_key(lhs) == _axis_dim_key(rhs)


def _axis_dim_key(value: str) -> str:
    return re.sub(r"tensor_\d+_size_", "tensor_size_", str(value))


def _dot_operand_coords(
    lhs_axes: tuple[str, ...],
    rhs_axes: tuple[str, ...],
    result_coords: tuple[str, ...],
    loop_var: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(lhs_axes) >= 2 and len(rhs_axes) >= 2:
        row = result_coords[0] if result_coords else "0"
        col = result_coords[1] if len(result_coords) > 1 else "0"

        return (row, loop_var), (loop_var, col)

    if len(lhs_axes) >= 2 and len(rhs_axes) == 1:
        row = result_coords[0] if result_coords else "0"

        return (row, loop_var), (loop_var,)

    if len(lhs_axes) == 1 and len(rhs_axes) >= 2:
        col = result_coords[0] if result_coords else "0"

        return (loop_var,), (loop_var, col)
    return (loop_var,), (loop_var,)


def _broadcast_coords(
    result_coords: tuple[str, ...],
    result_axes: tuple[str, ...],
    operand_axes: tuple[str, ...],
) -> tuple[str, ...]:
    if not operand_axes:
        return ()

    offset = len(result_axes) - len(operand_axes)
    coords: list[str] = []

    for index, axis in enumerate(operand_axes):
        result_index = index + offset

        if axis == "1" or result_index < 0 or result_index >= len(result_coords):
            coords.append("0")
        else:
            coords.append(result_coords[result_index])
    return tuple(coords)


def _view_base_coords(
    op: ssa.Operation, coords: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, ...]:
    subscript = str(op.attrs.get("subscript", ""))

    if "None" not in subscript:
        return coords

    parts = [part.strip() for part in subscript.strip("()").split(",") if part.strip()]
    base_coords: list[str] = []
    coord_index = 0

    for part in parts:
        if part == "None":
            coord_index += 1
            continue

        if coord_index < len(coords):
            base_coords.append(coords[coord_index])

        coord_index += 1

    if not base_coords and coords:
        base_coords.append(coords[0])
    return tuple(base_coords)


def _emit_reduce(local: str, op: ssa.Operation, ctx: _EmitContext) -> str:
    operator = op.opcode[len("reduce.") :]
    axis = op.attrs.get("axis")
    axis = None if axis is None else int(axis)
    lower = "0"
    step = "1"
    loop_var = f"{local}_i"
    operand_axes = _value_axes(op.operands[0], ctx) if op.operands else ctx.output_axes
    normalized_axis = None

    if ctx.target.vector_value_semantics and (
        (ctx.vector_program and op.results[0].name in ctx.scheduled_reductions)
        or ctx.block_program
    ):
        operand = _emit_value(op.operands[0], ctx)
        expr = ctx.target.vector_reduce(operator, operand, 0)
        ctx.lines.append(ctx.target.local_decl(op.results[0].type, local, expr))

        return local

    if axis is None:
        upper = _product(operand_axes)
    else:
        normalized_axis = axis if axis >= 0 else axis + len(operand_axes)
        upper = str(
            op.attrs.get("extent")
            or (
                operand_axes[normalized_axis]
                if 0 <= normalized_axis < len(operand_axes)
                else _axis_extent(ctx, normalized_axis)
            )
        )

    result_type = op.results[0].type
    init = _reduction_identity(operator, result_type, ctx.target)

    if ctx.target.vector_value_semantics and axis is not None:
        dtype = _normalize_dtype(result_type.dtype or "float32")
        init = ctx.target.vector_splat("(BLOCK,)", init, dtype)

    if _uses_mutable_scalar_slots(ctx.target):
        ctx.lines.extend(
            _mutable_scalar_decl_lines(ctx.target, result_type, local, init)
        )
        acc_expr = _mutable_scalar_read(ctx.target, local)
    else:
        ctx.lines.append(ctx.target.local_decl(result_type, local, init))
        acc_expr = local

    ctx.lines.append(ctx.target.loop_header(loop_var, lower, upper, step))
    inner_lines: list[str] = []
    inner = ctx.child(
        lines=inner_lines,
        memo={},
        reduce_axis=0 if normalized_axis is None else normalized_axis,
        reduce_index=loop_var,
        reduce_flattened=axis is None,
        mask_expr=ctx.mask_expr if axis is not None else None,
        indent=ctx.indent + _indent_unit(ctx.target),
        local_suffix=_nested_local_suffix(ctx, local),
    )
    term = _emit_value(op.operands[0], inner)
    update = ctx.target.reduce_update(operator, acc_expr, term)
    inner_lines.append(
        _assign_scalar(
            ctx.target, local, update, mutable=_uses_mutable_scalar_slots(ctx.target)
        )
    )
    ctx.lines.extend(_indent_lines(inner_lines, ctx.target))

    if ctx.target.c_style_syntax:
        ctx.lines.append("}")
    return acc_expr


def _emit_scf_for(local: str, op: ssa.Operation, ctx: _EmitContext) -> str | None:
    if ctx.native_block_program:
        fused = ctx.target.emit_reduction_loop(local, op, ctx)

        if fused is not None:
            return fused

    lower = _emit_loop_bound(op.operands[0], ctx)
    upper = _emit_loop_bound(op.operands[1], ctx)
    step = _emit_loop_bound(op.operands[2], ctx)
    iter_attrs = tuple(op.attrs.get("iter_args", ()))
    result_names = tuple(result.name for result in op.results)
    loop_locals: dict[str, str] = {}
    result_locals: dict[str, str] = {}

    for result, attr, value in zip(result_names, iter_attrs, op.results):
        initial_name = str(attr["initial"])
        init = _emit_value(initial_name, ctx)

        if (
            ctx.target.vector_value_semantics
            and ctx.mask_expr is not None
            and ctx.target.needs_block_init(initial_name, value, ctx)
        ):
            dtype = _loop_initializer_dtype(initial_name, value, ctx)
            init = ctx.target.vector_splat("(BLOCK,)", init, dtype)
        elif (
            ctx.target.vector_value_semantics
            and ctx.block_program
            and ctx.target.needs_block_init(initial_name, value, ctx)
        ):
            dtype = _loop_initializer_dtype(initial_name, value, ctx)
            shape = ctx.target.block_shape(tuple(str(dim) for dim in value.type.shape))
            init = ctx.target.vector_splat(shape, init, dtype)

        result_local = _local_symbol(result, ctx)
        result_locals[result] = result_local

        if _uses_mutable_scalar_slots(ctx.target):
            ctx.lines.extend(
                _mutable_scalar_decl_lines(ctx.target, value.type, result_local, init)
            )
            result_expr = _mutable_scalar_read(ctx.target, result_local)
        else:
            ctx.lines.append(ctx.target.local_decl(value.type, result_local, init))
            result_expr = result_local

        ctx.memo[result] = result_expr
        loop_locals[str(attr["block_arg"])] = result_expr
        loop_locals[result] = result_expr

    loop_var = f"{local}_i"
    induction = str(op.attrs.get("induction", "%iv"))
    loop_bindings = dict(ctx.bindings or {})
    loop_bindings[induction] = loop_var
    loop_bindings.update(loop_locals)
    ctx.lines.append(ctx.target.loop_header(loop_var, lower, upper, step))
    body_lines: list[str] = []
    body = ctx.child(
        lines=body_lines,
        memo=dict(ctx.memo),
        bindings=loop_bindings,
        local_suffix=_nested_local_suffix(ctx, local),
    )
    region = op.regions[0]
    yields: tuple[str, ...] = ()

    for inner_op in region.operations:
        if inner_op.opcode == "scf.yield":
            yields = inner_op.operands
            continue

        if _is_top_level_effect(inner_op):
            _emit_operation(inner_op, body)

    next_locals: list[tuple[str, str]] = []

    for result, yielded, value in zip(result_names, yields, op.results):
        next_local = _local_symbol(f"%next_{result[1:]}", body)
        body_lines.append(
            body.target.local_decl(value.type, next_local, _emit_value(yielded, body))
        )
        next_locals.append((result, next_local))

    for result, next_local in next_locals:
        body_lines.append(
            _assign_scalar(
                ctx.target,
                result_locals.get(result, _local_symbol(result, ctx)),
                next_local,
                mutable=_uses_mutable_scalar_slots(ctx.target),
            )
        )

    ctx.lines.extend(_indent_lines(body_lines, ctx.target))

    if ctx.target.c_style_syntax:
        ctx.lines.append("}")
    return ctx.memo.get(result_names[0]) if result_names else None


def _loop_initializer_dtype(
    initial_name: str, value: ssa.Value, ctx: _EmitContext
) -> str:
    producer = ctx.operations.get(initial_name)

    if producer is not None and producer.opcode in {
        "tensor.zeros",
        "tensor.empty",
        "tensor.full",
    }:
        dtype_ref = producer.attrs.get("dtype_ref")

        if isinstance(dtype_ref, str):
            info = ctx.tensor_infos.get(dtype_ref)

            if info is not None and info.ndim > 0:
                return f"{dtype_ref}.dtype"
            return value.type.dtype or "float32"

        dtype = producer.attrs.get("dtype")

        if isinstance(dtype, str) and dtype:
            return dtype
    return value.type.dtype or "float32"


def _emit_loop_bound(name: str, ctx: _EmitContext) -> str:
    op = ctx.operations.get(name)

    if op is not None and op.opcode == "arith.constant":
        return ctx.target.literal(op.attrs.get("value"))
    return _emit_value(name, ctx)


def _emit_scf_if_statement(op: ssa.Operation, ctx: _EmitContext) -> None:
    condition = _emit_value(op.operands[0], ctx)
    ctx.lines.append(_if_header(condition, ctx.target))
    then_lines: list[str] = []
    then_ctx = ctx.child(
        lines=then_lines,
        memo=dict(ctx.memo),
        local_suffix=_nested_local_suffix(ctx, "then"),
    )

    if op.regions:
        for inner_op in op.regions[0].operations:
            if _is_top_level_effect(inner_op):
                _emit_operation(inner_op, then_ctx)

    ctx.lines.extend(
        _indent_lines(then_lines or _empty_block_lines(ctx.target), ctx.target)
    )

    if len(op.regions) > 1:
        ctx.lines.append("} else {" if ctx.target.c_style_syntax else "else:")
        else_lines: list[str] = []
        else_ctx = ctx.child(
            lines=else_lines,
            memo=dict(ctx.memo),
            local_suffix=_nested_local_suffix(ctx, "else"),
        )

        for inner_op in op.regions[1].operations:
            if _is_top_level_effect(inner_op):
                _emit_operation(inner_op, else_ctx)

        ctx.lines.extend(
            _indent_lines(else_lines or _empty_block_lines(ctx.target), ctx.target)
        )

    if ctx.target.c_style_syntax:
        ctx.lines.append("}")


def _emit_scf_if_results(op: ssa.Operation, ctx: _EmitContext) -> None:
    result_locals: dict[str, str] = {}

    for result in op.results:
        local = _local_symbol(result.name, ctx)
        result_locals[result.name] = local
        init = _zero_value(result.type, ctx.target)

        if _uses_mutable_scalar_slots(ctx.target):
            ctx.lines.extend(
                _mutable_scalar_decl_lines(ctx.target, result.type, local, init)
            )
            ctx.memo[result.name] = _mutable_scalar_read(ctx.target, local)
        else:
            ctx.lines.append(ctx.target.local_decl(result.type, local, init))
            ctx.memo[result.name] = local

    condition = _emit_value(op.operands[0], ctx)
    ctx.lines.append(_if_header(condition, ctx.target))

    for region_index, region in enumerate(op.regions[:2]):
        if region_index == 1:
            ctx.lines.append("} else {" if ctx.target.c_style_syntax else "else:")

        lines: list[str] = []
        child = ctx.child(
            lines=lines,
            memo=dict(ctx.memo),
            local_suffix=_nested_local_suffix(ctx, f"if_{region.name}"),
        )
        yields: tuple[str, ...] = ()

        for inner_op in region.operations:
            if inner_op.opcode == "scf.yield":
                yields = inner_op.operands
                continue

            if _is_top_level_effect(inner_op):
                _emit_operation(inner_op, child)

        for result, yielded in zip(op.results, yields):
            lines.append(
                _assign_scalar(
                    ctx.target,
                    result_locals.get(result.name, _local_symbol(result.name, ctx)),
                    _emit_value(yielded, child),
                    mutable=_uses_mutable_scalar_slots(ctx.target),
                )
            )

        ctx.lines.extend(
            _indent_lines(lines or _empty_block_lines(ctx.target), ctx.target)
        )

    if ctx.target.c_style_syntax:
        ctx.lines.append("}")


def _scf_if_expr(op: ssa.Operation, ctx: _EmitContext) -> str:
    condition = _emit_value(op.operands[0], ctx)

    if len(op.regions) == 1:
        then_value = _region_yield_expr(op.regions[0], ctx)

        return ctx.target.where(condition, then_value, "0.0")

    then_region, else_region = op.regions[:2]
    then_value = _region_yield_expr(then_region, ctx)
    else_value = _region_yield_expr(else_region, ctx)

    return ctx.target.where(condition, then_value, else_value)


def _scf_if_element(
    op: ssa.Operation, coords: tuple[str, ...], ctx: _EmitContext
) -> str:
    if not ctx.target.vector_value_semantics:
        return _scf_if_element_control_flow(op, coords, ctx)

    condition = _emit_value(op.operands[0], ctx)

    if len(op.regions) == 1:
        then_value = _region_yield_element(op.regions[0], coords, ctx)

        return ctx.target.where(condition, then_value, "0.0")

    then_region, else_region = op.regions[:2]
    then_value = _region_yield_element(then_region, coords, ctx)
    else_value = _region_yield_element(else_region, coords, ctx)

    return ctx.target.where(condition, then_value, else_value)


def _scf_if_element_control_flow(
    op: ssa.Operation, coords: tuple[str, ...], ctx: _EmitContext
) -> str:
    if not op.results:
        _emit_scf_if_statement(op, ctx)

        return "0.0"

    result = op.results[0]
    local = f"{_local_symbol(result.name, ctx)}_if_{len(ctx.lines)}"
    init = _zero_value(result.type, ctx.target)
    mutable = _uses_mutable_scalar_slots(ctx.target)

    if mutable:
        ctx.lines.extend(
            _mutable_scalar_decl_lines(ctx.target, result.type, local, init)
        )
        result_expr = _mutable_scalar_read(ctx.target, local)
    else:
        ctx.lines.append(ctx.target.local_decl(result.type, local, init))
        result_expr = local

    condition = _emit_value(op.operands[0], ctx)
    ctx.lines.append(_if_header(condition, ctx.target))

    for region_index, region in enumerate(op.regions[:2]):
        if region_index == 1:
            ctx.lines.append("} else {" if ctx.target.c_style_syntax else "else:")

        lines: list[str] = []
        child = ctx.child(
            lines=lines,
            memo=dict(ctx.memo),
            local_suffix=_nested_local_suffix(ctx, f"if_{region.name}"),
        )
        yielded = None

        for inner_op in region.operations:
            if inner_op.opcode == "scf.yield":
                yielded = inner_op.operands[0] if inner_op.operands else None
                continue

            if _is_top_level_effect(inner_op):
                _emit_operation(inner_op, child)

        if yielded is not None:
            value = (
                _emit_element(yielded, coords, child)
                if yielded.startswith("%")
                else _emit_value(yielded, child)
            )
            lines.append(_assign_scalar(ctx.target, local, value, mutable=mutable))

        ctx.lines.extend(
            _indent_lines(lines or _empty_block_lines(ctx.target), ctx.target)
        )

    if len(op.regions) == 1:
        ctx.lines.append("} else {" if ctx.target.c_style_syntax else "else:")
        ctx.lines.extend(_indent_lines(_empty_block_lines(ctx.target), ctx.target))

    if ctx.target.c_style_syntax:
        ctx.lines.append("}")
    return result_expr


def _region_yield_expr(region: ssa.Block, ctx: _EmitContext) -> str:
    lines: list[str] = []
    child = ctx.child(
        lines=lines,
        memo=dict(ctx.memo),
        local_suffix=_nested_local_suffix(ctx, f"expr_{region.name}"),
    )
    yielded = None

    for op in region.operations:
        if op.opcode == "scf.yield":
            yielded = op.operands[0] if op.operands else None
            continue

        if _is_top_level_effect(op):
            _emit_operation(op, child)

    if yielded is None:
        ctx.lines.extend(lines)

        return "0.0"

    value = _emit_value(yielded, child)
    ctx.lines.extend(lines)

    return value


def _region_yield_element(
    region: ssa.Block, coords: tuple[str, ...], ctx: _EmitContext
) -> str:
    lines: list[str] = []
    child = ctx.child(
        lines=lines,
        memo=dict(ctx.memo),
        local_suffix=_nested_local_suffix(ctx, f"expr_{region.name}"),
    )
    yielded = None

    for op in region.operations:
        if op.opcode == "scf.yield":
            yielded = op.operands[0] if op.operands else None
            continue

        if _is_top_level_effect(op):
            _emit_operation(op, child)

    if yielded is None:
        ctx.lines.extend(lines)

        return "0.0"

    value = (
        _emit_element(yielded, coords, child)
        if yielded.startswith("%")
        else _emit_value(yielded, child)
    )
    ctx.lines.extend(lines)

    return value


def _tensor_value(name: str, ctx: _EmitContext) -> str:
    info = ctx.tensor_infos.get(name, _TensorInfo(name=name))

    if info.ndim == 0:
        if _is_bool_scalar_value(name, ctx) and ctx.target.tir_value_semantics:
            return f"({name} != 0)"
        return name

    if ctx.reduce_axis == 1:
        red = ctx.reduce_index or "col"
        axes = _value_axes(name, ctx)

        if len(axes) >= 2:
            row = ctx.row_expr or ctx.index_expr

            return _load_tensor(name, f"({row}) * ({axes[1]}) + ({red})", ctx)

        if name != ctx.output:
            return _load_tensor(name, red, ctx.child(mask_expr=None))
        return _load_tensor(name, ctx.row_expr or ctx.index_expr, ctx)

    if ctx.reduce_axis == 0:
        return _load_tensor(name, ctx.reduce_index or "i", ctx)
    return _load_tensor(name, _default_tensor_index(name, ctx), ctx)


def _load_tensor(name: str, view_index: str, ctx: _EmitContext) -> str:
    info = ctx.tensor_infos.get(name)
    source_index = _target_index_expr(
        ctx.target,
        _source_index_for_value(info, view_index, ctx, level=_dtype_level(name, ctx)),
    )
    source_index = _materialize_index_expr(source_index, ctx)
    mask = _combined_mask(
        ctx.target,
        _load_base_mask(source_index, ctx),
        info,
        view_index,
        ctx=ctx,
    )

    return _masked_load(name, source_index, mask, info, ctx)


def _load_base_mask(source_index: str, ctx: _EmitContext) -> str | None:
    if not ctx.target.vector_value_semantics:
        return None

    mask = ctx.mask_expr

    if mask is None or not _index_expr_is_vector(source_index, ctx):
        return None
    return mask


def _index_expr_is_vector(source_index: str, ctx: _EmitContext) -> bool:
    symbols = _expression_symbols(source_index)

    return bool(
        ctx.target.index_name in symbols
        or "offsets" in symbols
        or "tl.arange(" in source_index
    )


def _load_source_tensor(name: str, indices: tuple[str, ...], ctx: _EmitContext) -> str:
    info = ctx.tensor_infos.get(name)
    source_index = _target_index_expr(ctx.target, _source_linear_index(info, indices))
    base_mask = (
        _load_base_mask(source_index, ctx)
        if ctx.target.vector_value_semantics
        else ctx.mask_expr
    )
    mask = _source_bounds_mask(info, indices, base_mask=base_mask)

    return _masked_load(name, source_index, mask, info, ctx)


def _source_linear_index(info: _TensorInfo | None, indices: tuple[str, ...]) -> str:
    strides = _source_strides(info)
    terms = []

    for dim, index in enumerate(indices):
        stride = strides[dim] if dim < len(strides) else "1"
        terms.append(
            f"({index})" if _is_one_expr(stride) else f"({index}) * ({stride})"
        )
    return " + ".join(terms) if terms else "0"


def _source_bounds_mask(
    info: _TensorInfo | None,
    indices: tuple[str, ...],
    *,
    base_mask: str | None,
) -> str | None:
    checks = []

    if base_mask:
        checks.append(base_mask)

    shape = () if info is None else info.source_shape

    for dim, index in enumerate(indices):
        if dim < len(shape):
            checks.extend((f"({index}) >= 0", f"({index}) < ({shape[dim]})"))

    if not checks:
        return None
    return " & ".join(f"({check})" for check in checks)


def _load_other(info: _TensorInfo | None):
    if info is None or info.attrs is None:
        return 0.0

    value = info.attrs.get("other")

    return 0.0 if value is None else value


def _source_index(info: _TensorInfo | None, view_index: str) -> str:
    if info is None or not info.view_linear_offset:
        return view_index

    expr = info.view_linear_offset

    if expr == "index":
        return view_index
    return _replace_index_symbol(expr, view_index)


def _source_index_for_value(
    info: _TensorInfo | None,
    view_index: str,
    ctx: _EmitContext,
    *,
    level: int,
    extract_indices: tuple[str, ...] = (),
    value_coords: tuple[str, ...] = (),
) -> str:
    template = _access_template(info, level)

    if template is None:
        return _source_index(info, view_index)

    shape = tuple(str(dim) for dim in template.get("shape", ())) or _tensor_axes(
        info, fallback=ctx.output_axes
    )
    coords = (
        value_coords
        if len(value_coords) == len(shape)
        else _coords_from_linear(view_index, shape, ctx.target)
    )
    replacements = {"outer_index": ctx.outer_index_expr}
    replacements.update({f"value_{index}": coord for index, coord in enumerate(coords)})
    replacements.update(_jagged_extent_replacements(ctx))

    for index, value in enumerate(extract_indices):
        replacements[f"extract_0_{index}"] = value

    replacements.update(_jagged_runtime_replacements(template, replacements, ctx))

    split_index = _source_index_from_offsets(info, template, replacements, ctx)

    if split_index is not None:
        return _add_jagged_base_offset(split_index, template, replacements)
    return _add_jagged_base_offset(
        _replace_symbols(str(template.get("linear_offset", view_index)), replacements),
        template,
        replacements,
    )


def _jagged_runtime_replacements(
    template: Mapping[str, Any],
    replacements: Mapping[str, str],
    ctx: _EmitContext,
) -> dict[str, str]:
    jagged = template.get("jagged")

    if not isinstance(jagged, Mapping):
        return {}

    offsets_param = str(jagged["offsets_param"])
    batch_offset = _target_index_expr(
        ctx.target,
        _replace_symbols(str(jagged["batch_offset"]), replacements),
    )
    seq_start = ctx.target.load(
        offsets_param,
        batch_offset,
        mask=ctx.mask_expr if ctx.target.vector_value_semantics else None,
        other=0,
    )
    seq_end = ctx.target.load(
        offsets_param,
        f"({batch_offset}) + 1",
        mask=ctx.mask_expr if ctx.target.vector_value_semantics else None,
        other=0,
    )

    return {
        str(jagged["seq_start"]): seq_start,
        str(jagged["seq_len"]): f"(({seq_end}) - ({seq_start}))",
    }


def _jagged_extent_replacements(ctx: _EmitContext) -> dict[str, str]:
    replacements = {}

    for info in ctx.tensor_infos.values():
        attrs = info.attrs or {}
        seq_len = attrs.get("jagged_seq_len_param")
        max_seq_len = attrs.get("jagged_max_seq_len_param")

        if seq_len and max_seq_len:
            replacements[str(seq_len)] = str(max_seq_len)
    return replacements


def _add_jagged_base_offset(
    index: str,
    template: Mapping[str, Any],
    replacements: Mapping[str, str],
) -> str:
    jagged = template.get("jagged")

    if not isinstance(jagged, Mapping):
        return index

    seq_start = replacements.get(str(jagged["seq_start"]))

    if seq_start is None:
        return index

    stride = _replace_symbols(str(jagged["stride"]), replacements)
    base = seq_start if _is_one_expr(stride) else f"({seq_start}) * ({stride})"

    return f"({index}) + ({base})"


def _source_index_from_offsets(
    info: _TensorInfo | None,
    template: Mapping[str, Any],
    replacements: Mapping[str, str],
    ctx: _EmitContext,
) -> str | None:
    if ctx.target.vector_value_semantics:
        return None

    offsets = tuple(str(offset) for offset in template.get("offsets", ()))

    if not offsets:
        return None

    strides = (
        ("1",) * len(offsets)
        if ctx.layout_contiguous
        else _source_strides(info, prefer_default=False)
    )

    if len(strides) < len(offsets):
        return None

    terms: list[str] = []

    for dim, (offset, stride) in enumerate(zip(offsets, strides)):
        offset_expr = _target_index_expr(
            ctx.target, _replace_symbols(offset, replacements)
        )
        offset_expr = _materialize_index_expr(offset_expr, ctx, threshold=48)
        stride_expr = _target_index_expr(ctx.target, stride)

        if _is_zero_expr(offset_expr):
            continue

        if _is_one_expr(stride_expr):
            terms.append(offset_expr)
        else:
            terms.append(f"({offset_expr}) * ({stride_expr})")

    if not terms:
        return "0"
    return " + ".join(terms)


def _store_mask(
    target: _Target,
    base: str | None,
    info: _TensorInfo | None,
    view_index: str,
    *,
    ctx: _EmitContext,
    level: int | None = None,
    extract_indices: tuple[str, ...] = (),
) -> str | None:
    if target.tir_value_semantics:
        template_mask = _mask_from_template_offsets(info, view_index, ctx)
        masks = []

        if base:
            masks.append(_target_index_expr(target, base))

        if template_mask:
            masks.append(template_mask)
        elif info is not None and info.view_mask and info.view_mask != "True":
            masks.append(
                _target_index_expr(
                    target, _replace_index_symbol(info.view_mask, view_index)
                )
            )

        if masks:
            return " & ".join(f"({mask})" for mask in masks)
    return _combined_mask(
        target,
        base,
        info,
        view_index,
        ctx=ctx,
        level=level,
        extract_indices=extract_indices,
    )


def _mask_from_template_offsets(
    info: _TensorInfo | None,
    view_index: str,
    ctx: _EmitContext,
) -> str | None:
    template = _access_template(
        info, _dtype_level(info.name, ctx) if info is not None else 0
    )

    if template is None or info is None:
        return None

    offsets = tuple(str(offset) for offset in template.get("offsets", ()))
    source_shape = tuple(str(axis) for axis in info.source_shape if str(axis))

    if not offsets or len(source_shape) < len(offsets):
        return None

    shape = tuple(str(dim) for dim in template.get("shape", ())) or ctx.output_axes
    coords = _coords_from_linear(view_index, shape, ctx.target)
    replacements = {"outer_index": ctx.outer_index_expr}
    replacements.update({f"value_{index}": coord for index, coord in enumerate(coords)})
    replacements.update(_jagged_extent_replacements(ctx))
    replacements.update(_jagged_runtime_replacements(template, replacements, ctx))
    checks: list[str] = []

    for offset, dim in zip(offsets, source_shape):
        offset_expr = _target_index_expr(
            ctx.target, _replace_symbols(offset, replacements)
        )
        offset_expr = _materialize_index_expr(offset_expr, ctx, threshold=48)
        dim_expr = _replace_symbols(dim, replacements)
        checks.append(f"(({offset_expr}) >= 0)")
        checks.append(f"(({dim_expr}) > ({offset_expr}))")

    if not checks:
        return None
    return " & ".join(checks)


def _source_strides(
    info: _TensorInfo | None, *, prefer_default: bool = False
) -> tuple[str, ...]:
    if info is None:
        return ()

    source_shape = tuple(str(axis) for axis in info.source_shape if str(axis))

    if prefer_default and source_shape:
        return _default_strides(source_shape)

    strides = tuple(str(stride) for stride in info.source_strides if str(stride))

    if strides:
        return strides

    if source_shape:
        return _default_strides(source_shape)

    axes = tuple(str(axis) for axis in info.shape if str(axis))

    return _default_strides(axes)


def _buffer_storage_extent(info: _TensorInfo, *, fallback: str) -> str:
    values_numel = (info.attrs or {}).get("jagged_values_numel_param")

    if values_numel:
        return str(values_numel)

    shape = _source_axes(info, fallback=())
    strides = _source_strides(info)

    if not shape or len(shape) != len(strides):
        return fallback

    terms = [f"(({dim}) - 1) * ({stride})" for dim, stride in zip(shape, strides)]

    return "1 + " + " + ".join(terms)


def _combined_mask(
    target: _Target,
    base: str | None,
    info: _TensorInfo | None,
    view_index: str,
    *,
    ctx: _EmitContext | None = None,
    level: int | None = None,
    extract_indices: tuple[str, ...] = (),
    value_coords: tuple[str, ...] = (),
) -> str | None:
    masks = []

    if base:
        masks.append(_target_index_expr(target, base))

    template = None

    if ctx is not None:
        dtype_level = (
            _dtype_level(info.name, ctx)
            if level is None and info is not None
            else int(level or 0)
        )
        template = _access_template(info, dtype_level)

        if template is not None:
            shape = (
                tuple(str(dim) for dim in template.get("shape", ())) or ctx.output_axes
            )
            coords = (
                value_coords
                if len(value_coords) == len(shape)
                else _coords_from_linear(view_index, shape, target)
            )
            replacements = {"outer_index": ctx.outer_index_expr}
            replacements.update(
                {f"value_{index}": coord for index, coord in enumerate(coords)}
            )
            replacements.update(
                {
                    f"extract_0_{index}": value
                    for index, value in enumerate(extract_indices)
                }
            )
            replacements.update(_jagged_extent_replacements(ctx))
            replacements.update(
                _jagged_runtime_replacements(template, replacements, ctx)
            )
            template_mask = _replace_symbols(
                str(template.get("mask", "True")), replacements
            )

            if template_mask and template_mask != "True":
                masks.append(_target_index_expr(target, template_mask))

    if (
        template is None
        and info is not None
        and info.view_mask
        and info.view_mask != "True"
    ):
        masks.append(
            _target_index_expr(
                target, _replace_index_symbol(info.view_mask, view_index)
            )
        )

    if not masks:
        return None

    if len(masks) == 1:
        return masks[0]
    return " & ".join(f"({mask})" for mask in masks)


def _access_template(info: _TensorInfo | None, level: int) -> Mapping[str, Any] | None:
    if info is None or info.attrs is None:
        return None

    for template in info.attrs.get("access_templates", ()):
        if int(template.get("level", -1)) == level:
            return template
    return None


def _dtype_level(name: str, ctx: _EmitContext) -> int:
    type_ = ctx.value_types.get(name)

    if type_ is None:
        return 0
    return int(type_.attrs.get("dtype_level", 0))


def _coords_from_linear(
    index: str, axes: tuple[str, ...], target: _Target
) -> tuple[str, ...]:
    if not axes:
        return ()
    return tuple(
        _axis_offset_expr(axes, dim, index, target) for dim in range(len(axes))
    )


def _target_index_expr(target: _Target, expr: str) -> str:
    rewritten = _rewrite_index_math(expr, c_style=target.c_style_syntax)

    return rewritten


def _materialize_index_expr(
    expr: str, ctx: _EmitContext, *, threshold: int = 96
) -> str:
    if ctx.target.vector_value_semantics:
        return expr

    if len(expr) < threshold or _valid_symbol(expr):
        return expr

    cache_key = ("index", expr)

    if ctx.materialized is not None and cache_key in ctx.materialized:
        return ctx.materialized[cache_key]

    local = _fresh_temp(ctx, "nt_idx")
    ctx.lines.append(
        ctx.target.local_decl(ssa.Type(kind="index", dtype="index"), local, expr)
    )

    if ctx.materialized is not None:
        ctx.materialized[cache_key] = local
    return local


def _materialize_bool_expr(
    expr: str | None, ctx: _EmitContext, *, threshold: int = 96
) -> str | None:
    if expr is None:
        return None

    if ctx.target.vector_value_semantics:
        return expr

    if len(expr) < threshold or _valid_symbol(expr):
        return expr

    local = _fresh_temp(ctx, "nt_pred")
    ctx.lines.append(
        ctx.target.local_decl(ssa.Type(kind="scalar", dtype="bool"), local, expr)
    )

    return local


def _fresh_temp(ctx: _EmitContext, prefix: str) -> str:
    if ctx.temp_counter is None:
        ctx.temp_counter = [0]

    value = ctx.temp_counter[0]
    ctx.temp_counter[0] += 1

    return f"{prefix}_{value}"


def _default_tensor_index(name: str, ctx: _EmitContext) -> str:
    info = ctx.tensor_infos.get(name, _TensorInfo(name=name))
    axes = _value_axes(name, ctx)

    if info.ndim <= 1:
        if (
            len(ctx.output_axes) >= 2
            and name != ctx.output
            and ctx.col_expr is not None
        ):
            return ctx.col_expr
        return ctx.index_expr

    coords = _current_coords(axes, ctx)

    return _linearized_index(coords, axes) if coords else ctx.index_expr


def _store_index(op: ssa.Operation, ctx: _EmitContext) -> str:
    indices = op.attrs.get("indices", ())

    if isinstance(indices, str):
        indices = (indices,)

    if indices:
        rendered = tuple(_emit_index_value(str(index), ctx) for index in indices)

        return _linearized_index(rendered, _value_axes(op.operands[1], ctx))

    if ctx.vector_program:
        axes = _value_axes(op.operands[1], ctx)
        coords = _current_coords(axes, ctx)

        return _linearized_index(coords, axes) if coords else "0"

    if ctx.block_program and ctx.coordinate_exprs:
        return _linearized_index(ctx.coordinate_exprs, ctx.output_axes)
    return ctx.index_expr


def _emit_store_index_value(
    name: str, result_axes: tuple[str, ...], ctx: _EmitContext
) -> str:
    type_ = ctx.value_types.get(name)

    if type_ is None or type_.kind != "tensor":
        return _emit_index_value(name, ctx)

    coords = _current_coords(result_axes, ctx)
    operand_axes = _value_axes(name, ctx)
    value = _emit_element(
        name, _broadcast_coords(coords, result_axes, operand_axes), ctx
    )

    if ctx.target.c_style_syntax and not _integer_expr(value):
        return ctx.target.index_cast(value)
    return value


def _emit_index_value(name: str, ctx: _EmitContext) -> str:
    value = _emit_value(name, ctx)

    if ctx.target.c_style_syntax and not _integer_expr(value):
        return ctx.target.index_cast(value)
    return value


def _logical_ssa_audit(kernel: Kernel, target: _Target) -> str:
    if kernel.ssa is None:
        return ""

    operations = tuple(
        operation
        for block in kernel.ssa.blocks
        for operation in _walk_ops(block.operations)
    )
    by_result = {
        result.name: operation
        for operation in operations
        for result in operation.results
    }
    tensor_names = {tensor.name for tensor in kernel.tensors if tensor.ndim > 0}

    def operand(name: str, index: str) -> str:
        return (
            f"{target.tensor_ref(name)}[{index}]"
            if name in tensor_names
            else target.symbol(name)
        )

    lines = []

    for operation in operations:
        if operation.opcode.startswith("arith.") and operation.results:
            operator = operation.opcode[len("arith.") :]

            if operator not in _BINARY or len(operation.operands) != 2:
                continue

            operands = tuple(operand(name, "index") for name in operation.operands)
            lines.append(
                f"# {target.symbol(operation.results[0].name)} = "
                f"({operands[0]} {_BINARY[operator]} {operands[1]})"
            )
            continue

        if not operation.opcode.startswith("reduce.") or not operation.results:
            continue

        producer = by_result.get(operation.operands[0])

        if producer is None or not producer.opcode.startswith("arith."):
            continue

        operator = producer.opcode[len("arith.") :]

        if operator not in _BINARY or len(producer.operands) != 2:
            continue

        loop_index = f"{target.symbol(operation.results[0].name)}_i"
        operands = tuple(operand(name, loop_index) for name in producer.operands)
        lines.append(f"# {operands[0]} {_BINARY[operator]} {operands[1]}")
    return "\n".join(lines)


def _cooperative_dot_plan(
    operations: Mapping[str, ssa.Operation],
    stores: tuple[ssa.Operation, ...],
    value_types: Mapping[str, ssa.Type],
) -> _CooperativeDotPlan | None:
    unique_operations = {id(op): op for op in operations.values()}.values()

    for loop in unique_operations:
        if loop.opcode != "scf.for" or len(loop.results) != 1 or len(loop.regions) != 1:
            continue

        iter_args = tuple(loop.attrs.get("iter_args", ()))

        if len(iter_args) != 1 or len(loop.operands) < 4:
            continue

        initial = operations.get(str(iter_args[0].get("initial", "")))

        if initial is None or not _is_zero_initializer(initial):
            continue

        region = loop.regions[0]
        dots = tuple(
            op
            for op in region.operations
            if op.opcode in {"linalg.dot", "linalg.matmul"}
            and len(op.operands) >= 2
            and len(op.results) == 1
        )

        if len(dots) != 1:
            continue

        dot = dots[0]
        block_arg = str(iter_args[0].get("block_arg", ""))
        add = next(
            (
                op
                for op in region.operations
                if op.opcode == "arith.add"
                and len(op.results) == 1
                and set(op.operands) == {block_arg, dot.results[0].name}
            ),
            None,
        )
        yield_op = next(
            (op for op in region.operations if op.opcode == "scf.yield"), None
        )

        if (
            add is None
            or yield_op is None
            or yield_op.operands != (add.results[0].name,)
        ):
            continue

        lhs_axes = _value_axes_from_types(dot.operands[0], value_types)
        rhs_axes = _value_axes_from_types(dot.operands[1], value_types)
        result_axes = tuple(str(dim) for dim in dot.results[0].type.shape)

        if not _static_cooperative_dot_shape(lhs_axes, rhs_axes, result_axes):
            continue

        for store in stores:
            if _value_depends_on(store.operands[0], loop.results[0].name, operations):
                return _CooperativeDotPlan(loop=loop, dot=dot, store=store)
    return None


def _resolved_dot_operand_dtype(name: str, ctx: _EmitContext) -> str:
    producer = ctx.operations.get(name)

    if producer is None:
        info = ctx.tensor_infos.get(name)
        type_ = ctx.value_types.get(name)
        dtype = (
            info.dtype
            if info is not None
            else type_.dtype
            if type_ is not None
            else None
        )

        return _normalize_dtype(dtype)

    if producer.opcode == "tensor.cast":
        return _normalize_dtype(_resolved_cast_dtype(producer, ctx))

    if (
        producer.opcode
        in {
            "tensor.extract",
            "tensor.view",
            "linalg.transpose",
        }
        and producer.operands
    ):
        return _resolved_dot_operand_dtype(producer.operands[0], ctx)

    if producer.results and producer.results[0].type.dtype:
        return _normalize_dtype(producer.results[0].type.dtype)

    if producer.operands:
        return _resolved_dot_operand_dtype(producer.operands[0], ctx)
    return "float32"


def _value_axes_from_types(
    name: str, value_types: Mapping[str, ssa.Type]
) -> tuple[str, ...]:
    type_ = value_types.get(name)

    return () if type_ is None else tuple(str(dim) for dim in type_.shape)


def _static_cooperative_dot_shape(
    lhs_axes: tuple[str, ...],
    rhs_axes: tuple[str, ...],
    result_axes: tuple[str, ...],
) -> bool:
    if len(lhs_axes) != 2 or len(rhs_axes) != 2 or len(result_axes) != 2:
        return False

    if lhs_axes[-1] != rhs_axes[0]:
        return False

    try:
        dimensions = tuple(int(axis) for axis in (*lhs_axes, *rhs_axes, *result_axes))
    except ValueError:
        return False
    return all(dimension > 0 and dimension % 16 == 0 for dimension in dimensions)


def _is_zero_initializer(op: ssa.Operation) -> bool:
    if op.opcode in {"tensor.zeros", "tensor.empty"}:
        return op.opcode == "tensor.zeros"
    return (
        op.opcode in {"arith.constant", "tensor.full"} and op.attrs.get("value", 0) == 0
    )


def _tensor_info(tensor: TensorSpec) -> _TensorInfo:
    attrs = dict(tensor.attrs)
    source_shape = tuple(str(dim) for dim in attrs.get("source_shape", ()))
    source_strides = tuple(str(dim) for dim in attrs.get("source_strides", ()))

    return _TensorInfo(
        ndim=max(tensor.ndim, len(tensor.shape)),
        shape=tuple(str(dim) for dim in tensor.shape),
        dtype=_normalize_dtype(tensor.dtype or "float32"),
        name=tensor.name,
        source_name=str(attrs.get("source_name")) if attrs.get("source_name") else None,
        source_shape=source_shape,
        source_strides=source_strides,
        view_linear_offset=str(attrs.get("view_linear_offset"))
        if attrs.get("view_linear_offset")
        else None,
        view_mask=str(attrs.get("view_mask")) if attrs.get("view_mask") else None,
        attrs=attrs,
    )


def _auxiliary_pointer_bindings(
    tensors: tuple[TensorSpec, ...],
) -> tuple[Mapping[str, str], ...]:
    bindings = []

    for tensor in tensors:
        offsets_param = tensor.attrs.get("jagged_offsets_param")

        if offsets_param:
            bindings.append(
                {
                    "name": str(offsets_param),
                    "kind": "jagged_offsets",
                    "source": tensor.name,
                    "dtype": "int64",
                    "storage_extent": str(
                        tensor.attrs.get("jagged_offsets_numel_param", "n")
                    ),
                }
            )
    return tuple(bindings)


def _tensor_axes(
    info: _TensorInfo | None, *, fallback: tuple[str, ...]
) -> tuple[str, ...]:
    if info is None:
        return fallback

    shape = tuple(axis for axis in info.shape if axis != "")

    if shape:
        return shape

    if info.ndim > 0:
        return tuple(f"dim{i}" for i in range(info.ndim))
    return fallback


def _source_axes(
    info: _TensorInfo | None, *, fallback: tuple[str, ...]
) -> tuple[str, ...]:
    if info is None:
        return fallback

    shape = tuple(axis for axis in info.source_shape if axis != "")

    if shape:
        return shape
    return _tensor_axes(info, fallback=fallback)


def _shape_params(
    tensors: tuple[TensorSpec, ...],
    operations: tuple[ssa.Operation, ...] = (),
) -> tuple[str, ...]:
    include_source_shape = any(
        op.opcode == "shape.dim" and bool(op.attrs.get("source"))
        for op in _walk_ops(operations)
    )
    params: list[str] = []
    computed_symbols = {
        str(tensor.attrs["jagged_seq_len_param"])
        for tensor in tensors
        if tensor.attrs.get("jagged_seq_len_param")
    }

    for tensor in tensors:
        if tensor.constexpr and tensor.ndim == 0 and tensor.name not in params:
            params.append(tensor.name)

        for attr_name in (
            "jagged_values_numel_param",
            "jagged_offsets_numel_param",
        ):
            for symbol in _symbols_in_text(str(tensor.attrs.get(attr_name) or "")):
                if symbol not in params:
                    params.append(symbol)

        dims = list(tensor.shape)
        dims.extend(tuple(tensor.attrs.get("source_shape", ())))
        dims.extend(tuple(tensor.attrs.get("application_shape", ())))

        for dtype_shape in tensor.attrs.get("dtype_shapes", ()):
            dims.extend(tuple(dtype_shape))

        for template in tensor.attrs.get("access_templates", ()):
            dims.append(str(template.get("linear_offset", "")))
            dims.append(str(template.get("mask", "")))
            dims.extend(str(offset) for offset in template.get("offsets", ()))

        for attr_name in ("view_linear_offset", "view_mask"):
            value = tensor.attrs.get(attr_name)

            if value:
                dims.append(str(value))

        if include_source_shape:
            dims.extend(tuple(tensor.attrs.get("source_shape", ())))

        for dim in dims:
            text = str(dim)

            for symbol in _symbols_in_text(text):
                if symbol not in computed_symbols and symbol not in params:
                    params.append(symbol)
    return tuple(params)


def _axis_extent(ctx: _EmitContext, axis: int) -> str:
    if axis < len(ctx.output_axes):
        return ctx.output_axes[axis]

    for info in ctx.tensor_infos.values():
        axes = _tensor_axes(info, fallback=ctx.output_axes)

        if axis < len(axes):
            return axes[axis]
    return "n"


def _value_axes(name: str, ctx: _EmitContext) -> tuple[str, ...]:
    type_ = ctx.value_types.get(name)

    if type_ is not None:
        shape = _value_type_axes(type_)

        if shape:
            return shape

        if type_.kind == "scalar":
            return ()

    operation = ctx.operations.get(name)

    if operation is not None and operation.results:
        shape = tuple(str(dim) for dim in operation.results[0].type.shape if str(dim))

        if shape:
            return shape

    if name in ctx.tensor_infos:
        return _tensor_axes(ctx.tensor_infos[name], fallback=ctx.output_axes)
    return ctx.output_axes


def _value_type_axes(type_: ssa.Type | None) -> tuple[str, ...]:
    if type_ is None:
        return ()
    return tuple(str(dim) for dim in type_.shape if str(dim))


def _store_value_axes(
    store: ssa.Operation | None, value_types: Mapping[str, ssa.Type]
) -> tuple[str, ...] | None:
    if store is None:
        return None

    if not store.attrs.get("source") and "target_shape" in store.attrs:
        return tuple(str(dim) for dim in store.attrs.get("target_shape", ()))

    if store.operands:
        axes = _value_type_axes(value_types.get(store.operands[0]))

        if axes:
            return axes
    return None


def _operation_domain_axes(
    operations: tuple[ssa.Operation, ...], value_types: Mapping[str, ssa.Type]
) -> tuple[str, ...]:
    candidates = []

    for operation in operations:
        if operation.opcode.startswith("reduce.") and operation.operands:
            axes = _value_type_axes(value_types.get(operation.operands[0]))

            if axes:
                candidates.append(axes)

    if not candidates:
        return ()
    return max(
        candidates,
        key=lambda axes: (
            sum(axis.strip("() ") != "1" for axis in axes),
            len(axes),
        ),
    )


def _axis_offset_expr(
    axes: tuple[str, ...], dim: Any, index: str, target: _Target
) -> str:
    dim = int(dim or 0)

    if len(axes) <= 1:
        return index

    index_expr = index if _valid_symbol(str(index)) else f"({index})"

    if dim == len(axes) - 1:
        return _target_index_expr(target, f"({index_expr} % {axes[dim]})")

    stride = _product(axes[dim + 1 :])
    div = "/" if target.c_style_syntax else "//"
    base = f"({index_expr} {div} ({stride}))"
    expr = base if dim == 0 else f"({base} % {axes[dim]})"

    return _target_index_expr(target, expr)


def _indent_lines(lines: list[str], target: _Target) -> list[str]:
    prefix = _indent_unit(target)

    return [_indent_block(line, prefix) if line else line for line in lines]


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _indent_unit(target: _Target) -> str:
    return "    " if not target.c_style_syntax else "    "


def _if_header(condition: str, target: _Target) -> str:
    if target.c_style_syntax:
        return f"if ({condition}) {{"
    return f"if {condition}:"


def _empty_block_lines(target: _Target) -> list[str]:
    return ["/* no-op */"] if target.c_style_syntax else ["pass"]


def _zero_value(type_: ssa.Type, target: _Target) -> str:
    if type_.kind == "index" or _normalize_dtype(type_.dtype) in {
        "index",
        "int64",
        "int32",
    }:
        return "0"

    if _normalize_dtype(type_.dtype) == "bool":
        return "false" if target.c_style_syntax else "False"
    return "0.0"


def _uses_mutable_scalar_slots(target: _Target) -> bool:
    return target.uses_mutable_scalar_slots()


def _mutable_scalar_decl_lines(
    target: _Target,
    type_: ssa.Type,
    name: str,
    init: str,
) -> list[str]:
    return target.mutable_scalar_decl(type_, name, init)


def _mutable_scalar_read(target: _Target, name: str) -> str:
    return target.mutable_scalar_read(name)


def _assign_scalar(
    target: _Target, name: str, value: str, *, mutable: bool = False
) -> str:
    return target.assign_scalar(name, value, mutable=mutable)


def _resolved_cast_dtype(op: ssa.Operation, ctx: _EmitContext) -> str:
    attr = op.attrs.get("dtype")

    if isinstance(attr, str):
        text = attr.strip().strip("'\"")

        if text.endswith(".dtype"):
            base = text[: -len(".dtype")].split(".")[-1]
            info = ctx.tensor_infos.get(base)

            if info is not None:
                return info.dtype

            if op.operands:
                operand_op = ctx.operations.get(op.operands[0])

                if operand_op is not None and operand_op.results:
                    return _normalize_dtype(operand_op.results[0].type.dtype)

            if op.results:
                return _normalize_dtype(op.results[0].type.dtype)

        if text:
            return _normalize_dtype(text)

    if op.results:
        dtype = op.results[0].type.dtype

        if dtype:
            return _normalize_dtype(dtype)

    if op.operands:
        operand_op = ctx.operations.get(op.operands[0])

        if operand_op is not None and operand_op.results:
            return _normalize_dtype(operand_op.results[0].type.dtype)

        info = ctx.tensor_infos.get(op.operands[0])

        if info is not None:
            return info.dtype
    return "float32"


def _cast_value(op: ssa.Operation, value: str, ctx: _EmitContext) -> str:
    attr = op.attrs.get("dtype")

    if ctx.target.vector_value_semantics and isinstance(attr, str):
        text = attr.strip().strip("'\"")

        if text.endswith(".dtype"):
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", text)

            if match and match.group(1) in ctx.tensor_infos:
                return f"{value}.to({match.group(1)}.dtype.element_ty)"
    return ctx.target.cast(_resolved_cast_dtype(op, ctx), value)


# Public analysis and traversal hooks used by backend strategies.  Keeping this
# boundary explicit lets the shared walker move between modules without making
# backend implementations depend on private implementation names.
access_axes = _access_axes
buffer_storage_extent = _buffer_storage_extent
combined_mask = _combined_mask
cooperative_dot_plan = _cooperative_dot_plan
current_coords = _current_coords
default_strides = _default_strides
dot_accumulator_dtype = _dot_accumulator_dtype
dtype_level = _dtype_level
emit_element = _emit_element
emit_loop_bound = _emit_loop_bound
emit_operation = _emit_operation
emit_value = _emit_value
indent_block = _indent_block
indent_lines = _indent_lines
linearized_index = _linearized_index
load_other = _load_other
local_symbol = _local_symbol
logical_ssa_audit = _logical_ssa_audit
materialize_bool_expr = _materialize_bool_expr
materialize_index_expr = _materialize_index_expr
normalize_dtype = _normalize_dtype
product = _product
resolved_dot_operand_dtype = _resolved_dot_operand_dtype
rewrite_index_math = _rewrite_index_math
schedule_int = _schedule_int
source_index_for_value = _source_index_for_value
target_index_expr = _target_index_expr
value_axes = _value_axes
value_axes_from_types = _value_axes_from_types
view_base_coords = _view_base_coords

__all__ = [
    "access_axes",
    "buffer_storage_extent",
    "combined_mask",
    "cooperative_dot_plan",
    "current_coords",
    "default_strides",
    "dot_accumulator_dtype",
    "dtype_level",
    "emit",
    "emit_element",
    "emit_loop_bound",
    "emit_operation",
    "emit_value",
    "indent_block",
    "indent_lines",
    "linearized_index",
    "load_other",
    "local_symbol",
    "logical_ssa_audit",
    "materialize_bool_expr",
    "materialize_index_expr",
    "normalize_dtype",
    "product",
    "resolved_dot_operand_dtype",
    "rewrite_index_math",
    "schedule_int",
    "source_index_for_value",
    "target_index_expr",
    "value_axes",
    "value_axes_from_types",
    "view_base_coords",
]
