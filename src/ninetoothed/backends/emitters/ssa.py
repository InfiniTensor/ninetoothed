"""Unified SSA-to-source emitters for backend code generation.

This module is intentionally organized around SSA operations, values, blocks,
and regions.  It does not classify whole kernels into matmul/reduction/etc.
plans before lowering.  Backend-specific code is limited to spelling scalar
expressions, loops, buffers, and launch wrappers.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ninetoothed.backends.core import Artifact, Target
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

_REDUCE_INIT = {
    "sum": "0.0",
    "max": "-3.4028234663852886e+38",
    "min": "3.4028234663852886e+38",
}


@dataclass(frozen=True, kw_only=True)
class _TensorInfo:
    ndim: int = 1
    shape: tuple[str, ...] = ()
    dtype: str = "float32"
    name: str
    source_name: str | None = None
    source_shape: tuple[str, ...] = ()
    source_strides: tuple[str, ...] = ()
    view_linear_offset: str | None = None
    view_mask: str | None = None
    attrs: Mapping[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class _Target:
    backend: Target
    language: str
    suffix: str
    source_route: str
    buffer_suffix: str = ""
    index_name: str = "index"
    block_size: int = 256

    def symbol(self, name: str) -> str:
        return f"v{name[1:]}" if name.startswith("%") else name

    def literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if self.backend == Target.CUDA else "True"

        if isinstance(value, float) and math.isinf(value):
            if self.backend == Target.CUDA:
                return "INFINITY" if value > 0 else "-INFINITY"
            return "float('inf')" if value > 0 else "-float('inf')"

        if value == "inf":
            return "INFINITY" if self.backend == Target.CUDA else "float('inf')"

        if value == "-inf":
            return "-INFINITY" if self.backend == Target.CUDA else "-float('inf')"
        return repr(value)

    def tensor_ref(self, tensor: str) -> str:
        return f"{tensor}{self.buffer_suffix}"

    def load(self, tensor: str, index: str, *, mask: str | None = None) -> str:
        ref = self.tensor_ref(tensor)

        if self.backend == Target.TRITON:
            mask_text = "" if mask is None else f", mask={mask}, other=0.0"

            return f"tl.load({ref} + {index}{mask_text})"
        return f"{ref}[{index}]"

    def store(
        self, tensor: str, index: str, value: str, *, mask: str | None = None
    ) -> str:
        ref = self.tensor_ref(tensor)

        if self.backend == Target.TRITON:
            mask_text = "" if mask is None else f", mask={mask}"

            return f"tl.store({ref} + {index}, {value}{mask_text})"

        suffix = ";" if self.backend == Target.CUDA else ""
        assignment = f"{ref}[{index}] = {value}{suffix}"

        if mask is None:
            return assignment

        if self.backend == Target.CUDA:
            return f"if ({mask}) {{\n    {assignment}\n}}"
        return f"if {mask}:\n    {assignment}"

    def cast(self, dtype: str, value: str) -> str:
        dtype = _normalize_dtype(dtype)

        if self.backend == Target.TRITON:
            return f"{value}.to(tl.{dtype})"

        if self.backend == Target.CUDA:
            return f"static_cast<{_cuda_type(dtype)}>({value})"
        return f'T.Cast("{dtype}", {value})'

    def where(self, cond: str, yes: str, no: str) -> str:
        if self.backend == Target.TRITON:
            return f"tl.where({cond}, {yes}, {no})"

        if self.backend == Target.CUDA:
            return f"(({cond}) ? ({yes}) : ({no}))"
        return f"T.if_then_else({cond}, {yes}, {no})"

    def call(self, name: str, args: tuple[str, ...]) -> str:
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            if self.backend == Target.CUDA:
                return f"atomicAdd({args[0]}, {args[1]})"

            if self.backend == Target.TRITON:
                return f"tl.atomic_add({args[0]}, {args[1]})"
            return f"T.atomic_add({args[0]}, {args[1]})"

        if name == "dot" and self.backend == Target.CUDA and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "rand" and self.backend == Target.CUDA and len(args) >= 2:
            return f"fabsf(fmodf(sinf(static_cast<float>(({args[0]}) + ({args[1]})) * 12.9898f) * 43758.5453f, 1.0f))"

        if name == "expm1" and self.backend in {
            Target.TILELANG,
            Target.TVM,
            Target.TRITON,
        }:
            base = self.call("exp", args)

            return f"({base} - 1.0)"

        if self.backend == Target.CUDA:
            func = {
                "abs": "fabsf",
                "acos": "acosf",
                "asin": "asinf",
                "atan": "atanf",
                "atan2": "atan2f",
                "_atan2_approx": "atan2f",
                "ceil": "ceilf",
                "cos": "cosf",
                "cosh": "coshf",
                "dot": "dot",
                "erf": "erff",
                "exp": "expf",
                "exp2": "exp2f",
                "expm1": "expm1f",
                "floor": "floorf",
                "log": "logf",
                "log1p": "log1pf",
                "log2": "log2f",
                "log10": "log10f",
                "maximum": "fmaxf",
                "max": "fmaxf",
                "minimum": "fminf",
                "min": "fminf",
                "pow": "powf",
                "rsqrt": "rsqrtf",
                "sin": "sinf",
                "sinh": "sinhf",
                "sqrt": "sqrtf",
                "tan": "tanf",
                "tanh": "tanhf",
            }.get(name)
        elif self.backend == Target.TRITON:
            func = {
                "abs": "tl.abs",
                "acos": "tl.acos",
                "asin": "tl.asin",
                "atan": "tl.atan",
                "atan2": "tl.atan2",
                "_atan2_approx": "tl.atan2",
                "ceil": "tl.ceil",
                "cos": "tl.cos",
                "cosh": "tl.cosh",
                "erf": "tl.erf",
                "exp": "tl.exp",
                "exp2": "tl.exp2",
                "floor": "tl.floor",
                "log": "tl.log",
                "log1p": "tl.log",
                "log2": "tl.log2",
                "log10": "tl.log",
                "maximum": "tl.maximum",
                "max": "tl.maximum",
                "minimum": "tl.minimum",
                "min": "tl.minimum",
                "pow": "tl.pow",
                "rsqrt": "tl.rsqrt",
                "sin": "tl.sin",
                "sinh": "tl.sinh",
                "sqrt": "tl.sqrt",
                "tan": "tl.tan",
                "tanh": "tl.tanh",
            }.get(name)

            if name == "log1p":
                return f"tl.log(1.0 + {args[0]})"

            if name == "log10":
                return f"(tl.log({args[0]}) / 2.302585092994046)"

            if name == "dot" and len(args) == 2:
                return f"(({args[0]}) * ({args[1]}))"
        else:
            func = {
                "abs": "T.abs",
                "acos": "T.acos",
                "asin": "T.asin",
                "atan": "T.atan",
                "atan2": "T.atan2",
                "_atan2_approx": "T.atan2",
                "ceil": "T.ceil",
                "cos": "T.cos",
                "cosh": "T.cosh",
                "dot": "T.dot",
                "erf": "T.erf",
                "exp": "T.exp",
                "exp2": "T.exp2",
                "floor": "T.floor",
                "log": "T.log",
                "log1p": "T.log1p",
                "log2": "T.log2",
                "log10": "T.log10",
                "maximum": "T.max",
                "max": "T.max",
                "minimum": "T.min",
                "min": "T.min",
                "pow": "T.pow",
                "rsqrt": "T.rsqrt",
                "sin": "T.sin",
                "sinh": "T.sinh",
                "sqrt": "T.sqrt",
                "tan": "T.tan",
                "tanh": "T.tanh",
            }.get(name)

        if func is None:
            func = (
                name
                if self.backend == Target.CUDA
                else f"T.{name}"
                if self.backend in {Target.TILELANG, Target.TVM}
                else name
            )
        return f"{func}({', '.join(args)})"

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        if self.backend == Target.CUDA:
            return f"{_cuda_type(type_.dtype, type_.kind)} {name} = {expr};"
        return f"{name} = {expr}"

    def loop_header(self, var: str, lower: str, upper: str, step: str) -> str:
        if self.backend == Target.CUDA:
            return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

        if self.backend == Target.TRITON:
            return f"for {var} in range({lower}, {upper}, {step}):"

        serial = (
            f"T.serial({upper})"
            if lower == "0" and step == "1"
            else f"T.serial({lower}, {upper})"
            if step == "1"
            else f"T.serial({lower}, {upper}, {step})"
        )

        return f"for {var} in {serial}:"

    def reduce_update(self, operator: str, acc: str, term: str) -> str:
        if operator == "sum":
            return f"{acc} + {term}"

        if self.backend == Target.CUDA:
            return (
                f"fmaxf({acc}, {term})"
                if operator == "max"
                else f"fminf({acc}, {term})"
            )

        if self.backend == Target.TRITON:
            return (
                f"tl.maximum({acc}, {term})"
                if operator == "max"
                else f"tl.minimum({acc}, {term})"
            )
        return f"T.max({acc}, {term})" if operator == "max" else f"T.min({acc}, {term})"


@dataclass(kw_only=True)
class _EmitContext:
    target: _Target
    kernel: Kernel
    program: ssa.Program
    operations: Mapping[str, ssa.Operation]
    value_types: Mapping[str, ssa.Type]
    lines: list[str]
    memo: dict[str, str]
    tensor_infos: Mapping[str, _TensorInfo]
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
    bindings: Mapping[str, str] | None = None
    temp_counter: list[int] | None = None
    materialized: dict[tuple[str, str], str] | None = None
    indent: str = ""
    local_suffix: str = ""

    def child(
        self,
        *,
        lines: list[str] | None = None,
        memo: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> "_EmitContext":
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
            "bindings": self.bindings,
            "temp_counter": self.temp_counter,
            "materialized": self.materialized if lines is None else {},
            "indent": self.indent,
            "local_suffix": self.local_suffix,
        }
        data.update(kwargs)

        return _EmitContext(**data)


def emit(kernel: Kernel, backend: Target) -> Artifact:
    if kernel.ssa is None:
        raise ValueError("Backend emission requires ssa.Program.")

    target = _target(backend)
    block = kernel.ssa.blocks[0] if kernel.ssa.blocks else ssa.Block()
    shape_params = _shape_params(kernel.tensors, block.operations)
    source = _render_source(kernel, target)
    metadata = {
        "backend": backend.value,
        "kernel_name": kernel.kernel_name,
        "lowering_ir": "ssa.Program",
        "source_route": target.source_route,
        "shape_params": shape_params,
        "ssa": ir_to_dict(kernel.ssa),
        "ssa_metadata": dict(kernel.ssa.metadata),
        "tensors": [tensor.__dict__ for tensor in kernel.tensors],
    }

    return Artifact(
        backend=backend,
        kernel_name=kernel.kernel_name,
        language=target.language,
        sources={
            f"{kernel.kernel_name}.{target.suffix}": source,
            f"{kernel.kernel_name}.{backend.value}.json": json.dumps(
                metadata, indent=2
            ),
        },
        entrypoint=_entrypoint(kernel, backend),
        executable=True,
        metadata=metadata,
    )


def _target(backend: Target) -> _Target:
    return {
        Target.TRITON: _Target(
            backend=backend,
            language="python/triton",
            suffix="triton.py",
            source_route="ssa-unified-triton-emitter",
        ),
        Target.CUDA: _Target(
            backend=backend,
            language="cuda/c++",
            suffix="cu",
            source_route="ssa-unified-cuda-emitter",
        ),
        Target.TILELANG: _Target(
            backend=backend,
            language="python/tilelang",
            suffix="tilelang.py",
            source_route="ssa-unified-tilelang-emitter",
            buffer_suffix="_buf",
        ),
        Target.TVM: _Target(
            backend=backend,
            language="python/tvm-script",
            suffix="tvm.py",
            source_route="ssa-unified-tvm-emitter",
            buffer_suffix="_buf",
        ),
    }[backend]


def _render_source(kernel: Kernel, target: _Target) -> str:
    program = kernel.ssa
    assert program is not None
    block = program.blocks[0] if program.blocks else ssa.Block()
    tensor_infos = {tensor.name: _tensor_info(tensor) for tensor in kernel.tensors}
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
    variables = tuple(
        tensor.name
        for tensor in kernel.tensors
        if tensor.name not in outputs and not tensor.constexpr
    )
    shape_params = _shape_params(kernel.tensors, block.operations)

    if "index" in {*variables, *outputs, *shape_params}:
        target = replace(target, index_name="__nt_index")

    output = (
        outputs[0]
        if outputs
        else (kernel.tensors[-1].name if kernel.tensors else "out")
    )
    output_info = tensor_infos.get(output)
    outer_axes = _tensor_axes(output_info, fallback=("n",))
    value_axes = _value_type_axes(value_types.get(output)) or tuple(
        str(dim) for dim in (output_info.attrs or {}).get("application_shape", ())
    )
    output_attrs = output_info.attrs or {} if output_info is not None else {}
    split_outer_inner = (
        value_axes
        and output_info is not None
        and output_attrs.get("application_shape")
        and int(output_attrs.get("view_ndim", len(outer_axes)))
        > int(output_attrs.get("application_ndim", len(value_axes)))
    )

    if split_outer_inner:
        axes = value_axes
        inner_total = _product(value_axes)
        total = _target_index_expr(
            target, f"({_product(outer_axes)}) * ({inner_total})"
        )
        outer_index_expr = _target_index_expr(
            target, f"floor(({target.index_name})/({inner_total}))"
        )
        inner_index_expr = _target_index_expr(
            target, f"({target.index_name} % ({inner_total}))"
        )
    else:
        axes = outer_axes
        total = _target_index_expr(target, _product(axes))
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
    )

    if target.backend == Target.TRITON:
        return _render_triton_module(
            kernel, target, variables, outputs, shape_params, total, body
        )

    if target.backend == Target.CUDA:
        return _render_cuda_module(
            kernel, target, variables, outputs, shape_params, total, body, tensor_infos
        )

    if target.backend == Target.TILELANG:
        return _render_tilelang_module(
            kernel,
            target,
            variables,
            outputs,
            shape_params,
            total,
            body,
            tensor_infos,
            value_types,
        )
    return _render_tvm_module(
        kernel,
        target,
        variables,
        outputs,
        shape_params,
        total,
        body,
        tensor_infos,
        value_types,
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
) -> str:
    output = outputs[0] if outputs else "out"
    lines: list[str] = []
    coordinate_exprs: tuple[str, ...] = ()
    enable_index_cse = (
        target.backend in {Target.CUDA, Target.TILELANG, Target.TVM}
        and outer_index_expr != inner_index_expr
    )

    if enable_index_cse:
        index_type = ssa.Type(kind="index", dtype="index")

        if outer_index_expr != target.index_name:
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
        mask_expr="mask" if target.backend == Target.TRITON else None,
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
    )

    for op in operations:
        if _is_top_level_effect(op):
            _emit_operation(op, ctx)

    if not ctx.lines:
        ctx.lines.append("pass" if target.backend != Target.CUDA else "/* no-op */")
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


def _emit_operation(op: ssa.Operation, ctx: _EmitContext) -> None:
    if op.opcode == "scf.yield":
        return

    if op.opcode == "mem.store":
        value = _emit_value(op.operands[0], ctx)
        tensor = op.operands[1]
        view_index = _store_index(op, ctx)
        store_index = _target_index_expr(
            ctx.target,
            _source_index_for_value(
                ctx.tensor_infos.get(tensor),
                view_index,
                ctx,
                level=_dtype_level(tensor, ctx),
            ),
        )
        store_index = _materialize_index_expr(store_index, ctx)
        mask = _store_mask(
            ctx.target, ctx.mask_expr, ctx.tensor_infos.get(tensor), view_index, ctx=ctx
        )
        mask = _materialize_bool_expr(mask, ctx)
        ctx.lines.append(ctx.target.store(tensor, store_index, value, mask=mask))

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
            if _is_bool_scalar_value(name, ctx) and ctx.target.backend in {
                Target.TILELANG,
                Target.TVM,
            }:
                return f"({name} != 0)"
            return name
        return _tensor_value(name, ctx)

    op = ctx.operations[name]
    local = _local_symbol(name, ctx)

    if op.opcode.startswith("reduce."):
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
    elif _should_emit_tensor_value_as_element(op):
        expr = _emit_element(name, _current_coords(_value_axes(name, ctx), ctx), ctx)
    else:
        expr = _operation_expr(op, ctx)

    ctx.lines.append(ctx.target.local_decl(op.results[0].type, local, expr))
    ctx.memo[name] = local

    return local


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
        tensor = op.operands[0] if op.operands else ctx.output
        axes = _tensor_axes(ctx.tensor_infos.get(tensor), fallback=ctx.output_axes)

        return _axis_offset_expr(
            axes, op.attrs.get("dim", 0), ctx.index_expr, ctx.target
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
        axes = _tensor_axes(ctx.tensor_infos.get(tensor), fallback=ctx.output_axes)

        return _stride_dim(axes, op.attrs.get("dim", 0))

    if opcode == "mem.data_ptr":
        return ctx.target.tensor_ref(op.operands[0])

    if opcode == "mem.atomic_add":
        return target.call(
            "atomic_add", tuple(_emit_value(operand, ctx) for operand in op.operands)
        )

    if opcode == "tensor.view":
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
        index = _linearized_index(indices, _value_axes(tensor, ctx))

        return _load_tensor(tensor, index, ctx)

    if opcode == "tensor.cast":
        dtype = _resolved_cast_dtype(op, ctx)

        return target.cast(dtype, _emit_value(op.operands[0], ctx))

    if opcode == "select.where":
        args = tuple(_emit_value(operand, ctx) for operand in op.operands)
        args = (_materialize_bool_expr(args[0], ctx) or args[0], args[1], args[2])

        return target.where(args[0], args[1], args[2])

    if opcode.startswith("cmp."):
        return _binary_expr(opcode[len("cmp.") :], op.operands, ctx)

    if opcode.startswith("arith."):
        operator = opcode[len("arith.") :]
        args = tuple(_emit_value(operand, ctx) for operand in op.operands)

        if operator in _UNARY:
            return f"({_UNARY[operator]}{args[0]})"

        if operator == "floordiv":
            return (
                f"(({args[0]}) // ({args[1]}))"
                if target.backend != Target.CUDA
                else f"(({args[0]}) / ({args[1]}))"
            )

        if operator == "pow":
            return target.call("pow", args)

        if operator in {"maximum", "max"}:
            return target.call("maximum", args)

        if operator in {"minimum", "min"}:
            return target.call("minimum", args)
        return _binary_expr(operator, op.operands, ctx)

    if opcode.startswith("math."):
        return target.call(
            opcode[len("math.") :],
            tuple(_emit_value(operand, ctx) for operand in op.operands),
        )

    if opcode.startswith("call."):
        return target.call(
            opcode[len("call.") :],
            tuple(_emit_value(operand, ctx) for operand in op.operands),
        )

    if opcode == "symbol.attr":
        return str(op.attrs.get("expr", "0"))

    if opcode == "tuple.construct":
        return (
            "(" + ", ".join(_emit_value(operand, ctx) for operand in op.operands) + ")"
        )

    if opcode in {"linalg.matmul", "linalg.dot"}:
        return _emit_linalg_dot(op, ctx)

    if opcode == "linalg.transpose":
        return _emit_value(op.operands[0], ctx)

    raise ValueError(f"Unsupported SSA opcode `{opcode}` for unified backend emitter.")


def _binary_expr(operator: str, operands: tuple[str, ...], ctx: _EmitContext) -> str:
    args = tuple(_emit_value(operand, ctx) for operand in operands)
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

    if not lhs_axes or not rhs_axes:
        return ctx.target.call(
            "dot", tuple(_emit_value(operand, ctx) for operand in op.operands)
        )

    local = f"{_local_symbol(op.results[0].name, ctx)}_dot"
    acc_type = ssa.Type(kind="scalar", dtype=op.results[0].type.dtype or "float32")
    init = "0.0"

    if ctx.target.backend == Target.TRITON and ctx.mask_expr is not None:
        dtype = _normalize_dtype(acc_type.dtype or "float32")
        init = f"tl.full((BLOCK,), {init}, tl.{dtype})"

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
    lhs_value = _emit_element(lhs, lhs_coords, body)
    rhs_value = _emit_element(rhs, rhs_coords, body)
    body_lines.append(
        _assign_scalar(
            ctx.target,
            local,
            f"{acc_expr} + (({lhs_value}) * ({rhs_value}))",
            mutable=mutable,
        )
    )
    ctx.lines.extend(_indent_lines(body_lines, ctx.target))

    if ctx.target.backend == Target.CUDA:
        ctx.lines.append("}")
    return acc_expr


def _emit_element(name: str, coords: tuple[str, ...], ctx: _EmitContext) -> str:
    if ctx.bindings and name in ctx.bindings and not coords:
        return ctx.bindings[name]

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
            level = int(
                op.results[0].type.attrs.get("dtype_level", _dtype_level(base, ctx))
            )

            return _load_tensor_at(
                base, coords, ctx, level=level, extract_indices=extract_indices
            )
        return _emit_element(base, (*extract_indices, *coords), ctx)

    if op.opcode == "tensor.view":
        return _emit_element(op.operands[0], _view_base_coords(op, coords, ctx), ctx)

    if op.opcode == "linalg.transpose":
        return _emit_element(op.operands[0], tuple(reversed(coords)), ctx)

    if op.opcode == "tensor.cast":
        return ctx.target.cast(
            _resolved_cast_dtype(op, ctx), _emit_element(op.operands[0], coords, ctx)
        )

    if op.opcode == "index.offset":
        return _emit_offset_element(op, coords, ctx)

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
        return ctx.target.call(
            op.opcode[len("math.") :],
            tuple(
                _emit_element_arg(op, operand, coords, ctx) for operand in op.operands
            ),
        )

    if op.opcode.startswith("reduce."):
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
    return op.opcode in {"index.offset", "tensor.view"}


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
    args = _element_args(op, coords, ctx)

    if operator == "floordiv":
        return (
            f"(({args[0]}) // ({args[1]}))"
            if ctx.target.backend != Target.CUDA
            else f"(({args[0]}) / ({args[1]}))"
        )

    symbol = _BINARY[operator]

    return f"({args[0]} {symbol} {args[1]})"


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
    axis = op.attrs.get("axis")

    if axis is None:
        axis = 0

    axis = int(axis)

    if axis < 0:
        axis += len(operand_axes)

    upper = (
        operand_axes[axis] if 0 <= axis < len(operand_axes) else _axis_extent(ctx, axis)
    )

    if local is None:
        local = f"{_local_symbol(op.results[0].name, ctx)}_elem"

    result_type = ssa.Type(
        kind="scalar", dtype=op.results[0].type.dtype if op.results else "float32"
    )
    init = _REDUCE_INIT[operator]

    if ctx.target.backend == Target.TRITON and ctx.mask_expr is not None:
        dtype = _normalize_dtype(result_type.dtype or "float32")
        init = f"tl.full((BLOCK,), {init}, tl.{dtype})"

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

    if ctx.target.backend == Target.CUDA:
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
        ),
    )
    source_index = _materialize_index_expr(source_index, ctx)
    mask = _combined_mask(
        ctx.target,
        ctx.mask_expr if ctx.target.backend == Target.TRITON else None,
        info,
        view_index,
        ctx=ctx,
    )

    return ctx.target.load(name, source_index, mask=mask)


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
        return "0"

    offsets = tuple(str(offset) for offset in template.get("offsets", ()))
    source_ndim = len(offsets)

    if dim < 0:
        dim += source_ndim

    if dim < 0 or dim >= source_ndim:
        return "0"

    shape = tuple(str(axis) for axis in template.get("shape", ())) or ctx.output_axes
    value_index = _linearized_index(coords, shape) if coords else "0"
    value_coords = _coords_from_linear(value_index, shape, ctx.target)
    replacements = {"outer_index": ctx.outer_index_expr}
    replacements.update(
        {f"value_{index}": coord for index, coord in enumerate(value_coords)}
    )

    for index, value in enumerate(extract_indices):
        replacements[f"extract_0_{index}"] = value
    return _target_index_expr(ctx.target, _replace_symbols(offsets[dim], replacements))


def _current_coords(axes: tuple[str, ...], ctx: _EmitContext) -> tuple[str, ...]:
    if not axes:
        return ()

    output_axes = tuple(str(axis) for axis in ctx.output_axes)

    if output_axes:
        output_coords = ctx.coordinate_exprs or tuple(
            _axis_offset_expr(output_axes, dim, ctx.inner_index_expr, ctx.target)
            for dim in range(len(output_axes))
        )
        coords: tuple[str, ...] | None = None

        if len(axes) == len(output_axes):
            coords = tuple(
                "0" if axis == "1" else output_coords[index]
                for index, axis in enumerate(axes)
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

    if axis == 1:
        upper = operand_axes[1] if len(operand_axes) > 1 else _axis_extent(ctx, 1)
    else:
        upper = str(
            op.attrs.get("extent")
            or (operand_axes[0] if operand_axes else _axis_extent(ctx, 0))
        )

    result_type = op.results[0].type
    init = _REDUCE_INIT[operator]

    if ctx.target.backend == Target.TRITON and axis is not None:
        dtype = _normalize_dtype(result_type.dtype or "float32")
        init = f"tl.full((BLOCK,), {init}, tl.{dtype})"

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
        reduce_axis=0 if axis is None else axis,
        reduce_index=loop_var,
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

    if ctx.target.backend == Target.CUDA:
        ctx.lines.append("}")
    return acc_expr


def _emit_scf_for(local: str, op: ssa.Operation, ctx: _EmitContext) -> str | None:
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
            ctx.target.backend == Target.TRITON
            and ctx.mask_expr is not None
            and _needs_triton_block_init(initial_name, value, ctx)
        ):
            dtype = _normalize_dtype(value.type.dtype or "float32")
            init = f"tl.full((BLOCK,), {init}, tl.{dtype})"

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

    if ctx.target.backend == Target.CUDA:
        ctx.lines.append("}")
    return ctx.memo.get(result_names[0]) if result_names else None


def _is_scalar_seed(name: str, ctx: _EmitContext) -> bool:
    if not name.startswith("%"):
        return False

    op = ctx.operations.get(name)

    if op is None or not op.results:
        return False
    return op.results[0].type.kind == "scalar"


def _needs_triton_block_init(name: str, value: ssa.Value, ctx: _EmitContext) -> bool:
    if _is_scalar_seed(name, ctx):
        return True

    if value.type.kind != "tensor" or not name.startswith("%"):
        return False

    op = ctx.operations.get(name)

    if op is None:
        return False
    return op.opcode in {"arith.constant", "tensor.zeros", "tensor.full"}


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
        ctx.lines.append("} else {" if ctx.target.backend == Target.CUDA else "else:")
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

    if ctx.target.backend == Target.CUDA:
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
            ctx.lines.append(
                "} else {" if ctx.target.backend == Target.CUDA else "else:"
            )

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

    if ctx.target.backend == Target.CUDA:
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
    if ctx.target.backend != Target.TRITON:
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
            ctx.lines.append(
                "} else {" if ctx.target.backend == Target.CUDA else "else:"
            )

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
        ctx.lines.append("} else {" if ctx.target.backend == Target.CUDA else "else:")
        ctx.lines.extend(_indent_lines(_empty_block_lines(ctx.target), ctx.target))

    if ctx.target.backend == Target.CUDA:
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
        if _is_bool_scalar_value(name, ctx) and ctx.target.backend in {
            Target.TILELANG,
            Target.TVM,
        }:
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
    base_mask = ctx.mask_expr if ctx.target.backend == Target.TRITON else None

    if (
        ctx.target.backend == Target.TRITON
        and base_mask is not None
        and ctx.target.index_name not in source_index
        and "offsets" not in source_index
    ):
        base_mask = None

    mask = _combined_mask(
        ctx.target,
        base_mask,
        info,
        view_index,
        ctx=ctx,
    )

    return ctx.target.load(name, source_index, mask=mask)


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
) -> str:
    template = _access_template(info, level)

    if template is None:
        return _source_index(info, view_index)

    shape = tuple(str(dim) for dim in template.get("shape", ())) or _tensor_axes(
        info, fallback=ctx.output_axes
    )
    coords = _coords_from_linear(view_index, shape, ctx.target)
    replacements = {"outer_index": ctx.outer_index_expr}
    replacements.update({f"value_{index}": coord for index, coord in enumerate(coords)})

    for index, value in enumerate(extract_indices):
        replacements[f"extract_0_{index}"] = value

    split_index = _source_index_from_offsets(info, template, replacements, ctx)

    if split_index is not None:
        return split_index
    return _replace_symbols(
        str(template.get("linear_offset", view_index)), replacements
    )


def _source_index_from_offsets(
    info: _TensorInfo | None,
    template: Mapping[str, Any],
    replacements: Mapping[str, str],
    ctx: _EmitContext,
) -> str | None:
    if ctx.target.backend not in {
        Target.CUDA,
        Target.TILELANG,
        Target.TVM,
    }:
        return None

    offsets = tuple(str(offset) for offset in template.get("offsets", ()))

    if not offsets:
        return None

    strides = _source_strides(info, prefer_default=True)

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
) -> str | None:
    if target.backend in {Target.TILELANG, Target.TVM}:
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
    return _combined_mask(target, base, info, view_index, ctx=ctx)


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
    checks: list[str] = []

    for offset, dim in zip(offsets, source_shape):
        offset_expr = _target_index_expr(
            ctx.target, _replace_symbols(offset, replacements)
        )
        offset_expr = _materialize_index_expr(offset_expr, ctx, threshold=48)
        checks.append(f"(({offset_expr}) >= 0)")
        checks.append(f"(({dim}) > ({offset_expr}))")

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


def _default_strides(shape: tuple[str, ...]) -> tuple[str, ...]:
    strides: list[str] = []
    acc = "1"

    for dim in reversed(shape):
        strides.append(acc)
        acc = dim if _is_one_expr(acc) else f"({dim}) * ({acc})"
    return tuple(reversed(strides))


def _is_zero_expr(expr: str) -> bool:
    return expr.strip("() ") == "0"


def _is_one_expr(expr: str) -> bool:
    return expr.strip("() ") == "1"


def _combined_mask(
    target: _Target,
    base: str | None,
    info: _TensorInfo | None,
    view_index: str,
    *,
    ctx: _EmitContext | None = None,
) -> str | None:
    masks = []

    if base:
        masks.append(_target_index_expr(target, base))

    if ctx is not None:
        template = _access_template(
            info, _dtype_level(info.name, ctx) if info is not None else 0
        )

        if template is not None:
            shape = (
                tuple(str(dim) for dim in template.get("shape", ())) or ctx.output_axes
            )
            coords = _coords_from_linear(view_index, shape, target)
            replacements = {"outer_index": ctx.outer_index_expr}
            replacements.update(
                {f"value_{index}": coord for index, coord in enumerate(coords)}
            )
            template_mask = _replace_symbols(
                str(template.get("mask", "True")), replacements
            )

            if template_mask and template_mask != "True":
                masks.append(_target_index_expr(target, template_mask))

    if info is not None and info.view_mask and info.view_mask != "True":
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


def _replace_index_symbol(expr: str, value: str) -> str:
    return re.sub(r"\bindex\b", f"({value})", expr)


def _replace_symbols(expr: str, replacements: Mapping[str, str]) -> str:
    for name in sorted(replacements, key=len, reverse=True):
        expr = re.sub(rf"\b{re.escape(name)}\b", f"({replacements[name]})", expr)
    return expr


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


_TVM_INDEX_LITERAL_RE = re.compile(
    r"(?<!T\.int64\()(?<![A-Za-z0-9_.'\"])([0-9]+)(?![A-Za-z0-9_.'\"])"
)


def _target_index_expr(target: _Target, expr: str) -> str:
    rewritten = _rewrite_index_math(expr, cuda=target.backend == Target.CUDA)

    if target.backend == Target.TVM:
        return _tvm_index_literals(rewritten)
    return rewritten


def _tvm_index_literals(expr: str) -> str:
    return _TVM_INDEX_LITERAL_RE.sub(r"T.int64(\1)", expr)


def _materialize_index_expr(
    expr: str, ctx: _EmitContext, *, threshold: int = 96
) -> str:
    if ctx.target.backend not in {
        Target.CUDA,
        Target.TILELANG,
        Target.TVM,
    }:
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

    if ctx.target.backend not in {
        Target.CUDA,
        Target.TILELANG,
        Target.TVM,
    }:
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

    if info.ndim <= 1:
        if (
            len(ctx.output_axes) >= 2
            and name != ctx.output
            and ctx.col_expr is not None
        ):
            return ctx.col_expr
        return ctx.index_expr

    if ctx.row_expr is None or ctx.col_expr is None:
        return ctx.index_expr

    axes = _value_axes(name, ctx)

    if len(axes) >= 2:
        return f"({ctx.row_expr}) * ({axes[1]}) + ({ctx.col_expr})"
    return ctx.index_expr


def _store_index(op: ssa.Operation, ctx: _EmitContext) -> str:
    indices = op.attrs.get("indices", ())

    if isinstance(indices, str):
        indices = (indices,)

    if indices:
        rendered = tuple(_emit_index_value(str(index), ctx) for index in indices)

        return _linearized_index(rendered, _value_axes(op.operands[1], ctx))
    return ctx.index_expr


def _emit_index_value(name: str, ctx: _EmitContext) -> str:
    value = _emit_value(name, ctx)

    if ctx.target.backend == Target.CUDA and not _integer_expr(value):
        return f"static_cast<int64_t>({value})"
    return value


def _integer_expr(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", value))


def _render_triton_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    total: str,
    body: str,
) -> str:
    params = ",\n    ".join(
        (
            *variables,
            *outputs,
            *[f"{axis}: tl.constexpr" for axis in shape_params],
            "BLOCK: tl.constexpr",
        )
    )
    launch_params = ", ".join((*variables, *outputs, *shape_params))
    kernel_args = ",\n        ".join((*variables, *outputs, *shape_params))

    return f'''"""Triton lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
Lowering IR: ssa.Program
"""

import triton
import triton.language as tl
from math import floor


@triton.jit
def {kernel.kernel_name}_kernel(
    {params},
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    {target.index_name} = offsets
    mask = offsets < ({total})
{_indent_block(body, "    ")}


def launch_{kernel.kernel_name}({launch_params}):
    block = 256
    grid = (triton.cdiv({total}, block),)
    {kernel.kernel_name}_kernel[grid](
        {kernel_args},
        BLOCK=block,
        num_warps=4,
    )
    return {outputs[0] if outputs else "None"}
'''


def _render_cuda_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    total: str,
    body: str,
    tensors: Mapping[str, _TensorInfo],
) -> str:
    total = _cuda_integer_expr(total)
    body = _cuda_integer_expr(body)
    kernel_params = _render_c_signature_params(
        [
            *(
                f"const {_cuda_type(tensors[name].dtype)}* __restrict__ {name}"
                for name in variables
            ),
            *(
                f"{_cuda_type(tensors[name].dtype)}* __restrict__ {name}"
                for name in outputs
            ),
            *(f"int64_t {axis}" for axis in shape_params),
        ]
    )
    launch_signature_params = _render_c_signature_params(
        [
            *(f"const {_cuda_type(tensors[name].dtype)}* {name}" for name in variables),
            *(f"{_cuda_type(tensors[name].dtype)}* {name}" for name in outputs),
            *(f"int64_t {axis}" for axis in shape_params),
            "cudaStream_t stream",
        ]
    )
    args = ", ".join((*variables, *outputs, *shape_params))

    return f"""// Generated by NineToothed's unified CUDA SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program

#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void {kernel.kernel_name}_kernel(
{kernel_params}
) {{
    int64_t {target.index_name} = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if ({target.index_name} < {total}) {{
{_indent_block(body, "        ")}
    }}
}}

extern "C" int launch_{kernel.kernel_name}(
{launch_signature_params}
) {{
    constexpr int threads = 256;
    int64_t blocks = ({total} + threads - 1) / threads;
    {kernel.kernel_name}_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        {args}
    );
    return static_cast<int>(cudaGetLastError());
}}
"""


def _render_c_signature_params(params: list[str]) -> str:
    return ",\n".join(f"    {param}" for param in params)


def _render_tilelang_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    total: str,
    body: str,
    tensors: Mapping[str, _TensorInfo],
    value_types: Mapping[str, ssa.Type],
) -> str:
    total = _rewrite_index_math(total, cuda=False)
    body = _rewrite_index_math(body, cuda=False)
    handle_args = ", ".join(
        [f"{name}: T.handle" for name in (*variables, *outputs)]
        + [f"{axis}: {_tile_param_dtype(axis, value_types)}" for axis in shape_params]
    )
    buffer_extents = {
        name: _rewrite_index_math(
            _product(_source_axes(tensors[name], fallback=(total,))), cuda=False
        )
        for name in (*variables, *outputs)
    }
    buffers = "\n".join(
        f"        {name}_buf = T.match_buffer({name}, ({buffer_extents[name]},), {_tile_dtype(tensors[name].dtype)})"
        for name in (*variables, *outputs)
    )

    return f'''"""TileLang lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
"""

from math import floor

try:
    import tilelang
    import tilelang.language as T
except ImportError:
    tilelang = None
    T = None


def build_{kernel.kernel_name}():
    if tilelang is None:
        raise ImportError("TileLang is required to build this backend artifact.")

    @T.prim_func
    def {kernel.kernel_name}({handle_args}):
{buffers}
        for block_id in T.thread_binding(({total} + 255) // 256, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                {target.index_name} = block_id * 256 + tx
                if {target.index_name} < {total}:
{_indent_block(body, "                    ")}

    return {kernel.kernel_name}
'''


def _render_tvm_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    total: str,
    body: str,
    tensors: Mapping[str, _TensorInfo],
    value_types: Mapping[str, ssa.Type],
) -> str:
    total = _rewrite_index_math(total, cuda=False)
    body = _rewrite_index_math(body, cuda=False)
    guard_total = _target_index_expr(target, total)
    block_extent = f"(({guard_total} + T.int64(255)) // T.int64(256))"
    linear_index = (
        f'T.Cast("int64", block_id) * T.int64({target.block_size}) '
        '+ T.Cast("int64", tx)'
    )
    handle_args = ", ".join(
        [f"{name}: T.handle" for name in (*variables, *outputs)]
        + [f"{axis}: {_tile_param_dtype(axis, value_types)}" for axis in shape_params]
    )
    dtype = _normalize_dtype(tensors[outputs[0]].dtype if outputs else "float32")
    dtype_var = f"_{dtype.replace('.', '_')}_dtype"
    buffer_extents = {
        name: _rewrite_index_math(
            _product(_source_axes(tensors[name], fallback=(total,))), cuda=False
        )
        for name in (*variables, *outputs)
    }
    buffers = "\n".join(
        f"            {name}_buf = T.match_buffer({name}, ({buffer_extents[name]},), {dtype_var})"
        for name in (*variables, *outputs)
    )

    return f'''"""TVMScript lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
"""

from math import floor

try:
    import tvm
except ImportError:
    import tilelang
    import tvm

try:
    from tvm.script import tirx as T
except ImportError:
    from tvm.script import tir as T

{dtype_var} = tvm.DataType("{dtype}")


def build_{kernel.kernel_name}():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def {kernel.kernel_name}({handle_args}):
            T.func_attr({{'global_symbol': 'main', 'tir.noalias': True}})
{buffers}
            for block_id in T.thread_binding({block_extent}, thread='blockIdx.x'):
                for tx in T.thread_binding(256, thread='threadIdx.x'):
                    {target.index_name} = {linear_index}
                    if {target.index_name} < {guard_total}:
{_indent_block(body, "                        ")}

    return Module
'''


def _walk_ops(operations: tuple[ssa.Operation, ...]):
    for op in operations:
        yield op

        for region in op.regions:
            yield from _walk_ops(region.operations)


def _atomic_output_tensors(
    operations: tuple[ssa.Operation, ...],
    op_by_result: Mapping[str, ssa.Operation],
) -> tuple[str, ...]:
    outputs: list[str] = []

    for op in operations:
        if op.opcode != "mem.atomic_add" or not op.operands:
            continue

        pointer = op_by_result.get(op.operands[0])

        if pointer is None or pointer.opcode != "mem.data_ptr" or not pointer.operands:
            continue

        tensor = pointer.operands[0]

        if tensor not in outputs:
            outputs.append(tensor)
    return tuple(outputs)


def _program_value_types(program: ssa.Program) -> dict[str, ssa.Type]:
    value_types = {
        value.name: value.type for value in (*program.inputs, *program.outputs)
    }

    for block in program.blocks:
        _collect_value_types(block, value_types)
    return value_types


def _collect_value_types(block: ssa.Block, value_types: dict[str, ssa.Type]) -> None:
    value_types.update({arg.name: arg.type for arg in block.args})

    for operation in block.operations:
        value_types.update({result.name: result.type for result in operation.results})

        for region in operation.regions:
            _collect_value_types(region, value_types)


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

    for tensor in tensors:
        if tensor.constexpr and tensor.ndim == 0 and tensor.name not in params:
            params.append(tensor.name)

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
                if symbol not in params:
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
    div = "/" if target.backend == Target.CUDA else "//"
    base = f"({index_expr} {div} ({stride}))"
    expr = base if dim == 0 else f"({base} % {axes[dim]})"

    return _target_index_expr(target, expr)


def _shape_dim(axes: tuple[str, ...], dim: Any) -> str:
    dim = int(dim or 0)

    if dim < 0:
        dim += len(axes)
    return axes[dim]


def _stride_dim(axes: tuple[str, ...], dim: Any) -> str:
    dim = int(dim or 0)

    if dim < 0:
        dim += len(axes)

    if dim >= len(axes):
        return "1"
    return _product(axes[dim + 1 :])


def _linearized_index(indices: tuple[str, ...], axes: tuple[str, ...]) -> str:
    if len(indices) == 1 and len(axes) <= 1:
        return indices[0]

    terms: list[str] = []

    for position, index in enumerate(indices):
        stride = _product(axes[position + 1 :])
        terms.append(f"({index}) * ({stride})" if stride != "1" else f"({index})")
    return " + ".join(terms) if terms else "index"


def _product(terms: tuple[str, ...]) -> str:
    items = tuple(str(term) for term in terms if str(term) not in {"", "1"})

    return " * ".join(_factor(item) for item in items) if items else "1"


def _factor(term: str) -> str:
    return term if _valid_symbol(term) or term.isdecimal() else f"({term})"


def _cuda_integer_expr(expr: str) -> str:
    previous = None
    current = _rewrite_index_math(expr, cuda=True)
    pattern = re.compile(r"floor\(\(([^()]+)\)/([A-Za-z_][A-Za-z0-9_]*)\)")

    while current != previous:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def _rewrite_index_math(expr: str, *, cuda: bool) -> str:
    previous = None
    current = expr

    while current != previous:
        previous = current
        current = _rewrite_named_call(
            current,
            "Mod",
            lambda args: f"(({args[0]}) % ({args[1]}))" if len(args) == 2 else None,
        )
        current = _rewrite_named_call(
            current,
            "floor",
            lambda args: (
                _rewrite_floor_arg(args[0], cuda=cuda) if len(args) == 1 else None
            ),
        )
    return current


def _rewrite_floor_arg(arg: str, *, cuda: bool) -> str:
    split = _split_top_level_binary(arg, "/")

    if split is None:
        return f"floor({arg})"

    lhs, rhs = split
    operator = "/" if cuda else "//"

    return f"(({lhs}) {operator} ({rhs}))"


def _rewrite_named_call(
    expr: str,
    name: str,
    render,
) -> str:
    result: list[str] = []
    cursor = 0
    prefix = f"{name}("

    while True:
        start = expr.find(prefix, cursor)

        if start < 0:
            result.append(expr[cursor:])
            break

        result.append(expr[cursor:start])
        args_start = start + len(prefix)
        args_end = _matching_paren(expr, args_start - 1)

        if args_end is None:
            result.append(expr[start:])
            cursor = len(expr)
            break

        args = _split_call_args(expr[args_start:args_end])
        rendered = render(args)

        if rendered is None:
            result.append(expr[start : args_end + 1])
        else:
            result.append(rendered)

        cursor = args_end + 1
    return "".join(result)


def _matching_paren(expr: str, open_index: int) -> int | None:
    depth = 0

    for index in range(open_index, len(expr)):
        char = expr[index]

        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

            if depth == 0:
                return index
    return None


def _split_call_args(args: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0

    for index, char in enumerate(args):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(args[start:index].strip())
            start = index + 1

    tail = args[start:].strip()

    if tail:
        parts.append(tail)
    return parts


def _split_top_level_binary(expr: str, operator: str) -> tuple[str, str] | None:
    depth = 0

    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == operator and depth == 0:
            return expr[:index].strip(), expr[index + 1 :].strip()
    return None


def _indent_lines(lines: list[str], target: _Target) -> list[str]:
    prefix = _indent_unit(target)

    return [prefix + line if line else line for line in lines]


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _indent_unit(target: _Target) -> str:
    return "    " if target.backend != Target.CUDA else "    "


def _if_header(condition: str, target: _Target) -> str:
    if target.backend == Target.CUDA:
        return f"if ({condition}) {{"
    return f"if {condition}:"


def _empty_block_lines(target: _Target) -> list[str]:
    return ["/* no-op */"] if target.backend == Target.CUDA else ["pass"]


def _zero_value(type_: ssa.Type, target: _Target) -> str:
    if type_.kind == "index" or _normalize_dtype(type_.dtype) in {
        "index",
        "int64",
        "int32",
    }:
        return "0"

    if _normalize_dtype(type_.dtype) == "bool":
        return "false" if target.backend == Target.CUDA else "False"
    return "0.0"


def _uses_mutable_scalar_slots(target: _Target) -> bool:
    return target.backend in {Target.TILELANG, Target.TVM}


def _mutable_scalar_decl_lines(
    target: _Target,
    type_: ssa.Type,
    name: str,
    init: str,
) -> list[str]:
    dtype = _normalize_dtype(type_.dtype)

    if target.backend == Target.TILELANG:
        return [f'{name} = T.alloc_var("{dtype}", {init})']

    if target.backend == Target.TVM:
        return [
            f'{name} = T.alloc_buffer((1,), "{dtype}", scope="local")',
            f"{name}[0] = {init}",
        ]
    return [target.local_decl(type_, name, init)]


def _mutable_scalar_read(target: _Target, name: str) -> str:
    if target.backend == Target.TVM:
        return f"{name}[0]"
    return name


def _assign_scalar(
    target: _Target, name: str, value: str, *, mutable: bool = False
) -> str:
    lhs = f"{name}[0]" if mutable and target.backend == Target.TVM else name

    return f"{lhs} = {value}" + (";" if target.backend == Target.CUDA else "")


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


def _valid_symbol(value: str) -> bool:
    return value.isidentifier()


def _symbols_in_text(value: str) -> tuple[str, ...]:
    return tuple(
        symbol
        for symbol in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", value)
        if symbol
        not in {
            "True",
            "False",
            "None",
            "index",
            "outer_index",
            "floor",
            "ceil",
            "ceiling",
            "Mod",
        }
        and not re.fullmatch(r"value_\d+", symbol)
        and not re.fullmatch(r"extract_\d+_\d+", symbol)
    )


def _normalize_dtype(dtype: str | None) -> str:
    if dtype is not None:
        dtype = dtype.strip().strip("'\"")

        if "." in dtype:
            dtype = dtype.split(".")[-1]

    mapping = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
        "float": "float32",
    }

    return mapping.get(dtype or "float32", dtype or "float32")


def _cuda_type(dtype: str | None, kind: str | None = None) -> str:
    dtype = _normalize_dtype(dtype)

    if kind == "pointer":
        return f"{_cuda_type(dtype)}*"

    if kind == "index" or dtype in {"index", "int64"}:
        return "int64_t"
    return {
        "float32": "float",
        "float16": "half",
        "float64": "double",
        "int32": "int32_t",
        "bool": "bool",
    }.get(dtype, "float")


def _tile_dtype(dtype: str | None) -> str:
    dtype = _normalize_dtype(dtype)

    return {
        "float32": "T.float32",
        "float16": "T.float16",
        "bfloat16": "T.bfloat16",
        "float64": "T.float64",
        "int32": "T.int32",
        "int64": "T.int64",
        "bool": "T.bool",
    }.get(dtype, "T.float32")


def _tile_param_dtype(name: str, value_types: Mapping[str, ssa.Type]) -> str:
    type_ = value_types.get(name)

    if type_ is not None and type_.kind == "scalar" and type_.dtype:
        if _normalize_dtype(type_.dtype) == "bool":
            return "T.int64"
        return _tile_dtype(type_.dtype)
    return "T.int64"


def _entrypoint(kernel: Kernel, backend: Target) -> str:
    if backend in {Target.TILELANG, Target.TVM}:
        return f"build_{kernel.kernel_name}"
    return f"launch_{kernel.kernel_name}"
