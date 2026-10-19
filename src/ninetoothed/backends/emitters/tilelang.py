"""TileLang syntax hooks for the common SSA emitter."""

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.backends.emitters.context import TensorInfo as _TensorInfo
from ninetoothed.ir import Kernel, ssa

_Target = EmitterTarget
_buffer_storage_extent = common.buffer_storage_extent
_cooperative_dot_plan = common.cooperative_dot_plan
_dot_accumulator_dtype = common.dot_accumulator_dtype
_emit_element = common.emit_element
_emit_loop_bound = common.emit_loop_bound
_emit_operation = common.emit_operation
_fresh_temp = common.fresh_temp
_indent_block = common.indent_block
_indent_lines = common.indent_lines
_linearized_index = common.linearized_index
_logical_ssa_audit = common.logical_ssa_audit
_nested_local_suffix = common.nested_local_suffix
_normalize_dtype = common.normalize_dtype
_product = common.product
_reduction_identity = common.reduction_identity
_resolved_dot_operand_dtype = common.resolved_dot_operand_dtype
_rewrite_index_math = common.rewrite_index_math
_target_index_expr = common.target_index_expr
_value_axes = common.value_axes
_value_axes_from_types = common.value_axes_from_types


_TIR_FUNCTIONS = {
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
}


@dataclass(frozen=True, kw_only=True)
class TileLangTarget(EmitterTarget):
    backend: Target = Target.TILELANG
    language: str = "python/tilelang"
    suffix: str = "tilelang.py"
    source_route: str = "ssa-unified-tilelang-emitter"
    buffer_suffix: str = "_buf"
    entrypoint_prefix: str = "build_"
    tir_value_semantics: bool = True
    scalar_global_access: bool = False
    _accessed_buffers: set[str] | None = field(default=None, compare=False, repr=False)

    def uses_mutable_scalar_slots(self) -> bool:
        return True

    def mutable_scalar_decl(self, type_: ssa.Type, name: str, init: str) -> list[str]:
        dtype = (
            f'"{common.normalize_dtype(type_.dtype)}"'
            if type_.dtype is not None
            else self.value_dtype(init)
        )

        return [f"{name} = T.alloc_local((1,), {dtype})", f"{name}[0] = {init}"]

    def mutable_scalar_read(self, name: str) -> str:
        return f"{name}[0]"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        destination = self.mutable_scalar_read(name) if mutable else name

        return f"{destination} = {value}"

    def literal(self, value: Any) -> str:
        if isinstance(value, float) and math.isinf(value):
            return "float('inf')" if value > 0 else "-float('inf')"

        if value == "inf":
            return "float('inf')"

        if value == "-inf":
            return "-float('inf')"
        return repr(value)

    def load(self, tensor, index, *, mask=None, other=0.0):
        del mask, other

        if self._accessed_buffers is not None:
            self._accessed_buffers.add(self.tensor_ref(tensor))

        reference = f"{self.tensor_ref(tensor)}[{index}]"

        if self.scalar_global_access:
            return (
                f'T.call_extern(_nt_dtype({reference}), "_nt_load", '
                f'{self.tensor_ref(tensor)}.access_ptr("r", offset={index}, extent=1))'
            )

        return reference

    def store(self, tensor, index, value, *, mask=None):
        if self._accessed_buffers is not None:
            self._accessed_buffers.add(self.tensor_ref(tensor))

        reference = f"{self.tensor_ref(tensor)}[{index}]"
        assignment = (
            f'T.evaluate(T.call_extern("int32", "_nt_store", '
            f'{self.tensor_ref(tensor)}.access_ptr("w", offset={index}, extent=1), '
            f"{self.cast_like(value, reference)}))"
            if self.scalar_global_access
            else f"{reference} = {value}"
        )

        return assignment if mask is None else f"if {mask}:\n    {assignment}"

    def cast(self, dtype, value):
        return f'T.Cast("{common.normalize_dtype(dtype)}", {value})'

    def cast_like(self, value, reference):
        return f"T.Cast({self.value_dtype(reference)}, {value})"

    def value_dtype(self, value):
        return f"_nt_dtype({value})"

    def reduction_identity(self, dtype, operator, shape):
        del shape

        return f"_nt_identity({dtype}, {operator!r})"

    def where(self, cond, yes, no):
        return f"T.if_then_else({cond}, {yes}, {no})"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            return f"T.atomic_add({args[0]}[0], {args[1]})"

        if name == "load" and args:
            return f"{args[0]}[0]"

        if name == "block_dot" and len(args) == 2:
            return f"T.dot({args[0]}, {args[1]})"

        if name == "rand":
            mixed = (
                f'T.Cast("uint32", T.bitwise_xor({args[0]}, {args[1]})) '
                "* 1664525 + 1013904223"
            )

            return f'T.Cast("float32", T.bitwise_and({mixed}, 16777215)) / 16777216.0'

        if name == "expm1":
            return f"({self.call('exp', args)} - 1.0)"

        function = _TIR_FUNCTIONS.get(name, f"T.{name}")

        return f"{function}({', '.join(args)})"

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        del type_

        return f"{name} = {expr}"

    def loop_header(self, var, lower, upper, step):
        serial = (
            f"T.serial({upper})"
            if lower == "0" and step == "1"
            else f"T.serial({lower}, {upper})"
            if step == "1"
            else f"T.serial({lower}, {upper}, {step})"
        )

        return f"for {var} in {serial}:"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "T.max" if operator == "max" else "T.min"

        return f"{function}({acc}, {term})"

    def supports_cooperative_reduction(self, schedule: Mapping[str, Any]) -> bool:
        if not schedule.get("tilelang_cooperative_reduction"):
            return False

        try:
            threads = int(schedule.get("threads", 256))
        except (TypeError, ValueError):
            return False
        return 32 <= threads <= 1024 and threads % 32 == 0

    def program_id(self, axis: int = 0) -> str:
        if axis != 0:
            raise ValueError(
                "TileLang cooperative reduction uses a one-dimensional grid."
            )
        return "nt_block_id"

    def thread_id(self) -> str:
        return "nt_thread_id"

    def thread_count(self) -> str:
        return "nt_num_threads"

    def emit_cooperative_reduction(self, local, operation, context):
        return _emit_tilelang_cooperative_reduction(local, operation, context)

    def render_module(self, context: ModuleRenderContext) -> str:
        flash = _render_tilelang_flash_module(
            context.kernel,
            self,
            context.variables,
            context.outputs,
            context.shape_params,
            context.tensors,
            context.value_types,
            context.operations,
            context.stores,
            context.outer_axes,
        )

        if flash is not None:
            return flash

        cooperative = _render_tilelang_cooperative_dot_module(
            context.kernel,
            self,
            context.variables,
            context.outputs,
            context.shape_params,
            context.tensors,
            context.value_types,
            context.operations,
            context.stores,
            context.outer_axes,
        )

        if cooperative is not None:
            return cooperative
        return _render_tilelang_module(
            context.kernel,
            self,
            context.variables,
            context.outputs,
            context.shape_params,
            context.total,
            context.body,
            context.tensors,
            context.value_types,
            grid_total=(
                context.grid_total
                if (
                    context.kernel.ssa.metadata.get("schedule", {})
                    .get("row_group", {})
                    .get("enabled")
                    or context.cooperative_reduction_program
                )
                else None
            ),
            cooperative_reduction_program=context.cooperative_reduction_program,
        )


class _FlashPlan:
    """SSA operations participating in one online-softmax (flash) loop."""

    def __init__(
        self, loop, q_scaled, k_extract, v_extract, scores, probs, out, is_causal
    ):
        self.loop = loop
        self.q_scaled = q_scaled
        self.k_extract = k_extract
        self.v_extract = v_extract
        self.scores = scores
        self.probs = probs
        self.out = out
        self.is_causal = is_causal


def _flash_plan(
    operations: Mapping[str, ssa.Operation],
    stores: tuple[ssa.Operation, ...],
    value_types: Mapping[str, ssa.Type],
) -> _FlashPlan | None:
    """Pattern-match the canonical online-softmax attention loop.

    Matches an ``scf.for`` carrying ``(acc, m_i, l_i)`` iter args whose body
    computes ``scores = dot(q, k^T)``, an online ``reduce.max``/``reduce.sum``
    softmax over the score axis, and accumulates ``dot(p, v)`` into ``acc``.
    """
    loops = list(
        {
            id(op): op
            for op in operations.values()
            if op.opcode == "scf.for" and len(op.results) == 3 and len(op.regions) == 1
        }.values()
    )

    if len(loops) != 1:
        return None

    loop = loops[0]
    region = loop.regions[0]
    dots = list(
        {
            id(op): op
            for op in region.operations
            if op.opcode in {"linalg.dot", "linalg.matmul"} and len(op.operands) >= 2
        }.values()
    )

    if len(dots) != 2:
        return None

    reduce_max = next(
        (op for op in region.operations if op.opcode == "reduce.max"), None
    )
    reduce_sum = next(
        (op for op in region.operations if op.opcode == "reduce.sum"), None
    )

    if reduce_max is None or reduce_sum is None:
        return None

    scores_dot, pv_dot = dots[0], dots[1]

    # Compute scores = dot(q_scaled, trans(k_extract)); pv = dot(probs, v_extract).
    q_scaled = scores_dot.operands[0]
    k_rhs = scores_dot.operands[1]
    k_extract = _unwrap_to_extract(k_rhs, operations)
    v_extract = _unwrap_to_extract(pv_dot.operands[1], operations)

    if k_extract is None or v_extract is None:
        return None

    scores = reduce_max.operands[0]
    probs = reduce_sum.operands[0]

    # Identify the output store target of the accumulator loop result.
    out = None

    for op in stores:
        if op.operands and _value_depends_on_any(
            op.operands[0], {loop.results[0].name}, operations
        ):
            out = op.operands[1]

    if out is None:
        return None

    # Causal kernels carry an extra row>=col `select.where` mask (a `cmp.ge` /
    # `cmp.gt` comparison) on top of the in-bounds guard.  The mask may be
    # nested inside an `scf.if` (a runtime/constexpr `is_causal` guard), so the
    # region is walked recursively.
    def has_causal_cmp(block):
        for op in block.operations:
            if op.opcode in {"cmp.ge", "cmp.gt"}:
                return True

            for nested in op.regions:
                if has_causal_cmp(nested):
                    return True
        return False

    is_causal = has_causal_cmp(region)

    return _FlashPlan(
        loop, q_scaled, k_extract, v_extract, scores, probs, out, is_causal
    )


def _unwrap_to_extract(name, operations):
    current = name
    seen = set()

    while current not in seen:
        seen.add(current)
        op = operations.get(current)

        if op is None:
            return None

        if op.opcode == "tensor.extract":
            return op

        if (
            op.opcode in {"linalg.transpose", "tensor.view", "tensor.cast"}
            and op.operands
        ):
            current = op.operands[0]
            continue
        return None
    return None


def _value_depends_on_any(name, targets, operations) -> bool:
    if name in targets:
        return True

    seen = set()
    stack = [name]

    while stack:
        current = stack.pop()

        if current in targets:
            return True

        if current in seen:
            continue

        seen.add(current)
        op = operations.get(current)

        if op is not None:
            stack.extend(op.operands)
    return False


def _flash_source_4d(name, ctx):
    traced = _dot_source_tensor(name, ctx)

    if traced is None:
        return None

    src_name, info = traced
    shape = tuple(str(dim) for dim in (info.source_shape or info.shape) if str(dim))

    if len(shape) != 4:
        return None
    return src_name, shape


def _render_tilelang_flash_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    tensors: Mapping[str, _TensorInfo],
    value_types: Mapping[str, ssa.Type],
    operations: Mapping[str, ssa.Operation],
    stores: tuple[ssa.Operation, ...],
    outer_axes: tuple[str, ...],
) -> str | None:
    plan = _flash_plan(operations, stores, value_types)

    if plan is None or kernel.ssa is None:
        return None

    probe_lines: list[str] = []
    linear_block = (
        f"bz * ({outer_axes[1]} * {outer_axes[2]}) + by * {outer_axes[2]} + bx"
        if len(outer_axes) >= 3
        else "bx"
    )
    probe_ctx = _EmitContext(
        target=target,
        kernel=kernel,
        program=kernel.ssa,
        operations=operations,
        value_types=value_types,
        lines=probe_lines,
        memo={},
        tensor_infos=tensors,
        output=plan.out,
        output_axes=(),
        index_expr="0",
        outer_index_expr=linear_block,
        inner_index_expr="0",
        mask_expr=None,
        coordinate_exprs=(),
        bindings={},
        temp_counter=[0],
        materialized={},
        indent="",
    )

    k_src = _flash_source_4d(plan.k_extract.operands[0], probe_ctx)
    v_src = _flash_source_4d(plan.v_extract.operands[0], probe_ctx)
    o_src = _flash_source_4d(plan.out, probe_ctx)

    if not all((k_src, v_src, o_src)):
        return None

    k_name, k_shape = k_src
    v_name, _ = v_src
    o_name, _ = o_src
    seq_kv = k_shape[2]

    # Tile extents: scores are (bm, bn); head dim is q's last dim.
    scores_type = value_types.get(plan.scores)
    q_type = value_types.get(plan.q_scaled)

    if scores_type is None or q_type is None or len(q_type.shape) < 2:
        return None

    bm, bn = (str(dim) for dim in scores_type.shape[:2])
    head_dim = str(q_type.shape[-1])
    batch, heads, seq_tiles = (
        (str(axis) for axis in outer_axes[:3])
        if len(outer_axes) >= 3
        else (None, None, None)
    )

    if batch is None:
        return None

    threads = common.schedule_int(kernel, "threads", 256)
    num_stages = common.schedule_int(kernel, "num_stages", 2)
    q_dtype = _tile_dtype(tensors[k_name].dtype)
    kv_dtype = _tile_dtype(tensors[k_name].dtype)

    loop = plan.loop
    lower = _emit_loop_bound(loop.operands[0], probe_ctx)
    step = _emit_loop_bound(loop.operands[2], probe_ctx)

    if lower != "0" or step != "1":
        return None

    # Element-wise Q fill (scaled, cast to the GEMM operand dtype) via the SSA value.
    gemm_operand_dtype = _normalize_dtype(tensors[k_name].dtype)
    q_lines: list[str] = []
    q_ctx = probe_ctx.child(lines=q_lines, coordinate_exprs=("mi", "ni"))
    q_value = _emit_element(plan.q_scaled, ("mi", "ni"), q_ctx)
    q_lines.append(f'Q_shared[mi, ni] = T.Cast("{gemm_operand_dtype}", {q_value})')

    if plan.is_causal:
        score_init = f"""                for i, j in T.Parallel({bm}, {bn}):
                    acc_s[i, j] = T.if_then_else(bx * {bm} + i >= ko * {bn} + j, T.Cast(T.float32, 0), -T.infinity(T.float32))"""
    else:
        score_init = "                T.clear(acc_s)"

    # Region copies for K / V / O.
    def region_lines(src_name, shared, ko):
        corner = f"bz, by, {ko} * {bn}, 0"

        return f"T.copy({src_name}_tile[{corner}], {shared})"

    _, buffer_names, handle_args = _tilelang_signature(
        variables, outputs, shape_params, tensors, value_types
    )
    buffers = _tilelang_flat_buffers(buffer_names, tensors, _product(outer_axes))
    tile_decls = "\n".join(
        f"        {src}_tile = T.view({src}_buf, ({', '.join(shape)},))"
        for src, shape in ((k_name, k_src[1]), (v_name, v_src[1]), (o_name, o_src[1]))
    )
    q_fill = _indent_block("\n".join(q_lines), "                ")

    flash_body = f"""        with T.Kernel({seq_tiles}, {heads}, {batch}, threads={threads}) as (bx, by, bz):
            Q_shared = T.alloc_shared(({bm}, {head_dim}), {q_dtype})
            K_shared = T.alloc_shared(({bn}, {head_dim}), {kv_dtype})
            V_shared = T.alloc_shared(({bn}, {head_dim}), {kv_dtype})
            acc_s = T.alloc_fragment(({bm}, {bn}), T.float32)
            acc_o = T.alloc_fragment(({bm}, {head_dim}), T.float32)
            scores_max = T.alloc_fragment(({bm},), T.float32)
            scores_max_prev = T.alloc_fragment(({bm},), T.float32)
            scores_scale = T.alloc_fragment(({bm},), T.float32)
            scores_sum = T.alloc_fragment(({bm},), T.float32)
            logsum = T.alloc_fragment(({bm},), T.float32)
            p_cast = T.alloc_fragment(({bm}, {bn}), {kv_dtype})
            for mi, ni in T.Parallel({bm}, {head_dim}):
{q_fill}
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(T.float32))
            for ko in T.Pipelined(T.ceildiv({seq_kv}, {bn}), num_stages={num_stages}):
                {region_lines(k_name, "K_shared", "ko")}
{score_init}
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(T.float32))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel({bm}):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel({bm}):
                    scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel({bm}, {bn}):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel({bm}):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, p_cast)
                for i, j in T.Parallel({bm}, {head_dim}):
                    acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                {region_lines(v_name, "V_shared", "ko")}
                T.gemm(p_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel({bm}, {head_dim}):
                acc_o[i, j] = acc_o[i, j] / logsum[i]
            T.copy(acc_o, {o_name}_tile[bz, by, bx * {bm}, 0])"""

    return _render_tilelang_module_skeleton(
        kernel,
        target,
        note="online-softmax dot loop -> fused flash attention",
        handle_args=handle_args,
        buffers=buffers,
        body=flash_body,
        extra_decls=("\n" + tile_decls) if tile_decls else "",
    )


def _render_tilelang_cooperative_dot_module(
    kernel: Kernel,
    target: _Target,
    variables: tuple[str, ...],
    outputs: tuple[str, ...],
    shape_params: tuple[str, ...],
    tensors: Mapping[str, _TensorInfo],
    value_types: Mapping[str, ssa.Type],
    operations: Mapping[str, ssa.Operation],
    stores: tuple[ssa.Operation, ...],
    outer_axes: tuple[str, ...],
) -> str | None:
    plan = _cooperative_dot_plan(operations, stores, value_types)

    if plan is None or kernel.ssa is None:
        return None

    threads = common.schedule_int(kernel, "threads", 128)
    num_stages = common.schedule_int(kernel, "num_stages", 2)
    dot = plan.dot
    loop = plan.loop
    lhs, rhs = dot.operands[:2]
    lhs_axes = _value_axes_from_types(lhs, value_types)
    result_axes = tuple(str(dim) for dim in dot.results[0].type.shape)
    bm, bn = result_axes
    bk = lhs_axes[-1]
    output = plan.store.operands[1]
    _, buffer_names, handle_args = _tilelang_signature(
        variables, outputs, shape_params, tensors, value_types
    )

    def context(
        lines: list[str],
        *,
        coordinates: tuple[str, ...],
        bindings: Mapping[str, str] | None = None,
        access_target: _Target | None = None,
    ) -> _EmitContext:
        inner_index = _linearized_index(coordinates, result_axes)

        return _EmitContext(
            target=target if access_target is None else access_target,
            kernel=kernel,
            program=kernel.ssa,  # type: ignore[arg-type]
            operations=operations,
            value_types=value_types,
            lines=lines,
            memo={},
            tensor_infos=tensors,
            output=output,
            output_axes=result_axes,
            index_expr=inner_index,
            outer_index_expr="outer_index",
            inner_index_expr=inner_index,
            mask_expr=None,
            row_expr=coordinates[0],
            col_expr=coordinates[1],
            coordinate_exprs=coordinates,
            bindings={} if bindings is None else bindings,
            temp_counter=[0],
            materialized={},
            indent="",
        )

    bound_lines: list[str] = []
    bound_ctx = context(bound_lines, coordinates=("0", "0"))
    lower = _emit_loop_bound(loop.operands[0], bound_ctx)
    upper = _emit_loop_bound(loop.operands[1], bound_ctx)
    step = _emit_loop_bound(loop.operands[2], bound_ctx)

    if lower != "0" or step != "1":
        return None

    induction = str(loop.attrs.get("induction", "%iv"))
    loop_bindings = {induction: "ko"}
    probe_ctx = context([], coordinates=("0", "0"), bindings=loop_bindings)
    lhs_region = _dot_tile_region(lhs, probe_ctx, tile_shape=(bm, bk), loop_var="ko")
    rhs_region = _dot_tile_region(rhs, probe_ctx, tile_shape=(bk, bn), loop_var="ko")
    fills_copy = lhs_region is not None and rhs_region is not None

    lhs_lines: list[str] = []
    rhs_lines: list[str] = []
    tile_buffer_decls: list[str] = []

    if fills_copy:
        for region, shared, target_lines in (
            (lhs_region, "a_shared", lhs_lines),
            (rhs_region, "b_shared", rhs_lines),
        ):
            src_name, source_shape, strides, corners = region
            tile_buffer_decls.append(
                f"        {src_name}_tile = T.view({src_name}_buf, ({', '.join(source_shape)},))"
            )
            target_lines.append(
                f"T.copy({src_name}_tile[{', '.join(corners)}], {shared})"
            )
    else:
        lhs_ctx = context(
            lhs_lines,
            coordinates=("mi", "ki"),
            bindings=loop_bindings,
        )
        lhs_value = _emit_element(lhs, ("mi", "ki"), lhs_ctx)
        lhs_lines.append(f"a_shared[mi, ki] = {lhs_value}")
        rhs_ctx = context(
            rhs_lines,
            coordinates=("ki", "ni"),
            bindings=loop_bindings,
        )
        rhs_value = _emit_element(rhs, ("ki", "ni"), rhs_ctx)
        rhs_lines.append(f"b_shared[ki, ni] = {rhs_value}")

    output_lines: list[str] = []
    output_region = _dot_tile_region(
        output, probe_ctx, tile_shape=(bm, bn), loop_var="ko"
    )
    output_copy = (
        fills_copy
        and output_region is not None
        and _is_pure_loop_store(plan.store, loop, operations)
    )

    if output_copy:
        output_ctx = context(
            [],
            coordinates=("mi", "ni"),
            bindings={loop.results[0].name: "c_local[mi, ni]"},
        )
        src_name, source_shape, _, corners = output_region
        tile_buffer_decls.append(
            f"        {src_name}_tile = T.view({src_name}_buf, ({', '.join(source_shape)},))"
        )
        output_lines.append(f"T.copy(c_local, {src_name}_tile[{', '.join(corners)}])")
    else:
        output_ctx = context(
            output_lines,
            coordinates=("mi", "ni"),
            bindings={loop.results[0].name: "c_local[mi, ni]"},
        )
        _emit_operation(plan.store, output_ctx)

    lhs_storage_dtype = _resolved_dot_operand_dtype(lhs, output_ctx)
    rhs_storage_dtype = _resolved_dot_operand_dtype(rhs, output_ctx)
    supported_storage_dtypes = {
        "float16",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
        "float32",
    }

    if (
        lhs_storage_dtype not in supported_storage_dtypes
        or rhs_storage_dtype not in supported_storage_dtypes
    ):
        return None

    lhs_dtype = _tile_dtype(lhs_storage_dtype)
    rhs_dtype = _tile_dtype(rhs_storage_dtype)
    accumulator_dtype = _tile_dtype(_dot_accumulator_dtype(dot, output_ctx))
    grid_total = _rewrite_index_math(_product(outer_axes), c_style=False)

    buffers = _tilelang_flat_buffers(buffer_names, tensors, _product(outer_axes))

    bound_source = (
        "\n" + _indent_block("\n".join(bound_lines), "            ")
        if bound_lines
        else ""
    )
    tile_buffers = "\n".join(tile_buffer_decls)
    tile_buffers = ("\n" + tile_buffers) if tile_buffers else ""
    output_source = _indent_block("\n".join(output_lines), "                ")

    if fills_copy:
        fill_source = _indent_block(
            "\n".join((*lhs_lines, *rhs_lines)), "                "
        )
        pipeline_source = f"""            for ko in T.Pipelined({upper}, num_stages={num_stages}):
{fill_source}
                T.gemm(a_shared, b_shared, c_local)"""
    else:
        lhs_source = _indent_block("\n".join(lhs_lines), "                    ")
        rhs_source = _indent_block("\n".join(rhs_lines), "                    ")
        pipeline_source = f"""            for ko in T.Pipelined({upper}, num_stages={num_stages}):
                for mi, ki in T.Parallel({bm}, {bk}):
{lhs_source}
                for ki, ni in T.Parallel({bk}, {bn}):
{rhs_source}
                T.gemm(a_shared, b_shared, c_local)"""

    if output_copy:
        store_source = _indent_block("\n".join(output_lines), "            ")
    else:
        store_source = f"""            for mi, ni in T.Parallel({bm}, {bn}):
{output_source}"""

    dot_body = f"""        with T.Kernel({grid_total}, threads={threads}) as outer_index:{bound_source}
            a_shared = T.alloc_shared(({bm}, {bk}), {lhs_dtype})
            b_shared = T.alloc_shared(({bk}, {bn}), {rhs_dtype})
            c_local = T.alloc_fragment(({bm}, {bn}), {accumulator_dtype})
            T.clear(c_local)
{pipeline_source}
{store_source}"""

    return _render_tilelang_module_skeleton(
        kernel,
        target,
        note="cooperative linalg.dot -> T.gemm",
        handle_args=handle_args,
        buffers=buffers,
        body=dot_body,
        extra_decls=tile_buffers,
    )


def _tilelang_signature(variables, outputs, shape_params, tensors, value_types):
    """Build the prim_func parameter list and the flat-buffer name list.

    Shared by the generic, cooperative-dot, and flash TileLang renderers.
    """
    parameter_names = (*variables, *outputs)
    buffer_names = tuple(name for name in parameter_names if tensors[name].ndim != 0)
    handle_args = ", ".join(
        [
            f"{name}: T.handle"
            if tensors[name].ndim != 0
            else f"{name}: {_tile_scalar_abi_dtype(tensors[name].dtype)}"
            for name in parameter_names
        ]
        + [f"{axis}: {_tile_param_dtype(axis, value_types)}" for axis in shape_params]
    )

    return parameter_names, buffer_names, handle_args


def _tilelang_flat_buffers(buffer_names, tensors, fallback_total):
    """Emit the 1D ``T.match_buffer`` declarations for the given tensors."""
    extents = {
        name: _rewrite_index_math(
            _buffer_storage_extent(tensors[name], fallback=fallback_total),
            c_style=False,
        )
        for name in buffer_names
    }

    return "\n".join(
        f"        {name}_buf = T.match_buffer({name}, ({extents[name]},), {_tile_dtype(tensors[name].dtype)}, align=1)"
        for name in buffer_names
    )


def _render_tilelang_module_skeleton(
    kernel: Kernel,
    target: _Target,
    *,
    note: str,
    handle_args: str,
    buffers: str,
    body: str,
    extra_decls: str = "",
) -> str:
    """Emit the shared TileLang module skeleton (docstring/imports/build wrapper).

    ``body`` is the fully-indented kernel launch body; ``extra_decls`` are any
    additional buffer declarations (e.g. shaped tile views) and must carry
    their own leading newline when non-empty.
    """
    schedule_note = f"\nSchedule: {note}" if note else ""

    return f'''"""TileLang lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}{schedule_note}
"""

{_logical_ssa_audit(kernel, target)}

from math import floor

try:
    import tilelang
    import tilelang.language as T
except ImportError:
    tilelang = None
    T = None


{_TYPE_IDENTITY}

_NT_SCALAR_ACCESS = """
__device__ __forceinline__ bool _nt_aligned(const void* pointer) {{
    return (reinterpret_cast<unsigned long long>(pointer) & 63) == 0;
}}
template <typename V>
__device__ __forceinline__ V _nt_load(const V* pointer) {{ return *pointer; }}
template <typename V, typename W>
__device__ __forceinline__ int _nt_store(V* pointer, W value) {{
    *pointer = value;
    return 0;
}}
"""

def build_{kernel.kernel_name}():
    if tilelang is None:
        raise ImportError("TileLang is required to build this backend artifact.")

    @T.prim_func
    def {kernel.kernel_name}({handle_args}):
{buffers}{extra_decls}
{body}

    return {kernel.kernel_name}
'''


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
    *,
    grid_total: str | None = None,
    cooperative_reduction_program: bool = False,
) -> str:
    threads = common.schedule_int(kernel, "threads", 256)
    total = _rewrite_index_math(total, c_style=False)
    body = _rewrite_index_math(body, c_style=False)

    if grid_total is not None:
        grid_total = _rewrite_index_math(grid_total, c_style=False)

    _, buffer_names, handle_args = _tilelang_signature(
        variables, outputs, shape_params, tensors, value_types
    )
    buffers = _tilelang_flat_buffers(buffer_names, tensors, total)

    if cooperative_reduction_program:
        launch = f"""        with T.Kernel({grid_total}, threads={threads}) as nt_block_id:
            nt_thread_id = T.get_thread_binding(0)
            nt_num_threads = T.get_thread_extent(0)
{_indent_block(body, "            ")}"""
    elif grid_total is not None:
        launch = f"""        for {target.program_id(0)} in T.thread_binding({grid_total}, thread="blockIdx.x"):
            for tx in T.thread_binding({threads}, thread="threadIdx.x"):
                for tile_offset in T.serial(({total} + {threads - 1}) // {threads}):
                    {target.index_name} = tile_offset * {threads} + tx
                    if {target.index_name} < {total}:
{_indent_block(body, "                        ")}"""
    else:
        tile = _elementwise_tile_extent(threads)
        launch = f"""        with T.Kernel(({total} + {tile - 1}) // {tile}, threads={threads}) as nt_block_id:
            for nt_lane in T.Parallel({tile}):
                {target.index_name} = nt_block_id * {tile} + nt_lane
                if {target.index_name} < {total}:
{_indent_block(body, "                    ")}"""

    return _render_tilelang_module_skeleton(
        kernel, target, note=None, handle_args=handle_args, buffers=buffers, body=launch
    )


TARGET = TileLangTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["TARGET", "TileLangTarget", "emit"]


def _base_tensors(value: str, ctx: _EmitContext, _seen: set | None = None) -> set:
    """Return the set of source tensors a value transitively reads."""
    if _seen is None:
        _seen = set()

    if value in _seen:
        return set()

    _seen.add(value)

    if value in ctx.tensor_infos:
        return {value}

    op = ctx.operations.get(value)

    if op is None or not op.operands:
        return set()

    result = set()

    for operand in op.operands:
        result |= _base_tensors(operand, ctx, _seen)

    return result


def _emit_tilelang_cooperative_reduction(
    local: str, operation: ssa.Operation, ctx: _EmitContext
) -> str:
    """Lower a scheduled row reduction to a TileLang block reduction.

    The reduction domain is staged into a shared-memory buffer with a
    ``T.Parallel`` loop (which TileLang distributes across the block's
    threads), then reduced with the idiomatic ``T.reduce_*`` primitive, which
    TileLang lowers to an efficient shuffle/shared-memory tree reduction.
    """
    schedule = dict(ctx.reduction_schedule or {})
    axis = int(schedule["axis"]) if "axis" in schedule else None

    if axis is None or not operation.operands or not operation.results:
        raise ValueError("Malformed TileLang cooperative reduction operation.")

    operator = operation.opcode[len("reduce.") :]
    operand = operation.operands[0]
    operand_axes = _value_axes(operand, ctx)
    extent = str(schedule.get("extent", operand_axes[axis]))
    operand_type = ctx.value_types.get(operand)
    result_dtype = _normalize_dtype(operation.results[0].type.dtype or "float32")
    operand_dtype = _normalize_dtype(
        operand_type.dtype if operand_type is not None else result_dtype
    )
    accumulator_dtype = (
        "float32"
        if operand_dtype in {"float8_e4m3fn", "float8_e5m2", "float16", "bfloat16"}
        else result_dtype
    )
    reduction_index = _fresh_temp(ctx, "nt_reduce_index")
    staged = _fresh_temp(ctx, "nt_reduce_staged")
    reducer = _fresh_temp(ctx, "nt_reduce_acc")
    identity = _reduction_identity(
        operator, ssa.Type(kind="scalar", dtype=accumulator_dtype), ctx.target
    )
    reduce_function = {
        "sum": "T.reduce_sum",
        "max": "T.reduce_max",
        "min": "T.reduce_min",
    }.get(operator)

    if reduce_function is None:
        raise ValueError(
            f"Unsupported TileLang cooperative reduction operator: {operator!r}."
        )

    coordinates = list(ctx.coordinate_exprs)
    coordinates[axis] = reduction_index
    coordinates = tuple(coordinates)
    linear = _target_index_expr(
        ctx.target, _linearized_index(coordinates, operand_axes)
    )

    # Reduce-then-broadcast single-pass: if the reduction operand reads a
    # single source tensor that is also read by a downstream consumer (e.g.
    # rmsnorm's ``input / sqrt(mean(input^2))``), stage that source row into
    # shared memory once so both the reduction and the consumer read it from
    # shared instead of re-reading global memory.
    stash_source = None
    stash_dtype = None
    stash_buffer = None

    if extent.isdigit():
        candidate_bases = _base_tensors(operand, ctx)

        if len(candidate_bases) == 1:
            candidate = next(iter(candidate_bases))

            if ctx.shared_stash and candidate in ctx.shared_stash:
                # A previous reduction already staged this source row; reuse it.
                stash_source = candidate
            else:
                uses = sum(
                    1 for op in ctx.operations.values() if candidate in op.operands
                )
                candidate_info = ctx.tensor_infos.get(candidate)
                candidate_dtype = _normalize_dtype(
                    candidate_info.dtype if candidate_info is not None else "float16"
                )
                dtype_bytes = 2 if candidate_dtype in {"float16", "bfloat16"} else 4

                if uses >= 2 and int(extent) * (dtype_bytes + 4) <= 48 * 1024:
                    stash_source = candidate
                    stash_dtype = candidate_dtype
                    stash_buffer = _fresh_temp(ctx, "nt_reduce_src")

    if stash_source is not None and ctx.shared_stash is None:
        ctx.shared_stash = {}

    body_lines: list[str] = []
    body = ctx.child(
        lines=body_lines,
        memo=dict(ctx.memo),
        coordinate_exprs=coordinates,
        index_expr=linear,
        inner_index_expr=linear,
        reduce_axis=axis,
        reduce_index=reduction_index,
        mask_expr=None,
        local_suffix=_nested_local_suffix(ctx, staged),
    )

    if stash_buffer is not None:
        body_lines.append(
            f"{stash_buffer}[{reduction_index}] = "
            f"{_emit_element(stash_source, coordinates, body)}"
        )
        ctx.shared_stash[stash_source] = (stash_buffer, axis)

        # Drop any memoized global read of the source so that a downstream
        # consumer re-reads it from the staged shared buffer instead.
        ctx.memo.pop(stash_source, None)

    term = _emit_element(operand, coordinates, body)
    term = ctx.target.cast(accumulator_dtype, term)
    body_lines.append(f"{staged}[{reduction_index}] = {term}")

    if stash_buffer is not None:
        ctx.lines.append(
            f"{stash_buffer} = T.alloc_shared(({extent},), {_tile_dtype(stash_dtype)})"
        )

    ctx.lines.append(
        f"{staged} = T.alloc_shared(({extent},), {_tile_dtype(accumulator_dtype)})"
    )
    ctx.lines.append(f"for {reduction_index} in T.Parallel({extent}):")
    ctx.lines.extend(_indent_lines(body_lines, ctx.target))
    ctx.lines.append(
        f"{reducer} = T.alloc_fragment((1,), {_tile_dtype(accumulator_dtype)})"
    )
    ctx.lines.append(f"{reducer}[0] = {identity}")
    ctx.lines.append(f"{reduce_function}({staged}, {reducer}, dim=0, clear=False)")
    ctx.lines.append(
        ctx.target.local_decl(
            operation.results[0].type,
            local,
            ctx.target.cast(result_dtype, f"{reducer}[0]"),
        )
    )

    return local


def _elementwise_tile_extent(threads: int) -> int:
    """Return the per-block element count for a vectorized elementwise launch.

    Each block covers ``threads * elements_per_thread`` contiguous elements so
    that TileLang splits the inner ``T.Parallel`` loop into a serial x thread x
    vector partition (vectorized global loads) instead of one element per
    thread.  Eight elements per thread keeps a 16-byte vector access for
    16-bit dtypes and two for 32-bit dtypes, which matches hand-written
    TileLang elementwise kernels.
    """
    try:
        threads = int(threads)
    except (TypeError, ValueError):
        threads = 128

    return threads * 8


def _dot_source_tensor(name: str, ctx: _EmitContext):
    """Trace an SSA value through views/extracts/casts to its source tensor."""
    current = name
    seen = set()

    while current not in ctx.tensor_infos and current not in seen:
        seen.add(current)
        op = ctx.operations.get(current)

        if op is None or not op.operands:
            return None

        if op.opcode not in {
            "tensor.extract",
            "tensor.view",
            "tensor.cast",
            "mem.data_ptr",
            "linalg.transpose",
        }:
            return None

        current = op.operands[0]

    if current not in ctx.tensor_infos:
        return None
    return current, ctx.tensor_infos[current]


def _affine_coeff(expr: str, var: str) -> str | None:
    """Return the constant coefficient of ``var`` in an affine expression."""
    try:
        import sympy
    except ImportError:
        return None

    try:
        symbol = sympy.Symbol(var)
        expanded = sympy.expand(sympy.sympify(expr))
        coeff = expanded.coeff(symbol)

        if coeff.has(symbol):
            return None

        remainder = expanded - coeff * symbol

        if remainder.has(symbol):
            return None
        return str(coeff)
    except (SyntaxError, TypeError, ValueError, AttributeError):
        return None


def _index_exprs_equal(left: str, right: str) -> bool:
    """Return whether two index expressions are numerically equal."""
    if left == right:
        return True

    try:
        import sympy

        difference = sympy.simplify(sympy.sympify(left) - sympy.sympify(right))

        return difference == 0
    except (SyntaxError, TypeError, ValueError, AttributeError):
        return False


def _dot_tile_region(
    name: str,
    ctx: _EmitContext,
    *,
    tile_shape: tuple[str, ...],
    loop_var: str,
):
    """Reconstruct a shaped buffer region for a cooperative dot operand tile.

    Returns ``(source_name, source_shape, strides, corners)`` when the operand
    tile maps affinely (unit coefficient) onto a region of its source tensor,
    or ``None`` when it does not (e.g. gathered/im2col accesses), in which case
    the caller falls back to element-wise shared-memory fills.
    """
    traced = _dot_source_tensor(name, ctx)

    if traced is None:
        return None

    src_name, info = traced
    templates = (info.attrs or {}).get("access_templates", ())
    template = None

    for candidate in templates:
        shape = tuple(str(dim) for dim in candidate.get("shape", ()))

        if shape == tile_shape:
            template = candidate
            break

    if template is None:
        return None

    offsets = tuple(str(offset) for offset in template.get("offsets", ()))
    n_outer = len(offsets) - len(tile_shape)

    if n_outer < 0:
        return None

    source_shape = tuple(
        str(dim) for dim in (info.source_shape or info.shape) if str(dim)
    )
    strides = common.source_strides(info)

    if len(source_shape) != len(offsets) or len(strides) != len(offsets):
        return None

    default_strides = common.default_strides(source_shape)

    if not all(
        _index_exprs_equal(stride, default)
        for stride, default in zip(strides, default_strides)
    ):
        return None

    # Outer (e.g. batch) dims must not vary with the tile coordinates.
    for dim in range(n_outer):
        for value_dim in range(len(tile_shape)):
            if _affine_coeff(offsets[dim], f"value_{value_dim}") != "0":
                return None

    # Each tile dim must appear with unit coefficient in its own source dim
    # and zero coefficient in the others (a plain, contiguous tile region).
    for index in range(len(tile_shape)):
        dim = n_outer + index

        for value_dim in range(len(tile_shape)):
            expected = "1" if value_dim == index else "0"

            if _affine_coeff(offsets[dim], f"value_{value_dim}") != expected:
                return None

    corners = []

    for offset in offsets:
        corner = offset

        for value_dim in range(len(tile_shape)):
            corner = common.replace_symbols(corner, {f"value_{value_dim}": "0"})

        corner = common.replace_symbols(corner, {"extract_0_0": loop_var})
        corner = common.rewrite_index_math(corner, c_style=False)
        corners.append(corner)

    return src_name, source_shape, strides, tuple(corners)


def _is_pure_loop_store(
    store: ssa.Operation,
    loop: ssa.Operation,
    operations: Mapping[str, ssa.Operation],
) -> bool:
    """Return whether a store simply writes the loop accumulator (or a cast)."""
    if not store.operands or not loop.results:
        return False

    target = loop.results[0].name
    current = store.operands[0]

    for _ in range(8):
        if current == target:
            return True

        op = operations.get(current)

        if op is None or op.opcode != "tensor.cast" or len(op.operands) != 1:
            return False

        current = op.operands[0]
    return False


def _tile_dtype(dtype: str | None) -> str:
    dtype = common.normalize_dtype(dtype)
    types = {
        "float32": "T.float32",
        "float16": "T.float16",
        "bfloat16": "T.bfloat16",
        "float8_e4m3fn": "T.float8_e4m3fn",
        "float8_e5m2": "T.float8_e5m2",
        "float64": "T.float64",
        "int8": "T.int8",
        "uint8": "T.uint8",
        "int16": "T.int16",
        "uint16": "T.uint16",
        "int32": "T.int32",
        "uint32": "T.uint32",
        "int64": "T.int64",
        "uint64": "T.uint64",
        "bool": "T.bool",
    }

    if dtype not in types:
        raise ValueError(f"Unsupported TileLang SSA dtype: {dtype!r}.")
    return types[dtype]


def _tile_param_dtype(name: str, value_types: Mapping[str, ssa.Type]) -> str:
    type_ = value_types.get(name)

    if type_ is not None and type_.kind == "scalar" and type_.dtype:
        if common.normalize_dtype(type_.dtype) == "bool":
            return "T.int64"
        return _tile_scalar_abi_dtype(type_.dtype)
    return "T.int64"


def _tile_scalar_abi_dtype(dtype: str | None) -> str:
    dtype = common.normalize_dtype(dtype)

    if dtype in {"float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}:
        return "T.float32"
    return _tile_dtype(dtype)


_TYPE_IDENTITY = """

def _nt_dtype(value):
    return value.dtype if hasattr(value, "dtype") else T.const(value).dtype


def _nt_identity(dtype, operator):
    if operator == "sum":
        if dtype.startswith(("int", "uint")) and int(dtype.lstrip("uint")) < 32:
            dtype = "int32"
        return T.Cast(dtype, 0)
    if dtype.startswith(("float", "bfloat")) and dtype != "float8_e4m3fn":
        return T.Cast(dtype, float("-inf" if operator == "max" else "inf"))
    return T.min_value(dtype) if operator == "max" else T.max_value(dtype)
"""
