"""CUDA syntax hooks for the common SSA emitter."""

import math
import re
from dataclasses import dataclass, replace
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import walk_ops
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.c_types import CComputedTypesMixin
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.backends.emitters.task_launch import allocate_private_symbol
from ninetoothed.ir import Kernel, ssa

_current_coords = common.current_coords
_emit_element = common.emit_element
_emit_loop_bound = common.emit_loop_bound
_fresh_temp = common.fresh_temp
_indent_lines = common.indent_lines
_linearized_index = common.linearized_index
_local_symbol = common.local_symbol
_nested_local_suffix = common.nested_local_suffix
_normalize_dtype = common.normalize_dtype
_reduction_identity = common.reduction_identity
_reduction_input_mask = common.reduction_input_mask
_source_index_for_value = common.source_index_for_value
_target_index_expr = common.target_index_expr
_value_axes = common.value_axes


@dataclass(frozen=True, kw_only=True)
class CudaTarget(CComputedTypesMixin, EmitterTarget):
    backend: Target = Target.CUDA
    language: str = "cuda/c++"
    suffix: str = "cu"
    source_route: str = "ssa-unified-cuda-emitter"
    c_style_syntax: bool = True
    native_block_matmul: bool = True

    def supports_native_block(self, kernel) -> bool:
        operations = tuple(
            operation
            for block in kernel.ssa.blocks
            for operation in walk_ops(block.operations)
        )

        return (
            self.native_block_matmul
            and sum(
                operation.opcode in {"linalg.dot", "linalg.matmul"}
                for operation in operations
            )
            == 1
            and not any(
                operation.opcode.startswith("reduce.") for operation in operations
            )
        )

    def native_block_coords(self, axes):
        if len(axes) != 2:
            raise ValueError("CUDA WMMA blocks require a rank-two coordinate domain.")
        return "nt_matrix_row", "nt_matrix_col"

    def native_block_tile(self, kernel, axes):
        from ninetoothed.backends.emitters.cuda_cta_dot import selected

        return (32, 32) if selected(kernel) else super().native_block_tile(kernel, axes)

    def index_cast(self, value: str) -> str:
        return f"static_cast<int64_t>({value})"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value};"

    def literal(self, value: Any) -> str:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == -(1 << 63)
        ):
            return "(-9223372036854775807LL - 1LL)"

        if isinstance(value, bool):
            return "true" if value else "false"

        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return "NAN"

        if isinstance(value, float) and math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"

        if value == "inf":
            return "INFINITY"

        if value == "-inf":
            return "-INFINITY"
        return repr(value)

    def cast(self, dtype, value):
        return f"static_cast<{self.type_name(dtype)}>({value})"

    def inline_constant(self, value, type_):
        return f"static_cast<{self.type_name(type_.dtype, type_.kind)}>({self.literal(value)})"

    def cast_like(self, value, reference):
        return f"static_cast<{self.value_dtype(reference)}>({value})"

    def true_divide(self, lhs, rhs):
        return f"{self.global_helper('_nt_true_divide')}({lhs}, {rhs})"

    def reduction_identity(self, dtype, operator, shape):
        del shape

        if operator == "sum":
            dtype = (
                f"std::conditional<std::is_integral<{dtype}>::value && "
                f"(sizeof({dtype}) < 4), int32_t, {dtype}>::type"
            )

            return f"static_cast<{dtype}>(0)"

        bound = "low" if operator == "max" else "high"

        return f"{self.global_helper('_nt_limits')}<{dtype}>::{bound}"

    def type_name(self, dtype, kind=None):
        return _cuda_type(dtype, kind)

    def where(self, cond, yes, no):
        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            return f"atomicAdd({args[0]}, {args[1]})"

        if name == "load" and args:
            return f"*({args[0]})"

        if name in {"block_dot", "dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "rand" and len(args) >= 2:
            return f"{self.global_helper('ninetoothed_curand_uniform')}(({args[0]}), ({args[1]}))"

        functions = {
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
            "maximum": "nt_max",
            "max": "nt_max",
            "minimum": "nt_min",
            "min": "nt_min",
            "pow": "powf",
            "rsqrt": "rsqrtf",
            "sin": "sinf",
            "sinh": "sinhf",
            "sqrt": "sqrtf",
            "tan": "tanf",
            "tanh": "tanhf",
        }
        function = functions.get(name, name)

        if function in {"nt_max", "nt_min"}:
            function = self.global_helper(function)

        return f"{function}({', '.join(args)})"

    def supports_cooperative_reduction(self, schedule):
        try:
            threads = int(schedule.get("threads", 256))
        except (TypeError, ValueError):
            return False

        return bool(
            schedule.get("cuda_cooperative_reduction")
            and 32 <= threads <= 256
            and threads % 32 == 0
        )

    def program_id(self, axis=0):
        if axis != 0:
            raise ValueError("CUDA cooperative reduction uses a one-dimensional grid.")
        return "static_cast<int64_t>(blockIdx.x)"

    def thread_id(self):
        return "static_cast<int64_t>(threadIdx.x)"

    def thread_count(self):
        return "static_cast<int64_t>(blockDim.x)"

    def emit_cooperative_reduction(self, local, operation, context):
        return _emit_cuda_cooperative_reduction(local, operation, context)

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "nt_max" if operator == "max" else "nt_min"

        return f"{self.global_helper(function)}({acc}, {term})"

    def reduce_update_typed(self, operator, acc, term, dtype):
        if dtype is None:
            term = self.cast_like(term, acc)

            if operator == "sum":
                return f"({acc} + {term})"
            return f"{self.global_helper('_nt_reduce_' + operator)}({acc}, {term})"
        return _cuda_reduction_update(
            operator,
            acc,
            term,
            _cuda_reduction_accumulator_dtype(dtype, dtype),
            target=self,
        )

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        return _cuda_arithmetic_result_type(operation, context)

    def coerce_binary_args(self, operation, args, context):
        return _coerce_cuda_binary_args(operation, args, context)

    def emit_dot_operand(self, name, coords, context):
        return _emit_cuda_direct_dot_operand(name, coords, context)

    def emit_block_dot(self, operation, context, coords=None):
        return _emit_cuda_wmma_dot(operation, context, coords=coords)

    def emit_reduction_loop(self, local, operation, context):
        if not context.native_block_program:
            return None

        return _emit_cuda_wmma_reduction_loop(local, operation, context)

    def schedule_context(self, context):
        if not self.optimizes_native_execution(context.kernel):
            context.lowering_plans.update(row=None, reduce_broadcast=None, uniform=None)

            return context

        from ninetoothed.backends.emitters.cuda_cta_dot import CTA_DOT_PLAN, selected
        from ninetoothed.backends.emitters.cuda_uniform import render_uniform_store
        from ninetoothed.backends.emitters.patterns import (
            match_reduce_broadcast,
            match_row_reduce,
        )

        context.lowering_plans.update(
            row=match_row_reduce(context, allow_direct=True),
            reduce_broadcast=match_reduce_broadcast(context, allow_direct=True),
        )

        if selected(context.kernel):
            if not context.collective_body:
                raise ValueError(
                    "CUDA CTA dot candidate did not consume a verified SSA loop."
                )

            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "cuda_cta_dot": {
                        "selected": True,
                        "output_tile": (CTA_DOT_PLAN.rows, CTA_DOT_PLAN.columns),
                        "warps": CTA_DOT_PLAN.warps,
                        "shared_memory_bytes": CTA_DOT_PLAN.arena_bytes,
                        "input_slots": CTA_DOT_PLAN.input_slots,
                        "input_pitches": (
                            CTA_DOT_PLAN.left_pitch,
                            CTA_DOT_PLAN.right_pitch,
                        ),
                        "arena_lifetime": "input-readers-finish-before-output-overlay",
                        "fallback": "original-indexed-synchronous-cooperative-packets",
                        "compile_target_guard": "async-only-when-__CUDA_ARCH__>=800",
                        "output_ownership": "launch-guarded-disjoint-positive-stride-spans",
                    },
                },
            )

        if re.search(
            r"wmma::fragment<wmma::accumulator, 16, 16, 16, float> \w+_wmma_reduction_acc;",
            context.collective_body + context.body,
        ):
            accumulation = _wmma_accumulation_mode(context.kernel)
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "cuda_wmma_accumulation": {
                        "selected": True,
                        "mode": accumulation,
                        "input_dtype": "float16",
                        "accumulator_dtype": "float32",
                        "rounding": (
                            "continuous-wmma-across-source-K-groups"
                            if accumulation == "fused"
                            else "per-source-dot-partial-then-ordered-f32-add-rn"
                        ),
                        "source_dot_add_groups_preserved": accumulation == "ordered",
                        "region_contract": "zero-initialized-single-carry-exact-extract-extract-dot-add-yield",
                    },
                },
            )

        if "_dense_wmma" in context.collective_body + context.body:
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "cuda_wmma_direct_load": {
                        "selected": True,
                        "proof": "complete-original-tile-maps-and-predicates",
                        "dtype": "float16",
                        "fallback": "original-indexed-staging",
                        "guard": "input-dense-strides",
                    },
                },
            )

        uniform = render_uniform_store(context)
        context.lowering_plans["uniform"] = uniform

        if uniform is None:
            return context
        return replace(
            context,
            scheduled_metadata={
                **context.scheduled_metadata,
                "cuda_uniform_store": {
                    "selected": True,
                    "dtype": "float32",
                    "vector_elements": 4,
                    "max_blocks": 512,
                    "max_output_elements": 16777216,
                    "ownership": "disjoint-grid-stride-vectors-and-scalar-remainder",
                    "fallback": "original-indexed-ssa",
                },
            },
        )

    def _render_legacy_matrix(self, context, body, threads, grid_total):
        kernel = context.kernel

        if threads != 256:
            raise ValueError(
                "CUDA WMMA lowering currently requires exactly 256 threads."
            )

        schedule = kernel.ssa.metadata.get("schedule", {}) if kernel.ssa else {}
        mma_shape = schedule.get("mma_shape", {"m": 16, "n": 16, "k": 16})
        mma_m = int(mma_shape.get("m", 16))
        mma_n = int(mma_shape.get("n", 16))
        mma_k = int(mma_shape.get("k", 16))

        if (mma_m, mma_n, mma_k) != (16, 16, 16):
            raise ValueError(
                "The current CUDA WMMA intrinsic lowering supports only "
                "a 16x16x16 fragment; use a CTA tiling pass to compose fragments."
            )

        tile_rows = _cuda_integer_expr(f"(({context.axes[0]}) + {mma_m - 1}) / {mma_m}")
        tile_cols = _cuda_integer_expr(f"(({context.axes[1]}) + {mma_n - 1}) / {mma_n}")
        kernel_prelude = f"""    int64_t nt_tile_rows = {tile_rows};
    int64_t nt_tile_cols = {tile_cols};
    int64_t nt_tiles_per_outer = nt_tile_rows * nt_tile_cols;
    int64_t nt_outer_index = static_cast<int64_t>(blockIdx.x) / nt_tiles_per_outer;
    int64_t nt_tile_index = static_cast<int64_t>(blockIdx.x) % nt_tiles_per_outer;
    int64_t nt_tile_row = (nt_tile_index / nt_tile_cols) * {mma_m};
    int64_t nt_tile_col = (nt_tile_index % nt_tile_cols) * {mma_n};
    int64_t nt_matrix_row = nt_tile_row + static_cast<int64_t>(threadIdx.x) / {mma_n};
    int64_t nt_matrix_col = nt_tile_col + static_cast<int64_t>(threadIdx.x) % {mma_n};
    int64_t {self.index_name} = nt_matrix_row * ({context.axes[1]}) + nt_matrix_col;
    bool nt_matrix_active = nt_matrix_row < ({context.axes[0]}) && nt_matrix_col < ({context.axes[1]});
{common.indent_block(body, "    ")}"""
        blocks_expr = grid_total

        return kernel_prelude, blocks_expr

    def render_module(self, context: ModuleRenderContext) -> str:
        kernel = context.kernel
        threads = common.schedule_int(kernel, "threads", 256)
        total = _cuda_integer_expr(context.total)
        grid_total = _cuda_integer_expr(context.grid_total)
        body = _cuda_integer_expr(context.body)
        stream_name = allocate_private_symbol(
            "stream", *context.variables, *context.outputs, *context.shape_params
        )
        kernel_params = _render_signature_params(
            [
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=True, restrict=True
                    )
                    for name in context.variables
                ),
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=False, restrict=True
                    )
                    for name in context.outputs
                ),
                *(
                    _auxiliary_param(axis, context.tensors)
                    for axis in context.shape_params
                ),
            ]
        )
        launch_params = _render_signature_params(
            [
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=True, restrict=True
                    )
                    for name in context.variables
                ),
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=False, restrict=True
                    )
                    for name in context.outputs
                ),
                *(
                    _auxiliary_param(axis, context.tensors)
                    for axis in context.shape_params
                ),
                f"cudaStream_t {stream_name}",
            ]
        )
        args = ", ".join((*context.variables, *context.outputs, *context.shape_params))

        uniform = context.lowering_plans["uniform"]

        if uniform is not None:
            generic = f"""    int64_t {self.index_name} = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if ({self.index_name} < {total}) {{
{common.indent_block(body, "        ")}
    }}"""
            kernel_prelude = f"""    if ({uniform.guard}) {{
{common.indent_block(uniform.body, "    ")}
    }} else {{
{common.indent_block(generic, "    ")}
    }}"""
            blocks_expr = f"({uniform.guard}) ? ({uniform.blocks}) : (({total} + threads - 1) / threads)"
        elif context.block_program:
            from ninetoothed.backends.emitters.cuda_cta_dot import selected

            if selected(kernel):
                if threads != 128 or kernel.compiler_options.get("num_warps") not in (
                    None,
                    4,
                ):
                    raise ValueError("CUDA CTA dot requires exactly four warps.")

                tile_m, tile_n = self.native_block_tile(kernel, context.axes)
                tile_rows = _cuda_integer_expr(
                    f"(({context.axes[0]}) + {tile_m - 1}) / {tile_m}"
                )
                tile_cols = _cuda_integer_expr(
                    f"(({context.axes[1]}) + {tile_n - 1}) / {tile_n}"
                )
                prefix = _cuda_integer_expr(context.collective_body)
                kernel_prelude = f"""    int64_t nt_tile_rows = {tile_rows};
    int64_t nt_tile_cols = {tile_cols};
    int64_t nt_tiles_per_outer = nt_tile_rows * nt_tile_cols;
    int64_t nt_outer_index = static_cast<int64_t>(blockIdx.x) / nt_tiles_per_outer;
    int64_t nt_tile_index = static_cast<int64_t>(blockIdx.x) % nt_tiles_per_outer;
    int64_t nt_tile_row = (nt_tile_index / nt_tile_cols) * {tile_m};
    int64_t nt_tile_col = (nt_tile_index % nt_tile_cols) * {tile_n};
{common.indent_block(prefix, "    ")}
    for (int nt_cta_item = 0; nt_cta_item < 8; ++nt_cta_item) {{
        int nt_cta_element = threadIdx.x % 32 + nt_cta_item * 32;
        int64_t nt_matrix_row = nt_tile_row + (threadIdx.x / 32 / 2) * 16 + nt_cta_element / 16;
        int64_t nt_matrix_col = nt_tile_col + (threadIdx.x / 32 % 2) * 16 + nt_cta_element % 16;
        int64_t {self.index_name} = nt_matrix_row * ({context.axes[1]}) + nt_matrix_col;
        bool nt_matrix_active = nt_matrix_row < ({context.axes[0]}) && nt_matrix_col < ({context.axes[1]});
{common.indent_block(body, "        ")}
    }}"""
                blocks_expr = grid_total
            else:
                kernel_prelude, blocks_expr = self._render_legacy_matrix(
                    context, body, threads, grid_total
                )
        elif context.cooperative_reduction_program:
            group = kernel.ssa.metadata.get("schedule", {}).get("row_group", {})

            if group.get("enabled") and group.get("mapping") == "cooperative-loop":
                body = (
                    f"for (int64_t nt_group_row = 0; nt_group_row < {group['rows']}; "
                    f"++nt_group_row) {{\n"
                    + common.indent_block(body, "    ")
                    + "\n    __syncthreads();\n}"
                )

                kernel_prelude = body
                blocks_expr = grid_total
            else:
                reduce_match = context.lowering_plans["reduce_broadcast"]
                row_match = context.lowering_plans["row"]
                chosen_match = None

                if (
                    reduce_match is not None
                    and threads % 32 == 0
                    and 32 <= threads <= 1024
                ):
                    operator, x_name, out_name, rows, cols, scale = reduce_match
                    kernel_prelude = _render_shared_reduce_broadcast(
                        operator,
                        x_name,
                        out_name,
                        cols,
                        scale,
                        context,
                        broadcast_zero=reduce_match.broadcast_zero,
                    )
                    blocks_expr = rows
                    chosen_match = reduce_match
                elif (
                    row_match is not None
                    and threads % 32 == 0
                    and 32 <= threads <= 1024
                ):
                    operator, x_name, out_name, rows, cols, scale = row_match
                    kernel_prelude = _render_row_reduce(
                        operator,
                        x_name,
                        out_name,
                        cols,
                        scale,
                        context,
                        broadcast_zero=row_match.broadcast_zero,
                    )
                    blocks_expr = rows
                    chosen_match = row_match
                else:
                    kernel_prelude = body
                    blocks_expr = grid_total

                if chosen_match is not None:
                    guard = _row_memory_guard(
                        context, chosen_match, collapsed=chosen_match is row_match
                    )
                    kernel_prelude = f"""    if ({guard}) {{
{common.indent_block(kernel_prelude, "    ")}
    }} else {{
{common.indent_block(body, "        ")}
    }}"""
                    blocks_expr = f"({guard}) ? ({rows}) : ({grid_total})"
        else:
            fallback_row = None

            if context.cooperative_reduction_program is False and context.outputs:
                fallback_row = context.lowering_plans["row"]

            if fallback_row is not None and threads % 32 == 0 and 32 <= threads <= 1024:
                operator, x_name, out_name, rows, cols, scale = fallback_row
                kernel_prelude = _render_row_reduce(
                    operator,
                    x_name,
                    out_name,
                    cols,
                    scale,
                    context,
                    broadcast_zero=fallback_row.broadcast_zero,
                )
                blocks_expr = rows
                guard = _row_memory_guard(context, fallback_row, collapsed=True)
                kernel_prelude = f"""    if ({guard}) {{
{common.indent_block(kernel_prelude, "    ")}
    }} else {{
        int64_t {self.index_name} = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if ({self.index_name} < {total}) {{
{common.indent_block(body, "            ")}
        }}
    }}"""
                blocks_expr = (
                    f"({guard}) ? ({rows}) : (({total} + threads - 1) / threads)"
                )
            else:
                from ninetoothed.backends.emitters.staging import (
                    flatten_tiled_accesses,
                    split_stride_guard,
                    staging_extent,
                    staging_extent_guard,
                )

                guard_pred, guarded_body = split_stride_guard(body)
                flat_body = None

                if self.optimizes_native_execution(kernel) and guard_pred is not None:
                    flat_body = flatten_tiled_accesses(guarded_body, context)

                extent_expr = staging_extent(
                    context,
                    [(n, n) for n in (*context.variables, *context.outputs)],
                )

                if flat_body is not None and extent_expr is not None:
                    extent_c = _cuda_integer_expr(extent_expr)
                    flat_names = [
                        n
                        for n in (*context.variables, *context.outputs)
                        if re.search(rf"\b{re.escape(n)}\[", flat_body)
                    ]

                    eligible = (
                        len(context.outputs) == 1
                        and bool(flat_names)
                        and flat_names[-1] == context.outputs[0]
                        and all(
                            context.tensors[n].dtype == "float32" for n in flat_names
                        )
                    )
                    vectorized = (
                        _render_flat_vectorized(
                            flat_body, extent_c, flat_names, index_name=self.index_name
                        )
                        if eligible
                        else None
                    )

                    if vectorized is not None:
                        shapes = staging_extent_guard(
                            context, [(n, n) for n in flat_names]
                        )
                        alignment = " && ".join(
                            f"((uintptr_t){n} % alignof(float4) == 0)"
                            for n in flat_names
                        )
                        fast_guard = _cuda_integer_expr(
                            f"({guard_pred}) && ({shapes}) && ({alignment})"
                        )
                        generic = f"""    int64_t {self.index_name} = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if ({self.index_name} < {total}) {{
{common.indent_block(body, "        ")}
    }}"""
                        kernel_prelude = f"""    if ({fast_guard}) {{
{common.indent_block(vectorized, "    ")}
    }} else {{
{common.indent_block(generic, "    ")}
    }}"""
                        blocks_expr = f"({fast_guard}) ? (({extent_c} + threads * 16 - 1) / (threads * 16)) : (({total} + threads - 1) / threads)"
                    else:
                        kernel_prelude = f"""    int64_t {self.index_name} = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if ({self.index_name} < {total}) {{
{common.indent_block(body, "        ")}
    }}"""
                        blocks_expr = f"({total} + threads - 1) / threads"
                else:
                    kernel_prelude = f"""    int64_t {self.index_name} = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if ({self.index_name} < {total}) {{
{common.indent_block(body, "        ")}
    }}"""
                    blocks_expr = f"({total} + threads - 1) / threads"

        curand_header, curand_support = _curand_support(context)
        from ninetoothed.backends.emitters.cuda_cta_dot import render_launch_guard

        cta_launch_guard = render_launch_guard(context, self)

        return f"""// Generated by NineToothed's CUDA SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program

#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
#include <type_traits>
#include <limits>{curand_header}

using namespace nvcuda;

{_cuda_extremum_support(context)}

{curand_support}
{_ARITHMETIC_SUPPORT if "_nt_" in body else ""}

extern "C" __global__ void {kernel.kernel_name}_kernel(
{kernel_params}
) {{
{kernel_prelude}
}}

extern "C" int launch_{kernel.kernel_name}(
{launch_params}
) {{
{cta_launch_guard}    constexpr int threads = {threads};
    int64_t blocks = {blocks_expr};
    if (blocks <= 0) {{
        return static_cast<int>({self.global_helper("cudaSuccess")});
    }}
    {kernel.kernel_name}_kernel<<<static_cast<unsigned int>(blocks), threads, 0, {stream_name}>>>(
        {args}
    );
    return static_cast<int>({self.global_helper("cudaGetLastError")}());
}}
"""


def _row_memory_guard(context, pattern, *, collapsed):
    """Keep row intrinsics behind the source layout's runtime stride contract."""
    terms = []

    for name, width in (
        (pattern.input, pattern.cols),
        (pattern.output, "1" if collapsed else pattern.cols),
    ):
        info = context.tensors[name]
        strides = (info.attrs or {}).get("source_strides") or info.source_strides

        if not strides:
            continue

        if len(strides) != 2:
            return "false"

        for stride, expected in zip(strides, (width, "1")):
            if str(stride).isdigit():
                if str(stride) != str(expected):
                    terms.append(f"({stride}) == ({expected})")
            elif str(stride) in context.shape_params:
                terms.append(f"({stride}) == ({expected})")
            else:
                return "false"

    return _cuda_integer_expr(" && ".join(terms) or "true")


def _render_shared_reduce_broadcast(
    operator, x_name, out_name, cols, scale, context, broadcast_zero=None
):
    return _render_row_reduction(
        operator,
        x_name,
        out_name,
        cols,
        scale,
        context,
        broadcast=True,
        broadcast_zero=broadcast_zero,
    )


def _render_flat_vectorized(
    flat_body: str, extent: str, names, *, index_name="index"
) -> str | None:
    """Emit an explicit ``float4`` vectorized flat elementwise body.

    Every tensor is addressed as ``name[index]`` in the flattened body;
    each access maps onto a ``float4`` component so one thread moves
    128 bits per tensor, with a scalar tail covering the remainder.
    """
    import re

    if not names:
        return None

    residue = flat_body
    accesses = {name: re.compile(rf"\b{re.escape(name)}\[index\]") for name in names}

    for name in names:
        residue = accesses[name].sub("", residue)

    if re.search(r"\bindex\b", residue):
        return None

    comp = ("x", "y", "z", "w")
    groups = 4
    copies = []

    for g in range(groups):
        group_lines = []

        for k in range(4):
            body_k = flat_body

            for name in names:
                body_k = accesses[name].sub(f"nt4_{name}_{g}.{comp[k]}", body_k)

            if "[(nt_vi + " in body_k and re.search(r"\w+\[\(nt_vi \+ \d\)\]", body_k):
                return None

            group_lines.append(
                "{" + chr(10) + common.indent_block(body_k, "    ") + chr(10) + "}"
            )

        copies.append(chr(10).join(group_lines))

    loads = []

    for g in range(groups):
        loads.append(
            chr(10).join(
                [
                    *(
                        f"        float4 nt4_{name}_{g} = "
                        f"*reinterpret_cast<const float4*>({name} + nt_vi + {g * 4});"
                        for name in names[:-1]
                    ),
                    f"        float4 nt4_{names[-1]}_{g};",
                ]
            )
        )

    stores = chr(10).join(
        f"        *reinterpret_cast<float4*>({names[-1]} + nt_vi + {g * 4}) = "
        f"nt4_{names[-1]}_{g};"
        for g in range(groups)
    )
    span = groups * 4
    tail_body = flat_body

    for name in names:
        tail_body = accesses[name].sub(f"{name}[{index_name}]", tail_body)

    return f"""    const int64_t nt_extent = {extent};
    int64_t nt_vi = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * {span};
    if (nt_vi + {span} <= nt_extent) {{
{chr(10).join(loads)}
{chr(10).join(common.indent_block(c, "        ") for c in copies)}
{stores}
    }} else {{
        for (int nt_vk = 0; nt_vk < {span}; nt_vk++) {{
            int64_t {index_name} = nt_vi + nt_vk;
            if ({index_name} < nt_extent) {{
{common.indent_block(tail_body, "                ")}
            }}
        }}
    }}"""


def _render_row_reduce(
    operator, x_name, out_name, cols, scale, context, broadcast_zero=None
):
    return _render_row_reduction(
        operator,
        x_name,
        out_name,
        cols,
        scale,
        context,
        broadcast=False,
        broadcast_zero=broadcast_zero,
    )


def _render_row_reduction(
    operator, x_name, out_name, cols, scale, context, *, broadcast, broadcast_zero
):
    """Reduce the full row before applying scalar transforms and output terms."""
    neutral = (
        "-INFINITY"
        if operator == "max"
        else "INFINITY"
        if operator == "min"
        else "0.0f"
    )
    lines = [
        f"    const int64_t nt_cols = {cols};",
        f"    const float* nt_row_in = {x_name} + (int64_t)blockIdx.x * nt_cols;",
        f"    float* nt_row_out = {out_name} + (int64_t)blockIdx.x * {'nt_cols' if broadcast else '1'};",
        "    __shared__ float nt_red[32];",
        "    int64_t nt_tid = threadIdx.x;",
        f"    float nt_acc = {neutral};",
        "    for (int64_t nt_j = nt_tid; nt_j < nt_cols; nt_j += blockDim.x) {",
        "        float nt_value = nt_row_in[nt_j];",
    ]

    def combine(left, right):
        if operator in {"sum", "sum_sq"}:
            return f"({left}) + ({right})"

        fn = "fmaxf" if operator == "max" else "fminf"

        return f"(isnan({left}) || isnan({right})) ? NAN : {fn}({left}, {right})"

    term = "nt_value * nt_value" if operator == "sum_sq" else "nt_value"
    lines.extend(
        [
            f"        nt_acc = {combine('nt_acc', term)};",
            "    }",
            "    for (int offset = 16; offset > 0; offset >>= 1) {",
            "        float nt_peer = __shfl_down_sync(0xffffffff, nt_acc, offset);",
            f"        nt_acc = {combine('nt_acc', 'nt_peer')};",
            "    }",
            "    if ((nt_tid & 31) == 0) { nt_red[nt_tid >> 5] = nt_acc; }",
            "    __syncthreads();",
            "    if (nt_tid < 32) {",
            f"        float nt_warp_acc = (nt_tid < (blockDim.x + 31) / 32) ? nt_red[nt_tid] : {neutral};",
            "        for (int offset = 16; offset > 0; offset >>= 1) {",
            "            float nt_peer = __shfl_down_sync(0xffffffff, nt_warp_acc, offset);",
            f"            nt_warp_acc = {combine('nt_warp_acc', 'nt_peer')};",
            "        }",
            "        if (nt_tid == 0) {",
        ]
    )
    result = "sqrtf(nt_warp_acc)" if operator == "sum_sq" else "nt_warp_acc"
    lines.extend(
        [
            f"            nt_red[0] = ({result}){scale};",
            "        }",
            "    }",
            "    __syncthreads();",
        ]
    )

    if broadcast:
        result = "nt_red[0] + nt_row_in[nt_j] * 0.0f" if broadcast_zero else "nt_red[0]"
        lines.extend(
            [
                "    for (int64_t nt_j = nt_tid; nt_j < nt_cols; nt_j += blockDim.x) {",
                f"        nt_row_out[nt_j] = {result};",
                "    }",
            ]
        )
    else:
        result = "nt_red[0] + nt_row_in[0] * 0.0f" if broadcast_zero else "nt_red[0]"
        lines.append(f"    if (nt_tid == 0) {{ nt_row_out[0] = {result}; }}")

    return chr(10).join(lines)


def _emit_cuda_cooperative_reduction(
    local: str, operation: ssa.Operation, ctx: _EmitContext
) -> str:
    schedule = dict(ctx.reduction_schedule or {})
    axis = int(schedule["axis"]) if "axis" in schedule else None

    if axis is None or not operation.operands or not operation.results:
        raise ValueError("Malformed CUDA cooperative reduction operation.")

    operator = operation.opcode[len("reduce.") :]
    operand = operation.operands[0]
    operand_axes = _value_axes(operand, ctx)
    extent = str(schedule.get("extent", operand_axes[axis]))
    operand_type = ctx.value_types.get(operand)
    result_dtype = _normalize_dtype(operation.results[0].type.dtype or "float32")
    operand_dtype = _normalize_dtype(
        operand_type.dtype if operand_type is not None else result_dtype
    )
    accumulator_dtype = _cuda_reduction_accumulator_dtype(operand_dtype, result_dtype)
    accumulator_type = ssa.Type(kind="scalar", dtype=accumulator_dtype)
    accumulator = _fresh_temp(ctx, "nt_reduce_acc")
    reduction_index = _fresh_temp(ctx, "nt_reduce_index")
    shuffle_offset = _fresh_temp(ctx, "nt_reduce_offset")
    shared = _fresh_temp(ctx, "nt_reduce_shared")
    identity = _reduction_identity(operator, accumulator_type, ctx.target)
    coordinates = list(ctx.coordinate_exprs)
    coordinates[axis] = reduction_index
    coordinates = tuple(coordinates)
    linear = _target_index_expr(
        ctx.target, _linearized_index(coordinates, operand_axes)
    )
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
        local_suffix=_nested_local_suffix(ctx, accumulator),
    )
    term = _emit_element(operand, coordinates, body)
    term = ctx.target.cast(accumulator_dtype, term)

    if ctx.program.metadata.get("schedule", {}).get("row_group", {}).get("enabled"):
        mask = _reduction_input_mask(operand, coordinates, body)

        if mask is not None:
            term = ctx.target.where(mask, term, identity)

    body_lines.append(
        f"{accumulator} = "
        f"{_cuda_reduction_update(operator, accumulator, term, accumulator_dtype, target=ctx.target)};"
    )

    ctx.lines.append(ctx.target.local_decl(accumulator_type, accumulator, identity))
    ctx.lines.append(
        ctx.target.loop_header(
            reduction_index,
            ctx.target.thread_id(),
            extent,
            ctx.target.thread_count(),
        )
    )
    ctx.lines.extend(_indent_lines(body_lines, ctx.target))
    ctx.lines.append("}")
    ctx.lines.append(
        f"for (int {shuffle_offset} = 16; {shuffle_offset} > 0; "
        f"{shuffle_offset} >>= 1) {{"
    )
    shuffle = f"__shfl_down_sync(0xffffffffu, {accumulator}, {shuffle_offset})"
    ctx.lines.append(
        f"    {accumulator} = "
        f"{_cuda_reduction_update(operator, accumulator, shuffle, accumulator_dtype, target=ctx.target)};"
    )
    ctx.lines.append("}")
    ctx.lines.append(
        f"__shared__ {ctx.target.type_name(accumulator_dtype)} {shared}[8];"
    )
    ctx.lines.append(
        f"if ((threadIdx.x & 31) == 0) {{ "
        f"{shared}[threadIdx.x >> 5] = {accumulator}; }}"
    )
    ctx.lines.append("__syncthreads();")
    ctx.lines.append("if (threadIdx.x < 32) {")
    ctx.lines.append(
        f"    {accumulator} = threadIdx.x < ((blockDim.x + 31) / 32) "
        f"? {shared}[threadIdx.x] : {identity};"
    )
    ctx.lines.append(
        f"    for (int {shuffle_offset} = 16; {shuffle_offset} > 0; "
        f"{shuffle_offset} >>= 1) {{"
    )
    shuffle = f"__shfl_down_sync(0xffffffffu, {accumulator}, {shuffle_offset})"
    ctx.lines.append(
        f"        {accumulator} = "
        f"{_cuda_reduction_update(operator, accumulator, shuffle, accumulator_dtype, target=ctx.target)};"
    )
    ctx.lines.append("    }")
    ctx.lines.append(f"    if (threadIdx.x == 0) {{ {shared}[0] = {accumulator}; }}")
    ctx.lines.append("}")
    ctx.lines.append("__syncthreads();")
    result_type = ssa.Type(kind="scalar", dtype=result_dtype)
    ctx.lines.append(
        ctx.target.local_decl(
            result_type,
            local,
            ctx.target.cast(result_dtype, f"{shared}[0]"),
        )
    )

    return local


def _cuda_reduction_accumulator_dtype(operand_dtype: str, result_dtype: str) -> str:
    if operand_dtype in {
        "float8_e4m3fn",
        "float8_e5m2",
        "float16",
        "bfloat16",
    }:
        return "float32"

    if operand_dtype == "bool":
        return "int32"
    return result_dtype


def _cuda_reduction_update(
    operator: str, lhs: str, rhs: str, dtype: str, *, target=None
) -> str:
    del dtype

    if operator == "sum":
        return f"({lhs}) + ({rhs})"

    function = "nt_max" if operator == "max" else "nt_min"

    if target is not None:
        function = target.global_helper(function)

    return f"{function}({lhs}, {rhs})"


def _cuda_extremum_support(context):
    """Preserve integer width and propagate floating NaNs in every path."""
    opcodes = {operation.opcode for operation in context.operations.values()}

    if not opcodes.intersection(
        {
            "arith.maximum",
            "arith.minimum",
            "arith.max",
            "arith.min",
            "reduce.max",
            "reduce.min",
            "math.max",
            "math.min",
            "math.maximum",
            "math.minimum",
            "call.max",
            "call.min",
            "call.maximum",
            "call.minimum",
        }
    ):
        return ""

    return """template <typename T>
__device__ inline T nt_max(T a, T b) { return a > b ? a : b; }
template <typename T>
__device__ inline T nt_min(T a, T b) { return a < b ? a : b; }
__device__ inline float nt_max(float a, float b) {
    return isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
}
__device__ inline float nt_min(float a, float b) {
    return isnan(a) || isnan(b) ? NAN : fminf(a, b);
}
__device__ inline double nt_max(double a, double b) {
    return isnan(a) || isnan(b) ? NAN : fmax(a, b);
}
__device__ inline double nt_min(double a, double b) {
    return isnan(a) || isnan(b) ? NAN : fmin(a, b);
}
template <typename A, typename B>
__device__ inline auto nt_max(A a, B b) -> decltype(a + b) {
    using R = decltype(a + b);
    return nt_max(static_cast<R>(a), static_cast<R>(b));
}
template <typename A, typename B>
__device__ inline auto nt_min(A a, B b) -> decltype(a + b) {
    using R = decltype(a + b);
    return nt_min(static_cast<R>(a), static_cast<R>(b));
}"""


def _curand_support(context: ModuleRenderContext) -> tuple[str, str]:
    operations = {id(operation): operation for operation in context.operations.values()}

    if not any(operation.opcode == "math.rand" for operation in operations.values()):
        return "", ""
    return (
        "\n#include <curand_kernel.h>",
        """__device__ __forceinline__ float ninetoothed_curand_uniform(
    uint64_t seed, uint64_t offset
) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0ULL, offset, &state);
    return 1.0f - curand_uniform(&state);
}""",
    )


def _render_signature_params(params: list[str]) -> str:
    return ",\n".join(f"    {param}" for param in params)


def _signature_param(name, info, *, readonly: bool, restrict: bool) -> str:
    dtype = _cuda_type(info.dtype)

    if info.ndim == 0:
        return f"{dtype} {name}"

    const = "const " if readonly else ""
    qualifier = " __restrict__" if restrict else ""

    return f"{const}{dtype}*{qualifier} {name}"


def _auxiliary_param(name, tensors) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _signature_param(name, info, readonly=True, restrict=True)
    return f"int64_t {name}"


def _cuda_type(dtype: str | None, kind: str | None = None) -> str:
    dtype = common.normalize_dtype(dtype)

    if kind == "pointer":
        return f"{_cuda_type(dtype)}*"

    if kind == "index" or dtype in {"index", "int64"}:
        return "int64_t"

    types = {
        "float32": "float",
        "float16": "half",
        "bfloat16": "__nv_bfloat16",
        "float8_e4m3fn": "__nv_fp8_e4m3",
        "float8_e5m2": "__nv_fp8_e5m2",
        "float64": "double",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "int32": "int32_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
        "bool": "bool",
    }

    if dtype not in types:
        raise ValueError(f"Unsupported CUDA SSA dtype: {dtype!r}.")
    return types[dtype]


_cuda_integer_expr = common.c_integer_expr


def _cuda_arithmetic_result_type(op: ssa.Operation, ctx: _EmitContext) -> ssa.Type:
    result_type = op.results[0].type

    if op.opcode.startswith("math."):
        # CUDA math calls use the float32 intrinsics (sqrtf, expf, etc.).
        return replace(result_type, dtype="float32")

    dtype_source = ctx.tensor_infos.get(op.attrs.get("dtype_ref"))

    if result_type.dtype is None and dtype_source is not None:
        result_type = replace(result_type, dtype=dtype_source.dtype)

    if not op.opcode.startswith("arith."):
        return result_type

    if _normalize_dtype(result_type.dtype) not in {
        "float8_e4m3fn",
        "float8_e5m2",
        "float16",
        "bfloat16",
    }:
        return result_type
    return replace(result_type, dtype="float32")


_emit_cuda_direct_dot_operand = common.emit_direct_dot_operand


def _emit_cuda_wmma_dot(
    op: ssa.Operation, ctx: _EmitContext, *, coords: tuple[str, ...] | None
) -> str | None:
    if (
        len(op.results) != 1
        or _normalize_dtype(op.results[0].type.dtype) != "float32"
        or len(op.operands) != 2
        or any(
            (type_ := ctx.value_types.get(operand)) is None
            or _normalize_dtype(type_.dtype) != "float16"
            for operand in op.operands
        )
    ):
        return None

    lhs, rhs = op.operands[:2]
    lhs_axes = _value_axes(lhs, ctx)
    rhs_axes = _value_axes(rhs, ctx)
    result_axes = tuple(str(dim) for dim in op.results[0].type.shape)
    result_coords = (
        tuple(coords) if coords is not None else _current_coords(result_axes, ctx)
    )

    if len(result_coords) != 2:
        raise ValueError("CUDA WMMA dot lowering requires a rank-2 result domain.")

    local = f"{_local_symbol(op.results[0].name, ctx)}_wmma"
    row_base = f"(({result_coords[0]}) / 16) * 16"
    col_base = f"(({result_coords[1]}) / 16) * 16"
    k_extent = lhs_axes[-1]
    loop_var = f"{local}_k"
    lhs_row = f"({row_base}) + (static_cast<int64_t>(threadIdx.x) / 16)"
    lhs_col = f"{loop_var} + (static_cast<int64_t>(threadIdx.x) % 16)"
    rhs_row = f"{loop_var} + (static_cast<int64_t>(threadIdx.x) / 16)"
    rhs_col = f"({col_base}) + (static_cast<int64_t>(threadIdx.x) % 16)"

    ctx.lines.extend(
        (
            f"__shared__ half {local}_lhs[256];",
            f"__shared__ half {local}_rhs[256];",
            f"__shared__ float {local}_out[256];",
            (f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> {local}_acc;"),
            f"if (threadIdx.x < 32) {{ wmma::fill_fragment({local}_acc, 0.0f); }}",
            "__syncthreads();",
            ctx.target.loop_header(loop_var, "0", k_extent, "16"),
        )
    )

    body_lines: list[str] = []
    body = ctx.child(lines=body_lines, memo=dict(ctx.memo), mask_expr=None)
    lhs_value = _emit_element(lhs, (lhs_row, lhs_col), body)
    rhs_value = _emit_element(rhs, (rhs_row, rhs_col), body)
    lhs_active = f"({lhs_row}) < ({lhs_axes[0]}) && ({lhs_col}) < ({lhs_axes[1]})"
    rhs_active = f"({rhs_row}) < ({rhs_axes[0]}) && ({rhs_col}) < ({rhs_axes[1]})"
    body_lines.extend(
        (
            (
                f"{local}_lhs[threadIdx.x] = ({lhs_active}) "
                f"? __float2half_rn(static_cast<float>({lhs_value})) "
                ": __float2half_rn(0.0f);"
            ),
            (
                f"{local}_rhs[threadIdx.x] = ({rhs_active}) "
                f"? __float2half_rn(static_cast<float>({rhs_value})) "
                ": __float2half_rn(0.0f);"
            ),
            "__syncthreads();",
            "if (threadIdx.x < 32) {",
            (
                "    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, "
                f"wmma::row_major> {local}_a;"
            ),
            (
                "    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, "
                f"wmma::row_major> {local}_b;"
            ),
            f"    wmma::load_matrix_sync({local}_a, {local}_lhs, 16);",
            f"    wmma::load_matrix_sync({local}_b, {local}_rhs, 16);",
            f"    wmma::mma_sync({local}_acc, {local}_a, {local}_b, {local}_acc);",
            "}",
            "__syncthreads();",
        )
    )
    ctx.lines.extend(_indent_lines(body_lines, ctx.target))
    ctx.lines.extend(
        (
            "}",
            "if (threadIdx.x < 32) {",
            (
                f"    wmma::store_matrix_sync({local}_out, {local}_acc, 16, "
                "wmma::mem_row_major);"
            ),
            "}",
            "__syncthreads();",
        )
    )

    # Nested consumers can request a column that differs from the staging lane.
    # The fragment is stored in row-major order, so select the requested element.
    element = f"((({result_coords[0]}) % 16) * 16 + (({result_coords[1]}) % 16))"

    return f"{local}_out[{element}]"


def _coerce_cuda_binary_args(
    op: ssa.Operation, args: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, ...]:
    if len(args) != len(op.operands):
        return args

    result_type = op.results[0].type if op.results else None

    if result_type is not None and result_type.kind == "pointer":
        return args

    if any(
        (type_ := ctx.value_types.get(operand)) is not None and type_.kind == "pointer"
        for operand in op.operands
    ):
        return args

    dtype = result_type.dtype if result_type is not None else None

    if dtype is None or dtype == "bool":
        return tuple(
            f"_nt_arithmetic({value})"
            if (type_ := ctx.tensor_infos.get(operand) or ctx.value_types.get(operand))
            is not None
            and type_.dtype
            in {None, "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16"}
            else value
            for operand, value in zip(op.operands, args)
        )

    if dtype in {"float8_e4m3fn", "float8_e5m2", "float16", "bfloat16"}:
        dtype = "float32"
    return tuple(
        ctx.target.cast(dtype, value)
        if ctx.value_types.get(operand) is None
        or ctx.value_types[operand].dtype != dtype
        else value
        for operand, value in zip(op.operands, args)
    )


def _wmma_accumulation_mode(kernel: Kernel) -> str:
    return kernel.compiler_options.get("backend_options", {}).get(
        "cuda_wmma_accumulation", "fused"
    )


def _emit_cuda_wmma_reduction_loop(
    local: str, op: ssa.Operation, ctx: _EmitContext
) -> str | None:
    if len(op.results) != 1 or len(op.regions) != 1:
        return None

    iter_attrs = tuple(op.attrs.get("iter_args", ()))

    if len(iter_attrs) != 1:
        return None

    region = op.regions[0]
    dot = next(
        (
            inner
            for inner in region.operations
            if inner.opcode in {"linalg.dot", "linalg.matmul"}
            and len(inner.operands) >= 2
            and len(inner.results) == 1
        ),
        None,
    )
    add = next(
        (
            inner
            for inner in region.operations
            if inner.opcode == "arith.add" and len(inner.results) == 1
        ),
        None,
    )
    yield_op = next(
        (inner for inner in region.operations if inner.opcode == "scf.yield"), None
    )

    if dot is None or add is None or yield_op is None:
        return None

    # This hook replaces the complete region. It may only absorb the exact
    # extraction/dot/carry sequence; extra writes or control flow must remain
    # in ordinary SSA lowering. No inferred or wider carry may become float.
    if (
        tuple(inner.opcode for inner in region.operations)
        not in {
            (
                "tensor.extract",
                "tensor.extract",
                "linalg.dot",
                "arith.add",
                "scf.yield",
            ),
            (
                "tensor.extract",
                "tensor.extract",
                "linalg.matmul",
                "arith.add",
                "scf.yield",
            ),
        }
        or any(inner.regions for inner in region.operations)
        or len(dot.operands) != 2
        or any(len(inner.results) != 1 for inner in region.operations[:-1])
        or dot.operands
        != tuple(inner.results[0].name for inner in region.operations[:2])
        or len(op.results) != 1
        or len(region.args) != 2
    ):
        return None

    block_arg = str(iter_attrs[0]["block_arg"])
    dot_result = dot.results[0].name

    if set(add.operands) != {block_arg, dot_result}:
        return None

    if yield_op.operands != (add.results[0].name,):
        return None

    initial = str(iter_attrs[0]["initial"])
    initial_op = ctx.operations.get(initial)

    carry_types = (
        ctx.value_types.get(initial),
        region.args[1].type,
        op.results[0].type,
        dot.results[0].type,
        add.results[0].type,
    )

    if (
        region.args[1].name != block_arg
        or any(
            type_ is None
            or type_.kind != "tensor"
            or _normalize_dtype(type_.dtype) != "float32"
            or tuple(map(str, type_.shape))
            != tuple(map(str, dot.results[0].type.shape))
            for type_ in carry_types
        )
        or any(
            _normalize_dtype(inner.results[0].type.dtype) != "float16"
            for inner in region.operations[:2]
        )
    ):
        return None

    if initial_op is None or initial_op.opcode not in {
        "arith.constant",
        "tensor.zeros",
        "tensor.full",
    }:
        return None

    if initial_op.opcode != "tensor.zeros":
        initial_value = initial_op.attrs.get("value", 0)

        if initial_value != 0 or (
            isinstance(initial_value, float) and math.copysign(1.0, initial_value) < 0
        ):
            return None

    lhs, rhs = dot.operands[:2]
    lhs_axes = _value_axes(lhs, ctx)
    rhs_axes = _value_axes(rhs, ctx)
    result_axes = tuple(str(dim) for dim in dot.results[0].type.shape)
    result_coords = _current_coords(result_axes, ctx)

    if len(lhs_axes) != 2 or len(rhs_axes) != 2 or len(result_coords) != 2:
        return None

    lower = _emit_loop_bound(op.operands[0], ctx)
    upper = _emit_loop_bound(op.operands[1], ctx)
    step = _emit_loop_bound(op.operands[2], ctx)
    from ninetoothed.backends.emitters.cuda_cta_dot import emit_cta_dot, selected

    if selected(ctx.kernel):
        return emit_cta_dot(
            f"{_local_symbol(op.results[0].name, ctx)}_wmma_reduction",
            op,
            dot,
            ctx,
            bounds=(lower, upper, step),
            ordered=_wmma_accumulation_mode(ctx.kernel) == "ordered",
        )

    loop_var = f"{local}_i"
    induction = str(op.attrs.get("induction", "%iv"))
    bindings = dict(ctx.bindings or {})
    bindings[induction] = loop_var
    local_name = f"{_local_symbol(op.results[0].name, ctx)}_wmma_reduction"
    row_base = f"(({result_coords[0]}) / 16) * 16"
    col_base = f"(({result_coords[1]}) / 16) * 16"
    k_var = f"{local_name}_k"
    lhs_row = f"({row_base}) + (static_cast<int64_t>(threadIdx.x) / 16)"
    lhs_col = f"{k_var} + (static_cast<int64_t>(threadIdx.x) % 16)"
    rhs_row = f"{k_var} + (static_cast<int64_t>(threadIdx.x) / 16)"
    rhs_col = f"({col_base}) + (static_cast<int64_t>(threadIdx.x) % 16)"

    from ninetoothed.backends.emitters.cuda_dot_transfer import prove_dot_transfers

    transfers = prove_dot_transfers(
        op, dot, ctx, stage=loop_var, outer=ctx.outer_index_expr
    )

    if transfers is not None:
        ctx.lines.append(f"const bool {local_name}_dense_wmma = {transfers.guard};")

    ordered = _wmma_accumulation_mode(ctx.kernel) == "ordered"
    accumulator = f"{local_name}_carry" if ordered else f"{local_name}_acc"
    ctx.lines.extend(
        (
            f"__shared__ half {local_name}_lhs[256];",
            f"__shared__ half {local_name}_rhs[256];",
            f"__shared__ float {local_name}_out[256];",
            (f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> {local_name}_acc;"),
        )
    )

    if ordered:
        ctx.lines.append(
            f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> {local_name}_carry;"
        )

    ctx.lines.extend(
        (
            f"if (threadIdx.x < 32) {{ wmma::fill_fragment({accumulator}, 0.0f); }}",
            "__syncthreads();",
            ctx.target.loop_header(loop_var, lower, upper, step),
        )
    )

    if ordered:
        ctx.lines.append(
            f"    if (threadIdx.x < 32) {{ wmma::fill_fragment({local_name}_acc, 0.0f); }}"
        )

    ctx.lines.append(f"    {ctx.target.loop_header(k_var, '0', lhs_axes[-1], '16')}")

    body_lines: list[str] = []
    body = ctx.child(
        lines=body_lines, memo=dict(ctx.memo), bindings=bindings, mask_expr=None
    )
    lhs_value = _emit_element(lhs, (lhs_row, lhs_col), body)
    rhs_value = _emit_element(rhs, (rhs_row, rhs_col), body)
    lhs_active = f"({lhs_row}) < ({lhs_axes[0]}) && ({lhs_col}) < ({lhs_axes[1]})"
    rhs_active = f"({rhs_row}) < ({rhs_axes[0]}) && ({rhs_col}) < ({rhs_axes[1]})"
    indexed_assignments = (
        (
            f"{local_name}_lhs[threadIdx.x] = ({lhs_active}) "
            f"? __float2half_rn(static_cast<float>({lhs_value})) "
            ": __float2half_rn(0.0f);"
        ),
        (
            f"{local_name}_rhs[threadIdx.x] = ({rhs_active}) "
            f"? __float2half_rn(static_cast<float>({rhs_value})) "
            ": __float2half_rn(0.0f);"
        ),
    )

    if transfers is not None:
        direct_assignments = []

        for name, destination, bases, width, coordinates in (
            (
                transfers.left,
                "lhs",
                transfers.left_base,
                transfers.left_width,
                (lhs_row, lhs_col),
            ),
            (
                transfers.right,
                "rhs",
                transfers.right_base,
                transfers.right_width,
                (rhs_row, rhs_col),
            ),
        ):
            row, column = (
                common.target_index_expr(ctx.target, base).replace("//", "/")
                for base in bases
            )
            direct_assignments.append(
                f"{local_name}_{destination}[threadIdx.x] = "
                f"{ctx.target.tensor_ref(name)}[((({row}) + ({coordinates[0]})) * {width}) "
                f"+ ({column}) + ({coordinates[1]})];"
            )

        body_lines[:] = [
            f"if ({local_name}_dense_wmma) {{",
            *_indent_lines(direct_assignments, ctx.target),
            "} else {",
            *_indent_lines((*body_lines, *indexed_assignments), ctx.target),
            "}",
        ]
    else:
        body_lines.extend(indexed_assignments)

    body_lines.extend(
        (
            "__syncthreads();",
            "if (threadIdx.x < 32) {",
            (
                "    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, "
                f"wmma::row_major> {local_name}_a;"
            ),
            (
                "    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, "
                f"wmma::row_major> {local_name}_b;"
            ),
            f"    wmma::load_matrix_sync({local_name}_a, {local_name}_lhs, 16);",
            f"    wmma::load_matrix_sync({local_name}_b, {local_name}_rhs, 16);",
            (
                f"    wmma::mma_sync({local_name}_acc, {local_name}_a, "
                f"{local_name}_b, {local_name}_acc);"
            ),
            "}",
            "__syncthreads();",
        )
    )
    ctx.lines.extend(_indent_lines(_indent_lines(body_lines, ctx.target), ctx.target))
    ctx.lines.append("    }")

    if ordered:
        carry_value = f"{local_name}_carry.x[nt_wmma_component]"
        partial_value = f"{local_name}_acc.x[nt_wmma_component]"
        left, right = (
            (carry_value, partial_value)
            if add.operands[0] == block_arg
            else (partial_value, carry_value)
        )
        ctx.lines.extend(
            (
                "    if (threadIdx.x < 32) {",
                f"        for (int nt_wmma_component = 0; nt_wmma_component < {local_name}_carry.num_elements; ++nt_wmma_component) {{",
                f"            {carry_value} = __fadd_rn({left}, {right});",
                "        }",
                "    }",
                "    __syncthreads();",
            )
        )

    ctx.lines.extend(
        (
            "}",
            "if (threadIdx.x < 32) {",
            (
                f"    wmma::store_matrix_sync({local_name}_out, {accumulator}, "
                "16, wmma::mem_row_major);"
            ),
            "}",
            "__syncthreads();",
        )
    )
    element = f"((({result_coords[0]}) % 16) * 16 + (({result_coords[1]}) % 16))"
    result = f"{local_name}_out[{element}]"
    ctx.memo[op.results[0].name] = result

    return result


TARGET = CudaTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["CudaTarget", "TARGET", "emit"]


_ARITHMETIC_SUPPORT = """
template <class T> __device__ T _nt_reduce_max(T lhs, T rhs) {
    return lhs > rhs ? lhs : rhs;
}
template <class T> __device__ T _nt_reduce_min(T lhs, T rhs) {
    return lhs < rhs ? lhs : rhs;
}
__device__ inline float _nt_reduce_max(float lhs, float rhs) {
    return isnan(lhs) || isnan(rhs) ? NAN : fmaxf(lhs, rhs);
}
__device__ inline float _nt_reduce_min(float lhs, float rhs) {
    return isnan(lhs) || isnan(rhs) ? NAN : fminf(lhs, rhs);
}
__device__ inline double _nt_reduce_max(double lhs, double rhs) {
    return isnan(lhs) || isnan(rhs) ? NAN : fmax(lhs, rhs);
}
__device__ inline double _nt_reduce_min(double lhs, double rhs) {
    return isnan(lhs) || isnan(rhs) ? NAN : fmin(lhs, rhs);
}

template <class T>
__device__ typename std::enable_if<std::is_arithmetic<T>::value, T>::type
_nt_arithmetic(T value) { return value; }

template <class T>
__device__ typename std::enable_if<!std::is_arithmetic<T>::value, float>::type
_nt_arithmetic(T value) { return static_cast<float>(value); }

template <class T> struct _nt_limits {
    using R = typename std::conditional<std::is_arithmetic<T>::value, T, float>::type;
    static constexpr R low = std::numeric_limits<R>::has_infinity
        ? -std::numeric_limits<R>::infinity() : std::numeric_limits<R>::lowest();
    static constexpr R high = std::numeric_limits<R>::has_infinity
        ? std::numeric_limits<R>::infinity() : std::numeric_limits<R>::max();
};

template <class A, class B>
__device__ auto _nt_true_divide(A lhs, B rhs)
    -> typename std::conditional<std::is_integral<decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::value,
                                 float, decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::type {
    using R = typename std::conditional<
        std::is_integral<decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::value, float, decltype(_nt_arithmetic(lhs) + _nt_arithmetic(rhs))>::type;
    return static_cast<R>(_nt_arithmetic(lhs)) / static_cast<R>(_nt_arithmetic(rhs));
}
"""
