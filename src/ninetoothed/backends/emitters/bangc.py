"""BangC syntax hooks for the common SSA emitter.

The Cambricon BANG-C task model differs from CUDA threads: one kernel task
(``taskIdX``) executes on one MLU core, so the generic flat-index program is
rendered as a per-task scalar loop over a fixed element chunk.  Kernel launch
uses the ``<<<dim, ktype, queue>>>`` trigram with ``cnrtDim3_t`` grids and a
``cnrtQueue_t`` stream, matching the CUDA backend's C-ABI launcher contract.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import bangc_vector
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import dot_accumulation_plan
from ninetoothed.backends.emitters.bangc_launch import (
    MAX_BLOCK_TASKS,
    render_segmented_launch,
)
from ninetoothed.backends.emitters.bangc_ranges import (
    allocate_chunk_parameter,
    allocate_private_symbol,
    ceil_divide,
)
from ninetoothed.backends.emitters.base import (
    EmitterTarget,
    ModuleRenderContext,
    private_parameter_conflicts,
)
from ninetoothed.backends.emitters.c_types import (
    CComputedTypesMixin,
    render_c_type_support,
    render_int32_cast_support,
)
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.backends.emitters.math_helpers import render_float32_trig_support
from ninetoothed.backends.emitters.patterns import (
    match_reduce_broadcast,
    match_row_reduce,
)
from ninetoothed.backends.emitters.staging import (
    dense_staging_memory_guard,
    flatten_tiled_accesses,
    split_stride_guard,
    staging_extent_guard,
)
from ninetoothed.backends.emitters.task_launch import allocate_task_launch_symbols
from ninetoothed.backends.toolchain import resolve_bangc_arch
from ninetoothed.compiler.vector_resources import plan_vector_chunks
from ninetoothed.ir import Kernel, ir_to_dict, ssa

_emit_element = common.emit_element
_fresh_temp = common.fresh_temp
_indent_lines = common.indent_lines
_linearized_index = common.linearized_index
_nested_local_suffix = common.nested_local_suffix
_normalize_dtype = common.normalize_dtype
_reduction_identity = common.reduction_identity
_target_index_expr = common.target_index_expr
_value_axes = common.value_axes


@dataclass(frozen=True, kw_only=True)
class BangCTarget(CComputedTypesMixin, EmitterTarget):
    backend: Target = Target.BANGC
    language: str = "bangc/c++"
    suffix: str = "mlu"
    source_route: str = "ssa-unified-bangc-emitter"
    c_style_syntax: bool = True
    use_polynomial_memory_pair: bool = False
    use_bounded_domain_pool: bool = False

    def index_cast(self, value: str) -> str:
        return f"(int64_t)({value})"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value};"

    def literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, int) and abs(value) >= 1 << 48:
            from ninetoothed.targets import UnsupportedTargetCapabilityError

            raise UnsupportedTargetCapabilityError(
                "BangC cannot reliably represent 64-bit numerical integer "
                "literals on MLU590 with cncc 5.1.2; scalar values truncate "
                "to 48 bits."
            )

        if isinstance(value, float) and math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"

        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return "NAN"

        if value == "inf":
            return "INFINITY"

        if value == "-inf":
            return "-INFINITY"
        return repr(value)

    def cast(self, dtype, value):
        normalized = _normalize_dtype(dtype)

        if normalized in {"int32", "int64", "uint32", "uint64"} and re.fullmatch(
            r"[-+]?\d+\.0*[fF]?", value.strip()
        ):
            # Cncc device code miscompiles double literals cast to
            # integers (the masked-load `other` of 0.0 turned into
            # garbage); integer targets get an integer literal instead.
            value = str(int(float(value)))

        if normalized == "bfloat16":
            # C-style casts to bfloat16_t are ambiguous on cncc; use the
            # explicit conversion intrinsic instead.
            return f"{self.global_helper('__float2bfloat16_rn')}((float)({value}))"

        if normalized == "float16":
            return f"{self.global_helper('__float2half_rn')}((float)({value}))"

        if normalized == "int32":
            return f"{self.global_helper('nt_cast_int32')}({value})"

        return f"(({self.type_name(dtype)})({value}))"

    def type_name(self, dtype, kind=None):
        return _bangc_type(dtype, kind)

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        declaration = super().local_decl(type_, name, expr)

        if type_.dtype is None and type_.kind in {"scalar", "tensor"}:
            # A computed C++ dtype can become wide even if the original SSA
            # left it unspecified. Enforce the numerical contract on that
            # actual type rather than allowing auto to bypass the guard.
            declaration += (
                f"\nstatic_assert(!std::is_integral<decltype({name})>::value "
                f"|| sizeof(decltype({name})) <= 4, "
                '"BangC 64-bit numerical values truncate to 48 bits on '
                'MLU590 with cncc 5.1.2");'
            )

        return declaration

    def where(self, cond, yes, no):
        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            return (
                f"{self.global_helper('ninetoothed_atomic_add')}({args[0]}, {args[1]})"
            )

        if name == "load" and args:
            return f"*({args[0]})"

        if name in {"block_dot", "dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "rand" and len(args) >= 2:
            return f"{self.global_helper('ninetoothed_rand_uniform')}(({args[0]}), ({args[1]}))"

        if name == "rsqrt" and len(args) == 1:
            return f"(1.0f / sqrtf({args[0]}))"

        if name == "reciprocal" and len(args) == 1:
            return f"(1.0f / ({args[0]}))"

        if name == "sigmoid" and len(args) == 1:
            return f"(1.0f / (1.0f + expf(-({args[0]}))))"

        if name in {"sin", "cos"} and len(args) == 1:
            return f"{self.global_helper('nt_' + name)}({args[0]})"

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
            "round": "roundf",
            "sin": "sinf",
            "sinh": "sinhf",
            "sqrt": "sqrtf",
            "tan": "tanf",
            "tanh": "tanhf",
        }
        function = functions.get(name, name)

        return f"{function}({', '.join(args)})"

    def supports_cooperative_reduction(self, schedule):
        return bool(schedule.get("bangc_cooperative_reduction"))

    def thread_id(self):
        # One BANG task is one serial program; the cooperative lowering uses
        # a single lane so thread strided loops degenerate to full coverage.
        return "0"

    def thread_count(self):
        return "1"

    def emit_cooperative_reduction(self, local, operation, context):
        return _emit_bangc_cooperative_reduction(local, operation, context)

    def program_id(self, axis=0):
        if axis == 0:
            return "(int64_t)(taskIdX)"

        if axis == 1:
            return "(int64_t)(taskIdY)"

        if axis == 2:
            return "(int64_t)(taskIdZ)"

        raise ValueError("BangC supports at most three task-grid axes.")

    def atomic_add(self, operands: tuple[str, ...], dtype: str) -> str:
        suffix = _atomic_suffix(dtype)

        return (
            f"{self.global_helper('ninetoothed_atomic_add_' + suffix)}({operands[0]}, {operands[1]})"
            if suffix is not None
            else f"{self.global_helper('ninetoothed_atomic_add_f32')}({operands[0]}, {operands[1]})"
        )

    def emit_dot_operand(self, name, coords, context):
        return _emit_bangc_direct_dot_operand(name, coords, context)

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "nt_max" if operator == "max" else "nt_min"

        return f"{function}({acc}, {term})"

    def reduce_update_typed(self, operator, acc, term, dtype):
        if dtype is None:
            return super().reduce_update_typed(operator, acc, term, dtype)
        return _bangc_reduction_update(
            operator, acc, term, _bangc_reduction_accumulator_dtype(dtype, dtype)
        )

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        return _bangc_arithmetic_result_type(operation, context)

    def coerce_binary_args(self, operation, args, context):
        return _coerce_bangc_binary_args(operation, args, context)

    def check_numeric_result(self, operation, value, context):
        result_type = self.arithmetic_result_type(operation, context)

        if result_type.dtype is None and result_type.kind in {"scalar", "tensor"}:
            return f"{self.global_helper('_nt_checked_numeric')}({value})"

        return value

    def schedule_context(self, context: ModuleRenderContext) -> ModuleRenderContext:
        return self._plan_task_domain(self._schedule_routes(context))

    def _schedule_routes(self, context):
        conflicts = private_parameter_conflicts(
            context, ("nt_",), excluded_prefixes=("nt_materialized_",)
        )

        context.lowering_plans["private_parameters_safe"] = not conflicts

        if not self.optimizes_native_execution(context.kernel):
            return context

        if conflicts:
            return replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "bangc_private_parameter_fallback": {
                        "selected": True,
                        "parameters": conflicts,
                        "reason": "public-parameter-private-prefix",
                        "fallback": "original-indexed-ssa",
                    },
                },
            )

        options = context.kernel.compiler_options.get("backend_options", {})

        from ninetoothed.backends.emitters.bangc_row_tiles import plan_full_rows

        effective_arch = resolve_bangc_arch(options.get("arch", "native"))
        target = replace(
            self,
            use_polynomial_memory_pair=effective_arch == "compute_50",
            use_bounded_domain_pool=effective_arch == "compute_50",
        )
        body = _bangc_integer_expr(context.body)
        predicate, guarded = split_stride_guard(body)
        rows = plan_full_rows(context, budget=_NRAM_BUDGET_BYTES)
        cooperative_rows = bangc_vector.plan_rows(context, budget=_NRAM_BUDGET_BYTES)
        row_vectors = bangc_vector.plan_row_vectors(context, budget=_NRAM_BUDGET_BYTES)
        alias_rows = bangc_vector.plan_row_vectors(
            context, budget=_NRAM_BUDGET_BYTES, allow_reductions=True
        )
        scalar_rows = (
            bangc_vector.plan_rows(
                replace(context, cooperative_reduction_program=True),
                budget=_NRAM_BUDGET_BYTES,
            )
            if not context.cooperative_reduction_program
            and match_row_reduce(context) is not None
            else None
        )
        context.lowering_plans.update(
            target=target,
            effective_arch=effective_arch,
            full_rows=rows,
            cooperative_rows=cooperative_rows,
            row_vectors=row_vectors,
            alias_rows=alias_rows,
            scalar_rows=scalar_rows,
            staging=_nram_staging_plan(
                context, guarded if predicate is not None else body
            ),
        )

        if rows is not None:
            derived_ssa = ir_to_dict(rows.tiles.context.kernel.ssa)
            digest = hashlib.sha256(
                json.dumps(derived_ssa, sort_keys=True).encode()
            ).hexdigest()
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "bangc_full_row_tiling": {
                        "selected": True,
                        "route": "target-derived-ssa",
                        "source_shape": (rows.tiles.rows, rows.tiles.columns),
                        "output_tile": (1, rows.tiles.columns),
                        "shape_queries": "frozen-original",
                        "fallback_guard": _bangc_integer_expr(rows.vector.guard),
                        "fallback": "original-indexed-ssa",
                        "nram_bytes": rows.vector.capacity
                        * rows.vector.dag.bytes_per_element,
                        "derived_ssa_sha256": digest,
                        "derived_ssa": derived_ssa,
                    },
                },
            )

        from ninetoothed.backends.emitters.bangc_row_batch import render_row_batch

        batch = render_row_batch(
            context,
            target,
            _bangc_float_literal,
            _nram_halving_reduce,
            auto_batch=True,
        )
        context.lowering_plans["row_batch"] = batch

        from ninetoothed.backends.emitters.affine_transfer import plan_affine_transfer
        from ninetoothed.backends.emitters.bangc_direct_dot import render_direct_dot
        from ninetoothed.backends.emitters.bangc_dot import render_native_dot
        from ninetoothed.backends.emitters.bangc_transpose import (
            render_transpose_transfer,
        )
        from ninetoothed.backends.emitters.indexed_transfer import plan_indexed_transfer

        context.lowering_plans.update(
            native_dot=render_native_dot(
                context, target, dma=True, stage_output=True, vector_epilogue=True
            ),
            direct_dot=render_direct_dot(
                replace(context, grid_total="nt_fallback_tasks"), target
            ),
            transpose=render_transpose_transfer(context),
            affine=plan_affine_transfer(context),
            indexed=plan_indexed_transfer(context),
        )

        if batch is not None:
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "bangc_row_batch": {
                        "selected": True,
                        "route": "typed-row-dag",
                        "batch_rows": batch.batch_rows,
                        "tasks": batch.tasks,
                        "nram_bytes": batch.nram_bytes,
                        "fallback_guard": _bangc_integer_expr(batch.guard),
                        "fallback": "original-row-schedule",
                        "dot_accumulation": "unchanged",
                        "cost_target": batch.target_arch,
                        "cost_policy": {
                            "resource_fallback": True,
                            "original_batch8_budget_exceeded": True,
                            "minimum_rows": 512,
                            "columns": 1024,
                            "target_arch": "compute_50",
                            "program": "cooperative-reduction",
                            "requires_reduction": True,
                            "batch_budget_bytes": 416 * 1024,
                            "fallback_reserve_bytes": 96 * 1024,
                            "combined_budget_bytes": 512 * 1024,
                            "sum_schedule": "original-per-row-halving",
                            "existing_legal_auto_routes": "unchanged",
                        }
                        if batch.resource_fallback
                        else {
                            "minimum_rows": 512,
                            "columns": 1024,
                            "target_arch": "compute_50",
                            "program": "cooperative-reduction",
                            "requires_reduction": True,
                            "batch16_budget_bytes": 384 * 1024,
                            "fallback_reserve_bytes": 96 * 1024,
                            "combined_budget_bytes": 480 * 1024,
                            "default_budget_bytes": 192 * 1024,
                        }
                        if options.get("row_batch_rows", "auto") == "auto"
                        else {
                            "requested_rows": options["row_batch_rows"],
                            "target_arch": "compute_50",
                            "program": "cooperative-reduction",
                            "requires_reduction": True,
                            "batch_budget_bytes": 416 * 1024,
                            "fallback_reserve_bytes": 96 * 1024,
                            "combined_budget_bytes": 512 * 1024,
                            "default_policy": "unchanged",
                            "sum_schedule": "original-per-row-halving",
                        },
                    },
                },
            )

        if not _want_output_tile_fusion(context):
            return context

        from ninetoothed.backends.emitters.bangc_dot_fusion import render_fused_dot
        from ninetoothed.backends.emitters.dot_fusion import fuse_dot_output_tiles

        derived = fuse_dot_output_tiles(context)
        native = (
            render_fused_dot(
                replace(context, grid_total="nt_fallback_tasks"), target, tiled=derived
            )
            if derived is not None
            else None
        )
        context.lowering_plans["fused_dot"] = native

        if native is None:
            return context

        derived_ssa = ir_to_dict(derived.kernel.ssa)
        digest = hashlib.sha256(
            json.dumps(derived_ssa, sort_keys=True).encode()
        ).hexdigest()

        return replace(
            context,
            scheduled_metadata={
                **context.scheduled_metadata,
                "bangc_output_tile_fusion": {
                    "selected": True,
                    "route": "target-derived-ssa",
                    "source_tile": (32, 32),
                    "output_tile": (64, 64),
                    "output_tile_fusion": (2, 2),
                    "k_tile": 32,
                    "k_stage_grouping": "preserved",
                    "cost_policy": {
                        "selection": "automatic",
                        "min_output_elements": 65536,
                        "min_reduction_elements": 64,
                    },
                    "fallback_guard": _bangc_integer_expr(native.guard),
                    "fallback": "original-indexed-ssa",
                    "derived_ssa_sha256": digest,
                    "derived_ssa": derived_ssa,
                },
            },
        )

    def _plan_task_domain(self, context):
        kernel = context.kernel
        options = dict(kernel.compiler_options.get("backend_options", {}))
        fast_parameters_safe = context.lowering_plans["private_parameters_safe"]

        if not fast_parameters_safe and options.get("row_batch_rows", "auto") != "auto":
            raise ValueError(
                "The explicit BangC row packing request conflicts with private parameters."
            )

        default_chunk = int(options.get("task_chunk", 1024))
        chunk = common.schedule_int(kernel, "bangc_task_chunk", default_chunk)
        scalar_chunk = chunk
        uses_atomics = _program_uses_atomics(context)
        vector_request = kernel.compiler_options.get("backend_options", {}).get(
            "vector_chunk"
        )
        vector_resource_plan = None

        if not 1 <= chunk <= (1 << 63) - 1:
            raise ValueError("The BangC task chunk must fit positive int64.")

        total = _bangc_integer_expr(context.total)
        grid_total = _bangc_integer_expr(context.grid_total)
        body = _bangc_integer_expr(context.body)
        # The shared emitter wraps a contiguous fast path in a runtime
        # stride guard; staging only ever matches the contiguous branch,
        # so split the guard and keep the generic branch as the fallback.
        guard_predicate, guarded_body = split_stride_guard(body)
        staging = context.lowering_plans.get("staging")

        if staging is not None:
            if staging.dag is not None and "task_chunk" in options and chunk % 128:
                staging = replace(staging, dag=None)
            # Keep staged __nram__ buffers inside the MLU590 256 KB budget
            # (measured: 3 x 64 KB passes, 3 x 256 KB fails cnas).  Bigger
            # chunks amortize task launches and memcpy bursts (measured
            # 228 GB/s at 1024 vs 1214 GB/s at 16384), so staged kernels
            # default to the largest in-budget chunk unless overridden.

            staged_buffers = (
                (*staging.inputs,)
                if _staged_identity(staging)
                else (*staging.inputs, *staging.outputs)
            )
            buffer_bytes = sum(
                2 if staging.buffer_types.get(buffer) == "half" else 4
                for _, buffer in staged_buffers
            )

            if staging.dag is not None:
                buffer_bytes = staging.dag.bytes_per_element

            budget_chunk = max(
                128, (_NRAM_BUDGET_BYTES // max(buffer_bytes, 1)) // 128 * 128
            )

            if "task_chunk" not in options:
                chunk = budget_chunk
            else:
                chunk = min(chunk, budget_chunk)

            if (
                staging.dag is not None
                and not context.cooperative_reduction_program
                and not uses_atomics
            ):
                allocations = bangc_vector.nram_allocations(staging.dag)
                vector_resource_plan = plan_vector_chunks(
                    allocations,
                    budget_bytes=_NRAM_BUDGET_BYTES,
                    candidates=(128, 1024, 4096, 8192, budget_chunk),
                    default_chunk=chunk,
                    requested_chunk=vector_request,
                )
                chunk = vector_resource_plan.selected.elements
                context.scheduled_metadata["bangc_vector_resources"] = {
                    **vector_resource_plan.metadata(),
                    "route": "typed-flat-dag",
                    "budget_bytes": _NRAM_BUDGET_BYTES,
                    "allocation_names": tuple(item.name for item in allocations),
                    "fallback_task_chunk": scalar_chunk,
                    "fallback_task_partition": "unchanged-original-scalar",
                }

        if vector_request is not None and vector_resource_plan is None:
            raise ValueError(
                "The BangC vector resource chunk request requires a proved typed "
                "flat DAG; the request was not consumed."
            )

        indexed_alias_plan = context.lowering_plans.get("indexed")
        alias_parameters = (
            {
                indexed_alias_plan.source,
                indexed_alias_plan.indices,
                indexed_alias_plan.output,
            }
            if indexed_alias_plan is not None
            else set()
        )
        affine_alias_plan = context.lowering_plans.get("affine")

        if affine_alias_plan is not None:
            alias_parameters.update(
                (affine_alias_plan.source, affine_alias_plan.output)
            )

        alias_parameters.update(
            _staging_alias_parameters(
                context, guarded_body if guard_predicate is not None else body
            )
        )

        if staging is not None:
            equal_extents = staging_extent_guard(
                context, (*staging.inputs, *staging.outputs)
            )
            memory_guard = dense_staging_memory_guard(
                context,
                tuple(name for name, _ in staging.inputs),
                tuple(name for name, _ in staging.outputs),
            )

            guard_predicate = " && ".join(
                [
                    *([f"({guard_predicate})"] if guard_predicate is not None else []),
                    *([equal_extents] if equal_extents != "true" else []),
                    memory_guard,
                ]
            )

        if chunk != 1024:
            context.lowering_plans["native_dot"] = None

        context.lowering_plans["task_domain"] = _TaskDomain(
            body,
            total,
            grid_total,
            guard_predicate,
            staging,
            default_chunk,
            chunk,
            scalar_chunk,
            frozenset(alias_parameters),
            uses_atomics,
            fast_parameters_safe,
        )

        return context

    def render_module(self, context: ModuleRenderContext) -> str:
        if context.vector_program or context.block_program or context.scalar_program:
            raise ValueError(
                "The BangC backend renders every kernel through the generic "
                "flat-index task domain; vector/block program modes are not "
                "supported."
            )

        kernel = context.kernel
        effective_arch = context.lowering_plans.get("effective_arch")
        self = context.lowering_plans.get("target", self)
        domain = context.lowering_plans["task_domain"]
        body, total, grid_total = domain.body, domain.total, domain.grid_total
        guard_predicate, staging = domain.guard, domain.staging
        default_chunk, chunk = domain.default_chunk, domain.chunk
        scalar_chunk = domain.scalar_chunk
        alias_parameters = domain.alias_parameters
        fast_parameters_safe = domain.private_parameters_safe

        kernel_params = _render_signature_params(
            [
                *(
                    _signature_param(
                        name,
                        context.tensors[name],
                        readonly=True,
                        restrict=name not in alias_parameters,
                    )
                    for name in context.variables
                ),
                *(
                    _signature_param(
                        name,
                        context.tensors[name],
                        readonly=False,
                        restrict=name not in alias_parameters,
                    )
                    for name in context.outputs
                ),
                *(
                    _auxiliary_param(axis, context.tensors)
                    for axis in context.shape_params
                ),
            ]
        )
        queue_name = allocate_private_symbol(
            "queue", *context.variables, *context.outputs, *context.shape_params
        )
        launch_params = _render_signature_params(
            [
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=True, restrict=False
                    )
                    for name in context.variables
                ),
                *(
                    _signature_param(
                        name, context.tensors[name], readonly=False, restrict=False
                    )
                    for name in context.outputs
                ),
                *(
                    _auxiliary_param(axis, context.tensors)
                    for axis in context.shape_params
                ),
                f"cnrtQueue_t {queue_name}",
            ]
        )
        chunk_parameter = allocate_chunk_parameter(
            kernel_params, launch_params, body, total
        )
        private_chunk_needed = False
        private_chunk_bound = None

        if context.cooperative_reduction_program:
            row_dag = context.lowering_plans.get("cooperative_rows")

            if row_dag is not None:
                fast = bangc_vector.render_rows(
                    row_dag,
                    self,
                    _bangc_float_literal,
                    _nram_halving_reduce,
                )
                kernel_prelude = (
                    f"    if ({row_dag.guard}) {{\n"
                    + common.indent_block(fast, "    ")
                    + "\n    } else {\n"
                    + common.indent_block(body, "        ")
                    + "\n    }"
                )
                tasks_expr = row_dag.rows
                chunk_decl = ""
            else:
                kernel_prelude = body
                tasks_expr = grid_total
                chunk_decl = ""
        elif domain.uses_atomics:
            # Hardware scalar atomics trap on MLU590 and the plain GDRAM
            # read-modify-write fallback loses updates under concurrent tasks
            # (measured ~20/256 surviving), so atomic kernels are serialized
            # onto one task that walks the whole flat domain.
            kernel_prelude = f"""    for (int64_t {self.index_name} = 0; {self.index_name} < {total}; {self.index_name}++) {{
{common.indent_block(body, "        ")}
    }}"""
            tasks_expr = "1"
            chunk_decl = ""
        elif staging is not None:
            staged_body = _render_nram_staged_body(staging, total, chunk, target=self)

            if guard_predicate is not None:
                newline = chr(10)
                indented_staged = newline.join(
                    "    " + line for line in staged_body.split(newline)
                )
                generic_loop = common.indent_block(
                    _render_chunked_scalar_body(
                        body,
                        total,
                        chunk_parameter,
                        self.index_name,
                        reserved=kernel_params,
                    ),
                    "    ",
                )
                private_chunk_needed = True
                kernel_prelude = (
                    "    if ("
                    + guard_predicate
                    + ") {"
                    + newline
                    + indented_staged
                    + newline
                    + "    } else {"
                    + newline
                    + generic_loop
                    + newline
                    + "    }"
                )
            else:
                kernel_prelude = staged_body

            # Unsafe staging retains the original scalar task partition,
            # including when the NRAM budget selected a larger fast chunk.
            tasks_expr = (
                f"({guard_predicate}) ? ({ceil_divide(total, str(chunk))}) : "
                f"({ceil_divide(total, str(scalar_chunk))})"
            )
            private_chunk_bound = (
                f"(({guard_predicate}) ? {chunk}LL : {scalar_chunk}LL)"
            )
            chunk_decl = ""
        else:
            row_plan = context.lowering_plans.get("scalar_rows")

            if row_plan is not None:
                fast = bangc_vector.render_rows(
                    row_plan, self, _bangc_float_literal, _nram_halving_reduce
                )
                guard = row_plan.guard
                kernel_prelude = (
                    f"    if ({guard}) {{\n"
                    + common.indent_block(fast, "    ")
                    + "\n    } else {\n"
                    + f"        int64_t {self.index_name} = (int64_t)(taskIdX);\n"
                    + f"        if ({self.index_name} < {total}) {{\n"
                    + common.indent_block(body, "            ")
                    + "\n        }\n    }"
                )
                tasks_expr = row_plan.rows
                chunk_decl = ""
            else:
                kernel_prelude = _render_chunked_scalar_body(
                    body,
                    total,
                    chunk_parameter,
                    self.index_name,
                    reserved=kernel_params,
                )
                private_chunk_needed = True
                tasks_expr = ceil_divide(total, "nt_chunk")
                chunk_decl = f"    const int64_t nt_chunk = {chunk}LL;\n"

        if fast_parameters_safe:
            native_dot = context.lowering_plans.get("native_dot")

            if native_dot is not None:
                kernel_prelude = common.guarded_body(
                    _bangc_integer_expr(native_dot.guard),
                    _bangc_integer_expr(native_dot.body),
                    kernel_prelude,
                )
                tasks_expr = _bangc_integer_expr(native_dot.tasks)
                chunk_decl = ""

        if fast_parameters_safe:
            fallback_tasks = re.sub(r"\bnt_chunk\b", str(chunk), tasks_expr)
            fused = _bind_fallback_tasks(
                context.lowering_plans.get("fused_dot"), fallback_tasks
            )

            if fused is not None:
                kernel_prelude = common.guarded_body(
                    _bangc_integer_expr(fused.guard),
                    _bangc_integer_expr(fused.body),
                    kernel_prelude,
                )
                tasks_expr = _bangc_integer_expr(fused.tasks)
                chunk_decl = ""

        if fast_parameters_safe:
            fallback_tasks = re.sub(r"\bnt_chunk\b", str(chunk), tasks_expr)
            direct = _bind_fallback_tasks(
                context.lowering_plans.get("direct_dot"), fallback_tasks
            )

            if direct is not None:
                kernel_prelude = common.guarded_body(
                    _bangc_integer_expr(direct.guard),
                    _bangc_integer_expr(direct.body),
                    kernel_prelude,
                )
                tasks_expr = _bangc_integer_expr(direct.tasks)
                chunk_decl = ""

        if fast_parameters_safe:
            rows = context.lowering_plans.get("full_rows")

            if rows is not None:
                fallback_tasks = re.sub(r"\bnt_chunk\b", str(chunk), tasks_expr)
                fast = bangc_vector.render_rows(
                    rows.vector,
                    self,
                    _bangc_float_literal,
                    _nram_halving_reduce,
                )
                guard = _bangc_integer_expr(rows.vector.guard)
                kernel_prelude = common.guarded_body(
                    guard, _bangc_integer_expr(fast), kernel_prelude
                )
                tasks_expr = f"({guard}) ? ({rows.vector.rows}) : ({fallback_tasks})"
                chunk_decl = ""

        if fast_parameters_safe:
            row_vectors = context.lowering_plans.get("row_vectors")

            if row_vectors is not None:
                fallback_tasks = re.sub(r"\bnt_chunk\b", str(chunk), tasks_expr)
                fast = bangc_vector.render_rows(
                    row_vectors, self, _bangc_float_literal, _nram_halving_reduce
                )
                guard = _bangc_integer_expr(row_vectors.guard)
                kernel_prelude = common.guarded_body(
                    guard, _bangc_integer_expr(fast), kernel_prelude
                )
                tasks_expr = f"({guard}) ? ({row_vectors.rows}) : ({fallback_tasks})"
                chunk_decl = ""

        if fast_parameters_safe:
            from ninetoothed.backends.emitters.bangc_memory import (
                render_indexed_transfer,
            )

            indexed = render_indexed_transfer(
                context,
                chunk,
                fallback_body=kernel_prelude,
                fallback_tasks=re.sub(r"\bnt_chunk\b", str(chunk), tasks_expr),
                plan=context.lowering_plans.get("indexed"),
            )

            if indexed is not None:
                kernel_prelude = common.guarded_body(
                    _bangc_integer_expr(indexed.guard),
                    _bangc_integer_expr(indexed.body),
                    kernel_prelude,
                )
                tasks_expr = _bangc_integer_expr(indexed.tasks)
                chunk_decl = ""

        if fast_parameters_safe:
            from ninetoothed.backends.emitters.bangc_affine import (
                render_affine_transfer,
            )

            fallback_tasks = re.sub(r"\bnt_chunk\b", str(chunk), tasks_expr)
            affine = render_affine_transfer(
                context,
                fallback_tasks,
                coalesced=True,
                plan=context.lowering_plans.get("affine"),
            )

            if affine is not None:
                kernel_prelude = common.guarded_body(
                    _bangc_integer_expr(affine.guard),
                    _bangc_integer_expr(affine.body),
                    kernel_prelude,
                )
                tasks_expr = _bangc_integer_expr(affine.tasks)
                chunk_decl = ""

        if fast_parameters_safe:
            batch = context.lowering_plans.get("row_batch")

            if batch is not None:
                fallback_tasks = re.sub(r"\bnt_chunk\b", str(chunk), tasks_expr)
                guard = _bangc_integer_expr(batch.guard)
                kernel_prelude = common.guarded_body(
                    guard, _bangc_integer_expr(batch.body), kernel_prelude
                )
                tasks_expr = f"({guard}) ? ({batch.tasks}) : ({fallback_tasks})"
                chunk_decl = ""

        if fast_parameters_safe:
            transpose = context.lowering_plans.get("transpose")

            if transpose is not None:
                fallback_tasks = re.sub(r"\bnt_chunk\b", str(chunk), tasks_expr)
                guard = _bangc_integer_expr(transpose.guard)
                kernel_prelude = common.guarded_body(
                    guard, _bangc_integer_expr(transpose.body), kernel_prelude
                )
                tasks_expr = f"({guard}) ? ({transpose.tasks}) : ({fallback_tasks})"
                chunk_decl = ""

        predicate_consumer = (
            "::__bang_sumpool(nt_dag_scratch_float32_1, nt_dag_scratch_float32_0, 16, 1, (uint32_t)(nt_aligned / 16),"
            in kernel_prelude
        )
        context.scheduled_metadata["bangc_bounded_domain_pool"] = {
            "requested": True,
            "effective_arch": effective_arch,
            "enabled": self.use_bounded_domain_pool,
            "emitted_consumer": predicate_consumer,
            "selection_reason": (
                "exact-float01-predicate-sumpool"
                if predicate_consumer
                else "unverified-target"
                if effective_arch != "compute_50"
                else "no-bounded-vector-math-consumer"
            ),
            "runtime_guard": "0 < aligned <= 16384, aligned % 128 == 0, 0 <= count <= aligned",
            "fallback": "original-max-abs-domain",
            "extra_nram_bytes": 0,
            "pool_channels": 16,
            "integer_count_terms": 16,
            "precision_policy_changed": False,
            "numerical_reduction_changed": False,
        }
        support = "\n\n".join(
            filter(
                None,
                (
                    _bangc_support(context, kernel_prelude),
                    render_c_type_support(
                        body,
                        "__mlu_func__",
                        cast_conversions={
                            "half": "__float2half_rn((float)(value))",
                            "bfloat16_t": "__float2bfloat16_rn((float)(value))",
                        },
                    ),
                ),
            )
        )

        private_parameters = []
        private_arguments = ()

        if private_chunk_needed:
            private_parameters.append(f"int64_t {chunk_parameter}")
            private_arguments = (chunk_parameter,)
            chunk_bound = private_chunk_bound or f"{chunk}LL"
            chunk_decl = (
                f"    const int64_t {chunk_parameter} = ({total}) > 0 && "
                f"({total}) < {chunk_bound} ? ({total}) : {chunk_bound};\n" + chunk_decl
            )

        launch_symbols = allocate_task_launch_symbols(
            kernel_params,
            launch_params,
            kernel_prelude,
            support,
            tasks_expr,
            chunk_decl,
        )
        kernel_params += (",\n" if kernel_params else "") + _render_signature_params(
            [*launch_symbols.parameters, *private_parameters]
        )
        kernel_prelude = launch_symbols.kernel_body(
            kernel_prelude,
            task_pattern=r"\btaskIdX\b",
            physical_index="taskIdX",
            max_tasks=MAX_BLOCK_TASKS,
        )
        launcher = render_segmented_launch(
            kernel.kernel_name,
            (*context.variables, *context.outputs, *context.shape_params),
            tasks_expr,
            launch_symbols,
            extra_private_args=private_arguments,
            queue=queue_name,
        )
        context.scheduled_metadata["bangc_launch"] = {
            "max_block_tasks": MAX_BLOCK_TASKS,
            "logical_tasks": tasks_expr,
            "segmented": True,
            "public_abi_changed": False,
            "generic_chunk": (
                {
                    "requested": default_chunk,
                    "scheduled": chunk,
                    "effective": f"min({chunk}, ({total}))",
                    "private_kernel_argument": chunk_parameter,
                    "task_partition_preserved": True,
                }
                if private_chunk_needed
                else None
            ),
        }

        return f"""// Generated by NineToothed's BangC SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program

#include <bang.h>
#include <cnrt.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>
#include <limits>

{support}

__mlu_entry__ void {kernel.kernel_name}_kernel(
{kernel_params}
) {{
{kernel_prelude}
}}

extern "C" int launch_{kernel.kernel_name}(
{launch_params}
) {{
{chunk_decl}{launcher}
}}
"""


def _render_chunked_scalar_body(
    body, total, chunk_parameter, index_name, *, reserved=""
):
    """Walk only the remaining elements of the unchanged logical task chunk."""
    fragments = [body, total, chunk_parameter, reserved]
    names = []

    for base in ("nt_chunk", "nt_base", "nt_count", "nt_lane"):
        name = allocate_private_symbol(base, *fragments, *names)
        names.append(name)

    chunk, start, count, lane = names

    return f"""    const int64_t {chunk} = {chunk_parameter};
    const int64_t {start} = (int64_t)(taskIdX) * {chunk};
    if ({start} >= ({total})) {{ return; }}
    int64_t {count} = ({total}) - {start};
    if ({count} > {chunk}) {{ {count} = {chunk}; }}
    for (int64_t {lane} = 0; {lane} < {count}; {lane}++) {{
        int64_t {index_name} = {start} + {lane};
{common.indent_block(body, "        ")}
    }}"""


def _want_output_tile_fusion(context: ModuleRenderContext) -> bool:
    """Select a measured tile cost policy before the independent legality proof."""
    if len(context.outputs) != 1:
        return False

    plan = dot_accumulation_plan(context.kernel.ssa)

    if plan is None:
        return False

    try:
        m, n = map(int, context.tensors[context.outputs[0]].attrs["source_shape"])
        left = context.operations[plan.dot.operands[0]]
        k = int(left.results[0].type.attrs["source_shape"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return False

    return (
        m > 0
        and n > 0
        and m % 64 == 0
        and n % 64 == 0
        and m * n >= 65536
        and k >= 64
        and k % 32 == 0
    )


def _program_uses_atomics(context: ModuleRenderContext) -> bool:
    return any(
        operation.opcode == "mem.atomic_add"
        for operation in context.operations.values()
    )


_ELEMENTWISE_PREFIXES = (
    "arith.",
    "math.",
    "cmp.",
    "select.",
    "call.",
)


def _elementwise_only(context: ModuleRenderContext) -> bool:
    """Return whether every SSA operation is elementwise over the flat domain."""
    for operation in context.operations.values():
        opcode = operation.opcode

        if opcode in {"mem.load", "mem.store", "mem.data_ptr", "index.offset"}:
            continue

        if opcode.startswith(_ELEMENTWISE_PREFIXES):
            continue

        if opcode in {
            "tensor.zeros",
            "tensor.empty",
            "tensor.full",
            "tensor.cast",
            "tensor.extract",
            "tensor.view",
        }:
            continue
        return False
    return True


@dataclass(frozen=True)
class _TaskDomain:
    body: str
    total: str
    grid_total: str
    guard: str | None
    staging: Any
    default_chunk: int
    chunk: int
    scalar_chunk: int
    alias_parameters: frozenset[str]
    uses_atomics: bool
    private_parameters_safe: bool


@dataclass(frozen=True)
class _StagingPlan:
    inputs: tuple
    outputs: tuple
    body: str
    extent: str
    buffer_types: dict
    dag: Any = None


def _bind_fallback_tasks(plan, tasks):
    if plan is None:
        return None
    return replace(plan, tasks=plan.tasks.replace("nt_fallback_tasks", tasks))


def _staging_alias_parameters(context, body):
    """Keep original scalar fallbacks valid for strongly matched staged graphs."""
    names = set()

    if _elementwise_only(context):
        flat = flatten_tiled_accesses(body, context, allow_value_select=True)

        if flat is not None:
            names.update(
                name
                for name in (*context.variables, *context.outputs)
                if re.search(rf"\b{re.escape(name)}\[index\]", flat)
                and context.tensors[name].ndim != 0
            )

    for match in (
        match_row_reduce(context),
        match_reduce_broadcast(context),
    ):
        if match is not None:
            names.update((match[1], match[2]))

    row_plans = (
        context.lowering_plans.get("cooperative_rows"),
        context.lowering_plans.get("alias_rows"),
        context.lowering_plans.get("full_rows"),
    )

    if any(plan is not None for plan in row_plans):
        names.update(
            name
            for name in (*context.variables, *context.outputs)
            if context.tensors[name].ndim != 0
        )

    return names


_NRAM_BUDGET_BYTES = 192 * 1024


def _nram_halving_reduce(operator: str) -> list[str]:
    """Halve aligned vectors without dropping the unaligned remainder."""
    vector = {
        "sum": "__bang_add",
        "max": "__bang_nan_maximum",
        "min": "__bang_nan_minimum",
    }[operator]
    lhs = "nt_wrk[nt_j - 2 * nt_h]"
    rhs = "nt_wrk[nt_j]"
    scalar = (
        f"({lhs}) + ({rhs})" if operator == "sum" else f"nt_{operator}({lhs}, {rhs})"
    )

    return [
        "    while (nt_len >= 64) {",
        "        int64_t nt_h = (nt_len / 2) & ~7LL;",
        "        if (nt_h < 32) { break; }",
        f"        {vector}(nt_wrk, nt_wrk, nt_wrk + nt_h, (uint32_t)nt_h);",
        "        for (int64_t nt_j = 2 * nt_h; nt_j < nt_len; nt_j++) {",
        f"            {lhs} = {scalar};",
        "        }",
        "        nt_len = nt_h;",
        "    }",
    ]


def _nram_staging_plan(context: ModuleRenderContext, body: str) -> _StagingPlan | None:
    """Plan NRAM staging for pure contiguous float32 elementwise kernels.

    The generic scalar path reaches ~0.5 GB/s because every element access
    hits GDRAM directly; staging each tensor into an ``__nram__`` buffer via
    ``__memcpy`` and running the same per-element body on NRAM measures
    ~15 GB/s, and trivially mappable single-op kernels reach ~1.5 TB/s with
    ``__bang_*`` tensor instructions.  Staging applies only when every tensor
    access in the rendered body is the flat ``name[index]`` form (either the
    shared emitter's contiguous fast path or a tiled form whose predicates
    are provably subsumed by the outer bounds guard) so the rewrite is
    purely mechanical.
    """
    if not _elementwise_only(context):
        return None

    if "taskIdX" in body:
        return None

    typed_dag = bangc_vector.plan_flat(
        context,
        tuple(
            (name, "") for name in context.variables if context.tensors[name].ndim != 0
        ),
        tuple((name, "") for name in context.outputs),
    )
    body = flatten_tiled_accesses(
        body, context, allow_value_select=typed_dag is not None
    )

    if body is None:
        return None

    input_names = set(context.variables)
    output_names = set(context.outputs)
    rewritten = body
    inputs: list[tuple[str, str]] = []
    outputs: list[tuple[str, str]] = []

    for name in (*context.variables, *context.outputs):
        info = context.tensors.get(name)

        if info is None or info.ndim == 0:
            continue

        source_shape = (
            (info.attrs or {}).get("source_shape") or info.source_shape or info.shape
        )

        if info.ndim != 1 or len(source_shape) != 1:
            return None

        if _normalize_dtype(info.dtype) not in {"float32", "float16", "int32"}:
            return None

        direct = re.compile(rf"\b{re.escape(name)}\[index\]")

        if direct.search(rewritten) is None:
            continue

        buffer = f"nt_buf_{name}"
        rewritten = direct.sub(f"{buffer}[nt_j]", rewritten)

        if name in output_names:
            outputs.append((name, buffer))

        if name in input_names:
            inputs.append((name, buffer))

    if not outputs:
        return None

    # The flat domain is tile-padded (ceil(size / BLOCK) * BLOCK); staging
    # must clamp to the tensors' real element count or the memcpy and the
    # bang op read/write past the allocation.  Require every staged tensor
    # to expose the same size expression and clamp to it.
    limit_expr = _staged_extent_expression(context, (*inputs, *outputs))

    if limit_expr is None:
        return None

    residue = rewritten

    for _, buffer in (*inputs, *outputs):
        residue = residue.replace(f"{buffer}[nt_j]", "")

    if "index" in residue:
        return None

    # Staging is only valid for provably contiguous flat accesses, so any
    # contiguous fast-path guards on the staged tensors' strides fold away;
    # keep their symbols out of the kernel (they may not be launch params).
    for name, _ in (*inputs, *outputs):
        rewritten = re.sub(rf"\b{name}_stride_\d+\b", "1", rewritten)

    dag_usable = (
        typed_dag is not None
        and all(not store.value.scalar for store in typed_dag.stores)
        and typed_dag.bytes_per_element * 128 <= _NRAM_BUDGET_BYTES
    )

    if (
        any(
            _normalize_dtype(context.tensors[name].dtype) != "float32"
            for name, _ in (*inputs, *outputs)
        )
        and not dag_usable
    ):
        return None

    return _StagingPlan(
        tuple(inputs),
        tuple(outputs),
        rewritten,
        limit_expr,
        {
            buffer: _bangc_type(context.tensors[name].dtype)
            for name, buffer in (*inputs, *outputs)
        },
        typed_dag if dag_usable else None,
    )


def _is_staged_tensor(context, name: str) -> bool:
    info = context.tensors.get(name)

    return info is not None and info.ndim != 0


def _staged_extent_expression(
    context: ModuleRenderContext, staged: list[tuple[str, str]]
) -> str | None:
    """Return one shared runtime extent for all staged tensors."""
    extents: set[str] = set()

    for name, _ in staged:
        info = context.tensors.get(name)

        if info is None:
            return None

        attrs = info.attrs or {}
        source_shape = attrs.get("source_shape") or info.shape

        if not source_shape:
            return None

        extents.add(str(source_shape[0]))

    if len(extents) == 1:
        return extents.pop()

    # Unsized specializations name each tensor's extent parameter
    # separately (``<tensor>_size_0``), but a 1-D elementwise kernel
    # launches with equal lengths, so distinct size parameters are one
    # shared runtime extent.
    if all(re.fullmatch(r"[\w]*size_0", expr) for expr in extents):
        return sorted(extents)[0]

    return None


# Public vector math primitives preserve their scalar counterparts'
# domains, including negative reciprocal inputs and nonfinite values.
# The approximate active instructions remain confined to fast_math.


def _bangc_float_literal(value: float) -> str:
    if math.isnan(value):
        return "NAN"

    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"

    text = f"{value:.9g}"

    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def _staged_identity(plan: _StagingPlan) -> bool:
    return (
        len(plan.inputs) == 1
        and len(plan.outputs) == 1
        and "".join(plan.body.split())
        == f"{plan.outputs[0][1]}[nt_j]={plan.inputs[0][1]}[nt_j];"
    )


def _render_nram_staged_body(
    plan: _StagingPlan, total: str, chunk: int, *, target=None
) -> str:
    if target is None:
        target = TARGET

    extent = _bangc_integer_expr(plan.extent)

    if plan.dag is not None:
        return bangc_vector.render_flat(
            plan.dag, extent, chunk, target, _bangc_float_literal, _nram_halving_reduce
        )

    identity = _staged_identity(plan)
    lines: list[str] = []
    lines.append(f"    const int64_t nt_chunk = {chunk};")
    lines.append("    const int64_t nt_base = (int64_t)(taskIdX) * nt_chunk;")
    lines.append(f"    if (nt_base >= {extent}) {{ return; }}")
    lines.append(f"    int64_t nt_cnt = {extent} - nt_base;")
    lines.append("    if (nt_cnt > nt_chunk) { nt_cnt = nt_chunk; }")

    for _, buffer in (*plan.inputs, *(plan.outputs if not identity else ())):
        type_name = plan.buffer_types.get(buffer, "float")
        lines.append(f"    __nram__ {type_name} {buffer}[{chunk}];")

    for name, buffer in plan.inputs:
        lines.append(
            f"    __memcpy({buffer}, {name} + nt_base, "
            f"(uint32_t)(nt_cnt) * sizeof(float), GDRAM2NRAM);"
        )

    if not identity:
        lines.append("    for (int64_t nt_j = 0; nt_j < nt_cnt; nt_j++) {")
        lines.extend(common.indent_block(plan.body, "        ").splitlines())
        lines.append("    }")

    for name, buffer in plan.outputs:
        if identity:
            buffer = plan.inputs[0][1]

        type_name = plan.buffer_types.get(buffer, "float")
        lines.append(
            f"    __memcpy({name} + nt_base, {buffer}, "
            f"(uint32_t)(nt_cnt) * sizeof({type_name}), NRAM2GDRAM);"
        )

    return "\n".join(lines)


def _atomic_suffix(dtype: str | None) -> str | None:
    dtype = common.normalize_dtype(dtype)

    return {
        "float32": "f32",
        "float16": "f16",
        "bfloat16": "bf16",
        "int32": "i32",
        "int64": "i64",
        "uint32": "i32",
        "uint64": "i64",
    }.get(dtype)


def _bangc_support(context: ModuleRenderContext, rendered_body: str = "") -> str:
    """Emit helper functions required by the kernel body."""
    operations = {id(operation): operation for operation in context.operations.values()}
    lines: list[str] = []

    if "nt_cast_int32(" in context.body:
        lines.append(render_int32_cast_support("__mlu_func__"))

    if "_nt_checked_numeric(" in context.body:
        lines.append(
            """template <class T> static __mlu_func__ T _nt_checked_numeric(T value) {
    static_assert(!std::is_integral<T>::value || sizeof(T) <= 4,
                  "BangC 64-bit numerical values truncate to 48 bits on MLU590 with cncc 5.1.2");
    return value;
}"""
        )

    if any(operation.opcode == "arith.div" for operation in operations.values()):
        # MLU590's public header has vector division but no div_scalar.
        # Keep true division in NRAM rather than rounding a reciprocal
        # constant or emitting a nonexistent intrinsic.
        lines.append(
            """static __mlu_func__ void nt_bang_div_scalar(
    float *dst, const float *src, float scalar, uint32_t count
) {
    for (uint32_t i = 0; i < count; ++i) { dst[i] = src[i] / scalar; }
}"""
        )

    if any(
        operation.opcode in {"math.sin", "math.cos", "call.sin", "call.cos"}
        for operation in operations.values()
    ):
        lines.append(
            "union NtFloatBits { float f; int32_t i; };\n"
            "static __mlu_func__ float nt_inf() { return INFINITY; }\n"
            + render_float32_trig_support("__mlu_func__")
        )

    if any(operation.opcode == "math.rand" for operation in operations.values()):
        # NOTE: 64-bit immediate constants are truncated on the MLU590 scalar
        # path, so the generator is kept entirely in uint32 arithmetic.
        lines.append(
            """static __mlu_func__ float ninetoothed_rand_uniform(
    uint64_t seed, uint64_t offset
) {
    uint32_t z = (uint32_t)(seed) ^ ((uint32_t)(offset) * 0x9E3779B1u);
    z = z + 0x6D2B79F5u;
    z = (z ^ (z >> 15)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z = z ^ (z >> 16);
    return (float)(z >> 8) * (1.0f / 16777216.0f);
}"""
        )

    needs_extremum = bool(re.search(r"\bnt_(?:max|min)\s*\(", rendered_body)) or any(
        operation.opcode.startswith("reduce.")
        or operation.opcode in {"arith.maximum", "arith.minimum"}
        or operation.opcode in {"math.max", "math.min"}
        or operation.opcode
        in {
            "math.exp",
            "math.exp2",
            "math.sin",
            "math.cos",
            "math.tanh",
            "math.sigmoid",
            "call.exp",
            "call.exp2",
            "call.sin",
            "call.cos",
            "call.tanh",
            "call.sigmoid",
            "math.erf",
            "call.erf",
        }
        for operation in operations.values()
    )

    if needs_extremum:
        # Integer comparisons must not round-trip through fmaxf/fminf,
        # which loses precision beyond 2^24; float keeps NaN semantics.
        # cncc misresolves same-name __mlu_func__ overloads, so extremums
        # are type-generic macros: integer comparisons stay exact and the
        # ternary preserves a NaN operand for floats, matching vector
        # nan_maximum/nan_minimum and the language's extrema.
        lines.append(
            """#define nt_max(a, b) ((a) != (a) ? (a) : ((b) != (b) ? (b) : (((a) > (b)) ? (a) : (b))))
#define nt_min(a, b) ((a) != (a) ? (a) : ((b) != (b) ? (b) : (((a) < (b)) ? (a) : (b))))"""
        )

    atomic_dtypes = set()

    for operation in operations.values():
        if operation.opcode != "mem.atomic_add" or not operation.operands:
            continue

        operand_type = context.value_types.get(operation.operands[-1])
        atomic_dtypes.add(
            _normalize_dtype(operand_type.dtype if operand_type else None)
        )

    # NOTE: every `__bang_atomic_add` spelling (C intrinsics and raw
    # `atom.add.scalar` assembly, block and union task types) traps with
    # CN_INVOKE_ERROR on MLU590, so the update degrades to a plain GDRAM
    # read-modify-write.  Atomic kernels are therefore serialized onto a
    # single task by `render_module`, which keeps the accumulation exact.
    for dtype in sorted(atomic_dtypes):
        variant = _atomic_suffix(dtype)

        if variant is None:
            continue

        address_type = {
            "f32": "float",
            "f16": "half",
            "bf16": "bfloat16_t",
            "i32": "int32_t",
            "i64": "int64_t",
        }[variant]

        # Integer updates must stay in integer arithmetic: routing the
        # value through float32 loses every increment past 2^24.
        value_type = {
            "f32": "float",
            "f16": "float",
            "bf16": "float",
            "i32": "int32_t",
            "i64": "int64_t",
        }[variant]

        if variant in {"f16", "bf16"}:
            load = (
                "__bfloat162float(address[0])"
                if variant == "bf16"
                else "__half2float(address[0])"
            )
            store = "__float2bfloat16_rn" if variant == "bf16" else "__float2half_rn"
            body = f"""    float nt_prev = {load};
    address[0] = {store}(nt_prev + value);"""
            lines.append(
                f"""static __mlu_func__ void ninetoothed_atomic_add_{variant}(
    {address_type}* address, {value_type} value
) {{
{body}
}}"""
            )
        elif variant in {"i32", "i64"}:
            # The call site picks the variant from the update value's dtype,
            # which may differ from the buffer's; deduce the address type
            # and accumulate in the buffer's own integer width.
            lines.append(
                f"""template <typename T>
static __mlu_func__ void ninetoothed_atomic_add_{variant}(
    T* address, {value_type} value
) {{
    address[0] = (T)(address[0] + value);
}}"""
            )
        else:
            body = """    address[0] = address[0] + value;"""
            lines.append(
                f"""static __mlu_func__ void ninetoothed_atomic_add_{variant}(
    {address_type}* address, {value_type} value
) {{
{body}
}}"""
            )

    return "\n\n".join(lines)


def _render_signature_params(params: list[str]) -> str:
    return ",\n".join(f"    {param}" for param in params)


def _signature_param(name, info, *, readonly: bool, restrict: bool) -> str:
    dtype = _bangc_type(info.dtype)

    if info.ndim == 0:
        return f"{dtype} {name}"

    const = "const " if readonly else ""
    qualifier = " __restrict__" if restrict else ""

    return f"{const}{dtype}*{qualifier} {name}"


def _auxiliary_param(name, tensors) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _signature_param(name, info, readonly=True, restrict=False)
    return f"int64_t {name}"


def _bangc_type(dtype: str | None, kind: str | None = None) -> str:
    dtype = common.normalize_dtype(dtype)

    if kind == "pointer":
        return f"{_bangc_type(dtype)}*"

    if kind == "index" or dtype in {"index", "int64"}:
        return "int64_t"

    types = {
        "float32": "float",
        "float16": "half",
        "bfloat16": "bfloat16_t",
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

    if dtype in {"float8_e4m3fn", "float8_e5m2"}:
        raise ValueError(
            "The BangC backend does not support float8 dtypes on this toolchain."
        )

    if dtype not in types:
        raise ValueError(f"Unsupported BangC SSA dtype: {dtype!r}.")
    return types[dtype]


_bangc_integer_expr = common.c_integer_expr


def _validate_numeric_result_type(op: ssa.Operation, result_type: ssa.Type) -> None:
    if (
        (
            _normalize_dtype(result_type.dtype) in {"int64", "uint64"}
            or (
                _normalize_dtype(result_type.dtype) == "index"
                and op.opcode == "tensor.cast"
            )
        )
        and result_type.kind not in {"index", "pointer"}
        and op.opcode != "arith.constant"
    ):
        from ninetoothed.targets import UnsupportedTargetCapabilityError

        raise UnsupportedTargetCapabilityError(
            f"BangC cannot execute the 64-bit integer numerical result of "
            f"{op.opcode!r} on MLU590 with cncc 5.1.2; scalar values truncate "
            "to 48 bits."
        )


def _bangc_arithmetic_result_type(op: ssa.Operation, ctx: _EmitContext) -> ssa.Type:
    result_type = op.results[0].type
    _validate_numeric_result_type(op, result_type)

    if not op.opcode.startswith("arith."):
        return result_type

    if _normalize_dtype(result_type.dtype) not in {
        "float16",
        "bfloat16",
    }:
        return result_type
    return replace(result_type, dtype="float32")


def _coerce_bangc_binary_args(
    op: ssa.Operation, args: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, ...]:
    if len(args) != len(op.operands):
        return args

    result_type = op.results[0].type if op.results else None

    if result_type is not None and result_type.dtype is None:
        # Let C++ infer the actual promotion. Choosing a known integer
        # constant's dtype while another operand is unknown narrows even a
        # floating input to integer. check_numeric_result validates the
        # resulting computed type, including inline tensor expressions.
        return args

    if result_type is not None and result_type.kind == "pointer":
        return args

    if any(
        (type_ := ctx.value_types.get(operand)) is not None and type_.kind == "pointer"
        for operand in op.operands
    ):
        return args

    dtype = (
        _normalize_dtype(result_type.dtype)
        if result_type is not None and _normalize_dtype(result_type.dtype) != "bool"
        else _bangc_common_operand_dtype(op.operands, ctx)
    )

    if dtype is None:
        return args

    if dtype in {"float16", "bfloat16"}:
        dtype = "float32"

    coerced = []

    for operand, value in zip(op.operands, args):
        operand_type = ctx.value_types.get(operand)
        operand_dtype = _normalize_dtype(
            operand_type.dtype if operand_type is not None else None
        )
        coerced.append(
            ctx.target.cast(dtype, value) if operand_dtype != dtype else value
        )
    return tuple(coerced)


def _bangc_common_operand_dtype(
    operands: tuple[str, ...], ctx: _EmitContext
) -> str | None:
    ranks = {
        "bool": 0,
        "int32": 1,
        "int64": 2,
        "float16": 3,
        "bfloat16": 3,
        "float32": 4,
        "float64": 5,
    }
    dtypes = [
        _normalize_dtype(type_.dtype)
        for operand in operands
        if (type_ := ctx.value_types.get(operand)) is not None
    ]

    return max(dtypes, key=lambda dtype: ranks.get(dtype, -1)) if dtypes else None


_emit_bangc_direct_dot_operand = common.emit_direct_dot_operand


def _emit_bangc_cooperative_reduction(
    local: str, operation: ssa.Operation, ctx: _EmitContext
) -> str:
    """Reduce the full extent inside one task; no cross-thread steps needed."""
    schedule = dict(ctx.reduction_schedule or {})
    axis = int(schedule["axis"]) if "axis" in schedule else None

    if axis is None or not operation.operands or not operation.results:
        raise ValueError("Malformed BangC cooperative reduction operation.")

    operator = operation.opcode[len("reduce.") :]
    operand = operation.operands[0]
    operand_axes = _value_axes(operand, ctx)
    extent = str(schedule.get("extent", operand_axes[axis]))
    operand_type = ctx.value_types.get(operand)
    result_dtype = _normalize_dtype(operation.results[0].type.dtype or "float32")
    operand_dtype = _normalize_dtype(
        operand_type.dtype if operand_type is not None else result_dtype
    )
    accumulator_dtype = _bangc_reduction_accumulator_dtype(operand_dtype, result_dtype)
    accumulator_type = ssa.Type(kind="scalar", dtype=accumulator_dtype)
    accumulator = _fresh_temp(ctx, "nt_reduce_acc")
    reduction_index = _fresh_temp(ctx, "nt_reduce_index")
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
    body_lines.append(
        f"{accumulator} = "
        f"{_bangc_reduction_update(operator, accumulator, term, accumulator_dtype)};"
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
    result_type = ssa.Type(kind="scalar", dtype=result_dtype)
    ctx.lines.append(
        ctx.target.local_decl(
            result_type, local, ctx.target.cast(result_dtype, accumulator)
        )
    )

    return local


def _bangc_reduction_accumulator_dtype(operand_dtype: str, result_dtype: str) -> str:
    if operand_dtype in {"float16", "bfloat16"}:
        return "float32"

    if operand_dtype == "bool":
        return "int32"
    return result_dtype


def _bangc_reduction_update(operator: str, lhs: str, rhs: str, dtype: str) -> str:
    del dtype

    if operator == "sum":
        return f"({lhs}) + ({rhs})"

    # The same macro keeps integer comparisons exact and floating NaNs.
    # A computed dtype is a C++ expression, so it cannot be classified by
    # comparing its Python spelling with primitive dtype names.
    function = "nt_max" if operator == "max" else "nt_min"

    return f"{function}({lhs}, {rhs})"


TARGET = BangCTarget()


def _validate_integer_data_capability(kernel: Kernel) -> None:
    from ninetoothed.targets import UnsupportedTargetCapabilityError

    def operations(blocks):
        for block in blocks:
            for operation in block.operations:
                yield operation
                yield from operations(operation.regions)

    program = kernel.ssa
    uses = tuple(operations(program.blocks)) if program is not None else ()

    for operation in uses:
        for result in operation.results:
            _validate_numeric_result_type(operation, result.type)

    for tensor in kernel.tensors:
        if _normalize_dtype(tensor.dtype) not in {"int64", "uint64", "index"}:
            continue

        tensor_uses = tuple(op for op in uses if tensor.name in op.operands)

        if (
            tensor.ndim == 0
            and tensor_uses
            and all(
                op.opcode == "math.rand"
                and all(
                    index in {0, 1}
                    for index, operand in enumerate(op.operands)
                    if operand == tensor.name
                )
                for op in tensor_uses
            )
        ):
            # The deterministic generator explicitly consumes only uint32
            # seed/offset bits. Truncation of their upper bits cannot affect it.
            continue

        raise UnsupportedTargetCapabilityError(
            f"BangC cannot execute {tensor.dtype!r} data argument "
            f"{tensor.name!r} on MLU590 with cncc 5.1.2; even scalar loads "
            "and stores truncate to 48 bits. Use a supported numerical dtype."
        )


def emit(kernel: Kernel):
    _validate_integer_data_capability(kernel)

    return common.emit(kernel, TARGET)


__all__ = ["BangCTarget", "TARGET", "emit"]
