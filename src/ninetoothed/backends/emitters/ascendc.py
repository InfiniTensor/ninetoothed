"""AscendC syntax hooks for the common SSA emitter.

The Huawei AscendC task model maps one block per ``GetBlockIdx()`` on the
AI cores of Ascend 910B devices.  Kernels are declared ``__global__
__aicore__`` with ``GM_ADDR`` device pointers, host launchers use the
``<<<blocks, workspace, stream>>>`` trigram against ``aclrtStream``
streams, and the whole unit compiles with the CANN ``ccec`` compiler into
a shared library exposing the same C-ABI launcher contract as the CUDA and
BangC backends.
"""

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import walk_ops as _walk_ops
from ninetoothed.backends.emitters.ascendc_launch import (
    launch_symbols,
    render_native_vector_kernel,
    render_segmented_launch,
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
from ninetoothed.backends.emitters.task_launch import allocate_private_symbol
from ninetoothed.ir import Kernel, ssa

_ascendc_integer_expr = common.c_integer_expr
_emit_ascendc_direct_dot_operand = common.emit_direct_dot_operand
_ascendc_guarded_body = common.guarded_body
_emit_element = common.emit_element
_fresh_temp = common.fresh_temp
_indent_lines = common.indent_lines
_linearized_index = common.linearized_index
_nested_local_suffix = common.nested_local_suffix
_normalize_dtype = common.normalize_dtype
_reduction_identity = common.reduction_identity
_target_index_expr = common.target_index_expr
_value_axes = common.value_axes


_CUBE_PRIVATE_NAMES = frozenset(
    (
        "a1",
        "b1",
        "a2",
        "b2",
        "c0",
        "a_gm",
        "b_gm",
        "c_gm",
        "copy_a",
        "copy_b",
        "load_a",
        "load_b",
        "mm",
        "fix",
    )
)


def _ascendc_private_conflicts(context):
    # These generated public namespaces never declare adapter-local values.
    return private_parameter_conflicts(
        context,
        ("nt_",),
        excluded_prefixes=("nt_materialized_", "nt_dot_partials"),
    )


def _ascendc_cube_conflicts(context):
    parameters = dict.fromkeys(
        (*context.variables, *context.outputs, *context.shape_params)
    )

    return tuple(name for name in parameters if name in _CUBE_PRIVATE_NAMES)


def _ascendc_optimization_context(context):
    """Keep fixed private adapter names out of the public parameter scope."""
    conflicts = _ascendc_private_conflicts(context)
    cube_conflicts = _ascendc_cube_conflicts(context)

    if not conflicts and not cube_conflicts:
        return context

    candidates = (
        "ascendc_ordered_dot",
        "ascendc_ordered_dot_add",
        "ascendc_cube_dot",
        "ascendc_indexed_transfer",
        "ascendc_transpose_transfer",
        "ascendc_affine_transfer",
        "ascendc_full_row_dag",
        "ascendc_uniform_store",
    )

    if not conflicts:
        candidates = candidates[:3]

    metadata = context.scheduled_metadata

    for name in candidates:
        metadata.pop(name, None)

    metadata["ascendc_private_namespace"] = {
        "eligible": False,
        "conflicts": conflicts or cube_conflicts,
        "blocked_candidates": candidates,
        "reason": "public-parameter-overlaps-private-adapter-namespace",
        "fallback": "complete-original-kernel-and-launch-grid",
    }

    return context


def _ascendc_fallback_symbol(context, name):
    """Avoid shadowing public values in the retained indexed loops."""
    names = set(
        (*context.variables, *context.outputs, *getattr(context, "shape_params", ()))
    )
    names.update(re.findall(r"\b[A-Za-z_]\w*\b", getattr(context, "body", "")))
    candidate = name
    suffix = 0

    while candidate in names:
        suffix += 1
        candidate = f"{name}_{suffix}"

    return candidate


def _ascendc_serial_fallback(body, tensor_casts, total, index_name):
    return (
        "\n".join(tensor_casts)
        + f"""
    if (GetBlockIdx() != 0) {{ return; }}
    for (int64_t {index_name} = 0; {index_name} < {total}; {index_name}++) {{
{common.indent_block(body, "        ")}
    }}"""
    )


def _prepare_value_routes(context):
    """Choose expression domains once; renderers consume the selected plans."""
    from ninetoothed.backends.ascendc_resources import (
        configure_basic_division_strategy,
        configure_basic_validity_summary,
        configure_basic_vector,
        configure_serial_queue_storage,
    )
    from ninetoothed.backends.emitters.ascendc_reduce import (
        match_reduce_broadcast,
    )
    from ninetoothed.backends.emitters.ascendc_vector import match_vector_elementwise
    from ninetoothed.backends.emitters.ascendc_vector_dag import (
        match_flat_dag,
        match_row_dag,
    )

    context.lowering_plans["reduce_broadcast"] = match_reduce_broadcast(context)
    context.lowering_plans["row_dag"] = match_row_dag(context)
    vector_match = match_vector_elementwise(context, _normalize_dtype)
    total_param = _elementwise_total_param(context)
    flat_match = None

    if (
        vector_match is None
        or len(vector_match.stages) > 4
        or (
            len(vector_match.stages) > 1
            and any(operation == "Tanh" for operation, _, _ in vector_match.stages)
        )
    ):
        planned_flat = match_flat_dag(context)

        if vector_match is None or (
            planned_flat is not None
            and (
                (
                    len(vector_match.stages) > 4
                    and all(
                        step.node.result.scalar or step.node.lowering.mode != "scalar"
                        for step in planned_flat.plan.steps
                    )
                )
                or any(
                    step.node.lowering.mode == "bounded_tanh"
                    for step in planned_flat.plan.steps
                )
            )
        ):
            flat_match = planned_flat
            vector_match = None

    context.lowering_plans.update(
        packet=vector_match, flat=flat_match, packet_total=total_param
    )

    if vector_match is not None and total_param is not None:
        configure_serial_queue_storage(context, vector_match)
        resources = configure_basic_vector(context, vector_match)
        context.lowering_plans["packet_resources"] = resources
        configure_basic_validity_summary(context, vector_match)
        configure_basic_division_strategy(context, vector_match)
        consumers = vector_match.bound_consumers()

        if consumers:
            context.scheduled_metadata["ascendc_numeric_bounds"] = {
                "path": "basic-vector",
                "consumers": consumers,
                "repair": "upper-bound-and-NaN-Inf-with-original-scalar-tail",
            }


def _plan_selected_row_execution(context):
    from ninetoothed.backends.emitters.ascendc_vector_dag import plan_row_execution

    plans = context.lowering_plans

    if plans.get("uniform") is not None:
        return

    full_rows = plans.get("full_rows")

    if full_rows is not None:
        derived = replace(
            full_rows.tiles.context, scheduled_metadata=context.scheduled_metadata
        )
        plans["full_rows"] = replace(
            full_rows, dag=plan_row_execution(derived, full_rows.dag)
        )
    elif plans.get("row") is None:
        key = "row_dag" if context.cooperative_reduction_program else "flat"

        if plans.get(key) is not None:
            plans[key] = plan_row_execution(
                context,
                plans[key],
                route="typed-flat-dag" if key == "flat" else "typed-row-dag",
            )


@dataclass(frozen=True, kw_only=True)
class AscendCTarget(CComputedTypesMixin, EmitterTarget):
    backend: Target = Target.ASCENDC
    language: str = "ascendc/c++"
    suffix: str = "ascendc"
    source_route: str = "ssa-unified-ascendc-emitter"
    c_style_syntax: bool = True

    def index_cast(self, value: str) -> str:
        return f"(int64_t)({value})"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value};"

    def literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return f"{self.global_helper('nt_nan')}()"

        if isinstance(value, float) and math.isinf(value):
            function = self.global_helper("nt_inf")

            return f"{function}()" if value > 0 else f"-{function}()"

        if value == "inf":
            return f"{self.global_helper('nt_inf')}()"

        if value == "-inf":
            return f"-{self.global_helper('nt_inf')}()"

        if isinstance(value, float):
            # Ccec treats unsuffixed float literals as double, which is not
            # supported in aicore functions.
            return f"{value}f"

        if isinstance(value, int):
            if value == -(1 << 63):
                # A bare decimal 2^63 has unsigned type before unary minus.
                # In a masked term it would promote signed comparisons.
                return "(-9223372036854775807LL - 1LL)"

            return repr(value)

        return repr(value)

    def cast(self, dtype, value):
        normalized = _normalize_dtype(dtype)

        if normalized == "int32":
            return f"{self.global_helper('nt_cast_int32')}({value})"

        if normalized == "float16":
            return f"((half)(float)({value}))"

        return f"(({self.type_name(dtype)})({value}))"

    def type_name(self, dtype, kind=None):
        return _ascendc_type(dtype, kind)

    def where(self, cond, yes, no):
        no = _ascendc_float_suffix(no)

        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "load" and args:
            return f"*({args[0]})"

        if name in {"block_dot", "dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "rand" and len(args) >= 2:
            return f"{self.global_helper('ninetoothed_rand_uniform')}(({args[0]}), ({args[1]}))"

        if name == "sqrt" and len(args) == 1:
            return f"sqrt(({args[0]}))"

        function = _ASCENDC_CALL_FUNCS.get(name, name)

        if name in _ASCENDC_CALL_FUNCS and function.startswith("nt_"):
            function = self.global_helper(function)

        return f"{function}({', '.join(args)})"

    def supports_cooperative_reduction(self, schedule):
        return bool(schedule.get("ascendc_cooperative_reduction"))

    def thread_id(self):
        # One block is one serial program; the cooperative lowering uses a
        # single lane so thread-strided loops degenerate to full coverage.
        return "0"

    def thread_count(self):
        return "1"

    def emit_cooperative_reduction(self, local, operation, context):
        return _emit_ascendc_cooperative_reduction(local, operation, context)

    def program_id(self, axis=0):
        if axis != 0:
            raise ValueError("AscendC kernels use a one-dimensional block grid.")
        return "(int64_t)GetBlockIdx()"

    def emit_dot_operand(self, name, coords, context):
        return _emit_ascendc_direct_dot_operand(name, coords, context)

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "max" if operator == "max" else "min"

        return f"{function}({acc}, {term})"

    def reduce_update_typed(self, operator, acc, term, dtype):
        if dtype is None:
            return super().reduce_update_typed(operator, acc, term, dtype)
        return _ascendc_reduction_update(
            operator,
            acc,
            term,
            _ascendc_reduction_accumulator_dtype(dtype, dtype),
            target=self,
        )

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        return _ascendc_arithmetic_result_type(operation, context)

    def coerce_binary_args(self, operation, args, context):
        return _coerce_ascendc_binary_args(operation, args, context)

    def schedule_context(self, context):
        if not self.optimizes_native_execution(context.kernel):
            return context

        from ninetoothed.backends.ascendc_resources import ASCENDC_UB_BYTES

        context = replace(
            context,
            scheduled_metadata={
                **context.scheduled_metadata,
                "ascendc_ub_budget": {
                    "usable_bytes": ASCENDC_UB_BYTES,
                    "reserved_bytes": 256,
                    "source": "CANN PlatformAscendC(Ascend910B2C).GetCoreMemSize(UB)",
                },
            },
        )
        context = _ascendc_optimization_context(context)

        from ninetoothed.backends.emitters.ascendc_reduce import match_row_reduce

        context.lowering_plans["row"] = match_row_reduce(context)

        if _ascendc_private_conflicts(context):
            return context

        _prepare_value_routes(context)

        if not _ascendc_cube_conflicts(context):
            from ninetoothed.backends.emitters.ascendc_dot_partials import (
                render_ordered_dot_partials,
            )
            from ninetoothed.compiler.dot_materialization import (
                plan_ordered_dot_materialization,
            )

            partials = render_ordered_dot_partials(context, self)
            context.lowering_plans["partials"] = partials

            if partials is not None:
                return replace(
                    context,
                    scheduled_metadata={
                        **context.scheduled_metadata,
                        **partials.metadata,
                    },
                )

            from ninetoothed.backends.emitters.ascendc_dot_ordered_add import (
                render_ordered_dot_add,
            )

            ordered_add = render_ordered_dot_add(context, self)
            context.lowering_plans["ordered_add"] = ordered_add

            if ordered_add is not None:
                return replace(
                    context,
                    scheduled_metadata={
                        **context.scheduled_metadata,
                        **ordered_add.metadata,
                    },
                )

            ordered = plan_ordered_dot_materialization(
                context.kernel.ssa, context.kernel.tensors
            )

            if ordered is not None:
                context.execution_plans["ordered_dot"] = ordered

                return replace(
                    context,
                    scheduled_metadata={
                        **context.scheduled_metadata,
                        "ascendc_ordered_dot": {
                            "enabled": True,
                            "selection": "whole-call-runtime-layout-guard",
                            **ordered.metadata(),
                        },
                    },
                )

        from ninetoothed.backends.emitters.ascendc_indexed import (
            render_indexed_transfer,
        )

        indexed = render_indexed_transfer(context)
        context.lowering_plans["indexed"] = indexed

        if indexed is not None:
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "ascendc_indexed_transfer": {
                        "enabled": True,
                        "selection": "runtime-layout-guard-and-actual-address-batches",
                        "guard": indexed.guard,
                        "tasks": indexed.tasks,
                        "chunk": indexed.chunk,
                        "ub_bytes": indexed.ub_bytes,
                        "prefetch_dma": indexed.prefetch_dma,
                        "prefetch_batch": indexed.prefetch_batch,
                        "prefetch_min_extent": indexed.prefetch_min_extent,
                        "fallback": "complete-original-kernel-and-launch-grid",
                    },
                },
            )

        from ninetoothed.backends.emitters.ascendc_transpose import (
            render_transpose_transfer,
        )

        transpose = render_transpose_transfer(context)
        context.lowering_plans["transpose"] = transpose

        if transpose is not None:
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "ascendc_transpose_transfer": {
                        "enabled": True,
                        "selection": "runtime-layout-guard",
                        "guard": transpose.guard,
                        "tasks": transpose.tasks,
                        "ub_bytes": transpose.ub_bytes,
                        "tile_rows": transpose.tile_rows,
                        "tile_columns": transpose.tile_columns,
                        "fallback": "complete-original-kernel-and-launch-grid",
                    },
                },
            )

        from ninetoothed.backends.emitters.ascendc_affine import (
            render_affine_transfer,
        )

        affine = render_affine_transfer(context)
        context.lowering_plans["affine"] = affine

        if affine is not None:
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "ascendc_affine_transfer": {
                        "enabled": True,
                        "selection": "runtime-layout-guard",
                        "guard": affine.guard,
                        "chunk": affine.chunk,
                        "ub_bytes": affine.ub_bytes,
                        "fallback": "complete-original-kernel-and-launch-grid",
                    },
                },
            )

        from ninetoothed.backends.emitters.ascendc_row_tiles import plan_full_rows

        full_rows = plan_full_rows(context)
        context.lowering_plans["full_rows"] = full_rows

        if full_rows is not None:
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "ascendc_full_row_dag": {
                        "enabled": True,
                        "selection": "runtime-layout-guard",
                        "rows": full_rows.tiles.rows,
                        "columns": full_rows.tiles.columns,
                        "origin": full_rows.origin,
                        "buffer_dtypes": full_rows.dag.plan.buffer_dtypes,
                        "ub_bytes": full_rows.dag.plan.bytes_per_element
                        * full_rows.dag.cols,
                        **(
                            {
                                "input_layouts": tuple(
                                    {
                                        "name": name,
                                        "mode": layout.mode,
                                        "row_index": layout.row_index,
                                        "source_rows": layout.source_rows,
                                        "source_columns": layout.source_columns,
                                    }
                                    for name, layout in full_rows.dag.input_layouts
                                ),
                                "alias_policy": "all-input-output-and-output-output-dense-spans-disjoint",
                            }
                            if full_rows.dag.input_layouts
                            else {}
                        ),
                    },
                },
            )

        from ninetoothed.backends.emitters.ascendc_uniform import (
            render_uniform_store,
        )

        uniform = render_uniform_store(context)
        context.lowering_plans["uniform"] = uniform

        if uniform is not None:
            context = replace(
                context,
                scheduled_metadata={
                    **context.scheduled_metadata,
                    "ascendc_uniform_store": {
                        "enabled": True,
                        "selection": "runtime-layout-guard",
                        "chunk": uniform.chunk,
                        "ub_bytes": uniform.chunk * 4,
                        "fallback": "single-executor-complete-indexed-domain",
                    },
                },
            )

        _plan_selected_row_execution(context)

        if _ascendc_cube_conflicts(context):
            return context

        from ninetoothed.backends.emitters.ascendc_dot import (
            plan_cube_dot,
            render_direct_cube_dot,
        )

        direct = render_direct_cube_dot(context, self)
        context.lowering_plans["direct"] = direct

        if direct is not None:
            return replace(
                context,
                scheduled_metadata={**context.scheduled_metadata, **direct.metadata},
            )

        plan = plan_cube_dot(context, allow_reassociation=False)

        if plan is None:
            return context

        context.lowering_plans["cube"] = plan

        return replace(
            context,
            scheduled_metadata={**context.scheduled_metadata, **plan.metadata()},
        )

    def render_module(self, context: ModuleRenderContext) -> str:
        context = _ascendc_optimization_context(context)
        from ninetoothed.backends.ascendc_resources import (
            requested_vector_chunk,
        )

        resource_requested = requested_vector_chunk(context)
        resource_consumed = False

        if context.vector_program or context.block_program or context.scalar_program:
            raise ValueError(
                "The AscendC backend renders every kernel through the generic "
                "flat-index block domain; vector/block program modes are not "
                "supported."
            )

        _ascendc_validate_reduction_lowering(context)
        _ascendc_validate_atomics(context)

        kernel = context.kernel
        options = dict(kernel.compiler_options.get("backend_options", {}))
        default_chunk = int(options.get("task_chunk", 1024))
        chunk = common.schedule_int(kernel, "ascendc_task_chunk", default_chunk)
        total = _ascendc_integer_expr(context.total)
        grid_total = _ascendc_integer_expr(context.grid_total)
        body = _ascendc_integer_expr(context.body)
        # Shared masked-expression fallbacks can contain bare 0.0 literals.
        # Ccec promotes the whole conditional expression to unsupported
        # double unless every device floating literal has an f suffix.
        body = re.sub(
            r"(?<![\w.])(\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])",
            r"\1f",
            body,
        )

        kernel_parameters = [
            *(
                _signature_param(name, context.tensors[name])
                for name in context.variables
            ),
            *(
                _signature_param(name, context.tensors[name])
                for name in context.outputs
            ),
            *(_auxiliary_param(axis, context.tensors) for axis in context.shape_params),
        ]
        launches = launch_symbols(body, *kernel_parameters, kernel.kernel_name)
        context.scheduled_metadata["ascendc_launch"] = {
            "mapping": "consecutive-logical-task-segments",
            "max_block_dim": 65535,
            "internal_parameters": tuple(launches.parameters),
            "public_abi_changed": False,
        }
        kernel_params = _render_signature_params(
            [*kernel_parameters, *launches.parameters]
        )
        # CCE's host half ABI uses floating registers, whereas native_scalar
        # passes its raw storage with ctypes.c_uint16. Decode the storage at
        # the host boundary before any guard or device launch uses the value.
        half_storage = {}

        for name in dict.fromkeys(
            (*context.variables, *context.outputs, *context.shape_params)
        ):
            info = context.tensors.get(name)

            if (
                info is not None
                and info.ndim == 0
                and _normalize_dtype(info.dtype) == "float16"
            ):
                half_storage[name] = allocate_private_symbol(
                    f"nt_host_half_bits_{name}",
                    body,
                    *kernel_parameters,
                    *launches.parameters,
                    kernel.kernel_name,
                    *half_storage.values(),
                )

        half_decode = "\n".join(
            f"    half {name};\n"
            f'    static_assert(sizeof({name}) == sizeof({storage}), "half ABI width");\n'
            f"    ::memcpy(&{name}, &{storage}, sizeof({name}));"
            for name, storage in half_storage.items()
        )

        if half_storage:
            context.scheduled_metadata["ascendc_host_scalar_abi"] = {
                "version": "uint16-half-v1",
                "half_storage_bindings": tuple(half_storage),
            }

        stream_name = allocate_private_symbol(
            "stream", *context.variables, *context.outputs, *context.shape_params
        )
        launch_params = _render_signature_params(
            [
                *(
                    _launch_param(
                        name, context.tensors[name], half_storage=half_storage
                    )
                    for name in context.variables
                ),
                *(
                    _launch_param(
                        name, context.tensors[name], half_storage=half_storage
                    )
                    for name in context.outputs
                ),
                *(
                    _host_auxiliary_param(axis, context.tensors, half_storage)
                    for axis in context.shape_params
                ),
                f"aclrtStream {stream_name}",
            ]
        )

        tensor_casts = []
        launch_args = []

        for name in (*context.variables, *context.outputs):
            info = context.tensors.get(name)

            if info is not None and info.ndim != 0:
                tensor_casts.append(
                    f"    __gm__ {self.type_name(info.dtype)}* {name} = "
                    f"(__gm__ {self.type_name(info.dtype)}*){name}_gm;"
                )
                launch_args.append(f"(GM_ADDR)(uintptr_t){name}")
            else:
                launch_args.append(name)

        launch_args.extend(context.shape_params)

        from ninetoothed.backends.emitters.ascendc_reduce import (
            render_row_reduce,
        )

        row_match = context.lowering_plans.get("row")

        affine = context.lowering_plans.get("affine")
        uniform = context.lowering_plans.get("uniform")
        full_rows = context.lowering_plans.get("full_rows")

        if uniform is not None:
            fallback = f"""    if (GetBlockIdx() != 0) {{ return; }}
    for (int64_t {self.index_name} = 0; {self.index_name} < {total}; {self.index_name}++) {{
{common.indent_block(body, "        ")}
    }}"""
            kernel_prelude = (
                "\n".join(tensor_casts)
                + f"\n    if ({uniform.guard}) {{\n"
                + common.indent_block(_ascendc_integer_expr(uniform.body), "    ")
                + "\n    } else {\n"
                + common.indent_block(fallback, "    ")
                + "\n    }"
            )
            tasks_expr = f"(({uniform.guard}) ? ({uniform.tasks}) : 1)"
        elif full_rows is not None:
            from ninetoothed.backends.emitters.ascendc_vector_dag import (
                render_row_dag,
                row_dag_memory_guard,
            )

            native, native_tasks = render_row_dag(
                replace(
                    full_rows.tiles.context,
                    scheduled_metadata=context.scheduled_metadata,
                ),
                full_rows.dag,
                self,
            )
            predicate = row_dag_memory_guard(full_rows.tiles.context, full_rows.dag)

            if context.cooperative_reduction_program:
                fallback, _ = _ascendc_cooperative_fallback(
                    context, body, tensor_casts, grid_total
                )
            else:
                fallback = _ascendc_serial_fallback(
                    body, tensor_casts, total, self.index_name
                )

            kernel_prelude = _ascendc_guarded_body(predicate, native, fallback)
            host_predicate = predicate

            for name in (*context.variables, *context.outputs):
                host_predicate = host_predicate.replace(f"{name}_gm", name)

            tasks_expr = f"(({host_predicate}) ? ({native_tasks}) : 1)"
        elif row_match is not None:
            kernel_prelude = render_row_reduce(row_match)
            predicate = _ascendc_row_memory_guard(context, row_match, collapsed=True)

            if context.cooperative_reduction_program:
                fallback, _ = _ascendc_cooperative_fallback(
                    context, body, tensor_casts, grid_total
                )
            else:
                fallback = _ascendc_serial_fallback(
                    body, tensor_casts, total, self.index_name
                )

            kernel_prelude = _ascendc_guarded_body(predicate, kernel_prelude, fallback)
            tasks_expr = f"(({row_match[3]}) + 15) / 16"
        elif context.cooperative_reduction_program:
            from ninetoothed.backends.emitters.ascendc_reduce import (
                render_reduce_broadcast,
            )
            from ninetoothed.backends.emitters.ascendc_vector_dag import (
                render_row_dag,
                row_dag_memory_guard,
            )

            reduce_match = context.lowering_plans.get("reduce_broadcast")
            dag_match = context.lowering_plans.get("row_dag")

            if dag_match is not None:
                kernel_prelude, tasks_expr = render_row_dag(context, dag_match, self)
                predicate = row_dag_memory_guard(context, dag_match)
                fallback, _ = _ascendc_cooperative_fallback(
                    context, body, tensor_casts, grid_total
                )
                kernel_prelude = _ascendc_guarded_body(
                    predicate, kernel_prelude, fallback
                )
            elif reduce_match is not None:
                kernel_prelude = render_reduce_broadcast(reduce_match)
                predicate = _ascendc_row_memory_guard(
                    context, reduce_match, collapsed=False
                )
                fallback, _ = _ascendc_cooperative_fallback(
                    context, body, tensor_casts, grid_total
                )
                kernel_prelude = _ascendc_guarded_body(
                    predicate, kernel_prelude, fallback
                )
                tasks_expr = reduce_match[3]
            else:
                kernel_prelude, tasks_expr = _ascendc_cooperative_fallback(
                    context, body, tensor_casts, grid_total
                )
        else:
            from ninetoothed.backends.emitters.ascendc_vector import (
                render_vector_elementwise,
            )
            from ninetoothed.backends.emitters.ascendc_vector_dag import (
                flat_dag_memory_guard,
                render_flat_dag,
            )
            from ninetoothed.backends.emitters.staging import (
                dense_staging_memory_guard,
                split_stride_guard,
            )

            vector_match = context.lowering_plans.get("packet")
            flat_match = context.lowering_plans.get("flat")
            total_param = context.lowering_plans.get("packet_total")

            if vector_match is not None and total_param is not None:
                resource_consumed = (
                    context.lowering_plans.get("packet_resources") is not None
                )
                kernel_prelude = render_vector_elementwise(vector_match, total_param)

                tasks_expr = (
                    f"({total_param} + {vector_match.chunk} - 1) / {vector_match.chunk}"
                )
                predicate, _ = split_stride_guard(body)
                equal_extents = []

                for name in vector_match.inputs:
                    info = context.tensors[name]
                    shape = (info.attrs or {}).get("source_shape") or info.shape
                    equal_extents.append(f"({shape[0]}) == ({total_param})")

                # DMA sectors are 32 bytes, scalar GM writeback lines are 64.
                # Keep the prefix and any scalar tail in different lines.
                tail_alignment = 32 if vector_match.output_dtype == "float16" else 16
                tail_owner = (
                    f"((({total_param}) % {tail_alignment}) == 0 || "
                    f"(((uintptr_t){vector_match.output}_gm) & 63ULL) == 0)"
                )

                predicate = " && ".join(
                    [
                        *([f"({predicate})"] if predicate is not None else []),
                        *equal_extents,
                        dense_staging_memory_guard(
                            context,
                            vector_match.inputs,
                            (vector_match.output,),
                            pointer_suffix="_gm",
                        ),
                        tail_owner,
                        *(
                            f"(((uintptr_t){name}_gm) & 31ULL) == 0"
                            for name in (*vector_match.inputs, vector_match.output)
                        ),
                    ]
                )

                if predicate is not None:
                    # GM DataCopy alignment and scalar cache-line ownership
                    # are independent runtime obligations.
                    fallback, fallback_tasks = _ascendc_indexed_plain_fallback(
                        context,
                        body,
                        tensor_casts,
                        total,
                        vector_match.fallback_chunk,
                        self,
                    )
                    kernel_prelude = _ascendc_guarded_body(
                        predicate, kernel_prelude, fallback
                    )
                    host_predicate = predicate

                    for name in (*vector_match.inputs, vector_match.output):
                        host_predicate = host_predicate.replace(f"{name}_gm", name)

                    tasks_expr = (
                        f"(({host_predicate}) ? ({tasks_expr}) : ({fallback_tasks}))"
                    )
            elif flat_match is not None:
                resource_consumed = resource_requested is not None
                kernel_prelude, native_tasks = render_flat_dag(
                    context, flat_match, self
                )
                predicate = flat_dag_memory_guard(context, flat_match)
                fallback = _ascendc_serial_fallback(
                    body, tensor_casts, total, self.index_name
                )
                kernel_prelude = _ascendc_guarded_body(
                    predicate, kernel_prelude, fallback
                )
                host_predicate = predicate

                for name in (*context.variables, *context.outputs):
                    host_predicate = host_predicate.replace(f"{name}_gm", name)

                tasks_expr = f"(({host_predicate}) ? ({native_tasks}) : 1)"
            elif any(
                operation.opcode.startswith("reduce.")
                for block in context.kernel.ssa.blocks
                for operation in _walk_ops(block.operations)
            ):
                kernel_prelude, tasks_expr = _ascendc_indexed_reduction_fallback(
                    context,
                    body,
                    tensor_casts,
                    grid_total,
                    total,
                    index_name=self.index_name,
                )
            elif any(
                operation.opcode in {"linalg.dot", "linalg.matmul"}
                for block in context.kernel.ssa.blocks
                for operation in _walk_ops(block.operations)
            ):
                kernel_prelude, tasks_expr = _ascendc_indexed_dot_fallback(
                    context, body, tensor_casts, total, chunk, self
                )
            else:
                kernel_prelude, tasks_expr = _ascendc_indexed_plain_fallback(
                    context, body, tensor_casts, total, chunk, self
                )

        from ninetoothed.backends.ascendc_resources import (
            requested_division_strategy,
        )

        if requested_division_strategy(
            context
        ) == "native_division" and not context.scheduled_metadata.get(
            "ascendc_division_strategy", {}
        ).get("consumed"):
            raise ValueError(
                "The native division strategy was not consumed by a proved vector plan."
            )

        def native_vector_kernel(native, name):
            return render_native_vector_kernel(
                kernel.kernel_name + f"_{name}_kernel",
                kernel_params,
                launch_args,
                chr(10).join(tensor_casts)
                + chr(10)
                + _ascendc_integer_expr(native.body),
                native.guard,
                native.tasks,
                launches,
                stream=stream_name,
            )

        affine_source = ""
        affine_launch = ""
        ordered_add_source = ""
        ordered_add_launch = ""
        transpose_source = ""
        transpose_launch = ""
        indexed_source = ""
        indexed_launch = ""

        indexed = context.lowering_plans.get("indexed")

        if indexed is not None:
            (indexed_source, indexed_launch) = native_vector_kernel(indexed, "indexed")

        transpose = context.lowering_plans.get("transpose")

        if transpose is not None:
            (transpose_source, transpose_launch) = native_vector_kernel(
                transpose, "transpose"
            )

        ordered_add = context.lowering_plans.get("ordered_add")

        if ordered_add is not None:
            (ordered_add_source, ordered_add_launch) = native_vector_kernel(
                ordered_add, "ordered_add"
            )

        if affine is not None:
            (affine_source, affine_launch) = native_vector_kernel(affine, "affine")

        cube_source = ""
        cube_launch = ""

        cube = context.lowering_plans.get("partials") or context.lowering_plans.get(
            "direct"
        )
        cube_plan = context.lowering_plans.get("cube")

        if cube_plan is not None:
            from ninetoothed.backends.emitters.ascendc_dot import render_cube_plan

            cube = render_cube_plan(cube_plan, self)

        if cube is not None:
            cube_source = f"""
extern "C" __global__ __aicore__ void {kernel.kernel_name}_cube_kernel(
{kernel_params}
) {{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#ifndef __DAV_C220_CUBE__
    return;
#else
{launches.kernel_body(chr(10).join(tensor_casts) + chr(10) + cube.body)}
#endif
}}
"""
            aligned = " && ".join(
                f"(((uintptr_t){name}) & 31ULL) == 0"
                for name in (*context.variables, *context.outputs)
                if (info := context.tensors.get(name)) is not None and info.ndim != 0
            )
            from ninetoothed.backends.emitters.staging import dense_staging_memory_guard

            memory_guard = dense_staging_memory_guard(
                context, context.variables, context.outputs
            )
            predicate = f"({cube.guard}) && ({aligned}) && ({memory_guard})"
            cube_launch = f"""    if ({predicate}) {{
{render_segmented_launch(kernel.kernel_name + "_cube_kernel", launch_args, cube.tasks, launches, indent="        ", stream=stream_name)}
    }}
"""
            # The false guard has not proved output ownership and complete
            # reader/store independence. Execute the original domain once.
            kernel_prelude = _ascendc_serial_fallback(
                body, tensor_casts, total, self.index_name
            )
            tasks_expr = "1"

        if resource_requested is not None and not resource_consumed:
            raise ValueError(
                "The AscendC vector resource override has no proved flat vector route."
            )

        math_tail = any(
            f"nt_{name}(" in kernel_prelude
            for name in (
                "nan",
                "inf",
                "exp",
                "log",
                "tanh",
                "sigmoid",
                "sin",
                "cos",
                "tan",
                "atan",
                "erf",
                "floor",
                "ceil",
                "rsqrt",
                "fabs",
            )
        )

        support = "\n\n".join(
            filter(
                None,
                (
                    _ascendc_support(context, force_math=math_tail),
                    render_int32_cast_support("__aicore__ inline")
                    if "nt_cast_int32(" in body
                    else "",
                    render_c_type_support(body, "__aicore__ inline"),
                ),
            )
        )

        return f"""// Generated by NineToothed's AscendC SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program

#include "kernel_operator.h"
#include "acl/acl.h"
#include <type_traits>
#include <limits>
{("#include <string.h>" + chr(10)) if half_storage else ""}
using namespace AscendC;

{support}

extern "C" __global__ __aicore__ void {kernel.kernel_name}_kernel(
{kernel_params}
) {{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_CUBE__
    return;
#endif
{launches.kernel_body(kernel_prelude)}
}}

{cube_source}
{affine_source}
{ordered_add_source}
{transpose_source}
{indexed_source}
extern "C" int launch_{kernel.kernel_name}(
{launch_params}
) {{
{(half_decode + chr(10)) if half_decode else ""}{cube_launch}
{affine_launch}
{ordered_add_launch}
{transpose_launch}
{indexed_launch}
    const int64_t nt_chunk = {chunk};
{render_segmented_launch(kernel.kernel_name + "_kernel", launch_args, tasks_expr, launches, stream=stream_name)}
}}
"""


def _ascendc_float_suffix(value: str) -> str:
    """Add the float suffix to bare decimal literals for aicore.

    ccec treats unsuffixed decimal literals as double, which is not
    supported in aicore functions.
    """
    stripped = value.strip()

    if re.fullmatch(r"-?\d+\.\d+", stripped):
        return f"{stripped}f"

    return value


def _emit_ascendc_cooperative_reduction(
    local: str, operation: ssa.Operation, ctx: _EmitContext
) -> str:
    """Reduce the full extent inside one block; no cross-thread steps needed."""
    schedule = dict(ctx.reduction_schedule or {})
    axis = int(schedule["axis"]) if "axis" in schedule else None

    if axis is None or not operation.operands or not operation.results:
        raise ValueError("Malformed AscendC cooperative reduction operation.")

    operator = operation.opcode[len("reduce.") :]
    operand = operation.operands[0]
    operand_axes = _value_axes(operand, ctx)
    extent = str(schedule.get("extent", operand_axes[axis]))
    operand_type = ctx.value_types.get(operand)
    result_dtype = _normalize_dtype(operation.results[0].type.dtype or "float32")
    operand_dtype = _normalize_dtype(
        operand_type.dtype if operand_type is not None else result_dtype
    )
    accumulator_dtype = _ascendc_reduction_accumulator_dtype(
        operand_dtype, result_dtype
    )
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
        f"{_ascendc_reduction_update(operator, accumulator, term, accumulator_dtype, target=ctx.target)};"
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


def _ascendc_reduction_accumulator_dtype(operand_dtype: str, result_dtype: str) -> str:
    if operand_dtype in {"float16"}:
        return "float32"

    if operand_dtype == "bool":
        return "int32"
    return result_dtype


def _ascendc_reduction_update(
    operator: str, lhs: str, rhs: str, dtype: str, *, target=None
) -> str:
    del dtype

    if operator == "sum":
        return f"({lhs}) + ({rhs})"

    # Computed dtypes are C++ type text, so Python string tests cannot decide
    # floating versus integral. Explicitly keep the accumulator's value type
    # while the shared template propagates either NaN through the reduction.
    value_dtype = f"std::remove_cv<std::remove_reference<decltype({lhs})>::type>::type"

    function = f"_nt_reduce_{operator}"

    if target is not None:
        function = target.global_helper(function)
    return f"{function}<{value_dtype}>({lhs}, {rhs})"


_ASCENDC_MATH_SUPPORT = (
    """// Scalar transcendental functions are not hardware built-ins on AscendC,
// so the generated kernels carry compact software implementations.  Device
// functions cannot recurse, so every routine is range-reduced inline.
union NtFloatBits {
    float f;
    int32_t i;
};

__aicore__ inline float nt_inf() {
    NtFloatBits v;
    v.i = 0x7f800000;
    return v.f;
}

__aicore__ inline float nt_nan() {
    NtFloatBits v;
    v.i = 0x7fc00000;
    return v.f;
}

__aicore__ inline float nt_exp(float x) {
    if (x != x) { return x; }
    if (x > 88.72f) { return nt_inf(); }
    if (x < -103.97f) { return 0.0f; }
    float t = x * 1.442695041f;
    int k = (int)(t + (t > 0.0f ? 0.5f : -0.5f));
    float r = (float)k;
    x = x - r * 0.693147181f;
    // 10th-order minimax polynomial for exp on [-ln2/2, ln2/2]
    float p = 1.0f + x * (1.0f + x * (0.5000000000f + x * (0.1666666667f
        + x * (0.04166666667f + x * (0.008333333333f + x * (0.001388888889f
        + x * (0.0001984126984f + x * (0.00002480158730f
        + x * (0.000002755731922f + x * 0.0000002755731922f)))))))));
    // Scale in two steps: a single 2^k exponent field overflows into the
    // sign bit for subnormal results (k + 127 <= 0) and produces a huge
    // negative value instead.  Halving the exponent keeps both scale
    // factors in range, and IEEE multiplication rounds gradually.
    int k1 = k / 2;
    int k2 = k - k1;
    NtFloatBits s1;
    NtFloatBits s2;
    s1.i = (k1 + 127) << 23;
    s2.i = (k2 + 127) << 23;
    return (p * s1.f) * s2.f;
}

__aicore__ inline float nt_log(float x) {
    if (x != x || x < 0.0f) { return nt_nan(); }
    if (x == 0.0f) { return -nt_inf(); }
    if (x == nt_inf()) { return x; }
    NtFloatBits v;
    v.f = x;
    int biased = (v.i >> 23) & 0xff;
    float m;
    int e;
    if (biased == 0) {
        // Subnormal inputs carry no implicit bit: x = M * 2^-149.
        e = -126;
        m = (float)(v.i & 0x007fffff) * (1.0f / 8388608.0f);
    }
    else {
        e = biased - 127;
        v.i = (v.i & 0x007fffff) | 0x3f800000;
        m = v.f;
    }
    // Normalize m into [0.75, 1.5] so |u| <= 0.2 below.
    while (m > 1.5f) { m = m * 0.5f; e = e + 1; }
    while (m < 0.75f) { m = m * 2.0f; e = e - 1; }
    // log(m) = 2*artanh(u) with u = (m-1)/(m+1): truncation ~ u^11/11.
    float u = (m - 1.0f) / (m + 1.0f);
    float u2 = u * u;
    float p = u * (1.0f + u2 * (0.33333334f + u2 * (0.2f + u2 * (0.14285715f
        + u2 * 0.11111111f))));
    return e * 0.69314718f + 2.0f * p;
}

__aicore__ inline float nt_tanh(float x) {
    if (x == 0.0f) { return x; }
    // exp(2x) saturates around x ~ 44; (inf - 1) / (inf + 1) would be NaN.
    if (x > 22.0f) { return 1.0f; }
    if (x < -22.0f) { return -1.0f; }
    float e2 = nt_exp(2.0f * x);
    return (e2 - 1.0f) / (e2 + 1.0f);
}

__aicore__ inline float nt_sigmoid(float x) {
    return 1.0f / (1.0f + nt_exp(-x));
}

"""
    + render_float32_trig_support("__aicore__")
    + """

__aicore__ inline float nt_tan(float x) {
    return nt_sin(x) / nt_cos(x);
}

__aicore__ inline float nt_sinh(float x) {
    if (x == 0.0f) { return x; }
    float e = nt_exp(x);
    return 0.5f * (e - 1.0f / e);
}

__aicore__ inline float nt_cosh(float x) {
    float e = nt_exp(x);
    return 0.5f * (e + 1.0f / e);
}

// atanh-free, recursion-free atan: fold x > 1 through the reciprocal
// identity atan(x) = pi/2 - atan(1/x), then arguments above 0.35 through
// the additive sqrt(3) identity, so the odd Taylor polynomial only sees
// |v| <= 2 - sqrt(3) ~ 0.268.
__aicore__ inline float nt_atan_poly(float x) {
    float v2 = x * x;
    return x * (1.0f - v2 * (0.33333334f - v2 * (0.2f - v2 * (0.14285715f
        - v2 * (0.11111111f - v2 * 0.09090909f)))));
}

__aicore__ inline float nt_atan_pos(float x) {
    const float NT_HALF_PI = 1.5707964f;
    const float NT_PI_6 = 0.5235988f;
    const float NT_SQRT3 = 1.7320508f;

    if (x > 1.0f) {
        float v = 1.0f / x;
        float inner = nt_atan_poly(v);

        if (v > 0.35f) {
            inner = nt_atan_poly((NT_SQRT3 * v - 1.0f) / (NT_SQRT3 + v))
                + NT_PI_6;
        }
        return NT_HALF_PI - inner;
    }

    if (x > 0.35f) {
        return nt_atan_poly((NT_SQRT3 * x - 1.0f) / (NT_SQRT3 + x)) + NT_PI_6;
    }

    return nt_atan_poly(x);
}

__aicore__ inline float nt_atan(float x) {
    if (x < 0.0f) { return -nt_atan_pos(-x); }
    return nt_atan_pos(x);
}

__aicore__ inline float nt_asin(float x) {
    if (x != x || x > 1.0f || x < -1.0f) { return nt_nan(); }
    if (x == 0.0f) { return x; }
    float r = 1.0f - x * x;
    float root = r >= 0.0f ? sqrt(r) : 0.0f;
    float v = x >= 0.0f ? (root > 0.0f ? x / root : nt_inf()) : -1.0f;
    if (x < 0.0f) {
        v = root > 0.0f ? x / root : -nt_inf();
    }
    return nt_atan(v);
}

__aicore__ inline float nt_acos(float x) {
    return 1.5707964f - nt_asin(x);
}

__aicore__ inline float nt_atan2(float y, float x) {
    const float NT_PI_F = 3.14159265f;
    const float NT_HALF_PI = 1.5707964f;
    if (x != x || y != y) { return nt_nan(); }
    NtFloatBits bx;
    NtFloatBits by;
    bx.f = x;
    by.f = y;
    bool negative_x = bx.i < 0;
    bool negative_y = by.i < 0;
    if (y == 0.0f) {
        return negative_x ? (negative_y ? -NT_PI_F : NT_PI_F) : y;
    }
    if (x == 0.0f) { return negative_y ? -NT_HALF_PI : NT_HALF_PI; }
    bx.i &= 0x7fffffff;
    by.i &= 0x7fffffff;
    float angle;
    if (by.f == nt_inf()) {
        angle = bx.f == nt_inf()
            ? (negative_x ? 2.35619449f : 0.785398163f) : NT_HALF_PI;
    }
    else if (bx.f == nt_inf()) {
        angle = negative_x ? NT_PI_F : 0.0f;
    }
    else {
        angle = nt_atan_pos(by.f / bx.f);
        if (negative_x) { angle = NT_PI_F - angle; }
    }
    return negative_y ? -angle : angle;
}

__aicore__ inline float nt_floor(float x) {
    NtFloatBits v;
    v.f = x;
    int exponent = ((v.i >> 23) & 0xff) - 127;
    // Large finite float32 values already have no fractional bits.
    if (exponent >= 23 || (v.i & 0x7fffffff) == 0) { return x; }
    if (exponent < 0) { return x < 0.0f ? -1.0f : 0.0f; }
    int32_t fraction = (1 << (23 - exponent)) - 1;
    if ((v.i & fraction) == 0) { return x; }
    v.i &= ~fraction;
    return x < 0.0f ? v.f - 1.0f : v.f;
}

__aicore__ inline float nt_ceil(float x) {
    return -nt_floor(-x);
}

__aicore__ inline float nt_pow(float x, float y) {
    if (y == 0.0f) { return 1.0f; }
    if (x == 1.0f) { return 1.0f; }
    if (x != x || y != y) { return nt_nan(); }
    NtFloatBits base_bits;
    NtFloatBits exponent_bits;
    base_bits.f = x;
    exponent_bits.f = y;
    int exponent = ((exponent_bits.i >> 23) & 0xff) - 127;
    // Above 2^24 float32 exponents are all even; inspect the significand
    // without a large float-to-int conversion.
    bool odd = exponent >= 0 && exponent <= 23 && y == nt_floor(y)
        && ((((exponent_bits.i & 0x007fffff) | 0x00800000)
            >> (23 - exponent)) & 1);
    float ax = x < 0.0f ? -x : x;
    if (y == nt_inf() || y == -nt_inf()) {
        if (ax == 1.0f) { return 1.0f; }
        return ((ax > 1.0f) == (y > 0.0f)) ? nt_inf() : 0.0f;
    }
    if (ax == 0.0f || ax == nt_inf()) {
        float special = ((ax == 0.0f) == (y > 0.0f)) ? 0.0f : nt_inf();
        return base_bits.i < 0 && odd ? -special : special;
    }
    if (x < 0.0f && y != nt_floor(y)) {
        // Negative base with a non-integer exponent is undefined.
        return nt_nan();
    }
    NtFloatBits v;
    v.f = (x < 0.0f) ? -x : x;
    int biased = (v.i >> 23) & 0xff;
    float m;
    int e;
    if (biased == 0) {
        // Subnormal bases carry no implicit bit: x = M * 2^-149.
        e = -126;
        m = (float)(v.i & 0x007fffff) * (1.0f / 8388608.0f);
    }
    else {
        e = biased - 127;
        v.i = (v.i & 0x007fffff) | 0x3f800000;
        m = v.f;
    }
    float lg = e * 0.69314718f + nt_log(m);
    float r = nt_exp(((x < 0.0f) ? nt_floor(y) : y) * lg);
    if (x < 0.0f) {
        return odd ? -r : r;
    }
    return r;
}

__aicore__ inline float nt_fabs(float x) {
    NtFloatBits v;
    v.f = x;
    v.i &= 0x7fffffff;
    return v.f;
}

__aicore__ inline float nt_rsqrt(float x) {
    return 1.0f / sqrt(x);
}

__aicore__ inline float nt_exp2(float x) {
    return nt_exp(x * 0.69314718f);
}

__aicore__ inline float nt_log2(float x) {
    return nt_log(x) * 1.44269504f;
}

__aicore__ inline float nt_log10(float x) {
    return nt_log(x) * 0.43429448f;
}

__aicore__ inline float nt_log1p(float x) {
    return nt_log(1.0f + x);
}

__aicore__ inline float nt_expm1(float x) {
    return nt_exp(x) - 1.0f;
}

__aicore__ inline float nt_erf(float x) {
    if (x == 0.0f) { return x; }
    float s = x < 0.0f ? -1.0f : 1.0f;
    x = x * s;
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f)
        * t - 0.284496736f) * t + 0.254829592f) * t * nt_exp(-x * x);
    return s * y;
}"""
)


# Bare function names accepted by `call()`; names mapping to `nt_*` helpers
# require the software math library below to be emitted.
_ASCENDC_CALL_FUNCS = {
    "abs": "nt_fabs",
    "acos": "nt_acos",
    "asin": "nt_asin",
    "atan": "nt_atan",
    "atan2": "nt_atan2",
    "ceil": "nt_ceil",
    "cos": "nt_cos",
    "cosh": "nt_cosh",
    "erf": "nt_erf",
    "exp": "nt_exp",
    "exp2": "nt_exp2",
    "expm1": "nt_expm1",
    "floor": "nt_floor",
    "log": "nt_log",
    "log1p": "nt_log1p",
    "log2": "nt_log2",
    "log10": "nt_log10",
    "max": "max",
    "maximum": "max",
    "min": "min",
    "minimum": "min",
    "pow": "nt_pow",
    "rsqrt": "nt_rsqrt",
    "sigmoid": "nt_sigmoid",
    "sin": "nt_sin",
    "sinh": "nt_sinh",
    "tan": "nt_tan",
    "tanh": "nt_tanh",
}

_MATH_OPCODES = {
    "math.exp": "nt_exp",
    "math.log": "nt_log",
    "math.tanh": "nt_tanh",
    "math.sigmoid": "nt_sigmoid",
    "math.sin": "nt_sin",
    "math.cos": "nt_cos",
    "math.tan": "nt_tan",
    "math.sinh": "nt_sinh",
    "math.cosh": "nt_cosh",
    "math.asin": "nt_asin",
    "math.acos": "nt_acos",
    "math.atan": "nt_atan",
    "math.atan2": "nt_atan2",
    "math.pow": "nt_pow",
    "math.erf": "nt_erf",
    "math.abs": "nt_fabs",
    "math.floor": "nt_floor",
    "math.ceil": "nt_ceil",
    "math.rsqrt": "nt_rsqrt",
    "math.exp2": "nt_exp2",
    "math.log2": "nt_log2",
    "math.log10": "nt_log10",
    "math.log1p": "nt_log1p",
    "math.expm1": "nt_expm1",
}


def _ascendc_indexed_plain_fallback(context, body, tensor_casts, total, chunk, target):
    """Parallelize scalar stores only with exclusive physical cache-line owners."""
    from ninetoothed.backends.emitters.staging import canonical_flat_view

    casts = "\n".join(tensor_casts)
    serial = _ascendc_serial_fallback(body, tensor_casts, total, target.index_name)

    if not context.outputs or any(
        set(store.attrs) - {"target", "target_shape"} for store in context.stores
    ):
        return serial, "1"

    terms = []

    for name in context.outputs:
        info = context.tensors[name]

        if info.ndim != 1 or not canonical_flat_view(info):
            return serial, "1"

        stride = str(info.source_strides[0]) if info.source_strides else "1"
        terms.extend(
            (
                f"({stride}) == 1",
                f"(((uintptr_t){name}_gm) & 63ULL) == 0",
                f"({chunk} * sizeof({target.type_name(info.dtype)})) % 64 == 0",
            )
        )

    predicate = " && ".join(f"({term})" for term in terms)
    lane = _ascendc_fallback_symbol(context, "nt_lane")
    parallel = (
        casts
        + f"""
    for (int64_t {lane} = 0; {lane} < {chunk}; {lane}++) {{
        int64_t {target.index_name} = (int64_t)GetBlockIdx() * {chunk} + {lane};
        if ({target.index_name} < {total}) {{
{common.indent_block(body, "            ")}
        }}
    }}"""
    )
    source = _ascendc_guarded_body(predicate, parallel, serial)
    host_predicate = predicate

    for name in context.outputs:
        host_predicate = host_predicate.replace(f"{name}_gm", name)

    return source, f"(({host_predicate}) ? (({total} + {chunk} - 1) / {chunk}) : 1)"


def _ascendc_indexed_reduction_fallback(
    context, body, tensor_casts, grid_total, total, *, index_name="index"
):
    """Assign each logical row tile entirely to one scalar executor."""
    value_total = " * ".join(f"({axis})" for axis in context.axes) or "1"
    value = _ascendc_fallback_symbol(context, "nt_value")
    indexed_body = f"""
    for (int64_t {value} = 0; {value} < ({value_total}); {value}++) {{
        int64_t {index_name} = (int64_t)GetBlockIdx() * ({value_total}) + {value};
        if ({index_name} < ({total})) {{
{common.indent_block(body, "            ")}
        }}
    }}"""

    return _ascendc_cooperative_fallback(
        context, indexed_body, tensor_casts, grid_total
    )


def _ascendc_indexed_dot_fallback(context, body, tensor_casts, total, chunk, target):
    """Prove cache-line ownership before parallel scalar dot writeback."""
    from ninetoothed.backends.emitters.dot_layout import prove_dense_dot_access

    casts = "\n".join(tensor_casts)
    serial = _ascendc_serial_fallback(body, tensor_casts, total, target.index_name)

    if (
        len(context.stores) != 1
        or len(context.outputs) != 1
        or tuple(map(str, context.axes)) != ("32", "32")
        or set(context.stores[0].attrs) - {"target", "target_shape"}
    ):
        return serial, "1"

    name = context.outputs[0]
    info = context.tensors[name]
    shape = (info.attrs or {}).get("source_shape") or info.source_shape or info.shape

    try:
        rows, cols = map(int, shape)
    except (TypeError, ValueError):
        return serial, "1"

    if rows <= 0 or cols <= 0 or rows % 32 or cols % 32:
        return serial, "1"

    columns = cols // 32
    access = prove_dense_dot_access(
        info.attrs or {},
        shape=(32, 32),
        tasks=(rows // 32) * columns,
        stages=1,
        expected_offsets=(
            f"outer_index // {columns} * 32 + value_0",
            f"outer_index % {columns} * 32 + value_1",
        ),
    )

    if access is None:
        return serial, "1"

    cpp = target.type_name(info.dtype)
    predicate = " && ".join(
        (
            f"({access.strides[0]}) == {cols}",
            f"({access.strides[1]}) == 1",
            f"(((uintptr_t){name}_gm) & 63ULL) == 0",
            f"({chunk} * sizeof({cpp})) % 64 == 0",
        )
    )
    lane = _ascendc_fallback_symbol(context, "nt_lane")
    parallel = (
        casts
        + f"""
    for (int64_t {lane} = 0; {lane} < {chunk}; {lane}++) {{
        int64_t {target.index_name} = (int64_t)GetBlockIdx() * {chunk} + {lane};
        if ({target.index_name} < {total}) {{
{common.indent_block(body, "            ")}
        }}
    }}"""
    )
    source = _ascendc_guarded_body(predicate, parallel, serial)
    tasks = f"(({predicate}) ? (({total} + {chunk} - 1) / {chunk}) : 1)"
    # The launcher has typed pointers rather than the device GM_ADDR names.
    tasks = tasks.replace(f"{name}_gm", name)

    return source, tasks


def _ascendc_cooperative_fallback(context, body, tensor_casts, grid_total):
    """Assign scalar writeback ownership from geometry and runtime layout."""
    from ninetoothed.backends.emitters.row_geometry import canonical_row_view

    slice_name = _ascendc_fallback_symbol(context, "nt_slice")
    slice_start = _ascendc_fallback_symbol(context, "nt_slice0")
    grouped_body = body.replace("(int64_t)GetBlockIdx()", slice_name)
    serial = _ascendc_serial_fallback(
        grouped_body, tensor_casts, f"({grid_total})", slice_name
    )

    if not all(
        canonical_row_view(context.tensors[name])
        for name in (*context.variables, *context.outputs)
        if context.tensors.get(name) is not None and context.tensors[name].ndim != 0
    ):
        return serial, "1"

    # A full source row per slice gives 64-row boundaries aligned to the
    # cache line. Partial tiles and mapped/offset views have no such proof.
    grouped = (
        "\n".join(tensor_casts)
        + f"""
    int64_t {slice_start} = (int64_t)GetBlockIdx() * 64;
    for (int64_t {slice_name} = {slice_start}; {slice_name} < {slice_start} + 64 && {slice_name} < ({grid_total}); {slice_name}++) {{
{common.indent_block(grouped_body, "        ")}
    }}"""
    )
    terms = []

    for name in context.outputs:
        info = context.tensors[name]
        shape = (
            (info.attrs or {}).get("source_shape") or info.source_shape or info.shape
        )
        strides = tuple(map(str, info.source_strides)) or (str(shape[1]), "1")
        terms.extend((f"({strides[0]}) == ({shape[1]})", f"({strides[1]}) == 1"))
        terms.append(f"(((uintptr_t){name}_gm) & 63ULL) == 0")

    predicate = " && ".join(f"({term})" for term in terms) or "true"
    source = _ascendc_guarded_body(predicate, grouped, serial)

    return source, f"(({grid_total}) + 63) / 64"


def _ascendc_row_memory_guard(context, match, *, collapsed):
    """Prove row-contiguous, aligned GM before replacing indexed accesses."""
    from ninetoothed.backends.emitters.staging import dense_staging_memory_guard

    terms = []

    for name, width in (
        (match.input, match.cols),
        (match.output, "1" if collapsed else match.cols),
    ):
        info = context.tensors[name]
        strides = tuple(map(str, info.source_strides))

        if not strides:
            shape = (
                (info.attrs or {}).get("source_shape")
                or info.source_shape
                or info.shape
            )
            strides = (str(shape[1]), "1")

        terms.extend((f"({strides[0]}) == ({width})", f"({strides[1]}) == 1"))
        # Full collapsed batches own a cache line; a shifted base pointer
        # must retain the indexed fallback even with contiguous strides.
        terms.append(f"(((uintptr_t){name}_gm) & 63ULL) == 0")

    terms.append(
        dense_staging_memory_guard(
            context, (match.input,), (match.output,), pointer_suffix="_gm"
        )
    )

    return " && ".join(f"({term})" for term in terms)


def _ascendc_same_domain_loop_store(context: ModuleRenderContext) -> bool:
    """Prove a loop-carried result uses the named store's arranged domain."""
    domain = tuple(str(axis) for axis in context.axes)
    program = context.kernel.ssa

    if len(domain) != 2 or len(context.outputs) != 1 or len(context.stores) != 1:
        return False

    store = context.stores[0]

    if (
        program is None
        or len(program.blocks) != 1
        or store not in program.blocks[0].operations
        or store.attrs.get("source")
        or store.attrs.get("indices")
        or store.attrs.get("target_dtype_level", 0) != 0
        or store.attrs.get("base_dtype_level", 0) != 0
        or len(store.operands) != 2
    ):
        return False

    value, output = store.operands
    info = context.tensors.get(output)
    output_type = context.value_types.get(output)
    value_type = context.value_types.get(value)
    loop = context.operations.get(value)
    attrs = info.attrs or {} if info is not None else {}

    if (
        output != context.outputs[0]
        or store.attrs.get("target") != output
        or info is None
        or info.ndim != 2
        or output_type is None
        or value_type is None
        or tuple(str(axis) for axis in store.attrs.get("target_shape", ())) != domain
        or tuple(str(axis) for axis in output_type.shape) != domain
        or tuple(str(axis) for axis in value_type.shape) != domain
        or tuple(str(axis) for axis in attrs.get("application_shape", ())) != domain
        or tuple(
            tuple(str(axis) for axis in shape)
            for shape in attrs.get("dtype_shapes", ())
        )
        != (domain,)
        or loop is None
        or loop.opcode != "scf.for"
        or len(loop.results) != 1
        or len(loop.regions) != 1
        or len(loop.attrs.get("iter_args", ())) != 1
    ):
        return False

    region = loop.regions[0]
    carried = loop.attrs["iter_args"][0]

    if (
        not region.operations
        or region.operations[-1].opcode != "scf.yield"
        or len(region.operations[-1].operands) != 1
        or not any(
            op.opcode.startswith("reduce.") for op in _walk_ops(region.operations)
        )
    ):
        return False

    names = (
        carried.get("initial"),
        carried.get("block_arg"),
        region.operations[-1].operands[0],
    )

    return all(
        (type_ := context.value_types.get(name)) is not None
        and tuple(str(axis) for axis in type_.shape) == domain
        for name in names
    )


def _ascendc_validate_reduction_lowering(context: ModuleRenderContext) -> None:
    """Require a supported row schedule or a proven named store domain.

    Scalar loop-carried results may use the shared emitter's target-shaped
    domain when their shapes and store mapping agree. Unproven rank-dropped
    stores remain unsupported. Physical writeback ownership is checked
    separately by the indexed reduction fallback.
    """
    program = context.kernel.ssa
    schedule = dict(program.metadata.get("schedule", {})) if program else {}
    reduction = schedule.get("reduction", {})

    if not isinstance(reduction, Mapping):
        return

    mode = reduction.get("mode")

    if mode == "none":
        return

    if context.cooperative_reduction_program:
        # A store target with fewer dimensions than the reduction domain
        # still navigates through collapsed addressing and computes
        # out-of-domain output addresses.
        value_shape = tuple(reduction.get("value_shape", ()))
        outputs = tuple(context.outputs)

        if value_shape and outputs:
            info = context.tensors.get(outputs[0])

            if info is not None and info.ndim != len(value_shape):
                raise ValueError(
                    "The AscendC backend requires reduction store targets "
                    "to keep the dimensionality of the reduction domain."
                )

        return

    program_ops = context.kernel.ssa.blocks if program else ()
    has_tensor_stores = any(
        operation.opcode == "mem.store" and len(operation.operands) == 2
        for block in program_ops
        for operation in _walk_ops(block.operations)
    )

    if not has_tensor_stores:
        return

    if context.lowering_plans.get("row") is not None:
        # Collapsed (rows, 1) stores lower through the per-row kernel
        # instead of the generic flat-index path.
        return

    if _ascendc_same_domain_loop_store(context):
        # The shared emitter indexes the named target shape, including
        # loop-carried tensor results. Scalar writeback still uses the
        # geometry/physical-layout ownership guard in the indexed fallback.
        return

    # A reduction broadcast back to the full input domain is addressable
    # by the indexed scalar path.  This also permits the symbolic source
    # emission performed before tile specialization picks its schedule.
    if context.outputs:
        output = context.tensors.get(context.outputs[0])

        if output is not None and output.ndim == 2:
            attrs = output.attrs or {}
            shape = attrs.get("source_shape") or output.shape

            if shape and len(shape) == 2 and str(shape[1]) != "1":
                return

    raise ValueError(
        "The AscendC backend only supports reductions lowered through the "
        "cooperative row-vector schedule; tile the reduction axis so one "
        "tile spans its full extent (e.g. BLOCK >= axis length)."
    )


def _elementwise_total_param(context: ModuleRenderContext) -> str | None:
    """Use the output extent; vector staging proves input equality at runtime."""
    if len(context.outputs) != 1:
        return None

    info = context.tensors.get(context.outputs[0])

    if info is None or info.ndim != 1:
        return None

    shape = (info.attrs or {}).get("source_shape") or info.shape

    return str(shape[0]) if shape else None


def _ascendc_validate_atomics(context: ModuleRenderContext) -> None:
    """Reject mem.atomic_add: scalar atomics are unavailable on AscendC.

    The CANN runtime may execute one logical block on several cores without
    distinct block indices, so neither a serialized single block nor a
    block-indexed partition can guarantee exactly-once read-modify-write
    updates.  Fail at compile time instead of silently corrupting results.
    """
    program = context.kernel.ssa

    if program is None:
        return

    uses_atomics = any(
        operation.opcode == "mem.atomic_add"
        for block in program.blocks
        for operation in _walk_ops(block.operations)
    )

    if uses_atomics:
        raise ValueError(
            "The AscendC backend does not support atomic updates: scalar "
            "atomics are unavailable and block execution is not guaranteed "
            "to be exclusive, so read-modify-write updates cannot be made "
            "exact."
        )


def _ascendc_support(context: ModuleRenderContext, *, force_math: bool = False) -> str:
    """Emit helper functions required by the kernel body."""
    operations = {id(operation): operation for operation in context.operations.values()}
    lines: list[str] = []

    needed_math = sorted(
        {
            _MATH_OPCODES[operation.opcode]
            for operation in operations.values()
            if operation.opcode in _MATH_OPCODES
        }
    )
    derived = set()

    for name in needed_math:
        if name == "nt_log":
            derived.add("nt_log")
        elif name in {"nt_pow"}:
            derived.update({"nt_exp", "nt_log"})
        elif name in {"nt_tanh", "nt_sigmoid"}:
            derived.add("nt_exp")
        elif name in {"nt_sin"}:
            pass
        elif name == "nt_cos":
            derived.add("nt_sin")
        elif name == "nt_tan":
            derived.update({"nt_sin", "nt_cos"})
        elif name in {"nt_sinh", "nt_cosh"}:
            derived.add("nt_exp")
        elif name in {"nt_asin", "nt_acos"}:
            derived.update({"nt_atan"})
        elif name == "nt_atan2":
            derived.add("nt_atan")
        elif name == "nt_erf":
            derived.add("nt_exp")

    uses_math = (
        force_math
        or "nt_nan(" in context.body
        or any(operation.opcode in _MATH_OPCODES for operation in operations.values())
    )

    # An `x ** y` expression lowers to `arith.pow` -> `nt_pow`, and
    # namespace calls emit `call.{name}` ops; both reference software math
    # helpers without any `math.*` opcode being present.
    nt_funcs = {
        value for value in _ASCENDC_CALL_FUNCS.values() if value.startswith("nt_")
    }
    needed_calls = {
        _ASCENDC_CALL_FUNCS.get(operation.opcode[len("call.") :])
        for operation in operations.values()
        if operation.opcode.startswith("call.")
    }
    needed_opcodes = {operation.opcode for operation in operations.values()} & {
        "arith.pow"
    }
    uses_math = uses_math or bool(needed_calls & nt_funcs) or bool(needed_opcodes)

    if uses_math:
        lines.append(_ASCENDC_MATH_SUPPORT)
    else:
        # Nt_inf() is referenced by tile `other` defaults and reduction
        # identities even in math-free kernels.
        lines.append(
            """union NtFloatBits {
    float f;
    int32_t i;
};

__aicore__ inline float nt_inf() {
    NtFloatBits v;
    v.i = 0x7f800000;
    return v.f;
}"""
        )

    if any(operation.opcode == "math.rand" for operation in operations.values()):
        lines.append(
            """\
// 64-bit immediates cannot be loaded on the scalar path, so the generator
// is kept entirely in uint32 arithmetic.
__aicore__ inline float ninetoothed_rand_uniform(
    uint64_t seed, uint64_t offset
) {
    uint32_t z = (uint32_t)(seed) ^ ((uint32_t)(offset) * 0x9E3779B1u);
    z = z + 0x6D2B79F5u;
    z = (z ^ (z >> 15)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z = z ^ (z >> 16);
    // aicore disallows direct uint-to-float casts; the shift guarantees
    // the value fits in a positive int32 so the signed cast is exact.
    return (float)(int32_t)(z >> 8) * (1.0f / 16777216.0f);
}"""
        )

    return "\n\n".join(lines)


def _render_signature_params(params: list[str]) -> str:
    return ",\n".join(f"    {param}" for param in params)


def _signature_param(name, info) -> str:
    dtype = _ascendc_type(info.dtype)

    if info.ndim == 0:
        return f"{dtype} {name}"

    return f"GM_ADDR {name}_gm"


def _launch_param(name, info, *, half_storage=None) -> str:
    if info.ndim == 0:
        if half_storage is not None and name in half_storage:
            return f"uint16_t {half_storage[name]}"

        return f"{_ascendc_type(info.dtype)} {name}"

    return f"void* {name}"


def _host_auxiliary_param(name, tensors, half_storage) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _launch_param(name, info, half_storage=half_storage)
    return f"int64_t {name}"


def _auxiliary_param(name, tensors) -> str:
    info = tensors.get(name)

    if info is not None and info.ndim == 0:
        return _signature_param(name, info)
    return f"int64_t {name}"


def _ascendc_type(dtype: str | None, kind: str | None = None) -> str:
    dtype = common.normalize_dtype(dtype)

    if kind == "pointer":
        return f"{_ascendc_type(dtype)}*"

    if kind == "index" or dtype in {"index", "int64"}:
        return "int64_t"

    types = {
        "float32": "float",
        "float16": "half",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "int32": "int32_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
        "bool": "bool",
    }

    if dtype == "float64":
        raise ValueError(
            "The AscendC backend does not support `float64` device arithmetic "
            "or conversions on the dav-c220 target."
        )

    if dtype in {"bfloat16", "float8_e4m3fn", "float8_e5m2"}:
        raise ValueError(
            f"The AscendC backend does not support `{dtype}` in the scalar "
            "lowering yet."
        )

    if dtype not in types:
        raise ValueError(f"Unsupported AscendC SSA dtype: {dtype!r}.")
    return types[dtype]


def _ascendc_arithmetic_result_type(op: ssa.Operation, ctx: _EmitContext) -> ssa.Type:
    result_type = op.results[0].type

    if not op.opcode.startswith("arith."):
        return result_type

    if _normalize_dtype(result_type.dtype) not in {"float16"}:
        return result_type
    return replace(result_type, dtype="float32")


def _coerce_ascendc_binary_args(
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

    dtype = (
        _normalize_dtype(result_type.dtype)
        if result_type is not None and _normalize_dtype(result_type.dtype) != "bool"
        else _ascendc_common_operand_dtype(op.operands, ctx)
    )

    if dtype is None:
        return args

    if dtype == "float16":
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


def _ascendc_common_operand_dtype(
    operands: tuple[str, ...], ctx: _EmitContext
) -> str | None:
    ranks = {
        "bool": 0,
        "int32": 1,
        "int64": 2,
        "float16": 3,
        "float32": 4,
        "float64": 5,
    }
    dtypes = [
        _normalize_dtype(type_.dtype)
        for operand in operands
        if (type_ := ctx.value_types.get(operand)) is not None
    ]

    return max(dtypes, key=lambda dtype: ranks.get(dtype, -1)) if dtypes else None


TARGET = AscendCTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["AscendCTarget", "TARGET", "emit"]
