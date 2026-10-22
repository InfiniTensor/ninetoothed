"""TileLang syntax hooks for the common SSA emitter."""

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.context import CooperativeDotPlan
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.backends.emitters.context import TensorInfo as _TensorInfo
from ninetoothed.backends.tilelang_plan import plan_tiles
from ninetoothed.ir import Kernel, ssa

_Target = EmitterTarget
_buffer_storage_extent = common.buffer_storage_extent
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
    tile_values: Mapping[str, str] = field(
        default_factory=dict, compare=False, repr=False
    )

    def materialized_element(self, name, coords):
        buffer = self.tile_values.get(name)

        if buffer is None:
            return None
        return f"{buffer}[{', '.join(coords) if coords else '0'}]"

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
        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return "float('nan')"

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

    def schedule_context(self, context: ModuleRenderContext) -> ModuleRenderContext:
        schedule = context.kernel.ssa.metadata.get("schedule", {})
        threads = common.schedule_int(context.kernel, "threads", 256)
        grouped = schedule.get("row_group", {}).get("enabled")
        tile = _elementwise_tile_extent(threads)
        generic_grid = (
            context.grid_total
            if grouped or context.cooperative_reduction_program
            else f"(({context.total}) + {tile - 1}) // {tile}"
        )
        plan = plan_tiles(
            program=context.kernel.ssa,
            tensors=context.tensors,
            value_types=context.value_types,
            operations=context.operations,
            stores=context.stores,
            schedule=schedule,
            options=context.kernel.compiler_options.get("backend_options", {}),
            threads=threads,
            outer_grid=common.product(context.outer_axes),
            generic_grid=generic_grid,
        )
        metadata = dict(context.scheduled_metadata) | {
            "tilelang_emission": plan.as_metadata()
        }

        return replace(
            context,
            grid_total=plan.grid,
            block_program=context.block_program or plan.actual != "generic",
            scheduled_metadata=metadata,
        )

    def render_module(self, context: ModuleRenderContext) -> str:
        plan = context.scheduled_metadata["tilelang_emission"]

        if plan["actual"] == "materialized-tiles":
            from ninetoothed.backends.emitters.tilelang_block import render

            return render(context, self)

        if plan["actual"] == "cooperative-gemm":
            return _render_tilelang_cooperative_dot_module(
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
                CooperativeDotPlan(
                    loop=context.operations[plan["loop"]],
                    dot=context.operations[plan["dot"]],
                    store=context.stores[0],
                ),
                threads=plan["threads"],
                grid_total=plan["grid"],
            )
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
    plan: CooperativeDotPlan,
    *,
    threads: int,
    grid_total: str,
) -> str:
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
    buffers = _tilelang_flat_buffers(buffer_names, tensors, _product(outer_axes))

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

    assert lower == "0" and step == "1", "GEMM bounds must match the emission plan."

    induction = str(loop.attrs.get("induction", "%iv"))
    loop_bindings = {induction: "ko"}
    probe = context([], coordinates=("0", "0"), bindings=loop_bindings)
    regions = (
        _dot_tile_region(lhs, probe, tile_shape=(bm, bk), loop_var="ko"),
        _dot_tile_region(rhs, probe, tile_shape=(bk, bn), loop_var="ko"),
        _dot_tile_region(output, probe, tile_shape=(bm, bn), loop_var="ko")
        if _is_pure_loop_store(plan.store, loop, operations, tensors[output].dtype)
        else None,
    )
    tile_decls = {}
    copies = []

    for region, shared, writing in zip(
        regions, ("a_shared", "b_shared", "c_local"), (False, False, True)
    ):
        if region is None:
            copies.append(None)
            continue

        source, shape, _, corners = region
        tile_decls[source] = (
            f"        {source}_tile = T.view({source}_buf, ({', '.join(shape)},))"
        )
        region_expr = f"{source}_tile[{', '.join(corners)}]"
        copies.append(
            f"T.copy({shared}, {region_expr})"
            if writing
            else f"T.copy({region_expr}, {shared})"
        )

    transfer_buffers = (set(), set(), set())

    def transfer_sources(scalar_access: bool):
        access_target = replace(target, scalar_global_access=scalar_access)
        lhs_lines: list[str] = []
        lhs_ctx = context(
            lhs_lines,
            coordinates=("mi", "ki"),
            bindings=loop_bindings,
            access_target=replace(access_target, _accessed_buffers=transfer_buffers[0]),
        )
        lhs_value = _emit_element(lhs, ("mi", "ki"), lhs_ctx)
        lhs_lines.append(f"a_shared[mi, ki] = {lhs_value}")
        rhs_lines: list[str] = []
        rhs_ctx = context(
            rhs_lines,
            coordinates=("ki", "ni"),
            bindings=loop_bindings,
            access_target=replace(access_target, _accessed_buffers=transfer_buffers[1]),
        )
        rhs_value = _emit_element(rhs, ("ki", "ni"), rhs_ctx)
        rhs_lines.append(f"b_shared[ki, ni] = {rhs_value}")

        output_lines: list[str] = []
        output_ctx = context(
            output_lines,
            coordinates=("mi", "ni"),
            bindings={loop.results[0].name: "c_local[mi, ni]"},
            access_target=replace(access_target, _accessed_buffers=transfer_buffers[2]),
        )
        _emit_operation(plan.store, output_ctx)

        return lhs_lines, rhs_lines, output_lines, output_ctx

    lhs_lines, rhs_lines, output_lines, output_ctx = transfer_sources(False)
    scalar_lhs, scalar_rhs, scalar_output, _ = transfer_sources(True)

    lhs_storage_dtype = _resolved_dot_operand_dtype(lhs, output_ctx)
    rhs_storage_dtype = _resolved_dot_operand_dtype(rhs, output_ctx)
    supported_storage_dtypes = {
        "float16",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
        "float32",
    }

    assert not (
        lhs_storage_dtype not in supported_storage_dtypes
        or rhs_storage_dtype not in supported_storage_dtypes
    ), "GEMM dtypes must match the emission plan."

    lhs_dtype = _tile_dtype(lhs_storage_dtype)
    rhs_dtype = _tile_dtype(rhs_storage_dtype)
    accumulator_dtype = _tile_dtype(_dot_accumulator_dtype(dot, output_ctx))
    grid_total = _rewrite_index_math(grid_total, c_style=False)
    bound_source = (
        "\n" + _indent_block("\n".join(bound_lines), "            ")
        if bound_lines
        else ""
    )

    alignment_checks: list[str] = []

    def parallel_source(lines, loop_header, indent):
        return _indent_block(
            f"{loop_header}:\n" + _indent_block("\n".join(lines), "    "), indent
        )

    def transfer_source(native, scalar, loop_header, indent, accessed_buffers, copy):
        # Only the buffers accessed by this transfer select its scalar fallback.
        alignment = " & ".join(
            f'T.call_extern("bool", "_nt_aligned", {name}.data)'
            for name in sorted(accessed_buffers)
        )

        native_source = copy or (
            f"{loop_header}:\n" + _indent_block("\n".join(native), "    ")
        )

        if not alignment:
            return _indent_block(native_source, indent)

        alignment_name = f"nt_aligned_{len(alignment_checks)}"
        alignment_checks.append(f"{alignment_name} = {alignment}")
        scalar_source = f"{loop_header}:\n" + _indent_block("\n".join(scalar), "    ")
        branches = (
            f"if {alignment_name}:\n"
            + _indent_block(native_source, "    ")
            + "\nelse:\n"
            + _indent_block(scalar_source, "    ")
        )

        return _indent_block(branches, indent)

    lhs_loop = f"for mi, ki in T.Parallel({bm}, {bk})"
    rhs_loop = f"for ki, ni in T.Parallel({bk}, {bn})"
    lhs_source = transfer_source(
        lhs_lines,
        scalar_lhs,
        lhs_loop,
        "                ",
        transfer_buffers[0],
        copies[0],
    )
    rhs_source = transfer_source(
        rhs_lines,
        scalar_rhs,
        rhs_loop,
        "                ",
        transfer_buffers[1],
        copies[1],
    )
    load_source = f"{lhs_source}\n{rhs_source}"

    if len(alignment_checks) == 2:
        # Keep aligned copies together for TileLang's pipeline optimization.
        native_loads = "\n".join(
            (
                _indent_block(copies[0], "                    ")
                if copies[0]
                else parallel_source(lhs_lines, lhs_loop, "                    "),
                _indent_block(copies[1], "                    ")
                if copies[1]
                else parallel_source(rhs_lines, rhs_loop, "                    "),
            )
        )
        load_source = (
            "                if nt_aligned_0 & nt_aligned_1:\n"
            f"{native_loads}\n                else:\n"
            + _indent_block(load_source, "    ")
        )

    output_source = transfer_source(
        output_lines,
        scalar_output,
        f"for mi, ni in T.Parallel({bm}, {bn})",
        "            ",
        transfer_buffers[2],
        copies[2],
    )
    alignment_source = _indent_block("\n".join(alignment_checks), "            ")

    dot_body = f"""        with T.Kernel({grid_total}, threads={threads}) as outer_index:{bound_source}
            T.import_source(_NT_SCALAR_ACCESS)
{alignment_source}
            a_shared = T.alloc_shared(({bm}, {bk}), {lhs_dtype})
            b_shared = T.alloc_shared(({bk}, {bn}), {rhs_dtype})
            c_local = T.alloc_fragment(({bm}, {bn}), {accumulator_dtype})
            T.clear(c_local)
            for ko in T.Pipelined({upper}, num_stages={num_stages}):
{load_source}
                T.gemm(a_shared, b_shared, c_local)
{output_source}"""
    tile_buffers = "\n".join(tile_decls.values())

    return _render_tilelang_module_skeleton(
        kernel,
        target,
        note="cooperative linalg.dot -> T.gemm",
        handle_args=handle_args,
        buffers=buffers,
        body=dot_body,
        extra_decls=("\n" + tile_buffers) if tile_buffers else "",
    )


def _tilelang_signature(variables, outputs, shape_params, tensors, value_types):
    """Build the prim_func parameter list and the flat-buffer name list.

    Shared by the generic and cooperative-dot TileLang renderers.
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

    term = ctx.target.cast(accumulator_dtype, _emit_element(operand, coordinates, body))
    mask = common.reduction_input_mask(operand, coordinates, body)

    if mask is not None:
        term = ctx.target.where(mask, term, identity)

    body_lines.append(f"{staged}[{reduction_index}] = {term}")

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
    src_name = name

    if name not in ctx.tensor_infos:
        op = ctx.operations.get(name)
        # The access template's one extract placeholder is only equivalent to
        # this loop when the actual SSA index is its induction variable.

        if (
            op is None
            or op.opcode != "tensor.extract"
            or len(op.operands) != 2
            or op.attrs.get("source")
            or ctx.bindings.get(op.operands[1]) != loop_var
        ):
            return None

        src_name = op.operands[0]

    info = ctx.tensor_infos.get(src_name)
    attrs = info.attrs if info is not None else None

    if (
        attrs is None
        or attrs.get("layout_operations") is None
        or set(attrs["layout_operations"]) - {"tile", "expand", "squeeze", "unsqueeze"}
        or attrs.get("custom_tiling", True)
        or attrs.get("jagged_offsets_param") is not None
        or attrs.get("other") not in (None, 0)
    ):
        return None

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

        if "extract_" in corner:
            return None

        corner = common.rewrite_index_math(corner, c_style=False)
        corners.append(corner)

    return src_name, source_shape, strides, tuple(corners)


def _is_pure_loop_store(
    store: ssa.Operation,
    loop: ssa.Operation,
    operations: Mapping[str, ssa.Operation],
    output_dtype: str,
) -> bool:
    """Accept the accumulator or exactly the conversion performed by T.copy."""
    if not store.operands or not loop.results:
        return False

    target = loop.results[0].name
    current = store.operands[0]

    if current == target:
        return True

    op = operations.get(current)

    return bool(
        op is not None
        and op.opcode == "tensor.cast"
        and op.operands == (target,)
        and op.results
        and _normalize_dtype(op.results[0].type.dtype) == _normalize_dtype(output_dtype)
    )


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
