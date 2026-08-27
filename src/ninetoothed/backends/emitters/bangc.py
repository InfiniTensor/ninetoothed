"""BangC syntax hooks for the common SSA emitter.

The Cambricon BANG-C task model differs from CUDA threads: one kernel task
(``taskIdX``) executes on one MLU core, so the generic flat-index program is
rendered as a per-task scalar loop over a fixed element chunk.  Kernel launch
uses the ``<<<dim, ktype, queue>>>`` trigram with ``cnrtDim3_t`` grids and a
``cnrtQueue_t`` stream, matching the CUDA backend's C-ABI launcher contract.
"""

import math
import re
from dataclasses import dataclass, replace
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.backends.emitters.context import EmitContext as _EmitContext
from ninetoothed.ir import Kernel, ssa

_access_axes = common.access_axes
_combined_mask = common.combined_mask
_current_coords = common.current_coords
_dtype_level = common.dtype_level
_emit_element = common.emit_element
_emit_loop_bound = common.emit_loop_bound
_emit_value = common.emit_value
_fresh_temp = common.fresh_temp
_indent_lines = common.indent_lines
_linearized_index = common.linearized_index
_load_other = common.load_other
_local_symbol = common.local_symbol
_materialize_bool_expr = common.materialize_bool_expr
_materialize_index_expr = common.materialize_index_expr
_nested_local_suffix = common.nested_local_suffix
_normalize_dtype = common.normalize_dtype
_reduction_identity = common.reduction_identity
_source_index_for_value = common.source_index_for_value
_target_index_expr = common.target_index_expr
_value_axes = common.value_axes
_view_base_coords = common.view_base_coords


@dataclass(frozen=True, kw_only=True)
class BangCTarget(EmitterTarget):
    backend: Target = Target.BANGC
    language: str = "bangc/c++"
    suffix: str = "mlu"
    source_route: str = "ssa-unified-bangc-emitter"
    c_style_syntax: bool = True

    def index_cast(self, value: str) -> str:
        return f"(int64_t)({value})"

    def assign_scalar(self, name: str, value: str, *, mutable: bool) -> str:
        del mutable

        return f"{name} = {value};"

    def literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, float) and math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"

        if value == "inf":
            return "INFINITY"

        if value == "-inf":
            return "-INFINITY"
        return repr(value)

    def load(self, tensor, index, *, mask=None, other=0.0):
        del mask, other
        rendered = f"({index})" if index.startswith("nt_idx_") else index

        return f"{self.tensor_ref(tensor)}[{rendered}]"

    def store(self, tensor, index, value, *, mask=None):
        rendered = f"({index})" if index.startswith("nt_idx_") else index
        assignment = f"{self.tensor_ref(tensor)}[{rendered}] = {value};"

        return assignment if mask is None else f"if ({mask}) {{\n    {assignment}\n}}"

    def cast(self, dtype, value):
        normalized = _normalize_dtype(dtype)

        if normalized == "bfloat16":
            # C-style casts to bfloat16_t are ambiguous on cncc; use the
            # explicit conversion intrinsic instead.
            return f"__float2bfloat16_rn((float)({value}))"

        if normalized == "float16":
            return f"__float2half_rn((float)({value}))"

        return f"(({self.type_name(dtype)})({value}))"

    def type_name(self, dtype, kind=None):
        return _bangc_type(dtype, kind)

    def where(self, cond, yes, no):
        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where":
            return self.where(args[0], args[1], args[2])

        if name == "atomic_add":
            return f"ninetoothed_atomic_add({args[0]}, {args[1]})"

        if name == "load" and args:
            return f"*({args[0]})"

        if name in {"block_dot", "dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "rand" and len(args) >= 2:
            return f"ninetoothed_rand_uniform(({args[0]}), ({args[1]}))"

        if name == "rsqrt" and len(args) == 1:
            return f"(1.0f / sqrtf({args[0]}))"

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
            "maximum": "fmaxf",
            "max": "fmaxf",
            "minimum": "fminf",
            "min": "fminf",
            "pow": "powf",
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
            f"ninetoothed_atomic_add_{suffix}({operands[0]}, {operands[1]})"
            if suffix is not None
            else f"ninetoothed_atomic_add_f32({operands[0]}, {operands[1]})"
        )

    def emit_dot_operand(self, name, coords, context):
        return _emit_bangc_direct_dot_operand(name, coords, context)

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        if type_.kind == "pointer":
            return f"auto {name} = {expr};"
        return f"{self.type_name(type_.dtype, type_.kind)} {name} = {expr};"

    def loop_header(self, var, lower, upper, step):
        return f"for (int64_t {var} = {lower}; {var} < {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"{acc} + {term}"

        function = "fmaxf" if operator == "max" else "fminf"

        return f"{function}({acc}, {term})"

    def arithmetic_result_type(self, operation, context) -> ssa.Type:
        return _bangc_arithmetic_result_type(operation, context)

    def coerce_binary_args(self, operation, args, context):
        return _coerce_bangc_binary_args(operation, args, context)

    def render_module(self, context: ModuleRenderContext) -> str:
        if context.vector_program or context.block_program or context.scalar_program:
            raise ValueError(
                "The BangC backend renders every kernel through the generic "
                "flat-index task domain; vector/block program modes are not "
                "supported."
            )

        kernel = context.kernel
        options = dict(kernel.compiler_options.get("backend_options", {}))
        default_chunk = int(options.get("task_chunk", 1024))
        chunk = common.schedule_int(kernel, "bangc_task_chunk", default_chunk)
        total = _bangc_integer_expr(context.total)
        grid_total = _bangc_integer_expr(context.grid_total)
        body = _bangc_integer_expr(context.body)
        staging = _nram_staging_plan(context, body)

        if staging is not None:
            # Keep staged __nram__ buffers inside the MLU590 256 KB budget
            # (measured: 3 x 64 KB passes, 3 x 256 KB fails cnas).  Bigger
            # chunks amortize task launches and memcpy bursts (measured
            # 228 GB/s at 1024 vs 1214 GB/s at 16384), so staged kernels
            # default to the largest in-budget chunk unless overridden.
            buffer_count = len(staging.inputs) + len(staging.outputs)
            budget_chunk = max(128, _NRAM_BUDGET_BYTES // (4 * max(buffer_count, 1)))

            if "task_chunk" not in options:
                chunk = budget_chunk
            else:
                chunk = min(chunk, budget_chunk)

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
                "cnrtQueue_t queue",
            ]
        )
        args = ", ".join((*context.variables, *context.outputs, *context.shape_params))

        if context.cooperative_reduction_program:
            kernel_prelude = body
            tasks_expr = grid_total
            chunk_decl = ""
        elif _program_uses_atomics(context):
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
            kernel_prelude = _render_nram_staged_body(staging, total, chunk)
            tasks_expr = f"({total} + nt_chunk - 1) / nt_chunk"
            chunk_decl = f"    const int64_t nt_chunk = {chunk};\n"
        else:
            kernel_prelude = f"""    const int64_t nt_chunk = {chunk};
    for (int64_t nt_lane = 0; nt_lane < nt_chunk; nt_lane++) {{
        int64_t {self.index_name} = (int64_t)(taskIdX) * nt_chunk + nt_lane;
        if ({self.index_name} < {total}) {{
{common.indent_block(body, "            ")}
        }}
    }}"""
            tasks_expr = f"({total} + nt_chunk - 1) / nt_chunk"
            chunk_decl = f"    const int64_t nt_chunk = {chunk};\n"

        support = _bangc_support(context)

        return f"""// Generated by NineToothed's BangC SSA backend.
// Kernel: {kernel.kernel_name}
// Lowering IR: ssa.Program

#include <bang.h>
#include <cnrt.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

{support}

__mlu_entry__ void {kernel.kernel_name}_kernel(
{kernel_params}
) {{
{kernel_prelude}
}}

extern "C" int launch_{kernel.kernel_name}(
{launch_params}
) {{
{chunk_decl}    int64_t nt_tasks = {tasks_expr};
    if (nt_tasks <= 0) {{
        return 0;
    }}
    if (nt_tasks > 4294967295LL) {{
        return static_cast<int>(cnrtErrorInvalidKernel);
    }}
    cnrtDim3_t dim;
    dim.x = static_cast<unsigned int>(nt_tasks);
    dim.y = 1;
    dim.z = 1;
    cnrtFunctionType_t ktype = cnrtFuncTypeBlock;
    {kernel.kernel_name}_kernel<<<dim, ktype, queue>>>(
        {args}
    );
    return static_cast<int>(cnrtGetLastError());
}}
"""


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


class _StagingPlan:
    """Buffers, rewritten body, and bang-op mapping for an NRAM staged kernel."""

    def __init__(
        self,
        inputs: list[tuple[str, str]],
        outputs: list[tuple[str, str]],
        body: str,
        bang_op: tuple[str, str, str, str, str] | None,
        extent: str,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.body = body
        self.bang_op = bang_op
        self.extent = extent


_PRED_DECL = re.compile(r"bool (nt_pred_\d+) = ([^;]+);")

_NRAM_BUDGET_BYTES = 192 * 1024


def _flatten_tiled_accesses(body: str, context: ModuleRenderContext) -> str | None:
    """Rewrite the tiled predicated body into the flat ``name[index]`` form.

    Contiguous 1-D tiled renders access tensors as
    ``name[(nt_outer_index) * BLOCK + (nt_inner_index)]`` guarded by
    ``nt_pred`` bounds checks.  Inside one staged chunk those predicates are
    subsumed by ``nt_j < nt_cnt`` when (a) every predicate only references
    the tiling coordinates and size parameters and (b) every masked-load
    fallback value is zero.  Anything else (runtime flags, nonzero ``other``,
    strided access templates) bails out to the generic scalar path.
    """
    if "nt_pred" not in body:
        return body

    if "nt_idx_" in body:
        return None

    allowed_symbols = {
        "nt_outer_index",
        "nt_inner_index",
        "index",
        "true",
        "false",
        *context.shape_params,
    }

    for _, expression in _PRED_DECL.findall(body):
        symbols = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))

        if not symbols <= allowed_symbols:
            return None

    tensor_names = sorted((*context.variables, *context.outputs), key=len, reverse=True)
    flattened = body

    for name in tensor_names:
        tiled = re.compile(
            rf"\b{re.escape(name)}\[\(nt_outer_index\) \* \(?(\d+)\)? \+ \(nt_inner_index\)\]"
        )

        if not tiled.search(flattened):
            continue

        flattened = tiled.sub(f"{name}[__nt_flat__]", flattened)

    # Masked loads with a zero fallback collapse to the load itself.
    flattened = re.sub(
        r"\(\((nt_pred_\d+)\) \? \((\w+)\[__nt_flat__\]\) : \(\(\((?:float|int32_t|int64_t)\)\((?:0(?:\.0)?|0\.0f)\)\)\)\)",
        r"\2[__nt_flat__]",
        flattened,
    )

    # Masked stores drop their (bounds-only) predicate.
    flattened = re.sub(
        r"if \(nt_pred_\d+\) \{\n(\s+)(\w+)\[__nt_flat__\] = ([^;]+);\n\s*\}",
        r"\1\2[__nt_flat__] = \3;",
        flattened,
    )

    # Predicate declarations are no longer referenced.
    flattened = _PRED_DECL.sub("", flattened)

    # Tiling coordinate declarations become dead once accesses are flat.
    flattened = re.sub(r"int64_t nt_outer_index = [^;]+;\n", "", flattened)
    flattened = re.sub(r"int64_t nt_inner_index = [^;]+;\n", "", flattened)
    flattened = re.sub(r"int64_t nt_i0 = nt_inner_index;\n", "", flattened)

    residue = flattened

    for name in tensor_names:
        residue = residue.replace(f"{name}[__nt_flat__]", "")

    if (
        "nt_outer_index" in flattened
        or "nt_inner_index" in flattened
        or "nt_pred" in flattened
        or "? (" in residue
        or "__nt_flat__" in residue
    ):
        return None

    return flattened.replace("__nt_flat__", "index")


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

    body = _flatten_tiled_accesses(body, context)

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

        if _normalize_dtype(info.dtype) != "float32":
            return None

        direct = f"{name}[index]"

        if direct not in rewritten:
            continue

        buffer = f"nt_buf_{name}"
        rewritten = rewritten.replace(direct, f"{buffer}[nt_j]")

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

    bang_op = _match_bang_operation(context)

    return _StagingPlan(inputs, outputs, rewritten, bang_op, limit_expr)


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

    if len(extents) != 1:
        return None
    return extents.pop()


_BINARY_BANG_OPS = {
    "arith.add": "__bang_add",
    "arith.sub": "__bang_sub",
    "arith.mul": "__bang_mul",
    "arith.div": "__bang_div",
    "arith.maximum": "__bang_maximum",
    "arith.minimum": "__bang_minimum",
}

_BINARY_BANG_SCALAR_OPS = {
    "arith.add": "__bang_add_scalar",
    "arith.sub": "__bang_sub_scalar",
    "arith.mul": "__bang_mul_scalar",
    "arith.div": "__bang_div_scalar",
    "arith.maximum": "__bang_maximum_scalar",
    "arith.minimum": "__bang_minimum_scalar",
}

_COMMUTATIVE_BANG_OPS = {"arith.add", "arith.mul"}

_UNARY_BANG_OPS = {
    "math.exp": "__bang_active_exp",
    "math.log": "__bang_active_log",
    "math.sqrt": "__bang_active_sqrt",
    "math.rsqrt": "__bang_active_rsqrt",
    "math.tanh": "__bang_active_tanh",
    "math.abs": "__bang_abs",
    "math.relu": "__bang_active_relu",
    "math.sigmoid": "__bang_active_sigmoid",
}


def _match_bang_operation(
    context: ModuleRenderContext,
) -> tuple[str, str, str, str, str] | None:
    """Detect mappable single-op stores in float32 for direct bang calls.

    Returns ``(function, output, lhs, rhs, kind)`` where ``kind`` is
    ``"tensor"`` (all operands staged buffers) or ``"scalar"`` (rhs is a
    runtime scalar parameter spelled directly).
    """
    stores = [operation for operation in context.stores if len(operation.operands) == 2]

    if len(stores) != 1 or len(context.outputs) != 1:
        return None

    output = context.outputs[0]
    store = stores[0]

    if store.operands[1] != output:
        return None

    producer = context.operations.get(store.operands[0])

    if producer is None or len(producer.operands) not in {1, 2}:
        return None

    output_info = context.tensors.get(output)
    result_type = producer.results[0].type if producer.results else None

    if output_info is None or _normalize_dtype(output_info.dtype) != "float32":
        return None

    if result_type is not None and _normalize_dtype(result_type.dtype) != "float32":
        return None

    def operand_kind(operand: str) -> str | None:
        info = context.tensors.get(operand)

        if info is None:
            return None

        if _normalize_dtype(info.dtype) != "float32":
            return None

        return "scalar" if info.ndim == 0 else "tensor"

    operands = tuple((operand, operand_kind(operand)) for operand in producer.operands)

    if not operands or any(kind is None for _, kind in operands):
        return None

    if producer.opcode in _BINARY_BANG_OPS and len(operands) == 2:
        (lhs, lhs_kind), (rhs, rhs_kind) = operands

        if lhs_kind == "tensor" and rhs_kind == "tensor":
            return (_BINARY_BANG_OPS[producer.opcode], output, lhs, rhs, "tensor")

        if lhs_kind == "tensor" and rhs_kind == "scalar":
            return (
                _BINARY_BANG_SCALAR_OPS[producer.opcode],
                output,
                lhs,
                rhs,
                "scalar",
            )

        if lhs_kind == "scalar" and rhs_kind == "tensor":
            function = _BINARY_BANG_SCALAR_OPS.get(producer.opcode)

            if function is None or producer.opcode not in _COMMUTATIVE_BANG_OPS:
                return None

            return (function, output, rhs, lhs, "scalar")
        return None

    if producer.opcode in _UNARY_BANG_OPS and len(operands) == 1:
        operand, kind = operands[0]

        if kind != "tensor":
            return None

        return (_UNARY_BANG_OPS[producer.opcode], output, operand, operand, "tensor")
    return None


def _render_nram_staged_body(plan: _StagingPlan, total: str, chunk: int) -> str:
    extent = _bangc_integer_expr(plan.extent)
    lines: list[str] = []
    lines.append(f"    const int64_t nt_chunk = {chunk};")
    lines.append("    const int64_t nt_base = (int64_t)(taskIdX) * nt_chunk;")
    lines.append(f"    if (nt_base >= {extent}) {{ return; }}")
    lines.append(f"    int64_t nt_cnt = {extent} - nt_base;")
    lines.append("    if (nt_cnt > nt_chunk) { nt_cnt = nt_chunk; }")

    for _, buffer in (*plan.inputs, *plan.outputs):
        lines.append(f"    __nram__ float {buffer}[{chunk}];")

    for name, buffer in plan.inputs:
        lines.append(
            f"    __memcpy({buffer}, {name} + nt_base, "
            f"(uint32_t)(nt_cnt) * sizeof(float), GDRAM2NRAM);"
        )

    bang = plan.bang_op

    if bang is not None and chunk % 128 == 0:
        function, output, lhs, rhs, kind = bang
        lines.append(
            "    const uint32_t nt_aligned = (uint32_t)((nt_cnt + 127) / 128 * 128);"
        )

        if kind == "scalar":
            lines.append(
                f"    {function}(nt_buf_{output}, nt_buf_{lhs}, {rhs}, nt_aligned);"
            )
        else:
            lines.append(
                f"    {function}(nt_buf_{output}, nt_buf_{lhs}, nt_buf_{rhs}, "
                f"nt_aligned);"
            )
    else:
        lines.append("    for (int64_t nt_j = 0; nt_j < nt_cnt; nt_j++) {")
        lines.extend(common.indent_block(plan.body, "        ").splitlines())
        lines.append("    }")

    for name, buffer in plan.outputs:
        lines.append(
            f"    __memcpy({name} + nt_base, {buffer}, "
            f"(uint32_t)(nt_cnt) * sizeof(float), NRAM2GDRAM);"
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


def _bangc_support(context: ModuleRenderContext) -> str:
    """Emit helper functions required by the kernel body."""
    operations = {id(operation): operation for operation in context.operations.values()}
    lines: list[str] = []

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

        if variant in {"f16", "bf16"}:
            load = (
                "__bfloat162float(address[0])"
                if variant == "bf16"
                else "__half2float(address[0])"
            )
            store = "__float2bfloat16_rn" if variant == "bf16" else "__float2half_rn"
            body = f"""    float nt_prev = {load};
    address[0] = {store}(nt_prev + value);"""
        else:
            body = """    address[0] = address[0] + value;"""

        lines.append(
            f"""static __mlu_func__ void ninetoothed_atomic_add_{variant}(
    {address_type}* address, float value
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


def _bangc_integer_expr(expr: str) -> str:
    previous = None
    current = common.rewrite_index_math(expr, c_style=True).replace("//", "/")
    current = re.sub(r"\bTrue\b", "true", current)
    current = re.sub(r"\bFalse\b", "false", current)
    pattern = re.compile(r"floor\(\(([^()]+)\)/([A-Za-z_][A-Za-z0-9_]*)\)")

    while current != previous:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def _bangc_arithmetic_result_type(op: ssa.Operation, ctx: _EmitContext) -> ssa.Type:
    result_type = op.results[0].type

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


def _emit_bangc_direct_dot_operand(
    name: str, coords: tuple[str, ...], ctx: _EmitContext
) -> tuple[str, str | None]:
    original_name = name
    original_coords = coords
    producer = ctx.operations.get(name)

    while (
        producer is not None and producer.opcode == "tensor.view" and producer.operands
    ):
        coords = _view_base_coords(producer, coords, ctx)
        name = producer.operands[0]
        producer = ctx.operations.get(name)

    info = ctx.tensor_infos.get(name)

    if info is not None and info.ndim == 0:
        return _emit_value(name, ctx), None

    if info is None or _load_other(info) != 0.0:
        return _emit_element(original_name, original_coords, ctx), None

    dtype_level = _dtype_level(name, ctx)
    axes = _access_axes(info, ctx, dtype_level, fallback=_value_axes(name, ctx))
    view_index = _linearized_index(coords, axes) if coords else "0"
    source_index = _target_index_expr(
        ctx.target,
        _source_index_for_value(
            info,
            view_index,
            ctx,
            level=dtype_level,
            value_coords=coords,
        ),
    )
    source_index = _materialize_index_expr(source_index, ctx)
    mask = _combined_mask(
        ctx.target,
        None,
        info,
        view_index,
        ctx=ctx,
        level=dtype_level,
        value_coords=coords,
    )

    if mask is not None:
        mask = _materialize_bool_expr(mask, ctx) or mask
    return ctx.target.load(name, source_index), mask


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
    if operator == "sum":
        return f"({lhs}) + ({rhs})"

    if dtype == "float32":
        function = "fmaxf" if operator == "max" else "fminf"

        return f"{function}({lhs}, {rhs})"

    if dtype == "float64":
        function = "fmax" if operator == "max" else "fmin"

        return f"{function}({lhs}, {rhs})"

    comparison = ">" if operator == "max" else "<"

    return f"(({lhs}) {comparison} ({rhs}) ? ({lhs}) : ({rhs}))"


TARGET = BangCTarget()


def emit(kernel: Kernel):
    return common.emit(kernel, TARGET)


__all__ = ["BangCTarget", "TARGET", "emit"]
