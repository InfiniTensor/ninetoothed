"""Optional complete dot-loop lowering to CANN Cube primitives."""

import math
from dataclasses import dataclass

from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.dot_layout import (
    DenseDotAccess,
    prove_dense_dot_access,
)


@dataclass(frozen=True)
class CubeDotPlan:
    left: str
    right: str
    output: str
    left_access: DenseDotAccess
    right_access: DenseDotAccess
    output_access: DenseDotAccess
    stages: int
    tasks: int
    output_dtype: str
    origin: str = "source-k-group-loop"
    right_transposed: bool = False

    def metadata(self):
        return {
            "ascendc_cube_dot": {
                "enabled": True,
                "tile": (32, 32, 16),
                "input_dtype": "float16",
                "accumulator_dtype": "float32",
                "output_dtype": self.output_dtype,
                "stages": self.stages,
                "origin": self.origin,
                "right_transposed": self.right_transposed,
                "accumulation_order": "serial-k-fused"
                if self.stages > 1
                else "single-dot",
                "requires_reassociation": self.stages > 1
                and self.origin == "source-k-group-loop",
                "l1_bytes": 2048,
                "l0a_bytes": 1024,
                "l0b_bytes": 1024,
                "l0c_bytes": 4096,
                "partial_tiles": "indexed-fallback",
            }
        }


@dataclass(frozen=True)
class NativeDotBody:
    body: str
    tasks: str
    guard: str
    metadata: dict


def _bound(name, operations, types):
    op = operations.get(name)

    if op is None:
        return None

    if op.opcode == "arith.constant":
        value = op.attrs.get("value")

        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if (
        op.opcode == "shape.dim"
        and len(op.operands) == 1
        and not op.attrs.get("source")
    ):
        try:
            return int(types[op.operands[0]].shape[int(op.attrs["dim"])])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
    return None


def plan_cube_dot(context, *, allow_reassociation=False):
    """Keep user-authored multi-stage accumulation explicitly opt-in."""
    return _plan_cube_dot(context, allow_reassociation=allow_reassociation)


def _plan_cube_dot(
    context,
    *,
    allow_reassociation=False,
    origin="source-k-group-loop",
    right_transposed=False,
):
    """Reject effects, nested reductions, noncanonical maps and masked tails."""
    if (
        len(context.stores) != 1
        or len(context.outputs) != 1
        or len(context.kernel.ssa.blocks) != 1
        or context.cooperative_reduction_program
    ):
        return None

    top = context.kernel.ssa.blocks[0].operations
    loops = tuple(op for op in top if op.opcode == "scf.for")

    if len(loops) != 1:
        return None

    loop = loops[0]
    store = context.stores[0]

    if (
        not top
        or top[-1] is not store
        or len(loop.results) != 1
        or len(loop.regions) != 1
        or len(loop.operands) != 4
        or len(loop.attrs.get("iter_args", ())) != 1
        or len(store.operands) != 2
        or set(store.attrs) - {"target", "target_shape"}
        or tuple(map(str, store.attrs.get("target_shape", ()))) != ("32", "32")
        or tuple(map(str, context.axes)) != ("32", "32")
        or common.normalize_dtype(loop.results[0].type.dtype) != "float32"
        or tuple(map(str, loop.results[0].type.shape)) != ("32", "32")
    ):
        return None

    carry = loop.attrs["iter_args"][0]
    initial = context.operations.get(carry["initial"])

    if (
        initial is None
        or initial.opcode not in {"tensor.zeros", "tensor.full"}
        or loop.operands[-1] != carry["initial"]
        or initial.attrs.get("value", 0) != 0
        or common.normalize_dtype(initial.results[0].type.dtype) != "float32"
        or tuple(map(str, initial.results[0].type.shape)) != ("32", "32")
    ):
        return None

    lower, stages, step = (
        _bound(name, context.operations, context.value_types)
        for name in loop.operands[:3]
    )

    if lower != 0 or step != 1 or not isinstance(stages, int) or not 0 < stages <= 2047:
        return None

    if stages > 1 and not allow_reassociation:
        return None

    inner = loop.regions[0].operations
    dots = [op for op in inner if op.opcode in {"linalg.dot", "linalg.matmul"}]
    additions = [op for op in inner if op.opcode == "arith.add"]
    extracts = [op for op in inner if op.opcode == "tensor.extract"]

    if len(inner) != 5 or len(dots) != 1 or len(additions) != 1 or len(extracts) != 2:
        return None

    dot, addition = dots[0], additions[0]

    if (
        any(op.regions for op in inner)
        or len(dot.operands) != 2
        or len(dot.results) != 1
        or len(addition.results) != 1
        or set(addition.operands) != {carry["block_arg"], dot.results[0].name}
        or inner[-1].opcode != "scf.yield"
        or inner[-1].operands != (addition.results[0].name,)
        or common.normalize_dtype(dot.results[0].type.dtype) != "float32"
        or common.normalize_dtype(addition.results[0].type.dtype) != "float32"
        or any(
            tuple(map(str, op.results[0].type.shape)) != ("32", "32")
            for op in (dot, addition)
        )
    ):
        return None

    stored = store.operands[0]
    cast = context.operations.get(stored)
    allowed_cast = (
        cast is not None
        and cast.opcode == "tensor.cast"
        and cast.operands[0] == loop.results[0].name
    )

    if stored != loop.results[0].name and not allowed_cast:
        return None

    allowed_top = {"arith.constant", "shape.dim", "tensor.zeros", "tensor.full"}

    for op in top:
        if op is loop or op is store or (allowed_cast and op is cast):
            continue

        if op.opcode not in allowed_top or op.regions:
            return None

    try:
        tasks = math.prod(map(int, context.outer_axes))
        out_info = context.tensors[store.operands[1]]
        m, n = map(int, out_info.source_shape)
    except (KeyError, ValueError, TypeError):
        return None

    output_dtype = common.normalize_dtype(out_info.dtype)

    if (
        m <= 0
        or n <= 0
        or m % 32
        or n % 32
        or tasks != (m // 32) * (n // 32)
        or output_dtype not in {"float16", "float32"}
    ):
        return None

    if allowed_cast and (
        len(cast.operands) != 1
        or len(cast.results) != 1
        or tuple(map(str, cast.results[0].type.shape)) != ("32", "32")
        or common.normalize_dtype(cast.attrs.get("dtype") or cast.results[0].type.dtype)
        != output_dtype
        or (
            cast.attrs.get("dtype_ref") is not None
            and cast.attrs["dtype_ref"] != store.operands[1]
        )
    ):
        return None

    columns = n // 32
    row = f"outer_index // {columns} * 32 + value_0"
    column = f"outer_index % {columns} * 32 + value_1"
    stage_row = "extract_0_0 * 32 + value_0"
    stage_column = "extract_0_0 * 32 + value_1"
    accesses = []
    names = []

    for operand, expected, source_shape in (
        (dot.operands[0], (row, stage_column), (m, stages * 32)),
        (dot.operands[1], (stage_row, column), (stages * 32, n)),
    ):
        extract = context.operations.get(operand)

        if (
            extract is None
            or extract not in extracts
            or extract.operands[1:] != (str(loop.attrs.get("induction")),)
            or extract.operands[0] not in context.variables
            or common.normalize_dtype(extract.results[0].type.dtype) != "float16"
            or common.normalize_dtype(context.tensors[extract.operands[0]].dtype)
            != "float16"
        ):
            return None

        access = prove_dense_dot_access(
            extract.results[0].type.attrs,
            shape=(32, 32),
            tasks=tasks,
            stages=stages,
            expected_offsets=expected,
        )

        if access is None or access.source_shape != source_shape:
            return None

        accesses.append(access)
        names.append(extract.operands[0])

    output_access = prove_dense_dot_access(
        out_info.attrs or {},
        shape=(32, 32),
        tasks=tasks,
        stages=stages,
        expected_offsets=(row, column),
    )

    if output_access is None or output_access.source_shape != (m, n):
        return None
    return CubeDotPlan(
        *names,
        store.operands[1],
        *accesses,
        output_access,
        stages,
        tasks,
        output_dtype,
        origin,
        right_transposed,
    )


def render_cube_dot(context, target, *, allow_reassociation=False):
    """Render an AIC-only body; the owner supplies its separate kernel/launcher."""
    plan = plan_cube_dot(context, allow_reassociation=allow_reassociation)

    if plan is None:
        return None
    return _render_cube_dot(plan, target)


def render_direct_cube_dot(context, target):
    """Prove an original single dot before deriving its internal Cube tiles.

    This entry point cannot reinterpret a user-authored accumulation loop as a
    single dot. The full-matrix proof retains original maps, masks and ownership;
    the Cube proof additionally rejects any unimplemented epilogue.
    """
    from ninetoothed.backends.emitters.direct_dot import tile_direct_dot

    tiled = tile_direct_dot(context)

    if tiled is None:
        return None

    plan = _plan_cube_dot(
        tiled.context,
        allow_reassociation=True,
        origin="direct-linalg-dot",
        right_transposed=tiled.right_transposed,
    )

    if plan is None:
        return None
    return _render_cube_dot(plan, target)


def _render_cube_dot(plan, target, *, task_batch=1, output_tile=32):
    if (
        not isinstance(task_batch, int)
        or isinstance(task_batch, bool)
        or task_batch not in (1, 8)
        or (task_batch != 1 and plan.origin != "explicit-dot-group-partials")
        or not isinstance(output_tile, int)
        or isinstance(output_tile, bool)
        or output_tile not in (32, 64)
    ):
        return None

    left, right, output = plan.left_access, plan.right_access, plan.output_access
    logical_tasks = plan.tasks
    task_index = "nt_cube_task"
    mapping = ""

    if output_tile == 64:
        # Only the already proved Cartesian K32 partial plan can form exact
        # 2x2 unions. Source dimensions and the scratch shape preserve every
        # original group and prove that there are no partial output tiles.
        m, k = left.source_shape
        n = right.source_shape[1]

        if (
            plan.origin != "explicit-dot-group-partials"
            or plan.stages != 1
            or plan.right_transposed
            or m % 64
            or n % 64
            or k % 32
            or right.source_shape != (k, n)
            or output.source_shape != ((k // 32) * m, n)
            or plan.tasks != (m // 32) * (n // 32) * (k // 32)
        ):
            return None

        old_tiles = (m // 32) * (n // 32)
        new_tiles = (m // 64) * (n // 64)
        logical_tasks = plan.tasks // 4
        task_index = "nt_cube_original_task"
        mapping = (
            f"    const int64_t {task_index} = (nt_cube_task / {new_tiles}) * {old_tiles}"
            f" + ((nt_cube_task % {new_tiles}) / {n // 64}) * {2 * (n // 32)}"
            f" + ((nt_cube_task % {new_tiles}) % {n // 64}) * 2;\n"
        )

    if logical_tasks < task_batch:
        task_batch = 1

    def offset(access, stage=False):
        bases = access.base_offsets(
            {
                "outer_index": task_index,
                "extract_0_0": "nt_cube_stage" if stage else "0",
            }
        )
        # The proved grid coordinates/divisors are nonnegative. The shared
        # named-call rewriter leaves Python // tokens unchanged; C division
        # safely implements their floor over this canonical domain.
        row, column = (
            common.target_index_expr(target, value).replace("//", "/")
            for value in bases
        )

        return f"(({row}) * ({access.strides[0]}) + ({column}) * ({access.strides[1]}))"

    out_type = "half" if plan.output_dtype == "float16" else "float"
    quant = "fix.quantPre = QuantMode_t::F322F16;" if out_type == "half" else ""
    right_leading = right.strides[1] if plan.right_transposed else right.strides[0]
    right_k_stride = right.strides[0]
    right_n, right_d = (output_tile, 16) if plan.right_transposed else (16, output_tile)
    transpose_b = "false" if plan.right_transposed else "true"
    setup = f"""    TPipe nt_cube_pipe;
    TBuf<TPosition::A1> nt_cube_a1_buffer;
    TBuf<TPosition::B1> nt_cube_b1_buffer;
    TBuf<TPosition::A2> nt_cube_a2_buffer;
    TBuf<TPosition::B2> nt_cube_b2_buffer;
    TBuf<TPosition::CO1> nt_cube_c_buffer;
    nt_cube_pipe.InitBuffer(nt_cube_a1_buffer, {output_tile * 32});
    nt_cube_pipe.InitBuffer(nt_cube_b1_buffer, {output_tile * 32});
    nt_cube_pipe.InitBuffer(nt_cube_a2_buffer, {output_tile * 32});
    nt_cube_pipe.InitBuffer(nt_cube_b2_buffer, {output_tile * 32});
    nt_cube_pipe.InitBuffer(nt_cube_c_buffer, {output_tile * output_tile * 4});
    auto a1 = nt_cube_a1_buffer.Get<half>();
    auto b1 = nt_cube_b1_buffer.Get<half>();
    auto a2 = nt_cube_a2_buffer.Get<half>();
    auto b2 = nt_cube_b2_buffer.Get<half>();
    auto c0 = nt_cube_c_buffer.Get<float>();
    GlobalTensor<half> a_gm, b_gm;
    GlobalTensor<{out_type}> c_gm;
    a_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>({plan.left}), {math.prod(left.source_shape)});
    b_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>({plan.right}), {math.prod(right.source_shape)});
    c_gm.SetGlobalBuffer(reinterpret_cast<__gm__ {out_type}*>({plan.output}), {math.prod(output.source_shape)});
    Nd2NzParams copy_a;
    copy_a.ndNum = 1; copy_a.nValue = {output_tile}; copy_a.dValue = 16;
    copy_a.srcDValue = (uint16_t)({left.strides[0]});
    copy_a.dstNzC0Stride = {output_tile}; copy_a.dstNzNStride = 1; copy_a.dstNzMatrixStride = {output_tile * 16};
    Nd2NzParams copy_b;
    copy_b.ndNum = 1; copy_b.nValue = {right_n}; copy_b.dValue = {right_d};
    copy_b.srcDValue = (uint16_t)({right_leading});
    copy_b.dstNzC0Stride = {right_n}; copy_b.dstNzNStride = 1; copy_b.dstNzMatrixStride = {output_tile * 16};
    LoadData2DParams load_a;
    load_a.repeatTimes = {output_tile // 16}; load_a.srcStride = 1; load_a.ifTranspose = false;
    LoadData2DParams load_b = load_a; load_b.ifTranspose = {transpose_b};
"""
    task_body = (
        mapping
        + f"""    for (int nt_cube_stage = 0; nt_cube_stage < {plan.stages}; ++nt_cube_stage) {{
        for (int nt_cube_k = 0; nt_cube_k < 32; nt_cube_k += 16) {{
            DataCopy(a1, a_gm[{offset(left, True)} + nt_cube_k], copy_a);
            DataCopy(b1, b_gm[{offset(right, True)} + nt_cube_k * ({right_k_stride})], copy_b);
            PipeBarrier<PIPE_ALL>();
            LoadData(a2, a1, load_a); LoadData(b2, b1, load_b);
            PipeBarrier<PIPE_ALL>();
            MmadParams mm;
            mm.m = {output_tile}; mm.n = {output_tile}; mm.k = 16;
            mm.cmatrixInitVal = nt_cube_stage == 0 && nt_cube_k == 0;
            Mmad(c0, a2, b2, mm);
            PipeBarrier<PIPE_ALL>();
        }}
    }}
    FixpipeParamsV220 fix;
    fix.nSize = {output_tile}; fix.mSize = {output_tile}; fix.srcStride = {output_tile};
    fix.dstStride = (uint32_t)({output.strides[0]});
    {quant}
    Fixpipe<{out_type}, float, CFG_ROW_MAJOR>(c_gm[{offset(output)}], c0, fix);
    PipeBarrier<PIPE_ALL>();"""
    )
    tasks = (logical_tasks + task_batch - 1) // task_batch

    if task_batch == 1:
        body = (
            f"    const int64_t nt_cube_task = (int64_t)GetBlockIdx();\n"
            f"    if (nt_cube_task >= {logical_tasks}) {{ return; }}\n"
            + setup
            + task_body
        )
    else:
        # The original Cartesian partial tasks have disjoint stores. Reuse
        # only allocation/setup: each task still resets its K32 accumulator
        # and completes Fixpipe before another task touches the buffers.
        body = (
            f"    const int64_t nt_cube_job = (int64_t)GetBlockIdx();\n"
            f"    if (nt_cube_job >= {tasks}) {{ return; }}\n"
            + setup
            + f"    for (int64_t nt_cube_lane = 0; nt_cube_lane < {task_batch}; ++nt_cube_lane) {{\n"
            + f"        const int64_t nt_cube_task = nt_cube_job * {task_batch} + nt_cube_lane;\n"
            + f"        if (nt_cube_task >= {logical_tasks}) {{ break; }}\n"
            + common.indent_block(task_body, "    ")
            + "\n    }"
        )

    guard = " && ".join(
        [
            *(f"({access.strides[1]}) == 1" for access in (left, output)),
            *(
                f"({access.strides[0]}) == {access.source_shape[1]}"
                for access in (left, output)
            ),
            f"({right.strides[0] if plan.right_transposed else right.strides[1]}) == 1",
            f"({right_leading}) == {right.source_shape[0] if plan.right_transposed else right.source_shape[1]}",
            f"({left.strides[0]}) > 0 && ({left.strides[0]}) <= 65535",
            f"({right_leading}) > 0 && ({right_leading}) <= 65535",
            f"({output.strides[0]}) >= {output.source_shape[1]}",
            f"({output.strides[0]}) <= 4294967295LL",
            f"({output.strides[0]}) % 32 == 0",
        ]
    )

    metadata = plan.metadata()
    metadata["ascendc_cube_dot"].update(
        logical_tasks=logical_tasks, launch_tasks=tasks, task_batch=task_batch
    )

    if output_tile != 32:
        metadata["ascendc_cube_dot"].update(
            tile=(output_tile, output_tile, 16),
            original_tasks=plan.tasks,
            l1_bytes=output_tile * 64,
            l0a_bytes=output_tile * 32,
            l0b_bytes=output_tile * 32,
            l0c_bytes=output_tile * output_tile * 4,
            output_union="exact-two-by-two-original-tiles",
        )

    return NativeDotBody(body, str(tasks), guard, metadata)


def render_cube_plan(plan, target, *, task_batch=1, output_tile=32):
    """Render a proved primitive plan without selecting or rewriting SSA."""
    return _render_cube_dot(
        plan, target, task_batch=task_batch, output_tile=output_tile
    )


__all__ = [
    "CubeDotPlan",
    "NativeDotBody",
    "plan_cube_dot",
    "render_cube_dot",
    "render_direct_cube_dot",
    "render_cube_plan",
]
