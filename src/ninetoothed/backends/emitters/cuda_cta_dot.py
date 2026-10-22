"""Compose WMMA fragments from an already verified SSA reduction region."""

from dataclasses import dataclass
from math import prod

from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.cuda_dot_transfer import prove_dot_transfers


@dataclass(frozen=True)
class CudaCtaDotPlan:
    """One output tile, its fragment ownership, and its shared arena lifetime."""

    rows: int = 32
    columns: int = 32
    warps: int = 4
    reduction_packet: int = 16
    left_pitch: int = 24
    right_pitch: int = 40
    input_slots: int = 2

    @property
    def arena_bytes(self):
        input_bytes = 2 * (
            self.rows * self.left_pitch + self.reduction_packet * self.right_pitch
        )
        output_bytes = self.rows * self.columns * 4

        return max(input_bytes * self.input_slots, output_bytes)


CTA_DOT_PLAN = CudaCtaDotPlan()


def selected(kernel):
    return bool(
        kernel.ssa is not None
        and kernel.ssa.metadata.get("selected_schedule_candidate") == "wmma-32x32"
        and kernel.ssa.metadata.get("schedule", {}).get("cuda_cta_dot") is True
    )


def render_launch_guard(context, target):
    """Guard whole positive-stride spans before cooperative read/write traffic."""
    if not selected(context.kernel):
        return ""

    lines = []
    indices = {}
    invalid_value = target.global_helper("cudaErrorInvalidValue")

    for index, name in enumerate((*context.variables, *context.outputs)):
        info = context.tensors[name]

        if info.ndim == 0:
            continue

        shape = tuple(map(int, info.source_shape))
        strides = info.source_strides or tuple(
            str(prod(shape[axis + 1 :])) for axis in range(len(shape))
        )

        if (
            not shape
            or len(strides) != len(shape)
            or any(dimension <= 0 for dimension in shape)
        ):
            raise ValueError(
                "CUDA CTA dot launch requires known positive physical spans."
            )

        prefix = f"nt_cta_span_{index}"
        indices[name] = prefix
        element_size = f"sizeof({target.type_name(info.dtype)})"

        if name in context.outputs:
            if len(shape) != 2:
                raise ValueError(
                    "CUDA CTA dot output requires a rank-two physical span."
                )

            row_stride, column_stride = strides
            lines.append(
                f"if (({row_stride}) <= 0 || ({column_stride}) <= 0 || !((({row_stride}) / ({column_stride})) >= {shape[1]} || (({column_stride}) / ({row_stride})) >= {shape[0]})) {{ return static_cast<int>({invalid_value}); }}"
            )

        lines.extend(
            (
                f"if ({name} == nullptr || reinterpret_cast<uintptr_t>({name}) % {element_size} != 0) {{ return static_cast<int>({invalid_value}); }}",
                f"uintptr_t {prefix}_bytes = {element_size};",
            )
        )

        for axis, (dimension, stride) in enumerate(zip(shape, strides)):
            term = f"{prefix}_term_{axis}"
            lines.extend(
                (
                    f"if (({stride}) < 0 || static_cast<uintptr_t>({stride}) > std::numeric_limits<uintptr_t>::max() / {element_size} / {max(1, dimension - 1)}) {{ return static_cast<int>({invalid_value}); }}",
                    f"uintptr_t {term} = static_cast<uintptr_t>({stride}) * {dimension - 1} * {element_size};",
                    f"if ({prefix}_bytes > std::numeric_limits<uintptr_t>::max() - {term}) {{ return static_cast<int>({invalid_value}); }}",
                    f"{prefix}_bytes += {term};",
                )
            )

        lines.extend(
            (
                f"uintptr_t {prefix}_begin = reinterpret_cast<uintptr_t>({name});",
                f"if ({prefix}_begin > std::numeric_limits<uintptr_t>::max() - {prefix}_bytes) {{ return static_cast<int>({invalid_value}); }}",
                f"uintptr_t {prefix}_end = {prefix}_begin + {prefix}_bytes;",
            )
        )

    for output in context.outputs:
        for operand in context.variables:
            if operand not in indices:
                continue

            left, right = indices[output], indices[operand]
            lines.append(
                f"if ({left}_begin < {right}_end && {right}_begin < {left}_end) {{ return static_cast<int>({invalid_value}); }}"
            )
    return common.indent_block("\n".join(lines), "    ") + "\n"


def _prove_pointwise_epilogue(loop, top, output):
    """Reject consumers that change the collective result's coordinates."""
    axes = tuple(map(str, loop.results[0].type.shape))
    members = {loop.results[0].name}
    stored = False

    for operation in top:
        if operation is loop or not members.intersection(operation.operands):
            continue

        if operation.opcode in {"shape.dim", "tensor.stride"}:
            continue

        if operation.opcode == "mem.store" and operation.operands[0] in members:
            if (
                operation.operands[1] != output
                or operation.attrs.get("source")
                or operation.attrs.get("indices")
            ):
                raise ValueError("CUDA CTA dot requires a pointwise output store.")

            stored = True
            continue

        if (
            not operation.regions
            and len(operation.results) == 1
            and tuple(map(str, operation.results[0].type.shape)) == axes
            and (
                operation.opcode.startswith(("arith.", "cmp.", "math."))
                or operation.opcode in {"tensor.cast", "select.where"}
            )
        ):
            members.add(operation.results[0].name)
            continue

        raise ValueError(
            "CUDA CTA dot requires a same-coordinate pointwise epilogue; "
            + operation.opcode
        )

    if not stored:
        raise ValueError("CUDA CTA dot result does not reach a pointwise output store.")


def _async_copy(plan, local, transfers, context, *, position, slot):
    """Emit one aligned cooperative packet from proved physical access maps."""
    left_base, right_base = (
        tuple(
            common.target_index_expr(context.target, value).replace("//", "/")
            for value in bases
        )
        for bases in (transfers.left_base, transfers.right_base)
    )
    # The proof expresses K as a source stage. The async path uses its flat K
    # coordinate, while retaining each operand's proved physical row/column.
    stage = f"{local}_i"
    left_base = tuple(value.replace(stage, "0") for value in left_base)
    right_base = tuple(value.replace(stage, "0") for value in right_base)
    slot_elements = (
        plan.rows * plan.left_pitch + plan.reduction_packet * plan.right_pitch
    )
    left, right = (
        context.target.tensor_ref(transfers.left),
        context.target.tensor_ref(transfers.right),
    )

    return [
        "{",
        "    const int nt_copy_lane = threadIdx.x < 64 ? threadIdx.x : threadIdx.x - 64;",
        "    const int nt_copy_row = threadIdx.x < 64 ? nt_copy_lane / 2 : nt_copy_lane / 4;",
        "    const int nt_copy_col = (threadIdx.x < 64 ? nt_copy_lane % 2 : nt_copy_lane % 4) * 8;",
        f"    const half* nt_copy_source = threadIdx.x < 64 ? {left} : {right};",
        f"    nt_copy_source += threadIdx.x < 64 ? ((({left_base[0]}) + nt_copy_row) * {transfers.left_width} + ({left_base[1]}) + ({position}) + nt_copy_col) : ((({right_base[0]}) + ({position}) + nt_copy_row) * {transfers.right_width} + ({right_base[1]}) + nt_copy_col);",
        f"    half* nt_copy_target = {local}_lhs + ({slot}) * {slot_elements} + (threadIdx.x < 64 ? nt_copy_row * {plan.left_pitch} + nt_copy_col : {plan.rows * plan.left_pitch} + nt_copy_row * {plan.right_pitch} + nt_copy_col);",
        f"    const unsigned int nt_copy_shared = static_cast<unsigned int>({context.target.global_helper('__cvta_generic_to_shared')}(nt_copy_target));",
        '    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(nt_copy_shared), "l"(nt_copy_source) : "memory");',
        "}",
    ]


def _async_tiles(plan, local, transfers, context, *, upper, ordered, carry_first):
    slot_elements = (
        plan.rows * plan.left_pitch + plan.reduction_packet * plan.right_pitch
    )
    k = f"{local}_flat_k"
    acc = f"{local}_partial" if ordered else f"{local}_acc"
    lines = _async_copy(plan, local, transfers, context, position="0", slot="0")
    lines.append('asm volatile("cp.async.commit_group;" ::: "memory");')
    lines.append(f"for (int64_t {k} = 0; {k} < ({upper}) * 32; {k} += 16) {{")
    lines.append(f"    if ({k} + 16 < ({upper}) * 32) {{")
    lines.extend(
        "        " + value
        for value in _async_copy(
            plan,
            local,
            transfers,
            context,
            position=f"{k} + 16",
            slot=f"(({k} / 16 + 1) % 2)",
        )
    )
    lines.extend(
        (
            '        asm volatile("cp.async.commit_group;" ::: "memory");',
            '        asm volatile("cp.async.wait_group 1;" ::: "memory");',
            "    } else {",
            '        asm volatile("cp.async.wait_group 0;" ::: "memory");',
            "    }",
            "    __syncthreads();",
        )
    )

    if ordered:
        lines.append(
            f"    if ({k} % 32 == 0) {{ wmma::fill_fragment({local}_partial, 0.0f); }}"
        )

    lines.extend(
        (
            f"    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> {local}_a;",
            f"    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> {local}_b;",
            f"    wmma::load_matrix_sync({local}_a, {local}_lhs + ({k} / 16 % 2) * {slot_elements} + ({local}_warp / 2) * 16 * {plan.left_pitch}, {plan.left_pitch});",
            f"    wmma::load_matrix_sync({local}_b, {local}_rhs + ({k} / 16 % 2) * {slot_elements} + ({local}_warp % 2) * 16, {plan.right_pitch});",
            f"    wmma::mma_sync({acc}, {local}_a, {local}_b, {acc});",
            "    __syncthreads();",
        )
    )

    if ordered:
        carry, partial = (
            f"{local}_acc.x[nt_wmma_component]",
            f"{local}_partial.x[nt_wmma_component]",
        )
        left, right = (carry, partial) if carry_first else (partial, carry)
        lines.extend(
            (
                f"    if ({k} % 32 == 16) {{",
                f"        for (int nt_wmma_component = 0; nt_wmma_component < {local}_acc.num_elements; ++nt_wmma_component) {{",
                f"            {carry} = {context.target.global_helper('__fadd_rn')}({left}, {right});",
                "        }",
                "    }",
            )
        )

    lines.append("}")

    return lines


def emit_cta_dot(local, loop, dot, context, *, bounds, ordered):
    """Hoist only collective work; scalar effects remain in the SSA walker."""
    plan = CTA_DOT_PLAN
    program = context.program
    top = program.blocks[0].operations
    reserved_prefixes = (
        "nt_copy_",
        "nt_cta_",
        "nt_wmma_",
        "nt_matrix_",
        "nt_tile_",
        "nt_outer_index",
        "nt_i0",
        "nt_i1",
        "nt_pred_",
    )
    value_symbols = {
        context.target.symbol(name)
        for name in context.value_types
        if name.startswith("%")
    }
    conflicts = sorted(
        name
        for name in context.parameter_names
        if name.startswith(reserved_prefixes)
        or name in value_symbols
        or name == local
        or name.startswith(local + "_")
    )

    if conflicts:
        raise ValueError(
            "CUDA CTA dot private symbols conflict with ABI parameters: "
            + ", ".join(conflicts)
        )

    if (
        len(program.blocks) != 1
        or sum(operation.opcode == "scf.for" for operation in top) != 1
        or sum(operation.opcode == "mem.store" for operation in top) != 1
        or any(
            operation.regions
            and operation is not loop
            or operation.opcode.startswith("mem.")
            and operation.opcode not in {"mem.load", "mem.store"}
            for operation in top
        )
    ):
        raise ValueError(
            "CUDA CTA dot requires one verified loop and one output store."
        )

    _prove_pointwise_epilogue(loop, top, context.output)

    stage = f"{local}_i"
    transfers = prove_dot_transfers(
        loop, dot, context, stage=stage, outer=context.outer_index_expr
    )

    if transfers is None:
        raise ValueError("CUDA CTA dot requires complete original 32x32 access maps.")

    prefix = context.collective_lines
    k = f"{local}_k"
    copy_index = f"{local}_copy"
    a_count = plan.rows * plan.left_pitch
    prefix.extend(
        (
            f"__shared__ __align__(32) unsigned char {local}_arena[{plan.arena_bytes}];",
            f"half* {local}_lhs = reinterpret_cast<half*>({local}_arena);",
            f"half* {local}_rhs = {local}_lhs + {a_count};",
            f"float* {local}_out = reinterpret_cast<float*>({local}_arena);",
            f"const int {local}_warp = threadIdx.x / 32;",
            f"const bool {local}_dense_wmma = {transfers.guard};",
            f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> {local}_acc;",
            f"wmma::fill_fragment({local}_acc, 0.0f);",
        )
    )

    if ordered:
        prefix.extend(
            (
                f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> {local}_partial;",
                f"wmma::fill_fragment({local}_partial, 0.0f);",
            )
        )
    # The transfer proof established lower=0, step=1, and an exact static
    # number of 32-wide source groups. Use that witness here: a scalar SSA
    # spelling of shape.dim may otherwise be declared only in the epilogue.

    del bounds
    lower, upper, step = "0", str(transfers.left_width // 32), "1"
    aligned = " && ".join(
        f"reinterpret_cast<uintptr_t>({context.target.tensor_ref(name)}) % 32 == 0"
        for name in (transfers.left, transfers.right)
    )
    prefix.extend(
        ("#if __CUDA_ARCH__ >= 800", f"if ({local}_dense_wmma && {aligned}) {{")
    )
    carry_first = loop.regions[0].operations[3].operands[0] == str(
        loop.attrs["iter_args"][0]["block_arg"]
    )
    prefix.extend(
        "    " + value
        for value in _async_tiles(
            plan,
            local,
            transfers,
            context,
            upper=upper,
            ordered=ordered,
            carry_first=carry_first,
        )
    )
    prefix.extend(("} else", "#endif", "{"))
    prefix.append(context.target.loop_header(stage, lower, upper, step))

    if ordered:
        prefix.append(f"    wmma::fill_fragment({local}_partial, 0.0f);")

    prefix.append(f"    {context.target.loop_header(k, '0', '32', '16')}")
    bindings = {**(context.bindings or {}), str(loop.attrs["induction"]): stage}

    for operand, name, coordinates, pitch, bases, width in (
        (
            dot.operands[0],
            "lhs",
            (f"{copy_index} / 16", f"{k} + {copy_index} % 16"),
            plan.left_pitch,
            transfers.left_base,
            transfers.left_width,
        ),
        (
            dot.operands[1],
            "rhs",
            (f"{k} + {copy_index} / 32", f"{copy_index} % 32"),
            plan.right_pitch,
            transfers.right_base,
            transfers.right_width,
        ),
    ):
        packet_width = 16 if name == "lhs" else 32
        index = (
            f"({copy_index} / {packet_width}) * {pitch} + {copy_index} % {packet_width}"
        )
        source = transfers.left if name == "lhs" else transfers.right
        row, column = (
            common.target_index_expr(context.target, base).replace("//", "/")
            for base in bases
        )
        prefix.extend(
            (
                f"        for (int {copy_index} = threadIdx.x; {copy_index} < 512; {copy_index} += 128) {{",
                f"            if ({local}_dense_wmma) {{",
                f"                {local}_{name}[{index}] = {context.target.tensor_ref(source)}[((({row}) + ({coordinates[0]})) * {width}) + ({column}) + ({coordinates[1]})];",
                "            } else {",
            )
        )
        lines = []
        child = context.child(lines=lines, memo={}, bindings=bindings, mask_expr=None)
        value = common.emit_element(operand, coordinates, child)
        lines.append(
            f"{local}_{name}[{index}] = {context.target.global_helper('__float2half_rn')}(static_cast<float>({value}));"
        )
        prefix.extend("                " + line for line in lines)
        prefix.extend(("            }", "        }"))

    accumulator = f"{local}_partial" if ordered else f"{local}_acc"
    prefix.extend(
        (
            "        __syncthreads();",
            f"        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> {local}_a;",
            f"        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> {local}_b;",
            f"        wmma::load_matrix_sync({local}_a, {local}_lhs + ({local}_warp / 2) * 16 * {plan.left_pitch}, {plan.left_pitch});",
            f"        wmma::load_matrix_sync({local}_b, {local}_rhs + ({local}_warp % 2) * 16, {plan.right_pitch});",
            f"        wmma::mma_sync({accumulator}, {local}_a, {local}_b, {accumulator});",
            "        __syncthreads();",
            "    }",
        )
    )

    if ordered:
        add = loop.regions[0].operations[3]
        carry = f"{local}_acc.x[nt_wmma_component]"
        partial = f"{local}_partial.x[nt_wmma_component]"
        left, right = (
            (carry, partial)
            if add.operands[0] == str(loop.attrs["iter_args"][0]["block_arg"])
            else (partial, carry)
        )
        prefix.extend(
            (
                f"    for (int nt_wmma_component = 0; nt_wmma_component < {local}_acc.num_elements; ++nt_wmma_component) {{",
                f"        {carry} = {context.target.global_helper('__fadd_rn')}({left}, {right});",
                "    }",
            )
        )

    prefix.extend(
        (
            "}",
            "}",
            "__syncthreads();",
            f"wmma::store_matrix_sync({local}_out + {local}_warp * 256, {local}_acc, 16, wmma::mem_row_major);",
            "__syncthreads();",
        )
    )
    row, column = context.coordinate_exprs
    value = f"{local}_out[{local}_warp * 256 + (({row}) % 16) * 16 + (({column}) % 16)]"
    context.memo[loop.results[0].name] = value

    return value
