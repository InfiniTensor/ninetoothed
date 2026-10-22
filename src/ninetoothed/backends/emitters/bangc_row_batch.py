"""Channel-packed NRAM scheduling for a proved typed row graph."""

from dataclasses import dataclass, replace

from ninetoothed.backends.emitters import bangc_vector
from ninetoothed.backends.emitters.row_batch import plan_row_batch
from ninetoothed.backends.emitters.row_input_layout import prove_row_input
from ninetoothed.backends.toolchain import resolve_bangc_arch
from ninetoothed.compiler.reduction_domains import native_sum_domain


@dataclass(frozen=True)
class RowBatchBody:
    body: str
    guard: str
    tasks: int
    nram_bytes: int
    batch_rows: int = 8
    target_arch: str | None = None
    resource_fallback: bool = False


def _stat_name(value, target):
    return "nt_batch_stat_" + target.symbol(value)


def _ordered_packed_sum(source, dest, width, channels):
    """Pack independent rows without changing each row's halving tree."""
    return [
        "    {",
        f"        __memcpy(nt_batch_tmp, {source}, {width * channels} * sizeof(float), NRAM2NRAM);",
        f"        int64_t nt_len = {width};",
        "        while (nt_len >= 64) {",
        "            int64_t nt_h = (nt_len / 2) & ~7LL;",
        "            if (nt_h < 32) { break; }",
        f"            __bang_add(nt_batch_tmp, nt_batch_tmp, nt_batch_tmp + nt_h * {channels}, (uint32_t)(nt_h * {channels}));",
        "            for (int64_t nt_j = 2 * nt_h; nt_j < nt_len; ++nt_j) {",
        f"                for (int64_t nt_r = 0; nt_r < {channels}; ++nt_r) {{",
        f"                    nt_batch_tmp[(nt_j - 2 * nt_h) * {channels} + nt_r] = nt_batch_tmp[(nt_j - 2 * nt_h) * {channels} + nt_r] + nt_batch_tmp[nt_j * {channels} + nt_r];",
        "                }",
        "            }",
        "            nt_len = nt_h;",
        "        }",
        f"        for (int64_t nt_r = 0; nt_r < {channels}; ++nt_r) {{",
        "            float nt_result = 0.0f;",
        f"            for (int64_t nt_j = 0; nt_j < nt_len; ++nt_j) {{ nt_result = nt_result + nt_batch_tmp[nt_j * {channels} + nt_r]; }}",
        f"            {dest}[nt_r] = nt_result;",
        "        }",
        "    }",
    ]


def _pool_reduction(step, batch, target, halving_reduce, *, ordered_sum=False):
    node = step.node
    operator = node.operation.opcode.split(".")[1]
    source = f"nt_dag_buf_{step.operand_buffers[0]}"
    dest = _stat_name(node.result.name, target)
    width = batch.columns
    channels = batch.batch_rows
    scratch = "nt_dag_scratch_float32_0"
    lines = ["    {"]

    if operator == "sum" and ordered_sum:
        return _ordered_packed_sum(source, dest, width, channels)

    if operator == "sum":
        domain = native_sum_domain("reduce.sum", dtype="float32")
        bound = domain.maximum_absolute_input(width)
        lines.extend(
            (
                f"        __bang_abs({scratch}, {source}, nt_aligned);",
                f"        __bang_maxpool(nt_batch_domains, {scratch}, {channels}, 1, {width}, 1, {width}, {width}, 1);",
                f"        __bang_eq({scratch}, {source}, {source}, nt_aligned);",
                f"        __bang_sumpool(nt_batch_flags, {scratch}, {channels}, 1, {width}, 1, {width}, {width}, 1);",
                f"        __bang_sumpool({dest}, {source}, {channels}, 1, {width}, 1, {width}, {width}, 1);",
                f"        for (int64_t nt_r = 0; nt_r < {channels}; ++nt_r) {{",
                f"            if (!(nt_batch_domains[nt_r] <= {bound:.17g}f) || nt_batch_flags[nt_r] != {width}.0f) {{",
                f"                for (int64_t nt_j = 0; nt_j < {width}; ++nt_j) {{ nt_batch_tmp[nt_j] = {source}[nt_j * {channels} + nt_r]; }}",
                "                float* nt_wrk = nt_batch_tmp;",
                f"                int64_t nt_len = {width};",
            )
        )
        lines.extend("            " + line for line in halving_reduce("sum"))
        lines.extend(
            (
                "                float nt_result = 0.0f;",
                "                for (int64_t nt_j = 0; nt_j < nt_len; ++nt_j) { nt_result = nt_result + nt_wrk[nt_j]; }",
                f"                {dest}[nt_r] = nt_result;",
                "            }",
                f"            if ({dest}[nt_r] == 0.0f) {{ {dest}[nt_r] = 0.0f; }}",
                "        }",
            )
        )
    else:
        if operator == "min":
            lines.append(f"        __bang_neg({scratch}, {source}, nt_aligned);")
            pool_source = scratch
        else:
            pool_source = source

        lines.append(
            f"        __bang_maxpool({dest}, {pool_source}, {channels}, 1, {width}, 1, {width}, {width}, 1);"
        )

        if operator == "min":
            lines.append(
                f"        for (int64_t nt_r = 0; nt_r < {channels}; ++nt_r) {{ {dest}[nt_r] = -{dest}[nt_r]; }}"
            )

        lines.extend(
            (
                f"        __bang_eq({scratch}, {source}, {source}, nt_aligned);",
                f"        __bang_sumpool(nt_batch_flags, {scratch}, {channels}, 1, {width}, 1, {width}, {width}, 1);",
                f"        for (int64_t nt_r = 0; nt_r < {channels}; ++nt_r) {{",
                f"            if (nt_batch_flags[nt_r] != {width}.0f) {{ {dest}[nt_r] = NAN; }}",
                f"            else if ({dest}[nt_r] == 0.0f) {{",
                f"                for (int64_t nt_j = 0; nt_j < {width}; ++nt_j) {{ nt_batch_tmp[nt_j] = {source}[nt_j * {channels} + nt_r]; }}",
                "                float* nt_wrk = nt_batch_tmp;",
                f"                int64_t nt_len = {width};",
            )
        )
        lines.extend("            " + line for line in halving_reduce(operator))
        neutral = "-INFINITY" if operator == "max" else "INFINITY"
        lines.extend(
            (
                f"                float nt_result = {neutral};",
                f"                for (int64_t nt_j = 0; nt_j < nt_len; ++nt_j) {{ nt_result = nt_{operator}(nt_result, nt_wrk[nt_j]); }}",
                f"                {dest}[nt_r] = nt_result;",
                "            }",
                "        }",
            )
        )

    lines.append("    }")

    return lines


def render_row_batch(
    context,
    target,
    literal,
    halving_reduce,
    *,
    batch_rows=8,
    budget=192 * 1024,
    auto_batch=False,
):
    options = context.kernel.compiler_options.get("backend_options", {})
    requested_rows = options.get("row_batch_rows", "auto") if auto_batch else "auto"
    explicit_rows = requested_rows != "auto"
    target_arch = None

    if explicit_rows:
        target_arch = resolve_bangc_arch(options.get("arch", "native"))

        if (
            not isinstance(requested_rows, int)
            or isinstance(requested_rows, bool)
            or requested_rows not in {8, 16}
            or target_arch != "compute_50"
            or not context.cooperative_reduction_program
        ):
            raise ValueError(
                "Explicit BangC row packing requires 8 or 16 rows, compute_50 "
                "and a proved cooperative reduction."
            )

        batch_rows = requested_rows
        # Existing cooperative fallback is bounded by 96 KiB. This explicit
        # candidate keeps combined declarations within compute_50's 512 KiB;
        # existing legal default row plans and the global vector budget stay
        # unchanged.
        budget = 512 * 1024 - 96 * 1024

    row_plan = bangc_vector.plan_rows(context, budget=budget)

    if row_plan is None:
        row_plan = bangc_vector.plan_row_vectors(
            context, budget=budget, allow_reductions=True
        )

    if (
        row_plan is None
        or not str(row_plan.rows).isdecimal()
        or not str(row_plan.cols).isdecimal()
    ):
        if explicit_rows:
            raise ValueError("The explicit BangC row packing request was not consumed.")
        return None

    if explicit_rows and not any(
        step.node.lowering.mode == "reduce" for step in row_plan.dag.steps
    ):
        raise ValueError("Explicit BangC row packing requires a reduction DAG.")

    rows, cols = int(row_plan.rows), int(row_plan.cols)
    layouts = {}
    varying_inputs = []

    for item in row_plan.dag.inputs:
        name = item.value.name
        info = context.tensors.get(name)

        if info is None or info.ndim == 0:
            continue

        layout = prove_row_input(context, name, rows, cols)

        if layout is None:
            if explicit_rows:
                raise ValueError(
                    "The explicit BangC row packing input layout is illegal."
                )
            return None

        layouts[name] = layout

        if item.value.scalar and layout.row_index == "row":
            varying_inputs.append(name)

    batch = plan_row_batch(
        row_plan, varying_inputs, batch_rows=batch_rows, budget=budget
    )
    resource_fallback = False

    if (
        batch is None
        and auto_batch
        and not explicit_rows
        and batch_rows == 8
        and budget == 192 * 1024
        and context.cooperative_reduction_program
        and rows >= 512
        and cols == 1024
        and any(step.node.lowering.mode == "reduce" for step in row_plan.dag.steps)
    ):
        target_arch = resolve_bangc_arch(options.get("arch", "native"))

        if target_arch == "compute_50":
            # Probe the same eight-row plan with only its budget changed. A
            # valid, larger allocation proves the original failure was solely
            # resource capacity; masks, dtype and row provenance stay intact.
            capacity_probe = plan_row_batch(
                row_plan, varying_inputs, batch_rows=8, budget=416 * 1024
            )

            if capacity_probe is not None and capacity_probe.nram_bytes > budget:
                larger = plan_row_batch(
                    row_plan, varying_inputs, batch_rows=16, budget=416 * 1024
                )

                if larger is not None:
                    batch = larger
                    resource_fallback = True

    if batch is None:
        if explicit_rows:
            raise ValueError(
                "The explicit BangC row packing resource or dtype is illegal."
            )
        return None

    if auto_batch and not explicit_rows and not resource_fallback:
        target_arch = resolve_bangc_arch(options.get("arch", "native"))

        if (
            target_arch == "compute_50"
            and context.cooperative_reduction_program
            and any(step.node.lowering.mode == "reduce" for step in row_plan.dag.steps)
            and batch_rows == 8
            and budget == 192 * 1024
            and rows >= 512
            and cols == 1024
        ):
            # Cooperative fallback uses at most two 8192-float buffers,
            # or the same DAG with a single row. Reserve 96 KiB for it;
            # the combined declaration bound stays below 512 KiB.
            larger = plan_row_batch(
                row_plan, varying_inputs, batch_rows=16, budget=384 * 1024
            )

            if larger is not None:
                batch = larger

    dag = row_plan.dag
    capacity = batch.batch_rows * batch.columns
    lines = [
        f"    const int64_t nt_batch_base = (int64_t)(taskIdX) * {batch.batch_rows};",
        f"    const int64_t nt_batch_valid = ({rows} - nt_batch_base < {batch.batch_rows}) ? ({rows} - nt_batch_base) : {batch.batch_rows};",
        f"    const int64_t nt_cnt = {capacity};",
        f"    const uint32_t nt_aligned = {capacity};",
        *bangc_vector._buffers(dag, capacity),
        f"    __nram__ float nt_batch_tmp[{capacity}];",
        "    __nram__ float nt_batch_flags[128];",
        "    __nram__ float nt_batch_domains[128];",
    ]

    if batch.extra_scratch:
        lines.append(f"    __nram__ float nt_dag_scratch_float32_0[{capacity}];")

    for name in batch.varying_scalars:
        lines.append(f"    __nram__ float {_stat_name(name, target)}[128];")

    varying = set(batch.varying_scalars)

    for item in dag.inputs:
        value = item.value
        layout = layouts.get(value.name)

        if value.name in varying:
            stat = _stat_name(value.name, target)
            lines.extend(
                (
                    f"    __bang_write_zero({stat}, 128);",
                    f"    __memcpy({stat}, {value.name} + nt_batch_base, nt_batch_valid * sizeof(float), GDRAM2NRAM);",
                )
            )
        elif value.scalar:
            type_ = "int64_t" if value.dtype == "index" else "float"
            lines.append(
                f"    {type_} {bangc_vector._scalar_name(value, target)} = {row_plan.scalar_expressions[value.name]};"
            )
        else:
            offset = layout.base_offset("nt_batch_base")
            lines.append(f"    __bang_write_zero(nt_batch_tmp, {capacity});")

            if layout.row_index == "row":
                lines.append(
                    f"    __memcpy(nt_batch_tmp, {value.name} + ({offset}), nt_batch_valid * {cols} * sizeof(float), GDRAM2NRAM);"
                )
            else:
                lines.extend(
                    (
                        f"    __memcpy(nt_batch_tmp, {value.name} + ({offset}), {cols} * sizeof(float), GDRAM2NRAM);",
                        f"    for (int64_t nt_r = 1; nt_r < {batch.batch_rows}; ++nt_r) {{ __memcpy(nt_batch_tmp + nt_r * {cols}, nt_batch_tmp, {cols} * sizeof(float), NRAM2NRAM); }}",
                    )
                )

            lines.append(
                f"    __bang_transpose(nt_dag_buf_{item.buffer}, nt_batch_tmp, {batch.batch_rows}, {cols});"
            )

    for step in dag.steps:
        node = step.node

        if node.lowering.mode == "reduce":
            lines.extend(
                _pool_reduction(
                    step,
                    batch,
                    target,
                    halving_reduce,
                    ordered_sum=explicit_rows or resource_fallback,
                )
            )
            continue

        if node.result.scalar and node.result.name in varying:
            args = [
                f"{_stat_name(value.name, target)}[nt_r]"
                if value.name in varying
                else bangc_vector._operand(value, buffer, target, literal)
                for value, buffer in zip(node.operands, step.operand_buffers)
            ]
            expression = bangc_vector._expression(node, args, target)
            lines.append(
                f"    for (int64_t nt_r = 0; nt_r < {batch.batch_rows}; ++nt_r) {{ {_stat_name(node.result.name, target)}[nt_r] = {expression}; }}"
            )
            continue

        operands = []
        buffers = []
        broadcast = False

        for value, buffer in zip(node.operands, step.operand_buffers):
            if value.name in varying:
                broadcast = True
                lines.extend(
                    (
                        f"    __memcpy(nt_dag_scratch_float32_0, {_stat_name(value.name, target)}, {batch.batch_rows} * sizeof(float), NRAM2NRAM, {batch.batch_rows} * sizeof(float), 0, {cols - 1});",
                    )
                )
                operands.append(replace(value, scalar=False))
                buffers.append(len(dag.buffer_dtypes))
            else:
                operands.append(value)
                buffers.append(buffer)

        transformed = replace(
            step,
            node=replace(
                node,
                operands=tuple(operands),
                lowering=bangc_vector.classify(
                    node.operation, tuple(operands), node.result.dtype
                ),
            ),
            operand_buffers=tuple(buffers),
        )

        if transformed.node.lowering is None:
            return None

        single = replace(dag, steps=(transformed,))
        code = "\n".join(
            bangc_vector.render_steps(single, target, literal, halving_reduce)
        )

        if broadcast:
            code = code.replace(
                f"nt_dag_buf_{len(dag.buffer_dtypes)}", "nt_dag_scratch_float32_0"
            )

        lines.append(code)

    for store in dag.stores:
        if store.value.scalar:
            if store.value.name not in varying:
                return None

            if row_plan.output_widths[store.output] == "1":
                lines.append(
                    f"    __memcpy({store.output} + nt_batch_base, {_stat_name(store.value.name, target)}, nt_batch_valid * sizeof(float), NRAM2GDRAM);"
                )
            else:
                lines.extend(
                    (
                        f"    __memcpy(nt_dag_scratch_float32_0, {_stat_name(store.value.name, target)}, {batch.batch_rows} * sizeof(float), NRAM2NRAM, {batch.batch_rows} * sizeof(float), 0, {cols - 1});",
                        f"    __bang_transpose(nt_batch_tmp, nt_dag_scratch_float32_0, {cols}, {batch.batch_rows});",
                        f"    __memcpy({store.output} + nt_batch_base * {cols}, nt_batch_tmp, nt_batch_valid * {cols} * sizeof(float), NRAM2GDRAM);",
                    )
                )
        else:
            lines.extend(
                (
                    f"    __bang_transpose(nt_batch_tmp, nt_dag_buf_{store.buffer}, {cols}, {batch.batch_rows});",
                    f"    __memcpy({store.output} + nt_batch_base * {cols}, nt_batch_tmp, nt_batch_valid * {cols} * sizeof(float), NRAM2GDRAM);",
                )
            )

    guards = [row_plan.guard, *(layout.guard for layout in layouts.values())]

    for store in dag.stores:
        guards.append(f"(((uintptr_t){store.output} & 31ULL) == 0)")

        output_bytes = rows * int(row_plan.output_widths[store.output]) * 4

        for name, layout in layouts.items():
            source_bytes = layout.source_rows * layout.source_columns * 4
            src, dst = f"((uintptr_t){name})", f"((uintptr_t){store.output})"
            guards.append(
                f"(({src} <= {dst} && {dst} - {src} >= {source_bytes}ULL) || "
                f"({dst} <= {src} && {src} - {dst} >= {output_bytes}ULL))"
            )

    for position, store in enumerate(dag.stores):
        for other in dag.stores[position + 1 :]:
            first, second = (
                f"((uintptr_t){store.output})",
                f"((uintptr_t){other.output})",
            )
            first_bytes = rows * int(row_plan.output_widths[store.output]) * 4
            second_bytes = rows * int(row_plan.output_widths[other.output]) * 4
            guards.append(
                f"(({first} <= {second} && {second} - {first} >= {first_bytes}ULL) || "
                f"({second} <= {first} && {first} - {second} >= {second_bytes}ULL))"
            )

    return RowBatchBody(
        "\n".join(lines),
        " && ".join(guards),
        batch.tasks,
        batch.nram_bytes,
        batch.batch_rows,
        target_arch,
        resource_fallback,
    )
