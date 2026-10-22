"""Vector add of independently materialized original FP32 dot groups."""

import math

from ninetoothed.backends.emitters.ascendc_dot import NativeDotBody
from ninetoothed.backends.emitters.dot_layout import prove_dense_dot_access
from ninetoothed.dtype import normalize_dtype


def render_ordered_dot_add(context, target):
    """Keep the original initializer, operand order and serial K32 carry adds.

    A task owns one complete 32-by-32 output tile. Two small UB buffers retain
    the carry and one original partial. Only a terminal FP32/FP16 cast and one
    FP32 literal add are implemented; other epilogues remain with the fallback.
    """
    program = context.kernel.ssa

    if (
        len(program.blocks) != 1
        or len(program.inputs) != 2
        or len(program.outputs) != 1
    ):
        return None

    top = program.blocks[0].operations
    loops = [operation for operation in top if operation.opcode == "scf.for"]
    stores = [operation for operation in top if operation.opcode == "mem.store"]

    if len(loops) != 1 or len(stores) != 1 or top[-1] is not stores[0]:
        return None

    loop, store = loops[0], stores[0]
    definitions = {value.name: op for op in top for value in op.results}

    if (
        len(loop.operands) != 4
        or len(loop.results) != 1
        or len(loop.regions) != 1
        or len(loop.attrs.get("iter_args", ())) != 1
        or len(store.operands) != 2
        or store.results
        or store.operands[1] != program.outputs[0].name
        or store.attrs != {"target": store.operands[1], "target_shape": ("32", "32")}
    ):
        return None

    bounds = [definitions.get(name) for name in loop.operands[:3]]

    if any(
        operation is None or operation.opcode != "arith.constant"
        for operation in bounds
    ):
        return None

    lower, groups, step = (operation.attrs.get("value") for operation in bounds)

    if (
        lower != 0
        or step != 1
        or not isinstance(groups, int)
        or isinstance(groups, bool)
        or not 0 < groups <= 2047
    ):
        return None

    region = loop.regions[0]

    if (
        len(region.args) != 2
        or len(region.operations) != 3
        or tuple(operation.opcode for operation in region.operations)
        != ("tensor.extract", "arith.add", "scf.yield")
        or any(operation.regions for operation in region.operations)
    ):
        return None

    partial, addition, yielded = region.operations
    carry = loop.attrs["iter_args"][0]
    initial = definitions.get(carry["initial"])

    if (
        len(partial.results) != 1
        or len(addition.results) != 1
        or partial.operands[1:] != (region.args[0].name,)
        or region.args[0].name != loop.attrs.get("induction")
        or region.args[1].name != carry["block_arg"]
        or addition.operands
        not in {
            (carry["block_arg"], partial.results[0].name),
            (partial.results[0].name, carry["block_arg"]),
        }
        or yielded.operands != (addition.results[0].name,)
        or yielded.results
        or loop.operands[-1] != carry["initial"]
        or initial is None
        or initial.opcode not in {"tensor.full", "tensor.zeros"}
        or initial.operands
        or len(initial.results) != 1
        or initial.attrs.get("value", 0) != 0
    ):
        return None

    for value in (
        initial.results[0],
        region.args[1],
        partial.results[0],
        addition.results[0],
        loop.results[0],
    ):
        if (
            value.type.kind != "tensor"
            or normalize_dtype(value.type.dtype) != "float32"
            or tuple(map(str, value.type.shape)) != ("32", "32")
        ):
            return None

    scratch, output = partial.operands[0], store.operands[1]

    if scratch == output or {value.name for value in program.inputs} != {
        scratch,
        output,
    }:
        return None

    try:
        scratch_info, output_info = context.tensors[scratch], context.tensors[output]
        m, n = map(int, output_info.source_shape)
        scratch_shape = tuple(map(int, scratch_info.source_shape))
    except (KeyError, TypeError, ValueError):
        return None

    output_dtype = normalize_dtype(output_info.dtype)

    if (
        m <= 0
        or n <= 0
        or m % 32
        or n % 32
        or n > 65535
        or scratch_shape != (groups * m, n)
        or groups * m * n * 4 > 32 * 1024 * 1024
        or normalize_dtype(scratch_info.dtype) != "float32"
        or output_dtype not in {"float16", "float32"}
    ):
        return None

    rows = f"outer_index // {n // 32} * 32 + value_0"
    columns = f"outer_index % {n // 32} * 32 + value_1"
    accesses = []

    for attributes, expected, physical, info in (
        (
            partial.results[0].type.attrs,
            (f"extract_0_0 * {m} + {rows}", columns),
            scratch_shape,
            scratch_info,
        ),
        (output_info.attrs, (rows, columns), (m, n), output_info),
    ):
        access = prove_dense_dot_access(
            attributes,
            shape=(32, 32),
            tasks=(m // 32) * (n // 32),
            stages=groups,
            expected_offsets=expected,
        )

        if (
            access is None
            or access.source_shape != physical
            or access.strides != tuple(map(str, info.attrs.get("source_strides", ())))
        ):
            return None

        accesses.append(access)

    stored = store.operands[0]
    cast = definitions.get(stored)
    allowed = {id(operation) for operation in (*bounds, initial, loop, store)}

    if cast is not None and cast.opcode == "tensor.cast":
        if (
            len(cast.operands) != 1
            or len(cast.results) != 1
            or normalize_dtype(cast.attrs.get("dtype") or cast.results[0].type.dtype)
            != output_dtype
            or cast.attrs.get("dtype_ref") not in {None, output}
            or tuple(map(str, cast.results[0].type.shape)) != ("32", "32")
        ):
            return None

        allowed.add(id(cast))
        stored = cast.operands[0]
    elif output_dtype != "float32":
        return None

    bias = None
    bias_first = False

    if stored != loop.results[0].name:
        epilogue = definitions.get(stored)

        if (
            epilogue is None
            or epilogue.opcode != "arith.add"
            or epilogue.regions
            or len(epilogue.operands) != 2
            or len(epilogue.results) != 1
            or normalize_dtype(epilogue.results[0].type.dtype) != "float32"
            or tuple(map(str, epilogue.results[0].type.shape)) != ("32", "32")
            or epilogue.operands.count(loop.results[0].name) != 1
        ):
            return None

        bias_first = epilogue.operands[1] == loop.results[0].name
        constant = definitions.get(epilogue.operands[0 if bias_first else 1])

        if (
            constant is None
            or constant.opcode != "arith.constant"
            or constant.operands
            or constant.regions
        ):
            return None

        bias = constant.attrs.get("value")

        if (
            isinstance(bias, bool)
            or not isinstance(bias, (int, float))
            or not math.isfinite(bias)
        ):
            return None

        allowed.update((id(epilogue), id(constant)))

    if any(
        id(operation) not in allowed or (operation is not loop and operation.regions)
        for operation in top
    ):
        return None

    carry_first = addition.operands[0] == carry["block_arg"]
    add_args = (
        "nt_ordered_acc, nt_ordered_partial"
        if carry_first
        else "nt_ordered_partial, nt_ordered_acc"
    )
    bias_body = ""

    if bias is not None:
        value = target.cast("float32", target.literal(bias))
        bias_args = (
            "nt_ordered_partial, nt_ordered_acc"
            if bias_first
            else "nt_ordered_acc, nt_ordered_partial"
        )
        bias_body = f"""    Duplicate(nt_ordered_partial, {value}, 1024);
    PipeBarrier<PIPE_ALL>();
    Add(nt_ordered_acc, {bias_args}, 1024);
    PipeBarrier<PIPE_ALL>();"""

    out_type = "half" if output_dtype == "float16" else "float"
    extra = ""
    result = "nt_ordered_acc"

    if out_type == "half":
        extra = """    TBuf<TPosition::VECCALC> nt_ordered_half_buffer;
    nt_ordered_pipe.InitBuffer(nt_ordered_half_buffer, 2048);
    auto nt_ordered_half = nt_ordered_half_buffer.Get<half>();"""
        result = "nt_ordered_half"

    cast_body = (
        ""
        if out_type == "float"
        else """    Cast(nt_ordered_half, nt_ordered_acc, RoundMode::CAST_RINT, 1024);
    PipeBarrier<PIPE_ALL>();"""
    )
    block_len = 2 if out_type == "half" else 4
    divisor = 16 if out_type == "half" else 8
    initial_value = target.cast(
        "float32", target.literal(initial.attrs.get("value", 0))
    )
    body = f"""    const int64_t nt_ordered_task = (int64_t)GetBlockIdx();
    if (nt_ordered_task >= {(m // 32) * (n // 32)}) {{ return; }}
    const int64_t nt_ordered_base = (nt_ordered_task / {n // 32}) * 32 * {n} + (nt_ordered_task % {n // 32}) * 32;
    TPipe nt_ordered_pipe;
    TBuf<TPosition::VECCALC> nt_ordered_acc_buffer, nt_ordered_partial_buffer;
    nt_ordered_pipe.InitBuffer(nt_ordered_acc_buffer, 4096);
    nt_ordered_pipe.InitBuffer(nt_ordered_partial_buffer, 4096);
    auto nt_ordered_acc = nt_ordered_acc_buffer.Get<float>();
    auto nt_ordered_partial = nt_ordered_partial_buffer.Get<float>();
{extra}
    GlobalTensor<float> nt_ordered_scratch_gm;
    GlobalTensor<{out_type}> nt_ordered_output_gm;
    nt_ordered_scratch_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>({scratch}), {groups * m * n});
    nt_ordered_output_gm.SetGlobalBuffer(reinterpret_cast<__gm__ {out_type}*>({output}), {m * n});
    DataCopyParams nt_ordered_copy;
    nt_ordered_copy.blockCount = 32; nt_ordered_copy.blockLen = 4;
    nt_ordered_copy.srcStride = {n // 8 - 4}; nt_ordered_copy.dstStride = 0;
    Duplicate(nt_ordered_acc, {initial_value}, 1024);
    PipeBarrier<PIPE_ALL>();
    for (int nt_ordered_group = 0; nt_ordered_group < {groups}; ++nt_ordered_group) {{
        DataCopy(nt_ordered_partial, nt_ordered_scratch_gm[nt_ordered_group * {m * n} + nt_ordered_base], nt_ordered_copy);
        PipeBarrier<PIPE_ALL>();
        Add(nt_ordered_acc, {add_args}, 1024);
        PipeBarrier<PIPE_ALL>();
    }}
{bias_body}
{cast_body}
    DataCopyParams nt_ordered_store;
    nt_ordered_store.blockCount = 32; nt_ordered_store.blockLen = {block_len};
    nt_ordered_store.srcStride = 0; nt_ordered_store.dstStride = {n // divisor - block_len};
    DataCopy(nt_ordered_output_gm[nt_ordered_base], {result}, nt_ordered_store);
    PipeBarrier<PIPE_ALL>();"""
    guard = " && ".join(
        [
            *(
                f"({access.strides[0]}) == {n} && ({access.strides[1]}) == 1"
                for access in accesses
            ),
            *(f"((uintptr_t)({name})) % 32 == 0" for name in (scratch, output)),
        ]
    )
    metadata = {
        "ascendc_ordered_dot_add": {
            "enabled": True,
            "groups": groups,
            "tile": (32, 32),
            "accumulator_dtype": "float32",
            "output_dtype": output_dtype,
            "operand_order": "carry-first" if carry_first else "partial-first",
            "accumulation_order": "original-serial-k32-add",
            "reassociation": False,
            "ub_bytes": 10240 if out_type == "half" else 8192,
            "literal_add_epilogue": bias is not None,
            "origin": "typed-ssa-and-full-access-proof",
        }
    }

    return NativeDotBody(body, str((m // 32) * (n // 32)), guard, metadata)


__all__ = ["render_ordered_dot_add"]
