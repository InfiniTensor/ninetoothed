"""Bounded UB gathering for proved full-prefix float32 affine transfers."""

from dataclasses import dataclass

from ninetoothed.backends.ascendc_resources import ASCENDC_UB_BYTES
from ninetoothed.backends.emitters.affine_transfer import (
    affine_transfer_memory_guard,
    plan_affine_transfer,
)

_CHUNK = 6144
_MAX_STRIDE = 4
_UB_BYTES = (_MAX_STRIDE + 3) * _CHUNK * 4


@dataclass(frozen=True)
class AffineBody:
    body: str
    guard: str
    tasks: str
    chunk: int = _CHUNK
    ub_bytes: int = _UB_BYTES


def render_affine_transfer(context):
    """Copy only a proven valid source span, then gather bounded UB offsets.

    The caller retains one complete original indexed fallback when the guard
    is false. Source/output strides and output 32-byte ownership are runtime
    obligations. GM transfers use exact byte counts, including the last
    short task; padding can only occupy the task's private UB allocation.
    """
    if context.block_program or context.cooperative_reduction_program:
        return None

    plan = plan_affine_transfer(context)

    if (
        plan is None
        or _UB_BYTES > ASCENDC_UB_BYTES
        or plan.extent + plan.offset > (1 << 61) - _CHUNK
    ):
        return None

    source, output = plan.source, plan.output
    stride = f"({plan.source_stride})"
    predicates = [
        f"{stride} > 0",
        f"{stride} <= {_MAX_STRIDE}",
        f"({plan.output_stride}) == 1",
        f"(((uintptr_t){source}) & 3ULL) == 0",
        f"(((uintptr_t){output}) & 31ULL) == 0",
    ]

    if plan.offset == 0:
        # Keep the existing measured contiguous-copy schedule.
        predicates.append(f"{stride} != 1")

    predicates.append(affine_transfer_memory_guard(plan, max_source_stride=_MAX_STRIDE))

    guard = " && ".join(f"({predicate})" for predicate in predicates)
    tasks = str((plan.extent + _CHUNK - 1) // _CHUNK)
    body = f"""    const int64_t nt_affine_start = (int64_t)GetBlockIdx() * {_CHUNK};
    int64_t nt_affine_count = {plan.extent} - nt_affine_start;
    if (nt_affine_count > {_CHUNK}) {{ nt_affine_count = {_CHUNK}; }}
    if (nt_affine_count <= 0) {{ return; }}
    const int64_t nt_affine_span = (nt_affine_count - 1) * {stride} + 1;
    TPipe nt_affine_pipe;
    TQue<TPosition::VECIN, 1> nt_affine_input_queue;
    TQue<TPosition::VECOUT, 1> nt_affine_output_queue;
    TBuf<TPosition::VECCALC> nt_affine_float_offsets_buffer;
    TBuf<TPosition::VECCALC> nt_affine_int_offsets_buffer;
    nt_affine_pipe.InitBuffer(nt_affine_input_queue, 1, {_MAX_STRIDE * _CHUNK} * sizeof(float));
    nt_affine_pipe.InitBuffer(nt_affine_output_queue, 1, {_CHUNK} * sizeof(float));
    nt_affine_pipe.InitBuffer(nt_affine_float_offsets_buffer, {_CHUNK} * sizeof(float));
    nt_affine_pipe.InitBuffer(nt_affine_int_offsets_buffer, {_CHUNK} * sizeof(int32_t));
    GlobalTensor<float> nt_affine_source;
    GlobalTensor<float> nt_affine_output;
    nt_affine_source.SetGlobalBuffer({source}, ({plan.extent + plan.offset - 1}) * {stride} + 1);
    nt_affine_output.SetGlobalBuffer({output}, {plan.extent});
    LocalTensor<float> nt_affine_input = nt_affine_input_queue.AllocTensor<float>();
    DataCopyExtParams nt_affine_read(1, (uint32_t)(nt_affine_span * sizeof(float)), 0, 0, 0);
    DataCopyPadExtParams<float> nt_affine_padding(false, 0, 0, 0.0f);
    DataCopyPad(nt_affine_input, nt_affine_source[(nt_affine_start + {plan.offset}) * {stride}], nt_affine_read, nt_affine_padding);
    nt_affine_input_queue.EnQue(nt_affine_input);
    LocalTensor<float> nt_affine_ready = nt_affine_input_queue.DeQue<float>();
    DataCopyExtParams nt_affine_write(1, (uint32_t)(nt_affine_count * sizeof(float)), 0, 0, 0);
    if ({stride} == 1) {{
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(nt_affine_output[nt_affine_start], nt_affine_ready, nt_affine_write);
        PipeBarrier<PIPE_ALL>();
    }} else {{
        LocalTensor<float> nt_affine_offsets_float = nt_affine_float_offsets_buffer.Get<float>();
        LocalTensor<int32_t> nt_affine_offsets_int = nt_affine_int_offsets_buffer.Get<int32_t>();
        CreateVecIndex(nt_affine_offsets_float, 0.0f, (uint32_t)nt_affine_count);
        PipeBarrier<PIPE_V>();
        Muls(nt_affine_offsets_float, nt_affine_offsets_float, (float)({stride} * 4LL), (int32_t)nt_affine_count);
        PipeBarrier<PIPE_V>();
        Cast(nt_affine_offsets_int, nt_affine_offsets_float, RoundMode::CAST_TRUNC, (int32_t)nt_affine_count);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> nt_affine_values = nt_affine_output_queue.AllocTensor<float>();
        Gather(nt_affine_values, nt_affine_ready, nt_affine_offsets_int.ReinterpretCast<uint32_t>(), 0, (uint32_t)nt_affine_count);
        nt_affine_output_queue.EnQue(nt_affine_values);
        LocalTensor<float> nt_affine_written = nt_affine_output_queue.DeQue<float>();
        DataCopyPad(nt_affine_output[nt_affine_start], nt_affine_written, nt_affine_write);
        nt_affine_output_queue.FreeTensor<float>(nt_affine_written);
    }}
    nt_affine_input_queue.FreeTensor<float>(nt_affine_ready);"""

    return AffineBody(body, guard, tasks)
