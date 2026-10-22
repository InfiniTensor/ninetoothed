"""AscendC vector writes of proven uniform float32 SSA values."""

from dataclasses import dataclass

from ninetoothed.backends.ascendc_resources import ASCENDC_UB_BYTES
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.uniform_transfer import plan_uniform_transfer

_CHUNK = 32768


@dataclass(frozen=True)
class UniformBody:
    body: str
    guard: str
    tasks: str
    chunk: int = _CHUNK


def render_uniform_store(context):
    """Use one bounded output queue; preserve arithmetic and exact tail writes.

    Queue ownership orders the vector Duplicate before the GM copy. An aligned
    output base and a sixteen-float prefix keep scalar GM stores in separate
    64-byte cache lines from the preceding DMA and neighboring tasks. Pure
    DMA outputs still permit a 32-byte base; a scalar tail requires 64 bytes.
    At most fifteen final lanes keep the original scalar store value.
    The caller must keep the complete indexed fallback for a false guard.
    """
    plan = plan_uniform_transfer(context)

    if plan is None or _CHUNK * 4 > ASCENDC_UB_BYTES:
        return None

    output, extent = plan.output, plan.extent
    guard = (
        f"({plan.guard}) && (((uintptr_t){output}) & 31ULL) == 0"
        f" && ((({extent}) % 16) == 0 || (((uintptr_t){output}) & 63ULL) == 0)"
    )
    tasks = f"(({extent}) + {_CHUNK} - 1) / {_CHUNK}"
    body = f"""    const int64_t nt_uniform_extent = {extent};
    const int64_t nt_uniform_base = (int64_t)GetBlockIdx() * {_CHUNK};
    if (nt_uniform_base >= nt_uniform_extent) {{ return; }}
    int64_t nt_uniform_count = nt_uniform_extent - nt_uniform_base;
    if (nt_uniform_count > {_CHUNK}) {{ nt_uniform_count = {_CHUNK}; }}
    const int64_t nt_uniform_aligned = nt_uniform_count & ~15LL;
{common.indent_block(plan.value_body, "    ")}
    if (nt_uniform_aligned > 0) {{
        TPipe nt_uniform_pipe;
        TQue<TPosition::VECOUT, 1> nt_uniform_queue;
        nt_uniform_pipe.InitBuffer(nt_uniform_queue, 1, {_CHUNK} * sizeof(float));
        GlobalTensor<float> nt_uniform_gm;
        nt_uniform_gm.SetGlobalBuffer({output}, nt_uniform_extent);
        LocalTensor<float> nt_uniform_local = nt_uniform_queue.AllocTensor<float>();
        Duplicate(nt_uniform_local, nt_uniform_value, (int32_t)nt_uniform_aligned);
        nt_uniform_queue.EnQue(nt_uniform_local);
        LocalTensor<float> nt_uniform_ready = nt_uniform_queue.DeQue<float>();
        DataCopy(nt_uniform_gm[nt_uniform_base], nt_uniform_ready, (uint32_t)nt_uniform_aligned);
        PipeBarrier<PIPE_ALL>();
        nt_uniform_queue.FreeTensor<float>(nt_uniform_ready);
    }}
    for (int64_t nt_uniform_tail = nt_uniform_aligned; nt_uniform_tail < nt_uniform_count; nt_uniform_tail++) {{
        {output}[nt_uniform_base + nt_uniform_tail] = nt_uniform_value;
    }}"""

    return UniformBody(body, guard, tasks)
