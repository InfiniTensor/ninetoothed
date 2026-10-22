"""Expose independent raw-pointer loads without assuming bounded indices."""

from dataclasses import dataclass

from ninetoothed.backends.ascendc_resources import ASCENDC_UB_BYTES
from ninetoothed.backends.emitters.base import private_parameter_conflicts
from ninetoothed.backends.emitters.indexed_transfer import plan_indexed_transfer

_CHUNK = 1024
_BATCH = 8
_PREFETCH_MIN_EXTENT = 1 << 24
_AUTO_BATCH = object()


@dataclass(frozen=True)
class IndexedBody:
    body: str
    guard: str
    tasks: str
    chunk: int
    ub_bytes: int
    prefetch_dma: bool = False
    prefetch_min_extent: int = _PREFETCH_MIN_EXTENT
    prefetch_batch: int = _BATCH


def render_indexed_transfer(
    context, *, chunk=_CHUNK, batch=_AUTO_BATCH, prefetch_dma=None
):
    """Prefetch immutable indices and batch only disjoint actual read addresses.

    Indices retain their signed int32 values and raw pointer semantics. A source
    tensor's declared shape/stride does not bound those addresses. Each batch
    checks its actual four-byte reads against the complete output write domain;
    alignment makes the first-address test equivalent to interval disjointness.
    A potentially dependent batch keeps the original read/write order. Output
    stores never move across batches. The caller keeps the whole original
    kernel and launch domain when the host guard fails.
    """
    if private_parameter_conflicts(
        context,
        ("nt_",),
        excluded_prefixes=("nt_materialized_", "nt_dot_partials"),
    ):
        return None

    automatic_batch = batch is _AUTO_BATCH

    if automatic_batch:
        batch = _BATCH

    if (
        not isinstance(chunk, int)
        or isinstance(chunk, bool)
        or chunk <= 0
        or chunk > 16384
        or chunk % 16 != 0
        or not isinstance(batch, int)
        or isinstance(batch, bool)
        or batch not in (8, 32)
        or (prefetch_dma is not None and not isinstance(prefetch_dma, bool))
        or chunk * 4 + 256 > ASCENDC_UB_BYTES
        or getattr(context, "block_program", False)
        or getattr(context, "cooperative_reduction_program", False)
    ):
        return None

    plan = plan_indexed_transfer(context)

    if plan is None or context.tensors[plan.source].ndim <= 0:
        return None

    source, indices, output = plan.source, plan.indices, plan.output
    extent = plan.extent
    measured_large_domain = (
        extent.isascii() and extent.isdigit() and int(extent) >= _PREFETCH_MIN_EXTENT
    )

    if prefetch_dma is None:
        # Small and symbolic extents preserve the original scalar ILP body.
        # Four/five AIV waves regressed on device; only the measured 16M+
        # domain selects exact MTE2 reads, without altering raw index bounds.
        prefetch_dma = (
            measured_large_domain and chunk * 4 + batch * 32 + 256 <= ASCENDC_UB_BYTES
        )

    if automatic_batch and prefetch_dma and measured_large_domain:
        # Only the same measured 16M+ domain batches 32 exact MTE2 reads.
        # Small/symbolic bodies and all dependent read/write order stay intact.
        batch = 32

    if batch == 32 and not prefetch_dma:
        return None

    if prefetch_dma and chunk * 4 + batch * 32 + 256 > ASCENDC_UB_BYTES:
        return None

    index_ptr, output_ptr = f"((uintptr_t){indices})", f"((uintptr_t){output})"
    byte_span = f"((uint64_t)({extent}) * 4ULL)"
    disjoint_indices = (
        f"(({index_ptr} <= {output_ptr} && {output_ptr} - {index_ptr} >= {byte_span}) || "
        f"({output_ptr} <= {index_ptr} && {index_ptr} - {output_ptr} >= {byte_span}))"
    )
    guard = " && ".join(
        (
            f"({plan.guard})",
            f"(({extent}) > 0 && ({extent}) <= {((1 << 63) - 1 - chunk) // 4}LL)",
            f"((((uintptr_t){source}) & 3ULL) == 0)",
            f"(({index_ptr} & 3ULL) == 0)",
            f"(({output_ptr} & 63ULL) == 0)",
            disjoint_indices,
        )
    )
    pointers = "\n".join(
        f"        __gm__ float* nt_indexed_read_{lane} = {source} + (int64_t)nt_indexed_ready.GetValue(nt_indexed_j + {lane});"
        for lane in range(batch)
    )
    disjoint_reads = " && ".join(
        f"(((uintptr_t)nt_indexed_read_{lane}) < nt_indexed_output_begin || "
        f"((uintptr_t)nt_indexed_read_{lane}) - nt_indexed_output_begin >= nt_indexed_output_bytes)"
        for lane in range(batch)
    )
    loads = "\n".join(
        f"            float nt_indexed_value_{lane} = *nt_indexed_read_{lane};"
        for lane in range(batch)
    )
    setup = ""

    if prefetch_dma:
        setup = f"""
    TBuf<TPosition::VECIN> nt_indexed_prefetch_buffer;
    nt_indexed_pipe.InitBuffer(nt_indexed_prefetch_buffer, {batch * 32});
    LocalTensor<float> nt_indexed_prefetch = nt_indexed_prefetch_buffer.Get<float>();
    DataCopyExtParams nt_indexed_one_read(1, 4, 0, 0, 0);
    DataCopyPadExtParams<float> nt_indexed_one_padding(false, 0, 0, 0);"""
        loads = "\n".join(
            f"            GlobalTensor<float> nt_indexed_one_source_{lane};\n"
            f"            nt_indexed_one_source_{lane}.SetGlobalBuffer(nt_indexed_read_{lane}, 1);\n"
            f"            DataCopyPad(nt_indexed_prefetch[{lane * 8}], nt_indexed_one_source_{lane},\n"
            "                nt_indexed_one_read, nt_indexed_one_padding);"
            for lane in range(batch)
        )
        loads += "\n            PipeBarrier<PIPE_ALL>();\n"
        loads += "\n".join(
            f"            float nt_indexed_value_{lane} = nt_indexed_prefetch.GetValue({lane * 8});"
            for lane in range(batch)
        )

    stores = "\n".join(
        f"            {output}[nt_indexed_base + nt_indexed_j + {lane}] = nt_indexed_value_{lane};"
        for lane in range(batch)
    )

    if prefetch_dma:
        # Scalar consumes all slots before MTE2 can overwrite them on
        # another batch. The original ordered GM stores remain unchanged.
        stores += "\n            PipeBarrier<PIPE_ALL>();"

    body = f"""    const int64_t nt_indexed_base = (int64_t)GetBlockIdx() * {chunk};
    int64_t nt_indexed_count = ({extent}) - nt_indexed_base;
    if (nt_indexed_count > {chunk}) {{ nt_indexed_count = {chunk}; }}
    if (nt_indexed_count <= 0) {{ return; }}
    const uintptr_t nt_indexed_output_begin = (uintptr_t){output};
    const uint64_t nt_indexed_output_bytes = {byte_span};
    TPipe nt_indexed_pipe;
    TQue<TPosition::VECIN, 1> nt_indexed_queue;
    nt_indexed_pipe.InitBuffer(nt_indexed_queue, 1, {chunk} * sizeof(int32_t));{setup}
    GlobalTensor<int32_t> nt_indexed_positions;
    nt_indexed_positions.SetGlobalBuffer({indices}, ({extent}));
    LocalTensor<int32_t> nt_indexed_local = nt_indexed_queue.AllocTensor<int32_t>();
    DataCopyExtParams nt_indexed_read(1, (uint32_t)(nt_indexed_count * 4LL), 0, 0, 0);
    DataCopyPadExtParams<int32_t> nt_indexed_padding(false, 0, 0, 0);
    DataCopyPad(nt_indexed_local, nt_indexed_positions[nt_indexed_base],
        nt_indexed_read, nt_indexed_padding);
    nt_indexed_queue.EnQue(nt_indexed_local);
    LocalTensor<int32_t> nt_indexed_ready = nt_indexed_queue.DeQue<int32_t>();
    int64_t nt_indexed_j = 0;
    for (; nt_indexed_j + {batch} <= nt_indexed_count; nt_indexed_j += {batch}) {{
{pointers}
        if ({disjoint_reads}) {{
{loads}
{stores}
        }} else {{
            for (int64_t nt_indexed_lane = 0; nt_indexed_lane < {batch}; ++nt_indexed_lane) {{
                {output}[nt_indexed_base + nt_indexed_j + nt_indexed_lane] =
                    {source}[(int64_t)nt_indexed_ready.GetValue(nt_indexed_j + nt_indexed_lane)];
            }}
        }}
    }}
    for (; nt_indexed_j < nt_indexed_count; ++nt_indexed_j) {{
        {output}[nt_indexed_base + nt_indexed_j] =
            {source}[(int64_t)nt_indexed_ready.GetValue(nt_indexed_j)];
    }}
    nt_indexed_queue.FreeTensor<int32_t>(nt_indexed_ready);"""

    return IndexedBody(
        body,
        guard,
        f"(({extent}) + {chunk} - 1) / {chunk}",
        chunk,
        chunk * 4 + (batch * 32 if prefetch_dma else 0),
        prefetch_dma,
        _PREFETCH_MIN_EXTENT,
        batch,
    )
