"""DMA lowering for proved cropped and strided float32 copy domains."""

from ninetoothed.backends.emitters.affine_transfer import (
    affine_transfer_memory_guard,
    plan_affine_transfer,
)
from ninetoothed.backends.emitters.bangc_memory import IndexedTransferBody
from ninetoothed.backends.emitters.bangc_ranges import ceil_divide

_UNPLANNED = object()


def render_affine_transfer(
    context, fallback_tasks, *, chunk=32768, coalesced=False, plan=_UNPLANNED
):
    """Select task counts with the same runtime guard as the kernel branch."""
    if not isinstance(chunk, int) or not 1 <= chunk <= 49152:
        return None

    if plan is _UNPLANNED:
        plan = plan_affine_transfer(context)

    if plan is None:
        return None

    predicates = [
        f"({plan.source_stride}) > 0",
        f"({plan.source_stride}) <= 536870911",
        f"({plan.output_stride}) == 1",
    ]

    if plan.offset == 0:
        # Ordinary contiguous copies retain their existing staged schedule.
        predicates.append(f"({plan.source_stride}) != 1")

    predicates.append(affine_transfer_memory_guard(plan, max_source_stride=536870911))

    guard = " && ".join(predicates)

    # Short domains retain the measured lower-latency segmented DMA body.
    if coalesced and plan.extent > 8192:
        storage = max(chunk, 49152)
        packed = f"(({plan.source_stride}) > 1 && ({plan.source_stride}) <= 4)"
        width = f"(({plan.source_stride}) == 2 ? 16384 : ({plan.source_stride}) == 3 ? 12288 : ({plan.source_stride}) == 4 ? 9216 : {chunk})"
        body = f"""    __nram__ float nt_affine_storage[{storage}];
    int64_t nt_affine_width = {width};
    int64_t nt_affine_start = (int64_t)(taskIdX) * nt_affine_width;
    int64_t nt_affine_count = {plan.extent} - nt_affine_start;
    if (nt_affine_count > nt_affine_width) {{ nt_affine_count = nt_affine_width; }}
    if (nt_affine_count <= 0) {{ return; }}
    const float* nt_affine_input = {plan.source} + (nt_affine_start + {plan.offset}) * ({plan.source_stride});
    float* nt_affine_values = nt_affine_storage + ({packed} ? nt_affine_width * ({plan.source_stride}) : 0);
    if (({plan.source_stride}) == 1) {{
        __memcpy(nt_affine_values, nt_affine_input, nt_affine_count * sizeof(float), GDRAM2NRAM);
    }} else if ({packed}) {{
        int64_t nt_affine_span = (nt_affine_count - 1) * ({plan.source_stride}) + 1;
        __memcpy(nt_affine_storage, nt_affine_input, nt_affine_span * sizeof(float), GDRAM2NRAM);
        __memcpy(nt_affine_values, nt_affine_storage, sizeof(float), NRAM2NRAM,
                 sizeof(float), ({plan.source_stride}) * sizeof(float), nt_affine_count - 1);
    }} else {{
        __memcpy(nt_affine_values, nt_affine_input, sizeof(float), GDRAM2NRAM,
                 sizeof(float), ({plan.source_stride}) * sizeof(float), nt_affine_count - 1);
    }}
    __memcpy({plan.output} + nt_affine_start, nt_affine_values,
             nt_affine_count * sizeof(float), NRAM2GDRAM);"""
        tasks = f"({guard}) ? ({ceil_divide(plan.extent, width)}) : ({fallback_tasks})"

        return IndexedTransferBody(body, guard, tasks)

    body = f"""    __nram__ float nt_affine_values[{chunk}];
    int64_t nt_affine_start = (int64_t)(taskIdX) * {chunk};
    int64_t nt_affine_count = {plan.extent} - nt_affine_start;
    if (nt_affine_count > {chunk}) {{ nt_affine_count = {chunk}; }}
    if (nt_affine_count <= 0) {{ return; }}
    const float* nt_affine_input = {plan.source} + (nt_affine_start + {plan.offset}) * ({plan.source_stride});
    if (({plan.source_stride}) == 1) {{
        __memcpy(nt_affine_values, nt_affine_input, nt_affine_count * sizeof(float), GDRAM2NRAM);
    }} else {{
        __memcpy(nt_affine_values, nt_affine_input, sizeof(float), GDRAM2NRAM,
                 sizeof(float), ({plan.source_stride}) * sizeof(float), nt_affine_count - 1);
    }}
    __memcpy({plan.output} + nt_affine_start, nt_affine_values,
             nt_affine_count * sizeof(float), NRAM2GDRAM);"""
    tasks = f"({guard}) ? ({ceil_divide(plan.extent, chunk)}) : ({fallback_tasks})"

    return IndexedTransferBody(body, guard, tasks)
