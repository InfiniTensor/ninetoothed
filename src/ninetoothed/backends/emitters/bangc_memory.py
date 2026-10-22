"""Lower proved indexed transfers to the installed GDRAM gather primitive."""

from dataclasses import dataclass

from ninetoothed.backends.emitters.indexed_transfer import plan_indexed_transfer


@dataclass(frozen=True)
class IndexedTransferBody:
    body: str
    guard: str
    tasks: str


_UNPLANNED = object()


def render_indexed_transfer(
    context, chunk, *, fallback_body=None, fallback_tasks=None, plan=_UNPLANNED
):
    """Preserve the original ordered task whenever bulk reads are not independent."""
    if not isinstance(chunk, int) or chunk <= 0 or chunk > 16384:
        return None

    if fallback_body is None or fallback_tasks is None:
        return None

    if plan is _UNPLANNED:
        plan = plan_indexed_transfer(context)

    if plan is None:
        return None

    guard = (
        f"({plan.guard}) && "
        f"(((uintptr_t)({plan.indices}) + ({plan.extent}) * sizeof(int32_t) "
        f"<= (uintptr_t)({plan.output})) || "
        f"((uintptr_t)({plan.output}) + ({plan.extent}) * sizeof(float) "
        f"<= (uintptr_t)({plan.indices})))"
    )
    fallback = "\n".join("    " + line for line in fallback_body.splitlines())
    body = f"""    __nram__ int32_t nt_gather_indices[{chunk}];
    __nram__ uint32_t nt_gather_offsets[{chunk}];
    __nram__ float nt_gather_values[{chunk}];
    int64_t nt_gather_start = (int64_t)(taskIdX) * {chunk};
    int64_t nt_gather_count = ({plan.extent}) - nt_gather_start;
    if (nt_gather_count > {chunk}) {{ nt_gather_count = {chunk}; }}
    if (nt_gather_count <= 0) {{ return; }}
    __memcpy(nt_gather_indices, {plan.indices} + nt_gather_start,
             nt_gather_count * sizeof(int32_t), GDRAM2NRAM);
    int32_t nt_gather_min = 2147483647;
    int32_t nt_gather_max = (-2147483647 - 1);
    const uintptr_t nt_gather_output_begin = (uintptr_t)({plan.output});
    const uintptr_t nt_gather_output_end = nt_gather_output_begin
        + ({plan.extent}) * sizeof(float);
    for (int64_t nt_j = 0; nt_j < nt_gather_count; ++nt_j) {{
        int32_t nt_offset = nt_gather_indices[nt_j];
        if (nt_offset < nt_gather_min) {{ nt_gather_min = nt_offset; }}
        if (nt_offset > nt_gather_max) {{ nt_gather_max = nt_offset; }}
        nt_gather_offsets[nt_j] = (uint32_t)(nt_offset) * sizeof(float);
    }}
    bool nt_gather_narrow = nt_gather_min >= 0 && nt_gather_max <= 1073741823;
    bool nt_gather_independent = false;
    if (nt_gather_narrow) {{
        uintptr_t nt_gather_source_begin = (uintptr_t)({plan.source});
        uintptr_t nt_gather_read_begin = nt_gather_source_begin
            + (uintptr_t)(uint32_t)(nt_gather_min) * sizeof(float);
        uintptr_t nt_gather_read_end = nt_gather_source_begin
            + (uintptr_t)(uint32_t)(nt_gather_max) * sizeof(float) + sizeof(float);
        nt_gather_independent = nt_gather_read_end <= nt_gather_output_begin
            || nt_gather_read_begin >= nt_gather_output_end;
    }}
    if (nt_gather_narrow && nt_gather_independent) {{
        __gather(nt_gather_values, {plan.source}, nt_gather_offsets,
                 sizeof(float), GDRAM2NRAM, sizeof(float), nt_gather_count);
        __memcpy({plan.output} + nt_gather_start, nt_gather_values,
                 nt_gather_count * sizeof(float), NRAM2GDRAM);
    }} else {{
{fallback}
    }}"""

    return IndexedTransferBody(body, guard, str(fallback_tasks))
