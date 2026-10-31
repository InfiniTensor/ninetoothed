"""Split Block launches at the validated MLU590 physical task limit."""

MAX_BLOCK_TASKS = 65535
MAX_LOGICAL_TASKS = (1 << 32) - 1


def render_segmented_launch(kernel_name, arguments, tasks, symbols, *, queue="queue"):
    """Keep the public queue and stop at the first CNRT launch error."""
    args = ", ".join((*arguments, symbols.offset, symbols.total))

    return f"""    const int64_t {symbols.total} = {tasks};
    if ({symbols.total} <= 0) {{ return 0; }}
    if ({symbols.total} > {MAX_LOGICAL_TASKS}LL) {{
        return static_cast<int>(cnrtErrorInvalidKernel);
    }}
    cnrtFunctionType_t {symbols.kind} = cnrtFuncTypeBlock;
    for (int64_t {symbols.offset} = 0; {symbols.offset} < {symbols.total};) {{
        int64_t {symbols.segment} = {symbols.total} - {symbols.offset};
        if ({symbols.segment} > {MAX_BLOCK_TASKS}LL) {{ {symbols.segment} = {MAX_BLOCK_TASKS}LL; }}
        cnrtDim3_t {symbols.grid};
        {symbols.grid}.x = static_cast<unsigned int>({symbols.segment});
        {symbols.grid}.y = 1;
        {symbols.grid}.z = 1;
        {kernel_name}_kernel<<<{symbols.grid}, {symbols.kind}, {queue}>>>(
            {args}
        );
        const cnrtRet_t {symbols.error} = cnrtGetLastError();
        if ({symbols.error} != cnrtSuccess) {{ return static_cast<int>({symbols.error}); }}
        {symbols.offset} += {symbols.segment};
    }}
    return 0;"""
