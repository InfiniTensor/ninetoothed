"""Bound AscendC launches without changing logical tasks or the public ABI."""

from dataclasses import dataclass

from ninetoothed.backends.emitters.task_launch import (
    TaskLaunchSymbols,
    allocate_task_launch_symbols,
)

MAX_BLOCK_DIM = 65535


@dataclass(frozen=True)
class LaunchSymbols(TaskLaunchSymbols):
    def kernel_body(self, body):
        """Check the complete logical domain before any memory operation."""
        return super().kernel_body(
            body,
            task_pattern=r"GetBlockIdx\(\)",
            physical_index="GetBlockIdx()",
            max_tasks=MAX_BLOCK_DIM,
        )


def launch_symbols(*source_fragments):
    return allocate_task_launch_symbols(*source_fragments, symbol_type=LaunchSymbols)


def render_segmented_launch(
    kernel_name, arguments, tasks, symbols, *, indent="    ", stream="stream"
):
    """Launch consecutive <=65535 task ranges on the caller's stream."""
    args = ", ".join((*arguments, symbols.offset, symbols.total))
    lines = f"""const int64_t {symbols.total} = {tasks};
if ({symbols.total} <= 0) {{ return 0; }}
for (int64_t {symbols.offset} = 0; {symbols.offset} < {symbols.total};) {{
    int64_t {symbols.segment} = {symbols.total} - {symbols.offset};
    if ({symbols.segment} > {MAX_BLOCK_DIM}LL) {{ {symbols.segment} = {MAX_BLOCK_DIM}LL; }}
    {kernel_name}<<<(uint32_t)({symbols.segment}), nullptr, {stream}>>>(
        {args}
    );
    {symbols.offset} += {symbols.segment};
}}
return 0;"""

    return "\n".join(indent + line for line in lines.splitlines())


def render_native_vector_kernel(
    kernel_name, parameters, arguments, body, guard, tasks, symbols, *, stream="stream"
):
    """Wrap a proved Vector body and keep its original whole-call fallback."""
    source = f"""
extern "C" __global__ __aicore__ void {kernel_name}(
{parameters}
) {{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_CUBE__
    return;
#endif
{symbols.kernel_body(body)}
}}
"""
    launcher = f"""    if ({guard}) {{
{render_segmented_launch(kernel_name, arguments, tasks, symbols, indent="        ", stream=stream)}
    }}
"""

    return source, launcher
