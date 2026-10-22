"""Coalesced float4 writes for a proven lane-independent terminal store."""

from dataclasses import dataclass

from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.uniform_transfer import plan_uniform_transfer


@dataclass(frozen=True)
class UniformBody:
    body: str
    guard: str
    blocks: str


def render_uniform_store(context):
    """Keep the original scalar arithmetic and guard the physical flat write."""
    proof = plan_uniform_transfer(context)

    if proof is None or context.outputs != (proof.output,):
        return None

    extent = proof.extent
    # The low-CTA schedule is validated for cache-resident output. Larger
    # streaming writes retain the original schedule, whose bandwidth is
    # higher on the measured target. Symbolic extents keep the same guard.

    try:
        if int(extent) > 16777216:
            return None
    except ValueError:
        pass

    value_body = proof.value_body
    output = proof.output
    guard = f"({proof.guard}) && ({extent} <= 16777216) && ((uintptr_t){output} % alignof(float4) == 0)"
    block_count = f"(({extent} + threads * 4 - 1) / (threads * 4))"
    blocks = f"({block_count} < 512 ? {block_count} : 512)"
    body = f"""{common.indent_block(value_body, "    ")}
    const float4 nt_uniform_pack = make_float4(nt_uniform_value, nt_uniform_value, nt_uniform_value, nt_uniform_value);
    const int64_t nt_uniform_extent = {extent};
    const int64_t nt_uniform_lane = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nt_uniform_step = (int64_t)gridDim.x * blockDim.x;
    for (int64_t nt_uniform_vector = nt_uniform_lane; nt_uniform_vector < nt_uniform_extent / 4; nt_uniform_vector += nt_uniform_step) {{
        reinterpret_cast<float4*>({output})[nt_uniform_vector] = nt_uniform_pack;
    }}
    if (nt_uniform_lane < nt_uniform_extent % 4) {{
        {output}[(nt_uniform_extent / 4) * 4 + nt_uniform_lane] = nt_uniform_value;
    }}"""

    return UniformBody(body, guard, blocks)
