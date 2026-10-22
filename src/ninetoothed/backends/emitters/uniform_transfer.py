"""Shared geometry and scalar-code proof for a lane-independent store."""

import re
from dataclasses import dataclass

from ninetoothed.backends.emitters.base import private_parameter_conflicts
from ninetoothed.backends.emitters.staging import (
    flatten_tiled_accesses,
    split_stride_guard,
    staging_extent,
)
from ninetoothed.compiler.passes import program_facts
from ninetoothed.compiler.uniform import UniformStore


@dataclass(frozen=True)
class UniformTransfer:
    output: str
    extent: str
    guard: str
    value_body: str


def plan_uniform_transfer(context):
    """Retain original scalar arithmetic after proving one full flat store.

    The SSA proof excludes coordinates, memory reads, effects and regions.
    The access proof requires a canonical full prefix. Device alignment,
    resource limits and parallel ownership remain target obligations.
    """
    if context.block_program or context.cooperative_reduction_program:
        return None

    # These names remain part of the public ABI. A private local must not
    # shadow a tensor parameter inside the optimized store body.
    if private_parameter_conflicts(context, ("nt_uniform_",)):
        return None

    program = context.kernel.ssa
    proof = program_facts(program, "analysis").get("uniform_store") if program else None

    if not isinstance(proof, UniformStore) or context.outputs != (proof.output,):
        return None

    guard, guarded = split_stride_guard(context.body)

    if guard is None:
        return None

    flat = flatten_tiled_accesses(guarded, context)
    extent = staging_extent(context, [(proof.output, proof.output)])

    if flat is None or extent is None:
        return None

    pattern = re.compile(re.escape(proof.output) + r"\[index\]\s*=\s*([^;\n]+);")

    if len(pattern.findall(flat)) != 1:
        return None

    value_body = pattern.sub(r"float nt_uniform_value = \1;", flat)

    if re.search(
        r"\b(?:index|nt_outer_index|nt_inner_index|nt_pred\w*)\b|\[", value_body
    ):
        return None
    return UniformTransfer(proof.output, str(extent), guard, value_body)
