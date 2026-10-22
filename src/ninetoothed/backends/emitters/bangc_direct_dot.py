"""Reuse native LT lowering after proof-driven direct-dot tiling."""

from dataclasses import replace

from ninetoothed.backends.emitters.bangc_dot import render_native_dot
from ninetoothed.backends.emitters.direct_dot import tile_direct_dot


def render_direct_dot(context, target):
    """Keep the original fallback grid under the same native layout guard."""
    plan = tile_direct_dot(context)

    if plan is None:
        return None

    body = render_native_dot(
        plan.context,
        target,
        dma=True,
        stage_output=True,
        vector_epilogue=True,
        right_transposed=plan.right_transposed,
    )

    if body is None:
        return None
    return replace(
        body, tasks=f"({body.guard}) ? ({body.tasks}) : ({context.grid_total})"
    )
