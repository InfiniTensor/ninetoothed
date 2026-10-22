"""Execute proved output-tile fusion with the ordinary native LT schedule."""

from dataclasses import replace

from ninetoothed.backends.emitters.bangc_dot import render_native_dot
from ninetoothed.backends.emitters.dot_fusion import fuse_dot_output_tiles


def render_fused_dot(context, target, *, tiled=None):
    if tiled is None:
        tiled = fuse_dot_output_tiles(context)

    if tiled is None:
        return None

    body = render_native_dot(
        tiled, target, dma=True, stage_output=True, vector_epilogue=True
    )

    if body is None:
        return None
    return replace(
        body, tasks=f"({body.guard}) ? ({body.tasks}) : ({context.grid_total})"
    )
