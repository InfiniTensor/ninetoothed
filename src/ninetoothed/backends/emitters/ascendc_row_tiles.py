"""Ascend UB plans consuming the shared complete-row retile proof."""

from dataclasses import dataclass

from ninetoothed.backends.ascendc_resources import ASCENDC_UB_BYTES
from ninetoothed.backends.emitters.ascendc_vector_dag import (
    match_pointwise_rows,
    match_row_dag,
)
from ninetoothed.backends.emitters.row_tiles import FullRowTiles, tile_full_rows


@dataclass(frozen=True)
class FullRowDAG:
    tiles: object
    dag: object
    origin: str = "shared-full-row-access-proof"


def plan_full_rows(context, *, ub_bytes=ASCENDC_UB_BYTES):
    tiles = tile_full_rows(context)

    if tiles is None:
        dag = match_pointwise_rows(context, ub_bytes=ub_bytes)

        if dag is None:
            return None

        return FullRowDAG(
            FullRowTiles(context, int(dag.rows), dag.cols),
            dag,
            "shared-row-input-access-proof",
        )

    dag = match_row_dag(tiles.context, ub_bytes=ub_bytes)

    return FullRowDAG(tiles, dag) if dag is not None else None


__all__ = ["FullRowDAG", "plan_full_rows"]
