"""Typed NRAM plans consuming target-neutral full-row ownership proofs."""

from dataclasses import dataclass

from ninetoothed.backends.emitters import bangc_vector
from ninetoothed.backends.emitters.row_tiles import tile_full_rows


@dataclass(frozen=True)
class FullRowVectorPlan:
    tiles: object
    vector: object


def plan_full_rows(context, *, budget=192 * 1024):
    tiled = tile_full_rows(context)

    if tiled is None:
        return None

    vector = bangc_vector.plan_rows(tiled.context, budget=budget)

    return FullRowVectorPlan(tiled, vector) if vector is not None else None


__all__ = ["FullRowVectorPlan", "plan_full_rows"]
