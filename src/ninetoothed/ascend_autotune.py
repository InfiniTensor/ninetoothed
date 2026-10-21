"""Filter Ascend tuning candidates before an invalid device launch."""

import math
import os


def make_grid_pruner(grid):
    """Build a runtime-shape filter for Ascend's direct-launch grid limit."""

    def prune(configs, named_args, **kwargs):
        if os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "false").lower() in ("true", "1"):
            return configs

        valid = []

        for config in configs:
            extents = grid(named_args | kwargs | config.kwargs)

            if all(size >= 0 for size in extents) and math.prod(extents) <= 65535:
                valid.append(config)

        if not valid:
            raise ValueError(
                "No Ascend tuning configuration fits the direct-launch grid limit of 65535 blocks."
            )

        return valid

    return prune
