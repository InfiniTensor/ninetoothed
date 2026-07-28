"""Probe PyTorch jagged layout construction support."""

import functools

from tests.capabilities import run_python_probe


def _probe(jagged_dim):
    shapes = ((2, 3), (4, 3)) if jagged_dim == 1 else ((2, 3), (2, 5))
    source = f"""
import json
import torch

try:
    batches = tuple(torch.randn(shape, device="cuda") for shape in {shapes!r})
    torch.nested.nested_tensor(batches, layout=torch.jagged)
except Exception as error:
    print(json.dumps({{"status": "unavailable", "reason": str(error)}}))
else:
    print(json.dumps({{"status": "supported"}}))
"""

    return run_python_probe(f"torch jagged_dim={jagged_dim}", source)


@functools.cache
def jagged_dim_1():
    return _probe(1)


@functools.cache
def jagged_dim_2():
    return _probe(2)


__all__ = ["jagged_dim_1", "jagged_dim_2"]
