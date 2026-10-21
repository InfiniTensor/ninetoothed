import ast
from types import SimpleNamespace

import numpy as np
import pytest

from ninetoothed import Tensor
from ninetoothed.generation import CodeGenerator


def _pointers_and_mask(tensor, pid=0):
    generator = CodeGenerator()
    generator._invariants = {}
    pointers, mask = generator._generate_pointers_and_mask(tensor, ())
    language = SimpleNamespace(
        arange=np.arange,
        broadcast=np.broadcast_arrays,
        program_id=lambda axis: pid,
    )
    context = {"ninetoothed": SimpleNamespace(language=language)}
    context[tensor.source.pointer_string()] = 0
    strides = np.empty(tuple(int(str(size)) for size in tensor.source.shape)).strides
    itemsize = np.dtype(float).itemsize

    for dim, stride in enumerate(strides):
        context[tensor.source.stride_string(dim)] = stride // itemsize

    def evaluate(expression):
        tree = ast.parse(str(expression), mode="eval")

        return eval(compile(tree, "<generated-address>", "eval"), context)

    for name, expression in generator._invariants.items():
        context[str(name)] = evaluate(expression)

    return evaluate(pointers), evaluate(mask), str(pointers)


@pytest.mark.parametrize("pid", (0, 1, 2, 3))
def test_expanded_vector_load_retains_masked_rows(pid):
    vector = Tensor(shape=(5,), name="vector")
    tiled = vector[None, :].expand((7, -1)).tile((4, 4))
    pointers, mask, source = _pointers_and_mask(tiled, pid)
    rows = pid // 2 * 4 + np.arange(4)[:, None]
    columns = pid % 2 * 4 + np.arange(4)[None, :]

    assert pointers.shape == (4, 4)
    np.testing.assert_array_equal(pointers, np.broadcast_to(columns, (4, 4)))
    np.testing.assert_array_equal(mask, (rows < 7) & (columns < 5))
    assert "broadcast" in source


def test_expanded_permuted_flattened_vector_matches_spatial_bias():
    vector = Tensor(shape=(5,), name="bias")
    arranged = vector[None, :, None, None].expand((2, -1, 3, 3))
    tiled = arranged.permute((0, 2, 3, 1)).flatten(end_dim=3).tile((16, 8))
    pointers, mask, _ = _pointers_and_mask(tiled, pid=1)
    rows = 16 + np.arange(16)[:, None]
    columns = np.arange(8)[None, :]

    assert pointers.shape == (16, 8)
    np.testing.assert_array_equal(pointers, np.broadcast_to(columns, (16, 8)))
    np.testing.assert_array_equal(mask, (rows < 18) & (columns < 5))


def test_dense_tile_does_not_add_redundant_broadcast():
    tiled = Tensor(shape=(7, 5), name="matrix").tile((4, 4))
    pointers, mask, source = _pointers_and_mask(tiled, pid=3)
    rows = 4 + np.arange(4)[:, None]
    columns = 4 + np.arange(4)[None, :]

    np.testing.assert_array_equal(pointers, rows * 5 + columns)
    np.testing.assert_array_equal(mask, (rows < 7) & (columns < 5))
    assert "broadcast" not in source
