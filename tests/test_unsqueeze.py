import numpy as np

import ninetoothed
from ninetoothed import Tensor


def test_unsqueeze():
    x = Tensor(1)

    block_size_m = ninetoothed.block_size()
    block_size_n = ninetoothed.block_size()

    subs = {x: {"shape": (5,)}, block_size_m: 2, block_size_n: 3}

    x_unsqueezed = x.unsqueeze(0)

    assert np.array_equal(x_unsqueezed.eval(subs), np.array([[0, 1, 2, 3, 4]]))

    x_unsqueezed_expanded = x_unsqueezed.expand((3, -1))

    assert np.array_equal(
        x_unsqueezed_expanded.eval(subs),
        np.array(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ]
        ),
    )

    x_unsqueezed_expanded_tiled = x_unsqueezed_expanded.tile(
        (block_size_m, block_size_n)
    )

    assert np.array_equal(
        x_unsqueezed_expanded_tiled.eval(subs),
        np.array(
            [
                [[[0, 1, 2], [0, 1, 2]], [[3, 4, -1], [3, 4, -1]]],
                [[[0, 1, 2], [-1, -1, -1]], [[3, 4, -1], [-1, -1, -1]]],
            ]
        ),
    )

    x_tiled_unsqueezed = x.tile((block_size_n,)).unsqueeze(0)

    assert np.array_equal(
        x_tiled_unsqueezed.eval(subs), np.array([[[0, 1, 2], [3, 4, -1]]])
    )

    x_tiled_unsqueezed_expanded = x_tiled_unsqueezed.expand((3, -1))

    assert np.array_equal(
        x_tiled_unsqueezed_expanded.eval(subs),
        np.array(
            [[[0, 1, 2], [3, 4, -1]], [[0, 1, 2], [3, 4, -1]], [[0, 1, 2], [3, 4, -1]]]
        ),
    )
