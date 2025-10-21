import numpy as np

import ninetoothed
from ninetoothed import Tensor


def test_eval():
    x = Tensor(2)

    block_size_m = ninetoothed.block_size()
    block_size_n = ninetoothed.block_size()

    subs = {x: {"shape": (5, 6)}, block_size_m: 2, block_size_n: 3}

    x_tiled = x.tile((block_size_m, block_size_n))

    x_tiled_evaluated = x_tiled.eval(subs)

    assert (
        x_tiled_evaluated
        == np.array(
            [
                [[[0, 1, 2], [6, 7, 8]], [[3, 4, 5], [9, 10, 11]]],
                [[[12, 13, 14], [18, 19, 20]], [[15, 16, 17], [21, 22, 23]]],
                [[[24, 25, 26], [-1, -1, -1]], [[27, 28, 29], [-1, -1, -1]]],
            ]
        )
    ).all()

    x_tiled_tiled = x_tiled.tile((block_size_m, block_size_n))

    x_tiled_tiled_evaluated = x_tiled_tiled.eval(subs)

    assert (
        x_tiled_tiled_evaluated
        == np.array(
            [
                [
                    [
                        [
                            [[0, 1, 2], [6, 7, 8]],
                            [[3, 4, 5], [9, 10, 11]],
                            [[-1, -1, -1], [-1, -1, -1]],
                        ],
                        [
                            [[12, 13, 14], [18, 19, 20]],
                            [[15, 16, 17], [21, 22, 23]],
                            [[-1, -1, -1], [-1, -1, -1]],
                        ],
                    ]
                ],
                [
                    [
                        [
                            [[24, 25, 26], [-1, -1, -1]],
                            [[27, 28, 29], [-1, -1, -1]],
                            [[-1, -1, -1], [-1, -1, -1]],
                        ],
                        [
                            [[-1, -1, -1], [-1, -1, -1]],
                            [[-1, -1, -1], [-1, -1, -1]],
                            [[-1, -1, -1], [-1, -1, -1]],
                        ],
                    ]
                ],
            ]
        )
    ).all()
