import numpy as np
import pytest

import ninetoothed
from ninetoothed import Tensor


@pytest.mark.parametrize("subs_before_eval", (False, True))
@pytest.mark.parametrize("use_tensor_in_subs", (False, True))
def test_eval(use_tensor_in_subs, subs_before_eval):
    assert (Tensor(shape=(2, 3)).eval() == np.array([[0, 1, 2], [3, 4, 5]])).all()

    x = Tensor(2)

    block_size_m = ninetoothed.block_size()
    block_size_n = ninetoothed.block_size()

    subs = {x: {"shape": (5, 6)}, block_size_m: 2, block_size_n: 3}

    if use_tensor_in_subs:
        subs[x] = Tensor(shape=subs[x]["shape"])

    if subs_before_eval:
        x_substituted = x.subs(subs)
        x_evaluated = x_substituted.eval()
    else:
        x_evaluated = x.eval(subs)

    assert (
        x_evaluated
        == np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29],
            ]
        )
    ).all()

    x_tiled = x.tile((block_size_m, block_size_n))

    if subs_before_eval:
        x_tiled_substituted = x_tiled.subs(subs)
        x_tiled_evaluated = x_tiled_substituted.eval()
    else:
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

    if subs_before_eval:
        x_tiled_tiled_substituted = x_tiled_tiled.subs(subs)
        x_tiled_tiled_evaluated = x_tiled_tiled_substituted.eval()
    else:
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

    x_tiled_tiled.dtype = x_tiled_tiled.dtype.flatten()

    if subs_before_eval:
        x_tiled_tiled_substituted = x_tiled_tiled.subs(subs)
        x_tiled_tiled_evaluated = x_tiled_tiled_substituted.eval()
    else:
        x_tiled_tiled_evaluated = x_tiled_tiled.eval(subs)

    assert (
        x_tiled_tiled_evaluated
        == np.array(
            [
                [
                    [
                        [[0, 1, 2], [6, 7, 8]],
                        [[3, 4, 5], [9, 10, 11]],
                        [[-1, -1, -1], [-1, -1, -1]],
                        [[12, 13, 14], [18, 19, 20]],
                        [[15, 16, 17], [21, 22, 23]],
                        [[-1, -1, -1], [-1, -1, -1]],
                    ]
                ],
                [
                    [
                        [[24, 25, 26], [-1, -1, -1]],
                        [[27, 28, 29], [-1, -1, -1]],
                        [[-1, -1, -1], [-1, -1, -1]],
                        [[-1, -1, -1], [-1, -1, -1]],
                        [[-1, -1, -1], [-1, -1, -1]],
                        [[-1, -1, -1], [-1, -1, -1]],
                    ]
                ],
            ]
        )
    ).all()

    x_tiled_tiled.dtype.dtype = x_tiled_tiled.dtype.dtype.flatten()

    if subs_before_eval:
        x_tiled_tiled_substituted = x_tiled_tiled.subs(subs)
        x_tiled_tiled_evaluated = x_tiled_tiled_substituted.eval()
    else:
        x_tiled_tiled_evaluated = x_tiled_tiled.eval(subs)

    assert (
        x_tiled_tiled_evaluated
        == np.array(
            [
                [
                    [
                        [0, 1, 2, 6, 7, 8],
                        [3, 4, 5, 9, 10, 11],
                        [-1, -1, -1, -1, -1, -1],
                        [12, 13, 14, 18, 19, 20],
                        [15, 16, 17, 21, 22, 23],
                        [-1, -1, -1, -1, -1, -1],
                    ]
                ],
                [
                    [
                        [24, 25, 26, -1, -1, -1],
                        [27, 28, 29, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                    ]
                ],
            ]
        )
    ).all()
