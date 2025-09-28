import functools
import itertools

import torch

import tests.test_addmm as addmm
from ninetoothed import Tensor
from ninetoothed.debugging import simulate_arrangement
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_addmm():
    m = 3
    n = 4
    k = 5

    block_size_m = 2
    block_size_n = 2
    block_size_k = 2

    arrangement = functools.partial(
        addmm.arrangement,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    )
    tensors = (
        Tensor(shape=(m, n)),
        Tensor(shape=(m, k)),
        Tensor(shape=(k, n)),
        Tensor(0),
        Tensor(0),
        Tensor(shape=(m, n)),
    )

    reference_source_tensors = (
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ]
        ),
        torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
            ]
        ),
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19],
            ]
        ),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ]
        ),
    )

    reference_target_tensors = (
        torch.tensor(
            [
                [
                    [0, 1],
                    [4, 5],
                ],
                [
                    [2, 3],
                    [6, 7],
                ],
                [
                    [8, 9],
                    [-1, -1],
                ],
                [
                    [10, 11],
                    [-1, -1],
                ],
            ]
        ),
        torch.tensor(
            [
                [
                    [
                        [0, 1],
                        [5, 6],
                    ],
                    [
                        [2, 3],
                        [7, 8],
                    ],
                    [
                        [4, -1],
                        [9, -1],
                    ],
                ],
                [
                    [
                        [0, 1],
                        [5, 6],
                    ],
                    [
                        [2, 3],
                        [7, 8],
                    ],
                    [
                        [4, -1],
                        [9, -1],
                    ],
                ],
                [
                    [
                        [10, 11],
                        [-1, -1],
                    ],
                    [
                        [12, 13],
                        [-1, -1],
                    ],
                    [
                        [14, -1],
                        [-1, -1],
                    ],
                ],
                [
                    [
                        [10, 11],
                        [-1, -1],
                    ],
                    [
                        [12, 13],
                        [-1, -1],
                    ],
                    [
                        [14, -1],
                        [-1, -1],
                    ],
                ],
            ]
        ),
        torch.tensor(
            [
                [
                    [
                        [0, 1],
                        [4, 5],
                    ],
                    [
                        [8, 9],
                        [12, 13],
                    ],
                    [
                        [16, 17],
                        [-1, -1],
                    ],
                ],
                [
                    [
                        [2, 3],
                        [6, 7],
                    ],
                    [
                        [10, 11],
                        [14, 15],
                    ],
                    [
                        [18, 19],
                        [-1, -1],
                    ],
                ],
                [
                    [
                        [0, 1],
                        [4, 5],
                    ],
                    [
                        [8, 9],
                        [12, 13],
                    ],
                    [
                        [16, 17],
                        [-1, -1],
                    ],
                ],
                [
                    [
                        [2, 3],
                        [6, 7],
                    ],
                    [
                        [10, 11],
                        [14, 15],
                    ],
                    [
                        [18, 19],
                        [-1, -1],
                    ],
                ],
            ]
        ),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(
            [
                [
                    [0, 1],
                    [4, 5],
                ],
                [
                    [2, 3],
                    [6, 7],
                ],
                [
                    [8, 9],
                    [-1, -1],
                ],
                [
                    [10, 11],
                    [-1, -1],
                ],
            ]
        ),
    )

    source_tensors, target_tensors = simulate_arrangement(arrangement, tensors)

    for tensor, reference_tensor in itertools.chain(
        zip(source_tensors, reference_source_tensors),
        zip(target_tensors, reference_target_tensors),
    ):
        assert torch.equal(tensor.to(reference_tensor.device), reference_tensor)

    for tensor, reference_tensor in zip(
        (arranged.eval() for arranged in arrangement(*tensors)),
        reference_target_tensors,
    ):
        assert torch.equal(torch.from_numpy(tensor), reference_tensor)
