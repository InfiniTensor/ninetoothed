import numpy as np
import pytest

from ninetoothed import Tensor


@pytest.mark.parametrize(
    "key",
    [
        slice(None, None, None),
        slice(0, -1, -1),
        slice(0, -1, -1),
        slice(0, -1, 2),
        slice(1, None, 2),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (
            3,
            4,
            2,
        ),
        (
            2,
            3,
            4,
            5,
        ),
    ],
)
def test_slice(shape: tuple, key: slice):
    a = Tensor(shape=shape)
    a_ref = np.arange(0, np.prod(shape)).reshape(shape)

    b = a[key]
    b_ref = a_ref[key]

    assert np.allclose(b.eval(), b_ref)


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), 1, slice(None)),
        (1, slice(None), None),
        (slice(None, None, -1), 0, slice(1, -1)),
        (Ellipsis, 1),
        (None, slice(None), 1, slice(None)),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (
            3,
            4,
            2,
        )
    ],
)
def test_multi_dim_slice(shape: tuple, key):
    a = Tensor(shape=shape)
    a_ref = np.arange(0, np.prod(shape)).reshape(shape)

    b = a[key]
    b_ref = a_ref[key]

    assert np.allclose(b.eval(), b_ref)


def test_case_1():
    a = Tensor(shape=(3, 4, 2))
    a_ref = np.arange(0, np.prod(a.shape)).reshape(a.shape)

    b = a[:, 1, :]
    b_ref = a_ref[:, 1, :]

    assert np.allclose(b.eval(), b_ref)


def test_case_2():
    new_slice = (slice(2, 4, None), slice(1, 4, None))
    a = Tensor(shape=(6, 5))
    a_ref = np.arange(0, np.prod(a.shape)).reshape(a.shape)

    b = a[new_slice]
    b_ref = a_ref[new_slice]

    assert np.allclose(b.eval(), b_ref)
