"""Identity reads remain independent, masked snapshots for every storage layout."""

from types import SimpleNamespace

import numpy as np
import pytest

from ninetoothed.interpreter.memory import TensorRef


def _ref(array, explicit):
    spec = None

    if explicit:
        spec = SimpleNamespace(
            layout=SimpleNamespace(
                levels=(), application_shape=array.shape, view_access=None
            ),
            attrs={"other": 3},
        )

    return TensorRef(array, spec, {})


@pytest.mark.parametrize(
    "dtype", (np.bool_, np.uint8, np.int32, np.float16, np.float32, ">f4")
)
@pytest.mark.parametrize("explicit", (False, True))
@pytest.mark.parametrize("broadcast", (False, True))
def test_identity_read_owns_result_and_preserves_broadcast_mask(
    dtype, explicit, broadcast
):
    source = np.arange(35).reshape(5, 7).astype(dtype)
    array = np.broadcast_to(source[:1], (5, 7)) if broadcast else source[::-1, ::-1]
    before = array.copy()
    mask = np.array([True, False, True, False, False])[:, None]
    result = _ref(array, explicit).read(mask, other=2)
    expected = np.where(mask, before, np.array(2, dtype=dtype)).astype(dtype)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == array.dtype
    assert result.flags.c_contiguous
    assert result.flags.writeable
    assert not np.shares_memory(result, source)
    result[...] = 0
    np.testing.assert_array_equal(array, before)
    source[...] = 1
    np.testing.assert_array_equal(result, np.zeros_like(result))
    np.testing.assert_array_equal(_ref(array, explicit).read(), array)


@pytest.mark.parametrize("explicit", (False, True))
def test_equal_shape_extraction_still_applies_its_permutation(explicit):
    source = np.arange(12, dtype=np.int32).reshape(3, 4)
    ref = _ref(source, explicit)
    np.testing.assert_array_equal(
        ref.extract(([2, 0, 1], slice(None))), source[[2, 0, 1]]
    )
    np.testing.assert_array_equal(
        ref.extract((slice(None), [3, 2, 1, 0])), source[:, ::-1]
    )


@pytest.mark.parametrize("explicit", (False, True))
def test_identity_copy_preserves_signed_zero_and_nan_payloads(explicit):
    words = np.array(
        [0, 0x80000000, 0x7FC00123, 0xFFC00321, 0x7F800000, 0xFF800000], dtype=np.uint32
    )
    source = words.view(np.float32)[::-1]
    result = _ref(source, explicit).read()
    np.testing.assert_array_equal(result.view(np.uint32), words[::-1])


def test_invalid_identity_shape_is_rejected_before_direct_copy():
    ref = _ref(np.arange(12, dtype=np.float32).reshape(3, 4), True)
    ref.spec.layout.application_shape = (4, 3)

    with pytest.raises(ValueError, match="no access map"):
        ref.read()


def test_invalid_mask_is_rejected_before_read():
    ref = _ref(np.arange(12, dtype=np.float32).reshape(3, 4), False)

    with pytest.raises(ValueError):
        ref.read(np.ones((2, 3), dtype=bool))


class _CopySensitiveArray(np.ndarray):
    def __array_function__(self, function, types, args, kwargs):
        if function is np.copyto:
            raise AssertionError("A new copy protocol was dispatched to the subclass.")

        return super().__array_function__(function, types, args, kwargs)


@pytest.mark.parametrize("explicit", (False, True))
def test_array_subclass_keeps_its_existing_indexed_read_protocol(explicit):
    source = np.arange(12, dtype=np.float32).reshape(3, 4).view(_CopySensitiveArray)
    actual = _ref(source, explicit).read()
    np.testing.assert_array_equal(actual, np.asarray(source))
    assert type(actual) is np.ndarray
