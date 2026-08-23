import pytest

from ninetoothed import Tensor
from ninetoothed.compiler import lower
from ninetoothed.compiler.runtime import _validate_tensor_contract
from ninetoothed.frontend.layout import tensor_spec


def _arrangement(input, output):
    return input.tile((1, 128)), output.tile((1, 128))


def _application(input, output):
    output = input  # noqa: F841


def test_static_strides_are_removed_from_triton_launch_abi():
    artifact = lower(
        _arrangement,
        _application,
        (
            Tensor(shape=(64, 128), strides=(128, 1), dtype="float16"),
            Tensor(shape=(64, 128), strides=(128, 1), dtype="float16"),
        ),
        backend="triton",
        kernel_name="static_strides",
    )

    assert "stride_0" not in artifact.primary_source
    assert "stride_1" not in artifact.primary_source


class _FakeTensor:
    shape = (64, 128)
    dtype = "float16"
    device = "cuda:0"

    def stride(self):
        return (129, 1)


def test_runtime_rejects_a_layout_that_breaks_static_strides():
    spec = tensor_spec(
        "input",
        Tensor(shape=(64, 128), strides=(128, 1), dtype="float16"),
    )

    with pytest.raises(TypeError, match="expected stride 0 to be 128"):
        _validate_tensor_contract(spec, _FakeTensor(), None)
