from types import SimpleNamespace

import pytest

from ninetoothed.backends.materializers.cpp import supports_cpp_export
from ninetoothed.ir import LaunchABI, LaunchBinding, TensorSpec


@pytest.mark.parametrize(
    "key,jagged_dim,expected",
    (
        (("fp32", 64), None, True),
        (("int64", None), None, True),
        (("fast", 64), None, False),
        (("fp32", 64), 0, False),
        ((float("inf"),), None, False),
    ),
)
def test_cpp_export_support_preserves_python_only_builds(key, jagged_dim, expected):
    compilation = SimpleNamespace(
        artifact=SimpleNamespace(kernel_name="support_probe"),
        launch_abi=LaunchABI(
            public_args=("output",),
            kernel_args=(LaunchBinding(name="output", kind="tensor", source="output"),),
            outputs=("output",),
        ),
        kernel=SimpleNamespace(
            tensors=(
                TensorSpec(
                    name="output",
                    ndim=1,
                    shape=("size",),
                    dtype="float32",
                    jagged_dim=jagged_dim,
                ),
            ),
        ),
    )

    variants = ((key, compilation, "launch_support_probe_variant"),)

    assert supports_cpp_export(variants) is expected
