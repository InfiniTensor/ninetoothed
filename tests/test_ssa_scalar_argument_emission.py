import ninetoothed
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest


def _arrangement(input, scale, output):
    return input.tile((128,)), scale, output.tile((128,))


def _scale(input, scale, output):
    output = input * scale  # noqa: F841


def test_triton_scalar_input_is_passed_by_value():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_scale,
            tensors=(
                Tensor(1, dtype=ninetoothed.float32),
                Tensor(0, dtype=ninetoothed.float64),
                Tensor(1, dtype=ninetoothed.float32),
            ),
            backend="triton",
            max_num_configs=1,
        )
    )
    source = compilation.artifact.primary_source

    assert "tl.load(scale + 0)" not in source
    assert "* scale" in source
