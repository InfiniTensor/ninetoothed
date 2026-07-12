import os
import subprocess
import sys

import pytest

import ninetoothed
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, Compiler, CompileRequest


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


def test_public_entrypoints_are_functions_backed_by_default_compiler():
    assert isinstance(DEFAULT_COMPILER, Compiler)
    assert callable(ninetoothed.aot)
    assert callable(ninetoothed.jit)
    assert callable(ninetoothed.lower)
    assert callable(ninetoothed.make)
    assert not hasattr(ninetoothed, "load_built_artifact")


def test_package_import_does_not_create_compiler_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    env = dict(os.environ, NINETOOTHED_CACHE_DIR=str(cache_dir))
    subprocess.run(
        [sys.executable, "-c", "import ninetoothed"],
        check=True,
        env=env,
    )
    assert not cache_dir.exists()


def test_triton_launch_plan_contains_limited_runtime_variants():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
            num_warps=(4, 8),
            num_stages=(2, 3),
            max_num_configs=3,
        )
    )

    assert compilation.launch_plan.tuning_candidates == (
        {"id": "warps-4_stages-2", "num_warps": 4, "num_stages": 2},
        {"id": "warps-4_stages-3", "num_warps": 4, "num_stages": 3},
        {"id": "warps-8_stages-2", "num_warps": 8, "num_stages": 2},
    )


@pytest.mark.parametrize("backend", ("cuda", "tilelang", "tvm"))
@pytest.mark.parametrize(
    "options",
    (
        {"num_warps": (4, 8)},
        {"num_stages": (2, 3)},
        {"max_num_configs": 2},
    ),
)
def test_non_triton_backends_reject_unsupported_autotuning(backend, options):
    with pytest.raises(NotImplementedError, match="autotuning is not supported"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=(Tensor(1), Tensor(1), Tensor(1)),
                backend=backend,
                **options,
            )
        )
