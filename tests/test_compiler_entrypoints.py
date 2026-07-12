import os
import subprocess
import sys

import ninetoothed
from ninetoothed.compiler import DEFAULT_COMPILER, Compiler


def test_public_entrypoints_are_functions_backed_by_default_compiler():
    assert isinstance(DEFAULT_COMPILER, Compiler)
    assert callable(ninetoothed.aot)
    assert callable(ninetoothed.jit)
    assert callable(ninetoothed.lower)
    assert callable(ninetoothed.make)


def test_package_import_does_not_create_compiler_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    env = dict(os.environ, NINETOOTHED_CACHE_DIR=str(cache_dir))
    subprocess.run(
        [sys.executable, "-c", "import ninetoothed"],
        check=True,
        env=env,
    )
    assert not cache_dir.exists()
