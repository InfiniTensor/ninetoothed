import ast
import json
from types import SimpleNamespace

import pytest

from ninetoothed.ascendifier import Ascendifier
from ninetoothed.backends import Target, default_registry, emit
from ninetoothed.ir import ir_to_dict
from ninetoothed.targets import (
    resolve_target_context,
    runtime_device_types,
    validate_artifact_materialization,
)
from tests.test_backend_registry import _add_kernel


@pytest.mark.parametrize("platform", ("ascend-910b3", "ascend-910b4"))
def test_ascend_emits_python_with_platform_and_manifest(platform):
    context = resolve_target_context("ascend", platform=platform)
    assert runtime_device_types(SimpleNamespace(target=context)) == ("npu",)
    assert default_registry().get("ascend").capability.name == Target.ASCEND
    artifact = emit(_add_kernel(), "ascend", target_context=context)

    assert artifact.backend == Target.ASCEND
    assert artifact.language == "python/triton"
    assert artifact.primary_source_name.endswith(".ascend_triton.py")
    compile(artifact.primary_source, artifact.primary_source_name, "exec")
    tree = ast.parse(artifact.primary_source)
    assert artifact.entrypoint in {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "tl.store(" in artifact.primary_source
    assert artifact.metadata["target"]["platform"] == platform
    assert artifact.metadata["ssa_metadata"]["target_backend"] == "ascend"
    assert "ssa.ascend.optimize_schedule" in artifact.metadata["ssa_pass_trace"]
    manifest = json.loads(artifact.sources["add.ascend.json"])
    assert manifest["target"] == ir_to_dict(artifact.metadata["target"])
    validate_artifact_materialization(artifact, mode="jit")
    assert runtime_device_types(artifact) == ("npu",)
    validate_artifact_materialization(artifact, mode="aot")


@pytest.mark.parametrize("namespace", ("tl", "triton.language"))
def test_ascendifier_transforms_generated_triton_syntax(namespace):
    source = f"""
from triton.language.extra import libdevice
x = {namespace}.load(ptr, other=None)
y = {namespace}.clamp(x, 0, 1).to({namespace}.float64)
"""
    result = Ascendifier().transform(source)
    compile(result, "ascend.py", "exec")
    assert "from triton.language.extra.ascend import libdevice" in result
    assert "other=0.0" in result
    assert f"{namespace}.minimum({namespace}.maximum(x, 0), 1)" in result
    assert f"{namespace}.float32" in result


def test_ascendifier_preserves_autotune_configs():
    source = """
@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 512})], key=['x_size'])
def kernel(x):
    pass
"""
    result = Ascendifier().transform(source)
    assert "triton.Config({'BLOCK_SIZE': 512})" in result
    compile(result, "ascend.py", "exec")
