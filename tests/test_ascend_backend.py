import ast
import json
from types import SimpleNamespace

import pytest

from ninetoothed.ascendifier import Ascendifier
from ninetoothed.backends import Target, default_registry, emit
from ninetoothed.backends.materializers.ascend_aot import _compile_signature
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec, ir_to_dict
from ninetoothed.targets import (
    resolve_target_context,
    runtime_device_types,
    validate_artifact_materialization,
)
from tests.test_backend_registry import _add_kernel


def _exp_kernel():
    source = "\ndef exp_input(x, out):\n    out = ntl.exp(x)\n"
    tensors = (
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
    )

    return Kernel(
        kernel_name="exp_input",
        source=source,
        source_language="ninetoothed-python",
        entrypoint="exp_input",
        tensors=tensors,
        ssa=from_source(source, tensors),
    )


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


def test_ascend_exponential_helper_avoids_boolean_dtype_operations():
    context = resolve_target_context("ascend", platform="ascend-910b4")
    artifact = emit(_exp_kernel(), "ascend", target_context=context)
    tree = ast.parse(artifact.primary_source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_nt_exp_input"
    )

    assert (
        len(tuple(node for node in ast.walk(helper) if isinstance(node, ast.Return)))
        == 1
    )
    assert not tuple(node for node in ast.walk(helper) if isinstance(node, ast.BoolOp))
    assert "_nt_exp_input(" in artifact.primary_source


def test_ascend_aot_lowers_float_scalar_signatures():
    def triton_signature(_compilation):
        return "*fp64,fp64,i64"

    assert _compile_signature(None, triton_signature) == "*fp64,fp32,i64"


@pytest.mark.parametrize("namespace", ("tl", "triton.language"))
def test_ascendifier_transforms_generated_triton_syntax(namespace):
    source = f"""
from triton.language.extra import libdevice
x = {namespace}.load(ptr, other=None)
y = {namespace}.clamp(x, 0, 1).to({namespace}.float32)
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


def test_ascendifier_rejects_float64():
    source = "x = tl.float64"

    with pytest.raises(ValueError, match="float64"):
        Ascendifier().transform(source)


def test_ascendifier_omits_only_provable_static_bounds():
    source = "x = tl.load(ptr, mask=tl.arange(0, 64) < 64, other=0.0)"

    result = Ascendifier(omit_static_bounds=True).transform(source)

    assert "mask=" not in result
    assert "other=" not in result


def test_ascendifier_preserves_static_tail_bounds():
    source = "x = tl.load(ptr, mask=tl.arange(0, 64) < 1, other=0.0)"

    result = Ascendifier(omit_static_bounds=True).transform(source)

    assert "mask=tl.arange(0, 64) < 1" in result
    assert "other=0.0" in result


def test_ascendifier_rewrites_boolean_operators():
    source = "x = left or other\ny = left and other\n"

    result = Ascendifier().transform(source)

    assert "left | other" in result
    assert "left & other" in result
    assert " or " not in result
    assert " and " not in result


def test_ascendifier_substitutes_runtime_scalar_bounds():
    source = "x = tl.load(ptr, mask=index < size, other=0.0)"

    result = Ascendifier(
        omit_static_bounds=True,
        scalar_values={"index": 3, "size": 4},
    ).transform(source)

    assert "mask=" not in result
    assert "other=" not in result


def test_ascend_attention_selects_validated_tile_candidate():
    from ninetoothed import Tensor
    from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
    from tests.test_attention import application, arrangement

    shape = (
        None,
        None,
        {"constexpr": True},
        {"constexpr": True, "upper_bound": 128},
    )
    tensors = (
        Tensor(4, shape_options=shape),
        Tensor(4, shape_options=shape),
        Tensor(4, shape_options=shape),
        Tensor(0, constexpr=True),
        Tensor(4, shape_options=shape),
    )
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="ascend",
            platform="ascend-910b4",
        )
    )
    candidate = compilation.launch_plan.tuning_candidates[0]

    assert candidate["meta_parameters"] == {
        "ninetoothed_meta_prefix_ninetoothed_BLOCK_SIZE_0": 64,
        "ninetoothed_meta_prefix_ninetoothed_BLOCK_SIZE_1": 32,
    }
