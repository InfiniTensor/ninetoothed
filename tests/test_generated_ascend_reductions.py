import ast
import inspect
from pathlib import Path

import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.generation import CodeGenerator


def _sum(input, output):
    output = ntl.sum(input, axis=1)  # noqa: F841


def _min(input, output):
    output = ntl.min(input, axis=1)  # noqa: F841


def _max(input, output):
    output = ntl.max(input, axis=1)  # noqa: F841


def _generate(application, width, tmp_path, monkeypatch):
    tensors = (Tensor(shape=(None, width)), Tensor(1))
    application.__annotations__ = dict(
        zip(
            inspect.signature(application).parameters,
            (tensors[0].tile((1, -1)), tensors[1].tile((1,))),
        )
    )
    monkeypatch.setattr("ninetoothed.generation.CACHE_DIR", tmp_path)
    source = CodeGenerator()(
        application,
        caller="torch",
        kernel_name="generated_row_reduction",
        num_warps=8,
        num_stages=1,
        max_num_configs=None,
        prettify=False,
    )
    tree = ast.parse(Path(source).read_text())
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    return tree, functions


@pytest.mark.parametrize("application", (_sum, _min, _max))
@pytest.mark.parametrize("width", (None, 65536))
def test_generated_large_rows_use_ascend_chunk_loop(
    application, width, tmp_path, monkeypatch
):
    tree, functions = _generate(application, width, tmp_path, monkeypatch)
    npu = functions["generated_row_reduction_npu"]
    cuda = functions["generated_row_reduction"]

    assert any(isinstance(node, ast.For) for node in ast.walk(npu))
    assert not any(isinstance(node, ast.For) for node in ast.walk(cuda))

    if width is None:
        conditional = next(node for node in npu.body if isinstance(node, ast.If))
        assert isinstance(conditional.test, ast.Compare)
        assert isinstance(conditional.test.ops[0], ast.Gt)
        assert conditional.test.comparators[0].value == 32768
        assert conditional.orelse

    chunks = [
        node
        for node in ast.walk(npu)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "arange"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 2048
    ]

    assert chunks
    compile(tree, "<generated-ascend-reduction>", "exec")


def test_generated_small_row_stays_unchanged(tmp_path, monkeypatch):
    _, functions = _generate(_sum, 1024, tmp_path, monkeypatch)

    assert not any(
        isinstance(node, ast.For)
        for node in ast.walk(functions["generated_row_reduction_npu"])
    )
