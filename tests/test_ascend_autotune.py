import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from ninetoothed import Tensor, block_size
from ninetoothed.ascend_autotune import make_grid_pruner
from ninetoothed.generation import CodeGenerator


def test_prune_oversized_grid_before_launch(monkeypatch):
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "0")
    configs = [SimpleNamespace(kwargs={"BLOCK": block}) for block in (32, 64, 128)]
    prune = make_grid_pruner(
        lambda meta: ((meta["size"] + meta["BLOCK"] - 1) // meta["BLOCK"],)
    )

    assert prune(configs, {"size": 3670016}) == configs[1:]
    assert prune(configs, {"size": 1024}) == configs


def test_direct_launch_boundary_and_empty_candidates(monkeypatch):
    monkeypatch.delenv("TRITON_ALL_BLOCKS_PARALLEL", raising=False)
    configs = [SimpleNamespace(kwargs={})]
    prune = make_grid_pruner(lambda meta: (meta["rows"], meta["columns"]))

    assert prune(configs, {"rows": 65535, "columns": 1}) == configs

    with pytest.raises(ValueError, match="65535 blocks"):
        prune(configs, {"rows": 65536, "columns": 1})


@pytest.mark.parametrize("setting", ("1", "true", "True"))
def test_parallel_block_mapping_retains_original_candidates(monkeypatch, setting):
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", setting)
    configs = [SimpleNamespace(kwargs={})]
    prune = make_grid_pruner(lambda meta: (1000000,))

    assert prune(configs, {}) is configs


def _copy(input, output):
    output = input  # noqa: F841


def test_generated_pruner_uses_real_grid_and_runtime_arguments(tmp_path, monkeypatch):
    block = block_size(lower_bound=32, upper_bound=128)
    input = Tensor(1, name="input")
    output = Tensor(1, name="output")
    _copy.__annotations__ = dict(
        zip(
            inspect.signature(_copy).parameters,
            (input.tile((block,)), output.tile((block,))),
        )
    )
    monkeypatch.setattr("ninetoothed.generation.CACHE_DIR", tmp_path)
    source = CodeGenerator()(_copy, "torch", "copy_grid", 8, 1, None, False)
    tree = ast.parse(Path(source).read_text())
    compile(tree, "<grid-pruner>", "exec")
    factories = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ninetoothed_make_grid_pruner"
    ]

    assert len(factories) == 1
    grid = eval(compile(ast.Expression(factories[0].args[0]), "<grid>", "eval"))
    config_name = str(block)
    size_name = input.size_string(0)

    assert grid({config_name: 32, size_name: 3670016}) == (114688,)
    assert grid({config_name: 64, size_name: 3670016}) == (57344,)
