import ast
import inspect
from pathlib import Path

import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size
from ninetoothed.generation import CodeGenerator


def _sum_rows(input, output):
    output = ntl.sum(input, axis=1)  # noqa: F841


def _generate_row_sum(tmp_path, monkeypatch, *, rows=1, width_options=None):
    input = Tensor(2, shape_options=(None, width_options))
    output = Tensor(1)
    arranged = (input.tile((rows, -1)), output.tile((rows,)))
    _sum_rows.__annotations__ = dict(
        zip(inspect.signature(_sum_rows).parameters, arranged)
    )
    monkeypatch.setattr("ninetoothed.generation.CACHE_DIR", tmp_path)
    generator = CodeGenerator()
    generator._max_num_elements = 32768
    source = generator(
        _sum_rows,
        caller="torch",
        kernel_name="dynamic_tile_rows",
        num_warps=4,
        num_stages=1,
        max_num_configs=None,
        prettify=False,
    )

    return ast.parse(Path(source).read_text())


def _block_sizes(tree):
    return {
        ast.literal_eval(node.args[0]).popitem()[1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Config"
        and node.args
        and isinstance(node.args[0], ast.Dict)
    }


def test_unbounded_full_row_generates_runtime_specialization(tmp_path, monkeypatch):
    tree = _generate_row_sum(tmp_path, monkeypatch)
    source = ast.unparse(tree)

    assert "triton.next_power_of_2" in source
    assert "triton.language.sum" in source
    assert "autotune" not in source
    extents = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.arg) and "next_power_of_2" in node.arg
    ]

    assert extents
    assert all(
        isinstance(node.annotation, ast.Attribute)
        and node.annotation.attr == "constexpr"
        for node in extents
    )
    compile(tree, "<dynamic-row-sum>", "exec")


def test_unbounded_width_retains_row_block_candidates(tmp_path, monkeypatch):
    tree = _generate_row_sum(
        tmp_path, monkeypatch, rows=block_size(lower_bound=1, upper_bound=4)
    )

    assert _block_sizes(tree) == {1, 2, 4}


def test_explicit_width_bound_still_prunes_row_block_candidates(tmp_path, monkeypatch):
    tree = _generate_row_sum(
        tmp_path,
        monkeypatch,
        rows=block_size(lower_bound=1, upper_bound=8),
        width_options={"upper_bound": 8192},
    )

    assert _block_sizes(tree) == {1, 2, 4}


def test_impossible_bounded_tile_keeps_the_configuration_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Failed to generate auto-tuning"):
        _generate_row_sum(
            tmp_path,
            monkeypatch,
            rows=block_size(lower_bound=2, upper_bound=4),
            width_options={"upper_bound": 32768},
        )
