import ast
import re
from pathlib import Path

import pytest
import torch

import ninetoothed
import ninetoothed.aot as aot
import ninetoothed.naming as naming
from ninetoothed import Tensor
from ninetoothed.generation import CodeGenerator

TILE = 256


def arrangement(input, other, output):
    return tuple(tensor.tile((TILE,)) for tensor in (input, other, output))


def application(input, other, output):
    output = input + other  # noqa: F841


def arrangement_2d(input, other, output):
    return tuple(tensor.tile((32, 64)) for tensor in (input, other, output))


def application_2d(input, other, output):
    output = input + other  # noqa: F841


@pytest.fixture(scope="module")
def full_tile_kernel(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("competition_full_tile")
    tensors = tuple(Tensor(1, dtype=ninetoothed.float16) for _ in range(3))

    return ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller="cuda",
        kernel_name="competition_full_tile_add",
        output_dir=output_dir,
    )


@pytest.fixture(scope="module")
def full_tile_kernel_2d(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("competition_full_tile_2d")
    tensors = tuple(Tensor(2, dtype=ninetoothed.float16) for _ in range(3))

    return ninetoothed.make(
        arrangement_2d,
        application_2d,
        tensors,
        caller="cuda",
        kernel_name="competition_full_tile_add_2d",
        output_dir=output_dir,
    )


@pytest.mark.parametrize("size", (TILE * 16, TILE * 4096))
def test_specialization_hit_correctness(full_tile_kernel, size):
    input = torch.randn(size, device="cuda", dtype=torch.float16)
    other = torch.randn_like(input)
    output = torch.empty_like(input)

    full_tile_kernel(input, other, output)

    torch.testing.assert_close(output, input + other)


def test_nondivisible_fallback_correctness(full_tile_kernel):
    size = TILE * 16 + 1
    input = torch.randn(size, device="cuda", dtype=torch.float16)
    other = torch.randn_like(input)
    output = torch.empty_like(input)

    full_tile_kernel(input, other, output)

    torch.testing.assert_close(output, input + other)


def test_empty_fallback_correctness(full_tile_kernel):
    input = torch.empty(0, device="cuda", dtype=torch.float16)
    other = torch.empty_like(input)
    output = torch.empty_like(input)

    full_tile_kernel(input, other, output)
    torch.cuda.synchronize()

    assert output.numel() == 0


def test_noncontiguous_fallback_correctness(full_tile_kernel):
    size = TILE * 16
    input = torch.randn(size * 2, device="cuda", dtype=torch.float16)[::2]
    other = torch.randn(size, device="cuda", dtype=torch.float16)
    output = torch.empty(size, device="cuda", dtype=torch.float16)

    full_tile_kernel(input, other, output)

    torch.testing.assert_close(output, input + other)


def test_different_shape_fallback_correctness(full_tile_kernel):
    size = TILE * 16
    input = torch.randn(size, device="cuda", dtype=torch.float16)
    other = torch.randn(size * 2, device="cuda", dtype=torch.float16)
    output = torch.empty_like(input)

    full_tile_kernel(input, other, output)

    torch.testing.assert_close(output, input + other[:size])


def test_two_dimensional_specialization_hit_correctness(full_tile_kernel_2d):
    shape = (32 * 16, 64 * 16)
    input = torch.randn(shape, device="cuda", dtype=torch.float16)
    other = torch.randn_like(input)
    output = torch.empty_like(input)

    full_tile_kernel_2d(input, other, output)

    torch.testing.assert_close(output, input + other)


def test_two_dimensional_fallback_correctness(full_tile_kernel_2d):
    shape = (32 * 16 + 1, 64 * 16)
    input = torch.randn(shape, device="cuda", dtype=torch.float16)
    other = torch.randn_like(input)
    output = torch.empty_like(input)

    full_tile_kernel_2d(input, other, output)

    torch.testing.assert_close(output, input + other)


@pytest.mark.parametrize("shape", ((0, 64), (32, 0), (0, 0)))
def test_two_dimensional_empty_fallback_correctness(full_tile_kernel_2d, shape):
    input = torch.empty(shape, device="cuda", dtype=torch.float16)
    other = torch.empty_like(input)
    output = torch.empty_like(input)

    full_tile_kernel_2d(input, other, output)
    torch.cuda.synchronize()

    assert output.numel() == 0


def _generated_kernel_source():
    tensors = tuple(Tensor(1, dtype=ninetoothed.float16) for _ in range(3))
    arranged = arrangement(*tensors)
    application.__annotations__ = dict(zip(("input", "other", "output"), arranged))
    generator = CodeGenerator()
    source_file = generator(
        application,
        caller="cuda",
        kernel_name="competition_full_tile_source",
        num_warps=8,
        num_stages=3,
        max_num_configs=None,
        prettify=False,
    )

    return generator, source_file


def test_generated_source_removes_all_boundary_masks():
    generator, source_file = _generated_kernel_source()
    spec = aot._full_tile_spec(generator.kernel_func)
    specialized_file = aot._make_full_tile_source(source_file, spec)
    specialized = Path(specialized_file).read_text()

    assert len(spec) == 3
    assert "mask=" not in specialized


def test_generated_source_removes_two_dimensional_boundary_masks():
    tensors = tuple(Tensor(2, dtype=ninetoothed.float16) for _ in range(3))
    arranged = arrangement_2d(*tensors)
    application_2d.__annotations__ = dict(zip(("input", "other", "output"), arranged))
    generator = CodeGenerator()
    source_file = generator(
        application_2d,
        caller="cuda",
        kernel_name="competition_full_tile_source_2d",
        num_warps=8,
        num_stages=3,
        max_num_configs=None,
        prettify=False,
    )
    spec = aot._full_tile_spec(generator.kernel_func)
    launch_names = tuple(arg.arg for arg in generator.launch_func.args.args)

    assert len(spec) == 6
    assert aot._is_safe_full_tile_spec(
        spec,
        launch_names,
        generator.tensors,
        lambda values, name: next(
            tensor for tensor in values if tensor.source.name.endswith(name)
        ),
    )
    specialized_file = aot._make_full_tile_source(source_file, spec)
    specialized = Path(specialized_file).read_text()
    assert "mask=" not in specialized
    tree = ast.parse(specialized)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "competition_full_tile_source_2d"
    )
    used_size_sources = {
        naming.remove_prefixes(match.group(1))
        for node in ast.walk(kernel)
        if isinstance(node, ast.Name)
        if (match := Tensor.size_pattern().fullmatch(node.id)) is not None
    }
    assert len(used_size_sources) == 1


def test_dispatcher_has_strict_hit_and_fallback_guards():
    generator, _ = _generated_kernel_source()
    launch_names = tuple(arg.arg for arg in generator.launch_func.args.args)
    spec = aot._full_tile_spec(generator.kernel_func)
    variants = [
        (
            "fast_full_tile",
            tuple((name, 0) for name in launch_names),
            tuple((name, 0) for name in launch_names),
            ninetoothed.int32,
            ninetoothed.int32,
            spec,
        ),
        (
            "ordinary_divisible",
            tuple((name, 0) for name in launch_names),
            tuple((name, 0) for name in launch_names),
            ninetoothed.int32,
            ninetoothed.int32,
            (),
        ),
        (
            "fallback",
            (),
            (),
            ninetoothed.int64,
            ninetoothed.int64,
            (),
        ),
    ]
    source, _ = aot._generate_dispatcher(
        "competition_full_tile_source", launch_names, variants, (1, 1, 1)
    )

    assert source.count("% 256 == 0") == 1
    assert source.count("% 16 == 0") == 3
    assert source.count(".strides[0] == 1") == 6
    assert source.count(".shape[0] > 0") == 1
    assert source.count(".shape[0] ==") == 2
    assert re.search(r"if \(.*full_tile", source)
    assert "launch_competition_full_tile_source_fallback" in source


def test_non_multiple_of_16_tile_keeps_aot_divisibility_guard():
    variants = [
        (
            "fast_full_tile",
            (("input", 0),),
            (("input", 0),),
            ninetoothed.int32,
            ninetoothed.int32,
            (("input", 0, 10),),
        ),
        (
            "fallback",
            (),
            (),
            ninetoothed.int64,
            ninetoothed.int64,
            (),
        ),
    ]
    source, _ = aot._generate_dispatcher(
        "competition_full_tile_non_aligned", ("input",), variants, (1,)
    )

    assert "input.shape[0] % 16 == 0" in source
    assert "input.shape[0] % 10 == 0" in source
    assert "input.shape[0] > 0" in source


def test_non_program_multiplier_is_not_a_canonical_tile():
    expression = ast.parse("7 * 256 + triton.language.arange(0, 256)", mode="eval").body

    assert aot._canonical_tile_size(expression) is None


def test_unrelated_program_bound_is_not_removed(tmp_path):
    source = tmp_path / "unrelated_mask.py"
    source.write_text(
        """
import triton
import triton.language

@triton.jit
def unrelated_mask(pointer, unrelated_index_0, limit):
    triton.language.load(
        pointer,
        mask=unrelated_index_0 < limit,
        other=None,
    )
""".lstrip()
    )

    specialized = Path(
        aot._make_full_tile_source(source, (("input", 0, TILE),))
    ).read_text()

    assert "mask=" in specialized


def test_incomplete_high_rank_spec_is_not_eligible_for_mask_free_variant():
    tensors = tuple(Tensor(2) for _ in range(3))
    launch_names = ("input", "other", "output")

    assert not aot._is_safe_full_tile_spec(
        (("input", 1, 64), ("other", 1, 64), ("output", 1, 64)),
        launch_names,
        tensors,
        lambda values, name: values[launch_names.index(name)],
    )


def test_expanded_layout_is_not_eligible_for_mask_free_variant():
    source = Tensor(1)
    expanded = source.tile((TILE,)).expand((2, -1))

    assert not aot._is_safe_full_tile_spec(
        ((source.name, 0, TILE),),
        (source.name,),
        (expanded,),
        lambda values, name: values[0],
    )
