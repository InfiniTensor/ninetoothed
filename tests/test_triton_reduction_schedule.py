import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, float32
from ninetoothed.compiler import (
    DEFAULT_COMPILER,
    CompileRequest,
    load_built_artifact,
    make,
)
from ninetoothed.compiler.reductions import analyze_reductions
from ninetoothed.ir import ssa
from tests.utils import get_available_devices

WIDTH = Symbol("WIDTH", constexpr=True)
HEIGHT = Symbol("HEIGHT", constexpr=True)
TOO_WIDE = (1 << 20) + 1


def _row_arrangement(x, out, WIDTH=WIDTH):
    return x.tile((1, WIDTH)), out.tile((1, WIDTH))


def _row_reduced_arrangement(x, out, WIDTH=WIDTH):
    return x.tile((1, WIDTH)), out.tile((1,))


def _two_input_reduced_arrangement(x, y, out, WIDTH=WIDTH):
    return x.tile((1, WIDTH)), y.tile((1, WIDTH)), out.tile((1,))


def _fixed_row_reduced_arrangement(x, out):
    return x.tile((1, 127)), out.tile((1,))


def _column_arrangement(x, out, HEIGHT=HEIGHT):
    return x.tile((HEIGHT, 1)), out.tile((HEIGHT, 1))


def _middle_axis_arrangement(x, out):
    return x.tile((2, 3, 5)), out.tile((2, 5))


def _square_arrangement(x, out):
    return x.tile((4, 4)), out.tile((4, 4))


def _separate_reduction_arrangement(x, rows, columns):
    return x.tile((4, 8)), rows.tile((4,)), columns.tile((8,))


def _mixed_outer_arrangement(x, y, out_x, out_y):
    return (
        x.tile((1, 7)),
        y.tile((1, 7)),
        out_x.tile((1,)),
        out_y.tile((1,)),
    )


def _too_wide_arrangement(x, out):
    return x.tile((1, TOO_WIDE)), out.tile((1,))


def _row_normalize(x, out):
    maximum = ntl.max(x, axis=1)
    numerator = ntl.exp(x - maximum[:, None])
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def _row_layernorm(x, out):
    width = x.shape[1]
    mean = ntl.sum(x, axis=1) / width
    mean_square = ntl.sum(x * x, axis=1) / width
    variance = mean_square - mean * mean
    out = (x - mean[:, None]) * ntl.rsqrt(variance[:, None] + 1e-5)  # noqa: F841


def _row_min(x, out):
    out = ntl.min(x, axis=1)  # noqa: F841


def _row_product_sum(x, y, out):
    out = ntl.sum(x * y, axis=1)  # noqa: F841


def _column_max_broadcast(x, out):
    out = x + ntl.max(x, axis=0)[None, :]  # noqa: F841


def _middle_axis_sum(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def _incompatible_broadcast(x, out):
    out = x + ntl.sum(x, axis=1)[None, :]  # noqa: F841


def _separate_reduction_outputs(x, rows, columns):
    rows = ntl.sum(x, axis=1)  # noqa: F841
    columns = ntl.max(x, axis=0)  # noqa: F841


def _mixed_outer_reductions(x, y, out_x, out_y):
    out_x = ntl.sum(x, axis=1)  # noqa: F841
    out_y = ntl.sum(y, axis=1)  # noqa: F841


def _too_wide_sum(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def _mixed_tensor_scalar_reductions(x, out):
    row = ntl.sum(x, axis=1)
    scalar = ntl.sum(x[0, 0], axis=0)
    out = row + scalar  # noqa: F841


def _source_store_sum(x, out):
    out.source[out.offsets(0)] = ntl.sum(x, axis=1)


def _request():
    return CompileRequest(
        arrangement=_row_arrangement,
        application=_row_normalize,
        tensors=(Tensor(2), Tensor(2)),
        backend="triton",
        tensor_dtypes={"x": "float32", "out": "float32"},
    )


def test_reduction_domain_selects_triton_row_vector_schedule():
    compilation = DEFAULT_COMPILER.compile(_request())
    metadata = compilation.artifact.metadata["ssa_metadata"]
    domains = metadata["analysis"]["reduction_domains"]

    assert len(domains) == 2
    assert {domain["operator"] for domain in domains} == {"max", "sum"}
    assert all(domain["axis"] == 1 for domain in domains)
    assert all(domain["parallel_shape"] == ("1",) for domain in domains)
    assert all(len(domain["program_shapes"]) == 1 for domain in domains)
    assert metadata["schedule"]["reduction"]["mode"] == "row-vector"
    assert metadata["schedule"]["reduction"]["program_shape"]
    assert tuple(
        candidate["num_warps"]
        for candidate in compilation.launch_plan.tuning_candidates
    ) == (4, 8, 1)
    assert "tl.max(" in compilation.artifact.primary_source
    assert "tl.sum(" in compilation.artifact.primary_source
    assert "for v" not in compilation.artifact.primary_source

    mixed = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_row_reduced_arrangement,
            application=_mixed_tensor_scalar_reductions,
            tensors=(Tensor(2), Tensor(1)),
            backend="triton",
            max_num_configs=1,
        )
    )
    assert mixed.artifact.primary_source.count("tl.sum(") == 1

    source_store = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_row_reduced_arrangement,
            application=_source_store_sum,
            tensors=(Tensor(2), Tensor(1)),
            backend="triton",
            max_num_configs=1,
        )
    )
    store = next(
        line
        for line in source_store.artifact.primary_source.splitlines()
        if "tl.store(out +" in line
    )
    assert "offsets" not in store

    fallback = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_square_arrangement,
            application=_incompatible_broadcast,
            tensors=(Tensor(2), Tensor(2)),
            backend="triton",
            tensor_dtypes={"x": "float32", "out": "float32"},
        )
    )
    assert (
        fallback.artifact.metadata["ssa_metadata"]["schedule"]["reduction"]["mode"]
        == "scalar-fallback"
    )

    with pytest.raises(ValueError, match="separate kernels"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_separate_reduction_arrangement,
                application=_separate_reduction_outputs,
                tensors=(Tensor(2), Tensor(1), Tensor(1)),
                backend="triton",
                tensor_dtypes={
                    "x": "float32",
                    "rows": "float32",
                    "columns": "float32",
                },
            )
        )

    with pytest.raises(ValueError, match="separate kernels"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_mixed_outer_arrangement,
                application=_mixed_outer_reductions,
                tensors=(
                    Tensor(shape=(2, 7)),
                    Tensor(shape=(4, 7)),
                    Tensor(shape=(2,)),
                    Tensor(shape=(4,)),
                ),
                backend="triton",
            )
        )

    with pytest.raises(ValueError, match="exceeding the backend tensor numel limit"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_too_wide_arrangement,
                application=_too_wide_sum,
                tensors=(Tensor(shape=(1, TOO_WIDE)), Tensor(shape=(1,))),
                backend="triton",
            )
        )

    scalar = ssa.Type(kind="scalar", dtype="float32")
    operand = ssa.Value(name="x", type=scalar)
    result = ssa.Value(name="result", type=scalar)
    unsupported = ssa.Program(
        kind="unsupported-reduction",
        inputs=(operand,),
        outputs=(result,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="reduce.product",
                        operands=(operand.name,),
                        results=(result,),
                        attrs={"axis": 0},
                    ),
                )
            ),
        ),
    )

    with pytest.raises(ValueError, match="Unsupported SSA reduction `reduce.product`"):
        analyze_reductions(unsupported)


@pytest.mark.parametrize("device", get_available_devices())
def test_triton_row_vector_reduction_runtime(device, tmp_path):
    normalize = make(
        _row_arrangement,
        _row_normalize,
        (Tensor(2), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )
    layernorm = make(
        _row_arrangement,
        _row_layernorm,
        (Tensor(2), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )
    reduce_min = make(
        _row_reduced_arrangement,
        _row_min,
        (Tensor(2), Tensor(1)),
        backend="triton",
        max_num_configs=1,
    )
    product_sum = make(
        _two_input_reduced_arrangement,
        _row_product_sum,
        (Tensor(2), Tensor(2), Tensor(1)),
        backend="triton",
        max_num_configs=1,
    )
    column_max = make(
        _column_arrangement,
        _column_max_broadcast,
        (Tensor(2), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )
    middle_sum = make(
        _middle_axis_arrangement,
        _middle_axis_sum,
        (Tensor(3), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )

    width = 127
    base = torch.randn((37, width * 2), device=device)
    x = base[:, ::2]
    normalized = torch.empty_like(x)
    layernorm_output = torch.empty_like(x)
    minimum = torch.empty((x.shape[0],), device=device)

    normalize(x, normalized, WIDTH=width)
    layernorm(x, layernorm_output, WIDTH=width)
    reduce_min(x, minimum, WIDTH=width)

    torch.testing.assert_close(
        normalized,
        torch.softmax(x, dim=1),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        layernorm_output,
        torch.nn.functional.layer_norm(x, (width,)),
        rtol=2e-4,
        atol=2e-5,
    )
    torch.testing.assert_close(minimum, x.min(dim=1).values)

    x = torch.randn((193, 41), device=device)
    output = torch.empty_like(x)
    column_max(x, output, HEIGHT=x.shape[0])
    torch.testing.assert_close(output, x + x.max(dim=0).values, rtol=1e-5, atol=1e-6)

    x = torch.randn((2, 3, 5), device=device)
    output = torch.empty((2, 5), device=device)
    middle_sum(x, output)
    torch.testing.assert_close(output, x.sum(dim=1))

    mismatched_output = torch.empty((4,), device=device)

    with pytest.raises(ValueError, match="program domains do not match"):
        reduce_min(x=torch.randn((2, 7), device=device), out=mismatched_output, WIDTH=7)

    with pytest.raises(ValueError, match="program domains do not match"):
        product_sum(
            torch.randn((2, 7), device=device),
            torch.randn((4, 7), device=device),
            torch.empty((2,), device=device),
            WIDTH=7,
        )

    too_wide = torch.empty((1, TOO_WIDE), device=device)
    reduced = torch.empty((1,), device=device)

    with pytest.raises(ValueError, match="tensor numel limit"):
        reduce_min(too_wide, reduced, WIDTH=TOO_WIDE)

    aot_reduce_min = make(
        _fixed_row_reduced_arrangement,
        _row_min,
        (
            Tensor(shape=(37, 127), dtype=float32),
            Tensor(shape=(37,), dtype=float32),
        ),
        backend="triton",
        caller=device,
        kernel_name="row_min_reduction_aot",
        output_dir=tmp_path,
        max_num_configs=1,
    )
    x = torch.randn((37, 127), device=device)
    output = torch.empty((37,), device=device)
    aot_reduce_min(x, output)
    torch.testing.assert_close(output, x.min(dim=1).values)

    resized_x = torch.randn((38, 127), device=device)
    resized_output = torch.empty((38,), device=device)

    for launch in (aot_reduce_min, load_built_artifact(aot_reduce_min._built_artifact)):
        with pytest.raises(TypeError, match="has shape .* expected"):
            launch(resized_x, resized_output)

    dynamic_tensors = (
        Tensor(shape=(None, 127), dtype=float32),
        Tensor(shape=(None,), dtype=float32),
    )
    dynamic_jit = make(
        _fixed_row_reduced_arrangement,
        _row_min,
        dynamic_tensors,
        backend="triton",
        max_num_configs=1,
    )
    dynamic_aot = make(
        _fixed_row_reduced_arrangement,
        _row_min,
        dynamic_tensors,
        backend="triton",
        caller=device,
        kernel_name="dynamic_row_min_reduction_aot",
        output_dir=tmp_path,
        max_num_configs=1,
    )
    invalid_x = torch.randn((37, 64), device=device)
    invalid_output = torch.empty((37,), device=device)

    for launch in (
        dynamic_jit,
        dynamic_aot,
        load_built_artifact(dynamic_aot._built_artifact),
    ):
        with pytest.raises(TypeError, match="expected dimension 1 to be 127"):
            launch(invalid_x, invalid_output)


@pytest.mark.parametrize("device", get_available_devices())
def test_row_vector_only_vectorizes_scheduled_reductions_and_masks_source_store(
    device,
):
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

    for application, expected in (
        (_mixed_tensor_scalar_reductions, torch.tensor([7.0, 19.0], device=device)),
        (_source_store_sum, torch.tensor([6.0, 15.0], device=device)),
    ):
        kernel = make(
            _row_reduced_arrangement,
            application,
            (Tensor(2), Tensor(1)),
            backend="triton",
            max_num_configs=1,
        )
        output = torch.empty(2, device=device)
        kernel(x, output, WIDTH=3)
        torch.testing.assert_close(output, expected)
