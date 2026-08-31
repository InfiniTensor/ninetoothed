import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import lower
from ninetoothed.frontend.layout import tensor_spec


def _matmul_arrangement(lhs, rhs, output):
    output_tiled = output.tile((16, 128))
    lhs_tiled = (
        lhs.tile((16, 128))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)
    rhs_tiled = (
        rhs.tile((128, 128))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def _matmul_application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])

    output = accumulator  # noqa: F841


def _singleton_broadcast_arrangement(input, output):
    output_tiled = output.tile((16, 128))
    input_tiled = input.tile((16, 128), dilation=(0, 1))
    input_tiled = input_tiled.tile((1, -1)).expand(
        (-1, output_tiled.shape[1])
    )
    input_tiled.dtype = input_tiled.dtype.squeeze(0)

    return input_tiled, output_tiled


def _singleton_broadcast_application(input, output):
    row = input[0].to(ntl.float16)
    output = row.to(ntl.float16)  # noqa: F841


def test_exact_static_tile_is_marked_in_bounds():
    spec = tensor_spec("input", Tensor(shape=(16, 4096)).tile((16, 128)))
    template = spec.attrs["access_templates"][0]

    assert template["mask"] == "True"
    assert template["statically_in_bounds"] is True
    assert spec.attrs["view_mask"] == "True"
    assert spec.attrs["view_statically_in_bounds"] is True


def test_static_tail_tile_keeps_bounds_mask():
    spec = tensor_spec("input", Tensor(shape=(17, 4097)).tile((16, 128)))
    template = spec.attrs["access_templates"][0]

    assert template["mask"] != "True"
    assert "statically_in_bounds" not in template


def test_static_partial_tile_keeps_only_unproven_bounds():
    spec = tensor_spec("input", Tensor(shape=(1, 128)).tile((16, 128)))
    template = spec.attrs["access_templates"][0]

    assert template["mask"] != "True"
    assert template["mask"].count("<") == 1
    assert ">=" not in template["mask"]


def test_dynamic_tile_keeps_bounds_mask():
    spec = tensor_spec("input", Tensor(2).tile((16, 128)))
    template = spec.attrs["access_templates"][0]

    assert template["mask"] != "True"
    assert "statically_in_bounds" not in template


def test_triton_emission_elides_masks_only_for_exact_static_tile():
    exact = lower(
        _matmul_arrangement,
        _matmul_application,
        (
            Tensor(shape=(16, 128), dtype="float16"),
            Tensor(shape=(128, 128), dtype="float16"),
            Tensor(shape=(16, 128), dtype="float16"),
        ),
        backend="triton",
        kernel_name="static_exact_tile",
    )
    tail = lower(
        _matmul_arrangement,
        _matmul_application,
        (
            Tensor(shape=(17, 129), dtype="float16"),
            Tensor(shape=(129, 129), dtype="float16"),
            Tensor(shape=(17, 129), dtype="float16"),
        ),
        backend="triton",
        kernel_name="static_tail_tile",
    )

    assert "mask=" not in exact.primary_source
    assert "mask=" in tail.primary_source


def test_triton_emits_static_row_broadcast_for_singleton_tile():
    artifact = lower(
        _singleton_broadcast_arrangement,
        _singleton_broadcast_application,
        (
            Tensor(shape=(1, 128), dtype="float16"),
            Tensor(shape=(16, 128), dtype="float16"),
        ),
        backend="triton",
        kernel_name="singleton_block_broadcast",
    )

    assert "tl.load(input" in artifact.primary_source
