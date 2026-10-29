import pytest

from ninetoothed import Tensor
from ninetoothed.backends.core import Target
from ninetoothed.compiler.layout import LayoutTransfer, analyze_layout_transfer
from ninetoothed.compiler.passes import lower_for_target
from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import AccessMap, IndexExpr, TensorLayout, TensorSpec

_COPY_SOURCE = "\ndef copy(x, out):\n    out = x\n"


def _expr(value):
    return IndexExpr.parse(value)


def _access(indices, strides, predicate=True, linear_index=None):
    indices = tuple(_expr(index) for index in indices)
    strides = tuple(_expr(stride) for stride in strides)

    return AccessMap(
        source_indices=indices,
        linear_index=(
            _expr(linear_index)
            if linear_index is not None
            else IndexExpr(
                op="add",
                operands=(
                    IndexExpr(op="mul", operands=(indices[0], strides[0])),
                    IndexExpr(op="mul", operands=(indices[1], strides[1])),
                ),
            )
        ),
        predicate=_expr(predicate),
    )


def _spec(
    name,
    *,
    source_shape,
    value_shape=("m", "n"),
    program_shape=("p", "q"),
    indices=("value_0", "value_1"),
    strides=("ld", 1),
    predicate=True,
    tiled=True,
    linear_index=None,
):
    access = _access(indices, strides, predicate, linear_index)
    layout = TensorLayout(
        source_shape=tuple(_expr(dim) for dim in source_shape),
        source_strides=tuple(_expr(stride) for stride in strides),
        view_shape=tuple(_expr(dim) for dim in program_shape),
        application_shape=tuple(_expr(dim) for dim in value_shape),
        view_access=access,
        value_accesses=(access,) if tiled else (),
    )

    return TensorSpec(
        ndim=2,
        shape=tuple(value_shape),
        dtype="float32",
        name=name,
        layout=layout,
    )


def _permuted_specs(
    *,
    source_program=("p", "q"),
    destination_program=("p", "q"),
    source_indices=("outer_index + value_1", "value_0"),
    destination_indices=("value_0", "outer_index + value_1"),
    source_strides=("ld", 1),
    source_predicate=True,
    source_linear_index=None,
    source_tiled=True,
    destination_tiled=True,
    names=("x", "out"),
):
    return (
        _spec(
            names[0],
            source_shape=("n", "m"),
            program_shape=source_program,
            indices=source_indices,
            strides=source_strides,
            predicate=source_predicate,
            linear_index=source_linear_index,
            tiled=source_tiled,
        ),
        _spec(
            names[1],
            source_shape=("m", "n"),
            program_shape=destination_program,
            indices=destination_indices,
            tiled=destination_tiled,
        ),
    )


def _program(source, tensors, kind="layout_transfer"):
    program = from_source(source, tensors, kind=kind)
    assert program is not None

    return program


def _analyze_copy(specs):
    return analyze_layout_transfer(_program(_COPY_SOURCE, specs), specs)


def _assert_copy_rejected(specs):
    assert _analyze_copy(specs) is None


def test_explicit_transpose_builds_structured_immutable_access_contract():
    program = _program(
        "\ndef transpose(x, out):\n    out = x.T\n",
        (
            TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
            TensorSpec(
                ndim=2,
                shape=("output_rows", "output_columns"),
                dtype="float32",
                name="out",
            ),
        ),
    )

    transfer = analyze_layout_transfer(program)

    assert isinstance(transfer, LayoutTransfer)
    assert transfer.source_binding == "x"
    assert transfer.destination_binding == "out"
    assert transfer.permutation == (1, 0)
    assert transfer.source.access_map.source_indices == (
        _expr("value_1"),
        _expr("value_0"),
    )
    assert transfer.source.layout.application_shape == (_expr("n"), _expr("m"))
    assert transfer.value_constraints == (
        (
            (_expr("n"), _expr("m")),
            (_expr("output_rows"), _expr("output_columns")),
        ),
    )
    assert transfer.physical_constraints == (
        (_expr("n"), _expr("output_rows")),
        (_expr("m"), _expr("output_columns")),
    )
    assert transfer.requires_tiling
    assert transfer.schedulable


def test_dynamic_constraints_and_structured_contract_reach_schedule():
    specs = _permuted_specs(
        source_program=("source_p", "source_q"),
        destination_program=("destination_p", "destination_q"),
    )
    program = _program(_COPY_SOURCE, specs)

    lowered = lower_for_target(program, backend=Target.TRITON, tensors=specs)
    transfer = lowered.metadata["analysis"]["layout_transfer"]

    assert transfer.schedulable
    assert transfer.failure_reason is None
    assert transfer.program_constraints == (
        (
            (_expr("source_p"), _expr("source_q")),
            (_expr("destination_p"), _expr("destination_q")),
        ),
    )
    assert transfer.value_constraints == (
        (
            (_expr("m"), _expr("n")),
            (_expr("m"), _expr("n")),
        ),
    )
    assert transfer.physical_constraints == (
        (_expr("m"), _expr("m")),
        (_expr("n"), _expr("n")),
    )
    assert lowered.metadata["schedule"]["granularity"] == "layout-transfer"
    assert lowered.metadata["schedule"]["layout_transfer"] is transfer

    overridden = lower_for_target(
        program,
        backend=Target.TRITON,
        tensors=specs,
        pass_options={
            "ssa.select_schedule": {"granularity": "elementwise-grid"},
        },
    )

    assert "layout_transfer" not in overridden.metadata["schedule"]


@pytest.mark.parametrize(
    ("specs", "source", "reason"),
    (
        (
            _permuted_specs(source_program=(2, 3), destination_program=(3, 2)),
            "\ndef copy(x, out):\n    out = x\n",
            "program shapes are statically incompatible",
        ),
        (
            (
                TensorSpec(ndim=2, shape=(2, 3), dtype="float32", name="x"),
                TensorSpec(ndim=2, shape=(3, 4), dtype="float32", name="out"),
            ),
            "\ndef transpose(x, out):\n    out = x.T\n",
            "logical value shapes are statically incompatible",
        ),
    ),
)
def test_static_mismatch_is_rejected_from_layout_schedule(specs, source, reason):
    program = _program(source, specs)

    lowered = lower_for_target(program, backend=Target.TRITON, tensors=specs)
    transfer = analyze_layout_transfer(program, specs)

    assert not transfer.schedulable
    assert transfer.failure_reason == reason
    assert lowered.metadata["schedule"]["granularity"] == "elementwise-grid"
    assert "layout_transfer" not in lowered.metadata["schedule"]


def test_real_tiled_permute_uses_value_axis_contributions():
    source = Tensor(shape=(8, 12)).permute((1, 0)).tile((3, 2))
    destination = Tensor(shape=(12, 8)).tile((3, 2))
    specs = tensor_specs(("x", "out"), (source, destination))
    transfer = analyze_layout_transfer(
        _program(_COPY_SOURCE, specs),
        specs,
    )

    assert transfer.permutation == (1, 0)
    assert transfer.source.access_map is specs[0].layout.value_accesses[-1]
    assert transfer.destination.access_map is specs[1].layout.value_accesses[-1]
    assert not transfer.requires_tiling
    assert transfer.schedulable


def test_original_non_contiguous_stride_and_predicate_are_preserved():
    specs = _permuted_specs(
        source_strides=("row_stride", "column_stride"),
        source_predicate="value_1 < n",
    )
    transfer = _analyze_copy(specs)

    assert transfer.source.layout is specs[0].layout
    assert transfer.source.access_map is specs[0].layout.value_accesses[-1]
    assert transfer.source.layout.source_strides == (
        _expr("row_stride"),
        _expr("column_stride"),
    )
    assert transfer.source.access_map.predicate == _expr("value_1 < n")


def test_multiple_live_layout_transfers_are_ambiguous():
    first = _permuted_specs(names=("x0", "out0"))
    second = _permuted_specs(names=("x1", "out1"))
    specs = (*first, *second)
    program = _program(
        "\ndef two(x0, out0, x1, out1):\n    out0 = x0\n    out1 = x1\n",
        specs,
    )

    assert analyze_layout_transfer(program, specs) is None


def test_unproven_effect_or_coordinate_contract_is_rejected():
    specs = _permuted_specs()
    conditional = _program(
        "\ndef conditional(x, out):\n    if x > 0:\n        out = x\n",
        specs,
    )
    unknown_axes = _permuted_specs(
        source_indices=("j", "i"),
        destination_indices=("i", "j"),
    )
    mixed_specs = tensor_specs(
        ("x", "out"),
        (
            Tensor(shape=(8, 12)).permute((1, 0)).tile((3, 2)),
            Tensor(shape=(12, 8)),
        ),
    )
    scaled_axes = _permuted_specs(
        source_indices=("value_1", "2 * value_0"),
        destination_indices=("value_0", "value_1"),
    )
    inconsistent_index = _permuted_specs(
        source_indices=("value_1", "value_0"),
        destination_indices=("value_0", "value_1"),
        source_linear_index="value_0 + value_1",
    )
    inconsistent_outer_mapping = _permuted_specs(
        destination_indices=("value_0", "2 * outer_index + value_1"),
    )
    unmapped_outer_domain = _permuted_specs(
        source_indices=("value_1", "value_0"),
        destination_indices=("value_0", "value_1"),
    )
    missing_value_access = _permuted_specs(
        source_tiled=False,
        destination_tiled=False,
    )
    indexed_store = _program(
        """
def indexed(x, out):
    out.source[out.offsets(0), out.offsets(1)] = x
""",
        specs,
    )

    assert analyze_layout_transfer(conditional, specs) is None
    assert analyze_layout_transfer(indexed_store, specs) is None

    for rejected_specs in (
        inconsistent_index,
        inconsistent_outer_mapping,
        missing_value_access,
        mixed_specs,
        scaled_axes,
        unknown_axes,
        unmapped_outer_domain,
    ):
        _assert_copy_rejected(rejected_specs)
